#!/usr/bin/env python3
"""
Download papers from a DOI list for KnightGPT ingestion.

Resolves DOIs to full text via multiple strategies:
1. Unpaywall API (free, legal open-access PDFs)
2. PubMed Central full-text XML (via NCBI E-utilities)
3. DOI redirect + page scraping (fallback)

Strategy 2 fetches PMC's own full-text XML via efetch rather than
scraping a PDF URL: NCBI retired the old PMC Open Access Web Service
(oa.fcgi) in August 2026, and PMC's article "/pdf/" URLs now redirect
through a JS-based viewer that serves HTML, not a raw PDF stream, so a
plain PDF-URL fetch no longer works for PMC at all. efetch's full-text
XML isn't universal either -- some publishers opt out of XML
redistribution even for PMC-hosted articles -- but it recovers the
genuinely open-access subset without needing PDF conversion at all.

Usage:
    python scripts/download_papers.py --input data/paper_lists/initial_papers.txt
    python scripts/download_papers.py --input data/paper_lists/initial_papers.txt --output $SCRATCH/knightgpt/data/raw_pdfs
    python scripts/download_papers.py --input data/paper_lists/initial_papers.txt --run-pipeline
"""

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.web_scraper import MicrobiomeScraper, ScrapedDocument
from src.utils import get_logger, get_settings, setup_logging

logger = get_logger(__name__)
settings = get_settings()

# Email for Unpaywall API (required, but they just ask for a contact)
UNPAYWALL_EMAIL = "knightgpt@ucsd.edu"


def parse_doi_file(path: Path) -> list[str]:
    """Parse a DOI list file, skipping comments and blanks."""
    dois = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Normalize: strip leading https://doi.org/ if present
        line = re.sub(r"^https?://doi\.org/", "", line)
        dois.append(line)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for doi in dois:
        if doi not in seen:
            seen.add(doi)
            unique.append(doi)
    return unique


def resolve_doi_unpaywall(doi: str, session) -> str | None:
    """Resolve DOI to open-access PDF URL via Unpaywall API."""
    url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # Best OA location
        best = data.get("best_oa_location")
        if best and best.get("url_for_pdf"):
            return best["url_for_pdf"]
        # Check all OA locations
        for loc in data.get("oa_locations", []):
            if loc.get("url_for_pdf"):
                return loc["url_for_pdf"]
    except Exception as e:
        logger.debug(f"Unpaywall lookup failed for {doi}: {e}")
    return None


def resolve_doi_pmc_id(doi: str, session) -> str | None:
    """Look up a DOI's PMCID (e.g. "PMC1317376") via NCBI's ID converter,
    or None if the DOI isn't in PMC at all."""
    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    try:
        resp = session.get(
            url,
            params={"ids": doi, "format": "json", "tool": "knightgpt"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        records = data.get("records", [])
        if records and records[0].get("pmcid"):
            return records[0]["pmcid"]
    except Exception as e:
        logger.debug(f"PMC ID lookup failed for {doi}: {e}")
    return None


def extract_pmc_body_text(root: ET.Element) -> str:
    """Extract a plain-text-with-headings rendering of a PMC JATS XML
    article's <body> -- section titles become "## " headings, paragraphs
    are joined with blank lines. Returns "" if there's no <body> at all
    (the article isn't in PMC's full-text set, or the publisher opted
    out of XML redistribution for it -- PMC's efetch response still
    returns 200 with article metadata in that case, just no body)."""
    body = root.find(".//body")
    if body is None:
        return ""
    parts = []
    for el in body.iter():
        if el.tag == "title":
            text = "".join(el.itertext()).strip()
            if text:
                parts.append(f"## {text}")
        elif el.tag == "p":
            text = "".join(el.itertext()).strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def fetch_pmc_fulltext(pmcid: str, session) -> Optional[dict]:
    """Fetch a PMC article's full-text XML via NCBI's E-utilities efetch
    (the sanctioned replacement for the retired oa.fcgi service and for
    scraping PMC's own JS-rendered article pages) and extract its body
    text and title.

    Returns {"text": str, "title": str} on success, or None if the
    article has no extractable body (not everything in PMC permits full
    XML redistribution) or the request itself fails.
    """
    numeric_id = pmcid[3:] if pmcid.upper().startswith("PMC") else pmcid
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    try:
        resp = session.get(
            url,
            params={"db": "pmc", "id": numeric_id, "rettype": "full", "retmode": "xml"},
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        root = ET.fromstring(resp.content)
        text = extract_pmc_body_text(root)
        if not text:
            return None
        title_el = root.find(".//article-title")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        return {"text": text, "title": title}
    except Exception as e:
        logger.debug(f"PMC full-text fetch failed for {pmcid}: {e}")
        return None


def download_papers(
    doi_file: Path,
    output_dir: Path | None = None,
    delay: float = 1.5,
    on_paper_processed: Optional[
        Callable[[str, Optional[ScrapedDocument], Path], None]
    ] = None,
) -> dict:
    """
    Download papers from DOI list.

    Args:
        doi_file: Path to DOI list file
        output_dir: Where to save downloaded PDFs
        delay: Seconds between requests (be polite)
        on_paper_processed: optional callback invoked once per DOI after
            this function's own download+convert attempt, as
            (doi, doc_or_None, expected_pdf_path). doc is the converted
            ScrapedDocument on success (doc.file_path is the markdown
            output), or None on failure (in which case a partial PDF may
            still exist at expected_pdf_path). Never called for a DOI this
            function skips as already-downloaded. Lets a caller stream
            each paper straight into further processing (e.g. chunk ->
            embed -> insert) and clean up the raw PDF/markdown immediately,
            instead of keeping every paper's files on disk until the whole
            DOI list finishes.

    Returns:
        Stats dict with counts and details
    """
    dois = parse_doi_file(doi_file)
    logger.info(f"Parsed {len(dois)} unique DOIs from {doi_file}")

    download_dir = output_dir or settings.ingestion.raw_pdf_dir
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    markdown_dir = settings.ingestion.markdown_dir
    markdown_dir.mkdir(parents=True, exist_ok=True)

    scraper = MicrobiomeScraper(
        output_dir=markdown_dir,
        download_dir=download_dir,
        delay=delay,
    )

    stats = {
        "total_dois": len(dois),
        "downloaded": 0,
        "failed": 0,
        "skipped": 0,
        "details": [],
    }

    for i, doi in enumerate(dois, 1):
        logger.info(f"[{i}/{len(dois)}] Processing DOI: {doi}")

        # Check if already downloaded
        safe_name = doi.replace("/", "_").replace(".", "-")
        pdf_path = download_dir / f"{safe_name}.pdf"
        existing = list(download_dir.glob(f"*{safe_name}*"))
        if existing:
            logger.info(f"  Already downloaded: {existing[0].name}")
            stats["skipped"] += 1
            stats["details"].append({"doi": doi, "status": "skipped"})
            continue

        # Strategy 1: Unpaywall
        pdf_url = resolve_doi_unpaywall(doi, scraper.session)
        source = "unpaywall"

        doc: Optional[ScrapedDocument] = None

        # Strategy 2: PMC full-text XML (not a PDF fetch -- see module
        # docstring for why the old PDF-URL approach no longer works here)
        if not pdf_url:
            pmcid = resolve_doi_pmc_id(doi, scraper.session)
            if pmcid:
                fulltext = fetch_pmc_fulltext(pmcid, scraper.session)
                if fulltext:
                    output_path = markdown_dir / f"{safe_name}.md"
                    output_path.write_text(fulltext["text"], encoding="utf-8")
                    doc = ScrapedDocument(
                        url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                        title=fulltext["title"] or safe_name,
                        content_type="application/xml",
                        file_path=output_path,
                        metadata={"pmcid": pmcid},
                    )
                    source = "pmc_fulltext_xml"

        # Strategy 3: DOI redirect + scrape
        if doc is None and not pdf_url:
            doi_url = f"https://doi.org/{doi}"
            try:
                resp = scraper.session.get(doi_url, allow_redirects=True, timeout=15)
                if resp.status_code == 200:
                    # Try to find PDF link on the landing page
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(resp.text, "html.parser")
                    pdf_links = scraper._find_pdf_links(soup, resp.url)
                    if pdf_links:
                        pdf_url = pdf_links[0]
                        source = "page_scrape"
            except Exception as e:
                logger.debug(f"  DOI redirect scrape failed: {e}")

        if doc is None and pdf_url:
            logger.info(f"  Found PDF via {source}: {pdf_url}")
            doc = scraper._download_and_process(pdf_url, safe_name)

        if doc:
            doc.metadata["doi"] = doi
            doc.metadata["source"] = source
            stats["downloaded"] += 1
            stats["details"].append(
                {"doi": doi, "status": "downloaded", "source": source}
            )
            logger.info(f"  Downloaded and converted: {doc.file_path}")
        elif pdf_url:
            stats["failed"] += 1
            stats["details"].append(
                {"doi": doi, "status": "failed", "reason": "download_or_convert"}
            )
            logger.warning(f"  Download/convert failed")
        else:
            stats["failed"] += 1
            stats["details"].append(
                {"doi": doi, "status": "failed", "reason": "no_fulltext_found"}
            )
            logger.warning(f"  Could not resolve full text")

        if on_paper_processed is not None:
            on_paper_processed(doi, doc, pdf_path)

        # Rate limiting
        time.sleep(delay)

    logger.info(
        f"Done: {stats['downloaded']} downloaded, "
        f"{stats['skipped']} skipped, {stats['failed']} failed"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download papers from DOI list for KnightGPT"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/paper_lists/initial_papers.txt"),
        help="Path to DOI list file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for PDFs (default: from .env)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Delay between requests in seconds",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run full ingestion pipeline after download",
    )
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()
    setup_logging(level=args.log_level)

    # Download papers
    stats = download_papers(
        doi_file=args.input,
        output_dir=args.output,
        delay=args.delay,
    )

    # Save stats
    processed_dir = settings.ingestion.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    stats_file = processed_dir / "download_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved to {stats_file}")

    # Optionally run full pipeline
    if args.run_pipeline:
        from scripts.ingest_pipeline import run_pipeline

        logger.info("Running ingestion pipeline...")
        pipeline_stats = run_pipeline(
            input_dir=settings.ingestion.raw_pdf_dir,
            output_dir=settings.ingestion.processed_dir,
        )
        print("\nPipeline Summary:")
        for key, value in pipeline_stats.items():
            print(f"  {key}: {value}")

    # Print summary
    print(f"\nDownload Summary:")
    print(f"  Total DOIs: {stats['total_dois']}")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Skipped:    {stats['skipped']}")
    print(f"  Failed:     {stats['failed']}")


if __name__ == "__main__":
    main()
