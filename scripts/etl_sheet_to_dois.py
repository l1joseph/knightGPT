#!/usr/bin/env python3
"""
ETL: turn the four newly identified paper sources into DOI lists consumable
by scripts/download_papers.py.

Sources:
1. MMC 2025 Data Sheet (Google Sheet, export the "Final Data Sheet" and/or
   "Draft of Final Data Sheet" tab to local CSV first) — has a DOI column
   directly.
2. Cancer Qiita curation tracker (Google Sheet, export to local CSV) — no
   DOI column; resolves PMID/PMCID from article_link via OpenAlex.
3. Global Human Gut Microbiome Project (Google Sheet, export the
   study-level tab to local CSV) — DOI encoded in publisher URL path.
4. Long-read metagenomics BioProject table (checked in at
   data/paper_lists/sources/longread_bioprojects.tsv) — mostly bare
   doi.org URLs; skips rows explicitly flagged as having no publication.
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from src.utils import get_logger, setup_logging

logger = get_logger(__name__)

OPENALEX_BASE = "https://api.openalex.org"
USER_AGENT = "KnightGPT/1.0 (mailto:knightgpt@ucsd.edu)"


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _resolve_pmid_via_openalex(pmid: str, session: requests.Session) -> str | None:
    """Resolve a PubMed ID to a DOI via OpenAlex."""
    try:
        resp = session.get(
            f"{OPENALEX_BASE}/works",
            params={"filter": f"ids.pmid:{pmid}"},
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if results and results[0].get("doi"):
            return results[0]["doi"].replace("https://doi.org/", "")
    except Exception as e:
        logger.debug(f"OpenAlex PMID lookup failed for {pmid}: {e}")
    return None


DOI_PATH_RE = re.compile(r"/(?:doi|content)/(?:full|abs|pdf)?/?(10\.\d+/[^\s?#]+)")


def _extract_doi_from_url(url: str) -> str | None:
    """Extract a DOI from a doi.org URL, a publisher URL with a /doi/ path
    segment, or a preprint server's /content/ path segment (e.g. bioRxiv/
    medRxiv URLs not yet mirrored to doi.org)."""
    if not url:
        return None
    url = url.strip()
    if "doi.org/" in url:
        doi = url.split("doi.org/", 1)[1]
        return doi.rstrip("/")
    match = DOI_PATH_RE.search(url)
    if match:
        doi = match.group(1)
        # Strip trailing version suffix like v1 from bioRxiv/medRxiv content URLs
        doi = re.sub(r"v\d+$", "", doi)
        return doi
    return None


def resolve_longread_table(tsv_path: Path, session: requests.Session) -> list[str]:
    """Resolve DOIs from the long-read BioProject table. No HTTP calls needed —
    every row either has a resolvable Link or is explicitly flagged with no
    publication."""
    dois = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            citation = (row.get("citation") or "").strip().lower()
            if citation == "no publication" or "vendor reference" in citation:
                continue
            doi = _extract_doi_from_url(row.get("link", ""))
            if doi:
                dois.append(doi)
            else:
                logger.warning(f"Could not resolve DOI for row: {row}")
    return _dedupe_preserve_order(dois)


def resolve_mmc_sheet(csv_path: Path) -> list[str]:
    """Resolve DOIs from an MMC 2025 Data Sheet tab export (has a DOI column)."""
    dois = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doi = (row.get("DOI") or "").strip()
            doi = re.sub(r"^https?://doi\.org/", "", doi)
            if doi:
                dois.append(doi)
    return _dedupe_preserve_order(dois)


def resolve_qiita_tracker(csv_path: Path, session: requests.Session) -> list[str]:
    """Resolve DOIs from a Cancer Qiita tracker tab export. Dedupes by
    qiita_id first (the same study reappears across curation sections),
    then resolves each unique article_link's PMID via OpenAlex."""
    seen_qiita_ids = set()
    links = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qiita_id = (row.get("qiita_id") or "").strip()
            link = (row.get("article_link") or "").strip()
            if not qiita_id or qiita_id in seen_qiita_ids or not link:
                continue
            seen_qiita_ids.add(qiita_id)
            links.append(link)

    dois = []
    for link in links:
        pmid_match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", link)
        if pmid_match:
            doi = _resolve_pmid_via_openalex(pmid_match.group(1), session)
        else:
            doi = _extract_doi_from_url(link)
        if doi:
            dois.append(doi)
        else:
            logger.warning(f"Could not resolve DOI for article_link: {link}")
        time.sleep(0.2)  # be polite to OpenAlex

    return _dedupe_preserve_order(dois)


def resolve_global_gut_sheet(csv_path: Path, session: requests.Session) -> list[str]:
    """Resolve DOIs from the Global Human Gut Microbiome Project study-level
    tab export. Most 'Link to paper' URLs encode a DOI directly; falls back
    to OpenAlex title search for rows that don't."""
    dois = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            link_key = next((k for k in row if "link" in k.lower()), None)
            title_key = next((k for k in row if "title" in k.lower()), None)
            link = (row.get(link_key) or "").strip() if link_key else ""

            doi = _extract_doi_from_url(link)
            if not doi and title_key and row.get(title_key):
                doi = _resolve_title_via_openalex(row[title_key], session)
                time.sleep(0.2)
            if doi:
                dois.append(doi)
            else:
                logger.warning(f"Could not resolve DOI for row: {row}")
    return _dedupe_preserve_order(dois)


def _resolve_title_via_openalex(title: str, session: requests.Session) -> str | None:
    """Resolve a paper title to a DOI via OpenAlex search, as a fallback."""
    try:
        resp = session.get(
            f"{OPENALEX_BASE}/works",
            params={"search": title, "per_page": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if results and results[0].get("doi"):
            return results[0]["doi"].replace("https://doi.org/", "")
    except Exception as e:
        logger.debug(f"OpenAlex title lookup failed for '{title}': {e}")
    return None


def write_doi_file(dois: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for doi in dois:
            f.write(f"{doi}\n")
    logger.info(f"Wrote {len(dois)} DOIs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ETL paper sources into DOI lists")
    parser.add_argument(
        "--mmc-csv", type=Path, help="Local CSV export of an MMC 2025 Data Sheet tab"
    )
    parser.add_argument(
        "--qiita-csv", type=Path, help="Local CSV export of the Cancer Qiita tracker"
    )
    parser.add_argument(
        "--global-gut-csv",
        type=Path,
        help="Local CSV export of the Global Human Gut Microbiome Project study tab",
    )
    parser.add_argument(
        "--longread-tsv",
        type=Path,
        default=Path("data/paper_lists/sources/longread_bioprojects.tsv"),
        help="Long-read BioProject table (checked in by default)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/paper_lists"))
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    session = requests.Session()

    if args.mmc_csv:
        dois = resolve_mmc_sheet(args.mmc_csv)
        write_doi_file(dois, args.output_dir / "mmc_2025_papers.txt")

    if args.qiita_csv:
        dois = resolve_qiita_tracker(args.qiita_csv, session)
        write_doi_file(dois, args.output_dir / "cancer_qiita_papers.txt")

    if args.global_gut_csv:
        dois = resolve_global_gut_sheet(args.global_gut_csv, session)
        write_doi_file(dois, args.output_dir / "global_gut_papers.txt")

    if args.longread_tsv.exists():
        dois = resolve_longread_table(args.longread_tsv, session)
        write_doi_file(dois, args.output_dir / "longread_papers.txt")


if __name__ == "__main__":
    main()
