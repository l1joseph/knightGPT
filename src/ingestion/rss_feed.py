"""RSS feed discovery and paper ingestion for KnightGPT.

Dual-mode paper discovery:
1. Direct RSS parsing via feedparser (standalone)
2. kl-tools API integration for paper discovery (when available)

Tracks processed papers in SQLite to avoid duplicates.
"""

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from ..utils import get_logger, get_settings
from .web_scraper import MicrobiomeScraper, ScrapedDocument

logger = get_logger(__name__)
settings = get_settings()


# Default RSS feeds for microbiome-relevant journals
DEFAULT_FEEDS = {
    "nature_microbiome": "https://www.nature.com/natrevmicro.rss",
    "nature_methods": "https://www.nature.com/nmeth.rss",
    "nature_biotech": "https://www.nature.com/nbt.rss",
    "microbiome_journal": "https://microbiomejournal.biomedcentral.com/articles/most-recent/rss.xml",
    "isme_journal": "https://www.nature.com/ismej.rss",
    "mbio": "https://journals.asm.org/action/showFeed?type=etoc&feed=rss&jc=mbio",
    "msystems": "https://journals.asm.org/action/showFeed?type=etoc&feed=rss&jc=msystems",
    "cell_host_microbe": "https://www.cell.com/cell-host-microbe/rss",
    "pnas": "https://www.pnas.org/action/showFeed?type=etoc&feed=rss&jc=pnas",
    "elife": "https://elifesciences.org/rss/recent.xml",
    "gut": "https://gut.bmj.com/rss/recent.xml",
    "biorxiv_microbiology": "https://connect.biorxiv.org/biorxiv_xml.php?subject=microbiology",
}

# Keywords for filtering papers from general feeds
MICROBIOME_KEYWORDS = [
    "microbiome",
    "microbiota",
    "metagenom",
    "16s rrna",
    "amplicon",
    "gut bacteria",
    "oral microb",
    "skin microb",
    "dysbiosis",
    "probiotic",
    "prebiotic",
    "gnotobiotic",
    "fecal transplant",
    "qiime",
    "unifrac",
    "shotgun sequencing",
    "metabolomics",
    "metatranscriptom",
    "resistome",
    "virome",
    "mycobiome",
]


@dataclass
class DiscoveredPaper:
    """A paper discovered via RSS or API."""

    title: str
    doi: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[str] = None
    journal: Optional[str] = None
    published_date: Optional[str] = None
    source_feed: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def unique_id(self) -> str:
        """Generate unique ID from DOI or URL."""
        key = self.doi or self.url or self.title
        return hashlib.sha256(key.encode()).hexdigest()[:16]


class PaperTracker:
    """SQLite-based tracker for processed papers (dedup)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_papers (
                    unique_id TEXT PRIMARY KEY,
                    doi TEXT,
                    title TEXT,
                    url TEXT,
                    source_feed TEXT,
                    status TEXT DEFAULT 'discovered',
                    processed_at TEXT,
                    error_message TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doi ON processed_papers(doi)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON processed_papers(status)
            """)

    def is_processed(self, paper: DiscoveredPaper) -> bool:
        """Check if a paper has already been processed."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT status FROM processed_papers WHERE unique_id = ?",
                (paper.unique_id,),
            ).fetchone()
            return row is not None

    def mark_processed(
        self, paper: DiscoveredPaper, status: str = "completed", error: str = None
    ):
        """Mark a paper as processed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_papers
                (unique_id, doi, title, url, source_feed, status, processed_at, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper.unique_id,
                    paper.doi,
                    paper.title,
                    paper.url,
                    paper.source_feed,
                    status,
                    datetime.now().isoformat(),
                    error,
                ),
            )

    def get_stats(self) -> dict:
        """Get processing statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM processed_papers"
            ).fetchone()[0]
            by_status = dict(
                conn.execute(
                    "SELECT status, COUNT(*) FROM processed_papers GROUP BY status"
                ).fetchall()
            )
            return {"total": total, **by_status}


class RSSFeedIngester:
    """
    Discover and ingest papers from RSS feeds.

    Supports two modes:
    1. Direct RSS parsing via feedparser
    2. kl-tools API for paper discovery (when configured)
    """

    def __init__(
        self,
        feeds: dict[str, str] | None = None,
        keywords: list[str] | None = None,
        tracker_db: Path | None = None,
        kl_tools_url: str | None = None,
    ):
        self.feeds = feeds or DEFAULT_FEEDS
        self.keywords = keywords or MICROBIOME_KEYWORDS
        self.kl_tools_url = kl_tools_url or settings.rss.kl_tools_api_url
        self.tracker = PaperTracker(
            tracker_db
            or settings.ingestion.processed_dir / "processed_papers.db"
        )
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "KnightGPT Research Bot/1.0 (Academic research)"}
        )

    def discover_from_rss(
        self,
        max_per_feed: int = 20,
        filter_keywords: bool = True,
    ) -> list[DiscoveredPaper]:
        """
        Discover papers from configured RSS feeds.

        Args:
            max_per_feed: Max papers to fetch per feed
            filter_keywords: Whether to filter by microbiome keywords

        Returns:
            List of newly discovered papers (not already processed)
        """
        try:
            import feedparser
        except ImportError:
            logger.error("feedparser not installed: pip install feedparser")
            return []

        discovered = []

        for feed_name, feed_url in self.feeds.items():
            logger.info(f"Fetching RSS feed: {feed_name}")
            try:
                feed = feedparser.parse(feed_url)
                if feed.bozo and not feed.entries:
                    logger.warning(f"Failed to parse {feed_name}: {feed.bozo_exception}")
                    continue

                for entry in feed.entries[:max_per_feed]:
                    paper = self._parse_feed_entry(entry, feed_name)
                    if paper is None:
                        continue

                    # Keyword filtering
                    if filter_keywords and not self._matches_keywords(paper):
                        continue

                    # Dedup check
                    if self.tracker.is_processed(paper):
                        continue

                    discovered.append(paper)

            except Exception as e:
                logger.error(f"Error fetching feed {feed_name}: {e}")

        logger.info(f"Discovered {len(discovered)} new papers from RSS")
        return discovered

    def discover_from_kl_tools(
        self,
        max_results: int = 50,
    ) -> list[DiscoveredPaper]:
        """
        Discover papers via kl-tools API (when available).

        Uses /api/feed/{sourceId} and /api/new-works endpoints.

        Args:
            max_results: Max papers to fetch

        Returns:
            List of newly discovered papers
        """
        if not self.kl_tools_url:
            logger.debug("kl-tools API not configured, skipping")
            return []

        discovered = []
        try:
            # Fetch from new-works endpoint
            resp = self.session.get(
                f"{self.kl_tools_url}/api/new-works",
                params={"limit": max_results},
                timeout=30,
            )
            if resp.status_code == 200:
                works = resp.json()
                for work in works:
                    paper = DiscoveredPaper(
                        title=work.get("title", "Unknown"),
                        doi=work.get("doi"),
                        url=work.get("url"),
                        abstract=work.get("abstract"),
                        authors=work.get("authors"),
                        journal=work.get("journal"),
                        published_date=work.get("published_date"),
                        source_feed="kl-tools",
                    )
                    if not self.tracker.is_processed(paper):
                        discovered.append(paper)
        except Exception as e:
            logger.warning(f"kl-tools API unavailable: {e}")

        logger.info(f"Discovered {len(discovered)} papers from kl-tools")
        return discovered

    def _parse_feed_entry(
        self, entry, feed_name: str
    ) -> DiscoveredPaper | None:
        """Parse a feedparser entry into a DiscoveredPaper."""
        title = entry.get("title", "").strip()
        if not title:
            return None

        # Extract DOI from various locations
        doi = None
        link = entry.get("link", "")

        # Try DOI from link
        doi_match = re.search(r"10\.\d{4,9}/[^\s]+", link)
        if doi_match:
            doi = doi_match.group(0).rstrip(".")

        # Try prism:doi or dc:identifier
        if not doi:
            for key in ("prism_doi", "dc_identifier", "doi"):
                val = entry.get(key, "")
                if val and val.startswith("10."):
                    doi = val
                    break

        abstract = entry.get("summary", entry.get("description", ""))
        # Strip HTML tags from abstract
        abstract = re.sub(r"<[^>]+>", "", abstract).strip() if abstract else None

        authors = None
        if "authors" in entry:
            authors = ", ".join(a.get("name", "") for a in entry["authors"])
        elif "author" in entry:
            authors = entry["author"]

        published = entry.get("published", entry.get("updated"))

        return DiscoveredPaper(
            title=title,
            doi=doi,
            url=link,
            abstract=abstract,
            authors=authors,
            published_date=published,
            source_feed=feed_name,
        )

    def _matches_keywords(self, paper: DiscoveredPaper) -> bool:
        """Check if paper matches microbiome keywords."""
        text = " ".join(
            filter(None, [paper.title, paper.abstract, paper.authors])
        ).lower()
        return any(kw in text for kw in self.keywords)

    def download_and_ingest(
        self,
        papers: list[DiscoveredPaper],
        delay: float = 1.5,
    ) -> dict:
        """
        Download and ingest discovered papers.

        Args:
            papers: Papers to download and process
            delay: Delay between downloads (seconds)

        Returns:
            Stats dict
        """
        from .web_scraper import MicrobiomeScraper

        scraper = MicrobiomeScraper(
            output_dir=settings.ingestion.markdown_dir,
            download_dir=settings.ingestion.raw_pdf_dir,
            delay=delay,
        )

        stats = {"total": len(papers), "downloaded": 0, "failed": 0, "skipped": 0}

        for i, paper in enumerate(papers, 1):
            logger.info(f"[{i}/{len(papers)}] {paper.title[:60]}...")

            if self.tracker.is_processed(paper):
                stats["skipped"] += 1
                continue

            # Resolve PDF URL
            pdf_url = paper.pdf_url
            if not pdf_url and paper.doi:
                pdf_url = self._resolve_doi_to_pdf(paper.doi)
            if not pdf_url and paper.url:
                # Try scraping the paper page
                docs = scraper.scrape_url(paper.url)
                if docs:
                    self.tracker.mark_processed(paper, status="completed")
                    stats["downloaded"] += 1
                    time.sleep(delay)
                    continue

            if pdf_url:
                safe_name = (paper.doi or paper.unique_id).replace("/", "_")
                doc = scraper._download_and_process(pdf_url, safe_name)
                if doc:
                    self.tracker.mark_processed(paper, status="completed")
                    stats["downloaded"] += 1
                else:
                    self.tracker.mark_processed(
                        paper, status="failed", error="download_failed"
                    )
                    stats["failed"] += 1
            else:
                self.tracker.mark_processed(
                    paper, status="failed", error="no_pdf_url"
                )
                stats["failed"] += 1

            time.sleep(delay)

        logger.info(
            f"Ingestion complete: {stats['downloaded']} downloaded, "
            f"{stats['failed']} failed, {stats['skipped']} skipped"
        )
        return stats

    def _resolve_doi_to_pdf(self, doi: str) -> str | None:
        """Resolve DOI to PDF URL via Unpaywall."""
        try:
            resp = self.session.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": "knightgpt@ucsd.edu"},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                best = data.get("best_oa_location")
                if best and best.get("url_for_pdf"):
                    return best["url_for_pdf"]
                for loc in data.get("oa_locations", []):
                    if loc.get("url_for_pdf"):
                        return loc["url_for_pdf"]
        except Exception as e:
            logger.debug(f"Unpaywall lookup failed for {doi}: {e}")
        return None
