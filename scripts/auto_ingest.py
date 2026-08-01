#!/usr/bin/env python3
"""
Automated paper discovery and ingestion for KnightGPT.

Discovers papers from RSS feeds and kl-tools API, downloads PDFs,
runs the full ingestion pipeline (chunk → embed → graph).

Usage:
    # Discover + download only
    python scripts/auto_ingest.py

    # Full pipeline
    python scripts/auto_ingest.py --run-pipeline

    # Custom feeds
    python scripts/auto_ingest.py --feeds nature_microbiome mbio biorxiv_microbiology
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.rss_feed import RSSFeedIngester, DEFAULT_FEEDS
from src.utils import get_logger, get_settings, setup_logging

logger = get_logger(__name__)
settings = get_settings()


def run_auto_ingest(
    feed_names: list[str] | None = None,
    max_papers: int | None = None,
    run_pipeline: bool = False,
    delay: float = 1.5,
) -> dict:
    """
    Run automated paper discovery and ingestion.

    Args:
        feed_names: Specific feeds to check (None = all configured)
        max_papers: Max papers to process
        run_pipeline: Run full ingestion pipeline after download
        delay: Delay between downloads

    Returns:
        Combined stats
    """
    max_papers = max_papers or settings.rss.max_papers_per_run

    # Select feeds
    if feed_names:
        feeds = {k: v for k, v in DEFAULT_FEEDS.items() if k in feed_names}
        if not feeds:
            logger.error(f"No matching feeds. Available: {list(DEFAULT_FEEDS.keys())}")
            return {"error": "no matching feeds"}
    else:
        feeds = None  # Use all defaults

    ingester = RSSFeedIngester(feeds=feeds)

    # Discover papers
    logger.info("Phase 1: Discovering papers from RSS feeds...")
    rss_papers = ingester.discover_from_rss(
        filter_keywords=settings.rss.filter_keywords,
    )

    logger.info("Phase 1b: Checking kl-tools API...")
    kl_papers = ingester.discover_from_kl_tools(max_results=max_papers)

    all_papers = rss_papers + kl_papers
    if len(all_papers) > max_papers:
        all_papers = all_papers[:max_papers]

    logger.info(f"Total discovered: {len(all_papers)} papers")

    if not all_papers:
        logger.info("No new papers to ingest")
        return {"discovered": 0, "downloaded": 0}

    # Download and convert
    logger.info("Phase 2: Downloading and converting papers...")
    download_stats = ingester.download_and_ingest(all_papers, delay=delay)

    # Run pipeline if requested
    pipeline_stats = {}
    if run_pipeline and download_stats["downloaded"] > 0:
        logger.info("Phase 3: Running ingestion pipeline...")
        from scripts.ingest_pipeline import run_pipeline as run_ingest

        pipeline_stats = run_ingest(
            input_dir=settings.ingestion.raw_pdf_dir,
            output_dir=settings.ingestion.processed_dir,
        )

    # Tracker stats
    tracker_stats = ingester.tracker.get_stats()

    combined = {
        "discovered": len(all_papers),
        "rss_papers": len(rss_papers),
        "kl_tools_papers": len(kl_papers),
        **download_stats,
        "tracker": tracker_stats,
    }
    if pipeline_stats:
        combined["pipeline"] = pipeline_stats

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Automated paper discovery and ingestion"
    )
    parser.add_argument(
        "--feeds",
        nargs="+",
        default=None,
        help=f"Feed names to check. Available: {list(DEFAULT_FEEDS.keys())}",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Max papers to process",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run full ingestion pipeline after download",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Delay between downloads",
    )
    parser.add_argument(
        "--list-feeds",
        action="store_true",
        help="List available feeds and exit",
    )
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    setup_logging(level=args.log_level)

    if args.list_feeds:
        print("Available RSS feeds:")
        for name, url in DEFAULT_FEEDS.items():
            print(f"  {name}: {url}")
        return

    stats = run_auto_ingest(
        feed_names=args.feeds,
        max_papers=args.max_papers,
        run_pipeline=args.run_pipeline,
        delay=args.delay,
    )

    print("\nAuto-Ingest Summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save stats
    processed_dir = settings.ingestion.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    stats_file = processed_dir / "auto_ingest_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2, default=str)


if __name__ == "__main__":
    main()
