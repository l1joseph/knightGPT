#!/usr/bin/env python3
"""
Populate Zotero library with microbiome papers via OpenAlex,
then use ZoteroTool collections to extract DOIs for pipeline ingestion.

Usage:
    python scripts/populate_zotero.py
    python scripts/populate_zotero.py --run-pipeline
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from pyzotero import zotero

from src.utils import get_logger, get_settings, setup_logging
from src.tools.zotero import ZoteroTool

logger = get_logger(__name__)
settings = get_settings()

OPENALEX_BASE = "https://api.openalex.org"

# Microbiome research topics to search
SEARCH_TOPICS = {
    "Microbiome Methods & Tools": [
        "16S rRNA amplicon sequencing microbiome",
        "shotgun metagenomics microbiome analysis",
        "microbiome compositional data analysis",
    ],
    "Gut-Brain Axis": [
        "gut brain axis microbiome",
        "microbiota gut brain behavior",
    ],
    "Microbiome & Cancer": [
        "microbiome cancer immunotherapy",
        "tumor microbiome cancer",
    ],
    "Environmental Microbiome": [
        "soil microbiome ecology",
        "ocean microbiome metagenomics",
    ],
    "Microbiome & Metabolomics": [
        "microbiome metabolomics multi-omics",
        "short chain fatty acids gut microbiota",
    ],
}


def search_openalex(query: str, max_results: int = 10) -> list[dict]:
    """Search OpenAlex for papers, return metadata."""
    session = requests.Session()
    session.headers["User-Agent"] = "KnightGPT/1.0 (mailto:knightgpt@ucsd.edu)"

    resp = session.get(
        f"{OPENALEX_BASE}/works",
        params={
            "search": query,
            "per_page": max_results,
            "sort": "cited_by_count:desc",
            "filter": "is_oa:true,type:article",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    papers = []
    for work in data.get("results", []):
        doi = work.get("doi", "")
        if not doi:
            continue
        # Strip https://doi.org/ prefix
        doi = doi.replace("https://doi.org/", "")

        authors = [
            a.get("author", {}).get("display_name", "")
            for a in work.get("authorships", [])
        ]
        primary_loc = work.get("primary_location") or {}
        source = primary_loc.get("source") or {}

        papers.append({
            "doi": doi,
            "title": work.get("title", ""),
            "authors": authors,
            "journal": source.get("display_name", ""),
            "year": work.get("publication_year"),
            "cited_by_count": work.get("cited_by_count", 0),
            "pdf_url": primary_loc.get("pdf_url"),
        })

    return papers


def create_zotero_item(paper: dict) -> dict:
    """Convert paper metadata to a Zotero journal article item."""
    creators = []
    for name in paper.get("authors", [])[:10]:
        parts = name.rsplit(" ", 1)
        if len(parts) == 2:
            creators.append({
                "creatorType": "author",
                "firstName": parts[0],
                "lastName": parts[1],
            })
        else:
            creators.append({
                "creatorType": "author",
                "lastName": name,
                "firstName": "",
            })

    return {
        "itemType": "journalArticle",
        "title": paper.get("title", ""),
        "creators": creators,
        "DOI": paper.get("doi", ""),
        "publicationTitle": paper.get("journal", ""),
        "date": str(paper.get("year", "")),
        "url": f"https://doi.org/{paper['doi']}",
        "tags": [{"tag": "knightgpt-import"}, {"tag": "microbiome"}],
        "extra": f"Cited by: {paper.get('cited_by_count', 0)}",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Populate Zotero with microbiome papers and ingest"
    )
    parser.add_argument(
        "--papers-per-topic", type=int, default=8,
        help="Max papers per search query",
    )
    parser.add_argument(
        "--run-pipeline", action="store_true",
        help="Run download + ingestion pipeline after populating Zotero",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    # Connect to Zotero
    zt = zotero.Zotero("19943541", "user", "ScJFOVsMFophxs2pP645Wod0")
    logger.info("Connected to Zotero library")

    total_added = 0
    all_collection_keys = []

    for collection_name, queries in SEARCH_TOPICS.items():
        logger.info(f"\n=== Creating collection: {collection_name} ===")

        # Create collection
        coll_resp = zt.create_collections([{"name": collection_name}])
        if not coll_resp:
            logger.error(f"Failed to create collection: {collection_name}")
            continue

        # pyzotero returns a dict with success/failed keys
        if isinstance(coll_resp, dict) and "success" in coll_resp:
            coll_key = list(coll_resp["success"].values())[0] if coll_resp["success"] else None
        elif isinstance(coll_resp, list) and coll_resp:
            coll_key = coll_resp[0].get("data", {}).get("key") or coll_resp[0].get("key")
        else:
            coll_key = None

        if not coll_key:
            logger.error(f"Could not get collection key for: {collection_name}")
            continue

        logger.info(f"Created collection: {collection_name} (key: {coll_key})")
        all_collection_keys.append({"key": coll_key, "name": collection_name})

        # Search OpenAlex for each query in this topic
        seen_dois = set()
        items_to_add = []

        for query in queries:
            logger.info(f"  Searching OpenAlex: '{query}'")
            papers = search_openalex(query, max_results=args.papers_per_topic)
            logger.info(f"  Found {len(papers)} OA papers")

            for paper in papers:
                if paper["doi"] in seen_dois:
                    continue
                seen_dois.add(paper["doi"])
                item = create_zotero_item(paper)
                item["collections"] = [coll_key]
                items_to_add.append(item)

            time.sleep(0.5)  # Rate limit

        # Batch create items in Zotero (max 50 per request)
        if items_to_add:
            for batch_start in range(0, len(items_to_add), 50):
                batch = items_to_add[batch_start:batch_start + 50]
                try:
                    resp = zt.create_items(batch)
                    if isinstance(resp, dict):
                        n_success = len(resp.get("success", {}))
                        n_failed = len(resp.get("failed", {}))
                        logger.info(
                            f"  Added {n_success} items to '{collection_name}' "
                            f"({n_failed} failed)"
                        )
                        total_added += n_success
                    else:
                        total_added += len(batch)
                        logger.info(f"  Added {len(batch)} items to '{collection_name}'")
                except Exception as e:
                    logger.error(f"  Failed to add items: {e}")

            time.sleep(1)

    logger.info(f"\n=== Zotero Population Complete ===")
    logger.info(f"Total items added: {total_added}")
    logger.info(f"Collections created: {len(all_collection_keys)}")

    # Now use ZoteroTool to list collections and extract DOIs
    logger.info("\n=== Using ZoteroTool to extract DOIs ===")
    zt_tool = ZoteroTool(
        library_id="19943541", library_type="user", api_key="ScJFOVsMFophxs2pP645Wod0"
    )

    # List collections
    collections_result = zt_tool.execute(action="collections")
    if collections_result.success:
        logger.info(f"Collections in library:")
        for c in collections_result.data:
            logger.info(f"  {c['key']}: {c['name']} ({c['num_items']} items)")

    # Extract all DOIs and save to file
    doi_file = zt_tool.save_dois_to_file(
        output_path=Path("data/paper_lists/zotero_papers.txt"),
        max_results=500,
    )
    logger.info(f"DOI file saved: {doi_file}")

    # Optionally run the download + ingestion pipeline
    if args.run_pipeline:
        logger.info("\n=== Running download + ingestion pipeline ===")
        from scripts.download_papers import download_papers

        stats = download_papers(doi_file=doi_file)

        # Run pipeline on downloaded papers
        from scripts.ingest_pipeline import run_pipeline

        pipeline_stats = run_pipeline(
            input_dir=settings.ingestion.raw_pdf_dir,
            output_dir=settings.ingestion.processed_dir,
        )

        print("\nPipeline Summary:")
        for key, value in pipeline_stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
