"""Shared DOI resolution for chunks whose ``source_file`` is a markdown path
derived from ``download_papers.py``'s sanitized-DOI filename convention
(``doi.replace("/", "_").replace(".", "-")``), not a real DOI.

Maps that sanitized filename stem back to the real DOI using the checked-in
DOI list files (``data/paper_lists/*.txt``) as the source of truth. This is
what ``papers.doi`` / ``chunks.paper_doi`` must contain — the primary key
that both the one-time migration (``scripts/migrate_to_postgres.py``) and
live ingestion (``scripts/ingest_pipeline.py``, ``src/api/main.py``) write
to, so migrated and freshly-ingested rows for the same paper key identically.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_PAPER_LISTS_DIR = REPO_ROOT / "data" / "paper_lists"


def build_doi_lookup(paper_lists_dir: Path = DEFAULT_PAPER_LISTS_DIR) -> dict[str, str]:
    """Map sanitized-filename-safe DOI (as produced by download_papers.py's
    safe_name transform) back to the real DOI, using the checked-in DOI
    list files as the source of truth."""
    # Lazy import: scripts.download_papers runs get_settings() at module
    # import time, and importing src/ from scripts/ (rather than the usual
    # scripts/ -> src/ direction) is only safe if it stays deferred to call
    # time, not module load time.
    from scripts.download_papers import parse_doi_file

    lookup = {}
    for doi_file in paper_lists_dir.glob("*.txt"):
        for doi in parse_doi_file(doi_file):
            safe_name = doi.replace("/", "_").replace(".", "-")
            lookup[safe_name] = doi
    return lookup


def resolve_doi(source_file: str, doi_lookup: dict[str, str]) -> str:
    """Best-effort DOI resolution from a chunk's source_file. Falls back to
    the raw source_file (previous behavior) if no match is found — e.g. for
    papers ingested by a path that didn't go through the DOI-list-driven
    download flow."""
    stem = Path(source_file).stem if source_file else ""
    return doi_lookup.get(stem, source_file)
