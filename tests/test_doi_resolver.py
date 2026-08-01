"""Unit tests for the shared DOI resolver (src/ingestion/doi_resolver.py).

This module was extracted from scripts/migrate_to_postgres.py (which already
had this logic verified against the real 117-paper corpus) so that live
ingestion call sites (scripts/ingest_pipeline.py, src/api/main.py) resolve
`papers.doi` the same way the one-time migration does, instead of writing a
raw filesystem path into the `doi` primary key column.
"""

from pathlib import Path

import pytest

from src.ingestion.doi_resolver import build_doi_lookup, resolve_doi


@pytest.mark.unit
def test_build_doi_lookup_maps_sanitized_filename_to_real_doi(tmp_path):
    """build_doi_lookup must invert download_papers.py's safe_name transform
    (doi.replace("/", "_").replace(".", "-")) for every DOI in a paper list."""
    paper_lists_dir = tmp_path / "paper_lists"
    paper_lists_dir.mkdir()
    (paper_lists_dir / "initial_papers.txt").write_text(
        "# a comment, should be skipped\n"
        "10.1007/s00192-025-06144-8\n"
        "\n"  # blank line, should be skipped
    )

    lookup = build_doi_lookup(paper_lists_dir)

    assert lookup["10-1007_s00192-025-06144-8"] == "10.1007/s00192-025-06144-8"


@pytest.mark.unit
def test_resolve_doi_matches_sanitized_source_file_stem():
    """A chunk's source_file (a markdown path named from the sanitized DOI)
    must resolve to the real DOI, not the raw path."""
    doi_lookup = {"10-1007_s00192-025-06144-8": "10.1007/s00192-025-06144-8"}
    source_file = (
        "/cosmos/vast/scratch/l1joseph/knightgpt/data/processed/markdown/"
        "10-1007_s00192-025-06144-8.md"
    )

    resolved = resolve_doi(source_file, doi_lookup)

    assert resolved == "10.1007/s00192-025-06144-8"
    assert resolved != source_file


@pytest.mark.unit
def test_resolve_doi_falls_back_to_source_file_when_unmatched():
    """Papers ingested outside the DOI-list-driven download flow (no match
    in any paper list) fall back to the raw source_file rather than
    raising or silently dropping the paper."""
    resolved = resolve_doi("some/unrelated/path/not_a_known_doi.md", {})
    assert resolved == "some/unrelated/path/not_a_known_doi.md"


@pytest.mark.unit
def test_resolve_doi_handles_empty_source_file():
    """An empty/falsy source_file must not raise."""
    assert resolve_doi("", {}) == ""


@pytest.mark.unit
def test_migrate_and_live_ingest_resolve_the_same_doi_for_the_same_source_file(
    tmp_path,
):
    """Strongest regression form of the fix: the migration script's DOI
    resolution (scripts/migrate_to_postgres._resolve_doi, now re-exported
    from this module) and the shared resolver used by live ingestion
    call sites must produce IDENTICAL papers.doi values for identical
    source_file inputs. Before this fix, the migration path resolved a
    real DOI while live ingestion wrote the raw filesystem path — the two
    ingestion routes would key the same paper under two different primary
    keys."""
    from scripts.migrate_to_postgres import _build_doi_lookup, _resolve_doi

    paper_lists_dir = tmp_path / "paper_lists"
    paper_lists_dir.mkdir()
    (paper_lists_dir / "initial_papers.txt").write_text("10.1128/mbio.00519-19\n")

    source_file = "/cosmos/vast/scratch/l1joseph/knightgpt/data/processed/markdown/10-1128_mbio-00519-19.md"

    migration_lookup = _build_doi_lookup(paper_lists_dir)
    live_lookup = build_doi_lookup(paper_lists_dir)

    migration_doi = _resolve_doi(source_file, migration_lookup)
    live_doi = resolve_doi(source_file, live_lookup)

    assert migration_doi == live_doi == "10.1128/mbio.00519-19"
    # And, crucially, neither is the raw markdown path that the pre-fix
    # live-ingestion call sites wrote directly into papers.doi.
    assert migration_doi != source_file
    assert Path(source_file).stem not in (migration_doi, live_doi)
