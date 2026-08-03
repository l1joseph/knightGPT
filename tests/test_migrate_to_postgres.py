"""Tests for the one-time file-to-Postgres migration script.

Most of this module needs a live Postgres (see the `integration`-marked test
below). The DOI-resolution tests do not touch Postgres at all — they exercise
`_build_doi_lookup` / `_resolve_doi` and `migrate(dry_run=True)` directly, so
they run (and must pass) even in environments with no database available.
"""

import json
import os

import asyncpg
import networkx as nx
import pytest

from scripts.migrate_to_postgres import _build_doi_lookup, _resolve_doi, migrate

DSN = os.environ.get(
    "TEST_POSTGRES_DSN", "postgresql://postgres:password@localhost:5432/knightgpt"
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dry_run_counts_chunks_edges_papers(tmp_path):
    """Test that dry_run mode counts chunks, edges, and papers correctly,
    and accepts the new duckdb_path parameter."""
    chunks_path = tmp_path / "chunks_with_emb.json"
    chunks_path.write_text(
        json.dumps(
            [
                {
                    "id": "c1",
                    "text": "hello",
                    "source_file": "p1.md",
                    "section": "Intro",
                    "token_count": 5,
                    "embedding": [0.1] * 4,
                    "metadata": {},
                },
                {
                    "id": "c2",
                    "text": "world",
                    "source_file": "p1.md",
                    "section": "Methods",
                    "token_count": 5,
                    "embedding": [0.2] * 4,
                    "metadata": {},
                },
            ]
        )
    )

    graph_path = tmp_path / "graph.graphml"
    g = nx.Graph()
    g.add_edge("c1", "c2", similarity=0.8)
    nx.write_graphml(g, str(graph_path))

    paper_lists_dir = tmp_path / "paper_lists"
    paper_lists_dir.mkdir()

    result = await migrate(
        dsn="postgresql://unused",
        chunks_path=chunks_path,
        graph_path=graph_path,
        duckdb_path=tmp_path / "unused.duckdb",
        dry_run=True,
        paper_lists_dir=paper_lists_dir,
    )

    assert result["dry_run"] is True
    assert result["chunks_migrated"] == 2
    assert result["edges_migrated"] == 1
    assert result["papers_migrated"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_migrate_preserves_chunk_and_edge_counts(tmp_path):
    """migrate() must insert exactly as many chunks and edges as the source files hold."""
    try:
        conn = await asyncpg.connect(DSN)
    except (OSError, asyncpg.PostgresError):
        pytest.skip("No live Postgres available at TEST_POSTGRES_DSN")

    try:
        from scripts.apply_schema import apply_schema

        await conn.execute("DROP TABLE IF EXISTS chunk_edges, chunks, papers CASCADE")
        await apply_schema(DSN)

        chunks_data = [
            {
                "id": f"chunk_{i}",
                "text": f"text {i}",
                "source_file": "paper1.md",
                "page_number": None,
                "section": "Intro",
                "metadata": {},
                "token_count": 5,
                "embedding": [0.1] * 3584,
            }
            for i in range(3)
        ]
        chunks_path = tmp_path / "chunks_with_emb.json"
        chunks_path.write_text(json.dumps(chunks_data))

        graph = nx.Graph()
        for c in chunks_data:
            graph.add_node(c["id"])
        graph.add_edge("chunk_0", "chunk_1", similarity=0.8)
        graph.add_edge("chunk_1", "chunk_2", similarity=0.75)
        graph_path = tmp_path / "graph.graphml"
        nx.write_graphml(graph, str(graph_path))

        result = await migrate(
            DSN, chunks_path, graph_path, tmp_path / "test.duckdb", dry_run=False
        )

        chunk_count = await conn.fetchval("SELECT count(*) FROM chunks")
        edge_count = await conn.fetchval("SELECT count(*) FROM chunk_edges")

        assert chunk_count == 3
        assert edge_count == 2
        assert result["chunks_migrated"] == 3
        assert result["edges_migrated"] == 2
    finally:
        await conn.execute("DROP TABLE IF EXISTS chunk_edges, chunks, papers CASCADE")
        await conn.close()


@pytest.mark.unit
def test_build_doi_lookup_maps_sanitized_filename_to_real_doi(tmp_path):
    """_build_doi_lookup must invert download_papers.py's safe_name transform
    (doi.replace("/", "_").replace(".", "-")) for every DOI in a paper list."""
    paper_lists_dir = tmp_path / "paper_lists"
    paper_lists_dir.mkdir()
    (paper_lists_dir / "initial_papers.txt").write_text(
        "# a comment, should be skipped\n"
        "10.1007/s00192-025-06144-8\n"
        "\n"  # blank line, should be skipped
    )

    lookup = _build_doi_lookup(paper_lists_dir)

    assert lookup["10-1007_s00192-025-06144-8"] == "10.1007/s00192-025-06144-8"


@pytest.mark.unit
def test_build_doi_lookup_merges_all_list_files(tmp_path):
    """Every *.txt file in paper_lists_dir must contribute to the lookup,
    matching how the three real DOI list files (initial/zotero/mmc) are
    all sources of truth for previously-downloaded papers."""
    paper_lists_dir = tmp_path / "paper_lists"
    paper_lists_dir.mkdir()
    (paper_lists_dir / "a.txt").write_text("10.1111/aaa.1\n")
    (paper_lists_dir / "b.txt").write_text("10.2222/bbb.2\n")

    lookup = _build_doi_lookup(paper_lists_dir)

    assert lookup["10-1111_aaa-1"] == "10.1111/aaa.1"
    assert lookup["10-2222_bbb-2"] == "10.2222/bbb.2"


@pytest.mark.unit
def test_resolve_doi_matches_sanitized_source_file_stem():
    """A chunk's source_file (a markdown path named from the sanitized DOI)
    must resolve to the real DOI, not the raw path."""
    doi_lookup = {"10-1007_s00192-025-06144-8": "10.1007/s00192-025-06144-8"}
    source_file = (
        "/cosmos/vast/scratch/l1joseph/knightgpt/data/processed/markdown/"
        "10-1007_s00192-025-06144-8.md"
    )

    resolved = _resolve_doi(source_file, doi_lookup)

    assert resolved == "10.1007/s00192-025-06144-8"


@pytest.mark.unit
def test_resolve_doi_falls_back_to_source_file_when_unmatched():
    """A source_file whose stem doesn't appear in any DOI list (e.g. papers
    ingested outside the DOI-list-driven download flow) must fall back to
    the raw source_file rather than raising or dropping the paper."""
    doi_lookup = {"10-1007_s00192-025-06144-8": "10.1007/s00192-025-06144-8"}
    source_file = "some/unrelated/path/not_a_known_doi.md"

    resolved = _resolve_doi(source_file, doi_lookup)

    assert resolved == source_file


@pytest.mark.unit
def test_resolve_doi_handles_empty_source_file():
    """An empty/falsy source_file must not raise."""
    assert _resolve_doi("", {}) == ""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_migrate_dry_run_dedupes_papers_by_resolved_doi(tmp_path):
    """papers_migrated must count distinct resolved DOIs, not distinct raw
    source_file paths — two chunks whose source_file both sanitize to the
    same DOI are the same paper and must count once."""
    paper_lists_dir = tmp_path / "paper_lists"
    paper_lists_dir.mkdir()
    (paper_lists_dir / "initial_papers.txt").write_text("10.1007/s00192-025-06144-8\n")

    chunks_data = [
        {
            "id": "chunk_0",
            "text": "text 0",
            "source_file": "/data/markdown/10-1007_s00192-025-06144-8.md",
            "page_number": None,
            "section": "Intro",
            "metadata": {},
            "token_count": 5,
            "embedding": [0.1] * 3584,
        },
        {
            "id": "chunk_1",
            "text": "text 1",
            "source_file": "/other/data/markdown/10-1007_s00192-025-06144-8.md",
            "page_number": None,
            "section": "Methods",
            "metadata": {},
            "token_count": 5,
            "embedding": [0.2] * 3584,
        },
        {
            "id": "chunk_2",
            "text": "text 2",
            "source_file": "/data/markdown/unrelated_paper.md",
            "page_number": None,
            "section": "Intro",
            "metadata": {},
            "token_count": 5,
            "embedding": [0.3] * 3584,
        },
    ]
    chunks_path = tmp_path / "chunks_with_emb.json"
    chunks_path.write_text(json.dumps(chunks_data))

    graph = nx.Graph()
    for c in chunks_data:
        graph.add_node(c["id"])
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, str(graph_path))

    result = await migrate(
        "unused-dsn-not-connected-to-in-dry-run",
        chunks_path,
        graph_path,
        tmp_path / "unused.duckdb",
        dry_run=True,
        paper_lists_dir=paper_lists_dir,
    )

    # chunk_0 and chunk_1 both resolve to the same real DOI; chunk_2's
    # source_file has no match and falls back to its raw path. That's 2
    # distinct papers, not 3.
    assert result["chunks_migrated"] == 3
    assert result["papers_migrated"] == 2
