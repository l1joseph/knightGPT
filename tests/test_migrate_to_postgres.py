"""Integration test for the one-time file-to-Postgres migration script."""

import json
import os

import asyncpg
import networkx as nx
import pytest

from scripts.migrate_to_postgres import migrate

DSN = os.environ.get(
    "TEST_POSTGRES_DSN", "postgresql://postgres:password@localhost:5432/knightgpt"
)


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

        result = await migrate(DSN, chunks_path, graph_path, dry_run=False)

        chunk_count = await conn.fetchval("SELECT count(*) FROM chunks")
        edge_count = await conn.fetchval("SELECT count(*) FROM chunk_edges")

        assert chunk_count == 3
        assert edge_count == 2
        assert result["chunks_migrated"] == 3
        assert result["edges_migrated"] == 2
    finally:
        await conn.execute("DROP TABLE IF EXISTS chunk_edges, chunks, papers CASCADE")
        await conn.close()
