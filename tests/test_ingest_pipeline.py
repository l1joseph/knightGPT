"""Unit tests for the Postgres-backed ingestion pipeline wiring."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.unit
def test_run_pipeline_calls_insert_chunks_not_build_graph_from_chunks(tmp_path):
    """run_pipeline's graph step should call the Postgres insert helper."""
    from scripts.ingest_pipeline import run_pipeline

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    with patch("scripts.ingest_pipeline.batch_convert_pdfs", return_value=[]), patch(
        "scripts.ingest_pipeline.SemanticChunker"
    ) as MockChunker, patch(
        "scripts.ingest_pipeline.VLLMEmbedder"
    ) as MockEmbedder, patch(
        "scripts.ingest_pipeline.get_pg_pool", new_callable=AsyncMock
    ) as mock_get_pool, patch(
        "scripts.ingest_pipeline.insert_chunks", new_callable=AsyncMock
    ) as mock_insert:

        MockChunker.return_value.chunk_directory.return_value = []
        mock_embedder = MockEmbedder.return_value
        mock_embedder.check_health.return_value = True
        mock_embedder.embed_chunks.return_value = []
        mock_insert.return_value = {
            "chunks_inserted": 0,
            "edges_inserted": 0,
            "papers_inserted": 0,
        }

        run_pipeline(input_dir=input_dir, output_dir=output_dir)

        mock_insert.assert_called_once()
