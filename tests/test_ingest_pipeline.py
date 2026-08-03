"""Unit tests for the Postgres-backed ingestion pipeline wiring."""

from unittest.mock import AsyncMock, patch

import pytest

from src.chunking import Chunk


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
    ), patch(
        "scripts.ingest_pipeline.insert_chunks", new_callable=AsyncMock
    ) as mock_insert, patch(
        "scripts.ingest_pipeline.DuckDBStore"
    ):

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


@pytest.mark.unit
def test_run_pipeline_resolves_real_doi_not_source_file_path(tmp_path):
    """Regression test for the bug where papers.doi got the chunk's
    filesystem source_file instead of a real DOI. run_pipeline's papers
    dict must key "doi" off resolve_doi(), not off c.source_file directly,
    for every chunk it passes to insert_chunks()."""
    from scripts.ingest_pipeline import run_pipeline

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    source_file = (
        "/cosmos/vast/scratch/l1joseph/knightgpt/data/processed/markdown/"
        "10-1128_mbio-00519-19.md"
    )
    real_doi = "10.1128/mbio.00519-19"
    embedded_chunk = Chunk(
        id="c1",
        text="chunk text",
        source_file=source_file,
        section="Intro",
        metadata={"title": "Some Paper"},
        token_count=5,
        embedding=[0.1] * 3584,
    )

    with patch("scripts.ingest_pipeline.batch_convert_pdfs", return_value=[]), patch(
        "scripts.ingest_pipeline.SemanticChunker"
    ) as MockChunker, patch(
        "scripts.ingest_pipeline.VLLMEmbedder"
    ) as MockEmbedder, patch(
        "scripts.ingest_pipeline.get_pg_pool", new_callable=AsyncMock
    ), patch(
        "scripts.ingest_pipeline.insert_chunks", new_callable=AsyncMock
    ) as mock_insert, patch(
        "scripts.ingest_pipeline.build_doi_lookup",
        return_value={"10-1128_mbio-00519-19": real_doi},
    ), patch(
        "scripts.ingest_pipeline.DuckDBStore"
    ):

        MockChunker.return_value.chunk_directory.return_value = [embedded_chunk]
        mock_embedder = MockEmbedder.return_value
        mock_embedder.check_health.return_value = True
        mock_embedder.embed_chunks.return_value = [embedded_chunk]
        mock_insert.return_value = {
            "chunks_inserted": 1,
            "edges_inserted": 0,
            "papers_inserted": 1,
        }

        run_pipeline(input_dir=input_dir, output_dir=output_dir)

        mock_insert.assert_called_once()
        _pool, _chunks, papers, _store = mock_insert.call_args.args
        assert papers[source_file]["doi"] == real_doi
        assert papers[source_file]["doi"] != source_file
