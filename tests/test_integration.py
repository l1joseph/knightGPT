"""Integration tests for KnightGPT."""

import pytest
import tempfile
from pathlib import Path

from src.chunking import SemanticChunker, save_chunks
from src.graph import KnowledgeGraphBuilder
from src.chunking import Chunk


class TestIngestionPipeline:
    """Test full ingestion pipeline."""

    def test_pipeline_flow(self):
        """Test complete pipeline flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Create test markdown
            md_dir = Path(tmpdir) / "markdown"
            md_dir.mkdir()
            test_file = md_dir / "test.md"
            test_file.write_text("# Test Document\n\nThis is test content.")

            # Step 2: Chunk
            chunker = SemanticChunker(max_tokens=100)
            chunks_file = Path(tmpdir) / "chunks.json"
            chunks = chunker.chunk_directory(md_dir, chunks_file)
            assert len(chunks) > 0

            # Step 3: Add mock embeddings
            for chunk in chunks:
                chunk.embedding = [0.1] * 768

            chunks_with_emb_file = Path(tmpdir) / "chunks_with_emb.json"
            save_chunks(chunks, chunks_with_emb_file)

            # Step 4: Build graph
            graph_file = Path(tmpdir) / "graph.graphml"
            builder = KnowledgeGraphBuilder(similarity_threshold=0.5)
            graph = builder.build_graph(chunks)
            builder.save_graph(graph_file)

            assert graph.number_of_nodes() > 0
            assert graph_file.exists()

    def test_chunk_to_graph_consistency(self):
        """Test consistency between chunks and graph."""
        chunks = [
            Chunk(
                id=f"chunk_{i}",
                text=f"Test chunk {i}",
                source_file="test.md",
                embedding=[float(i % 10) / 10.0] * 768,
            )
            for i in range(10)
        ]

        builder = KnowledgeGraphBuilder()
        graph = builder.build_graph(chunks)

        # All chunk IDs should be in graph
        chunk_ids = {c.id for c in chunks}
        graph_ids = set(graph.nodes())
        assert chunk_ids == graph_ids

