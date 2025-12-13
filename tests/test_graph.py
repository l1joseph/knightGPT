"""Unit tests for graph module."""

import pytest
import numpy as np

from src.graph import KnowledgeGraphBuilder
from src.chunking import Chunk


class TestKnowledgeGraphBuilder:
    """Test KnowledgeGraphBuilder class."""

    def test_empty_chunks(self):
        """Test building graph with empty chunks."""
        builder = KnowledgeGraphBuilder()
        graph = builder.build_graph([])
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_chunks_without_embeddings(self):
        """Test building graph with chunks without embeddings."""
        chunks = [
            Chunk(id="1", text="Test", source_file="test.md"),
            Chunk(id="2", text="Test2", source_file="test.md"),
        ]
        builder = KnowledgeGraphBuilder()
        graph = builder.build_graph(chunks)
        assert graph.number_of_nodes() == 0

    def test_chunks_with_embeddings(self):
        """Test building graph with chunks with embeddings."""
        chunks = [
            Chunk(
                id="1",
                text="Test",
                source_file="test.md",
                embedding=[1.0, 0.0, 0.0],
            ),
            Chunk(
                id="2",
                text="Test2",
                source_file="test.md",
                embedding=[0.0, 1.0, 0.0],
            ),
            Chunk(
                id="3",
                text="Test3",
                source_file="test.md",
                embedding=[1.0, 0.0, 0.0],  # Similar to chunk 1
            ),
        ]
        builder = KnowledgeGraphBuilder(similarity_threshold=0.5)
        graph = builder.build_graph(chunks)
        assert graph.number_of_nodes() == 3
        # Should have edges between similar chunks

    def test_get_neighbors_nonexistent(self):
        """Test getting neighbors for non-existent node."""
        builder = KnowledgeGraphBuilder()
        neighbors = builder.get_neighbors("nonexistent", hops=1)
        assert neighbors == []

    def test_get_graph_stats_empty(self):
        """Test graph stats for empty graph."""
        builder = KnowledgeGraphBuilder()
        stats = builder.get_graph_stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0

    def test_find_similar_chunks(self):
        """Test finding similar chunks."""
        chunks = [
            Chunk(
                id="1",
                text="Test",
                source_file="test.md",
                embedding=[1.0, 0.0, 0.0],
            ),
            Chunk(
                id="2",
                text="Test2",
                source_file="test.md",
                embedding=[0.9, 0.1, 0.0],  # Similar to chunk 1
            ),
            Chunk(
                id="3",
                text="Test3",
                source_file="test.md",
                embedding=[0.0, 0.0, 1.0],  # Different
            ),
        ]
        builder = KnowledgeGraphBuilder()
        builder.build_graph(chunks)

        query_embedding = [1.0, 0.0, 0.0]
        results = builder.find_similar_chunks(query_embedding, chunks, top_k=2)
        assert len(results) == 2
        # First result should be most similar
        assert results[0][0].id == "1" or results[0][0].id == "2"

    def test_expand_context(self):
        """Test context expansion."""
        chunks = [
            Chunk(
                id="1",
                text="Test",
                source_file="test.md",
                embedding=[1.0, 0.0, 0.0],
            ),
            Chunk(
                id="2",
                text="Test2",
                source_file="test.md",
                embedding=[0.9, 0.1, 0.0],
            ),
            Chunk(
                id="3",
                text="Test3",
                source_file="test.md",
                embedding=[0.0, 0.0, 1.0],
            ),
        ]
        builder = KnowledgeGraphBuilder(similarity_threshold=0.5)
        builder.build_graph(chunks)

        initial = [chunks[0]]
        expanded = builder.expand_context(initial, chunks, hops=1)
        assert len(expanded) >= 1
        # Should include chunk 2 if connected

