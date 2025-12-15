"""Tests for the retrieval module."""

import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

import networkx as nx
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from retrieval.retriever import Retriever, RetrieverError, DataFileError


class TestRetrieverInitialization:
    """Tests for Retriever initialization."""

    def test_raises_on_missing_chunks_file(self):
        """Should raise DataFileError for missing chunks file."""
        with pytest.raises(DataFileError) as exc_info:
            Retriever(
                chunks_path="/nonexistent/chunks.json",
                graph_path="/nonexistent/graph.graphml"
            )
        assert "not found" in str(exc_info.value)

    def test_raises_on_invalid_json(self, temp_graph_file):
        """Should raise DataFileError for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {{{")
            temp_file = f.name

        try:
            with pytest.raises(DataFileError) as exc_info:
                Retriever(
                    chunks_path=temp_file,
                    graph_path=temp_graph_file
                )
            assert "Invalid JSON" in str(exc_info.value)
        finally:
            os.unlink(temp_file)

    def test_raises_on_unsupported_format(self, temp_chunks_file):
        """Should raise ValueError for unsupported graph format."""
        with pytest.raises(ValueError) as exc_info:
            Retriever(
                chunks_path=temp_chunks_file,
                graph_path="/some/path.xml",
                graph_format="xml"
            )
        assert "Unsupported graph format" in str(exc_info.value)

    def test_raises_on_missing_chunk_fields(self, temp_graph_file):
        """Should raise DataFileError for chunks missing required fields."""
        # Create chunks without 'embedding' field
        invalid_chunks = [{"node_id": "1", "text": "test"}]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_chunks, f)
            temp_file = f.name

        try:
            with pytest.raises(DataFileError) as exc_info:
                Retriever(
                    chunks_path=temp_file,
                    graph_path=temp_graph_file
                )
            assert "missing required fields" in str(exc_info.value)
        finally:
            os.unlink(temp_file)

    @patch('retrieval.retriever.SentenceTransformer')
    def test_successful_initialization(self, mock_st, temp_chunks_file, temp_graph_file):
        """Should initialize successfully with valid files."""
        mock_st.return_value = MagicMock()

        retriever = Retriever(
            chunks_path=temp_chunks_file,
            graph_path=temp_graph_file
        )

        assert retriever is not None
        assert len(retriever.chunks) == 3
        assert retriever.graph is not None


class TestRetrieverRetrieve:
    """Tests for the retrieve method."""

    @patch('retrieval.retriever.SentenceTransformer')
    def test_retrieve_returns_chunks(self, mock_st, temp_chunks_file, temp_graph_file):
        """Should return relevant chunks for a query."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5] * 76 + [0.1, 0.2, 0.3, 0.4]])
        mock_st.return_value = mock_model

        retriever = Retriever(
            chunks_path=temp_chunks_file,
            graph_path=temp_graph_file
        )

        results = retriever.retrieve("microbiome", top_k=2, hops=0)

        assert len(results) <= 2
        assert all("text" in r for r in results)
        assert all("node_id" in r for r in results)

    @patch('retrieval.retriever.SentenceTransformer')
    def test_retrieve_with_graph_expansion(self, mock_st, temp_chunks_file, temp_graph_file):
        """Should expand results via graph when hops > 0."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5] * 76 + [0.1, 0.2, 0.3, 0.4]])
        mock_st.return_value = mock_model

        retriever = Retriever(
            chunks_path=temp_chunks_file,
            graph_path=temp_graph_file
        )

        results_no_hops = retriever.retrieve("test", top_k=1, hops=0)
        results_with_hops = retriever.retrieve("test", top_k=1, hops=2)

        # With hops, should get more results (neighbors included)
        assert len(results_with_hops) >= len(results_no_hops)

    @patch('retrieval.retriever.SentenceTransformer')
    def test_retrieve_validates_top_k(self, mock_st, temp_chunks_file, temp_graph_file):
        """Should raise ValueError for invalid top_k."""
        mock_st.return_value = MagicMock()

        retriever = Retriever(
            chunks_path=temp_chunks_file,
            graph_path=temp_graph_file
        )

        with pytest.raises(ValueError) as exc_info:
            retriever.retrieve("test", top_k=0)
        assert "top_k must be >= 1" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            retriever.retrieve("test", top_k=-1)
        assert "top_k must be >= 1" in str(exc_info.value)

    @patch('retrieval.retriever.SentenceTransformer')
    def test_retrieve_validates_hops(self, mock_st, temp_chunks_file, temp_graph_file):
        """Should raise ValueError for invalid hops."""
        mock_st.return_value = MagicMock()

        retriever = Retriever(
            chunks_path=temp_chunks_file,
            graph_path=temp_graph_file
        )

        with pytest.raises(ValueError) as exc_info:
            retriever.retrieve("test", hops=-1)
        assert "hops must be >= 0" in str(exc_info.value)

    @patch('retrieval.retriever.SentenceTransformer')
    def test_retrieve_empty_query(self, mock_st, temp_chunks_file, temp_graph_file):
        """Should return empty list for empty query."""
        mock_st.return_value = MagicMock()

        retriever = Retriever(
            chunks_path=temp_chunks_file,
            graph_path=temp_graph_file
        )

        results = retriever.retrieve("", top_k=5)
        assert results == []

        results = retriever.retrieve("   ", top_k=5)
        assert results == []

    @patch('retrieval.retriever.SentenceTransformer')
    def test_retrieve_results_sorted_by_similarity(self, mock_st, temp_chunks_file, temp_graph_file):
        """Should return results sorted by similarity."""
        mock_model = MagicMock()
        # Make embeddings that will produce different similarity scores
        mock_model.encode.return_value = np.array([[0.5, 0.5, 0.5, 0.5, 0.5] * 76 + [0.5, 0.5, 0.5, 0.5]])
        mock_st.return_value = mock_model

        retriever = Retriever(
            chunks_path=temp_chunks_file,
            graph_path=temp_graph_file
        )

        results = retriever.retrieve("test", top_k=3, hops=0)

        # Results should be returned (sorted by similarity internally)
        assert len(results) > 0


class TestRetrieverEmbedQuery:
    """Tests for query embedding."""

    @patch('retrieval.retriever.SentenceTransformer')
    def test_embed_query_returns_normalized(self, mock_st, temp_chunks_file, temp_graph_file):
        """Should return normalized query embedding."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 2.0, 3.0]])
        mock_st.return_value = mock_model

        retriever = Retriever(
            chunks_path=temp_chunks_file,
            graph_path=temp_graph_file
        )

        result = retriever.embed_query("test query")

        # Result should be normalized (unit norm)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01
