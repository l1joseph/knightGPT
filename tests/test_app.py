"""Tests for the API/app module."""

import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.app import (
    RAGChatbot,
    Citation,
    _validate_positive_int,
    _load_pdf_metadata,
    _get_ollama_response_content
)


class TestValidatePositiveInt:
    """Tests for input validation."""

    def test_validates_positive_value(self):
        """Should not raise for positive values."""
        _validate_positive_int(1, "test")
        _validate_positive_int(100, "test")

    def test_raises_for_zero_when_not_allowed(self):
        """Should raise ValueError for zero when allow_zero=False."""
        with pytest.raises(ValueError) as exc_info:
            _validate_positive_int(0, "test_param", allow_zero=False)
        assert "test_param must be >= 1" in str(exc_info.value)

    def test_allows_zero_when_specified(self):
        """Should not raise for zero when allow_zero=True."""
        _validate_positive_int(0, "test", allow_zero=True)

    def test_raises_for_negative(self):
        """Should raise ValueError for negative values."""
        with pytest.raises(ValueError) as exc_info:
            _validate_positive_int(-1, "test_param")
        assert "test_param must be >= 1" in str(exc_info.value)


class TestLoadPdfMetadata:
    """Tests for metadata loading."""

    def test_loads_valid_metadata(self, temp_metadata_file):
        """Should load title and DOI from valid file."""
        title, doi = _load_pdf_metadata(temp_metadata_file)
        assert title == "Microbiome Research Paper"
        assert doi == "10.1234/microbiome.2024"

    def test_returns_defaults_for_missing_file(self):
        """Should return defaults for nonexistent file."""
        title, doi = _load_pdf_metadata("/nonexistent/file.json")
        assert title == "Unknown Title"
        assert doi == "Unknown DOI"

    def test_returns_defaults_for_invalid_json(self):
        """Should return defaults for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json")
            temp_file = f.name

        try:
            title, doi = _load_pdf_metadata(temp_file)
            assert title == "Unknown Title"
            assert doi == "Unknown DOI"
        finally:
            os.unlink(temp_file)

    def test_returns_defaults_for_missing_fields(self):
        """Should return defaults when fields are missing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"metadata": {}}, f)
            temp_file = f.name

        try:
            title, doi = _load_pdf_metadata(temp_file)
            assert title == "Unknown Title"
            assert doi == "Unknown DOI"
        finally:
            os.unlink(temp_file)


class TestCitation:
    """Tests for Citation dataclass."""

    def test_citation_creation(self):
        """Should create citation with all fields."""
        cit = Citation(
            text="Test text",
            node_id="node-001",
            page=1,
            paragraph_index=2,
            chunk_index=3
        )
        assert cit.text == "Test text"
        assert cit.node_id == "node-001"
        assert cit.page == 1
        assert cit.paragraph_index == 2
        assert cit.chunk_index == 3

    def test_citation_with_none_values(self):
        """Should allow None for optional fields."""
        cit = Citation(
            text="Test",
            node_id="node-001",
            page=None,
            paragraph_index=None,
            chunk_index=None
        )
        assert cit.page is None
        assert cit.paragraph_index is None
        assert cit.chunk_index is None


class TestRAGChatbot:
    """Tests for RAGChatbot class."""

    def test_init_with_defaults(self):
        """Should initialize with default paths."""
        chatbot = RAGChatbot()
        assert chatbot.chunks_path == "data/chunks.json"
        assert chatbot.graph_path == "data/graph.graphml"

    def test_init_with_custom_paths(self):
        """Should use custom paths when provided."""
        chatbot = RAGChatbot(
            chunks_path="/custom/chunks.json",
            graph_path="/custom/graph.graphml"
        )
        assert chatbot.chunks_path == "/custom/chunks.json"
        assert chatbot.graph_path == "/custom/graph.graphml"

    def test_lazy_retriever_not_loaded(self):
        """Should not load retriever on init."""
        chatbot = RAGChatbot()
        assert chatbot._retriever is None

    def test_lazy_metadata_not_loaded(self):
        """Should not load metadata on init."""
        chatbot = RAGChatbot()
        assert chatbot._pdf_title is None
        assert chatbot._pdf_doi is None

    def test_pdf_title_lazy_loads(self, temp_metadata_file):
        """Should lazy load PDF title on access."""
        chatbot = RAGChatbot(metadata_path=temp_metadata_file)
        assert chatbot._pdf_title is None

        title = chatbot.pdf_title  # Trigger lazy load

        assert title == "Microbiome Research Paper"
        assert chatbot._pdf_title is not None

    def test_process_query_empty_query(self, capsys):
        """Should handle empty query gracefully."""
        chatbot = RAGChatbot()
        result = chatbot.process_query("")

        assert result is None
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower() or "empty" in captured.out

    def test_process_query_invalid_top_k(self, capsys):
        """Should reject invalid top_k."""
        chatbot = RAGChatbot()
        result = chatbot.process_query("test", top_k=0)

        assert result is None
        captured = capsys.readouterr()
        assert "top_k" in captured.out

    def test_process_query_invalid_hops(self, capsys):
        """Should reject invalid hops."""
        chatbot = RAGChatbot()
        result = chatbot.process_query("test", hops=-1)

        assert result is None
        captured = capsys.readouterr()
        assert "hops" in captured.out


class TestGetOllamaResponseContent:
    """Tests for Ollama response content extraction."""

    def test_extracts_from_message_content(self):
        """Should extract from resp.message.content."""
        resp = MagicMock()
        resp.message = MagicMock()
        resp.message.content = "Test answer"
        assert _get_ollama_response_content(resp) == "Test answer"

    def test_extracts_from_dict(self):
        """Should extract from dict response."""
        resp = {"message": {"content": "Test answer"}}
        assert _get_ollama_response_content(resp) == "Test answer"

    def test_returns_empty_for_none(self):
        """Should return empty string for None."""
        assert _get_ollama_response_content(None) == ""
