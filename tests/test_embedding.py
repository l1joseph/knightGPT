"""Unit tests for embedding module."""

import pytest
from unittest.mock import Mock, patch

from src.embedding import VLLMEmbedder


class TestVLLMEmbedder:
    """Test VLLMEmbedder class."""

    def test_empty_text(self):
        """Test embedding empty text."""
        embedder = VLLMEmbedder(api_base="http://test:8001/v1")
        with pytest.raises(ValueError):
            embedder.embed_text("")

    def test_none_text(self):
        """Test embedding None text."""
        embedder = VLLMEmbedder(api_base="http://test:8001/v1")
        with pytest.raises((TypeError, ValueError)):
            embedder.embed_text(None)

    def test_empty_batch(self):
        """Test embedding empty batch."""
        embedder = VLLMEmbedder(api_base="http://test:8001/v1")
        result = embedder.embed_batch([])
        assert result == []

    def test_invalid_batch(self):
        """Test embedding invalid batch."""
        embedder = VLLMEmbedder(api_base="http://test:8001/v1")
        with pytest.raises(TypeError):
            embedder.embed_batch("not a list")

    def test_batch_with_none(self):
        """Test batch with None values."""
        embedder = VLLMEmbedder(api_base="http://test:8001/v1")
        # Should filter out None values
        result = embedder.embed_batch([None, "test", None])
        # Result depends on implementation, but shouldn't crash

    @patch("src.embedding.OpenAI")
    def test_embed_text_success(self, mock_openai):
        """Test successful embedding."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        embedder = VLLMEmbedder(api_base="http://test:8001/v1")
        result = embedder.embed_text("test text")
        assert result == [0.1, 0.2, 0.3]

    @patch("src.embedding.OpenAI")
    def test_embed_batch_success(self, mock_openai):
        """Test successful batch embedding."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(index=0, embedding=[0.1, 0.2]),
            Mock(index=1, embedding=[0.3, 0.4]),
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        embedder = VLLMEmbedder(api_base="http://test:8001/v1")
        result = embedder.embed_batch(["text1", "text2"])
        assert len(result) == 2

    def test_check_health_invalid_server(self):
        """Test health check with invalid server."""
        embedder = VLLMEmbedder(api_base="http://invalid:8001/v1")
        # Should return False or raise exception
        result = embedder.check_health()
        assert result is False

