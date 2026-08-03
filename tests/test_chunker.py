"""Tests for text chunking module."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chunking.chunker import Chunker


class TestChunker:
    """Tests for the Chunker class."""

    def test_chunk_single_paragraph(self):
        """Should create single chunk for short paragraph."""
        chunker = Chunker(max_tokens=500, min_tokens=0)  # Disable min filter for test
        pages = ["This is a short paragraph."]
        chunks = chunker.chunk_pages(pages)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "This is a short paragraph."
        assert chunks[0]["page"] == 1
        assert chunks[0]["paragraph_index"] == 1
        assert chunks[0]["chunk_index"] == 1
        assert "node_id" in chunks[0]

    def test_chunk_multiple_paragraphs(self):
        """Should create separate chunks for multiple paragraphs."""
        chunker = Chunker(max_tokens=500, min_tokens=0)  # Disable min filter for test
        pages = ["First paragraph.\n\nSecond paragraph."]
        chunks = chunker.chunk_pages(pages)

        assert len(chunks) == 2
        assert chunks[0]["text"] == "First paragraph."
        assert chunks[1]["text"] == "Second paragraph."
        assert chunks[0]["paragraph_index"] == 1
        assert chunks[1]["paragraph_index"] == 2

    def test_chunk_multiple_pages(self):
        """Should track page numbers correctly."""
        chunker = Chunker(max_tokens=500, min_tokens=0)  # Disable min filter for test
        pages = ["Page 1 content.", "Page 2 content."]
        chunks = chunker.chunk_pages(pages)

        assert len(chunks) == 2
        assert chunks[0]["page"] == 1
        assert chunks[1]["page"] == 2

    def test_chunk_long_paragraph(self):
        """Should split long paragraphs into multiple chunks."""
        chunker = Chunker(max_tokens=10, min_tokens=0)  # Very small limit, disable min filter
        # Create text with multiple sentences
        long_text = "This is sentence one. This is sentence two. This is sentence three."
        pages = [long_text]
        chunks = chunker.chunk_pages(pages)

        # Should create multiple chunks
        assert len(chunks) > 1
        # All should be from same paragraph
        assert all(c["paragraph_index"] == 1 for c in chunks)
        # Chunk indices should increment
        assert chunks[0]["chunk_index"] == 1
        assert chunks[1]["chunk_index"] == 2

    def test_unique_node_ids(self):
        """Should generate unique node IDs for each chunk."""
        chunker = Chunker(max_tokens=500, min_tokens=0)  # Disable min filter for test
        pages = ["First para.\n\nSecond para.\n\nThird para."]
        chunks = chunker.chunk_pages(pages)

        node_ids = [c["node_id"] for c in chunks]
        assert len(node_ids) == len(set(node_ids))  # All unique

    def test_empty_pages(self):
        """Should handle empty pages gracefully."""
        chunker = Chunker(max_tokens=500, min_tokens=0)
        pages = ["", "   ", "\n\n"]
        chunks = chunker.chunk_pages(pages)

        assert len(chunks) == 0

    def test_mixed_empty_and_content(self):
        """Should skip empty paragraphs but process content."""
        chunker = Chunker(max_tokens=500, min_tokens=0)  # Disable min filter for test
        pages = ["Content here.\n\n\n\nMore content."]
        chunks = chunker.chunk_pages(pages)

        assert len(chunks) == 2
        assert chunks[0]["text"] == "Content here."
        assert chunks[1]["text"] == "More content."

    def test_token_count(self):
        """Should count tokens approximately by whitespace."""
        chunker = Chunker(max_tokens=500)
        assert chunker._token_count("one two three") == 3
        assert chunker._token_count("single") == 1
        assert chunker._token_count("") == 0

    def test_preserves_text_content(self):
        """Should preserve original text content."""
        chunker = Chunker(max_tokens=500, min_tokens=0)  # Disable min filter for test
        original_text = "Special chars: $#@! and numbers 12345."
        pages = [original_text]
        chunks = chunker.chunk_pages(pages)

        assert chunks[0]["text"] == original_text

    def test_min_tokens_filters_short_chunks(self):
        """Should filter out chunks below min_tokens threshold."""
        chunker = Chunker(max_tokens=500, min_tokens=5)
        pages = ["Short.\n\nThis is a longer paragraph with more words."]
        chunks = chunker.chunk_pages(pages)

        # "Short." has 1 token, should be filtered
        # The longer paragraph has 8 tokens, should pass
        assert len(chunks) == 1
        assert "longer paragraph" in chunks[0]["text"]

    def test_min_tokens_default_filters_garbage(self):
        """Default min_tokens should filter garbage chunks like arXiv sidebar text."""
        chunker = Chunker(max_tokens=500)  # Default min_tokens=10
        pages = ["5202\n\nrpA2\n\nThis is actual content with enough words to pass the filter easily."]
        chunks = chunker.chunk_pages(pages)

        # "5202" and "rpA2" should be filtered out (1 token each)
        assert len(chunks) == 1
        assert "actual content" in chunks[0]["text"]
