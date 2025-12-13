"""Unit tests for chunking module."""

import pytest
from pathlib import Path
import tempfile
import json

from src.chunking import SemanticChunker, Chunk, load_chunks, save_chunks


class TestSemanticChunker:
    """Test SemanticChunker class."""

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_text("", "test.md")
        assert len(chunks) == 0

    def test_none_text(self):
        """Test chunking None text."""
        chunker = SemanticChunker()
        with pytest.raises((TypeError, AttributeError)):
            chunker.chunk_text(None, "test.md")

    def test_simple_text(self):
        """Test chunking simple text."""
        chunker = SemanticChunker(max_tokens=100)
        text = "This is a test paragraph. It has multiple sentences."
        chunks = chunker.chunk_text(text, "test.md")
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_long_text(self):
        """Test chunking long text."""
        chunker = SemanticChunker(max_tokens=50)
        text = " ".join(["Sentence"] * 100)
        chunks = chunker.chunk_text(text, "test.md")
        assert len(chunks) > 1  # Should be split into multiple chunks

    def test_section_headers(self):
        """Test chunking with section headers."""
        chunker = SemanticChunker()
        text = """
# Introduction
This is the introduction text.

## Methods
This is the methods section.
"""
        chunks = chunker.chunk_text(text, "test.md")
        assert len(chunks) >= 2

    def test_invalid_source_file(self):
        """Test with invalid source file."""
        chunker = SemanticChunker()
        with pytest.raises(ValueError):
            chunker.chunk_text("test", "")

    def test_chunk_markdown_file(self):
        """Test chunking markdown file."""
        chunker = SemanticChunker()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test\n\nThis is test content.")
            temp_path = Path(f.name)

        try:
            chunks = chunker.chunk_markdown_file(temp_path)
            assert len(chunks) > 0
        finally:
            temp_path.unlink(missing_ok=True)

    def test_chunk_directory(self):
        """Test chunking directory."""
        chunker = SemanticChunker()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_file = Path(tmpdir) / "output.json"
            input_dir.mkdir()

            # Create test file
            test_file = input_dir / "test.md"
            test_file.write_text("# Test\n\nContent here.")

            chunks = chunker.chunk_directory(input_dir, output_file)
            assert len(chunks) > 0
            assert output_file.exists()


class TestChunkIO:
    """Test chunk I/O operations."""

    def test_save_and_load_chunks(self):
        """Test saving and loading chunks."""
        chunks = [
            Chunk(
                id="1",
                text="Test chunk 1",
                source_file="test.md",
                embedding=[0.1, 0.2, 0.3],
            ),
            Chunk(
                id="2",
                text="Test chunk 2",
                source_file="test.md",
                embedding=None,
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_chunks(chunks, temp_path)
            assert temp_path.exists()

            loaded = load_chunks(temp_path)
            assert len(loaded) == 2
            assert loaded[0].id == "1"
            assert loaded[1].id == "2"
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_chunks(Path("/nonexistent/file.json"))

    def test_load_invalid_json(self):
        """Test loading invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json {")
            temp_path = Path(f.name)

        try:
            with pytest.raises(json.JSONDecodeError):
                load_chunks(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

