"""Pytest configuration and fixtures."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    from src.chunking import Chunk

    return [
        Chunk(
            id="chunk_1",
            text="This is the first test chunk with some content.",
            source_file="test.md",
            section="Introduction",
            embedding=[0.1] * 768,
            token_count=10,
        ),
        Chunk(
            id="chunk_2",
            text="This is the second test chunk with different content.",
            source_file="test.md",
            section="Methods",
            embedding=[0.2] * 768,
            token_count=12,
        ),
        Chunk(
            id="chunk_3",
            text="This is the third test chunk with more content.",
            source_file="test2.md",
            section="Results",
            embedding=[0.3] * 768,
            token_count=11,
        ),
    ]


@pytest.fixture
def sample_markdown_file(temp_dir):
    """Create sample markdown file."""
    md_file = temp_dir / "test.md"
    md_file.write_text(
        """# Test Document

## Introduction
This is the introduction section with some content.

## Methods
This section describes the methods used in the study.

## Results
The results are presented here.

## Discussion
Discussion of the findings.
"""
    )
    return md_file

