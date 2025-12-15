"""Pytest fixtures for knightGPT tests."""

import json
import os
import sys
import tempfile
from typing import Dict, List
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_chunks() -> List[Dict]:
    """Sample chunks with embeddings for testing."""
    return [
        {
            "node_id": "chunk-001",
            "page": 1,
            "paragraph_index": 1,
            "chunk_index": 1,
            "text": "The human microbiome is a complex ecosystem.",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 76 + [0.1, 0.2, 0.3, 0.4]  # 384 dims
        },
        {
            "node_id": "chunk-002",
            "page": 1,
            "paragraph_index": 2,
            "chunk_index": 1,
            "text": "Gut bacteria play a crucial role in digestion.",
            "embedding": [0.2, 0.3, 0.4, 0.5, 0.1] * 76 + [0.2, 0.3, 0.4, 0.5]  # 384 dims
        },
        {
            "node_id": "chunk-003",
            "page": 2,
            "paragraph_index": 1,
            "chunk_index": 1,
            "text": "Probiotics can help maintain gut health.",
            "embedding": [0.3, 0.4, 0.5, 0.1, 0.2] * 76 + [0.3, 0.4, 0.5, 0.1]  # 384 dims
        },
    ]


@pytest.fixture
def sample_graph() -> nx.Graph:
    """Sample graph for testing."""
    G = nx.Graph()
    G.add_node("chunk-001", text="The human microbiome...")
    G.add_node("chunk-002", text="Gut bacteria play...")
    G.add_node("chunk-003", text="Probiotics can help...")
    G.add_edge("chunk-001", "chunk-002", weight=0.85)
    G.add_edge("chunk-002", "chunk-003", weight=0.75)
    return G


@pytest.fixture
def temp_chunks_file(sample_chunks) -> str:
    """Create a temporary chunks JSON file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_chunks, f)
        return f.name


@pytest.fixture
def temp_graph_file(sample_graph) -> str:
    """Create a temporary GraphML file."""
    with tempfile.NamedTemporaryFile(suffix='.graphml', delete=False) as f:
        nx.write_graphml(sample_graph, f.name)
        return f.name


@pytest.fixture
def temp_metadata_file() -> str:
    """Create a temporary metadata JSON file."""
    metadata = {
        "metadata": {
            "title": "Microbiome Research Paper",
            "authors": ["John Doe", "Jane Smith"],
            "publication_date": "2024-01-15",
            "doi": "10.1234/microbiome.2024"
        },
        "pages": ["Page 1 text...", "Page 2 text..."]
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(metadata, f)
        return f.name


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama chat response."""
    response = MagicMock()
    response.message = MagicMock()
    response.message.content = '{"title": "Test Paper", "authors": ["Test Author"]}'
    return response


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model."""
    mock_model = MagicMock()
    # Return 384-dimensional embeddings
    mock_model.encode.return_value = np.random.randn(1, 384)
    return mock_model


@pytest.fixture(autouse=True)
def cleanup_temp_files(request):
    """Cleanup temporary files after tests."""
    temp_files = []

    def register_temp_file(filepath):
        temp_files.append(filepath)
        return filepath

    request.node.register_temp_file = register_temp_file

    yield

    for filepath in temp_files:
        if os.path.exists(filepath):
            os.unlink(filepath)
