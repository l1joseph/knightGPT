"""Unit tests for SearchCorpusTool (src/tools/search_corpus.py).

The corpus connection (HybridRetriever.retrieve) is mocked -- no real
Postgres/DuckDB, matching this project's established tool-test convention
(see tests/test_tools_ingest_paper.py, tests/test_tools_base.py).
"""

from unittest.mock import MagicMock

import pytest

from src.chunking import Chunk
from src.retrieval.base import RetrievalResult


@pytest.mark.unit
def test_execute_returns_matches_with_scores():
    from src.tools.search_corpus import SearchCorpusTool

    chunks = [
        Chunk(id="c1", text="Akkermansia is depleted in AD.", source_file="10.1/a"),
        Chunk(id="c2", text="SCFAs modulate neuroinflammation.", source_file="10.1/b"),
    ]
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        chunks=chunks, query_embedding=[0.1, 0.2], similarity_scores=[0.91, 0.82]
    )

    tool = SearchCorpusTool(retriever=mock_retriever)
    result = tool.execute("Akkermansia AD")

    assert result.success is True
    assert result.data == [
        {
            "chunk_id": "c1",
            "source": "10.1/a",
            "section": None,
            "similarity": 0.91,
            "text": "Akkermansia is depleted in AD.",
        },
        {
            "chunk_id": "c2",
            "source": "10.1/b",
            "section": None,
            "similarity": 0.82,
            "text": "SCFAs modulate neuroinflammation.",
        },
    ]
    assert result.metadata == {"query": "Akkermansia AD", "total_results": 2}
    mock_retriever.retrieve.assert_called_once_with("Akkermansia AD", top_k=5)


@pytest.mark.unit
def test_execute_passes_through_custom_top_k():
    from src.tools.search_corpus import SearchCorpusTool

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        chunks=[], query_embedding=[], similarity_scores=[]
    )

    tool = SearchCorpusTool(retriever=mock_retriever, default_top_k=5)
    tool.execute("query", top_k=20)

    mock_retriever.retrieve.assert_called_once_with("query", top_k=20)


@pytest.mark.unit
def test_execute_with_no_query_fails_without_calling_retriever():
    from src.tools.search_corpus import SearchCorpusTool

    mock_retriever = MagicMock()
    tool = SearchCorpusTool(retriever=mock_retriever)

    result = tool.execute("")

    assert result.success is False
    assert "No search query" in result.error
    mock_retriever.retrieve.assert_not_called()


@pytest.mark.unit
def test_execute_with_no_retriever_configured_fails_gracefully():
    from src.tools.search_corpus import SearchCorpusTool

    tool = SearchCorpusTool(retriever=None)
    result = tool.execute("query")

    assert result.success is False
    assert "no corpus connection" in result.error.lower()


@pytest.mark.unit
def test_execute_handles_retrieve_exception():
    from src.tools.search_corpus import SearchCorpusTool

    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = RuntimeError("connection lost")

    tool = SearchCorpusTool(retriever=mock_retriever)
    result = tool.execute("query")

    assert result.success is False
    assert "connection lost" in result.error


@pytest.mark.unit
def test_execute_handles_mismatched_chunk_and_score_lengths():
    """retrieve() can return more chunks than scores (or vice versa) in
    edge cases -- mirrors the same min-length truncation /api/v1/search
    already does for the same reason."""
    from src.tools.search_corpus import SearchCorpusTool

    chunks = [
        Chunk(id="c1", text="first", source_file="10.1/a"),
        Chunk(id="c2", text="second", source_file="10.1/b"),
    ]
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        chunks=chunks, query_embedding=[], similarity_scores=[0.5]
    )

    tool = SearchCorpusTool(retriever=mock_retriever)
    result = tool.execute("query")

    assert result.success is True
    assert len(result.data) == 1
    assert result.data[0]["chunk_id"] == "c1"


@pytest.mark.unit
def test_schema_requires_query():
    from src.tools.search_corpus import SearchCorpusTool

    schema = SearchCorpusTool().schema
    assert schema["parameters"]["required"] == ["query"]
    assert "query" in schema["parameters"]["properties"]
    assert "top_k" in schema["parameters"]["properties"]
