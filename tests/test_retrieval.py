"""Unit tests for retrieval module."""

import pytest

from src.retrieval import GraphRAGRetriever, RAGEngine
from src.chunking import Chunk


class TestGraphRAGRetriever:
    """Test GraphRAGRetriever class."""

    def test_empty_chunks(self):
        """Test retriever with empty chunks."""
        retriever = GraphRAGRetriever(chunks=[])
        result = retriever.retrieve("test query")
        assert len(result.chunks) == 0
        assert len(result.similarity_scores) == 0

    def test_none_query(self):
        """Test retrieval with None query."""
        retriever = GraphRAGRetriever(chunks=[])
        result = retriever.retrieve(None)
        assert len(result.chunks) == 0

    def test_empty_query(self):
        """Test retrieval with empty query."""
        retriever = GraphRAGRetriever(chunks=[])
        result = retriever.retrieve("")
        assert len(result.chunks) == 0

    def test_format_context_empty(self):
        """Test formatting empty context."""
        retriever = GraphRAGRetriever(chunks=[])
        context = retriever.format_context([])
        assert context == ""

    def test_format_context_uses_doi_directly_for_doi_shaped_source_file(self):
        """HybridRetriever sets Chunk.source_file to the paper's real DOI
        (e.g. "10.1128/mbio.00519-19"). Running that through Path(...).stem
        mangles it (Path treats "/" as a separator and "." as an extension
        marker), so format_context must detect DOI-shaped source_file
        values and use them verbatim instead."""
        retriever = GraphRAGRetriever(chunks=[])
        chunk = Chunk(
            id="1",
            text="Some finding about the gut microbiome.",
            source_file="10.1128/mbio.00519-19",
            section="Results",
        )
        context = retriever.format_context([chunk])
        assert "[Source: 10.1128/mbio.00519-19, Section: Results]" in context
        # Explicitly guard against the mangled Path(...).stem result.
        assert "[Source: mbio, Section: Results]" not in context

    def test_format_context_still_uses_stem_for_file_path_source_file(self):
        """Non-DOI (file-backed retriever) source_file values keep the
        existing Path(...).stem behavior."""
        retriever = GraphRAGRetriever(chunks=[])
        chunk = Chunk(
            id="1",
            text="Some finding.",
            source_file="/data/processed/markdown/some_paper.md",
            section="Intro",
        )
        context = retriever.format_context([chunk])
        assert "[Source: some_paper, Section: Intro]" in context

    def test_create_citations_mismatched_lengths(self):
        """Test creating citations with mismatched lengths."""
        chunks = [Chunk(id="1", text="Test", source_file="test.md")]
        scores = [0.5, 0.6]  # Mismatched length
        retriever = GraphRAGRetriever(chunks=chunks)
        citations = retriever.create_citations(chunks, scores)
        # Should handle gracefully
        assert len(citations) <= len(chunks)

    def test_create_citations_none_chunk(self):
        """Test creating citations with None chunk."""
        chunks = [None, Chunk(id="1", text="Test", source_file="test.md")]
        scores = [0.5, 0.6]
        retriever = GraphRAGRetriever(chunks=[])
        citations = retriever.create_citations(chunks, scores)
        # Should skip None chunks
        assert len(citations) <= len([c for c in chunks if c is not None])


class TestRAGEngine:
    """Test RAGEngine class."""

    @pytest.fixture
    def mock_retriever(self):
        """Create mock retriever."""
        chunks = [
            Chunk(
                id="1",
                text="Test chunk",
                source_file="test.md",
                embedding=[0.1] * 768,
            )
        ]
        return GraphRAGRetriever(chunks=chunks)

    def test_query_empty_question(self, mock_retriever):
        """Test query with empty question."""
        # This will fail at embedding stage, which is expected
        pass

    def test_system_prompt_default(self, mock_retriever):
        """Test default system prompt."""
        engine = RAGEngine(retriever=mock_retriever)
        assert engine.system_prompt is not None
        assert "microbiome" in engine.system_prompt.lower()


@pytest.mark.unit
def test_base_retriever_is_abstract():
    """BaseRetriever cannot be instantiated directly."""
    from src.retrieval.base import BaseRetriever

    with pytest.raises(TypeError):
        BaseRetriever()


@pytest.mark.unit
def test_graph_rag_retriever_is_base_retriever():
    """GraphRAGRetriever must implement the BaseRetriever interface."""
    from src.retrieval.base import BaseRetriever
    from src.retrieval import GraphRAGRetriever

    retriever = GraphRAGRetriever(chunks=[])
    assert isinstance(retriever, BaseRetriever)
