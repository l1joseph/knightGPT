"""Unit tests for IngestPaperTool (src/tools/ingest_paper.py).

External calls (DOI download/resolution, embedding server) and the live
corpus connection (HybridRetriever.insert_paper) are all mocked -- no
network access, no real Postgres/DuckDB, matching this project's
established tool-test convention (see tests/test_tools_base.py,
tests/test_orchestrator.py's FakeTool pattern).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.api.request_context import RequestContext
from src.chunking import Chunk


def _fake_settings(tmp_path):
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            raw_pdf_dir=tmp_path / "raw_pdfs",
            markdown_dir=tmp_path / "markdown",
        )
    )


def _make_doc(tmp_path, title="Gut Microbiome in Health"):
    md_path = tmp_path / "markdown" / "10-1038_x.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("## Introduction\n\nThe gut microbiome is complex.\n")
    return SimpleNamespace(file_path=md_path, title=title, metadata={})


@pytest.mark.unit
def test_execute_success_downloads_chunks_embeds_and_inserts(tmp_path):
    """Happy path: download -> chunk -> embed -> insert, all mocked to
    succeed, should return a success ToolResult summarizing what was
    added."""
    from src.tools.ingest_paper import IngestPaperTool

    doc = _make_doc(tmp_path)
    chunk = Chunk(
        id="c1", text="The gut microbiome is complex.", source_file=str(doc.file_path)
    )

    mock_retriever = MagicMock()
    mock_retriever.insert_paper.return_value = {
        "papers_inserted": 1,
        "chunks_inserted": 1,
        "edges_inserted": 3,
    }

    tool = IngestPaperTool(retriever=mock_retriever)

    mock_chunker_instance = MagicMock()
    mock_chunker_instance.chunk_markdown_file.return_value = [chunk]

    mock_embedder_instance = MagicMock()
    mock_embedder_instance.check_health.return_value = True

    def _embed_chunks(chunks, **kwargs):
        for c in chunks:
            c.embedding = [0.1, 0.2, 0.3]
        return chunks

    mock_embedder_instance.embed_chunks.side_effect = _embed_chunks

    admin_ctx = RequestContext(
        email="admin@example.com", is_admin=True, collection_id=None
    )

    with (
        patch("src.tools.ingest_paper.settings", _fake_settings(tmp_path)),
        patch(
            "scripts.download_papers.download_single_paper",
            return_value={
                "status": "downloaded",
                "doi": "10.1038/x",
                "doc": doc,
                "source": "unpaywall",
            },
        ),
        patch(
            "src.tools.ingest_paper.SemanticChunker", return_value=mock_chunker_instance
        ),
        patch(
            "src.tools.ingest_paper.VLLMEmbedder", return_value=mock_embedder_instance
        ),
    ):
        result = tool.execute("10.1038/x", request_context=admin_ctx)

    assert result.success is True
    assert result.tool_name == "ingest_paper"
    assert "10.1038/x" in result.data
    assert "1 chunks inserted" in result.data
    assert result.metadata["doi"] == "10.1038/x"
    assert result.metadata["chunks_inserted"] == 1
    assert result.metadata["edges_inserted"] == 3

    mock_retriever.insert_paper.assert_called_once()
    call_kwargs = mock_retriever.insert_paper.call_args.kwargs
    assert call_kwargs["doi"] == "10.1038/x"
    assert call_kwargs["title"] == doc.title
    assert call_kwargs["chunks"] == [chunk]


@pytest.mark.unit
def test_execute_accepts_doi_org_url_and_strips_prefix(tmp_path):
    """A doi.org URL should be normalized to a bare DOI before any
    downstream call."""
    from src.tools.ingest_paper import IngestPaperTool

    doc = _make_doc(tmp_path)
    chunk = Chunk(id="c1", text="text", source_file=str(doc.file_path))

    mock_retriever = MagicMock()
    mock_retriever.insert_paper.return_value = {
        "chunks_inserted": 1,
        "edges_inserted": 0,
    }
    tool = IngestPaperTool(retriever=mock_retriever)

    mock_chunker_instance = MagicMock()
    mock_chunker_instance.chunk_markdown_file.return_value = [chunk]
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.check_health.return_value = True
    mock_embedder_instance.embed_chunks.side_effect = lambda chunks, **kw: (
        [setattr(c, "embedding", [0.1]) or c for c in chunks]
    )

    captured_doi = {}

    def _fake_download(doi, scraper, download_dir, markdown_dir):
        captured_doi["doi"] = doi
        return {"status": "downloaded", "doi": doi, "doc": doc, "source": "unpaywall"}

    admin_ctx = RequestContext(
        email="admin@example.com", is_admin=True, collection_id=None
    )

    with (
        patch("src.tools.ingest_paper.settings", _fake_settings(tmp_path)),
        patch(
            "scripts.download_papers.download_single_paper", side_effect=_fake_download
        ),
        patch(
            "src.tools.ingest_paper.SemanticChunker", return_value=mock_chunker_instance
        ),
        patch(
            "src.tools.ingest_paper.VLLMEmbedder", return_value=mock_embedder_instance
        ),
    ):
        result = tool.execute("https://doi.org/10.1038/x", request_context=admin_ctx)

    assert result.success is True
    assert captured_doi["doi"] == "10.1038/x"


@pytest.mark.unit
def test_execute_no_doi_provided_returns_failure_not_exception():
    from src.tools.ingest_paper import IngestPaperTool

    tool = IngestPaperTool(retriever=MagicMock())
    result = tool.execute("")

    assert result.success is False
    assert "No DOI" in result.error


@pytest.mark.unit
def test_execute_no_retriever_configured_returns_failure_not_exception():
    from src.tools.ingest_paper import IngestPaperTool

    tool = IngestPaperTool(retriever=None)
    admin_ctx = RequestContext(
        email="admin@example.com", is_admin=True, collection_id=None
    )
    result = tool.execute("10.1038/x", request_context=admin_ctx)

    assert result.success is False
    assert "unavailable" in result.error.lower()


@pytest.mark.unit
def test_execute_doi_not_resolvable_returns_clear_failure(tmp_path):
    """download_single_paper reporting no_fulltext_found should surface as
    a clear, honest ToolResult(success=False, ...), not an exception."""
    from src.tools.ingest_paper import IngestPaperTool

    tool = IngestPaperTool(retriever=MagicMock())
    admin_ctx = RequestContext(
        email="admin@example.com", is_admin=True, collection_id=None
    )

    with (
        patch("src.tools.ingest_paper.settings", _fake_settings(tmp_path)),
        patch(
            "scripts.download_papers.download_single_paper",
            return_value={
                "status": "failed",
                "doi": "10.1038/nonexistent",
                "reason": "no_fulltext_found",
            },
        ),
    ):
        result = tool.execute("10.1038/nonexistent", request_context=admin_ctx)

    assert result.success is False
    assert "10.1038/nonexistent" in result.error
    assert result.metadata["reason"] == "no_fulltext_found"


@pytest.mark.unit
def test_execute_already_downloaded_returns_success_with_no_op_message(tmp_path):
    from src.tools.ingest_paper import IngestPaperTool

    tool = IngestPaperTool(retriever=MagicMock())
    admin_ctx = RequestContext(
        email="admin@example.com", is_admin=True, collection_id=None
    )

    with (
        patch("src.tools.ingest_paper.settings", _fake_settings(tmp_path)),
        patch(
            "scripts.download_papers.download_single_paper",
            return_value={"status": "skipped", "doi": "10.1038/x"},
        ),
    ):
        result = tool.execute("10.1038/x", request_context=admin_ctx)

    assert result.success is True
    assert result.metadata["status"] == "already_downloaded"


@pytest.mark.unit
def test_execute_embedding_server_down_returns_failure(tmp_path):
    from src.tools.ingest_paper import IngestPaperTool

    doc = _make_doc(tmp_path)
    chunk = Chunk(id="c1", text="text", source_file=str(doc.file_path))

    tool = IngestPaperTool(retriever=MagicMock())
    admin_ctx = RequestContext(
        email="admin@example.com", is_admin=True, collection_id=None
    )

    mock_chunker_instance = MagicMock()
    mock_chunker_instance.chunk_markdown_file.return_value = [chunk]
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.check_health.return_value = False

    with (
        patch("src.tools.ingest_paper.settings", _fake_settings(tmp_path)),
        patch(
            "scripts.download_papers.download_single_paper",
            return_value={
                "status": "downloaded",
                "doi": "10.1038/x",
                "doc": doc,
                "source": "unpaywall",
            },
        ),
        patch(
            "src.tools.ingest_paper.SemanticChunker", return_value=mock_chunker_instance
        ),
        patch(
            "src.tools.ingest_paper.VLLMEmbedder", return_value=mock_embedder_instance
        ),
    ):
        result = tool.execute("10.1038/x", request_context=admin_ctx)

    assert result.success is False
    assert "embedding server" in result.error.lower()


@pytest.mark.unit
def test_execute_download_exception_returns_failure_not_exception(tmp_path):
    """An unexpected exception from the download step must not propagate
    out of execute() -- it would otherwise kill the whole agent loop turn."""
    from src.tools.ingest_paper import IngestPaperTool

    tool = IngestPaperTool(retriever=MagicMock())
    admin_ctx = RequestContext(
        email="admin@example.com", is_admin=True, collection_id=None
    )

    with (
        patch("src.tools.ingest_paper.settings", _fake_settings(tmp_path)),
        patch(
            "scripts.download_papers.download_single_paper",
            side_effect=RuntimeError("network exploded"),
        ),
    ):
        result = tool.execute("10.1038/x", request_context=admin_ctx)

    assert result.success is False
    assert "network exploded" in result.error


@pytest.mark.unit
def test_execute_non_admin_with_collection_attached_writes_to_that_collection(tmp_path):
    from src.tools.ingest_paper import IngestPaperTool

    doc = _make_doc(tmp_path)
    chunk = Chunk(id="c1", text="text", source_file=str(doc.file_path))

    mock_retriever = MagicMock()
    mock_retriever.insert_paper.return_value = {
        "chunks_inserted": 1,
        "edges_inserted": 0,
    }
    tool = IngestPaperTool(retriever=mock_retriever)

    mock_chunker_instance = MagicMock()
    mock_chunker_instance.chunk_markdown_file.return_value = [chunk]
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.check_health.return_value = True
    mock_embedder_instance.embed_chunks.side_effect = lambda chunks, **kw: (
        [setattr(c, "embedding", [0.1]) or c for c in chunks]
    )

    ctx = RequestContext(
        email="alice@example.com", is_admin=False, collection_id="know-123"
    )

    with (
        patch("src.tools.ingest_paper.settings", _fake_settings(tmp_path)),
        patch(
            "scripts.download_papers.download_single_paper",
            return_value={
                "status": "downloaded",
                "doi": "10.1038/x",
                "doc": doc,
                "source": "unpaywall",
            },
        ),
        patch(
            "src.tools.ingest_paper.SemanticChunker", return_value=mock_chunker_instance
        ),
        patch(
            "src.tools.ingest_paper.VLLMEmbedder", return_value=mock_embedder_instance
        ),
    ):
        result = tool.execute("10.1038/x", request_context=ctx)

    assert result.success is True
    mock_retriever.insert_paper.assert_called_once()
    assert mock_retriever.insert_paper.call_args.kwargs["collection_id"] == "know-123"


@pytest.mark.unit
def test_execute_non_admin_with_no_collection_attached_fails_clearly():
    from src.tools.ingest_paper import IngestPaperTool

    tool = IngestPaperTool(retriever=MagicMock())
    ctx = RequestContext(
        email="mallory@example.com", is_admin=False, collection_id=None
    )

    result = tool.execute("10.1038/x", request_context=ctx)

    assert result.success is False
    assert "collection" in result.error.lower()
    tool.retriever.insert_paper.assert_not_called()


@pytest.mark.unit
def test_execute_admin_with_no_collection_attached_defaults_to_global(tmp_path):
    """Preserves today's existing single-corpus behavior for casual admin
    use -- no collection attached and no also_global still works for an
    admin, going straight to the global corpus."""
    from src.tools.ingest_paper import IngestPaperTool

    doc = _make_doc(tmp_path)
    chunk = Chunk(id="c1", text="text", source_file=str(doc.file_path))

    mock_retriever = MagicMock()
    mock_retriever.insert_paper.return_value = {
        "chunks_inserted": 1,
        "edges_inserted": 0,
    }
    tool = IngestPaperTool(retriever=mock_retriever)

    mock_chunker_instance = MagicMock()
    mock_chunker_instance.chunk_markdown_file.return_value = [chunk]
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.check_health.return_value = True
    mock_embedder_instance.embed_chunks.side_effect = lambda chunks, **kw: (
        [setattr(c, "embedding", [0.1]) or c for c in chunks]
    )

    ctx = RequestContext(email="admin@example.com", is_admin=True, collection_id=None)

    with (
        patch("src.tools.ingest_paper.settings", _fake_settings(tmp_path)),
        patch(
            "scripts.download_papers.download_single_paper",
            return_value={
                "status": "downloaded",
                "doi": "10.1038/x",
                "doc": doc,
                "source": "unpaywall",
            },
        ),
        patch(
            "src.tools.ingest_paper.SemanticChunker", return_value=mock_chunker_instance
        ),
        patch(
            "src.tools.ingest_paper.VLLMEmbedder", return_value=mock_embedder_instance
        ),
    ):
        result = tool.execute("10.1038/x", request_context=ctx)

    assert result.success is True
    mock_retriever.insert_paper.assert_called_once()
    assert mock_retriever.insert_paper.call_args.kwargs["collection_id"] is None


@pytest.mark.unit
def test_execute_also_global_by_non_admin_fails_with_exact_error(tmp_path):
    from src.tools.ingest_paper import IngestPaperTool

    tool = IngestPaperTool(retriever=MagicMock())
    ctx = RequestContext(
        email="mallory@example.com", is_admin=False, collection_id="know-123"
    )

    result = tool.execute("10.1038/x", also_global=True, request_context=ctx)

    assert result.success is False
    assert result.error == "Only admin can add to the global corpus."
    tool.retriever.insert_paper.assert_not_called()


@pytest.mark.unit
def test_execute_admin_also_global_writes_twice_with_derived_global_chunk_ids(tmp_path):
    """The admin also_global path must call insert_paper() twice: once
    for the attached collection with the chunk's ORIGINAL id, once for
    'global' with a DERIVED id (f'{id}:global') -- chunks.id/
    chunk_embeddings.id are both PRIMARY KEY columns, so reusing the
    exact same id for both writes would make the second one a silent
    ON CONFLICT DO NOTHING no-op (see this plan's design note)."""
    from src.tools.ingest_paper import IngestPaperTool

    doc = _make_doc(tmp_path)
    chunk = Chunk(id="c1", text="text", source_file=str(doc.file_path))

    mock_retriever = MagicMock()
    mock_retriever.insert_paper.return_value = {
        "chunks_inserted": 1,
        "edges_inserted": 0,
    }
    tool = IngestPaperTool(retriever=mock_retriever)

    mock_chunker_instance = MagicMock()
    mock_chunker_instance.chunk_markdown_file.return_value = [chunk]
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.check_health.return_value = True
    mock_embedder_instance.embed_chunks.side_effect = lambda chunks, **kw: (
        [setattr(c, "embedding", [0.1]) or c for c in chunks]
    )

    ctx = RequestContext(
        email="admin@example.com", is_admin=True, collection_id="know-123"
    )

    with (
        patch("src.tools.ingest_paper.settings", _fake_settings(tmp_path)),
        patch(
            "scripts.download_papers.download_single_paper",
            return_value={
                "status": "downloaded",
                "doi": "10.1038/x",
                "doc": doc,
                "source": "unpaywall",
            },
        ),
        patch(
            "src.tools.ingest_paper.SemanticChunker", return_value=mock_chunker_instance
        ),
        patch(
            "src.tools.ingest_paper.VLLMEmbedder", return_value=mock_embedder_instance
        ),
    ):
        result = tool.execute("10.1038/x", also_global=True, request_context=ctx)

    assert result.success is True
    assert mock_retriever.insert_paper.call_count == 2

    first_call, second_call = mock_retriever.insert_paper.call_args_list
    assert first_call.kwargs["collection_id"] == "know-123"
    assert [c.id for c in first_call.kwargs["chunks"]] == ["c1"]

    assert second_call.kwargs["collection_id"] == "global"
    assert [c.id for c in second_call.kwargs["chunks"]] == ["c1:global"]
    # Content reused, not re-chunked/re-embedded -- same text/embedding.
    assert second_call.kwargs["chunks"][0].text == "text"
    assert second_call.kwargs["chunks"][0].embedding == [0.1]


@pytest.mark.unit
def test_model_supplied_collection_id_kwarg_is_never_read_for_tenancy(tmp_path):
    """Even if the model's JSON tool-call arguments include a
    collection_id key (ingest_paper's schema doesn't declare one, but a
    model can still hallucinate extra arguments), it must land in
    **kwargs and be ignored -- only request_context.collection_id decides
    where the paper is written."""
    from src.tools.ingest_paper import IngestPaperTool

    doc = _make_doc(tmp_path)
    chunk = Chunk(id="c1", text="text", source_file=str(doc.file_path))

    mock_retriever = MagicMock()
    mock_retriever.insert_paper.return_value = {
        "chunks_inserted": 1,
        "edges_inserted": 0,
    }
    tool = IngestPaperTool(retriever=mock_retriever)

    mock_chunker_instance = MagicMock()
    mock_chunker_instance.chunk_markdown_file.return_value = [chunk]
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.check_health.return_value = True
    mock_embedder_instance.embed_chunks.side_effect = lambda chunks, **kw: (
        [setattr(c, "embedding", [0.1]) or c for c in chunks]
    )

    ctx = RequestContext(
        email="alice@example.com", is_admin=False, collection_id="know-123"
    )

    with (
        patch("src.tools.ingest_paper.settings", _fake_settings(tmp_path)),
        patch(
            "scripts.download_papers.download_single_paper",
            return_value={
                "status": "downloaded",
                "doi": "10.1038/x",
                "doc": doc,
                "source": "unpaywall",
            },
        ),
        patch(
            "src.tools.ingest_paper.SemanticChunker", return_value=mock_chunker_instance
        ),
        patch(
            "src.tools.ingest_paper.VLLMEmbedder", return_value=mock_embedder_instance
        ),
    ):
        result = tool.execute(
            "10.1038/x", request_context=ctx, collection_id="attacker-chosen-collection"
        )

    assert result.success is True
    assert mock_retriever.insert_paper.call_args.kwargs["collection_id"] == "know-123"


@pytest.mark.unit
def test_execute_also_global_partial_failure_reports_success_with_note(tmp_path):
    """If the PRIMARY (attached-collection) insert_paper() call succeeds
    but the SECOND (global-copy) call then raises, the primary collection
    write is already durably committed -- the overall ToolResult must be
    success=True (the paper really is in the corpus and searchable there),
    with a clear note that the global copy specifically failed and why.
    This must not look like a total failure to the caller."""
    from src.tools.ingest_paper import IngestPaperTool

    doc = _make_doc(tmp_path)
    chunk = Chunk(id="c1", text="text", source_file=str(doc.file_path))

    mock_retriever = MagicMock()
    mock_retriever.insert_paper.side_effect = [
        {"chunks_inserted": 1, "edges_inserted": 0},
        RuntimeError("duckdb: database is locked"),
    ]
    tool = IngestPaperTool(retriever=mock_retriever)

    mock_chunker_instance = MagicMock()
    mock_chunker_instance.chunk_markdown_file.return_value = [chunk]
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.check_health.return_value = True
    mock_embedder_instance.embed_chunks.side_effect = lambda chunks, **kw: (
        [setattr(c, "embedding", [0.1]) or c for c in chunks]
    )

    ctx = RequestContext(
        email="admin@example.com", is_admin=True, collection_id="know-123"
    )

    with (
        patch("src.tools.ingest_paper.settings", _fake_settings(tmp_path)),
        patch(
            "scripts.download_papers.download_single_paper",
            return_value={
                "status": "downloaded",
                "doi": "10.1038/x",
                "doc": doc,
                "source": "unpaywall",
            },
        ),
        patch(
            "src.tools.ingest_paper.SemanticChunker", return_value=mock_chunker_instance
        ),
        patch(
            "src.tools.ingest_paper.VLLMEmbedder", return_value=mock_embedder_instance
        ),
    ):
        result = tool.execute("10.1038/x", also_global=True, request_context=ctx)

    # Primary write succeeded -- the paper is genuinely in the corpus, so
    # this must NOT be reported as an overall failure.
    assert result.success is True
    assert mock_retriever.insert_paper.call_count == 2

    # The result must clearly communicate the global-copy failure
    # somewhere a caller/admin would see it.
    assert result.metadata["global_copy_failed"] is True
    assert "duckdb: database is locked" in result.metadata["global_copy_error"]
    assert "global" in result.data.lower()
    assert "fail" in result.data.lower()

    # The primary collection's successful insert is still reflected.
    assert result.metadata["collection_id"] == "know-123"
