"""Unit tests for the streaming full-corpus ingestion script."""

import pytest
from unittest.mock import MagicMock


@pytest.mark.unit
def test_build_full_doi_list_dedupes_across_files(tmp_path):
    from scripts.stream_corpus_ingest import build_full_doi_list

    list_a = tmp_path / "a.txt"
    list_a.write_text("10.1/a\n10.1/b\n")
    list_b = tmp_path / "b.txt"
    list_b.write_text("10.1/b\n10.1/c\n")

    result = build_full_doi_list([list_a, list_b])

    # First-seen order preserved, duplicate across files collapsed to one.
    assert result == ["10.1/a", "10.1/b", "10.1/c"]


@pytest.mark.unit
def test_build_full_doi_list_empty_input():
    from scripts.stream_corpus_ingest import build_full_doi_list

    assert build_full_doi_list([]) == []


@pytest.mark.unit
def test_clear_stale_downloads_deletes_leftover_pdf_for_remaining_doi(tmp_path):
    """A DOI that isn't in Postgres yet but has a leftover PDF (from a
    prior run that failed after downloading, e.g. a conversion failure)
    must have that file cleared -- otherwise download_papers()'s own
    file-existence "already downloaded" check skips it forever, even
    though it was never actually ingested."""
    from scripts.stream_corpus_ingest import clear_stale_downloads

    stale_pdf = tmp_path / "10-1_x.pdf"
    stale_pdf.write_bytes(b"leftover from a failed run")

    deleted = clear_stale_downloads(["10.1/x"], tmp_path)

    assert deleted == 1
    assert not stale_pdf.exists()


@pytest.mark.unit
def test_clear_stale_downloads_leaves_unrelated_files_alone(tmp_path):
    from scripts.stream_corpus_ingest import clear_stale_downloads

    unrelated = tmp_path / "10-1_other-paper.pdf"
    unrelated.write_bytes(b"a different DOI's file")

    deleted = clear_stale_downloads(["10.1/x"], tmp_path)

    assert deleted == 0
    assert unrelated.exists()


@pytest.mark.unit
def test_clear_stale_downloads_handles_missing_directory(tmp_path):
    from scripts.stream_corpus_ingest import clear_stale_downloads

    assert clear_stale_downloads(["10.1/x"], tmp_path / "does-not-exist") == 0


@pytest.mark.unit
def test_process_one_paper_download_failed_deletes_partial_pdf(tmp_path):
    from scripts.stream_corpus_ingest import process_one_paper

    pdf_path = tmp_path / "10-1_x.pdf"
    pdf_path.write_bytes(b"partial")

    result = process_one_paper(
        doi="10.1/x",
        doc=None,
        pdf_path=pdf_path,
        chunker=MagicMock(),
        embedder=MagicMock(),
        store=MagicMock(),
    )

    assert result == {"status": "download_failed"}
    assert not pdf_path.exists()


@pytest.mark.unit
def test_process_one_paper_download_failed_no_pdf_present(tmp_path):
    from scripts.stream_corpus_ingest import process_one_paper

    pdf_path = tmp_path / "does-not-exist.pdf"

    result = process_one_paper(
        doi="10.1/x",
        doc=None,
        pdf_path=pdf_path,
        chunker=MagicMock(),
        embedder=MagicMock(),
        store=MagicMock(),
    )

    assert result == {"status": "download_failed"}


@pytest.mark.unit
def test_process_one_paper_no_chunks_keeps_files(tmp_path):
    from scripts.stream_corpus_ingest import process_one_paper
    from src.ingestion.web_scraper import ScrapedDocument

    md_path = tmp_path / "paper.md"
    md_path.write_text("content")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"pdf")
    doc = ScrapedDocument(
        url="http://x", title="t", content_type="application/pdf", file_path=md_path
    )

    chunker = MagicMock()
    chunker.chunk_markdown_file.return_value = []

    result = process_one_paper(
        doi="10.1/x",
        doc=doc,
        pdf_path=pdf_path,
        chunker=chunker,
        embedder=MagicMock(),
        store=MagicMock(),
    )

    assert result == {"status": "chunk_failed"}
    # Files are NOT deleted -- only a confirmed insert triggers cleanup.
    assert md_path.exists()
    assert pdf_path.exists()


@pytest.mark.unit
def test_process_one_paper_embedding_failure_keeps_files(tmp_path):
    from scripts.stream_corpus_ingest import process_one_paper
    from src.chunking import Chunk
    from src.ingestion.web_scraper import ScrapedDocument

    md_path = tmp_path / "paper.md"
    md_path.write_text("content")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"pdf")
    doc = ScrapedDocument(
        url="http://x", title="t", content_type="application/pdf", file_path=md_path
    )

    unembedded = [Chunk(id="c1", text="a", source_file=str(md_path), embedding=None)]
    chunker = MagicMock()
    chunker.chunk_markdown_file.return_value = unembedded
    embedder = MagicMock()
    embedder.embed_chunks.return_value = unembedded  # embedding stays None

    result = process_one_paper(
        doi="10.1/x",
        doc=doc,
        pdf_path=pdf_path,
        chunker=chunker,
        embedder=embedder,
        store=MagicMock(),
    )

    assert result == {"status": "process_failed"}
    assert md_path.exists()
    assert pdf_path.exists()


@pytest.mark.unit
def test_process_one_paper_success_deletes_files_and_reports_counts(tmp_path, monkeypatch):
    from scripts.stream_corpus_ingest import process_one_paper
    from src.chunking import Chunk
    from src.ingestion.web_scraper import ScrapedDocument

    md_path = tmp_path / "paper.md"
    md_path.write_text("content")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"pdf")
    doc = ScrapedDocument(
        url="http://x",
        title="t",
        content_type="application/pdf",
        file_path=md_path,
        metadata={"title": "Real Paper"},
    )

    embedded = [Chunk(id="c1", text="a", source_file=str(md_path), embedding=[0.1])]
    chunker = MagicMock()
    chunker.chunk_markdown_file.return_value = embedded
    embedder = MagicMock()
    embedder.embed_chunks.return_value = embedded

    monkeypatch.setattr(
        "scripts.stream_corpus_ingest._insert_batch",
        lambda chunks, papers, store: _fake_insert_stats(),
    )

    result = process_one_paper(
        doi="10.1/x",
        doc=doc,
        pdf_path=pdf_path,
        chunker=chunker,
        embedder=embedder,
        store=MagicMock(),
    )

    assert result == {
        "status": "ingested",
        "chunks_inserted": 1,
        "edges_inserted": 2,
    }
    assert not md_path.exists()
    assert not pdf_path.exists()


async def _fake_insert_stats():
    return {"chunks_inserted": 1, "edges_inserted": 2}
