"""Unit tests for convert_pdf_to_markdown's marker-pdf-unavailable fallback.

Covers a real regression: convert_pdf_to_markdown() used to re-raise
immediately when marker-pdf's import chain failed (whether truly
uninstalled, or "installed" but broken by a dependency mismatch -- e.g. a
transformers version marker doesn't support, which raises ImportError with
a misleading, unrelated-looking message), bypassing the PyMuPDF4LLM
fallback entirely. Confirmed live: a full-corpus ingestion run hit this
exact path and silently failed every single PDF conversion, 0 of which
ever reached the fallback.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_marker_import_error_falls_back_to_pymupdf(tmp_path):
    from src.ingestion.pdf_to_markdown import convert_pdf_to_markdown

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    output_dir = tmp_path / "out"

    fake_fallback_result = {
        "success": True,
        "output_path": str(output_dir / "paper.md"),
        "metadata": {"source_file": str(pdf_path)},
        "content": "fallback text",
    }

    # Force `from marker.converters.pdf import PdfConverter` to raise
    # ImportError deterministically, regardless of whether marker-pdf (or
    # a broken version of it) actually happens to be installed in
    # whatever environment runs this test.
    with patch.dict(sys.modules, {"marker.converters.pdf": None}):
        with patch(
            "src.ingestion.pdf_to_markdown._fallback_pymupdf",
            return_value=fake_fallback_result,
        ) as mock_fallback:
            result = convert_pdf_to_markdown(pdf_path=pdf_path, output_dir=output_dir)

    mock_fallback.assert_called_once_with(pdf_path, output_dir)
    assert result == fake_fallback_result


@pytest.mark.unit
def test_missing_pdf_file_raises_before_attempting_any_conversion(tmp_path):
    from src.ingestion.pdf_to_markdown import convert_pdf_to_markdown

    with pytest.raises(FileNotFoundError):
        convert_pdf_to_markdown(
            pdf_path=tmp_path / "does-not-exist.pdf", output_dir=tmp_path
        )
