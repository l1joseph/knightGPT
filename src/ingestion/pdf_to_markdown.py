"""PDF to Markdown conversion using marker-pdf for high-quality extraction."""

import json
from pathlib import Path
from typing import Optional

from ..utils import get_logger

logger = get_logger(__name__)


def convert_pdf_to_markdown(
    pdf_path: Path,
    output_dir: Path,
    force_ocr: bool = False,
    use_llm: bool = False,
    output_format: str = "markdown",
) -> dict:
    """
    Convert a PDF file to Markdown using marker-pdf.

    marker-pdf provides high-quality PDF extraction with:
    - Table detection and formatting
    - Math equation extraction (LaTeX)
    - Code block detection
    - Multi-language support
    - Figure/image extraction

    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory for output files
        force_ocr: Force OCR on all pages (useful for scanned PDFs)
        use_llm: Use LLM enhancement for better table/equation handling
        output_format: Output format ('markdown', 'json', 'html', 'chunks')

    Returns:
        Dictionary with conversion results and metadata
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.config.parser import ConfigParser
        from marker.output import save_output
    except ImportError:
        logger.error("marker-pdf not installed. Install with: pip install marker-pdf")
        raise ImportError("marker-pdf is required for PDF conversion")

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info(f"Converting PDF: {pdf_path}")

    # Configure marker
    config = {
        "force_ocr": force_ocr,
        "output_format": output_format,
    }

    if use_llm:
        config["use_llm"] = True
        logger.info("LLM enhancement enabled for better parsing")

    # Create converter and process
    try:
        converter = PdfConverter()
        result = converter.convert(str(pdf_path))

        # Generate output filename
        stem = pdf_path.stem
        if output_format == "markdown":
            output_path = output_dir / f"{stem}.md"
            content = result.markdown
        elif output_format == "json":
            output_path = output_dir / f"{stem}.json"
            content = json.dumps(result.model_dump(), indent=2)
        elif output_format == "html":
            output_path = output_dir / f"{stem}.html"
            content = result.html
        else:
            output_path = output_dir / f"{stem}.md"
            content = result.markdown

        # Write output
        output_path.write_text(content, encoding="utf-8")
        logger.info(f"Saved output to: {output_path}")

        # Save metadata
        metadata = {
            "source_file": str(pdf_path),
            "output_file": str(output_path),
            "pages": result.metadata.get("pages", 0) if hasattr(result, "metadata") else 0,
            "output_format": output_format,
            "force_ocr": force_ocr,
            "use_llm": use_llm,
        }

        metadata_path = output_dir / f"{stem}_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "success": True,
            "output_path": str(output_path),
            "metadata": metadata,
            "content": content,
        }

    except Exception as e:
        logger.error(f"Marker conversion failed: {e}")
        # Fallback to PyMuPDF4LLM
        return _fallback_pymupdf(pdf_path, output_dir)


def _fallback_pymupdf(pdf_path: Path, output_dir: Path) -> dict:
    """
    Fallback PDF conversion using PyMuPDF4LLM.

    Args:
        pdf_path: Path to input PDF
        output_dir: Output directory

    Returns:
        Conversion results dictionary
    """
    logger.info("Falling back to PyMuPDF4LLM for conversion")

    try:
        import pymupdf4llm
    except ImportError:
        logger.error("pymupdf4llm not installed. Install with: pip install pymupdf4llm")
        raise ImportError("pymupdf4llm is required as fallback")

    try:
        # Convert to markdown
        md_text = pymupdf4llm.to_markdown(str(pdf_path))

        # Save output
        output_path = output_dir / f"{pdf_path.stem}.md"
        output_path.write_text(md_text, encoding="utf-8")

        metadata = {
            "source_file": str(pdf_path),
            "output_file": str(output_path),
            "converter": "pymupdf4llm",
        }

        return {
            "success": True,
            "output_path": str(output_path),
            "metadata": metadata,
            "content": md_text,
        }

    except Exception as e:
        logger.error(f"PyMuPDF4LLM conversion failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def batch_convert_pdfs(
    input_dir: Path,
    output_dir: Path,
    force_ocr: bool = False,
    use_llm: bool = False,
    recursive: bool = True,
) -> list[dict]:
    """
    Batch convert all PDFs in a directory to Markdown.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Output directory for markdown files
        force_ocr: Force OCR on all pages
        use_llm: Use LLM enhancement
        recursive: Search subdirectories

    Returns:
        List of conversion results
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if recursive:
        pdf_files = list(input_dir.rglob("*.pdf"))
    else:
        pdf_files = list(input_dir.glob("*.pdf"))

    logger.info(f"Found {len(pdf_files)} PDF files to convert")

    results = []
    for pdf_path in pdf_files:
        # Preserve directory structure in output
        rel_path = pdf_path.relative_to(input_dir)
        file_output_dir = output_dir / rel_path.parent
        file_output_dir.mkdir(parents=True, exist_ok=True)

        result = convert_pdf_to_markdown(
            pdf_path=pdf_path,
            output_dir=file_output_dir,
            force_ocr=force_ocr,
            use_llm=use_llm,
        )
        results.append(result)

    success_count = sum(1 for r in results if r.get("success"))
    logger.info(f"Conversion complete: {success_count}/{len(results)} successful")

    return results


def extract_metadata_from_pdf(pdf_path: Path) -> dict:
    """
    Extract metadata from a PDF file.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dictionary with PDF metadata (title, authors, DOI, etc.)
    """
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf

    pdf_path = Path(pdf_path)
    metadata = {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
    }

    try:
        doc = pymupdf.open(str(pdf_path))
        pdf_metadata = doc.metadata

        metadata.update({
            "title": pdf_metadata.get("title", ""),
            "author": pdf_metadata.get("author", ""),
            "subject": pdf_metadata.get("subject", ""),
            "keywords": pdf_metadata.get("keywords", ""),
            "creator": pdf_metadata.get("creator", ""),
            "producer": pdf_metadata.get("producer", ""),
            "creation_date": pdf_metadata.get("creationDate", ""),
            "modification_date": pdf_metadata.get("modDate", ""),
            "page_count": doc.page_count,
        })

        # Try to extract DOI from first page
        first_page = doc[0]
        text = first_page.get_text()
        doi = _extract_doi(text)
        if doi:
            metadata["doi"] = doi

        doc.close()

    except Exception as e:
        logger.warning(f"Could not extract metadata from {pdf_path}: {e}")

    return metadata


def _extract_doi(text: str) -> Optional[str]:
    """Extract DOI from text using regex."""
    import re

    # DOI patterns
    patterns = [
        r'10\.\d{4,}/[^\s]+',
        r'doi:\s*(10\.\d{4,}/[^\s]+)',
        r'DOI:\s*(10\.\d{4,}/[^\s]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            doi = match.group(1) if match.lastindex else match.group(0)
            # Clean up DOI
            doi = doi.rstrip('.')
            return doi

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown")
    parser.add_argument("input", type=Path, help="Input PDF file or directory")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM enhancement")

    args = parser.parse_args()

    if args.input.is_dir():
        batch_convert_pdfs(
            input_dir=args.input,
            output_dir=args.output,
            force_ocr=args.force_ocr,
            use_llm=args.use_llm,
        )
    else:
        convert_pdf_to_markdown(
            pdf_path=args.input,
            output_dir=args.output,
            force_ocr=args.force_ocr,
            use_llm=args.use_llm,
        )
