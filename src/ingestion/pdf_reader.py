"""PDF metadata extraction and page processing."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class PDFMetadata:
    """Structured PDF metadata."""

    file_name: str
    file_path: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    doi: Optional[str] = None
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    abstract: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    page_count: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "file_name": self.file_name,
            "file_path": self.file_path,
            "title": self.title,
            "authors": self.authors,
            "doi": self.doi,
            "publication_date": self.publication_date,
            "journal": self.journal,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "page_count": self.page_count,
        }


@dataclass
class ProcessedPage:
    """Processed page content."""

    page_number: int
    text: str
    tables: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)


@dataclass
class ProcessedDocument:
    """Complete processed document."""

    metadata: PDFMetadata
    pages: list[ProcessedPage]
    full_text: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "pages": [
                {
                    "page_number": p.page_number,
                    "text": p.text,
                    "tables": p.tables,
                    "figures": p.figures,
                }
                for p in self.pages
            ],
            "full_text": self.full_text,
        }


def process_pdf(
    pdf_path: Path,
    extract_tables: bool = True,
    extract_images: bool = False,
) -> ProcessedDocument:
    """
    Process a PDF file and extract metadata and content.

    Args:
        pdf_path: Path to PDF file
        extract_tables: Whether to extract tables
        extract_images: Whether to extract images

    Returns:
        ProcessedDocument with metadata and pages
    """
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf

    pdf_path = Path(pdf_path)
    logger.info(f"Processing PDF: {pdf_path}")

    doc = pymupdf.open(str(pdf_path))

    # Extract metadata
    metadata = _extract_metadata(doc, pdf_path)

    # Process pages
    pages = []
    full_text_parts = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        page_text = page.get_text("text")
        full_text_parts.append(page_text)

        tables = []
        if extract_tables:
            tables = _extract_tables_from_page(page)

        figures = []
        if extract_images:
            figures = _extract_images_from_page(page, page_num)

        processed_page = ProcessedPage(
            page_number=page_num + 1,
            text=page_text,
            tables=tables,
            figures=figures,
        )
        pages.append(processed_page)

    doc.close()

    # Extract abstract from first pages
    first_pages_text = "\n".join(full_text_parts[:3])
    metadata.abstract = _extract_abstract(first_pages_text)

    return ProcessedDocument(
        metadata=metadata,
        pages=pages,
        full_text="\n\n".join(full_text_parts),
    )


def _extract_metadata(doc, pdf_path: Path) -> PDFMetadata:
    """Extract metadata from PyMuPDF document."""
    pdf_metadata = doc.metadata

    # Parse authors
    author_str = pdf_metadata.get("author", "")
    authors = _parse_authors(author_str)

    # Extract DOI from document
    doi = None
    if doc.page_count > 0:
        first_page_text = doc[0].get_text()
        doi = _extract_doi(first_page_text)

    # Parse date
    creation_date = pdf_metadata.get("creationDate", "")
    publication_date = _parse_date(creation_date)

    return PDFMetadata(
        file_name=pdf_path.name,
        file_path=str(pdf_path),
        title=pdf_metadata.get("title", pdf_path.stem),
        authors=authors,
        doi=doi,
        publication_date=publication_date,
        keywords=_parse_keywords(pdf_metadata.get("keywords", "")),
        page_count=doc.page_count,
    )


def _parse_authors(author_str: str) -> list[str]:
    """Parse author string into list of names."""
    if not author_str:
        return []

    # Common separators
    separators = [";", ",", " and ", " & "]
    authors = [author_str]

    for sep in separators:
        new_authors = []
        for a in authors:
            new_authors.extend(a.split(sep))
        authors = new_authors

    # Clean up
    authors = [a.strip() for a in authors if a.strip()]
    return authors


def _parse_keywords(keywords_str: str) -> list[str]:
    """Parse keywords string into list."""
    if not keywords_str:
        return []

    keywords = keywords_str.split(",")
    return [k.strip() for k in keywords if k.strip()]


def _parse_date(date_str: str) -> Optional[str]:
    """Parse PDF date format to ISO format."""
    if not date_str:
        return None

    # PDF date format: D:YYYYMMDDHHmmSS
    match = re.match(r"D:(\d{4})(\d{2})?(\d{2})?", date_str)
    if match:
        year = match.group(1)
        month = match.group(2) or "01"
        day = match.group(3) or "01"
        return f"{year}-{month}-{day}"

    return None


def _extract_doi(text: str) -> Optional[str]:
    """Extract DOI from text."""
    patterns = [
        r"10\.\d{4,}/[^\s]+",
        r"doi:\s*(10\.\d{4,}/[^\s]+)",
        r"DOI:\s*(10\.\d{4,}/[^\s]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            doi = match.group(1) if match.lastindex else match.group(0)
            doi = doi.rstrip(".,;")
            return doi

    return None


def _extract_abstract(text: str) -> Optional[str]:
    """Extract abstract from document text."""
    # Common abstract patterns
    patterns = [
        r"Abstract[:\s]*(.+?)(?:Introduction|Keywords|1\.|Background)",
        r"ABSTRACT[:\s]*(.+?)(?:INTRODUCTION|KEYWORDS|1\.|BACKGROUND)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = match.group(1).strip()
            # Clean up whitespace
            abstract = " ".join(abstract.split())
            if len(abstract) > 100:  # Reasonable abstract length
                return abstract[:2000]  # Limit length

    return None


def _extract_tables_from_page(page) -> list[str]:
    """Extract tables from a page."""
    tables = []
    try:
        # PyMuPDF table extraction
        tabs = page.find_tables()
        for tab in tabs:
            df = tab.to_pandas()
            tables.append(df.to_markdown())
    except Exception as e:
        logger.debug(f"Table extraction failed: {e}")

    return tables


def _extract_images_from_page(page, page_num: int) -> list[str]:
    """Extract image references from a page."""
    images = []
    try:
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            images.append(f"Image {page_num + 1}-{img_index + 1}")
    except Exception as e:
        logger.debug(f"Image extraction failed: {e}")

    return images


def save_processed_document(
    doc: ProcessedDocument,
    output_path: Path,
) -> None:
    """Save processed document to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Saved processed document to: {output_path}")


def load_processed_document(input_path: Path) -> ProcessedDocument:
    """Load processed document from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = PDFMetadata(**data["metadata"])
    pages = [ProcessedPage(**p) for p in data["pages"]]

    return ProcessedDocument(
        metadata=metadata,
        pages=pages,
        full_text=data.get("full_text", ""),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process PDF and extract metadata")
    parser.add_argument("pdf_path", type=Path, help="Input PDF file")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file")

    args = parser.parse_args()

    doc = process_pdf(args.pdf_path)

    if args.output:
        save_processed_document(doc, args.output)
    else:
        print(json.dumps(doc.to_dict(), indent=2))
