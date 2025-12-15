import logging
import json
import os
from typing import Optional, Dict
import pdfplumber
import re

logger = logging.getLogger(__name__)

class PDFMetadataExtractor:
    """
    Extracts metadata (title, authors, publication date) directly from PDF metadata fields.
    """

    def __init__(self):
        pass

    def extract_metadata(self, pdf_path: str) -> Optional[Dict]:
        """
        Reads PDF metadata using pdfplumber and returns a dict with
        'title', 'authors', and 'publication_date'.
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF not found: {pdf_path}")
            return None
        try:
            with pdfplumber.open(pdf_path) as pdf:
                meta = pdf.metadata or {}
        except Exception as e:
            logger.error(f"Error opening PDF {pdf_path}: {e}")
            return None

        title = meta.get('Title')
        authors_raw = meta.get('Author')
        if authors_raw:
            # split on common delimiters
            authors = [a.strip() for a in re.split(r'[;,]', authors_raw) if a.strip()]
        else:
            authors = []

        # Parse creation date 'D:YYYYMMDDHHmmSS'
        creation = meta.get('CreationDate')
        pub_date = None
        if creation and creation.startswith('D:'):
            # Extract YYYYMMDD
            date_str = creation[2:10]
            pub_date = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"

        metadata = {
            'title': title,
            'authors': authors,
            'publication_date': pub_date
        }
        logger.info(f"Extracted metadata from PDF: {metadata}")
        return metadata

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract metadata from PDF file')
    parser.add_argument('--pdf', type=str, required=True, help='Path to PDF file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON metadata file')
    args = parser.parse_args()

    extractor = PDFMetadataExtractor()
    meta = extractor.extract_metadata(args.pdf)
    if meta:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"Metadata saved to {args.output}")
    else:
        print("Failed to extract metadata.")
