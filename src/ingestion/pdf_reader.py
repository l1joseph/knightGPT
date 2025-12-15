import logging
import json
import os
import re
from typing import List, Dict, Any, Optional

from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from ollama import chat

logger = logging.getLogger(__name__)

# Configurable LLM model via environment variable
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")


def _fix_letter_spaced_text(text: str) -> str:
    """
    Fix letter-spaced text that appears in some PDFs (especially arXiv headers).
    Converts patterns like "a r X i v" to "arXiv", "A B S T R A C T" to "ABSTRACT".
    Also handles newline-separated characters common in arXiv PDF sidebars.
    """
    result = text

    # First, fix newline-separated single characters (common in arXiv sidebars)
    # Pattern: single chars separated by newlines, e.g., "a\nr\nX\ni\nv"
    # Look for sequences of at least 3 single-char lines
    newline_spaced_pattern = re.compile(r'((?:[A-Za-z0-9.:]\n){2,}[A-Za-z0-9.:])')

    def join_newline_chars(match):
        spaced = match.group(1)
        return spaced.replace('\n', '')

    result = newline_spaced_pattern.sub(join_newline_chars, result)

    # Pattern: single letters/digits separated by single spaces
    # Must be at least 3 characters to avoid false positives
    letter_spaced_pattern = re.compile(r'\b((?:[A-Za-z0-9]\s){2,}[A-Za-z0-9])\b')

    def join_letters(match):
        spaced = match.group(1)
        # Join by removing spaces between single characters
        joined = spaced.replace(' ', '')
        return joined

    result = letter_spaced_pattern.sub(join_letters, result)

    # Also fix spaced colons like "1 0 . 4 8 5 5 0" (DOI-like patterns)
    # Pattern for spaced numbers with dots
    spaced_doi_pattern = re.compile(r'(\d(?:\s+[\d.])+\d)')

    def join_doi_numbers(match):
        return match.group(1).replace(' ', '')

    result = spaced_doi_pattern.sub(join_doi_numbers, result)

    return result

def _sanitize_text_for_prompt(text: str) -> str:
    """
    Sanitize text before injecting into LLM prompts to prevent prompt injection.
    Removes or escapes potentially dangerous patterns.
    """
    # Remove common prompt injection patterns
    sanitized = text
    # Remove instruction-like patterns that could manipulate LLM
    dangerous_patterns = [
        r'ignore\s+(previous|above|all|the\s+above)(\s+instructions?)?',
        r'disregard\s+(previous|above|all|the\s+above|everything)',
        r'forget\s+(previous|above|all|everything)',
        r'new\s+instructions?:',
        r'system\s*prompt',
        r'<\s*/?\s*(system|user|assistant)\s*>',
        r'do\s+not\s+follow\s+(previous|above|prior)',
        r'override\s+(previous|all|any)\s+(instructions?|rules?)',
    ]
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '[REMOVED]', sanitized, flags=re.IGNORECASE)

    # Limit length to prevent context overflow
    max_len = 2000
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len] + "..."

    return sanitized


def _extract_json_from_response(content: str) -> Optional[Dict]:
    """
    Safely extract JSON from LLM response using regex and validation.
    Returns None if extraction fails.
    """
    if not content:
        return None

    # Try to find JSON object pattern with regex (handles nested braces properly)
    json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')
    matches = json_pattern.findall(content)

    # Try each match until we find valid JSON
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # Fallback: try parsing the entire content if it looks like JSON
    content_stripped = content.strip()
    if content_stripped.startswith('{') and content_stripped.endswith('}'):
        try:
            parsed = json.loads(content_stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def _get_ollama_response_content(resp) -> str:
    """
    Safely extract content from Ollama response with proper type checking.
    """
    # Try resp.message.content (newer Ollama API)
    if hasattr(resp, 'message') and hasattr(resp.message, 'content'):
        return str(resp.message.content)
    # Try resp.content (older API)
    if hasattr(resp, 'content'):
        return str(resp.content)
    # Try dict-like access
    if isinstance(resp, dict):
        if 'message' in resp and isinstance(resp['message'], dict):
            return str(resp['message'].get('content', ''))
        return str(resp.get('content', ''))
    return ''


class PDFReader:
    """
    PDFReader uses pdfminer.six for metadata and text extraction,
    including DOI detection via regex from the first page of text.
    Falls back to LLM for missing metadata fields, using robust response parsing.
    """

    # More precise DOI regex - must have alphanumeric after the slash
    DOI_REGEX = re.compile(r"10\.\d{4,9}/[A-Z0-9][-._;()/:A-Z0-9]*[A-Z0-9]", re.IGNORECASE)

    # arXiv paper ID pattern (e.g., 2504.02148, 2504.02148v1, arXiv:2504.02148)
    ARXIV_ID_REGEX = re.compile(r"(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)

    def __init__(self):
        pass

    def _decode_pdf_string(self, value: Any) -> Any:
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8', errors='ignore')
            except Exception:
                return value.decode('latin-1', errors='ignore')
        return value

    def extract_metadata(self, pdf_path: str, first_page_text: str) -> Dict[str, Any]:
        """
        Extracts metadata using pdfminer; falls back to llama3 for any missing fields.
        """
        metadata: Dict[str, Any] = {"title": None, "authors": [], "publication_date": None, "doi": None}
        # 1) PDF document info
        with open(pdf_path, 'rb') as fp:
            parser = PDFParser(fp)
            doc = PDFDocument(parser)
            info = doc.info[0] if doc.info else {}

            # Title
            raw_title = info.get('Title')
            metadata['title'] = self._decode_pdf_string(raw_title)

            # Author(s)
            raw_author = info.get('Author')
            author_str = self._decode_pdf_string(raw_author)
            if author_str:
                metadata['authors'] = [a.strip() for a in re.split(r'[;,]', author_str) if a.strip()]

            # Publication Date
            raw_date = info.get('CreationDate') or info.get('ModDate')
            date_str = self._decode_pdf_string(raw_date)
            if date_str and date_str.startswith('D:'):
                parts = date_str[2:10]
                if len(parts) == 8 and parts.isdigit():
                    metadata['publication_date'] = f"{parts[0:4]}-{parts[4:6]}-{parts[6:8]}"

        # 2) DOI detection via regex
        doi_match = self.DOI_REGEX.search(first_page_text)
        if doi_match:
            metadata['doi'] = doi_match.group(0).rstrip('.')
            logger.info(f"Detected DOI via regex: {metadata['doi']}")
        else:
            # Try to detect arXiv paper ID and construct DOI
            arxiv_match = self.ARXIV_ID_REGEX.search(first_page_text)
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)
                # Remove version suffix for DOI (e.g., v1)
                arxiv_id_base = re.sub(r'v\d+$', '', arxiv_id)
                metadata['doi'] = f"10.48550/arXiv.{arxiv_id_base}"
                logger.info(f"Constructed DOI from arXiv ID: {metadata['doi']}")

        # 3) Fallback: LLM for missing fields
        missing = [key for key, val in metadata.items() if not val]
        if missing:
            logger.info(f"Falling back to LLM ({LLM_MODEL}) for metadata fields: {missing}")

            # Sanitize text to prevent prompt injection
            sanitized_text = _sanitize_text_for_prompt(first_page_text)

            # Use structured prompt with clear delimiters
            prompt = (
                "You are a metadata extraction assistant. Extract ONLY the requested fields from the provided document text.\n\n"
                f"REQUIRED FIELDS: {', '.join(missing)}\n\n"
                "INSTRUCTIONS:\n"
                "- Return ONLY a valid JSON object with the requested fields\n"
                "- Use null for fields you cannot find\n"
                "- For 'authors', return a list of strings\n"
                "- For 'publication_date', use YYYY-MM-DD format if possible\n\n"
                "DOCUMENT TEXT (between triple backticks):\n"
                f"```\n{sanitized_text}\n```\n\n"
                "JSON RESPONSE:"
            )

            try:
                resp = chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
                content = _get_ollama_response_content(resp)

                if not content:
                    logger.warning("LLM returned empty response for metadata extraction")
                else:
                    # Safely extract JSON from response
                    llm_meta = _extract_json_from_response(content)

                    if llm_meta is None:
                        logger.warning(f"Could not parse JSON from LLM response: {content[:200]}...")
                    else:
                        # Only update missing fields with non-empty values
                        for key in missing:
                            if key in llm_meta and llm_meta[key]:
                                metadata[key] = llm_meta[key]
                                logger.debug(f"Extracted {key} from LLM: {llm_meta[key]}")

            except ConnectionError as e:
                logger.error(f"Could not connect to Ollama service: {e}")
            except Exception as e:
                logger.error(f"LLM metadata extraction failed: {type(e).__name__}: {e}")

        return metadata

    def read_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Reads a PDF file and returns metadata and page texts using pdfminer.six.
        Splits on form-feed characters to separate pages.
        Applies post-processing to fix letter-spaced text from stylized headers.
        """
        logger.info(f"Extracting text and metadata from PDF: {pdf_path}")
        full_text = extract_text(pdf_path)
        raw_pages = full_text.split("\f")

        # Apply letter-spacing fix to each page
        pages = []
        for p in raw_pages:
            stripped = p.strip()
            if stripped:
                fixed = _fix_letter_spaced_text(stripped)
                pages.append(fixed)

        logger.info(f"Extracted {len(pages)} pages from PDF")

        first_page_text = pages[0] if pages else ''
        metadata = self.extract_metadata(pdf_path, first_page_text)

        return {"metadata": metadata, "pages": pages}

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract metadata (including DOI) and text from PDF into JSON'
    )
    parser.add_argument('pdf_path', type=str, help='Path to PDF file to read')
    parser.add_argument('-o', '--output_json', type=str, help='Path to write output JSON file')
    args = parser.parse_args()

    reader = PDFReader()
    data = reader.read_pdf(args.pdf_path)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote output JSON: {args.output_json}")
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))
