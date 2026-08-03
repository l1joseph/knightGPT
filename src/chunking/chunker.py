import uuid
import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

# Try to load spaCy, fall back to simple sentence splitting if unavailable
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
        nlp.add_pipe("sentencizer")
        SPACY_AVAILABLE = True
    except OSError:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        SPACY_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"spaCy not available, using simple sentence splitting: {e}")
    nlp = None
    SPACY_AVAILABLE = False


def _simple_sentence_split(text: str) -> List[str]:
    """Simple sentence splitting fallback when spaCy is not available."""
    # Split on common sentence-ending punctuation followed by space and capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

class Chunker:
    """
    Chunker splits page texts into paragraph-sized chunks suitable for RAG nodes.
    - Splits on blank lines for paragraphs
    - Enforces a max token count per chunk
    - Splits oversized paragraphs on sentence boundaries
    - Filters out chunks below minimum token threshold
    """

    def __init__(self, max_tokens: int = 500, min_tokens: int = 10):
        """
        :param max_tokens: Approximate maximum number of tokens (words) per chunk
        :param min_tokens: Minimum number of tokens required for a chunk (filters garbage)
        """
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def _token_count(self, text: str) -> int:
        """Approximate token count by splitting on whitespace"""
        return len(text.split())

    def chunk_pages(self, pages: List[str]) -> List[Dict]:
        """
        Splits each page's text into chunks with metadata.

        :param pages: List of page text strings
        :return: List of chunk dicts: {
            "node_id": str,
            "page": int,
            "paragraph_index": int,
            "chunk_index": int,
            "text": str
        }
        """
        all_chunks: List[Dict] = []
        for page_num, page_text in enumerate(pages, start=1):
            # Split into paragraphs by two or more newlines
            paras = [p.strip() for p in page_text.split("\n\n") if p.strip()]
            logger.info(f"Page {page_num}: {len(paras)} paragraphs detected")
            for para_idx, para in enumerate(paras, start=1):
                tokens = self._token_count(para)
                if tokens <= self.max_tokens:
                    # fits in one chunk
                    chunk = {
                        "node_id": str(uuid.uuid4()),
                        "page": page_num,
                        "paragraph_index": para_idx,
                        "chunk_index": 1,
                        "text": para
                    }
                    all_chunks.append(chunk)
                else:
                    # split paragraph into sentences and form sub-chunks
                    if SPACY_AVAILABLE and nlp is not None:
                        doc = nlp(para)
                        sentences = [sent.text.strip() for sent in doc.sents]
                    else:
                        sentences = _simple_sentence_split(para)
                    logger.debug(f"Paragraph {para_idx} on page {page_num} has {len(sentences)} sentences")
                    chunk_text = ""
                    chunk_i = 1
                    for sent in sentences:
                        if self._token_count(chunk_text + " " + sent) <= self.max_tokens:
                            chunk_text = (chunk_text + " " + sent).strip()
                        else:
                            # emit previous chunk
                            all_chunks.append({
                                "node_id": str(uuid.uuid4()),
                                "page": page_num,
                                "paragraph_index": para_idx,
                                "chunk_index": chunk_i,
                                "text": chunk_text
                            })
                            chunk_i += 1
                            chunk_text = sent
                    # emit final chunk_text
                    if chunk_text:
                        all_chunks.append({
                            "node_id": str(uuid.uuid4()),
                            "page": page_num,
                            "paragraph_index": para_idx,
                            "chunk_index": chunk_i,
                            "text": chunk_text
                        })
        # Filter out chunks below minimum token threshold
        filtered_chunks = [c for c in all_chunks if self._token_count(c['text']) >= self.min_tokens]
        filtered_count = len(all_chunks) - len(filtered_chunks)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} chunks below {self.min_tokens} tokens")
        logger.info(f"Generated {len(filtered_chunks)} total chunks (from {len(all_chunks)} raw)")
        return filtered_chunks

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Chunk PDF page texts into paragraph-sized nodes")
    parser.add_argument("--pages", type=str, required=True,
                        help="Path to JSON file containing list of page texts")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output JSON chunks file")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Maximum tokens per chunk")
    parser.add_argument("--min_tokens", type=int, default=10,
                        help="Minimum tokens per chunk (filters garbage)")
    args = parser.parse_args()

    # Load pages
    with open(args.pages, 'r') as f:
        data = json.load(f)

    # Handle both formats: list of pages or dict with "pages" key
    if isinstance(data, dict) and 'pages' in data:
        pages = data['pages']
    elif isinstance(data, list):
        pages = data
    else:
        raise ValueError(f"Invalid pages format: expected list or dict with 'pages' key")

    chunker = Chunker(max_tokens=args.max_tokens, min_tokens=args.min_tokens)
    chunks = chunker.chunk_pages(pages)

    # Save to output
    with open(args.output, 'w') as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {args.output}")
