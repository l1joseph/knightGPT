"""Storage-agnostic retriever interface for RAG."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..chunking import Chunk
from ..utils import get_logger

logger = get_logger(__name__)

# DOIs always start with the "10." directory indicator followed by a
# registrant code and a slash (https://www.doi.org/doi_handbook/2_Numbering.html).
# HybridRetriever sets Chunk.source_file to the paper's real DOI (e.g.
# "10.1128/mbio.00519-19"); the file-backed retriever sets it to a markdown
# path. Running a DOI through Path(...).stem mangles it badly -- Path
# treats "/" as a separator and "." as an extension marker, so
# Path("10.1128/mbio.00519-19").stem yields just "mbio".
_DOI_PATTERN = re.compile(r"^10\.\d+/")


@dataclass
class RetrievalResult:
    """Result from retrieval."""

    chunks: list[Chunk]
    query_embedding: list[float]
    similarity_scores: list[float]


@dataclass
class Citation:
    """Citation information."""

    chunk_id: str
    source_file: str
    section: Optional[str]
    text_snippet: str
    similarity: float


@dataclass
class RAGResponse:
    """Complete RAG response."""

    answer: str
    citations: list[Citation]
    context_chunks: list[Chunk]


class BaseRetriever(ABC):
    """Storage-agnostic retriever interface consumed by RAGEngine."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        expand_context: bool = True,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query."""
        raise NotImplementedError

    def format_context(
        self,
        chunks: list[Chunk],
        max_tokens: int = 4000,
    ) -> str:
        """Format chunks as context string."""
        context_parts = []
        total_tokens = 0

        for chunk in chunks:
            chunk_tokens = chunk.token_count or len(chunk.text) // 4

            if total_tokens + chunk_tokens > max_tokens:
                break

            if not chunk.source_file:
                source = "Unknown"
            elif _DOI_PATTERN.match(chunk.source_file):
                source = chunk.source_file
            else:
                source = Path(chunk.source_file).stem
            section = chunk.section or "General"

            context_parts.append(
                f"[Source: {source}, Section: {section}]\n{chunk.text}"
            )
            total_tokens += chunk_tokens

        return "\n\n---\n\n".join(context_parts)

    def create_citations(
        self,
        chunks: list[Chunk],
        scores: list[float],
    ) -> list[Citation]:
        """Create citation objects from chunks."""
        citations = []

        min_len = min(len(chunks), len(scores))
        chunks = chunks[:min_len]
        scores = scores[:min_len]

        for chunk, score in zip(chunks, scores):
            if not chunk:
                continue
            try:
                citations.append(
                    Citation(
                        chunk_id=chunk.id or "unknown",
                        source_file=chunk.source_file or "unknown",
                        section=chunk.section,
                        text_snippet=(
                            chunk.text[:200] + "..."
                            if len(chunk.text) > 200
                            else chunk.text if chunk.text else ""
                        ),
                        similarity=float(score) if score is not None else 0.0,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to create citation: {e}")
                continue

        return citations
