"""Semantic text chunking for RAG pipeline."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from ..utils import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class Chunk:
    """A text chunk with metadata."""

    id: str
    text: str
    source_file: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    token_count: int = 0
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "section": self.section,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Create from dictionary."""
        return cls(**data)


class SemanticChunker:
    """
    Semantic paragraph-level chunker.
    
    Splits documents into meaningful chunks based on:
    - Paragraph boundaries
    - Section headers
    - Token limits
    - Semantic coherence
    """

    def __init__(
        self,
        max_tokens: int = 500,
        overlap_tokens: int = 50,
        min_chunk_size: int = 100,
        tokenizer: str = "cl100k_base",
    ):
        """
        Initialize chunker.
        
        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Token overlap between chunks
            min_chunk_size: Minimum chunk size in characters
            tokenizer: Tiktoken tokenizer name
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_size = min_chunk_size
        
        # Initialize tokenizer
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding(tokenizer)
        except ImportError:
            logger.warning("tiktoken not available, using approximate token counting")
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Approximate: ~4 characters per token
        return len(text) // 4

    def chunk_text(
        self,
        text: str,
        source_file: str,
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Chunk text into semantic paragraphs.
        
        Args:
            text: Input text to chunk
            source_file: Source file path
            metadata: Additional metadata
            
        Returns:
            List of Chunk objects
        """
        metadata = metadata or {}
        chunks = []
        
        # Split into paragraphs
        paragraphs = self._split_paragraphs(text)
        
        # Track current section
        current_section = None
        current_chunk_text = ""
        current_chunk_tokens = 0
        
        for para in paragraphs:
            para_text = para.strip()
            if not para_text:
                continue
            
            # Check for section header
            if self._is_section_header(para_text):
                # Save current chunk
                if current_chunk_text:
                    chunks.append(self._create_chunk(
                        text=current_chunk_text,
                        source_file=source_file,
                        section=current_section,
                        metadata=metadata,
                    ))
                    current_chunk_text = ""
                    current_chunk_tokens = 0
                
                current_section = para_text
                continue
            
            para_tokens = self.count_tokens(para_text)
            
            # Check if paragraph fits in current chunk
            if current_chunk_tokens + para_tokens <= self.max_tokens:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para_text
                else:
                    current_chunk_text = para_text
                current_chunk_tokens += para_tokens
            else:
                # Save current chunk and start new one
                if current_chunk_text:
                    chunks.append(self._create_chunk(
                        text=current_chunk_text,
                        source_file=source_file,
                        section=current_section,
                        metadata=metadata,
                    ))
                
                # Handle oversized paragraphs
                if para_tokens > self.max_tokens:
                    sub_chunks = self._split_large_paragraph(
                        para_text, source_file, current_section, metadata
                    )
                    chunks.extend(sub_chunks)
                    current_chunk_text = ""
                    current_chunk_tokens = 0
                else:
                    current_chunk_text = para_text
                    current_chunk_tokens = para_tokens
        
        # Add final chunk
        if current_chunk_text:
            chunks.append(self._create_chunk(
                text=current_chunk_text,
                source_file=source_file,
                section=current_section,
                metadata=metadata,
            ))
        
        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        # Split on double newlines or page breaks
        paragraphs = re.split(r"\n\s*\n|\f", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _is_section_header(self, text: str) -> bool:
        """Check if text is a section header."""
        # Common header patterns
        patterns = [
            r"^#{1,6}\s+",  # Markdown headers
            r"^\d+\.\s+[A-Z]",  # Numbered sections
            r"^[A-Z][A-Z\s]{2,}$",  # ALL CAPS
            r"^(Abstract|Introduction|Methods|Results|Discussion|Conclusion|References)$",
        ]
        
        for pattern in patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # Short, title-case text
        if len(text) < 100 and text.istitle():
            return True
            
        return False

    def _split_large_paragraph(
        self,
        text: str,
        source_file: str,
        section: Optional[str],
        metadata: dict,
    ) -> list[Chunk]:
        """Split oversized paragraph into smaller chunks."""
        chunks = []
        sentences = self._split_sentences(text)
        
        current_text = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens <= self.max_tokens:
                current_text += " " + sentence if current_text else sentence
                current_tokens += sentence_tokens
            else:
                if current_text:
                    chunks.append(self._create_chunk(
                        text=current_text,
                        source_file=source_file,
                        section=section,
                        metadata=metadata,
                    ))
                current_text = sentence
                current_tokens = sentence_tokens
        
        if current_text:
            chunks.append(self._create_chunk(
                text=current_text,
                source_file=source_file,
                section=section,
                metadata=metadata,
            ))
        
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunk(
        self,
        text: str,
        source_file: str,
        section: Optional[str],
        metadata: dict,
    ) -> Chunk:
        """Create a Chunk object."""
        # Generate unique ID
        chunk_id = hashlib.md5(
            f"{source_file}:{section}:{text[:100]}".encode()
        ).hexdigest()[:16]
        
        return Chunk(
            id=chunk_id,
            text=text,
            source_file=source_file,
            section=section,
            metadata=metadata,
            token_count=self.count_tokens(text),
        )

    def chunk_markdown_file(
        self,
        file_path: Path,
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Chunk a markdown file.
        
        Args:
            file_path: Path to markdown file
            metadata: Additional metadata
            
        Returns:
            List of chunks
        """
        file_path = Path(file_path)
        text = file_path.read_text(encoding="utf-8")
        
        file_metadata = metadata or {}
        file_metadata["file_name"] = file_path.name
        
        return self.chunk_text(
            text=text,
            source_file=str(file_path),
            metadata=file_metadata,
        )

    def chunk_directory(
        self,
        input_dir: Path,
        output_path: Path,
        pattern: str = "*.md",
    ) -> list[Chunk]:
        """
        Chunk all files in a directory.
        
        Args:
            input_dir: Input directory
            output_path: Output JSON file
            pattern: File pattern to match
            
        Returns:
            List of all chunks
        """
        input_dir = Path(input_dir)
        all_chunks = []
        
        for file_path in input_dir.rglob(pattern):
            chunks = self.chunk_markdown_file(file_path)
            all_chunks.extend(chunks)
        
        # Save chunks
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [c.to_dict() for c in all_chunks],
                f,
                indent=2,
                ensure_ascii=False,
            )
        
        logger.info(f"Saved {len(all_chunks)} chunks to {output_path}")
        return all_chunks


def load_chunks(path: Path) -> list[Chunk]:
    """Load chunks from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk.from_dict(d) for d in data]


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    """Save chunks to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [c.to_dict() for c in chunks],
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunk documents")
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    
    args = parser.parse_args()
    
    chunker = SemanticChunker(max_tokens=args.max_tokens)
    
    if args.input.is_dir():
        chunks = chunker.chunk_directory(args.input, args.output)
    else:
        chunks = chunker.chunk_markdown_file(args.input)
        save_chunks(chunks, args.output)
    
    print(f"Created {len(chunks)} chunks")
