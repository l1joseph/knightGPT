import os
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass
from ollama import chat
import json

logger = logging.getLogger(__name__)

# Configurable LLM model via environment variable
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")


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


def _validate_positive_int(value: int, name: str, allow_zero: bool = False) -> None:
    """Validate that a value is a positive integer."""
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    else:
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")


def _load_pdf_metadata(metadata_path: str) -> Tuple[str, str]:
    """
    Load PDF metadata from a JSON file.
    Returns (title, doi) tuple with fallback values on error.
    """
    try:
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found: {metadata_path}")
            return 'Unknown Title', 'Unknown DOI'

        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logger.warning(f"Invalid metadata format in {metadata_path}: expected dict")
            return 'Unknown Title', 'Unknown DOI'

        md = data.get('metadata', {})
        title = md.get('title', 'Unknown Title') or 'Unknown Title'
        doi = md.get('doi', 'Unknown DOI') or 'Unknown DOI'
        return title, doi

    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in metadata file {metadata_path}: {e}")
        return 'Unknown Title', 'Unknown DOI'
    except Exception as e:
        logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        return 'Unknown Title', 'Unknown DOI'


class RAGChatbot:
    """
    RAG Chatbot that retrieves relevant chunks and queries LLM for answers.
    Uses lazy initialization to avoid side effects on import.
    """

    # Default system prompt - generic to work with any domain
    DEFAULT_SYSTEM_PROMPT = (
        "You are a knowledgeable research assistant. "
        "Answer the question using ONLY the provided context paragraphs. "
        "Include inline citations like [^1], [^2], etc. "
        "If the context doesn't contain enough information to answer, say so."
    )

    def __init__(
        self,
        chunks_path: str = None,
        graph_path: str = None,
        graph_format: str = None,
        model_name: str = None,
        metadata_path: str = None,
        llm_model: str = None,
        system_prompt: str = None
    ):
        """
        Initialize the chatbot. Paths default to environment variables or standard locations.
        """
        # Use environment variables as defaults
        self.chunks_path = chunks_path or os.getenv("CHUNKS_PATH", "data/chunks.json")
        self.graph_path = graph_path or os.getenv("GRAPH_PATH", "data/graph.graphml")
        self.graph_format = graph_format or os.getenv("GRAPH_FORMAT", "graphml")
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.metadata_path = metadata_path or os.getenv("METADATA_PATH", "data/processed_pages.json")
        self.llm_model = llm_model or LLM_MODEL
        self.system_prompt = system_prompt or os.getenv("SYSTEM_PROMPT", self.DEFAULT_SYSTEM_PROMPT)

        # Lazy initialization - retriever created on first use
        self._retriever = None
        self._pdf_title = None
        self._pdf_doi = None

    @property
    def retriever(self):
        """Lazy load the retriever on first access."""
        if self._retriever is None:
            # Import here to avoid circular imports and module-level side effects
            from retrieval.retriever import Retriever
            logger.info(f"Initializing Retriever with chunks={self.chunks_path}, graph={self.graph_path}")
            self._retriever = Retriever(
                chunks_path=self.chunks_path,
                graph_path=self.graph_path,
                graph_format=self.graph_format,
                model_name=self.model_name
            )
        return self._retriever

    @property
    def pdf_title(self) -> str:
        """Lazy load PDF title."""
        if self._pdf_title is None:
            self._pdf_title, self._pdf_doi = _load_pdf_metadata(self.metadata_path)
        return self._pdf_title

    @property
    def pdf_doi(self) -> str:
        """Lazy load PDF DOI."""
        if self._pdf_doi is None:
            self._pdf_title, self._pdf_doi = _load_pdf_metadata(self.metadata_path)
        return self._pdf_doi

    def process_query(self, query: str, top_k: int = 5, hops: int = 1) -> Optional[str]:
        """
        Retrieve relevant chunks and query LLM interactively.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve initially (must be >= 1)
            hops: Number of graph hops for context expansion (must be >= 0)

        Returns:
            The LLM-generated answer, or None on error
        """
        # Validate inputs
        if not query or not query.strip():
            logger.error("Query cannot be empty")
            print("Error: Please enter a non-empty question.")
            return None

        try:
            _validate_positive_int(top_k, "top_k", allow_zero=False)
            _validate_positive_int(hops, "hops", allow_zero=True)
        except ValueError as e:
            logger.error(str(e))
            print(f"Error: {e}")
            return None

        # Retrieve chunks
        try:
            chunks = self.retriever.retrieve(query=query, top_k=top_k, hops=hops)
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            print(f"Error: Required data file not found. Please ensure the data files exist.")
            return None
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            print(f"Error during retrieval: {e}")
            return None

        if not chunks:
            logger.warning("No relevant documents found for query")
            print("No relevant documents found. Try rephrasing your question or adjusting retrieval parameters.")
            return None

        # Build context and citations list
        context_lines: List[str] = []
        citations: List[Citation] = []
        for idx, c in enumerate(chunks, start=1):
            tag = f"[^{idx}]"
            context_lines.append(f"{tag} {c['text']}")
            citations.append(Citation(
                text=c['text'],
                node_id=c['node_id'],
                page=c.get('page'),
                paragraph_index=c.get('paragraph_index'),
                chunk_index=c.get('chunk_index')
            ))
        context = "\n\n".join(context_lines)

        # Prepare prompts
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        # Query LLM via Ollama
        try:
            resp = chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
        except ConnectionError as e:
            logger.error(f"Could not connect to Ollama: {e}")
            print("Error: Could not connect to Ollama service. Please ensure it is running.")
            return None
        except Exception as e:
            logger.error(f"LLM service error: {e}")
            print(f"Error: LLM service error: {e}")
            return None

        # Extract and display answer
        content = _get_ollama_response_content(resp)
        if not content:
            logger.warning("LLM returned empty response")
            print("Warning: LLM returned an empty response.")
            return None

        answer = content.strip()
        print(f"\n=== Answer (based on '{self.pdf_title}' DOI: {self.pdf_doi}) ===")
        print(answer)
        print("\n=== Citations ===")
        for idx, cit in enumerate(citations, start=1):
            page_str = f"Page {cit.page}" if cit.page is not None else "Page N/A"
            para_str = f"Para {cit.paragraph_index}" if cit.paragraph_index is not None else "Para N/A"
            chunk_str = f"Chunk {cit.chunk_index}" if cit.chunk_index is not None else "Chunk N/A"
            print(f"[^{idx}] Title: {self.pdf_title}, DOI: {self.pdf_doi}, {page_str}, {para_str}, {chunk_str}")

        return answer

@dataclass
class Citation:
    """Represents a citation to a document chunk."""
    text: str
    node_id: str
    page: Optional[int]
    paragraph_index: Optional[int]
    chunk_index: Optional[int]


# Legacy function for backward compatibility
def process_query(query: str, top_k: int = 5, hops: int = 1) -> Optional[str]:
    """
    Legacy function - creates a RAGChatbot instance and processes the query.
    Prefer using RAGChatbot directly for better control.
    """
    chatbot = RAGChatbot()
    return chatbot.process_query(query, top_k=top_k, hops=hops)


def main():
    """Main entry point for the CLI."""
    import argparse
    import sys

    # Add parent directory to path for imports when running directly
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Configure logging
    from utils.logging import configure_logging
    configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))

    parser = argparse.ArgumentParser(description="CLI for Microbiome RAG Chatbot")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve initially (must be >= 1)")
    parser.add_argument("--hops", type=int, default=1, help="Number of graph hops for context expansion (must be >= 0)")
    parser.add_argument("--chunks", type=str, default=None, help="Path to chunks JSON file")
    parser.add_argument("--graph", type=str, default=None, help="Path to graph file")
    parser.add_argument("--metadata", type=str, default=None, help="Path to metadata JSON file")
    args = parser.parse_args()

    # Validate CLI arguments
    if args.top_k < 1:
        print("Error: --top_k must be >= 1")
        sys.exit(1)
    if args.hops < 0:
        print("Error: --hops must be >= 0")
        sys.exit(1)

    # Create chatbot with optional custom paths
    chatbot = RAGChatbot(
        chunks_path=args.chunks,
        graph_path=args.graph,
        metadata_path=args.metadata
    )

    print("Microbiome RAG Chatbot CLI (type 'exit' to quit)")
    while True:
        try:
            query = input("\nEnter your question: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query.lower().strip() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not query.strip():
            print("Please enter a question.")
            continue

        chatbot.process_query(query, top_k=args.top_k, hops=args.hops)


if __name__ == "__main__":
    main()
