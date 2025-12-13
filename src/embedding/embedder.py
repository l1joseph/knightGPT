"""vLLM-based embedding generation using OpenAI-compatible API."""

import asyncio
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..chunking import Chunk, load_chunks, save_chunks
from ..utils import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


class VLLMEmbedder:
    """
    Embedding generator using vLLM's OpenAI-compatible API.
    
    vLLM supports embedding models with the --task embed flag.
    Recommended models for high-quality embeddings:
    - Alibaba-NLP/gte-Qwen2-7B-instruct (7B, 8192 context)
    - BAAI/bge-large-en-v1.5 (335M, 512 context)
    - intfloat/multilingual-e5-large (560M, 512 context)
    
    Start vLLM embedding server:
    ```bash
    vllm serve Alibaba-NLP/gte-Qwen2-7B-instruct \\
        --task embed \\
        --trust-remote-code \\
        --port 8001 \\
        --tensor-parallel-size 1
    ```
    
    For AMD MI300A with ROCm:
    ```bash
    VLLM_ROCM_USE_AITER=1 vllm serve Alibaba-NLP/gte-Qwen2-7B-instruct \\
        --task embed \\
        --trust-remote-code \\
        --port 8001
    ```
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        api_key: str = "EMPTY",
        batch_size: int = 32,
        timeout: float = 60.0,
    ):
        """
        Initialize embedder.
        
        Args:
            api_base: vLLM server URL (default from settings)
            model: Embedding model name
            api_key: API key (vLLM typically uses "EMPTY")
            batch_size: Batch size for embedding generation
            timeout: Request timeout in seconds
        """
        self.api_base = api_base or settings.vllm.embedding_url
        self.model = model or settings.vllm.embedding_model
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout
        
        # Sync client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
        )
        
        # Async client
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Input validation
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")
        
        if not text.strip():
            raise ValueError("text cannot be empty or whitespace only")
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            if not response.data or len(response.data) == 0:
                raise ValueError("Empty response from embedding API")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Input validation
        if not isinstance(texts, list):
            raise TypeError("texts must be a list")
        
        # Filter out invalid entries
        valid_texts = [t for t in texts if t and isinstance(t, str) and t.strip()]
        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} invalid texts")
        
        if not valid_texts:
            return []
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=valid_texts,
            )
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise
        
        # Sort by index to maintain order (if index is available)
        # Some APIs may not return index, so handle gracefully
        try:
            sorted_data = sorted(response.data, key=lambda x: getattr(x, 'index', 0))
        except (AttributeError, TypeError):
            # If sorting fails, return in original order
            sorted_data = response.data
        
        return [item.embedding for item in sorted_data]

    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True,
    ) -> list[Chunk]:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: List of Chunk objects
            show_progress: Whether to show progress
            
        Returns:
            Chunks with embeddings added
        """
        total = len(chunks)
        logger.info(f"Embedding {total} chunks in batches of {self.batch_size}")
        
        for i in range(0, total, self.batch_size):
            batch = chunks[i:i + self.batch_size]
            texts = [c.text for c in batch]
            
            try:
                embeddings = self.embed_batch(texts)
                
                for chunk, embedding in zip(batch, embeddings):
                    chunk.embedding = embedding
                    
                if show_progress:
                    progress = min(i + self.batch_size, total)
                    logger.info(f"Embedded {progress}/{total} chunks")
                    
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Try individual embeddings as fallback
                for chunk in batch:
                    try:
                        chunk.embedding = self.embed_text(chunk.text)
                    except Exception as e2:
                        logger.error(f"Single embedding failed: {e2}")
        
        return chunks

    async def embed_text_async(self, text: str) -> list[float]:
        """Async version of embed_text."""
        response = await self.async_client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """Async version of embed_batch."""
        if not texts:
            return []
        
        response = await self.async_client.embeddings.create(
            model=self.model,
            input=texts,
        )
        
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    async def embed_chunks_async(
        self,
        chunks: list[Chunk],
        max_concurrent: int = 4,
    ) -> list[Chunk]:
        """
        Async version of embed_chunks with concurrency control.
        
        Args:
            chunks: List of Chunk objects
            max_concurrent: Maximum concurrent batch requests
            
        Returns:
            Chunks with embeddings added
        """
        total = len(chunks)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch_chunks: list[Chunk]) -> None:
            async with semaphore:
                texts = [c.text for c in batch_chunks]
                embeddings = await self.embed_batch_async(texts)
                for chunk, embedding in zip(batch_chunks, embeddings):
                    chunk.embedding = embedding
        
        # Create batch tasks
        tasks = []
        for i in range(0, total, self.batch_size):
            batch = chunks[i:i + self.batch_size]
            tasks.append(process_batch(batch))
        
        await asyncio.gather(*tasks)
        logger.info(f"Embedded {total} chunks")
        
        return chunks

    def check_health(self) -> bool:
        """Check if embedding server is healthy."""
        try:
            self.embed_text("health check")
            return True
        except Exception as e:
            logger.error(f"Embedding server health check failed: {e}")
            return False


def embed_chunks_file(
    input_path: str,
    output_path: str,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    batch_size: int = 32,
) -> None:
    """
    Embed chunks from a JSON file and save results.
    
    Args:
        input_path: Path to input chunks JSON
        output_path: Path for output JSON with embeddings
        api_base: vLLM server URL
        model: Embedding model name
        batch_size: Batch size
    """
    chunks = load_chunks(Path(input_path))
    logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
    
    embedder = VLLMEmbedder(
        api_base=api_base,
        model=model,
        batch_size=batch_size,
    )
    
    embedded_chunks = embedder.embed_chunks(chunks)
    save_chunks(embedded_chunks, Path(output_path))
    logger.info(f"Saved embedded chunks to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for chunks")
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    
    args = parser.parse_args()
    
    embed_chunks_file(
        input_path=args.input,
        output_path=args.output,
        api_base=args.api_base,
        model=args.model,
        batch_size=args.batch_size,
    )
