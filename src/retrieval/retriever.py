"""Graph-based RAG retriever with vLLM inference (file-backed)."""

from pathlib import Path
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI, OpenAI

from ..chunking import Chunk, load_chunks
from ..embedding import VLLMEmbedder
from ..graph import KnowledgeGraphBuilder
from ..utils import get_logger, get_settings
from .base import BaseRetriever, RAGResponse, RetrievalResult

logger = get_logger(__name__)
settings = get_settings()


class GraphRAGRetriever(BaseRetriever):
    """
    Graph-enhanced RAG retriever.
    
    Combines semantic similarity search with knowledge graph
    traversal for improved context retrieval.
    """

    def __init__(
        self,
        chunks: Optional[list[Chunk]] = None,
        chunks_path: Optional[Path] = None,
        graph_path: Optional[Path] = None,
        embedder: Optional[VLLMEmbedder] = None,
        top_k: int = 5,
        graph_hops: int = 1,
    ):
        """
        Initialize retriever.
        
        Args:
            chunks: Pre-loaded chunks
            chunks_path: Path to chunks JSON file
            graph_path: Path to graph GraphML file
            embedder: VLLMEmbedder instance
            top_k: Number of initial chunks to retrieve
            graph_hops: Number of hops for context expansion
        """
        self.top_k = top_k
        self.graph_hops = graph_hops
        
        # Load chunks
        if chunks:
            self.chunks = chunks
        elif chunks_path:
            self.chunks = load_chunks(Path(chunks_path))
        else:
            self.chunks = []
        
        # Create chunk lookup
        self.chunk_map = {c.id: c for c in self.chunks}
        
        # Initialize graph builder
        self.graph_builder = KnowledgeGraphBuilder()
        if graph_path and Path(graph_path).exists():
            self.graph_builder.load_graph(Path(graph_path))
        elif self.chunks:
            self.graph_builder.build_graph(self.chunks)
        
        # Initialize embedder
        self.embedder = embedder or VLLMEmbedder()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        expand_context: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Override default top_k
            expand_context: Whether to expand with graph neighbors
            
        Returns:
            RetrievalResult with chunks and scores
        """
        # Input validation
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Empty or invalid query provided")
            return RetrievalResult(chunks=[], query_embedding=[], similarity_scores=[])
        
        if not self.chunks:
            logger.warning("No chunks available for retrieval")
            return RetrievalResult(chunks=[], query_embedding=[], similarity_scores=[])
        
        top_k = top_k or self.top_k
        top_k = max(1, min(top_k, len(self.chunks)))  # Clamp to valid range
        
        # Generate query embedding
        try:
            query_embedding = self.embedder.embed_text(query)
            if not query_embedding or len(query_embedding) == 0:
                logger.error("Failed to generate query embedding")
                return RetrievalResult(chunks=[], query_embedding=[], similarity_scores=[])
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return RetrievalResult(chunks=[], query_embedding=[], similarity_scores=[])
        
        # Find similar chunks
        results = self.graph_builder.find_similar_chunks(
            query_embedding=query_embedding,
            chunks=self.chunks,
            top_k=top_k,
        )
        
        initial_chunks = [chunk for chunk, _ in results]
        similarity_scores = [score for _, score in results]
        
        # Expand context using graph
        if expand_context and self.graph_hops > 0:
            expanded_chunks = self.graph_builder.expand_context(
                initial_chunks=initial_chunks,
                all_chunks=self.chunks,
                hops=self.graph_hops,
            )
            
            # Re-score expanded chunks
            expanded_scores = []
            for chunk in expanded_chunks:
                try:
                    idx = initial_chunks.index(chunk)
                    expanded_scores.append(similarity_scores[idx])
                except ValueError:
                    # Chunk not in initial_chunks, calculate similarity
                    from scipy.spatial.distance import cosine
                    import numpy as np
                    
                    if chunk.embedding:
                        sim = 1 - cosine(
                            np.array(query_embedding),
                            np.array(chunk.embedding)
                        )
                        expanded_scores.append(sim)
                    else:
                        expanded_scores.append(0.0)
            
            # Sort by similarity
            sorted_results = sorted(
                zip(expanded_chunks, expanded_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            
            initial_chunks = [c for c, _ in sorted_results]
            similarity_scores = [s for _, s in sorted_results]
        
        return RetrievalResult(
            chunks=initial_chunks,
            query_embedding=query_embedding,
            similarity_scores=similarity_scores,
        )


class RAGEngine:
    """
    Complete RAG engine with retrieval and generation.
    
    Uses vLLM for both embedding and inference.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        inference_url: Optional[str] = None,
        inference_model: Optional[str] = None,
        api_key: str = "EMPTY",
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize RAG engine.

        Args:
            retriever: BaseRetriever instance
            inference_url: vLLM inference server URL
            inference_model: Model name for generation
            api_key: API key
            system_prompt: Custom system prompt
        """
        self.retriever = retriever
        self.inference_url = inference_url or settings.vllm.inference_url
        self.inference_model = inference_model or settings.vllm.inference_model
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.inference_url,
        )
        
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.inference_url,
        )
        
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a helpful research assistant specializing in microbiome science.
You answer questions based on the provided context from scientific literature.

Guidelines:
- Answer based on the provided context
- Cite sources when making specific claims
- If the context doesn't contain enough information, say so
- Be precise and scientific in your language
- Highlight any uncertainties or conflicting information

When citing, use the format: [Source: filename, Section: section_name]"""

    def query(
        self,
        question: str,
        top_k: int = 5,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> RAGResponse:
        """
        Process a RAG query.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
            
        Returns:
            RAGResponse with answer and citations
        """
        # Retrieve relevant chunks
        retrieval = self.retriever.retrieve(question, top_k=top_k)
        
        # Format context
        context = self.retriever.format_context(retrieval.chunks)
        
        # Create prompt
        user_message = f"""Context:
{context}

Question: {question}

Please answer the question based on the provided context. Cite your sources."""
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.inference_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        answer = response.choices[0].message.content
        
        # Create citations
        citations = self.retriever.create_citations(
            retrieval.chunks,
            retrieval.similarity_scores,
        )
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            context_chunks=retrieval.chunks,
        )

    async def query_async(
        self,
        question: str,
        top_k: int = 5,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> RAGResponse:
        """Async version of query."""
        retrieval = self.retriever.retrieve(question, top_k=top_k)
        context = self.retriever.format_context(retrieval.chunks)
        
        user_message = f"""Context:
{context}

Question: {question}

Please answer the question based on the provided context. Cite your sources."""
        
        response = await self.async_client.chat.completions.create(
            model=self.inference_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        answer = response.choices[0].message.content
        
        citations = self.retriever.create_citations(
            retrieval.chunks,
            retrieval.similarity_scores,
        )
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            context_chunks=retrieval.chunks,
        )

    async def query_stream(
        self,
        question: str,
        top_k: int = 5,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """
        Stream RAG response.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            
        Yields:
            Response tokens as they're generated
        """
        retrieval = self.retriever.retrieve(question, top_k=top_k)
        context = self.retriever.format_context(retrieval.chunks)
        
        user_message = f"""Context:
{context}

Question: {question}

Please answer the question based on the provided context. Cite your sources."""
        
        stream = await self.async_client.chat.completions.create(
            model=self.inference_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG query interface")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--query", "-q", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    
    args = parser.parse_args()
    
    retriever = GraphRAGRetriever(
        chunks_path=args.chunks,
        graph_path=args.graph,
    )
    
    engine = RAGEngine(retriever=retriever)
    response = engine.query(args.query, top_k=args.top_k)
    
    print("Answer:")
    print(response.answer)
    print("\nCitations:")
    for cit in response.citations:
        print(f"  - {cit.source_file} ({cit.section}): {cit.similarity:.3f}")
