"""FastAPI application for KnightGPT RAG API."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..chunking import SemanticChunker
from ..embedding import VLLMEmbedder
from ..graph import KnowledgeGraphBuilder
from ..ingestion import (
    GoogleFormWebhook,
    batch_convert_pdfs,
    get_webhook_handler,
)
from ..retrieval import GraphRAGRetriever, RAGEngine
from ..utils import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


# Global instances
_retriever: Optional[GraphRAGRetriever] = None
_rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _retriever, _rag_engine
    
    # Load data on startup
    chunks_path = settings.ingestion.processed_dir / "chunks_with_emb.json"
    graph_path = settings.graph.graph_path
    
    if chunks_path.exists():
        logger.info(f"Loading chunks from {chunks_path}")
        _retriever = GraphRAGRetriever(
            chunks_path=chunks_path,
            graph_path=graph_path if graph_path.exists() else None,
        )
        _rag_engine = RAGEngine(retriever=_retriever)
        logger.info("RAG engine initialized")
    else:
        logger.warning(f"No chunks file found at {chunks_path}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")


app = FastAPI(
    title="KnightGPT API",
    description="Microbiome RAG API with knowledge graph retrieval",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ChatRequest(BaseModel):
    """Chat completion request."""
    
    message: str = Field(..., description="User message")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    max_tokens: int = Field(default=1024, description="Maximum response tokens")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    stream: bool = Field(default=False, description="Stream response")


class Citation(BaseModel):
    """Citation information."""
    
    source_file: str
    section: Optional[str]
    text_snippet: str
    similarity: float


class ChatResponse(BaseModel):
    """Chat completion response."""
    
    answer: str
    citations: list[Citation]


class IngestRequest(BaseModel):
    """Document ingestion request."""
    
    pdf_path: Optional[str] = None
    pdf_directory: Optional[str] = None
    force_ocr: bool = False


class SearchRequest(BaseModel):
    """Semantic search request."""
    
    query: str
    top_k: int = Field(default=10)
    expand_context: bool = Field(default=True)


class SearchResult(BaseModel):
    """Search result item."""
    
    chunk_id: str
    text: str
    source_file: str
    section: Optional[str]
    similarity: float


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    embedding_server: bool
    inference_server: bool
    chunks_loaded: int
    graph_nodes: int


def get_rag_engine() -> RAGEngine:
    """Dependency for RAG engine."""
    if _rag_engine is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not initialized. No data loaded.",
        )
    return _rag_engine


def get_retriever() -> GraphRAGRetriever:
    """Dependency for retriever."""
    if _retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialized. No data loaded.",
        )
    return _retriever


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    embedding_healthy = False
    inference_healthy = False
    
    try:
        embedder = VLLMEmbedder()
        embedding_healthy = embedder.check_health()
    except Exception as e:
        logger.error(f"Embedding health check failed: {e}")
    
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key="EMPTY",
            base_url=settings.vllm.inference_url,
        )
        # Quick health check
        client.models.list()
        inference_healthy = True
    except Exception as e:
        logger.error(f"Inference health check failed: {e}")
    
    chunks_count = len(_retriever.chunks) if _retriever else 0
    graph_nodes = (
        _retriever.graph_builder.graph.number_of_nodes()
        if _retriever and _retriever.graph_builder and _retriever.graph_builder.graph
        else 0
    )
    
    status = "healthy" if embedding_healthy and inference_healthy else "degraded"
    
    return HealthResponse(
        status=status,
        embedding_server=embedding_healthy,
        inference_server=inference_healthy,
        chunks_loaded=chunks_count,
        graph_nodes=graph_nodes,
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """
    RAG-enhanced chat completion.
    
    Retrieves relevant context from the knowledge graph and generates
    a response using the LLM.
    """
    if request.stream:
        # Return streaming response
        async def generate():
            async for token in rag_engine.query_stream(
                question=request.message,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )
    
    # Non-streaming response
    response = await rag_engine.query_async(
        question=request.message,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    
    citations = [
        Citation(
            source_file=c.source_file,
            section=c.section,
            text_snippet=c.text_snippet,
            similarity=c.similarity,
        )
        for c in response.citations
    ]
    
    return ChatResponse(
        answer=response.answer,
        citations=citations,
    )


@app.post("/api/v1/search")
async def semantic_search(
    request: SearchRequest,
    retriever: GraphRAGRetriever = Depends(get_retriever),
) -> list[SearchResult]:
    """
    Semantic search over the knowledge base.
    
    Returns relevant chunks without LLM generation.
    """
    result = retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        expand_context=request.expand_context,
    )
    
    # Ensure chunks and scores have matching lengths
    min_len = min(len(result.chunks), len(result.similarity_scores))
    chunks = result.chunks[:min_len]
    scores = result.similarity_scores[:min_len]
    
    return [
        SearchResult(
            chunk_id=chunk.id,
            text=chunk.text,
            source_file=chunk.source_file,
            section=chunk.section,
            similarity=score,
        )
        for chunk, score in zip(chunks, scores)
    ]


@app.post("/api/v1/ingest")
async def ingest_documents(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Ingest new documents into the knowledge base.
    
    Runs asynchronously in the background.
    """
    async def process_ingestion():
        try:
            if request.pdf_path:
                from ..ingestion import convert_pdf_to_markdown
                
                result = convert_pdf_to_markdown(
                    pdf_path=Path(request.pdf_path),
                    output_dir=settings.ingestion.markdown_dir,
                    force_ocr=request.force_ocr,
                )
                logger.info(f"Ingested: {result}")
                
            elif request.pdf_directory:
                results = batch_convert_pdfs(
                    input_dir=Path(request.pdf_directory),
                    output_dir=settings.ingestion.markdown_dir,
                    force_ocr=request.force_ocr,
                )
                logger.info(f"Ingested {len(results)} files")
            
            # Re-chunk, embed, and rebuild graph
            from ..chunking import SemanticChunker, save_chunks, load_chunks
            from ..embedding import VLLMEmbedder
            from ..graph import build_graph_from_chunks
            
            # Load existing chunks
            chunks_file = settings.ingestion.processed_dir / "chunks_with_emb.json"
            existing_chunks = []
            if chunks_file.exists():
                existing_chunks = load_chunks(chunks_file)
            
            # Chunk new markdown files
            chunker = SemanticChunker()
            new_chunks = chunker.chunk_directory(
                settings.ingestion.markdown_dir,
                settings.ingestion.processed_dir / "chunks_new.json"
            )
            
            # Generate embeddings for new chunks
            embedder = VLLMEmbedder()
            if embedder.check_health():
                new_chunks = embedder.embed_chunks(new_chunks)
            
            # Merge with existing chunks
            all_chunks = existing_chunks + new_chunks
            
            # Save updated chunks
            save_chunks(all_chunks, chunks_file)
            
            # Rebuild graph
            graph_file = settings.graph.graph_path
            build_graph_from_chunks(
                chunks_path=chunks_file,
                output_path=graph_file,
                threshold=settings.graph.similarity_threshold,
            )
            
            # Reload retriever and engine
            global _retriever, _rag_engine
            _retriever = GraphRAGRetriever(
                chunks_path=chunks_file,
                graph_path=graph_file if graph_file.exists() else None,
            )
            _rag_engine = RAGEngine(retriever=_retriever)
            logger.info("Knowledge base updated and reloaded")
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
    
    background_tasks.add_task(process_ingestion)
    
    return {"status": "accepted", "message": "Ingestion started in background"}


@app.post("/api/v1/webhook/google-form")
async def google_form_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Webhook endpoint for Google Form submissions.
    
    Automatically processes uploaded PDF files.
    """
    webhook = get_webhook_handler()
    return await webhook.handle_submission(request, background_tasks)


# OpenAI-compatible endpoints for Open WebUI integration
@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "knightgpt-rag",
                "object": "model",
                "created": 1700000000,
                "owned_by": "knight-lab",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def openai_chat_completions(
    request: Request,
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """
    OpenAI-compatible chat completions endpoint.
    
    For integration with Open WebUI and other OpenAI-compatible clients.
    """
    data = await request.json()
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    
    # Get the last user message
    user_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content")
            break
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    if stream:
        async def generate():
            async for token in rag_engine.query_stream(
                question=user_message,
                top_k=5,
                max_tokens=data.get("max_tokens", 1024),
                temperature=data.get("temperature", 0.7),
            ):
                chunk = {
                    "id": "chatcmpl-knightgpt",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {__import__('json').dumps(chunk)}\n\n"
            
            # Final chunk
            final_chunk = {
                "id": "chatcmpl-knightgpt",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {__import__('json').dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )
    
    # Non-streaming
    response = await rag_engine.query_async(
        question=user_message,
        top_k=5,
        max_tokens=data.get("max_tokens", 1024),
        temperature=data.get("temperature", 0.7),
    )
    
    return {
        "id": "chatcmpl-knightgpt",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = False,
):
    """Run the FastAPI server."""
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run KnightGPT API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, reload=args.reload)
