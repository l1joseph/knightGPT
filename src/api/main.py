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
from ..graph import insert_chunks
from ..ingestion import (
    GoogleFormWebhook,
    batch_convert_pdfs,
    get_webhook_handler,
)
from ..retrieval import PostgresRetriever, RAGEngine
from ..utils import get_logger, get_pg_pool, get_settings

logger = get_logger(__name__)
settings = get_settings()


# Global instances
_pool = None
_retriever: Optional[PostgresRetriever] = None
_rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _pool, _retriever, _rag_engine

    # _pool backs the /api/v1/ingest background task's insert_chunks() calls
    # (always awaited from this loop). _retriever manages its own separate
    # pool internally on a private background loop — see
    # src/retrieval/postgres_retriever.py — because its retrieve() must stay
    # callable synchronously from code that may already be inside a running
    # event loop (e.g. /api/v1/search), which this loop's own pool can't
    # support.
    _pool = await get_pg_pool()
    _retriever = PostgresRetriever()
    _rag_engine = RAGEngine(retriever=_retriever)
    logger.info("RAG engine initialized (Postgres-backed)")

    yield

    logger.info("Shutting down...")
    if _pool is not None:
        await _pool.close()
    if _retriever is not None:
        _retriever.close()


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


def get_retriever() -> PostgresRetriever:
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

    chunks_count = 0
    graph_nodes = 0
    if _pool is not None:
        async with _pool.acquire() as conn:
            chunks_count = await conn.fetchval("SELECT count(*) FROM chunks")
            graph_status = await conn.fetchrow("SELECT node_count FROM graph.status()")
            graph_nodes = graph_status["node_count"] if graph_status else 0

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
    retriever: PostgresRetriever = Depends(get_retriever),
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

            # Chunk new markdown files and embed
            from ..chunking import SemanticChunker
            from ..embedding import VLLMEmbedder

            chunker = SemanticChunker()
            new_chunks = chunker.chunk_directory(
                settings.ingestion.markdown_dir,
                settings.ingestion.processed_dir / "chunks_new.json",
            )

            embedder = VLLMEmbedder()
            if embedder.check_health():
                new_chunks = embedder.embed_chunks(new_chunks)

            papers = {
                c.source_file: {
                    "doi": c.source_file,
                    "title": "",
                    "metadata": c.metadata,
                }
                for c in new_chunks
            }
            insert_stats = await insert_chunks(_pool, new_chunks, papers)
            logger.info(f"Knowledge base updated: {insert_stats}")

        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)

    background_tasks.add_task(process_ingestion)

    return {"status": "accepted", "message": "Ingestion started in background"}


class RSSIngestRequest(BaseModel):
    """RSS feed ingestion request."""

    feeds: Optional[list[str]] = Field(
        default=None,
        description="Feed names to check (None = all)",
    )
    max_papers: int = Field(default=20, description="Max papers to ingest")


@app.post("/api/v1/ingest/rss")
async def ingest_from_rss(
    request: RSSIngestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger RSS feed discovery and ingestion.

    Discovers papers from configured RSS feeds, downloads PDFs,
    and runs the ingestion pipeline.
    """

    async def process_rss():
        try:
            from ..ingestion.rss_feed import RSSFeedIngester, DEFAULT_FEEDS

            feeds = None
            if request.feeds:
                feeds = {k: v for k, v in DEFAULT_FEEDS.items() if k in request.feeds}

            ingester = RSSFeedIngester(feeds=feeds)
            papers = ingester.discover_from_rss()
            papers = papers[: request.max_papers]

            if papers:
                stats = ingester.download_and_ingest(papers)
                logger.info(f"RSS ingestion: {stats}")
            else:
                logger.info("No new papers found in RSS feeds")

        except Exception as e:
            logger.error(f"RSS ingestion failed: {e}", exc_info=True)

    background_tasks.add_task(process_rss)

    return {"status": "accepted", "message": "RSS ingestion started in background"}


class BriefingRequest(BaseModel):
    """Briefing ingestion request."""

    text: str = Field(..., description="Briefing text content")
    papers: Optional[list[dict]] = Field(
        default=None,
        description="Explicit paper list [{doi, url, title, summary}]",
    )
    source: str = Field(default="manual", description="Source identifier")


@app.post("/api/v1/ingest/briefing")
async def ingest_briefing(
    request: BriefingRequest,
    background_tasks: BackgroundTasks,
):
    """
    Ingest papers from a briefing (email, bot, or manual paste).

    Parses text for DOIs, downloads PDFs, and runs ingestion.
    Accepts both free-form text and structured paper lists.
    """
    from ..ingestion.briefing_parser import parse_briefing_text, parse_briefing_json

    # Parse the briefing
    if request.papers:
        parsed = parse_briefing_json(
            {
                "text": request.text,
                "papers": request.papers,
                "source": request.source,
            }
        )
    else:
        parsed = parse_briefing_text(request.text, source=request.source)

    if not parsed.papers:
        return {
            "status": "no_papers",
            "message": "No paper references found in briefing",
        }

    async def process_briefing():
        try:
            from ..ingestion.rss_feed import RSSFeedIngester, DiscoveredPaper

            ingester = RSSFeedIngester()
            papers = []
            for ref in parsed.papers:
                papers.append(
                    DiscoveredPaper(
                        title=ref.title or "Unknown",
                        doi=ref.doi,
                        url=ref.url,
                        abstract=ref.summary,
                        source_feed=f"briefing:{request.source}",
                    )
                )

            stats = ingester.download_and_ingest(papers)
            logger.info(f"Briefing ingestion: {stats}")

        except Exception as e:
            logger.error(f"Briefing ingestion failed: {e}", exc_info=True)

    background_tasks.add_task(process_briefing)

    return {
        "status": "accepted",
        "papers_found": len(parsed.papers),
        "dois": [p.doi for p in parsed.papers if p.doi],
        "message": f"Processing {len(parsed.papers)} papers from briefing",
    }


@app.post("/api/v1/webhook/briefing")
async def briefing_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Webhook for direct briefing bot integration.

    Requires shared secret in X-Webhook-Secret header.
    Accepts JSON payload with text and optional paper list.
    """
    # Verify shared secret
    webhook_secret = settings.api.api_key
    if webhook_secret:
        provided_secret = request.headers.get("X-Webhook-Secret")
        if provided_secret != webhook_secret:
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

    data = await request.json()

    briefing_req = BriefingRequest(
        text=data.get("text", ""),
        papers=data.get("papers"),
        source=data.get("source", "webhook"),
    )

    return await ingest_briefing(briefing_req, background_tasks)


class AgentChatRequest(BaseModel):
    """Agent chat request."""

    message: str = Field(..., description="User message")
    top_k: int = Field(default=5, description="RAG retrieval depth")


@app.post("/api/v1/agent/chat")
async def agent_chat(request: AgentChatRequest):
    """
    Multi-agent RAG chat with tool use.

    Uses a 4-stage pipeline (plan → execute → verify → generate)
    with access to PubMed, OpenAlex, KEGG, and QIIME2 tools.
    """
    from ..agents import AgentOrchestrator

    orchestrator = AgentOrchestrator(
        retriever=_retriever,
        rag_engine=_rag_engine,
    )

    result = orchestrator.run(request.message, top_k=request.top_k)

    return {
        "answer": result.final_answer,
        "plan": {
            "tools_used": result.plan.tools_to_use if result.plan else [],
            "sub_queries": result.plan.sub_queries if result.plan else [],
            "reasoning": result.plan.reasoning if result.plan else "",
        },
        "citations": result.verified_citations[:10],
        "tool_results_count": len(result.tool_results),
    }


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
