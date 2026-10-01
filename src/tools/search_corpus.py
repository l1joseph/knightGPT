"""Search-corpus agent tool for KnightGPT agent system.

Lets a chat user ask the model to explicitly search the live corpus mid-
conversation, instead of relying solely on the one automatic RAG
retrieval pass AgentOrchestrator.run() does up front against the
original query. This matters most right after ingest_paper adds a new
paper: the automatic retrieval pass already ran before that paper
existed, so without this tool the model has no way to go back and pull
the newly-ingested chunks within the same turn.

Thin wrapper around HybridRetriever.retrieve() -- the exact same
semantic-search + pgGraph-expansion logic /api/v1/search already uses --
so this duplicates no retrieval logic of its own.
"""

from typing import TYPE_CHECKING

from ..retrieval.base import BaseRetriever
from ..utils import get_logger
from .base import BaseTool, ToolResult

if TYPE_CHECKING:
    # Deferred at runtime (see execute()'s body) -- a module-level
    # `from ..api.request_context import RequestContext` would be a
    # circular import: src.api.__init__ imports .main, which imports
    # src.agents, which imports this module (same pattern as
    # src/tools/base.py, src/agents/orchestrator.py, and
    # src/tools/ingest_paper.py).
    from ..api.request_context import RequestContext

logger = get_logger(__name__)


class SearchCorpusTool(BaseTool):
    """Search the live KnightGPT corpus for relevant chunks."""

    name = "search_corpus"
    description = (
        "Search the knowledge base's own corpus of ingested papers for "
        "relevant text chunks, with similarity scores. Use this to look "
        "up specific findings, taxa, pathways, or methods already in the "
        "corpus -- including a paper just added via ingest_paper, which "
        "won't otherwise be reachable until a later turn. Not for "
        "general literature search outside the corpus (use pubmed_search "
        "or openalex_search for that)."
    )

    def __init__(self, retriever: BaseRetriever | None = None, default_top_k: int = 5):
        """
        Args:
            retriever: a HybridRetriever instance (or any BaseRetriever)
                to search against. Normally the same
                AgentOrchestrator.retriever this tool is registered
                alongside.
            default_top_k: chunks to return when the model doesn't
                specify top_k.
        """
        self.retriever = retriever
        self.default_top_k = default_top_k

    def execute(
        self,
        query: str,
        top_k: int | None = None,
        *,
        request_context: "RequestContext | None" = None,
        **kwargs,
    ) -> ToolResult:
        """Search the corpus and return matching chunks with scores,
        scoped to request_context.collection_id (global if no collection
        is attached). No model-facing collection override in v1 -- see
        the spec's Decisions section; request_context is injected by
        AgentOrchestrator.run(), never read from this method's own
        **kwargs even if the model's JSON arguments happen to include a
        collection_id-shaped key."""
        from ..api.request_context import RequestContext

        ctx = request_context or RequestContext()
        query = (query or "").strip()
        if not query:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="No search query was provided.",
            )

        if self.retriever is None:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    "Corpus search is unavailable: no corpus connection "
                    "(HybridRetriever) is configured for this tool."
                ),
            )

        try:
            result = self.retriever.retrieve(
                query,
                top_k=top_k or self.default_top_k,
                collection_id=ctx.collection_id,
            )
        except Exception as e:
            logger.error(f"search_corpus: retrieve failed for {query!r}: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Corpus search failed: {e}",
                metadata={"query": query},
            )

        min_len = min(len(result.chunks), len(result.similarity_scores))
        chunks = result.chunks[:min_len]
        scores = result.similarity_scores[:min_len]

        matches = [
            {
                "chunk_id": chunk.id,
                "source": chunk.source_file,
                "section": chunk.section,
                "similarity": round(float(score), 4),
                "text": chunk.text,
            }
            for chunk, score in zip(chunks, scores)
        ]

        return ToolResult(
            tool_name=self.name,
            success=True,
            data=matches,
            metadata={"query": query, "total_results": len(matches)},
        )

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the corpus.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum chunks to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        }
