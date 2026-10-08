"""Create-collection agent tool for KnightGPT agent system.

Lets a chat user self-serve create a new collection from inside a
conversation (e.g. "create a new workspace called qiita-pilot"), instead
of the only existing path being a raw curl call to
POST /api/v1/collections (see src/api/main.py). Reuses that same route's
underlying validation and insert logic -- src.utils.db
.validate_collection_slug() / .insert_collection() -- via
HybridRetriever.create_collection(), so both self-serve paths enforce
identical rules and run the identical INSERT, rather than two
slightly-different copies.

execute() runs synchronously inside AgentOrchestrator.run(), itself
dispatched to a worker thread via run_in_threadpool from the async
/v1/chat/completions and /api/v1/agent/chat handlers (see
src/api/main.py). The Postgres write below therefore goes through
HybridRetriever's private event-loop bridge (self._run/self._pool) rather
than awaiting main.py's own `_pool` directly from this sync method -- the
same reasoning src/tools/ingest_paper.py's Postgres insert documents.
"""

from typing import TYPE_CHECKING

import asyncpg

from ..utils import get_logger
from ..utils.db import validate_collection_slug
from .base import BaseTool, ToolResult

if TYPE_CHECKING:
    # Deferred at runtime (see execute()'s body) -- a module-level
    # `from ..api.request_context import RequestContext` would be a
    # circular import, same reason as src/tools/base.py,
    # src/agents/orchestrator.py, src/tools/ingest_paper.py, and
    # src/tools/search_corpus.py.
    from ..api.request_context import RequestContext

logger = get_logger(__name__)

_MODEL_ID_PREFIX = "knightgpt-rag-"


class CreateCollectionTool(BaseTool):
    """Register a new collection, selectable afterward in Open WebUI."""

    name = "create_collection"
    description = (
        "Create a new collection (project/workspace) that scopes its "
        "own RAG corpus. slug must match ^[a-z0-9][a-z0-9-]{0,39}$ "
        "(lowercase alphanumerics and hyphens, starting with an "
        "alphanumeric, max 40 chars); 'global' is reserved and not "
        "allowed. Use this when the user explicitly asks to create a "
        "new project, collection, or workspace."
    )

    def __init__(self, retriever=None):
        """
        Args:
            retriever: a HybridRetriever instance (or any object exposing
                the same create_collection() method) that owns the live
                Postgres pool to insert into. Normally the same
                AgentOrchestrator.retriever this tool is registered
                alongside, so no second DB connection is ever opened.
        """
        self.retriever = retriever

    def execute(
        self,
        query: str,
        slug: str | None = None,
        display_name: str | None = None,
        *,
        request_context: "RequestContext | None" = None,
        **kwargs,
    ) -> ToolResult:
        """Validate and register a new collection, owned by the caller.

        Args:
            query: unused -- BaseTool's dispatch convention always passes
                the model's "query" argument positionally, but this tool
                reads "slug" instead (see this tool's schema).
            slug: the desired collection id/slug.
            display_name: optional human-readable name.
            request_context: identity, injected by AgentOrchestrator.run()
                -- request_context.email becomes the new collection's
                owner_email, never read from this method's own **kwargs
                even if the model's JSON tool-call arguments happen to
                include an owner_email- or request_context-shaped key.
        """
        from ..api.request_context import RequestContext

        ctx = request_context or RequestContext()

        slug_str = (slug or "").strip()
        if not slug_str:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="No slug was provided.",
            )

        try:
            validate_collection_slug(slug_str)
        except ValueError as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e),
                metadata={"slug": slug_str},
            )

        if self.retriever is None or not hasattr(self.retriever, "create_collection"):
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    "Collection creation is unavailable: no corpus "
                    "connection (HybridRetriever) is configured for "
                    "this tool."
                ),
                metadata={"slug": slug_str},
            )

        try:
            row = self.retriever.create_collection(
                slug_str, display_name, owner_email=ctx.email
            )
        except asyncpg.UniqueViolationError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Collection '{slug_str}' already exists.",
                metadata={"slug": slug_str, "already_exists": True},
            )
        except Exception as e:
            logger.error(f"create_collection: insert failed for {slug_str}: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Could not create collection '{slug_str}': {e}",
                metadata={"slug": slug_str},
            )

        model_id = f"{_MODEL_ID_PREFIX}{slug_str}"
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=(
                f"Created collection '{slug_str}'. Select '{model_id}' "
                "from Open WebUI's model picker to start using it."
            ),
            metadata={
                "slug": slug_str,
                "display_name": row.get("display_name"),
                "owner_email": row.get("owner_email"),
                "model_id": model_id,
            },
        )

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "slug": {
                        "type": "string",
                        "description": (
                            "Collection id, e.g. 'test-a'. Must match "
                            "^[a-z0-9][a-z0-9-]{0,39}$ (lowercase "
                            "alphanumerics and hyphens, starting with "
                            "an alphanumeric, max 40 chars). 'global' "
                            "is reserved."
                        ),
                    },
                    "display_name": {
                        "type": "string",
                        "description": "Human-readable name for the collection.",
                    },
                },
                "required": ["slug"],
            },
        }
