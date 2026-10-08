"""Rename-collection agent tool for KnightGPT agent system.

Lets a chat user self-serve rename a collection's display name (e.g.
"rename test-pilot to 'Qiita Pilot Study'"). Reuses
HybridRetriever.rename_collection() -- the same bridge create_collection/
ingest_paper already use -- so this tool never opens a second Postgres
connection.

Scope is deliberately narrow: only display_name can change, never the
slug/id itself -- renaming the id would require migrating
collection_id across every papers/chunks/chunk_edges row in Postgres
AND every row in DuckDB, out of scope. Any authenticated caller may
rename any collection -- collections are shareable/collaborative by
design, same policy as GET /api/v1/collections.

execute() runs synchronously inside AgentOrchestrator.run(), itself
dispatched to a worker thread via run_in_threadpool from the async
/v1/chat/completions and /api/v1/agent/chat handlers (see
src/api/main.py). The Postgres write below therefore goes through
HybridRetriever's private event-loop bridge (self._run/self._pool)
rather than awaiting main.py's own `_pool` directly from this sync
method -- the same reasoning src/tools/ingest_paper.py's Postgres
insert documents.
"""

from typing import TYPE_CHECKING

from ..utils import get_logger
from ..utils.db import RESERVED_COLLECTION_SLUGS
from .base import BaseTool, ToolResult

if TYPE_CHECKING:
    # Deferred at runtime -- same circular-import reason as
    # src/tools/base.py, src/agents/orchestrator.py,
    # src/tools/ingest_paper.py, and src/tools/create_collection.py.
    from ..api.request_context import RequestContext

logger = get_logger(__name__)


class RenameCollectionTool(BaseTool):
    """Rename a collection's display name (never its slug/id)."""

    name = "rename_collection"
    description = (
        "Rename a collection's display name (NOT its slug/id -- the id "
        "used in Open WebUI's model picker never changes here). Any "
        "user may rename any collection -- collections are shareable by "
        "design. 'global' cannot be renamed. Use this when the user "
        "explicitly asks to rename a project, collection, or workspace."
    )

    def __init__(self, retriever=None):
        """
        Args:
            retriever: a HybridRetriever instance (or any object
                exposing the same rename_collection() method) that owns
                the live Postgres pool to update. Normally the same
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
        """Rename one collection's display_name.

        Args:
            query: unused -- BaseTool's dispatch convention always
                passes the model's "query" argument positionally, but
                this tool reads "slug" instead (see this tool's
                schema).
            slug: the collection id/slug to rename.
            display_name: the new human-readable name.
            request_context: unused -- any authenticated caller may
                rename any collection (see this module's docstring),
                accepted only for dispatch-convention parity with every
                other tool.
        """
        slug_str = (slug or "").strip()
        if not slug_str:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="No slug was provided.",
            )

        if slug_str in RESERVED_COLLECTION_SLUGS:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="'global' is reserved and implicit -- it cannot be renamed.",
                metadata={"slug": slug_str},
            )

        display_name_str = (display_name or "").strip()
        if not display_name_str:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="No display_name was provided.",
                metadata={"slug": slug_str},
            )

        if self.retriever is None or not hasattr(self.retriever, "rename_collection"):
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    "Collection rename is unavailable: no corpus "
                    "connection (HybridRetriever) is configured for "
                    "this tool."
                ),
                metadata={"slug": slug_str},
            )

        try:
            row = self.retriever.rename_collection(slug_str, display_name_str)
        except Exception as e:
            logger.error(f"rename_collection: update failed for {slug_str}: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Could not rename collection '{slug_str}': {e}",
                metadata={"slug": slug_str},
            )

        if row is None:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Collection '{slug_str}' not found.",
                metadata={"slug": slug_str},
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            data=f"Renamed collection '{slug_str}' to '{row.get('display_name')}'.",
            metadata={"slug": slug_str, "display_name": row.get("display_name")},
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
                            "Collection id/slug to rename. 'global' cannot "
                            "be renamed."
                        ),
                    },
                    "display_name": {
                        "type": "string",
                        "description": "New human-readable name for the collection.",
                    },
                },
                "required": ["slug", "display_name"],
            },
        }
