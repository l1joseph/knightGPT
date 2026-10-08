"""Delete-collection agent tool for KnightGPT agent system.

Lets a chat user self-serve remove a collection from Open WebUI's model
picker (e.g. "delete the test-pilot collection"), and lets an admin
additionally wipe its underlying corpus data. Reuses
HybridRetriever.delete_collection_registry_row() /
.delete_collection_data() -- the same bridge create_collection/
ingest_paper already use -- so this tool never opens a second Postgres/
DuckDB connection.

Two modes, mirroring DELETE /api/v1/collections/{slug} (src/api/main.py):

- Default (delete_data=False): removes only the collections registry
  row. papers/chunks/chunk_edges/DuckDB rows under this collection_id
  are left untouched and stay fully searchable -- they are just no
  longer listed via /v1/models. Recreating the same slug later makes
  that old data reachable again through the model picker; this is a
  deliberate, non-destructive property of collection_id being
  free-form text with no FK to the registry table, not a bug.
- delete_data=True, admin-only (gated the same way ingest_paper.py
  gates also_global): ALSO permanently deletes every papers/chunks/
  chunk_edges row tagged with this collection_id and every matching
  DuckDB embedding row. A non-admin caller setting delete_data=True
  gets a clear ToolResult(success=False, ...) error, never silent
  ignoring.

execute() runs synchronously inside AgentOrchestrator.run(), itself
dispatched to a worker thread via run_in_threadpool from the async
/v1/chat/completions and /api/v1/agent/chat handlers (see
src/api/main.py). The Postgres/DuckDB writes below therefore go
through HybridRetriever's private event-loop bridge (self._run/
self._pool) rather than awaiting main.py's own `_pool` directly from
this sync method -- the same reasoning src/tools/ingest_paper.py's
Postgres insert documents.
"""

from typing import TYPE_CHECKING

from ..utils import get_logger
from ..utils.db import RESERVED_COLLECTION_SLUGS
from .base import BaseTool, ToolResult

if TYPE_CHECKING:
    # Deferred at runtime (see execute()'s body) -- a module-level
    # `from ..api.request_context import RequestContext` would be a
    # circular import, same reason as src/tools/base.py,
    # src/agents/orchestrator.py, src/tools/ingest_paper.py, and
    # src/tools/create_collection.py.
    from ..api.request_context import RequestContext

logger = get_logger(__name__)


class DeleteCollectionTool(BaseTool):
    """Remove a collection from the registry, optionally wiping its
    underlying corpus data (admin-only)."""

    name = "delete_collection"
    description = (
        "Remove a collection (project/workspace) so it no longer "
        "appears in Open WebUI's model picker. By default this only "
        "removes the collection's registry entry -- its papers/chunks "
        "stay in the corpus under that collection id, just no longer "
        "listed. Set delete_data=true to ALSO permanently delete every "
        "paper, chunk, and embedding under this collection -- admin "
        "only, irreversible. 'global' can never be deleted. Use this "
        "when the user explicitly asks to delete, remove, or wipe a "
        "collection/project/workspace."
    )

    def __init__(self, retriever=None):
        """
        Args:
            retriever: a HybridRetriever instance (or any object
                exposing the same delete_collection_registry_row() /
                delete_collection_data() methods) that owns the live
                Postgres pool + DuckDB store to delete from. Normally
                the same AgentOrchestrator.retriever this tool is
                registered alongside, so no second DB connection is
                ever opened.
        """
        self.retriever = retriever

    def execute(
        self,
        query: str,
        slug: str | None = None,
        delete_data: bool = False,
        *,
        request_context: "RequestContext | None" = None,
        **kwargs,
    ) -> ToolResult:
        """Remove a collection's registry row, and optionally wipe its
        underlying corpus data.

        Args:
            query: unused -- BaseTool's dispatch convention always
                passes the model's "query" argument positionally, but
                this tool reads "slug" instead (see this tool's schema).
            slug: the collection id/slug to remove.
            delete_data: if True, ALSO permanently delete every
                papers/chunks/chunk_edges row tagged with this
                collection_id and every matching DuckDB embedding row.
                Honored only when request_context.is_admin is True --
                anyone else setting it gets a clear
                ToolResult(success=False, ...) error, never silent
                ignoring (same convention as ingest_paper's
                also_global).
            request_context: identity, injected by
                AgentOrchestrator.run() -- never read from this
                method's own **kwargs even if the model's JSON
                tool-call arguments happen to include an is_admin- or
                request_context-shaped key.
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

        if slug_str in RESERVED_COLLECTION_SLUGS:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="'global' is reserved and implicit -- it cannot be deleted.",
                metadata={"slug": slug_str},
            )

        if delete_data and not ctx.is_admin:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="Only admin can delete the underlying collection data.",
                metadata={"slug": slug_str},
            )

        if self.retriever is None or not hasattr(
            self.retriever, "delete_collection_registry_row"
        ):
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    "Collection deletion is unavailable: no corpus "
                    "connection (HybridRetriever) is configured for "
                    "this tool."
                ),
                metadata={"slug": slug_str},
            )

        # Data wipe runs BEFORE the registry row delete, deliberately: the
        # registry row is the user-visible signal that a collection
        # "exists" (it's what /v1/models and GET /api/v1/collections show).
        # If the dangerous papers/chunks/chunk_edges/DuckDB delete fails
        # partway, leaving the registry row intact means the collection
        # still shows up normally and nothing is silently orphaned --
        # worst case it's an empty-looking collection, never a vanished
        # one with real data quietly left behind under an id nobody can
        # see anymore. Reversing this order was a deliberate fix after an
        # earlier version ran the registry delete first.
        stats = None
        if delete_data:
            try:
                stats = self.retriever.delete_collection_data(slug_str)
            except Exception as e:
                logger.error(f"delete_collection: data wipe failed for {slug_str}: {e}")
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=(
                        f"Deleting the underlying data for '{slug_str}' "
                        f"failed: {e}. Nothing was removed -- its registry "
                        "entry (if any) is still intact."
                    ),
                    metadata={"slug": slug_str},
                )

        try:
            deleted_row = self.retriever.delete_collection_registry_row(slug_str)
        except Exception as e:
            logger.error(
                f"delete_collection: registry delete failed for {slug_str}: {e}"
            )
            extra = (
                " Its underlying data has already been permanently deleted."
                if delete_data
                else ""
            )
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Could not delete collection '{slug_str}': {e}.{extra}",
                metadata={"slug": slug_str, "data_deleted": delete_data},
            )

        if deleted_row is None and not delete_data:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Collection '{slug_str}' is not registered -- nothing to delete.",
                metadata={"slug": slug_str, "registry_row_deleted": False},
            )

        registry_note = (
            f"Removed '{slug_str}' from the collection registry"
            if deleted_row is not None
            else f"No registry entry existed for '{slug_str}'"
        )
        data_note = ""
        if delete_data and stats is not None:
            data_note = (
                f". Permanently deleted {stats['chunks_deleted']} chunks, "
                f"{stats['chunk_edges_deleted']} graph edges, "
                f"{stats['papers_deleted']} orphaned papers, and "
                f"{stats['duckdb_rows_deleted']} embeddings"
            )

        metadata = {
            "slug": slug_str,
            "registry_row_deleted": deleted_row is not None,
            "data_deleted": delete_data,
        }
        if stats is not None:
            metadata["stats"] = stats

        return ToolResult(
            tool_name=self.name,
            success=True,
            data=f"{registry_note}{data_note}.",
            metadata=metadata,
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
                            "Collection id/slug to delete. 'global' cannot "
                            "be deleted."
                        ),
                    },
                    "delete_data": {
                        "type": "boolean",
                        "description": (
                            "If true, ALSO permanently delete every "
                            "paper/chunk/embedding stored under this "
                            "collection. Only honored for admin users -- "
                            "set this only if the user explicitly asks to "
                            "permanently delete the underlying data, not "
                            "by default."
                        ),
                    },
                },
                "required": ["slug"],
            },
        }
