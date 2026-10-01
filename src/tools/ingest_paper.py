"""Ingest-paper agent tool for KnightGPT agent system.

Lets a chat user ask to add a specific paper to the live corpus by DOI
(e.g. "please add this paper to the corpus: 10.1038/s41586-023-12345-6").
Reuses the same download/chunk/embed/insert building blocks the batch
ingestion scripts use, rather than re-deriving them:

- Download/full-text resolution: scripts.download_papers.download_single_paper
  (Unpaywall -> PMC full-text XML -> DOI-redirect scrape), the same
  per-DOI logic download_papers() runs for a whole DOI-list file.
- Chunking: src.chunking.SemanticChunker (same as src/api/main.py's
  /api/v1/ingest and scripts/ingest_pipeline.py).
- Embedding: src.embedding.VLLMEmbedder (same as every other ingestion
  call site).
- Postgres+DuckDB+pgGraph insert: HybridRetriever.insert_paper(), which
  reuses the orchestrator's EXISTING pool/DuckDBStore/private event-loop
  bridge (see src/retrieval/hybrid_retriever.py) instead of opening a
  second DuckDB connection -- DuckDBStore supports exactly one read-write
  connection per file, and the API server's single _duckdb_store
  singleton already holds it for the process lifetime.

execute() runs synchronously inside AgentOrchestrator.run(), which itself
is dispatched to a worker thread via run_in_threadpool from the async
/v1/chat/completions and /api/v1/agent/chat handlers (see
src/api/main.py) -- so ordinary blocking I/O here (requests.get, disk
writes) is safe and does not block the event loop. Only the final
Postgres/DuckDB insert needs the HybridRetriever bridge, which is itself
safe to call from a thread already inside a running event loop.
"""

import re
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from ..chunking import SemanticChunker
from ..embedding import VLLMEmbedder
from ..ingestion.web_scraper import MicrobiomeScraper
from ..utils import get_logger, get_settings
from .base import BaseTool, ToolResult

if TYPE_CHECKING:
    # Deferred at runtime (see execute()'s body) -- a module-level
    # `from ..api.request_context import RequestContext` would be a
    # circular import: src.api.__init__ imports .main, which imports
    # src.agents, which imports this module (same pattern as
    # src/tools/base.py and src/agents/orchestrator.py).
    from ..api.request_context import RequestContext

logger = get_logger(__name__)
settings = get_settings()

_DOI_URL_PREFIX = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)


class IngestPaperTool(BaseTool):
    """Download a paper by DOI and add it to the live KnightGPT corpus."""

    name = "ingest_paper"
    description = (
        "Add a specific research paper to the knowledge base by its DOI "
        "(or a doi.org URL). Downloads the paper's full text, converts, "
        "chunks, and embeds it, then inserts it into the live corpus so "
        "it becomes searchable and citable in future answers. Use this "
        "when the user explicitly asks to add, ingest, or include a "
        "particular paper (identified by DOI) in the corpus -- not for "
        "general literature search (use pubmed_search or openalex_search "
        "for that)."
    )

    def __init__(self, retriever=None):
        """
        Args:
            retriever: a HybridRetriever instance (or any object exposing
                the same insert_paper() method) that owns the live
                Postgres pool + DuckDB store to insert into. Normally the
                same AgentOrchestrator.retriever this tool is registered
                alongside, so no second DB connection is ever opened.
        """
        self.retriever = retriever

    def execute(
        self,
        query: str,
        doi: str | None = None,
        also_global: bool = False,
        *,
        request_context: "RequestContext | None" = None,
        **kwargs,
    ) -> ToolResult:
        """Download, chunk, embed, and insert one paper by DOI, scoped to
        the collection attached to the current chat (or the global
        corpus for an admin with also_global or no collection attached).

        Args:
            query: the DOI or doi.org URL (accepted here for consistency
                with BaseTool's default schema / the orchestrator's
                dispatch convention of always passing the model's "query"
                argument positionally).
            doi: same as query, offered as an explicit alternative in
                case the model supplies it as a named "doi" argument
                instead (this tool's schema asks for "doi").
            also_global: if True, ALSO insert into the global corpus in
                addition to the attached collection. Honored only when
                request_context.is_admin is True -- anyone else setting
                it gets a clear ToolResult(success=False, ...) error,
                never silent ignoring (see the spec's Decisions section).
            request_context: identity + collection scope, injected by
                AgentOrchestrator.run() -- never sourced from this
                method's own **kwargs even if the model's JSON
                tool-call arguments happen to include a collection_id-
                or request_context-shaped key.
        """
        from ..api.request_context import RequestContext

        ctx = request_context or RequestContext()

        doi_str = (doi or query or "").strip()
        doi_str = _DOI_URL_PREFIX.sub("", doi_str)

        if not doi_str:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="No DOI was provided.",
            )

        if also_global and not ctx.is_admin:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="Only admin can add to the global corpus.",
                metadata={"doi": doi_str},
            )

        if ctx.collection_id is not None:
            primary_collection_id: str | None = ctx.collection_id
            write_global_too = also_global and ctx.is_admin
        elif ctx.is_admin:
            # No collection attached, admin caller: default to global,
            # preserving today's existing single-corpus behavior for
            # casual admin use. Already global -- no second write needed
            # even if also_global was also set.
            primary_collection_id = None
            write_global_too = False
        else:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    "No Knowledge collection is attached to this chat, and "
                    "you are not an admin. Attach a collection in Open WebUI "
                    "before adding a paper to the corpus, or ask an admin to "
                    "add it to the global corpus."
                ),
                metadata={"doi": doi_str},
            )

        if self.retriever is None or not hasattr(self.retriever, "insert_paper"):
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    "Paper ingestion is unavailable: no corpus connection "
                    "(HybridRetriever) is configured for this tool."
                ),
                metadata={"doi": doi_str},
            )

        try:
            # Deferred import: scripts/download_papers.py runs
            # get_settings() at module import time and sys.path.inserts
            # the repo root, same reason src/ingestion/doi_resolver.py
            # defers its `from scripts.download_papers import
            # parse_doi_file` -- importing scripts/ from src/ (the
            # reverse of the usual scripts/ -> src/ direction) is only
            # safe deferred to call time, not module load time.
            from scripts.download_papers import download_single_paper
        except Exception as e:
            logger.error(f"ingest_paper: could not import download_papers: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Ingestion pipeline is unavailable: {e}",
                metadata={"doi": doi_str},
            )

        try:
            download_dir = Path(settings.ingestion.raw_pdf_dir)
            markdown_dir = Path(settings.ingestion.markdown_dir)
            download_dir.mkdir(parents=True, exist_ok=True)
            markdown_dir.mkdir(parents=True, exist_ok=True)

            scraper = MicrobiomeScraper(
                output_dir=markdown_dir, download_dir=download_dir
            )
            result = download_single_paper(doi_str, scraper, download_dir, markdown_dir)
        except Exception as e:
            logger.error(f"ingest_paper: download failed for {doi_str}: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Download failed for DOI {doi_str}: {e}",
                metadata={"doi": doi_str},
            )

        if result["status"] == "failed":
            reason = result["reason"]
            human_reason = (
                "no open-access full text could be found (Unpaywall, PMC, "
                "and DOI-page scraping all failed)"
                if reason == "no_fulltext_found"
                else "a PDF was found but downloading/converting it failed"
            )
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Could not obtain the full text for DOI {doi_str}: {human_reason}.",
                metadata={"doi": doi_str, "reason": reason},
            )

        doc = result.get("doc")
        if result["status"] == "skipped" or doc is None or doc.file_path is None:
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=(
                    f"DOI {doi_str} was already downloaded previously "
                    "(likely already in the corpus); no new ingestion was performed."
                ),
                metadata={"doi": doi_str, "status": "already_downloaded"},
            )

        try:
            chunker = SemanticChunker()
            chunks = chunker.chunk_markdown_file(
                doc.file_path, metadata={"doi": doi_str, "title": doc.title}
            )
        except Exception as e:
            logger.error(f"ingest_paper: chunking failed for {doi_str}: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Downloaded {doi_str} but chunking its text failed: {e}",
                metadata={"doi": doi_str},
            )

        if not chunks:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    f"Downloaded {doi_str} but it produced no usable text "
                    "chunks (empty or unparseable full text)."
                ),
                metadata={"doi": doi_str},
            )

        try:
            embedder = VLLMEmbedder()
            if not embedder.check_health():
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=(
                        "The embedding server is unavailable, so "
                        f"{doi_str} could not be added to the corpus."
                    ),
                    metadata={"doi": doi_str},
                )
            chunks = embedder.embed_chunks(chunks)
        except Exception as e:
            logger.error(f"ingest_paper: embedding failed for {doi_str}: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Downloaded {doi_str} but embedding it failed: {e}",
                metadata={"doi": doi_str},
            )

        embedded_count = sum(1 for c in chunks if c.embedding)
        if embedded_count == 0:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    f"Downloaded {doi_str} but no chunks could be embedded "
                    "-- nothing was added to the corpus."
                ),
                metadata={"doi": doi_str},
            )

        try:
            insert_stats = self.retriever.insert_paper(
                doi=doi_str,
                chunks=chunks,
                title=doc.title,
                collection_id=primary_collection_id,
            )
            if write_global_too:
                # Same content, DERIVED chunk ids (":global" suffix) --
                # chunks.id/chunk_embeddings.id are PRIMARY KEY columns,
                # so reusing the exact same id for a second collection's
                # row would make this second write a silent
                # ON CONFLICT DO NOTHING no-op. Text/embedding are reused
                # unchanged -- no second chunk/embed pass (see the
                # plan's design note on this double-write).
                global_chunks = [replace(c, id=f"{c.id}:global") for c in chunks]
                global_stats = self.retriever.insert_paper(
                    doi=doi_str,
                    chunks=global_chunks,
                    title=doc.title,
                    collection_id="global",
                )
                insert_stats = {
                    "papers_inserted": insert_stats.get("papers_inserted", 0)
                    + global_stats.get("papers_inserted", 0),
                    "chunks_inserted": insert_stats.get("chunks_inserted", 0)
                    + global_stats.get("chunks_inserted", 0),
                    "edges_inserted": insert_stats.get("edges_inserted", 0)
                    + global_stats.get("edges_inserted", 0),
                }
        except Exception as e:
            logger.error(f"ingest_paper: insert failed for {doi_str}: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    f"Downloaded and embedded {doi_str}, but inserting it "
                    f"into the corpus failed: {e}"
                ),
                metadata={"doi": doi_str},
            )

        chunks_inserted = insert_stats.get("chunks_inserted", 0)
        edges_inserted = insert_stats.get("edges_inserted", 0)
        title_display = doc.title or doi_str
        also_global_note = (
            " (also added to the global corpus)" if write_global_too else ""
        )

        return ToolResult(
            tool_name=self.name,
            success=True,
            data=(
                f"Added '{title_display}' ({doi_str}) to the corpus{also_global_note}: "
                f"{chunks_inserted} chunks inserted, {edges_inserted} "
                "graph edges created."
            ),
            metadata={
                "doi": doi_str,
                "title": doc.title,
                "source": result.get("source"),
                "collection_id": primary_collection_id,
                "also_global": write_global_too,
                **insert_stats,
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
                    "doi": {
                        "type": "string",
                        "description": (
                            "The paper's DOI, e.g. '10.1038/s41586-023-12345-6', "
                            "or a doi.org URL."
                        ),
                    },
                    "also_global": {
                        "type": "boolean",
                        "description": (
                            "If true, also add this paper to the global corpus "
                            "(shared by every collection) in addition to the "
                            "currently attached collection. Only honored for "
                            "admin users -- set this only if the user "
                            "explicitly asks to add a paper globally/for "
                            "everyone, not by default."
                        ),
                    },
                },
                "required": ["doi"],
            },
        }
