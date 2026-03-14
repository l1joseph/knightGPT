"""OpenAlex search tool for KnightGPT agent system."""

import requests

from ..utils import get_logger
from .base import BaseTool, ToolResult

logger = get_logger(__name__)

OPENALEX_BASE = "https://api.openalex.org"


class OpenAlexTool(BaseTool):
    """Search OpenAlex for academic papers, authors, and concepts."""

    name = "openalex_search"
    description = (
        "Search OpenAlex for academic works, authors, and institutions. "
        "Returns papers with citation counts, open access info, and DOIs."
    )

    def __init__(self, max_results: int = 10, email: str = "knightgpt@ucsd.edu"):
        self.max_results = max_results
        self.session = requests.Session()
        self.session.headers["User-Agent"] = f"KnightGPT/1.0 (mailto:{email})"

    def execute(
        self,
        query: str,
        entity_type: str = "works",
        max_results: int | None = None,
        **kwargs,
    ) -> ToolResult:
        """Search OpenAlex."""
        max_results = max_results or self.max_results

        try:
            resp = self.session.get(
                f"{OPENALEX_BASE}/{entity_type}",
                params={
                    "search": query,
                    "per_page": max_results,
                    "sort": "relevance_score:desc",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            papers = []
            for work in results:
                authors = [
                    a.get("author", {}).get("display_name", "")
                    for a in work.get("authorships", [])[:5]
                ]
                primary_loc = work.get("primary_location") or {}
                source = primary_loc.get("source") or {}

                papers.append({
                    "openalex_id": work.get("id", ""),
                    "title": work.get("title", ""),
                    "doi": work.get("doi", ""),
                    "authors": ", ".join(authors),
                    "journal": source.get("display_name", ""),
                    "year": work.get("publication_year"),
                    "cited_by_count": work.get("cited_by_count", 0),
                    "is_oa": work.get("open_access", {}).get("is_oa", False),
                    "pdf_url": primary_loc.get("pdf_url"),
                    "type": work.get("type", ""),
                })

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=papers,
                metadata={
                    "query": query,
                    "total_count": data.get("meta", {}).get("count", 0),
                    "returned": len(papers),
                },
            )

        except Exception as e:
            logger.error(f"OpenAlex search failed: {e}")
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "entity_type": {
                        "type": "string",
                        "enum": ["works", "authors", "sources"],
                        "description": "Entity type to search (default: works)",
                    },
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
        }
