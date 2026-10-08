"""Web search tool for KnightGPT agent system.

Backed by a self-hosted SearXNG instance (see docker/docker-compose.yaml's
searxng service and docker/searxng/settings.yml) rather than a third-party
search API, consistent with this project's self-hosted-first infra.
"""

import requests

from ..utils import get_logger, get_settings
from .base import BaseTool, ToolResult

logger = get_logger(__name__)
settings = get_settings()


class WebSearchTool(BaseTool):
    """Search the web via a self-hosted SearXNG instance."""

    name = "web_search"
    description = (
        "Search the web for current information not covered by PubMed, "
        "OpenAlex, KEGG, or the ingested corpus. Returns titles, URLs, "
        "and snippets. Use web_fetch afterward to read a specific result "
        "in full."
    )

    def __init__(self, base_url: str | None = None, max_results: int = 10):
        self.base_url = base_url or settings.searxng.url
        self.max_results = max_results
        self.session = requests.Session()

    def execute(self, query: str, count: int | None = None, **kwargs) -> ToolResult:
        """Search SearXNG and return the top results."""
        count = count or self.max_results

        try:
            resp = self.session.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                }
                for r in data.get("results", [])[:count]
            ]

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=results,
                metadata={"query": query, "total_results": len(results)},
            )

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Web search query"},
                    "count": {
                        "type": "integer",
                        "description": "Maximum results to return (default 10)",
                    },
                },
                "required": ["query"],
            },
        }
