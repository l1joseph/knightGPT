"""Web fetch tool for KnightGPT agent system.

Lets the model read a full page after finding it via web_search (see
src/tools/websearch.py) -- fetches a URL and returns its plain-text
content, truncated to a manageable size for the LLM context.
"""

import re

import lxml.html
import requests

from ..utils import get_logger
from .base import BaseTool, ToolResult

logger = get_logger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


class WebFetchTool(BaseTool):
    """Fetch a URL and extract its plain-text content."""

    name = "web_fetch"
    description = (
        "Fetch a web page by URL and return its plain-text content. Use "
        "this to read a specific page in full after finding it via "
        "web_search."
    )

    def __init__(self):
        self.session = requests.Session()

    def execute(
        self,
        query: str,
        url: str | None = None,
        max_chars: int = 8000,
        **kwargs,
    ) -> ToolResult:
        """Fetch a URL and return collapsed plain text, truncated to max_chars.

        Args:
            query: the URL to fetch, accepted here for consistency with
                the orchestrator's dispatch convention of always passing
                the model's "query" argument positionally.
            url: same as query, offered as an explicit alternative in
                case the model supplies it as a named "url" argument
                instead (this tool's schema asks for "url").
            max_chars: truncate extracted text to this many characters
                (default 8000).
        """
        target_url = (url or query or "").strip()

        if not target_url:
            return ToolResult(
                tool_name=self.name, success=False, error="No URL was provided."
            )

        try:
            resp = self.session.get(target_url, timeout=15)
            resp.raise_for_status()

            doc = lxml.html.fromstring(resp.content)
            text = _WHITESPACE_RE.sub(" ", doc.text_content()).strip()

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=text,
                metadata={"url": target_url, "truncated": truncated},
            )

        except Exception as e:
            logger.error(f"Web fetch failed for {target_url}: {e}")
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters of extracted text to return (default 8000)",
                    },
                },
                "required": ["url"],
            },
        }
