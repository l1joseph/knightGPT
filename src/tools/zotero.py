"""Zotero collections tool for KnightGPT agent system.

Connects to Zotero libraries to browse collections, search items,
and extract DOIs/PDF URLs for ingestion into the KnightGPT pipeline.
"""

import os
from pathlib import Path
from typing import Optional

from pyzotero import zotero

from ..utils import get_logger
from .base import BaseTool, ToolResult

logger = get_logger(__name__)


class ZoteroTool(BaseTool):
    """Browse and search Zotero libraries and collections.

    Supports listing collections, fetching items from collections,
    searching across the library, and extracting DOIs for pipeline ingestion.
    """

    name = "zotero"
    description = (
        "Browse Zotero library collections and search for papers. "
        "Can list collections, get items from a collection, search the library, "
        "and extract DOIs for ingestion."
    )

    def __init__(
        self,
        library_id: str | None = None,
        library_type: str | None = None,
        api_key: str | None = None,
    ):
        self.library_id = library_id or os.environ.get("ZOTERO_LIBRARY_ID", "")
        self.library_type = library_type or os.environ.get("ZOTERO_LIBRARY_TYPE", "user")
        self.api_key = api_key or os.environ.get("ZOTERO_API_KEY", "")

        if not self.library_id or not self.api_key:
            raise ValueError(
                "Zotero credentials required. Set ZOTERO_LIBRARY_ID and "
                "ZOTERO_API_KEY in .env or pass directly."
            )

        self.zot = zotero.Zotero(self.library_id, self.library_type, self.api_key)

    def execute(
        self,
        query: str = "",
        action: str = "collections",
        collection_id: str | None = None,
        max_results: int = 50,
        **kwargs,
    ) -> ToolResult:
        """Execute a Zotero action.

        Args:
            query: Search query (for 'search' action)
            action: One of 'collections', 'items', 'search', 'dois'
            collection_id: Collection key (for 'items' and 'dois' actions)
            max_results: Max items to return
        """
        try:
            if action == "collections":
                return self._list_collections()
            elif action == "items":
                return self._get_collection_items(collection_id, max_results)
            elif action == "search":
                return self._search_library(query, max_results)
            elif action == "dois":
                return self._extract_dois(collection_id, max_results)
            else:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Unknown action: {action}. Use: collections, items, search, dois",
                )
        except Exception as e:
            logger.error(f"Zotero {action} failed: {e}")
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    def _list_collections(self) -> ToolResult:
        """List all collections in the library."""
        collections = self.zot.collections()
        result = []
        for c in collections:
            data = c["data"]
            result.append({
                "key": data["key"],
                "name": data["name"],
                "num_items": data.get("numItems", 0),
                "parent": data.get("parentCollection", None),
            })

        # Sort by name
        result.sort(key=lambda x: x["name"])

        return ToolResult(
            tool_name=self.name,
            success=True,
            data=result,
            metadata={"action": "collections", "count": len(result)},
        )

    def _get_collection_items(
        self, collection_id: str | None, max_results: int
    ) -> ToolResult:
        """Get items from a specific collection."""
        if not collection_id:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="collection_id required for 'items' action",
            )

        items = self.zot.collection_items(collection_id, limit=max_results)
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=self._format_items(items),
            metadata={
                "action": "items",
                "collection_id": collection_id,
                "count": len(items),
            },
        )

    def _search_library(self, query: str, max_results: int) -> ToolResult:
        """Search across the entire library."""
        if not query:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="query required for 'search' action",
            )

        items = self.zot.items(q=query, limit=max_results)
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=self._format_items(items),
            metadata={"action": "search", "query": query, "count": len(items)},
        )

    def _extract_dois(
        self, collection_id: str | None, max_results: int
    ) -> ToolResult:
        """Extract DOIs from a collection (or entire library) for pipeline ingestion."""
        if collection_id:
            items = self.zot.collection_items(collection_id, limit=max_results)
        else:
            items = self.zot.items(limit=max_results)

        dois = []
        items_without_doi = []
        for item in items:
            data = item.get("data", {})
            item_type = data.get("itemType", "")
            if item_type in ("attachment", "note"):
                continue

            doi = data.get("DOI", "")
            title = data.get("title", "")
            if doi:
                dois.append({"doi": doi, "title": title})
            elif title:
                # Try to find DOI in extra field
                extra = data.get("extra", "")
                if "DOI:" in extra:
                    doi_match = extra.split("DOI:")[1].strip().split()[0]
                    dois.append({"doi": doi_match, "title": title})
                else:
                    items_without_doi.append(title)

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"dois": dois, "items_without_doi": items_without_doi},
            metadata={
                "action": "dois",
                "collection_id": collection_id,
                "doi_count": len(dois),
                "missing_doi_count": len(items_without_doi),
            },
        )

    def _format_items(self, items: list) -> list[dict]:
        """Format Zotero items into a clean list."""
        result = []
        for item in items:
            data = item.get("data", {})
            item_type = data.get("itemType", "")
            if item_type in ("attachment", "note"):
                continue

            creators = data.get("creators", [])
            authors = ", ".join(
                f"{c.get('lastName', '')}, {c.get('firstName', '')}"
                for c in creators[:5]
                if c.get("creatorType") == "author"
            )

            result.append({
                "key": data.get("key", ""),
                "title": data.get("title", ""),
                "authors": authors,
                "year": data.get("date", "")[:4] if data.get("date") else "",
                "doi": data.get("DOI", ""),
                "item_type": item_type,
                "journal": data.get("publicationTitle", ""),
                "url": data.get("url", ""),
                "abstract": data.get("abstractNote", "")[:200] if data.get("abstractNote") else "",
                "tags": [t.get("tag", "") for t in data.get("tags", [])],
            })
        return result

    def save_dois_to_file(
        self,
        collection_id: str | None = None,
        output_path: Path | None = None,
        max_results: int = 200,
    ) -> Path:
        """Extract DOIs from Zotero and save to a DOI list file for the pipeline.

        Args:
            collection_id: Optional collection to extract from (None = whole library)
            output_path: Where to save the file
            max_results: Max items to fetch

        Returns:
            Path to the saved DOI file
        """
        result = self._extract_dois(collection_id, max_results)
        if not result.success:
            raise RuntimeError(f"Failed to extract DOIs: {result.error}")

        dois_data = result.data["dois"]
        if not dois_data:
            raise RuntimeError("No DOIs found in the specified collection/library")

        output_path = output_path or Path("data/paper_lists/zotero_papers.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# KnightGPT Paper List — Imported from Zotero",
            f"# Collection: {collection_id or 'entire library'}",
            f"# {len(dois_data)} papers with DOIs",
            "",
        ]
        for entry in dois_data:
            lines.append(f"# {entry['title']}")
            lines.append(entry["doi"])
            lines.append("")

        output_path.write_text("\n".join(lines))
        logger.info(f"Saved {len(dois_data)} DOIs to {output_path}")
        return output_path

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "action": {
                        "type": "string",
                        "enum": ["collections", "items", "search", "dois"],
                        "description": "Action to perform (default: collections)",
                    },
                    "collection_id": {
                        "type": "string",
                        "description": "Zotero collection key",
                    },
                    "max_results": {"type": "integer"},
                },
                "required": [],
            },
        }
