"""KEGG pathway/compound lookup tool for KnightGPT agent system."""

import requests

from ..utils import get_logger
from .base import BaseTool, ToolResult

logger = get_logger(__name__)

KEGG_BASE = "https://rest.kegg.jp"


class KEGGTool(BaseTool):
    """Query KEGG for pathways, compounds, and organisms."""

    name = "kegg_lookup"
    description = (
        "Search KEGG database for metabolic pathways, compounds, enzymes, "
        "and organisms. Useful for connecting microbiome functions to "
        "metabolic activity."
    )

    def __init__(self):
        self.session = requests.Session()

    def execute(
        self,
        query: str,
        database: str = "pathway",
        **kwargs,
    ) -> ToolResult:
        """Search KEGG."""
        try:
            # KEGG find endpoint
            resp = self.session.get(
                f"{KEGG_BASE}/find/{database}/{query}",
                timeout=15,
            )
            resp.raise_for_status()

            results = []
            for line in resp.text.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("\t", 1)
                entry_id = parts[0] if parts else ""
                description = parts[1] if len(parts) > 1 else ""
                results.append({"id": entry_id, "description": description})

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=results[:20],
                metadata={"query": query, "database": database, "total": len(results)},
            )

        except Exception as e:
            logger.error(f"KEGG search failed: {e}")
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    def get_entry(self, entry_id: str) -> ToolResult:
        """Get detailed KEGG entry."""
        try:
            resp = self.session.get(f"{KEGG_BASE}/get/{entry_id}", timeout=15)
            resp.raise_for_status()
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=resp.text[:3000],
                metadata={"entry_id": entry_id},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"},
                    "database": {
                        "type": "string",
                        "enum": ["pathway", "compound", "enzyme", "organism", "module"],
                        "description": "KEGG database (default: pathway)",
                    },
                },
                "required": ["query"],
            },
        }
