"""PubMed search tool for KnightGPT agent system."""

import requests

from ..utils import get_logger
from .base import BaseTool, ToolResult

logger = get_logger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedTool(BaseTool):
    """Search PubMed for biomedical literature."""

    name = "pubmed_search"
    description = (
        "Search PubMed for biomedical and microbiome research papers. "
        "Returns titles, abstracts, DOIs, and PMIDs."
    )

    def __init__(self, max_results: int = 10, email: str = "knightgpt@ucsd.edu"):
        self.max_results = max_results
        self.email = email
        self.session = requests.Session()

    def execute(self, query: str, max_results: int | None = None, **kwargs) -> ToolResult:
        """Search PubMed and return paper summaries."""
        max_results = max_results or self.max_results

        try:
            # Step 1: Search for IDs
            search_resp = self.session.get(
                f"{EUTILS_BASE}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json",
                    "email": self.email,
                },
                timeout=15,
            )
            search_resp.raise_for_status()
            ids = search_resp.json().get("esearchresult", {}).get("idlist", [])

            if not ids:
                return ToolResult(
                    tool_name=self.name, success=True, data=[], metadata={"query": query}
                )

            # Step 2: Fetch summaries
            summary_resp = self.session.get(
                f"{EUTILS_BASE}/esummary.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "json",
                    "email": self.email,
                },
                timeout=15,
            )
            summary_resp.raise_for_status()
            result_data = summary_resp.json().get("result", {})

            papers = []
            for pmid in ids:
                doc = result_data.get(pmid, {})
                papers.append({
                    "pmid": pmid,
                    "title": doc.get("title", ""),
                    "authors": ", ".join(
                        a.get("name", "") for a in doc.get("authors", [])[:5]
                    ),
                    "journal": doc.get("fulljournalname", ""),
                    "year": doc.get("pubdate", "")[:4],
                    "doi": next(
                        (
                            aid["value"]
                            for aid in doc.get("articleids", [])
                            if aid.get("idtype") == "doi"
                        ),
                        None,
                    ),
                })

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=papers,
                metadata={"query": query, "total_results": len(papers)},
            )

        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "PubMed search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (default 10)",
                    },
                },
                "required": ["query"],
            },
        }
