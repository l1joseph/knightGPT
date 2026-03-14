"""QIIME 2 documentation and method lookup tool for KnightGPT."""

import requests

from ..utils import get_logger
from .base import BaseTool, ToolResult

logger = get_logger(__name__)


class QIIME2Tool(BaseTool):
    """Look up QIIME 2 plugins, methods, and documentation."""

    name = "qiime2_docs"
    description = (
        "Search QIIME 2 documentation for plugins, methods, pipelines, "
        "and tutorials. Useful for answering questions about microbiome "
        "analysis workflows, parameters, and best practices."
    )

    # Core QIIME 2 plugin info (static, updated periodically)
    PLUGINS = {
        "diversity": {
            "description": "Alpha and beta diversity analyses",
            "key_methods": [
                "alpha", "beta", "alpha-group-significance",
                "beta-group-significance", "core-metrics",
                "core-metrics-phylogenetic",
            ],
        },
        "feature-table": {
            "description": "Feature table manipulation and summarization",
            "key_methods": [
                "filter-samples", "filter-features", "summarize",
                "rarefy", "merge", "group",
            ],
        },
        "taxa": {
            "description": "Taxonomic classification and visualization",
            "key_methods": ["barplot", "collapse", "filter-table"],
        },
        "phylogeny": {
            "description": "Phylogenetic tree construction",
            "key_methods": [
                "align-to-tree-mafft-fasttree", "fasttree", "raxml",
            ],
        },
        "composition": {
            "description": "Compositional data analysis (ANCOM, etc.)",
            "key_methods": ["ancombc", "add-pseudocount"],
        },
        "emperor": {
            "description": "Interactive 3D ordination visualization",
            "key_methods": ["plot", "biplot"],
        },
        "demux": {
            "description": "Demultiplexing and quality control",
            "key_methods": ["emp-single", "emp-paired", "summarize"],
        },
        "dada2": {
            "description": "Denoising with DADA2",
            "key_methods": ["denoise-single", "denoise-paired"],
        },
        "deblur": {
            "description": "Denoising with Deblur",
            "key_methods": ["denoise-16S", "denoise-other"],
        },
        "feature-classifier": {
            "description": "Taxonomic classification of sequences",
            "key_methods": [
                "classify-sklearn", "classify-consensus-blast",
                "fit-classifier-naive-bayes",
            ],
        },
        "longitudinal": {
            "description": "Longitudinal study analysis",
            "key_methods": [
                "volatility", "linear-mixed-effects",
                "first-differences", "pairwise-differences",
            ],
        },
        "sample-classifier": {
            "description": "Machine learning on microbiome data",
            "key_methods": [
                "classify-samples", "regress-samples",
                "fit-classifier", "predict-classification",
            ],
        },
        "gemelli": {
            "description": "Compositional tensor factorization (CTF/RPCA)",
            "key_methods": ["ctf", "rpca", "auto-rpca"],
        },
    }

    def __init__(self):
        self.session = requests.Session()

    def execute(self, query: str, **kwargs) -> ToolResult:
        """Search QIIME 2 plugins and methods."""
        query_lower = query.lower()
        matches = []

        for plugin_name, info in self.PLUGINS.items():
            # Match by plugin name or description
            if (
                query_lower in plugin_name
                or query_lower in info["description"].lower()
            ):
                matches.append({
                    "plugin": plugin_name,
                    "description": info["description"],
                    "methods": info["key_methods"],
                    "docs_url": f"https://docs.qiime2.org/plugins/available/q2-{plugin_name}/",
                })
                continue

            # Match by method name
            for method in info["key_methods"]:
                if query_lower in method:
                    matches.append({
                        "plugin": plugin_name,
                        "method": method,
                        "description": info["description"],
                        "docs_url": f"https://docs.qiime2.org/plugins/available/q2-{plugin_name}/",
                    })

        if not matches:
            # Return general info
            matches = [
                {
                    "info": "No specific match found. Available plugins:",
                    "plugins": list(self.PLUGINS.keys()),
                    "docs_url": "https://docs.qiime2.org/",
                }
            ]

        return ToolResult(
            tool_name=self.name,
            success=True,
            data=matches,
            metadata={"query": query},
        )

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "QIIME 2 plugin, method, or concept to look up",
                    },
                },
                "required": ["query"],
            },
        }
