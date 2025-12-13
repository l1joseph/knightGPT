#!/usr/bin/env python3
"""
Health check script for KnightGPT services.

Checks:
- API server status
- vLLM embedding server
- vLLM inference server
- Data availability
- System resources
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import requests
from rich.console import Console
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils import get_logger, get_settings

logger = get_logger(__name__)
console = Console()
settings = get_settings()


class HealthChecker:
    """Health check for all KnightGPT services."""

    def __init__(self, api_url: Optional[str] = None):
        self.api_url = api_url or f"http://{settings.api.host}:{settings.api.port}"
        self.results = {}

    def check_api_health(self) -> Dict:
        """Check API server health."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            response.raise_for_status()
            data = response.json()
            return {
                "status": "healthy" if data.get("status") == "healthy" else "degraded",
                "embedding_server": data.get("embedding_server", False),
                "inference_server": data.get("inference_server", False),
                "chunks_loaded": data.get("chunks_loaded", 0),
                "graph_nodes": data.get("graph_nodes", 0),
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "embedding_server": False,
                "inference_server": False,
                "chunks_loaded": 0,
                "graph_nodes": 0,
            }

    def check_embedding_server(self) -> Dict:
        """Check vLLM embedding server."""
        try:
            url = settings.vllm.embedding_url.replace("/v1", "/v1/models")
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return {"status": "healthy", "url": settings.vllm.embedding_url}
        except requests.exceptions.RequestException as e:
            return {"status": "unhealthy", "error": str(e), "url": settings.vllm.embedding_url}

    def check_inference_server(self) -> Dict:
        """Check vLLM inference server."""
        try:
            url = settings.vllm.inference_url.replace("/v1", "/v1/models")
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return {"status": "healthy", "url": settings.vllm.inference_url}
        except requests.exceptions.RequestException as e:
            return {"status": "unhealthy", "error": str(e), "url": settings.vllm.inference_url}

    def check_data_files(self) -> Dict:
        """Check if required data files exist."""
        chunks_file = settings.ingestion.processed_dir / "chunks_with_emb.json"
        graph_file = settings.graph.graph_path

        result = {
            "chunks_file": {
                "exists": chunks_file.exists(),
                "path": str(chunks_file),
            },
            "graph_file": {
                "exists": graph_file.exists(),
                "path": str(graph_file),
            },
        }

        # Check file sizes if they exist
        if chunks_file.exists():
            result["chunks_file"]["size_mb"] = chunks_file.stat().st_size / (1024 * 1024)

        if graph_file.exists():
            result["graph_file"]["size_mb"] = graph_file.stat().st_size / (1024 * 1024)

        return result

    def check_neo4j(self) -> Dict:
        """Check Neo4j connection (optional)."""
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                settings.neo4j.uri,
                auth=(settings.neo4j.user, settings.neo4j.password),
            )

            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()

            driver.close()
            return {"status": "healthy", "uri": settings.neo4j.uri}
        except ImportError:
            return {"status": "not_configured", "message": "neo4j package not installed"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "uri": settings.neo4j.uri}

    def run_all_checks(self) -> Dict:
        """Run all health checks."""
        console.print("[bold blue]Running Health Checks...[/bold blue]\n")

        results = {
            "timestamp": time.time(),
            "api": self.check_api_health(),
            "embedding_server": self.check_embedding_server(),
            "inference_server": self.check_inference_server(),
            "data_files": self.check_data_files(),
            "neo4j": self.check_neo4j(),
        }

        self.results = results
        return results

    def print_summary(self):
        """Print health check summary."""
        table = Table(title="KnightGPT Health Check Summary")

        table.add_column("Service", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Details", style="green")

        # API
        api_status = self.results["api"]["status"]
        status_emoji = "✅" if api_status == "healthy" else "⚠️" if api_status == "degraded" else "❌"
        api_details = f"Chunks: {self.results['api']['chunks_loaded']}, Graph: {self.results['api']['graph_nodes']}"
        table.add_row("API Server", f"{status_emoji} {api_status}", api_details)

        # Embedding Server
        emb_status = self.results["embedding_server"]["status"]
        emb_emoji = "✅" if emb_status == "healthy" else "❌"
        emb_details = self.results["embedding_server"].get("url", "N/A")
        table.add_row("Embedding Server", f"{emb_emoji} {emb_status}", emb_details)

        # Inference Server
        inf_status = self.results["inference_server"]["status"]
        inf_emoji = "✅" if inf_status == "healthy" else "❌"
        inf_details = self.results["inference_server"].get("url", "N/A")
        table.add_row("Inference Server", f"{inf_emoji} {inf_status}", inf_details)

        # Data Files
        chunks_exists = self.results["data_files"]["chunks_file"]["exists"]
        graph_exists = self.results["data_files"]["graph_file"]["exists"]
        data_status = "✅ Available" if chunks_exists and graph_exists else "⚠️ Partial" if chunks_exists or graph_exists else "❌ Missing"
        data_details = f"Chunks: {'✓' if chunks_exists else '✗'}, Graph: {'✓' if graph_exists else '✗'}"
        table.add_row("Data Files", data_status, data_details)

        # Neo4j
        neo4j_status = self.results["neo4j"]["status"]
        neo4j_emoji = "✅" if neo4j_status == "healthy" else "⚪" if neo4j_status == "not_configured" else "❌"
        neo4j_details = self.results["neo4j"].get("uri", "N/A")
        table.add_row("Neo4j", f"{neo4j_emoji} {neo4j_status}", neo4j_details)

        console.print(table)

        # Overall status
        all_healthy = (
            self.results["api"]["status"] == "healthy"
            and self.results["embedding_server"]["status"] == "healthy"
            and self.results["inference_server"]["status"] == "healthy"
            and self.results["data_files"]["chunks_file"]["exists"]
        )

        if all_healthy:
            console.print("\n[bold green]✓ All systems operational[/bold green]")
        else:
            console.print("\n[bold yellow]⚠ Some services need attention[/bold yellow]")

    def save_report(self, output_path: Path):
        """Save health check report to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        console.print(f"\n[dim]Report saved to: {output_path}[/dim]")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="KnightGPT Health Check")
    parser.add_argument("--api-url", type=str, help="API server URL")
    parser.add_argument("--output", "-o", type=Path, help="Save report to JSON file")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    checker = HealthChecker(api_url=args.api_url)
    results = checker.run_all_checks()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        checker.print_summary()

    if args.output:
        checker.save_report(args.output)

    # Exit code: 0 if healthy, 1 if degraded, 2 if unhealthy
    api_status = results["api"]["status"]
    if api_status == "healthy":
        sys.exit(0)
    elif api_status == "degraded":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()

