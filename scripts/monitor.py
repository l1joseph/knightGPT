#!/usr/bin/env python3
"""
Continuous monitoring script for KnightGPT.

Monitors system health and sends alerts if issues are detected.
Can run as a daemon or cron job.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests
from rich.console import Console
from rich.live import Live
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.healthcheck import HealthChecker

console = Console()


class Monitor:
    """Continuous monitoring for KnightGPT."""

    def __init__(self, api_url: str = None, check_interval: int = 60, alert_threshold: int = 3):
        self.checker = HealthChecker(api_url=api_url)
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold
        self.failure_count = 0
        self.history: List[Dict] = []

    def check(self) -> Dict:
        """Run health check and return results."""
        results = self.checker.run_all_checks()
        results["timestamp"] = datetime.now().isoformat()
        self.history.append(results)

        # Keep only last 100 checks
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return results

    def is_healthy(self, results: Dict) -> bool:
        """Check if system is healthy."""
        return (
            results["api"]["status"] == "healthy"
            and results["embedding_server"]["status"] == "healthy"
            and results["inference_server"]["status"] == "healthy"
            and results["data_files"]["chunks_file"]["exists"]
        )

    def should_alert(self, results: Dict) -> bool:
        """Determine if alert should be sent."""
        if not self.is_healthy(results):
            self.failure_count += 1
            return self.failure_count >= self.alert_threshold
        else:
            self.failure_count = 0
            return False

    def send_alert(self, results: Dict):
        """Send alert notification."""
        console.print(f"\n[bold red]ALERT: System Health Issues Detected[/bold red]")
        console.print(f"API Status: {results['api']['status']}")
        console.print(f"Embedding Server: {results['embedding_server']['status']}")
        console.print(f"Inference Server: {results['inference_server']['status']}")

        # TODO: Add email/Slack/webhook notifications here
        # Example:
        # send_email_alert(results)
        # send_slack_webhook(results)

    def create_status_table(self, results: Dict) -> Table:
        """Create status table for display."""
        table = Table(title=f"KnightGPT Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        table.add_column("Service", style="cyan", width=20)
        table.add_column("Status", style="magenta", width=15)
        table.add_column("Details", style="green", width=30)

        # API
        api_status = results["api"]["status"]
        status_icon = "✅" if api_status == "healthy" else "⚠️" if api_status == "degraded" else "❌"
        api_details = f"Chunks: {results['api']['chunks_loaded']}"
        table.add_row("API Server", f"{status_icon} {api_status}", api_details)

        # Embedding
        emb_status = results["embedding_server"]["status"]
        emb_icon = "✅" if emb_status == "healthy" else "❌"
        table.add_row("Embedding", f"{emb_icon} {emb_status}", "")

        # Inference
        inf_status = results["inference_server"]["status"]
        inf_icon = "✅" if inf_status == "healthy" else "❌"
        table.add_row("Inference", f"{inf_icon} {inf_status}", "")

        # Data
        chunks_exists = results["data_files"]["chunks_file"]["exists"]
        data_icon = "✅" if chunks_exists else "❌"
        table.add_row("Data Files", f"{data_icon} {'Available' if chunks_exists else 'Missing'}", "")

        # Failure count
        table.add_row("Failures", f"{self.failure_count}/{self.alert_threshold}", "")

        return table

    def run_continuous(self):
        """Run continuous monitoring."""
        console.print("[bold blue]Starting KnightGPT Monitor[/bold blue]")
        console.print(f"Check interval: {self.check_interval}s")
        console.print(f"Alert threshold: {self.alert_threshold} failures\n")

        try:
            with Live(console=console, refresh_per_second=1) as live:
                while True:
                    results = self.check()

                    if self.should_alert(results):
                        self.send_alert(results)

                    table = self.create_status_table(results)
                    live.update(table)

                    time.sleep(self.check_interval)
        except KeyboardInterrupt:
            console.print("\n[dim]Monitoring stopped[/dim]")

    def save_history(self, output_path: Path):
        """Save monitoring history to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.history, f, indent=2)
        console.print(f"History saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="KnightGPT Monitor")
    parser.add_argument("--api-url", type=str, help="API server URL")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--threshold", type=int, default=3, help="Alert threshold (failures)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--output", type=Path, help="Save history to file")
    args = parser.parse_args()

    monitor = Monitor(
        api_url=args.api_url,
        check_interval=args.interval,
        alert_threshold=args.threshold,
    )

    if args.once:
        results = monitor.check()
        monitor.checker.print_summary()
        if args.output:
            monitor.save_history(args.output)
    else:
        monitor.run_continuous()
        if args.output:
            monitor.save_history(args.output)


if __name__ == "__main__":
    main()

