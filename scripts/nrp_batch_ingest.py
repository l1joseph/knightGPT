#!/usr/bin/env python3
"""NRP batch ingestion orchestrator.

Runs a full ingestion of all four paper source lists into the NRP
Postgres+DuckDB storage, in fixed-size batches through chunk -> embed ->
insert (not via scripts/ingest_pipeline.py::run_pipeline(), which would
redundantly reconvert PDFs download_papers() already converted -- see
docs/superpowers/plans/2026-08-06-nrp-ingestion-job.md's Global
Constraints for why).
"""

from pathlib import Path


def partition_into_batches(items: list, batch_size: int) -> list[list]:
    """Split items into consecutive batches of at most batch_size each.

    The final batch may be smaller than batch_size if len(items) isn't an
    exact multiple. Returns an empty list for empty input.
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
