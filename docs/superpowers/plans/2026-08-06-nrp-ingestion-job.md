# GPU-Backed Ingestion Job on NRP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A GPU-backed Kubernetes Job on NRP that runs a fresh, full ingestion of the entire paper corpus (all four source lists) into the already-deployed Postgres+DuckDB storage.

**Architecture:** A single Job with two containers — a native sidecar running `vllm/vllm-openai` (CUDA, 1 A100) serving the embedding model, and a main container running a new batch orchestrator. The orchestrator derives the long-read source's DOI list, downloads+converts all papers (reusing the existing, already-resumable `download_papers()`), then processes markdown files in fixed-size batches through chunk → embed → insert directly (not through `run_pipeline()`, which would redundantly reconvert PDFs `download_papers()` already converted — see Global Constraints).

**Tech Stack:** Kubernetes Job with native sidecar containers (stable since 1.29, confirmed supported on this cluster's v1.33.8), `vllm/vllm-openai:v0.26.0` (current stable CUDA release as of this plan), GitHub Actions → GHCR for the ingestion image, existing `src/chunking`, `src/embedding`, `src/graph` modules.

## Global Constraints

- Namespace: `knightlab-ml`, cluster context `nautilus`.
- GHCR image target for this plan: `ghcr.io/l1joseph/knightgpt-ingestion`.
- vLLM sidecar image: `vllm/vllm-openai:v0.26.0` (current stable release, confirmed via Docker Hub — NVIDIA CUDA build, NOT the ROCm image used on Cosmos, since NRP's GPUs are NVIDIA A100s).
- GPU resource: `nvidia.com/a100: 1` (this namespace has 16 allocated, 0 used — confirmed available; H100/H200/GH200 are all zero-quota here, do not request those).
- This namespace's Gatekeeper admission policy requires `resources.requests`/`resources.limits` on every container — budget for this on both the sidecar and main container in the Job manifest task.
- **Never test a workflow by pushing to the shared `vllm` branch.** Use `gh workflow run <file> --ref <this-feature-branch>` — confirmed to work correctly once the workflow file exists on that branch (learned the hard way during the storage deployment plan; do not repeat that incident).
- GitHub Actions `docker/metadata-action` tag rule for `latest` must be plain `type=raw,value=latest` with NO `enable={{is_default_branch}}` condition — this repo's GitHub-configured default branch is `main`, not `vllm`, so that condition would never fire (already discovered and fixed once this session in `build-postgres-image.yml`; do not reintroduce it here).
- **`run_pipeline()` must NOT be called directly by the new orchestrator.** `scripts/ingest_pipeline.py::run_pipeline()`'s Step 1 unconditionally globs and reconverts every PDF in `input_dir` via `batch_convert_pdfs`. `download_papers()` (via `MicrobiomeScraper._download_and_process()`) already downloads AND converts each PDF to markdown in one pass. Calling `run_pipeline()` afterward would redundantly reconvert every PDF a second time. The orchestrator instead composes the same underlying pieces `run_pipeline()` uses (`SemanticChunker.chunk_markdown_file()`, `VLLMEmbedder.embed_chunks()`, `insert_chunks()`) directly, scoped per-batch, starting from the markdown `download_papers()` already produced.
- Env vars (verified against `src/utils/config.py`): `POSTGRES_DSN`, `VLLM_EMBEDDING_URL`, `VLLM_EMBEDDING_MODEL`, `INGEST_RAW_PDF_DIR`, `INGEST_MARKDOWN_DIR`, `INGEST_PROCESSED_DIR`, `INGEST_DUCKDB_PATH`. No `.env` file needed — pydantic-settings falls back cleanly to process env vars. Per `k8s/nrp/README.md`'s documented trap, `INGEST_DUCKDB_PATH` must always be set explicitly alongside `INGEST_PROCESSED_DIR` (leaving both unset is the only genuinely broken combination, but setting the path explicitly is the safe habit regardless).
- Follow existing code conventions: type hints, Google-style docstrings, `get_logger(__name__)` from `..utils`/`.utils`, `pytest.mark.unit` on new tests.

---

### Task 1: Ingestion image — Dockerfile and GitHub Actions workflow

**Files:**
- Create: `docker/Dockerfile.ingestion`
- Create: `.github/workflows/build-ingestion-image.yml`

**Interfaces:**
- Produces: a container image at `ghcr.io/l1joseph/knightgpt-ingestion:<tag>`, consumed by Task 4's Job manifest.

- [ ] **Step 1: Create the Dockerfile**

```dockerfile
# docker/Dockerfile.ingestion
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/paper_lists/ ./data/paper_lists/

RUN mkdir -p data/raw_pdfs data/markdown data/processed logs && \
    chown -R appuser:appuser /app

USER appuser
```

No `CMD`/`ENTRYPOINT` — Task 4's Job manifest specifies the exact command to run, matching this Dockerfile's role as a general-purpose ingestion image rather than a single-purpose server image (unlike `Dockerfile.api`, which always runs the same server). `data/paper_lists/` is baked into the image (the checked-in DOI list `.txt` files and the long-read `sources/*.tsv` — small, versioned, needed at runtime) rather than mounted from a PVC, since these are static repo content, not generated data.

- [ ] **Step 2: Create the GitHub Actions workflow**

```yaml
# .github/workflows/build-ingestion-image.yml
name: Build and push ingestion image

on:
  push:
    branches: [vllm]
    paths:
      - "docker/Dockerfile.ingestion"
      - "requirements.txt"
      - "scripts/**"
      - "src/**"
      - "data/paper_lists/**"
      - ".github/workflows/build-ingestion-image.yml"
  workflow_dispatch: {}

env:
  IMAGE_NAME: ghcr.io/${{ github.repository_owner }}/knightgpt-ingestion

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.IMAGE_NAME }}
          tags: |
            type=sha,format=long
            type=raw,value=latest

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/Dockerfile.ingestion
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

Note the `tags:` block is `type=raw,value=latest` with no `enable={{is_default_branch}}` condition — per Global Constraints, that condition never fires on this repo (default branch is `main`, this workflow triggers on `vllm`), so including it would silently mean `latest` never gets produced (already discovered and fixed once this session for the Postgres image — this workflow is written correctly from the start).

- [ ] **Step 3: Commit**

```bash
git add docker/Dockerfile.ingestion .github/workflows/build-ingestion-image.yml
git commit -m "ci: build and push ingestion image to GHCR"
```

- [ ] **Step 4: Push the branch and trigger the workflow via workflow_dispatch**

```bash
git push -u origin feature/nrp-ingestion-job
gh workflow run build-ingestion-image.yml --ref feature/nrp-ingestion-job
```

Wait for it to complete:

```bash
gh run list --workflow=build-ingestion-image.yml --limit 1
```

Expected: `completed`, `success`. If it fails, read the actual error from `gh run view <run-id> --log-failed` and fix — do not push to `vllm` to work around anything (see Global Constraints).

- [ ] **Step 5: Verify the image is pullable**

```bash
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:l1joseph/knightgpt-ingestion:pull" | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])")
curl -s "https://ghcr.io/v2/l1joseph/knightgpt-ingestion/tags/list" -H "Authorization: Bearer $TOKEN"
```

Expected: a JSON object listing both a `sha-<hash>` tag and `latest`. If the package is private (check via `gh api /users/l1joseph/packages/container/knightgpt-ingestion --jq '.visibility'` — may 403 due to token scope, as happened for the Postgres image; fall back to the anonymous-pull check above, expecting `200` on a manifest fetch), make it public the same way the Postgres image was:

```bash
gh api --method PATCH /user/packages/container/knightgpt-ingestion -f visibility=public
```

---

### Task 2: Batch-partitioning utility

**Files:**
- Create: `scripts/nrp_batch_ingest.py` (this task only adds the partitioning function; Task 3 adds the rest)
- Test: `tests/test_nrp_batch_ingest.py`

**Interfaces:**
- Produces: `partition_into_batches(items: list, batch_size: int) -> list[list]` — pure function, no I/O, consumed by Task 3's orchestration logic.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_nrp_batch_ingest.py
"""Unit tests for the NRP batch ingestion orchestrator."""

import pytest


@pytest.mark.unit
def test_partition_into_batches_even_split():
    from scripts.nrp_batch_ingest import partition_into_batches

    items = list(range(10))
    batches = partition_into_batches(items, batch_size=5)

    assert batches == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


@pytest.mark.unit
def test_partition_into_batches_uneven_remainder():
    from scripts.nrp_batch_ingest import partition_into_batches

    items = list(range(7))
    batches = partition_into_batches(items, batch_size=3)

    assert batches == [[0, 1, 2], [3, 4, 5], [6]]


@pytest.mark.unit
def test_partition_into_batches_empty_input():
    from scripts.nrp_batch_ingest import partition_into_batches

    assert partition_into_batches([], batch_size=5) == []


@pytest.mark.unit
def test_partition_into_batches_batch_size_larger_than_input():
    from scripts.nrp_batch_ingest import partition_into_batches

    items = [1, 2, 3]
    assert partition_into_batches(items, batch_size=50) == [[1, 2, 3]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nrp_batch_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.nrp_batch_ingest'`

- [ ] **Step 3: Create the file with the partitioning function**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_nrp_batch_ingest.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/nrp_batch_ingest.py tests/test_nrp_batch_ingest.py
git commit -m "feat(nrp): batch-partitioning utility for ingestion orchestrator"
```

---

### Task 3: Full orchestrator — long-read ETL, download, batched chunk/embed/insert

**Files:**
- Modify: `scripts/nrp_batch_ingest.py`
- Modify: `tests/test_nrp_batch_ingest.py`

**Interfaces:**
- Consumes: `partition_into_batches` (Task 2); `scripts.download_papers.download_papers`, `scripts.etl_sheet_to_dois.resolve_longread_table`, `scripts.etl_sheet_to_dois.write_doi_file`; `src.chunking.SemanticChunker`, `src.chunking.Chunk`, `src.embedding.VLLMEmbedder`, `src.graph.insert_chunks`, `src.graph.DuckDBStore`; `src.ingestion.doi_resolver.build_doi_lookup`, `src.ingestion.doi_resolver.resolve_doi`; `src.utils.get_pg_pool`, `src.utils.get_settings`, `src.utils.get_logger`, `src.utils.setup_logging`.
- Produces: `run_batch_ingestion(paper_lists: list[Path], batch_size: int = 50, max_papers: int | None = None) -> dict` (the `main()` CLI entry point calls this) — `max_papers` caps the total number of papers processed, used by Task 4's small-scale validation run before Task 5's full run.

- [ ] **Step 1: Write the failing tests for the paper-dict-building helper**

The only piece of new logic in this task that's meaningfully unit-testable without live Postgres/DuckDB/vLLM is the per-batch "build the `papers` dict from a batch's chunks" logic (mirrors `run_pipeline()`'s existing pattern at `scripts/ingest_pipeline.py:90-102`, extracted here as its own function so it's testable in isolation). Add to `tests/test_nrp_batch_ingest.py`:

```python
@pytest.mark.unit
def test_build_papers_dict_resolves_doi_per_chunk():
    from scripts.nrp_batch_ingest import build_papers_dict
    from src.chunking import Chunk

    chunks = [
        Chunk(id="c1", text="a", source_file="/data/markdown/10-1234_x.md", metadata={"title": "Paper X"}),
        Chunk(id="c2", text="b", source_file="/data/markdown/10-1234_x.md", metadata={"title": "Paper X"}),
    ]
    doi_lookup = {"10-1234_x": "10.1234/x"}

    papers = build_papers_dict(chunks, doi_lookup)

    assert papers["/data/markdown/10-1234_x.md"]["doi"] == "10.1234/x"
    assert papers["/data/markdown/10-1234_x.md"]["title"] == "Paper X"
    # Same source_file across multiple chunks must produce exactly one entry.
    assert len(papers) == 1


@pytest.mark.unit
def test_build_papers_dict_handles_missing_source_file():
    from scripts.nrp_batch_ingest import build_papers_dict
    from src.chunking import Chunk

    chunks = [Chunk(id="c1", text="a", source_file="", metadata={})]
    papers = build_papers_dict(chunks, doi_lookup={})

    assert papers == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nrp_batch_ingest.py -v`
Expected: FAIL — `build_papers_dict` not defined.

- [ ] **Step 3: Replace `scripts/nrp_batch_ingest.py` with the full orchestrator**

```python
#!/usr/bin/env python3
"""NRP batch ingestion orchestrator.

Runs a full ingestion of all four paper source lists into the NRP
Postgres+DuckDB storage, in fixed-size batches through chunk -> embed ->
insert (not via scripts/ingest_pipeline.py::run_pipeline(), which would
redundantly reconvert PDFs download_papers() already converted -- see
docs/superpowers/plans/2026-08-06-nrp-ingestion-job.md's Global
Constraints for why).

Three phases:
  0. Derive the long-read source's DOI list from its checked-in TSV
     (skipped if already derived).
  1. Download + convert every paper across all four source lists.
     download_papers() (via MicrobiomeScraper._download_and_process) does
     both download and PDF-to-markdown conversion in one resumable pass --
     already-downloaded PDFs are skipped on a re-run.
  2. Partition the resulting markdown files into fixed-size batches and
     run each batch through chunk -> embed -> insert.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.download_papers import download_papers
from scripts.etl_sheet_to_dois import resolve_longread_table, write_doi_file
from src.chunking import Chunk, SemanticChunker
from src.embedding import VLLMEmbedder
from src.graph import DuckDBStore, insert_chunks
from src.ingestion.doi_resolver import build_doi_lookup, resolve_doi
from src.utils import get_logger, get_pg_pool, get_settings, setup_logging

logger = get_logger(__name__)
settings = get_settings()

DEFAULT_PAPER_LISTS = [
    Path("data/paper_lists/initial_papers.txt"),
    Path("data/paper_lists/zotero_papers.txt"),
    Path("data/paper_lists/mmc_papers.txt"),
]
LONGREAD_TSV = Path("data/paper_lists/sources/longread_bioprojects.tsv")
LONGREAD_DERIVED = Path("data/paper_lists/longread_papers.txt")


def partition_into_batches(items: list, batch_size: int) -> list[list]:
    """Split items into consecutive batches of at most batch_size each.

    The final batch may be smaller than batch_size if len(items) isn't an
    exact multiple. Returns an empty list for empty input.
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def build_papers_dict(chunks: list[Chunk], doi_lookup: dict[str, str]) -> dict:
    """Build the papers dict insert_chunks() expects: source_file -> {doi,
    title, metadata}. One entry per distinct source_file, regardless of how
    many chunks share it. Mirrors scripts/ingest_pipeline.py::run_pipeline()'s
    existing inline logic, extracted here so it's independently testable.
    """
    papers = {}
    for chunk in chunks:
        if not chunk.source_file or chunk.source_file in papers:
            continue
        papers[chunk.source_file] = {
            "doi": resolve_doi(chunk.source_file, doi_lookup),
            "title": chunk.metadata.get("title", ""),
            "metadata": chunk.metadata,
        }
    return papers


def ensure_longread_dois_derived() -> Path:
    """Phase 0: derive data/paper_lists/longread_papers.txt from the
    checked-in TSV, unless already present from a prior run."""
    if LONGREAD_DERIVED.exists():
        logger.info(f"Long-read DOI list already derived at {LONGREAD_DERIVED}, skipping")
        return LONGREAD_DERIVED

    import requests

    session = requests.Session()
    dois = resolve_longread_table(LONGREAD_TSV, session)
    write_doi_file(dois, LONGREAD_DERIVED)
    logger.info(f"Derived {len(dois)} DOIs from {LONGREAD_TSV} -> {LONGREAD_DERIVED}")
    return LONGREAD_DERIVED


def download_all_sources(paper_lists: list[Path]) -> dict:
    """Phase 1: download + convert every paper across all source lists.
    Resumable -- download_papers() skips DOIs whose PDF already exists."""
    combined_stats = {"total_dois": 0, "downloaded": 0, "skipped": 0, "failed": 0}
    for doi_file in paper_lists:
        logger.info(f"Downloading from {doi_file}")
        stats = download_papers(doi_file=doi_file)
        for key in ("total_dois", "downloaded", "skipped", "failed"):
            combined_stats[key] += stats[key]
    logger.info(f"Phase 1 complete: {combined_stats}")
    return combined_stats


async def _insert_batch(chunks: list[Chunk], papers: dict, store: DuckDBStore) -> dict:
    """Insert one batch's chunks into Postgres+DuckDB. Owns its own pool
    for the lifetime of this one batch (matches run_pipeline()'s existing
    single-asyncio.run()-call pattern for asyncpg pool/loop binding)."""
    pool = await get_pg_pool()
    try:
        return await insert_chunks(pool, chunks, papers, store, similarity_threshold=0.7)
    finally:
        await pool.close()


def run_batch_ingestion(
    paper_lists: list[Path],
    batch_size: int = 50,
    max_papers: int | None = None,
) -> dict:
    """Run the full three-phase ingestion. max_papers caps the total
    number of markdown files processed in Phase 2, for a small-scale
    validation run before committing to the full corpus."""
    all_lists = list(paper_lists) + [ensure_longread_dois_derived()]

    stats = {"start_time": datetime.now().isoformat()}
    stats["download"] = download_all_sources(all_lists)

    markdown_dir = settings.ingestion.markdown_dir
    markdown_files = sorted(markdown_dir.rglob("*.md"))
    if max_papers is not None:
        markdown_files = markdown_files[:max_papers]
    logger.info(f"Phase 2: processing {len(markdown_files)} markdown files")

    batches = partition_into_batches(markdown_files, batch_size)
    doi_lookup = build_doi_lookup()
    chunker = SemanticChunker()
    embedder = VLLMEmbedder()
    store = DuckDBStore(str(settings.ingestion.duckdb_path))

    batch_results = []
    try:
        for i, batch_files in enumerate(batches, 1):
            logger.info(f"Batch {i}/{len(batches)}: {len(batch_files)} papers")
            batch_chunks: list[Chunk] = []
            paper_failures = 0
            for md_file in batch_files:
                # Per-paper failures are caught, logged, and counted -- the batch
                # continues with its remaining papers (spec's Error Handling
                # section: "Per-paper failures ... are caught, logged, and
                # counted within their batch"). Only batch-level failures below
                # (embedding, insert) are allowed to propagate and fail the Job.
                try:
                    batch_chunks.extend(chunker.chunk_markdown_file(md_file))
                except Exception:
                    logger.exception(f"Batch {i}: failed to chunk {md_file}, skipping paper")
                    paper_failures += 1

            if not batch_chunks:
                logger.warning(f"Batch {i}: no chunks produced, skipping")
                batch_results.append({"batch": i, "papers": len(batch_files), "paper_failures": paper_failures})
                continue

            batch_chunks = embedder.embed_chunks(batch_chunks)
            papers = build_papers_dict(batch_chunks, doi_lookup)
            insert_stats = asyncio.run(_insert_batch(batch_chunks, papers, store))

            logger.info(f"Batch {i}/{len(batches)} done: {insert_stats}, paper_failures={paper_failures}")
            batch_results.append({"batch": i, "papers": len(batch_files), "paper_failures": paper_failures, **insert_stats})
    finally:
        store.close()

    stats["batches"] = batch_results
    stats["end_time"] = datetime.now().isoformat()

    processed_dir = settings.ingestion.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "nrp_batch_ingest_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Ingestion complete: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Run NRP batch ingestion")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Cap total papers processed (for a small-scale validation run)",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    stats = run_batch_ingestion(
        paper_lists=DEFAULT_PAPER_LISTS,
        batch_size=args.batch_size,
        max_papers=args.max_papers,
    )

    print("\nIngestion Summary:")
    print(f"  Download: {stats['download']}")
    print(f"  Batches processed: {len(stats['batches'])}")
    total_chunks = sum(b.get("chunks_inserted", 0) for b in stats["batches"])
    total_edges = sum(b.get("edges_inserted", 0) for b in stats["batches"])
    print(f"  Total chunks inserted: {total_chunks}")
    print(f"  Total edges inserted: {total_edges}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_nrp_batch_ingest.py -v`
Expected: PASS (6 tests — the 4 from Task 2 plus the 2 new ones)

- [ ] **Step 5: Commit**

```bash
git add scripts/nrp_batch_ingest.py tests/test_nrp_batch_ingest.py
git commit -m "feat(nrp): full batch ingestion orchestrator (download, chunk, embed, insert)"
```

---

### Task 4: Kubernetes Job manifest and small-scale validation run

**Files:**
- Create: `k8s/nrp/ingestion-job.yaml`
- Create: `k8s/nrp/ingestion-pvc.yaml`

**Interfaces:**
- Consumes: the GHCR image from Task 1 (`ghcr.io/l1joseph/knightgpt-ingestion:latest`), the orchestrator from Task 3 (`scripts/nrp_batch_ingest.py`), the existing Postgres/DuckDB storage (`knightgpt-postgres` Service, `knightgpt-postgres-credentials` Secret, `knightgpt-duckdb` PVC — all from the prior storage sub-project).
- Produces: a validated, working Job manifest — Task 5 removes the `--max-papers` cap and runs it at full scale.

- [ ] **Step 1: Create the intermediates PVC manifest**

```yaml
# k8s/nrp/ingestion-pvc.yaml
#
# WARNING: this PVC's storage class (rook-ceph-block) has reclaimPolicy:
# Delete. Deleting this resource destroys all downloaded PDFs/markdown/
# chunks irreversibly. Kept in its own file, separate from the Job
# manifest, so `kubectl delete -f k8s/nrp/ingestion-job.yaml` (a natural
# cleanup command once the Job is done) does not also delete this data.
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: knightgpt-ingestion-data
  namespace: knightlab-ml
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: rook-ceph-block
  resources:
    requests:
      storage: 30Gi
```

- [ ] **Step 2: Create the Job manifest**

```yaml
# k8s/nrp/ingestion-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: knightgpt-ingestion
  namespace: knightlab-ml
spec:
  backoffLimit: 2
  activeDeadlineSeconds: 21600  # 6 hours -- generous ceiling for a run of this size; NRP
                                  # policy requires Jobs to run a command that terminates on
                                  # its own, this is the hard backstop if something hangs
  template:
    spec:
      restartPolicy: Never
      initContainers:
        - name: vllm-embedding
          restartPolicy: Always  # native sidecar (K8s 1.29+): starts before the main
                                  # container, torn down automatically when the main
                                  # container exits, regardless of exit status
          image: vllm/vllm-openai:v0.26.0
          args:
            - "--model"
            - "Alibaba-NLP/gte-Qwen2-7B-instruct"
            - "--port"
            - "8001"
            - "--trust-remote-code"
          ports:
            - containerPort: 8001
          resources:
            requests:
              cpu: "4"
              memory: 32Gi
              nvidia.com/a100: 1
            limits:
              cpu: "4"
              memory: 32Gi
              nvidia.com/a100: 1
          readinessProbe:
            httpGet:
              path: /health
              port: 8001
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 30  # model load can take several minutes
      containers:
        - name: ingestion
          image: ghcr.io/l1joseph/knightgpt-ingestion:latest
          imagePullPolicy: Always
          command: ["python", "scripts/nrp_batch_ingest.py", "--max-papers", "20", "--log-level", "INFO"]
          env:
            - name: POSTGRES_DSN
              value: "postgresql://postgres:$(POSTGRES_PASSWORD)@knightgpt-postgres.knightlab-ml.svc.cluster.local:5432/knightgpt"
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: knightgpt-postgres-credentials
                  key: POSTGRES_PASSWORD
            - name: INGEST_RAW_PDF_DIR
              value: /data/raw_pdfs
            - name: INGEST_MARKDOWN_DIR
              value: /data/markdown
            - name: INGEST_PROCESSED_DIR
              value: /data/processed
            - name: INGEST_DUCKDB_PATH
              value: /duckdb-data/embeddings.duckdb
            - name: VLLM_EMBEDDING_URL
              value: http://localhost:8001/v1
            - name: VLLM_EMBEDDING_MODEL
              value: Alibaba-NLP/gte-Qwen2-7B-instruct
          resources:
            requests:
              cpu: "2"
              memory: 4Gi
            limits:
              cpu: "4"
              memory: 8Gi
          volumeMounts:
            - name: ingestion-data
              mountPath: /data
            - name: duckdb-data
              mountPath: /duckdb-data
      volumes:
        - name: ingestion-data
          persistentVolumeClaim:
            claimName: knightgpt-ingestion-data
        - name: duckdb-data
          persistentVolumeClaim:
            claimName: knightgpt-duckdb
```

Note `command: [..., "--max-papers", "20", ...]` — this task validates the Job's wiring cheaply (20 papers, not the full ~900) before Task 5 commits to a full-scale, multi-hour, GPU-consuming run. Note also that `POSTGRES_DSN`'s value uses `$(POSTGRES_PASSWORD)` Kubernetes env-var interpolation syntax, referencing the `POSTGRES_PASSWORD` env var defined immediately above it in the same `env:` list — confirm this resolves correctly in Step 4 below; if the generated password happens to contain a character that's invalid in this position of a URI (`@`, `:`, `/`, `#` — the same caveat `k8s/nrp/README.md` already documents), this will produce a malformed DSN and needs handling (see Step 4's troubleshooting note).

- [ ] **Step 3: Apply both manifests**

```bash
kubectl apply -f k8s/nrp/ingestion-pvc.yaml -f k8s/nrp/ingestion-job.yaml
```

If this fails on the Gatekeeper `resources.requests`/`limits` policy or any other admission error, read the actual error message and adjust the manifest accordingly — this cluster's exact policy requirements were discovered empirically during the storage deployment plan and may need re-confirming here (e.g. the `container-must-meet-memory-and-cpu-ratio` policy, ~1.2x max ratio, may require tightening the `requests`/`limits` values above — check `kubectl create job --from=... ` dry-run style verification, matching the pattern used in the storage plan's backup CronJob task, if a warning appears).

- [ ] **Step 4: Watch the Job and verify it completes successfully**

```bash
kubectl get job knightgpt-ingestion -n knightlab-ml -w
```

Expected: reaches `Complete` (Ctrl-C once it does). If it doesn't within ~30 minutes (20 papers should be fast once the vLLM sidecar is up — model load itself can take several minutes, budget for that), check:

```bash
kubectl describe job knightgpt-ingestion -n knightlab-ml
kubectl logs -n knightlab-ml -l job-name=knightgpt-ingestion -c ingestion --tail=200
kubectl logs -n knightlab-ml -l job-name=knightgpt-ingestion -c vllm-embedding --tail=100
```

If the DSN interpolation issue flagged in Step 2 occurs (malformed DSN due to special characters in the password), the ingestion container's logs will show a Postgres connection error mentioning URL parsing — if so, don't force a workaround blindly; check the actual error, and if it's specifically the interpolation problem, the fix is generating a new password without URI-special characters for the existing Secret (`kubectl create secret generic knightgpt-postgres-credentials --dry-run=client -o yaml` with a re-generated `openssl rand -base64 24 | tr -d '/@:#'`-style value, then `kubectl apply`) — but this affects the live Postgres deployment's existing credential, so treat this as a real, careful action requiring the same caution as any credential rotation, not a casual fix. If you hit this, stop and report the exact situation rather than unilaterally rotating a live credential.

- [ ] **Step 5: Verify results with a live-query check**

```bash
kubectl exec -n knightlab-ml deploy/knightgpt-postgres -- psql -U postgres -d knightgpt -c "SELECT count(*) FROM papers; SELECT count(*) FROM chunks; SELECT count(*) FROM chunk_edges;"
```

Expected: non-zero counts roughly proportional to ~20 papers (a handful of papers, tens to low hundreds of chunks depending on paper length — exact numbers will vary with real download success rates, this is a plausibility check not an exact-match one).

- [ ] **Step 6: Clean up the validation Job (but not the PVCs)**

```bash
kubectl delete job knightgpt-ingestion -n knightlab-ml
```

This deletes the completed Job object only — `knightgpt-ingestion-data` (intermediates) and the existing Postgres/DuckDB storage are untouched, so the 20 validation papers' downloaded PDFs/markdown remain on the PVC (harmless — Task 5's full run will just skip re-downloading them, per the resumability design) and their rows remain in Postgres/DuckDB (intentional — Task 5 is additive, not a fresh start).

- [ ] **Step 7: Commit**

```bash
git add k8s/nrp/ingestion-pvc.yaml k8s/nrp/ingestion-job.yaml
git commit -m "feat(nrp): Kubernetes Job manifest for GPU-backed ingestion, validated at small scale"
```

---

### Task 5: Full-scale ingestion run

**Files:**
- Modify: `k8s/nrp/ingestion-job.yaml`

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: the fully populated NRP Postgres+DuckDB storage — the actual deliverable this whole plan exists for.

- [ ] **Step 1: Remove the `--max-papers` cap**

In `k8s/nrp/ingestion-job.yaml`, change:
```yaml
          command: ["python", "scripts/nrp_batch_ingest.py", "--max-papers", "20", "--log-level", "INFO"]
```
to:
```yaml
          command: ["python", "scripts/nrp_batch_ingest.py", "--log-level", "INFO"]
```

- [ ] **Step 2: Apply and run the full-scale Job**

```bash
kubectl apply -f k8s/nrp/ingestion-job.yaml
kubectl get job knightgpt-ingestion -n knightlab-ml -w
```

Expected: this is a genuinely long-running operation (hundreds of papers, network-bound downloads with a polite per-request delay, GPU-bound embedding) — budget realistically for this to take a substantial fraction of the `activeDeadlineSeconds: 21600` (6 hour) ceiling, not a quick check. Monitor periodically rather than blocking continuously:

```bash
kubectl logs -n knightlab-ml -l job-name=knightgpt-ingestion -c ingestion -f --tail=50
```

Per-paper and per-download failures are expected and non-fatal (see the design spec's Error Handling section — PDF fetch success won't be 100%) — do not treat individual failures in the log stream as something to stop and fix; only a Job-level `Failed` status (visible via `kubectl get job`) warrants investigation.

- [ ] **Step 3: Verify final results**

```bash
kubectl exec -n knightlab-ml deploy/knightgpt-postgres -- psql -U postgres -d knightgpt -c "SELECT count(*) FROM papers; SELECT count(*) FROM chunks; SELECT count(*) FROM chunk_edges;"
```

Expected: paper count in the hundreds (realistically less than the ~900 raw source entries, given expected PDF-fetch failures, dedup across lists sharing DOIs, and the pre-existing 117 already having been part of the source lists), chunk count in the thousands, edge count larger still.

Also verify the `nrp_batch_ingest_stats.json` summary written to the intermediates PVC, via a throwaway debug pod (same pattern as `k8s/nrp/duckdb-verify-pod.yaml`):

```bash
kubectl run ingestion-stats-check --image=busybox:latest --restart=Never -n knightlab-ml --overrides='{"spec":{"containers":[{"name":"check","image":"busybox:latest","command":["cat","/data/processed/nrp_batch_ingest_stats.json"],"resources":{"requests":{"cpu":"50m","memory":"64Mi"},"limits":{"cpu":"200m","memory":"128Mi"}},"volumeMounts":[{"name":"data","mountPath":"/data"}]}],"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"knightgpt-ingestion-data"}}],"restartPolicy":"Never"}}'
kubectl logs ingestion-stats-check -n knightlab-ml
kubectl delete pod ingestion-stats-check -n knightlab-ml
```

- [ ] **Step 4: Spot-check retrieval end-to-end**

From a throwaway debug pod (or a local port-forward to both `knightgpt-postgres` and a way to reach the DuckDB PVC — a debug pod inside the cluster is simpler since it can mount `knightgpt-duckdb` directly):

```bash
kubectl run retrieval-check --image=ghcr.io/l1joseph/knightgpt-ingestion:latest --restart=Never -n knightlab-ml \
  --overrides='{"spec":{"containers":[{"name":"check","image":"ghcr.io/l1joseph/knightgpt-ingestion:latest","command":["sleep","300"],"resources":{"requests":{"cpu":"250m","memory":"512Mi"},"limits":{"cpu":"1","memory":"1Gi"}},"env":[{"name":"POSTGRES_DSN","value":"postgresql://postgres:PLACEHOLDER@knightgpt-postgres.knightlab-ml.svc.cluster.local:5432/knightgpt"},{"name":"INGEST_DUCKDB_PATH","value":"/duckdb-data/embeddings.duckdb"},{"name":"VLLM_EMBEDDING_URL","value":"http://localhost:8001/v1"}],"volumeMounts":[{"name":"duckdb-data","mountPath":"/duckdb-data"}]}],"volumes":[{"name":"duckdb-data","persistentVolumeClaim":{"claimName":"knightgpt-duckdb"}}],"restartPolicy":"Never"}}'
```

Replace `PLACEHOLDER` with the real password (`kubectl get secret knightgpt-postgres-credentials -n knightlab-ml -o jsonpath='{.data.POSTGRES_PASSWORD}' | base64 -d`) before running — this spot-check needs live embedding for the query text too, which requires either the vLLM sidecar pattern again (heavier than needed for a one-off check) or, more practically, running this check from a machine that can reach an already-running vLLM instance. If no embedding server is conveniently reachable for this one-off check, a simpler DuckDB-only spot-check (bypassing `HybridRetriever`'s query-embedding step) is an acceptable substitute: directly query DuckDB for a handful of stored chunk embeddings' nearest neighbors to each other (confirms the HNSW index and data are coherent) and separately confirm a `graph.expand()` call from Postgres returns real neighbor chunk IDs — together these two checks cover the same "data is real and usable" intent as a full end-to-end retrieval call, without needing a live embedding server just for verification. Use judgment on which is more practical to actually execute at implementation time; the goal (confirm ingested data is genuinely queryable and coherent, not just present as row counts) is what matters, not the exact mechanism.

- [ ] **Step 5: Clean up any debug pods**

```bash
kubectl get pods -n knightlab-ml | grep -E "check|debug"
```

Delete anything left over from Step 4's verification.

- [ ] **Step 6: Commit**

```bash
git add k8s/nrp/ingestion-job.yaml
git commit -m "feat(nrp): run full-scale ingestion (max-papers cap removed)"
```

---

## Final Verification

After all 5 tasks, confirm the storage is genuinely populated and ready for the third sub-project (the MCP server, out of scope for this plan) to build on:

```bash
kubectl exec -n knightlab-ml deploy/knightgpt-postgres -- psql -U postgres -d knightgpt -c "SELECT count(*) FROM papers; SELECT count(*) FROM chunks; SELECT count(*) FROM chunk_edges;"
kubectl get job knightgpt-ingestion -n knightlab-ml
```

Expected: Job `Complete`, all three counts substantially larger than the pre-ingestion state (117 papers / 6,179 chunks / 37,286 edges was the old Cosmos-based corpus size — this run's counts should be in that neighborhood or larger, reflecting the full ~900-entry source list rather than just the original subset).
