# Qiita Study Registry Ingestion (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A one-time Kubernetes Job on NRP that seeds a `qiita_studies` Postgres table with every study ID redbiom's public index exposes, plus per-study sample counts.

**Architecture:** A plain Kubernetes Job (no GPU, no sidecar, no PVC) runs a Python script that shells out to the `redbiom` CLI against its public backend (`http://qiita.ucsd.edu:7329`), aggregates per-study sample counts across all redbiom contexts, and upserts the results into a new Postgres table alongside the existing `papers` table.

**Tech Stack:** Python 3.11, `redbiom` (PyPI, public unauthenticated backend), `asyncpg`, existing `src/utils` settings/logging, Kubernetes Job on NRP (`knightlab-ml` namespace), GitHub Actions → GHCR.

## Global Constraints

- Namespace: `knightlab-ml`, cluster context `nautilus`.
- This is Stage 1 only: study IDs + sample counts + which redbiom contexts each study appears in. `title`/`abstract`/`principal_investigator`/`funding`/`metadata`/`metadata_backfilled_at` columns exist in the schema but stay `NULL` — populated by a future, separate Stage 2 once direct Postgres access to Qiita's own database is granted. Do not attempt to backfill them here.
- **One-time run, not recurring.** No CronJob, no scheduling — a plain Job, matching the paper-ingestion Job's pattern.
- **Every study redbiom's public index exposes** — no filtering to a subset (e.g. no Knight-Lab-only filter).
- **Do not add per-study sample-metadata category rollups** (e.g. environment-type distribution) — deliberately descoped in the spec. Keep to study ID + sample count + context list.
- Redbiom's backend (`http://qiita.ucsd.edu:7329`) is public and requires no credentials — confirmed reachable from outside the lab's internal network. This Job needs no VPN, no Barnacle2 access, no special network configuration beyond normal internet egress.
- Confirmed real `redbiom` CLI contract (verified live, exact formats — do not deviate):
  - `redbiom summarize contexts` → stdout TSV, header row `ContextName\tSamplesWithData\tFeaturesWithData\tDescription`, one row per context. 358 contexts observed as of this session (not ~50 — an earlier rough estimate was wrong; verify the real count fresh if this matters, don't hardcode 358 into any code).
  - `redbiom fetch samples-contained --context <ctx>` → stdout, one sample ID per line, no header. Redirect to a file for the next step (the next command requires a real file path via `--from`, not stdin).
  - `redbiom summarize samples --category qiita_study_id --from <file>` → stdout TSV: one `<study_id>\t<count>` row per study, **then a blank line, then a trailing `Total samples\t<N>` summary line** — code that parses this output must skip the blank line and the trailing `Total samples` line, not treat them as data rows.
- Kubernetes env-var ordering: any env var referencing `$(POSTGRES_PASSWORD)` must be defined **after** `POSTGRES_PASSWORD` in the same container's `env:` list — Kubernetes only expands `$(VAR)` for vars defined earlier in the list. Getting this backwards caused a real, reproducible `InvalidPasswordError` in the prior ingestion-job sub-project (see `k8s/nrp/ingestion-job.yaml`'s own comment on this).
- This namespace's Gatekeeper admission policy requires `resources.requests`/`resources.limits` on every container, plus a container-must-meet-memory-and-cpu-ratio policy (max ~1.2x ratio, warn-only not deny) — set `requests` equal to `limits` to trivially satisfy this, matching the established convention in `k8s/nrp/ingestion-job.yaml`.
- **Never test a workflow by pushing to the shared `vllm` branch.** Use `gh workflow run <file> --ref feature/qiita-registry-ingestion` — confirmed to work correctly once the workflow file exists on that branch.
- GitHub Actions `docker/metadata-action` tag rule for `latest` must be plain `type=raw,value=latest` with NO `enable={{is_default_branch}}` condition — this repo's default branch is `main`, not `vllm`.
- `src/utils/` (config, db, logging) is fully self-contained — no imports outside `pydantic`, `pydantic-settings`, `asyncpg`, `loguru`. A minimal Docker image only needs `src/__init__.py` + `src/utils/` copied in, not the full `src/` tree or the full `requirements.txt` (which pulls in ~3GB of PDF/ML dependencies this job doesn't need).
- `PostgresSettings` (`src/utils/config.py`, `env_prefix="POSTGRES_"`, field `dsn`) is the existing settings class for Postgres connectivity — no new settings class needed. Redbiom itself needs no settings/credentials.
- Follow existing code conventions: type hints, Google-style docstrings, `get_logger(__name__)` from `src.utils`, `pytest.mark.unit` on new tests.

---

### Task 1: `qiita_studies` Postgres schema

**Files:**
- Modify: `sql/schema.sql`

**Interfaces:**
- Produces: table `qiita_studies(study_id bigint PK, sample_count integer, contexts jsonb, title text, abstract text, principal_investigator text, funding text, metadata jsonb, ingested_at timestamptz, metadata_backfilled_at timestamptz)`, consumed by Task 3's upsert.

- [ ] **Step 1: Add the table to `sql/schema.sql`**

Append to the end of `sql/schema.sql` (after the existing `chunk_edges` table, before the pgGraph registration `DO $$` blocks — this table needs no pgGraph registration, it's not part of the chunk similarity graph):

```sql

CREATE TABLE IF NOT EXISTS qiita_studies (
    study_id                bigint PRIMARY KEY,
    sample_count            integer NOT NULL,
    contexts                jsonb NOT NULL DEFAULT '[]'::jsonb,
    title                   text,
    abstract                text,
    principal_investigator  text,
    funding                 text,
    metadata                jsonb NOT NULL DEFAULT '{}'::jsonb,
    ingested_at             timestamptz NOT NULL DEFAULT now(),
    metadata_backfilled_at  timestamptz
);
```

- [ ] **Step 2: Apply the schema to the live NRP Postgres**

Port-forward to the live Postgres Service, then run the existing schema-apply script against it:

```bash
kubectl port-forward -n knightlab-ml svc/knightgpt-postgres 5432:5432 &
PORT_FORWARD_PID=$!
sleep 2

PG_PASSWORD=$(kubectl get secret knightgpt-postgres-credentials -n knightlab-ml -o jsonpath='{.data.POSTGRES_PASSWORD}' | base64 -d)
python scripts/apply_schema.py --dsn "postgresql://postgres:${PG_PASSWORD}@localhost:5432/knightgpt" --log-level INFO

kill $PORT_FORWARD_PID
```

Expected: log line `Schema applied` followed by `pgGraph registration verified: chunks table and similar_to edge present` (this verification is pre-existing, unrelated to the new table, but `apply_schema.py` always runs it after applying the whole file — seeing it succeed confirms the whole file, including the new table, executed without error).

- [ ] **Step 3: Verify the table exists with the right columns**

```bash
kubectl exec -n knightlab-ml deploy/knightgpt-postgres -- psql -U postgres -d knightgpt -c "\d qiita_studies"
```

Expected: output listing all 10 columns (`study_id`, `sample_count`, `contexts`, `title`, `abstract`, `principal_investigator`, `funding`, `metadata`, `ingested_at`, `metadata_backfilled_at`) with `study_id` as the primary key.

- [ ] **Step 4: Commit**

```bash
git add sql/schema.sql
git commit -m "feat(qiita): add qiita_studies table for Stage 1 registry ingestion"
```

---

### Task 2: Redbiom output parsing and cross-context merge (pure functions, TDD)

**Files:**
- Create: `scripts/qiita_registry_ingest.py` (this task adds only the two pure functions; Task 3 adds the rest)
- Test: `tests/test_qiita_registry_ingest.py`

**Interfaces:**
- Produces: `parse_study_counts(summarize_output: str) -> dict[int, int]`, `merge_context_results(context_results: dict[str, dict[int, int]]) -> dict[int, dict]` — both consumed by Task 3's orchestrator.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_qiita_registry_ingest.py
"""Unit tests for the Qiita study registry ingestion script."""

import pytest


@pytest.mark.unit
def test_parse_study_counts_multiple_studies():
    from scripts.qiita_registry_ingest import parse_study_counts

    output = "10317\t31556\n12949\t17277\n\nTotal samples\t48833\n"
    result = parse_study_counts(output)

    assert result == {10317: 31556, 12949: 17277}


@pytest.mark.unit
def test_parse_study_counts_single_study():
    from scripts.qiita_registry_ingest import parse_study_counts

    output = "10333\t1\n\nTotal samples\t1\n"
    result = parse_study_counts(output)

    assert result == {10333: 1}


@pytest.mark.unit
def test_parse_study_counts_empty_context():
    from scripts.qiita_registry_ingest import parse_study_counts

    assert parse_study_counts("") == {}


@pytest.mark.unit
def test_parse_study_counts_ignores_total_line_only():
    from scripts.qiita_registry_ingest import parse_study_counts

    # "Total samples" itself is never a valid numeric study_id, so a
    # naive int() cast would raise -- confirms the line is skipped, not
    # silently mis-parsed as a study.
    output = "10317\t5\n\nTotal samples\t5\n"
    result = parse_study_counts(output)

    assert "Total samples" not in result
    assert result == {10317: 5}


@pytest.mark.unit
def test_merge_context_results_sums_across_contexts():
    from scripts.qiita_registry_ingest import merge_context_results

    context_results = {
        "ctxA": {10317: 100, 12949: 50},
        "ctxB": {10317: 20, 99999: 5},
    }

    result = merge_context_results(context_results)

    assert result == {
        10317: {"sample_count": 120, "contexts": ["ctxA", "ctxB"]},
        12949: {"sample_count": 50, "contexts": ["ctxA"]},
        99999: {"sample_count": 5, "contexts": ["ctxB"]},
    }


@pytest.mark.unit
def test_merge_context_results_empty_input():
    from scripts.qiita_registry_ingest import merge_context_results

    assert merge_context_results({}) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_qiita_registry_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.qiita_registry_ingest'`

- [ ] **Step 3: Create the file with the two pure functions**

```python
#!/usr/bin/env python3
"""Qiita study registry ingestion (Stage 1).

Seeds the qiita_studies Postgres table with every study ID redbiom's
public index exposes, plus per-study sample counts. redbiom's backend
(http://qiita.ucsd.edu:7329) is public and requires no credentials.

This is Stage 1 only: study_id, sample_count, and which redbiom contexts
each study appears in. title/abstract/principal_investigator/funding/
metadata stay NULL here -- a separate, future Stage 2 backfills them
once direct Postgres access to Qiita's own database is available.
"""


def parse_study_counts(summarize_output: str) -> dict[int, int]:
    """Parse `redbiom summarize samples --category qiita_study_id`'s TSV
    stdout into {study_id: sample_count}.

    The real output is one "<study_id>\\t<count>" row per study, then a
    blank line, then a trailing "Total samples\\t<N>" summary row -- both
    the blank line and the summary row are skipped, not treated as data.
    """
    counts = {}
    for line in summarize_output.splitlines():
        line = line.strip()
        if not line or line.startswith("Total samples"):
            continue
        study_id_str, count_str = line.split("\t")
        counts[int(study_id_str)] = int(count_str)
    return counts


def merge_context_results(
    context_results: dict[str, dict[int, int]],
) -> dict[int, dict]:
    """Merge per-context {study_id: count} maps into a single registry.

    A study can appear in multiple redbiom contexts (different processing
    pipelines run on the same study's samples) -- this sums sample_count
    across all contexts a study appears in and records which contexts it
    appeared in, in first-seen order.
    """
    merged: dict[int, dict] = {}
    for context_name, study_counts in context_results.items():
        for study_id, count in study_counts.items():
            if study_id not in merged:
                merged[study_id] = {"sample_count": 0, "contexts": []}
            merged[study_id]["sample_count"] += count
            merged[study_id]["contexts"].append(context_name)
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_qiita_registry_ingest.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/qiita_registry_ingest.py tests/test_qiita_registry_ingest.py
git commit -m "feat(qiita): redbiom output parsing and cross-context merge logic"
```

---

### Task 3: Full orchestrator — redbiom calls + Postgres upsert

**Files:**
- Modify: `scripts/qiita_registry_ingest.py`

**Interfaces:**
- Consumes: `parse_study_counts`, `merge_context_results` (Task 2); `src.utils.get_logger`, `src.utils.get_settings`, `src.utils.get_pg_pool`, `src.utils.setup_logging`.
- Produces: `list_contexts() -> list[str]`, `fetch_and_summarize_context(context_name: str) -> dict[int, int]`, `run_registry_ingest() -> dict` (the `main()` CLI entry point calls this) — consumed by Task 5's Job manifest via `python scripts/qiita_registry_ingest.py`.

- [ ] **Step 1: Replace `scripts/qiita_registry_ingest.py` with the full orchestrator**

Replace the whole file (the module docstring and the two functions from Task 2 carry over unchanged, moved below the new top-of-file imports; the orchestrator functions are new):

```python
#!/usr/bin/env python3
"""Qiita study registry ingestion (Stage 1).

Seeds the qiita_studies Postgres table with every study ID redbiom's
public index exposes, plus per-study sample counts. redbiom's backend
(http://qiita.ucsd.edu:7329) is public and requires no credentials.

This is Stage 1 only: study_id, sample_count, and which redbiom contexts
each study appears in. title/abstract/principal_investigator/funding/
metadata stay NULL here -- a separate, future Stage 2 backfills them
once direct Postgres access to Qiita's own database is available.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, get_pg_pool, setup_logging

logger = get_logger(__name__)

REDBIOM_TIMEOUT_S = 120


def parse_study_counts(summarize_output: str) -> dict[int, int]:
    """Parse `redbiom summarize samples --category qiita_study_id`'s TSV
    stdout into {study_id: sample_count}.

    The real output is one "<study_id>\\t<count>" row per study, then a
    blank line, then a trailing "Total samples\\t<N>" summary row -- both
    the blank line and the summary row are skipped, not treated as data.
    """
    counts = {}
    for line in summarize_output.splitlines():
        line = line.strip()
        if not line or line.startswith("Total samples"):
            continue
        study_id_str, count_str = line.split("\t")
        counts[int(study_id_str)] = int(count_str)
    return counts


def merge_context_results(
    context_results: dict[str, dict[int, int]],
) -> dict[int, dict]:
    """Merge per-context {study_id: count} maps into a single registry.

    A study can appear in multiple redbiom contexts (different processing
    pipelines run on the same study's samples) -- this sums sample_count
    across all contexts a study appears in and records which contexts it
    appeared in, in first-seen order.
    """
    merged: dict[int, dict] = {}
    for context_name, study_counts in context_results.items():
        for study_id, count in study_counts.items():
            if study_id not in merged:
                merged[study_id] = {"sample_count": 0, "contexts": []}
            merged[study_id]["sample_count"] += count
            merged[study_id]["contexts"].append(context_name)
    return merged


def list_contexts() -> list[str]:
    """Enumerate all redbiom context names via `redbiom summarize contexts`."""
    result = subprocess.run(
        ["redbiom", "summarize", "contexts"],
        capture_output=True,
        text=True,
        timeout=REDBIOM_TIMEOUT_S,
        check=True,
    )
    lines = result.stdout.splitlines()
    # First line is the header: ContextName\tSamplesWithData\tFeaturesWithData\tDescription
    return [line.split("\t")[0] for line in lines[1:] if line.strip()]


def fetch_and_summarize_context(context_name: str) -> dict[int, int]:
    """Fetch every sample ID in a context and summarize by qiita_study_id.

    Runs `redbiom fetch samples-contained --context <ctx>` to a temp file,
    then `redbiom summarize samples --category qiita_study_id --from
    <file>` on that file (summarize requires a real file path via --from,
    not stdin).
    """
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt") as tmp:
        fetch_result = subprocess.run(
            ["redbiom", "fetch", "samples-contained", "--context", context_name],
            capture_output=True,
            text=True,
            timeout=REDBIOM_TIMEOUT_S,
            check=True,
        )
        tmp.write(fetch_result.stdout)
        tmp.flush()

        if not fetch_result.stdout.strip():
            return {}

        summarize_result = subprocess.run(
            ["redbiom", "summarize", "samples", "--category", "qiita_study_id", "--from", tmp.name],
            capture_output=True,
            text=True,
            timeout=REDBIOM_TIMEOUT_S,
            check=True,
        )
        return parse_study_counts(summarize_result.stdout)


async def _upsert_registry(registry: dict[int, dict]) -> int:
    """Upsert the merged registry into qiita_studies. Deliberately does
    NOT touch title/abstract/principal_investigator/funding/metadata/
    metadata_backfilled_at, so a re-run never clobbers a future Stage 2
    backfill. Returns the number of rows upserted."""
    pool = await get_pg_pool()
    try:
        async with pool.acquire() as conn:
            for study_id, data in registry.items():
                await conn.execute(
                    """
                    INSERT INTO qiita_studies (study_id, sample_count, contexts)
                    VALUES ($1, $2, $3::jsonb)
                    ON CONFLICT (study_id) DO UPDATE SET
                        sample_count = EXCLUDED.sample_count,
                        contexts = EXCLUDED.contexts,
                        ingested_at = now()
                    """,
                    study_id,
                    data["sample_count"],
                    json.dumps(data["contexts"]),
                )
        return len(registry)
    finally:
        await pool.close()


def run_registry_ingest() -> dict:
    """Run the full Stage 1 ingestion: enumerate contexts, fetch+summarize
    each, merge across contexts, upsert into Postgres. Per-context
    failures are logged and non-fatal; if every context fails, raises
    (whole-run failure) rather than upserting an empty registry."""
    contexts = list_contexts()
    logger.info(f"Found {len(contexts)} redbiom contexts")

    context_results: dict[str, dict[int, int]] = {}
    context_failures = 0
    for i, context_name in enumerate(contexts, 1):
        try:
            counts = fetch_and_summarize_context(context_name)
            context_results[context_name] = counts
            logger.info(f"[{i}/{len(contexts)}] {context_name}: {len(counts)} studies")
        except Exception:
            logger.exception(f"[{i}/{len(contexts)}] {context_name} failed, skipping")
            context_failures += 1

    if not context_results and context_failures > 0:
        raise RuntimeError(
            f"All {context_failures} redbiom contexts failed -- treating as a "
            "whole-run failure rather than upserting an empty registry"
        )

    registry = merge_context_results(context_results)
    logger.info(f"Merged registry: {len(registry)} distinct studies across {len(context_results)} contexts")

    rows_upserted = asyncio.run(_upsert_registry(registry))
    logger.info(f"Upserted {rows_upserted} rows into qiita_studies")

    return {
        "contexts_total": len(contexts),
        "contexts_succeeded": len(context_results),
        "contexts_failed": context_failures,
        "studies_found": len(registry),
        "rows_upserted": rows_upserted,
    }


def main():
    parser = argparse.ArgumentParser(description="Seed qiita_studies from redbiom")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    summary = run_registry_ingest()

    print("\nQiita Registry Ingestion Summary:")
    print(f"  Contexts: {summary['contexts_succeeded']}/{summary['contexts_total']} succeeded")
    print(f"  Studies found: {summary['studies_found']}")
    print(f"  Rows upserted: {summary['rows_upserted']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the existing tests to confirm nothing broke**

Run: `pytest tests/test_qiita_registry_ingest.py -v`
Expected: PASS (still 6 tests — this task adds orchestration code, not new pure-logic tests; the orchestrator's live behavior is verified in Task 5)

- [ ] **Step 3: Commit**

```bash
git add scripts/qiita_registry_ingest.py
git commit -m "feat(qiita): redbiom orchestration and Postgres upsert"
```

---

### Task 4: Minimal Docker image + CI workflow

**Files:**
- Create: `docker/Dockerfile.qiita-registry`
- Create: `.github/workflows/build-qiita-registry-image.yml`

**Interfaces:**
- Produces: image `ghcr.io/l1joseph/knightgpt-qiita-registry:<tag>`, consumed by Task 5's Job manifest.

- [ ] **Step 1: Create the Dockerfile**

```dockerfile
# docker/Dockerfile.qiita-registry
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install \
        redbiom>=0.3.9 \
        asyncpg>=0.29.0 \
        pydantic>=2.5.0 \
        pydantic-settings>=2.1.0 \
        python-dotenv>=1.0.0 \
        loguru>=0.7.2

COPY src/__init__.py ./src/__init__.py
COPY src/utils/ ./src/utils/
COPY scripts/qiita_registry_ingest.py ./scripts/qiita_registry_ingest.py

RUN chown -R appuser:appuser /app

USER appuser
```

No `CMD`/`ENTRYPOINT` — the Job manifest specifies the exact command, matching the established `Dockerfile.ingestion` pattern. This image intentionally does NOT install the full `requirements.txt` (which pulls in ~3GB of PDF/ML/CUDA dependencies this job never uses) — only the handful of packages `redbiom` and `src/utils` actually need.

- [ ] **Step 2: Create the GitHub Actions workflow**

```yaml
# .github/workflows/build-qiita-registry-image.yml
name: Build and push Qiita registry image

on:
  push:
    branches: [vllm]
    paths:
      - "docker/Dockerfile.qiita-registry"
      - "scripts/qiita_registry_ingest.py"
      - "src/utils/**"
      - "src/__init__.py"
      - ".github/workflows/build-qiita-registry-image.yml"
  workflow_dispatch: {}

env:
  IMAGE_NAME: ghcr.io/${{ github.repository_owner }}/knightgpt-qiita-registry

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
          file: docker/Dockerfile.qiita-registry
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

Note the `tags:` block is `type=raw,value=latest` with no `enable={{is_default_branch}}` condition — per Global Constraints, that condition never fires on this repo (default branch is `main`, this workflow triggers on `vllm`).

- [ ] **Step 3: Commit**

```bash
git add docker/Dockerfile.qiita-registry .github/workflows/build-qiita-registry-image.yml
git commit -m "ci: build and push Qiita registry image to GHCR"
```

- [ ] **Step 4: Push the branch and trigger the workflow via workflow_dispatch**

```bash
git push -u origin feature/qiita-registry-ingestion
gh workflow run build-qiita-registry-image.yml --ref feature/qiita-registry-ingestion
```

Wait for it to complete:

```bash
gh run list --workflow=build-qiita-registry-image.yml --limit 1
```

Expected: `completed`, `success`. If it 404s on first dispatch (a brand-new workflow file sometimes isn't immediately dispatchable until GitHub indexes it via a push-triggered run — this happened once in the prior ingestion-job sub-project), do NOT merge or push to `vllm` to work around it. Wait a minute and retry `gh workflow run` with the same command — if it still fails after a couple of retries, stop and report the exact error rather than improvising a workaround.

- [ ] **Step 5: Verify the image is pullable**

```bash
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:l1joseph/knightgpt-qiita-registry:pull" | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])")
curl -s "https://ghcr.io/v2/l1joseph/knightgpt-qiita-registry/tags/list" -H "Authorization: Bearer $TOKEN"
```

Expected: a JSON object listing both a `sha-<hash>` tag and `latest`. If the package is private, make it public the same way prior images were:

```bash
gh api --method PATCH /user/packages/container/knightgpt-qiita-registry -f visibility=public
```

---

### Task 5: Kubernetes Job manifest, live run, verification, README update

**Files:**
- Create: `k8s/nrp/qiita-registry-ingest-job.yaml`
- Modify: `k8s/nrp/README.md`

**Interfaces:**
- Consumes: the image from Task 4 (`ghcr.io/l1joseph/knightgpt-qiita-registry:latest`), the orchestrator from Task 3, the existing Postgres storage (`knightgpt-postgres` Service, `knightgpt-postgres-credentials` Secret).
- Produces: the live, populated `qiita_studies` table — the actual deliverable this plan exists for.

- [ ] **Step 1: Create the Job manifest**

```yaml
# k8s/nrp/qiita-registry-ingest-job.yaml
#
# One-time Job: seeds qiita_studies from redbiom's public index. No GPU,
# no sidecar, no PVC -- lightweight HTTP calls to redbiom's public backend
# (http://qiita.ucsd.edu:7329) plus a handful of Postgres writes.
#
# Re-running: Kubernetes Jobs are immutable, so re-applying this over an
# already-Complete Job of the same name fails. Delete first:
#   kubectl delete job knightgpt-qiita-registry-ingest -n knightlab-ml && kubectl apply -f k8s/nrp/qiita-registry-ingest-job.yaml
# This is idempotent to re-run -- the upsert in scripts/qiita_registry_ingest.py
# only updates sample_count/contexts/ingested_at, never touching any
# Stage-2-backfilled metadata columns.
apiVersion: batch/v1
kind: Job
metadata:
  name: knightgpt-qiita-registry-ingest
  namespace: knightlab-ml
spec:
  backoffLimit: 2
  activeDeadlineSeconds: 3600  # 1 hour -- generous ceiling; the confirmed-working recipe
                                 # processed a 300k-sample context in ~12s, but redbiom's
                                 # public endpoint can be slow/flaky at times (observed
                                 # during planning), so this leaves real margin across
                                 # ~358 contexts. NRP policy requires Jobs to run a command
                                 # that terminates on its own -- this is the hard backstop.
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: qiita-registry-ingest
          image: ghcr.io/l1joseph/knightgpt-qiita-registry:latest
          imagePullPolicy: Always
          command: ["python", "scripts/qiita_registry_ingest.py", "--log-level", "INFO"]
          env:
            # POSTGRES_PASSWORD must come BEFORE POSTGRES_DSN -- Kubernetes only
            # expands $(VAR) references to env vars defined earlier in the same
            # container's env list. Getting this backwards caused a real,
            # reproducible InvalidPasswordError in the prior ingestion-job
            # sub-project (see k8s/nrp/ingestion-job.yaml's own comment on this).
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: knightgpt-postgres-credentials
                  key: POSTGRES_PASSWORD
            - name: POSTGRES_DSN
              value: "postgresql://postgres:$(POSTGRES_PASSWORD)@knightgpt-postgres.knightlab-ml.svc.cluster.local:5432/knightgpt"
          resources:  # requests == limits (ratio 1.0) to satisfy this namespace's Gatekeeper
                      # container-must-meet-memory-and-cpu-ratio policy. This is a lightweight
                      # HTTP+Postgres job with no GPU/large-file work, so modest values suffice.
            requests:
              cpu: "1"
              memory: 2Gi
            limits:
              cpu: "1"
              memory: 2Gi
```

- [ ] **Step 2: Apply and run the Job**

```bash
kubectl apply -f k8s/nrp/qiita-registry-ingest-job.yaml
kubectl get job knightgpt-qiita-registry-ingest -n knightlab-ml -w
```

Expected: reaches `Complete` (Ctrl-C once it does). If it fails on admission (Gatekeeper resources policy or similar), read the actual error and adjust the manifest to comply rather than working around the policy. Monitor progress with:

```bash
kubectl logs -n knightlab-ml -l job-name=knightgpt-qiita-registry-ingest -f --tail=50
```

Expect per-context progress lines (`[i/358] <context-name>: N studies`) and occasional context failures logged as non-fatal (per Global Constraints, this is expected — redbiom's public endpoint isn't perfectly reliable). Only a Job-level `Failed` status warrants stopping to investigate; individual context failures in the log stream do not.

- [ ] **Step 3: Verify results**

```bash
kubectl exec -n knightlab-ml deploy/knightgpt-postgres -- psql -U postgres -d knightgpt -c "SELECT count(*) FROM qiita_studies;"
kubectl exec -n knightlab-ml deploy/knightgpt-postgres -- psql -U postgres -d knightgpt -c "SELECT study_id, sample_count, jsonb_array_length(contexts) AS context_count FROM qiita_studies WHERE study_id = 10317;"
```

Expected: a study count well over 500 (one context alone had 573 distinct studies; the full run across all contexts should exceed that). Study 10317 (the American Gut Project, a well-known large Qiita study) should show a substantial `sample_count` (tens of thousands) and appear in multiple contexts.

- [ ] **Step 4: Clean up the completed Job**

```bash
kubectl get job knightgpt-qiita-registry-ingest -n knightlab-ml
```

Leave the completed Job in place (matches the established convention for `knightgpt-ingestion` — a completed Job's pod is not a debug artifact and doesn't need deletion).

- [ ] **Step 5: Update `k8s/nrp/README.md`**

Read the current file first, then add a row to the "Manifests and what they own" table for `qiita-registry-ingest-job.yaml` (owning `Job/knightgpt-qiita-registry-ingest`), and a short paragraph near the existing ingestion-Job documentation noting: this Job seeds `qiita_studies` from redbiom's public index (no lab-network/credential dependency), Stage 1 only (title/abstract/PI/funding/metadata columns stay `NULL` until a future Stage 2 backfill via direct Postgres access to Qiita's own database), and it's a one-time run, not a CronJob.

- [ ] **Step 6: Commit**

```bash
git add k8s/nrp/qiita-registry-ingest-job.yaml k8s/nrp/README.md
git commit -m "feat(qiita): Kubernetes Job for Stage 1 registry ingestion, live-verified"
```

---

## Final Verification

```bash
kubectl exec -n knightlab-ml deploy/knightgpt-postgres -- psql -U postgres -d knightgpt -c "SELECT count(*) FROM qiita_studies; SELECT count(*) FROM qiita_studies WHERE title IS NULL;"
kubectl get job knightgpt-qiita-registry-ingest -n knightlab-ml
```

Expected: `Job` shows `Complete`; `qiita_studies` has a substantial row count (500+); every row has `title IS NULL` (confirming Stage 1 correctly left metadata backfill untouched, ready for a future Stage 2).
