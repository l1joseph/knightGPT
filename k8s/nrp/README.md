# k8s/nrp — wiring contract for consumers

This directory provisions knightGPT's persistent storage on NRP (cluster
`nautilus`, namespace `knightlab-ml`): a Postgres+pgGraph Deployment, a
DuckDB PVC, a weekly backup CronJob, and a GPU-backed ingestion Job. This file
documents the fixed resource names and the exact environment variables **any
future workload in this namespace** (the not-yet-built MCP server, ad hoc
debug pods, etc.) must set to connect correctly.

Manifests and what they own:

| File | Resource(s) |
|---|---|
| `postgres-pvc.yaml` | `PersistentVolumeClaim/knightgpt-postgres-data` (RWO, linstor-igrok) — **never delete**, see warning comment in that file |
| `postgres-cluster.yaml` | `Deployment/knightgpt-postgres`, `Service/knightgpt-postgres` |
| `duckdb-pvc.yaml` | `PersistentVolumeClaim/knightgpt-duckdb` (RWX, rook-cephfs) |
| `backup-pvc.yaml` | `PersistentVolumeClaim/knightgpt-backup-dest` (RWO, rook-ceph-block) — **never delete**, see warning comment in that file |
| `backup-cronjob.yaml` | `CronJob/knightgpt-backup` |
| `duckdb-verify-pod.yaml` | throwaway debug pod pattern for inspecting the DuckDB PVC |
| `ingestion-pvc.yaml` | `PersistentVolumeClaim/knightgpt-ingestion-data` (RWO, rook-ceph-block) — raw PDFs/markdown/processed intermediates for the ingestion Job; **never delete**, see warning comment in that file |
| `ingestion-job.yaml` | `Job/knightgpt-ingestion` — GPU-backed (native `vllm/vllm-openai` sidecar + batch orchestrator main container), built and run successfully in production (see "Re-running this Job" below before re-applying) |
| `qiita-registry-ingest-job.yaml` | `Job/knightgpt-qiita-registry-ingest` — lightweight, no GPU/sidecar/PVC; ran to `Complete` in production with partial coverage (32/500+ expected studies) — see "Qiita registry ingestion Job" below before re-applying or relying on this data |

Apply order matters where a PVC was split out from its owning workload:
`kubectl apply -f k8s/nrp/postgres-pvc.yaml -f k8s/nrp/postgres-cluster.yaml`,
`kubectl apply -f k8s/nrp/backup-pvc.yaml -f k8s/nrp/backup-cronjob.yaml`,
and `kubectl apply -f k8s/nrp/ingestion-pvc.yaml -f k8s/nrp/ingestion-job.yaml`.

## Fixed resource names

- **Postgres Service DNS (in-cluster):** `knightgpt-postgres.knightlab-ml.svc.cluster.local:5432`
  (short form `knightgpt-postgres:5432` works for pods in the same namespace)
- **Postgres credentials Secret:** `knightgpt-postgres-credentials`, key `POSTGRES_PASSWORD`
  (user is always `postgres`, database is always `knightgpt` — see `postgres-cluster.yaml` env block)
- **DuckDB PVC:** `knightgpt-duckdb` (RWX — mountable read-write by exactly one writer at a
  time by convention; `DuckDBStore.__init__` raises a clear error on a lock conflict, there
  is no K8s-level enforcement). **Ownership convention: uid/gid 1000.** The ingestion Job's
  `fix-duckdb-permissions` init container (`chown -R 1000:1000`, see `ingestion-job.yaml`)
  established this on first write, because this PVC's storage class (rook-cephfs) does not
  honor `fsGroup` the way `ingestion-data`'s does — the mount stayed root-owned (0755)
  without it, and the ingestion container (non-root, uid 1000) failed with a `PermissionError`
  until this was added. **Any future workload that mounts this PVC as a different uid (e.g.
  the MCP server) will hit the same `PermissionError` unless it either runs as uid 1000 too,
  or adds its own equivalent chown init container.** This is not enforced by Kubernetes —
  it's a convention, easy to silently violate.

## Required environment variables

The app config is Pydantic `BaseSettings` in `src/utils/config.py`. Two settings classes
matter for any pod that needs to read/write knightGPT's storage:

- `PostgresSettings` (`env_prefix="POSTGRES_"`) — field `dsn` → env var **`POSTGRES_DSN`**.
  Format is the asyncpg connection string:
  `postgresql://postgres:<password>@knightgpt-postgres.knightlab-ml.svc.cluster.local:5432/knightgpt`
- `IngestionSettings` (`env_prefix="INGEST_"`) — field `duckdb_path` → env var
  **`INGEST_DUCKDB_PATH`**. Point this at the DuckDB PVC mount, e.g.
  `/data/embeddings.duckdb` if the PVC is mounted at `/data`.

### The `INGEST_DUCKDB_PATH` trap — read this before deploying anything

`IngestionSettings.model_post_init` (see `src/utils/config.py:161-178`) re-derives
`duckdb_path` from `processed_dir` **only when `processed_dir` is customized and
`duckdb_path` is left at its class-level default** — i.e. if a future Deployment/Job sets
`INGEST_PROCESSED_DIR` (e.g. to a PVC mount path) without also setting
`INGEST_DUCKDB_PATH`, the derivation *does* kick in correctly:
`duckdb_path = processed_dir / "embeddings.duckdb"`, which resolves inside the mounted PVC
as expected. That combination is actually safe.

**The real trap is leaving both unset.** If neither `INGEST_PROCESSED_DIR` nor
`INGEST_DUCKDB_PATH` is set, `processed_dir` stays at its own class-level default too, so
the `model_post_init` condition above never fires — `duckdb_path` falls straight through to
its own **CWD-relative** default (`data/processed/embeddings.duckdb`), which almost
certainly does **not** resolve inside the DuckDB PVC mount in a container. The result: a
fresh, empty, wrong-path DuckDB file gets created silently, with **no error, no warning**.
The only symptom is every future search returning zero results, because chunks exist in
Postgres but this wrong-path DuckDB file has nothing in it.

**Always set `INGEST_DUCKDB_PATH` explicitly** to a path inside the mounted PVC, rather
than relying on the `INGEST_PROCESSED_DIR` derivation to save you — it's a convenience for
local/non-containerized use, not a guarantee for deployment manifests, and is one line to
just set directly and never think about again.

## Example: a future Job/Deployment's `env:` block, wired correctly

```yaml
spec:
  containers:
    - name: knightgpt-ingest  # or knightgpt-mcp-server, etc.
      image: ghcr.io/l1joseph/knightgpt:<pinned-tag>
      env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: knightgpt-postgres-credentials
              key: POSTGRES_PASSWORD
        # Kubernetes expands $(POSTGRES_PASSWORD) here because it's defined earlier in
        # this same container's env list (dependent env vars) -- see
        # https://kubernetes.io/docs/tasks/inject-data-application/define-interdependent-environment-variables/
        #
        # CAVEAT: this only produces a valid DSN if the password contains none of
        # @ : / # (URI-reserved characters). The Secret's password is
        # openssl-rand-generated, so this is a real if low-probability edge case, not
        # hypothetical -- if the current password ever happens to contain one of those
        # characters, this interpolation breaks silently (connects to the wrong
        # host/db, or fails to parse) and the password component needs URL-encoding
        # first.
        - name: POSTGRES_DSN
          value: "postgresql://postgres:$(POSTGRES_PASSWORD)@knightgpt-postgres.knightlab-ml.svc.cluster.local:5432/knightgpt"
        # Set this explicitly -- don't leave it (and INGEST_PROCESSED_DIR) both unset
        # (see "the INGEST_DUCKDB_PATH trap" above) -- that combination silently falls
        # back to a CWD-relative default and returns zero search results with no error.
        - name: INGEST_DUCKDB_PATH
          value: "/data/embeddings.duckdb"
      volumeMounts:
        - name: duckdb-data
          mountPath: /data
  volumes:
    - name: duckdb-data
      persistentVolumeClaim:
        claimName: knightgpt-duckdb
```

## Operational notes for future consumers

- **Readiness window:** the Postgres pod is not immediately ready to accept connections
  after it starts (a `readinessProbe`/`livenessProbe` now gate the Service, see
  `postgres-cluster.yaml`), but any client with an eagerly-built connection pool and no
  retry logic (e.g. `HybridRetriever.__init__`) can still hard-fail if it starts and
  connects before the Service has routable, ready endpoints. Build in retry/backoff on
  initial pool creation if this matters to your workload.
- **DuckDB PVC is RWX but single-writer by convention**, not by K8s enforcement. Never run
  the ingestion Job and an MCP server (or two ingestion Jobs) concurrently against it.
- **DuckDB PVC is owned by uid/gid 1000** — see the "Fixed resource names" section above.
  Match that uid or add your own chown init container; don't assume a fresh mount is
  writable by whatever uid your container runs as.
- **Node failure recovery is manual**, not automatic — see the Error Handling section of
  `docs/superpowers/specs/2026-08-03-nrp-storage-deployment-design.md`. If Postgres is down
  and its pod is stuck `Terminating`, check for a dead node before assuming a simple crash.
- **Restoring from backup has a documented, tested procedure** — see `k8s/nrp/RESTORE.md`
  before assuming a plain `psql < backup.sql` restore is safe for pgGraph's registration
  tables.

## Re-running the ingestion Job

`kubectl apply -f k8s/nrp/ingestion-job.yaml` a second time against an already-`Complete`
Job of the same name **fails** — Kubernetes Jobs are immutable once created (the API
server rejects most field changes on an existing Job object), so you must delete the old
Job first:

```bash
kubectl delete job knightgpt-ingestion -n knightlab-ml
kubectl apply -f k8s/nrp/ingestion-job.yaml
```

Deleting the Job does not delete `knightgpt-ingestion-data` or `knightgpt-duckdb` (separate
PVC manifests, by design — see the warning comment in `ingestion-pvc.yaml`) or anything
already written to Postgres, so a re-run resumes/adds to existing data rather than starting
from empty storage; `download_papers()`'s per-DOI skip-if-already-downloaded behavior means
a re-run is safe to do, not just possible.

**`:latest` is a moving tag.** `ingestion-job.yaml` pins `image:
ghcr.io/l1joseph/knightgpt-ingestion:latest` with `imagePullPolicy: Always`, so every
`kubectl apply` pulls whatever `latest` currently points at — merging any branch that
touches the paths watched by `.github/workflows/build-ingestion-image.yml` (`scripts/**`,
`src/**`, `docker/Dockerfile.ingestion`, `requirements.txt`, `data/paper_lists/**`)
triggers a rebuild that moves `latest` to a new image. If reproducing the *exact* bits
that produced a specific past run matters, use that run's `sha-<commit>` tag instead of
`latest` — the same workflow pushes both tags on every build (`type=sha,format=long` in
its `docker/metadata-action` step), so any prior commit's image is still pullable by tag
even after `latest` has moved on. This isn't done automatically as part of the manifest —
pinning `ingestion-job.yaml` to a specific `sha-` tag is a judgment call for whoever runs
the Job next (trade reproducibility against always-latest-code), not something forced by
default.

The image that produced the corpus currently in Postgres (160 papers, 8,299 chunks, 58,458
edges, as of this writing) was digest
`sha256:b19c80cc3ae08140fc61b88330dd3ff6b42c2d064de7b84a3486814f205c6b51` — recorded here
since nothing else in the repo captures it, and `latest` will have moved past it as soon as
this branch merges and CI rebuilds.

## Qiita registry ingestion Job

`qiita-registry-ingest-job.yaml` seeds `qiita_studies` (Stage 1 of a two-stage
plan) from redbiom's public index at `http://qiita.ucsd.edu:7329` — no
lab-network or credential dependency, unlike the main ingestion Job's Postgres
access. It enumerates every redbiom context, fetches the samples in each, and
summarizes them by `qiita_study_id`, upserting `study_id`, `sample_count`, and
which contexts each study appeared in. It deliberately leaves
`title`/`abstract`/`principal_investigator`/`funding`/`metadata` `NULL` — those
are a separate, future Stage 2 backfill via direct Postgres access to Qiita's
own database, not attempted here. This is a one-time run, not a CronJob.

### Live-run history — two runs, read both before touching this Job again

**Run 1 (2026-08-08 03:59-05:02 UTC): failed, zero rows.** The original
single-threaded orchestrator (`REDBIOM_TIMEOUT_S = 120`, no concurrency, a
single whole-run Postgres upsert only after all 357 contexts finished) hit
`activeDeadlineSeconds: 3600` (1 hour) with `qiita_studies` still empty.
Root cause, confirmed by live diagnosis inside the pod: the per-context
`redbiom summarize samples --category qiita_study_id --from <file>` call
did not complete even with a manually-extended 580-second timeout against a
~12,900-sample context — directly contradicting the "~12s for a 300k-sample
context" estimate the original 1-hour deadline was sized around. Because the
orchestrator only wrote to Postgres once, at the very end, the
deadline-killed run lost 100% of its progress.

**Fix applied**: `scripts/qiita_registry_ingest.py` was redesigned (commits
`f5d0943`, `8c63a9a`) to process contexts concurrently (`MAX_WORKERS = 5`
thread pool), checkpoint to Postgres incrementally after every context
(cumulative recompute-and-upsert, not additive — safe and idempotent even on
re-run), reuse a single connection pool across the whole run, and tolerate
individual checkpoint-write failures without aborting. `REDBIOM_TIMEOUT_S`
was also raised 120s → 300s. The image was rebuilt from this code and this
manifest's `activeDeadlineSeconds` raised 3600 → 21600 (6 hours) and
`resources` bumped 1 CPU/2Gi → 2 CPU/4Gi for the 5-worker concurrency.

**Run 2 (2026-08-08 05:46-10:29 UTC, 4h43m): reached `Complete`, but with
substantial, not full, coverage.** The incremental-checkpointing fix worked
as designed — Postgres rows grew steadily over the run's ~4.7 hours, visible
live via `SELECT count(*) FROM qiita_studies`, and the Job finished cleanly
within the 6-hour deadline this time. But the *underlying* redbiom
`summarize samples` slowness that caused Run 1's failure was not actually
resolved by raising the per-call timeout to 300s — it was only made
survivable. Final result: **80/357 contexts (22%) succeeded, 277/357 (78%)
still failed via the same `subprocess.TimeoutExpired` at 300s** (every one
of the 277 failures was a plain timeout, confirmed by grepping the full pod
log — no other exception type appeared), yielding **32 distinct studies**
in `qiita_studies` — far short of the ~500+ this Job was expected to
produce (one single context alone was expected to contribute 573 distinct
studies on its own; that context, along with most others, is presumably
among the 277 that timed out here). Study 10317 (American Gut Project) is
present but severely undercounted as a direct consequence: `sample_count =
32` across only 5 contexts, versus the tens-of-thousands and many-contexts
expected if the large contexts it actually belongs to hadn't mostly timed
out.

All 32 persisted rows are correctly Stage-1-only (`title IS NULL` for all
of them, confirmed by query), so what's there is *correct*, just
incomplete. No checkpoint-upsert failures occurred (0 occurrences of
"checkpoint upsert failed" in the full log) — every context that completed
its redbiom fetch+summarize successfully was reliably persisted.

**Status of this sub-project as of Run 2**: the "runs to completion without
losing all progress" bug from Run 1 is fixed and confirmed working live.
The "most contexts are too slow against redbiom's live public backend to
finish even in 300s" problem is a distinct, still-open issue — likely
requires either a fundamentally different query strategy against redbiom
(e.g. its Python API directly instead of shelling out to the CLI per call,
or batching/paginating within a context rather than one `summarize` call
over the whole sample list) or accepting that a full-coverage run may need
hours per individual large context rather than minutes. **Re-running this
Job as-is (`kubectl delete job ... && kubectl apply -f ...`) is
idempotent and will re-attempt all 357 contexts from scratch** (the
incremental upsert overwrites cumulative totals, it does not skip
already-succeeded contexts), so a re-run might pick up different contexts
succeeding/failing on a given day (redbiom's public backend responsiveness
appears to vary), but is not guaranteed to do meaningfully better without
addressing the root performance issue first. See
`.superpowers/sdd/2026-08-07-qiita-registry-ingestion/task-5-report.md` for
the full command-by-command record of both live runs.

## Known gap: NRP corpus is not a strict superset of the old Cosmos corpus

The design spec for the ingestion Job assumed the old Cosmos-based 117-paper corpus was
entirely a subset of `initial_papers.txt`/`zotero_papers.txt`, so a fresh full NRP
ingestion would naturally re-cover all of it with no separate migration step needed. In
practice, of those 117 papers, only 111 are present in the new NRP Postgres storage — 6
failed to re-fetch on this run (transient download failures, not permanently unavailable):

- `10.1038/nature13793`
- `10.1038/s41579-018-0029-9`
- `10.1038/s41591-019-0458-7`
- `10.1080/03014460.2025.2509606`
- `10.1101/gr.186072.114`
- `10.1101/gr.213959.116`

These 6 papers' original PDFs/markdown still exist on Cosmos scratch
(`/cosmos/vast/scratch/l1joseph/knightgpt/data/`) — they are not lost, just absent from the
NRP storage. If full parity with the old Cosmos corpus matters for a future use case (e.g.
the MCP server), these 6 would need to be sourced separately (e.g. copied from Cosmos
scratch and ingested directly) rather than assumed already covered. This is a known,
accepted gap as of the current run, not an open bug to chase.
