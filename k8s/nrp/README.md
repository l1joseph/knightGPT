# k8s/nrp — wiring contract for consumers

This directory provisions knightGPT's persistent storage on NRP (cluster
`nautilus`, namespace `knightlab-ml`): a Postgres+pgGraph Deployment, a
DuckDB PVC, and a weekly backup CronJob. This file documents the fixed
resource names and the exact environment variables **any future workload in
this namespace** (the not-yet-built GPU ingestion Job, the not-yet-built MCP
server, ad hoc debug pods, etc.) must set to connect correctly.

Manifests and what they own:

| File | Resource(s) |
|---|---|
| `postgres-pvc.yaml` | `PersistentVolumeClaim/knightgpt-postgres-data` (RWO, linstor-igrok) — **never delete**, see warning comment in that file |
| `postgres-cluster.yaml` | `Deployment/knightgpt-postgres`, `Service/knightgpt-postgres` |
| `duckdb-pvc.yaml` | `PersistentVolumeClaim/knightgpt-duckdb` (RWX, rook-cephfs) |
| `backup-pvc.yaml` | `PersistentVolumeClaim/knightgpt-backup-dest` (RWO, rook-ceph-block) — **never delete**, see warning comment in that file |
| `backup-cronjob.yaml` | `CronJob/knightgpt-backup` |
| `duckdb-verify-pod.yaml` | throwaway debug pod pattern for inspecting the DuckDB PVC |

Apply order matters where a PVC was split out from its owning workload:
`kubectl apply -f k8s/nrp/postgres-pvc.yaml -f k8s/nrp/postgres-cluster.yaml`
and `kubectl apply -f k8s/nrp/backup-pvc.yaml -f k8s/nrp/backup-cronjob.yaml`.

## Fixed resource names

- **Postgres Service DNS (in-cluster):** `knightgpt-postgres.knightlab-ml.svc.cluster.local:5432`
  (short form `knightgpt-postgres:5432` works for pods in the same namespace)
- **Postgres credentials Secret:** `knightgpt-postgres-credentials`, key `POSTGRES_PASSWORD`
  (user is always `postgres`, database is always `knightgpt` — see `postgres-cluster.yaml` env block)
- **DuckDB PVC:** `knightgpt-duckdb` (RWX — mountable read-write by exactly one writer at a
  time by convention; `DuckDBStore.__init__` raises a clear error on a lock conflict, there
  is no K8s-level enforcement)

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
- **Node failure recovery is manual**, not automatic — see the Error Handling section of
  `docs/superpowers/specs/2026-08-03-nrp-storage-deployment-design.md`. If Postgres is down
  and its pod is stuck `Terminating`, check for a dead node before assuming a simple crash.
- **Restoring from backup has a documented, tested procedure** — see `k8s/nrp/RESTORE.md`
  before assuming a plain `psql < backup.sql` restore is safe for pgGraph's registration
  tables.
