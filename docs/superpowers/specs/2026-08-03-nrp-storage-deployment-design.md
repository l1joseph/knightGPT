# NRP Storage Deployment (Postgres+pgGraph + DuckDB PVC)

## Context

knightGPT's Postgres+pgGraph+DuckDB code (`sql/schema.sql`, `docker/postgres/Dockerfile`, `src/graph/duckdb_store.py`, `src/graph/postgres_builder.py`, `src/retrieval/hybrid_retriever.py`, `scripts/migrate_to_postgres.py`) merged into `vllm` via PR #2, validated against real production data (117 papers, 6,179 chunks, 37,286 edges) through ephemeral Singularity-hosted Postgres instances on the SDSC Cosmos SLURM cluster — but has never run anywhere persistent.

The broader goal is a three-part project: (1) this storage deployment, (2) a GPU-backed ingestion Job to bulk-ingest the full paper corpus (~900 combined entries across `initial_papers.txt`, `zotero_papers.txt`, `mmc_papers.txt`, plus the long-read BioProject table), (3) an MCP server exposing retrieval to any MCP-compatible client, independent of which model is being used. All three run on NRP (National Research Platform, Kubernetes cluster `nautilus`, namespace `knightlab-ml`), chosen because: Cosmos/Barnacle2 have no persistent, always-reachable hosting; NRP explicitly forbids GPU access for long-running Deployments but permits it for bounded-duration Jobs (confirmed via NRP's own policy docs: *"Such a deployment can not request a GPU"*); the namespace already has real, unused A100 GPU quota (16 allocated, 0 used) sufficient for the embedding step.

This spec covers only the first part — provisioning the storage. The ingestion Job and MCP server both depend on it existing first and are separate specs.

## Decisions

- **Postgres operator: CloudNativePG (CNPG).** One of NRP's two documented, supported operators (the other being the Zalando operator). CNPG explicitly supports custom container images with third-party extensions baked in — verified via CNPG's own documentation, not just NRP's (which doesn't cover custom-extension specifics). Chosen over Zalando as the more actively-developed option, and over self-managing Postgres directly (a plain Deployment/StatefulSet) to get managed restart/health behavior without owning that logic ourselves.
- **Custom image, not CNPG's newer ImageVolume extensions feature.** CNPG 1.27+ supports mounting an extension as a separate OCI image at pod startup (Kubernetes ImageVolume + PostgreSQL 18's `extension_control_path`), but that requires PostgreSQL 18. Our existing `docker/postgres/Dockerfile` already builds a full custom Postgres 17 + pgGraph image — reusing it via CNPG's standard custom-image support (`imageName` in the Cluster CRD) avoids rebuilding pgGraph against a newer major version for no functional benefit right now.
- **Image build: GitHub Actions → GHCR**, not local Docker/Podman (unavailable on both Cosmos and Barnacle2) and not NRP in-cluster build tooling (unconfirmed/undocumented, adds a dependency on NRP-specific mechanics for something GitHub Actions already solves cleanly with real Docker on hosted runners).
- **DuckDB storage: a standalone RWX PVC (`rook-cephfs`), not part of the Postgres deployment.** DuckDB isn't a Postgres extension — it's a separate embedded file opened directly by the Python process. The PVC needs `ReadWriteMany` so different pods (the future ingestion Job, the future MCP server) can mount it at different times, even though DuckDB's own single-writer lock (already documented in this repo's `CLAUDE.md`) means only one of them can hold it read-write at once. That constraint is enforced by convention/documentation here, not new coordination code — consistent with the equivalent decision already made for the Cosmos-side deployment.
- **Backup: basic (PVC durability + snapshots if available), not a full pipeline.** Source PDFs plus re-embedding could regenerate the data if lost, so this doesn't need production-grade DR from day one. CSI volume snapshot support for this namespace's storage classes isn't confirmed (a cluster-scoped `VolumeSnapshotClass` listing returned Forbidden at the available permission level) — attempt it first; if unavailable, fall back to a periodic `pg_dump` + DuckDB file copy to Cosmos scratch via a lightweight CronJob, tracked as a documented follow-up rather than blocking this deployment.
- **No external network exposure.** Only in-cluster reachability (ClusterIP) is needed — the future ingestion Job and MCP server both run inside the same namespace. External exposure is the MCP server's concern in its own later spec, not this one.

## Architecture

Two resources in the `knightlab-ml` namespace: a CNPG `Cluster` running the custom Postgres 17 + pgGraph image (built by GitHub Actions, pulled from GHCR), and a standalone `rook-cephfs` PVC for the DuckDB file. Neither is exposed outside the cluster. This spec provisions both and verifies them; it does not run ingestion or serve queries against them yet.

## Components

- **`.github/workflows/build-postgres-image.yml`** (new): builds `docker/postgres/Dockerfile` on push to `vllm` (path-filtered to `docker/postgres/**`) and on manual dispatch, tags and pushes to `ghcr.io/l1joseph/knightgpt-postgres`, using `docker/build-push-action` with GHCR login via the built-in `GITHUB_TOKEN`.
- **`k8s/nrp/postgres-cluster.yaml`** (new): a CNPG `Cluster` custom resource — 1 instance (no replicas yet, this is new infrastructure not yet serving traffic), `imageName: ghcr.io/l1joseph/knightgpt-postgres:<tag>`, `storage.storageClass: linstor-igrok` (NRP's documented preferred class for Postgres performance), `storage.size: 20Gi`, `postgresql.parameters.shared_preload_libraries: graph` matching the existing Dockerfile's `CMD ["postgres", "-c", "shared_preload_libraries=graph"]`. Sizing: chunk text/metadata/graph-edges (no embeddings, those live in DuckDB) is currently ~516MB for 117 papers (605MB combined `chunks_with_emb.json` minus its ~89MB embedding portion); scaled to the ~900-paper target corpus, ~4GB — 20Gi leaves 5x headroom.
- **`k8s/nrp/duckdb-pvc.yaml`** (new): a plain `PersistentVolumeClaim`, `storageClassName: rook-cephfs`, `accessModes: [ReadWriteMany]`, `resources.requests.storage: 20Gi`. Sizing: 6,179 chunks × 3584-dim float32 embeddings is ~89MB today; scaled to ~900 papers (~47,500 projected chunks), ~680MB — 20Gi leaves roughly 29x headroom for HNSW index overhead and well beyond the current corpus target.
- **Schema application**: reuse `scripts/apply_schema.py` (already exists, already validated this session) against the new CNPG-created Postgres instance — no new schema-application code, just pointing the existing script at a new DSN (via a `kubectl port-forward` or a throwaway debug pod, since there's no external exposure).
- **Backup CronJob** (`k8s/nrp/backup-cronjob.yaml`, new — only if CSI snapshots turn out unavailable): periodic `pg_dump` of the Postgres cluster plus a copy of the DuckDB file, written to `/cosmos/vast/scratch/l1joseph/knightgpt/backups/` (requires the CronJob's pod to reach Cosmos scratch — needs confirming NRP pods can reach that path at all, likely via an NFS-backed PV or similar; flagged as a real open question for the implementation plan to resolve, not assumed solvable here).

## Data Flow

None yet — this spec is provisioning-only. Once complete: a future ingestion Job connects to Postgres via CNPG's in-cluster service DNS (`<cluster-name>-rw.knightlab-ml.svc.cluster.local`) and mounts the DuckDB PVC read-write to populate it; a future MCP server does the same for reads, never running concurrently with the ingestion Job against the DuckDB PVC.

## Error Handling

CNPG handles Postgres-level process failures (restart behavior is the operator's job). The DuckDB PVC's single-writer constraint has no K8s-level enforcement in this spec (e.g. no admission webhook preventing two pods from mounting it simultaneously) — it's a documented operational convention, matching the "fail loudly, document the limitation" approach already taken for the equivalent constraint on Cosmos (`DuckDBStore.__init__` already raises a clear error on a lock conflict; the same code runs unchanged here).

## Testing

Infrastructure verification checklist, not a pytest suite:
- CNPG `Cluster` resource reaches `Cluster` status `Healthy`/instances `Running`.
- `psql` against the in-cluster service (via port-forward) confirms the `graph` extension is loaded (`\dx`) and `graph.registered_tables()`/`graph.registered_edges()` are callable.
- `scripts/apply_schema.py` runs clean against the new instance, including its existing post-apply pgGraph registration verification step.
- A throwaway debug pod mounting the DuckDB PVC can `duckdb.connect()` a test file successfully with `INSTALL vss; LOAD vss;` working (confirms the PVC's filesystem semantics are compatible with DuckDB — CephFS is POSIX-compliant so this is expected to work, but not yet verified on this specific storage class).
- If a second debug pod attempts to open the same DuckDB file read-write while the first still holds it, confirm it fails with the documented lock-conflict error (not a hang or a silent corruption) — direct verification that the Cosmos-validated behavior carries over to NRP's CephFS-backed RWX volumes unchanged.
