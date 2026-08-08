# Qiita Study Registry Ingestion (Stage 1)

## Context

Part 3 of knightGPT's NRP deployment plan (see `docs/superpowers/plans/2026-08-06-nrp-ingestion-job.md` for parts 1-2, already merged) was originally scoped narrowly as "an MCP server over the paper corpus." Leo confirmed on 2026-08-06 that it now absorbs a previously-separate Qiita ingestion initiative, decomposed into: Qiita structured ingestion → paper↔study cross-referencing → a unified MCP server exposing both the paper corpus (already live: 160 papers / 8,299 chunks / 58,458 edges on NRP Postgres+DuckDB) and Qiita study data.

This spec covers only the first slice of the first phase: a Qiita study **registry** — the set of study IDs that exist, with sample counts, seeded from a data source that's usable *today*. The richer study metadata (title, abstract, PI, funding, publication links) needed for actual cross-referencing and retrieval is a separate, follow-on stage blocked on external access (see Decisions).

### Landscape investigated this session

- **QiitaExplore** (an existing lab chatbot for natural-language study search over Qiita's Postgres tables) is explicitly out of scope — not something knightGPT integrates with or depends on.
- **the-miint/Qiita** (`github.com/the-miint`) is a ground-up rewrite of Qiita itself (FastAPI control plane + Rust/Arrow-Flight/DuckDB data plane), under active development, explicitly "do not use" per its own README — no live data, and its schema (verified directly) has no publication/DOI linkage anyway. Not a near-term dependency.
- **Direct Postgres access to production Qiita's own database** (`kl-db`/`terrance.ucsd.edu:5432`, database `qiita`) is the right long-term source for real study metadata — it has title/abstract/PI/funding fields (confirmed via the-miint's carried-forward schema shape; the *original* production schema's exact columns weren't directly confirmed since no DB connection was achieved). It is **not reachable today**: investigated directly from Barnacle2 compute node `b1` (read-only, no research data read) and found `kl-db` doesn't resolve there at all (no DNS, no route). A separate, not-fully-confirmed check found Postgres reachable from some other host but rejected at the `pg_hba.conf` level — a different, more easily fixable failure mode. Either way, unblocking this requires a lab admin to grant a read-only role and confirm/fix network routing. This is an external, human-coordination dependency outside this project's control.
- **redbiom** (a Knight Lab CLI/library for querying processed Qiita sample data) is usable **today**, and its backend is genuinely public: `redbiom`'s default host (`redbiom/__init__.py`'s `REDBIOM_HOST` default) is `http://qiita.ucsd.edu:7329`, a public Webdis/Redis REST endpoint. Verified by installing `redbiom` and running `redbiom summarize contexts` successfully from a network entirely separate from Barnacle2 (the Cosmos cluster) — confirming this is not gated to lab-internal network access at all. redbiom is sample/feature-search-oriented and exposes **no study-level metadata** (no title, abstract, PI, funding, or DOI) — but it does let us enumerate every study ID currently represented in Qiita's processed data, with per-study sample counts, via a confirmed-working two-step recipe (see Decisions).

## Decisions

- **Two-stage phase 1, this spec covers only Stage 1.** Stage 1 (this spec): seed a `qiita_studies` registry table from redbiom — study IDs, sample counts, which processing contexts each appears in. Stage 2 (future, separate spec): backfill title/abstract/PI/funding/publication links once direct Postgres access to Qiita's own database is granted. Splitting this way means real, useful progress happens now instead of blocking entirely on an external admin ask, without permanently settling for degraded data — Stage 2's backfill is a straightforward `UPDATE` pass over the same rows once unblocked, not a redesign.
- **Runs on NRP, not Barnacle2.** Since redbiom's backend is a public internet endpoint (confirmed above), Stage 1 needs no lab-network access at all. It runs as a plain Kubernetes Job in the `knightlab-ml` namespace, following the exact pattern established by the paper-ingestion Job (`k8s/nrp/ingestion-job.yaml`) — except with **no GPU sidecar**, since redbiom calls are lightweight HTTP requests, not embedding computation.
- **One-time run for now, not recurring.** Qiita grows continuously, so an eventual scheduled refresh makes sense — but building that now, around only half the eventual data (no metadata until Stage 2), would be premature. Revisit cadence once Stage 2 exists.
- **Scope: every study redbiom's public index exposes**, not a narrower Knight-Lab-only filter. redbiom's index is presumably already scoped to what's publicly processed/available; no additional filtering logic in Stage 1.
- **Deliberately descoped from Stage 1: per-study sample-metadata category rollups** (e.g. environment-type distribution per study) — a real, confirmed redbiom capability (`redbiom summarize samples --category <cat>`), but applying it per-study across 573+ studies would multiply the number of redbiom calls substantially for a signal that's a rough proxy at best (real abstracts from Stage 2 will be strictly better). Worth a deliberate follow-up once the core registry exists, not part of this first cut.
- **redbiom's cross-context aggregation, not a database dump.** A study can have samples processed under multiple contexts (e.g. different trimming lengths or reference sets) — the registry aggregates a study's total sample count across all contexts it appears in, plus records which contexts it appeared in (for provenance/debugging), rather than treating each context separately.

## Architecture

A single Kubernetes Job named `knightgpt-qiita-registry-ingest` in `knightlab-ml`: one container, Python, `redbiom` installed as a dependency, network egress to the public `qiita.ucsd.edu:7329` endpoint, connects to the existing NRP Postgres via the same `POSTGRES_DSN` wiring contract every other workload in this namespace uses (`k8s/nrp/README.md`). No GPU, no sidecar, no new PVC — this is a lightweight metadata-only job with no large intermediate files to persist across a restart.

## Components

- **`sql/schema.sql`** (extended): new `qiita_studies` table.
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
  `title`/`abstract`/`principal_investigator`/`funding`/`metadata_backfilled_at` all stay `NULL` after Stage 1 — populated only by the future Stage 2 backfill. `metadata` is reserved for Stage 2's publication/DOI links, following the same free-form-JSONB convention as `papers.metadata`.
- **`scripts/qiita_registry_ingest.py`** (new): the orchestrator.
  - Enumerates all redbiom contexts (`redbiom summarize contexts`).
  - Per context: `redbiom fetch samples-contained --context <ctx>` → `redbiom summarize samples --category qiita_study_id --from <file>` → per-context study_id → sample_count mapping.
  - Merges across all contexts into one study_id → {total sample_count, [contexts]} map (pure function, unit-testable without live redbiom).
  - Upserts into `qiita_studies` (`ON CONFLICT (study_id) DO UPDATE SET sample_count = ..., contexts = ...` — idempotent, so a re-run is safe and doesn't clobber any Stage-2-backfilled columns since those aren't touched by this UPDATE).
- **`k8s/nrp/qiita-registry-ingest-job.yaml`** (new): the Job manifest — plain Python container, `POSTGRES_DSN` wired per the established contract (`POSTGRES_PASSWORD` before `POSTGRES_DSN` in the env list — a real bug from part 2 that must not be repeated), Gatekeeper-compliant `resources.requests`/`limits`, no GPU, no sidecar, no new PVC.
- **`docker/Dockerfile.qiita-registry`** (new, or possibly reuse `Dockerfile.ingestion` if its base image already suits this) + a CI workflow mirroring `build-ingestion-image.yml` — decide during planning whether a new lightweight image is warranted or whether extending the existing ingestion image (adding `redbiom` to its `requirements.txt`) is simpler, given this job doesn't need any of the heavy PDF/embedding dependencies the ingestion image carries.

## Data Flow

Job starts → enumerate contexts → for each context, fetch sample IDs and aggregate to per-study counts (each context's fetch/aggregate is independent and could in principle run concurrently, though a first cut can do this sequentially given ~50 contexts and the confirmed ~12-second-per-300k-sample-context performance) → merge all contexts' results into one registry → upsert into Postgres → log a summary (total studies found, total samples covered, any context failures) → exit.

## Error Handling

A single context's redbiom calls failing (network hiccup, malformed context name, etc.) is logged and non-fatal — the job continues with the remaining contexts, and the final summary reports how many contexts succeeded vs. failed. If **every** context fails, that's treated as a whole-job failure (raises, non-zero exit) rather than silently upserting an empty or trivial registry — mirroring the `all_embeddings_missing` total-outage pattern from the paper-ingestion job (part 2).

## Testing

- Unit tests for the pure cross-context merge function: given fake per-context `{study_id: count}` dicts (including a study appearing in multiple contexts, to verify correct summing), verify the merged registry is correct. No live redbiom needed for this.
- Live verification once implemented: run the Job, check `SELECT count(*) FROM qiita_studies` is in the right neighborhood (500+ given one context alone had 573), and spot-check a well-known study ID (e.g. 10317, the American Gut Project) for a plausible sample count.
