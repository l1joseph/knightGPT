# Postgres/pgGraph/pgContext Migration + Ingestion Scale-Up

**Status:** Draft — approved for planning
**Date:** 2026-08-01

## Problem

knightGPT's knowledge-graph build and query-time retrieval are both O(n²)-ish
brute force: `KnowledgeGraphBuilder` computes pairwise chunk similarity with a
nested Python loop calling `scipy.spatial.distance.cosine()` once per pair
(~38M scalar calls at the current 6,179-chunk scale), and
`GraphRAGRetriever.find_similar_chunks()` does a linear Python-loop scan at
query time — no vector index of any kind. The whole corpus
(`chunks_with_emb.json`, 578MB; `graph.graphml`, 17MB) is loaded fully into
memory at API startup with no storage abstraction underneath
`GraphRAGRetriever`.

The user wants to scale ingestion by an open-ended, potentially large amount:
four additional paper sources (three Google Sheets identified during design,
plus a pasted long-read-sequencing BioProject table) on top of the current
117 papers. At that scale the current architecture doesn't degrade
gracefully — it stops working (O(n²) build time, unbounded memory growth).

Separately, `docker-compose.yaml` already runs a Neo4j service
(`src/storage/storage.py`'s `sync_to_neo4j()`), but it's a write-only mirror —
nothing in the retrieval path reads from it. It's live infrastructure that
isn't earning its keep.

## Decisions made during brainstorming

- **Architecture:** migrate to self-hosted Postgres with the `pgGraph`
  (graph traversal) and `pgContext` (HNSW vector search) extensions, both
  Apache-2.0, both from Evokoa/Polygres (`github.com/evokoa`). Self-hosted via
  `docker-compose.yaml`, not the managed Polygres service — no external
  billing or network dependency for retrieval calls, consistent with how the
  rest of the stack (Neo4j today) is run.
- **Neo4j:** retire entirely. Replace with pgGraph. One graph-ish system
  instead of two.
- **Scope of this batch:** paper-level ingestion only, from all four sources
  identified during design (see ETL section). Sample/accession-level metadata
  (e.g. the NCBI/ENA attributes tab in the third sheet, and BioProjects in
  the fourth source with no associated publication) is out of scope.
- **DOI resolution for sheets without a DOI column:** auto-resolve via
  OpenAlex/PMID lookup, the same pattern `scripts/populate_zotero.py` already
  uses for OpenAlex discovery.
- **Existing 117 papers / 6,179 chunks:** migrate the already-computed
  embeddings and graph edges into Postgres as-is. Do not re-embed or
  recompute similarity — avoids vLLM compute cost and avoids introducing
  embedding drift from a model rerun.
- **Storage location:** move everything — Postgres data directory, raw PDFs,
  markdown intermediates — from `/cosmos/vast/scratch/l1joseph/knightgpt`
  (shared Vast scratch, 675TB, subject to different retention behavior) to
  `/sdsc/scc/ddp478/l1joseph/knightgpt` (dedicated 1.7PB Ceph-backed project
  allocation, currently empty, not general scratch).

## Architecture

Replace the file-based NetworkX/JSON storage and the write-only Neo4j service
with a single self-hosted Postgres instance running `pgGraph` and `pgContext`,
deployed as a new `postgres` service in `docker-compose.yaml` in place of the
existing `neo4j` service. The API (`src/api/main.py`) connects to it via a
connection pool instead of loading flat files into memory at startup.

## Data model

```sql
-- one row per paper
papers (
  doi          text PRIMARY KEY,
  title        text,
  metadata     jsonb  -- authors, year, journal, source_sheet, etc.
)

-- one row per chunk
chunks (
  id           bigint PRIMARY KEY,
  paper_doi    text REFERENCES papers(doi),
  text         text,
  embedding    vector(3584),  -- pgContext HNSW index, cosine
  section      text,
  token_count  int
)

-- similarity edges, registered with pgGraph as edge-table-style relationships
chunk_edges (
  src_chunk_id bigint REFERENCES chunks(id),
  dst_chunk_id bigint REFERENCES chunks(id),
  similarity   real
)
```

`chunk_edges` is registered with pgGraph via
`graph.add_edge(from_table := 'chunk_edges', from_column := 'src_chunk_id',
to_table := 'chunks', to_column := 'dst_chunk_id', weight_column :=
'similarity')` — pgGraph's documented "edge-table style registration," which
exists exactly for this case (one table holding relationship rows, as opposed
to FK-discovered relationships). The `weight_column` also enables
`graph.weighted_shortest_path()` if that's ever useful later.

## Ingestion pipeline change

This is what actually removes the O(n²) bottleneck. Per new chunk:

1. Embed via vLLM (unchanged).
2. Insert the row into `chunks`.
3. Issue one pgContext HNSW query for the top-10 neighbors ≥0.7 similarity
   among chunks already indexed.
4. Insert the resulting rows into `chunk_edges`.

This replaces the nested-loop `scipy.cosine()` brute force with one ANN query
per chunk, and — importantly for "open-ended scale" — makes ingestion
**incremental**: ingesting paper #5,000 costs one query against the existing
index, not a full recompute over the whole corpus. `graph.build()` runs once
at the end of each ingestion batch; pgGraph's graph is derived/rebuildable
state, not live-updated on individual writes, and "manual build after batch"
is its documented operational fit for a mostly-batch write pattern like
ingestion (as opposed to `trigger` sync mode, which is meant for continuous
write workloads).

Query-time retrieval changes correspondingly:
- `find_similar_chunks` → one pgContext HNSW query.
- `expand_context` (k-hop neighbor expansion) → one pgGraph traversal query.

## ETL for the four paper sources → DOI lists

`scripts/etl_sheet_to_dois.py`, one resolver per source, all converging on the
existing `data/paper_lists/*.txt` DOI-list format that `download_papers.py`
already consumes — no changes needed to the download/convert/chunk stages.

- **MMC 2025 Data Sheet** (source for the existing `mmc_datasheet.tsv`, much
  larger than the 194 DOIs currently ingested from it): DOI column present
  directly. Dedupe to one row per study — the full sheet mixes study-level
  and per-sample rows across tabs; only study-level rows are in scope.
- **Cancer Qiita curation tracker**: not a paper list — a working curation
  doc, ~780 rows across 13 sections deduplicating to ~100-150 unique Qiita
  studies, no DOI column (only PubMed/PMC/Nature URLs), some exact
  duplicates and free-text curator notes. Dedupe by `qiita_id` first, then
  extract PMID/PMCID from `article_link` and resolve to DOI via OpenAlex.
- **Global Human Gut Microbiome Project**: has two relevant tabs — a
  sample/accession-level tab (out of scope) and a study-level tab
  (`Study | Title | Publication Year | Link to paper | Accession#`). Most
  `Link to paper` URLs directly encode a DOI in the path (e.g.
  `.../doi/full/10.1128/mbio.00519-19` → `10.1128/mbio.00519-19`) — extract
  via regex; OpenAlex fallback for rows without a doi-shaped URL.
- **Long-read metagenomics BioProject table** (pasted directly, not a Drive
  sheet — 72 rows of ONT/PacBio-sequenced BioProjects): the cleanest of the
  four sources. 50 of 72 rows have a `Link` that's already a bare
  `doi.org/<doi>` URL — the DOI is the literal path suffix, no extraction
  logic needed. 2 more rows (bioRxiv/medRxiv preprints not yet mirrored to
  `doi.org`) need the same URL-path DOI-regex extraction as source 3. The
  remaining 20 rows are explicitly flagged in the `Citation` column as
  `no publication` or a non-scholarly vendor blog post — these have no paper
  text to ingest and are skipped (not treated as ETL failures; the source
  data itself says there's nothing to fetch).

Rows that fail DOI resolution are logged to `etl_failures.csv` for manual
follow-up rather than silently dropped or blocking the batch. Rows explicitly
marked as having no publication (source 4) are logged separately as skipped,
not failed.

## Migrating the existing 117 papers

One-time `scripts/migrate_to_postgres.py`: reads `chunks_with_emb.json` and
`graph.graphml` and inserts into `papers`/`chunks`/`chunk_edges` as-is. No
re-embedding, no re-computed similarity — existing edges carry over verbatim.
Dry-run mode asserts pre/post chunk and edge counts match before any
destructive step (e.g. decommissioning the old file-based path).

## Retriever refactor

`GraphRAGRetriever` is currently hard-coupled to file paths (`chunks_path`,
`graph_path` constructor args) with no interface underneath it. Introduce a
thin storage interface so `RAGEngine` and the agent tools in `src/agents/`
depend on an abstraction rather than a concrete file-backed class. Implement
`PostgresRetriever` against that interface via `asyncpg`. API startup
(`src/api/main.py` lifespan handler) opens a connection pool instead of
loading 578MB + 17MB into memory — this is also what makes "open-ended" scale
actually feasible, since memory-loading a corpus 10x the current size
wouldn't work at all under the current design.

## Decommissioning

- Remove the `neo4j` service from `docker-compose.yaml`.
- Remove `Neo4jStorage` / `sync_to_neo4j()` from `src/storage/storage.py`.
- Remove `--sync-neo4j` flags from `scripts/download_papers.py`,
  `scripts/ingest_pipeline.py`, and the corresponding `sync_neo4j` params on
  the RSS/briefing ingest endpoints in `src/api/main.py`.
- Remove `neo4j` from `requirements.txt` and `environment.mi300a.yml`.

## Error handling

- ETL: unresolvable rows (broken URLs, PMIDs with no OpenAlex match) are
  logged to `etl_failures.csv`, not dropped silently and not fatal to the
  batch.
- Ingestion: existing `tenacity` retry pattern in `VLLMEmbedder` is
  unchanged. Each paper's chunk+edge inserts run in one transaction, so a
  failed paper doesn't leave partial chunks/edges in the graph.
- Migration: dry-run mode validates chunk/edge counts match between the old
  files and the new Postgres tables before anything old is removed.

## Testing

- Unit tests for each source resolver (`tests/test_etl.py`) using mocked
  rows → expected DOI list, covering the direct-DOI, PMID-resolution,
  URL-regex-extraction, and no-publication-skip paths independently.
- Integration test standing up Postgres+pgGraph+pgContext (docker-compose
  test profile), verifying the round trip: insert chunk → pgContext HNSW
  query → pgGraph traversal query. This also gives a home for logic currently
  blocked by the GraphML-dict-serialization failure in the existing
  `test_integration.py`.
- Migration script dry-run against a copy of current production data,
  asserting chunk and edge counts match exactly.

## Out of scope for this design

- Ingesting the sample/accession-level metadata from any of the four
  sources (BioSample/BioProject attributes, per-sample SRA data, BioProjects
  with no associated publication) — paper-level only, per the scoping
  decision above.
- Managed Polygres hosting.
- Changes to the vLLM embedding/inference SLURM pipeline itself — this design
  only touches storage and retrieval, not the embedding/generation path.
