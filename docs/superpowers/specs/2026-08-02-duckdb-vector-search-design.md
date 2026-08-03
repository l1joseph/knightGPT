# DuckDB Vector-Search Redesign

## Context

The original design (`2026-08-01-postgres-graph-migration-design.md`) migrated knightGPT's knowledge-graph storage to Postgres with two extensions: pgGraph (graph traversal) and pgContext (vector search via HNSW). That work landed on `feature/postgres-graph-migration` (19 commits, PR #2 against `vllm`) and was validated end-to-end against real production data (117 papers, 6,179 chunks, 37,286 edges — exact match).

Before merging, we ran a series of benchmarks to validate pgContext's HNSW index at increasing scale (6,179 / 61,790 / 617,900 rows, real diverse embeddings, both NFS and local-disk storage). Findings:

- pgContext's HNSW index showed **no reliable median speedup** over sequential scan at any tested scale, with high query-time variance at the largest scale (one query 7.6x slower than exact search, alongside occasional genuine sub-millisecond wins).
- pgContext has a hard, documented limitation: HNSW node records are capped at 8,064 bytes (one Postgres page), which flatly blocks native 3584-dim float32 vectors (14,336 bytes) — the current production embedding dimension (gte-Qwen2-7B-instruct). This is why validation used a separately re-embedded 1024-dim dataset (Qwen3-Embedding-4B via MRL truncation).
- A comparative benchmark of DuckDB's `vss` extension (HNSW via usearch) against the same real data showed a clean, textbook result: HNSW query time stayed flat (~7ms) regardless of scale while exact search grew with data size, delivering 1.75x/9.7x/25x speedups at the three tested scales — with zero high-variance outliers. DuckDB's exact search was also 7-14x faster than Postgres's at every scale tested, independent of HNSW.
- A follow-up validation at the **native production dimension (3584-dim, real `chunks_with_emb.json`, no re-embedding)** confirmed DuckDB handles this dimension natively with no page-size constraint, and HNSW still delivers real speedup: 3.3x at 6,179 rows (39.8ms → 12.0ms), 13.9x at 61,790 rows (241.4ms → 17.3ms).

Separately, a related lab tool (ezredbiom → QiitaExplore, org "the-miint" / Miint) has independently standardized its own data plane on DuckDB, reinforcing this as a credible choice for this domain and scale (Qiita's real target scale is ~437,000+ samples).

**Conclusion:** pgContext is not a solid foundation for the production vector-search decision, independent of scale. DuckDB+vss is. This spec redesigns the vector-search half of the migration to use DuckDB while keeping the validated Postgres+pgGraph half (graph traversal, `papers`/`chunk_edges` tables) as-is — they are cleanly separable in the existing codebase (confirmed via code exploration: `schema.sql`, `BaseRetriever`, and the Dockerfile's build stages all separate pgGraph from pgContext already). The one clear consequence: this also means the previously-scoped "Phase 0" embedding-dimension-switch project (re-embedding all chunks with Qwen3-Embedding-4B at 1024-dim to work around pgContext's page-size limit) is no longer necessary — DuckDB works fine at the current native 3584-dim.

## Decisions

- **Scope:** Hybrid only. Keep Postgres+pgGraph for graph traversal; swap pgContext → DuckDB+vss for vector search. Not reconsidering pgGraph itself (out of scope — no benchmark evidence against it).
- **Deployment model:** DuckDB embedded in the same single Python process that already runs ingestion and query serving (confirmed: no multi-worker/multi-replica deployment exists or is planned). One read-write connection, no locking layer needed.
- **Branch:** Continue on `feature/postgres-graph-migration` — new commits on top of the validated pgGraph/papers/schema work, not a fresh branch.
- **Dimension:** Native 3584-dim (current production `gte-Qwen2-7B-instruct` embeddings). No re-embedding, no model swap.
- **Storage split (Approach A — split ownership):** Postgres owns chunk text/metadata + graph edges (unchanged responsibility). DuckDB owns exactly one lean table: chunk ID + embedding. No duplication of chunk text across stores. Rejected the alternative (DuckDB owning full chunk records, Postgres holding only graph structure) because it depends on an unverified assumption — whether pgGraph's `graph.add_table('public.chunks', ..., columns=[text, section])` registration actually needs live data in those columns, or just typed columns. Approach A sidesteps that question entirely.

## Architecture

Postgres+pgGraph stays exactly as validated for text storage and graph traversal (`papers`, `chunks` minus its `embedding` column, `chunk_edges`, pgGraph registration DO blocks). A new DuckDB file, `/cosmos/vast/scratch/l1joseph/knightgpt/data/processed/embeddings.duckdb`, owns exactly one table: `chunk_embeddings(id VARCHAR, embedding FLOAT[3584])`, with an HNSW index via the `vss` extension (`metric = 'cosine'`, matching the `array_cosine_distance` function used for both exact and index-accelerated queries — the function must match the index's configured metric or the query optimizer silently falls back to sequential scan, a real footgun discovered during benchmarking). DuckDB is embedded in-process — no server, no container, no Singularity sandboxing needed (a significant deployment simplification over the Postgres-extension-build path).

## Components

- **`sql/schema.sql`**: Drop the `embedding pgcontext.vector(3584)` column and the `chunks_embedding_hnsw` index from `chunks`. Drop the pgContext `CREATE EXTENSION` statement. `chunk_edges` and pgGraph registration DO blocks are unchanged.
- **New `src/graph/duckdb_store.py`**: Thin wrapper around a DuckDB connection. Public interface:
  - `insert_embeddings(rows: list[tuple[str, list[float]]]) -> None` — batch insert (id, embedding) pairs.
  - `search(query_embedding: list[float], top_k: int) -> list[tuple[str, float]]` — returns (id, cosine_similarity) pairs via the HNSW index.
  - `get_embeddings(ids: list[str]) -> dict[str, list[float]]` — direct ID lookup, used for graph-neighbor rescoring.
  - `ensure_index() -> None` — idempotent HNSW index creation, called once at startup/after bulk load.
  - Owns the DuckDB connection lifecycle (`connect()`/`close()`), analogous to how `PostgresRetriever` owns its asyncpg pool.
- **`src/graph/postgres_builder.py::insert_chunks()`**: Per chunk — insert paper (if new) + chunk row into Postgres (no embedding column), insert the embedding into DuckDB via `duckdb_store.insert_embeddings()`, query DuckDB (not pgContext) for nearest-neighbor candidates, insert qualifying results as `chunk_edges` rows into Postgres. Same overall per-chunk-transaction shape as today; only the neighbor-query backend changes. Batching note: DuckDB inserts should be batched (not one `INSERT` per chunk in isolation) since per-row inserts via list/array unnesting proved catastrophically slow in benchmarking (90s+ for 6,179 rows) versus a bulk DataFrame-backed `CREATE TABLE AS SELECT` (0.3s for the same data) — the implementation must use DuckDB's fast bulk-insert path even when chunks arrive incrementally, e.g. buffering a small batch per paper rather than truly one-row-at-a-time.
- **`src/retrieval/postgres_retriever.py` → `src/retrieval/hybrid_retriever.py`** (rename reflects the two-store reality): Vector search step queries DuckDB for top-k (id, score) pairs, then fetches full chunk rows from Postgres by ID (`WHERE id = ANY($1)`). Graph-expansion step is unchanged (pgGraph `graph.expand()`), but the neighbor-rescoring step — which today recomputes cosine similarity via pgContext for pgGraph-discovered neighbor IDs — now fetches those neighbors' embeddings from DuckDB via `get_embeddings()` and computes cosine similarity in Python (a handful of IDs at a time; no need for the HNSW index at this step, exact computation over ~10s of vectors is trivially fast).
- **`scripts/migrate_to_postgres.py`**: Forked embedding-write path — chunk text/metadata inserted into Postgres as today, embeddings inserted into DuckDB via the new store module. Dry-run mode and existing count-verification logic preserved unchanged.
- **`docker/postgres/Dockerfile`**: Drop the `pgcontext-builder` build stage and its two `COPY --from=pgcontext-builder` lines. The `pggraph-builder` stage and its `COPY --from=pggraph-builder` lines, and `shared_preload_libraries=graph` in `CMD`, are untouched (confirmed independent in exploration).
- **API image**: `duckdb` added as a pip dependency (already confirmed installable, no build complications). The DuckDB file lives on the same mounted scratch volume (`/cosmos/vast/scratch/l1joseph/knightgpt/data/processed/`) as other processed artifacts — no new volume/mount needed.

## Data Flow

**Ingestion:** PDF → chunks (existing chunker, unchanged) → embed via vLLM (existing embedder, unchanged, still 3584-dim) → Postgres write (paper + chunk text/metadata) + DuckDB write (chunk embedding) → DuckDB ANN query for edge candidates (replaces the old live pgContext query) → Postgres write (`chunk_edges`) → `graph.build()` (unchanged, called once after the full batch).

**Retrieval:** query text → embed (unchanged) → DuckDB top-k search → Postgres fetch of full chunk rows by ID → pgGraph `graph.expand()` for hop-neighbor IDs (unchanged) → DuckDB embedding fetch for those neighbor IDs → cosine rescoring in Python → merge, sort, return `RetrievalResult` (unchanged shape, satisfies `BaseRetriever.retrieve()`).

## Error Handling

Postgres and DuckDB are not covered by one transaction — there is no two-phase commit across the two systems, and none is being added (this is a research/RAG system, not a system requiring cross-store atomicity guarantees). Write order is: Postgres chunk row first (this makes Postgres the source of truth for "does this chunk exist"), then the DuckDB embedding write, then the Postgres `chunk_edges` write (which depends on the DuckDB write having succeeded, since it queries DuckDB for neighbors). If the DuckDB write fails, the chunk exists in Postgres but has no embedding and no edges — a detectable, re-ingestable degraded state (a chunk ID present in Postgres with no corresponding row in `chunk_embeddings`), not silent data corruption. This is the same class of partial-failure risk any single-store ingestion pipeline already has (e.g. a crash between the chunk insert and the edge-building query in the current pgContext-based design) — the two-store split doesn't introduce a fundamentally new failure mode, just a slightly wider window and a different detection query.

## Testing

- **`tests/graph/test_duckdb_store.py`** (new): insert/search/get round-trip on a temp DuckDB file; `EXPLAIN`-based assertion that `search()` actually uses the HNSW index once created (same technique validated during benchmarking — check for `"HNSW"` in the plan text, case-insensitive); assert index build is idempotent (`ensure_index()` called twice doesn't error).
- **`tests/retrieval/test_hybrid_retriever.py`** (new, replaces/extends the existing Postgres retriever test): integration test covering the two-step merge (DuckDB vector search + Postgres chunk fetch + pgGraph expansion + DuckDB neighbor rescoring) against a small fixture dataset in both stores.
- **Re-run `scripts/migrate_to_postgres.py --dry-run`** against real production data (117 papers, 6,179 chunks, native 3584-dim, no re-embedding) once implemented, to reconfirm the exact counts already validated for the pre-fork version of this script, now that the embedding path is forked into two stores.
