-- sql/schema.sql
CREATE EXTENSION IF NOT EXISTS graph;

CREATE TABLE IF NOT EXISTS papers (
    doi        text PRIMARY KEY,
    title      text,
    metadata   jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS chunks (
    id           text PRIMARY KEY,
    paper_doi    text REFERENCES papers(doi),
    text         text NOT NULL,
    section      text,
    token_count  integer
);

CREATE TABLE IF NOT EXISTS chunk_edges (
    src_chunk_id text NOT NULL REFERENCES chunks(id),
    dst_chunk_id text NOT NULL REFERENCES chunks(id),
    similarity   real NOT NULL,
    PRIMARY KEY (src_chunk_id, dst_chunk_id)
);

-- Per-user/per-project collections migration -- see
-- docs/superpowers/specs/2026-10-01-per-user-collections-design.md.
-- NOT NULL with a constant default is a fast, metadata-only change on
-- Postgres 17 (no table rewrite) and makes every existing row an
-- explicit, queryable member of the 'global' collection rather than an
-- implicit NULL -- NULL = anything is never true in SQL, which would
-- make pgGraph's tenant_column scoping unable to ever match existing
-- rows if collection_id were nullable instead.
ALTER TABLE papers ADD COLUMN IF NOT EXISTS collection_id text NOT NULL DEFAULT 'global';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS collection_id text NOT NULL DEFAULT 'global';
ALTER TABLE chunk_edges ADD COLUMN IF NOT EXISTS collection_id text NOT NULL DEFAULT 'global';

-- Collections registry -- model-id-per-collection follow-up to the
-- per-user/per-project collections migration above. Purely a
-- discoverability aid for GET /api/v1/collections and /v1/models (which
-- lists one knightgpt-rag-<id> model entry per row here, see
-- src/api/main.py's list_models()) -- it is NOT used for enforcement.
-- collection_id on papers/chunks/chunk_edges stays free-form text: any
-- string works there whether or not a matching row exists here, and
-- 'global' is implicit and never needs a row of its own.
CREATE TABLE IF NOT EXISTS collections (
    id           text PRIMARY KEY,
    display_name text,
    owner_email  text,
    created_at   timestamptz NOT NULL DEFAULT now()
);

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

COMMENT ON TABLE qiita_studies IS
    'Stage 1 registry of Qiita studies, seeded from redbiom''s public index by '
    'scripts/qiita_registry_ingest.py. title/abstract/principal_investigator/'
    'funding/metadata are left NULL by Stage 1 -- a separate, future Stage 2 '
    'backfills them via direct Postgres access to Qiita''s own database.';

COMMENT ON COLUMN qiita_studies.sample_count IS
    'Sum of per-context sample counts, NOT a distinct/deduplicated sample '
    'count. A study reprocessed under multiple redbiom contexts (different '
    'pipeline variants run over the same physical samples) has its samples '
    'counted once per context, inflating this total -- e.g. study 12949 has '
    'sample_count = 103662 across 6 contexts, versus 17277 for any single '
    'context (6x). See merge_context_results() in '
    'scripts/qiita_registry_ingest.py.';

COMMENT ON COLUMN qiita_studies.study_id IS
    'Parsed from the leading numeric prefix of each redbiom sample ID '
    '(format <study_id>.<sample_name>), NOT queried from redbiom''s official '
    'qiita_study_id metadata field -- a deliberate Stage 1 simplification for '
    'performance, with a small chance of mismatch for "ambiguous" sample IDs. '
    'See the "KNOWN SIMPLIFICATION" block in '
    'scripts/qiita_registry_ingest.py for full detail.';

-- pgGraph's tenant-scoping mechanism is an indirection: graph.tenant_setting
-- holds the NAME of another GUC to read the actual tenant value from (read
-- by graph.enforce_tenant_scope at query time), it does not carry the
-- tenant value itself. This is a one-time, database-level configuration --
-- ALTER DATABASE ... SET takes effect for new sessions, so it must run
-- before anything opens a connection expecting tenant scoping to work, and
-- it does not need to be (and should not be) re-set per request. Confirmed
-- live on kl-remote: an earlier version of this codebase set
-- graph.tenant_setting ITSELF to the per-request collection_id value
-- (mistaking it for the value-holding GUC), which made every
-- graph.expand() call fail with "tenant scope is required for registered
-- tables with tenant_column" -- pgGraph was correctly looking up
-- current_setting(graph.tenant_setting's own value) and finding no real
-- GUC by that name. The actual per-request value goes into
-- knightgpt.collection_id (see HybridRetriever._retrieve_async), the GUC
-- this line points graph.tenant_setting at.
ALTER DATABASE knightgpt SET graph.tenant_setting = 'knightgpt.collection_id';

-- pgGraph registration: chunks as nodes, chunk_edges as an edge-table relationship.
-- Idempotency is not assumed from pgGraph itself; each registration is wrapped
-- in its own DO block so re-running this file against an already-provisioned
-- database is always safe.
--
-- We do NOT know pgGraph's exact duplicate-registration SQLSTATE (no live
-- Postgres has been available to determine it during this migration), so we
-- deliberately do not narrow the WHEN OTHERS catch here. Instead we surface
-- every caught exception via RAISE NOTICE so a genuine failure (bad argument
-- name, type mismatch, etc.) is visible in the Postgres logs rather than
-- silently swallowed. scripts/apply_schema.py additionally verifies
-- registration succeeded by querying graph.registered_tables() /
-- graph.registered_edges() after applying this file and raises a Python
-- exception if either registration is missing, so a swallowed failure here
-- is caught loudly at the one point in the runbook where it's still cheap
-- to catch.
DO $$
BEGIN
  PERFORM graph.add_table(
      table_name := 'public.chunks'::regclass,
      id_column := 'id',
      columns := ARRAY['text', 'section'],
      tenant_column := 'collection_id'
  );
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'graph.add_table(public.chunks) raised % (%) -- ignored, assumed already registered; verify with SELECT * FROM graph.registered_tables()', SQLERRM, SQLSTATE;
END $$;

DO $$
BEGIN
  PERFORM graph.add_edge(
      from_table := 'public.chunk_edges'::regclass,
      from_column := 'src_chunk_id',
      to_table := 'public.chunks'::regclass,
      to_column := 'dst_chunk_id',
      label := 'similar_to',
      bidirectional := true,
      weight_column := 'similarity'
  );
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'graph.add_edge(public.chunk_edges) raised % (%) -- ignored, assumed already registered; verify with SELECT * FROM graph.registered_edges()', SQLERRM, SQLSTATE;
END $$;
