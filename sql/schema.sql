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
      columns := ARRAY['text', 'section']
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
