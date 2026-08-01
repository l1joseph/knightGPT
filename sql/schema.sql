-- sql/schema.sql
CREATE EXTENSION IF NOT EXISTS pgcontext;
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
    embedding    pgcontext.vector(3584) NOT NULL,
    section      text,
    token_count  integer
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
ON chunks
USING pgcontext_hnsw (embedding pgcontext.vector_hnsw_cosine_ops);

CREATE TABLE IF NOT EXISTS chunk_edges (
    src_chunk_id text NOT NULL REFERENCES chunks(id),
    dst_chunk_id text NOT NULL REFERENCES chunks(id),
    similarity   real NOT NULL,
    PRIMARY KEY (src_chunk_id, dst_chunk_id)
);

-- pgGraph registration: chunks as nodes, chunk_edges as an edge-table relationship.
-- Idempotency is not assumed from pgGraph itself; each registration is wrapped
-- in its own DO block that swallows re-registration errors, so re-running this
-- file against an already-provisioned database is always safe.
DO $$
BEGIN
  PERFORM graph.add_table(
      table_name := 'public.chunks'::regclass,
      id_column := 'id',
      columns := ARRAY['text', 'section']
  );
EXCEPTION WHEN OTHERS THEN
  NULL; -- already registered; safe to ignore on re-run
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
  NULL;
END $$;
