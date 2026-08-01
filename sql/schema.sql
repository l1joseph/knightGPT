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
-- These calls are idempotent registration metadata writes; safe to re-run.
SELECT graph.add_table(
    table_name := 'public.chunks'::regclass,
    id_column := 'id',
    columns := ARRAY['text', 'section']
);

SELECT graph.add_edge(
    from_table := 'public.chunk_edges'::regclass,
    from_column := 'src_chunk_id',
    to_table := 'public.chunks'::regclass,
    to_column := 'dst_chunk_id',
    label := 'similar_to',
    bidirectional := true,
    weight_column := 'similarity'
);
