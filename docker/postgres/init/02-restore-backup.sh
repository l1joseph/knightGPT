#!/bin/bash
# Restores qiita_studies/qiita_study_publications/paper_study_links from
# the 2026-09-04 verified backup, then truncates papers/chunks/chunk_edges
# -- those were embedded with gte-Qwen2-7B and are incompatible with the
# qwen3-embedding switch (see docs/superpowers/specs/2026-09-14-knightgpt-webui-deploy-design.md,
# "Postgres" decision). Runs once, only against a fresh (empty) data
# directory, after 01-create-extensions.sql, per the standard Postgres
# image /docker-entrypoint-initdb.d/ convention.
set -euo pipefail

pg_restore -U postgres -d knightgpt --no-owner --if-exists --clean \
  /docker-entrypoint-initdb.d/backup.dump

# CASCADE is required here because paper_study_links/qiita_study_publications
# (defined only in the restored dump, not in sql/schema.sql) reference
# papers/chunks by foreign key -- a plain TRUNCATE of just
# papers/chunks/chunk_edges is refused by Postgres unless every table with
# such a reference is truncated in the same statement. CASCADE grants that
# by truncating the referencing tables too, which would silently wipe the
# very Qiita data this restore exists to preserve if it ever reached tables
# outside papers/chunks/chunk_edges themselves. Snapshot counts before and
# verify after, rather than trusting CASCADE only touched what we intended.
QIITA_STUDIES_BEFORE=$(psql -U postgres -d knightgpt -tAc "SELECT count(*) FROM qiita_studies")
QIITA_PUBS_BEFORE=$(psql -U postgres -d knightgpt -tAc "SELECT count(*) FROM qiita_study_publications")
PAPER_LINKS_BEFORE=$(psql -U postgres -d knightgpt -tAc "SELECT count(*) FROM paper_study_links")

psql -U postgres -d knightgpt -c "TRUNCATE papers, chunks, chunk_edges CASCADE;"

QIITA_STUDIES_AFTER=$(psql -U postgres -d knightgpt -tAc "SELECT count(*) FROM qiita_studies")
QIITA_PUBS_AFTER=$(psql -U postgres -d knightgpt -tAc "SELECT count(*) FROM qiita_study_publications")
PAPER_LINKS_AFTER=$(psql -U postgres -d knightgpt -tAc "SELECT count(*) FROM paper_study_links")

if [ "$QIITA_STUDIES_BEFORE" != "$QIITA_STUDIES_AFTER" ] \
  || [ "$QIITA_PUBS_BEFORE" != "$QIITA_PUBS_AFTER" ] \
  || [ "$PAPER_LINKS_BEFORE" != "$PAPER_LINKS_AFTER" ]; then
  echo "FATAL: TRUNCATE ... CASCADE reached the Qiita tables it must not touch." >&2
  echo "  qiita_studies: $QIITA_STUDIES_BEFORE -> $QIITA_STUDIES_AFTER" >&2
  echo "  qiita_study_publications: $QIITA_PUBS_BEFORE -> $QIITA_PUBS_AFTER" >&2
  echo "  paper_study_links: $PAPER_LINKS_BEFORE -> $PAPER_LINKS_AFTER" >&2
  exit 1
fi

# pg_restore assigns brand-new OIDs to the recreated chunks/chunk_edges
# tables, but the restored graph._registered_tables/_registered_edges rows
# still reference the OLD source-database OIDs -- pgGraph resolves
# registration by OID, not by table name, so a naive restore leaves
# graph.expand() (and anything that calls it, including this API's own
# /health check's `SELECT node_count FROM graph.status()`) broken with
# "registered table relation no longer exists" until this is fixed up.
# Confirmed and documented end-to-end in k8s/nrp/RESTORE.md ("The risk:
# stale OIDs in pgGraph's registration tables"). graph.add_table()/
# graph.add_edge() are upsert-by-name (not insert-only), so re-running
# sql/schema.sql's registration blocks re-points the stored OIDs at the
# tables' actual current OIDs; graph.build() then rebuilds the graph
# projection against the corrected registration (a projection built
# against the old OIDs would be stale too).
# -v ON_ERROR_STOP=1 is required here: without it, an error inside this
# script (e.g. a broken registration block) does not make psql exit
# non-zero, so `set -euo pipefail` above would silently let a real
# failure through.
psql -v ON_ERROR_STOP=1 -U postgres -d knightgpt -f /opt/knightgpt/schema.sql
psql -v ON_ERROR_STOP=1 -U postgres -d knightgpt -c "SELECT graph.build();"

# Real verification, not just "the script didn't error": per
# k8s/nrp/RESTORE.md step 4, graph.registered_tables()/
# graph.registered_edges() are name-based views that look completely
# normal even when the underlying OIDs are stale -- only an actual
# graph.expand() call surfaces a broken registration. sql/schema.sql's
# own DO $$ ... EXCEPTION WHEN OTHERS THEN RAISE NOTICE ... $$; blocks
# deliberately swallow every registration error (the real verification
# was always meant to live in scripts/apply_schema.py, which this
# init-script restore path doesn't use), so without an explicit check
# here a genuine registration failure would be completely silent.
#
# chunks is intentionally empty at this point (truncated above), and
# pgGraph does NOT treat a nonexistent starting id as "zero rows" the way
# RESTORE.md's wording first suggested -- confirmed directly: it raises a
# hard error ("Node not found: <oid>.schema-verify-probe", diagnostic
# PG010) for any id that isn't an actual row, which is a completely
# different (and here, false-positive) failure mode from the OID-staleness
# bug this check exists to catch. RESTORE.md's own tested procedure hit
# this identical empty-table situation and solved it the same way this
# does: insert a throwaway probe row to have something real to resolve
# against. Wrapped in BEGIN/ROLLBACK rather than explicit DELETEs so the
# probe row is never actually persisted either way: with
# -v ON_ERROR_STOP=1, a failing graph.expand() stops the script before
# the ROLLBACK line runs, but the transaction was never committed, so
# Postgres rolls it back automatically when this psql session's
# connection closes -- cleanup happens on every path, not just the
# happy one.
psql -v ON_ERROR_STOP=1 -U postgres -d knightgpt -c "
  BEGIN;
  INSERT INTO papers (doi, title) VALUES ('schema-verify-probe-doi', 'schema verify probe');
  INSERT INTO chunks (id, paper_doi, text) VALUES ('schema-verify-probe-chunk', 'schema-verify-probe-doi', 'probe');
  SELECT * FROM graph.expand('public.chunks'::regclass, 'schema-verify-probe-chunk', 1);
  ROLLBACK;
"

echo "Postgres restore complete: qiita_studies/qiita_study_publications/paper_study_links restored, papers/chunks/chunk_edges truncated for re-ingestion, pgGraph registration re-pointed at post-restore OIDs and verified via a real graph.expand() call."
