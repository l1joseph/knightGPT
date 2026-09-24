#!/bin/bash
# Restores papers/chunks/chunk_edges/qiita_studies/qiita_study_publications/
# paper_study_links from the mounted backup.dump. Runs once, only against a
# fresh (empty) data directory, after 01-create-extensions.sql, per the
# standard Postgres image /docker-entrypoint-initdb.d/ convention.
#
# IMPORTANT: this does NOT truncate papers/chunks/chunk_edges after
# restoring. Earlier versions of this script did, because the original
# 2026-09-04 backup's papers/chunks were embedded with the old self-hosted
# gte-Qwen2-7B model (3584-dim) and incompatible with the qwen3-embedding
# switch (4096-dim) -- that was a one-time migration step for that specific
# dump, not a general property of "restoring a backup". Every dump produced
# since (via scripts/stream_corpus_ingest.py and friends) is already
# embedded at the current dimension, so truncating here would silently
# destroy real, correctly-embedded corpus data on every fresh deploy.
# If a future embedding-model change needs the same one-time treatment,
# do it as a deliberate, explicit migration step against the live database
# -- not by re-adding an unconditional truncate here.
set -euo pipefail

pg_restore -U postgres -d knightgpt --no-owner --if-exists --clean \
  /docker-entrypoint-initdb.d/backup.dump

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
# A throwaway probe row is used rather than an existing chunk because
# pgGraph does NOT treat a nonexistent starting id as "zero rows" the way
# RESTORE.md's wording first suggested -- confirmed directly: it raises a
# hard error ("Node not found: <oid>.schema-verify-probe", diagnostic
# PG010) for any id that isn't an actual row, which is a completely
# different (and here, false-positive) failure mode from the OID-staleness
# bug this check exists to catch. A dedicated probe row sidesteps needing
# to know a real chunk id up front (and still works if chunks happens to
# be empty). Wrapped in BEGIN/ROLLBACK rather than explicit DELETEs so the
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

echo "Postgres restore complete: papers/chunks/chunk_edges/qiita_studies/qiita_study_publications/paper_study_links restored as-is from backup.dump, pgGraph registration re-pointed at post-restore OIDs and verified via a real graph.expand() call."
