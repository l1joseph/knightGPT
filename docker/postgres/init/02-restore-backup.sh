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

psql -U postgres -d knightgpt -c "TRUNCATE papers, chunks, chunk_edges CASCADE;"

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
psql -U postgres -d knightgpt -f /opt/knightgpt/schema.sql
psql -U postgres -d knightgpt -c "SELECT graph.build();"

echo "Postgres restore complete: qiita_studies/qiita_study_publications/paper_study_links restored, papers/chunks/chunk_edges truncated for re-ingestion, pgGraph registration re-pointed at post-restore OIDs."
