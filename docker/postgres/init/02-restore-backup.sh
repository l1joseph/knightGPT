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

echo "Postgres restore complete: qiita_studies/qiita_study_publications/paper_study_links restored, papers/chunks/chunk_edges truncated for re-ingestion."
