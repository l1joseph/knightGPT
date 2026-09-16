#!/bin/bash
#SBATCH --job-name=nrp-embed-manuscript-validation
#SBATCH --output=/cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy/logs/%x_%j.out
#SBATCH --error=/cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy/logs/%x_%j.err
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#
# Stands up Postgres+pgGraph via SingularityPRO (same proven recipe as
# slurm/singularity_stack_proof_test.slurm), restores the verified Sept 4
# backup, then -- instead of the proof test's short-lived API/webui check
# and full teardown -- runs a small, real validation of the now-fixed NRP
# embedding path (src/embedding/embedder.py's api_key bug, see PR #7)
# against real already-converted manuscripts already sitting on scratch,
# then takes a fresh pg_dump for later transfer to kl-remote.
#
# Deliberately uses scripts/validate_nrp_ingestion.py, NOT
# scripts/nrp_batch_ingest.py directly: nrp_batch_ingest.py's Phase 1
# (download_all_sources()) downloads/converts every DOI across all paper
# lists regardless of --max-papers, which would kick off ~800 real
# downloads for what is meant to be a quick, cheap validation run. The
# validation script skips straight to chunk -> embed -> insert against
# the 117 manuscripts already converted from a prior run.
#
# This job does NOT delete $PERSIST_ROOT at the end (unlike the proof
# test) -- the built Postgres+pgGraph binary is kept so a follow-up,
# larger ingestion run doesn't pay the ~10min build cost again.

set -uo pipefail

REPO=/cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy
cd "$REPO"

BUILD_ROOT="/cosmos/vast/scratch/l1joseph/knightgpt/singularity-proof-test/${SLURM_JOB_ID}"
mkdir -p "$BUILD_ROOT"/{logs,pgdata,singularity-tmp}

PERSIST_ROOT="/cosmos/vast/scratch/l1joseph/knightgpt/singularity-proof-test/persistent"
mkdir -p "$PERSIST_ROOT"/{pgwork,tools}

DUMP_OUT_DIR="/cosmos/vast/scratch/l1joseph/knightgpt/deploy"
mkdir -p "$DUMP_OUT_DIR"

VALIDATION_DUCKDB="/cosmos/vast/scratch/l1joseph/knightgpt/data/processed/embeddings_qwen3_4096.duckdb"

export SINGULARITY_TMPDIR="$BUILD_ROOT/singularity-tmp"
mkdir -p "$PERSIST_ROOT/singularity-cache"
export SINGULARITY_CACHEDIR="$PERSIST_ROOT/singularity-cache"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"

echo "=== $(date -Is) job ${SLURM_JOB_ID} on $(hostname) ==="
echo "BUILD_ROOT=$BUILD_ROOT"

module purge
module use /opt/modulefiles
module load singularitypro
singularity --version

fail() {
    echo "=== FAIL: $1 ==="
    echo "BUILD_ROOT retained for postmortem: $BUILD_ROOT"
    exit 1
}

trap 'echo "--- trap: stopping all singularity instances for this job ---"; singularity instance stop -a >/dev/null 2>&1 || true' EXIT

VLLM_API_KEY=$(grep -m1 '^VLLM_API_KEY=' "$REPO/.env" | cut -d= -f2-)
VLLM_EMBEDDING_DIM=$(grep -m1 '^VLLM_EMBEDDING_DIM=' "$REPO/.env" | cut -d= -f2-)
[ -n "$VLLM_API_KEY" ] || fail "VLLM_API_KEY not found in $REPO/.env"
[ -n "$VLLM_EMBEDDING_DIM" ] || fail "VLLM_EMBEDDING_DIM not found in $REPO/.env"
echo "VLLM_API_KEY: present, length=${#VLLM_API_KEY} (not printed)"
echo "VLLM_EMBEDDING_DIM=$VLLM_EMBEDDING_DIM"

VLLM_EMBEDDING_URL="https://ellm.nrp-nautilus.io/v1"
VLLM_INFERENCE_URL="https://ellm.nrp-nautilus.io/v1"
VLLM_EMBEDDING_MODEL="qwen3-embedding"
VLLM_INFERENCE_MODEL="qwen3"

POSTGRES_DSN="postgresql://postgres:password@localhost:5432/knightgpt"
DUMP_FILE="/cosmos/vast/scratch/l1joseph/knightgpt/deploy/knightgpt-postgres-20260904-024751.dump"
[ -f "$DUMP_FILE" ] || fail "backup dump not found: $DUMP_FILE"
ls -l "$DUMP_FILE"

# ============================================================
# 1. Build Postgres 17 (from source) + pgGraph (cargo-pgrx), fully
#    unprivileged, into $PERSIST_ROOT/pgwork. Identical to
#    slurm/singularity_stack_proof_test.slurm's proven build (idempotent:
#    reruns skip already-built steps).
# ============================================================
echo "=== [build] conda-forge tools (bison/flex/m4/clang/libclang/zlib) ==="
TOOLS_ENV="$PERSIST_ROOT/tools/pgbuild-tools"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
[ -x "$TOOLS_ENV/bin/bison" ] || mamba create -y -p "$TOOLS_ENV" -c conda-forge bison flex m4 clang libclang zlib 2>&1 | tail -n 30
"$TOOLS_ENV/bin/bison" --version | head -1

echo "=== [build] overlay + builder instance ==="
singularity overlay create --size 8192 --sparse "$BUILD_ROOT/build-overlay.img"
singularity instance start --no-home --overlay "$BUILD_ROOT/build-overlay.img" \
    --bind "$PERSIST_ROOT/pgwork:/work" \
    --bind "$TOOLS_ENV:$TOOLS_ENV" \
    docker://rust:1.96.0-bookworm builder
singularity instance list

BUILD_ENV=(--env HOME=/work --env PATH="$TOOLS_ENV/bin:/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
           --env LD_LIBRARY_PATH="$TOOLS_ENV/lib" --env LIBCLANG_PATH="$TOOLS_ENV/lib" --env TOOLS_ENV="$TOOLS_ENV")
runb() { singularity exec "${BUILD_ENV[@]}" instance://builder bash -c "$1"; }

echo "=== [build] postgres 17.6 from source (unprivileged local prefix /work/pg17) ==="
runb '
set -e
cd /work
PGVER=17.6
[ -d "postgresql-${PGVER}" ] || { curl -fSL -o postgresql.tar.bz2 "https://ftp.postgresql.org/pub/source/v${PGVER}/postgresql-${PGVER}.tar.bz2"; tar xjf postgresql.tar.bz2; }
cd postgresql-${PGVER}
[ -x /work/pg17/bin/pg_config ] || {
    ./configure --prefix=/work/pg17 --without-readline --without-icu --without-openssl \
        --with-includes="$TOOLS_ENV/include" --with-libraries="$TOOLS_ENV/lib" > /work/configure.log 2>&1
    make -j"$(nproc)" > /work/make.log 2>&1
    make install > /work/make-install.log 2>&1
}
/work/pg17/bin/pg_config --version
'
PG_BUILD_RC=$?
if [ "$PG_BUILD_RC" != "0" ]; then
    runb 'tail -n 60 /work/configure.log 2>/dev/null; echo ---MAKE---; tail -n 150 /work/make.log 2>/dev/null; echo ---INSTALL---; tail -n 60 /work/make-install.log 2>/dev/null'
    fail "postgres-from-source build failed"
fi

echo "=== [build] cargo-pgrx + pgGraph, merged into /work/pg17 ==="
runb '
set -e
cargo install cargo-pgrx --version 0.19.1 --locked > /work/cargo-pgrx-install.log 2>&1
cargo pgrx --version
[ -d /work/pggraph ] || git clone --depth 1 --branch v1.0.0 https://github.com/evokoa/pggraph.git /work/pggraph
cd /work/pggraph/graph
cargo pgrx init --pg17=/work/pg17/bin/pg_config > /work/pgrx-init.log 2>&1
cargo pgrx package --pg-config=/work/pg17/bin/pg_config > /work/pgrx-package.log 2>&1
cp -a target/release/graph-pg17/work/pg17/. /work/pg17/.
ls -la /work/pg17/lib/postgresql/graph.so
ls -la /work/pg17/share/postgresql/extension/graph*
'
PGGRAPH_RC=$?
if [ "$PGGRAPH_RC" != "0" ]; then
    runb 'echo ---CARGO-PGRX-INSTALL---; tail -n 100 /work/cargo-pgrx-install.log 2>/dev/null; echo ---INIT---; tail -n 80 /work/pgrx-init.log 2>/dev/null; echo ---PACKAGE---; tail -n 200 /work/pgrx-package.log 2>/dev/null'
    fail "pgGraph build/merge failed"
fi

echo "=== [build] stopping builder instance ==="
singularity instance stop builder

# ============================================================
# 2. Runtime: start Postgres, initialize, restore, verify (unchanged from
#    the proof test).
# ============================================================
echo "=== [pg] starting postgres runtime instance ==="
singularity instance start --no-home \
    --bind "$PERSIST_ROOT/pgwork:/work:ro" \
    --bind "$BUILD_ROOT/pgdata:/pgdata" \
    --bind "${DUMP_FILE}:/dump/backup.dump:ro" \
    --bind "$TOOLS_ENV:$TOOLS_ENV:ro" \
    --bind "$REPO/sql/schema.sql:/dump/schema.sql:ro" \
    docker://rust:1.96.0-bookworm pg
singularity instance list

PG_ENV=(--env HOME=/pgdata --env LD_LIBRARY_PATH="$TOOLS_ENV/lib")
runpg() { singularity exec "${PG_ENV[@]}" instance://pg bash -c "$1"; }

echo "=== [pg] initdb + start ==="
runpg '
set -e
PGBIN=/work/pg17/bin
"$PGBIN/initdb" -D /pgdata -U postgres -A trust --locale=C > /tmp/initdb.log 2>&1
"$PGBIN/pg_ctl" -D /pgdata -l /pgdata/server.log -o "-c shared_preload_libraries=graph -c listen_addresses=localhost -p 5432 -k /tmp" start -w -t 60
'
PG_START_RC=$?
if [ "$PG_START_RC" != "0" ]; then
    runpg 'tail -n 60 /tmp/initdb.log 2>/dev/null; echo ---SERVER---; tail -n 100 /pgdata/server.log 2>/dev/null'
    fail "postgres initdb/start failed"
fi

echo "=== [pg] waiting for readiness ==="
PG_READY=0
for i in $(seq 1 30); do
    if runpg '/work/pg17/bin/pg_isready -h localhost -p 5432 -U postgres' >/dev/null 2>&1; then
        echo "postgres ready after $((i*3))s"; PG_READY=1; break
    fi
    sleep 3
done
[ "$PG_READY" = "1" ] || { runpg 'tail -n 100 /pgdata/server.log'; fail "postgres never became ready"; }

echo "=== [pg] create db + extension + restore + truncate + re-register pgGraph ==="
cat > "$BUILD_ROOT/pgdata/verify-graph-expand.sql" <<'SQL'
BEGIN;
INSERT INTO papers (doi, title) VALUES ('schema-verify-probe-doi', 'schema verify probe');
INSERT INTO chunks (id, paper_doi, text) VALUES ('schema-verify-probe-chunk', 'schema-verify-probe-doi', 'probe');
SELECT * FROM graph.expand('public.chunks'::regclass, 'schema-verify-probe-chunk', 1);
ROLLBACK;
SQL

runpg '
set -e
PGBIN=/work/pg17/bin
"$PGBIN/createdb" -h localhost -p 5432 -U postgres knightgpt
"$PGBIN/psql" -h localhost -p 5432 -U postgres -d knightgpt -c "CREATE EXTENSION IF NOT EXISTS graph;"
"$PGBIN/pg_restore" -h localhost -p 5432 -U postgres -d knightgpt --no-owner --if-exists --clean /dump/backup.dump
"$PGBIN/psql" -h localhost -p 5432 -U postgres -d knightgpt -c "TRUNCATE papers, chunks, chunk_edges CASCADE;"
"$PGBIN/psql" -v ON_ERROR_STOP=1 -h localhost -p 5432 -U postgres -d knightgpt -f /dump/schema.sql
"$PGBIN/psql" -v ON_ERROR_STOP=1 -h localhost -p 5432 -U postgres -d knightgpt -c "SELECT graph.build();"
"$PGBIN/psql" -v ON_ERROR_STOP=1 -h localhost -p 5432 -U postgres -d knightgpt -f /pgdata/verify-graph-expand.sql
' 2>&1 | tee "$BUILD_ROOT/logs/pg-init-sequence.log"
RESTORE_RC=${PIPESTATUS[0]}
[ "$RESTORE_RC" = "0" ] || fail "postgres create/extension/restore/truncate sequence failed (see pg-init-sequence.log above)"

echo "=== [pg] verifying restore (real row counts) ==="
cat > "$BUILD_ROOT/pgdata/verify.sql" <<'SQL'
SELECT 'qiita_studies=' || count(*) FROM qiita_studies
UNION ALL SELECT 'qiita_study_publications=' || count(*) FROM qiita_study_publications
UNION ALL SELECT 'paper_study_links=' || count(*) FROM paper_study_links
UNION ALL SELECT 'papers=' || count(*) FROM papers
UNION ALL SELECT 'chunks=' || count(*) FROM chunks;
SQL
runpg '/work/pg17/bin/psql -h localhost -p 5432 -U postgres -d knightgpt -t -f /pgdata/verify.sql' 2>&1 | tee "$BUILD_ROOT/logs/verify-counts-before-ingest.log"
[ -s "$BUILD_ROOT/logs/verify-counts-before-ingest.log" ] || fail "row-count verification produced no output"

# ============================================================
# 3. Real validation: chunk -> embed (via NRP) -> insert on real,
#    already-converted manuscripts. Runs as a plain conda-env python
#    process on this same allocated node -- Postgres (above) shares the
#    host network namespace, so localhost:5432 reaches it directly, no
#    singularity instance needed for this step.
# ============================================================
echo "=== [validate] chunk -> embed -> insert on real manuscripts via NRP ==="
export VLLM_API_KEY VLLM_EMBEDDING_DIM VLLM_EMBEDDING_URL VLLM_INFERENCE_URL VLLM_EMBEDDING_MODEL VLLM_INFERENCE_MODEL POSTGRES_DSN
export INGEST_MARKDOWN_DIR=/cosmos/vast/scratch/l1joseph/knightgpt/data/markdown

"$HOME/miniforge3/envs/knightGPT/bin/python" scripts/validate_nrp_ingestion.py \
    --max-papers 10 \
    --duckdb-path "$VALIDATION_DUCKDB" \
    --log-level INFO 2>&1 | tee "$BUILD_ROOT/logs/validate-ingestion.log"
VALIDATE_RC=${PIPESTATUS[0]}
[ "$VALIDATE_RC" = "0" ] || fail "validation ingestion failed (see validate-ingestion.log above)"

echo "=== [pg] verifying restore (real row counts, after validation ingest) ==="
runpg '/work/pg17/bin/psql -h localhost -p 5432 -U postgres -d knightgpt -t -f /pgdata/verify.sql' 2>&1 | tee "$BUILD_ROOT/logs/verify-counts-after-ingest.log"

# ============================================================
# 4. Fresh pg_dump for later transfer to kl-remote (a real, up-to-date
#    backup, replacing the Sept 4 one this job restored from).
# ============================================================
DUMP_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
NEW_DUMP_FILE="$DUMP_OUT_DIR/knightgpt-postgres-${DUMP_TIMESTAMP}.dump"
echo "=== [dump] taking fresh pg_dump -> $NEW_DUMP_FILE ==="
runpg "/work/pg17/bin/pg_dump -h localhost -p 5432 -U postgres -d knightgpt -Fc -f /pgdata/fresh-backup.dump" \
    2>&1 | tee "$BUILD_ROOT/logs/pg-dump.log"
DUMP_RC=${PIPESTATUS[0]}
[ "$DUMP_RC" = "0" ] || fail "pg_dump failed (see pg-dump.log above)"
cp "$BUILD_ROOT/pgdata/fresh-backup.dump" "$NEW_DUMP_FILE"
ls -l "$NEW_DUMP_FILE"

echo "=== VALIDATION RUN COMPLETE ==="

# ============================================================
# 5. Teardown: stop Postgres, remove job-scoped scratch. $PERSIST_ROOT
#    (the built Postgres+pgGraph binary) is deliberately KEPT so a
#    follow-up, larger ingestion run doesn't pay the ~10min build cost
#    again.
# ============================================================
echo "=== Tearing down ==="
runpg '/work/pg17/bin/pg_ctl -D /pgdata stop -m fast' 2>&1 || true
singularity instance stop pg 2>&1 || true
echo "--- instance list after stop (should be empty) ---"
singularity instance list

echo "=== Removing job-scoped scratch: $BUILD_ROOT ==="
rm -rf "$BUILD_ROOT"
if [ -d "$BUILD_ROOT" ]; then
    echo "WARNING: $BUILD_ROOT still exists after rm -rf"
else
    echo "Confirmed removed: $BUILD_ROOT"
fi

echo "Keeping build cache for reuse: $PERSIST_ROOT"
echo "New backup dump: $NEW_DUMP_FILE"
echo "Validation DuckDB: $VALIDATION_DUCKDB"

echo "=== $(date -Is) job ${SLURM_JOB_ID} complete ==="
