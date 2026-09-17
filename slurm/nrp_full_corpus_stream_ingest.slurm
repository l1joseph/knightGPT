#!/bin/bash
#SBATCH --job-name=nrp-full-corpus-stream-ingest
#SBATCH --output=/cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy/logs/%x_%j.out
#SBATCH --error=/cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy/logs/%x_%j.err
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=12:00:00
#
# Full-corpus run of scripts/stream_corpus_ingest.py: streams every
# remaining DOI (across all paper lists, minus whatever's already in
# Postgres) through download -> chunk -> embed (NRP) -> insert -> delete,
# so raw PDFs/markdown never accumulate for all ~900 papers at once.
#
# Builds on job 101511 (slurm/nrp_embedding_validation.slurm), which
# validated the pipeline on 10 real manuscripts and left behind:
#   - a built Postgres+pgGraph binary cache ($PERSIST_ROOT, reused below)
#   - a fresh pg_dump already containing those 10 papers + the Qiita
#     tables, which this job restores from (NOT the original Sept 4 dump)
#     so those 10 papers aren't reprocessed and nothing already-validated
#     is lost.
#
# This is a long, network-bound job (real HTTP calls to Unpaywall/PMC/
# journal sites for ~900 DOIs, with a polite 1.5s delay between attempts)
# with real, if hard to predict precisely, wall-clock risk. To keep a
# multi-hour run from losing everything if it dies partway:
#   - A background loop takes a fresh pg_dump every 30 minutes to scratch.
#   - stream_corpus_ingest.py itself writes a running stats JSON every 10
#     papers and is resumable from Postgres state (not from files on
#     disk), so a rerun of this same script naturally continues instead
#     of reprocessing already-ingested DOIs.
# If this job is killed or times out, resubmit this same script -- it
# will restore from whichever pg_dump is newest and pick up where it
# left off.

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

PERIODIC_DUMP_PID=""
cleanup() {
    echo "--- trap: stopping periodic dump loop and all singularity instances ---"
    [ -n "$PERIODIC_DUMP_PID" ] && kill "$PERIODIC_DUMP_PID" 2>/dev/null || true
    singularity instance stop -a >/dev/null 2>&1 || true
}
trap cleanup EXIT

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

# Restore from the NEWEST available dump, not the original Sept 4 one --
# picks up job 101511's fresh dump (10 papers + Qiita tables already in
# it) on first run, and this job's own periodic dumps on a resubmit.
DUMP_FILE=$(ls -t "$DUMP_OUT_DIR"/knightgpt-postgres-*.dump 2>/dev/null | head -1)
[ -n "$DUMP_FILE" ] && [ -f "$DUMP_FILE" ] || fail "no backup dump found in $DUMP_OUT_DIR"
echo "Restoring from: $DUMP_FILE"
ls -l "$DUMP_FILE"

# ============================================================
# 1. Build Postgres 17 + pgGraph (idempotent -- reuses $PERSIST_ROOT/pgwork
#    from job 101511, should be near-instant here).
# ============================================================
echo "=== [build] conda-forge tools ==="
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

echo "=== [build] postgres 17.6 from source ==="
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

echo "=== [build] cargo-pgrx + pgGraph ==="
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
'
PGGRAPH_RC=$?
if [ "$PGGRAPH_RC" != "0" ]; then
    runb 'echo ---CARGO-PGRX-INSTALL---; tail -n 100 /work/cargo-pgrx-install.log 2>/dev/null; echo ---INIT---; tail -n 80 /work/pgrx-init.log 2>/dev/null; echo ---PACKAGE---; tail -n 200 /work/pgrx-package.log 2>/dev/null'
    fail "pgGraph build/merge failed"
fi

echo "=== [build] stopping builder instance ==="
singularity instance stop builder

# ============================================================
# 2. Runtime: start Postgres, restore from the newest dump, verify.
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

echo "=== [pg] create db + extension + restore + re-register pgGraph ==="
# NOTE: no TRUNCATE here, unlike the Sept-4-dump restore scripts -- this
# dump is already at the current embedding generation (built by job
# 101511 and this job's own prior runs), so its papers/chunks are meant
# to be kept, not wiped. Restoring still requires re-registering pgGraph
# (pg_restore assigns fresh OIDs every time).
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
"$PGBIN/psql" -v ON_ERROR_STOP=1 -h localhost -p 5432 -U postgres -d knightgpt -f /dump/schema.sql
"$PGBIN/psql" -v ON_ERROR_STOP=1 -h localhost -p 5432 -U postgres -d knightgpt -c "SELECT graph.build();"
"$PGBIN/psql" -v ON_ERROR_STOP=1 -h localhost -p 5432 -U postgres -d knightgpt -f /pgdata/verify-graph-expand.sql
' 2>&1 | tee "$BUILD_ROOT/logs/pg-init-sequence.log"
RESTORE_RC=${PIPESTATUS[0]}
[ "$RESTORE_RC" = "0" ] || fail "postgres create/extension/restore sequence failed (see pg-init-sequence.log above)"

cat > "$BUILD_ROOT/pgdata/verify.sql" <<'SQL'
SELECT 'qiita_studies=' || count(*) FROM qiita_studies
UNION ALL SELECT 'qiita_study_publications=' || count(*) FROM qiita_study_publications
UNION ALL SELECT 'paper_study_links=' || count(*) FROM paper_study_links
UNION ALL SELECT 'papers=' || count(*) FROM papers
UNION ALL SELECT 'chunks=' || count(*) FROM chunks;
SQL
echo "=== [pg] row counts before this run's ingestion ==="
runpg '/work/pg17/bin/psql -h localhost -p 5432 -U postgres -d knightgpt -t -f /pgdata/verify.sql' 2>&1 | tee "$BUILD_ROOT/logs/verify-counts-before.log"

# ============================================================
# 3. Background periodic pg_dump every 30 minutes, so a killed/timed-out
#    job doesn't lose everything ingested so far.
# ============================================================
(
    while true; do
        sleep 1800
        ts=$(date -u +%Y%m%d-%H%M%S)
        echo "=== [periodic dump] $(date -Is) ==="
        singularity exec "${PG_ENV[@]}" instance://pg bash -c \
            "/work/pg17/bin/pg_dump -h localhost -p 5432 -U postgres -d knightgpt -Fc -f /pgdata/periodic-backup.dump" \
            && cp "$BUILD_ROOT/pgdata/periodic-backup.dump" "$DUMP_OUT_DIR/knightgpt-postgres-${ts}.dump" \
            && echo "=== [periodic dump] saved knightgpt-postgres-${ts}.dump ==="
    done
) &
PERIODIC_DUMP_PID=$!
echo "periodic dump loop backgrounded as PID $PERIODIC_DUMP_PID"

# ============================================================
# 4. The actual full-corpus streaming ingestion.
# ============================================================
echo "=== [ingest] streaming full corpus via NRP ==="
export VLLM_API_KEY VLLM_EMBEDDING_DIM VLLM_EMBEDDING_URL VLLM_INFERENCE_URL VLLM_EMBEDDING_MODEL VLLM_INFERENCE_MODEL POSTGRES_DSN
export INGEST_MARKDOWN_DIR=/cosmos/vast/scratch/l1joseph/knightgpt/data/markdown
export INGEST_RAW_PDF_DIR=/cosmos/vast/scratch/l1joseph/knightgpt/data/raw_pdfs
export INGEST_PROCESSED_DIR=/cosmos/vast/scratch/l1joseph/knightgpt/data/processed

"$HOME/miniforge3/envs/knightGPT/bin/python" scripts/stream_corpus_ingest.py \
    --duckdb-path "$VALIDATION_DUCKDB" \
    --delay 1.5 \
    --stats-every 10 \
    --log-level INFO 2>&1 | tee "$BUILD_ROOT/logs/stream-ingest.log"
INGEST_RC=${PIPESTATUS[0]}

echo "=== [pg] row counts after this run's ingestion ==="
runpg '/work/pg17/bin/psql -h localhost -p 5432 -U postgres -d knightgpt -t -f /pgdata/verify.sql' 2>&1 | tee "$BUILD_ROOT/logs/verify-counts-after.log"

# ============================================================
# 5. Final pg_dump regardless of ingest exit code (partial progress is
#    still worth keeping for a resubmit).
# ============================================================
kill "$PERIODIC_DUMP_PID" 2>/dev/null || true
PERIODIC_DUMP_PID=""

FINAL_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
FINAL_DUMP_FILE="$DUMP_OUT_DIR/knightgpt-postgres-${FINAL_TIMESTAMP}.dump"
echo "=== [dump] final pg_dump -> $FINAL_DUMP_FILE ==="
runpg "/work/pg17/bin/pg_dump -h localhost -p 5432 -U postgres -d knightgpt -Fc -f /pgdata/final-backup.dump" \
    2>&1 | tee "$BUILD_ROOT/logs/pg-dump-final.log"
DUMP_RC=${PIPESTATUS[0]}
if [ "$DUMP_RC" = "0" ]; then
    cp "$BUILD_ROOT/pgdata/final-backup.dump" "$FINAL_DUMP_FILE"
    ls -l "$FINAL_DUMP_FILE"
else
    echo "WARNING: final pg_dump failed -- most recent periodic dump in $DUMP_OUT_DIR is the latest good backup"
fi

if [ "$INGEST_RC" != "0" ]; then
    echo "=== ingestion exited non-zero ($INGEST_RC) -- resubmit this same script to resume ==="
else
    echo "=== FULL CORPUS INGESTION COMPLETE ==="
fi

# ============================================================
# 6. Teardown: stop Postgres, remove job-scoped scratch. $PERSIST_ROOT
#    (build cache) is kept.
# ============================================================
echo "=== Tearing down ==="
runpg '/work/pg17/bin/pg_ctl -D /pgdata stop -m fast' 2>&1 || true
singularity instance stop pg 2>&1 || true
singularity instance list

echo "=== Removing job-scoped scratch: $BUILD_ROOT ==="
rm -rf "$BUILD_ROOT"
[ -d "$BUILD_ROOT" ] && echo "WARNING: $BUILD_ROOT still exists" || echo "Confirmed removed: $BUILD_ROOT"

echo "=== $(date -Is) job ${SLURM_JOB_ID} complete ==="
exit "$INGEST_RC"
