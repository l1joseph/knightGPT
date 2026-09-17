#!/bin/bash
#SBATCH --job-name=resume-backfill-chunk-edges
#SBATCH --output=/cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy/logs/%x_%j.out
#SBATCH --error=/cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy/logs/%x_%j.err
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#
# job 101691 (slurm/backfill_chunk_edges.slurm) hit its 1-hour time limit
# partway through (11300/16894 candidates done, ~85204 edges backfilled)
# and got killed before it could take a final dump. Its pgdata directory
# is still intact on scratch though (Postgres wasn't mid-write-corrupted,
# just SIGTERM'd -- normal WAL crash recovery on next startup), so this
# job REUSES that exact directory directly instead of restoring from the
# older dump and redoing ~85k edges worth of work.
#
# Longer time limit this time: observed rate was ~200 candidates/min,
# so the full 16894 needs ~85min, plus ~15-20min build+restore -- 1 hour
# wasn't enough. 2.5 hours gives real margin.

set -uo pipefail

REPO=/cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy
cd "$REPO"

PREV_JOB_PGDATA="/cosmos/vast/scratch/l1joseph/knightgpt/singularity-proof-test/101691/pgdata"
[ -d "$PREV_JOB_PGDATA" ] || { echo "FAIL: $PREV_JOB_PGDATA not found -- nothing to resume"; exit 1; }

BUILD_ROOT="/cosmos/vast/scratch/l1joseph/knightgpt/singularity-proof-test/${SLURM_JOB_ID}"
mkdir -p "$BUILD_ROOT"/{logs,singularity-tmp}

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
echo "Reusing pgdata from job 101691: $PREV_JOB_PGDATA"

module purge
module use /opt/modulefiles
module load singularitypro
singularity --version

fail() {
    echo "=== FAIL: $1 ==="
    echo "BUILD_ROOT retained for postmortem: $BUILD_ROOT"
    echo "PREV_JOB_PGDATA retained (NOT deleted on failure): $PREV_JOB_PGDATA"
    exit 1
}

trap 'echo "--- trap: stopping all singularity instances for this job ---"; singularity instance stop -a >/dev/null 2>&1 || true' EXIT

VLLM_API_KEY=$(grep -m1 '^VLLM_API_KEY=' "$REPO/.env" | cut -d= -f2-)
VLLM_EMBEDDING_DIM=$(grep -m1 '^VLLM_EMBEDDING_DIM=' "$REPO/.env" | cut -d= -f2-)
[ -n "$VLLM_API_KEY" ] || fail "VLLM_API_KEY not found in $REPO/.env"
POSTGRES_DSN="postgresql://postgres:password@localhost:5432/knightgpt"

# ============================================================
# 1. Build Postgres 17 + pgGraph (idempotent, cached).
# ============================================================
TOOLS_ENV="$PERSIST_ROOT/tools/pgbuild-tools"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
[ -x "$TOOLS_ENV/bin/bison" ] || mamba create -y -p "$TOOLS_ENV" -c conda-forge bison flex m4 clang libclang zlib 2>&1 | tail -n 30

singularity overlay create --size 8192 --sparse "$BUILD_ROOT/build-overlay.img"
singularity instance start --no-home --overlay "$BUILD_ROOT/build-overlay.img" \
    --bind "$PERSIST_ROOT/pgwork:/work" \
    --bind "$TOOLS_ENV:$TOOLS_ENV" \
    docker://rust:1.96.0-bookworm builder

BUILD_ENV=(--env HOME=/work --env PATH="$TOOLS_ENV/bin:/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
           --env LD_LIBRARY_PATH="$TOOLS_ENV/lib" --env LIBCLANG_PATH="$TOOLS_ENV/lib" --env TOOLS_ENV="$TOOLS_ENV")
runb() { singularity exec "${BUILD_ENV[@]}" instance://builder bash -c "$1"; }

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
' || fail "postgres-from-source build failed"

runb '
set -e
cargo install cargo-pgrx --version 0.19.1 --locked > /work/cargo-pgrx-install.log 2>&1
[ -d /work/pggraph ] || git clone --depth 1 --branch v1.0.0 https://github.com/evokoa/pggraph.git /work/pggraph
cd /work/pggraph/graph
cargo pgrx init --pg17=/work/pg17/bin/pg_config > /work/pgrx-init.log 2>&1
cargo pgrx package --pg-config=/work/pg17/bin/pg_config > /work/pgrx-package.log 2>&1
cp -a target/release/graph-pg17/work/pg17/. /work/pg17/.
' || fail "pgGraph build/merge failed"

singularity instance stop builder

# ============================================================
# 2. Start Postgres directly against the REUSED pgdata (no restore --
#    the data is already there; Postgres runs normal WAL crash recovery
#    on startup since it was SIGTERM'd, not corrupted).
# ============================================================
singularity instance start --no-home \
    --bind "$PERSIST_ROOT/pgwork:/work:ro" \
    --bind "$PREV_JOB_PGDATA:/pgdata" \
    --bind "$TOOLS_ENV:$TOOLS_ENV:ro" \
    docker://rust:1.96.0-bookworm pg
singularity instance list

PG_ENV=(--env HOME=/pgdata --env LD_LIBRARY_PATH="$TOOLS_ENV/lib")
runpg() { singularity exec "${PG_ENV[@]}" instance://pg bash -c "$1"; }

runpg '
set -e
"/work/pg17/bin/pg_ctl" -D /pgdata -l /pgdata/server-resume.log -o "-c shared_preload_libraries=graph -c listen_addresses=localhost -p 5432 -k /tmp" start -w -t 60
' || { runpg 'tail -n 150 /pgdata/server-resume.log'; fail "postgres failed to start against the reused pgdata (crash recovery issue?)"; }

PG_READY=0
for i in $(seq 1 30); do
    if runpg '/work/pg17/bin/pg_isready -h localhost -p 5432 -U postgres' >/dev/null 2>&1; then
        PG_READY=1; break
    fi
    sleep 3
done
[ "$PG_READY" = "1" ] || { runpg 'tail -n 150 /pgdata/server-resume.log'; fail "postgres never became ready"; }
echo "=== Postgres started successfully against the recovered pgdata ==="

cat > "$BUILD_ROOT/verify.sql" <<'SQL'
SELECT 'papers=' || count(*) FROM papers
UNION ALL SELECT 'chunks=' || count(*) FROM chunks
UNION ALL SELECT 'chunk_edges=' || count(*) FROM chunk_edges
UNION ALL SELECT 'edgeless_chunks=' || count(*) FROM chunks c
    WHERE NOT EXISTS (SELECT 1 FROM chunk_edges e WHERE e.src_chunk_id = c.id);
SQL
singularity exec "${PG_ENV[@]}" --bind "$BUILD_ROOT/verify.sql:/verify.sql:ro" instance://pg bash -c \
    '/work/pg17/bin/psql -h localhost -p 5432 -U postgres -d knightgpt -t -f /verify.sql' \
    2>&1 | tee "$BUILD_ROOT/logs/verify-recovered.log"

# Immediate safety-net dump BEFORE continuing, so the ~85k already-recovered
# edges are durably saved even if the rest of this job also fails somehow.
RECOVERY_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
RECOVERY_DUMP_FILE="$DUMP_OUT_DIR/knightgpt-postgres-${RECOVERY_TIMESTAMP}.dump"
runpg "/work/pg17/bin/pg_dump -h localhost -p 5432 -U postgres -d knightgpt -Fc -f /pgdata/recovery-backup.dump" \
    2>&1 | tee "$BUILD_ROOT/logs/pg-dump-recovery.log"
RECOVERY_DUMP_RC=${PIPESTATUS[0]}
[ "$RECOVERY_DUMP_RC" = "0" ] || fail "recovery dump failed -- refusing to continue without a safety net"
cp "$PREV_JOB_PGDATA/recovery-backup.dump" "$RECOVERY_DUMP_FILE"
echo "=== Safety-net dump saved: $RECOVERY_DUMP_FILE ==="
ls -l "$RECOVERY_DUMP_FILE"

# ============================================================
# 3. Continue the backfill -- naturally resumes on whatever's still
#    edgeless, no separate bookkeeping needed.
# ============================================================
echo "=== [backfill] resuming scripts/backfill_chunk_edges.py ==="
export VLLM_API_KEY VLLM_EMBEDDING_DIM POSTGRES_DSN
"$HOME/miniforge3/envs/knightGPT/bin/python" scripts/backfill_chunk_edges.py \
    --duckdb-path "$VALIDATION_DUCKDB" \
    --log-level INFO 2>&1 | tee "$BUILD_ROOT/logs/backfill-resume.log"
BACKFILL_RC=${PIPESTATUS[0]}
[ "$BACKFILL_RC" = "0" ] || echo "WARNING: backfill exited non-zero ($BACKFILL_RC) -- the recovery dump above is still a safe checkpoint to resume from again"

singularity exec "${PG_ENV[@]}" --bind "$BUILD_ROOT/verify.sql:/verify.sql:ro" instance://pg bash -c \
    '/work/pg17/bin/psql -h localhost -p 5432 -U postgres -d knightgpt -t -f /verify.sql' \
    2>&1 | tee "$BUILD_ROOT/logs/verify-final.log"

# ============================================================
# 4. Final dump, teardown.
# ============================================================
FINAL_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
FINAL_DUMP_FILE="$DUMP_OUT_DIR/knightgpt-postgres-${FINAL_TIMESTAMP}.dump"
runpg "/work/pg17/bin/pg_dump -h localhost -p 5432 -U postgres -d knightgpt -Fc -f /pgdata/final-backup.dump" \
    2>&1 | tee "$BUILD_ROOT/logs/pg-dump-final.log"
DUMP_RC=${PIPESTATUS[0]}
if [ "$DUMP_RC" = "0" ]; then
    cp "$PREV_JOB_PGDATA/final-backup.dump" "$FINAL_DUMP_FILE"
    ls -l "$FINAL_DUMP_FILE"
fi

echo "=== BACKFILL RESUME COMPLETE ==="

runpg '/work/pg17/bin/pg_ctl -D /pgdata stop -m fast' 2>&1 || true
singularity instance stop pg 2>&1 || true
rm -rf "$BUILD_ROOT"

# Only remove the reused pgdata once a fresh dump of it is confirmed safely
# on scratch -- never delete a working directory whose contents aren't
# durably saved elsewhere yet.
if [ "$DUMP_RC" = "0" ] && [ -s "$FINAL_DUMP_FILE" ]; then
    rm -rf "/cosmos/vast/scratch/l1joseph/knightgpt/singularity-proof-test/101691"
    echo "Removed recovered job 101691's scratch dir (superseded by $FINAL_DUMP_FILE)"
else
    echo "Keeping job 101691's pgdata at $PREV_JOB_PGDATA -- final dump was not confirmed"
fi

echo "=== $(date -Is) job ${SLURM_JOB_ID} complete ==="
