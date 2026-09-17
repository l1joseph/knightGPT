#!/bin/bash
#SBATCH --job-name=backfill-chunk-edges
#SBATCH --output=/cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy/logs/%x_%j.out
#SBATCH --error=/cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy/logs/%x_%j.err
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#
# Runs scripts/backfill_chunk_edges.py against the current corpus:
# builds Postgres+pgGraph (reusing the cached build), restores the
# newest dump, backfills chunk_edges for every chunk with none (the
# ~340 papers ingested during the poisoned-DuckDB-row window, jobs
# 101550/101647), takes a fresh dump, tears down. No NRP calls, no
# downloads -- purely local Postgres+DuckDB work, so this should run
# in minutes, not hours.

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

POSTGRES_DSN="postgresql://postgres:password@localhost:5432/knightgpt"

DUMP_FILE=$(ls -t "$DUMP_OUT_DIR"/knightgpt-postgres-*.dump 2>/dev/null | head -1)
[ -n "$DUMP_FILE" ] && [ -f "$DUMP_FILE" ] || fail "no backup dump found in $DUMP_OUT_DIR"
echo "Restoring from: $DUMP_FILE"
ls -l "$DUMP_FILE"

# ============================================================
# 1. Build Postgres 17 + pgGraph (idempotent -- should be near-instant,
#    reusing the persistent build cache).
# ============================================================
echo "=== [build] conda-forge tools ==="
TOOLS_ENV="$PERSIST_ROOT/tools/pgbuild-tools"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
[ -x "$TOOLS_ENV/bin/bison" ] || mamba create -y -p "$TOOLS_ENV" -c conda-forge bison flex m4 clang libclang zlib 2>&1 | tail -n 30

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
ls -la /work/pg17/lib/postgresql/graph.so
' || fail "pgGraph build/merge failed"

singularity instance stop builder

# ============================================================
# 2. Runtime: start Postgres, restore, verify.
# ============================================================
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

runpg '
set -e
PGBIN=/work/pg17/bin
"$PGBIN/initdb" -D /pgdata -U postgres -A trust --locale=C > /tmp/initdb.log 2>&1
"$PGBIN/pg_ctl" -D /pgdata -l /pgdata/server.log -o "-c shared_preload_libraries=graph -c listen_addresses=localhost -p 5432 -k /tmp" start -w -t 60
' || { runpg 'tail -n 60 /tmp/initdb.log; echo ---SERVER---; tail -n 100 /pgdata/server.log'; fail "postgres initdb/start failed"; }

PG_READY=0
for i in $(seq 1 30); do
    if runpg '/work/pg17/bin/pg_isready -h localhost -p 5432 -U postgres' >/dev/null 2>&1; then
        PG_READY=1; break
    fi
    sleep 3
done
[ "$PG_READY" = "1" ] || { runpg 'tail -n 100 /pgdata/server.log'; fail "postgres never became ready"; }

runpg '
set -e
PGBIN=/work/pg17/bin
"$PGBIN/createdb" -h localhost -p 5432 -U postgres knightgpt
"$PGBIN/psql" -h localhost -p 5432 -U postgres -d knightgpt -c "CREATE EXTENSION IF NOT EXISTS graph;"
"$PGBIN/pg_restore" -h localhost -p 5432 -U postgres -d knightgpt --no-owner --if-exists --clean /dump/backup.dump
"$PGBIN/psql" -v ON_ERROR_STOP=1 -h localhost -p 5432 -U postgres -d knightgpt -f /dump/schema.sql
"$PGBIN/psql" -v ON_ERROR_STOP=1 -h localhost -p 5432 -U postgres -d knightgpt -c "SELECT graph.build();"
' 2>&1 | tee "$BUILD_ROOT/logs/pg-init-sequence.log"
RESTORE_RC=${PIPESTATUS[0]}
[ "$RESTORE_RC" = "0" ] || fail "postgres restore sequence failed"

cat > "$BUILD_ROOT/pgdata/verify.sql" <<'SQL'
SELECT 'papers=' || count(*) FROM papers
UNION ALL SELECT 'chunks=' || count(*) FROM chunks
UNION ALL SELECT 'chunk_edges=' || count(*) FROM chunk_edges
UNION ALL SELECT 'edgeless_chunks=' || count(*) FROM chunks c
    WHERE NOT EXISTS (SELECT 1 FROM chunk_edges e WHERE e.src_chunk_id = c.id);
SQL
echo "=== [pg] state before backfill ==="
runpg '/work/pg17/bin/psql -h localhost -p 5432 -U postgres -d knightgpt -t -f /pgdata/verify.sql' 2>&1 | tee "$BUILD_ROOT/logs/verify-before.log"

# ============================================================
# 3. Run the backfill.
# ============================================================
echo "=== [backfill] running scripts/backfill_chunk_edges.py ==="
export VLLM_API_KEY VLLM_EMBEDDING_DIM POSTGRES_DSN
"$HOME/miniforge3/envs/knightGPT/bin/python" scripts/backfill_chunk_edges.py \
    --duckdb-path "$VALIDATION_DUCKDB" \
    --log-level INFO 2>&1 | tee "$BUILD_ROOT/logs/backfill.log"
BACKFILL_RC=${PIPESTATUS[0]}
[ "$BACKFILL_RC" = "0" ] || fail "backfill failed (see backfill.log above)"

echo "=== [pg] state after backfill ==="
runpg '/work/pg17/bin/psql -h localhost -p 5432 -U postgres -d knightgpt -t -f /pgdata/verify.sql' 2>&1 | tee "$BUILD_ROOT/logs/verify-after.log"

# ============================================================
# 4. Fresh dump, teardown.
# ============================================================
FINAL_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
FINAL_DUMP_FILE="$DUMP_OUT_DIR/knightgpt-postgres-${FINAL_TIMESTAMP}.dump"
runpg "/work/pg17/bin/pg_dump -h localhost -p 5432 -U postgres -d knightgpt -Fc -f /pgdata/final-backup.dump" \
    2>&1 | tee "$BUILD_ROOT/logs/pg-dump.log"
DUMP_RC=${PIPESTATUS[0]}
if [ "$DUMP_RC" = "0" ]; then
    cp "$BUILD_ROOT/pgdata/final-backup.dump" "$FINAL_DUMP_FILE"
    ls -l "$FINAL_DUMP_FILE"
else
    echo "WARNING: final pg_dump failed"
fi

echo "=== BACKFILL COMPLETE ==="

runpg '/work/pg17/bin/pg_ctl -D /pgdata stop -m fast' 2>&1 || true
singularity instance stop pg 2>&1 || true
rm -rf "$BUILD_ROOT"

echo "=== $(date -Is) job ${SLURM_JOB_ID} complete ==="
