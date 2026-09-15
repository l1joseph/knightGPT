#!/bin/bash
#SBATCH --job-name=stack-proof-singularity
#SBATCH --output=/cosmos/nfs/home/l1joseph/knightGPT/logs/%x_%j.out
#SBATCH --error=/cosmos/nfs/home/l1joseph/knightGPT/logs/%x_%j.err
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#
# Local proof test of the full knightGPT stack (Postgres+pgGraph, API,
# Open WebUI) via singularitypro on SDSC Cosmos, ahead of the real
# kl-remote deploy.
#
# Supersedes the earlier podman-compose-based attempt
# (slurm/podman_stack_proof_test.slurm, removed): rootless podman is
# structurally blocked on this cluster ("cannot re-exec process to join
# the existing user namespace" / "no subuid ranges found").
#
# IMPORTANT DEVIATION FROM THE ORIGINAL PLAN, discovered and worked
# through in this session (see the Task 9 report for the full blow by
# blow): `singularity build` of a custom multi-stage .def (the direct
# equivalent of docker/postgres/Dockerfile) is ALSO structurally blocked
# on this cluster -- confirmed directly:
#   - plain `singularity build docker/postgres/postgres.def` (non-root):
#     "FATAL: --remote, --fakeroot, or the proot command are required to
#     build this source as a non-root user"
#   - `singularity build --fakeroot ...`: "FATAL: could not use fakeroot:
#     no mapping entry found in /etc/subuid for l1joseph" -- confirmed
#     /etc/subuid and /etc/subgid are completely empty cluster-wide, and
#     that --fakeroot fails identically for plain `exec`, not just
#     `build`. This is the SAME root cause as the podman blocker (no
#     subuid/subgid allocation for this account, cluster-wide), just
#     surfacing through a different tool. `singularity remote` (Sylabs
#     Cloud build) is configured but needs auth this session doesn't
#     have (and shipping this repo's source to a third-party builder
#     isn't desirable anyway); `proot` isn't installed anywhere on the
#     cluster.
#
# So: no root, ever, inside any singularity container on this cluster,
# by any built-in mechanism. `docker/postgres/postgres.def` and
# docker/api.def (which both assumed `singularity build` would work)
# were removed as unusable dead weight. The WORKING alternative used
# below needs no root at all:
#   - `singularity instance start docker://...` (plain pull) + a real
#     disk-backed overlay (`singularity overlay create`, itself
#     unprivileged) gives a writable container without fakeroot.
#   - Postgres 17 is compiled FROM SOURCE into a self-owned prefix
#     (./configure && make && make install, all unprivileged -- building
#     from a plain user prefix has never needed root) instead of
#     apt-installing postgresql-server-dev-17.
#   - The only two extra build tools missing from the rust:1.96.0-bookworm
#     image (bison/flex, needed by Postgres's ./configure even against a
#     release tarball with pre-generated parser files; clang/libclang,
#     needed by cargo-pgrx's bindgen step) are pulled from conda-forge
#     into an isolated scratch-only env via mamba -- conda never needs
#     root, unlike apt.
#   - pgGraph is built with cargo-pgrx against that self-built Postgres,
#     then its packaged output is merged directly into the same prefix
#     (guaranteed ABI match, since both came from the identical build).
#   - At runtime, Postgres is *our own compiled binary*, started and
#     initialized with a short hand-written init sequence
#     (initdb -> start -> CREATE EXTENSION -> pg_restore -> TRUNCATE)
#     standing in for the official image's docker-entrypoint.sh
#     (which we no longer have, since we're not using that image).
#   - The API needs no build-time root either: `pip install --target=`
#     into a bind-mounted scratch directory works against a stock
#     `python:3.11-slim` pull with no overlay needed at all.
#
# Singularity instances share the host network namespace by default (no
# bridge network needed, unlike Docker Compose) -- all three services
# run on this one allocated node and talk to each other over
# localhost/127.0.0.1, each on its own port:
#   postgres -> 5432, api -> 8080, open-webui -> 3000

set -uo pipefail

REPO=/cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy
cd "$REPO"

BUILD_ROOT="/cosmos/vast/scratch/l1joseph/knightgpt/singularity-proof-test/${SLURM_JOB_ID}"
mkdir -p "$BUILD_ROOT"/{logs,pgdata,duckdb,api-app-logs,webui-data,singularity-tmp}

# The Postgres-from-source + pgGraph build (proven working, ~10min) and
# its conda-forge build-tool env are kept in a STABLE, non-job-ID-scoped
# location and are idempotent ([ -x pg_config ] || build, etc.) so that
# iterating on the remaining (runtime) steps does not pay the full
# rebuild cost every single job submission. Removed as part of this
# same script's teardown once the full proof test actually passes.
PERSIST_ROOT="/cosmos/vast/scratch/l1joseph/knightgpt/singularity-proof-test/persistent"
mkdir -p "$PERSIST_ROOT"/{pgwork,tools,api-packages}

export SINGULARITY_TMPDIR="$BUILD_ROOT/singularity-tmp"
# Cache dir kept in PERSIST_ROOT (not job-ID-scoped) so the docker://
# image pulls (rust/python/open-webui layers) are reused across
# iterations instead of re-downloaded every single job submission.
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

# --- Read required secrets/config from .env (sbatch does not source it) ---
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
#    unprivileged, into $PERSIST_ROOT/pgwork -- real compute, why this runs
#    as a SLURM job and not on the login node.
# ============================================================
echo "=== [build] conda-forge tools (bison/flex/m4/clang/libclang) -- no apt/root available on this cluster ==="
TOOLS_ENV="$PERSIST_ROOT/tools/pgbuild-tools"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
# zlib added after job 101287s real error: pg_restore refused to read
# the (gzip-compressed, standard pg_dump -Fc default) backup dump
# because Postgres was built --without-zlib. Same no-root conda-forge
# route as bison/flex/clang.
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

echo "=== [build] stopping builder instance (pgwork already persisted on scratch, outside the overlay) ==="
singularity instance stop builder

# ============================================================
# 2. Runtime: start Postgres (our self-built binary, run inside a plain
#    unprivileged instance -- no overlay needed, only bind-mounted
#    writable dirs), initialize it ourselves (standing in for the
#    official image's docker-entrypoint.sh, which we no longer have),
#    verify the restore.
# ============================================================
echo "=== [pg] starting postgres runtime instance ==="
# $TOOLS_ENV is bound in too (read-only) because our self-built postgres
# binaries (postgres/initdb/pg_restore/psql/...) are now dynamically
# linked against the conda-forge zlib in that env (see the configure
# --with-includes/--with-libraries change above) -- without it and
# LD_LIBRARY_PATH below, every one of those binaries would fail at
# process start with a shared-library-not-found error, not just pg_restore.
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

echo "=== [pg] initdb + start (our own init sequence, standing in for docker-entrypoint.sh) ==="
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
# pg_restore assigns brand-new OIDs to the recreated chunks/chunk_edges
# tables, but the restored graph._registered_tables/_registered_edges
# rows still reference the OLD source-database OIDs -- pgGraph resolves
# registration by OID, not by table name, confirmed and documented in
# k8s/nrp/RESTORE.md ("The risk: stale OIDs in pgGraph's registration
# tables"), which is exactly the "registered table relation no longer
# exists" error hit in job 101295's /health check. Fixed the same way
# documented there (and now also in docker/postgres/init/02-restore-backup.sh
# for the real deployment): re-apply sql/schema.sql (its DO $$ blocks
# upsert-by-name, re-pointing the stored OIDs) then rebuild the graph
# projection.
runpg '
set -e
PGBIN=/work/pg17/bin
"$PGBIN/createdb" -h localhost -p 5432 -U postgres knightgpt
"$PGBIN/psql" -h localhost -p 5432 -U postgres -d knightgpt -c "CREATE EXTENSION IF NOT EXISTS graph;"
"$PGBIN/pg_restore" -h localhost -p 5432 -U postgres -d knightgpt --no-owner --if-exists --clean /dump/backup.dump
"$PGBIN/psql" -h localhost -p 5432 -U postgres -d knightgpt -c "TRUNCATE papers, chunks, chunk_edges CASCADE;"
"$PGBIN/psql" -h localhost -p 5432 -U postgres -d knightgpt -f /dump/schema.sql
"$PGBIN/psql" -h localhost -p 5432 -U postgres -d knightgpt -c "SELECT graph.build();"
' 2>&1 | tee "$BUILD_ROOT/logs/pg-init-sequence.log"
RESTORE_RC=${PIPESTATUS[0]}
[ "$RESTORE_RC" = "0" ] || fail "postgres create/extension/restore/truncate sequence failed (see pg-init-sequence.log above)"

echo "=== [pg] verifying restore (real row counts) ==="
# Written to a file (rather than inlined through several layers of shell
# quoting: outer script -> runpg() -> singularity exec -> nested bash -c)
# to avoid a fragile nested-quoting bug.
cat > "$BUILD_ROOT/pgdata/verify.sql" <<'SQL'
SELECT 'qiita_studies=' || count(*) FROM qiita_studies
UNION ALL SELECT 'qiita_study_publications=' || count(*) FROM qiita_study_publications
UNION ALL SELECT 'paper_study_links=' || count(*) FROM paper_study_links
UNION ALL SELECT 'papers=' || count(*) FROM papers;
SQL
runpg '/work/pg17/bin/psql -h localhost -p 5432 -U postgres -d knightgpt -t -f /pgdata/verify.sql' 2>&1 | tee "$BUILD_ROOT/logs/verify-counts.log"
[ -s "$BUILD_ROOT/logs/verify-counts.log" ] || fail "row-count verification produced no output"

# ============================================================
# 3. Runtime: API (stock python:3.11-slim, pip install --target=, no
#    overlay needed)
# ============================================================
echo "=== [api] starting api instance ==="
# DuckDB's own default extension-cache path is $HOME/.duckdb, resolved
# internally regardless of anything we set (confirmed in job 101289:
# "Failed to create directory '$HOME/.duckdb': Read-only file system" --
# $HOME fell back to the real, read-only-in-container NFS home, same
# SINGULARITYENV_HOME restriction as before, but this time it's DuckDB's
# own internal default, not something our env vars or the app's code
# control). Since HOME itself can't be overridden on this cluster, bind a
# writable scratch dir directly over that exact path instead.
mkdir -p "$BUILD_ROOT/duckdb-home"
singularity instance start --no-home \
    --bind "$REPO/data:/app/data:ro" \
    --bind "$REPO/src:/app/src:ro" \
    --bind "$REPO/requirements.txt:/app/requirements.txt:ro" \
    --bind "$PERSIST_ROOT/api-packages:/packages" \
    --bind "$BUILD_ROOT/duckdb:/app/duckdb_data" \
    --bind "$BUILD_ROOT/api-app-logs:/app/logs" \
    --bind "$BUILD_ROOT/duckdb-home:$HOME/.duckdb" \
    docker://python:3.11-slim api
singularity instance list

echo "=== [api] pip install --target=/packages (no root/apt needed -- wheel-only deps) ==="
# This cluster's singularity config refuses to let SINGULARITYENV_HOME
# override HOME at all ("Overriding HOME environment variable ... is not
# permitted", confirmed in job 101288's log) -- HOME silently falls back
# to the real (read-only-inside-container) NFS home. That's harmless for
# psql/pg_restore (only used for optional ~/.pgpass-style lookups) but
# broke `pip install --upgrade pip` specifically, since with no --target
# it falls back to a $HOME/.local user-site install. Dropped that
# upgrade step entirely -- unneeded; the real install below is fully
# explicit via --target and never touches $HOME.
singularity exec --no-home instance://api bash -c '
set -e
pip install --no-cache-dir --target=/packages -r /app/requirements.txt
' 2>&1 | tee "$BUILD_ROOT/logs/pip-install.log"
PIP_RC=${PIPESTATUS[0]}
if [ "$PIP_RC" != "0" ]; then
    tail -n 100 "$BUILD_ROOT/logs/pip-install.log"
    fail "api pip install failed"
fi

API_ENV=(--env HOME=/tmp --env PYTHONPATH=/packages:/app \
    --env VLLM_API_KEY="$VLLM_API_KEY" \
    --env VLLM_EMBEDDING_URL="$VLLM_EMBEDDING_URL" \
    --env VLLM_INFERENCE_URL="$VLLM_INFERENCE_URL" \
    --env VLLM_EMBEDDING_MODEL="$VLLM_EMBEDDING_MODEL" \
    --env VLLM_INFERENCE_MODEL="$VLLM_INFERENCE_MODEL" \
    --env VLLM_EMBEDDING_DIM="$VLLM_EMBEDDING_DIM" \
    --env POSTGRES_DSN="$POSTGRES_DSN" \
    --env INGEST_DUCKDB_PATH="/app/duckdb_data/embeddings.duckdb" \
    --env LOG_LEVEL="INFO")

singularity exec "${API_ENV[@]}" instance://api bash -c 'cd /app && exec python -m src.api.main --host 0.0.0.0 --port 8080' \
    > "$BUILD_ROOT/logs/api.out" 2> "$BUILD_ROOT/logs/api.err" &
API_PID=$!
echo "api backgrounded as host PID $API_PID"

echo "=== [api] waiting for readiness (http://localhost:8080/health) ==="
API_READY=0
for i in $(seq 1 60); do
    if curl -sf -o "$BUILD_ROOT/logs/health-response.json" http://localhost:8080/health; then
        echo "api ready after $((i*5))s"; API_READY=1; break
    fi
    sleep 5
done
if [ "$API_READY" != "1" ]; then
    echo "--- last 200 lines of api.out ---"; tail -n 200 "$BUILD_ROOT/logs/api.out"
    echo "--- last 200 lines of api.err ---"; tail -n 200 "$BUILD_ROOT/logs/api.err"
    fail "api never became ready"
fi
echo "--- /health response ---"; cat "$BUILD_ROOT/logs/health-response.json"; echo

# ============================================================
# 4. Runtime: Open WebUI (no custom build -- pulled directly)
# ============================================================
echo "=== [webui] starting open-webui instance ==="
singularity instance start --no-home \
    --bind "$BUILD_ROOT/webui-data:/app/backend/data" \
    docker://ghcr.io/open-webui/open-webui:main webui \
    2>&1 | tee "$BUILD_ROOT/logs/instance-start-webui.log"
singularity instance list

singularity exec --no-home \
    --env HOME=/tmp --env PORT=3000 --env HOST=0.0.0.0 \
    --env OPENAI_API_BASE_URL="http://localhost:8080/v1" \
    --env OPENAI_API_KEY="EMPTY" \
    --env WEBUI_AUTH="false" \
    --env ENABLE_OLLAMA_API="false" \
    instance://webui bash -c '
        set -e
        if [ -f /app/backend/start.sh ]; then
            cd /app/backend && exec bash start.sh
        elif [ -f /start.sh ]; then
            exec bash /start.sh
        else
            exec open-webui serve --host "${HOST:-0.0.0.0}" --port "${PORT:-3000}"
        fi
    ' > "$BUILD_ROOT/logs/webui.out" 2> "$BUILD_ROOT/logs/webui.err" &
WEBUI_PID=$!
echo "open-webui backgrounded as host PID $WEBUI_PID"

echo "=== [webui] waiting for readiness (http://localhost:3000/) ==="
WEBUI_READY=0
for i in $(seq 1 60); do
    # curl's -w '%{http_code}' already prints "000" on a connection
    # failure, so a `|| echo "000"` fallback here double-appends a second
    # "000" (-> the bogus string "000000", which then passed the numeric
    # `-lt 500` check as a false positive -- confirmed in job 101296's
    # log: "open-webui responding (HTTP 000000)" after only 5s, when it
    # almost certainly wasn't actually up yet). curl itself never fails
    # to produce -w output here, so no fallback is needed at all.
    code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/)
    if [ "$code" != "000" ] && [ "$code" -lt 500 ]; then
        echo "open-webui responding (HTTP $code) after $((i*5))s"; WEBUI_READY=1; break
    fi
    sleep 5
done
if [ "$WEBUI_READY" != "1" ]; then
    echo "--- last 200 lines of webui.out ---"; tail -n 200 "$BUILD_ROOT/logs/webui.out"
    echo "--- last 200 lines of webui.err ---"; tail -n 200 "$BUILD_ROOT/logs/webui.err"
    echo "WARNING: open-webui did not respond -- continuing to the api chat-completion test anyway (item 8 allows testing directly against the api)."
fi

# ============================================================
# 5. Real end-to-end check: a genuine chat completion through the
#    deployed API (real multi-tool agent loop, real NRP LLM call)
# ============================================================
echo "=== Real end-to-end check: chat completion via /v1/chat/completions ==="
HTTP_CODE=$(curl -sS -o "$BUILD_ROOT/logs/chat-completion-response.json" -w '%{http_code}' \
    -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"What is the KEGG pathway for the citrate cycle?"}],"stream":false}')
echo "chat completion HTTP status: $HTTP_CODE"
echo "--- chat completion response body ---"
cat "$BUILD_ROOT/logs/chat-completion-response.json"; echo

if [ "$HTTP_CODE" != "200" ]; then
    fail "chat completion returned HTTP $HTTP_CODE (see response above)"
fi
if [ ! -s "$BUILD_ROOT/logs/chat-completion-response.json" ]; then
    fail "chat completion returned HTTP 200 but an empty body"
fi

echo "=== PROOF TEST PASSED ==="

# ============================================================
# 6. Clean teardown (reverse order), then remove scratch artifacts
# ============================================================
echo "=== Tearing down ==="
singularity instance stop webui 2>&1 || true
singularity instance stop api 2>&1 || true
runpg '/work/pg17/bin/pg_ctl -D /pgdata stop -m fast' 2>&1 || true
singularity instance stop pg 2>&1 || true
echo "--- instance list after stop (should be empty) ---"
singularity instance list

echo "=== Removing scratch build/run artifacts: $BUILD_ROOT ==="
rm -rf "$BUILD_ROOT"
if [ -d "$BUILD_ROOT" ]; then
    echo "WARNING: $BUILD_ROOT still exists after rm -rf"
else
    echo "Confirmed removed: $BUILD_ROOT"
fi

echo "=== Removing persistent build cache now that the full proof test has passed: $PERSIST_ROOT ==="
rm -rf "$PERSIST_ROOT"
if [ -d "$PERSIST_ROOT" ]; then
    echo "WARNING: $PERSIST_ROOT still exists after rm -rf"
else
    echo "Confirmed removed: $PERSIST_ROOT"
fi

echo "=== $(date -Is) job ${SLURM_JOB_ID} complete ==="
