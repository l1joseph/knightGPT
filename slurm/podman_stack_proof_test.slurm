#!/bin/bash
#SBATCH --job-name=proof-test-podman-stack
#SBATCH --output=/cosmos/nfs/home/l1joseph/knightGPT/logs/%x_%j.out
#SBATCH --error=/cosmos/nfs/home/l1joseph/knightGPT/logs/%x_%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

REPO=/cosmos/nfs/home/l1joseph/knightGPT/.worktrees/qiita-knightgpt-webui-deploy
VENV=/cosmos/vast/scratch/l1joseph/knightgpt/podman-compose-venv
COMPOSE_FILES=(-f docker/docker-compose.yaml -f docker/docker-compose.local-test.yaml)
cd "$REPO"

export PATH="$VENV/bin:$PATH"

# Rootless podman's default storage root (~/.local/share/containers/storage)
# lives on Cosmos's NFS-backed home (/cosmos/nfs/home) -- confirmed via
# `podman info` ("Backing Filesystem: nfs"). Extracting image layers that
# carry xattrs (e.g. rust:1.96.0-bookworm's /boot content) onto NFS fails
# with "lsetxattr ...: operation not supported" (confirmed failure from
# job 101197's run). CONTAINERS_STORAGE_CONF does NOT override this on
# this podman version (confirmed: graphRoot stayed on NFS with it set --
# job 101206 failed identically). The mechanism that actually works is
# podman's own --root/--runroot CLI flags (confirmed via a cheap `podman
# info` check on the login node). podman-compose shells out to a plain
# `podman` resolved via $PATH for every operation (build/up/inspect/exec/
# logs/ps), so wrap that binary to always inject --root/--runroot rather
# than trying to thread the flags through podman-compose's own
# --podman-args (which wouldn't cover the direct `podman inspect`/`exec`/
# `logs`/`ps` calls below). This makes the storage node-local and
# invisible from the login node, so teardown (podman-compose down, podman
# system prune) must also happen inside this same job, not as a separate
# later step.
#
# Second, separate issue hit after fixing the above (job 101220): this
# account has no /etc/subuid or /etc/subgid entries at all (confirmed via
# `grep '^l1joseph:' /etc/subuid /etc/subgid` -- no output), so rootless
# podman's user namespace can only map a very narrow UID/GID range.
# Extracting a real Debian-based layer (rust:1.96.0-bookworm) needs to
# chown files up to GID 42 (/etc/gshadow, the "shadow" group) and fails
# with "potentially insufficient UIDs or GIDs available in user
# namespace ... lchown /etc/gshadow: invalid argument". This is a host
# account provisioning gap (no sysadmin access from here to fix
# /etc/subuid/subgid), not a knightGPT bug. Confirmed workaround (tested
# directly against the exact failing image on the login node, ~12s,
# succeeded): --storage-opt overlay.ignore_chown_errors=true, which lets
# the overlay driver proceed past chown failures during layer extraction.
#
# Third, separate issue hit after fixing both of the above (job 101221):
# image pull/extraction now succeeded fully (all 5 blobs copied), but the
# very next step -- actually spawning a container to run a Dockerfile RUN
# instruction -- failed with "default OCI runtime \"runc\" not found:
# invalid argument", and every subsequent podman call (network, volume,
# ps) failed identically. Also present: repeated warnings "Failed to get
# rootless runtime dir for DefaultAPIAddress: lstat /run/user/537937: no
# such file or directory". /run/user/<uid> is normally created by a
# systemd-logind session (present on an interactive SSH login to the
# login node -- confirmed there via `podman info` showing runc correctly
# at /usr/bin/runc); a non-interactive sbatch job on a compute node gets
# no such session, so XDG_RUNTIME_DIR points at a directory that was
# never created, and rootless podman's default "systemd" cgroup manager
# then fails in a way that surfaces as the confusing "OCI runtime not
# found" message rather than a clearer cgroup/session error. Fix: give
# podman its own real XDG_RUNTIME_DIR under this job's local root, and
# force the "cgroupfs" cgroup manager, which doesn't require a systemd
# user session at all.
PROOF_TEST_LOCAL_ROOT="/tmp/knightgpt-proof-test-${SLURM_JOB_ID}"
mkdir -p "$PROOF_TEST_LOCAL_ROOT/storage" "$PROOF_TEST_LOCAL_ROOT/run" "$PROOF_TEST_LOCAL_ROOT/bin" "$PROOF_TEST_LOCAL_ROOT/xdg-runtime"
chmod 700 "$PROOF_TEST_LOCAL_ROOT/xdg-runtime"
export XDG_RUNTIME_DIR="$PROOF_TEST_LOCAL_ROOT/xdg-runtime"
REAL_PODMAN="$(command -v podman)"
cat > "$PROOF_TEST_LOCAL_ROOT/bin/podman" <<WRAPPER
#!/bin/bash
exec "$REAL_PODMAN" --root "$PROOF_TEST_LOCAL_ROOT/storage" --runroot "$PROOF_TEST_LOCAL_ROOT/run" --storage-opt overlay.ignore_chown_errors=true --cgroup-manager=cgroupfs "\$@"
WRAPPER
chmod +x "$PROOF_TEST_LOCAL_ROOT/bin/podman"
export PATH="$PROOF_TEST_LOCAL_ROOT/bin:$PATH"

export DOCKER_HOST="unix://$(podman info --format '{{.Host.RemoteSocket.Path}}' 2>/dev/null || echo /tmp/podman.sock)"

SCRATCH_TEST_DIRS=(
  /cosmos/vast/scratch/l1joseph/knightgpt/podman-test-pgdata
  /cosmos/vast/scratch/l1joseph/knightgpt/podman-test-duckdb
)

cleanup() {
  echo "=== Tearing down (cleanup trap) ==="
  podman-compose "${COMPOSE_FILES[@]}" down -v || true
  podman system prune -af || true
  echo "podman ps -a after teardown:"
  podman ps -a || true
  rm -rf "${SCRATCH_TEST_DIRS[@]}"
  # Some overlay layer content is owned (inside the rootless user
  # namespace) by IDs the plain host user can't unlink directly, since
  # this account has no /etc/subuid/subgid range (see above). Use
  # `podman unshare` to remove local storage from inside the same
  # namespace mapping podman itself used; fall back to a plain rm if
  # unshare itself isn't usable for any reason.
  podman unshare rm -rf "$PROOF_TEST_LOCAL_ROOT" 2>/dev/null || rm -rf "$PROOF_TEST_LOCAL_ROOT" || true
  echo "Scratch test dirs after cleanup (du should fail / report gone):"
  for d in "${SCRATCH_TEST_DIRS[@]}"; do
    du -sh "$d" 2>&1 || echo "  $d: gone (expected)"
  done
}
trap cleanup EXIT

# podman-compose reads docker-compose.yaml by default; -f order matters,
# later files override earlier ones.
podman-compose "${COMPOSE_FILES[@]}" up -d --build postgres

echo "Waiting for postgres healthcheck..."
status="starting"
for i in $(seq 1 30); do
  status=$(podman inspect --format '{{.State.Health.Status}}' knightgpt-postgres 2>/dev/null || echo "starting")
  [ "$status" = "healthy" ] && break
  sleep 5
done
if [ "$status" != "healthy" ]; then
  echo "postgres never became healthy"
  podman logs knightgpt-postgres || true
  exit 1
fi

echo "Verifying restore..."
podman exec knightgpt-postgres psql -U postgres -d knightgpt -t -c \
  "SELECT 'qiita_studies=' || count(*) FROM qiita_studies
   UNION ALL SELECT 'qiita_study_publications=' || count(*) FROM qiita_study_publications
   UNION ALL SELECT 'paper_study_links=' || count(*) FROM paper_study_links
   UNION ALL SELECT 'papers=' || count(*) FROM papers;"

echo "Bringing up api + open-webui..."
podman-compose "${COMPOSE_FILES[@]}" \
  --env VLLM_API_KEY="$VLLM_API_KEY" --env VLLM_EMBEDDING_DIM="$VLLM_EMBEDDING_DIM" \
  up -d --build api open-webui

echo "Waiting for api healthcheck..."
status="starting"
for i in $(seq 1 30); do
  status=$(podman inspect --format '{{.State.Health.Status}}' knightgpt-api 2>/dev/null || echo "starting")
  [ "$status" = "healthy" ] && break
  sleep 5
done
if [ "$status" != "healthy" ]; then
  echo "api never became healthy"
  podman logs knightgpt-api || true
  exit 1
fi

echo "Real end-to-end check: a chat completion through the deployed stack"
chat_response=$(podman exec knightgpt-api curl -sS -w '\nHTTP_STATUS:%{http_code}' -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the KEGG pathway for the citrate cycle?"}],"stream":false}')
echo "$chat_response"

http_status=$(echo "$chat_response" | grep -o 'HTTP_STATUS:[0-9]*' | cut -d: -f2)
if [ "$http_status" != "200" ]; then
  echo "chat completion returned non-200 status ($http_status) -- NOT a pass"
  exit 1
fi

echo "PROOF TEST PASSED"
