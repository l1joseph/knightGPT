# NRP Storage Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provision a persistent, durable Postgres+pgGraph (self-managed Kubernetes Deployment, no operator) and DuckDB (via a standalone PVC) storage deployment on NRP's Kubernetes cluster, ready for a future ingestion Job and MCP server to use.

**Architecture:** GitHub Actions builds the existing `docker/postgres/Dockerfile` image and pushes it to GHCR (no local Docker/Podman needed). A plain Kubernetes `Deployment` (1 replica) in the `knightlab-ml` namespace runs that image directly against its own PVC, fronted by a `Service`. The `graph` extension is created via a direct `psql` statement during Task 2's verification and again (idempotently) by `scripts/apply_schema.py` in Task 4. Two Postgres operators (CloudNativePG, then Zalando) were tried and abandoned during live implementation — see Task 2's amendment note and the design spec for full reasoning; neither's core value (managed multi-replica failover) applies at a single instance. A separate `rook-cephfs` PVC holds the DuckDB file, mountable by different pods at different times per DuckDB's single-writer constraint.

**Tech Stack:** GitHub Actions, GHCR, Kubernetes (NRP "nautilus" cluster) — plain `Deployment`/`PersistentVolumeClaim`/`Service`, no Postgres operator (CloudNativePG has no RBAC grant in this namespace; Zalando's operator requires a Spilo-based image incompatible with our vanilla-postgres-based one) — Rook-Ceph (`rook-cephfs` storage class) and Linstor (`linstor-igrok` storage class, both confirmed available in the `knightlab-ml` namespace).

## Global Constraints

- Namespace: `knightlab-ml` (kubectl already configured, context `nautilus`, this is the default namespace).
- GHCR image target: `ghcr.io/l1joseph/knightgpt-postgres`.
- Postgres storage class: `linstor-igrok` (NRP's documented preference for Postgres performance; confirmed present via `kubectl get storageclass`).
- DuckDB PVC storage class: `rook-cephfs` (RWX required — multiple pods at different times mount it; confirmed present).
- Postgres storage size: `20Gi` (current data ~516MB at 117 papers, ~4GB projected at ~900 papers — 5x headroom).
- DuckDB PVC size: `20Gi` (current embeddings ~89MB at 117 papers, ~680MB projected at ~900 papers — ~29x headroom).
- No external network exposure for either resource in this plan — in-cluster (ClusterIP) only.
- Postgres deployment: **self-managed `Deployment`, no operator.** CloudNativePG has no RBAC access in this namespace (confirmed via `kubectl auth can-i`); Zalando requires a Spilo-based image (ours is vanilla `postgres:17-bookworm`, incompatible). See Task 2's amendment note and the design spec for full reasoning.
- Never push directly to a shared branch (`vllm`, `main`) to work around an obstacle — this happened once already in this plan's execution and had to be reverted. If a documented step doesn't work, find another way within this task's own branch/scope, or stop and report BLOCKED.
- The Postgres service is `knightgpt-postgres.knightlab-ml.svc.cluster.local`, credentials in Secret `knightgpt-postgres-credentials` (key `POSTGRES_PASSWORD`), user `postgres`, database `knightgpt` — fixed names, not requiring live discovery (unlike the abandoned operator attempts). Every later task uses these exact names.

---

### Task 1: GitHub Actions workflow — build and push the Postgres+pgGraph image to GHCR

**Files:**
- Create: `.github/workflows/build-postgres-image.yml`

**Interfaces:**
- Produces: a container image at `ghcr.io/l1joseph/knightgpt-postgres:<tag>`, pullable by the NRP cluster, consumed by Task 2's `Cluster.spec.imageName`.

- [ ] **Step 1: Create the workflow file**

```yaml
# .github/workflows/build-postgres-image.yml
name: Build and push Postgres+pgGraph image

on:
  push:
    branches: [vllm]
    paths:
      - "docker/postgres/**"
      - ".github/workflows/build-postgres-image.yml"
  workflow_dispatch: {}

env:
  IMAGE_NAME: ghcr.io/${{ github.repository_owner }}/knightgpt-postgres

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.IMAGE_NAME }}
          tags: |
            type=sha,format=long
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/postgres/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

- [ ] **Step 2: Commit and push to trigger the workflow**

```bash
git add .github/workflows/build-postgres-image.yml
git commit -m "ci: build and push Postgres+pgGraph image to GHCR"
git push -u origin feature/nrp-storage-deployment
```

Note: the workflow triggers on push to `vllm`, not this feature branch — pushing the feature branch alone won't run it. Use `workflow_dispatch` (Step 3) to test it before this branch merges.

- [ ] **Step 3: Manually trigger the workflow to verify it works before merge**

```bash
gh workflow run build-postgres-image.yml --ref feature/nrp-storage-deployment
```

Wait for it to complete (`gh run watch` or check the Actions tab), then verify:

```bash
gh run list --workflow=build-postgres-image.yml --limit 1
```

Expected: the run succeeds (`completed`, `success`).

- [ ] **Step 4: Verify the image is pullable and check its visibility**

```bash
gh api /users/l1joseph/packages/container/knightgpt-postgres --jq '.visibility'
```

Expected: `public`. If it prints `private`, make it public (new GHCR packages default to the repository's visibility, but container packages sometimes default to private regardless — this needs to be public since Task 2's Postgres manifest has no `imagePullSecrets` configured, per the Global Constraints' "no external network exposure" scope keeping this plan simple):

```bash
gh api --method PATCH /user/packages/container/knightgpt-postgres -f visibility=public
```

Re-run the check above to confirm it now reports `public`.

- [ ] **Step 5: No test framework step here (this is a CI workflow, not application code) — verification is Steps 3-4 above. Commit is already done in Step 2.**

---

### Task 2: Self-managed Postgres Deployment for Postgres+pgGraph

> **Amendment (second):** this task tried CloudNativePG first (no RBAC access to its CRD in this namespace, confirmed via `kubectl auth can-i`), then Zalando (RBAC confirmed working, but its operator requires images built on a "Spilo" base bundling Patroni — our image is `FROM postgres:17-bookworm`, the vanilla upstream image, which CrashLoopBackOff'd under Zalando's management). Both attempts are abandoned. This version uses a plain Kubernetes `Deployment` + `PersistentVolumeClaim` + `Service` — no operator. Neither operator's core value (managed multi-replica failover) applies at `replicas: 1` anyway; see the design spec's twice-amended Decisions section for full reasoning. This is simpler than either operator attempt: the vanilla `postgres` image's own `POSTGRES_DB`/`POSTGRES_USER`/`POSTGRES_PASSWORD` env-var bootstrap mechanism applies directly, no operator-specific CRD schema or bootstrap hook to satisfy.

**Files:**
- Create: `k8s/nrp/postgres-cluster.yaml` (Deployment, PVC, Service)

**Interfaces:**
- Consumes: the GHCR image from Task 1 (`ghcr.io/l1joseph/knightgpt-postgres:latest`).
- Produces: a reachable Postgres service at `knightgpt-postgres.knightlab-ml.svc.cluster.local:5432`, database `knightgpt`, user `postgres` — consumed by Task 4 (schema application) and, in a future plan, the ingestion Job and MCP server. The credentials Secret is created imperatively (Step 1, not committed to git) — its name (`knightgpt-postgres-credentials`) is fixed and used consistently by every later task, unlike the operator attempts where names needed live discovery.

- [ ] **Step 1: Generate and create the credentials Secret (not committed to git — plaintext passwords don't belong in version control)**

```bash
POSTGRES_PASSWORD=$(openssl rand -base64 24)
kubectl create secret generic knightgpt-postgres-credentials \
  -n knightlab-ml \
  --from-literal=POSTGRES_PASSWORD="$POSTGRES_PASSWORD"
echo "Password generated and stored in the knightgpt-postgres-credentials Secret (not printed here to avoid leaking it into shell history/logs — retrieve it from the Secret when needed: kubectl get secret knightgpt-postgres-credentials -n knightlab-ml -o jsonpath='{.data.POSTGRES_PASSWORD}' | base64 -d)"
```

- [ ] **Step 2: Create the Deployment, PVC, and Service manifest**

```yaml
# k8s/nrp/postgres-cluster.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: knightgpt-postgres-data
  namespace: knightlab-ml
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: linstor-igrok
  resources:
    requests:
      storage: 20Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: knightgpt-postgres
  namespace: knightlab-ml
spec:
  replicas: 1
  strategy:
    type: Recreate  # RWO volume -- avoid two pods ever trying to mount it during a rolling update
  selector:
    matchLabels:
      app: knightgpt-postgres
  template:
    metadata:
      labels:
        app: knightgpt-postgres
    spec:
      containers:
        - name: postgres
          image: ghcr.io/l1joseph/knightgpt-postgres:latest
          imagePullPolicy: Always
          command: ["postgres", "-c", "shared_preload_libraries=graph"]
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_DB
              value: knightgpt
            - name: POSTGRES_USER
              value: postgres
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: knightgpt-postgres-credentials
                  key: POSTGRES_PASSWORD
            - name: PGDATA
              value: /var/lib/postgresql/data/pgdata  # subdirectory, not the mount root -- the vanilla postgres image's initdb refuses a non-empty mount root (e.g. a lost+found dir some storage backends create), this is the documented workaround
          volumeMounts:
            - name: postgres-data
              mountPath: /var/lib/postgresql/data
      volumes:
        - name: postgres-data
          persistentVolumeClaim:
            claimName: knightgpt-postgres-data
---
apiVersion: v1
kind: Service
metadata:
  name: knightgpt-postgres
  namespace: knightlab-ml
spec:
  selector:
    app: knightgpt-postgres
  ports:
    - port: 5432
      targetPort: 5432
```

Note on `imagePullPolicy: Always` and the mutable `latest` tag — same reasoning as both prior operator attempts: acceptable for now, a future follow-up could pin to an immutable SHA tag once this deployment is stable.

- [ ] **Step 3: Apply the manifest**

```bash
kubectl apply -f k8s/nrp/postgres-cluster.yaml
```

If this fails with a validation error, read the actual error message and adjust the manifest accordingly — don't force a workaround outside this task's scope (no pushing to shared branches, no modifying cluster-scoped resources, no requesting elevated permissions). If you can't resolve it from the error message alone, stop and report BLOCKED with the exact error.

- [ ] **Step 4: Wait for the pod to become ready**

```bash
kubectl get pods -n knightlab-ml -l app=knightgpt-postgres -w
```

Expected: reaches `1/1 Running` (may take a minute or two for image pull + initdb — much faster than either operator attempt since there's no operator reconciliation loop involved, just a direct Deployment). Press Ctrl-C once it does. If it doesn't stabilize within ~5 minutes, check:

```bash
kubectl describe pod -n knightlab-ml -l app=knightgpt-postgres
kubectl logs -n knightlab-ml -l app=knightgpt-postgres --tail=100
```

- [ ] **Step 5: Verify the `graph` extension loads, via port-forward**

```bash
kubectl port-forward -n knightlab-ml svc/knightgpt-postgres 5433:5432 &
PF_PID=$!
sleep 3
PGPASSWORD=$(kubectl get secret knightgpt-postgres-credentials -n knightlab-ml -o jsonpath='{.data.POSTGRES_PASSWORD}' | base64 -d) \
  psql -h localhost -p 5433 -U postgres -d knightgpt -c 'CREATE EXTENSION IF NOT EXISTS graph;'
PGPASSWORD=$(kubectl get secret knightgpt-postgres-credentials -n knightlab-ml -o jsonpath='{.data.POSTGRES_PASSWORD}' | base64 -d) \
  psql -h localhost -p 5433 -U postgres -d knightgpt -c '\dx'
kill $PF_PID
```

Note: like the vanilla-image approach this always was, there's no operator bootstrap-SQL hook — `CREATE EXTENSION IF NOT EXISTS graph;` is run directly here via psql as a one-time verification step. Task 4's `apply_schema.py` runs this same idempotent statement again as part of `sql/schema.sql`, so running it twice is safe.

Expected: the `\dx` output lists `graph` (and `plpgsql`, which ships by default). If `graph` is missing, `shared_preload_libraries=graph` may not have taken effect — check `kubectl logs` for the pod's startup output for any error mentioning the `graph` library.

- [ ] **Step 6: Commit**

```bash
git add k8s/nrp/postgres-cluster.yaml
git commit -m "feat(nrp): self-managed Postgres Deployment+PVC+Service for Postgres+pgGraph"
```

---

### Task 3: DuckDB PVC and verification

**Files:**
- Create: `k8s/nrp/duckdb-pvc.yaml`
- Create: `k8s/nrp/duckdb-verify-pod.yaml` (throwaway verification pod, not meant to persist long-term — delete after use)

**Interfaces:**
- Consumes: nothing from Task 1/2 (independent of the Postgres piece).
- Produces: a PVC named `knightgpt-duckdb` in `knightlab-ml`, mountable RWX — consumed by a future ingestion Job and MCP server plan.

- [ ] **Step 1: Create the PVC manifest**

```yaml
# k8s/nrp/duckdb-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: knightgpt-duckdb
  namespace: knightlab-ml
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: rook-cephfs
  resources:
    requests:
      storage: 20Gi
```

- [ ] **Step 2: Apply and verify it binds**

```bash
kubectl apply -f k8s/nrp/duckdb-pvc.yaml
kubectl get pvc knightgpt-duckdb -n knightlab-ml -w
```

Expected: `STATUS` reaches `Bound`. Press Ctrl-C once it does.

- [ ] **Step 3: Create a throwaway verification pod**

```yaml
# k8s/nrp/duckdb-verify-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: duckdb-verify
  namespace: knightlab-ml
spec:
  restartPolicy: Never
  containers:
    - name: verify
      image: python:3.11-slim
      command: ["sleep", "600"]
      volumeMounts:
        - name: duckdb-data
          mountPath: /data
  volumes:
    - name: duckdb-data
      persistentVolumeClaim:
        claimName: knightgpt-duckdb
```

```bash
kubectl apply -f k8s/nrp/duckdb-verify-pod.yaml
kubectl wait --for=condition=Ready pod/duckdb-verify -n knightlab-ml --timeout=120s
```

- [ ] **Step 4: Verify DuckDB + vss work on this PVC**

```bash
kubectl exec -n knightlab-ml duckdb-verify -- pip install --quiet duckdb
kubectl exec -n knightlab-ml duckdb-verify -- python3 -c "
import duckdb
con = duckdb.connect('/data/test.duckdb')
con.execute('INSTALL vss')
con.execute('LOAD vss')
con.execute('CREATE TABLE t (id VARCHAR, embedding FLOAT[4])')
con.execute(\"INSERT INTO t VALUES ('a', [1.0, 0.0, 0.0, 0.0])\")
print(con.execute('SELECT * FROM t').fetchall())
con.close()
print('OK')
"
```

Expected: prints the inserted row and `OK`, confirming CephFS's POSIX semantics work correctly with DuckDB's file locking (`INSTALL vss`/`LOAD vss` also confirms this pod has network access to fetch the extension, same as verified on Cosmos earlier).

- [ ] **Step 5: Verify the single-writer lock-conflict error carries over correctly**

Run a long-lived read-write connection in the background, then attempt a second one while the first still holds the file:

```bash
kubectl exec -n knightlab-ml duckdb-verify -- python3 -c "
import duckdb
import time
con = duckdb.connect('/data/test.duckdb')
print('holding connection...')
time.sleep(20)
" &
sleep 3
kubectl exec -n knightlab-ml duckdb-verify -- python3 -c "
import duckdb
try:
    con2 = duckdb.connect('/data/test.duckdb')
    print('UNEXPECTED: second connection succeeded')
except duckdb.IOException as e:
    print(f'EXPECTED lock conflict: {e}')
"
wait
```

Expected: the second connection attempt prints `EXPECTED lock conflict: ...` (a `duckdb.IOException` mentioning the file is locked), confirming this repo's `DuckDBStore.__init__`'s existing lock-conflict error handling (from the earlier DuckDB redesign work — it catches exactly `duckdb.IOException` and re-raises a clear `RuntimeError`) will correctly trigger under NRP's CephFS-backed RWX volume the same way it did in local/Cosmos testing.

- [ ] **Step 6: Clean up the verification pod**

```bash
kubectl delete pod duckdb-verify -n knightlab-ml
kubectl exec -n knightlab-ml duckdb-verify -- rm -f /data/test.duckdb 2>/dev/null || true
```

(The `rm` will fail harmlessly since the pod is already gone by the time it'd run — if cleanup of the test file itself matters, delete it before deleting the pod, by inserting `kubectl exec -n knightlab-ml duckdb-verify -- rm -f /data/test.duckdb` as its own step before `kubectl delete pod`.)

Do NOT delete `k8s/nrp/duckdb-verify-pod.yaml` from the repo — keep it as a reusable verification tool for future debugging, just don't leave the actual pod running in the cluster.

- [ ] **Step 7: Commit**

```bash
git add k8s/nrp/duckdb-pvc.yaml k8s/nrp/duckdb-verify-pod.yaml
git commit -m "feat(nrp): DuckDB RWX PVC and verification pod"
```

---

### Task 4: Apply the application schema to the new NRP Postgres instance

**Files:**
- Modify: none (reuses existing `scripts/apply_schema.py` and `sql/schema.sql` as-is)

**Interfaces:**
- Consumes: the Task 2 Postgres service (`knightgpt-postgres.knightlab-ml.svc.cluster.local`, credentials in the `knightgpt-postgres-credentials` Secret, user `postgres`), reached via port-forward for this one-time application.
- Produces: `papers`, `chunks`, `chunk_edges` tables and pgGraph registration in the `knightgpt` database on NRP — the actual schema state a future ingestion Job and MCP server will read/write.

- [ ] **Step 1: Port-forward to the new Postgres instance**

```bash
kubectl port-forward -n knightlab-ml svc/knightgpt-postgres 5433:5432 &
PF_PID=$!
sleep 3
```

- [ ] **Step 2: Get the connection password and build a DSN**

```bash
PGPASSWORD=$(kubectl get secret knightgpt-postgres-credentials -n knightlab-ml -o jsonpath='{.data.POSTGRES_PASSWORD}' | base64 -d)
DSN="postgresql://postgres:${PGPASSWORD}@localhost:5433/knightgpt"
```

- [ ] **Step 3: Run apply_schema.py against it**

```bash
cd /cosmos/nfs/home/l1joseph/knightGPT
source ~/miniforge3/etc/profile.d/conda.sh && conda activate knightGPT
python scripts/apply_schema.py --dsn "$DSN"
```

Expected output ends with: `pgGraph registration verified: chunks table and similar_to edge present` (this is `apply_schema.py`'s own built-in verification, already implemented and tested in earlier work this session — no new test code needed here, this step exercises existing, trusted code against a new target).

- [ ] **Step 4: Verify the tables exist independently, via psql**

```bash
PGPASSWORD="$PGPASSWORD" psql -h localhost -p 5433 -U postgres -d knightgpt -c '\dt'
```

Expected: lists `papers`, `chunks`, `chunk_edges`.

- [ ] **Step 5: Stop the port-forward**

```bash
kill $PF_PID
```

- [ ] **Step 6: No commit needed — this task applies existing, already-committed code against live infrastructure and doesn't change any files.**

---

### Task 5: Investigate and document backup strategy

**Files:**
- Create: `docs/superpowers/specs/2026-08-03-nrp-storage-deployment-design.md` — Modify (append a "Backup: resolved" note, since this task resolves the spec's open question)
- Create (conditionally, only if snapshots are unavailable): `k8s/nrp/backup-cronjob.yaml`

**Interfaces:**
- Consumes: the Task 2 Postgres Cluster and Task 3 DuckDB PVC.
- Produces: either a working `VolumeSnapshotClass`-based snapshot mechanism (documented, not necessarily automated in this plan), or a CronJob manifest, or a documented decision to defer entirely — this task's job is to resolve the open question from the spec, not to guarantee a specific mechanism exists.

- [ ] **Step 1: Check whether VolumeSnapshotClass is usable in this namespace**

```bash
kubectl get volumesnapshotclass 2>&1
```

If this returns a list (not `Forbidden`), note the snapshot classes available for `rook-system.rbd.csi.ceph.com` (for the Postgres PVC, `linstor-igrok`'s underlying driver is `linstor.csi.linbit.com` — check if a matching snapshot class exists for that driver too) and `rook-system.cephfs.csi.ceph.com` (for the DuckDB PVC) drivers.

If it returns `Forbidden` (as it did during brainstorming for this permission level) or an empty list with no matching driver, snapshots aren't usable here — proceed to Step 3 (fallback).

- [ ] **Step 2: If snapshots ARE available, create a one-off test snapshot to confirm it actually works end-to-end**

```bash
cat <<'EOF' | kubectl apply -f -
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: knightgpt-duckdb-test-snapshot
  namespace: knightlab-ml
spec:
  volumeSnapshotClassName: <the class name found in Step 1>
  source:
    persistentVolumeClaimName: knightgpt-duckdb
EOF
kubectl get volumesnapshot knightgpt-duckdb-test-snapshot -n knightlab-ml -w
```

Expected: `READYTOUSE` becomes `true` within a few minutes. If it does, document this working pattern in the spec (Step 4 below) and skip the CronJob fallback (Step 3) — periodic snapshotting can be automated later as a lightweight follow-up (e.g. a scheduled `VolumeSnapshot` creation via a simple CronJob calling `kubectl`), not required as part of this plan per the spec's "basic" backup rigor decision. Delete the test snapshot after confirming: `kubectl delete volumesnapshot knightgpt-duckdb-test-snapshot -n knightlab-ml`.

- [ ] **Step 3: If snapshots are NOT available, create the fallback CronJob manifest**

```yaml
# k8s/nrp/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: knightgpt-backup
  namespace: knightlab-ml
spec:
  schedule: "0 8 * * 0"  # weekly, Sunday 08:00 UTC
  jobTemplate:
    spec:
      backoffLimit: 2
      template:
        spec:
          restartPolicy: Never
          containers:
            - name: backup
              image: ghcr.io/l1joseph/knightgpt-postgres:latest
              command:
                - /bin/bash
                - -c
                - |
                  set -euo pipefail
                  DATE=$(date +%Y%m%d)
                  pg_dump -h knightgpt-postgres -U postgres -d knightgpt \
                    -f /backup/knightgpt-postgres-${DATE}.sql
                  cp /duckdb-data/embeddings.duckdb /backup/knightgpt-embeddings-${DATE}.duckdb
              env:
                - name: PGPASSWORD
                  valueFrom:
                    secretKeyRef:
                      name: knightgpt-postgres-credentials
                      key: POSTGRES_PASSWORD
              volumeMounts:
                - name: duckdb-data
                  mountPath: /duckdb-data
                  readOnly: true
                - name: backup-dest
                  mountPath: /backup
          volumes:
            - name: duckdb-data
              persistentVolumeClaim:
                claimName: knightgpt-duckdb
            - name: backup-dest
              persistentVolumeClaim:
                claimName: knightgpt-backup-dest
```

This assumes a separate `knightgpt-backup-dest` PVC (not Cosmos scratch directly — reaching `/cosmos/vast/scratch/...` from an NRP pod requires network connectivity between the two clusters that hasn't been confirmed to exist; the spec flagged this as a real open question, and this plan resolves it by keeping the backup destination inside NRP itself rather than assuming cross-cluster reachability). Add the destination PVC:

```yaml
# append to k8s/nrp/backup-cronjob.yaml
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: knightgpt-backup-dest
  namespace: knightlab-ml
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: rook-ceph-block
  resources:
    requests:
      storage: 20Gi
```

Apply and verify the CronJob is scheduled correctly (don't wait for the actual weekly trigger — trigger a manual run to test it):

```bash
kubectl apply -f k8s/nrp/backup-cronjob.yaml
kubectl create job --from=cronjob/knightgpt-backup knightgpt-backup-test -n knightlab-ml
kubectl wait --for=condition=complete job/knightgpt-backup-test -n knightlab-ml --timeout=300s
kubectl logs -n knightlab-ml job/knightgpt-backup-test
kubectl delete job knightgpt-backup-test -n knightlab-ml
```

Expected: the manual test job completes successfully with no errors in its logs.

- [ ] **Step 4: Update the design spec with the resolved decision**

Add a short section to the end of `docs/superpowers/specs/2026-08-03-nrp-storage-deployment-design.md`:

```markdown
## Backup: Resolved

<Either:>
CSI volume snapshots ARE available in this namespace (VolumeSnapshotClass `<name>` for the relevant CSI drivers), confirmed via a working end-to-end test snapshot. Automated periodic snapshotting is a follow-up, not implemented in this plan.

<Or:>
CSI volume snapshots are NOT available in this namespace (VolumeSnapshotClass listing returns Forbidden/empty for the relevant drivers). Implemented the documented fallback instead: a weekly CronJob (`k8s/nrp/backup-cronjob.yaml`) running `pg_dump` + a DuckDB file copy to a separate in-cluster PVC (`knightgpt-backup-dest`), verified via a manual test run.
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-08-03-nrp-storage-deployment-design.md
# Plus, if the fallback path was taken:
git add k8s/nrp/backup-cronjob.yaml
git commit -m "docs(nrp): resolve backup strategy decision, implement fallback if needed"
```

---

## Final Verification

After all 5 tasks, confirm the full storage tier is ready for the next plan (ingestion Job) to build on:

```bash
kubectl get pods -n knightlab-ml -l app=knightgpt-postgres
kubectl get pvc knightgpt-postgres-data knightgpt-duckdb -n knightlab-ml
```

Expected: Postgres cluster healthy, DuckDB PVC bound, schema applied (from Task 4), backup strategy resolved one way or the other (from Task 5). No pods left over from verification steps (Task 3's `duckdb-verify` pod deleted, Task 5's test snapshot/job deleted).
