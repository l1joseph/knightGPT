# Restoring knightGPT Postgres from a `knightgpt-backup` CronJob dump

This procedure was tested end-to-end on 2026-08-05 against a real backup produced by
`k8s/nrp/backup-cronjob.yaml` and a disposable throwaway Postgres+pgGraph instance (not the
live `knightgpt-postgres` Deployment). It confirms and works around a real, previously
unverified risk in pgGraph's registration tables.

## The risk: stale OIDs in pgGraph's registration tables

pgGraph tracks which tables/columns participate in the graph via internal tables
`graph._registered_tables` and `graph._registered_edges`, which pgGraph marks as extension
config tables — so `pg_dump` **does** include their row data (not just the
`CREATE EXTENSION graph` statement). Those rows store the **hard Postgres OID** of the
registered tables (`table_oid`, `from_table_oid`, `to_table_oid`), captured at
registration time on the source database.

A restore into a fresh database (`CREATE TABLE public.chunks ...` etc., run in dump order
after `CREATE EXTENSION graph`) assigns **brand-new OIDs** to the recreated `chunks` and
`chunk_edges` tables. The restored `graph._registered_tables`/`_registered_edges` rows still
reference the **old, source-database OIDs**, which point to nothing in the new database.

**Confirmed: pgGraph resolves registrations by OID, not by table name, and a naive restore
breaks it.** Evidence from the test below: after a plain `psql < backup.sql` restore,
`graph.registered_tables()`/`graph.registered_edges()` (the friendly view functions) looked
completely normal —

```
 table_name | id_columns |    columns     | tenant_column
------------+------------+----------------+---------------
 chunks     | {id}       | {text,section} |
```

— but `graph.expand()` on a real row failed outright:

```
$ psql -c "SELECT * FROM graph.expand('public.chunks'::regclass, 'c1', 1);"
ERROR:  Internal error: registered table relation no longer exists; re-register it
DETAIL:  pgGraph diagnostic: PG000
HINT:  This is a bug. Please report it with the full error message.
```

So `registered_tables()`/`registered_edges()` alone are **not sufficient** to verify a
restore is usable — they don't check OID liveness. Only an actual `graph.expand()` call
surfaced the break.

Direct OID comparison confirmed the mechanism:

```
-- source DB (live knightgpt-postgres), from the dump:
--   graph._registered_tables: chunks table_oid = 16842
--   graph._registered_edges:  chunk_edges table_oid = 16854, chunks table_oid = 16842

-- restored DB, actual current OIDs after CREATE TABLE:
SELECT 'chunks'::regclass::oid, 'chunk_edges'::regclass::oid;
 chunks_oid | chunk_edges_oid
------------+-----------------
      16834 |           16829

-- restored DB, what's still stored in the registration tables (unchanged by the restore):
SELECT table_name, table_oid FROM graph._registered_tables;
  table_name   | table_oid
---------------+-----------
 public.chunks |     16842   <- stale, points at the OLD database's OID, not this one
```

## The working procedure

A plain restore (`psql < backup.sql`) is safe for the actual data (`papers`, `chunks`,
`chunk_edges` restore cleanly with no errors) but leaves pgGraph's graph functionality
broken until you **re-run the registration step**. `graph.add_table()`/`graph.add_edge()`
turn out to be idempotent-by-name (upsert), not insert-only: calling them again with the
same table/column arguments updates the stored OID to whatever the table's *current* OID
actually is, rather than erroring as a duplicate.

```bash
# 1. Restore the plain-SQL dump as-is. No special flags needed -- this is the same
#    dump the CronJob already produces, no changes to backup-cronjob.yaml required.
PGPASSWORD=<postgres password> psql -h <target host> -U postgres -d knightgpt \
    -f knightgpt-postgres-<DATE>.sql

# 2. Re-run pgGraph's registration step -- this is exactly what sql/schema.sql's
#    DO $$ ... graph.add_table(...) / graph.add_edge(...) ... $$; blocks do, and exactly
#    what scripts/apply_schema.py already runs (it re-applies sql/schema.sql in full and
#    then verifies registration). Running it again against a database that already has
#    the graph.* internal tables from step 1's restore is safe -- it does NOT error as a
#    duplicate, it fixes up the stale OIDs in place.
python scripts/apply_schema.py --dsn "postgresql://postgres:<password>@<target host>:5432/knightgpt"

# 3. Build the graph projection. This is a normal step after any registration change
#    (the same call src/graph/postgres_builder.py and scripts/migrate_to_postgres.py
#    already make after ingestion), not something restore-specific -- but it's easy to
#    forget after a restore, since a projection built against the old OIDs is stale too.
psql -h <target host> -U postgres -d knightgpt -c "SELECT graph.build();"

# 4. Verify with a real graph.expand() call, not just registered_tables()/registered_edges()
#    -- those two return name-based views that look fine even when the underlying OIDs are
#    stale (see "The risk" above). Pick any real chunk id:
psql -h <target host> -U postgres -d knightgpt -c \
    "SELECT * FROM graph.expand('public.chunks'::regclass, '<some real chunk id>', 1);"
```

If step 4 returns rows (or, for a chunk with no edges, an empty result *without an error*),
the restore is good. If it raises `registered table relation no longer exists`, step 2 was
skipped or failed.

## What was actually tested (2026-08-05)

1. Triggered a real backup: `kubectl create job --from=cronjob/knightgpt-backup
   knightgpt-backup-restore-test -n knightlab-ml`, waited for `Complete`.
2. Retrieved the dump via a throwaway debug pod mounting `knightgpt-backup-dest`
   (same pattern as `k8s/nrp/duckdb-verify-pod.yaml`) and `kubectl cp`.
3. Stood up a disposable Postgres+pgGraph pod (`pg-restore-test`, same image as the live
   Deployment, `emptyDir` storage — never touched the live `knightgpt-postgres` Deployment
   or its PVC).
4. Ran `psql -f backup.sql` against it — succeeded with no errors (`CREATE TABLE`,
   `COPY N`, etc., all clean).
5. Confirmed the live source database's `chunks`/`chunk_edges` tables were actually empty
   (0 rows -- no ingestion Job has run yet in this deployment), so inserted 1 test paper, 2
   test chunks, and 1 test edge directly into the *restored* throwaway database to have
   real rows to call `graph.expand()` against.
6. Reproduced the OID break exactly as described above (`graph.expand()` failed).
7. Applied the fix (re-ran the `graph.add_table()`/`graph.add_edge()` calls from
   `sql/schema.sql`), confirmed the stored OIDs updated to match the restored table's
   actual current OIDs, ran `graph.build()`, and confirmed `graph.expand()` then succeeded
   and returned the expected traversal:
   ```
    root_id | node_id | depth |            readable_path
   ---------+---------+-------+-------------------------------------
    c1      | c2      |     1 | chunks:c1 --similar_to--> chunks:c2
   ```
8. Cleaned up all throwaway resources: `pg-restore-test` pod, `backup-debug` pod, and the
   test Job(s) created from the CronJob. Nothing from this test was left running in the
   cluster. The live `knightgpt-postgres` Deployment/PVC/data was never touched.

## Notes / caveats

- This was tested against an **empty** source `chunks`/`chunk_edges` (no ingestion has run
  yet in this deployment). The OID mechanism tested here is structural (how pgGraph stores
  and resolves table identity) and does not depend on row count, so the finding generalizes
  to a populated database — but re-run step 4's verification after the first real restore
  of a populated database as a sanity check, since this test used only synthetic rows.
- `graph.add_table()`/`graph.add_edge()` being upsert-by-name (rather than insert-only) is
  itself pgGraph's own behavior, not something this repo controls. If a future pgGraph
  version changes that (e.g. treats a second call as an error), step 2 above would need
  `graph.remove_table()`/`graph.remove_edge()` first — verify this still holds when
  `docker/postgres/Dockerfile`'s `PGGRAPH_REF` is next bumped.
- No changes to `k8s/nrp/backup-cronjob.yaml`'s `pg_dump` command were needed or made — the
  existing plain-SQL dump restores fine once step 2 (re-registration) is included in the
  runbook.
