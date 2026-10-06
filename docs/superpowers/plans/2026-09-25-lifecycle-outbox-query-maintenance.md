# Lifecycle Outbox Query and Maintenance Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Extract lifecycle outbox reads and housekeeping from `lifecycle_outbox.py` while preserving its durable state, security, and operator contracts.

**Architecture:** Keep `_LifecycleOutboxRepository` as the narrow composition service that opens and secures connections, validates schema, owns transaction boundaries, and maps typed results to existing callers. Move read-only status/snapshot query and projection logic to `lifecycle_outbox_queries.py`; move bounded pruning and opportunistic housekeeping to `lifecycle_outbox_maintenance.py`. Do not add internal forwarding modules or per-operation repositories.

**Tech Stack:** Python 3.11, `sqlite3`, dataclasses, `unittest`, existing lifecycle/outbox types.

**Spec:** `docs/superpowers/specs/2026-09-25-lifecycle-outbox-query-maintenance-design.md`

## Global Constraints

- `status` remains read-only and does not create, initialize, migrate, or repair outbox state.
- Snapshot reads remain complete and deterministic; a poison row rejects the whole snapshot.
- Maintenance stays bounded and deletes only acknowledged rows older than the retention boundary.
- SQLite transactions stay inside repository operations; no Taskwarrior command runs while a transaction is active.
- Busy, corrupt-schema, malformed-row, filesystem-security, and interrupted-operation outcomes retain their current typed result behavior.
- Existing command output and lifecycle semantics remain unchanged.
- Query and projection results use typed intermediate results; the repository maps them to the existing operator payload.
- Schema, codec, claim/lease policy, and transition policy remain in their current owner modules.
- Do not add compatibility forwarding modules for removed internal paths.

## Review Focus

- Missing database: status succeeds without creating `.nautical-state`; pin in Task 1.
- Unsupported schema or invalid schema: status and snapshot return their current rejected outcomes; pin in Task 1.
- Poison lifecycle row or integrity envelope: status reports the row as poison, while a complete snapshot rejects; pin in Task 1.
- Busy or locked SQLite database: callers receive retryable results with lock status preserved; pin in Task 1.
- Retention, cooldown, and size boundaries: maintenance removes only eligible acknowledged rows, honors bounds, and leaves all other states untouched; pin in Task 1.

---

### Task 1: Characterize read and maintenance contracts

**Files:**
- Modify: `tests/test_lifecycle_outbox_contract.py`

**Interfaces:**
- Consumes: existing `LifecycleOutboxRepository`, `OutboxResultKind`, `OutboxProcessingState`, and `OutboxMaintenanceResult`.
- Produces: regression tests that pin current status, snapshot, prune, and housekeeping behavior before extraction.

- [ ] **Step 1: Add status filtering and ordering characterization**

Add a test that enqueues records in ready, retry, claimed, quarantined, and manual-review states with controlled `updated_at` values. Assert `status(limit=2)` returns the two highest-priority rows in stable `updated_at`, then `intent_id`, order; assert `status(intent_id=...)` returns only that record.

Use the existing repository fixture and pin the public projection directly:

```python
result, payload = repository.status(limit=2)
self.assertTrue(result.ok)
self.assertEqual(
    [row["state"] for row in payload["records"]],
    ["manual_review", "quarantined"],
)
```

- [ ] **Step 2: Add deterministic snapshot and poison-row characterization**

Add tests that insert two valid rows in reverse identity order and assert `snapshot_records()` returns ascending `intent_id`. Corrupt one lifecycle row's encoded plan and one integrity envelope in separate cases; assert each poison row rejects the entire snapshot with no partial records. Keep the existing missing-state non-mutation assertion.

- [ ] **Step 3: Add housekeeping boundary characterization**

With a fixed repository clock, test cooldown skip, no-work skip, size-triggered bounded cleanup, and checkpoint reporting. Assert retry, claimed, manual-review, quarantined, and ready records remain after pruning and housekeeping.

- [ ] **Step 4: Pin schema and busy-read failures**

Create an outbox, change `PRAGMA user_version` to a newer unsupported value, and assert both `status()` and `snapshot_records()` return rejected results. For a valid existing outbox, patch `nautical_core.lifecycle_outbox.sqlite3.connect` to raise `sqlite3.OperationalError("database is locked")`; assert both read methods return retryable results and set `lock_busy=True`. These repository-contract cases stay together in `tests/test_lifecycle_outbox_contract.py`; separate failure-boundary files are unnecessary for this behavior.

- [ ] **Step 5: Run the focused tests before extraction**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_lifecycle_outbox_contract tests.test_lifecycle_failure_injection tests.test_structured_failure_boundaries -q`

Expected: PASS against the current repository; any failure indicates a characterization test that does not match current behavior and must be corrected before extraction.

- [ ] **Step 6: Commit the characterization tests**

Run: `git add tests/test_lifecycle_outbox_contract.py tests/test_lifecycle_failure_injection.py tests/test_structured_failure_boundaries.py && git commit -m "test: characterize outbox reads and housekeeping"`

Expected: the commit contains only behavior-preserving regression tests.

### Task 2: Extract typed read/query and projection ownership

**Files:**
- Create: `nautical_core/lifecycle_outbox_queries.py`
- Modify: `nautical_core/lifecycle_outbox.py`
- Modify: `tests/test_lifecycle_outbox_contract.py`
- Create: `tests/test_lifecycle_outbox_queries.py`

**Interfaces:**
- Consumes: an already-open read-only `sqlite3.Connection`, `now`, query limits, and the repository's row decoder.
- Produces: frozen `OutboxStatusSummary` fields `schema_version: int`, `integrity: str`, `states: Mapping[str, int]`, `stale_claims: int`, `max_attempts: int`, `retention_seconds: float`, `acknowledged: int`, `eligible: int`, `oldest_age_s: int`, and `records: tuple[OutboxStatusRecordSummary, ...]`; frozen typed failure, plan, and record summaries; `status_summary(...)`; and generic `snapshot_rows(...)`. Repository entry points continue returning current `OutboxResult` contracts and the existing status payload.

- [ ] **Step 1: Add typed query projection tests**

Test `OutboxStatusSummary` field values, stable record order, row poison projection, and deterministic snapshot order using temporary SQLite databases. Assert the query functions do not open or close the connection passed by the caller.

Pin a typed projection with a disposable connection:

```python
view = status_summary(
    connection,
    now=100.0,
    limit=10,
    stale_after=30.0,
    retention_seconds=60.0,
    intent_id=None,
    decode_row=decode_row,
)
self.assertEqual(view.stale_claims, 1)
self.assertEqual(view.records[0].intent_id, expected_intent_id)
```

- [ ] **Step 2: Run the focused query tests and confirm failure**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_lifecycle_outbox_queries -q`

Expected: FAIL because the new module and typed result classes do not yet exist.

- [ ] **Step 3: Implement query result types and connection-scoped functions**

Add frozen, slotted `OutboxStatusSummary`, `OutboxStatusRecordSummary`, `OutboxStatusFailureSummary`, and `OutboxStatusPlanSummary` dataclasses. Record summaries expose `intent_id`, `state`, `stage`, `attempts`, `lease_expires_at`, `lease_age_s`, `failure`, and `plan`. Failure summaries expose `code` and `message`. Plan summaries expose `schema_version`, `action`, `event`, `chainID`, `parent_uuid`, `source_link`, `target_link`, `parent_guard`, and `child_uuid`. Implement `status_summary(connection: sqlite3.Connection, *, now: float, limit: int, stale_after: float, retention_seconds: float, intent_id: str | None, decode_row: Callable[[sqlite3.Row], StatusLifecycleRecord]) -> OutboxStatusSummary`. `StatusLifecycleRecord` is a Protocol for the corresponding typed record fields, including the plan identity and parent guard; `status_summary` decodes and projects records through this interface. Implement `snapshot_rows(connection: sqlite3.Connection, *, decode_lifecycle_row: Callable[[sqlite3.Row], T], decode_integrity_row: Callable[[sqlite3.Row], U]) -> tuple[T | U, ...]` with `T` and `U` module-level TypeVars. Keep schema validation and connection lifecycle in `_LifecycleOutboxRepository`; do not initialize or mutate through these read functions.

The summary type is immutable and contains only query output:

```python
@dataclass(frozen=True, slots=True)
class OutboxStatusSummary:
    schema_version: int
    integrity: str
    states: Mapping[str, int]
    stale_claims: int
    max_attempts: int
    retention_seconds: float
    acknowledged: int
    eligible: int
    oldest_age_s: int
    records: tuple[OutboxStatusRecordSummary, ...]
```

- [ ] **Step 4: Delegate repository reads to the query owner**

Replace the SQL and record projection bodies in `_LifecycleOutboxRepository.status()` and `.snapshot_records()` with calls to the new query functions. Keep current error-to-`OutboxResult` mapping and operator payload shape at the composition boundary. Remove the old SQL/projection code in the same edit.

- [ ] **Step 5: Run read-side contract tests**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_lifecycle_outbox_queries tests.test_lifecycle_outbox_contract tests.test_lifecycle_read_service tests.test_queue_review tests.test_queue_status_budget -q`

Expected: PASS, including read-only missing-state and poison-row behavior.

- [ ] **Step 6: Commit the query extraction**

Run: `git add nautical_core/lifecycle_outbox_queries.py nautical_core/lifecycle_outbox.py tests/test_lifecycle_outbox_queries.py tests/test_lifecycle_outbox_contract.py && git commit -m "refactor: extract outbox read queries"`

Expected: the commit contains the query owner, composition delegation, and direct query contracts.

### Task 3: Extract pruning and opportunistic housekeeping

**Files:**
- Create: `nautical_core/lifecycle_outbox_maintenance.py`
- Create: `tests/test_lifecycle_outbox_maintenance.py`
- Modify: `nautical_core/lifecycle_outbox.py`
- Modify: `tests/test_lifecycle_outbox_contract.py`
- Modify: `tests/test_structured_failure_boundaries.py`

**Interfaces:**
- Consumes: an open repository-owned `sqlite3.Connection`, the repository transaction context, current time, retention/cooldown/size/row limits, and database size.
- Produces: `prune_acknowledged_rows(connection: sqlite3.Connection, *, cutoff: float, limit: int, transaction: TransactionFactory) -> int` and `housekeeping_rows(connection: sqlite3.Connection, *, now: float, cutoff: float, interval_seconds: float, size_threshold_bytes: int, limit: int, checkpoint: bool, database_size: Callable[[], int], transaction: TransactionFactory) -> HousekeepingOutcome`; the repository retains validation, filesystem setup/security, exception classification, and `OutboxMaintenanceResult` creation.

- [ ] **Step 1: Add unit tests for bounded SQL maintenance**

Test eligible acknowledged-row selection and deletion at exact retention cutoffs, deterministic row ordering, row limits, cooldown, no-work, and checkpoint decisions. Assert non-acknowledged rows are never deleted.

Pin the retention boundary through the repository contract:

```python
result = repository.prune_acknowledged(retention_seconds=10.0, limit=1)
self.assertTrue(result.ok)
self.assertEqual(result.removed, 1)
self.assertEqual(repository.status()[1]["states"]["manual_review"], 1)
```

- [ ] **Step 2: Run maintenance tests and confirm failure**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_lifecycle_outbox_maintenance -q`

Expected: FAIL because the maintenance owner and its typed outcome do not exist.

- [ ] **Step 3: Implement connection-scoped maintenance operations**

Define `TransactionFactory` as `Callable[[sqlite3.Connection], AbstractContextManager[None]]` and `HousekeepingOutcome` as a frozen, slotted dataclass with `removed: int`, `skipped: bool`, `reason: str`, and `checkpoint: str`. Move only SQL selection, deletion, cooldown persistence, and checkpoint decision logic to the maintenance module. Keep connection opening, schema initialization, secure file validation, clock sampling, argument validation, and exception-to-result mapping in the repository.

- [ ] **Step 4: Delegate repository maintenance methods**

Replace SQL bodies in `prune_acknowledged()` and `opportunistic_housekeeping()` with calls to the maintenance module. Preserve current defaults, return types, and error classifications. Delete the original SQL implementation as part of this change.

- [ ] **Step 5: Run focused maintenance and failure tests**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_lifecycle_outbox_maintenance tests.test_lifecycle_outbox_contract tests.test_structured_failure_boundaries -q`

Expected: PASS for retention, cooldown, storage busy, filesystem-security failure, and bounded cleanup cases.

- [ ] **Step 6: Commit the maintenance extraction**

Run: `git add nautical_core/lifecycle_outbox_maintenance.py nautical_core/lifecycle_outbox.py tests/test_lifecycle_outbox_maintenance.py tests/test_lifecycle_outbox_contract.py tests/test_structured_failure_boundaries.py && git commit -m "refactor: extract outbox housekeeping"`

Expected: the commit contains maintenance SQL ownership, repository delegation, and boundary tests.

### Task 4: Verify composition boundaries and full behavior

**Files:**
- Modify: `tests/test_architecture_contract.py`
- Modify: `tests/test_lifecycle_outbox_contract.py`
- Modify: `docs/superpowers/specs/2026-09-25-lifecycle-outbox-query-maintenance-design.md` only if implementation reveals a necessary design correction.

**Interfaces:**
- Consumes: the extracted query and maintenance modules and repository composition methods.
- Produces: architecture checks preventing duplicated SQL owners, plus final verification evidence for the work package.

- [ ] **Step 1: Add owner-boundary architecture assertions**

Assert query SQL is owned by `lifecycle_outbox_queries.py`, maintenance SQL by `lifecycle_outbox_maintenance.py`, and connection security/transaction setup remains in `lifecycle_outbox.py`. Spy on the focused owners to prove the repository composition methods delegate once and preserve their established result contracts.

- [ ] **Step 2: Run focused architecture and outbox tests**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_architecture_contract tests.test_lifecycle_outbox_queries tests.test_lifecycle_outbox_maintenance tests.test_lifecycle_outbox_contract tests.test_lifecycle_failure_injection tests.test_lifecycle_execution_capabilities tests.test_lifecycle_terminal_plans tests.test_reconcile_snapshot_budget tests.test_queue_review tests.test_queue_status_budget -q`

Expected: PASS.

- [ ] **Step 3: Run full unit, golden, and static checks**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest discover -s tests -q`

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 dev_tools/nautical_golden_tests.py`

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 dev_tools/nautical_golden_tests.py --shuffle-seed 20260925`

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m mypy --config-file mypy.ini nautical_core`

- [ ] **Step 4: Measure full-suite branch coverage against the baseline**

Run:

```bash
coverage_root=$(mktemp -d /tmp/nautical-outbox-coverage.XXXXXX)
COVERAGE_FILE="$coverage_root/parent" \
NAUTICAL_SUBPROCESS_COVERAGE_DIR="$coverage_root/children" \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 -m coverage run -m unittest discover -s tests -q
COVERAGE_FILE="$coverage_root/parent" \
  python3 -m coverage combine --append "$coverage_root/children"
COVERAGE_FILE="$coverage_root/parent" \
  python3 -m coverage report --fail-under=67 --skip-empty
```

Expected: all tests and mypy pass, branch coverage remains at least the measured 67% baseline, and no performance or output contract changes appear.

- [ ] **Step 5: Update local checklist evidence**

Record the new module ownership, focused/full verification results, and any remaining WP3 items in the local-only checklist. Do not stage or commit that checklist.

- [ ] **Step 6: Commit architecture boundary assertions**

Run: `git add tests/test_architecture_contract.py tests/test_lifecycle_outbox_contract.py && git commit -m "test: enforce outbox ownership boundaries"`

Expected: only the owner-contract tests enter this commit; local checklist evidence remains unstaged.
