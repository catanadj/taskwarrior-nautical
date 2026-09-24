# WP4 Lifecycle Application and Outbox Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make lifecycle persistence, recovery policy, and application orchestration independently testable while preserving the current SQLite schema, outbox API, mutation order, leases, retries, and fail-closed behavior.

**Architecture:** Keep `LifecycleOutboxRepository` as the compatibility façade, but move pure serialization/state-transition decisions and schema metadata behind focused collaborators. Keep `LifecycleApplicationService` responsible for orchestration only; move retry/manual-review decisions and batch accounting into typed policy helpers. Taskwarrior remains outside SQLite transactions.

**Tech Stack:** Python 3.11, dataclasses, enums, SQLite, unittest, mypy.

**Spec:** `checklists/REPOSITORY_UPGRADE_REFACTOR_CHECKLIST.md`, Work Package 4.

## Global Constraints

- Preserve `OUTBOX_SCHEMA_VERSION = 2` and migration support for legacy schema v1.
- Preserve strict JSON/Unicode behavior and existing Taskwarrior hook contracts.
- Keep Taskwarrior reads and mutations outside open SQLite transactions.
- Preserve the public `LifecycleOutboxRepository` API during internal migration.
- Keep the system offline and allow intermediate commits to be non-functional.

## Review Focus

- A concurrent first opener must converge on one valid schema without corrupting WAL state; test concurrent initialization.
- A claimed row whose lease expires must be reclaimable without another owner mutating it; test stale claims and lease loss.
- A malformed or poisoned row must be quarantined, not repeatedly retried; test quarantine and status reporting.
- A failure between every durable lifecycle stage must resume idempotently; extend the existing failure-injection matrix.
- SQLite transactions must never contain Taskwarrior commands; add a transaction-boundary capability test.

### Task 1: Freeze the persistence contract

**Files:**
- Create: `tests/test_lifecycle_outbox_contract.py`
- Modify: `checklists/REPOSITORY_UPGRADE_REFACTOR_CHECKLIST.md`
- Modify: `.superpowers/sdd/2026-09-21-wp4-lifecycle-outbox/progress.md`

Record schema v2 columns, supported processing states (`ready`, `claimed`, `retry`, `manual_review`, `quarantined`, `acknowledged`), execution stages, and round-trip behavior for `LifecyclePlan`, `OutboxFailure`, and `LifecycleOutboxRecord`. Add tests for schema mismatch, malformed failure JSON, and record validation before changing implementation.

### Task 2: Extract pure outbox value operations

**Files:**
- Create: `nautical_core/lifecycle_outbox_codec.py`
- Modify: `nautical_core/lifecycle_outbox.py`
- Test: `tests/test_lifecycle_outbox_contract.py`

Move plan/failure JSON encoding, decoding, canonical JSON, state/stage transition predicates, and row-to-record decoding into the new module. Keep repository methods as delegating wrappers with identical signatures and error types. Verify Unicode round trips and invalid payload failures.

### Task 3: Isolate schema and transaction ownership

**Files:**
- Create: `nautical_core/lifecycle_outbox_schema.py`
- Modify: `nautical_core/lifecycle_outbox.py`
- Test: `tests/test_structured_failure_boundaries.py`, `tests/test_lifecycle_outbox_contract.py`

Move schema version constants, initialization/migration, schema validation, and transaction context ownership into focused helpers. Repository methods must explicitly pass a connection into transaction callbacks; no helper may invoke Taskwarrior. Verify concurrent initialization, v1 migration, newer-schema refusal, and transaction-boundary behavior.

### Task 4: Extract lifecycle retry and progress policy

**Files:**
- Create: `nautical_core/lifecycle_execution_policy.py`
- Modify: `nautical_core/lifecycle_application.py`
- Test: `tests/test_lifecycle_failure_injection.py`, `tests/test_lifecycle_recovery_policy.py`

Move retry budget, manual-review, terminal outcome, stage-progress accounting, and failure classification into typed pure functions/classes. Keep serial mutation order, authoritative post-mutation snapshots, and fail-closed unavailable reads unchanged. Replace only the deeply nested policy closures in `_drain_batched`.

### Task 5: Verification and compatibility gate

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest \
  tests.test_lifecycle_outbox_contract \
  tests.test_lifecycle_failure_injection \
  tests.test_lifecycle_execution_capabilities \
  tests.test_lifecycle_recovery_policy \
  tests.test_lifecycle_terminal_plans \
  tests.test_taskwarrior_uow_contracts -q
/home/pooK/venv/test_1/bin/mypy --config-file mypy.ini \
  --disallow-untyped-defs --disallow-incomplete-defs \
  nautical_core/lifecycle_application.py nautical_core/lifecycle_outbox.py \
  nautical_core/lifecycle_models.py nautical_core/lifecycle_outbox_codec.py \
  nautical_core/lifecycle_outbox_schema.py nautical_core/lifecycle_execution_policy.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 dev_tools/nautical_reliability_smoke.py --load 10
git diff --check
```

WP4 is complete only when the existing failure-injection suite, direct extracted-module tests, strict typing, reliability smoke, and compatibility façade tests all pass.
