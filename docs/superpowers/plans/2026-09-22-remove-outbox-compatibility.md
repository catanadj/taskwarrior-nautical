# Remove Lifecycle Outbox Compatibility Delegation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the unsupported lifecycle outbox compatibility wrappers and migrate the repository to one concrete internal owner.

**Architecture:** `_LifecycleOutboxRepository` remains the sole SQLite store. Typed protocols remain as dependency boundaries, but no concrete adapter or factory delegates to the store. All callers and tests use the canonical owner directly.

**Tech Stack:** Python, SQLite, unittest, mypy.

**Spec:** `docs/superpowers/specs/2026-09-22-remove-outbox-compatibility-design.md`

## Global Constraints

- Keep hook stdout strict JSON and diagnostics opt-in.
- Preserve lifecycle idempotence, recovery, leases, retries, and postcondition checks.
- Do not change recurrence, lifecycle, query, or presentation behavior.
- Do not add dependencies or network access.
- Keep unsupported internal imports unsupported; no compatibility bridge is added.

## Review Focus

- Queue review/status patches must target `_LifecycleOutboxRepository.status`.
- Integrity query/audit must still satisfy typed evidence-port contracts.
- Reconcile and exit hooks must receive a store implementing every execution method.
- Restore and failure-injection tests must continue using disposable Taskdata.
- Installed-layout checks must not require removed symbols.

### Task 1: Prove the removal surface with failing import/reference tests

**Files:**
- Modify: `tests/test_lifecycle_outbox_contract.py`
- Modify: `tests/test_queue_review.py`
- Modify: `tests/test_queue_status_budget.py`
- Add: `tests/test_outbox_compatibility_removal.py`

- [x] Assert removed names are absent from the operations module and the old public alias is absent from `lifecycle_outbox`.
- [x] Update queue test seams to patch `_LifecycleOutboxRepository` instead of the removed wrapper.
- [x] Run the focused tests and observe import/reference failures before production edits.

### Task 2: Remove wrapper classes and factory

**Files:**
- Modify: `nautical_core/lifecycle_outbox.py`
- Modify: `nautical_core/lifecycle_outbox_operations.py`

- [x] Delete the `LifecycleOutboxRepository` alias and export entry.
- [x] Delete `RepositoryOutboxOperations`, `RepositoryLifecycleExecution`, and `repository_for_taskdata`.
- [x] Retain `LifecycleOutboxOperationsPort`, `LifecycleOutboxEvidencePort`, and `LifecycleExecutionOutboxPort`.
- [x] Update `__all__` to expose protocols only.

### Task 3: Migrate production callers

**Files:**
- Modify: `nautical_core/queue_status_service.py`
- Modify: `nautical_core/integrity_audit_service.py`
- Modify: `nautical_core/integrity_query_service.py`
- Modify: `nautical_core/hooks/exit_impl.py`
- Modify: `nautical_core/tools/nautical_queue_status.py`
- Modify: `nautical_core/tools/nautical_reconcile.py`
- Modify: `nautical_core/modify_spawn_effects.py`

- [x] Replace wrapper construction with `_LifecycleOutboxRepository` construction.
- [x] Replace `repository_for_taskdata` factory use with direct canonical construction.
- [x] Keep protocol-typed parameters unchanged where they provide a real boundary.
- [x] Run focused lifecycle/query/queue/reconcile tests.

### Task 4: Migrate test seams and verify behavior

**Files:**
- Modify all tests referencing removed wrapper names.

- [ ] Update mocks/patches to target the canonical repository methods.
- [ ] Run lifecycle failure injection, outbox contract, integrity, queue, restore, and hook suites.
- [ ] Run configured and strict mypy for affected modules.
- [ ] Run deployment sanity and verify removed symbols are not required by installed layout.

### Task 5: Full acceptance and checklist closure

- [ ] Run full unit discovery and both golden-suite modes.
- [ ] Run `git diff --check`.
- [ ] Update the repository checklist to record the intentional internal breaking cleanup and evidence.
