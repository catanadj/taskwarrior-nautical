# Remove Lifecycle Outbox Compatibility Delegation

## Goal

Remove the lifecycle outbox compatibility/delegation layer because
`nautical_core` internals are explicitly unsupported for direct user imports.
Leave one concrete repository owner and migrate every in-repository caller to
it.

## Scope

- Keep `_LifecycleOutboxRepository` as the sole concrete SQLite repository.
- Remove `LifecycleOutboxRepository` alias from `lifecycle_outbox.py`.
- Remove `RepositoryOutboxOperations`, `RepositoryLifecycleExecution`, and
  `repository_for_taskdata` from `lifecycle_outbox_operations.py`.
- Keep the typed operation protocols in `lifecycle_outbox_operations.py`.
- Update hooks, query/audit services, queue tools, reconciliation, and tests
  to construct `_LifecycleOutboxRepository` directly where a concrete store is
  required.
- Preserve all method behavior, SQLite schema/recovery semantics, and port
  interfaces.

## Non-goals

- No recurrence or lifecycle behavior changes.
- No schema migration.
- No new public replacement API.
- No dependency or version changes.

## Acceptance criteria

1. No source or test file references the removed compatibility names.
2. Lifecycle, integrity, queue, reconcile, hook, and restore tests pass.
3. Configured and strict mypy pass for affected modules.
4. Deployment/installed-layout sanity confirms no removed symbol is required.
5. Full unit and golden suites pass with unchanged counts apart from tests
   intentionally changed to target the canonical owner.
