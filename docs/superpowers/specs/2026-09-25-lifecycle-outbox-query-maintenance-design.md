# Lifecycle Outbox Query and Maintenance Extraction

## Purpose

Reduce the responsibility and size of `lifecycle_outbox.py` without changing
durable lifecycle behavior. The outbox repository currently owns SQLite
security and transactions, writes, claims, operator reads, and maintenance.
Schema, codec, claim policy, and operation policy already have separate
modules; read/query projection and maintenance remain the extraction targets.

## Approved boundary

Create two focused owners:

- `lifecycle_outbox_queries.py` owns the read-only status and immutable
  snapshot queries, including query-specific projection.
- `lifecycle_outbox_maintenance.py` owns acknowledged-row pruning and bounded
  opportunistic housekeeping.

Keep one narrow repository composition service responsible for secure
connection creation, schema validation, transaction boundaries, and wiring
these operations to callers. Keep schema ownership in
`lifecycle_outbox_schema.py`, serialization in `lifecycle_outbox_codec.py`,
claim/lease policy in `lifecycle_outbox_claims.py`, and state-transition policy
in `lifecycle_outbox_operations.py`.

This is the narrow split selected by the user. Do not create per-operation
repository classes, duplicate connection/security helpers, or forwarding
modules for removed internal paths.

## Contracts and invariants

- `status` remains read-only: it must not create directories, initialize or
  migrate schema, or repair database state.
- Snapshot reads remain complete and deterministic. A poison row rejects the
  snapshot instead of disappearing from it.
- Maintenance stays bounded and deletes only acknowledged rows older than the
  requested retention boundary.
- SQLite transactions stay inside repository operations. No Taskwarrior
  command runs while a transaction is active.
- Busy, corrupt-schema, malformed-row, filesystem-security, and interrupted
  operation outcomes retain their current typed result behavior.
- Existing command output and lifecycle semantics remain unchanged.
- Keep typed intermediate query/projection results. Convert to the existing
  operator-facing payload only at the composition/presentation boundary.

## Implementation sequence

1. Characterize status, snapshot, pruning, and housekeeping behavior with
   focused tests, including missing state, poison rows, retention boundaries,
   busy/corrupt storage, and bounded cleanup.
2. Extract query execution and projection into the query owner. Keep connection
   lifecycle and schema checks in the repository composition service.
3. Extract pruning and opportunistic housekeeping into the maintenance owner.
   Inject or pass only the narrow connection, transaction, clock, and security
   operations those routines require.
4. Migrate repository-owned callers directly if any interface changes. Keep
   no compatibility forwarding module or duplicate implementation.
5. Run focused outbox, queue, review, reconcile, lifecycle, and operator tests,
   then the full unit and golden suites.

## Acceptance criteria

- Query and maintenance behavior have distinct owners and focused tests.
- The composition service contains no copied query or housekeeping logic.
- Transaction and filesystem-security behavior remains owned in one place.
- Existing lifecycle/outbox contract tests and CLI output tests pass unchanged
  or receive intentional, reviewed updates.
- CI and normal/shuffled golden verification remain green; coverage does not
  regress.

## Non-goals

- Redesign lifecycle state machines, retry policy, schema, or persisted data.
- Split every write/claim operation into a separate repository.
- Change the public Nautical query API, CLI contracts, or Navigator internals.
- Add compatibility support for unsupported imports from `nautical_core`.
