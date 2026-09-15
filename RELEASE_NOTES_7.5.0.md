# Nautical 7.5.0

Nautical 7.5.0 is a reliability and architecture release. It keeps the
Taskwarrior hook experience familiar while making recurrence, lifecycle
recovery, and operator tooling safer to evolve and easier to verify.

## Highlights

- Added a typed task domain model with one codec, immutable observations,
  explicit drafts and patches, and a single authoritative Taskwarrior read
  boundary.
- Reworked recurrence scheduling around explicit parser, scheduler, cache,
  calendar, astronomy, and occurrence-provider owners.
- Unified `cp` completion, calendar-anchor completion, carry-forward, omission,
  multi-time, expiration, and terminal-bound behavior behind typed scheduling
  contracts.
- Added deterministic, idempotent lifecycle planning and application. Successor
  creation is backed by durable outbox intents, leases, retries, postcondition
  checks, and crash-safe replay.
- Introduced the chain-integrity engine for authoritative snapshots, immutable
  graphs, invariant evaluation, repair planning, and guarded reconciliation.

## Reliability and safety

- Read-only Doctor, query, Navigator, queue, and reconcile paths are separated
  from mutation owners and cannot silently broaden their scope.
- Mutation operations require an explicit capable unit of work and fail closed
  when required evidence, configuration, or persistence is unavailable.
- Unexpected scheduler, anchor, calendar, outbox, and Taskwarrior failures now
  retain typed evidence instead of being mistaken for ordinary absence.
- Added bounded file and JSON parsing, cache size limits, SQLite contention
  handling, stale-lock cleanup, resource limits, and Unicode-preserving output.
- Preserved strict hook protocol behavior: valid hooks emit one JSON document on
  stdout; diagnostics are opt-in and go to stderr.
- Added backup/restore manifests, checksum validation, offline kits, deep
  Doctor checks, and an offline recovery runbook for disconnected devices.

## Architecture and maintainability

- Replaced broad facade and callback dictionaries with explicit API bindings,
  protocols, runtime ports, and composition roots.
- Removed obsolete scheduler, parser, preview, reconcile, and integration
  ownership paths rather than retaining shadow implementations.
- Consolidated preview collection into one typed occurrence pipeline and kept
  rendering separate from scheduling decisions.
- Reorganized parsing, hooks, operator, lifecycle, and domain modules around
  stable ownership boundaries.
- Centralized short-UUID formatting, omission state, configuration loading,
  failure classification, and scheduler terminal evidence.

## Performance

- Reduced Taskwarrior calls in queue draining and reconciliation through bounded
  snapshots, hydration, batching, compiled evaluator reuse, and shared cursors.
- Added call-count, row-count, memory, import, cache, wall-time, desktop, and
  slow-device performance budgets.
- Hardened the performance harness so cache benchmarks use isolated temporary
  state and remain reliable on read-only CI checkouts.
- Added absent-field allocation, immutable-task, resource-limit, and snapshot
  memory benchmarks without changing recurrence semantics.

## Testing and compatibility

- Expanded direct contract coverage for hooks, lifecycle transitions, mutation
  guards, parser/scheduler boundaries, renderer behavior, outbox recovery,
  chain integrity, offline restore, and dynamic effect services.
- Added deterministic shuffled golden suites, black-box Taskwarrior scenarios,
  stress and soak campaigns, fault injection, concurrency checks, and strict
  stdout/stderr protocol tests.
- Enabled strict mypy checks across the migrated package boundaries; the current
  local run reports no issues across 244 source files.
- Expanded CI compatibility coverage for Python 3.13 and 3.14 and Astral 3.2
  plus the current release.
- Consolidated runtime dependency pins into `requirements.txt`; astronomy
  remains an optional install through `requirements-astronomy.txt`.

## Documentation and operations

- Made the `docs/` site the primary documentation source, covering recurrence
  grammar, calendars, omissions, multiple times, completion periods, lifecycle
  integrity, diagnostics, backups, synchronization, and offline recovery.
- Documented scheduling/wait-delta carry-forward behavior and the operator
  contracts for Doctor, query, reconcile, Navigator, and the lifecycle outbox.
- Added deployment sanity checks, runtime manifests, installation validation,
  and release-oriented recovery guidance.

## Verification

The release candidate has passed the local full unittest suite (1,237 tests,
3 skipped), package-wide mypy, targeted golden and contract checks, and
repository diff validation. CI workflows for type checking, performance,
stress, compatibility, and astronomy are included in the release gate.

## Upgrade notes

- Upgrade through the normal Nautical installer. Existing Taskwarrior data and
  configuration are preserved.
- Astral is optional; install the astronomy requirements only when configured
  astronomy locations are used.
- Before an offline upgrade or device transfer, build and verify an offline kit
  and a current Taskwarrior/lifecycle backup.
- No Taskwarrior task-data migration is required. Keep the previous managed
  Nautical release available as the rollback path until the new installation
  passes Doctor and reconcile dry-run checks.
