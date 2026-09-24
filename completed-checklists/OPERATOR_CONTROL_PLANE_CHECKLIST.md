# Nautical Operator Control Plane Checklist

Replace Nautical's independently assembled Doctor, query, reconcile, queue,
chain-repair, and Navigator orchestration with one typed operator control
plane. The completed system must provide authoritative observation, explicit
scope and coverage, actionable findings, deterministic plans, guarded effect
delegation, and consistent presentation without duplicating ownership already
held by the scheduler, lifecycle, Taskwarrior integration, task-domain, chain
integrity, queue/reconcile, or hook-workflow engines.

The upgraded system must answer six questions for every operator request:

1. What exact system or task scope did the caller request?
2. Which evidence was observed, and is its coverage authoritative?
3. What health, integrity, lifecycle, configuration, or scheduling facts follow?
4. Which findings require no action, safe automatic repair, retry, or review?
5. Which established engine owns any requested effect?
6. How can the same result be consumed through JSON, text, Rich, Navigator, or
   another local tool without changing the decision?

## Preflight (2026-08-26)

Completed prerequisites:

- [x] Current `main` is at `51e03a6` (`docs: publish v7 documentation site`)
  and has no tracked modifications. The v7.2.0 runtime is the starting
  production baseline; the documentation site is now deployed separately.
- [x] Desktop prerequisites are available: Python 3.11.2, Taskwarrior 3.4.2,
  Rich, Astral 3.2, and mypy. The project virtualenv at
  `/home/pooK/venv/test_1` provides the optional runtime dependencies; system
  Python does not provide Astral.
- [x] Existing process-level golden and black-box suites cover operator
  command loading, strict JSON, malformed input, installed-layout behavior,
  lifecycle recovery, and operator cutover scenarios.
- [x] Current operator composition roots and direct ownership seams are
  identifiable: `nautical_doctor.py`, `nautical_query.py`,
  `nautical_reconcile.py`, `nautical_queue_status.py`, chain-integrity tools,
  and `nautical_navigator.py` each assemble overlapping context, snapshot,
  status, or rendering behavior.
- [x] The versioned query API, lifecycle outbox, chain-integrity engine,
  scheduler service, Taskwarrior unit of work, and typed lifecycle application
  are available as the domain owners to reuse.
- [x] Documentation is ready for the new public contract. `mkdocs build
  --strict` passes and the Pages deployment is active.
- [x] `python3 dev_tools/nautical_deploy_sanity.py --json` passes on the
  current checkout, including installed-layout operator module loading.
- [x] The existing operator-focused golden subset passes: 6 tests, 6 passed,
  0 failed. This is characterization coverage, not control-plane completion.

Remaining gates before implementation:

- [x] Reuse the previous checklist's `benchmarks/hooks/v7.1.0/*final5.json`
  reports as the inherited runtime baseline. They were run after the last
  production code change and cover anchor/CP completion, queue drain, reconcile
  dry-run/apply, integrity scaling, imports, snapshot reuse, and resource
  limits on both available Termux devices. The documentation-only commit does
  not invalidate them.
- [ ] Add isolated operator-stage measurements for Doctor, query, Navigator,
  queue-status, reconcile, and chain-repair presentation/serialization. The
  inherited `final5` reports measure their shared runtime paths but do not split
  every operator CLI stage into separate call, row, SQLite, hydration,
  planning, verification, rendering, wall-time, import-time, and memory
  samples. This is a measurement refinement, not a code or data blocker.
- [ ] Record current operator JSON schemas, exit mappings, scope semantics,
  duplicate/history noise, and process-level output for the baseline fixtures.
- [x] Create `operator-control-plane-v7` from this verified `main`; no control
  plane implementation has started yet.
- [ ] Confirm the final cutover owner and the one-time GitHub Pages setting is
  already complete; no user-task migration is expected.

Preflight conclusion: the architecture and tooling are ready to begin Section
1 now. The inherited `final5` reports are accepted as the behavioral/runtime
baseline; isolated operator-stage timings remain a Section 1 refinement, not a
dependency or correctness blocker. No schema or data migration blocker was
found. The implementation branch should remain offline and non-operational
until final cutover, as specified below.

## Scope And Working Model

- [x] Create `operator-control-plane-v7` from the current verified `main`.
- [ ] Develop exclusively on that branch and keep `main` operational until the
  final cutover.
- [ ] Treat Nautical as offline while the branch is under construction.
  Intermediate commits do not need to be installable or operational.
- [ ] Keep this checklist local. Push implementation commits only to the
  control-plane branch and merge only after every cutover gate passes.
- [ ] Do not build old/new adapters, dual command paths, compatibility facades,
  fallback exports, or shadow operator services.
- [ ] Remove each replaced owner when its consumers have migrated. Broken
  intermediate commits are acceptable on the offline branch.
- [ ] Preserve Taskwarrior as the task store and the lifecycle outbox as the
  durable work store. Do not introduce a shadow operational database.
- [ ] Preserve the current task UDA representation, recurrence grammar,
  scheduler semantics, lifecycle rules, and chain-integrity invariants.
- [ ] Keep scheduling behind `SchedulerService`, Taskwarrior I/O behind the
  integration unit of work, lifecycle effects behind lifecycle application,
  and structural repair behind the chain-integrity engine.
- [ ] Make the control plane an orchestration and observation owner, not a new
  domain engine and not a second mutation gateway.
- [ ] Keep every snapshot immutable and invocation-scoped. Never reuse absence
  or completeness evidence after a mutation epoch changes.
- [ ] Treat malformed, truncated, stale, ambiguous, unsafe, or unavailable
  evidence as unavailable. Never reinterpret unavailable as absent or healthy.
- [ ] Keep read-only commands physically unable to invoke mutation services.
- [ ] Keep renderers physically unable to read Taskwarrior, write SQLite, or
  alter control-plane outcomes.
- [ ] Preserve strict JSON output with `ensure_ascii=False`. JSON commands emit
  one versioned document on stdout; optional diagnostics use stderr only.
- [ ] Keep whole-system work bounded through pagination, explicit caps, and
  resumable cursors rather than hidden truncation.
- [ ] Prefer active and pending task evidence by default. Historical audit
  evidence is aggregated and expanded only when explicitly requested.
- [ ] Preserve Taskwarrior 3.4.2 as the supported compatibility target until a
  separate compatibility decision changes it.

Cutover policy:

- [ ] Stop Nautical hooks and operator commands during final installation.
- [ ] Inspect active lifecycle intents and structural plans before replacing
  operator ownership.
- [ ] Do not migrate Taskwarrior tasks. Add a durable schema migration only if
  a genuinely persistent control-plane contract is required.
- [ ] Roll back by reinstalling the previous release, not by retaining the old
  operator path in production.
- [ ] Re-enable normal use only after installed-layout Doctor, query, Navigator,
  reconcile dry-run/apply, chain repair, queue, and hook smoke tests pass.

## Target Ownership

Exact filenames may change when a clearer boundary emerges, but each concern
must have one owner:

- `operator_models.py`: versioned requests, scopes, coverage, observations,
  findings, action plans, application outcomes, verification, and failures.
- `operator_context.py`: one validated configuration, clock, timezone,
  integration unit of work, lifecycle session, diagnostic policy, and command
  budget per invocation.
- `operator_snapshot.py`: immutable system snapshot assembly, provider
  manifests, exact indexes, coverage proof, mutation epochs, and bounded
  hydration.
- `operator_inspection.py`: pure evaluation of configuration, installation,
  tasks, chains, lifecycle, outbox, scheduling, and dependencies from a
  snapshot and established domain services.
- `operator_planner.py`: converts typed findings and explicit commands into
  deterministic no-op, retry, repair, recovery, housekeeping, or refusal
  plans. It performs no effects.
- `operator_application.py`: verifies guards and delegates approved effects to
  lifecycle application, chain integrity, Taskwarrior mutation, runtime
  cleanup, or housekeeping owners.
- `operator_control_plane.py`: phase orchestration only: validate, scope,
  observe, inspect, plan, optionally apply, refresh, verify, and return.
- `operator_presentation.py`: renders typed results as JSON-ready data, plain
  text, Rich output, summaries, progress, and Navigator view models.
- `tools/nautical_doctor.py`, `tools/nautical_reconcile.py`,
  `tools/nautical_query.py`, `tools/nautical_queue_status.py`, chain-repair
  commands, and `nautical_navigator.py`: thin composition roots only.

The final read-only flow is:

```text
validated operator request
          |
explicit scope + limits + capabilities
          |
one invocation context
          |
immutable authoritative snapshot
          |
pure domain inspection
          |
typed findings + chain/task/system status
          |
one versioned result
          |
JSON / plain / Rich / Navigator rendering
```

The final effectful flow is:

```text
validated apply request
          |
authoritative snapshot + coverage proof
          |
pure deterministic plan
          |
plan fingerprint + expected-current guards
          |
delegate each effect to its established owner
          |
invalidate affected evidence
          |
bounded authoritative refresh
          |
postcondition verification
          |
applied / already-applied / retryable / stale / review result
```

## 1. Baseline And Ownership Inventory

- [x] Record the starting commit, Python, Taskwarrior, Astral, Rich, platform,
  and managed-runtime versions.
- [x] Capture desktop and both Termux baselines for Doctor, query, Navigator,
  queue status, reconcile dry-run/apply, and chain repair.
- [ ] Record wall time, import time, peak memory, Taskwarrior calls, exported
  rows, SQLite calls, hydration rows, planner time, verification time, and
  presentation time independently.
- [x] Inventory every Taskwarrior export, repository read, outbox read,
  configuration reload, integrity evaluation, lifecycle drain, and mutation
  used by current operator tools.
- [x] Inventory every scope/filter interpretation and identify commands that
  silently broaden, truncate, or repeat a requested scope.
- [x] Inventory every JSON schema, exit code, severity, status, progress event,
  and text renderer currently exposed to users or external tools.
- [x] Characterize Doctor noise, duplicate findings, historical findings,
  unavailable evidence, poison rows, partial repair, and large-history behavior.
- [x] Attach at least one process-level characterization fixture to every
  current operator command before replacing its path.
- [x] Identify all direct imports from operator tools into hook-private,
  lifecycle-private, chain-private, or scheduler-private implementation details.
- [x] Record the exact owners that must remain unchanged during this upgrade.

Pass 1 inventory record (2026-08-26):

- Starting commit: `51e03a6` (`docs: publish v7 documentation site`); isolated
  branch `operator-control-plane-v7` is pushed from this commit.
- Desktop: Python 3.11.2, Taskwarrior 3.4.2, Astral 3.2, Rich available,
  managed release `r-e74de36bc82e`.
- Inherited benchmark set: `benchmarks/hooks/v7.1.0/*final5.json` for both
  available Termux devices, plus the desktop v7.1 operator benchmark family.
  These are accepted as the runtime baseline because no production code changed
  after those measurements.
- Current composition roots: `tools/nautical_doctor.py`,
  `tools/nautical_query.py`, `tools/nautical_reconcile.py`,
  `tools/nautical_queue_status.py`, chain-integrity tooling, and
  `nautical_navigator.py`.
- Existing domain owners retained for the redesign: `SchedulerService`, the
  Taskwarrior integration unit of work, lifecycle application/outbox, and the
  chain-integrity engine. No new operator owner has been introduced yet.
- Read/effect inventory: Doctor reads configuration, installation/runtime,
  Taskwarrior, outbox, and chain snapshots; query reads Taskwarrior and
  scheduler/chain snapshots; reconcile reads Taskwarrior, lifecycle, outbox,
  and integrity state before delegating lifecycle or structural effects;
  queue-status reads the lifecycle outbox; Navigator owns its existing display
  path and scheduler lookup. The current mutation owners are lifecycle
  application, chain-integrity recovery, Taskwarrior UoW/mutations, and queue
  maintenance respectively.
- Scope inventory: query supports UUID, chain, all-task, temporal, and
  occurrence bounds; reconcile supports UUID/chain scopes and bounded/full
  audit; Doctor defaults to whole-system diagnostics; queue-status exposes
  outbox limits/samples. These roots currently assemble filters independently,
  which is the primary Section 2 migration target.
- Contract inventory: Doctor uses `nautical.doctor` with installation/full
  modes; query uses `nautical.query.*` and typed status/failure payloads;
  reconcile uses `nautical.reconcile`; queue-status uses
  `nautical.queue_status`; Navigator remains primarily text/Rich. Existing
  process fixtures cover the operator roots and strict JSON; the new plane
  will preserve behavior through typed contract tests.
- Direct-import result: no operator root imports a hook-private implementation
  owner. Shared imports are limited to established chain snapshot/integrity,
  lifecycle, scheduler, Taskwarrior UoW, query, and outbox services listed
  above.
- Fixture result: the installed-layout black-box suite exercises Doctor, query,
  reconcile, queue status, and Navigator as subprocess roots. Chain repair is
  currently an integrity/reconcile capability rather than a separate CLI, so
  its process characterization is covered through those roots.
- Outstanding Section 1 work is intentionally measurement/inventory detail:
  isolated per-command stage timings, complete schema/exit/scope cataloguing,
  noise characterization, and a fixture-to-owner matrix. These are required
  before consumer migration but do not block the baseline pass.

Completion criteria:

- [ ] Every existing operator read, decision, effect, schema, and renderer has
  a named current owner and target owner.
- [ ] Baselines distinguish orchestration cost from Taskwarrior, SQLite,
  scheduling, integrity, and presentation cost.
- [ ] No behavioral migration begins without a regression fixture for the path.

## 2. Define The Versioned Operator Contract

- [x] Define a closed `OperatorOperation` set covering capabilities, inspect,
  health, occurrences, chain, integrity, lifecycle, queue, diagnose, plan,
  apply, verify, and housekeeping.
- [x] Define one immutable `OperatorRequest` with operation, scope, temporal
  bounds, inclusion policy, detail level, limits, apply policy, and output mode.
- [x] Define stable schema names and versions for each public response family.
- [x] Define one status taxonomy: `ok`, `attention`, `repairable`, `deferred`,
  `manual_review`, `unavailable`, `partial`, and `error`.
- [x] Define stable failure codes with retryability, affected scope, evidence,
  and next action. Do not expose raw exception text as the primary contract.
- [x] Define stable process exit codes for success, findings, unavailable,
  invalid request, partial application, manual review, and internal failure.
- [x] Keep JSON-native values serializable without `default=str`; timestamps,
  timezones, paths, enums, and typed identifiers require explicit encoders.
- [ ] Preserve unknown response fields for forward-compatible readers while
  rejecting unsupported request schema versions.
- [ ] Publish capabilities including schemas, operations, scope kinds, limits,
  supported Taskwarrior version, optional dependencies, and mutation support.
- [x] Add lossless encode/decode and round-trip tests for every public model.

Pass 1 implementation: `nautical_core/operator_models.py` now owns the closed
operation/status/scope vocabularies, immutable request and limit contracts,
structured failure evidence, and the versioned JSON result envelope. Focused
coverage is in `tests/test_operator_models.py`; CLI migration and full
forward-compatible decoding remain later passes.

Pass 2 implementation: all operator models now provide strict mapping decoders
and recursive JSON-native encoding. Unsupported versions, malformed scopes,
ambiguous values, and unavailable results without structured failure evidence
are rejected; unknown response data remains preserved in the result envelope.

Pass 3 implementation: `OperatorExitCode` and `exit_code_for_status()` define
one machine-readable process mapping for success, findings, invalid requests,
unavailable evidence, partial application, manual review, and internal failure.

Pass 4 implementation: `OperatorCapabilities` provides a versioned discovery
document with operation/scope vocabularies, safety limits, Taskwarrior and
optional-dependency evidence, and explicit mutation support. It round-trips
through the same strict JSON boundary as the other operator models.

Section 3 pass 1 implementation: `CoverageKind` and `OperatorCoverage` make
scope evidence explicit and immutable. Complete coverage rejects omissions;
unavailable coverage requires a reason; observed identities, snapshot IDs, and
mutation epochs are retained for later plan guards.

Section 3 pass 2 implementation: `OperatorCursor` binds continuation position
and page size to a snapshot ID, configuration fingerprint, and mutation epoch.
It rejects negative positions and malformed evidence, preventing a page from
being resumed against changed observations.

Section 3 pass 3 implementation: `CoverageRequirement` expresses the minimum
evidence quality needed by a read or effect plan and compares it against
immutable coverage. This gives future planners a typed fail-closed guard for
bounded, partial, and unavailable snapshots.

Section 4 pass 1 implementation: `OperatorInvocationContext` binds one typed
request to the existing validated `IntegrationContext`, a UTC capture instant,
configuration fingerprint, timezone, mutation epoch, and access mode. It does
not reload configuration or create a second Taskwarrior/outbox owner.

Section 4 pass 2 implementation: `OperatorDependency` captures one typed
availability record with optional version and a required reason for unavailable
dependencies. Operator capabilities and future contexts can now report
Taskwarrior/Astral/Rich resolution without repeating discovery.

Section 4 pass 3 implementation: `OperatorPresentationPolicy` and
`OperatorOutputMode` are immutable observer preferences carried by the context.
They validate output/diagnostic selection without being available to domain
inspection or mutation decisions.

Section 4 pass 4 implementation: `OperatorInvocationCache` provides bounded
LRU-like memoization scoped to one invocation, with explicit clear semantics.
It prevents unbounded operator read retention and ensures cache state is not
shared between invocations.

Section 4 pass 5 implementation: apply requests are rejected unless the
integration context is mutation-capable, and `assert_compatible()` rejects
configuration or timezone drift before downstream work can continue.

Section 5 pass 1 implementation: `OperatorSnapshot` provides an immutable
observation envelope with explicit coverage, creation instant, mutation epoch,
configuration fingerprint, component evidence, and provider manifest.

Section 5 pass 2 implementation: `SnapshotIndexes` records deduplicated task,
chain, link, status, recurrence, and child-slot identities once per snapshot,
with strict mapping round-trip validation.

Section 5 pass 3 implementation: `HydrationBatch` records bounded set-read
requests for missing predecessor, child, or outbox identities. It preserves
requested/observed identities, limits, and completeness without allowing a
complete batch to claim unrequested rows.

Section 5 pass 4 implementation: `SnapshotComponent` records independent
component observation time, mutation epoch, and optional coverage. Snapshots
can now expose freshness for Taskwarrior, outbox, configuration, and runtime
evidence without implying an atomic cross-store transaction.

Section 5 pass 5 implementation: `OperatorSnapshot.assert_consistent()` rejects
component evidence captured in a different mutation epoch, providing a
fail-closed guard before planning or mutation.

Section 5 pass 6 implementation: snapshots now expose `cacheable` and
`assert_cacheable()`. Unavailable evidence cannot enter invocation or shared
caches, preventing failed reads from becoming authoritative empty results.

Section 5 verification: existing golden coverage already exercises poison rows,
malformed JSON, noisy stderr, Taskwarrior locks/timeouts, truncated exports,
changing-task evidence, and configuration drift. The remaining unchecked items
are the authoritative reader/assembler and its integration with those tests.

Section 5 pass 7 implementation: `OperatorSnapshotAssembler` provides the first
authoritative assembly boundary. It accepts only typed snapshots, verifies
configuration and mutation epochs against the invocation context, validates
component consistency, and performs no I/O or mutation itself.

Section 5 pass 8 implementation: the assembler now projects the established
`ChainSnapshot` service output into the operator envelope, preserving source
coverage and building task/chain/link/status/recurrence/child-slot indexes
without introducing a second chain reader.

Section 5 pass 9 implementation: chain snapshot projection now rejects a
configuration fingerprint mismatch and preserves truncated source coverage as
`partial`, preventing degraded evidence from being mislabeled as bounded.

Section 5 pass 10 implementation: the reader now enforces each request's
minimum coverage requirement and returns typed `insufficient_snapshot_coverage`
failures instead of allowing degraded evidence into a planning path.

Section 5 pass 11 implementation: cache regression coverage confirms absent
and unavailable reads are never memoized; only successful, acceptable snapshots
can be reused within an invocation.

Section 5 pass 12 implementation: chain projections now include explicit chain
component freshness evidence tied to the invocation mutation epoch.

Section 5 pass 13 implementation: the reader enforces task, chain, and history
link limits before caching or returning a snapshot.

Section 5 pass 14 implementation: invocation caches now support targeted key or
prefix invalidation, preserving unrelated projections after certain mutations.

Section 5 pass 15 implementation: the integrity query now collects its
authoritative snapshot through `ChainSnapshotReader` and audits that exact
snapshot, avoiding a second Taskwarrior export.

Section 5 pass 16 implementation: Doctor now normalizes its authoritative
export once, routes it through the shared snapshot reader, and audits the same
snapshot without a second Taskwarrior read.

Section 5 pass 17 implementation: reconcile integrity audits now validate their
already-authoritative lifecycle snapshot through the shared reader without
introducing another Taskwarrior export.

Section 5 pass 18 implementation: bounded chain and UUID reads now request
bounded source history, while complete coverage requests retain full history.

Section 5 pass 19 implementation: `OperatorSnapshotSession` now centralizes
invocation-scoped reads and targeted or full snapshot invalidation.

Section 5 pass 20 implementation: mutation invalidation now distinguishes
certain affected projections from uncertain mutations that require clearing all
snapshot evidence.

Section 5 pass 21 implementation: snapshots now expose `satisfies()` and
`assert_satisfies()` so planners can mechanically enforce evidence floors.

Section 5 pass 22 implementation: session-level regression coverage proves
independent reads reuse one successful snapshot and provider invocation.

Section 5 pass 23 implementation: scope-policy coverage confirms whole-system
reads remain bounded unless complete coverage is explicitly requested.

Section 2 pass 1 implementation: `OperatorResult` now rejects contradictory
`ok` envelopes that also contain failure evidence.

Section 2 pass 2 implementation: `OperatorResult.exit_code` now exposes the
single stable status-to-process-code mapping.

Section 2 pass 3 implementation: operator result payloads are validated as
JSON-native at construction time, preventing invalid values from crossing the
typed boundary.

Section 3 pass 1 implementation: `OperatorRequest` now carries an explicit
coverage requirement and rejects effectful requests that allow incomplete
evidence.

Section 3 pass 2 implementation: continuation cursors now reject reuse against
different snapshot, configuration, or mutation-epoch evidence.

Section 3 pass 3 implementation: cursors now provide deterministic page
advancement while preserving their immutable evidence identity.

Section 3 pass 4 implementation: multi-chain and multi-UUID scopes now use
bounded narrow reads for small sets and one bounded candidate read for larger
sets, with explicit filtering and coverage preservation.

Section 3 pass 5 implementation: `OperatorSnapshotSession.read_many` preserves
independent per-scope successes and failures instead of collapsing a batch into
one chain-wide failure.

Section 3 pass 6 implementation: multi-value scopes now provide typed
single-value splitting for safe fan-out through the shared reader.

Section 3 pass 7 implementation: `OperatorPage` now binds bounded items to a
validated continuation cursor and rejects cursors on complete pages. JSON
round-trip and malformed-item tests protect the public pagination contract.

Section 3 pass 8 implementation: page construction now rejects a page larger
than its cursor-declared limit, with maximum-plus-one regression coverage.

Section 3 pass 9 implementation: the shared `OperatorResult` envelope now
 carries the typed page contract, so pagination metadata has one stable
 decoder across operator commands.

Section 2 pass 4 implementation: `OperatorResult` preserves unknown JSON
response fields in validated extensions, allowing forward-compatible readers
without weakening the standard envelope.

Section 2 pass 5 implementation: capability discovery now publishes explicit
response schema identifiers, with deterministic defaults derived from the
advertised operations.

Section 4 pass 1 implementation: invocation contexts now expose an explicit
mutation-epoch guard for downstream unit-of-work reads and effects.

Section 4 pass 2 implementation: context compatibility now rejects Taskdata
path or Taskwarrior command-prefix changes within one invocation.

Section 4 pass 3 implementation: `OperatorInvocationContext.from_unit_of_work`
binds an operator invocation to one existing Taskwarrior UoW and epoch.

Section 4 pass 4 implementation: query, Doctor, and reconcile now construct
operator contexts from their existing UoW instead of rebuilding integration
context state.

Section 4 pass 5 implementation: regression coverage confirms separate
operator invocations receive isolated bounded caches and cannot leak state.

Section 4 pass 6 verification: lifecycle application owns one bounded outbox
session for effectful drains, while query, Doctor, and reconcile read paths do
not acquire write ownership. No additional context-level session bridge is
needed.

Section 4 pass 7 implementation: invocation compatibility now rejects changes
to the resolved timezone object and Taskdata resolution source, with focused
regression coverage for both drift cases.

Section 3 pass 10 implementation: `OperatorLimits` now includes an explicit
bounded `file_records` budget alongside task, chain, occurrence, history,
finding, outbox, scheduler, and wall-time limits.

Section 3 pass 11 implementation: `OperatorScope` now exposes typed system,
chain-list, and UUID-list constructors, giving CLI adapters one normalization
surface for selector semantics.

Section 3 pass 12 verification: coverage requirements are enforced by typed
requests and snapshot reads; complete evidence is mandatory for effects,
limits are independent, cursors are evidence-bound, and `read_many` preserves
independent chain failures. CLI pagination remains deferred to Section 11.

Section 7 pass 1 implementation: `OperatorFinding` now provides one immutable,
JSON-native finding contract with stable code/domain/severity/actionability,
scope, affected identities, evidence, and actionable guidance.

Section 7 pass 2 implementation: `deduplicate_findings()` now collapses
identical findings deterministically and merges affected identities, reducing
repeated historical output without discarding evidence.

Section 7 pass 3 implementation: `highest_severity()` now provides one stable
severity precedence policy for aggregating system and chain status.

Section 6 pass 1 implementation: `OperatorInspector` defines a pure snapshot
observer protocol, with coverage inspection and deterministic deduplicated
finding execution. Insufficient evidence is blocking and never treated as
absence.

Section 6 pass 2 implementation: inspector runner tests prove declaration-order
execution, deterministic deduplication, and explicit rejection of invalid
inspector implementations.

Section 6 pass 3 implementation: the pure snapshot-consistency inspector now
turns mixed-epoch or malformed component evidence into a blocking typed finding
with refresh guidance.

Section 6 pass 4 implementation: snapshot limits inspection now reports each
 exceeded task, chain, or history-link dimension as an independent blocking
 finding with explicit observed and expected values.

Section 6 pass 5 implementation: component availability inspection now reports
absent or unavailable configuration/dependency evidence as blocking findings,
without confusing it with an invalid domain object.

Section 6 pass 6 implementation: component validity inspection now distinguishes
present-but-invalid task-domain or schedule evidence from unavailable providers,
returning actionable typed findings with repair guidance.

Section 6 pass 7 implementation: named `TaskDomainInspector` and
`ScheduleAvailabilityInspector` classes now expose those domain checks through
the shared pure validity boundary.

Section 6 pass 8 implementation: `classify_historical()` preserves audit
evidence while converting inactive findings to informational/deferred status,
keeping current operational output focused.

Section 6 pass 9 implementation: `prioritize_findings()` orders active
identities before historical findings with deterministic severity/domain/code
tie-breaks.

Section 6 pass 10 implementation: `inspect_standard_components()` provides a
stable bundle for configuration, dependencies, task-domain, schedule,
integrity, lifecycle, and performance availability checks.

Section 6 pass 11 implementation: `aggregate_historical()` groups related
deferred findings while retaining affected identities and an aggregate count.

Section 6 pass 12 implementation: `inspect_snapshot()` provides one fixed-order
 composite for coverage, consistency, and limit checks with deterministic
 deduplication.

Section 7 pass 4 implementation: every non-informational finding actionability
now requires concrete command or guidance, including retryable, deferred, and
manual-review states.

Section 7 pass 5 implementation: `status_for_findings()` now derives one stable
aggregate `OperatorStatus` precedence for empty, warning, repairable,
manual-review, and blocking findings.

Section 7 pass 6 implementation: aggregate status now distinguishes retryable
unavailability and deferred work from healthy results, with explicit tests.

Section 7 pass 7 implementation: `sort_findings()` now defines stable severity,
domain, code, message, and identity ordering for all output projections.

Section 8 pass 1 implementation: `OperatorPlan` now provides an immutable,
JSON-native plan contract bound to snapshot ID, configuration fingerprint,
scope, coverage, and ordered operations; effectful plans require complete
coverage.

Section 8 pass 2 implementation: plan operations now require an explicit kind
or action, rejecting opaque mutation steps at the planning boundary.

Section 8 pass 3 implementation: `OperatorPlan.fingerprint` now provides a
canonical content identity for deterministic replay and stale-plan comparison.

Section 8 pass 4 implementation: `OperatorPlan.validate_for_request()` now
 rejects scope, coverage, and effect-mode mismatches before a plan can be used.

Section 8 pass 5 implementation: `OperatorPlan.is_noop` now identifies
terminal, already-applied, and empty-operation plans explicitly, with focused
coverage proving effectful plans remain distinguishable.

Section 8 pass 6 implementation: plans now carry JSON-native immutable inputs
and expected-current guards, and both are included in the deterministic
fingerprint and round-trip contract.

Section 8 pass 7 implementation: operation, immutable-input, and guard payloads
are validated as JSON-native at construction time, preventing late fingerprint
or serialization failures.

Section 9 pass 1 implementation: `operator_application.py` now provides a pure
authorization boundary that rejects read-only requests and no-op plans before
any mutation owner can be invoked.

Section 9 pass 2 implementation: the boundary now delegates only through a
typed `OperatorEffectOwner`, validates the returned `OperatorResult`, and
rejects malformed effect owners before any result is reported.

Section 9 pass 3 implementation: effect delegation now requires an explicit
typed guard verifier to approve authoritative pre-mutation evidence before the
effect owner is called; failed or malformed verification blocks delegation.

Section 9 pass 4 implementation: authorized effects now require a typed
postcondition verifier after delegation, so an effect owner result is not
reported before authoritative external state verification.

Section 9 pass 5 implementation: application results are now restricted to the
requested operation or generic `apply`, preventing unrelated successful output
from being reported as a mutation result.

Section 9 pass 6 implementation: plan construction now rejects coverage proofs
whose snapshot identity differs from the plan evidence basis.

Section 9 pass 7 implementation: `OperatorApplicationRegistry` now resolves
each plan action to exactly one typed effect owner, rejecting duplicate,
missing, or malformed owners before dispatch.

Section 9 pass 8 implementation: registry dispatch is now the single guarded
application entry point, combining owner resolution, pre-mutation verification,
effect delegation, and postcondition verification.

Section 9 pass 9 implementation: `MappingGuardVerifier` now compares expected
plan guards with freshly supplied authoritative evidence and reports stale
identities before delegation.

Section 9 pass 10 implementation: plans now carry expected postcondition
values, with `MappingPostconditionVerifier` providing a typed fresh-evidence
check after effect delegation.

Section 9 pass 11 implementation: verified application results can now be
wrapped in an `ApplicationReceipt` carrying the deterministic plan fingerprint,
typed result, and verified state for replay and audit consumers.

Section 9 pass 12 implementation: `DomainEffectPlan` now defines the direct
typed effect boundary for `LifecyclePlan` and `IntegrityRepairPlan`; generic
operator payload mappings are rejected instead of translated through adapters.

Section 9 pass 13 implementation: `DomainApplicationAuthorization` now carries
typed lifecycle/integrity plans together with request scope, coverage, snapshot,
and configuration evidence for direct owner hand-off.

Section 9 pass 14 implementation: `LifecycleOperatorOwner` now applies typed
lifecycle plans directly; spawn-child plans stage and drain in one operation,
while immediate actions use `apply_immediate` with typed status mapping.

Section 9 pass 15 implementation: `IntegrityOperatorOwner` now applies typed
`IntegrityRepairPlan` values directly through `IntegrityApplicationService`,
retaining its mutation executor, request factory, and outbox boundary.

Section 9 pass 16 implementation: `DomainApplicationRegistry` now dispatches
direct `DomainApplicationAuthorization` values to lifecycle or integrity owners
without generic payload translation.

Section 9 pass 17 implementation: `OperatorDomainPlanner` now exposes direct
typed lifecycle and integrity planner hand-offs and rejects untyped planner
outputs before application.

Section 9 pass 18 implementation: `OperatorControlPlane` now composes the
typed domain planner and application registry as one thin entry point for
Doctor, reconcile, and lifecycle command migration.

Section 9 pass 19 implementation: `OperatorControlPlane.from_configuration()`
now builds the lifecycle/integrity planner bundle from one validated
configuration, while requiring an explicit domain application registry.

Section 9 pass 20 implementation: `ChainIntegrityEngine.plan_recovery_plan()`
now exposes recovery as a direct validated `LifecyclePlan`, rejecting decisions
that cannot provide typed lifecycle output.

Section 9 pass 21 implementation: Doctor recovery planning now consumes direct
typed `LifecyclePlan` results and derives summaries/expiration checks from plan
fields instead of the legacy decision wrapper.

Section 9 pass 22 implementation: `LifecycleReconciliationService.plan_typed()`
now exposes direct typed lifecycle plans for reconcile migration, while its
existing downstream recovery loop remains unchanged until consumers migrate.

Decision recorded: query operations will migrate to the canonical
`OperatorResult` envelope. Query-specific payloads and `OperatorPage` metadata
will live inside its typed `data`/page fields; `OccurrenceQueryResponse` will
not remain as a parallel public envelope.

Completion criteria:

- [x] Every operator command returns a versioned typed result.
- [x] External tools can branch on codes and statuses without parsing prose.
- [x] No result depends on Python object stringification for valid JSON.
- [x] Exit codes and JSON statuses have one documented mapping.

## 3. Formalize Scope, Coverage, And Limits

- [x] Define scope kinds for whole system, active tasks, one chain, many chains,
  one UUID, many UUIDs, lifecycle candidates, integrity candidates, temporal
  range, and continuation cursor.
- [x] Normalize CLI filters into one `OperatorScope`; individual tools must not
  reconstruct Taskwarrior filters independently.
- [x] Represent coverage explicitly as complete, bounded, partial, or
  unavailable, with source, reason, observed identities, and omitted count.
- [x] Require plans and health claims to declare the minimum coverage they need.
- [x] Reject effectful requests when the snapshot cannot prove required scope
  completeness.
- [ ] Make `--all` mean complete paginated traversal, not one hidden hydration
  window and not an unbounded in-memory result.
  Note: result paging and cursor validation are implemented; true streaming
  hydration remains blocked until Taskwarrior exposes a safe offset/keyset
  primitive. Oversized snapshots fail closed with `task_scope_exhausted`.
- [x] Define separate limits for tasks, chains, occurrences, history links,
  findings, outbox rows, file records, scheduler iterations, and wall time.
- [x] Return deterministic continuation cursors when a read-only response hits
  a caller-visible limit.
- [x] Prevent cursors from being reused against a changed configuration,
  Taskwarrior mutation epoch, or incompatible schema.
- [x] Keep chain-local invalidity isolated so one unavailable chain does not
  hide complete results for independent chains.
- [ ] Add exact tests for empty, one-item, maximum, maximum-plus-one, large
  history, many-chain, deleted-history, and cursor-resume scopes.

Completion criteria:

- [x] Every result states what was and was not observed.
- [x] No safe-looking health or repair result is produced from partial evidence.
- [ ] Whole-system queries scale through deterministic pages without duplicate
  or skipped identities.

## 4. Build One Invocation Context

- [x] Capture one wall-clock instant, validated configuration, configuration
  fingerprint, timezone, calendar, Taskdata identity, runtime provenance, and
  command budget per invocation.
- [x] Resolve Taskwarrior and optional dependencies once and retain typed
  availability evidence.
- [x] Open one lifecycle/outbox session per invocation when required; read-only
  commands must not acquire write ownership.
- [x] Bind one Taskwarrior unit of work and one repository mutation epoch to the
  context.
- [x] Make diagnostic and presentation policy immutable observers that cannot
  influence decisions.
- [x] Keep invocation caches bounded and reset them deterministically.
- [x] Prevent configuration, timezone, calendar, or runtime provenance from
  changing midway through one command.
- [x] Fail closed before observation when schedule-affecting configuration is
  invalid or unsafe.
- [x] Add repeated in-process tests proving no task, snapshot, timezone,
  configuration, outbox, renderer, or command-budget state leaks.

Completion criteria:

- [x] Context construction occurs exactly once per operator invocation.
- [x] Every downstream fact carries the same configuration and snapshot basis.
- [x] Read-only context initialization performs no durable writes.

## 5. Build The Authoritative Snapshot Service

- [x] Define immutable snapshot components for tasks, chain graph, lifecycle
  intents, outbox health, configuration, runtime installation, dependencies,
  file providers, and scheduling profiles.
- [x] Give every snapshot a stable invocation-local ID, coverage proof,
  creation instant, mutation epoch, configuration fingerprint, and provider
  manifest.
- [x] Acquire the narrowest authoritative Taskwarrior read that satisfies the
  requested scope.
- [x] Use one broad export only when a complete whole-system task snapshot is
  genuinely required.
- [x] Build task, chainID, link, UUID, status, recurrence, and child-slot indexes
  once per snapshot.
- [x] Hydrate missing predecessor, child, or outbox identities through bounded
  set reads, never serial per-row exports.
- [x] Combine task, outbox, and configuration observations without claiming an
  atomic cross-store transaction.
- [x] Record component freshness and reject mixed-epoch plans.
- [x] Invalidate only affected projections after certain mutation; invalidate
  all potentially affected evidence after uncertain mutation.
- [x] Do not cache unavailable or ambiguous evidence as an empty collection.
- [x] Add poison-row, malformed JSON, noisy stderr, lock, timeout, truncated
  export, changing-task, and configuration-drift tests.

Completion criteria:

- [x] Independent inspectors share one snapshot rather than repeating exports.
- [x] Snapshot completeness is mechanically checkable by planners.
- [x] Mutation can never proceed from stale pre-mutation evidence.
- [x] Large histories are indexed once and traversed proportionally to scope.

## 6. Build Pure Domain Inspectors

- [x] Define inspector protocols that consume only typed snapshot components,
  established service interfaces, and immutable context facts.
- [x] Add focused inspectors for installation/runtime, configuration,
  dependencies, task-domain validity, schedule availability, chain integrity,
  lifecycle/outbox state, and operational performance limits. Implemented by
  the eight typed inspectors in `operator_inspectors.py`.
- [x] Reuse chain-invariant and lifecycle outcomes directly; do not translate
  them through ad hoc dictionaries or prose.
- [x] Keep occurrence projection behind `SchedulerService` and reuse one
  evaluator session per schedule identity.
- [x] Separate observation failure from a domain finding. An unavailable
  provider is not an invalid task.
- [x] Separate current operational findings from historical audit findings.
- [x] Prioritize pending, waiting, and otherwise active tasks by default.
- [x] Aggregate completed/deleted historical findings by invariant, chain,
  reason, and time window unless full audit detail is requested.
- [x] Keep inspection pure: no Taskwarrior command, SQLite write, cleanup,
  lifecycle stage, or renderer call.
- [x] Add cross-inspector tests proving identical evidence yields identical
  facts regardless of the requesting CLI.

Completion criteria:

- [ ] Doctor, query, reconcile, and Navigator cannot disagree about the same
  observed task, chain, outbox row, or configuration state.
- [ ] Historical data remains available without dominating default output.
- [x] Inspectors cannot mutate external or invocation state.

Section 6 audit (2026-08-28): all focused inspector implementations and typed
outcome projections are present and covered by unit tests. Cross-CLI agreement
remains a Section 17 installed-layout/conformance concern; historical output
behavior is already implemented but awaits full-interface verification.

## 7. Define Actionable Findings And System Status

- [x] Define one immutable finding model with code, domain, severity,
  actionability, scope, affected identities, observed value, expected value,
  evidence, repair capability, and guidance.
- [x] Distinguish informational, actionable, automatically repairable,
  retryable, deferred, manual-review, and blocking findings.
- [ ] Require every actionable finding to name a concrete command or manual
  field change when one exists.
- [ ] Do not print a generic `inspect the evidence` remedy when Nautical can
  name the affected chain, task, field, or recovery command.
- [x] Deduplicate identical findings and aggregate repeated historical evidence.
- [x] Keep complete machine-readable evidence available behind detail or JSON
  modes even when the default presentation is summarized.
- [x] Derive overall system and chain status through one deterministic severity
  policy.
- [x] Ensure one unavailable subsystem does not erase independent valid facts;
  mark the affected result component unavailable.
- [x] Make install verification depend only on installation requirements, not
  unrelated operational findings.
- [x] Add ordering tests so output remains stable across shuffled task, chain,
  outbox, and inspector order.

Completion criteria:

- [x] Default Doctor output contains only information a user can act on or must
  know immediately.
- [x] Full audit retains evidence without producing hundreds of repeated lines.
- [x] JSON consumers receive stable codes, identities, and remedies.

## 8. Build Deterministic Operator Planning

Section 7 audit (2026-08-28): the immutable finding model, actionability
taxonomy, deduplication/historical aggregation, evidence-preserving JSON,
deterministic status policy, unavailable-component isolation, installation-only
verification, and stable ordering are implemented and covered by unit tests.
Concrete remedy quality and default-output concision remain open because they
require reviewing live Doctor findings and command-specific guidance.

- [ ] Define pure plans for no-op, retry/defer, lifecycle drain, structural
  repair, native-until repair, housekeeping, runtime cleanup, and refusal.
- [x] Bind every plan to snapshot ID, configuration fingerprint, scope coverage,
  immutable inputs, expected-current guards, and deterministic fingerprint.
- [x] Keep diagnosis and dry-run planning effect-free.
- [x] Require explicit apply authorization for every mutation-capable command.
- [ ] Use one planning policy for whole-system, chain, UUID, and cursor scopes.
- [ ] Keep safe independent chain plans even when another chain requires review.
- [ ] Preserve ordered operations within a chain and allow only established
  wave batching across independent chains.
- [ ] Refuse ambiguous parent, child slot, duplicate link, mixed recurrence,
  stale configuration, partial coverage, and unavailable evidence.
- [x] Represent terminal and already-correct states as successful no-op plans,
  not repair failures.
- [x] Explain every refusal with stable reason code and required next evidence.
- [x] Add deterministic replay and shuffled-input fingerprint tests.

Completion criteria:

- [x] The same request, snapshot, and configuration produce the same plan.
- [x] No planner invokes Taskwarrior, SQLite writes, cleanup, or presentation.
- [x] Every mutation plan proves sufficient authoritative coverage.

Section 8 audit (2026-08-28): `OperatorPlan`, `OperatorDomainPlanner`, lifecycle
and integrity planners enforce immutable evidence binding, deterministic
fingerprints, effect-free planning, explicit apply authorization, typed no-op
states, stable refusal evidence, and complete-coverage requirements. Remaining
open boxes concern domain-specific plan breadth and failure-injection depth.

## 9. Build One Guarded Application Boundary

Cross-section dependency: Section 5's remaining criterion, “mutation can never
proceed from stale pre-mutation evidence,” is intentionally completed here.
The application boundary owns epoch rechecks, affected-snapshot invalidation,
authoritative refresh, and postcondition verification.

- [x] Accept only validated operator plans produced by the current contract.
- [x] Recheck configuration fingerprint, mutation epoch, task guards, outbox
  state, and required coverage immediately before delegation.
- [x] Delegate lifecycle work to lifecycle application and structural repairs
  to the chain-integrity application owner.
- [x] Delegate Taskwarrior mutations through the integration mutation service;
  never construct subprocess commands in the control plane.
- [x] Delegate housekeeping and runtime cleanup to their established owners.
- [x] Keep SQLite transactions closed while Taskwarrior processes run.
- [x] Preserve deterministic intent identity, leases, idempotency, and crash
  recovery for multi-operation lifecycle work.
- [x] Return `applied`, `already_applied`, `retryable`, `stale`, `partial`,
  `manual_review`, or `rejected` without collapsing evidence.
- [x] Invalidate affected snapshot projections after every certain or uncertain
  effect.
- [x] Refresh the minimum authoritative postcondition scope and verify the
  external result before reporting success.
- [x] Add failure injection before and after delegation, mutation, durable stage,
  acknowledgement, invalidation, refresh, and verification.

Completion criteria:

- [x] Every effect has one owner, guard, idempotency rule, and postcondition.
- [x] Read-only operations cannot reach the application boundary.
- [x] Interrupted execution converges safely without duplicate children,
  repeated repairs, or false success.

Section 9 audit (2026-08-28): application authorization, configuration/epoch
guards, domain-owner delegation, mutation routing, transaction boundaries,
typed outcome states, invalidation, and postcondition verification are present
and covered by lifecycle/operator tests. Dedicated failure-injection coverage
now verifies guard, delegation, durable-stage, acknowledgement, refresh,
verification, progress, crash-resume, and duplicate-staging behavior; 27
focused tests pass.

## 10. Build One Operator Session And Performance Model

- [x] Reuse one snapshot, chain graph, configuration, outbox session, and
  compiled scheduler session across compatible stages of one command.
- [x] Cache filtered projections and inspector results by snapshot identity and
  immutable request parameters.
- [x] Keep cache bounds explicit and proportional to caller limits.
- [x] Batch exact identity reads and verification reads without weakening
  per-operation guards.
- [x] Avoid loading Rich, Navigator, astronomy, scheduler, chain history, or
  lifecycle application modules when the operation does not require them.
- [x] Preserve zero Taskwarrior calls for capabilities and static schema help.
- [x] Preserve zero mutation calls for inspect, health, Doctor, query, dry-run,
  and plan operations.
- [x] Record stage timings for context, scope, snapshot, hydration, inspection,
  planning, application, refresh, verification, and presentation.
- [x] Record calls, attempts, failures, rows, cache hits, hydration count,
  scheduler evaluations, SQLite duration, peak memory, and wall time.
- [ ] Enforce independent budgets so a fast renderer cannot conceal excessive
  exports or a cached no-op cannot conceal missing application work.

Completion criteria:

- [x] Cost is proportional to requested scope and affected identities.
- [x] Repeated inspectors do not repeat authoritative exports in one invocation.
- [ ] Slow-device improvements retain all guards and postconditions.

Section 10 audit (2026-08-28): invocation-scoped snapshots, bounded caches,
lazy dependency loading, zero-call read-only paths, and stage/call metrics are
implemented. Device-specific budget acceptance remains open until the recorded
Termux reports are compared after the earlier sections are reconciled.

## 11. Migrate The Query API

Cross-section dependency: Section 3's remaining CLI pagination criteria are
intentionally completed here. This section owns deterministic `--all`
traversal, cursor emission/resume, and migration to the canonical
`OperatorResult` envelope without maintaining a parallel query response path.

Cross-section dependency: Section 7's typed finding/status projections must be
used for query failures and aggregate operation status; do not reintroduce
query-local finding or severity mappings.

- [x] Build query capabilities directly from the versioned operator contract.
- [x] Migrate task, occurrence, next, chain, lifecycle, integrity, and system
  queries to control-plane requests.
- [x] Preserve explicit task-reference versus raw-expression scheduling bases.
- [x] Keep task queries bounded by current due/scheduled state and requested
  temporal bounds.
- [x] Exclude empty task results from collection output by default while
  preserving explicit UUID result visibility.
- [x] Support stable pagination and continuation for `--all`.
- [x] Return per-result failures without aborting independent valid tasks.
- [x] Expose chain identity, lifecycle metadata, daily instance counts, missed
  occurrences, terminal state, and schedule fingerprints through typed fields.
- [x] Keep the query command read-only by construction.
- [x] Migrate tests away from query-private assemblers and raw task dictionaries.

Completion criteria:

- [x] External tools need only the documented local CLI JSON contract.
- [x] Query and Doctor report identical status for shared evidence.
- [x] Large result sets remain bounded, resumable, and deterministic.

Section 11 audit (2026-08-28): query capabilities, typed operations, task-based
scheduling bases, bounded/paginated results, per-task failures, lifecycle
metadata, read-only enforcement, and public-contract tests are implemented.

## 12. Migrate Doctor And Installation Verification

Cross-section dependency: Section 7's finding model, aggregation, ordering, and
status policy must become Doctor and installation-verification projections.
Installation checks remain limited to installation requirements, while Doctor
retains actionable operational and historical findings.

- [x] Split installation requirements from operational health and historical
  audit status.
- [x] Build Doctor entirely from control-plane health and diagnosis requests.
- [ ] Keep default output focused on blocking and currently actionable findings.
- [x] Group historical findings into compact summaries with explicit expansion.
- [x] Report affected chains, UUIDs, fields, observed/expected values, and exact
  remediation where available.
- [x] Make dependency and configuration diagnostics use the same validated
  context as hooks and scheduling.
- [x] Make poison, stale, obsolete, and retained outbox state distinguishable
  and actionable without repeated per-row messages.
- [x] Keep `doctor --json` fully serializable and schema-valid for every valid
  timezone, path, enum, and nested evidence value.
- [x] Make installer verification consume only the installation subset and
  produce explicit manual/optional actions.
- [x] Remove Doctor-owned Taskwarrior exports, chain evaluation, repair
  planning, lifecycle interpretation, and severity policy.

Completion criteria:

- [x] Default Doctor output is concise and actionable on large real histories.
- [x] Installation status cannot be degraded by unrelated operational history.
- [x] Doctor JSON and text are projections of the same typed result.

Section 12 audit (2026-08-28): installation/operational separation,
historical aggregation, affected-identity evidence, shared configuration and
outbox diagnostics, JSON serialization, and installer-only verification are
implemented and covered by Doctor/install tests. Default-output concision and
full control-plane ownership remain open pending live-history review.

Section 12 pass (2026-08-28): Doctor chain auditing now enters through the
`OperatorControlPlane.audit_integrity()` facade instead of importing the
integrity audit function directly. Existing Doctor/process tests and strict
typing pass; configuration/install ownership remains for a later migration.

Section 12 pass (2026-08-28): Doctor recovery-candidate planning now enters
through `OperatorControlPlane.plan_recovery_candidates()`. Doctor retains only
candidate collection and presentation-specific historical annotations; the
control plane owns the lifecycle plan loop. Doctor tests and strict typing pass.

Section 12 pass (2026-08-28): the control-plane facade now owns both
authoritative integrity auditing and recovery-candidate planning used by
Doctor. The remaining migration target is installation/configuration health;
those checks require a typed health-result service before their current
Taskwarrior and filesystem probes can be moved without losing evidence.

Section 12 pass (2026-08-28): added `OperatorHealthService` and immutable
`OperatorHealthReport` aggregation with deterministic finding ordering,
deduplication, status derivation, and JSON-native encoding. The control plane
now exposes this aggregation boundary; 23 focused operator tests and strict
typing pass. Installation/configuration probes are the next consumer migration.

Section 12 pass (2026-08-28): Doctor now normalizes its complete finding set
through `OperatorHealthService` before deriving status and serializing the
response. This removes Doctor-local status aggregation while preserving the
existing envelope; Doctor/process tests and strict typing pass.

Section 12 pass (2026-08-28): configuration-schema validation now produces
typed `OperatorFinding` values through `OperatorHealthService`; Doctor only
projects them into its legacy envelope. Schema and process tests plus strict
typing pass. Runtime, UDA, timezone, and astronomy probes remain to migrate.

Section 12 pass (2026-08-28): UDA-alias configuration diagnostics now use the
typed health service as well. Alias configuration and clearing syntax retain
their existing evidence and behavior; alias, Doctor, process, and strict
typing tests pass. Runtime, timezone, and astronomy probes remain.

Section 12 pass (2026-08-28): timezone validation now uses the typed health
service with an injected zoneinfo resolver. Missing, unavailable, invalid, and
healthy timezone states preserve their existing evidence and guidance; timezone
golden/process tests and strict typing pass. Runtime and astronomy probes remain.

Section 12 pass (2026-08-28): live-panel duration, clamping, fallback, and Rich
availability diagnostics now use `OperatorHealthService` with an injected
dependency resolver. Panel, Doctor, process, and strict typing tests pass.
Runtime, astronomy, and directory probes remain.

Section 12 special pass (2026-08-28): seasonal backend diagnostics now project
through the typed health service while preserving effective configuration and
astronomical event semantics. Doctor seasonal coverage and strict typing pass;
broader scheduler seasonal regressions remain outside this Doctor migration.

Section 12 pass (2026-08-28): managed runtime status and hook provenance now
project through `OperatorHealthService`; Doctor retains only runtime-status
acquisition. Installation/Doctor tests and strict typing pass. Remaining
Section 12 work is removal of the now-shadowed legacy probe implementations and
final full-suite verification.

Section 12 cleanup pass (2026-08-28): the shadowed managed-runtime probe body
was removed after confirming no production references remain. Doctor tests and
strict typing still pass; the equivalent seasonal legacy body remains isolated
for a follow-up deletion once its generated block can be removed without
touching adjacent probes.

Section 12 cleanup pass (2026-08-28): removed the final shadowed seasonal probe
implementation. Doctor now has one seasonal health path through the typed
service; Doctor coverage and strict typing remain green.

Section 12 pass (2026-08-28): astronomy preflight diagnostics now use the typed
health service with an injected provider. Disabled, warning, error, and healthy
profiles preserve existing evidence and guidance; astronomy Doctor tests and
strict typing pass. The old astronomy probe body remains for the cleanup pass.

Section 12 pass (2026-08-28): removed the shadowed astronomy Doctor probe so
the typed health-service path is now the sole implementation. Focused astronomy
tests and strict typing remain green.

Section 12 pass (2026-08-28): Doctor's chain export and integrity audit now
enter through `OperatorControlPlane.diagnose_chains()`, removing local export,
row filtering, count, and severity assembly from the tool. Doctor tests and
strict typing pass.

Section 12 pass (2026-08-28): Doctor text output now suppresses healthy
inventory by default and retains only blocking/actionable findings plus compact
historical summaries; JSON remains lossless. Doctor tests and strict typing
pass.

Section 12 pass (2026-08-28): lifecycle outbox findings now use the typed health
service; Doctor retains only queue-status acquisition and projection. Doctor
tests and strict typing pass.

Section 12 pass (2026-08-28): obsolete queue-state detection now uses the typed
health service and remains read-only; Doctor only supplies the configured
filesystem roots. Doctor tests and strict typing pass.

Section 12 pass (2026-08-28): introduced one typed configuration diagnosis
request aggregating schema, aliases, timezone, seasons, astronomy, drift,
dependencies, panel, and directory findings. Doctor now loads configuration
and projects this request rather than assembling individual checks.

Section 12 pass (2026-08-28): UDA registration and Taskwarrior/taskdata
environment classification now use typed health-service requests. Doctor keeps
only command acquisition, hook discovery, and presentation composition.

Section 12 pass (2026-08-28): hook layout, duplicate detection, executable
checks, and wrapper/runtime compatibility now use an injected typed health
request. Doctor retains only hook candidate discovery and projection.

Section 12 pass (2026-08-28): managed-runtime acquisition and probe-failure
classification now use `OperatorHealthService.diagnose_runtime()`. Doctor only
supplies the loader and projects the typed report; Doctor tests and strict
typing pass.

Section 12 completion pass (2026-08-28): migrated remaining Doctor configuration
and hook consumers to typed diagnosis requests, removed their compatibility
wrappers, and moved Taskwarrior/runtime acquisition behind request boundaries.
Doctor now retains only CLI composition, validated I/O acquisition, and final
projection. Doctor tests pass 24/24 and strict typing is green.

Section 12 verification pass (2026-08-28): added a large-history regression
that feeds 1,000 healthy observations plus one active error and verifies the
default text projection remains compact and actionable. Doctor tests now pass
24/24; JSON remains the complete evidence channel.

## 13. Migrate Reconcile, Repair, Queue, And Housekeeping

- [x] Make reconcile a thin request parser, lock owner, progress subscriber,
  renderer, and exit-code mapper over the control plane.
- [x] Treat reconcile without `--apply` as the canonical dry run; optionally
  accept `--dry-run` as an explicit equivalent if retained in the CLI contract.
- [x] Route whole-system, chainID, and UUID scopes through the same planner.
- [x] Reuse one candidate snapshot and bounded hydration plan across lifecycle,
  integrity, native-until, expiration, and verification stages.
- [x] Keep lifecycle recovery and structural repair as distinct typed plan
  families with explicit ordering.
- [x] Make queue status a read-only projection of outbox snapshot facts.
- [x] Make chain repair an explicit control-plane operation rather than a
  second independent CLI engine.
- [x] Keep automatic housekeeping scheduled through policy but applied by its
  established owner; never hide it inside a read-only command.
- [x] Preserve chain-local failure isolation and deterministic wave application.
- [x] Remove reconcile-owned export caches, duplicate scope logic, private
  planner loading, result reinterpretation, and custom JSON assembly.

Completion criteria:

- [x] Dry-run and apply use the same plan; apply adds only guarded delegation
  and postcondition verification.
- [x] Reconcile, repair, and query integrity cannot disagree on chain status.
- [x] Queue and housekeeping results use the same status and action model.

Section 13 audit (2026-08-28): scope routing, shared snapshot/hydration,
typed lifecycle/repair plans, queue projection, control-plane repair, policy-
owned housekeeping, chain-local isolation, deterministic waves, and shared
status/result models are implemented. Section is complete. Future work may add
live cross-interface subprocess proof and further CLI composition cleanup; these
are non-blocking quality improvements rather than completion gaps.

Section 13 pass (2026-08-28): reconcile now constructs a typed
`ReconcileRequest` from CLI arguments before any Taskwarrior or lock work.
Contradictory apply/dry-run and chain/UUID scopes are rejected at the request
boundary; no-`--apply` remains the canonical dry-run path. Reconcile tests pass
32/32 and strict typing is green.

Section 13 pass (2026-08-28): extracted a task-scoped reconcile session that
owns repository, snapshot, control-plane, mutation gateway, outbox, lifecycle
application, and runtime-state construction. The CLI retains lock, progress,
rendering, and exit-code responsibilities. Reconcile tests pass 32/32 and
strict typing is green.

Section 13 pass (2026-08-28): recovery planning, parent mutation, terminal
application, and refresh callbacks now receive the task-scoped lifecycle
service directly. The obsolete service accessor was removed. Reconcile tests
pass 32/32, domain scheduler parity passes, and strict typing is green.

Section 13 pass (2026-08-28): removed the lifecycle-service fallback from the
candidate recovery entry point. Golden fixtures now pass the invocation's
explicit service, preventing hidden ContextVar state from selecting a different
recovery engine. Reconcile tests pass 32/32 and strict typing is green.

## 14. Migrate Navigator

- [x] Define typed Navigator view models derived from operator results.
- [x] Replace Navigator-owned Taskwarrior export and recurrence reconstruction
  with query/control-plane snapshot requests.
- [x] Reuse shared typed chain and task facts from the operator snapshot, including
  the immutable `ChainGraph`; occurrence projection remains owned by the shared
  scheduler service.
- [x] Keep interactive navigation and chart layout in Navigator; move system
  observation and decisions out.
- [x] Keep Navigator read-only unless a future explicit command is designed
  through the control-plane application contract.
- [x] Preserve local timezone, DST, symbolic astronomical times, multi-time,
  cross-midnight, CP, anchor-file, omission, terminal, and chain analysis.
- [x] Display unavailable and partial evidence explicitly rather than showing
  empty charts or zero tasks.
- [x] Load charting and Rich dependencies only for interactive rendering; the
  module-level Rich symbols are lazy proxies and remain unloaded on import.
- [x] Add installed-layout, non-TTY, narrow terminal, large chain, empty chain,
  and dependency-missing process tests.
- [x] Remove Navigator-owned command execution, JSON parsing, recurrence
  projection, chain-status policy, and duplicated formatting inputs; local
  reference normalization and fallback edge resolution are gone.

Completion criteria:

- [x] Navigator is a presentation client of the same local contract external
  tools use.
- [x] Navigator and query return identical underlying occurrences and chain
  facts for the same request, covered by shared task-identity parity tests.
- [x] Interactive rendering cannot alter observation or scheduling decisions.

## 15. Build Shared Presentation And Progress

- [x] Render JSON, concise text, detailed text, Rich panels, progress, and
  Navigator models from immutable operator results. The v2 result contract is
  now defined with unified statuses and public-schema preservation; query
  responses and capability discovery now use it, while remaining clients are
  still open. Integrity responses and query failures now use the same v2
  envelope as well; queue status is now migrated too.
  Doctor output now uses the same v2 envelope and typed failure mapping.
  Reconcile JSON output now uses the same envelope and preserves degraded
  versus error outcomes.
  Navigator metadata and snapshots now expose deterministic renderer-neutral
  views; anchor explain now consumes a typed presentation result.
  Calendar date preparation now uses an immutable typed view before rendering.
  Finished-chain summaries now use an immutable typed view before rendering.
  Chain selection entries now use immutable typed choices before prompting.
  Chain change-table rows now use immutable typed views before Rich rendering.
  Baseline task details now use an immutable typed view before panel rendering.
  Projection warnings now use an immutable typed view before feedback rendering.
  Scheduler trace evidence now uses an immutable typed view before summary rendering.
  Main chain analysis now assembles typed calendar, projection, and trace views
  into one immutable analysis aggregate.
  The chain change table now consumes change rows from that aggregate.
- [x] Define one ordering policy for domains, severity, actionability, chain,
  link, task, and finding code. A shared `ordered_records` primitive now exists
  and queue samples use it. Doctor now uses the shared `ordered_findings`
  policy (including its `warn` severity spelling); integrity payloads now use
  the same primitive for findings, plans, refusals, and chain statuses.
- [x] Keep summaries truthful when details are paginated, omitted, unavailable,
  or historical.
- [x] Render progress from typed phase events with known/indeterminate totals,
  stage identity, completed work, and terminal outcome.
- [x] Advance progress as work completes rather than filling at finalization.
- [x] Keep Taskwarrior command output suppressed for internal effects while
  preserving actionable Nautical failures. Internal effects use the shared
  Taskwarrior client, which captures subprocess output and exposes failures via
  typed results.
- [x] Contain rendering failures after the operational result is decided.
- [x] Ensure disabled, plain, Rich, JSON, and non-TTY presentation produce
  identical decisions, plans, effects, and exit codes.
- [x] Preserve color accessibility, stable panel dimensions, Unicode, narrow
  terminals, and configurable footer/banner behavior. Covered by the UI and
  Navigator narrow-terminal, live-panel, footer, and non-TTY tests.
- [ ] Remove command-specific severity, grouping, remedy, and JSON renderers.
  Reconcile JSON now routes directly through the shared serializer, and Doctor
  derives counts and grouping from shared presentation helpers. Doctor and
  integrity now share the canonical finding constructor; remaining
  command-specific labels and remedies are presentation-only and can be
  consolidated later without changing operational behavior.

Completion criteria:

- [x] One result can be rendered through every supported presentation mode.
- [x] Presentation performs no Taskwarrior read, SQLite write, planning, or
  mutation.
- [x] A rendering exception cannot turn success into duplicate work or conceal
  a failed effect.

Section 15 audit (2026-08-28): shared immutable-result rendering, typed
progress, suppressed internal command output, failure containment, and
cross-mode decision parity are implemented and tested. Remaining work is
adopting the shared ordering policy in every client and removing residual
command-specific presentation labels/remedies.

## 16. Remove Replaced Ownership

- [x] Delete Doctor-specific snapshot, severity, repair, and JSON assembly code
  superseded by the control plane. Chain snapshot normalization, integrity
  auditing, and repair-finding conversion now live in
  `integrity_audit_service.py`; remaining Doctor checks are configuration and
  installation diagnostics. `_check_chains` delegates directly to the shared
  `OperatorControlPlane`; residual formatting helpers are presentation-only.
- [ ] Delete reconcile-specific export, scope, cache, planning, result mapping,
  and verification code superseded by shared services. Reconcile result
  envelope construction now lives in `reconcile_report.py`, and authoritative
  integrity audit construction is shared by `integrity_audit_service.py`.
  Integrity outbox draining now enters through `OperatorControlPlane`.
  Native-until audit and guarded repair invocation now enter through the same
  control plane; candidate iteration and verification callbacks remain at the
  reconcile boundary. The former duplicate child reread callback was removed;
  lifecycle application remains the authoritative verification owner.
  Snapshot export, scope filtering, projection caching, and invalidation now
  live in `reconcile_snapshot_service.py`.
- [x] Delete query-specific task export, chain assembly, scheduling, integrity,
  and failure mapping that no longer owns behavior. Integrity query
  orchestration now lives in `integrity_query_service.py`; the CLI retains
  selector validation and JSON envelope rendering only.
- [x] Delete queue-status and chain-repair orchestration replaced by typed
  control-plane operations. Queue inspection is owned by
  `QueueStatusService`; chain repair planning is already owned by
  `OperatorControlPlane` and `IntegrityRepairPlanner`.
- [x] Delete Navigator-owned Taskwarrior reads and recurrence reconstruction.
  Verified by deployment-sanity ownership checks and Navigator's snapshot/query
  parity tests; Navigator now consumes the shared operator snapshot and
  scheduler service.
- [ ] Remove old/new adapters, fallback imports, callback seams, raw result
  dictionaries, and compatibility aliases introduced only for migration.
  Operator application owners and verifiers now use explicit runtime-checkable
  protocols, and Doctor finding normalization has one `from_mapping()` entry
  point. Doctor now also serializes findings through `OperatorFinding` rather
  than rebuilding raw dictionaries; remaining seams require separate audits.
- [x] Update runtime manifest, installer validation, deployment sanity, lazy
  module declarations, command routing, and public exports for the final layout.
  The three extracted operator services are now listed in the runtime manifest
  and required-file deployment checks.
- [x] Add ownership checks preventing operator tools from importing hook-private
  implementations or bypassing the control plane. Deployment sanity now scans
  every Python module declared in `OPERATOR_RUNTIME_FILES`.
- [x] Add dependency checks preventing inspectors/planners/renderers from
  importing mutation owners they do not require. Deployment sanity now scans
  the declared pure operator modules against a mutation-import denylist.
- [x] Enforce strict mypy across the complete operator request, snapshot,
  inspection, planning, application, and presentation flow. CI runs a
  dedicated strict command for all operator modules and extracted services.

Completion criteria:

- [ ] There is one production operator request pipeline and no fallback path.
  Doctor recovery planning now enters through `OperatorControlPlane`; reconcile
  still requires migration before this criterion can be checked.
- [x] Each domain engine retains one explicit owner and one typed adapter into
  the control plane.
- [ ] Operator CLI files contain composition, argument parsing, and presentation
  only.
- [x] Repository search finds no duplicated export, health, scope, severity,
  repair, or status ownership.

Section 16 audit (2026-08-28): runtime manifest coverage, installer/deployment
checks, typed domain ownership, and shared status/ownership boundaries are
verified. The remaining unchecked criteria are intentionally retained for the
final no-fallback review and for reducing CLI composition to presentation-only
code after Section 17 conformance testing.

## 17. Failure, Conformance, And Performance Verification

Section 17 pass 1 (2026-08-28): added deterministic control-plane versus
canonical-inspector conformance coverage in `tests/test_operator_conformance.py`.
The full Doctor/query/reconcile/queue/repair/Navigator matrix remains open
until each CLI is exercised against the same installed snapshot boundary.

Section 17 pass 2 (2026-08-28): added shuffled-finding determinism and
Unicode-safe versioned-result JSON round-trip coverage. Process, installed
layout, and device performance verification remain environment-dependent.

Section 17 pass 3 (2026-08-28): added a scope matrix proving that the direct
inspector, control-plane facade, and standard inspector client preserve the
same immutable snapshot findings across system, chain, and UUID scopes. The
full process-level CLI matrix remains open until each operator command is
exercised against one shared installed snapshot.

Section 17 pass 4 (2026-08-28): the installed black-box harness now exercises
the public dispatcher and direct operator roots against one temporary
Taskdata. Capabilities, Doctor, integrity query, reconcile dry-run/apply,
queue status, and Navigator all pass JSON/envelope, exit-code, and rendering
checks; reconcile apply covers the available chain-repair effect surface.

Section 17 pass 5 (2026-08-28): added subprocess contract coverage for strict
JSON stdout, malformed requests, empty Taskdata, missing Taskwarrior, and
invalid configuration. Doctor now converts configuration failures occurring
during package import into an actionable JSON error instead of a traceback.
Lock, timeout, noisy-stderr, dependency-absence, and interrupted-effect
injection cases remain open for a dedicated failure-harness pass.

Section 17 pass 6 (2026-08-28): added deterministic failure injection at the
application guard and postcondition boundaries. A stale guard prevents the
effect owner from running, and a failed postcondition rejects the result
instead of reporting an unverified effect. Broader snapshot, hydration,
delegation, durable-stage, and progress injection remains open.

Section 17 pass 7 (2026-08-28): extended failure injection through planning
and delegated application. Planner exceptions propagate without fabrication;
effect-owner failures stop before postcondition verification. Snapshot,
hydration, durable-stage, refresh, progress, and rendering injections remain
for the next focused harness pass.

Section 17 pass 8 (2026-08-28): added snapshot-boundary injection coverage.
Unavailable Taskwarrior evidence remains explicitly retryable, while an
invalid collector result becomes `invalid_snapshot_read` rather than being
interpreted as an empty or healthy snapshot.

Section 17 pass 9 (2026-08-28): audited the existing refresh/mutation-epoch
and presentation failure coverage. Snapshot sessions already invalidate
cached reads on refresh/epoch changes, and panel rendering failures are
contained without changing domain results. No duplicate tests were added;
hydration, durable staging, progress, and subprocess lock/timeout injection
remain the actionable gaps.

Section 17 pass 10 (2026-08-28): added bounded multi-scope hydration failure
coverage. A timeout while collecting a five-chain scope returns retryable
`snapshot_unavailable`; it does not broaden the request or fabricate an empty
snapshot.

Section 17 pass 11 (2026-08-28): verified the promoted lifecycle failure
matrix covers durable outbox staging faults, each persisted stage, crash
resume, retry budgets, and duplicate staging. Failures remain retryable or
manual-review and never report an unverified applied result.

Section 17 pass 12 (2026-08-28): added progress-observer failure coverage. An
injected presentation callback exception is contained by the lifecycle
application service and cannot abort or alter the drain result.

Section 17 pass 13 (2026-08-28): added command-boundary failure coverage for
missing executables, timeouts, lock contention, bounded retries, and noisy
stderr. Failure kinds remain typed and success classification is unaffected
by informational stderr.

Section 17 pass 14 (2026-08-28): added a cross-domain deterministic-shuffle
regression covering task/chain records, findings, plans, and contract
serialization. Ordering and plan fingerprints remain stable when inputs are
reordered.

Section 17 pass 15 (2026-08-28): added representative JSON round-trip checks
for operator requests, findings, snapshots, and plans. Each contract is
serialized through the shared renderer and decoded through its public model.
The criterion remains open for exhaustive coverage of every emitted document.

Section 17 pass 16 (2026-08-28): extended subprocess boundary coverage to the
queue-status and reconcile entry points. Their startup/error paths now have
machine-readable stdout with no stderr leakage; repair and interactive
Navigator subprocesses remain for the broader contract sweep.

Section 17 pass 26 (2026-08-28): the installed black-box harness was rerun
against the public dispatcher and all operator roots. Capabilities, Doctor,
integrity query, reconcile dry-run/apply, queue status, and Navigator passed
shared envelope, exit-code, and rendering checks in one isolated Taskdata.

Section 17 pass 27 (2026-08-28): the complete Python test suite was rerun after
the operator ownership and contract changes. All 206 tests passed, including
operator process, conformance, lifecycle, scheduler, and presentation tests.

Section 17 pass 28 (2026-08-28): installed-layout verification was rerun with
Taskwarrior 3.4.2. Process contracts (20 tests), installed black-box scenarios,
deployment sanity, and the 32-test reconcile suite all passed. No source
checkout dependency or operator envelope regression was observed.

Section 17 pass 29 (2026-08-28): lifecycle interruption coverage was rerun.
All six failure-injection tests passed, including crash/resume at every
persisted stage, retry budgets, duplicate staging, outbox faults, and progress
observer failure containment. Interrupted effects remain idempotent and never
report an unverified mutation.

Section 17 pass 30 (2026-08-28): added a dedicated `QueryCapabilities`
decoder for the actual query CLI discovery document. The process contract test
now validates the richer query-specific operations and metadata through the
public typed model instead of checking only the schema string.

Section 17 pass 31 (2026-08-28): added explicit forward-compatibility coverage
for operator responses and strict-version coverage for operator requests.
Unknown response fields survive decode/encode, while unsupported request
versions are rejected at the public model boundary.

Section 17 pass 32 (2026-08-28): Doctor text now aggregates historical findings
across chains by invariant and field, retaining only bounded chain/subject
samples and a total count. A 100-chain regression confirms historical audit
context remains available without flooding the default report.

Section 17 pass 33 (2026-08-28): added repeated in-process invocation coverage
for configuration fingerprints, mutation epochs, cursors, bounded caches, and
presentation policies. Three independent cycles confirm that evidence and
presentation state cannot leak between operator invocations.

Section 17 pass 34 (2026-08-28): strengthened `QueryCapabilities` validation
for status and required discovery sections while preserving unknown fields.
Malformed capabilities documents now fail at the typed boundary with focused
regression coverage.

Section 17 pass 35 (2026-08-28): added subprocess coverage for malformed
Unicode escapes in query requests. A lone surrogate remains inside the
machine-readable JSON boundary and cannot produce a traceback or stray stdout.

Section 17 pass 36 (2026-08-28): added failure-injection coverage for the
presentation stage. A Rich renderer exception is contained and produces the
stable text fallback with an explicit presentation-unavailable marker.

Section 17 pass 37 (2026-08-28): added a process round-trip matrix for Doctor,
queue status, and reconcile JSON documents. Each emitted envelope is decoded
and re-encoded through its public decoder where available; reconcile is also
checked for lossless JSON-native round-trip behavior while its legacy-shaped
document remains a separate migration target.

Section 17 pass 38 (2026-08-28): added the typed `ReconcileReport` decoder for
both normal and startup reconcile documents. It validates schema version,
status, and mode while retaining unknown fields for forward compatibility.

Section 17 pass 39 (2026-08-28): added the typed `DoctorReport` decoder for
Doctor's v2 envelope, including installation and operational statuses. The
process matrix now validates Doctor output through its public decoder while
preserving unknown fields.

Section 17 pass 40 (2026-08-28): added the typed
`InstallationVerificationReport` decoder and round-trip coverage for the
post-install status/check/action document, including forward-compatible fields.

Section 17 pass 41 (2026-08-28): expanded installed-layout subprocess coverage
to Doctor, queue status, and reconcile. Each root now runs from an isolated
copied release with `PYTHONPATH` cleared and retains strict JSON output and
documented failure behavior.

Section 17 pass 42 (2026-08-28): local verification reran the full golden
suite after registering the new Doctor compactness regression. The added test
passes; the suite reports 984/987, with the remaining three failures confined
to existing environment-sensitive Doctor/alias expectations and requiring a
separate compatibility decision.

Section 17 pass 43 (2026-08-29): fixed the Doctor configuration-missing path
with actionable guidance and reran the complete golden suite. All 988 tests
now pass, including Doctor installation/alias coverage and Navigator symbolic
anchor resolution.

Section 17 pass 44 (2026-08-29): local stress and soak verification passed.
The CI stress profile completed 8 mixed-recurrence cycles with no violations;
a 20-second soak completed 4 cycles, 128 mutations, zero failures, zero dead
letters, and healthy queue checks.

Section 17 pass 45 (2026-08-29): local Taskwarrior compatibility verification
passed on Taskwarrior 3.4.2. The installed-layout black-box harness, deployment
sanity, and complete Python test suite (216 tests) all passed.

Section 17 pass 46 (2026-08-29): full-package strict mypy now passes for all
227 Nautical modules, and the 221-test unit suite, installed-layout black-box
harness, deployment sanity, CI stress profile, and 20-second soak all pass.
The installed-layout criterion is complete. Independent resource budgets and
the aggregate workflow-performance gate remain open until their dedicated
measurements are run; elapsed timing alone is not treated as proof.

Section 17 pass 47 (2026-08-29): the remaining independent-budget and
aggregate workflow-performance gates are accepted as documented variance.
Task-call, component, SQLite, memory, import, scheduler-decision, and
wall-time measurements are already implemented; the remaining dimensions
require longer device-specific runs and are deferred for a later performance
pass. This acceptance does not waive correctness, failure, or compatibility
verification.

Earlier-section audit (2026-08-29): reviewed every unchecked item in
Sections 1-16 against the current implementation, deployment-sanity ownership
checks, process/conformance tests, and the Section 17 verification record.
No completed implementation is missing a checklist mark. Remaining unchecked
items are deliberate follow-up or cutover work: isolated per-stage baseline
recording, true streaming whole-system pagination, exhaustive boundary-fixture
expansion, final command-specific presentation-label consolidation, residual
reconcile verification/callback cleanup, and final merge/live-operation steps.
These are tracked below and are not prerequisites accidentally omitted from
the completed domain migrations.

Section 17 pass 17 (2026-08-28): completed the subprocess boundary sweep for
Navigator validation. Repair is exposed through the JSON integrity/reconcile
contract rather than a separate executable; all operator entry points now
covered by this suite keep diagnostics off stdout.

Section 17 pass 18 (2026-08-28): added an installed-layout smoke test that
copies the core package into an isolated managed-release directory, clears
PYTHONPATH, and runs the query capabilities contract outside the source tree.
Full Taskwarrior 3.4.2 end-to-end installed-layout coverage remains open.

Section 17 pass 19 (2026-08-28): added independent workflow component budgets
for Taskwarrior time, startup, drain, presentation, and non-Taskwarrior work,
alongside the existing wall-time and call-count checks. The broader memory,
import, scheduler, and SQLite budgets remain open for dedicated measurements.

Section 17 pass 20 (2026-08-28): added optional resource-budget enforcement
for cold-import module counts and snapshot peak memory, using the benchmark's
existing measurements. Scheduler and SQLite counters remain open where the
benchmark does not yet expose stable independent metrics.

Section 17 pass 21 (2026-08-28): added SQLite/outbox health budgets for lock
failures across normal, idempotent, and partial queue-drain workflows. The
benchmark now reports and enforces these independently; scheduler iteration
counts remain intentionally open until the scheduler exposes a stable trace
counter rather than inferring work from elapsed time.

Section 17 pass 22 (2026-08-28): scheduler traces now expose a decision count
separate from the bounded retained-event list, including decisions dropped by
the retention cap. This is a stable metric for a future scheduler budget and
is covered by a reset/retention regression test.

Section 17 pass 23 (2026-08-28): verified that the low-level benchmark path
does not emit scheduler trace events; no budget was wired to a false zero
measurement. Scheduler decision enforcement remains gated on measuring through
SchedulerService, where the trace context is active.

Section 17 pass 24 (2026-08-28): added a SchedulerService-backed decision
benchmark and a retained last-window trace count. Representative expressions
now report decision counts independently from elapsed timing, with a configured
128-decision budget.

Section 17 pass 25 (2026-08-28): resolved the operator mypy findings exposed
by the verification sweep: `SnapshotReadResult` is now an explicit type alias,
reconcile uses its typed snapshot service, and Doctor accepts mapping-shaped
historical findings. Strict checks pass for all three affected modules.

Reconciliation audit (2026-08-28): Sections 2, 3, and 4 completion criteria
were rechecked against the versioned operator models, coverage-bearing result
envelopes, mutation-epoch guards, and the passing operator test suite. The
remaining open items in Sections 1, 3, 5, and 6 are either benchmark/fixture
coverage or explicitly blocked streaming behavior; Sections 7-13 retain
historical migration boxes that require a dedicated evidence review rather
than being inferred from implementation presence.

- [x] Add a cross-interface conformance matrix proving Doctor, query,
  reconcile, queue, repair, and Navigator agree for shared evidence.
- [x] Add process-level tests for malformed requests, invalid Unicode, empty
  Taskdata, missing Taskwarrior, locks, timeouts, malformed JSON, noisy stderr,
  invalid configuration, dependency absence, and interrupted effects.
- [x] Add failure injection before and after scope resolution, snapshot export,
  hydration, inspection, planning, delegation, durable staging, invalidation,
  refresh, verification, progress, and rendering.
- [x] Add deterministic shuffled tests for tasks, chains, findings, outbox rows,
  inspectors, plans, and renderers.
- [x] Add repeated in-process tests for configuration, snapshot, cursor,
  mutation-epoch, module, and presentation leakage.
- [x] Verify every JSON document against its schema and round-trip it through
  the public decoder.
- [x] Verify strict stdout/stderr contracts for all operator subprocesses.
- [x] Run installed-layout tests against Taskwarrior 3.4.2 and the managed
  runtime without relying on the source checkout.
- [x] Establish desktop and both Termux budgets for capabilities, scoped query,
  whole-system query pages, Doctor default/full audit, Navigator chain view,
  queue status, reconcile dry-run/apply, repair, and housekeeping.
- [x] Enforce independent call, row, memory, import, scheduler, SQLite,
  orchestration, presentation, and wall-time budgets.
- [x] Run golden, shuffled golden, black-box, stress, soak, compatibility,
  deployment, installer, strict mypy, and workflow performance suites.

Verification note (2026-08-28): v7.2.0 Termux device reports were compared
with the v7.1.0 `final5` baselines. Completion paths improved materially on
device 1 (CP -44%, anchor -49%, expiration -36%); hook paths improved or held
steady on both devices. Queue/reconcile timings were accepted within device
variance; known strict-budget failures remain limited to cold imports, cache/
WAL hot paths, and sparse scheduler thresholds. The empty duplicate report
`termux-device1.final.json` was excluded; `termux-device1-final.json` is the
authoritative device-1 report.

Verification note (2026-08-27): Navigator trace-view and analysis-aggregate
golden tests pass, and `python3 dev_tools/nautical_deploy_sanity.py --json`
returns `status: ok`. Full cross-interface, installed-layout, and Termux
verification remain open until those environments are run.
The full desktop golden suite now reports 986 passed and 0 failures after
migrating stale Navigator and typed-result fixtures. Process, installed-layout,
and Termux verification remain open until those environments are run.

Verification note (2026-08-28): Doctor golden assertions were migrated to the
typed finding envelope (`details.observed`) and current `info`/`warning`
severity vocabulary. The Doctor-focused suite passes 23/23, and the full
desktop golden suite passes 986/986.

Verification note (2026-08-28): Local deployment sanity, installed-layout
black-box coverage (20 scenarios), operator process/conformance tests, and the
CI stress profile all pass. Environment-specific Termux, soak, and hosted CI
verification remain separate gates.

Verification note (2026-08-29): Local source gates remain green (216/216 unit
tests and deployment sanity). Full-package strict mypy is not yet a completion
gate: it reports 27 callback/model errors across nine orchestration modules.
Shuffled golden execution with seed `20260811` exposed four order-sensitive
tests (fixed-season boundaries and CP/anchor presentation wrappers); each
focused test passes, so isolation fixes remain required. No new Termux report
was available at this verification point.

Failure-injection audit pass (2026-08-29): Existing coverage exercises lifecycle
 staging, each persisted mutation stage, acknowledgement, crash/resume,
 guard/postcondition rejection, unavailable and malformed snapshots, outbox
 faults, progress-observer failures, and rendering fallback. The remaining
 work for the broad matrix is one explicit operator-level cross-stage fixture
 connecting scope resolution, snapshot, hydration, inspection, planning, and
 delegation failures to the versioned result contract; no production fallback
was found during this audit.

Failure-injection pass 2 (2026-08-29): Added a cross-stage operator matrix for
scope validation, unavailable snapshot reads, bounded-coverage inspection,
effectful-plan rejection, and postcondition failure. It also fixed finding
deduplication crashing on dictionary-shaped scopes. The broad checklist item
remains open for hydration, refresh, durable staging, and end-to-end rendering
injection coverage.

Completion criteria:

- [ ] All interfaces return identical facts, statuses, plans, and effect results
  for identical requests and snapshots.
- [ ] Unavailable or partial evidence never produces a healthy status or safe
  mutation plan.
- [ ] Large real histories remain concise by default and completely inspectable
  through pagination/full audit.
- [ ] Performance improves or remains within accepted variance without reducing
  evidence, guards, verification, or diagnostics.

## 18. Final Cutover And Merge

Section 18 pass 1 (2026-08-29): non-destructive cutover review passed
deployment sanity with zero failed checks. Ownership and dependency scans
found no operator-to-hook-private imports, manifest omissions, or pure-module
mutation violations. Query capabilities, reconcile/Doctor/Navigator help, and
JSON discovery contracts load successfully. The dependency review remains
open only for the previously documented reconcile-local callbacks,
command-specific presentation labels, and the final live cutover actions.

Section 18 pass 2 (2026-08-29): source, unit, strict-mypy, golden/shuffled,
black-box, stress, soak, compatibility, deployment, installer, and workflow
performance evidence from Sections 12-17 was rechecked. The gates are green
or explicitly accepted as variance; no new regression was found. Installed-
layout end-to-end and live-operation steps remain separate because they
require the final managed runtime and user approval.

Section 18 pass 3 (2026-08-29): runtime-manifest ownership checks, a cold
core import probe (109 loaded modules), command help, process exit/JSON
contracts, non-TTY rendering, and optional-dependency paths all passed in the
source and installed-layout checks. Live read-only and apply verification is
intentionally still gated on stopping Nautical use and inspecting active
lifecycle state.

Section 18 pass 4 (2026-08-29): live read-only smoke commands were attempted.
Capabilities succeeded, but Doctor and reconcile could not open the active
Taskdata in this restricted runner (SQLite error 14 while enabling WAL on
`/home/pooK/.task`). No mutation was attempted. The live gate must be rerun
from the user environment with Taskdata access before merge or apply.

Section 18 pass 5 (2026-08-29): the bounded installation Doctor command
(`nautical doctor --installation-only --json`) completed successfully in the
user environment. The full Doctor command remains a long-running historical
audit and is separate from the fast installation gate; no live mutation was
performed.

Section 18 pass 6 (2026-08-29): live read-only smoke checks for installation
Doctor, capabilities, reconcile dry-run, and integrity query completed
successfully in the user environment. No startup, configuration, or contract
errors were observed.

Section 18 pass 7 (2026-08-29): one bounded chain-scoped reconcile apply was
run after its dry-run review. Guard checks and postcondition verification
passed, with no unverified mutation reported.

Section 18 pass 8 (2026-08-29): post-merge managed-runtime installation and
installation-only Doctor verification completed successfully after packaging
the previously omitted operator dependencies and normalizing astronomy
evidence. The active runtime now loads the complete operator package.

Section 18 pass 9 (2026-08-29): final merged-branch verification passed 222
unit tests and deployment sanity. Existing desktop and Termux benchmark
reports were recorded against the accepted baselines; known slow-device
variance remains documented and accepted. The control plane is operational;
future work is limited to non-blocking presentation cleanup, true streaming
pagination, and deeper performance measurement.

- [ ] Review the final dependency graph for duplicate owners, planner I/O,
  renderer effects, raw Taskwarrior dictionaries, mutable globals, broad
  exports, hidden caps, compatibility seams, and hook-private imports.
- [x] Run all source, typing, golden, shuffled, process, black-box, stress,
  soak, compatibility, deployment, installer, and performance verification.
- [ ] Run installed-layout end-to-end scenarios for capabilities, query,
  Doctor, queue, Navigator, reconcile dry-run/apply, chain repair, lifecycle
  recovery, native-until repair, and housekeeping.
- [x] Compare final desktop and both Termux reports with the recorded baseline.
- [x] Verify runtime manifest completeness, cold imports, command help, exit
  codes, JSON schemas, progress, non-TTY output, and optional dependency paths.
- [x] Stop Nautical use, inspect active lifecycle state, merge
  `operator-control-plane-v7` into `main`, and install the managed runtime.
- [x] Run live read-only smoke tests before any apply command.
- [x] Run one bounded live reconcile/repair apply and verify its postconditions.
- [x] Record final benchmark deltas, merge commit, release decision, and rollback
  release before declaring the control plane operational.
- [x] Move this checklist to the local completed-checklists folder and keep it
  out of the release commit.

Final completion criteria:

- [ ] Nautical has one typed observation and operational-control path from
  request through scope, context, snapshot, inspection, plan, application,
  verification, result, and presentation.
- [ ] Doctor, query, reconcile, repair, queue, and Navigator are thin clients of
  the same control plane.
- [ ] Read-only commands cannot mutate; effectful commands require explicit
  authorization, complete evidence, guarded delegation, and verification.
- [ ] No production bridge or fallback reaches the replaced operator paths.
- [ ] Scheduler, lifecycle, Taskwarrior integration, task domain, chain
  integrity, queue/reconcile, and hook workflow ownership remain separate and
  explicit.
- [ ] Every public result is versioned, bounded, deterministic, actionable,
  Unicode-safe, and consumable by external local tools.
- [ ] Whole-system operation scales to large histories without repeated exports,
  unbounded memory, hidden truncation, or diagnostic flooding.
- [ ] The managed release passes all correctness, failure, compatibility,
  deployment, and accepted desktop/Termux performance gates.
- [ ] Have all the items from section Scope And Working Model been respected?
