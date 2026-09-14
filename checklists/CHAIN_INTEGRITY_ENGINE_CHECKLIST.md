# Nautical Chain Integrity Engine Checklist

Replace Nautical's fragmented chain auditing, reconciliation, metadata repair,
and recovery orchestration with one typed integrity engine. The scheduler
remains authoritative for when an occurrence belongs, the lifecycle planner
remains authoritative for what transition is required, and the Taskwarrior
integration engine remains the only boundary allowed to read or mutate
Taskwarrior.

The integrity engine becomes authoritative for answering two questions:

1. Is the persisted Nautical chain state internally consistent and complete
   enough to trust?
2. When it is not, which deterministic, guarded operations can safely converge
   it without guessing?

## Scope And Working Model

- [x] Develop exclusively on `chain-integrity-engine-v6`; keep `main`
  operational until every final gate passes.
- [x] Treat Nautical as offline while the branch is under construction.
  Intermediate commits do not need to be installable or operational.
- [x] Do not build legacy bridges, old/new engine adapters, duplicate repair
  commands, fallback planners, tuple compatibility layers, or dual JSON
  serializers for replaced internal APIs.
- [x] Keep comparisons with the previous implementation in characterization
  tests and benchmark fixtures only. Production must have one integrity path.
- [x] Remove an old diagnosis or repair path as soon as its replacement owns
  the behavior; do not retain shadows solely to keep intermediate commits
  functional.
- [x] Keep Taskwarrior as the durable source of task and chain state. The
  lifecycle outbox may retain mutation intent and verification evidence, but
  the integrity engine must not create a shadow task database.
- [x] Do not introduce a daemon or persistent cache of Taskwarrior rows.
  Snapshot and graph reuse is invocation-scoped; persisted data may contain
  only repair intent, fingerprints, and bounded operational evidence.
- [x] Preserve strict hook JSON on stdout with `ensure_ascii=False`.
  Diagnostics remain silent unless `NAUTICAL_DIAG=1`, and then go to stderr.
- [x] Keep ordinary hooks outside the integrity engine. Reconcile, Doctor, and
  explicit integrity queries are operator paths; hooks may only consume narrow
  typed integrity outcomes when lifecycle recovery requires them.
- [x] Prefer deterministic refusal over speculative repair. Ambiguous identity,
  incomplete snapshot coverage, unavailable reads, or stale guards must never
  be interpreted as absence.
- [x] Keep this checklist local. Push implementation commits only to the
  integrity branch and merge it into `main` in the final stage.

Cutover policy:

- [ ] Stop Taskwarrior hooks and Nautical workers before installing the
  completed engine.
- [ ] Quarantine obsolete repair state instead of maintaining runtime schema
  bridges.
- [ ] Let the new engine reconstruct required repair work from authoritative
  Taskwarrior state and the current lifecycle outbox.
- [ ] Run Doctor, queue status, integrity dry-run, reconcile apply smoke, and
  lifecycle black-box tests before re-enabling hooks.
- [ ] Roll back by restoring the previous release, not by retaining both
  integrity engines.

## Target Ownership

Exact filenames may change if a clearer boundary emerges, but these
responsibilities must remain separate:

- `chain_integrity_models.py`: immutable graph, coverage, invariant, finding,
  repair-plan, guard, application, and report models.
- `chain_snapshot.py`: authoritative snapshot acquisition, normalization,
  provenance, coverage, and invocation-local hydration.
- `chain_graph.py`: immutable nodes, edges, indexes, ambiguity tracking, and
  graph construction without Taskwarrior access.
- `chain_invariants.py`: pure invariant registry and deterministic evidence.
- `chain_repair_planner.py`: pure finding-to-repair conversion, safety policy,
  dependencies, preconditions, and postconditions.
- `chain_integrity_engine.py`: audit and planning orchestration over one
  snapshot and graph.
- `chain_integrity_application.py`: guarded application through the lifecycle
  application service and Taskwarrior mutation gateway.
- `chain_integrity_render.py`: human and versioned JSON presentation only.
- `tools/nautical_reconcile.py`: thin CLI composition, argument parsing,
  locking, and exit-code mapping.

The final flow is:

```text
reconcile / doctor / query integrity
                 |
       TaskwarriorUnitOfWork
                 |
        authoritative snapshot
                 |
          immutable chain graph
                 |
         pure invariant registry
                 |
          pure repair planner
                 |
     chain integrity application
          /                 \
lifecycle application   mutation gateway
          \                 /
       TaskwarriorUnitOfWork
```

## Baseline And Inventory

- [ ] Record full golden, deterministic shuffled golden, black-box,
  deployment, mypy, hook-protocol, Doctor, chain-repair, reconcile dry-run,
  and reconcile-apply results from `main`.
- [ ] Record desktop and both Termux reconcile profiles for an empty database,
  healthy chains, missing successors, delayed expiration recovery, structural
  metadata repair, native-until repair, and mixed healthy/unhealthy chains.
- [ ] Record Taskwarrior call counts by purpose and exported row counts for
  each profile.
- [ ] Inventory every chain-health rule and mutation currently implemented in
  `reconcile.py`, `tools/nautical_reconcile.py`, `chain_repair.py`, Doctor,
  lifecycle reads, native-until audit, and hook recovery paths.
- [ ] Inventory every place that groups rows by chainID, indexes short UUIDs,
  infers missing links, detects reciprocal-link errors, or classifies an
  orphan completion/deletion.
- [ ] Inventory every configuration, read, JSON, timeout, lock, and mutation
  failure that can currently become an empty snapshot, skipped audit, generic
  error, or manual-review state.
- [ ] Add characterization coverage for every legitimate repair and refusal
  before deleting its old path.
- [ ] Capture current human output and JSON schema fixtures separately;
  presentation parity must not constrain the new internal model.

Completion criteria:

- [ ] Every current integrity check and repair operation has one named owner,
  one characterization test, and one recorded Taskwarrior call budget.
- [ ] Baselines run in isolated Taskdata and state directories without reading
  or mutating the user's tasks.

## 1. Define The Integrity Contract

- [x] Add immutable identifiers for chain, node, slot, edge, invariant,
  finding, repair plan, and repair operation.
- [x] Represent a node with full UUID, short UUID, chainID, link, status,
  recurrence identity, lifecycle fields, scheduling fields, and normalized
  Taskwarrior timestamps.
- [x] Represent `prevLink` and `nextLink` as typed references with resolution
  states: resolved, absent, ambiguous, outside coverage, or unavailable.
- [x] Define snapshot coverage explicitly: complete task universe, complete
  chain, candidate-only, narrow hydration, truncated, and unavailable.
- [x] Define finding states: healthy, repairable, blocked, manual review, and
  unavailable. Severity and repairability must be separate properties.
- [x] Define stable invariant IDs rather than user-facing message strings.
- [x] Make every finding carry the exact node/edge, observed value, expected
  value, evidence provenance, snapshot identity, and invariant version.
- [x] Define typed repair operations for lifecycle recovery, link metadata,
  chain disablement, native-until repair, recurrence metadata repair, and
  outbox recovery.
- [x] Make every repair plan carry deterministic identity, configuration and
  schedule fingerprints, source snapshot identity, operation dependencies,
  guards, postconditions, and safety classification.
- [x] Define application outcomes: applied, already applied, stale, retryable,
  rejected, conflicted, manual review, and unavailable.
- [x] Validate all models at construction so incomplete identities and invalid
  state combinations cannot enter the engine.

Completion criteria:

- [x] Integrity contracts contain no raw task tuples, nullable success values,
  free-form action strings, or `Any` callback bundles.
- [x] A partial or unavailable snapshot cannot construct an authoritative
  absence finding.
- [x] Contract tests cover every valid and invalid model combination.

## 2. Build One Authoritative Chain Snapshot

- [x] Build snapshots exclusively through `TaskwarriorUnitOfWork` and
  `TaskReadRepository`; no integrity module may invoke `task` directly.
- [x] Resolve validated configuration, timezone, Taskdata, and command policy
  once before acquiring snapshot rows. `ChainSnapshotService` binds its
  fingerprint to the invocation's validated configuration context.
- [x] Normalize Taskwarrior's literal `null`, empty UDAs, timestamps, numeric
  links, statuses, and UUID references once at the snapshot boundary.
- [x] Preserve original values beside normalized values when evidence may be
  needed for diagnostics or mutation guards.
- [x] Acquire one broad authoritative candidate snapshot per invocation and
  reuse it across every invariant.
- [x] Hydrate predecessors, referenced children, or historical links narrowly
  only when a rule requires evidence outside broad coverage. The engine now
  hydrates only chains with outside-coverage references, caps hydration at 32
  chains per audit, and fails closed when a required narrow read is unavailable.
- [x] Mark each chain and edge with its actual coverage. A failed narrow read
  must downgrade the relevant result to unavailable, never absent.
- [x] Detect truncated exports, ambiguous short UUIDs, and inconsistent
  Taskwarrior status data before graph construction. Providers now carry an
  explicit `truncated` marker; marked snapshots fail closed. Taskwarrior's
  current unbounded export remains unmarked because it exposes no trustworthy
  truncation signal.
- [x] Reject malformed rows, duplicate full UUIDs, and mismatched rows from a
  narrow chain snapshot before graph construction.
- [x] Include relevant lifecycle outbox rows through the existing outbox
  repository without mixing outbox state into task truth.
- [x] Fingerprint the snapshot from normalized identity and guard fields, not
  descriptions, urgency, IDs, or other volatile presentation data.
- [x] Keep snapshot caches invocation-scoped and invalidate them after every
  successful mutation epoch.

Completion criteria:

- [x] Doctor, reconcile dry-run, reconcile apply, and integrity query acquire
  chain state through the same snapshot service. The read-only `nautical query
  integrity` operation now uses the same engine and snapshot boundary.
- [x] No mutation decision can be made from malformed, truncated, stale, or
  unavailable snapshot evidence. The engine rejects malformed/status-invalid
  snapshots and downgrades unavailable hydration before planning.
- [x] Snapshot tests cover empty Taskdata, large history, partial chains,
  malformed JSON, command failure, ambiguous short UUIDs, bounded hydration,
  and mutation-epoch invalidation. Date-exhaustion fixtures remain part of the
  lifecycle-context pass.

## 3. Construct One Immutable Chain Graph

- [x] Build immutable indexes by full UUID, short UUID, chainID, link slot,
  and status in one pass. (Recurrence-kind and outbox indexes require their
  dedicated graph metadata passes.)
- [x] Represent unresolved references as graph evidence rather than silently
  dropping them.
- [x] Preserve duplicate slot occupants and ambiguous short references as
  first-class graph state.
- [x] Model active, completed, deleted, terminal, and intentionally disabled
  nodes without treating status alone as lifecycle intent. `ChainNode.lifecycle_intent`
  derives semantic intent from chain/terminal fields before status.
- [x] Keep graph construction free of scheduling, Taskwarrior reads,
  mutations, SQLite writes, configuration reloads, and presentation.
- [x] Expose focused graph queries for adjacent links, roots, tips, orphan
  candidates, referenced children, outbox transitions, and coverage. Graph
  exposes lifecycle, root/tip, orphan, referenced-child, chain, slot, status,
  UUID, and edge queries; outbox and coverage remain explicit context data.
- [x] Prevent graph consumers from receiving mutable task dictionaries.
- [x] Add deterministic graph serialization for tests and structured
  diagnostics; do not expose it as a durable database format. (`ChainGraph.to_dict()`
  is invocation-local and includes sorted nodes and reference evidence.)

Completion criteria:

- [x] The same normalized rows always produce byte-for-byte equivalent graph
  evidence regardless of input order.
- [x] Graph construction is linear in row and edge count.
- [x] Unit tests cover healthy, branched, cyclic, duplicate, disconnected,
  partially covered, and ambiguous-reference graphs.

## 4. Define One Pure Invariant Registry

- [x] Give every rule a stable ID, required coverage, deterministic evaluator,
  and typed evidence schema.
- [x] Evaluate identity invariants: mandatory chainID, positive numeric link,
  and exclusive root recurrence identity (`anchor` or `cp`). (Root identity
  consistency across historical nodes follows with lifecycle metadata.)
- [x] Evaluate slot invariants: one authoritative occupant per chain/link.
- [x] Evaluate edge invariants: reciprocal prev/next links, same-chain
  targets, adjacent slot targets, cycles, and forks.
- [x] Evaluate lifecycle successor expectations for completed/deleted chain-on
  tips while respecting explicit terminal bounds and chain disablement.
- [x] Evaluate lifecycle invariants: completed or proven-expired active tips
  receive a successor unless a legitimate terminal condition applies;
  deletion without reliable expiration evidence is manual review, and manual
  disablement remains terminal.
- [x] Evaluate terminal invariants for date exhaustion without fabricating
  successors. `date_limit`/`search_limit` are typed, durable, guard-verified,
  and checked against the terminal postcondition before acceptance.
- [x] Verify terminal exhaustion plans against the persisted parent guard and
  recurrence fingerprint; stale or contradictory scheduler evidence now
  becomes manual review.
- [x] Validate chain-off finalization against acknowledged lifecycle outbox
  evidence; a terminal plan now requires an unlinked parent with `chain:off`.
- [x] Validate terminal chainMax and chainUntil values before treating an
  unlinked completed/deleted tip as legitimately final.
- [x] Evaluate child continuity only at a specific parent-to-child transition.
  The registry rejects a resolved child target that is not later than its
  parent target, without comparing historical recurrence expressions.
- [x] Evaluate carry invariants for due/scheduled basis, wait, scheduled, and
  native until using canonical field normalization. Direct parent→child edges
  now reject changed relative offsets when both sides are parseable.
- [x] Require direct children to retain the parent recurrence kind without
  comparing historical expression text.
- [x] Evaluate native-until ordering after canonical timestamp normalization.
  (Predecessor-derived repair evidence belongs to the structural repair pass.)
- [x] Evaluate lifecycle outbox invariants: deterministic identity, plan
  agreement, stage/postcondition agreement, stale/manual-review evidence, and
  acknowledged intent retention. Acknowledged child/parent postconditions and
  terminal finalization are now checked against the graph; repository model
  validation covers deterministic identity and stage constraints.
- [x] Evaluate configuration fingerprint drift before proposing any mutation
  when an outbox intent is present. (Schedule fingerprint comparison belongs
  with lifecycle planning.)
- [x] Ensure rules return unavailable when required evidence cannot be read,
  rather than returning healthy or repairable.
- [x] Run rules in a deterministic dependency order and deduplicate findings
  that share one underlying fault.

Completion criteria:

- [x] Invariant evaluation is pure and cannot import Taskwarrior, SQLite,
  hooks, Rich, or CLI modules.
- [x] Every current Doctor, chain-repair, native-until, and reconcile check is
  represented once in the registry or explicitly documented as presentation
  only. `INVARIANT_OWNERSHIP` is validated by a golden test.
- [x] Mutation paths never parse human-readable finding messages. Reconcile
  terminal event selection now uses the typed `terminal_kind` field emitted by
  lifecycle planning.

## 5. Build The Pure Repair Planner

- [x] Convert uniquely derivable repairable findings into complete typed repair
  plans without Taskwarrior calls, SQLite writes, locks, clocks, or
  presentation. (Current pass covers reciprocal links and missing numeric
  links uniquely implied by adjacent neighbors.)

- [x] Centralize the safety policy that distinguishes automatic repair,
  deferred retry, and manual review.
- [x] Permit automatic repair only when one result follows uniquely from
  authoritative evidence and all required coverage is complete.
- [x] Refuse repair for incomplete coverage, ambiguous evidence, and findings
  without a unique guarded operation.
- [x] Convert all remaining automatically repairable findings into complete typed
  repair plans without Taskwarrior calls, SQLite writes, locks, clocks, or
  presentation. Findings without a uniquely derivable value are now explicit
  manual-review findings rather than misleadingly repairable.
- [x] Refuse repair for duplicate slots, ambiguous short UUIDs, conflicting
  recurrence identities, incomplete coverage, and findings without unique
  guarded evidence. (Stale configuration and predecessor policy remain in
  lifecycle planning.)
- [x] Generate deterministic repair identity from invariant, chain, affected
  nodes, expected values, schedule/configuration fingerprints, and operation
  version.
- [x] Express multi-operation repair order as explicit dependencies, not
  procedural fallthrough.
- [x] Attach exact parent/task guards and postconditions to every operation.
- [x] Collapse equivalent findings into one operation and reject plans that
  attempt incompatible mutations of the same field.
- [x] Reuse `LifecyclePlanner` for completion, expiration, terminal, and
  successor decisions; the integrity planner must not recalculate schedules or
  build children itself by explicitly refusing lifecycle successor findings.
- [x] Use named Taskwarrior mutation operations for structural field repairs;
  the planner must not construct command argv.
- [x] Make dry-run return the same plan identity and operation order that apply
  will use when the snapshot remains unchanged.

Completion criteria:

- [x] Planning is deterministic under shuffled findings and graph row order.
- [x] No repair plan exists without complete evidence, guards,
  postconditions, and a stable reason code.
- [x] Planner tests cover safe repair, incomplete/ambiguous evidence, stable
  identities, and manual-review boundaries. Dependency and guard validation are
  covered by the integrity contract/application tests.

## 6. Introduce The Chain Integrity Engine

- [x] Add one `ChainIntegrityEngine` that accepts a validated unit of work,
  audit request, clock, and explicit read-only or mutation-capable access.
- [x] Make the engine own snapshot acquisition, graph construction, invariant
  evaluation, repair planning, and typed summary assembly.
- [x] Make `ChainIntegrityEngine.plan_recovery()` the single lifecycle planning
  entry point; the implementation now lives in `chain_integrity_lifecycle.py`.
- [x] Keep audit and planning separate from application. A dry-run must never
  acquire mutation services or write lifecycle state.
- [x] Support whole-system and explicit chain/UUID scopes through the same
  pipeline.
- [x] Treat configuration or broad snapshot unavailability as a run-level
  failure that prevents all mutation.
- [x] Isolate chain-local invalidity so one manual-review chain does not hide
  independent safe plans for other chains.
- [x] Preserve deterministic ordering by chainID, link, invariant ID, and
  operation identity.
- [x] Produce one typed report containing coverage, findings, plans, blocked
  work, application outcomes, command statistics, and timing.
- [x] Keep rendering and exit-code policy outside the engine.

Completion criteria:

- [x] Reconcile, Doctor, and integrity query receive the same findings for the
  same snapshot and configuration. The engine parity regression verifies that
  the supplied snapshot and freshly collected snapshot serialize identical
  findings; Doctor presentation mapping remains isolated to Section 11.
- [x] The engine has no hook-module imports, global repository variables,
  process-global snapshot state, or presentation callbacks.

## 7. Migrate Successor And Terminal Recovery

Progress note: the lifecycle planner and typed decision model have moved out of
`reconcile.py`, and reconcile calls the engine-owned planning entry point. The
remaining unchecked work is routing hookless completion, terminal recovery, and
application execution through that same engine path.

- [x] Route hookless completion recovery through the engine and shared
  `LifecyclePlanner`.
- [x] Route native-until expiration recovery through the shared lifecycle
  policy while preserving due/scheduled recurrence basis and bounded delayed-
  expiration hops.
- [x] Resolve existing deterministic children through typed repository reads;
  unavailable lookup defers rather than creating a duplicate.
- [x] Preserve chainMax, chainUntil, date exhaustion, and intentional manual
  deletion as explicit terminal findings and plans.
- [x] Preserve already-created child and already-linked parent recovery as
  idempotent `already_applied`, not manual review.
- [x] Make each expiration hop a separately guarded lifecycle plan while one
  integrity report retains their dependency order.
- [x] Remove successor/terminal orchestration from `reconcile.py` and
  `tools/nautical_reconcile.py` after parity tests pass; the standalone
  chain-repair command and dispatch path are removed, while lifecycle
  successor/terminal decisions remain owned by `LifecyclePlanner`.

Completion criteria:

- [x] Completion, expiration, delayed recovery, terminal, duplicate-child,
  and partial-application cases use one lifecycle planning and application
  path; the lifecycle parity matrix and reconcile recovery fixtures cover
  each case through the shared planner/application services.
- [x] No integrity module computes an occurrence independently of the
  scheduler service.

## 8. Migrate Structural And Carry Repairs

- [x] Replace `chain_repair.py` link inference with graph invariants and typed
  repair plans. Doctor now consumes `ChainIntegrityEngine` plans.
- [x] Repair missing numeric links only when adjacent resolved edges imply one
  unoccupied positive slot.
- [x] Repair prevLink/nextLink only when full UUID evidence resolves to one
  adjacent same-chain node and both guards remain current.
- [x] Move native-until audit, predecessor inference, day-end fallback, carry
  comparison, and repair verification behind typed integrity evidence. The
  policy-owned carry calculation remains isolated in `native_until_integrity`.
- [x] Preserve the policy: consult predecessor evidence first; use local
  due-date 23:00 only when inference is unavailable; refuse when due is at or
  after the fallback boundary. Focused native-until recovery tests cover this
  policy and its manual-review boundary.
- [x] Normalize Taskwarrior date encodings and literal `null` before immutable
  comparison so volatile representation differences do not create conflicts.
- [x] Exclude ID, urgency, modified, end, status transitions, and other
  explicitly volatile fields from immutable plan comparison while retaining
  them in mutation guards where relevant.
- [x] Move recurrence metadata and manual chain-off repair behind named
  mutation requests and postcondition verification.
- [x] Remove standalone chain-repair planning and application after Doctor,
  reconcile, and the operator consume the integrity engine. The obsolete
  module and stale characterization tests were removed; the operator now uses
  only typed integrity plans and guarded mutation requests.

Completion criteria:

- [x] Every structural repair is planned from graph evidence and applied
  through the mutation gateway; native-until carry remains the explicitly
  isolated policy-owned exception pending its extraction pass.
- [x] A representation-only Taskwarrior difference cannot create manual
  review, while a semantic difference cannot be hidden as volatility.

## 9. Build Guarded Integrity Application

- [x] Add one application service that accepts only validated integrity repair
  plans and a mutation-capable unit of work. `IntegrityApplicationService` now
  rejects non-safe or unfingerprinted plans before invoking the typed mutation
  boundary.
- [x] Route lifecycle operations through `LifecycleApplicationService` and
  structural operations through `TaskwarriorMutationService`. Reconcile uses
  the lifecycle service for lifecycle plans and the integrity engine accepts
  only structural mutation operations; a regression test rejects lifecycle
  operations at that boundary.
- [x] Persist every multi-step repair in the lifecycle outbox before its first
  external mutation; do not invent a second repair queue. The engine now sends
  every plan through `IntegrityApplicationService`, which applies this shared
  outbox policy for multi-operation plans.
- [x] Acquire one reconcile lease for apply and narrow per-parent locks only
  where lifecycle mutation ordering requires them. Reconcile's lease and
  parent-link lock paths are covered by concurrency regressions.
- [x] Re-read narrow guards immediately before mutation and verify explicit
  postconditions immediately afterward. The mutation gateway owns both checks.
- [x] Invalidate invocation snapshots after mutation, then use narrow fresh
  reads for dependent operations. `TaskwarriorUnitOfWork.record_mutation()`
  advances the epoch and invalidates the invocation read cache.
- [x] Convert stale guards into deterministic stale outcomes. Replanning is
  deliberately caller-driven (zero implicit retries in one apply invocation),
  so a fresh invocation is the bounded replan and the old plan is never
  applied to fresh state.
- [x] Make crash recovery converge after every boundary: before persistence,
  after persistence, after child import, after parent link, after structural
  mutation, after verification, and before acknowledgement. The shared
  lifecycle outbox stages/claims/acknowledges work and the mutation gateway
  verifies each boundary.
- [x] Keep already-applied plans idempotent even when Taskwarrior changed the
  `modified` timestamp during the original mutation. Guarded mutation replay
  returns `ALREADY_APPLIED` for converged state and preserves the parent-link
  crash recovery exception.
- [x] Persist manual-review transitions durably and report persistence failure
  as unavailable rather than claiming review was recorded. Integrity drain now
  checks the outbox transition result and emits a retryable outcome when it
  cannot persist manual review.
- [x] Enforce configuration and schedule fingerprint agreement again before
  each mutation group. Integrity outbox drain now refuses drifted envelopes
  before invoking the mutation gateway.

Completion criteria:

- [x] Re-running apply after interruption converges without duplicate child,
  duplicate link, lost carry field, or repeated mutation. Shared outbox
  acknowledgement and guarded mutation replay provide idempotent recovery.
- [x] No integrity application code invokes `task`, opens SQLite directly, or
  parses Taskwarrior text output. Those responsibilities remain in the
  Taskwarrior repository/client and outbox adapters.

## 10. Replace The Reconcile Front End

- [x] Reduce `tools/nautical_reconcile.py` to argument parsing, validated unit
  of work construction, lease composition, engine invocation, rendering, and
  exit-code mapping. Candidate selection, child hydration, and lifecycle
  schedule projection and parent action dispatch now live in
  `LifecycleReconciliationService`; the CLI supplies only typed Taskwarrior
  callbacks.
- [x] Preserve dry-run as the default and require `--apply` for mutation.
- [x] Preserve bounded delayed-expiration recovery and explicit housekeeping
  control as typed audit/application options.
- [x] Add explicit chainID and UUID scopes so operators can diagnose one chain
  without scanning unrelated history. `reconcile --chain-id` and `reconcile
  --uuid` are mutually exclusive and become repository-level Taskwarrior
  filters.
- [x] Define one versioned JSON report schema containing coverage, findings,
  repair plans, outcomes, summaries, and command statistics.
- [x] Version the new schema once; do not maintain simultaneous old and new
  serializers. Document the breaking operator-contract change for the major
  architecture release.
- [x] Render actionable human output from stable finding/operation codes and
  evidence, including the exact fields requiring manual review.
- [x] Keep color and Rich presentation outside engine models.
- [x] Make exit codes distinguish healthy, repair available, repairs applied,
  manual review, unavailable, and startup/configuration failure.
- [x] Keep stdout machine-clean in `--json` mode and diagnostics on stderr only
  under `NAUTICAL_DIAG=1`.

Completion criteria:

- [x] The CLI contains no chain grouping, scheduling, repair inference,
  Taskwarrior parsing, or mutation logic; those responsibilities are owned by
  the lifecycle service and its repository/application ports.
- [x] Human and JSON subprocess tests cover the startup-error contract, while
  in-process tests cover healthy, degraded, partial, and manual-review states.

## 11. Unify Doctor And Public Integrity Queries

- [x] Make Doctor consume a read-only integrity report instead of independently
  exporting, grouping, repairing, or building reconcile plans.
- [x] Preserve Doctor's installation, dependency, configuration, and runtime
  checks outside the chain integrity engine.
- [x] Map integrity findings to Doctor severity and fixes in one presentation
  adapter.
- [x] Remove Doctor's duplicate chain indexes, link checks, repair planner,
  orphan planning, and native-until interpretations.
- [x] Add a read-only `nautical query integrity` operation for external tools,
  scoped by UUID, chainID, or bounded all-task audit.
- [x] Reuse the integrity JSON models and stable invariant IDs; do not expose
  internal task dictionaries or private module names.
- [x] Advertise integrity-query schema, limits, statuses, and failure kinds in
  `nautical query capabilities`.
- [x] Keep occurrence queries focused on scheduling. Do not inflate every
  occurrence response with a full chain audit.
- [x] Ensure query integrity cannot construct mutation-capable services.

Completion criteria:

- [x] Doctor, reconcile, and query integrity agree on finding identity,
  evidence, and repairability for the same fixture.
- [x] External tools can inspect chain health without importing Nautical
  internals or replicating repair logic.

## 12. Protect Performance And Slow Devices

- [x] Add desktop and Termux benchmarks for empty, healthy, candidate-heavy,
  long-history, corrupted, and mixed-chain audits.
- [ ] Budget Taskwarrior calls, exported rows, graph-build time, invariant time,
  planning time, application time, peak memory, and total wall time.
- [x] Require one broad snapshot per run; invariant count must not increase
  Taskwarrior export count. Reconcile reports export calls/rows and the
  benchmark rejects workflows that exceed the bounded snapshot budget.
- [x] Use narrow hydration only for unresolved evidence needed by a candidate
  finding; deduplicate identical reads through the unit of work.
- [x] Avoid loading complete chain history for healthy active tips when current
  node, adjacent references, terminal policy, and outbox state are sufficient.
- [x] Keep a full-audit mode for explicit deep validation; do not weaken
  correctness silently to meet a fast-path budget.
- [x] Build graph indexes and normalized immutable records once per snapshot.
- [x] Add scale fixtures for 100, 1,000, and 10,000 chains plus long individual
  histories.
- [x] Add benchmark assertions that the expected finding or mutation occurred;
  do not measure an empty or stale cleanup path by accident.
- [x] Treat performance optimizations as provider-certified only after
  conformance tests prove identical findings and plans.

Completion criteria:

- [x] Healthy reconcile performs no per-chain Taskwarrior subprocess loop.
- [x] Candidate-heavy apply stays within the established integration-engine
  command budget plus the minimum guarded mutation calls.
- [x] Both Termux devices complete the accepted profile without timeout,
  excessive memory growth, or correctness drift.

## 13. Remove Replaced Ownership

- [x] Delete the old `ReconcilePlan` string-action model and repair orchestration
  from `reconcile.py`; retain no facade aliases for internal callers.
- [x] Delete `chain_repair.py` after all consumers use integrity findings and
  repair plans.
- [x] Remove global reconcile repository/snapshot state and private runtime
  loader coupling.
- [x] Remove duplicate native-until audit and candidate grouping from the CLI
  and Doctor.
- [x] Remove short-UUID indexes and existing-child scans from the CLI and
  Doctor.
- [x] Keep repair rendering in the presentation front ends only; remove any
  duplicated repair classification or formatting from them.
- [x] Remove direct imports of hook implementations from reconcile and Doctor.
- [x] Update runtime manifests, installer validation, deployment sanity, AST
  boundary checks, mypy targets, and test registry for the new modules.
- [x] Add an AST/deployment rule preventing new chain diagnosis or Taskwarrior
  mutation ownership inside operator front ends.
- [x] Remove obsolete characterization tests only after equivalent engine-level
  and black-box coverage exists.
- [x] Run an import graph audit to ensure pure graph/invariant/planner modules
  cannot import UI, hooks, Taskwarrior client, SQLite, or operator tools.

Completion criteria:

- [x] One production path owns snapshot, graph, invariants, repair planning,
  and application.
- [x] No compatibility bridge or shadow implementation remains.
- [x] Operator front ends are composition and presentation only.

## 14. Failure, Concurrency, And Recovery Verification

- [x] Add fault injection for every Taskwarrior read and mutation boundary.
- [x] Add fault injection for outbox persist, claim, stage, verification,
  acknowledgement, and manual-review persistence.
- [x] Test concurrent reconcile apply, hook exit drain, Doctor, and integrity
  query processes against one Taskdata directory.
- [x] Test configuration drift between snapshot, planning, persistence, and
  mutation.
- [x] Test task modification, deletion, completion, sync replacement, and
  child creation between snapshot and apply.
- [x] Test malformed JSON, empty output, Taskwarrior busy/locked, timeout,
  missing binary, invalid timezone, unavailable astronomy, and malformed
  calendar files.
- [x] Test ambiguous short UUIDs, duplicate links, cycles, forks, missing
  references, cross-chain edges, and partial snapshot coverage.
- [x] Test crashes after each repair stage and verify that the next apply
  converges idempotently.
- [x] Test deterministic shuffled input, invariant order, plan order, and
  repeated process execution.
- [x] Test strict stdout/stderr contracts for every reconcile and integrity
  query state.
- [x] Add black-box tests using a real temporary Taskwarrior database for each
  automatically repairable invariant and each manual-review boundary.

Completion criteria:

- [x] Unavailable evidence never becomes healthy, absent, or repairable.
- [x] Concurrent or interrupted operation cannot duplicate children, fork a
  chain, overwrite a legitimate edit, lose outbox intent, or falsely report a
  durable repair.
- [x] Full golden and deterministic shuffled suites pass without state leaks or
  unexpected warnings.

## 15. Documentation, Cutover, And Final Verification

- [x] Update the Systems Manual with the integrity model, dry-run/apply flow,
  finding statuses, manual-review guidance, and query contract.
- [x] Keep README changes concise: mention self-auditing and recovery without
  exposing internal architecture.
- [x] Update the cheatsheet only with operator commands that help users recover
  quickly.
- [ ] Update `nautical query capabilities`, Doctor help, reconcile help, and
  release notes together.
- [x] Run full golden, deterministic shuffled golden, black-box, deployment,
  strict mypy, hook protocol, Doctor, reconcile, query, and installer suites.
- [ ] Run desktop performance and both Termux profiles; compare correctness,
  Taskwarrior calls, exported rows, memory, and wall time with baseline.
- [x] Exercise dry-run and apply against representative copies of real chain
  data containing healthy, completed, expired, manually deleted, repaired,
  and historical chains.
- [ ] Stop hooks, install the branch release, quarantine obsolete repair state,
  run Doctor and reconcile dry-run, then apply only reviewed plans.
- [x] Verify a second dry-run is healthy or contains only explicitly deferred
  manual-review findings.
- [ ] Merge `chain-integrity-engine-v6` into `main` only after every completion
  criterion passes.
- [ ] Verify local and remote `main` equality after the merge.
- [ ] Remove this completed local checklist after the operational release is
  published.

Final completion criteria:

- [ ] Scheduler, lifecycle, integration, and integrity ownership are mutually
  exclusive and enforced by imports and tests.
- [ ] Reconcile, Doctor, and public integrity queries report one authoritative
  view of chain health.
- [ ] Every automatic repair is deterministic, guarded, idempotent, and
  postcondition-verified.
- [ ] Every ambiguous or unavailable case fails closed with actionable
  evidence.
- [ ] Healthy systems pay one bounded audit cost rather than one subprocess
  loop per chain.
- [ ] No old chain repair, reconcile planner, or duplicate Doctor ownership
  remains in production.
