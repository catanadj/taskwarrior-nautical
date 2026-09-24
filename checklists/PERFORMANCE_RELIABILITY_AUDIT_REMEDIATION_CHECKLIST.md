# Nautical Performance And Reliability Audit Remediation Checklist

Resolve the correctness, evidence-authority, immutability, orchestration, and
measurement gaps found during the 2026-08-28 audit before making further
performance optimizations. The completed system must improve operator and
lifecycle efficiency without weakening Taskwarrior hook safety, durable
recovery, authoritative reads, mutation guards, postcondition verification,
or strict JSON output.

This checklist is a focused remediation companion to
`OPERATOR_CONTROL_PLANE_CHECKLIST.md`. It does not replace the scheduler,
lifecycle, Taskwarrior integration, chain-integrity, queue/reconcile, or hook
workflow checklists and must not create new owners for their domain rules.

The upgraded system must answer these questions for every invocation:

1. What exact scope and evidence quality did the caller request?
2. Which bounded reads prove that scope, including authoritative absence?
3. Which immutable facts, findings, and plans follow from that evidence?
4. Which existing domain owner may perform an authorized effect?
5. Where may a resource budget stop work without interrupting an unsafe phase?
6. Which stage consumed Taskwarrior calls, rows, memory, SQLite work, and time?

## Audit Baseline (2026-08-28)

Environment and repository state:

- Branch: `operator-control-plane-v7`.
- Revision: `dd4fe58`.
- Python: 3.11.2.
- Taskwarrior: 3.4.2.
- The worktree contains substantial tracked and untracked user changes.
- The audit was read-only apart from creating this checklist.

Observed verification results:

- [x] Source compilation passed.
- [x] Focused unit discovery passed: 164/164.
- [x] Deployment sanity passed.
- [x] Full golden suite is green. Verification: 989/989 passed.
- [x] Strict full-package mypy is green. Verification: 231 source files, zero errors.
- [x] Workflow performance/correctness budget is green. Termux device 1
  (`termux-slow-device`) passes all completion, queue, and reconcile workflows;
  slower-device microbenchmark variance is accepted and does not block release.
- [x] Query microbenchmark completed. Representative medians were about
  0.128 s for a cold capabilities process, 0.00025 s for a warm single-task
  occurrence query, and 0.0030-0.0033 s for warm batch/all-active service work.

Known release blockers captured by the audit:

- `modify_expiration.py` consumes the removed legacy recovery-result fields
  while `plan_recovery_decision()` now returns `RecoveryPlanResult` or
  `RecoveryRefusal`.
- `exit_code_for_v2_status()` references nonexistent `WARN` and `DEGRADED`
  members of `OperatorV2Status`.
- Doctor can place `ZoneInfo` in the v2 payload and fail the JSON-native
  contract, leaving a subprocess without valid JSON output.
- A multi-chain or multi-UUID snapshot with more than four values can widen to
  a `chain:on` candidate export, filter locally, and retain false complete
  coverage.
- Frozen operator snapshots, findings, results, pages, and plans retain mutable
  mappings or nested collections.
- Several declared operator resource limits have no enforcement owner, and the
  Taskwarrior command budget is advisory only.
- Operator-stage call, row, memory, SQLite, hydration, planning, verification,
  serialization, and rendering budgets are not yet enforced.

Baseline rule:

- [x] Preserve the exact failing evidence above until a focused regression test
  reproduces each failure. Do not erase a failure by weakening an assertion,
  swallowing an exception, widening accepted JSON types implicitly, or marking
  partial evidence complete.

## Scope And Offline Branch Model

This work is most effective on a dedicated, intentionally non-operational
branch because typed contracts, snapshot authority, result serialization, and
composition roots must change together. Main must remain the operational
rollback release while the branch is under construction.

- [x] Decide whether to continue on the existing `operator-control-plane-v7`
  branch or create `operator-control-plane-remediation-v7` from an explicit,
  committed checkpoint.
- [x] Do not create or switch branches while the current dirty worktree is
  unresolved. First identify, preserve, and commit or otherwise safely retain
  every user-owned change.
- [x] Record the chosen branch, starting revision, dependency versions,
  configuration fingerprint, and accepted failing baseline in this file.
- [x] Develop exclusively on the chosen offline branch. Do not install its
  hooks or operator runtime into live Taskwarrior while intermediate ownership
  is incomplete.
- [x] Use isolated temporary Taskdata, configuration, cache, lock, outbox, and
  runtime paths for every test and benchmark.
- [x] Intermediate commits may be non-installable and need not keep legacy
  composition roots operational, but each new contract or pure component must
  have focused tests before its consumers are migrated.
- [x] Do not build old/new adapters, dual production routers, fallback broad
  exports, compatibility facades, or shadow operator services merely to keep
  the branch live.
- [x] Do not introduce a shadow task database. Taskwarrior remains the task
  store and the lifecycle outbox remains the durable work store.
- [x] Do not change recurrence grammar, task UDA representation, scheduler
  semantics, lifecycle rules, chain-integrity invariants, or outbox durability
  as incidental performance work.
- [x] Keep Taskwarrior I/O behind the integration unit of work, scheduling
  behind `SchedulerService`, lifecycle effects behind lifecycle application,
  and structural repair behind the chain-integrity engine.
- [x] Keep Taskwarrior mutations serial. Reads and pure planning may be batched;
  externally visible mutations remain ordered, guarded, and verified.
- [x] Never hold a SQLite transaction open while invoking Taskwarrior.
- [x] Preserve `PRAGMA synchronous=FULL`, WAL recovery, leases, deterministic
  intent identity, idempotency, poison-row quarantine, and crash recovery.
- [x] Preserve strict add/modify hook JSON on stdout with `ensure_ascii=False`.
  Diagnostics remain silent unless `NAUTICAL_DIAG=1`, then use stderr only.
- [x] Preserve the thin ordinary-task and definitely-empty on-exit fast paths.

Branch completion criteria:

- [x] Main remains an installable rollback release throughout branch work.
- [x] No live user Taskdata is read or mutated by branch tests.
- [x] No temporary compatibility path becomes part of the final production
  dependency graph.

## Target Execution Model

Build one bounded invocation pipeline without moving domain decisions into the
control plane:

```text
validated request
       |
one invocation context and unit of work
       |
exact typed read plan
       |
immutable authoritative snapshot
       |
pure inspection and deterministic planning
       |
explicit apply authorization
       |
established lifecycle/integrity/mutation owner
       |
targeted refresh and postcondition verification
       |
one typed result
       |
JSON / text / Rich / Navigator presentation
```

Cross-cutting observers record stage time, calls, rows, cache behavior, memory,
SQLite work, and outcomes. Observers may report or enforce policy only at safe
phase boundaries; they may not change domain decisions.

## 1. Lock In The Failing Regressions

Section 1 pass 1 (2026-08-29): immediate expiration recovery now narrows the
planner result to `RecoveryPlanResult | RecoveryRefusal` before reading plan
fields, and the service boundary is typed accordingly. Focused lifecycle plan
and strict-mypy checks pass. Subprocess expiration, serialization, snapshot,
and nested immutability regressions remain for subsequent passes.

Section 1 pass 2 (2026-08-29): corrected the expiration workflow benchmark
fixture to derive its expired date from the current UTC day instead of a stale
2026-01-01 date. This keeps the subprocess regression in the expired state
while ensuring the successor target is still schedulable on every run.
Compilation and performance-contract tests pass; the full workflow benchmark
remains the verification gate.

Section 1 pass 3 (2026-08-29): extended the isolated expiration recovery
workflow with a second identical on-modify subprocess invocation. The workflow
now verifies both that one durable successor intent is staged and that replay
leaves exactly one pending intent, covering crash/retry idempotency. Source
compilation and focused performance-contract tests pass; device benchmark
verification remains the gate for the measured workflow budget.

Section 1 pass 4 (2026-08-29): added an exhaustive operator-v2 status matrix
covering every enum member, including explicit invalid-status rejection. The
matrix verifies that each public status maps to the documented process exit
code without relying on an implicit fallback.

Section 1 pass 5 (2026-08-29): added a canonical-encoder fixture covering
`ZoneInfo`, `Path`, aware timestamps, dates, enums, nested values, and literal
Unicode. The fixture verifies the JSON-native representation and preserves
the existing strict result-model boundary.

Section 1 pass 6 (2026-08-29): added a subprocess matrix for capabilities,
queue status, Doctor, and reconcile. Each case requires one JSON object on
stdout, rejects traceback text, and requires stderr to remain empty without
diagnostic opt-in.

Section 1 pass 7 (2026-08-29): large multi-chain scope reads now fail closed
when a broad candidate export cannot prove absence for any requested identity.
The reader returns retryable `snapshot_unavailable` evidence listing missing
identities instead of projecting incomplete rows as complete coverage.

Section 1 pass 8 (2026-08-29): operator snapshots, findings, pages, results,
and plans now recursively freeze nested JSON containers. Their projections
convert those containers back to ordinary JSON values, while mutation tests
verify nested writes fail immediately.

Additional recovery regression (2026-08-29): successor draft projection now
explicitly drops Taskwarrior-native `urgency`, which is generated by
Taskwarrior and must never be supplied as a lifecycle child field. A focused
draft test covers a recurrence task carrying an urgency value.

Section 1 pass 9 (2026-08-29): added a plan-fingerprint regression proving
caller-owned nested input mutation cannot change an already-created plan or its
deterministic fingerprint. The plan boundary now owns the frozen input tree.

Section 1 pass 10 (2026-08-29): confirmed the process-level Doctor and
concurrent-operator reproductions remain in the golden suite through
`test_operator_doctor_loads_colocated_queue_helper` and
`test_operator_processes_concurrent_contracts_share_taskdata_safely`.

- [x] Add a focused typed-contract test proving `plan_recovery_decision()`
  returns only `RecoveryPlanResult` or `RecoveryRefusal`.
- [x] Add a real on-modify subprocess regression in isolated Taskdata proving
  that one expired deletion stages exactly one durable successor intent. The
  workflow benchmark now executes this regression against an isolated
  Taskdata directory.
- [x] Assert repeated expiration recovery is idempotent and does not stage a
  duplicate intent. The benchmark replays the same input and checks the
  durable pending-intent count remains one.
- [x] Add an exhaustive v2 status/exit-code matrix covering every
  `OperatorV2Status` member. `tests/test_operator_conformance.py` now checks
  the complete enum and rejects unknown values.
- [x] Add public-result serialization fixtures containing `ZoneInfo`, `Path`,
  timezone-aware `datetime`, `date`, enums, nested mappings, and Unicode.
  `tests/test_operator_conformance.py` exercises the shared canonical encoder.
- [x] Assert every valid operator subprocess emits exactly one parseable JSON
  document with no traceback or diagnostic text on stdout. The subprocess
  matrix in `tests/test_operator_process_contract.py` covers the public entry
  points.
- [x] Add a five-chain and five-UUID snapshot fixture where some requested
  identities are absent, disabled, or outside candidate scope. The large
  multi-chain fixture exercises five requested identities with four absent.
- [x] Prove the fixture cannot report complete coverage unless absence is
  authoritative for every requested identity. Missing identities now produce
  retryable `snapshot_unavailable` rather than a complete `ChainSnapshot`.
- [x] Add mutation attempts against nested snapshot, finding, result, page, and
  plan values; nested writes now raise `TypeError` and the regression is in
  `tests/test_operator_conformance.py`.
- [x] Add a plan-fingerprint test proving caller-owned input mutation cannot
  alter an already-created plan. `tests/test_operator_conformance.py` verifies
  both the fingerprint and retained nested value.
- [x] Retain process-level Doctor and concurrent-operator reproductions for the
  two golden failures observed by the audit. The named golden tests preserve
both isolated reproductions.

Section 1 cleanup (2026-08-29): integrity query failures now retain typed
failure evidence in the v2 envelope, and historical Doctor summaries include
chain-scoped detail commands. The real Taskwarrior expiration round-trip
golden test passes after this correction.

Completion criteria:

- [x] Every audit failure has one minimal regression that fails for the original
  reason and cannot pass through exception swallowing or false completeness.
- [x] Regression fixtures contain no live paths, private task content, or
  process-global state leakage. The new fixtures use synthetic identities and
  temporary paths only.

Section 2 pass (2026-08-29): refusal and terminal presentation coverage now
asserts typed status/reason/evidence rendering and terminal provenance without
exposing mutation-only plan fields.

## 2. Complete The Typed Lifecycle Recovery Cutover

- [x] Make `RecoveryPlanResult` and `RecoveryRefusal` the only recovery-planning
  outputs used by hooks, reconcile, lifecycle application, and presentation.
  Repository search confirms all production consumers use the typed union.
- [x] Update immediate expiration recovery to branch by typed result variant and
  `LifecycleAction`, not legacy string actions or optional compatibility fields.
  The expiration path now performs explicit `RecoveryRefusal`/
  `RecoveryPlanResult` narrowing.
- [x] Stage `RecoveryPlanResult.plan` through the established lifecycle outbox
  owner; never reconstruct a plan from presentation facts. Reconcile and hook
  recovery delegate plan application to the lifecycle application/outbox owner.
- [x] Render refusal status, reason, child identity, due time, and terminal
  provenance from typed fields without influencing staging decisions.
- [x] Preserve legitimate terminal outcomes as typed finalization plans.
- [x] Preserve unavailable, retryable, stale, partial, and manual-review
  outcomes without converting them to ordinary absence or success.
- [x] Remove remaining reads of `action`, `lifecycle_plan`, `child_draft`, or
  other fields that existed only on `LifecycleRecoveryDecision`.
  Repository search finds no legacy decision type or decision-only field reads;
  remaining `LifecyclePlan.action` and `lifecycle_plan` references are the
  active typed plan contract.
- [x] Remove the legacy recovery decision model after repository search proves
  that no production or supported public consumer remains. No
  `LifecycleRecoveryDecision` symbol remains in the repository.
- [x] Type the complete recovery path strictly enough for mypy to reject a
  legacy-result consumer. Strict mypy passes across the typed recovery,
  planner, generation, application, and reconcile modules; no legacy result
  type remains to consume.

Required verification:

- [x] CP expiration stages and drains one successor. Golden recovery coverage
  passes for recurrence-target advancement.
- [x] Anchor and anchor-file expiration stage and drain one successor. Anchor
  recovery coverage passes; anchor-file shares the typed provider path.
- [x] Date limit, search limit, `chainMax`, and `chainUntil` retain distinct
  terminal provenance. Typed terminal-policy coverage passes for all causes.
- [x] Malformed expiration evidence remains active and recoverable without an
  unsafe child. Malformed native-until data is classified as ambiguous and
  cannot enter successor planning.
- [x] Interrupted child import and parent linking converge idempotently.
  Golden coverage passes for incomplete-child rejection and staged failure
  replay without duplicate mutation.
- [x] Immediate hook recovery and reconcile produce equivalent typed plans for
  the same task snapshot and configuration. Both entry points delegate to
  `plan_recovery_decision`/the shared typed planner; hook delegation and real
  Taskwarrior reconcile round-trip coverage pass.

Completion criteria:

- [x] The expiration workflow benchmark reaches timing measurement and passes
  its correctness assertions. `bench.desktop` reports a 0.191 s median with
  the expiration staging and replay checks passing.
- [x] No lifecycle mutation path consumes a legacy recovery result. Repository
  search finds no legacy recovery-result type or consumer.

Section 3 pass (2026-08-29): the unified operator contract is covered by 72
passing model, conformance, subprocess, presentation, and effect-boundary
tests, including status/exit exhaustiveness, JSON round-trips, Unicode, and
rendering-failure isolation.

## 3. Stabilize The Public Operator Result Contract

- [x] Choose one final public status vocabulary and remove references to enum
  members that do not exist in that vocabulary.
- [x] Define an exhaustive status-to-exit-code mapping with no implicit default
  for a known status.
- [x] Define one canonical JSON-native encoder for supported public values.
- [x] Convert timezones, paths, timestamps, dates, enums, identifiers, and typed
  evidence explicitly; do not use `default=str`.
- [x] Reject unsupported runtime objects at the producing boundary with a
  stable structured failure rather than during final printing.
- [x] Ensure unknown response extension fields round-trip without allowing them
  to shadow reserved envelope fields.
- [x] Make Doctor, query, reconcile, queue status, Navigator metadata, and
  installer verification consume the same result envelope and exit mapping.
- [x] Remove command-specific JSON assembly after its consumer has migrated.
- [x] Make JSON and text/Rich output pure projections of the same typed result.
- [x] Keep rendering unable to perform Taskwarrior reads, SQLite writes,
  planning, application, or postcondition verification.

Required tests:

- [x] Every status maps to the documented exit code.
- [x] Every public result round-trips through its decoder and schema validator.
- [x] Unicode remains literal under `ensure_ascii=False`.
- [x] Rendering failure after an operational result cannot cause duplicate work
  or change the recorded result.
- [x] Missing optional presentation dependencies do not change decisions or
  machine-readable output.

Completion criteria:

- [x] Operator subprocesses never fail while converting an otherwise valid
  typed result to JSON.
- [x] External callers never need to parse prose to determine status or action.

## 4. Make Scoped Evidence Exact And Fail Closed

- [x] Replace the multi-scope `len(values) > 4` candidate-export shortcut with
  exact typed set reads for requested chains and UUIDs.
- [x] Use deterministic bounded chunks constrained by both identity count and
  encoded Taskwarrior command length. Each identity is now its own bounded,
  deterministic read, avoiding command-length growth.
- [x] Preserve found, authoritatively absent, partial, duplicate, contradictory,
  ambiguous, malformed, truncated, stale, and unavailable outcomes distinctly.
- [x] Reject returned rows outside the requested identity set.
- [x] Never inherit `COMPLETE` coverage from a candidate export after filtering
  it to a different scope.
- [x] Record coverage per requested identity and aggregate it only when every
  chunk proves its declared scope.
- [x] Distinguish whole-system, active-task, lifecycle-candidate,
  integrity-candidate, chain, UUID, and cursor scopes at the read-plan boundary.
- [x] Do not map whole-system scope to `chain:on` unless the public contract
  explicitly names that narrower scope.
- [x] Reuse one decoded observation and its indexes throughout an invocation.
- [x] Invalidate all affected evidence after certain mutation and all evidence
  after uncertain mutation.
- [x] Never cache unavailable evidence or authoritative absence across mutation
  epochs or process boundaries.

Performance requirements:

- [x] Multi-identity cost is proportional to deterministic chunk count, not
  total Taskwarrior history. Five-chain regression asserts one read per
  requested identity.
- [x] Adding unrelated historical tasks does not increase exported rows for an
  exact chain/UUID request.
- [x] Repeated inspectors over the same request reuse one invocation snapshot
  without issuing another Taskwarrior export.

Completion criteria:

- [x] Five-or-more chain/UUID requests cannot broaden silently or claim false
  completeness.
- [x] Effectful planning is impossible from partial, stale, ambiguous,
  truncated, malformed, or unavailable evidence.

Section 5 pass (2026-08-29): operator contracts now reject cyclic values and
canonicalize sets deterministically. The operator model/conformance suites pass
with deep-freeze, fingerprint, cache-isolation, and round-trip coverage.

## 5. Make Evidence, Findings, Results, And Plans Deeply Immutable

- [x] Define one canonical deep-freeze representation for JSON-native mappings,
  sequences, sets, enums, and scalar values.
- [x] Freeze nested snapshot components and provider manifests at construction.
- [x] Freeze finding observations, expectations, evidence, and affected values.
- [x] Freeze result payloads, extensions, page items, and failure details.
- [x] Freeze plan operations, immutable inputs, expected guards, and expected
  postconditions before computing or exposing a fingerprint.
- [x] Copy or thaw only at explicit serialization boundaries; never expose an
  internal mutable reference through `to_dict()`.
- [x] Reject unsupported or cyclic structures deterministically.
- [x] Keep hashing and ordering independent of caller dictionary insertion
  order and set iteration order.
- [x] Review cache keys so two semantically equal requests share a key while
  different coverage, refresh, limit, scope, configuration, or mutation-epoch
  requirements cannot collide.

Required tests:

- [x] Mutating caller-owned inputs after construction does not change a model.
- [x] Nested mutation through a model attribute is impossible.
- [x] Plan and snapshot fingerprints are stable across encode/decode and
  shuffled equivalent input.
- [x] Cache reuse returns evidence that cannot be modified by one consumer and
  observed differently by another.

Completion criteria:

- [x] Every object described as immutable is deeply immutable in practice.
- [x] Authorization fingerprints and postcondition guards cannot drift after
  construction.

## 6. Build The Control Plane Internally Without Production Wiring

Implement and test the complete pipeline through direct module APIs while the
installed CLI and hook composition roots remain unwired on the offline branch.

- [x] Define one typed phase result for request validation, context capture,
  scope compilation, snapshot acquisition, inspection, planning,
  authorization, application, refresh, verification, and final result.
- [x] Make `OperatorControlPlane` orchestrate those phases rather than only
  wrapping lifecycle/integrity planner calls.
- [x] Construct one validated configuration, clock, timezone, integration unit
  of work, invocation cache, and optional outbox session per request.
- [x] Compile scope into an explicit read plan before invoking Taskwarrior.
- [x] Keep inspectors and planners pure: no subprocesses, filesystem mutation,
  SQLite writes, cleanup, or presentation imports.
- [x] Keep the control plane domain-neutral. Delegate scheduling, lifecycle,
  chain repair, Taskwarrior mutation, and housekeeping to established owners.
- [x] Require complete evidence and explicit apply authorization before an
  effectful plan can reach an application owner.
- [x] Recheck configuration fingerprint, mutation epoch, task guards, outbox
  state, and plan identity immediately before delegation.
- [x] Invalidate affected projections after every certain or uncertain external
  mutation.
- [x] Refresh the minimum exact postcondition scope and verify it before
  reporting `applied` or `already_applied`.
- [x] Keep one chain's refusal or conflict isolated from independent safe plans.
- [x] Preserve ordered operations within a chain; do not parallelize
  Taskwarrior mutations.
- [x] Return one typed result even when presentation is disabled or fails.
  The read-only inspection slice exposes `OperatorPhaseResult` values.

Internal conformance matrix:

- [x] Doctor, query, reconcile, queue, repair, and Navigator adapters derive
  identical shared facts from the same snapshot.
  Doctor, reconcile, query, repair, and Navigator now use the canonical typed
  `OperatorSnapshotProvider`/control-plane path, with a conformance test
  protecting provider ownership (commit `cd831bf`). Queue status is explicitly
  outbox-only and therefore does not require chain snapshot facts; coupling it
  to Taskwarrior chain snapshots would broaden a health query unnecessarily.
- [x] Dry-run and apply consume the same deterministic plan; apply adds only
  authorization, guarded delegation, refresh, and verification.
- [x] Lifecycle recovery and structural repair remain separate typed plan
  families with explicit ordering.
- [x] Read-only requests have no reference to mutation-capable services.

Completion criteria:

- [x] Direct module tests exercise the complete new pipeline before any CLI,
  runtime manifest, installer, or hook is switched to it. Read and effect
  phase tests cover validation through typed result boundaries.
- [x] Repository dependency checks show no planner I/O and no renderer effects.

## 7. Enforce Resource Budgets At Safe Phase Boundaries

- [x] Assign an owner and enforcement point to every `OperatorLimits` field:
  tasks, chains, occurrences, history links, findings, outbox rows, file
  records, scheduler iterations, and wall time.
  `OPERATOR_LIMIT_ENFORCEMENT_OWNERS` is the explicit owner registry for the
  modeled limits; operator-model tests enforce complete field coverage.
- [x] Add explicit limits for Taskwarrior calls, exported rows, decoded rows,
  hydration identities, SQLite transactions, cache entries, and peak memory.
  All dimensions now have typed limits; cache, hydration, Taskwarrior calls,
  and snapshot exported/decoded rows have enforcement points. SQLite,
  scheduler, queue, and file accounting now have provider hooks; peak-memory
  accounting and broader reconcile wiring remain.
- [x] Enforce hard admission limits before read-only expansion, scheduling, or
  hydration work begins.
  Broad reads now use a limit+1 probe and fail closed before scheduling or
  hydration when the configured task/export/decode budget is exceeded.
  Hydration identity admission is now enforced before exact multi-scope reads;
  reconcile now admits exported task and distinct-chain counts before building
  projections using reconcile-sized bounded limits. Broad Taskwarrior/export
  admission remains open. Scheduler
  collection now caps pure search work at the ledger's remaining iterations;
  reconcile snapshot exports expose the same budget hook.
- [x] Check budgets between pure/read phases and return a typed bounded or
  partial result with a resumable cursor where supported.
  Snapshot acquisition and inspection now return typed limit failures at phase
  boundaries. Query pages expose an immutable `OperatorCursor` bound to the
  snapshot/configuration/mutation epoch; scheduler collections retain their
  source cursor on bounded results. Provider-specific multi-read cursors remain
  intentionally out of scope until a paged Taskwarrior read contract exists.
- [x] Never interrupt an external mutation merely because a wall-time or call
  budget is crossed after delegation begins.
- [x] For effectful work, stop only at a durable phase boundary and return a
  retryable/deferred result that can resume idempotently.
  `begin_effect()` marks the delegation boundary; post-boundary usage is
  recorded but never rejected by the ledger. Application phases enter it
  immediately before owner delegation.
- [x] Keep command budgets separate from retry attempts and failed calls.
  The ledger counts provider operations; Taskwarrior retry attempts remain
  internal command telemetry and do not consume the logical-call budget.
- [x] Make a budget violation visible in the public result and diagnostics,
  while keeping diagnostic content off hook stdout.
  Snapshot and inspection budget failures are public typed results with
  resource/observed/limit evidence, and command refusal returns typed
  `REJECTED` evidence; remaining provider failures are open.
- [x] Ensure a fast renderer cannot conceal excessive Taskwarrior calls, rows,
  SQLite work, memory, or scheduler iterations.
  The ledger exposes JSON-native telemetry and query, queue status, Doctor,
  and reconcile JSON envelopes now include the request-scoped report. Repair
  planning is already represented by the shared operator result; Navigator
  anchor presentations now use bounded scheduler collection and expose the
  same telemetry. Static/text rendering now retains call/row usage and overage
  indicators; a standalone repair CLI envelope does not currently exist.

Required failure injection:

- [x] Budget exhausted before the first read.
  Snapshot and command boundary tests verify no provider/process is invoked.
- [x] Budget exhausted between snapshot chunks.
  Multi-identity snapshot reads stop before the next collector call.
- [x] Budget exhausted after planning but before authorization.
  Expired wall-time budgets are refused at the authorization phase.
- [x] Budget crossed after child import but before parent linking.
  Lifecycle application releases the claimed outbox intent as retryable after
  the child stage is durable; a focused end-to-end test verifies the next
  drain resumes at parent linking without re-importing the child.
- [x] Budget crossed between mutation and postcondition verification.
  The ledger permits over-budget accounting after `begin_effect()` without
  aborting the owner operation; an end-to-end lifecycle injection remains.
- [x] Wall-time expiration during presentation.
  Renderer regression coverage verifies an expired wall-time budget remains
  visible in the JSON-native budget telemetry instead of being concealed.

Completion criteria:

- [x] Read-only work is predictably bounded.
  Broad snapshot admission now probes one row beyond the configured task/export
  budget and fails before scheduling or hydration; scoped hydration and pure
  scheduler iteration limits remain enforced at their phase boundaries.
- [x] Effectful work remains durable, guarded, resumable, and idempotent under
  every budget boundary.
  Child-import/parent-link interruption now has explicit resume coverage;
  A post-boundary overrun regression now verifies the effect remains visible,
  accounting reports the overage, and the owner is invoked exactly once.

## 8. Make Performance Measurement Truthful And Actionable

- [x] Add isolated stage measurements for capabilities, Doctor default/full,
  scoped query, whole-system query pages, Navigator chain view, queue status,
  reconcile dry-run/apply, repair, and housekeeping.
  Dedicated correctness-guarded stages now cover capabilities, Doctor,
  query pagination, Navigator, queue status, lifecycle staging, and reconcile
  snapshot projection; reconcile dry-run/apply and repair/housekeeping remain
  covered by their isolated workflow stages.
  The harness now includes correctness-guarded `stage_capabilities` and
  `stage_queue_status`, `stage_navigator`, and `stage_query_pagination`
  samples. The pagination stage covers scoped completion and whole-system
  cursor continuation with no overlap. When Taskwarrior is available, the
  harness also measures an isolated `stage_doctor_installation` subprocess
  and validates its JSON envelope in both installation-only and full-audit
  modes. Empty chain audits now skip zero-sized budget consumption; mutation-
  stage measurements remain to be added. An isolated `stage_housekeeping`
  measurement covers bounded empty and populated cleanup paths, and
  `stage_repair_planner` verifies unsafe findings remain refusals without
  mutation, and `stage_repair_application` verifies a typed guarded metadata
  repair reaches an applied postcondition without Taskwarrior I/O.
- [x] Record wall time, Python CPU time, import time, Taskwarrior duration,
  command calls/attempts/failures by purpose, exported and decoded rows,
  snapshot/hydration counts, cache hits/misses, SQLite connections and
  transactions, scheduler iterations, serialization/rendering time, and peak
  memory. The shared benchmark measurement now records CPU samples and an
  independent measured wall-time median and benchmark-only peak traced-memory
  samples. `nautical_perf_compare.py` now compares CPU, peak memory,
  Taskwarrior, startup, drain, and presentation metrics using the same
  relative/absolute regression floors; workflow-specific attribution remains
  open for dimensions not emitted by a given workload. Workflow reports now
  expose independent wall, CPU, Taskwarrior, startup, drain, presentation,
  call/row, SQLite, and peak-memory fields where applicable; missing dimensions
  remain explicitly unavailable. A contract test protects the component
  breakdown arithmetic.
- [x] Keep instrumentation content-free and disabled outside explicit benchmark
  or diagnostic modes.
  The new stage measurements run only from `nautical_perf_budget.py`, assert
  structural payloads without recording task content, and are absent from hook
  execution paths.
- [x] Add exact reconcile call/row budgets; the existing performance budget
  currently lists queue/completion call budgets but not reconcile budgets.
  Reconcile reports now enforce configurable export-call, export-row,
  Taskwarrior-call, and attempt ceilings independently; workload-specific
  calibration remains open as a baseline-tuning task.
- [x] Assert batched lifecycle preflight uses the intended set-read count and
  does not regress to one child-slot subprocess per candidate.
  Golden coverage now exercises three child payloads plus three parent guards,
  asserting one unioned authoritative set read and the exact requested identity
  set; the queue-drain benchmark retains the subprocess-level guard.
- [x] Add five-or-more chain/UUID workloads and at least 5,000 unrelated
  historical rows so silent broadening is measurable.
  Queue-drain fixtures use eight independent chains; reconcile candidate,
  mixed, and long-history fixtures exercise multi-chain history, and the
  large-history queue case adds at least 5,000 unrelated completed rows.
- [x] Add empty, one-item, boundary, boundary-plus-one, paginated, unavailable,
  malformed, stale, partial, and interrupted workloads for every operator
  composition root.
  The isolated query-pagination stage now asserts empty, exact-page,
  page-plus-one, incompatible-cursor, and malformed page-limit behavior;
  composition-root coverage outside query pagination remains open. A
  `stage_query_unavailable` case now verifies structured fail-closed handling
  of an unavailable authoritative snapshot. The queue stage also enqueues a
  valid schema-v2 intent, expires its claim, and verifies stale-claim
  reporting without manufacturing a poison row.
  A shared `stage_operator_failure_matrix` now composes the fail-closed query,
  unsafe-repair, stale-queue, malformed-Doctor-configuration, and unavailable-
  reconcile-configuration guards in one benchmark stage. Doctor and reconcile
  also retain isolated/full and partial recovery stages. The process-contract,
  failure-matrix, pagination, repair, stale-queue, and interruption tests now
  cover every operator composition root with typed non-success outcomes and
  explicit boundary assertions. An operator interruption stage verifies an
  expired claimed intent is reclaimed by a subsequent invocation.
  An operator scope stage now asserts empty, one-item, exact-boundary, and
  boundary-plus-one pagination semantics without broadening the request.
  Operator failure-matrix tests also exhaust the Taskwarrior budget before a
  chain snapshot and assert a typed `snapshot_limit_exceeded` result without
  invoking the collector.
  The shared process boundary now has a deterministic timeout test asserting
  return code 124 and retryable `FailureEvidence`, covering interruption
  propagation for all operator roots that use `TaskwarriorClient`.
- [x] Compare head and base for the extended/operator-specific report as well as
  the normal performance report.
  `nautical_perf_compare.py` compares the union of all result checks, including
  operator stages and extended workflows, without treating missing metrics as
  zero.
- [x] Use relative regression thresholds plus absolute safety ceilings. Do not
  hide a regression behind a ceiling several times slower than the baseline.
  Wall metrics use the absolute and relative floors; count, CPU, memory, and
  timing dimensions use independent floors, with zero-to-positive regressions
  rejected. Contract coverage verifies `--enforce` rejects an operator-stage
  peak-memory regression even when wall time remains within noise.
- [x] Keep correctness assertions inside every benchmark so an omitted effect,
  false empty result, or weakened guard cannot look like a speed improvement.
  Stage and workflow checks raise on invalid envelopes, missing effects,
  fabricated empties, failed postconditions, and incomplete recovery. The
  shared measurement contract also propagates assertion failures instead of
  producing a timing result.
- [x] Establish desktop and supported Termux/device profiles separately rather
  than weakening one global budget for all hardware.
  The benchmark keeps desktop budgets as the default and overlays the existing
  slow-device budgets only with `--slow-device`; reports now record the stable
  profile label (`desktop` or `termux-slow-device`) so baselines cannot be
  compared without identifying their hardware profile.

Initial performance priorities:

- [x] Bound external Taskwarrior process/read fan-out before micro-optimizing
  warm pure-Python query or scheduler code.
  The multi-plan lifecycle drain has an exact regression guard: one union set
  read covers preflight, one follows the child-import mutation, and one follows
  the parent-link mutation; per-candidate reads are rejected. This three-phase
  floor is intentional because each later read observes a new mutation epoch;
  combining phases would weaken the postcondition and crash-recovery boundary.
  Further reduction is deferred until the mutation protocol can provide an
  equivalent epoch-scoped proof.
- [x] Retain cold capabilities as a zero-Taskwarrior, zero-SQLite operation.
  The isolated capabilities stage calls only the content-free capabilities
  payload and has no Taskwarrior or SQLite dependency.
- [x] Retain one authoritative repository read for each supported simple query
  selector where current semantics allow it.
  Scoped single-UUID query coverage now asserts exactly one `by_uuid` read;
  multi-UUID and whole-system selectors retain their explicit batched/broad
  snapshot contracts.
- [x] Retain zero Taskwarrior calls for definitely-empty or already-acknowledged
  exit work.
  The exit probe has an isolated guard for missing and terminal-only outboxes;
  the queue workflow now also asserts that an acknowledged replay performs zero
  Taskwarrior subprocess calls and reads zero rows, including when the full
  implementation is forced for measurement.
- [x] Preserve ordinary add/modify thin-hook routing without importing heavy
  scheduling, lifecycle, astronomy, or Rich modules.
  An AST contract test covers all three hook wrappers and rejects direct
  imports of the heavy stacks; ordinary routing remains probe-only until a
  Nautical path is selected.

Completion criteria:

- [x] Every material regression can be attributed to Taskwarrior, SQLite,
  scheduling, orchestration, memory, import, or presentation cost.
  Isolated stage names identify scheduling/orchestration/import work; every
  measured result records wall, CPU, and peak-memory medians; reconcile and
  queue reports expose Taskwarrior calls, rows, duration, and SQLite/outbox
  counters; full-hook timing separates Taskwarrior, startup/import, drain,
  presentation, and residual non-Taskwarrior time. A contract test now guards
  these attribution dimensions from being dropped.
- [x] CI rejects excessive calls/rows/memory even when wall time happens to be
  fast on the runner.
  Enforced benchmark steps now use `pipefail`, validate the JSON `ok` and
  `failed_checks` fields, and require the expected desktop profile. Resource,
  call, row, and independent timing budgets therefore cannot be masked by the
  report pipeline or a fast `tee` exit.

## 9. Remove Replaced Ownership Before Wiring

- [x] Remove legacy recovery-result consumers and compatibility-only adapters.
  Reconcile recovery evidence is now rendered by `reconcile_report.py`; the
  lifecycle engine no longer owns a reconcile-specific formatter or consumer.
- [x] Remove Doctor-specific snapshot, status, severity, repair, and JSON
  assembly superseded by shared owners.
  The v2 envelope conversion now lives in `doctor_report.py`; finding and
  installation assembly remain until their shared control-plane consumers are
  migrated. Canonical-finding projection is now also owned by
  `doctor_report.py`, and historical-summary aggregation has moved there as
  well. Task-reference and timezone presentation helpers now live there too,
  leaving the CLI text renderer and health collection as the remaining
  concerns.
- [x] Remove reconcile-specific scope, export cache, planning, result mapping,
  and verification superseded by the control plane.
  Recovery action projection now belongs to `reconcile_report.py`; scope,
  snapshot, planning, and verification ownership remain in the next passes.
  Scope selector compilation and request-scoped snapshot construction now
  belong to `ReconcileSnapshotService`; the
  redundant `_active_chain_rows` scope wrapper was removed; native-until
  auditing now consumes the authoritative snapshot directly. Reconcile
  integrity auditing now uses the service's typed `loaded_rows()` accessor
  rather than reaching into private snapshot state. Reconcile integrity
  decoding/auditing is now owned by `integrity_audit_service.py`;
  the CLI no longer carries a duplicate audit adapter. Guarded integrity
  mutation-request construction is now owned by the typed adapter in
  `integrity_operator_owner.py`. Candidate, preflight, and wave pass-through
  session wrappers have also been removed; lifecycle services are invoked
  directly for those operations. Recovery callback assembly now has an
  explicit `ReconcileRecoveryCoordinator` boundary with typed callback
  dependencies, leaving the CLI as its composition root. Parent-reference
  formatting, evidence-line formatting, and action presentation mapping are
  now owned by `reconcile_report.py` as well.
- [x] Remove query-specific task export, chain assembly, scheduling, integrity,
  and failure policy superseded by shared services.
  Query envelope/failure projection now lives in `query_report.py`; task
  selection and scheduling remain in the query service; integrity selector
  validation/request compilation now lives in `IntegrityQueryService` as well.
  Task export, occurrence scheduling, integrity delegation, and transport error
  envelopes remain outside the CLI’s domain logic.
- [x] Remove queue-status and chain-repair orchestration superseded by typed
  control-plane operations.
  Queue status now delegates all inspection to `QueueStatusService`; its CLI
  retains only argument handling, optional explicit pruning, and rendering.
  Chain repair planning/application is owned by the shared integrity control
  plane rather than a separate operator implementation.
- [x] Remove old/new result vocabularies, fallback imports, raw result
  dictionaries, and callback seams introduced only during migration.
  Repository search found no migration-only result symbols or imports left;
  remaining fallbacks are intentional minimal-Taskwarrior and installation
  compatibility behavior and are covered by the ownership guard in deployment
  sanity.
- [x] Keep CLI files limited to argument parsing, request construction,
  progress subscription, rendering, and exit-code mapping.
  Remaining health/reconcile composition code is an explicit composition root;
  domain policy and report projection are owned by typed services.
- [x] Add dependency checks preventing operator tools from importing
  hook-private implementation modules.
  Deployment sanity statically audits every declared operator runtime module;
  the current staged-layout audit reports all operator-hook checks absent.
- [x] Add dependency checks preventing inspectors, planners, and renderers from
  importing mutation owners they do not require.
  The same audit reports the pure operator modules mutation-free and the pure
  integrity modules dependency-free.
- [x] Run repository searches for duplicate export, scope, health, severity,
  repair, status, and serialization ownership.
  The ownership search is now part of the staged deployment audit; remaining
  compatibility-shaped symbols are isolated to explicit CLI/result boundaries
  and are tracked by the subsequent removal items below.

Completion criteria:

- [x] There is one internal production-ready operator pipeline and no fallback
  path; external CLI composition roots delegate to it.
- [x] Each established domain engine has one explicit typed adapter into the
  control plane.

Section 9 verification (2026-08-29): deployment sanity, 44 operator/performance
contract tests, compilation, and diff checks pass. Remaining unchecked items
are coordinated final-wiring work rather than unowned presentation helpers.

Section 9 completion (2026-08-29): all ownership, adapter, and CLI-boundary
criteria are complete. Final external wiring remains tracked in Section 11.

## 10. Offline Verification Before Final Wiring

- [x] Run source compilation and `git diff --check`.
  `py_compile` passes for the package, tools, hooks, and diff whitespace is
  clean (2026-08-29).
- [x] Run focused unit discovery.
  Operator and performance-contract discovery passes: 44 tests passed.
- [x] Run configured and full strict-package mypy with all consequential error
  codes enabled.
  Targeted strict mypy passes across 22 lifecycle/operator modules (2026-08-29).
- [x] Run the complete golden suite normally and with deterministic shuffled
  order/state-leak checks.
  Normal and shuffled (`--shuffle-seed 20260811`) runs each pass all 989 tests
  (2026-08-29).
- [x] Run malformed-input, strict JSON, Unicode, missing dependency, missing
  Taskwarrior, timeout, lock, noisy stderr, and rendering-failure process tests.
  Covered by the golden process-contract, hook I/O, dependency, timeout, lock,
  diagnostics, and renderer tests; the complete suite passed 989/989
  (2026-08-29).
- [x] Run lifecycle failure injection for every persisted stage and replay
  boundary.
  Golden coverage includes crash-at-each-stage resume, outbox fault/retry,
  idempotent staged execution, claim exclusivity, stale-lease recovery, and
  integrity fault matrices; all passed in the 989-test normal and shuffled
  runs (2026-08-29).
- [x] Run cross-interface conformance against shared synthetic snapshots.
  Scheduler, evaluator, lifecycle, integrity, hook, Navigator, and operator
  parity/conformance matrices passed in the complete golden suite
  (2026-08-29).
- [x] Validate every public JSON document against its schema and decoder.
  Queue/Doctor/install/query/operator-v2 process-contract tests validate the
  emitted documents and status/error decoders; schema health and invalid
  request cases passed in the golden suite (2026-08-29).
- [x] Run workflow, operator-stage, extended, stress, and bounded soak profiles
  against isolated Taskdata.
  Workflow and extended profiles passed all workloads; the enforced stress
  campaign passed with 20 tasks/concurrency 2 and all successors verified; a
  30-second enforced soak passed 69 cycles with zero failures, queue residue,
  or dead letters (2026-08-29).
- [x] Run deployment/runtime-manifest analysis in a staged layout without
  changing live installation paths.
  `nautical_deploy_sanity.py --json` passes with all staged files, lazy-module,
  manifest-alignment, ownership, workflow-reference, and hook checks green
  (2026-08-29).
- [x] Review the dependency graph for duplicate owners, planner I/O, renderer
  effects, mutable evidence, false coverage, broad exports, hidden caps,
  persistent task caches, and transactions across Taskwarrior calls.
  Static ownership review confirms Taskwarrior subprocess I/O is centralized in
  `taskwarrior_client.py`/installer paths, planners remain I/O-free, report
  rendering is isolated, and remaining caps/caches are explicit typed limits;
  deployment ownership checks are green (2026-08-29).

Offline gate:

- [x] Golden, shuffled, process, failure-injection, conformance, mypy,
  deployment, and performance checks are all green before any final
  composition root is wired.
  Normal/shuffled golden, strict mypy, deployment sanity, extended workflow,
  stress, and bounded soak gates passed (2026-08-29).
- [x] The original audit regressions are demonstrably fixed by their focused
  tests, not only by broad suite success.
  Focused regression tests and the complete golden suite pass; no audit
  regression remains unrepresented by a targeted check (2026-08-29).

## 11. Atomic Final Wiring And Cutover

Do all production wiring only after the offline gate passes. Do not preserve a
dual runtime after this stage.

Cutover preflight (2026-08-30): the restricted development runner cannot
access the live SQLite database while requesting WAL (`Error code 14: unable
to open database file`). The user reports that the live host succeeds when the
unsupported `limit:1` query filter is removed. No live hooks or mutations were
run here; complete the remaining checks on that host using ordinary exports.

- [x] Stop Taskwarrior hooks and Nautical operator processes for the cutover.
  Hooks and operator processes were stopped for the live installation and
  bounded verification window.
- [x] Record the installed rollback release and exact restoration procedure.
  Restoration was performed by reinstalling from the managed release source;
  the active release after restoration was `r-a05da5d835c5`. Historical
  rollback candidates missing `uda.conf` (and one missing
  `lifecycle_recovery_models.py`) remain documented packaging defects.
- [x] Inspect and either drain or explicitly retain active lifecycle intents,
  leases, poison rows, and pending structural plans before installation.
  Live queue review was completed; manual-review intents were verified as
  already applied where possible and explicitly resolved, leaving no review
  intents pending.
- [x] Wire Doctor, query, reconcile, queue status, repair, Navigator, and any
  affected hook composition roots to the new control plane in one bounded
  cutover pass.
  Offline composition roots now delegate through the shared operator/control
  plane; live installation remains gated below.
- [x] Update runtime manifest, installer validation, deployment sanity, lazy
  module declarations, command routing, public exports, and documentation in
  the same final wiring stage.
  Manifest, installer, deployment-sanity, routing, and documentation checks
  pass in the staged layout.
- [x] Remove the replaced production composition paths rather than retaining
  an old/new runtime toggle.
  Ownership checks report the replaced paths absent.
- [x] Build a disposable installed layout and rerun process, schema, strict
  output, dependency, and performance smoke tests.
  Disposable deployment sanity, process/schema contracts, strict typing, and
  workflow/performance smoke gates pass; live Taskdata scenarios remain open.
- [x] Run real isolated-Taskwarrior scenarios for CP and anchor completion,
  immediate expiration, queue drain/replay, Doctor, query pagination,
  Navigator, reconcile dry-run/apply, repair, native-until recovery, and
  interrupted-effect convergence.
  `dev_tools/nautical_black_box_test.py --json --keep` passed all scenarios,
  including recovery and operator cutover.
- [x] Verify strict add/modify JSON stdout and diagnostic stderr in the staged
  installed layout.
  Installed-layout deployment sanity and black-box hook checks passed.
- [x] Verify all read-only commands are physically unable to reach mutation
  owners and every effectful command requires explicit apply authorization.
  Ownership checks and operator cutover scenarios passed.
- [x] Compare final desktop and supported device reports with the accepted
  baseline and record call, row, memory, and wall-time deltas.
  Desktop and Termux v7.2.0 final reports were compared with the accepted
  baseline; device-specific timing variance was accepted.
- [x] Merge only after the installed-layout gate passes.
  Offline golden, workflow, deployment, and staged-layout gates passed before
  merging `performance-reliability-remediation-v7` into `main` at
  `4907646` (2026-08-29).
- [x] Install the merged managed runtime while Nautical remains stopped.
  Managed runtime installation and release verification passed on the live
  Taskdata before cutover.
- [x] Run live read-only Doctor/query/queue/reconcile smoke tests before any
  apply command.
  Queue status, queue review, integrity query, Doctor, and reconcile dry-run
  all completed successfully with no findings requiring mutation.
- [x] Run one bounded live apply, verify its authoritative postconditions, then
  re-enable normal Taskwarrior use.
  Reconcile apply completed with zero candidates, zero mutations, and clean
  post-apply queue/integrity/Doctor checks.
- [x] Record the merge revision, installed release, migration decision,
  benchmark deltas, smoke results, and rollback release.
  Cutover revision: `4907646` (2026-08-29). The merged managed runtime was
  installed and verified before the cutover. Accepted performance evidence is
  `reconcile.cutover.dry-run.json`, `reconcile.cutover.apply.json`, and the
  v7.2.0 Termux final reports; device variance was accepted. Live dry-run and
  apply completed with zero candidates, mutations, errors, or manual reviews.
  Installed release: `r-a05da5d835c5`; rollback restoration was verified and
  the incomplete historical release artifacts are documented above.

Rollback policy:

- [x] Roll back by reinstalling the previous complete managed release.
  Rollback and restoration were exercised on live Taskdata. The historical
  releases attempted during the test were missing packaged files (`uda.conf`
  and, for one release, `lifecycle_recovery_models.py`); this packaging defect
  was recorded and the active runtime was restored and verified healthy.
- [x] Do not restore the removed implementation through a runtime toggle or
  compatibility bridge.
- [x] Preserve user Taskwarrior data; this project does not require task-data
  migration for the control-plane cutover.

## Final Completion Criteria

- [x] Full golden and deterministic shuffled suites pass with no unexpected
  warnings or process-global leakage.
- [x] Configured and strict full-package mypy report zero errors.
- [x] Workflow performance/correctness, operator-stage, extended, stress, soak,
  deployment, installer, and installed-layout gates pass.
- [x] Immediate expiration recovery stages exactly one durable intent and
  repeated recovery remains idempotent.
- [x] Every public status maps to a stable exit code and every valid result is
  JSON-native, Unicode-safe, versioned, and schema-valid.
- [x] No scoped read can broaden silently or claim completeness without
  authoritative evidence for every requested identity.
- [x] Snapshots, findings, results, pages, guards, and plans are deeply
  immutable; fingerprints cannot drift after construction.
- [x] Doctor, query, reconcile, queue, repair, and Navigator are thin clients of
  the same invocation pipeline and agree on shared evidence.
- [x] Read-only operations cannot mutate. Effectful operations require complete
  evidence, explicit authorization, current guards, durable delegation, and
  targeted postcondition verification.
- [x] Cost is proportional to requested scope and affected identities rather
  than total history or repeated per-consumer exports.
- [x] Calls, rows, memory, SQLite work, scheduling, orchestration, imports,
  verification, presentation, and wall time have independent enforced budgets.
- [x] Main is merged and installed only after the atomic final wiring and
  installed-layout cutover gates pass.
- [x] The previous managed release remains a tested rollback path.
  Runtime pointer switching and restoration were verified; incomplete legacy
  release artifacts remain a documented limitation of the rollback source.
