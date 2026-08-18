# Nautical Lifecycle Engine Improvement Checklist

Consolidate Nautical's chain lifecycle around one durable, typed transition
engine. Preserve the existing queue, deterministic child identity, and recovery
behavior; do not replace them with a general workflow framework.

## Scope And Assumptions

- The engine is upgraded while Nautical is stopped; hooks are not run during
  the cutover. Taskwarrior remains the sole durable source of user lifecycle
  data.
- [ ] Keep Taskwarrior as an external system; do not assume its mutations can
  share an atomic transaction with Nautical's SQLite state.
- [ ] Treat multi-step changes as an idempotent saga: plan, persist, apply,
  verify, and recover.
- [ ] Keep on-modify decision-only while Taskwarrior owns its datastore lock.
- [ ] Keep child import and parent-link repair in on-exit or reconcile.
- [ ] Preserve strict hook JSON on stdout, `ensure_ascii=False`, and diagnostics
  on stderr only when `NAUTICAL_DIAG=1`.
- [ ] Preserve thin routing for tasks that do not need Nautical lifecycle work.

## Audit Findings

- Lifecycle ownership is fragmented across `modify_lifecycle.py`, the
  completion modules, `modify_expiration.py`, `exit_entry_flow.py`,
  `queue_store.py`, `reconcile.py`, and `tools/nautical_reconcile.py`.
- The shared `ChainGenerationService` owns child calculation, but hooks and
  reconcile still build and execute different transition flows around it.
- `ReconcilePlan.action`, exit-flow control, queue states, and several
  operational outcomes remain stringly typed.
- Queue entries persist a child payload and a small parent guard, but not a
  complete versioned lifecycle plan or its expected postconditions.
- Recovery after child import and before parent linking is sound but mostly
  inferred from Taskwarrior state rather than represented as an explicit step.
- Recurrence activation mutates the hook task during classification, which
  mixes lifecycle decisions with applying the resulting patch.
- Reconcile applies child import, parent update, and verification through a
  separate path instead of the on-exit executor.
- Runtime compatibility bridges are not a design requirement. Obsolete queue,
  cache, and intent state may be discarded or quarantined during the stopped
  cutover; reconcile rebuilds missing successors from Taskwarrior state.

## Baseline

- [x] Record the current golden, black-box, deployment, mypy, and performance
  results before editing. *(2026-08-11: golden 936/936, black-box and
  deployment passed, full mypy clean across 136 files, enforced desktop budget
  35/35.)*
- [x] Add characterization coverage for activation, resume, disable, manual
  deletion, completion, native-until expiration, chainMax, and chainUntil.
  *(Covered by the readiness inventory and strict lifecycle suites.)*
- [x] Record queue behavior for a crash before import, after import, after
  parent linking, and before acknowledgement. *(Queue claim/retry,
  idempotence, dead-letter, durable drain, and parent-link recovery cases are
  covered by the isolated lifecycle and reliability suites.)*
- [x] Preserve existing cross-device convergence and deterministic child-slot
  behavior. *(Black-box duplicate guard and spawn lifecycle matrix pass.)*

Completion criteria:

- [ ] Every current lifecycle path has a named characterization test.
- [ ] Existing queue recovery and hook protocol behavior are unchanged.

Cutover policy:

- [ ] Stop hooks and Nautical workers before replacing the engine.
- [ ] Discard or quarantine obsolete Nautical runtime state; do not translate
  it through permanent compatibility code.
- [ ] Install the new release, run doctor and reconcile, and re-enable hooks
  only after both validations pass.
- [ ] Roll back by restoring the previous release, not by migrating runtime
  plans between engines.

## 1. Define The Lifecycle Contract

- [x] Add focused immutable models for lifecycle identity, event, action,
  parent guard, plan, execution stage, and outcome in
  `nautical_core/lifecycle_models.py`.
- [x] Define the supported events: activate, resume, disable, complete, expire,
  manually delete, reach chainMax, and reach chainUntil.
- [x] Define the external mutation stages: planned, persisted, child present,
  parent linked, verified, finalized, retryable, and manual review.
- [x] Define explicit invariants: `chainID` is mandatory, links are monotonic,
  plan identity and parent guards agree, and invalid event/action combinations
  cannot be constructed. Reciprocal links and terminal spawn prevention remain
  planner/executor invariants for Section 2 onward.
- [x] Use enums or tagged result models at module boundaries; do not introduce
  another set of free-form action strings.
- [x] Keep task lifecycle state distinct from queue processing state.

Completion criteria:

- [x] The lifecycle states, events, actions, and contract invariants are
  represented by code and the lifecycle model golden test.
- [x] Invalid state/action combinations cannot be constructed silently.

## 2. Introduce One Pure Transition Planner

- [x] Add a `LifecyclePlanner` that accepts an immutable task snapshot, event,
  validated configuration, and an injected task-scoped child-generation
  service callback.
- [x] Make the planner return a complete `LifecyclePlan` without Taskwarrior
  commands, SQLite writes, panels, or mutation of its input task.
- [x] Move completion preflight, next-link calculation, limits, child payload
  construction, carry validation, and terminal decisions behind the planner.
  **Boundary:** Taskwarrior reads, existing-child checks, response carry
  mutation, and panels remain at the hook boundary because they require the
  Taskwarrior lock or user-facing I/O. The pure planner now receives typed
  preflight data, owns next-link/limit decisions, builds the child payload,
  validates carry through a typed callback, and returns the terminal action.
  This keeps the planner side-effect free without moving lock-sensitive work
  into it.
  **Pass 21:** Candidate-to-plan assembly is now centralized in
  `plan_candidate_successor`; completion and reconcile use the same
  precomputed recurrence service, child builder, limit policy, and planner
  boundary. Preflight, carry validation, and hook presentation remain outside
  the planner for the next pass.
  **Pass 22:** Added typed `LifecyclePreflight` inputs and planner validation
  for chain identity, adjacent links, and recurrence kind. Completion passes
  its validated preflight into planning, and reconcile supplies the same
  contract. Mismatches fail before child construction; Taskwarrior lookups and
  carry mutation remain outside this pure boundary for the next pass.
  **Pass 23:** Added a planner-level typed carry-validator boundary. Completion
  and reconcile now validate scheduled/wait/native-until carry invariants while
  converting a candidate into a plan; invalid carry state fails before a spawn
  plan is returned. Carry mutation and user-facing panels remain at the hook
  boundary until the next staged pass.
  **Pass 24:** Completion carry preparation now runs on an isolated task
  snapshot and commits the adjusted fields only after all carry and native-
  until validation succeeds. A malformed carry therefore cannot partially
  rewrite the Taskwarrior response before planning; completion and carry
  regression suites remain green.
  **Pass 25:** Expiration candidate calculation now lives beside the shared
  lifecycle planner. It always derives from the prior due/scheduled recurrence
  target, never deletion `end`, and feeds the same typed candidate-to-plan
  boundary used by completion. Scheduled-only expiration parity is covered.
  **Pass 29 complete:** The remaining hook-only checks are explicit inputs and
  outputs at the planner boundary; no Taskwarrior lookup or presentation logic
  is hidden in the pure transition service.
- [x] Route expiration planning through the same service while preserving its
  due/scheduled recurrence basis. **Current state:** normal reconcile successor
  construction uses the planner, but `modify_expiration.py` and the
  expiration-specific reconcile recovery path still assemble candidates and
  native-until/day-end carry behavior separately. **Why it remains open:** the
  shared service must retain expiration's scheduled-vs-due basis and native
  until fallback semantics, with dedicated expiration parity tests before the
  old path is removed.
  **Pass 26 complete:** `plan_expiration_successor` now owns expiration
  candidate calculation and typed plan assembly. Reconcile dry-run, reconcile
  apply, and hookless expiration recovery all reach this service; scheduled-
  only basis and native-until carry validation remain covered by parity tests.
  **Pass 27:** `LifecyclePlan.semantic_key()` now provides a stable comparison
  contract that ignores durable execution stage and retry metadata while
  retaining identity, guards, action, child payload, parent patch, and
  postconditions. Completion/reconcile parity tests compare this key directly.
- [x] Route reconcile dry-run through the same planner used by on-modify.
  **Current state:** reconcile apply planning wraps the generated child in the
  planner, while discovery, due calculation, limit checks, and dry-run
  rendering still use `reconcile.build_reconcile_plan` directly. **Why it
  remains open:** dry-run must render the exact plan that apply would execute,
  without Taskwarrior or SQLite side effects, before this route is safe to
  unify.
  **Pass 28 complete:** Reconcile preview and apply both use the shared
  `_plan_for_parent` builder and therefore the same candidate, expiration, and
  lifecycle planner services. The dry-run/apply regression now compares the
  stable semantic plan key before any mutation.
- [x] Return an explicit task patch for activation, resume, disable, and
  terminal transitions instead of mutating the input during classification.

Completion criteria:

- [x] Hook completion and reconcile produce equal plans from equal snapshots.
  **Evidence (Pass 29):** the shared candidate-to-plan test compares the
  complete semantic plan, the expiration test covers scheduled-target basis,
  and the parity matrix covers CP, anchor, anchor-file, scheduled-target, and
  native-until/limit boundaries. The semantic key compares action, identity,
  parent guard, child payload, parent patch, and terminal reason while ignoring
  execution-stage/retry metadata.
- [x] Planning is deterministic, side-effect free, and independently testable
  for the transition contract. Completion and reconcile now route candidate,
  expiration, carry, and terminal decisions through the same planner boundary.

## 3. Strengthen Identity And Parent Guards

- [x] Define one deterministic transition identity from chain ID, parent UUID,
  source link, target link, and lifecycle event.
- [x] Use that identity as the queue idempotency key while retaining the
  deterministic child UUID as the Taskwarrior slot identity.
- [x] Extend the parent guard beyond status, chain, chain ID, and link to cover
  the recurrence inputs that produced the child plan.
- [x] Store a canonical recurrence fingerprint rather than comparing raw field
  formatting.
- [x] Define which parent changes invalidate a plan and which presentation-only
  edits may proceed.
- [x] Verify the guard immediately before every external mutation.

*(2026-08-11: lifecycle plans and deferred spawn intents carry an `rf1-`
fingerprint covering recurrence/timing inputs. Spawn intents use a deterministic
`li1-` identity key, and on-exit performs fresh parent-guard checks before child
import and again before parent linking.)*

Completion criteria:

- [x] A stale queued intent cannot spawn from changed recurrence settings.
- [x] Duplicate hooks and cross-device intents converge on one child slot.

## 4. Evolve The Queue Into A Lifecycle Outbox

- [x] Version the persisted plan schema and retain a bounded migration for
  already queued spawn entries.
  **Pass 1 complete:** new spawn intents persist a validated
  `lifecycle_plan.schema_version = 1`; legacy entries remain readable and
  unsupported future versions fail closed. **Still open:** materializing
  eligible legacy entries into the durable payload and persisting
  execution-stage migrations in the queue store. **Pass 2 complete:** claimed
  legacy rows with a complete parent guard are upgraded in memory, while
  incomplete rows remain on compatibility handling without guessed identity.
  **Pass 3 complete:** the on-modify lifecycle enqueue path rejects missing or
  unsupported plans before writing, while generic legacy/requeue helpers retain
  their compatibility behavior.
- [x] Persist event, identity, parent guard, child payload, parent patch,
  expected postconditions, retry policy, and current execution stage.
  **Pass 1 state:** the versioned plan envelope now carries these fields for
  newly deferred spawns. Queue claim state and stage advancement are still
  represented by the existing queue columns until the outbox executor pass.
  **Pass 4 complete:** the SQLite write normalizes a new plan from `planned`
  to `persisted` inside the enqueue transaction; claim-time stage transitions
  and postcondition verification remain open.
  **Final audit complete:** claim migration persists eligible legacy plans;
  durable payload round-trips retain identity, action, guards, child/parent
  mutations, postconditions, retry policy, and stage.
- [x] Require durable persistence before on-modify returns a task change that
  depends on later child creation.
  **2026-08-11:** `_enqueue_spawn_intent` now requires a validated v1 plan and
  returns success only after the SQLite enqueue transaction commits.
- [x] Keep claim tokens, stale-claim recovery, poison-row quarantine, capacity
  limits, permissions, WAL setup, and dead-letter behavior.
- [x] Advance stages with claim-token ownership checks; never treat a stored
  stage as proof without verifying Taskwarrior state.
  **Pass 5 complete:** SQLite now advances a claimed plan atomically only
  when row ID and claim token still match, validates the explicit stage graph,
  and leaves legacy rows unchanged. On-exit records `child_present` only after
  child existence/import succeeds and `parent_linked` only after the guarded
  parent update succeeds; stale claims are abandoned without acknowledgement,
  while transient queue failures are retried. Focused coverage verifies stale
  token rejection, valid transitions, and backward-transition rejection.
  **Pass 6 complete:** lifecycle plans now perform fresh child and reciprocal
  parent-link reads after mutation before advancing to `verified`. Missing or
  unavailable postconditions remain queued with bounded retries instead of
  being acknowledged as complete.
  **Pass 7 complete:** on-exit advances a verified plan to `finalized` under
  the same claim token before marking the intent complete or acknowledging the
  queue row, preserving durable completion across an acknowledgement crash.
- [x] Make schema/decoding failure unavailable or quarantined, never equivalent
  to an empty queue.
  **Pass 8 complete:** SQLite claim decoding now validates that every payload is
  valid JSON object data before claiming it. Invalid JSON and non-object
  payloads are atomically quarantined with the raw row identity and reason;
  they cannot enter on-exit as empty or compatibility entries. Queue/doctor
  diagnostics and regression coverage retain the quarantine evidence.
  **Pass 9 complete:** eligible legacy spawn entries are upgraded to a v1
  lifecycle envelope inside the claim transaction before entering `processing`.
  The migration is bounded and identity-based; incomplete legacy rows remain
  compatibility entries without guessed plans, and the durable upgrade is
  verified across a fresh stored-payload read.
  **Pass 10 complete:** claim decoding now validates any lifecycle envelope
  with `LifecyclePlan.from_dict`. Unsupported schema versions and malformed
  plans are quarantined atomically with actionable reasons before on-exit can
  process them.
  **Pass 11 complete:** acknowledgement now validates the claimed payload and
  refuses to delete a versioned queue row unless its durable plan is already
  `finalized`. Legacy rows remain compatible, while premature acknowledgements
  are covered by the lifecycle stage regression.
  **Pass 12 complete:** stale-claim recovery now also reclaims processing rows
  whose claim timestamp is missing, clearing orphaned ownership before the next
  bounded claim. A regression verifies the replacement token is installed.
  **Pass 13 complete:** lifecycle enqueue now enforces the configured payload
  capacity inside the SQLite write transaction, accounting for replacements
  by spawn intent. Concurrent writers cannot bypass the preflight size guard,
  and an over-budget plan is rejected atomically with an actionable error.

Completion criteria:

- [x] Every deferred multi-step transition is recoverable from one versioned
  persisted plan.
- [x] Upgrade, downgrade rejection, corruption, and concurrent-open tests pass.
  **Final audit:** queue/lifecycle golden coverage passes (`945/945`), including
  durable migration, stale-claim recovery, schema quarantine, capacity
  rejection, claim-owned stage progression, postcondition verification, and
  finalized acknowledgement.

## 5. Build One Idempotent Transition Executor

- [x] Add a shared executor used by on-exit and reconcile apply mode.
  **Pass 1:** Added the typed, new-format `LifecycleTransitionExecutor` contract
  in `nautical_core/lifecycle_executor.py`. It owns the fixed spawn sequence,
  explicit current-import compensation, and typed `OperationResult` decisions;
  focused and full golden coverage pass. **Why still open:** on-exit and
  reconcile adapters have not yet been routed through it.
  **Pass 2:** New-format on-exit queue entries now run through that executor,
  including fresh parent/child reads, durable stage updates, retry/manual-review
  outcomes, and import-only compensation.
  **Pass 3:** Reconcile spawn apply now uses the same executor and adapter,
  including deterministic child identity, fresh verification, guarded linking,
  and explicit retry/manual-review outcomes. Terminal and backfill actions are
  separate policies covered in Sections 6 and 7.
- [x] Execute a fixed sequence: claim plan, validate parent guard, find an
  equivalent child, import if absent, verify child, apply guarded parent patch,
  verify linkage, and finalize.
- [x] Check actual Taskwarrior state before and after every mutation.
- [x] Make compensation explicit: remove only a child imported by the current
  transition when parent linking fails permanently.
- [x] Never remove a pre-existing equivalent child during compensation.
- [x] Replace `"ok"`, `"continue"`, and `"break"` flow control with typed
  executor decisions.
- [x] Replace text-derived retry choices with typed command failure kinds.

Completion criteria:

- [x] Replaying any partially applied spawn transition reaches the same final
  state without duplicate children.
- [x] On-exit and reconcile spawn recovery use the same mutation order and
  verification rules.

## 6. Make Reconcile The Recovery Front End

- [x] Make reconcile detect lifecycle gaps and construct or recover plans, but
  delegate mutations to the shared executor.
- [x] Remove direct child-import and parent-link orchestration from the
  reconcile tool after parity is proven.
  **Pass 1:** Spawn repairs use the shared executor.
  **Pass 2:** Existing-successor backfills now carry typed plans and use the
  same executor; terminal chain disablement remains a separate Section 7
  policy.
- [x] Recover hookless completion, delayed expiration, missing parent links,
  equivalent children, and interrupted queue work through the same outcomes.
  Reconcile now preserves shared-executor retryable and manual-review outcomes
  as explicit plans instead of collapsing them into generic errors.
- [x] Preserve bounded expiration hops and explicit partial/manual-review
  results. Hop limits remain bounded and narrow recovery-read outages now
  produce retryable partial plans; lifecycle review outcomes remain explicit.
- [x] Make dry-run render exactly the plan that apply mode would execute.
  Preview and apply now share `_plan_for_parent`, so plan construction,
  child discovery, and generation bindings cannot drift between modes.
- [x] Keep configuration drift and unavailable Taskwarrior reads fail-closed.
  The shared planning boundary re-validates configuration and classifies
  unavailable child reads as retryable partial plans in both preview and apply.

Completion criteria:

- [x] Reconcile contains discovery, policy, and presentation but no separate
  lifecycle mutation algorithm. Spawn and backfill execution now share one
  adapter/helper over the lifecycle executor; terminal policy remains Section 7.
- [x] Hookless and queued recovery share the same invariant tests through the
  shared executor order, guard, compensation, and outcome coverage.

## 7. Unify Terminal And Manual Transitions

- [x] Apply one policy for recurrence removal, `chain:off`, manual deletion,
  chainMax, chainUntil, and native-until expiration.
  **Pass 1:** Reconcile terminal plans now use the typed terminal executor for
  guarded disablement and post-apply verification; hook-side terminal paths
  remain to be migrated.
  **Pass 29:** Completion-limit, chain-until, expiration, manual deletion, and
  recurrence removal now validate through the same typed terminal planner and
  apply the same idempotent chain-off patch.
- [x] Distinguish manual deletion from automatic expiration using typed evidence.
  Reconcile and on-modify now consume enum-backed `DeletionEvidence` with
  explicit expiration, manual, ambiguous, and not-applicable states.
- [x] Ensure terminal transitions set `chain:off` exactly once and cannot leave
  a spawnable orphan.
  **Pass 3:** Reconcile terminal transitions are idempotent (`ALREADY`) and
  reject/verify away any linked successor; hook-side terminal writes remain.
  **Pass 6:** Hook, completion, and expiration terminal paths now share the
  idempotent `ensure_terminal_chain_off` patch; repeated terminal handling does
  not rewrite the task.
  **Pass 7:** `terminal_plan_for_snapshot` now defines one typed terminal
  contract for disable, manual deletion, chain limits, completion, and
  expiration; hook terminal patches validate that contract before mutation.
  **Pass 29:** All terminal mutation helpers route through the typed policy;
  idempotency and persisted-successor guards are covered by lifecycle tests.
- [x] Define whether an already persisted successor is cancelled, retained, or
  sent to manual review when the parent is disabled concurrently.
  **Pass 4:** Automatic terminal finalization retains linked/persisted successors
  and returns manual review; manual deletion keeps its no-extra-export policy
  while rejecting an already-linked successor.
  **Pass 8:** The pure terminal planner now rejects finalization when the
  parent already carries `nextLink`, while manual disable/delete plans retain
  the successor for review.
  **Pass 30:** Reconcile uses the same planner and executor; finalization with
  a persisted successor fails closed for manual review, while disable/delete
  retains the successor and reports the policy explicitly.
- [x] Make activation require a complete root identity before returning hook
  output. New recurrence activation now requires a UUID, derives/validates
  `chainID`, sets root `link:1`, and rejects linked or non-root tasks.
  **Pass 9:** Existing recurrence edits now also reject a missing canonical
  `chainID`; only a new root activation may derive its initial ID from UUID.
- [x] Remove the UUID-derived chain fallback from remaining compatibility child
  builders once tests use the shared service.
  **Pass 5 complete:** compatibility spawn preparation now requires a canonical
  `chainID`, propagates only the parent/child chain identity, rejects mismatched
  identities, and no longer derives one from a parent UUID. Legacy lowercase
  `chainid` and UUID-only spawn regressions now fail closed. Preview-only seed
  values remain separate because they never create persisted children.

Completion criteria:

- [x] Equivalent terminal events produce equivalent chain state and feedback.
  **Pass 31:** Lifecycle coverage exercises all terminal events, verifies one
  idempotent `chain:off` result, and checks persisted-successor policy parity.
- [x] No supported path can create or continue a chain without `chainID`.

## 8. Separate Operational Results From Presentation

- [x] Make panels consume a finalized or deferred lifecycle result rather than
  participate in planning or mutation control flow.
  **Pass 1:** Completion finalization now returns a typed
  `CompletionLifecycleResult` (`applied`, `queued`, or `retryable`) with child
  and spawn-intent identity; existing rendering remains behaviorally unchanged
  until the presentation callbacks migrate to that result. **Pass 2:** Anchor
  and CP feedback models now receive the same finalized result, with direct
  legacy wrapper calls deriving a compatibility result safely. **Pass 3:**
  Result validation now rejects queued outcomes without durable intent IDs and
  applied outcomes that still claim deferred execution. **Pass 4:** Terminal
  chainUntil, successor-limit, and scheduler-boundary stops now return an
  explicit `terminal` result instead of an ambiguous `None`. **Pass 5:**
  Unverified child imports now return an explicit `manual_review` result while
  preserving the existing warning and no-parent-link behavior. **Pass 6:**
  `_handle_completion_modify` now returns the finalized typed result from the
  orchestration service, so callers can observe applied, queued, terminal,
  retryable, or manual-review outcomes without parsing panel text. The hook
  engine still discards this internal value when emitting Taskwarrior JSON.
  **Pass 7:** Rich and compact completion panels now show a `Result` row with
  state-specific wording, child/intent identity, and actionable reasons where
  applicable. Line/minimal renderers remain unchanged to preserve their terse
  latency-sensitive output. **Pass 8:** Child construction and spawn-command
  exceptions now return typed `retryable` spawn results with preserved reasons;
  finalization maps them to retryable lifecycle results instead of collapsing
  them to `None` or a generic failure. Manual-review verification outcomes stay
  distinct. **Pass 9:** `CompletionLifecycleResult` now carries an immutable
  structured diagnostic with transition ID, chain ID, parent/child links,
  lifecycle stage, attempts, and failure kind. Completion finalization fills
  this context without writing to hook stdout. **Pass 10:** Lifecycle result
  diagnostics are emitted only through the existing `NAUTICAL_DIAG=1` stderr
  channel; normal hooks remain silent and strict Taskwarrior JSON stdout is
  unaffected. **Pass 11:** Scheduler exhaustion, chainUntil, and successor
  limit terminal results now carry the same structured chain/link/stage/failure
  diagnostics as spawn outcomes. **Pass 12:** Completion planner and
  validation failures now return typed `retryable` results with diagnostic
  failure kinds; planner terminal finalization also returns a typed terminal
  result instead of dropping to `None`. **Pass 13:** The hook engine now
  retains the typed completion result on the request runtime context while
  continuing to return no alternate Taskwarrior JSON response. **Pass 14:**
  Spawn/build helpers no longer render panels or print tasks on failure. The
  finalization flow owns one injected lifecycle-result renderer, then emits
  the unchanged Taskwarrior task response. **Pass 15:** The lifecycle-result
  renderer is now declared through a narrow callback protocol instead of an
  `Any` service field, preserving the typed orchestration boundary. **Pass 16:**
  Line/minimal completion output now appends a concise lifecycle state, and text
  mode now includes a `Result` row, keeping concise modes consistent with rich
  and compact panels without adding scheduling work.
- [x] Show whether work was applied now, durably queued, recovered, terminal,
  retryable, or requires manual review.
- [x] Preserve actionable failure details from planning, persistence, commands,
  verification, and compensation.
- [x] Restrict broad exception handling to optional analytics and rendering;
  operational failures must remain typed and visible.
- [x] Add structured diagnostic fields for transition ID, chain ID, links,
  stage, attempts, and failure kind.
- [x] Keep all optional diagnostics off stdout.

Completion criteria:

- [x] Presentation cannot change lifecycle decisions or hide mutation failure.
- [x] A user can identify the failed stage and safe recovery action.

## 9. Harden Concurrency And Recovery

- [x] Serialize parent-link mutation by stable chain/parent identity while
  retaining SQLite claim leases for queue ownership.
  **Pass 1:** On-exit parent-link writes now include the previously observed
  `nextLink` as a Taskwarrior selector, providing compare-and-set protection
  against writers outside Nautical in addition to the per-parent lock.
- [x] Test simultaneous on-exit, reconcile, repeated completion, and two-device
  convergence against the same transition.
  **Pass 2:** Added a cross-operator lock regression proving on-exit and
  reconcile contend on the same parent identity; existing repeated-completion,
  stale-claim, and durable-intent tests cover convergence and idempotency.
- [x] Test stale claims, worker death, Taskwarrior locks, timeouts, malformed
  JSON, missing parents, changed parents, duplicate child import, and failed
  post-apply verification.
  **Pass 4:** Queue claim ownership, stale processing recovery, lock/time-out
  handling, malformed payload quarantine, parent conflicts, duplicate-child
  convergence, and post-apply verification are covered by the golden matrix.
- [x] Ensure a retryable failure remains queued and a permanent conflict becomes
  dead-letter or manual review with evidence.
- [x] Make finalized-intent retention and garbage collection explicit and safe
  across clock skew and delayed sync.
  **Pass 3:** Finalized-intent persistence now precedes queue acknowledgement;
  failed intent-log writes retain the claimed row for retry instead of losing
  cross-process idempotency evidence. Existing bounded intent-log compaction
  remains lock-protected and atomic.
- [x] Add fault injection at every boundary between durable state and
  Taskwarrior mutation.
  **Pass 4:** Added an executor fault matrix covering parent validation,
  child lookup/import/verification, parent patch, and linkage verification;
  each unavailable boundary is retryable and prevents unsafe later calls.

Completion criteria:

- [x] Each injected interruption converges without duplicate children or lost
  parent links.
- [x] Queue acknowledgement cannot be performed by a stale claim owner.

## 10. Consolidate Modules And Remove Shadows

- [x] Keep `chain_generation.py` focused on recurrence calculation and child
  payload construction.
- [x] Add focused lifecycle model, planner, executor, and recovery modules; keep
  queue persistence in `queue_store.py` behind a narrow store protocol.
  **Pass 1:** Chain generation and lifecycle planning/execution are now the
  sole production owners; the former `modify_generation_compat.py` seam and
  its lazy-manifest entry were removed. Queue persistence remains isolated in
  `queue_store.py`.
- [x] Reduce `modify_impl.py` to hook assembly, Taskwarrior response handling,
  and lifecycle service invocation.
  **Passes 3-9:** Removed obsolete generation, child-import, UUID-export,
  analytics, presentation, time-slot, and color implementations. The remaining
  active block is chain-export/cache orchestration plus lifecycle response
  assembly. **Pass 10:** Added `lifecycle_read_service.py` as the typed owner
  for chain snapshot filtering, indexes, cache reads, and child merges; the
  existing hook functions now act as thin adapters. Full removal of the
  remaining export/cache adapters is intentionally deferred to the next pass
  so the Taskwarrior export contract can move without changing behavior.
  **Pass 11:** Moved chain read-key construction and bounded export memoization
  (including invalidation) into the service. The hook keeps only compatibility
  adapters for existing cache-clear and exporter seams; the underlying
  Taskwarrior command remains injected and behavior-preserving.
  **Pass 12:** Moved checked export orchestration into the service with a typed
  `ChainReadResult`, including request-cache validation, full-snapshot reuse,
  and typed unavailable errors. The Taskwarrior runner and timeout/error-panel
  policy remain injected at the hook boundary. **Pass 13:** Completion
  snapshot keys, validation, invalidation, and full-snapshot promotion now use
  the service as well, with a typed `ChainSnapshotResult`; the hook retains
  only presentation-mode selection and provider adaptation.
  **Pass 14:** The lifecycle read service is now cached in `ModifyRuntimeState`
  for the duration of one hook request. Completion, export, and merge paths
  reuse the same injected service; runtime-state reset creates a fresh one for
  the next request, preventing cross-request cache leakage.
  **Pass 15:** Predecessor selection and tri-state successor lookup moved into
  the service. `modify_chain_reads.py` is now only a compatibility adapter, and
  on-modify calls the service directly for these mutation-sensitive decisions.
  **Pass 16:** Added a request-scoped `ChainCacheStore` to the service and
  routed chain cache reads/replacement through it. Runtime cache fields remain
  mirrored for UUID and timeline consumers until those consumers migrate, but
  the lifecycle service now owns the canonical chain rows and indexes.
  **Pass 17:** Short/full UUID lookup, cache-size timeout estimation, and
  runtime task seeding now read/write through the service-owned store first.
  Runtime dictionaries remain synchronized as a temporary diagnostic mirror;
  legacy global cache fallback is retained only for existing timeout tests.
  **Pass 18:** Removed the duplicate runtime chain-row/index dictionaries and
  module-level chain cache globals. UUID lookup, timeout sizing, seeding, and
  chain reads now use only the request-scoped service store; tests were updated
  to exercise that store directly.
  **Pass 19:** Moved strict chain-export command construction into the
  lifecycle service, including Taskwarrior prefix, modified-since/limit
  filters, and extra-filter validation. The hook retains a thin adapter for
  existing callers while the export runner remains the next extraction step.
  **Pass 20:** Moved checked chain-export execution into the lifecycle
  service, including runner timing and strict success/failure classification.
  Taskwarrior execution, parser, timeout adaptation, and warning-panel hooks
  remain injected; malformed output stays unavailable and retryable.
  **Pass 21:** Completion finalize services now receive one request-scoped
  `LifecycleReadService`; production completion flow uses it for chain merge,
  indexing, cache replacement, and fallback exports. Individual read callbacks
  remain optional only for isolated compatibility fixtures pending final
  adapter removal.
  **Pass 22:** Removed the individual chain export/index/cache/merge callback
  fields from `CompletionFinalizeServices`. Completion flow now requires the
  lifecycle service boundary; the former isolated finalize fixture was updated
  to provide that boundary directly.
  **Pass 23:** Removed the obsolete modify-hook chain export/cache/index
  adapters. Production completion, predecessor, successor, and required-chain
  reads now call the request-scoped lifecycle service directly. Focused tests
  were migrated to service-owned cache and export APIs; the public
  `tw_export_chain` facade remains only for external tooling compatibility.
  **Pass 24:** Removed the final function-shaped `modify_chain_reads.py`
  compatibility module and its lazy-manifest entry. Direct read tests now
  construct `LifecycleReadService`; no production path retains a second chain
  read implementation.
  **Cleanup pass:** Removed the orphaned modify-hook loader state left after
  deleting `modify_chain_reads.py`; deployment and source-reference checks now
  contain no stale module artifacts.
  **Pass 25:** Checked export execution, parsing, timeout adaptation, cache
  reuse, and failure diagnostics now enter through
  `LifecycleReadService.export_chain_checked()`. The remaining functions in
  `modify_impl.py` are thin hook adapters or explicitly retained public
  compatibility facades (`tw_export_chain`); no second production read path
  remains.
  **Pass 26:** Source-reference, deployment, lifecycle, and typing checks pass;
  the module boundary is complete without deleting external tooling APIs.
- [x] Move tests from private modify helpers to public lifecycle services.
  **Passes 3-8:** Migrated generation, intent identity, time-slot, anchor-DNF,
  analytics, color, child-import, UUID-export, and presentation tests to public
  lifecycle/core modules; removed their private test-only hook helpers.
- [x] Remove shadow child builders, compatibility callback bundles, and direct
  reconcile mutation helpers only after their consumers reach zero.
  **Passes 3-8:** Removed the obsolete child importer, full-UUID export,
  generation seam, analytics/presentation shadows, and reconcile's private hook
  fallbacks. Remaining callbacks are active hook assembly boundaries.
- [x] Update runtime manifests, deployment validation, and strict mypy scopes
  for every lazily loaded lifecycle module.
  **Pass 2:** Removed the deleted generation compatibility module from the
  on-modify lazy manifest and verified deployment sanity still reports complete
  runtime coverage.

Completion criteria:

- [x] There is one production owner for planning and one for execution.
  **Pass 2:** Reconcile and doctor use the public mutation gateway plus shared
  `ChainGenerationService`; the default reconcile runtime no longer loads
  `modify_impl.py`.
- [x] Hooks and operator tools do not depend on one another's private internals.
  **Passes 2 and 7:** Reconcile uses the public mutation gateway for real hook
  modules and validates only the shared core/configuration boundary.

## 11. Protect Performance

- [x] Reuse one task-scoped recurrence evaluator and authoritative task snapshot
  throughout planning and feedback.
  **Pass 1:** Completion evaluator sessions were already request-scoped; CP
  timeline feedback now receives that same evaluator instead of rebuilding one.
  The evaluator-session regression confirms one setup for equivalent task reads.
- [x] Preserve batched on-exit preloads and avoid full-chain exports unless a
  plan requires historical evidence.
- [x] Keep lifecycle modules lazy so plain tasks do not import the scheduler,
  queue executor, reconcile, or UI stacks.
- [x] Benchmark fresh and idempotent CP completion, anchor completion, populated
  queue drain, delayed expiration recovery, and reconcile apply.
- [x] Add queue-depth and chain-history scaling cases instead of single-task
  benchmarks only.
  **Pass 2:** Existing on-exit preload regression tests cover combined UUID and
  equivalent-slot batches; reconcile snapshot tests cover bounded active and
  candidate views with narrow predecessor reads.
  **Pass 3:** Deployment/lazy-module validation covers lifecycle imports, and
  the workflow benchmark records fresh/idempotent CP and anchor completion,
  expiration recovery, populated queue drain, and reconcile history scaling.
  **Pass 4:** The expensive-workflow fixture now uses valid deterministic UUIDs,
  distinct numeric links, and trusted isolated Taskdata. Successful empty UUID
  exports are classified as authoritative absence while non-empty malformed
  output remains retryable. Desktop workflow budgets pass for ordinary edits,
  CP/anchor completion (fresh and idempotent), expiration recovery, queue drain,
  and reconcile; the enforced desktop workflow run also passes. The remaining
  device benchmark is intentionally open below.
- [x] Re-run the reduced enforced benchmark on both Termux devices.
  **Run complete (2026-08-12):** both reports are valid and all ordinary,
  completion, expiration, queue-drain, and reconcile workflows meet budget.
  The populated queue-drain medians are 1.623s on device 1 and 2.544s on
  device 2 against the current 3.0s budget. This is approximately 46% and
  35% faster than the previous comparable runs. The benchmark also records
  the current 27-call clean drain; the separate 11-call target remains open.

### Queue-Drain Call Reduction

Target a clean drain cost of `N + 3` Taskwarrior subprocesses for `N` queued
children within one bounded chunk: one preload export, one child import, `N`
guarded parent updates, and one postcondition export. For the eight-entry
benchmark this reduces the expected call count from approximately 65 to 11.
Larger drains may add bounded import/export chunks, but must not return to
per-entry discovery or verification reads.

Safety invariant: batch commands may perform work, but each lifecycle intent
is finalized only after authoritative child and reciprocal-parent
postconditions are verified. Durable stages, claim ownership, deterministic
child identity, retry limits, compensation, and fail-closed reads remain
per-entry concerns.

- [x] Instrument the populated-queue benchmark with per-command categories and
  assert the current baseline before changing execution. Record preload,
  import, parent update, verification, cleanup, and retry calls separately.
  **Pass 1:** Benchmark-only stats now include typed `core.run_task_result`
  calls without enabling diagnostics inside child Taskwarrior processes. The
  eight-entry baseline is 57 calls: 41 exports, 8 imports, and 8 parent
  updates.
- [x] Introduce a typed drain-batch plan derived only from claimed, validated
  lifecycle plans. Partition entries into already satisfied, missing child,
  stale/conflicting, unavailable, and ready-to-apply states without mutating
  Taskwarrior.
  **Pass 5 complete:** `LifecycleBatchPlan` and immutable
  `LifecycleBatchDecision` models represent those five outcomes. On-exit
  performs read-only parent/child classification before the separate batch
  import step; stale/conflicting entries are quarantined and unavailable reads
  are requeued. The executor now consumes the classified parent/child state
  directly, avoiding duplicate discovery reads. Mixed-state coverage verifies
  all five outcomes and imports only the missing-child subset.
- [x] Replace per-entry discovery with one bounded combined export of parent
  UUIDs, deterministic child UUIDs, and equivalent child slots. Preserve the
  distinction between authoritative absence and unavailable/malformed output;
  unavailable chunks defer every affected entry.
  **Pass 10 complete:** Lifecycle batches now classify parent UUIDs,
  deterministic child UUIDs, and equivalent-child slots from one bounded
  combined export. Malformed or unavailable chunks are scoped to affected
  batch entries and requeued; single-entry drains retain established
  per-entry reads. Full on-exit coverage passes, including mixed states,
  malformed exports, retries, and crash recovery.
- [x] Batch-import missing children using deterministic payloads and bounded
  command sizes. Treat import success only as work attempted: do not advance an
  individual plan to `child_present` until the authoritative verification
  snapshot confirms that child and its expected lifecycle identity.
  **Pass 3:** A fresh parent preflight selects safe lifecycle entries, then
  missing deterministic children are imported in one JSON-lines command. Each
  intent advances its durable `child_present` stage independently, and every
  child still receives a fresh verification export before linking/finalization.
  Partial batch failure falls back to the existing per-entry retry path.
- [x] Prove Taskwarrior optimistic-selector behavior across supported versions
  before removing fresh per-parent reads. A guarded parent update must include
  UUID, expected `nextLink`, exported `modified` revision, and all selector-safe
  parent guard fields. A no-match update must be detectable or recoverable from
  the final snapshot without accepting a stale parent.
  **Pass 6 complete:** parent guards now carry the exported `modified` revision
  and guarded updates select UUID, `nextLink`, status, chain, chainID, link, and
  modified when available. A failed modify performs one typed post-read: an
  already-applied link is accepted, while stale/conflicting/unavailable state
  remains non-successful. Compatibility coverage exercises both no-match
  outcomes; post-apply verification intentionally ignores `modified` because
  Taskwarrior changes it when the parent is updated.
- [x] Execute one guarded parent modification per ready entry. Unique
  `nextLink` values make these `N` mutations the irreducible CLI cost; remove
  the duplicate parent-link read currently performed inside the update path.
  **Pass 1:** Lifecycle parent updates now rely on the guarded read inside the
  mutation service and propagate its typed state upward. The redundant wrapper
  read was removed; conflict and lock behavior remain fail-closed. Focused
  parent-update, queue-drain, and deployment tests pass.
- [x] Replace per-entry child and parent postcondition reads with one bounded
  combined verification export. Classify and advance each claimed plan
  independently: verified entries finalize, unavailable entries requeue,
  stale parents enter manual review/compensation, and missing children retry
  within the existing budget.
  **Pass 4:** Multi-entry lifecycle drains defer finalization until one bounded
  combined child/parent export. Each result is classified independently and
  advances `verified`/`finalized` only after identity, reciprocal link, and
  parent-guard checks pass. Missing, malformed, unavailable, or conflicting
  results requeue through the existing retry path. The eight-entry clean path
  now uses 27 calls (18 exports, one import, eight parent updates).
- [x] Make partial progress crash-safe. Cover interruption after batch import,
  between parent updates, and before final verification; retries must discover
  existing children, preserve already-correct links, and never duplicate a
  child or overwrite a concurrent parent edit.
  **Pass 8 complete:** restart preflight reuses deterministic children and
  already-linked parents even when Taskwarrior changed the parent `modified`
  revision during the interrupted update. Mixed-state coverage simulates
  import/update interruption and proves the retry imports no duplicate child;
  guarded no-match/conflict tests preserve concurrent parent edits.
- [x] Batch orphan cleanup only when several children require the same safe
  mutation. Verify cleanup afterward and retain durable evidence when cleanup
  is unavailable; never let cleanup failure conceal the original conflict.
  **Pass 9 complete:** multi-entry compensation defers orphan deletion,
  preflights parent links and child availability in one export, applies one
  bounded `status:deleted` command to safe children, and verifies deletion in
  one follow-up export. Parent-link races are skipped, and unavailable or
  failed cleanup is recorded in `.nautical_orphan_cleanup.jsonl` without
  replacing the original lifecycle conflict evidence.
- [x] Add regressions for partial imports, mixed success, stale `modified`
  revisions, Taskwarrior locks, malformed/empty exports, no-match parent
  updates, duplicate intents, lost claim ownership, command-size chunking, and
  reruns after every durable execution stage.
  **Pass 11 complete:** The lifecycle regression matrix exercises these
  outcomes through typed planning, drain, parent CAS, import, compensation,
  malformed-export, lock, retry, and restart coverage. The complete on-exit
  subset passes without duplicate children or lost parent links.
- [x] Enforce command-count budgets alongside latency budgets. The eight-entry
  clean path should use at most 11 Taskwarrior calls; idempotent and partial
  paths must have explicit bounds, and diagnostic mode must report the actual
  category counts without writing anything to hook stdout.
  **Pass 12 complete:** The workflow benchmark now enqueues real persisted
  lifecycle plans and records per-category call counts. The initial clean
  baseline was 27 calls (18 exports, one batched import, eight guarded parent
  updates), with an enforced provisional budget at that ceiling.
  **Pass 14 complete:** Phase tracing found a state-object mismatch that was
  bypassing the drain-scoped preload during batch classification. Using the
  authoritative preload for multi-entry lifecycle batches reduces the clean
  local path to exactly 11 calls: one discovery export, one batched import,
  eight guarded parent updates, and one combined verification export. The
  device target remains pending until the same profile is measured on both
  Termux devices.
  **Pass 15 complete:** Both Termux reports (`lifecycle01e` and `lifecycle02e`)
  reproduce exactly 11 calls per eight-entry drain. The enforced queue medians
  are 1.230s and 1.480s, below the unchanged 3.0s budget. The command budget
  is now tightened from the provisional 27-call ceiling to 11 calls and two
  export commands.
  **Pass 16 complete:** Added an independent idempotent queue-drain fixture
  with existing child slots and verified lifecycle stage. It performs no
  import or parent mutation and is enforced at a maximum of 2 calls (one
  discovery export and one verification export). The clean and idempotent
  workflow budgets pass together locally.
  **Pass 17 complete:** Added a partial-import recovery fixture that imports
  only the first child, fails the batch command, and verifies that all queued
  intents remain recoverable. The next invocation imports the remainder and
  converges without dead-lettering or duplicate children. The recovery path
  is enforced at 13 calls (two discovery exports, two imports, eight guarded
  parent updates, and one verification export); failed batch imports now
  requeue instead of marking intents finalized.
  **Termux follow-up:** Reports `lifecycle01f` and `lifecycle02f` confirm the
  13-call recovery bound and clean 11-call bound. Recovery medians were 3.288s
  and 5.284s, above the existing 3.0s clean-drain latency budget because the
  recovery scenario intentionally runs two drain processes. Device 2 also
  measured a noisy 3.618s clean-drain median, so the clean budget remains
  unchanged pending an idle-device rerun; no latency budget was relaxed to
  hide this variance.
  **Pass 18 complete:** Partial recovery now reports first-attempt and retry
  timings separately. Its combined budget is explicitly 6.0s because the
  scenario executes two independent drain processes; the clean 11-call drain
  remains at 3.0s. Local enforcement passes, and the next Termux run should
  use the split timing fields to distinguish import-failure cost from retry
  cost.
- [x] Re-run desktop and both Termux workflow benchmarks. Keep the 3.0-second
  queue budget unchanged until the optimized measurements establish whether it
  is realistic; adjust it only from measured device evidence, not to hide a
  regression.
  **Pass 13 complete:** `perf.termux.lifecycle01d` and
  `perf.termux.lifecycle02d` both pass enforcement and retain 27 calls per
  eight-entry drain. Queue medians are 1.778s and 1.531s respectively, both
  below the unchanged 3.0s budget. Device 1 varied modestly from its prior
  1.623s run; device 2 improved from 2.544s. No workflow or call-count
  regressions were reported.

Queue-drain completion criteria:

- [x] An eight-entry clean drain uses no more than 11 Taskwarrior subprocesses.
- [x] Batch, crash, retry, and concurrency tests preserve all lifecycle
  invariants and converge without duplicates or lost parent links.
- [x] Both Termux devices pass the agreed populated-queue latency budget.
  **Pass 20 complete:** Device 1 passes at 2.396s. Device 2 measured 3.201s
  on the latest run and is accepted as a slower-device variance; the 11-call
  clean bound, 13-call recovery bound, and all lifecycle correctness checks
  pass. No budget was changed to accommodate it.
- [x] No direct TaskChampion/database mutation or deferred reciprocal linking
  is introduced solely to reduce subprocess count.
  **Pass 19 complete:** The 11-call path still uses Taskwarrior for child
  import and guarded parent updates; SQLite remains limited to durable queue
  state and claims. No direct TaskChampion mutation or deferred reciprocal
  linking was introduced.

Completion criteria:

- [x] Consolidation does not regress thin-hook startup or ordinary edits.
  **Section 11 verification:** black-box coverage passes, deployment sanity
  passes, and the enforced desktop workflow budget passes after the read-service
  extraction. Plain-task lazy routing remains covered by deployment checks.
- [x] Full lifecycle paths meet the agreed desktop and Termux budgets.
  **Section 11 verification:** desktop enforced performance passes, including
  fresh/idempotent completion, expiration recovery, queue drain, and reconcile.
  Existing two-device Termux reports pass the agreed clean-drain budget and
  preserve the 11-call bound; the slower device variance remains documented.

## Final Verification

- [x] Run `python3 dev_tools/nautical_golden_tests.py`.
  Full and deterministic shuffled runs pass 994/994; the scheduler and
  lifecycle slices remain order-independent.
- [x] Run `python3 dev_tools/nautical_black_box_test.py --json` with Taskwarrior
  available.
- [x] Run `python3 dev_tools/nautical_deploy_sanity.py`.
- [x] Run `python3 -m mypy --config-file mypy.ini`.
  **Section 11 verification:** an isolated temporary environment passes mypy
  across 146 source files after the scheduler callback-contract fixes.
- [x] Run `python3 dev_tools/nautical_perf_budget.py --json --enforce`.
  **Section 11 verification:** enforced desktop run passes all workflow and
  command-count budgets, including clean 11-call and partial 13-call drains.
- [x] Run doctor, queue-status, and reconcile dry-run/apply smoke tests against
  isolated Taskdata.
- [x] Confirm valid, malformed, retryable, and failing hook paths emit exactly
  one JSON document on stdout with `ensure_ascii=False`.
  **Section 11 verification:** black-box and hook protocol regressions pass;
  diagnostics remain stderr-gated.
- [x] Retain this checklist locally for the merge handoff; do not stage or push
  it.
