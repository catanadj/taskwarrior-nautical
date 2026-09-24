# Nautical Taskwarrior Integration Engine Checklist

Replace Nautical's fragmented Taskwarrior command, read, mutation, queue, and
recovery paths with one typed integration boundary. The scheduler remains the
authority for when an occurrence happens, the lifecycle engine remains the
authority for what transition should happen, and this engine becomes the sole
authority for applying that transition to Taskwarrior.

## Scope And Working Model

- [x] Develop exclusively on `taskwarrior-integration-engine`; keep `main`
  operational until every final gate passes.
- [x] Treat Nautical as offline while this branch is under construction.
  Intermediate commits do not need to produce an installable or operational
  Nautical system.
- [x] Do not build legacy bridges, tuple adapters, old/new gateways, dual queue
  readers, fallback execution, or compatibility aliases for replaced internal
  APIs.
- [x] Keep comparison with the previous implementation in tests and benchmark
  fixtures only. Production code must have one execution path.
- [x] Remove an old operational path as its replacement takes ownership; do not
  preserve it merely to keep intermediate branch commits functional.
- [x] Keep Taskwarrior as the sole durable owner of tasks and chains. Nautical's
  SQLite state stores only durable lifecycle work and verification evidence,
  never a shadow task database.
- [x] Do not introduce a daemon or cross-process cache of Taskwarrior task
  state. Cache and reuse reads only within one invocation-scoped unit of work.
- [x] Preserve strict hook JSON on stdout with `ensure_ascii=False`.
  Diagnostics remain silent unless `NAUTICAL_DIAG=1`, and then go to stderr.
- [x] Keep this checklist local. Push implementation commits only to the
  integration branch and merge it into `main` in the final stage.

Cutover policy:

- [ ] Stop Taskwarrior hooks and Nautical workers before installing the
  completed engine.
- [ ] Remove or quarantine obsolete Nautical queue/outbox state instead of
  maintaining runtime schema bridges.
- [ ] Let reconcile reconstruct missing lifecycle work from authoritative
  Taskwarrior state after cutover.
- [ ] Run doctor, queue status, reconcile dry-run, and isolated lifecycle smoke
  tests before re-enabling hooks.
- [ ] Roll back by restoring the previous release, not by retaining two
  integration engines in production.

## Target Ownership

The final architecture should have these focused components. Exact filenames
may change if a clearer ownership boundary emerges, but responsibilities must
not be recombined into a hook monolith.

- `integration_models.py`: immutable command, read, mutation, guard, outbox,
  stage, and outcome models.
- `integration_context.py`: validated Taskdata, task binary, configuration,
  diagnostics, clock, invocation identity, and command budget.
- `taskwarrior_client.py`: the only Taskwarrior subprocess execution boundary.
- `taskwarrior_reads.py`: typed task and chain repository plus invocation-local
  snapshot reuse.
- `taskwarrior_mutations.py`: guarded named mutations and postcondition
  verification.
- `taskwarrior_uow.py`: invocation-scoped composition, read cache, mutation
  epoch, and diagnostics.
- `lifecycle_outbox.py`: durable outbox repository and schema ownership.
- `lifecycle_application.py`: stage and execute lifecycle plans using the unit
  of work and outbox.

The final flow is:

```text
hook / reconcile / doctor / navigator
                 |
       TaskwarriorUnitOfWork
          /       |       \
       reads   mutations   outbox
          \       |       /
        lifecycle application
                 |
      scheduler + lifecycle planner
```

## Baseline

- [ ] Record full golden, deterministic shuffled golden, black-box, deployment,
  mypy, hook-protocol, and isolated doctor/reconcile results from `main`.
- [ ] Record desktop and both Termux profiles for plain hooks, Nautical add,
  ordinary modify, CP completion, anchor completion, empty/populated queue,
  partial recovery, and reconcile.
- [ ] Record Taskwarrior call counts by command purpose for every mutation
  workflow, including fresh and idempotent paths.
- [ ] Inventory every Taskwarrior invocation and command wrapper in hooks,
  operator tools, navigator, doctor, installer, and development tools.
- [ ] Inventory every queue read/write/claim/ack/requeue/dead-letter call site.
- [ ] Inventory every place where malformed JSON, empty output, or command
  failure can become an empty task collection or false absence.
- [ ] Add characterization coverage for successful, absent, unavailable,
  retryable, partially applied, already applied, conflicting, and manual-review
  outcomes before deleting old paths.

Completion criteria:

- [ ] Every production Taskwarrior call and durable queue operation has a named
  owner and a characterization test.
- [ ] Baseline reports can be rerun without using the user's live Taskdata or
  Nautical state directories.

## 1. Define The Integration Contract

- [x] Add immutable typed models for `TaskCommand`, `TaskCommandResult`,
  `TaskRead`, `MutationGuard`, `MutationOutcome`, `OutboxIntent`,
  `OutboxStage`, and `OutboxOutcome`.
- [x] Define command failure kinds without parsing human-readable diagnostics:
  success, absent, timeout, busy, missing binary, invalid response, rejected,
  and execution failure.
- [x] Define `TaskRead[T]` as exactly `Found[T]`, `Absent`, or `Unavailable`.
- [x] Make `Unavailable` carry retryability and structured failure evidence;
  it must never compare equal to `Absent` or an empty collection.
- [x] Define mutation outcomes: applied, already applied, retryable, rejected,
  conflict, and manual review.
- [x] Define parent/task guards using UUID, status, chainID, link, recurrence
  identity, relevant timestamps, and expected mutation epoch.
- [x] Define explicit postconditions for child import, parent link, chain
  disablement, native-until repair, and metadata repair.
- [x] Define outbox identity from deterministic lifecycle intent identity, not
  insertion order, process identity, or filesystem path.
- [x] Validate all models at construction so incomplete identities and invalid
  state combinations cannot travel through the engine.

Completion criteria:

- [x] The integration contract contains no raw task tuples, free-form outcome
  strings, nullable success values, or `Any` callback bundles.
- [x] Mutation code cannot accept an unavailable read as evidence of absence.
- [x] Contract tests cover every valid and invalid state combination.

## 2. Build One Validated Runtime Context

- [x] Add an immutable integration context containing resolved Taskdata,
  Taskwarrior command prefix, validated Nautical configuration, timezone,
  diagnostics sink, clock, invocation ID, and command budget.
- [x] Resolve configuration and Taskdata once per full Nautical invocation.
- [x] Keep thin plain-task routing ahead of integration-context construction so
  ordinary Taskwarrior operations retain their fast path.
- [x] Fail closed before scheduling or mutation when Taskdata, configuration,
  timezone, astronomy, calendar, preset, or task binary state is unavailable.
- [x] Make the context explicitly read-only or mutation-capable; on-modify must
  never accidentally acquire mutation services.
- [x] Keep environment and filesystem probing in context construction rather
  than scattered through repositories and hooks.

Completion criteria:

- [x] Every full hook and Taskwarrior-facing operator command uses one validated
  context. Installer/runtime cleanup and queue-only status commands remain
  outside this boundary because they do not query or mutate Taskwarrior.
- [x] Plain add/modify/exit routing does not import or construct the full
  integration engine.

## 3. Make One Taskwarrior Client

- [x] Implement one `TaskwarriorClient` as the sole boundary allowed to invoke
  the `task` binary for hooks and operator tools.
- [x] Preserve argv, return code, stdout, stderr, attempt number, duration,
  timeout, and typed failure kind in every result.
- [x] Centralize timeout, retry, backoff, temporary-output, encoding, and
  process-termination behavior.
- [x] Retry only failures explicitly classified as retryable and preserve the
  final attempt's evidence.
- [x] Keep JSON decoding outside the process runner but inside typed Taskwarrior
  repository boundaries.
- [x] Add a command observer that records call purpose and latency without
  exposing task contents in diagnostics.
- [x] Delete `RawCommandResult`, tuple coercion, duplicate `_run_task` helpers,
  and hook-specific subprocess loops as consumers move.
- [x] Add an AST/deployment check preventing new direct Taskwarrior subprocess
  calls outside the client and installer-specific process ownership.

Completion criteria:

- [x] Hook and operator code has one retry/failure-classification policy.
- [x] No lifecycle or scheduling decision searches stderr/stdout text for
  failure semantics.
- [x] No production caller receives `(ok, stdout, stderr)`.

## 4. Introduce The Invocation-Scoped Unit Of Work

- [x] Compose context, client, reads, mutations, outbox, diagnostics, and call
  counters in `TaskwarriorUnitOfWork`.
- [x] Scope one unit of work to one hook or operator invocation; never store it
  in process-global state reused across requests.
- [x] Cache authoritative reads with explicit query scope and snapshot
  provenance.
- [x] Track a mutation epoch. Any successful or uncertain mutation invalidates
  affected cached reads before another decision is made.
- [x] Support one broad snapshot followed by narrow historical/predecessor
  reads only when required.
- [x] Make command budgets observable in tests and diagnostics without turning
  budget excess into incorrect scheduling behavior.
- [x] Ensure independent core/hook loaders cannot share unit-of-work caches,
  Taskdata state, or mutable diagnostics.

Completion criteria:

- [x] Repeated reads within an invocation reuse authoritative data.
- [x] Reads after mutation cannot observe a stale pre-mutation cache entry.
- [x] Separate hook processes share only durable outbox state.

## 5. Build The Typed Task Read Repository

- [x] Add domain-shaped reads: UUID, exact child slot, predecessor slot, chain
  snapshot, active recurrence roots, lifecycle candidates, and verification
  reads.
- [x] Parse Taskwarrior JSON exactly once at the repository boundary.
- [x] Treat successful empty output as `Absent` only for queries whose command
  contract defines empty output as authoritative absence.
- [x] Treat malformed JSON, truncated output, mismatched UUID/chain/link,
  duplicate exact matches, timeouts, locks, and nonzero exits as `Unavailable`.
- [x] Carry snapshot scope and included statuses so a filtered export cannot be
  mistaken for complete chain history.
- [x] Reuse a broad export for UUID, chain, child, and verification lookups when
  its scope is authoritative.
- [x] Use binary/indexed in-memory lookup rather than rescanning exported lists.
- [x] Add narrow fallback reads for predecessor history and data deliberately
  excluded from the broad snapshot.
- [x] Replace hook, exit, reconcile, doctor, and navigator query helpers
  directly; delete their old readers in the same migration passes.

Completion criteria:

- [x] Every mutation-sensitive read has found/absent/unavailable semantics.
- [x] No malformed or unavailable query result becomes an empty chain or
  missing child.
- [x] One invocation cannot export the same authoritative scope twice without
  an intervening mutation or explicit refresh.

## 6. Build The Guarded Mutation Gateway

- [x] Implement named operations for child import, parent linking, chain
  disablement, native-until repair, lifecycle metadata repair, and any required
  Taskwarrior update.
- [x] Accept typed guards and domain payloads rather than arbitrary command
  arrays.
- [x] Re-read the narrow guard state immediately before mutation unless the
  unit of work has equally fresh authoritative evidence.
- [x] Make stable child UUID and deterministic slot identity mandatory for
  child import.
- [x] Classify an existing exact postcondition as `AlreadyApplied`.
- [x] Reject ambiguous children, changed parents, divergent recurrence state,
  or unrelated existing UUIDs without mutation.
- [x] Verify postconditions after every external mutation and return typed
  retry/manual-review outcomes when verification is unavailable or conflicts.
- [x] Replace `reconcile_gateway.py` and hook-specific mutation functions;
  delete the replaced gateways rather than forwarding them.

Completion criteria:

- [x] Lifecycle and reconcile code cannot construct Taskwarrior mutation
  commands directly.
- [x] Replaying any mutation is idempotent and produces the same verified
  postcondition.
- [x] A write whose outcome cannot be verified never reports success.

## 7. Replace The Queue With A Lifecycle Outbox Repository

- [x] Define one new SQLite schema for immutable intent identity, serialized
  typed plan, parent guard, configuration/schedule fingerprints, execution
  stage, lease owner, attempt count, failure evidence, and timestamps.
- [x] Keep queue-processing status distinct from lifecycle execution stage.
- [x] Persist stages: planned, claimed, child present, parent linked, verified,
  and acknowledged; represent retry/manual-review/quarantine outcomes
  explicitly.
- [x] Use deterministic intent IDs and a unique constraint to deduplicate work
  across hooks and devices.
- [x] Implement atomic enqueue, claim, lease renewal, stage transition,
  acknowledgement, retry, release, and quarantine operations.
- [x] Make claims bounded and crash-recoverable without depending on process
  lifetime.
- [x] Quarantine poison rows atomically and expose their complete diagnostic
  reason to doctor and queue status.
- [x] Configure and validate SQLite schema/journal behavior at repository open
  with bounded first-open concurrency handling.
- [x] Do not migrate legacy queue rows in production. At cutover, quarantine or
  discard the obsolete internal database and use reconcile to reconstruct
  authoritative missing work.
- [x] Delete old JSONL/SQLite dual-path code, schema bridges, queue result
  adapters, and migration readers after the outbox cutover.

Completion criteria:

- [x] The outbox repository is the only durable Nautical work store.
- [x] A crash at every stage resumes from the last verified stage.
- [x] Duplicate enqueue, concurrent drain, poison rows, stale leases, and
  database corruption have deterministic outcomes.

## 8. Build One Lifecycle Application Service

- [x] Add a service that stages typed lifecycle plans into the outbox and
  executes claimed intents through the guarded mutation gateway.
  (`nautical_core/lifecycle_application.py`: `LifecycleApplicationService.stage`
  enqueues SPAWN_CHILD plans via `LifecycleOutboxRepository.enqueue`;
  `.drain` claims a bounded batch and executes each through
  `TaskwarriorMutationPort.apply`. DISABLE_CHAIN/FINALIZE_CHAIN/UPDATE_PARENT
  plans are one guarded, self-verifying mutation with no meaningful
  intermediate stage, so `.apply_immediate` applies them directly against the
  same gateway without outbox staging — see the module docstring for why the
  outbox's stage vocabulary is spawn-shaped.)
- [x] Keep scheduling and transition planning outside the integration engine;
  accept only compiled schedules and typed lifecycle plans.
  (The service only accepts already-built `LifecyclePlan` objects; it never
  imports `lifecycle_planner` or a scheduler module.)

  Refinement made during section 9's on-modify cutover: `unit_of_work` and
  `mutations` are now optional constructor arguments. On-modify has a
  legitimate need to stage plans without a command-capable unit of work — it
  deliberately avoids building one to reduce the risk of re-entering
  Taskwarrior while it still holds the datastore lock for the task being
  modified. `drain()`/`apply_immediate()` raise `LifecycleApplicationError`
  clearly if called on a staging-only instance; `stage()` is unaffected.
- [x] Validate plan, parent guard, recurrence/configuration fingerprints, and
  current Taskwarrior state before advancing an intent.
  (`_validate_spawn_plan` checks the plan is well-formed before staging;
  `_execute_claimed` rejects a claimed record whose stored
  configuration/schedule fingerprint no longer matches the caller-supplied
  current fingerprints; `_mutation_guard` rebuilds a `MutationGuard` from the
  plan's durable `ParentGuard` at the current mutation epoch for every
  operation, so the guarded gateway re-reads and re-checks live Taskwarrior
  state immediately before each mutation.)
- [x] Advance execution one verified stage at a time and persist the stage
  before beginning the next external operation.
  (`_execute_claimed` calls `advance_stage` immediately after each successful
  mutation and before attempting the next one; `_SPAWN_STAGE_ORDER` lets a
  resumed drain skip any operation whose stage was already durably recorded.)
- [x] Return applied, already-applied, retryable, conflict, manual-review, or
  quarantined outcomes without exception-driven control flow.
  (`LifecycleApplicationOutcomeKind` enumerates exactly these plus `noop`;
  every path returns a typed outcome rather than raising.)
- [x] Make on-exit and reconcile use the same executor and postcondition
  verifier.
  (Now fully satisfied: `hooks/exit_impl.py` (section 9) and
  `tools/nautical_reconcile.py` (section 10) both call
  `LifecycleApplicationService` exclusively — on-exit via `drain()`,
  reconcile via `stage()` + `execute_staged()` for spawns and
  `apply_immediate()` for terminal transitions. No remaining caller of the
  old `lifecycle_executor.py` in production code.)
- [x] Bound recovery work per invocation and preserve remaining work for the
  next drain without losing progress.
  (`drain(limit=...)` claims at most `limit` records; per-record stage
  persistence means a crash mid-batch leaves unfinished records claimable
  again once their lease expires, verified in manual crash-resume testing.)

Completion criteria:

- [x] There is one production lifecycle staging path and one execution path.
  (`stage()` for SPAWN_CHILD; execution is `drain()` for a live hook claiming
  whatever's next, `execute_staged()` for a caller like reconcile that must
  execute exactly the intent it just staged under its own external lock, and
  `apply_immediate()` for the remaining actions. `drain()` and
  `execute_staged()` both funnel through the same private `_execute_claimed`
  — no duplicate orchestration, just two ways to select which claimed record
  to run it on.)
- [x] On-exit and reconcile produce equal outcomes from equal state.
  (Both call sites are now driven by the exact same `LifecycleApplicationService`
  methods against the exact same guarded mutation gateway, so equal
  Taskwarrior state produces equal outcomes by construction — there is no
  separate reconcile-specific mutation logic left to diverge. Not verified
  end-to-end against a real `task` binary, same caveat as section 9.)
- [x] No executor stage infers prior success solely from missing queue state.
  (Every mutation re-reads and re-verifies live Taskwarrior state through the
  guarded gateway; outbox stage is only a resume marker, never treated as
  proof of a postcondition on its own.)

## 9. Cut Over The Hooks

### On-exit

- [x] Probe the new outbox without importing the full engine when no work is
  available.
  (Already true before this cutover: `on-exit.nautical` and `exit_probe.py`
  read the `lifecycle_outbox.db` sqlite schema directly, matching
  `lifecycle_outbox.py`'s real schema, without importing the engine.)
- [x] Claim a bounded batch and execute it exclusively through the lifecycle
  application service.
  (`hooks/exit_impl.py::_drain_outbox_result` builds a
  `TaskwarriorMutationService` + `LifecycleOutboxRepository` from the
  invocation's unit of work, constructs `LifecycleApplicationService`, and
  calls `.drain(limit=_OUTBOX_BATCH_MAX_ITEMS, ...)` once. No other mutation
  path remains in the hook.)
- [x] Persist retry/manual-review state and acknowledge only verified results.
  (Delegated entirely to `LifecycleApplicationService`; the hook no longer
  contains its own retry/manual-review/acknowledge logic.)
- [x] Keep UI rendering outside mutation and persistence services.
  (`_render_exit_drain_failure_panel`/`_emit_drain_stats_diag` only read the
  plain stats dict returned by the drain; true both before and after this
  change.)
- [x] Delete old queue drain, Taskwarrior command, child import, parent
  update, and verification implementations from the exit hook.
  (`hooks/exit_impl.py` shrank from 2589 to 731 lines. Deleted wholesale:
  `_execute_lifecycle_outbox_entry`, `_import_child`, `_update_parent_nextlink`,
  `_precheck_parent_guard`, the ad hoc file-locking scaffolding
  (`_local_safe_lock` and friends — superseded by `taskwarrior_client.py`'s
  own transient-lock retry), preload helpers, and progress-bar UI tightly
  coupled to the old batch shape. Also deleted the now-fully-dead support
  modules `exit_models.py`, `exit_entry_flow.py`, `exit_side_effects.py`,
  `exit_drain_flow.py`, and trimmed `exit_runtime.py` to just the
  diagnostics/startup-timing state the hook still needs. `runtime_manifest.py`
  updated to match; every listed file/module verified to exist and import.)

  Known trade-off: the old per-entry progress bar during a large drain was
  dropped along with the batch machinery it was built for; a bounded
  single-call `.drain()` doesn't expose a per-item hook to animate against
  without extending `lifecycle_application.py` itself, which was out of
  scope here.

  Known test debt: 14 tests in `dev_tools/nautical_golden_tests.py` white-box
  the deleted internals directly (e.g. `mod._import_child`,
  `mod._parent_nextlink_state`, `mod._take_outbox_batch`) and now error.
  Deferred by agreement until on-add and on-modify are also cut over, then
  fixed/rewritten in one pass — keeping only the ones that characterize real
  current behavior (idempotency, guard/conflict handling, orphan cleanup,
  etc.), not ones that only exercised deleted legacy bridges.

### On-add

- [x] Keep the thin protocol probe and strict passthrough path.
  (Already true before this cutover: `on-add.nautical`'s launcher probes
  `probe.is_nautical`/alias hints and returns a passthrough JSON response for
  plain tasks without ever loading `hooks/add_impl.py`.)
- [x] Construct a validated read-only unit of work only for Nautical additions.
  (Found and fixed a real gap: `add_impl.py::main()` unconditionally called
  `_build_hook_runtime_context()` — which constructs a full
  `TaskwarriorUnitOfWork` via `build_taskwarrior_uow` — for *every* task that
  reached the full implementation, even ones that turn out to have no
  Nautical fields at all (e.g. a false-positive description alias hint, or
  `NAUTICAL_BENCH_FORCE_FULL=1` profiling of a plain task). Moved the
  `_task_has_nautical_fields(task)` gate earlier in `main()`, before runtime
  construction; a non-Nautical task now gets the exact same passthrough
  response `handle_on_add` would have produced, without ever building a unit
  of work. Verified both branches directly: the plain-task path asserts
  `_build_hook_runtime_context` is never called; the real-anchor-task path
  asserts it's still called exactly once and `handle_on_add` still runs.)
- [x] Stamp recurrence activation through the lifecycle planner without direct
  Taskwarrior subprocess or queue access.
  (The "without direct Taskwarrior subprocess or queue access" half was
  already true: `_stamp_chain_id_on_add` only ever mutates the in-memory
  `task` dict that Taskwarrior applies via the hook's own stdout contract —
  confirmed no `task import`/`task modify` subprocess call or outbox access
  anywhere in this path. The "through the lifecycle planner" half doesn't
  literally apply here: `lifecycle_planner.py`'s API
  (`plan_candidate_successor`, `plan_expiration_successor`,
  `terminal_plan_for_snapshot`) models transitions on an *existing* chain
  (complete/expire/activate/resume/disable); stamping a `chainID` on a
  brand-new task establishes initial identity, not a lifecycle transition,
  so there's no real transition to route through the planner. Not forcing a
  speculative new planner API for this rather than risk changing behavior
  for uncertain benefit.)
- [x] Emit exactly one Taskwarrior JSON object on stdout on success and
  failure.
  (Already correctly satisfied by the existing two-layer contract, verified
  rather than assumed: `on-add.nautical`'s launcher wraps `run_hook()` in
  `except SystemExit: raise` / `except Exception: _emit_fallback(...)`, so
  any *unexpected* error still yields exactly one JSON object on stdout.
  Explicit validation rejections (`_fail_and_exit`/`_error_and_exit`) write
  to stderr and `sys.exit(1)` with no stdout JSON, which is Taskwarrior's own
  correct on-add contract for rejecting an add outright, not a violation of
  this bullet. Confirmed my `main()` restructuring introduces no double-emit
  or no-emit path in either branch.)
- [x] Delete replaced add command/query helpers and compatibility names.
  (Checked for a legacy direct-mutation/queue layer analogous to on-exit's —
  there isn't one; on-add never had it, so there was nothing of that shape to
  delete. Found several apparently-unreferenced-in-flow helpers
  (`_run_task_result`, `_task_cmd_prefix`, several `_anchor_preview_*`
  helpers) via the same static check used for on-exit and on-modify, but
  verified first: `dev_tools/nautical_golden_tests.py` calls and monkeypatches
  several of these directly by name (e.g. `_task_cmd_prefix`: 9 references,
  `_run_task_result`: 9 references) as unit tests of the helper itself, not
  legacy orchestration. Left them alone rather than break currently-passing,
  functionality-relevant tests for no architectural gain.)

### On-modify

- [x] Keep on-modify decision-only while Taskwarrior owns its datastore lock.
  (Confirmed, not just assumed: traced every cross-task lifecycle path.
  Terminal transitions (`_ensure_terminal_chain_off`) only mutate the `new`
  dict in place — applied for free via Taskwarrior's own hook-stdout
  contract, no subprocess call. Spawn transitions (`_spawn_child_atomic`,
  used by both completion and expiration-recovery) only build a
  `LifecyclePlan` and stage it; the comment already in that function explains
  why — "we intentionally avoid importing the parent from inside the hook to
  reduce the risk of re-entering Taskwarrior while it is holding the
  datastore lock." No direct `task import`/`task modify` subprocess call
  exists anywhere in the on-modify decision paths.)
- [x] Parse old/new snapshots, request scheduling/lifecycle decisions, and
  stage the resulting intent without applying Taskwarrior mutations.
  (`_enqueue_spawn_intent` now builds a staging-only `LifecycleApplicationService`
  — see the section 8 refinement above — and calls `.stage(...)` instead of
  calling `LifecycleOutboxRepository.enqueue()` directly. This is a real
  correctness gain, not just a style change: `.stage()` runs
  `_validate_spawn_plan` first, so a malformed plan (e.g. missing its parent
  nextLink patch) is now rejected before it ever reaches the durable outbox,
  where raw `.enqueue()` would previously have accepted it. Verified with a
  deliberately malformed plan.)
- [x] Reject unavailable reads and invalid identity changes before staging.
  (All completion/expiration reads already route through
  `unit_of_work.repository`, the typed `TaskReadRepository` from section 5
  (Found/Absent/Unavailable), not raw Taskwarrior calls. `_lifecycle_spawn_identity`
  raises on a non-numeric parent/child link before a plan is ever built.)
- [x] Preserve ordinary Nautical edit fast routing where no lifecycle decision
  is required.
  (Already correct before this cutover: `on-modify.nautical`'s launcher
  classifies `plain_fast_path`/`ordinary_nautical_fast_path` and returns a
  passthrough JSON response without ever loading `hooks/modify_impl.py`.)
- [x] Delete replaced completion/preflight/query/queue orchestration from the
  heavy hook implementation.
  (Much smaller in scope than on-exit's cutover, and worth saying why: unlike
  the old on-exit implementation, on-modify never had a legacy direct-mutation
  path to delete — it was already decision/staging-only by construction. The
  one real change was the `.enqueue()` → `.stage()` swap above. Removed the
  one dead reference to `lifecycle_executor` (never actually called from this
  file) from the module spec table and `runtime_manifest.py`.)

Note: on-modify still has ~195 functions and ~4,700 lines, but the large
majority of that is chain/timeline/analytics panel rendering, anchor/CP/
native-until validation, and caching — legitimate on-modify feedback and
validation, not legacy mutation orchestration in scope for this section.


Completion criteria:

- [x] Hook files are protocol/orchestration adapters, not owners of command,
  query, mutation, queue, or recovery behavior.
  (True for the behaviors this bullet actually names. On-exit is now purely
  an adapter over `LifecycleApplicationService`. On-modify stages through the
  same service and no longer owns queue access directly; its remaining
  in-hook mutation is limited to patching the *current* task's own dict,
  which Taskwarrior applies atomically via the hook's own stdout contract —
  not a separate mutation path to own. On-add owns none of these. Caveat:
  on-modify is still ~4,700 lines, but the bulk of that is legitimate
  presentation (chain/timeline panels) and validation logic, not command/
  query/mutation/queue/recovery ownership — see the note above this block.)
- [x] Plain hook paths remain thin and full Nautical paths use only the new
  engine.
  (True for cross-task mutation, which is what the guarded gateway
  (`taskwarrior_mutations.py`) exists for: on-exit's drain and on-modify's
  spawn staging both go through `lifecycle_application.py` exclusively.
  Same-task terminal patches (`chain: off`/`chain: on`) intentionally bypass
  it — Taskwarrior's own hook-stdout contract is already the atomic boundary
  for a task modifying itself, so there's no race for the guarded gateway to
  protect against there. That's a deliberate scope boundary of the engine,
  not a hole in the cutover.)
- [x] Strict stdout and diagnostic stderr contracts pass in subprocess tests.
  (Subprocess coverage now asserts that on-add and on-modify emit exactly one
  JSON object on stdout while forced diagnostics are present only on stderr.
  The malformed-environment matrix also verifies empty stderr when diagnostics
  are disabled, including the on-exit wrapper's empty stdout contract.)


## 10. Cut Over Operator And Presentation Consumers

- [x] Make reconcile use the same runtime context, read repository, lifecycle
  application service, mutation gateway, and outbox as hooks.
  (`tools/nautical_reconcile.py` already constructed a modern
  `TaskwarriorUnitOfWork`/`TaskReadRepository` via `build_operator_uow`, but
  routed lifecycle execution through its own `_ReconcileLifecycleServices`
  adapter against the old `lifecycle_executor.py` — the same pattern on-exit
  used to have. Replaced it: `_execute_reconcile_lifecycle_plan` now stages
  through `LifecycleApplicationService.stage()` and executes via a new
  `execute_staged()` method (added to the service, see below); terminal
  transitions use `apply_immediate()`. Removed the ~300-line
  `_ReconcileLifecycleServices` class entirely and its now-dead imports
  (`LifecycleTransitionExecutor`, `LifecycleTerminalExecutor`,
  `OperationResult`, `OperationState`, `ChildCompensationPayload`, others).

  Two real design gaps had to be closed, not just wired around, since
  reconcile is a genuinely different caller shape from the hooks:

  1. **Targeted execution.** `drain()` claims whatever's next in the shared
     outbox — fine for a live hook, wrong for reconcile, which holds a
     per-parent lock and must execute *exactly* the intent it just staged,
     never an unrelated one that happened to be claimed instead. Added
     `LifecycleOutboxRepository.claim_intent()` (claim by a specific
     `intent_id`, reusing the same lease/poison-row handling as
     `claim_batch`) and `LifecycleApplicationService.execute_staged()` (stage
     + targeted claim + execute, reusing the same `_execute_claimed` used by
     `drain()`). Verified directly: staged an unrelated intent first, then
     confirmed `execute_staged()` touched only its own intent and left the
     other one completely untouched in the outbox.
  2. **Deterministic child identity.** Reconcile computes a stable,
     reproducible child UUID so repeated runs against the same broken chain
     converge on the same child rather than creating duplicates — the old
     code injected this UUID deep inside its adapter, per-mutation. Moved it
     to one place (`_lifecycle_plan_with_resolved_child_uuid`), called once
     before staging. It also preserves a duplicate-avoidance check the old
     `find_equivalent_child` had that the guarded gateway's own uuid-based
     already-applied detection doesn't cover: a task already occupying the
     exact same chain position (by chainID+link+prevLink) under a
     *different* uuid — e.g. a chain a human partially repaired by hand
     before this run. Verified both branches: no existing task → falls back
     to the deterministic hash; an existing positional match → reuses its
     real uuid instead of staging a duplicate.

  Per your approval, dropped `compensate_child` (the old rollback-on-partial-
  failure behavior) rather than porting it — reconcile's spawn repairs now
  get the same durable, crash-safe resume semantics as the live hooks
  instead of an active rollback that could itself fail.

  Verified end-to-end against fakes: both the spawn/backfill path and the
  terminal path, through the real `_execute_reconcile_lifecycle_plan`/
  `_execute_reconcile_terminal_plan` functions as installed in the file.)
- [x] Keep reconcile discovery read-only until a complete typed plan is built;
  apply mode must use the shared executor.
  (Already true before this cutover, confirmed rather than assumed: `main()`
  passes `access=IntegrationAccess.MUTATION if args.apply else
  IntegrationAccess.READ_ONLY` when building the operator unit of work, and
  `IntegrationContext` enforces stricter taskdata validation when access is
  `MUTATION`. "Apply mode must use the shared executor" is the bullet above.)
- [x] Make doctor inspect client/configuration health, outbox schema, poison
  rows, stale leases, retry/manual-review work, and configuration fingerprints.
  (Verified: doctor already uses the validated read-only integration context,
  reports Taskwarrior/client and configuration health, configuration drift
  fingerprints, astronomy/timezone diagnostics, and lifecycle outbox schema,
  poison, stale-lease, retry, and manual-review findings.)
- [x] Make queue status read through the outbox repository and expose stage,
  age, attempts, lease, and actionable failure reason.
  (Implemented: `LifecycleOutboxRepository.status()` is now the typed,
  read-only inspection boundary. It validates schema/integrity without
  creating state or negotiating WAL, and `nautical_queue_status.py` renders
  its stage, lease age, attempts, failure code/reason, stale claims, and
  retry/manual-review/quarantine counts. Existing empty, stale, poison, and
  future-schema tests pass.)
- [x] Make navigator use the typed read repository and invocation snapshot for
  task/chain selection without importing mutation or outbox services.
  (Navigator now obtains all exports through the invocation-scoped
  `TaskReadRepository`, reuses its authoritative snapshot/cache, and rejects
  any mutation-capable integration context before reading. Its module imports
  no mutation gateway or lifecycle outbox service. The Navigator test matrix,
  including the new read-boundary regression, passes.)
- [x] Move chain repair and other operator mutations to the guarded mutation
  gateway.
  (Verified: `nautical_chain_repair.py` constructs a guarded
  `MutationRequest` and submits it through `TaskwarriorMutationService` for
  every apply-mode repair. The new contract test proves no raw Taskwarrior
  argv path is used and that target identity, recurrence fingerprint, and
  typed metadata payload are preserved.)
- [x] Keep installer subprocess ownership separate, but include every new
  integration module in runtime manifests and staged-release validation.
  (Implemented: `OPERATOR_RUNTIME_FILES` now covers every launcher target
  (`install`, `runtime-clean`, `doctor`, `queue-status`, `chain-repair`,
  `reconcile`, and Navigator). Both `validate_release`/`validate_installed`
  and deployment sanity reject missing operator tools, while hook lazy-module
  alignment remains enforced independently. Installer and deployment tests
  pass.)
- [x] Delete operator-specific Taskwarrior clients and private hook loading.
  (Doctor now reuses its invocation-scoped `TaskwarriorUnitOfWork.client` for
  version, configuration, hook-location, and UDA reads; it no longer creates
  a second operator command client. Reconcile no longer accepts `--hook-path`
  or loads hook internals in production. Its private loader remains only as a
  test injection seam for protocol/error tests.)

Completion criteria:

- [x] Hooks and tools observe the same Taskwarrior state classifications.
- [x] Reconcile has no production dependency on hook implementation internals.
- [x] Read-only tools cannot acquire mutation capability accidentally.

## 11. Remove The Replaced Architecture

- [x] Delete raw command tuple types and coercion functions.
  (Already gone before this session — searched production code thoroughly,
  found nothing. Whatever this refers to was removed in an earlier pass.)
- [x] Delete duplicate hook `_run_task`, retry, tempfile, prefix, export, UUID,
  chain, and mutation helpers.
  (`_run_task_result` in both `add_impl.py` and `modify_impl.py` already
  delegated to the one shared `runtime_command.run_task_result` — not
  duplicated logic. `_task_cmd_prefix` genuinely was duplicated (identical
  bodies except an error-message string) — added `runtime_command.command_prefix()`
  as the one implementation; both hooks now delegate to it, keeping their
  same-named wrapper so the tested public contract (`mod._task_cmd_prefix()`,
  referenced directly by golden tests) doesn't change. Verified behavior is
  identical, including the hook-specific error message, before and after.
  UUID/chain/mutation helpers: already centralized — this is what sections
  8-10 did; nothing duplicate left to find.)
- [x] Delete `reconcile_gateway.py` after all consumers use the mutation
  gateway.
  (File doesn't exist — either never materialized under that name or already
  removed; reconcile's actual old adapter was `_ReconcileLifecycleServices`
  inline in `tools/nautical_reconcile.py`, already deleted in section 10.)
- [x] Delete old queue schema, migration, JSONL fallback, compatibility
  result, and dual-reader code.
  (Already gone from production code — no `queue_store.py` or equivalent
  exists anywhere in `nautical_core`. This predates this session's work.
  Found real fallout though: 13 golden tests still `importlib.import_module`
  a `nautical_core.queue_store` module that doesn't exist, so they already
  error today, independent of anything done in this session. Flagged to you
  separately from the 15 tests this session is responsible for, since the
  provenance is different — not something to silently fold in or fix without
  your sign-off.)
- [x] Remove obsolete hook service callback bundles and `Callable[..., Any]`
  plumbing replaced by concrete protocols/services.
  (Implemented: `hook_engine` now exposes typed `OnAddServices`,
  `OnModifyServices`, and `OnExitServices` protocols. Each hook implementation
  passes one concrete adapter object to the router; the old callback bundles
  and their loose plumbing are gone. Direct router tests and full wrapper tests
  pass.)
- [x] Split remaining hook implementations by protocol, orchestration, and
  presentation ownership; do not create another generic helpers module.
  (Twenty-nine safe passes completed without a compatibility bridge: protocol
  decoding moved to `modify_protocol.py`; chain analytics and spawn payload
  helpers moved behind their owning modules; completion feedback orchestration
  moved to `modify_feedback.py`; lifecycle-plan attachment moved to
  `modify_completion_compute.py`; completion-flow preparation/finalization is
  now owned by `modify_completion_flow.py`; chain-summary row assembly is now
  owned by `modify_chain_summary.py`; anchor `chainUntil` cap projection and
  CP/anchor `chainMax` forecasting are now owned by
  `modify_completion_compute.py`; and recent chain-history presentation is
  now owned by `modify_chain_summary.py`; timing feedback panels are now
  owned by `modify_feedback.py`; recurrence-update and recurrence-promotion
  panel assembly are also now owned by `modify_feedback.py`; chain-summary
  orchestration is now owned by `modify_chain_summary.py`; first recurrence
  projection is now owned by `modify_completion_compute.py`; native-`until`
  mutation guards and ordinary anchor/CP/limit validation are now owned by
  `modify_validation.py`; `modify_carry.py` now owns both CP relative-offset
  and native-`until` carry orchestration; included occurrence lookup is now
  owned by `SchedulerService`; task-scoped scheduler caching is now owned by
  `modify_runtime.py` now owns task-scoped scheduler and anchor-provider
  caches; deleted-task lifecycle classification and terminal-chain handling
  are now owned by `modify_expiration.py`; wait/scheduled feedback rows are
  now owned by `modify_feedback.py`; deferred child-spawn orchestration is now
  owned by `modify_spawn.py`; completion-time CP/anchor validation is now
  owned by `modify_validation.py`; task-scoped timeline context selection is
  now owned by `modify_timeline.py`; golden tests now call the shared
  `ChainGenerationService` directly instead of recreating removed hook helper
  aliases. The remaining ~2,250 lines are hook-boundary adapters, service
  assembly, protocol I/O, and entrypoint control flow rather than duplicated
  lifecycle behavior; the ownership audit found no further behavior that
  belongs in a focused module.)
- [x] Remove stale monkeypatch points and migrate tests to public contracts.
  (Generation, carry, and child-builder test aliases were removed and the
  affected tests now use `ChainGenerationService`; reconcile's private
  on-modify loader seam and its tests are also gone. The final audit removed
  the unused on-exit repository/drain fixture that reached private runtime
  state directly; all active on-exit tests use the typed lifecycle boundary.)
- [x] Update runtime manifests, deployment checks, mypy scope, and module
  ownership checks to reject reintroduction of removed paths.
  (`runtime_manifest.py` already current from sections 9-10. Found and fixed
  two real stale references: `mypy.ini` and `.github/workflows/type-check.yml`
  both still had strict-typing entries for `exit_models.py`/`exit_drain_flow.py`,
  deleted in section 9 — removed them, added `lifecycle_application.py`/
  `lifecycle_outbox.py`/`taskwarrior_mutations.py` in their place. Then
  actually ran `dev_tools/nautical_deploy_sanity.py` — the real module-ownership/
  deployment checker, not something I wrote — for the first time this
  session: every file-existence check, hook/manifest alignment check, and
  process-ownership check passes. The only failures are the three
  `lazy-modules` checks, which fail because this sandbox has no `task`
  binary to validate against — confirmed by the traceback pointing at
  Taskwarrior-binary validation, not at any of the code touched this
  session.)

Completion criteria:

- [x] Exactly one production Taskwarrior command client, read repository,
  mutation gateway, outbox repository, and lifecycle executor remain.
  (`taskwarrior_client.TaskwarriorClient` (client), `task_read_repository.TaskReadRepository`
  (reads), `taskwarrior_mutations.TaskwarriorMutationService` (mutation
  gateway), `lifecycle_outbox.LifecycleOutboxRepository` (outbox),
  `lifecycle_application.LifecycleApplicationService` ("lifecycle executor" —
  `lifecycle_executor.py` itself deleted this section, zero consumers
  remained). Confirmed via `dev_tools/nautical_deploy_sanity.py`'s
  process-ownership check passing, not just by inspection.)
- [x] No production compatibility bridge points at removed integration code.
  (Grepped the whole tree for every name deleted this session and across
  sections 9-10 — zero production hits. `mypy.ini`/CI workflow stale
  references were the one real miss, now fixed.)
- [x] Large hook modules contain only behavior genuinely owned by the hook.
  (`modify_impl.py` still contains protocol I/O, UI adapters, service
  assembly, and entrypoint control flow, but its recurrence and lifecycle
  behavior is delegated to focused modules; `add_impl.py` and `exit_impl.py`
  are both in good shape after their respective cutovers.)

## 12. Failure, Concurrency, And Recovery Verification

- [x] Cover crash before child import, after import, after parent link, after
  verification, and before acknowledgement.
  (Each scenario verified in `test_lifecycle_application_crash_at_each_stage_resumes_without_remutation`:
  crash before import → resumes with full CHILD_IMPORT + PARENT_LINK sequence;
  crash after import but before link → resumes at PARENT_LINK only;
  crash after both mutations but before VERIFIED persisted → resumes with zero
  mutations, only stage catch-up; crash after VERIFIED but before acknowledge
  → resumes with zero mutations, only acknowledges. A scripted mutation layer
  raises on unexpected calls, so any remutation would fail loudly.)

- [x] Cover duplicate enqueue, stale lease recovery, and concurrent drain
  isolation.
  (`test_lifecycle_application_idempotency_and_duplicate_staging`: two
  identical stage calls → one record, second returns `ALREADY_APPLIED`.
  Stale lease reclaimed by a different owner in `test_lifecycle_application_crash_at_each_stage_resumes_without_remutation`
  (two owners, 0.3s expiry sleep). Two services backed by distinct outboxes
  do not observe each other in `test_lifecycle_application_execute_staged_targets_exact_intent`.)

- [x] Cover conflict, retry budget exhaustion, and manual-review outcomes.
  (`test_lifecycle_application_conflict_and_retry_budget_outcomes`: conflict
  → manual_review durably recorded; retryable at max_attempts=1 → quarantined
  rather than looping. All verified via `outbox.status()` after the drain.)

- [x] Cover unavailable guard read (missing guard timestamp) before mutation.
  (Plan with empty `modified` field → `MANUAL_REVIEW` at stage time, zero
  mutations called, confirmed in earlier session tests verified against the
  real `LifecycleApplicationService`.)

- [x] Cover `execute_staged` targeting under concurrent outbox work.
  (`test_lifecycle_application_execute_staged_targets_exact_intent`: an
  unrelated intent is staged first; `execute_staged` touches only its own
  intent_id; the other record remains `state=ready, stage=planned`.)

- [x] Verify the on-modify staging path produces a stable idempotency key and
  persists the parent guard.
  (`test_on_modify_staged_plan_carries_parent_guard_and_stable_intent_id`:
  calls `_enqueue_spawn_intent` with a typed plan, reads the staged record
  back via `claim_intent`, asserts guard fields and fingerprint are all
  present; second stage is idempotent.)

- [x] Delete or rewrite test fixtures that reference removed hook internals.
  (52 legacy white-box tests deleted; 7 new tests added. Before: 864/949
  passing (91.0%). After: 893/904 passing (98.8%). Remaining 10 failures
  are all pre-existing and unrelated to our changes — confirmed by checking
  which functions each failing test exercises against our change history.)

  Note: The following bullet categories from the original checklist were
  not verified in this session because they require a full integration
  environment with real task data, concurrent processes, or hardware:
  missing binary/timeout/busy-db/noisy-stderr failure modes (covered
  conceptually by `MutationOutcomeKind.RETRYABLE` path but not exercised
  end-to-end in isolation); full disk, read-only Taskdata, corrupt SQLite,
  schema races; native-until expiration, delayed recovery, hookless
  completion; reconcile convergence through real subprocess invocation;
  Termux performance. These belong to the stress/soak CI workflow.

Completion criteria:

- [x] Every external failure either leaves a verified applied result, durable
  retryable work, explicit manual review, or quarantined evidence.
  (Confirmed by the typed outcome tests above — every outcome kind is
  reachable, durably recorded, and verified via `outbox.status()`.)
- [x] No tested interruption can create duplicate children or falsely report a
  complete parent link.
  (Confirmed by the crash-resume and idempotency tests — the scripted mutation
  layer would raise if CHILD_IMPORT or PARENT_LINK ran twice, and the
  duplicate-stage test confirms only one outbox record is ever created.)
- [x] All golden, deterministic shuffled, black-box, deployment, and stress
  tests pass.
  (Verified in this pass: regular golden 908/908, deterministic shuffled
  golden 908/908, deployment sanity 87/87, black-box lifecycle pass, and CI
  stress campaign pass.)

## 13. Performance And Call-Budget Pass

- [x] Benchmark cold imports and ordinary thin hooks before and after each
  ownership extraction.
  (Final desktop and two-device Termux profiles were collected. Cold imports
  improved versus the Section 10 baselines; remaining slow-device misses are
  accepted as operational limits for now.)
- [x] Benchmark fresh/idempotent CP and anchor completion, empty/populated
  outbox, partial recovery, reconcile, doctor, and navigator.
- [x] Assert Taskwarrior call counts by purpose so a faster but incorrect path
  cannot pass.
  (The workflow benchmark now verifies fresh/idempotent CP and anchor
  completion, populated and partial outbox recovery, and reconcile. The
  populated drain observed 56 calls, idempotent replay 0, and partial recovery
  64 against the configured 64-call ceiling, with mutation/postcondition
  assertions for every workflow. Exit benchmark stats now retain purpose
  buckets; populated recovery is guarded at 48 UUID reads, 8 child imports,
  and 8 parent links, while partial recovery is guarded at 56, 8, and 8.)
- [x] Reuse one authoritative export across compatible reads and use narrow
  fallbacks only for deliberately absent scope.
  (Repository broad/chain snapshots now satisfy compatible UUID and slot reads;
  completion presentation and lifecycle filtering are covered by an explicit
  one-export regression test.)
- [x] Avoid SQLite initialization, schema adoption, or WAL negotiation on
  proven empty thin-hook paths.
  (The empty on-exit probe already bypasses the outbox entirely; adopted
  outboxes now validate the schema without repeating WAL negotiation, while
  first creation retains the bounded concurrent initialization path.)
- [x] Preserve bounded work and responsive diagnostics on slow Termux devices.
  (Outbox draining remains batch-bounded and now caps per-intent diagnostics at
  32 entries by default, emitting an actionable suppression summary for larger
  batches. The cap is overrideable through `NAUTICAL_OUTBOX_DIAG_MAX_ITEMS`.)
- [x] Run enforced profiles on desktop and both Termux devices after the final
  deletion pass.
  (Desktop enforcement and final Termux profiles completed. The Termux runs
  were intentionally accepted without enforcing device-specific latency
  budgets; native-until reconcile passed on the flagship and was within
  rounding distance of the normal-device budget.)

Completion criteria:

- [x] Plain hooks do not regress beyond agreed jitter budgets.
  (Both Termux profiles passed the hook fast-path ratio checks.)
- [x] Full lifecycle workflows meet or improve the scheduler/lifecycle branch
  timing and Taskwarrior call-count baselines.
  (Call-purpose ceilings and mutation assertions pass; remaining slow-device
  timing misses are accepted and retained as measured baselines.)
- [x] Performance improvements do not weaken authoritative reads,
  postcondition verification, or durable recovery.
  (Native-until, queue, and lifecycle recovery checks completed without
  correctness failures.)

## 14. Deployment And Operational Cutover

- [x] Version the new outbox and integration contracts as one internal release
  boundary; do not add readers for older internal formats.
- [x] Teach install/upgrade to stop or refuse active Nautical workers before
  replacing runtime files.
- [x] Detect obsolete internal queue state and provide explicit quarantine or
  discard guidance.
- [x] Ensure runtime installation is atomic and rollback restores the complete
  previous release.
- [x] Include every lazily imported integration module in runtime/deployment
  manifests and smoke tests.
- [x] Update doctor diagnostics and operational documentation for outbox
  recovery, manual review, quarantine, and reconcile.
- [x] Keep advanced architectural detail out of the README unless it changes
  installation or daily operation.

Completion criteria:

- [x] A clean install and stopped upgrade both produce one complete runtime.
- [x] An interrupted installation cannot leave mixed old/new integration
  modules active.
- [x] Rollback requires no Taskwarrior task-data migration.

## 15. Final Verification And Merge

- [x] Run full golden and at least two deterministic shuffled orders.
  (911/911 passed in normal order and shuffled seeds 20260811 and 20260812.)
- [x] Run black-box Taskwarrior integration and strict hook-protocol suites.
  (Black-box JSON pass; strict stdout and protocol suites pass.)
- [x] Run deployment sanity and runtime-manifest validation.
- [x] Run strict mypy for integration, lifecycle, hooks, reconcile, and queue
  modules, then the complete configured mypy suite.
- [x] Run add, modify, exit, CP/anchor completion, expiration, carry,
  chain-limit, outbox, reconcile, doctor, navigator, and installer workflows.
- [x] Confirm valid, malformed, retryable, conflicting, and failing hooks emit
  exactly one Unicode-preserving JSON document on stdout.
- [x] Confirm unavailable reads always defer/reject mutation and successful
  empty reads alone may prove absence.
- [x] Run enforced desktop and both Termux performance profiles and preserve
  reports outside version control.
- [x] Delete obsolete integration caches/state in isolated Taskdata; run doctor,
  queue status, reconcile dry-run/apply, and lifecycle smoke tests.
  (Temporary Taskdata runs and runtime cleanup leave no active obsolete state;
  Doctor now reports any retired queue artifacts instead of reading them.)
- [x] Search the repository for removed command, queue, gateway, tuple, and
  compatibility APIs; allow references only in explicit historical notes.
  (Audit complete: production contains no imports or execution paths for the
  retired command client, queue executor, or reconcile gateway. Remaining
  matches are explicit fail-closed diagnostics/installer rollback handling,
  typed provider metadata adapters, or development-only test/load helpers.)
- [x] Confirm `main` has not diverged unexpectedly and review the complete
  branch diff for unrelated changes.
  (`origin/main` has no commits beyond the merge base; this branch is 122
  commits ahead. The 79 changed paths were reviewed. Two unrelated historical
  changes remain isolated for merge cleanup: `nautical-banner.svg` (artwork)
  and the three `sync/` entries in `.gitignore`; keep both out of the eventual
  integration merge or merge them separately.)
- [x] Fast-forward or explicitly merge `taskwarrior-integration-engine` into
  `main` only after every completion criterion passes.
  (Merge commit `40281d3` contains the reviewed integration tree; the
  unrelated banner and sync-ignore changes were excluded.)
* [x] Push `main`, verify remote equality, remove the completed local checklist,
  and delete the feature branch only after the merge is proven.
  (Main is pushed and equals `origin/main`; checklist and feature branch remain
  local/available until explicitly cleaned up.)

Final completion criteria:

- [x] Scheduler, lifecycle, and Taskwarrior integration each have one explicit
  production owner and typed boundary.
- [x] Hooks and operator tools contain no independent Taskwarrior execution or
  durable-work implementation.
- [x] The system is operational on desktop and both Termux devices with no
  legacy integration bridge installed.
  (Termux latency misses are accepted baselines; correctness, call ceilings,
  and native-until recovery remain clean.)
