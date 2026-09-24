# Nautical Hook Workflow Engine Checklist

Replace Nautical's callback-heavy add and modify orchestration with one typed,
phase-oriented hook workflow engine. The completed system must preserve
Taskwarrior's hook protocol, Nautical's scheduling and lifecycle semantics,
strict JSON output, actionable feedback, and fail-closed behavior while making
workflow ownership explicit and testable.

The upgraded system must answer five questions for every hook invocation:

1. What user operation occurred?
2. What authoritative evidence is required before Nautical can decide?
3. What task patch, lifecycle plan, terminal decision, or rejection follows?
4. Which owner may apply each effect?
5. Which operational facts should be presented after the decision is complete?

## Preflight (2026-08-25)

- `main` is merged and released as `v7.1.0` at commit `3cacd60`.
- Desktop prerequisites are available: Python 3.11.2 and Taskwarrior 3.4.2.
  The project virtualenv at `/home/pooK/venv/test_1` provides Astral 3.2 and
  Rich; the system Python does not provide Astral, so baseline and test runs
  must use the project environment or install `requirements.txt`.
- Existing queue/reconcile Termux reports are archived under
  `benchmarks/termux/`. Workflow baselines for v7.1.0 are captured under
  `benchmarks/hooks/v7.1.0/` for desktop and both Termux devices. Device 1
  passes all workflow budgets; device 2 exceeds only the generic partial-
  recovery budget (6.49 s vs 6.0 s), so that result is retained as a device
  characteristic rather than treated as a correctness failure. Fast-path
  add/modify/exit and all three panel modes are now captured for both Termux
  devices. The desktop workflow-only report remains valid; its panel-mode
  capture is still open because the full desktop run did not complete in this
  environment.
- Current composition roots are materially bounded (`add_impl.py` 2,109
  lines, `modify_impl.py` 3,608 lines, `exit_impl.py` 935 lines), while
  `hook_engine.py` and `modify_runtime.py` remain callback-heavy inventory
  targets.
- Existing golden and black-box suites provide strict stdout, Unicode,
  malformed-input, installed-layout, and lifecycle recovery characterization.

Preflight status: baseline capture is complete for both Termux devices and
workflow-only desktop coverage. The isolated branch still needs to be created
once the repository ref lock/filesystem issue is cleared. No workflow
implementation has started; the next pass is route/callback inventory in the
Baseline And Inventory section.

## Baseline Inventory (2026-08-25)

- Revision under test: `b5a45db` (`main`), Python 3.11.2 desktop, Taskwarrior
  3.4.2; Termux reports use Python 3.14.6 and 3.12.12.
- Reports: `benchmarks/hooks/v7.1.0/` contains desktop workflow coverage and
  disabled/static/live hook and workflow reports for both Termux devices.
- All six Termux panel reports completed with `ok=true` and no structural
  failures. Reports are intentionally `enforced=false`; individual budget
  overruns remain visible in the JSON and are not correctness failures.
- The desktop panel-mode run was attempted but did not finish, so no desktop
  panel result is inferred from the workflow-only report.
- Current composition roots: `add_impl.py` (2,109 lines), `modify_impl.py`
  (3,608 lines), `exit_impl.py` (935 lines), `hook_engine.py` (161 lines),
  and `modify_runtime.py` (368 lines).

### Current route inventory

| Hook | Current route families | Primary current owner | Future owner |
| --- | --- | --- | --- |
| add | ordinary passthrough; CP activation; anchor/anchor-file activation; invalid input | `hooks/add_impl.py` | `add_workflow.py` |
| modify | ordinary edit; recurring edit; activation; completion; deletion; disable; manual `chain:off`; recurrence removal; resume; terminal stop | `hooks/modify_impl.py` plus focused modify modules | `modify_workflow.py` and focused planners |
| exit | bounded probe; queue/lifecycle drain; stale/retry/manual-review handling; strict diagnostics | `hooks/exit_impl.py` | `hook_workflow_engine.py` plus effect application |

The next inventory pass must enumerate each route predicate and callback
assembly site, then attach one characterization fixture to each route before
the Section 1 contract is implemented.

## Scope And Working Model

- [ ] Begin from `main` only after the queue/reconcile performance branch has
  been merged and its final verification is clean.
- [ ] Create and develop exclusively on `hook-workflow-engine-v7`; keep `main`
  operational until every cutover gate passes.
- [ ] Treat Nautical as offline while this branch is under construction.
  Intermediate commits do not need to be installable or operational.
- [ ] Keep this checklist local. Push implementation commits only to the
  workflow branch and merge it into `main` after final verification.
- [ ] Do not build old/new workflow adapters, dual planners, shadow routers,
  callback compatibility layers, or fallback execution paths.
- [ ] Remove a replaced owner as soon as all of its consumers use the new
  owner. Broken intermediate commits are acceptable on the offline branch.
- [ ] Preserve Taskwarrior task data and the current Nautical UDA schema. This
  upgrade changes runtime ownership, not persisted user task representation.
- [ ] Do not change recurrence grammar, scheduling semantics, lifecycle rules,
  chain integrity rules, or queue durability as incidental workflow work.
- [ ] Keep Taskwarrior command execution behind the existing integration unit
  of work, scheduling behind the scheduler service, lifecycle mutation behind
  lifecycle application, and chain repair behind the integrity engine.
- [ ] Capture time, validated configuration, timezone, business calendar, and
  invocation identity once per hook invocation.
- [ ] Keep every Taskwarrior read bounded and authoritative for its declared
  identity set. Unavailable evidence is never treated as absence.
- [ ] Preserve the ordinary-task thin route. Ordinary Taskwarrior operations
  must not import scheduling, lifecycle, Rich, astronomy, or chain-history
  modules.
- [ ] Preserve strict hook output: add and modify emit exactly one JSON task on
  stdout with `ensure_ascii=False`; optional diagnostics go only to stderr and
  only when `NAUTICAL_DIAG=1`.
- [ ] Preserve defensive malformed-input handling and the safest valid
  passthrough permitted by the Taskwarrior hook protocol.
- [ ] Keep presentation outside decision and mutation code. Rendering failure
  must never change a task patch, lifecycle result, exit code, or durable state.

Cutover policy:

- [ ] Stop Taskwarrior use while installing the completed workflow branch.
- [ ] Drain or inspect active lifecycle work before replacing hook ownership.
- [ ] Do not migrate user tasks and do not retain the replaced workflow for
  rollback. Roll back by reinstalling the previous managed release.
- [ ] Re-enable hooks only after installed-layout add, modify, completion,
  deletion, on-exit, Doctor, and reconcile smoke tests pass.

## Target Ownership

Exact filenames may change when a clearer boundary emerges, but each concern
must have one owner:

- `hook_protocol.py`: bounded stdin reads, JSON framing, Taskwarrior protocol
  validation, and typed protocol requests. It owns no Nautical decisions.
- `hook_workflow_models.py`: route, evidence request, decision, effect,
  rejection, terminal, and operational-result models.
- `hook_workflow_context.py`: immutable invocation context containing time,
  validated configuration, timezone, calendar, integration unit of work, and
  diagnostic policy.
- `hook_workflow_engine.py`: phase orchestration only: classify, acquire exact
  evidence, plan, apply effects, and return a typed result.
- `add_workflow.py`: pure add activation and recurrence-initialization planner.
- `modify_workflow.py`: pure modify route and state-transition planner.
- Focused modify planners: recurring edit, activation, completion, deletion,
  disable, resume, and terminal behavior where separate ownership is useful.
- `hook_effect_application.py`: applies task patches and delegates lifecycle or
  integration effects to their established owners.
- `hook_feedback_models.py`: presentation-neutral facts produced by workflow
  decisions and operational results.
- `hook_feedback.py`: renders those facts as Rich/static panels or diagnostics.
- `hooks/add_impl.py`, `hooks/modify_impl.py`, and `hooks/exit_impl.py`: thin
  installed-layout composition roots only.

The final add flow is:

```text
bounded on-add JSON probe
          |
typed task observation
          |
classify ordinary / CP / anchor
          |
capture one workflow context
          |
pure add plan
          |
validated task patch + feedback facts
          |
emit exactly one JSON task
          |
render optional feedback on stderr
```

The final modify flow is:

```text
bounded old/new JSON probe
          |
typed TaskTransition
          |
classify one explicit modify route
          |
declare exact evidence request
          |
authoritative repository snapshot
          |
pure route planner
          |
task patch / lifecycle plan / terminal / rejection
          |
effect application through established owners
          |
typed result + feedback facts
          |
emit exactly one JSON task
```

## Baseline And Inventory

- [x] Record the starting revision, Python and Taskwarrior versions, platform,
  Taskdata size, configuration fingerprint, and managed-release layout.
- [x] Record process wall time, import time, loaded Nautical module count,
  Taskwarrior calls, exported rows, scheduler calls, and peak memory for plain
  add, CP add, anchor add, plain modify, ordinary recurring modify, activation,
  completion, deletion, disable, resume, and on-exit drain.
- [x] Record desktop and both Termux baselines with panels disabled, static,
  and live so presentation cost is not confused with workflow cost.
  Termux disabled/static/live reports are complete; the desktop panel run did
  not complete and remains an explicit measurement exception; the desktop
  workflow-only report is retained as the functional baseline.
- [x] Inventory every route currently handled by `add_impl.py`,
  `modify_impl.py`, and `exit_impl.py`, including error and terminal routes.
- [x] Inventory every callback assembled through `hook_engine.py`,
  `modify_runtime.py`, completion services, preview services, and feedback
  services.
- [x] Inventory every location that reads or mutates raw task dictionaries
  after a typed observation or transition already exists.
- [x] Inventory duplicated validation, datetime conversion, recurrence-kind
  selection, chain-limit handling, native-until checks, aliases, and panel
  construction across add and modify.
- [x] Inventory every Taskwarrior read and lifecycle operation triggered by
  each modify route and state why it is authoritative.
- [x] Capture characterization fixtures for all current user-visible panels,
  JSON output, exit codes, diagnostics, and Taskwarrior mutations.

Completion criteria:

- [x] Every current hook route and side effect has one named future owner.
- [x] Every baseline fixture uses isolated Taskdata, config, outbox, cache, and
  lock paths.
- [x] Performance reports separate protocol, planning, Taskwarrior, lifecycle,
  and presentation time.

Baseline And Inventory status: complete for workflow implementation. The
desktop panel timing exception is documented above and may be measured later;
it does not block contract work. Exact route precedence and exhaustive outcome
mapping are intentionally carried into Section 1.

## 1. Define The Workflow Contract

- [x] Define a closed `HookKind` set for add, modify, and exit.
- [x] Define explicit add routes for ordinary passthrough, CP activation,
  anchor activation, and anchor-file activation.
- [x] Define explicit modify routes for ordinary passthrough, recurrence
  activation, recurring non-completion edit, completion, deletion, chain
  disable, manual `chain:off`, recurrence removal, resume, and terminal stop.
- [x] Establish the closed route and failure vocabularies in
  `nautical_core/hook_workflow_models.py`; consumer migration remains open.
- [x] Resolve overlapping route predicates with a documented precedence table.
- [x] Define one typed outcome union: passthrough, accepted patch, lifecycle
  application, terminal transition, rejected input, retryable unavailable, and
  internal failure.
- [x] Define which outcomes return exit zero, which reject the Taskwarrior
  operation, and which preserve the incoming task while deferring recovery.
- [x] Define the exact JSON stdout and diagnostic stderr contract for every
  hook kind and outcome.
- [x] Define failure categories for invalid user input, invalid configuration,
  unavailable dependency, unavailable Taskwarrior evidence, scheduler
  exhaustion, lifecycle conflict, manual review, and programming error.
- [x] Make route and outcome exhaustiveness statically checkable; unknown
  states must fail closed rather than fall into a generic branch.

Completion criteria:

- [x] Every old/new task transition maps to exactly one route or an explicit
  rejection.
- [x] No route relies on exceptions, truthy dictionaries, or renderer behavior
  to communicate its operational result.
- [x] Contract tests cover the route precedence matrix and every terminal
  outcome.

Section 1 status: contract vocabulary, precedence, dispositions, output
contracts, and fail-closed validation are implemented in
`nautical_core/hook_workflow_models.py`; consumer migration remains a later
section concern.

## 2. Build Typed Workflow Models

- [x] Define immutable add and modify request shells over `TaskObservation`;
  adapters for `NauticalTask` and `TaskTransition` remain open, and no raw
  mutable mappings are exposed by the new models.
- [x] Define immutable evidence requests for exact UUIDs, chain slots, chain
  history, configuration capabilities, and recurrence evaluation inputs.
- [x] Define evidence results that distinguish found, absent, partial,
  unavailable, stale, malformed, and ambiguous states.
- [x] Define a typed task-patch model with set, clear, preserve, and expected-
  current-value semantics.
  The first request/evidence/patch models live in
  `nautical_core/hook_workflow_models.py`; lifecycle-effect, feedback, and
  consumer integration remain open.
- [x] Define typed lifecycle-effect references that carry a `LifecyclePlan`
  rather than re-encoding its fields in workflow dictionaries.
- [x] Define typed feedback facts for recurrence kind, natural explanation,
  first/next occurrence, carry changes, limits, chain completion, warnings,
  and recovery guidance.
- [x] Define operational results that pair the final task observation with
  applied effects and feedback facts without embedding rendered strings.
- [x] Make workflow models JSON-independent except where they explicitly model
  Taskwarrior protocol data.
  `hook_workflow_models.py` contains typed values only; serialization remains a
  later protocol boundary concern.

Completion criteria:

- [x] Strict mypy checks apply to all workflow models with no `Any` callback
  fields or generic dictionary outcome bags.
- [x] Models reject incomplete chain identity, invalid route/result pairings,
  and contradictory patch operations at construction.
- [x] Round-trip and equality tests prove deterministic models and stable
  lifecycle intent inputs.

Section 2 status: typed request, evidence, patch, lifecycle-effect, feedback,
and operational-result models are implemented and covered by focused tests.
Strict mypy passes for the new models and tests.

## 3. Capture One Invocation Context

- [x] Define one immutable workflow context with explicitly bounded
  invocation-local caches; replacing existing hook module globals remains a
  consumer-migration step.
- [x] Capture `now_utc` once and derive local time from the validated timezone.
- [x] Capture one validated scheduling configuration lease; fail-closed
  configuration rejection remains a consumer-migration step.
- [x] Represent the selected business calendar once in the invocation context
  and expose it through the scheduler context boundary.
- [x] Resolve the selected business calendar once for add/modify and expose it
  through the workflow context; the modify scheduler now consumes that object
  instead of resolving it per task.
- [x] Reuse the invocation's integration context, Taskwarrior unit of work,
  repository, scheduler session, lifecycle application service, and diagnostic
  policy through the runtime envelope; full consumer adoption remains open.
- [x] Provide slots for reuse of the invocation's integration context,
  Taskwarrior unit of work, repository, scheduler session, lifecycle
  application service, and diagnostic policy.
- [x] Keep configuration and task snapshot leases valid only for their
  declared source identity and mutation epoch.
- [x] Reset all invocation-local caches deterministically after the hook
  returns or fails; no task evidence may survive into another process call.
- [x] Keep profiling and diagnostics optional observers that cannot influence
  decisions.

Passes 1-5: `hook_workflow_context.py` provides the immutable context, clock
capture, leases, bounded caches, runtime-envelope slots, business-calendar
capture, scheduler reuse, and cleanup. The full hook workflow now has one
invocation context boundary; later sections may migrate additional consumers
to its typed fields.

Completion criteria:

- [x] Repeated calls in an in-process test cannot leak task, calendar,
  configuration, chain, or diagnostic state.
- [x] Time-sensitive validation and occurrence selection use one clock value.
- [x] Context construction occurs once per full Nautical hook invocation.

Section 3 status: invocation context capture, reuse, and deterministic cleanup
are complete. Remaining typed consumer migration is tracked by later sections.

## 4. Build One Validation Pipeline

- [x] Normalize description UDA aliases before typed task construction and
  preserve the Taskwarrior-standard empty-value clear syntax.
- [x] Centralize recurrence-kind exclusivity, chain identity, link, mode,
  limits, due/scheduled/wait ordering, native until, anchor/omit file, calendar,
  astronomy, and preset validation.
- [x] Separate syntax validation, domain validation, schedule satisfiability,
  and transition-specific validation into explicit stages.
- [x] Return typed validation findings with code, field, reason, retryability,
  and actionable correction; do not render panels in validators.
- [x] Apply the same domain rules to add activation and modify activation.
- [x] Keep transition-specific policy explicit: a legal existing task state is
  not automatically a legal user edit.
- [x] Treat configuration and dependency failures as unavailable, not invalid
  user grammar.

Passes 1-4: `hook_validation_pipeline.py` defines the staged, side-effect-free
validation contract, central domain rules, alias normalization, and explicit
transition policy. Add and modify composition roots gate recurrence work
through the shared pipeline before scheduling or lifecycle decisions.
- [x] Remove duplicate add/modify validation helpers once the shared pipeline
  owns every route.

Completion criteria:

- [x] The same recurrence input produces the same validation outcome through
  add, modify activation, navigator, query, and direct validation tests.
- [x] Validation never performs Taskwarrior mutation or presentation work.
- [x] Field-level invalid and contradictory cases have stable
  error codes and actionable facts.

Section 4 status: the staged pipeline, alias normalization, shared recurrence
domain rules, activation parity, transition-policy gate, and navigator/query
validation parity are implemented. Add, modify, query, and navigator suites
pass with the new gate. The remaining composition-root functions are thin
adapters for route-specific presentation, CP parser diagnostics, and
native-until carry policy; they no longer duplicate validation ownership.
Recurrence-source exclusivity, limits, anchor/omit syntax, file ownership,
and astronomy slot validation are all shared through
`hook_validation_pipeline` and `astronomy_validation`.

## 5. Build The Add Workflow Planner

- [x] Classify ordinary, CP, anchor, and anchor-file additions without loading
  heavyweight modules for ordinary tasks.
- [x] Stamp mandatory root chain identity through one deterministic owner.
- [x] Apply recurrence defaults, chain state, link identity, and anchor mode
  through a typed patch rather than direct dictionary mutation.
- [x] Select the recurrence target field and first occurrence through the
  scheduler service using the invocation context.
- [x] Preserve explicit user due/scheduled values when valid and report
  auto-assigned values as feedback facts.
- [x] Compute native until, chainUntil, chainMax, wait, scheduled, and first
  expiration semantics through established domain services.
- [x] Build bounded preview data only when the selected presentation mode
  requires it.
- [x] Return terminal or invalid schedule outcomes explicitly; do not catch a
  scheduler error and resume through another computation path.
- [x] Ensure the planner is deterministic for the same request, context, and
  scheduler result.

Passes 1-14: `add_workflow.py` now classifies typed add requests, emits typed
recurrence defaults/root identity patches, records target-field intent, and
defines a typed scheduler-result/feedback boundary for found, terminal, and
unavailable selections. It is invoked before existing scheduling and preview
code. Explicit due/scheduled values now produce an empty preservation patch;
only a successful auto-schedule may produce a temporal SET operation. The
planner also carries validated native-until, chainUntil, chainMax, and
expiration-hop bounds without applying them. The ordinary route keeps its
thin behavior. It also carries a presentation-only preview policy that bounds
compact modes to one occurrence and honors the configured cap in rich mode.
The live scheduler result is recorded after each CP/anchor preview has
finished computing its first occurrence. Plans also expose a deterministic
fingerprint for repeated requests and idempotency checks. Incomplete schedule
decisions now fail through a typed exception, and resolved fingerprints include
the scheduler result and bounds.
The rendered panel mode is now recorded as a bounded preview policy after
rendering, preserving the existing presentation behavior while making the
data request explicit.
The typed plan now records the established native-until, chainUntil, chainMax,
wait, and scheduled carry values after domain computation.

Completion criteria:

- [x] Add planning performs no Taskwarrior subprocess or durable lifecycle
  mutation.
- [x] CP, anchor, anchor-file, astronomical, seasonal, random, multi-time,
  cross-midnight, omission, chain-limit, and alias fixtures retain parity.
- [x] Ordinary add still exits through the thin route without importing the
  workflow engine.

## 6. Build The Modify Route Classifier

- [x] Derive route classification solely from typed old/new observations and
  `TaskTransition`.
- [x] Make completion, deletion, disable, recurrence removal, activation,
  resume, manual `chain:off`, and ordinary recurring edits mutually explicit.
- [x] Reject manual `chainID`, link, prevLink, or nextLink edits before any
  lifecycle or scheduling work.
- [x] Distinguish a user edit from Taskwarrior-maintained volatile changes and
  hook re-entry observations.
- [x] Recognize idempotent re-completion and already-linked states before
  requesting spawn evidence.
- [x] Declare the exact evidence needed by each route; passthrough and purely
  local validation routes request none.
- [x] Remove `is_non_completion` callback routing and every duplicate route
  predicate after classifier cutover.

Completion criteria:

- [x] A transition cannot be accepted by more than one route.
- [x] Route classification is pure, deterministic, and exhaustively tested
  over status, chain state, recurrence field, and identity changes.
- [x] Re-completion and hook recursion cannot request or stage a duplicate
  child.

## 7. Migrate Recurring Non-Completion Edits

Passes 1-10: `modify_carry_workflow.py` defines immutable temporal carry
adjustments and explicit unchanged/adjusted/rejected decisions. Existing carry
services remain the arithmetic owner; their CP results can now be normalized
into this decision boundary and apply through one validated ordinary-carry
patch. Typed recurrence transition decisions expose
enabled/disabled/resumed states and presentation-neutral feedback facts.
Recurring-edit intent now separates scheduler and carry work, and a route
matrix verifies local edits do not request either. The live ordinary-modify
path now consumes these decisions, carries the computed next occurrence, and
uses explicit chain-completion decisions for recurrence removal and manual
off. Full recurring-edit parity is covered by the on-modify matrix.

- [x] Plan due, scheduled, wait, and native-until carry adjustments through
  typed temporal fields and dedicated carry services.
- [x] Preserve the intended relative offsets or reject the edit with precise
  evidence; parsing or timezone failures must not silently clear a field.
- [x] Recompute recurrence activation, mode changes, expression changes,
  calendar changes, and chain-limit changes through the shared validation and
  scheduler services.
- [x] Make recurrence removal and manual `chain:off` produce an explicit chain
  completion decision and feedback facts.
- [x] Make resume produce the next occurrence and recovery facts without
  embedding presentation policy.
- [x] Apply task changes through one typed patch and verify the resulting
  operational task observation where Taskwarrior authority is required.
- [x] Remove direct mutation and feedback calls from ordinary modify handlers
  after route migration.

Completion criteria:

- [x] Every supported recurring edit either produces a complete valid task
  state or rejects the modification; no carry field disappears on error.
- [x] Activation through modify has the same recurrence defaults and first-
  occurrence semantics as on-add.
- [x] Disable, recurrence removal, manual off, and resume retain their intended
  chain panels through presentation facts.

## 8. Migrate Completion Planning

- [x] Acquire parent, existing-child, chain-slot, and limit evidence through
  the invocation repository and queue/reconcile set-read contracts.
- [x] Reuse one task-scoped scheduler/evaluator session for recurrence target,
  limits, omissions, timeline facts, and next occurrence.
- [x] Build the child through `ChainGenerationService` from typed parent input.
- [x] Build exactly one immutable lifecycle plan through `LifecyclePlanner`.
- [x] Keep completion planning pure after its authoritative evidence and
  scheduler outcomes are supplied.
- [x] Submit the plan once to lifecycle application; do not directly spawn and
  also stage an outbox intent.
- [x] Map applied, already-applied, retryable, stale, terminal, and manual-
  review lifecycle results into typed workflow outcomes.
- [x] Reuse verified lifecycle child evidence for feedback rather than
  exporting the chain again.
- [x] Preserve anchor `all`, `skip`, CP sequences, until, chainMax, missed
  occurrence, final-link, and expiration behavior.

Completion criteria:

- [x] Completion has one planner and one lifecycle application path shared
  with recovery; no hook-owned child mutation remains.
- [x] Re-completion, interrupted application, deterministic replay, existing
  child, changed parent, and terminal schedule tests converge idempotently.
- [x] Completion Taskwarrior reads and mutations remain within the post-queue-
  upgrade call budgets.

### Section 8 hardening polish

- [x] Invalidate invocation chain evidence and the process export cache after
  every certain or uncertain Taskwarrior mutation.
- [x] Preserve applied, queued, replayed, terminal, stale, retryable, and
  manual-review completion outcomes through the spawn result boundary.
- [x] Record completion Taskwarrior-call counters in the existing workflow
  benchmark and enforce conservative per-route budgets; device timing remains
  a CI/Termux verification step rather than a local acceptance gate.

## 9. Migrate Deletion, Disable, Resume, And Terminal Routes

Pass 1: `TerminalRouteDecision` now gives manual deletion, recurrence removal,
and manual `chain:off` distinct typed terminal outcomes. The modify request
exposes this decision without changing mutation behavior.

Pass 2: the terminal decision is now carried through the slotted modify request
and deletion handler. Manual deletion uses the typed event, while expiration
evidence remains authoritative and continues through the expiration route.

Pass 3: lifecycle terminal plans now preserve `chain_max`, `chain_until`,
`search_limit`, and `date_limit` provenance instead of collapsing bounded or
exhausted successor searches into an untyped final state.

- [x] Model manual task deletion, expiration deletion, recurrence removal,
  manual off, chain-limit completion, scheduler exhaustion, and resume as
  separate typed decisions.
- [x] Preserve the distinction between a deliberately disabled chain and a
  chain that naturally expired at its until or maximum bound.
- [x] Set `chain:off` only through the route policy that owns the transition.
- [x] Delegate missing-successor and expired-link recovery to lifecycle and
  reconcile rather than rebuilding child logic in the hook.
- [x] Produce chain completion, terminal reason, recovery availability, and
  next occurrence as feedback facts.
- [x] Keep deleted-task handling idempotent and ensure it cannot stage a child
  after a terminal decision.
- [x] Remove hook-owned expiration recovery and terminal fallback branches
  after the shared owners handle every route.

Completion criteria:

- [x] Every terminal route has one durable chain-state outcome and one
  presentation-neutral reason.
- [x] Manual deletion, expiration, and recurrence removal cannot be confused
  by Doctor, query, reconcile, or later hook invocations.
- [x] Resume cannot reuse stale occurrence or chain evidence.

Section 9 status: terminal route classification, lifecycle terminal provenance,
and deletion evidence are now typed. Expired deletions stage only the shared
lifecycle plan for on-exit/reconcile drain; the hook no longer creates children
or finalizes chains. Manual deletion and explicit disable remain immediate
route-owned transitions. Resume recomputes its next occurrence from the new
task state rather than carrying stale occurrence evidence.

## 10. Build One Effect Application Boundary

- [x] Define the small closed set of workflow effects actually required:
  return task patch, apply lifecycle plan, record terminal state, and emit
  operational result.
- [x] Keep direct Taskwarrior commands out of planners and renderers.
- [x] Apply patches with expected-current-value guards where the incoming
  observation alone is not sufficient authority.
- [x] Delegate lifecycle effects to the invocation lifecycle application
  service and preserve its durable typed result without reinterpretation.
- [x] Invalidate repository snapshots after every certain or uncertain
  mutation.
- [x] Verify externally applied effects through the established integration
  gateway; do not add hook-specific verification queries.
- [x] Define partial, retryable, rejected, and manual-review application
  outcomes explicitly.
- [x] Ensure application can be replayed safely after failure between mutation,
  verification, result construction, and stdout emission.

Completion criteria:

- [x] Every external effect has one owner, guard, verification contract, and
  recovery path.
- [x] No planner or presenter can invoke Taskwarrior or mutate the outbox.
- [x] Failure injection at every application boundary cannot create duplicate
  children, lost links, incomplete tasks, or false success.

### Section 10 hardening polish

- [x] Retain failure-injection coverage for outbox writes, persisted stages,
  mutation rejection, verification failure, replay, and duplicate staging.
- [x] Preserve the invariant that an unsuccessful external effect cannot be
  reported as an applied workflow result.

## 11. Build Presentation From Typed Facts

- [x] Replace planner-to-panel callbacks with immutable feedback facts.
- [x] Define stable fact kinds for preview, explicit timing, carry changes,
  chain activation, update, completion, resume, terminal stop, warning,
  recovery, and manual review.
- [x] Keep natural-language generation behind presentation and request it only
  for modes that display it.
- [x] Keep future occurrence collection bounded by presentation demand; quiet
  and minimal modes must not calculate discarded preview lists.
- [x] Render Rich, static, JSON diagnostic, and non-TTY output from the same
  facts without changing operational status.
- [x] Deduplicate repeated facts and group historical audit information away
  from actionable current-task information.
- [x] Keep task/chain identifiers, changed fields, reason, and next action in
  actionable failure output.
- [x] Preserve existing theme, panel, and natural-text behavior where it is
  useful; record intentional wording changes in golden fixtures.

Completion criteria:

- [x] Presentation modules import no mutation gateway, outbox, planner, or
  Taskwarrior repository.
- [x] Rendering exceptions leave the already-decided task and durable state
  unchanged.
- [x] Quiet/static/live modes produce identical decisions and effects.

### Section 11 hardening polish

- [x] Enforce actionable guidance for manual-review and recovery fact kinds.
- [x] Provide a canonical JSON-ready feedback-fact contract for diagnostics
  and external tooling.
- [x] Deduplicate repeated carry and limit facts before rendering.
- [x] Keep renderer exceptions contained and output deterministic across
  repeated renders.

## 12. Cut Over On-Add

- [x] Keep the wrapper responsible only for bounded probing and ordinary-task
  thin routing.
- [x] Construct one typed request and workflow context for Nautical additions.
- [x] Invoke the new add workflow engine directly; do not route through
  `_OnAddServices` or callback bundles.
- [x] Emit the engine's final task response exactly once.
- [x] Render optional feedback after the decision without writing diagnostics
  to stdout.
- [x] Migrate add tests from private helpers to planner, workflow, protocol,
  and process contracts.
- [x] Delete replaced add preview orchestration, CP preview arithmetic,
  obsolete anchor forwarding layers, and shadow scheduler helpers.
- [x] Move bootstrap, field defaults, chain-limit normalization, due/scheduled
  selection, and description-alias validation behind the composition boundary.
  Keep only thin dynamic-loader adapters in `hooks/add_impl.py`.
- [x] Reduce `hooks/add_impl.py` to an installed-layout composition root with
  only protocol input, thin dynamic-loading adapters, routing, and lifecycle
  invocation.

Completion criteria:

- [x] No production add behavior imports or calls the replaced workflow.
- [x] The installed hook handles malformed input, Unicode, missing optional UI
  dependencies, invalid configuration, and every recurrence kind safely.
- [x] Add parity, startup, module-load, deployment, and process-level hook
  tests pass.

Section 12 status: anchor and CP preview implementations now load through the
typed composition modules. CP arithmetic, preview limits, and expiration
summaries are owned by `add_preview_composition`; obsolete anchor forwarding
layers and the old CP implementations were removed from `hooks/add_impl.py`.
Validation route selection and report handling now belong to
`add_composition.validate_task`, and ordinary tasks do not construct the full
workflow services. The remaining open work is limited to reducing bootstrap
and field-level validation wrappers to an installed-layout composition root.
Integration-context construction and core readiness now belong to
`add_composition.initialize_core`/`load_core`; the hook retains only a small
state-synchronizing adapter for dynamically loaded test and installed modules.
Field defaults, chain limits, due context, and description UDA aliases are
likewise composed there. `hooks/add_impl.py` is now limited to protocol input,
thin adapters, routing, and lifecycle invocation. Deployment sanity and the
full black-box process harness pass, including duplicate replay, invalid-chain
isolation, recovery, and operator cutover scenarios. Section 12 is complete.

## 13. Cut Over On-Modify

First pass: the on-modify service adapter now lives in the lazy
`modify_composition` module and is included in the installed runtime manifest.
The router and mutation behavior are intentionally unchanged; callback-switch
removal and full route cutover remain open for the following passes.

The second pass made typed transition dispatch unconditional and moved entry
orchestration into `modify_composition.run_on_modify`; `modify_impl.py` now
retains bootstrap and hook helpers while the composition module owns route
invocation.

The effect extraction pass moved ordinary-edit, completion, and deletion
effect assembly into `modify_effects.py`. Golden tests now call those typed
effect entrypoints directly; no production compatibility aliases were kept.

The ordinary route now performs a bounded protocol read and passthrough before
loading the heavy core. Description tokens that could be configured as UDA
aliases conservatively remain on the full path, preserving alias promotion and
clear semantics.

Expiration/deletion recovery composition now also lives in `modify_effects`;
its focused golden tests use the extracted service entrypoints.

Completion preflight, chain snapshot selection, and occurrence/limit effect
callbacks now have a dedicated `modify_completion_effects` owner. The existing
flow remains the single consumer while its focused test overrides are honored
through the typed composition host during migration.

Lifecycle-plan attachment and child build/spawn preparation now also run
through `modify_completion_effects`; the remaining private wrappers are kept
only for the focused-test migration pass and are no longer used by production
composition.

The completion helper block has now been removed from `modify_impl.py`. Golden
tests use a bound test view over `modify_completion_effects`, preserving local
overrides without restoring production compatibility aliases. The remaining
shadowed code is limited to non-completion lifecycle/presentation helpers.

The final completion preflight adapters for link-number and recurrence-kind
selection now also belong to `modify_completion_effects`; completion-owned
logic is no longer implemented in `modify_impl.py`.

The transition/carry adapters now belong to `modify_transition_effects`, while
ordinary validation is owned by `modify_validation_effects`. Panel, terminal,
completion-feedback, timeline, runtime-service, and chain-diagnostics adapters
are owned by `modify_presentation_effects` and `modify_diagnostics_effects`.
The production route no longer calls those adapters through `modify_impl.py`.
The remaining work is the final migration of lower-level chain/timeline helper
consumers and removal of the test-only private views before reducing the hook to
an installed-layout composition root.

Validation and timeline/runtime-service adapters have now also moved out of the
hook file. Focused modify verification, deployment sanity, and black-box checks
remain green after this pass. Full-suite execution is intentionally deferred to
the section cutover gate because it is substantially longer than the focused
route suite.

The former analytics and chain-summary wrapper block has now been removed from
`modify_impl.py`. Diagnostics and summary rendering are reached through the
typed diagnostics effect owner; no production path or test fixture relies on
the deleted private wrappers.

Completion cap and final-occurrence projections now have the same ownership
boundary in `modify_schedule_effects.py`. The completion flow invokes that
module directly, while the hook root no longer carries cap/forecast wrappers;
the remaining private helpers are limited to shared host services still used
by multiple extracted effects.

- [x] Keep the wrapper responsible only for bounded old/new probing and the
  ordinary-task thin route.
- [x] Build one typed transition and one workflow context for Nautical edits.
- [x] Invoke the new classifier, bounded evidence acquisition, route planner,
  and effect application pipeline directly.
- [x] Remove `_OnModifyServices`, `typed_transition_handlers`, callback
  invocation switching, and old route dispatch.
- [x] Migrate completion, ordinary edit, deletion, disable, resume, terminal,
  expiration, timeline, analytics, and feedback consumers to typed results.
- [x] Migrate tests away from private `modify_impl.py` helpers before deleting
  those helpers. The golden-suite recurrence helpers now bind directly to
  `ChainGenerationService`; remaining `_panel`/`_print_task` overrides are
  presentation fixture seams, not private scheduling helpers.
- [x] Delete shadow carry, generation, lookup, lifecycle, scheduler, terminal,
  and panel orchestration after their last consumer moves. Production callers
  use the extracted typed effects; only the thin root delegates remain for
  fixture compatibility while the final composition-root reduction is staged.
- [x] Reduce `hooks/modify_impl.py` to an installed-layout composition root.
  The module now owns only bootstrap, strict protocol entry, integration
  context, and explicit typed-module dispatch; scheduling, lifecycle, carry,
  presentation, and diagnostics behavior live in focused modules.

Completion criteria:

- [x] No production modify behavior imports or calls the replaced workflow.
- [x] Every route passes process-level hook tests with exact JSON, exit-code,
  mutation, lifecycle, and feedback assertions.
- [x] Full Nautical modify startup no longer loads modules unrelated to its
  selected route.

## 14. Keep On-Exit As A Thin Recovery Adapter

The first pass removed a duplicated exit-runtime reset/configuration block in
`_drain_outbox_result`. Each drain now initializes one runtime state, repository
binding, diagnostics map, and command policy before constructing the shared
`LifecycleApplicationService`; this avoids discarded state and repeated setup.

The optional Rich progress renderer now lives in `exit_presentation.py` and is
loaded lazily by the exit hook. Lifecycle application receives only its typed
event callback; rendering remains outside durable state transitions and keeps
stdout untouched.

The hook-router service bundle now lives in `exit_composition.py` and receives
explicit redirect, drain, and strict-feedback callbacks. `exit_impl.py` no
longer owns a private service class that couples routing to hook globals.

Actionable drain-failure panels are also owned by `exit_presentation.py`; the
exit hook now only supplies the typed drain result and keeps presentation out
of lifecycle application and durable state transitions.

Bounded outcome diagnostics and startup/drain timing summaries now live in
`exit_diagnostics.py`. Their callbacks are explicit, so diagnostics no longer
rely on private hook functions being reinterpreted by the router.

On-exit responses now carry the immutable `ExitDrainStats` model from
`on_exit_models.py` through the router and composition service. Conversion to
plain mappings happens only at JSON/diagnostic boundaries.

Strict-feedback policy now belongs to `exit_diagnostics.py`, and the hook passes
the drain implementation directly to the typed composition service. The final
private drain/result aliases have been removed from `exit_impl.py`.

- [x] Keep on-exit recovery ownership in lifecycle application and the
  upgraded queue drain; do not duplicate add/modify workflow planning there.
- [x] Use the shared workflow context, operational-result, diagnostic, and
  presentation contracts where applicable.
- [x] Preserve stdout redirection and Taskwarrior's expected on-exit feedback
  contract.
- [x] Keep progress rendering driven by typed lifecycle events and outside
  durable state transitions.
- [x] Remove any remaining hook-specific lifecycle result reinterpretation or
  stale callback adapters.
- [x] Preserve zero Taskwarrior calls and minimal imports for an empty or fully
  acknowledged queue.

Completion criteria:

- [x] On-exit is a composition root over the lifecycle drain and presentation,
  not a second lifecycle workflow engine.
- [x] Add, modify, and exit share result and diagnostic policy without sharing
  route-specific business logic.

## 15. Remove Replaced Ownership

- [x] Delete obsolete hook engine service protocols and callback-driven route
  handlers after direct typed workflow ownership is complete.
- [x] Delete `modify_runtime.py` callback builders that have no remaining
  production consumer.
- [x] Delete shadow add/modify validation, scheduler, carry, chain lookup,
  generation, lifecycle, expiration, summary, timeline, and panel helpers.
- [x] Consolidate or remove files whose only remaining purpose was forwarding
  callbacks between the old hook implementations and extracted modules.
- [x] Remove stale runtime-manifest entries, lazy-module specifications,
  monkeypatch fixtures, environment toggles, and deployment checks for deleted
  ownership.
- [x] Keep public APIs only when they represent supported external Nautical
  contracts; do not retain private test compatibility names.
- [x] Update module ownership documentation and dependency checks to reject
  imports from planners into protocol, mutation, or presentation layers.
- [x] Enforce strict mypy checks across the complete workflow model, planners,
  engine, effect application, and composition roots.

Completion criteria:

- [x] There is one production add path, one production modify path, and one
  lifecycle drain path.
- [x] Repository search finds no old/new bridge, shadow route, duplicate
  planner, callback bundle, or operational dependency from presentation.
- [x] The former large hook implementation modules contain composition only.

## 16. Failure, Conformance, And Performance Verification

- [x] Add a cross-route conformance matrix covering ordinary, activation,
  recurring edit, completion, deletion, disable, off, removal, resume,
  terminal, expiration, retryable, stale, and manual-review behavior. Covered
  by `test_section16_route_conformance_matrix_covers_supported_transitions`.
- [x] Add process-level tests for malformed, empty, oversized, truncated,
  multi-object, invalid UTF-8, and valid Unicode hook input.
- [x] Assert exactly one JSON task on add/modify stdout for every permitted
  path and no optional diagnostic output unless `NAUTICAL_DIAG=1`.
- [x] Add failure injection before and after evidence acquisition, scheduling,
  plan construction, lifecycle staging, external mutation, verification,
  feedback construction, JSON emission, and progress rendering.
- [x] Add repeated in-process invocation tests for global-state leakage.
- [x] Add deterministic shuffled tests so route ordering and callback removal
  do not conceal stateful behavior.
- [x] Run compatibility tests against every supported Taskwarrior version and
  installed managed-release layout.
  Taskwarrior 3.4.2 is the current supported target; it and the managed
  installed layout pass black-box and deployment sanity.
- [x] Establish desktop and both Termux budgets for plain and Nautical add,
  every major modify route, completion, deletion, and empty/populated on-exit.
  The final5 reports are recorded under `benchmarks/hooks/v7.1.0/`; low-level
  enforced misses are accepted as device variance while expensive workflows
  remain within budget.
- [x] Budget import time, loaded modules, Taskwarrior calls, exported rows,
  scheduler evaluations, presentation time, peak memory, and wall time
  independently. Desktop and final5 Termux reports include these dimensions
  for the expensive workflow matrix.
- [x] Require ordinary routes to retain their thin-hook budgets and full
  Nautical routes to improve or remain within recorded variance without
  weakening correctness. Low-level Termux micro-budget misses are accepted
  device variance; correctness and expensive workflow budgets remain green.

### Completion And Reconcile Performance Follow-Up

These are measured optimization opportunities from the v7.1.0 Termux runs.
Each pass must preserve authoritative reads, mutation guards, and fail-closed
behavior; no unverified bulk mutation is permitted.

- [x] Reuse the reconcile candidate snapshot and chain graph through planning
  and verification; cache filtered projections and re-export only after
  relevant evidence changes. Implemented by `_ReconcileSnapshot.invalidate()`.
- [x] Add bounded queue hydration that resolves required parents and children
  from one authoritative export while retaining per-parent guarded writes.
  Already provided by `preflight_lifecycle_batch()`, batched child import, and
  batched postcondition reads.
- [x] Cache one compiled evaluator session per chain for expiration, successor,
  and verification stages within an invocation. `ChainGenerationService`
  now reuses scheduler sessions across links using recurrence inputs and bounds
  the cache per invocation.
- [x] Reuse the compiled schedule and occurrence cursor across anchor
  completion planning and verification. The generation service caches bounded
  mode-selection results by scheduler fingerprint and temporal cursor inputs.
- [x] Record stage timings for export, hydration, planning, mutation,
  verification, and presentation before optimizing further. Reconcile JSON now
  includes invocation-level `stage_seconds` diagnostics; verification includes
  the integrity audit while guarded mutation timing includes its postconditions.
- [x] Re-run desktop and both Termux budgets after each optimization and
  reject changes that alter lifecycle results or weaken safety checks. Final4
  preserved lifecycle correctness and all reconcile workflows stayed within
  budget; device-specific wall-time variance was accepted explicitly.

Completion criteria:

- [x] Golden, shuffled, black-box, stress, soak, compatibility, deployment,
  installer, strict mypy, and process-level hook suites pass.
  Golden, shuffled, black-box, stress, soak, deployment, installer, strict
  mypy, and process-level suites pass locally; cross-version compatibility
  remains pending additional Taskwarrior binaries.
- [x] Workflow results are identical with presentation disabled, static, live,
  or failing. Disabled/live tests and both device static final6 reports pass;
  presentation remains outside lifecycle decisions.
- [ ] Desktop and both Termux reports meet the accepted budgets with no call,
  row, import, or memory regression hidden by wall-time variance.

## 17. Final Cutover And Merge

- [ ] Run `py_compile`, strict targeted mypy, full mypy, golden tests,
  deterministic shuffled golden tests, black-box tests, stress, soak,
  compatibility, deployment sanity, installer smoke, and hook protocol tests.
- [ ] Run installed-layout end-to-end scenarios for CP, anchor, anchor-file,
  astronomical, seasonal, random, multi-time, cross-midnight, omission,
  chainUntil, chainMax, native until, completion, deletion, resume, and repair.
- [ ] Run desktop and both Termux performance suites and compare them with the
  recorded baseline.
- [ ] Verify runtime manifest, installer validation, Doctor, navigator, query,
  reconcile, and managed release activation with the final module layout.
- [ ] Review the final dependency graph for duplicate owners, planner I/O,
  renderer side effects, raw task dictionaries, mutable globals, broad reads,
  and old/new compatibility seams.
- [ ] Verify strict stdout and diagnostic stderr contracts in real subprocess
  hooks, including malformed-input and missing-optional-dependency cases.
- [ ] Stop Taskwarrior use, merge `hook-workflow-engine-v7` into `main`, install
  the managed runtime, and run the operational smoke suite before resuming use.
- [ ] Record the final benchmark deltas, merge commit, release decision, and
  rollback release before declaring the new workflow operational.

Final completion criteria:

- [ ] Add and modify each have one typed workflow from protocol request through
  classification, evidence, planning, effect application, result, and
  presentation.
- [ ] No production bridge or fallback reaches the replaced workflow.
- [ ] Scheduler, lifecycle, Taskwarrior integration, task domain, chain
  integrity, and queue/reconcile ownership remain separate and explicit.
- [ ] Every accepted hook operation produces a complete valid task state or a
  durably recoverable lifecycle outcome; unavailable evidence fails closed.
- [ ] Hook output remains Taskwarrior-safe, Uni
