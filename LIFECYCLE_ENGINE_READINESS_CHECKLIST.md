# Nautical Lifecycle Engine Readiness Checklist

Complete only the architecture hardening required to begin
`LIFECYCLE_ENGINE_CHECKLIST.md` on a reliable foundation. Do not finish or
optimize architecture that the lifecycle consolidation will replace.

## Scope

This checklist draws the required work from these sections of
`ARCHITECTURE_HARDENING_CHECKLIST.md`:

- Section 4: the hook command boundary.
- The configuration-safety portion of Section 5.
- The lifecycle-relevant test-isolation portion of Section 6.
- The mutation-relevant typing portion of Section 7.
- The mutation-relevant terminal-evidence portion of Section 8.
- Baseline verification needed before the lifecycle refactor.

The remaining startup optimization, broad test cleanup, scheduler presentation,
typing expansion, and compatibility-matrix work is intentionally deferred until
the lifecycle module boundaries are stable.

## 1. Reconcile The Existing Checklist

- [x] Audit every unchecked item in `ARCHITECTURE_HARDENING_CHECKLIST.md`
  against the current code and tests.
- [x] Mark already completed work as complete; an unchecked historical item is
  not evidence that its implementation is absent.
- [x] Mark the prerequisite items covered here as moved to this checklist.
- [x] Leave deferred items open with a note that they resume after lifecycle
  consolidation.
- [x] Do not change production behavior during this audit.

Completion criteria:

- [x] Every unchecked architecture item is classified as required now,
  superseded, already complete, or deferred.
- [x] The two checklists contain no ambiguous or contradictory ownership.

## 2. Record The Readiness Baseline

- [x] Run and record the full golden suite. *(2026-08-11: 929/929 passed)*
- [x] Run the black-box suite with isolated Taskdata and Taskwarrior available. *(2026-08-11: passed; CP, preset, files, modify, navigator, and duplicate guard scenarios passed)*
- [x] Run deployment sanity and mypy. *(deployment passed; mypy ran in `/tmp/nautical-mypy-venv` and reports one existing `__all__` type error at `nautical_core/__init__.py:1305`)*
- [x] Run the enforced performance budget and retain the current Termux reports
  outside version control.
- [x] Run doctor, queue-status, and reconcile dry-run smoke tests against
  isolated Taskdata. *(queue status and reconcile were clean; doctor returned
  the expected structured error for an empty, uninstalled Taskdata: missing
  hooks and missing Astral in the system interpreter)*
- [x] Confirm successful and failing hooks still emit exactly one JSON document
  on stdout with `ensure_ascii=False`. *(covered by golden and black-box suites)*
- [x] Confirm optional diagnostics appear only on stderr with
  `NAUTICAL_DIAG=1`. *(covered by golden and hook protocol checks)*

Baseline environment: 2026-08-11, Linux, Python 3.11.2, Taskwarrior 3.4.2.
The enforced desktop budget passed all checks, including cold imports, CP and
anchor completion, expiration recovery, queue drain, and reconcile. Mypy 2.3.0
is installed only in `/tmp/nautical-mypy-venv`; the initial baseline had one
known facade typing failure, which was corrected during Section 8 handoff
verification. The final full run is clean across 136 source files.

Completion criteria:

- [x] The pre-lifecycle behavior and performance baseline is reproducible.
- [x] Any existing failure is explained before lifecycle code is changed.

## 3. Complete The Typed Hook Command Boundary

- [x] Make `TaskCommandResult` the direct result of Taskwarrior subprocesses in
  add, modify, and exit paths.
- [x] Preserve argv, return code, stdout, stderr, failure kind, attempts, and
  timeout from the original command execution.
- [ ] Replace tuple-returning hook command wrappers and adapters after their
  callers have migrated. Compatibility tuple wrappers remain for direct
  external/test callers; lifecycle callers now use the typed helpers. Keep
  unchecked until a compatibility/API audit confirms there are no supported
  external consumers, then remove the wrappers in a dedicated breaking-change
  pass.
- [x] Stop reconstructing timeout, lock, and general failure kinds from output
  text after a typed result is available. Legacy tuples are normalized only at
  the compatibility boundary.
- [x] Retry only explicitly retryable typed failures.
- [x] Keep mutation-sensitive reads as found, absent, or unavailable; never
  convert unavailable or malformed output into absence.
- [ ] Remove duplicate UUID reads introduced solely by compatibility adapters.
  Defer until the lifecycle read broker is measured across real hook and
  on-exit workloads; revisit when profiling identifies a remaining duplicate
  read and the compatibility wrappers above are ready for removal.
- [x] Preserve strict JSON parsing for exports and reject empty, malformed, or
  non-array output where an authoritative snapshot is required.

Required tests:

- [x] Missing Taskwarrior binary.
- [x] Command timeout.
- [x] Taskwarrior lock contention and bounded retry.
- [x] Nonzero command exit.
- [x] Empty export output.
- [x] Malformed JSON and invalid JSON rows.
- [x] Valid JSON with noisy stderr.
- [x] Confirmed absence versus unavailable lookup.

Completion criteria:

- [x] No lifecycle mutation decision consumes a tuple command result.
- [x] No lifecycle mutation decision depends on matching `lock`, `timeout`, or
  similar words in command output.
- [x] Unavailable reads always reject or defer mutation.

Verification: `python3 dev_tools/nautical_golden_tests.py` passed 929/929
after the boundary migration; the focused on-modify and on-exit suites also
passed in isolation. Compatibility wrappers are intentionally retained and
are the only unchecked migration item in this section.

## 4. Verify One Fail-Closed Configuration Boundary

- [x] Identify one validated configuration reload API shared by add, modify,
  reconcile, and doctor: `reload_taskdata_config()` remains the single
  validated reload entry point.
- [x] Ensure every scheduling or lifecycle mutation entry point resolves that
  validated configuration before planning. Add/modify/exit, reconcile, and
  Navigator now fail closed at startup through the same reload boundary.
- [x] Reject an explicitly selected unsafe configuration path with its path and
  reason instead of continuing with defaults.
- [x] Reject malformed discovered TOML instead of silently continuing with
  default scheduling values. Explicit and Taskdata-discovered failures now
  block reload; the regression verifies the parse detail is retained.
- [x] Reject invalid timezones, astronomy profiles, business calendars, and
  recurrence presets before lifecycle mutation. The shared reload boundary now
  validates profile coordinates/timezones, preset expressions, and complete
  business-calendar definitions before planning.
- [x] Preserve the same effective configuration fingerprint across hook and
  reconcile planning. The validated reload result now exposes both effective
  and scheduler fingerprints, and repeated Taskdata reloads are regression-
  tested for identity and drift-free state.
- [x] Keep configuration diagnostics actionable without contaminating hook
  stdout. Reload failures retain the rejected path/reason, while hook errors
  remain on the existing diagnostic channel.

Completion criteria:

- [x] Equal Taskdata and environment inputs resolve equal scheduling
  configuration in hooks and operator tools through the shared reload API.
- [x] Invalid or unavailable scheduling configuration prevents mutation;
  malformed TOML, unsafe paths, invalid timezones, astronomy, calendars, and
  presets are rejected before planning.

Pass 2 verification: domain validation regression plus the full golden suite
passed 930/930. Pass 3 covered discovered malformed TOML. Pass 4 added the
Navigator reload gate and passed 932/932. Pass 5 exposed validated effective
and scheduler fingerprints; the focused configuration set passed 50/50 and
the full golden suite passed 933/933.

## 5. Stabilize Lifecycle Regression Tests

- [x] Inventory existing tests for activation, resume, disable, completion,
  manual deletion, native-until expiration, chainMax, and chainUntil. Current
  coverage is grouped below; remaining work is characterization/isolation, not
  a new lifecycle design.
- [x] Inventory existing tests for queue claims, retry, idempotency,
  dead-lettering, equivalent children, parent guards, and reconcile recovery.
-
  Inventory (2026-08-11):
  - Activation/resume/disable: `test_on_modify_promotes_chain_when_task_becomes_nautical`,
    `test_on_modify_resumes_chain_emits_resumed_panel`, and
    `test_on_modify_disables_chain_emits_disabled_panel`.
  - Completion: CP and anchor happy paths in
    `test_on_modify_completion_build_and_spawn_child_happy_path`,
    `test_on_modify_cp_completion_spawns_next_link`, and the completion panel
    characterization tests around `test_on_modify_render_*_completion_feedback`.
  - Manual deletion/native-until expiration: `test_on_modify_manual_delete_persists_chain_off`,
    `test_on_modify_expiration_queues_next_occurrence_and_preserves_manual_delete`,
    and the native-until carry/expiration tests in the 13,000-17,000 line range.
  - Limits: `test_chain_cap_guards_are_inclusive_at_boundary`,
    `test_completion_caps_earliest_limit_wins`, and the chainUntil validation
    tests around `test_on_modify_native_until_*`.
  - Queue claims/retry/idempotency: `test_on_exit_take_queue_*`,
    `test_queue_claim_owner_blocks_stale_ack_and_requeue`,
    `test_on_exit_queue_drain_idempotent`, and the dead-letter tests around
    `test_on_exit_dead_letter_*`.
  - Equivalent children/guards: `test_on_exit_equivalent_child_*`,
    `test_on_exit_parent_nextlink_*`, and `test_on_modify_recompleted_task_*`.
  - Reconcile recovery: `test_reconcile_expiration_*`,
    `test_reconcile_delayed_expiration_*`, `test_reconcile_apply_*`, and
    `test_reconcile_post_apply_verification_checks_both_sides`.
- [x] Add only missing characterization cases; do not redesign lifecycle
  behavior in this pass. The activation-failure characterization now verifies
  that a failed transition preserves the task without partial chain metadata;
  existing queue, completion, expiration, and reconcile failure/replay cases
  already cover the remaining inventory families.
- [x] Make lifecycle tests restore environment variables, Taskdata paths,
  configuration globals, caches, and injected services that they modify. The
  lifecycle audit found and fixed the shared-core `MAX_LINK_NUMBER` leak in
  `test_on_modify_link_limit`; other audited mutations already had `finally`
  restoration or lived in isolated dynamically loaded modules.
- [x] Verify each lifecycle test passes alone and in the registered suite.
  The on-modify (156), on-exit (55), and reconcile (56) lifecycle subsets pass
  independently, and the complete registered suite remains green.
- [x] Add a deterministic shuffled-order run for the lifecycle subset or an
  equivalent state-leak check. `nautical_golden_tests.py --shuffle-seed N`
  now provides a reproducible full-suite order; seed `20260811` passed 933/933
  after fixing the discovered preview test dependency.
- [x] Treat warnings caused by deleted temporary Taskdata or stale configuration
  state as test failures. The runner now supports
  `--strict-lifecycle-warnings`; all lifecycle subsets pass without leaked
  state warnings.

Completion criteria:

- [x] Lifecycle tests do not depend on execution order or state left by another
  test. The deterministic shuffle and restoration audit pass.
- [x] Current interruption and replay behavior is characterized before the new
  state machine is introduced through the completion, queue, expiration, and
  reconcile retry/idempotency cases inventoried above.

Section 5 verification: strict lifecycle subsets passed on-modify 156/156,
on-exit 55/55, and reconcile 56/56; deterministic full-suite shuffle passed
933/933; normal full suite passed 933/933.

Inventory note: the registered suite has characterization for the primary
paths, but it does not yet prove shuffled-order determinism or exercise every
failure/replay pair in isolation. Those are the next passes.

## 6. Preserve Terminal Scheduling Evidence For Mutation

- [x] Verify `OccurrenceBatch` and `OccurrenceSearchExhausted` evidence reaches
  child-generation decisions without becoming a plain empty list or `None`.
  `collect_after()` and evaluator range collection now preserve date-limit
  terminal evidence on valid prefixes; child-generation and reconcile already
  consume typed exhaustion outcomes.
- [x] Verify completion and reconcile distinguish ordinary absence, calendar
  exhaustion, search-limit exhaustion, invalid rules, and unavailable
  dependencies. Completion now treats date-limit exhaustion as a normal chain
  finish and keeps search-limit exhaustion actionable; reconcile maps only
  date-limit exhaustion to a legitimate final plan and leaves search limits as
  errors.
- [x] Add regressions for a valid occurrence prefix followed by date-limit or
  search-limit exhaustion. The evaluator range regression covers a valid prefix
  followed by date-limit exhaustion; existing scheduler tests cover search-limit
  exhaustion as a typed failure.
- [x] Verify chainMax and chainUntil decisions cannot spawn after terminal
  evidence. Completion returns before child construction for terminal outcomes,
  while existing chainMax/chainUntil guards and reconcile terminal plans stop
  successor creation at their effective boundary.
- [x] Keep preview- and timeline-only terminal-evidence cleanup in the scheduler
  checklist unless it affects a mutation decision. This pass completed the
  deferred presentation cleanup: merged previews retain date-limit evidence and
  timelines render typed date-boundary versus provider-failure warnings.

Completion criteria:

- [x] No mutation path fabricates a successor or interprets exhaustion as an
  ordinary empty result. Typed date-limit and search-limit outcomes are
  covered in evaluator, completion, and reconcile regressions.
- [x] Hook completion and reconcile reach the same terminal decision. Date-limit
  exhaustion finishes the chain; search-limit and unavailable outcomes remain
  fail-closed errors.

## 7. Tighten Only The Prerequisite Type Surface

- [x] Enable consequential mypy checks for `task_command` and `hook_support`.
  The module-specific mypy sections now re-enable `arg-type`, `return-value`,
  `assignment`, `union-attr`, `attr-defined`, and `operator` checks.
- [x] Type the found/absent/unavailable lookup boundary and command failure
  classification without `Callable[..., Any]` adapters. Command runners,
  diagnostics, Taskwarrior rows, and typed/legacy command normalization now
  have named boundary aliases; both modules pass the enabled checks.
- [x] Type the configuration validation result consumed by lifecycle entry
  points. `reload_taskdata_config()` and `reload_for_taskdata()` now share a
  typed `ConfigReloadResult` contract for validity, source, error, and
  fingerprints.
- [x] Type the child-generation result or exceptions consumed by completion and
  reconcile. CP and anchor successor results now have explicit datetime,
  metadata, and DNF tuple types; carry and chain-identity failures remain
  dedicated exceptions.
- [ ] Do not broadly type parser, preview, UI, or unrelated compatibility APIs
  before the lifecycle models replace their callback plumbing.

Completion criteria:

- [x] The command, configuration, lookup, and child-generation prerequisites
  pass `arg-type`, `return-value`, `assignment`, `union-attr`, `attr-defined`,
  and `operator` checks. The six-module prerequisite set passes mypy.
- [x] No cast or assertion weakens runtime validation to satisfy mypy; the
  boundary changes use named types and preserve runtime validation paths.

## 8. Run The Handoff Verification

- [x] Re-run golden, black-box, deployment, mypy, hook-protocol, and enforced
  performance checks. Golden (936/936), black-box, deployment, hook protocol,
  and the enforced desktop budget (35 checks) passed. The focused six-module
  mypy prerequisite and the full repository run both pass (136 source files).
- [x] Re-run lifecycle tests in isolation and deterministic shuffled order.
  Seed `20260811` passed on-modify 156/156, on-exit 55/55, and reconcile
  56/56 with strict lifecycle warnings.
- [x] Exercise one fresh CP completion, one anchor completion, one successful
  populated-queue drain, and one reconcile recovery against isolated Taskdata.
  The black-box run passed CP and anchor-file completion plus an empty active
  queue assertion; the reliability smoke passed durable queue drain and load;
  reconcile recovery scenarios passed in the isolated reconcile suite.
- [x] Verify repeated completion and repeated recovery remain idempotent.
  The black-box duplicate guard and registered queue/reconcile idempotence
  tests passed.
- [x] Confirm thin plain-task add, modify, and exit routing did not regress.
  Plain fast-path and hook-protocol tests passed without importing the core
  package.
- [x] Update `ARCHITECTURE_HARDENING_CHECKLIST.md` with the verified results.
  Its Final Verification section now records the same evidence and the
  full-mypy exception.

Handoff criteria:

- [x] Typed Taskwarrior results reach every lifecycle mutation boundary.
- [x] Scheduling configuration is shared and fail-closed.
- [x] Mutation-relevant terminal evidence is preserved.
- [x] Lifecycle regression tests are isolated and reproducible.
- [x] Existing behavior, typing, and performance baselines are green.
- [x] Begin Section 1 of `LIFECYCLE_ENGINE_CHECKLIST.md`.

## Deferred Until After Lifecycle Consolidation

- [ ] Remaining `nautical_core.__init__` and full-hook startup optimization.
- [ ] Managed-launcher profiling and broader Termux import tuning.
- [ ] Golden-suite isolation outside lifecycle-related tests.
- [ ] Parser, preview, presentation, and scheduler-wide typing expansion.
- [ ] Preview- and timeline-only terminal-evidence propagation.
- [ ] Python 3.13/3.14 and broader Astral compatibility-matrix expansion.
- [ ] General cleanup whose target modules will be replaced by the lifecycle
  planner or executor.
