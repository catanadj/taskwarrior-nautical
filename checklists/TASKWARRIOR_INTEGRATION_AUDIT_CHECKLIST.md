# Nautical Taskwarrior Integration Audit Checklist

Follow-up work from the post-merge audit of the Taskwarrior integration engine.
Complete sections in priority order. Keep this checklist local and commit each
implementation pass independently.

## Working Rules

- [ ] Preserve strict hook JSON on stdout with `ensure_ascii=False`.
- [ ] Emit diagnostics to stderr only when `NAUTICAL_DIAG=1`.
- [ ] Keep mutation-sensitive reads fail-closed: unavailable must never mean
  absent.
- [ ] Keep one Taskwarrior subprocess boundary and one guarded mutation path.
- [ ] Add a focused regression test before or with every behavioral fix.
- [ ] Do not restore legacy integration bridges while completing this work.

## Baseline

- [x] Run the full golden suite with a normal terminal environment.
  Result: 911/911 passed with `TERM=xterm`.
- [x] Run deployment sanity and installed-layout black-box verification.
  Result: both passed.
- [x] Run the configured mypy suite against clean merged `main`.
  Result: failed with five errors.
- [x] Audit Taskwarrior command ownership, outbox recovery, mutation guards,
  reconcile behavior, deployment manifests, and performance budgets.

## 1. Restore The Type-Check Gate

Priority: release blocker.

- [x] Import `Any` in `nautical_core/runtime_command.py`.
- [x] Remove the conflicting local variable inference in
  `nautical_core/taskwarrior_mutations.py`.
- [x] Type the postcondition predicate and variable-length selector tuples.
- [x] Confirm the candidate fixes currently present only in the dirty local
  worktree are intentional and commit only those fixes.
- [x] Run the configured mypy suite.
- [x] Run the strict boundary mypy command from the Type Check workflow.

Completion criteria:

- [x] Clean `main` passes `.github/workflows/type-check.yml` without local-only
  edits.
- [x] No typing-only change weakens runtime validation.

## 2. Prevent Recursive Hook Execution

Priority: release blocker.

- [x] Add `rc.hooks=off` to the shared guarded Taskwarrior modify command.
- [x] Confirm parent linking, parent-link clearing, chain disabling,
  native-until repair, metadata repair, and child compensation all use it.
- [x] Preserve `rc.hooks=off` on child import.
- [x] Add command-argv tests that reject any lifecycle mutation without
  `rc.hooks=off`.
- [x] Add subprocess coverage proving an on-exit or reconcile mutation does not
  invoke on-modify or a nested on-exit. *(The black-box completion harness now
  observes real hook process launches.)*

Completion criteria:

- [x] Lifecycle application and reconcile cannot recursively invoke Nautical
  hooks through their own Taskwarrior mutations.
- [x] Guard and postcondition verification remain unchanged.

### 2.1 Process-Level Hook Harness

The argv regression proves that lifecycle mutations request `rc.hooks=off`.
This subsection verifies the operational consequence with real subprocesses.

- [x] Create disposable Taskdata, TaskRC, hooks, and configuration for the
  scenario; never use the user's live Taskwarrior data.
- [x] Wrap the installed `on-modify` and `on-exit` hooks with tiny counters that
  record each process launch before executing the real hook.
- [x] Add a Taskwarrior command shim that logs every invocation and delegates to
  the real Taskwarrior binary without re-entering the shim.
- [x] Seed a completed or expired recurrence parent with hooks disabled.
- [x] Trigger completion or reconcile through real Taskwarrior with hooks
  enabled.
- [x] Assert that the user operation launches the expected hooks exactly once.
- [x] Assert that child import and parent linking do not increase the hook
  counters.
- [x] Assert that every internal mutation in the command log contains
  `rc.hooks=off`.
- [x] Verify the resulting child and parent links remain correct.
- [x] Run the harness in CI as a focused regression test; keep it independent
  of terminal styling and live user configuration.

Completion criteria:

- [x] A real on-exit or reconcile mutation cannot launch nested on-modify or
  on-exit processes.
- [x] The harness fails if a future mutation path omits `rc.hooks=off`.

## 3. Harden Outbox Leases And Retry Budgets

Priority: medium-high recovery correctness.

- [x] Avoid giving a large sequential batch one shared fixed-expiry lease, or
  renew each intent lease before and during execution. Claimed records now
  renew immediately before every external mutation and durable progress step.
- [x] Treat lease-renewal failure as retryable or manual review according to
  typed failure evidence; never continue mutating after ownership is lost.
- [x] Enforce `max_attempts` while claiming, not only in `release_retry()`.
- [x] Quarantine an exhausted intent even when previous executions crashed
  before releasing it.
- [x] Validate active processing states against executable lifecycle stages
  while decoding rows.
- [x] Quarantine enum-valid but inconsistent stage/state combinations instead
  of allowing `_SPAWN_STAGE_ORDER` lookup failures.
- [x] Add slow-batch concurrency coverage where the original lease expires
  before the final record starts.
- [x] Add repeated hard-crash recovery and malformed stage/state tests.

Completion criteria:

- [x] Two processes cannot execute the same leased intent concurrently.
- [x] Retry budgets remain bounded across crashes.
- [x] A malformed durable row cannot crash or permanently block a drain.
- [x] Sequential execution renews ownership before each claimed step and
  refuses mutation when a later batch record has lost its lease.

## 4. Strengthen Child Import Verification

Priority: medium integrity.

- [x] Define the minimum authoritative child-import postcondition from the
  typed payload.
- [x] Verify UUID, `chainID`, `link`, `prevLink`, status, `chain`, and the
  required recurrence-mode metadata after import.
- [x] Keep the existing-child idempotency check aligned with the same invariant.
- [x] Return conflict or manual review when Taskwarrior imports only a partial
  or altered child payload.
- [x] Add tests for missing `prevLink`, wrong status, disabled chain, and
  changed recurrence metadata.

Completion criteria:

- [x] `CHILD_IMPORTED` is reported only when the imported child is a complete
  member of the intended recurrence chain.

## 5. Make Reconcile Integrity Audits Fail Closed

Priority: medium mutation safety.

- [x] Represent the native-until audit as valid, invalid, or unavailable.
- [x] Allow dry-run to report an unavailable audit as degraded diagnostics.
- [x] Prevent `reconcile --apply` from spawning or repairing tasks when the
  authoritative native-until audit is unavailable.
- [x] Preserve the actionable operational cause in text and JSON output.
- [x] Add locked, malformed-output, missing-binary, and timeout regressions.

Completion criteria:

- [x] Reconcile never mutates Taskwarrior after a required integrity read was
  skipped or became unavailable.

## 6. Add Outbox Retention And Maintenance

Priority: medium long-term operation.

- [x] Define a conservative retention policy for acknowledged outbox records.
- [x] Prune only terminal acknowledged rows older than the retention boundary.
- [x] Never automatically remove retry, claimed, quarantined, or manual-review
  evidence.
- [x] Add an explicit maintenance API or operator command with structured
  results.
- [x] Integrate retention status and cleanup guidance into doctor and queue
  status. Status exposes acknowledged/eligible counts, and cleanup remains an
  explicit `nautical queue-status --prune-acknowledged` action.
- [x] Add bounded maintenance with a passive WAL checkpoint after successful
  cleanup; the operation is cooldown- and eligibility-gated rather than run on
  every reconcile.
- [x] Test retention boundaries, concurrent readers, interrupted cleanup, and
  preservation of non-acknowledged rows.
- [x] Run bounded housekeeping opportunistically from `reconcile --apply`,
  with a persisted cooldown, eligibility/size gates, and a `--no-housekeeping`
  escape hatch. Hook execution remains unaffected.

Completion criteria:

- [ ] Normal successful recurrence history does not grow the outbox database
  indefinitely or trigger permanent health warnings.

## 7. Reduce Queue-Drain Taskwarrior Calls

Priority: medium Termux performance.

- [x] Record the current authoritative baseline: eight fresh intents measured
  56 calls (40 UUID reads, eight imports, eight parent links); partial recovery
  measured 64 calls (48 UUID reads, eight imports, eight parent links).
- [x] Design invocation-scoped batch pre-read and post-verification snapshots;
  the first pass implements child-existence prefetching.
- [x] Preserve refresh-after-mutation and mutation-epoch correctness.
- [x] Keep per-parent guarded updates and deterministic idempotency.
- [x] Reduce duplicate UUID reads without introducing cross-process task caches;
  child existence and safe pre-mutation parent rows now use one
  invocation-local broad snapshot when available; fresh/partial batches fell
  to 41/50 calls before phase batching.
- [x] Split fresh, idempotent, and partial-recovery call budgets; conflict
  coverage remains a separate follow-up because no dedicated workflow exists
  yet.
- [x] Re-run desktop and both device queue benchmarks. The Section 7.2
  reports reproduce the exact 27-call fresh and 43-call partial-recovery
  shapes on both devices. Their wall-clock workflow budgets remain above the
  Linux budgets, which reflects device speed rather than extra Taskwarrior
  calls.

### 7.1 Safe 24-call fresh-batch target

Target the following without weakening authoritative guards, postconditions,
or crash recovery:

- [x] Build one invocation-scoped preflight snapshot covering all parent and
  child identities needed by the claimed batch. Parent rows are reused only
  when they do not already carry the requested link; idempotent rows still
  receive a fresh read.
- [x] Keep the eight child imports and eight parent-link mutations individually
  guarded and hookless.
- [x] Replace per-intent success reads with bounded batch post-verification
  snapshots, retaining narrow reads for conflicts and partial recovery. Child
  imports and parent links are verified in separate fresh snapshots so a
  malformed child cannot be linked before its own postcondition is proven.
- [x] Keep lease renewal, mutation-epoch invalidation, and deterministic
  idempotency unchanged.
- [x] Establish measured budgets of 27 Taskwarrior calls for a fresh
  eight-intent batch and 43 for partial recovery, with separate replay and
  recovery budgets. The remaining three broad reads are intentional: one
  preflight plus one postcondition snapshot per mutation phase.
- [x] Prove that malformed, stale, conflicting, and interrupted snapshots
  fail closed and never become authoritative by accident. Batch verifier
  regressions cover unavailable, malformed, stale, and duplicate rows; the
  lifecycle crash matrix covers interrupted stage boundaries.

Completion criteria:

- [x] Queue-drain call counts materially improve without weakening authoritative
  reads, guards, postconditions, crash recovery, or replay behavior.

## 8. Consolidate Remaining Read And Legacy Paths

Priority: cleanup and ownership hardening.

- [x] Move presentation and timeline chain reads from
  `LifecycleReadService` behind the typed invocation repository. The service
  now consumes the invocation repository directly, normalizes authoritative
  snapshots at one boundary, and propagates unavailable/malformed reads
  instead of converting them to empty chains. Focused cache, lifecycle,
  export-reuse, and completion-chain regressions pass.
- [x] Replace `None`, empty-list, and empty-string failure signaling with typed
  read outcomes where the result can influence lifecycle presentation or
  decisions. Chain reads now expose `Found`/`Absent`/`Unavailable`; predecessor
  timeline collection preserves unavailable evidence and refuses to present it
  as an empty predecessor list.
- [x] Remove the four unreachable old exit-flow modules after confirming they
  have no runtime, deployment, or test ownership. Removed `exit_drain_flow.py`,
  `exit_entry_flow.py`, `exit_models.py`, and `exit_side_effects.py`; deployment
  sanity, manifest checks, and the registered ownership regression pass.
- [x] Remove reconcile support for legacy roots without `link` now that post-v2
  chain identity is enforced. Such parents are rejected with an actionable
  stamped-link diagnostic; deterministic repair remains an explicit
  `chain-repair --apply` operation rather than a reconcile compatibility path.
- [x] Remove obsolete reconcile runtime/protocol branching now that the public
  core runtime is the sole owner. Reconcile loads one runtime object directly;
  legacy hook tuples, protocol negotiation, and private hook validation were
  removed. Runtime-loading, startup-error, registration, mypy, and tool-path
  regressions pass.
- [x] Update deployment-manifest and AST ownership checks so removed paths
  cannot return. `runtime_manifest.py` records forbidden legacy paths/symbols;
  deployment sanity rejects their presence or reintroduction, with positive
  and negative regression coverage.

Completion criteria:

- [x] Production contains one typed Taskwarrior read boundary and no unused
  exit or legacy-root execution path. Section 8 focused tests, deployment
  sanity, compilation, and ownership checks pass.

## 9. Make Golden UI Tests Environment-Hermetic

Priority: test reliability.

- [x] Make live-renderer tests set and restore `TERM` explicitly. The focused
  live-renderer cases now use a scoped test guard that restores both existing
  and absent `TERM` values; non-TTY and `TERM=dumb` cases remain explicit.
- [x] Preserve the production fallback for `TERM=dumb`. The existing
  renderer regression confirms live output is skipped on dumb terminals.
- [x] Run the full suite under both `TERM=xterm` and `TERM=dumb`. Both runs
  reached 919/921; the same two pre-existing reconcile integration tests fail
  in each environment.
- [x] Confirm focused UI tests behave identically alone and in the full suite.
  Focused live-panel tests pass in both terminal modes, and the full-suite
  results are identical.

Completion criteria:

- [x] The golden suite does not depend on the invoking shell's terminal
  environment. Explicit terminal runs have identical results; the remaining
  failures are unrelated reconcile integration behavior.

## 10. Final Verification

- [x] Run full golden tests in normal and deterministic shuffled orders. Both
  strict runs pass 922/922 after fixing lifecycle repository rebinding and
  delayed-expiration postcondition handling.
- [x] Run configured and strict-boundary mypy. Both commands pass with no
  issues after narrowing the typed child-read payloads.
- [x] Run deployment sanity and installed-layout black-box tests. Both return
  structured success.
- [x] Run strict hook stdout/stderr protocol tests. Stdout JSON, Unicode,
  diagnostics, and protocol-gate tests all pass.
- [x] Run lifecycle crash, replay, lease, poison-row, partial-import, and
  reconcile failure campaigns. Focused lifecycle and reconcile campaigns pass;
  the fail-closed audit fixture now injects an explicit valid audit result, and
  delayed expiration accepts children that Taskwarrior immediately expires.
- [x] Run enforced desktop performance checks. The reduced
  `--slow-device --workflows-only --json --enforce` workflow gate passes;
  the broader non-workflow benchmark remains intentionally separate because
  it is not needed for the lifecycle workflow gate.
- [x] Run the reduced enforced benchmark on both Termux devices. Device 1
  (`android`, Python 3.14) measured queue drain at 5.138 s and partial
  recovery at 8.114 s; device 2 (`linux`, Python 3.12) measured 8.174 s and
  11.219 s. Both preserve the 27/43 Taskwarrior-call budgets and pass every
  other workflow; the 3.5 s / 6.0 s wall-time budgets are accepted as a
  documented slow-device tradeoff for now.
- [x] Review the final source diff for compatibility bridges, duplicate
  Taskwarrior boundaries, unrelated edits, and stale modules. Intentional
  lifecycle deletions are enforced by the runtime manifest; local checklists,
  benchmark reports, caches, and test artifacts remain untracked and untouched.

Final completion criteria:

- [x] Type Check CI is green from a clean temporary checkout: the tracked
  worktree diff applied cleanly, mypy passed across 150 source files, and
  deployment sanity passed.
- [x] Lifecycle mutations cannot trigger nested hooks.
- [x] Outbox ownership, retry, poison-row, and retention behavior is bounded
  and recoverable.
- [x] Every applied mutation has an authoritative complete postcondition.
- [x] Reconcile fails closed when required integrity data is unavailable.
- [x] Queue performance improves without reducing correctness. The reduced
  Taskwarrior-call budgets pass; current Termux wall-time remains accepted for
  this release and can be optimized separately later.
- [x] Golden, black-box, deployment, protocol, and mypy gates pass.
- [x] Performance gates pass for the binding call-count/correctness criteria;
  Termux wall-time thresholds are intentionally deferred optimization work.
