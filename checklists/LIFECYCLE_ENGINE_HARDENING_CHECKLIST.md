# Nautical Lifecycle Engine Reliability Hardening Checklist

> **For implementation:** Complete the sections in order. Each section has its
> own red/green verification gate and should be committed independently. Do not
> combine this work with the offline-kit changes already active in the current
> worktree.

**Goal:** Close the remaining lifecycle reliability gaps without changing
Nautical recurrence semantics, weakening Taskwarrior guards, or replacing the
existing lifecycle engine.

**Architecture:** Keep the current typed planner, guarded Taskwarrior mutation
gateway, and SQLite outbox. Harden only the boundaries that decide whether
manual work is safe to close, whether integrity work still owns its lease, and
whether persisted postconditions are meaningful. Prefer shared pure validators
and small compare-and-set changes over another workflow abstraction.

**Reference:** `checklists/LIFECYCLE_ENGINE_CHECKLIST.md`

## Global Safety Constraints

- [ ] Keep Taskwarrior as the authoritative task store. Never modify its data
  files or TaskChampion database directly.
- [ ] Keep SQLite limited to Nautical plans, claims, stages, failures, and
  operator evidence.
- [ ] Preserve deterministic child UUID generation and idempotent replay.
- [ ] Preserve parent compare-and-set guards and authoritative post-mutation
  reads.
- [ ] Treat missing, malformed, stale, or unavailable evidence as non-success.
- [ ] Keep manual review sticky unless a narrowly classified recovery policy
  explicitly allows retry.
- [ ] Never acknowledge an intent merely because a command returned success.
- [ ] Keep hook stdout to exactly one JSON document, preserve
  `ensure_ascii=False`, and emit diagnostics to stderr only when
  `NAUTICAL_DIAG=1`.
- [ ] Make small, targeted changes. Do not refactor the lifecycle planner,
  scheduler, or hook routing while completing this checklist.
- [ ] Do not add a general workflow framework, direct TaskChampion mutation,
  background daemon, network dependency, or automatic destructive repair.

## Audit Baseline

The 2026-09-02 audit found that the main lifecycle engine remains structurally
sound: typed plans, deterministic identities, guarded writes, durable stages,
lease ownership, batched verification, replay, and crash recovery are present.
The remaining work is concentrated at manual-review closure and the integrity
variant of the shared outbox.

Current verification evidence from the active worktree:

- 47 focused lifecycle, queue-review, failure-matrix, and effect-boundary unit
  tests passed.
- The shared integrity-outbox golden test passed.
- The lifecycle-outbox golden slice passed 6/7. Its existing recovery case
  failed with `known stale postcondition review was not reopened` after the
  active manual-review reopening condition was changed.

This failure is the first implementation gate; it is not an accepted variance.

## Change Map

| Section | Primary files | Deliverable |
| --- | --- | --- |
| 0 | Current branch and tests | Isolated work and reproducible baseline |
| 1 | `lifecycle_outbox.py`, `lifecycle_application.py` | Explicit manual-review reopening policy |
| 2 | `queue_status_service.py`, `taskwarrior_mutations.py` | Authoritative proof before review resolution |
| 3 | `chain_integrity_engine.py`, `lifecycle_outbox.py` | Lease-safe, bounded integrity recovery |
| 4 | `lifecycle_models.py`, `chain_invariants.py` | Closed, checked postcondition vocabulary |
| 5 | `lifecycle_outbox.py`, lifecycle-outbox documentation | Preserved failure and resolution evidence |
| 6 | `lifecycle_models.py`, `lifecycle_operator_owner.py` | Small correctness polish |
| 7 | Full verification surfaces | Cutover evidence |

---

## 0. Isolate The Work And Capture The Baseline

The current `extended-offline-reliability-v1` worktree contains unrelated
offline-kit edits, including an uncommitted edit to `lifecycle_outbox.py`.
Lifecycle hardening must start from an intentional state so neither body of
work silently absorbs the other.

- [ ] Finish, commit, or safely shelve the offline-kit work before lifecycle
  implementation begins.
- [ ] Create a dedicated lifecycle-hardening branch or isolated git worktree.
- [ ] Record the starting commit and branch in this section.
- [ ] Confirm that `git status --short` contains no unexplained changes.
- [ ] Run the focused unit baseline:

  ```bash
  PYTHONPYCACHEPREFIX=/tmp/nautical-lifecycle-hardening-pycache \
    python3 -m unittest \
      tests.test_queue_review \
      tests.test_lifecycle_failure_injection \
      tests.test_lifecycle_terminal_plans \
      tests.test_lifecycle_read_service \
      tests.test_operator_failure_matrix \
      tests.test_effect_boundary -v
  ```

- [ ] Run the lifecycle-outbox golden slice and retain its output:

  ```bash
  python3 dev_tools/nautical_golden_tests.py --only lifecycle_outbox --verbose
  ```

- [ ] Run the shared integrity-outbox golden case:

  ```bash
  python3 dev_tools/nautical_golden_tests.py \
    --only shared_outbox_persists_integrity_work_without_lifecycle_claiming \
    --verbose
  ```

Section 0 completion gate:

- [ ] The lifecycle work is isolated from the offline-kit work.
- [ ] Every baseline failure is recorded; no pre-existing failure is silently
  attributed to a hardening change.
- [ ] The active manual-review reopening regression is reproducible before its
  fix.

---

## 1. Make Manual-Review Reopening Explicit And Fail-Closed

### Required behavior

Use the following policy. Do not infer recoverability from matching schedule or
configuration fingerprints alone.

| Stored review cause | Fresh plan semantically identical | Required action |
| --- | --- | --- |
| `mutation_conflict` | yes | Reopen for retry, including when configuration or schedule fingerprints changed |
| `mutation_rejected` | yes | Reopen for retry, including when configuration or schedule fingerprints changed |
| explicit `plan_environment_drift` | yes | Reopen for retry after a fresh equivalent replan |
| `invalid_intent` | any | Keep in manual review |
| guard, identity, compensation, poison, or unknown failure | any | Keep in manual review |
| any cause | no | Return conflict and preserve the existing record |

Implementation tasks:

- [ ] In `tests/test_lifecycle_failure_injection.py`, replace the broad
  configuration-drift test that uses `invalid_intent: guard modified changed`
  with an explicit `plan_environment_drift` case.
- [ ] Add `test_manual_review_reopen_policy_matrix` covering every row in the
  table above.
- [ ] Assert that an `invalid_intent` caused by a changed guard remains
  `manual_review` even when plan and schedule fingerprints match.
- [ ] Assert that the existing `mutation_conflict` case reopens when both
  configuration and schedule fingerprints change.
- [ ] Run the focused test and confirm the new assertions fail against the
  pre-fix behavior:

  ```bash
  PYTHONPYCACHEPREFIX=/tmp/nautical-lifecycle-hardening-pycache \
    python3 -m unittest \
      tests.test_lifecycle_failure_injection.LifecycleFailureInjectionTests.test_manual_review_reopen_policy_matrix \
      -v
  ```

- [ ] In `nautical_core/lifecycle_application.py`, persist configuration,
  schedule, or combined mismatch with the single specific code
  `plan_environment_drift`. Record which fingerprint changed in bounded
  evidence; do not use generic `invalid_intent` for these cases.
- [ ] In `nautical_core/lifecycle_outbox.py`, implement one private reopening
  predicate that accepts only the three recoverable causes in the table and
  requires `same_plan`.
- [ ] Do not make equality of configuration or schedule fingerprints an
  independent reason to reopen manual review.
- [ ] Ensure non-recoverable review records retain their original state,
  stage, lease fields, failure, and timestamps after a duplicate enqueue.
- [ ] Run the focused unit file:

  ```bash
  PYTHONPYCACHEPREFIX=/tmp/nautical-lifecycle-hardening-pycache \
    python3 -m unittest tests.test_lifecycle_failure_injection -v
  ```

- [ ] Run the lifecycle-outbox golden slice:

  ```bash
  python3 dev_tools/nautical_golden_tests.py --only lifecycle_outbox --verbose
  ```

Section 1 completion gate:

- [ ] The focused lifecycle-outbox golden slice passes 7/7 or its new total
  with zero failures.
- [ ] Generic `invalid_intent` and guard-change failures cannot reopen
  automatically.
- [ ] Known stale mutation failures and explicitly classified drift recover
  after an equivalent fresh replan.
- [ ] Reopening never executes a plan whose compatibility key changed.
- [ ] Commit this section independently with a message such as
  `Harden lifecycle manual-review reopening`.

---

## 2. Require Authoritative Proof Before Resolving Manual Review

`queue-review --resolve-applied` currently treats a child UUID lookup plus a
matching parent `nextLink` as high-confidence proof. The lifecycle mutation
gateway already uses a stronger child predicate that also checks `chainID`,
link, `prevLink`, status, active chain state, and recurrence mode.

### Internal interfaces

- [ ] Add an exact, read-only repository operation such as
  `LifecycleOutboxRepository.read_intent(intent_id: str) -> OutboxResult` that
  returns the validated typed `LifecycleOutboxRecord` in `result.record`.
- [ ] The exact read must return `CONFLICT` for an absent ID, `REJECTED` for a
  poisoned payload, and must never claim or modify the row.
- [ ] Extract the existing child identity predicate from
  `nautical_core/taskwarrior_mutations.py` into a shared pure lifecycle
  postcondition helper, or expose it there under a non-private name. Both the
  mutation gateway and queue review must call the same implementation.
- [ ] Provide a pure spawn assessment that consumes a typed `LifecyclePlan`,
  one parent observation, and one child observation and returns a typed match
  or mismatch with bounded reasons.

The authoritative spawn assessment must prove all of the following:

- [ ] The child UUID exactly matches the plan.
- [ ] The child `chainID` equals the plan chain.
- [ ] The child link equals the plan target link.
- [ ] The child `prevLink` equals the parent UUID prefix.
- [ ] The child's `prevLink` and parent's `nextLink` prefixes each resolve
  uniquely to the expected full UUID; a prefix collision is insufficient
  evidence.
- [ ] The child status is a valid imported occurrence.
- [ ] The child remains in the expected recurrence mode (`cp`, `anchor`, or
  `anchor_file`, including `anchor_mode`).
- [ ] The parent UUID, status, chain state, `chainID`, link, and recurrence
  fingerprint still match the immutable plan guard, excluding only the
  `modified` timestamp that parent linking legitimately changes.
- [ ] The parent `nextLink` exactly matches the expected child UUID prefix.
- [ ] Every unavailable, malformed, absent, ambiguous, or mismatched read
  produces `needs_review`; it never produces `already_applied`.

### Red/green coverage

- [ ] Replace the minimal high-confidence fixture in
  `tests/test_queue_review.py` with complete parent, child, guard, and child
  payload evidence.
- [ ] Add one negative test for each of these child changes: wrong UUID, wrong
  `chainID`, wrong link, wrong `prevLink`, inactive chain, invalid status, and
  different recurrence mode.
- [ ] Add a negative test for a colliding UUID prefix on each side of the
  reciprocal link.
- [ ] Add one negative test for each of these parent changes: wrong chain,
  wrong link, changed recurrence fingerprint, and different `nextLink`.
- [ ] Add tests for child export failure, malformed child JSON, absent child,
  parent export failure, malformed parent JSON, and absent parent.
- [ ] In every negative case, assert that `--resolve-applied` leaves the durable
  record in manual review.
- [ ] Add a positive test proving a fully matching parent and child resolve
  exactly one requested intent.
- [ ] Add a test proving `--all --resolve-applied` resolves matching intents
  independently and leaves every ambiguous intent untouched.
- [ ] Run the tests before implementation and confirm the incomplete-child
  fixture fails closed only after the intended code change:

  ```bash
  PYTHONPYCACHEPREFIX=/tmp/nautical-lifecycle-hardening-pycache \
    python3 -m unittest tests.test_queue_review -v
  ```

### Resolution boundary

- [ ] In `nautical_core/queue_status_service.py`, use the typed exact record;
  do not reconstruct authority from the abbreviated status payload.
- [ ] Add a single service operation for resolution that performs a fresh
  exact assessment immediately before calling `resolve_manual_review`.
- [ ] In `nautical_core/tools/nautical_queue_review.py`, call that service
  operation rather than trusting an earlier rendered assessment object.
- [ ] Never rely on a prior listing assessment. Re-read immediately at the
  resolution boundary and store the observed identity evidence. Taskwarrior
  and SQLite cannot share a transaction, so do not claim atomicity across
  them; the acknowledged-intent integrity audit remains the backstop for a
  later concurrent edit.
- [ ] Preserve existing bounded limits for `--all` and exact intent selection.
- [ ] Preserve JSON-only output and `ensure_ascii=False`.
- [ ] Update `docs/tools/lifecycle-outbox.md` to enumerate the evidence required
  for high-confidence resolution.

Section 2 completion gate:

- [ ] Queue review and the mutation gateway share the same child identity
  predicate.
- [ ] Only a complete reciprocal parent/child postcondition can be resolved.
- [ ] No unavailable or mismatched case changes outbox state.
- [ ] `python3 -m unittest tests.test_queue_review -v` passes.
- [ ] The lifecycle-outbox golden slice passes.
- [ ] Commit this section independently with a message such as
  `Require authoritative lifecycle review proof`.

---

## 3. Bring Integrity Outbox Work Up To Lifecycle Lease Standards

Integrity work shares the lifecycle outbox table but currently has weaker
execution rules: leases are not renewed while operations run, expired leases
can transition state, retryable mutation outcomes become manual review, and a
failed durable acknowledgement is ignored.

### Fixed retry policy

- [ ] Define one fixed integrity retry ceiling of three claims. Keep it local
  to the integrity outbox path; do not introduce user configuration.
- [ ] On claim, quarantine an integrity row whose durable `attempts` already
  reached the ceiling.
- [ ] Persist `retry_exhausted` evidence containing the last failure message.
- [ ] Keep poisoned-envelope quarantine distinct from retry exhaustion.

### Lease compare-and-set behavior

- [ ] Add `renew_integrity_lease(intent_id, owner, lease_seconds)` to
  `LifecycleOutboxRepository`.
- [ ] Require state `CLAIMED`, exact owner, and `lease_expires_at > now` for
  renewal.
- [ ] Update `_integrity_transition` to reject expired leases before changing
  state.
- [ ] Add `lease_expires_at > now` to the transition SQL `WHERE` clause and
  require `rowcount == 1` before returning `APPLIED`.
- [ ] Return `CONFLICT` when ownership or expiry changed during the
  compare-and-set.
- [ ] Keep acknowledgement idempotent: a repeated acknowledgement of an
  already acknowledged intent returns `ALREADY_APPLIED`.

### Drain behavior

- [ ] Add an optional pre-operation heartbeat callback to
  `IntegrityApplicationService.apply`; it must run before every integrity
  operation and stop the plan if ownership cannot be renewed.
- [ ] Have `ChainIntegrityEngine.drain` renew the exact intent before each
  operation and again before acknowledgement.
- [ ] When an application result is `RETRYABLE`, call
  `release_integrity_retry`; do not convert the transient failure directly to
  manual review.
- [ ] Keep `CONFLICT`, `REJECTED`, and `MANUAL_REVIEW` outcomes in durable
  manual review with their original reason.
- [ ] Inspect the result of `acknowledge_integrity`.
- [ ] If acknowledgement returns `RETRYABLE` or raises, append an explicit
  retryable application result and do not report the intent as durably
  acknowledged.
- [ ] If acknowledgement returns an ownership conflict, report non-success and
  leave the row recoverable by the current owner or stale-lease recovery.
- [ ] Never repeat an already verified Taskwarrior effect merely to repair the
  SQLite acknowledgement; rely on guarded idempotent postcondition reads.

### Failure-injection coverage

- [ ] Create `tests/test_integrity_outbox_failure_injection.py`.
- [ ] Add `test_integrity_transition_rejects_expired_lease`.
- [ ] Add `test_stale_integrity_worker_cannot_ack_after_reclaim` using two
  distinct owners and an injected clock.
- [ ] Add `test_integrity_lease_renews_between_operations` with a two-operation
  plan.
- [ ] Add `test_integrity_heartbeat_failure_stops_before_next_mutation`.
- [ ] Add `test_integrity_retryable_mutation_returns_to_retry_state`.
- [ ] Add `test_integrity_retry_exhaustion_quarantines_without_mutation`.
- [ ] Add `test_integrity_ack_lock_never_reports_durable_success`.
- [ ] Add `test_integrity_restart_verifies_effect_before_acknowledging`.
- [ ] Run the new suite before implementation and confirm its lease, retry, and
  acknowledgement assertions fail for the expected reasons:

  ```bash
  PYTHONPYCACHEPREFIX=/tmp/nautical-lifecycle-hardening-pycache \
    python3 -m unittest tests.test_integrity_outbox_failure_injection -v
  ```

- [ ] Run the new suite again after implementation and require zero failures.
- [ ] Run the existing shared-outbox golden case:

  ```bash
  python3 dev_tools/nautical_golden_tests.py \
    --only shared_outbox_persists_integrity_work_without_lifecycle_claiming \
    --verbose
  ```

Section 3 completion gate:

- [ ] Integrity work cannot mutate or acknowledge after losing its lease.
- [ ] Temporary failures remain automatically recoverable within a bounded
  retry budget.
- [ ] Exhaustion is quarantined with evidence rather than retried forever.
- [ ] A Taskwarrior effect followed by SQLite acknowledgement failure is
  reported as retryable, then converges by verification without duplication.
- [ ] Commit this section independently with a message such as
  `Harden integrity outbox leases and retries`.

---

## 4. Close And Verify The Lifecycle Postcondition Vocabulary

`LifecyclePlan.expected_postconditions` currently accepts arbitrary strings.
The acknowledged-intent integrity rule initializes every condition as
satisfied and checks only a subset, so unknown values and several planner
values can pass silently.

### Contract

- [ ] Add a `LifecyclePostcondition` string enum to
  `nautical_core/lifecycle_models.py` containing the supported external
  conditions:
  `child_present`, `parent_linked`, `parent_chain_on`, `parent_chain_off`,
  `terminal_chain`, and `no_successor`.
- [ ] Normalize the legacy aliases `child_exists` to `child_present` and
  `chain_off` to `parent_chain_off` during supported v1 deserialization.
- [ ] Remove `verified` from newly constructed plans because it is an execution
  stage, not an independently observable Taskwarrior postcondition.
- [ ] During supported v1 deserialization, consume and drop `verified` as a
  documented legacy execution marker after validating the required external
  spawn conditions. Do not retain or re-emit it from new plans.
- [ ] Reject every other unknown postcondition during plan construction and
  deserialization.
- [ ] Validate required conditions by action:
  - `SPAWN_CHILD`: `child_present` and `parent_linked`.
  - activation/resume `UPDATE_PARENT`: `parent_chain_on`.
  - `DISABLE_CHAIN`: `parent_chain_off`.
  - `FINALIZE_CHAIN`: `terminal_chain` and `no_successor`.
  - `NOOP`: no external postconditions.
- [ ] Keep the serialized values stable strings and preserve
  `ensure_ascii=False`.

### Integrity evaluation

- [ ] Rewrite `_acknowledged_postcondition_rule` so it begins fail-closed for
  each condition; no default `satisfied = True` path may remain.
- [ ] Verify every enum member against the authoritative graph.
- [ ] Emit a manual-review finding for an unsupported persisted value rather
  than treating it as satisfied.
- [ ] Keep terminal checks consistent with `_finalization_rule`; share a pure
  predicate if that avoids two definitions of terminal state.

### Red/green coverage

- [ ] Add construction tests for all valid action-specific sets.
- [ ] Add construction and deserialization tests rejecting an unknown string.
- [ ] Add alias-normalization tests for `child_exists` and `chain_off`.
- [ ] Add a planner test proving new spawn plans omit `verified`.
- [ ] Add acknowledged-intent integrity tests for every supported condition in
  both matching and mismatching states.
- [ ] Add a corrupted-row test proving an unknown condition is quarantined or
  reported for manual review, never accepted.
- [ ] Run the lifecycle model and integrity slices identified by:

  ```bash
  python3 dev_tools/nautical_golden_tests.py --only lifecycle_plan --verbose
  python3 dev_tools/nautical_golden_tests.py \
    --only chain_integrity_finalization_evidence --verbose
  ```

Section 4 completion gate:

- [ ] No producer writes a free-form lifecycle postcondition.
- [ ] No unknown or unimplemented postcondition can pass integrity evaluation.
- [ ] Every lifecycle action carries the postconditions needed to prove its
  result.
- [ ] Existing retained v1 evidence remains readable during its retention
  period.
- [ ] Commit this section independently with a message such as
  `Validate lifecycle postcondition contracts`.

---

## 5. Preserve Failure Evidence When Review Is Resolved

The operator-resolution path currently replaces the original failure with an
`operator_resolved` failure. Preserve the original cause without adding an
event-sourcing subsystem or an unbounded history.

- [ ] Add a unit test that creates manual review with structured Unicode
  evidence, resolves it, reloads the row, and proves the original code,
  message, and evidence remain available.
- [ ] Add a repeated-resolution test proving the second call is idempotent and
  does not rewrite evidence.
- [ ] Add a bounded-evidence test proving resolution stores exactly one
  `original_failure` object, never a recursive history or nested resolution.
- [ ] Before implementation, run those tests and confirm the original failure
  is currently replaced.
- [ ] Keep `OutboxFailure("operator_resolved", reason)` as the top-level
  resolution result, but place the complete prior failure under a single
  bounded `evidence["original_failure"]` object.
- [ ] Include the operator reason and acknowledgement timestamp without
  copying task payloads or command output into the evidence.
- [ ] Ensure `OutboxFailure.to_json()` continues using `ensure_ascii=False`.
- [ ] Return `REJECTED` without changing state if the existing failure cannot
  be decoded or bounded resolution evidence cannot be encoded.
- [ ] Update `docs/tools/lifecycle-outbox.md` so “immutable audit evidence”
  precisely means that the original review failure survives resolution.
- [ ] Run the focused outbox and queue-review tests.

Section 5 completion gate:

- [ ] An acknowledged operator resolution retains its original failure cause.
- [ ] Resolution evidence is deterministic, bounded, Unicode-safe, and
  idempotent.
- [ ] No task payload or unrestricted stderr is copied into the outbox.
- [ ] Commit this section independently with a message such as
  `Preserve lifecycle review evidence`.

---

## 6. Apply Small Lifecycle Correctness Polish

### Preserve immutable plan fields during stage changes

- [ ] Add `test_with_stage_preserves_every_non_stage_field` using a plan with a
  non-empty `terminal_kind`.
- [ ] Assert that identity, action, parent guard, child payload, parent patch,
  postconditions, `max_attempts`, and `terminal_kind` are unchanged.
- [ ] Replace the hand-copied constructor in `LifecyclePlan.with_stage` with
  `dataclasses.replace(self, stage=ExecutionStage(stage))`, or explicitly copy
  `terminal_kind` if project compatibility requires the existing style.
- [ ] Run lifecycle terminal-plan and model tests.

### Retire the unsafe dormant operator owner

`LifecycleOperatorOwner` is not imported by a current production path. Its
protocol omits the required drain limit, and its FIFO `outcomes[-1]` selection
can attribute an unrelated intent's outcome to the requested plan.

- [ ] Reconfirm production usage with:

  ```bash
  rg -n "LifecycleOperatorOwner|LifecycleApplicationPort" . \
    --glob '!completed-checklists/**' \
    --glob '!checklists/**'
  ```

- [ ] If the result still contains only
  `nautical_core/lifecycle_operator_owner.py`, remove the dormant module and any
  package export; do not repair an unused abstraction.
- [ ] If a production import now exists, replace FIFO drain ownership with
  `execute_staged(plan, configuration_fingerprint=...,
  schedule_fingerprint=...)` and add a regression proving unrelated ready
  intents remain untouched.
- [ ] Run operator control-plane, exact-claim, and reconcile-wave tests after
  the selected path.

Section 6 completion gate:

- [ ] `with_stage` cannot silently discard present or future immutable fields.
- [ ] No callable owner can stage one intent and report the result of another.
- [ ] No production behavior or hook output changes from this polish.
- [ ] Commit this section independently with a message such as
  `Polish lifecycle immutable transitions`.

---

## 7. Full Verification And Cutover Evidence

### Static and focused checks

- [ ] Compile the affected Python modules:

  ```bash
  PYTHONPYCACHEPREFIX=/tmp/nautical-lifecycle-hardening-pycache \
    python3 -m py_compile \
      nautical_core/lifecycle_models.py \
      nautical_core/lifecycle_outbox.py \
      nautical_core/lifecycle_application.py \
      nautical_core/queue_status_service.py \
      nautical_core/taskwarrior_mutations.py \
      nautical_core/chain_integrity_engine.py \
      nautical_core/chain_integrity_application.py \
      nautical_core/chain_invariants.py \
      nautical_core/tools/nautical_queue_review.py
  ```

- [ ] Run the focused reliability suites:

  ```bash
  PYTHONPYCACHEPREFIX=/tmp/nautical-lifecycle-hardening-pycache \
    python3 -m unittest \
      tests.test_queue_review \
      tests.test_lifecycle_failure_injection \
      tests.test_integrity_outbox_failure_injection \
      tests.test_lifecycle_terminal_plans \
      tests.test_lifecycle_read_service \
      tests.test_operator_failure_matrix \
      tests.test_effect_boundary -v
  ```

- [ ] Run the complete golden suite:

  ```bash
  python3 dev_tools/nautical_golden_tests.py --strict-lifecycle-warnings
  ```

- [ ] Run it once with deterministic shuffle:

  ```bash
  python3 dev_tools/nautical_golden_tests.py \
    --strict-lifecycle-warnings --shuffle-seed 20260903
  ```

- [ ] Run mypy with the repository configuration:

  ```bash
  python3 -m mypy --config-file mypy.ini
  ```

- [ ] Run whitespace validation:

  ```bash
  git diff --check
  ```

### Hook and deployment checks

- [ ] Run the black-box suite with Taskwarrior available:

  ```bash
  python3 dev_tools/nautical_black_box_test.py --json
  ```

- [ ] Run deployment sanity:

  ```bash
  python3 dev_tools/nautical_deploy_sanity.py
  ```

- [ ] Run the enforced desktop performance budget and confirm the hardening did
  not add Taskwarrior calls to the normal lifecycle drain:

  ```bash
  python3 dev_tools/nautical_perf_budget.py --json --enforce
  ```

- [ ] Confirm valid, malformed, retryable, and rejected hook inputs each emit
  exactly one JSON document on stdout.
- [ ] Confirm non-diagnostic hook stderr remains empty.
- [ ] Confirm diagnostic text appears only on stderr when `NAUTICAL_DIAG=1`.
- [ ] Confirm Unicode hook fields remain unescaped.

### Isolated operational smoke test

- [ ] Create disposable Taskdata and exercise one complete recurrence through
  child import, parent linking, verification, and acknowledgement.
- [ ] Interrupt one run after child import and prove the next run resumes
  without creating another child.
- [ ] Produce one manual-review conflict and prove queue review refuses it when
  child or parent evidence differs.
- [ ] Produce one already-applied manual-review case and prove exact resolution
  retains the original failure evidence.
- [ ] Expire an integrity lease, reclaim it with a new owner, and prove the old
  owner cannot mutate or acknowledge it.
- [ ] Inject an acknowledgement lock after a successful effect and prove the
  next run verifies and acknowledges without duplicating the effect.
- [ ] Run the read-only operational commands against the disposable Taskdata:

  ```bash
  nautical doctor --deep --json
  nautical queue-status --json
  nautical query integrity --all
  nautical reconcile --dry-run --json --no-housekeeping
  ```

- [ ] Review the scoped plan, then run `nautical reconcile --apply --json`
  against only the disposable Taskdata and repeat all four read-only commands.

Final completion gate:

- [ ] All focused and full verification commands pass with zero failures.
- [ ] The lifecycle-outbox regression from the audit is closed rather than
  accepted as a variance.
- [ ] Manual-review closure requires complete authoritative evidence.
- [ ] Lifecycle and integrity work enforce equivalent lease ownership and
  acknowledgement truthfulness.
- [ ] Unknown postconditions fail closed.
- [ ] Original review failures survive operator resolution.
- [ ] Clean and recovery lifecycle subprocess-count budgets are unchanged.
- [ ] `git diff --check` is clean and `git status --short` contains only the
  intended lifecycle-hardening changes.
- [ ] Record the final golden count, mypy file count, performance result, and
  operational smoke evidence in this checklist before merge.

## Deliberate Deferrals

The following changes are intentionally outside this checklist because they do
not improve lifecycle safety enough to justify their complexity:

- [ ] Do not move single guarded terminal/update mutations into the durable
  saga solely for architectural uniformity.
- [ ] Do not redesign persisted leases around monotonic clocks. Use existing
  suspicious-clock Doctor/offline-readiness checks unless a reproducible
  lifecycle failure demonstrates the need.
- [ ] Do not split large lifecycle modules by line count alone. Extract only a
  shared validator required by the hardening above.
- [ ] Do not add an append-only event store. Preserve the one original failure
  inside bounded resolution evidence.
- [ ] Do not add directory-fsync behavior to the lifecycle path as part of this
  work; cover filesystem durability in the offline-kit fault campaign.
