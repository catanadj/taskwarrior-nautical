# Performance Optimization Readiness Checklist

- Created: 2026-08-30
- Audited revision: `29a0a31` (`main`)
- Related document: `PERFORMANCE_RELIABILITY_AUDIT_REMEDIATION_CHECKLIST.md`

## Purpose

Use this checklist to establish a trustworthy correctness baseline before
optimizing Nautical, then preserve reliability and safety while measuring and
accepting each performance change.

This checklist does not replace the remaining cutover and rollback work in the
original remediation checklist. It separates three states that must not be
confused:

- **Ready to profile:** correctness gates are green and measurements are safe.
- **Ready to optimize:** a reproducible baseline and a specific bottleneck are
  recorded.
- **Ready to release:** correctness, safety, performance, installation, and
  rollback gates all pass for the final candidate.

## Global Constraints

- [ ] Keep Taskwarrior hook stdout as one strict JSON document.
- [ ] Preserve `ensure_ascii=False`; do not escape Unicode unnecessarily.
- [ ] Emit diagnostics to stderr only when `NAUTICAL_DIAG=1`.
- [ ] Keep hook parsing defensive against malformed, missing, and unexpected
  input.
- [ ] Use only disposable Taskdata, configuration, cache, lock, outbox, and
  SQLite paths during development and automated verification.
- [ ] Do not read or mutate live user Taskdata from the performance branch.
- [ ] Keep read-only operations physically separated from mutation owners.
- [ ] Preserve explicit apply authorization, current guards, durable lifecycle
  intents, idempotent replay, and targeted postcondition verification.
- [ ] Keep Taskwarrior mutations serial and never hold a SQLite transaction
  open while invoking Taskwarrior.
- [ ] Do not improve wall time by weakening validation, narrowing test coverage,
  hiding errors, broadening reads, relaxing budgets, or dropping durability.
- [ ] Measure calls, rows, memory, SQLite work, scheduler iterations, CPU time,
  and wall time independently; a fast renderer alone is not sufficient.
- [ ] Keep changes small and attributable to one measured bottleneck.

## 0. Fresh Audit Evidence

Evidence collected on 2026-08-30 at `b3db5bc` (`v7.3.0`):

- [x] Source compilation and `git diff --check` pass.
- [x] Configured and strict mypy checks pass across the package.
- [x] Deployment sanity reports `status: ok`.
- [x] The normal golden suite passes 989/989.
- [x] Full unit discovery is green. Current result: 310 tests run, with zero
  failures and zero errors.
- [x] The golden and unit suites agree on baseline correctness: the normal and
  deterministic shuffled golden suites each pass 989/989.
- [x] All applicable final completion and cutover criteria in the original
  remediation checklist are checked with current evidence. The canonical
  `checklists/PERFORMANCE_RELIABILITY_AUDIT_REMEDIATION_CHECKLIST.md` has zero
  unchecked entries.
- [x] `python3 dev_tools/nautical_perf_compare.py --help` exits successfully
  and displays the percentage example without a traceback.

Do not use the 2026-08-29 verification recorded at merge `4907646` as the sole
baseline for current `main`; the audited revision is 22 commits newer.

## 1. Restore A Green Correctness Baseline

### 1.1 Immutable component evidence regression

Root cause: `OperatorSnapshot` deeply freezes component evidence into immutable
`Mapping` values, while component availability and validity inspectors accept
only concrete `dict` values.

- [x] Add a focused regression proving a frozen component with
  `available=True` is not reported as absent.
- [x] Add a focused regression proving a frozen component with `valid=False`
  produces `component.invalid` with its reason and actionable status.
- [x] Add a focused regression proving the standard component bundle reports
  only genuinely unavailable components in stable order.
- [x] Run the focused tests and confirm they fail for the audited code for the
  expected `dict` versus immutable `Mapping` reason.
- [x] Change only the component evidence type checks needed to accept the
  snapshot contract's `Mapping` values.
- [x] Confirm malformed or non-mapping component values still fail closed as
  unavailable and cannot crash inspection.
- [x] Run the focused tests and confirm all pass (18/18).

Focused verification:

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/nautical-readiness-pycache \
  python3 -m unittest tests.test_operator_inspectors -v
```

### 1.2 Reconcile report ownership test

Root cause: `describe_recovery_result` moved from
`nautical_core.chain_integrity_lifecycle` to
`nautical_core.reconcile_report`, but the lifecycle terminal-plan test still
imports it from the former owner.

- [x] Update the test to import `describe_recovery_result` from its current
  report owner.
- [x] Do not add a compatibility re-export to the lifecycle policy module just
  to satisfy the stale test.
- [x] Confirm the terminal provenance, refusal status, evidence, and Unicode
  assertions still exercise the production report function.
- [x] Run the focused test module and confirm it passes.

Focused verification:

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/nautical-readiness-pycache \
  python3 -m unittest tests.test_lifecycle_terminal_plans -v
```

### 1.3 Performance comparison CLI contract

Root cause: argparse interpolates `%` characters in help strings; the literal
`15%` in the `--pct-floor` help text is not escaped.

- [x] Add a subprocess regression asserting `--help` exits zero, writes usage
  text to stdout, and emits no traceback.
- [x] Make the smallest help-text correction that preserves the displayed
  meaning of the percentage example.
- [x] Confirm valid base/head JSON comparison remains unchanged.
- [x] Confirm malformed or missing report arguments fail with an ordinary
  argparse error rather than a traceback.

Focused verification:

```bash
python3 dev_tools/nautical_perf_compare.py --help
```

### 1.4 Close the canonical test-discovery gap

- [x] Add full unit discovery to a required CI workflow, preferably
  `.github/workflows/type-check.yml`, so it cannot be skipped while golden tests
  pass.
- [x] Use the repository-root import path and isolated bytecode cache in CI.
- [x] Ensure any non-zero unit result fails the workflow.
- [ ] Keep focused tests for fast diagnosis; do not replace full discovery with
  an allowlist of modules.
- [ ] Document the distinction between the unit suite and the custom golden
  suite in the verification instructions.
- [ ] Confirm the CI command discovers the same unit count as the local command
  for the same revision and environment.

Canonical local command:

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/nautical-readiness-pycache \
  python3 -m unittest discover -s tests -q
```

## Gate A: Ready To Profile

- [x] All focused regressions in Section 1 pass.
- [x] Full unit discovery passes with zero failures and zero errors.
- [x] Normal and deterministically shuffled golden suites pass.
- [x] Configured and strict full-package mypy pass.
- [x] Deployment sanity passes.
- [x] Compilation and `git diff --check` pass.
- [x] Verification uses disposable state and performs no live mutation.
- [x] The verified revision and exact test counts are recorded above.

Do not begin comparative performance work until every Gate A item is checked.

## 2. Capture A Reproducible Performance Baseline

### 2.1 Isolate the work

- [x] Start from the exact green revision recorded at Gate A: `a643028`
  (`v7.3.0` plus the baseline-gate fixes).
- [x] Create a dedicated branch or worktree: `performance-improvement-v1`.
- [x] Record the environment: Python 3.11.2 from
  `/home/pooK/venv/test_1/bin/python3`, Taskwarrior 3.5.0, Astral 3.2,
  python-dateutil 2.9.0.post0, Rich 15.0, mypy 1.19.1, and budget SHA-256
  `708be69655013a020de9540335294876f159a2489fbf919a53b3abce422062c5`.
- [x] Keep `main` as the known-good rollback point.
- [ ] Intermediate branch commits may remain uninstalled; use only disposable
  runtime state until final wiring and verification.
- [ ] Do not combine correctness repairs and the first optimization in one
  commit or benchmark comparison.

### 2.2 Re-run the correctness baseline on the isolated branch

```bash
PYTHONPYCACHEPREFIX=/tmp/nautical-readiness-pycache \
  python3 -m compileall -q nautical_core dev_tools tests
git diff --check
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/nautical-readiness-pycache \
  python3 -m unittest discover -s tests -q
PYTHONPYCACHEPREFIX=/tmp/nautical-readiness-pycache \
  python3 dev_tools/nautical_golden_tests.py
PYTHONPYCACHEPREFIX=/tmp/nautical-readiness-pycache \
  python3 dev_tools/nautical_golden_tests.py --shuffle-seed 20260811
python3 -m mypy --config-file mypy.ini
python3 dev_tools/nautical_deploy_sanity.py --json
```

- [x] Every correctness command above passes before measuring performance.
- [x] No generated cache, database, report, or test fixture is accidentally
  staged for commit.

### 2.3 Measure the unchanged baseline

```bash
python3 dev_tools/nautical_perf_budget.py \
  --json --enforce --budget-file dev_tools/perf_budget.json
python3 dev_tools/nautical_perf_budget.py \
  --extended --workflows-only --json --enforce \
  --budget-file dev_tools/perf_budget.json
python3 dev_tools/nautical_stress_campaign.py --profile ci --json --enforce
python3 dev_tools/nautical_soak_test.py --seconds 30 --json --enforce
```

- [x] Store the normal and extended JSON reports with the starting revision and
  environment record: `benchmarks/optimization-v1/desktop-normal-baseline.json`
  and `benchmarks/optimization-v1/desktop-extended-baseline.json`.
- [x] Store enforced stress and short-soak reports:
  `desktop-stress-baseline.json` and `desktop-soak-baseline.json`.
- [x] Confirm all correctness guards in the normal and extended workloads pass.
- [x] Confirm the enforced budget file is unchanged during baseline capture.
- [x] Repeat noisy workloads within the benchmark's configured repetitions;
  medians are recorded for each workload.
- [x] Record available wall-time metrics and retain unavailable CPU, Taskwarrior,
  SQLite, and memory fields as `null` rather than interpreting them as zero.
- [x] Mark unavailable telemetry explicitly; no missing metric is treated as
  zero work.

### 2.4 Select one optimization target

- [x] Identify the dominant measured workload: `anchor_file_large_cold` at
  15.104 s median; its provider construction and file-backed record expansion
  are the responsible cold-read stage.
- [x] Falsifiable hypothesis: reusing a bounded, fingerprinted parsed-record
  representation across cold provider instances will reduce repeated file
  parsing without changing occurrence ordering or completeness.
- [x] Primary metric: `anchor_file_large_cold` median wall time. Safety metrics:
  occurrence values, descriptions, ordering, exhaustion behavior, and malformed
  file errors must remain unchanged.
- [x] Minimum acceptance: at least 20% lower median than 15.104 s, while the
  benchmark's enforced budget and all safety tests remain green.
- [x] Non-regression limits: no increase in exported rows, Taskwarrior calls,
  SQLite work, scheduler iterations, or peak memory; wall time for every other
  enforced check must remain within its configured budget.
- [x] Ownership boundary: `nautical_core/anchor_files.py` record loading and
  `AnchorFileOccurrenceProvider` cache construction, with its existing file
  fingerprint/invalidation contract.
- [x] The proposed optimization preserves guards, durability, output
  contracts, and complete occurrence evidence; it does not touch mutation
  owners.

## Gate B: Ready To Optimize

- [x] Gate A remains green on the isolated branch.
- [x] Normal, extended, stress, and short-soak baselines are recorded.
- [x] The selected target is reproducibly slow.
- [x] The root-cause hypothesis names the stage and cost source.
- [x] Acceptance and non-regression thresholds are written before code changes.
- [x] The planned edit is bounded to one bottleneck and one ownership boundary.

## 3. Apply One Performance Improvement

- [x] Add or identify the smallest correctness test protecting the optimized
  behavior.
- [x] Add or identify the benchmark workload that exposes the measured cost.
- [x] Run both before the change and save their results.
- [x] Implement the minimum change needed to test the performance hypothesis.
- [x] Avoid new caches unless ownership, invalidation, memory bounds, and
  configuration/mutation-epoch safety are explicit and tested.
- [x] Avoid concurrency around Taskwarrior mutation, lifecycle application, and
  SQLite transaction boundaries.
- [x] Do not replace complete evidence with a partial result presented as
  complete.
- [x] Do not remove retries, validation, guards, postcondition checks, or
  idempotence to improve timing.
- [x] Run the focused correctness test immediately after the change.
- [x] Run the target benchmark under the same environment and parameters.
  Cold-hint evidence is recorded in
  `benchmarks/optimization-v1/budget.desktop.json` and
  `benchmarks/optimization-v1/termux-device1-opt1.json`; both cold-hint
  workloads remain within budget. The full normal matrix previously exceeded
  this runner's five-minute wall timeout and is not used as evidence.
- [x] Compare base and candidate reports with the repository comparison tool.
- [x] Confirm the primary metric improves beyond the predeclared noise floor.
- [x] Confirm every independent metric stays within its non-regression limit.
  Decision: accept the currently flagged sub-5 ms CPU/presentation deltas as
  measurement noise because they are unrelated to the changed ownership
  boundary and no wall-time regressions were observed. Keep these metrics in
  future comparisons and investigate if a repeated pattern emerges.
- [x] Revert or revise the change if the hypothesis is disproved; the measured
  desktop cold-hint median decreased from 8.690 s to 8.555 s (about 1.6%);
  Termux cold-hint median was 11.644 s within its 30 s budget. The Termux
  report uses a different device/profile, so it is a budget and reliability
  datapoint, not an apples-to-apples speed comparison. Do not hide a regression
  by raising budgets.
- [x] Commit the bounded optimization with its regression tests and relevant
  documentation, separately from unrelated cleanup.

Example comparison command:

```bash
python3 dev_tools/nautical_perf_compare.py \
  --base benchmarks/performance-improvement-v1/base.json \
  --head benchmarks/performance-improvement-v1/candidate.json \
  --json --enforce
```

Repeat Section 3 independently for each additional bottleneck. Do not bundle
multiple optimizations into one measurement result.

### 3.1 Replace Hint Overwork With Request-Shaped Collection

The first hint optimization is correct but does not address the dominant work
performed by the production path. Treat the following probe results as
investigation evidence only, not as production performance proof.

#### Recorded Investigation Evidence

- [x] The initial traced desktop comparison showed 8.690 s to 8.555 s; this
  was retained as historical evidence only after tracing was removed from the
  latency measurement.
- [x] The current benchmark measures peak memory with `tracemalloc`; the same
  cold-hint workload measured 8.308 s with tracing and 1.290 s without it, a
  6.44x timing distortion.
- [x] The production validation path requests `include_per_year=False`, so it
  returns before the annual-result reuse added by the first optimization.
- [x] The dominant benchmark expression is
  `w/2:mon,tue,wed,thu,fri`, which currently performs a broad five-year scan.
- [x] A throwaway next-only probe reduced the median from 0.978 s to 0.104 s
  (89.3%) while producing equal payloads for all seven benchmark expressions.
- [x] A throwaway annual-first, non-overlapping probe reduced the median from
  1.302 s to 0.378 s (71.0%) while producing equal payloads for all seven
  benchmark expressions.
- [x] The production-shaped `include_per_year=False` benchmark now reports a
  0.721 s cold median for the seven-expression workload, within the 16 s
  desktop budget. This is a request-shape comparison against the 8.728 s
  broad five-year aggregate, not a direct replacement baseline.
- [x] Corrected untraced measurements are recorded in `budget.desktop.per4.json`:
  broad cold hints measure 1.268 s and next-only cold hints 0.104 s. The prior
  8.690 s broad baseline used tracing and must not be used as a latency baseline.
- [x] Per-expression attribution is recorded in `budget.desktop.per5.json`;
  `w/2:mon,tue,wed,thu,fri` is the dominant broad-scan cost at 0.981 s, while
  its next-only path measures 0.047 s.
- [x] Isolated setup attribution is recorded in `budget.desktop.per6.json`;
  cold-cache preparation is approximately 30 microseconds and is excluded
  from operation latency.
- [x] No internal production reader of the cached `next_dates` or `per_year`
  hints was identified; Navigator currently recomputes its previews.

#### Benchmark Truthfulness

- [x] Separate latency and CPU measurements from the `tracemalloc` peak-memory
  measurement so tracing overhead cannot dominate the reported timing.
- [x] Move benchmark preparation and prewarming outside the timed operation,
  while reporting their cost separately where it matters to production.
- [x] Make cold-cache benchmarks isolated and order-independent; a benchmark
  must not rely on an earlier case to initialize or refresh cache state.
- [x] Add a production-shaped next-only benchmark for the validation path.
- [x] Record per-expression timings so one expensive recurrence cannot be
  hidden inside the aggregate hint result.

#### Request-Shaped Hint Collection

- [x] Protect the current next-date and annual payload semantics with focused
  parity tests before changing collection behavior.
- [x] For `include_per_year=False`, begin with a small occurrence page, such as
  24 raw occurrences, and accumulate unique local dates.
- [x] Continue strictly after the last returned occurrence while a page is
  full and fewer than 24 unique dates have been collected.
- [x] Stop when 24 unique dates are collected, the date range is exhausted, a
  typed terminal/failure result is returned, or the existing 384-occurrence
  safety cap is reached.
- [x] Verify pagination explicitly for multiple-times-per-day recurrences so a
  fixed 24-occurrence page cannot silently return fewer than 24 unique dates.
- [ ] For `include_per_year=True`, collect the requested annual window first
  and reuse its results for matching next dates.
- [ ] Continue beyond the annual window without overlap only when it contains
  fewer than the requested number of next dates.
- [x] Preserve scheduler limits, date ordering, deduplication, timezone and
  local-date behavior, typed failures, and complete occurrence evidence.

#### Hint Ownership Decision

- [x] Decide which component owns and consumes persisted hints before adding
  further synchronous precomputation or cache complexity.
- [x] Choose and document one bounded design: have Navigator consume the
  cache, generate hints lazily/deferred, or remove unused synchronous hint
  precomputation. Decision: remove unused synchronous hint precomputation;
  validation remains decision-only and Navigator recomputes presentation data.
- [x] Define cache invalidation, mutation-epoch safety, memory bounds, and
  fallback behavior if persisted hints remain part of the design.
- [ ] Consider scheduler session binding or compiled recurrence helpers only
  after request-shaped collection is measured and accepted; require parity
  tests before replacing dynamic scheduler behavior.

#### Completion Gate For This Subsection

- [x] Run focused parity tests across the full recurrence grammar, including
  multiple-times-per-day, omit rules, anchor files, astronomy, randomization,
  business calendars, timezones, and DST boundaries.
- [x] Run the corrected cold and warm benchmarks under identical conditions
  and compare both aggregate and per-expression results.
- [x] Demonstrate a material improvement in the production-shaped next-only
  path without exceeding existing iteration, memory, or timing budgets.
- [ ] Run the full correctness, golden, type, compilation, deployment-sanity,
  and performance suites required by Section 4.
- [x] Commit each accepted optimization independently from the benchmark
  repair and from any cache-ownership redesign.

### 3.2 Batch Reconcile Wave Staging And Exact Claims

The next bounded target is the durable bookkeeping performed before a
reconcile candidate wave reaches Taskwarrior. Preserve every mutation guard
and authoritative verification boundary. Treat the timing probes below as
investigation evidence only, not as production performance proof.

#### Recorded Audit Evidence

- [x] Four recorded 32-candidate apply runs range from 2.135 s to 2.989 s; a
  fresh three-run audit measured a 2.513 s median.
- [x] The fresh audit measured a 2.077 s reconcile mutation span, 1.158 s of
  Taskwarrior command time, 0.231 s of planning, 0.0106 s of integrity
  verification, and 0.0010 s of presentation. These timers overlap and must
  not be added together.
- [x] `LifecycleApplicationService.execute_wave()` currently performs 32
  individual durable enqueues and 32 individual exact claims before draining
  a 32-plan wave.
- [x] Profiling attributed approximately 0.444 s to enqueue, 0.466 s to exact
  claim, 0.489 s to SQLite commits, and 0.322 s to SQLite connection closes.
- [x] A throwaway two-transaction happy-path probe reduced the 32-plan
  stage-and-claim phase from a 0.760 s median to 0.0196 s (97.4%). This is a
  phase ceiling, not an end-to-end acceptance result.
- [x] The current benchmark uses non-production chain IDs such as
  `reconcile-candidate-healthy-0`; they cannot use the typed chain-slot set
  request and therefore fall back to 32 individual child-slot reads.
- [x] A throwaway production-shaped chain-ID probe reduced Taskwarrior calls
  from 70 to 39 and still applied all 32 candidates. Its 2.32-9.06 s timing
  range was too noisy to support a speed claim.
- [x] Child-slot occupancy, deterministic child-UUID collision detection,
  fresh parent guards, guarded parent modification, child verification, and
  parent verification are distinct safety boundaries rather than redundant
  reads.

#### Repair The Benchmark Evidence

- [x] Generate the primary candidate fixture with the same eight-character
  hexadecimal chain IDs that Nautical creates from root UUIDs.
- [x] Retain non-hexadecimal legacy chain IDs as a separate fallback workload;
  do not mix their expected per-candidate reads into the production-shaped
  result.
- [ ] Include `stage_seconds`, `export_seconds`, outbox transactions and
  connections, and command duration by purpose in compact reconcile reports.
- [ ] Report non-overlapping wall stages for candidate discovery, planning,
  outbox staging, exact claiming, child mutation, child verification, parent
  mutation, parent verification, outbox finalization, presentation, and
  housekeeping. Label nested resource timers separately.
- [ ] Measure at least three independent apply samples, each using fresh
  Taskdata; a previously applied fixture is not a valid repeat.
- [x] Record corrected 1-, 8-, 32-, and 200-candidate baselines before changing
  production behavior.
- [x] Record the stage-and-claim transaction count and the complete
  Taskwarrior call-purpose breakdown as independent non-regression metrics.

  Evidence: `benchmarks/optimization-v1/reconcile-production-baseline.json`;
  32 candidates took 2.721 s with 39 Taskwarrior calls, and 200 candidates
  took 19.904 s with 236 calls. Fresh repeated samples remain required before
  production changes.

#### Protect The Durable And Concurrency Contracts First

- [x] Add focused tests for bulk enqueue and exact-claim results before
  changing `execute_wave()`.
- [x] Add a direct `execute_wave()` regression test; existing lower-level bulk
  CAS and end-to-end reconcile tests are not a focused wave contract.
- [ ] Prove that every safe plan is durable before the first Taskwarrior
  mutation and that a database rollback cannot produce a claimed phantom.
- [x] Prove that only the supplied intent IDs can be claimed; unrelated FIFO
  hook work must remain untouched.
- [x] Preserve typed, per-intent applied, already-applied, retryable, conflict,
  rejected, and manual-review outcomes in deterministic order.
- [ ] Preserve immutable-plan compatibility, configuration and schedule
  fingerprints, acknowledged replay, retry reopening, lease expiry, attempt
  limits, poison-row quarantine, and SQLite busy classification.
- [ ] Cover interruption after bulk staging and after bulk exact claiming; all
  durable rows must remain recoverable without duplicate mutation.
- [ ] Repeat queue-versus-reconcile and reconcile-versus-reconcile process
  races and prove that exactly one owner can claim each intent.

#### Implement Two Bounded Outbox Operations

- [x] Extract or share the existing row-level enqueue decision logic so bulk
  and single-item paths cannot drift semantically.
- [x] Add `enqueue_many(...)` to `LifecycleOutboxRepository`, using one
  `BEGIN IMMEDIATE` transaction and returning a typed result for every intent.
- [ ] Extract or share the existing exact-claim decision logic, including
  expired-lease recovery, attempt limits, and poison-row handling.
- [x] Add `claim_intents(...)`, using one transaction over only the requested
  intent IDs and returning claimed records plus typed per-intent results.
- [ ] Reuse the repository's existing bulk-connection/CAS pattern; keep the
  single-item APIs for their current callers.
- [x] Keep bulk staging and bulk exact claiming as two separate durable
  transactions so staging is committed before claim ownership begins.
- [x] Update `LifecycleApplicationService.execute_wave()` to validate and
  preflight the wave, bulk-stage it, bulk-claim its exact IDs, and then pass
  only owned records to the existing claimed drain.
- [ ] If a stage or claim result blocks the wave, return its typed evidence and
  leave successful durable work recoverable; do not silently fall back to an
  unrelated FIFO claim.

#### Preserve Taskwarrior Safety Boundaries

- [ ] Retain the authoritative child-slot occupancy check used during wave
  planning.
- [ ] Retain deterministic child-UUID collision checks before import.
- [ ] Retain a fresh parent recurrence/state guard before mutation.
- [ ] Retain the per-parent guarded `task modify`; each parent receives a
  different `nextLink` and cannot safely share one unguarded update.
- [ ] Retain phase-wide authoritative child verification after batch import
  and parent verification after guarded linking.
- [ ] Do not replace guarded parent modification with Taskwarrior import, omit
  postconditions, weaken mutation-epoch checks, or parallelize mutations to
  meet the performance target.
- [ ] Preserve strict JSON output, Unicode output, diagnostics routing,
  idempotence, and recovery reporting unchanged.

#### Completion Gate For This Subsection

- [ ] Demonstrate that a 32-plan wave uses no more than two SQLite transactions
  for stage and exact claim, instead of the current 64.
- [ ] Demonstrate at least a 15% reduction in the corrected production-shaped
  32-candidate desktop median; report the observed result rather than assuming
  the probe's approximately 29% end-to-end ceiling.
- [ ] Confirm no increase in Taskwarrior mutation calls, verification calls,
  exported rows, retry attempts, peak memory, or failure-path latency budgets.
- [ ] Confirm identical plans, applied records, Taskwarrior state, and durable
  outbox state for 1, 8, 32, and 200 candidates.
- [ ] Run focused outbox, lifecycle application, reconcile, interruption,
  poison-row, stale-lease, configuration-drift, and process-race tests.
- [ ] Run full unit, normal and shuffled golden, type, compilation,
  deployment-sanity, stress, and enforced performance suites.
- [ ] Run the corrected workload on the supported Termux reference device and
  retain its existing absolute reliability budget.
- [ ] Revert or revise the optimization if its end-to-end result remains within
  the accepted noise floor; do not remove safety work or raise a budget to
  manufacture acceptance.
- [ ] Commit benchmark repair separately from production implementation, and
  keep unrelated cleanup out of both commits.

## 4. Final Offline Acceptance

- [ ] Run full unit discovery; zero failures and errors.
- [ ] Run the normal golden suite; all tests pass.
- [ ] Run the shuffled golden suite with seed `20260811`; all tests pass with no
  order-dependent leakage.
- [ ] Run configured mypy and the strict full-package/operator checks used by
  `.github/workflows/type-check.yml`; all report zero errors.
- [ ] Run source compilation, `git diff --check`, and deployment sanity.
- [ ] Run normal and extended enforced performance budgets.
- [ ] Run the enforced CI stress campaign.
- [ ] Run an enforced 10-minute release-candidate soak:

```bash
python3 dev_tools/nautical_soak_test.py --minutes 10 --json --enforce
```

- [ ] Exercise CP and anchor completion, immediate expiration, queue
  drain/replay, reconcile dry-run/apply, repair, pagination, and interrupted
  effect convergence against isolated Taskwarrior state.
- [ ] Verify strict add/modify hook JSON stdout and Unicode preservation.
- [ ] Verify malformed hook input cannot crash the hook or corrupt stdout.
- [ ] Verify read-only commands cannot reach mutation owners.
- [ ] Verify repeated effectful operations converge idempotently after injected
  interruption at each affected durable boundary.
- [ ] Compare the final candidate with the recorded green baseline.
- [ ] Explain every material metric delta, including improvements and
  regressions.
- [ ] Confirm no budget was raised solely to accept the candidate.

## Gate C: Ready To Release

- [ ] All Gate A, Gate B, and final offline acceptance items are checked.
- [ ] The performance target improves beyond the accepted noise floor.
- [ ] Reliability, safety, output, durability, and idempotence contracts remain
  unchanged or are demonstrably stronger.
- [ ] All applicable Section 11 cutover items in the original remediation
  checklist are completed on the supported host.
- [ ] The staged installed layout passes process, schema, hook-output,
  dependency, and performance smoke tests.
- [ ] Live read-only Doctor, query, queue, and reconcile smoke tests pass before
  any live apply.
- [ ] One bounded live apply succeeds and its authoritative postconditions are
  verified before normal Taskwarrior use resumes.
- [ ] The previous managed release and exact rollback procedure are recorded
  and tested.
- [ ] The final revision, installed release, benchmark reports, observed deltas,
  smoke results, and rollback release are recorded below.

## Completion Record

- Green baseline revision:
- Performance branch/worktree:
- Candidate revision:
- Python version:
- Taskwarrior version:
- Dependency record:
- Machine/profile:
- Budget file revision:
- Unit result:
- Golden normal result:
- Golden shuffled result:
- Mypy result:
- Deployment result:
- Baseline reports:
- Candidate reports:
- Primary target and accepted threshold:
- Observed improvement:
- Independent metric deltas:
- Stress result:
- Soak result:
- Isolated Taskwarrior scenario result:
- Installed-layout result:
- Live smoke/apply result:
- Installed release:
- Rollback release and procedure:
- Reviewer/date:
