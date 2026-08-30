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
- [ ] All applicable final completion and cutover criteria in the original
  remediation checklist are checked with current evidence. It currently has
  51 unchecked entries, including every final completion criterion.
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

- [ ] Start from the exact green revision recorded at Gate A.
- [ ] Create a dedicated branch or worktree, for example
  `performance-improvement-v1`.
- [ ] Record the branch name, starting revision, Python version, Taskwarrior
  version, dependency versions, machine profile, and budget file revision.
- [ ] Keep `main` as the known-good rollback point.
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

- [ ] Every command above passes before measuring performance.
- [ ] No generated cache, database, report, or test fixture is accidentally
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

- [ ] Store the normal and extended JSON reports with the starting revision and
  environment record.
- [ ] Confirm all correctness guards inside the performance workloads pass.
- [ ] Confirm the enforced budget file is unchanged during baseline capture.
- [ ] Repeat noisy workloads until medians are stable enough to distinguish a
  real change from timing noise.
- [ ] Record wall time, CPU time, Taskwarrior calls and duration, exported and
  decoded rows, SQLite connections and transactions, scheduler iterations,
  startup/import time, drain time, presentation time, and peak memory wherever
  the workload exposes them.
- [ ] Mark unavailable telemetry explicitly; do not interpret missing metrics
  as zero work.

### 2.4 Select one optimization target

- [ ] Identify the slow workload and the responsible stage from measurements,
  not intuition.
- [ ] State one falsifiable hypothesis explaining the cost.
- [ ] Choose one primary success metric and list all safety/correctness metrics
  that must remain unchanged.
- [ ] Define the minimum improvement worth accepting, accounting for measured
  noise.
- [ ] Define independent non-regression limits for calls, rows, memory, SQLite
  work, scheduler iterations, and wall time.
- [ ] Identify the files and ownership boundary involved.
- [ ] Confirm the proposed optimization does not require weakening a guard,
  durability boundary, output contract, or completeness claim.

## Gate B: Ready To Optimize

- [ ] Gate A remains green on the isolated branch.
- [ ] Normal, extended, stress, and short-soak baselines are recorded.
- [ ] The selected target is reproducibly slow.
- [ ] The root-cause hypothesis names the stage and cost source.
- [ ] Acceptance and non-regression thresholds are written before code changes.
- [ ] The planned edit is bounded to one bottleneck and one ownership boundary.

## 3. Apply One Performance Improvement

- [ ] Add or identify the smallest correctness test protecting the optimized
  behavior.
- [ ] Add or identify the benchmark workload that exposes the measured cost.
- [ ] Run both before the change and save their results.
- [ ] Implement the minimum change needed to test the performance hypothesis.
- [ ] Avoid new caches unless ownership, invalidation, memory bounds, and
  configuration/mutation-epoch safety are explicit and tested.
- [ ] Avoid concurrency around Taskwarrior mutation, lifecycle application, and
  SQLite transaction boundaries.
- [ ] Do not replace complete evidence with a partial result presented as
  complete.
- [ ] Do not remove retries, validation, guards, postcondition checks, or
  idempotence to improve timing.
- [ ] Run the focused correctness test immediately after the change.
- [ ] Run the target benchmark under the same environment and parameters.
- [ ] Compare base and candidate reports with the repository comparison tool.
- [ ] Confirm the primary metric improves beyond the predeclared noise floor.
- [ ] Confirm every independent metric stays within its non-regression limit.
- [ ] Revert or revise the change if the hypothesis is disproved; do not hide a
  regression by raising budgets.
- [ ] Commit the bounded optimization with its regression tests and relevant
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
