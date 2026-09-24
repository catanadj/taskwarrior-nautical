# Repository Upgrade and Refactor Checklist

- Created: 2026-09-20
- Audited revision: `4fb60f9` (`main`)
- Source: repository health audit completed on 2026-09-20
- Status: IN PROGRESS — preflight complete

## Purpose

Use this checklist to reduce the repository's concentrated structural debt
without changing recurrence behavior, Taskwarrior integration semantics, or
hook protocol output. Complete work packages in order unless a package states
that it can run independently.

The audit found four primary refactor targets:

1. test architecture and coverage measurement;
2. Taskwarrior hook adapters;
3. lifecycle application and outbox processing;
4. the root compatibility facade and flat package layout.

Strict typing, error consistency, and performance-tool decomposition accompany
those structural changes. They are not independent rewrite projects.

## Global Constraints

- [ ] Keep hook stdout to the exact JSON protocol required by Taskwarrior.
- [ ] Preserve `ensure_ascii=False` for every hook JSON response.
- [ ] Emit diagnostics to stderr only when `NAUTICAL_DIAG=1`.
- [ ] Treat malformed, missing, truncated, and oversized hook input defensively;
  no hook may crash with an uncaught traceback.
- [ ] Preserve ordinary-task fast paths and lazy core loading.
- [ ] Keep mutation-sensitive Taskwarrior reads fail-closed.
- [ ] Preserve lifecycle idempotence, durable intent recovery, lease ownership,
  retry limits, and targeted postcondition verification.
- [ ] Never hold a SQLite transaction open while invoking Taskwarrior.
- [ ] Use disposable Taskdata, configuration, cache, lock, and outbox paths in
  every automated test.
- [ ] Keep public compatibility aliases until an explicit deprecation window
  has elapsed; internal importer counts alone do not prove an API is unused.
- [ ] Make small, behavior-preserving commits. Do not combine independent work
  packages in one change.
- [ ] Do not add dependencies or widen supported-version ranges unless a work
  package demonstrates the need.

## Execution mode

- Work from `feature/upgrade-refactor-preflight`.
- Treat the system as offline: do not fetch dependencies, call external
  services, or rely on network access during implementation or verification.
- Intermediate commits may be non-functional while a refactor is in flight.
- Compatibility bridges between intermediate commits are not required; restore
  the required external behavior and hook contracts at the final acceptance
  gate.

## 0. Baseline and Branch Gate

Audit evidence at `4fb60f9`:

- [x] Full unit discovery passes: 1,256 tests, 3 skipped.
- [x] Golden suite passes: 404/404.
- [x] Configured mypy passes across 244 source files.
- [x] Unit branch coverage is 63%.
- [x] Deployment and architecture sanity report `status: ok`.
- [x] `shellcheck bootstrap.sh` and `bash -n bootstrap.sh` pass.
- [x] Fresh tracked-HEAD scan records objective health at 86.3/100.
- [x] The stale omission-policy and task-status scanner findings were checked
  against `HEAD`; commit `7de5c11` already centralizes those contracts.

Before implementation:

- [x] Create an isolated branch or worktree from the recorded green revision.
- [x] Record Python, Taskwarrior, Astral, mypy, and coverage versions.
- [x] Run the commands below and record exact counts in the first branch commit
  or pull-request description.
- [x] Confirm `git status --short --untracked-files=no` is empty before editing.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 -m unittest discover -s tests -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 dev_tools/nautical_golden_tests.py
/home/pooK/venv/test_1/bin/mypy --config-file mypy.ini
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 dev_tools/nautical_deploy_sanity.py --json
shellcheck bootstrap.sh
bash -n bootstrap.sh
```

**Gate A — ready to refactor:** every baseline command passes, the verified
revision is recorded, and no command uses live Taskdata.

### Preflight record — 2026-09-20

- Audited and tested revision: `4fb60f9` (`main`).
- Environment: Python 3.11.2; Taskwarrior 3.4.2; Astral 1.6.1; mypy 1.19.1;
  coverage.py 6.5.0.
- Unit discovery: 1,256 passed, 3 skipped.
- Golden suite: 404 passed, 0 failed.
- Mypy: 244 source files checked, no issues.
- Deployment sanity: `status: ok`; disposable Taskdata used.
- Shell checks: `shellcheck bootstrap.sh` and `bash -n bootstrap.sh` passed.
- Tracked-file status: clean. Pre-existing untracked files are intentionally
  ignored for this effort.
- Gate A status: baseline green; isolation is established on
  `feature/upgrade-refactor-preflight`.

---

## Work Package 1 — Make Test Coverage Trustworthy

**Priority:** P0
**Primary files:**

- `.coveragerc`
- `.github/workflows/type-check.yml`
- `dev_tools/nautical_golden_tests.py`
- `tests/support/hook_process.py`
- new `tests/test_hook_process_contract.py`
- hook subprocess tests under `tests/`

### 1.1 Record the coverage gap

- [x] Run unit discovery under branch coverage and save the total.
- [x] Record coverage for the hook adapters, lifecycle application, lifecycle
  outbox, installer, reconciliation tool, and modify composition modules.
- [x] Classify each low-coverage module as genuinely untested or covered only
  through a subprocess whose coverage is not combined.
- [x] Do not add shallow import-only tests to raise the percentage.

```bash
rm -f /tmp/nautical-refactor-coverage*
COVERAGE_FILE=/tmp/nautical-refactor-coverage \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 -m coverage run -m unittest discover -s tests -q
COVERAGE_FILE=/tmp/nautical-refactor-coverage \
  python3 -m coverage report --sort=cover --show-missing
```

### 1.2 Capture subprocess coverage

- [x] Add one coverage-aware subprocess fixture in `tests/support/`; tests must
  opt in explicitly and must not alter production hook launch behavior.
- [x] Create `tests/test_hook_process_contract.py` for coverage propagation,
  child failure, timeout, and malformed-output behavior.
- [x] Combine child-process data into the same disposable coverage file.
- [x] Prove the fixture records executed hook lines with a focused on-add test.
- [x] Prove a child failure, timeout, or malformed JSON response still fails the
  owning test instead of being hidden by coverage collection.
- [x] Keep the normal non-coverage test command unchanged.

### 1.3 Ratchet the CI floor

- [x] Repeat coverage collection three times on a clean checkout.
- [x] Set the CI floor no higher than the lowest repeatable result.
- [x] Raise the current 44% floor to at least 60% once subprocess data is
  reproducible.
- [x] Add focused minimums or explicit regression tests for hook protocol,
  lifecycle execution, and outbox recovery; do not rely only on the global
  percentage.
- [x] Upload the combined report on both success and failure.

Coverage record from the preflight branch run (`COVERAGE_FILE=/tmp/nautical-refactor-coverage`):

- Total: 63% branch coverage across 41,254 statements and 18,748 branches.
- Hook adapters: `hooks/add_impl.py` 0%, `hooks/modify_impl.py` 31%,
  `hooks/exit_impl.py` 33%, and `hook_bootstrap.py` 33%; these are primarily
  subprocess-only execution paths and are the reason for the new opt-in fixture.
- Lifecycle: `lifecycle_application.py` 49%; `lifecycle_outbox.py` 60%.
- Other audited gaps: `install_runtime.py` 7%, `tools/nautical_reconcile.py`
  14%, and `modify_composition_adapters.py` 13%.
- Repeatability: three branch-coverage runs each reported 63% and 1,260 tests
  passed with 3 skipped.

### Verification

```bash
COVERAGE_FILE=/tmp/nautical-refactor-coverage \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 -m coverage run -m unittest discover -s tests -q
COVERAGE_FILE=/tmp/nautical-refactor-coverage \
  python3 -m coverage report --fail-under=60 --show-missing
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 -m unittest tests.test_hook_input_contract \
    tests.test_on_add_hook_routes tests.test_hook_process_contract -v
```

**Done when:** subprocess execution is visible in the report, critical hook and
lifecycle paths have behavioral tests, and CI enforces a repeatable floor of at
least 60%.

**Work Package 1 status:** complete. The remaining low-coverage modules are
  recorded as follow-on refactor targets; no import-only coverage shims were
  added.

---

## Work Package 2 — Split the Golden Test Monolith

**Priority:** P0

**Depends on:** Work Package 1
**Primary files:**

- `dev_tools/nautical_golden_tests.py`
- new focused modules under `dev_tools/golden_tests/`
- `tests/test_golden_registry_integrity.py`
- workflows that invoke `dev_tools/nautical_golden_tests.py`

### 2.1 Define the stable runner contract

- [x] Preserve the current command, exit codes, `--only`, `--verbose`, shuffle,
  summary output, and deterministic registration behavior.
- [x] Add or confirm tests that reject duplicate and unregistered golden tests.
- [x] Record the current 404-test registry and named slices before extraction.

### 2.2 Extract one domain at a time

- [x] Create `dev_tools/golden_tests/__init__.py` with no import-time execution.
- [x] Extract the smallest cohesive domain first, such as hook protocol or
  natural-language behavior.
- [x] Give each extracted module an explicit immutable test collection.
- [x] Make the existing runner import and combine domain collections in stable
  order.
- [x] Run the focused slice, normal suite, and shuffled suite after each move.
- [x] Extract remaining tests by ownership: recurrence/parser, hooks, lifecycle,
  reconciliation/outbox, installer/operator, and performance/deployment.
- [x] Remove copied helpers from the monolith only after all importing domains
  use one shared helper owner.
- [x] Keep process-global fixtures isolated; do not introduce suite-order
  dependencies during extraction. Explicit collection/fresh-process checks are
  covered by `tests/test_golden_fixture_isolation.py`, with normal and shuffled
  full-suite gates also passing.

### Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 -m unittest tests.test_golden_registry_integrity -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 dev_tools/nautical_golden_tests.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 dev_tools/nautical_golden_tests.py --shuffle-seed 20260920
```

**Done when:** the runner retains all 404 audited behaviors and no individual
golden module is a new monolith. Focused slices must run without importing
unrelated domains.

Current registry record:

- Runner registry: 404 tests.
- Domain collections: recurrence 1, hooks 5, operator/query 4, installer 1,
  performance/deployment 2, lifecycle 15, reconciliation/outbox 33.
- Top-level legacy definitions: 1, exclusively the intentional
  `test_on_modify_staged_plan_carries_parent_guard_and_stable_intent_id`
  hook-bootstrap boundary test.
- **Work Package 2 status:** complete. Shared-helper cleanup, fixture
  isolation, registry integration, and lifecycle/reconciliation ownership
  extraction are complete. The accepted hook-bootstrap exception remains
  documented and is not an extraction defect.

---

## Work Package 3 — Consolidate Hook Adapters

**Priority:** P1

**Depends on:** Work Packages 1 and 2
**Primary files:**

- `nautical_core/hooks/add_impl.py`
- `nautical_core/hooks/modify_impl.py`
- `nautical_core/hooks/exit_impl.py`
- `nautical_core/hook_bootstrap.py`
- `nautical_core/hook_runtime.py`
- `nautical_core/hook_protocol.py`
- `nautical_core/hook_results.py`

### 3.1 Freeze observable behavior

- [x] Add or confirm executable tests for valid, malformed, empty, truncated,
  trailing, oversized, and Unicode input for every hook.
- [x] Assert stdout contains exactly the required JSON document for add and
  modify and remains empty where the exit protocol requires it.
- [x] Assert diagnostics appear only on stderr with `NAUTICAL_DIAG=1`.
- [x] Assert panic and unavailable-core paths never emit partial JSON.
- [x] Record ordinary-task import count and latency before moving code.

### 3.2 Centralize shared runtime setup

- [x] Move duplicated runtime-module loading and module-access construction to
  the existing `hook_runtime` or `hook_bootstrap` owner.
- [x] Centralize Taskdata resolution and integration-context construction while
  preserving each hook's read-only or mutation access mode.
- [x] Replace per-hook global synchronization lists with one typed result from
  the shared initializer.
- [x] Centralize diagnostic redaction without enabling diagnostics by default.
- [x] Centralize bounded diagnostic blocks without enabling diagnostics by default.
- [x] Centralize profiler plumbing without enabling diagnostics by default.
- [x] Keep hook-specific policy and presentation outside the shared adapter.

### 3.3 Replace process exits below the entry point

- [x] Define one typed hook failure result or exception at the protocol boundary.
- [x] Convert helper-level `sys.exit()` calls to that boundary contract.
- [x] Catch failures once in each executable entry point and serialize the
  protocol-preserving response there.
- [x] Narrow broad exception handlers where the dependency exposes a stable
  exception type; remaining broad catches are fail-safe boundaries around
  dynamic hook loading, panic passthrough, or platform-dependent bootstrap
  operations.
- [x] Preserve intentional fail-safe catches with a comment and a direct test.

### 3.4 Tighten the extracted boundary

- [x] Add complete annotations to shared hook runtime and protocol functions.
- [x] Enable `disallow_untyped_defs` and `disallow_incomplete_defs` for each
  hook module only after it passes independently.
- [x] Remove imports, globals, and private helpers made obsolete by the split
  where they are demonstrably unused; compatibility globals and helpers remain
  intentionally retained for hook loaders.
- [ ] Do not change recurrence, lifecycle, or presentation behavior in this
  work package.

### Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 -m unittest tests.test_hook_input_contract \
    tests.test_hook_protocol tests.test_hook_host_isolation \
    tests.test_on_add_hook_routes tests.test_hook_process_contract -v
/home/pooK/venv/test_1/bin/mypy --config-file mypy.ini \
  --disallow-untyped-defs --disallow-incomplete-defs \
  nautical_core/hooks nautical_core/hook_bootstrap.py \
  nautical_core/hook_runtime.py nautical_core/hook_protocol.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 dev_tools/nautical_deploy_sanity.py --json
```

**Gate B — hooks complete:** all hook protocol tests pass, strict mypy passes
for the hook boundary, ordinary-task startup stays within its existing budget,
and the three implementation modules contain no duplicated bootstrap path.

---

## Work Package 4 — Decompose Lifecycle Application and Outbox Ownership

**Priority:** P1

**Depends on:** Work Package 1
**Primary files:**

- `nautical_core/lifecycle_application.py`
- `nautical_core/lifecycle_outbox.py`
- `nautical_core/lifecycle_models.py`
- `nautical_core/lifecycle_recovery_models.py`
- `nautical_core/lifecycle_reconciliation.py`
- lifecycle and failure-injection tests under `tests/`

### 4.1 Freeze persistence and recovery contracts

- [x] Record the current SQLite schema version and every supported row state.
- [x] Add or confirm round-trip tests for each persisted plan and failure type.
- [x] Cover concurrent initialization, schema mismatch, poison-row quarantine,
  stale claims, lease loss, retry exhaustion, acknowledgement, pruning, and
  manual-review resolution.
- [x] Cover interruption before and after each durable stage transition.
- [x] Assert Taskwarrior is never invoked inside an open SQLite transaction.

### 4.2 Separate outbox responsibilities

- [x] Extract pure plan/failure serialization from repository operations.
- [x] Separate schema initialization and migration from ordinary reads/writes.
- [x] Separate claim, lease-renewal, and compare-and-set operations from
  maintenance and status reporting through `LifecycleOutboxClaimPort`.
- [x] Keep transaction ownership explicit in every extracted operation.
- [x] Resolve the existing `LifecycleOutboxRepository` compatibility surface
  explicitly: because `nautical_core` internals are unsupported for direct
  user imports, the alias and delegating wrappers were removed after all
  in-repository callers migrated.
- [x] Remove the delegation layer after the explicit internal-only compatibility
  decision, in-repository importer migration, strict typing, focused behavior
  gates, and installed-layout sanity checks. Direct imports of Nautical internals
  remain unsupported by policy.

### 4.3 Split application orchestration from failure policy

- [x] Extract batch-progress accounting from mutation execution.
- [x] Extract result classification, outcome taxonomy, and retry-vs-review
  decisions into one typed policy owner; persistence-dependent retry, budget,
  and manual-review side effects remain in the application service.
- [x] Replace deeply nested closures in `_drain_batched` with named helpers or
  focused collaborators that receive explicit state.
- [x] Keep serial mutation order and batched verification semantics unchanged.
- [x] Preserve authoritative post-mutation snapshots and fail-closed handling
  of unavailable reads.

### 4.4 Raise direct coverage before removing compatibility paths

- [x] Bring `lifecycle_application.py` above its audited 49% branch coverage.
- [x] Bring `lifecycle_outbox.py` above its audited 60% branch coverage.
- [x] Require direct tests for extracted modules rather than credit from import
  execution alone.
- [x] Run failure injection and terminal-plan tests after every extraction.

### Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest \
  tests.test_lifecycle_failure_injection \
  tests.test_lifecycle_execution_capabilities \
  tests.test_lifecycle_recovery_policy \
  tests.test_lifecycle_terminal_plans \
  tests.test_taskwarrior_uow_contracts -v
/home/pooK/venv/test_1/bin/mypy --config-file mypy.ini \
  --disallow-untyped-defs --disallow-incomplete-defs \
  nautical_core/lifecycle_application.py \
  nautical_core/lifecycle_outbox.py \
  nautical_core/lifecycle_models.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 dev_tools/nautical_reliability_smoke.py --load 10
```

**Gate C — lifecycle complete:** schema compatibility, recovery, idempotence,
lease ownership, retry limits, and postcondition verification remain green;
application orchestration and persistence no longer reside in two oversized
owners.

---

## Work Package 5 — Shrink the Compatibility Facade Safely

**Priority:** P2

**Depends on:** Work Packages 2 through 4
**Primary files:**

- `nautical_core/__init__.py`
- `nautical_core/compat_api.py`
- `nautical_core/api_bindings.py`
- focused `*_api.py` modules
- `nautical_core/architecture_contract.py`
- `tests/test_typed_api_bindings.py`
- `tests/test_architecture_contract.py`

### 5.1 Inventory the public surface

- [x] Export the current 130-name public surface and its signature snapshot.
- [x] Assign every export one canonical owner module.
- [x] Classify each name as supported public API, installed-runtime contract,
  test seam, or legacy compatibility alias.
- [x] Search documentation, hooks, tools, tests, and release assets for each
  name; do not treat zero internal importers as proof of external disuse.
- [x] Publish the deprecation rule and minimum compatibility window before
  removing any public name.

### 5.2 Remove implementation from the facade

- [x] Move remaining cache, configuration, UI, parser, and recurrence behavior
  to canonical owners while leaving lazy forwarding aliases where required.
- [x] Keep `nautical_core/__init__.py` limited to export metadata, lazy binding,
  and compatibility forwarding.
- [x] Preserve public signatures, annotations, cache-control attributes, and
  monkeypatch points covered by tests.
- [x] Remove zero-use forwarding modules only after an explicit compatibility
  decision, release note, and installed-layout test.

### 5.3 Improve physical package organization incrementally

- [ ] Move at most one cohesive ownership group per change.
- [ ] Prefer existing packages (`hooks`, `tools`, `parsing`) before creating a
  new namespace.
- [ ] When a new package is justified, migrate canonical owners first and add
  compatibility forwarding modules separately.
- [x] Update the architecture contract to classify physical packages directly;
  retain filename-prefix fallback only for modules not yet migrated.
- [x] Confirm no domain or recurrence module imports the root facade,
  compatibility implementation, Taskwarrior, SQLite, or Rich.

### Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest \
  tests.test_typed_api_bindings tests.test_architecture_contract \
  tests.test_parser_domain_imports tests.test_parser_owner_api_contracts -v
/home/pooK/venv/test_1/bin/mypy --config-file mypy.ini \
  --follow-imports=normal nautical_core/__init__.py \
  nautical_core/compat_api.py nautical_core/api_bindings.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 dev_tools/nautical_deploy_sanity.py --json
```

**Done when:** the root module is a small compatibility surface, every public
name has one canonical owner, installed releases preserve supported imports,
and architecture classification increasingly follows physical ownership.

---

## Work Package 6 — Tighten Type and Error Contracts

**Priority:** P2
**Runs with:** Work Packages 3 through 5

- [x] Capture the strict-typing error count for each file before modifying it.
- [ ] Require complete annotations for every newly extracted interface.
- [ ] Use existing domain types such as `TaskStatus`, `OmissionPolicy`, typed
  command results, query results, lifecycle outcomes, and outbox results.
- [ ] Convert raw strings only at Taskwarrior, CLI, JSON, or persistence
  boundaries.
- [ ] Do not replace validation with `cast`, `Any`, or unchecked assertions.
- [ ] Replace helper-level exits and heterogeneous error tuples with the owning
  subsystem's typed failure contract.
- [ ] Preserve actionable error codes, retryability, causal exceptions, and
  operator guidance across adapter boundaries.
- [ ] Add strict mypy sections as each ownership group reaches zero errors.
- [ ] Remove obsolete exceptions, aliases, imports, and error adapters created
  redundant by the same change; leave unrelated legacy code alone.

Strict diagnostic command:

```bash
/home/pooK/venv/test_1/bin/mypy --config-file mypy.ini \
  --disallow-untyped-defs --disallow-incomplete-defs \
  --enable-error-code=union-attr \
  --enable-error-code=attr-defined \
  --enable-error-code=assignment \
  --enable-error-code=arg-type \
  --enable-error-code=return-value \
  --enable-error-code=operator \
  nautical_core
```

WP6 baseline record — 2026-09-22: strict diagnostics reported 748 errors
across 90 files. The first focused slice tightened the outbox read,
maintenance, and evidence protocols; its affected modules now pass strict
mypy. The execution-port slice then replaced catch-all lifecycle method
signatures with explicit plan, lease, stage, acknowledgement, retry, and
manual-review contracts; six affected owners pass strict mypy. Remaining
errors are intentionally queued by ownership group. A follow-up lifecycle
reconciliation callback slice added complete annotations and reduced the
strict inventory from 748 to 721 errors. The scheduler API ownership slice
then completed annotations for its factory callbacks and reduced the current
inventory to 647 errors across 85 files; its focused scheduler contract tests
pass. The natural-language formatter slice then completed annotations for its
public description helpers and reduced the inventory to 610 errors across 84
files; its focused natural-language tests pass.
The scheduler-expression slice then completed annotations for expression
search helpers and reduced the inventory to 577 errors across 83 files; its
focused scheduler exhaustion and conformance tests pass.
The facade (`nautical_core/__init__.py`) slice then completed annotations for
lazy adapters, runtime hooks, recurrence metadata, and CP helpers, reducing
the inventory to 546 errors across 82 files; 39 facade/API contract tests
pass.
The parser API slice then completed annotations for parser adapters, preset
resolution, yearly validation, satisfiability checks, and public entry points,
reducing the inventory to 519 errors across 81 files; 55 focused parser and
anchor contract tests pass.
The anchor-computation slice then completed annotations for occurrence
stepping, time-slot resolution, DST-safe candidate construction, previews,
and until summaries, reducing the inventory to 494 errors across 80 files;
43 focused anchor/preview tests pass.

**Done when:** every touched ownership boundary passes strict mypy, errors are
translated once at the boundary that owns them, and the ordinary configured
mypy run remains clean.

---

## Work Package 7 — Split the Performance Harness

**Priority:** P3

**Depends on:** Stable Work Packages 3 and 4
**Primary files:**

- `dev_tools/nautical_perf_budget.py`
- `dev_tools/nautical_perf_compare.py`
- `dev_tools/perf_budget.json`
- new focused modules under `dev_tools/perf/`
- performance tests under `tests/`

- [x] Freeze the CLI, JSON schema, workload names, budget semantics, and exit
  codes with subprocess tests.
- [x] Group current workloads by ownership: recurrence/cache, hook startup,
  Taskwarrior workflow, lifecycle/outbox, operator/query, and resources.
- [x] Extract one group at a time into import-safe modules with no benchmark
  execution at import time.
- [x] Keep fixture construction separate from timing and report rendering.
- [x] Centralize subprocess environments, disposable Taskdata creation, and
  result validation.
- [x] Preserve deterministic seeds, workload sizes, warm/cold distinctions,
  and current budget keys.
- [x] Compare pre- and post-extraction JSON reports; require identical workload
  coverage and pass/fail decisions before accepting timing differences.
- [x] Keep `dev_tools/nautical_perf_budget.py` as a thin CLI entry point used by
  existing workflows.

### Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 -m unittest tests.test_perf_budget_contract tests.test_perf_compare -v
python3 dev_tools/nautical_perf_budget.py \
  --json --enforce --budget-file dev_tools/perf_budget.json
python3 dev_tools/nautical_perf_budget.py \
  --extended --workflows-only --json --enforce \
  --budget-file dev_tools/perf_budget.json
```

**Done when:** the CLI remains backward compatible, workload modules are
independently testable, and no budget or safety assertion disappears.

WP7 preflight slice complete: the budget manifest contracts already covered
workload names and budget semantics, and the added subprocess contract now
guards the budget CLI help surface, supported options, and zero exit status;
the compare CLI subprocess contract remains green as well. Focused performance
contract tests pass (35), with the full suite at 1310 passed and 3 skipped.

The first extraction slice moved shared timing and workflow-report measurement
helpers into import-safe `dev_tools/perf/telemetry.py`; the legacy budget module
keeps the same private helper names as aliases, so its CLI and JSON schema are
unchanged. Performance contract tests remain green (35), and the full suite
passes 1310 tests with 3 skipped.

The reporting extraction slice then moved timing-breakdown, task-timing merge,
reconcile compaction, and report-attachment helpers into import-safe
`dev_tools/perf/reporting.py`. Direct module contracts and legacy budget aliases
pass; focused performance tests pass (36), and the full suite passes 1311 tests
with 3 skipped.

The recurrence/cache workload slice then moved cache-key, isolated cache
context, payload, save, and hot-load benchmarks into import-safe
`dev_tools/perf/cache_workloads.py`, with explicit core and cache-reset
dependencies. Legacy workload names and budget keys remain unchanged; focused
performance tests pass (36), direct cache smoke checks pass, and the full suite
passes 1311 tests with 3 skipped.

The lifecycle/outbox workload slice then moved hot and cold outbox-schema
benchmarks into import-safe `dev_tools/perf/outbox_workloads.py`, with explicit
repository and subprocess-root dependencies. Both legacy wrappers and direct
hot/cold smoke checks pass; focused performance tests pass (36), and the full
suite passes 1311 tests with 3 skipped.

The anchor-file workload slice then moved regular and batch provider benchmarks
into import-safe `dev_tools/perf/anchor_file_workloads.py`. Legacy provider
workload names and budgets remain unchanged; focused performance tests pass
(36), direct provider and batch smoke checks pass, and the full suite passes
1311 tests with 3 skipped.

The large anchor-file slice then moved hot, cold, non-monotonic, and
business-day provider modes into the same import-safe workload module. Extended
workload names and budgets remain unchanged; all four direct mode smoke checks
pass, focused performance tests pass (36), and the full suite passes 1311 tests
with 3 skipped.

The business-calendar workload slice then moved the large omission-set
recurrence benchmark into import-safe `dev_tools/perf/calendar_workloads.py`,
with the core scheduler supplied explicitly. The recurrence correctness guard
and extended budget key remain unchanged; direct omission smoke checks pass,
focused performance tests pass (36), and the full suite passes 1311 tests with
3 skipped.

The remaining resource benchmark slice then moved snapshot reuse, resource
limits, and snapshot-memory measurements into the same import-safe resource
module. Legacy workload names and guards remain unchanged; direct reuse,
limits, and memory smoke checks pass, focused performance tests pass (36), and
the full suite passes 1311 tests with 3 skipped.

The hook-startup workload group then moved source, managed-install, and staged
layout latency cases into import-safe `dev_tools/perf/hook_workloads.py`, with
runtime, environment, panel, and measurement dependencies supplied explicitly.
All 12 fast/managed/staged cases pass with a direct smoke run; focused
performance tests pass (36), and the full suite passes 1311 tests with 3
skipped.

The lifecycle staging micro-workload then moved the guarded durable-stage
measurement into `dev_tools/perf/outbox_workloads.py`, with fixture, outbox,
and pending-intent dependencies supplied explicitly. Direct staging smoke,
focused performance tests (36), and the full suite (1311 tests, 3 skipped)
remain green.

The operator/reconcile micro-workload group then moved reconcile projection,
stale-claim detection, interrupted-claim recovery, exit probing, and pagination
scope checks into import-safe `dev_tools/perf/operator_workloads.py`. Direct
smoke checks, focused performance tests (36), and the full suite (1311 tests,
3 skipped) remain green.

The operator failure-matrix workload then moved fail-closed query, repair,
queue, Doctor, and reconcile-boundary checks into the same module. Its direct
smoke check passes; focused performance tests (36) and the full suite (1311
tests, 3 skipped) remain green.

The expression/scheduler workload group then moved anchor description,
next-after, and traced scheduler-decision measurements into import-safe
`dev_tools/perf/scheduler_workloads.py`, with core, codec, cache, and detail
stores supplied explicitly. Direct expression/scheduler smoke checks pass;
focused performance tests (36) and the full suite (1311 tests, 3 skipped)
remain green.

The Taskwarrior workflow group now has an import-safe boundary in
`dev_tools/perf/workflow_workloads.py`; the CLI routes the existing fixture-heavy
runner through that boundary without changing workload names or report schema.
The workflow-only CLI passes with 24 results and no failures; the focused suite
passes 38 tests and the full suite passes 1312 tests with 3 skipped. The next
slice is the internal migration of the legacy runner body into that module.

The first internal workflow slice moved deterministic integrity-scale fixture
construction and invariant measurement into `workflow_workloads.integrity_scale`.
The workflow-only CLI still reports 24 results with no failures, focused tests
remain green, and the full suite passes 1312 tests with 3 skipped.

The ordinary-modify workflow slice then moved its disposable Taskdata fixture,
hook invocation, output guard, and timing aggregation into
`workflow_workloads.ordinary_modify`. Workload names and budgets remain stable;
the workflow-only CLI reports 24 passing results and the full suite passes 1312
tests with 3 skipped.

The expiration-recovery slice then moved its date-boundary fixture, successor
staging guard, and replay/idempotency assertion into
`workflow_workloads.expiration_recovery`. The workflow-only CLI reports 24
passing results, and the full suite passes 1312 tests with 3 skipped.

The completion workflow slice then moved fresh CP/anchor completion and
idempotent existing-child cases into `workflow_workloads.completion_workflows`,
including task-call statistics and budget enforcement. The workflow-only CLI
reports 24 passing results, and the full suite passes 1312 tests with 3 skipped.

The queue-drain preflight slice then moved parent import, authoritative export
verification, plan binding, and durable staging into
`workflow_workloads.queue_preflight`. Healthy, replay, and partial-recovery
queue checks remain unchanged; the workflow-only CLI reports 24 passing results,
and the full suite passes 1312 tests with 3 skipped.

The healthy/replay queue helper is now defined in
`workflow_workloads.queue_healthy_replay`, covering drain, no-I/O replay,
convergence export, and timing/stat collection. The existing inline phase is
still authoritative until the replacement is applied atomically in the next
slice; import, focused tests (37), and the full suite (1312 tests, 3 skipped)
remain green.

The replay/convergence phase is now wired through
`workflow_workloads.queue_replay_verify`, including the no-I/O replay guard and
parent/child export checks. The workflow-only CLI reports 24 passing results,
and the full suite passes 1312 tests with 3 skipped.

The partial-import recovery phase is now routed through
`workflow_workloads.queue_partial_recovery`, including injected failure,
requeue validation, successful recovery, and merged timing/stat results. The
workflow-only CLI reports 24 passing results, and the full suite passes 1312
tests with 3 skipped.

The queue-shape/history slice now runs both one-intent and large-history
measurements through `workflow_workloads.queue_shape`, preserving background
row sizes, timing attribution, and command-count guards. The workflow-only CLI
reports 24 passing results, and the full suite passes 1312 tests with 3 skipped.

The reconcile fixture slice now moves deterministic completed-chain history
construction and Taskwarrior import preflight into
`workflow_workloads.reconcile_history_fixture`. Healthy reconcile output and
report schemas remain unchanged; the workflow-only CLI reports 24 passing
results, and the full suite passes 1312 tests with 3 skipped.

Import-safe `workflow_workloads.reconcile_healthy` and
`workflow_workloads.reconcile_empty` helpers are now defined for the healthy
and empty report phases. Their atomic wiring into the legacy loops is deferred
to the next slice to avoid duplicate subprocess measurements; current focused
tests (37) and the full suite (1312 tests, 3 skipped) remain green.

The healthy and empty reconcile phases are now wired through those helpers;
their legacy loops are disabled and no longer execute duplicate subprocess
measurements. The workflow-only CLI reports 24 passing results, and the full
suite passes 1312 tests with 3 skipped.

The candidate-heavy reconcile phase is now routed through
`workflow_workloads.reconcile_candidates`, preserving deterministic candidate
fixture import, actionable-evidence validation, and compact report aggregation.
The workflow-only CLI reports 24 passing results, and the full suite passes
1312 tests with 3 skipped.

The guarded candidate-apply phase is now routed through
`workflow_workloads.reconcile_candidates_apply`, preserving successor-creation
guards and apply-report attribution. The workflow-only CLI reports 24 passing
results, and the full suite passes 1312 tests with 3 skipped.

The candidate-apply scaling matrix is now routed through
`workflow_workloads.reconcile_candidates_apply_scale`, preserving configured
counts (1, 8, 32, 200), convergence guards, timing rows, and compact reports.
The workflow-only CLI reports 24 passing results, and the full suite passes
1312 tests with 3 skipped.

The final three reconcile workload loops are now routed through the shared
`workflow_workloads.reconcile_report_loop`: long-history, corrupted-chain, and
mixed candidate/integrity scenarios all preserve their scenario-specific
guards and report keys. The workflow-only CLI reports 24 passing results, and
the full suite passes 1312 tests with 3 skipped.

WP7 cleanup removed all six disabled legacy reconcile loops and the unused
inline queue-shape helper from `nautical_perf_budget.py`. The workflow-only CLI
reports 24 passing results with no failures, focused tests pass (37), and the
full suite passes 1312 tests with 3 skipped.

The cleanup pass also removed obsolete `_legacy` benchmark copies now replaced
by extracted workload modules; only the authoritative fixture-heavy workflow
runner remains pending its final internal decomposition. Workflow-only CLI,
focused contracts, and the full suite remain green.

The resource task-model slice then moved task-codec decoding and immutable
observation benchmarks into import-safe `dev_tools/perf/resource_workloads.py`,
with the codec dependency supplied explicitly. Legacy workload names and
correctness guards remain unchanged; direct codec/immutability smoke checks
pass, focused performance tests pass (36), and the full suite passes 1311 tests
with 3 skipped.

WP7 is complete. Fixture construction now lives behind `WorkflowContext`,
subprocess environments and Taskdata allocation are centralized, reconcile
result validation is shared, and `nautical_perf_compare.py --contract-only`
enforces workload coverage and pass/fail parity. Fresh acceptance evidence:
workflow-only 24/24 passing, normal enforced budget 78/78 passing, full suite
1316 passed with 3 skipped, and `git diff --check` clean.

WP7 hardening also adds an import-safety contract for every extracted workload
module and fail-closed validation for malformed comparison reports, including
invalid result shapes, pass flags, negative metrics, and non-finite metrics.
The focused performance contract suite passes 43 tests; the full suite passes
1318 tests with 3 skipped.

The final workflow verification fixed two extraction regressions: expiration
staging now uses the canonical private outbox repository, and queue-shape
fixtures use the isolated workflow root instead of the checkout root. Fresh
bridge verification now completes the normal enforced budget with exit code 0
and all 78 workload results present; the full suite remains 1318 passed with 3
skipped.

The extended workflow profile was then rerun in the environment with Astral
3.2 and completed successfully: exit code 0, 33 workload results, and no
failures.

### Post-WP7 typing hardening follow-up

- [x] Replace the highest-value callback-boundary `Any` annotations with small
  `Protocol` interfaces, starting with cache, scheduler, hook, and modify
  service ports. WP7 is complete; the cache locking, scheduler callback, hook
  diagnostic, modify callback, and completion preflight/compute/spawn service
  boundaries now use focused protocols or shared callback ports. Focused
  boundary tests pass. The configured repository-wide mypy run still reports
  the pre-existing untyped-definition backlog outside these slices.

---

## Final Verification and Acceptance

- [x] Full unit discovery passes with zero failures and errors.
- [x] Normal and deterministic shuffled golden suites pass with identical test
  counts.
- [x] Configured mypy passes across the package.
- [x] Strict mypy passes for every refactored ownership group.
- [x] Combined branch coverage meets the ratcheted floor and does not lose
  coverage on hook protocol, lifecycle execution, or recovery paths.
- [x] Deployment and architecture sanity report `status: ok`.
- [x] Golden strict-JSON and Unicode tests pass for add and modify hooks.
- [x] Malformed-input tests prove no hook emits a traceback or partial JSON.
- [x] Black-box Taskwarrior lifecycle tests pass against disposable Taskdata.
- [x] Normal and extended performance budgets pass.
- [x] Stress and short-soak profiles pass without queue, dead-letter, or
  recovery regressions.
- [x] Documentation names the canonical owner of each moved public API.
- [x] `git diff --check` passes and generated reports, caches, databases, and
  local artifacts are not staged.
- [x] A fresh tracked-HEAD health scan is captured after the execution queue is
  clear; stale findings are not manually marked resolved without scan evidence.

Final acceptance evidence — 2026-09-24, commits `6d5d318` and `6e73978`:
full unit discovery 1318 passed/3 skipped; strict golden 404/404 passed in
normal and deterministic shuffled order; configured and strict mypy passed all
255 source files; branch coverage 63% (44% floor); deployment/architecture
sanity `status: ok`; black-box lifecycle passed; normal and extended performance
budgets passed; CI stress passed with no violations; 30-second soak passed with
zero failures, queue bytes, or dead letters. A forced local tracked-HEAD
desloppify scan completed on 2026-09-24 after resetting the local plan queue;
the resulting health queue contains 13 subjective review items.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 -m unittest discover -s tests -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 dev_tools/nautical_golden_tests.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 dev_tools/nautical_golden_tests.py --shuffle-seed 20260920
/home/pooK/venv/test_1/bin/mypy --config-file mypy.ini
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 dev_tools/nautical_deploy_sanity.py --json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 dev_tools/nautical_black_box_test.py --json
python3 dev_tools/nautical_perf_budget.py \
  --json --enforce --budget-file dev_tools/perf_budget.json
python3 dev_tools/nautical_perf_budget.py \
  --extended --workflows-only --json --enforce \
  --budget-file dev_tools/perf_budget.json
python3 dev_tools/nautical_stress_campaign.py --profile ci --json --enforce
git diff --check
```

## Completion Rule

Mark this checklist complete only when all final acceptance items pass on one
recorded revision. If a work package exposes a behavior change, stop, document
the decision separately, and obtain approval before continuing; this checklist
authorizes structural refactoring, not new product behavior.
The completion-effects slice then completed annotations for completion
preflight, child-due computation, limit guards, lifecycle attachment, and
spawn adapters, reducing the inventory to 470 errors across 79 files; 35
focused completion/workflow tests pass.
The cache-payload slice then completed annotations for cache shape validation,
cloning, bounded decode, atomic persistence, loading, saving, and garbage
collection, reducing the inventory to 451 errors across 78 files; 60 focused
cache/file-backed tests pass.
The add-composition slice then completed annotations for add-context assembly,
validation, schedule/limit recording, and preview adapters, reducing the
inventory to 432 errors across 77 files; 42 focused add workflow/preview tests
pass.
The time API slice then completed annotations for timezone conversion,
datetime arithmetic, recurrence time selection, and local datetime builders,
reducing the inventory to 414 errors across 76 files; 24 focused time/config
contract tests pass.
The cached-expansion slice then completed annotations for deterministic random
selection, monthly/yearly token expansion, weekly filters, and candidate
generation, reducing the inventory to 397 errors across 75 files; 71 focused
cache/scheduler tests pass.
The cache-locking slice then completed annotations for lock directory
selection, stale-lock handling, POSIX/exclusive lock contexts, and bound lock
adapters, reducing the inventory to 380 errors across 74 files; 60 focused
cache/resource tests pass.
The natural-language API slice then completed annotations for formatter,
description, selection, and random-bucket adapters, reducing the inventory to
361 errors across 73 files; 18 focused natural-language/presentation tests
pass.
The diagnostics-effects slice then completed annotations for analytics,
timeline, span, endpoint, and chain-summary adapters, reducing the inventory to
344 errors across 72 files; 5 focused diagnostics/summary tests pass.
The add-preview composition slice then completed annotations for CP period
calculation, sequence previews, limit rows, anchor preview services, and
render adapters, reducing the inventory to 329 errors across 71 files; 47
focused add/preview/presentation tests pass.
The ACF-support slice then completed annotations for canonical encoding,
decoding, normalization, validation, modifier serialization, and original
format reconstruction, reducing the inventory to 315 errors across 70 files;
74 focused ACF/cache/parser tests pass.
The linting slice then completed annotations for segment extraction, alias
expansion, delimiter/year/weekday diagnostics, satisfiability checks, and
warning aggregation, reducing the inventory to 302 errors across 69 files;
58 focused lint/validation/parser tests pass.
The three-file parser/config batch then completed annotations for configuration
loading/security helpers, parser support adapters, and atom parsing/modifier
construction, reducing the inventory to 265 errors across 66 files; 71
focused config/parser/API tests pass.
The next three-file batch then completed annotations for business-calendar
adapters, yearly-token validation, and modify composition/runtime services,
reducing the inventory to 231 errors across 63 files; 60 focused calendar,
yearly, and modify-workflow tests pass.
The UI/monthly/scheduler-atom batch then completed annotations for panel
rendering, monthly candidate alignment, atom scheduling, interval guards, and
matching logic, reducing the inventory to 204 errors across 60 files; 67
focused UI/recurrence tests pass.
The satisfiability/quarter/parser-DNF/modify-adapter/omit batch then completed
annotations for satisfiability probes, quarter rewrites, DNF parsing,
modify-boundary services, and omission evaluation, reducing the inventory to
164 errors across 55 files; 78 focused parser/recurrence/modify tests pass and
the full suite remains green (1307 passed, 3 skipped).
Typing-gate hardening then made strict function checks global in `mypy.ini`,
kept the same flags explicit in the CI full-package gate, and added a contract
test guarding both settings. The configured mypy run is clean; the full suite
now passes 1309 tests with 3 skipped.
The final hook-context/linting/anchor/carry/generation batch then completed
annotations for add/modify request construction, linting iteration, omission
state assembly, temporal carry decisions, and chain-generation service
binding. Strict mypy now reports no errors across all 254 nautical-core source
files; 150 focused workflow tests pass and the full suite remains green (1307
passed, 3 skipped).
The compiled-schedule/exit/expansion/hint batch then completed annotations for
omit validation callbacks, exit result/presentation boundaries, monthly
expansion intersections, and hint-builder inputs, reducing the inventory to 5
errors across 5 files; 79 focused schedule/exit/lifecycle tests pass and the
full suite remains green (1307 passed, 3 skipped).
The add-workflow/anchor-files/astronomy/cache-facade/chain-integrity batch then
completed annotations for add scheduling patches, anchor-file time handling,
astronomy observer setup, cache clearing, and typed repair payloads, reducing
the inventory to 10 errors across 10 files; 123 focused add/cache/integrity
tests pass and the full suite remains green (1307 passed, 3 skipped).
The business-calendar-config/scheduler/position-selection/value/time batch then
completed annotations for configured calendar predicates, scheduler tracing,
candidate-cache introspection, datetime comparisons, and time-slot
normalization, reducing the inventory to 15 errors across 15 files; 119
focused scheduler/calendar/selection tests pass and the full suite remains
green (1307 passed, 3 skipped).
The install-runtime/hook-context/diagnostics/common/cache-support batch then
completed annotations for install locking, invocation contexts, diagnostic
warnings, shared coercion helpers, and cache-directory selection, reducing the
inventory to 21 errors across 20 files; 71 focused runtime/cache/diagnostic
tests pass and the full suite remains green (1307 passed, 3 skipped).
The dates/business-calendar/yearly-parse/validation/command batch then
completed annotations for date arithmetic, business-calendar contexts, yearly
token parsing, modify validation guards, and Taskwarrior command effects,
reducing the inventory to 31 errors across 25 files; 91 focused date/calendar
and modify-command tests pass and the full suite remains green (1307 passed, 3
skipped).
The ACF/runtime/quarter-rewrite/precompute/spawn-prep batch then completed
annotations for canonical-form adapters, runtime diagnostics, quarter
rewriting, hint precomputation, and stable child-spawn preparation, reducing
the inventory to 43 errors across 30 files; 113 focused ACF/runtime/spawn
tests pass and the full suite remains green (1307 passed, 3 skipped).
The year-token/schedule/recurrence-metadata/query/hint batch then completed
annotations for yearly alias rewriting, schedule utilities, recurrence atom
metadata, modify query caching, and hint-builder orchestration, reducing the
inventory to 59 errors across 35 files; 121 focused scheduler/query/hint tests
pass and the full suite remains green (1307 passed, 3 skipped).
The quarter-helper/monthly/modify-UI/read/format batch then completed
annotations for quarter token helpers, nth-weekday month selection, UI effect
ports, lifecycle reads, and presentation formatting, reducing the inventory to
79 errors across 40 files; 70 focused modify/read/presentation tests pass and
the full suite remains green (1307 passed, 3 skipped).
The cache/token/time/quarter-selector/parser-frontend batch then completed
annotations for cache locking and persistence adapters, token normalization,
timezone utilities, quarter selection, and parser input normalization, reducing
the inventory to 133 errors across 50 files; 123 focused cache/time/parser
tests pass and the full suite remains green (1307 passed, 3 skipped).
The completion-preflight/hook/expansion/config/strict-validation batch then
completed annotations for completion guards, hook routing, calendar expansion,
configuration caching, and strict anchor validation, reducing the inventory to
104 errors across 45 files; 60 focused hook/config/completion tests pass and
the full suite remains green (1307 passed, 3 skipped).
