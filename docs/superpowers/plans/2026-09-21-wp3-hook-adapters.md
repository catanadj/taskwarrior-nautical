# WP3 Hook Adapter Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate add, modify, and exit hook startup/protocol behavior behind the shared hook runtime without changing observable hook behavior.

**Architecture:** Keep hook-specific policy and presentation in `nautical_core/hooks/*`. Make `hook_runtime.py` the single owner of module loading, Taskdata/integration-context setup, diagnostics, and profiler state. Make `hook_protocol.py`/`hook_results.py` the single owner of parsing, typed failures, and stdout/stderr serialization; executable entry points catch and serialize once.

**Tech Stack:** Python 3.11, `unittest`, Taskwarrior hook subprocess fixtures, strict mypy, JSON protocol.

**Spec:** `checklists/REPOSITORY_UPGRADE_REFACTOR_CHECKLIST.md`, Work Package 3.

## Global Constraints

- Hook stdout must contain only the required JSON document; diagnostics go to stderr only when `NAUTICAL_DIAG=1`.
- Malformed, truncated, oversized, and ordinary non-Nautical input must remain fail-safe and must not emit partial JSON.
- Preserve `ensure_ascii=False` for task JSON output and preserve read-only versus mutation access modes.
- Do not alter recurrence, lifecycle, or presentation behavior in WP3.
- Work offline on `feature/upgrade-refactor-preflight`; ignore unrelated untracked files.
- Compatibility bridges between intermediate commits are unnecessary; the branch may remain non-functional until the final gate.

## Review Focus

- Empty/truncated/extra JSON input: one typed protocol failure, no partial stdout.
- Unicode task fields: exact UTF-8 JSON round trip with `ensure_ascii=False`.
- Ordinary tasks: passthrough behavior and bounded startup cost remain unchanged.
- Core import/configuration failure: fail-safe response with diagnostics isolated to stderr.
- Diagnostic mode and panic paths: no stdout contamination and no sensitive-field leakage.

### Task 1: Freeze hook observables and baseline startup

**Files:**
- Modify: `tests/test_hook_input_contract.py`, `tests/test_hook_protocol.py`, `tests/test_hook_host_isolation.py`, `tests/test_on_add_hook_routes.py`, `tests/test_hook_process_contract.py`
- Test support: `tests/support/hook_process.py`

- [ ] Add/confirm explicit tests for empty, truncated, trailing, oversized, malformed, and Unicode input across add/modify/exit.
- [ ] Add a subprocess assertion that ordinary-task startup records import count/latency without changing stdout.
- [ ] Run the focused hook suite and record the baseline output in the checklist.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest \
  tests.test_hook_input_contract tests.test_hook_protocol \
  tests.test_hook_host_isolation tests.test_on_add_hook_routes \
  tests.test_hook_process_contract -v
```

### Task 2: Centralize runtime initialization

**Files:**
- Modify: `nautical_core/hook_runtime.py`, `nautical_core/hook_bootstrap.py`
- Modify callers: `nautical_core/hooks/add_impl.py`, `nautical_core/hooks/modify_impl.py`, `nautical_core/hooks/exit_impl.py`
- Test: `tests/test_hook_host_isolation.py`, `tests/test_bootstrap_contract.py`

- [ ] Define one typed runtime result carrying core target, Taskdata context, module access, access mode, and diagnostics/profiler state.
- [ ] Move duplicated `_resolve_task_data_context`, `_build_hook_runtime_context`, module loading, and integration-context setup into the shared owner.
- [ ] Update each hook to consume that result while retaining add read-only and modify/exit mutation capabilities.
- [ ] Add tests proving all three hooks use the same initializer and do not duplicate bootstrap paths.

### Task 3: Consolidate protocol failures and serialization

**Files:**
- Modify: `nautical_core/hook_protocol.py`, `nautical_core/hook_results.py`
- Modify entry points: `nautical_core/hooks/add_impl.py`, `nautical_core/hooks/modify_impl.py`, `nautical_core/hooks/exit_impl.py`
- Test: `tests/test_hook_input_contract.py`, `tests/test_hook_protocol.py`, `tests/test_hook_process_contract.py`

- [ ] Define the single typed failure/result contract used below executable entry points.
- [ ] Convert helper-level exits into returned failures or the typed boundary exception.
- [ ] Catch once in each executable `main`/`run_hook`, serialize the protocol-preserving response, and keep diagnostics on stderr.
- [ ] Add direct tests for panic, unavailable-core, and diagnostic-redaction paths.

### Task 4: Remove obsolete adapter state and complete annotations

**Files:**
- Modify: `nautical_core/hooks/add_impl.py`, `nautical_core/hooks/modify_impl.py`, `nautical_core/hooks/exit_impl.py`, `nautical_core/hook_runtime.py`, `nautical_core/hook_protocol.py`, `nautical_core/hook_results.py`
- Test: existing focused hook suites

- [ ] Remove imports, globals, synchronization lists, and private helpers made obsolete by the shared initializer.
- [ ] Add complete annotations to shared runtime/protocol functions and narrow broad exception handlers where stable exception types exist.
- [ ] Keep intentional fail-safe catches documented and directly tested.

Run:

```bash
/home/pooK/venv/test_1/bin/mypy --config-file mypy.ini \
  --disallow-untyped-defs --disallow-incomplete-defs \
  nautical_core/hooks nautical_core/hook_bootstrap.py \
  nautical_core/hook_runtime.py nautical_core/hook_protocol.py \
  nautical_core/hook_results.py
```

### Task 5: Final WP3 verification and checklist closure

- [ ] Run the focused hook suite, full unit suite, and deployment sanity check.
- [ ] Run the golden suite normally and shuffled to confirm no cross-domain behavior changed.
- [ ] Run `git diff --check` and update WP3 checklist counts/evidence.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest discover -s tests -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 dev_tools/nautical_deploy_sanity.py --json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 dev_tools/nautical_golden_tests.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 dev_tools/nautical_golden_tests.py --shuffle-seed 20260920
git diff --check
```
