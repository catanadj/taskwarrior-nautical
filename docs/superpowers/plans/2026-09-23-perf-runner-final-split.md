# Performance Runner Final Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Leave `nautical_perf_budget.py` as a thin dispatcher while preserving every workload key, report field, correctness guard, and budget decision.

**Architecture:** Move workflow context/setup and final result assembly into `dev_tools/perf/workflow_workloads.py`; retain the CLI parser and output formatting in `nautical_perf_budget.py`. Existing extracted workload helpers remain the ownership boundaries.

**Tech Stack:** Python 3, `unittest`, Taskwarrior subprocess fixtures, JSON reports.

## Global Constraints

- Preserve strict JSON on machine-readable output.
- Preserve workload names, budget keys, exit codes, deterministic fixtures, and report schemas.
- Do not run benchmark work when workload modules are imported.
- Keep subprocess diagnostics out of benchmark JSON output.

## Review Focus

- Workflow configuration disabled: returns an empty result without creating fixtures.
- Taskwarrior unavailable or failing: surfaces an actionable benchmark error.
- Empty and candidate reconcile reports: retain schema and compact report fields.
- Partial queue recovery: preserves merged timing and task/outbox statistics.
- CLI JSON output: remains valid JSON with identical workload coverage.

### Task 1: Extract workflow context and result assembly

**Files:** `dev_tools/perf/workflow_workloads.py`, `dev_tools/nautical_perf_budget.py`, `tests/test_perf_budget_contract.py`

- Add an import-safe orchestration function receiving explicit fixture, workload, measurement, and reporting dependencies.
- Route the legacy runner through it and remove duplicate setup/result code.
- Add a contract asserting the orchestration entry point is callable and import-safe.
- Verify focused contracts, workflow-only JSON, and full discovery.

### Task 2: Thin the CLI and remove migration scaffolding

**Files:** `dev_tools/nautical_perf_budget.py`, checklist documentation

- Delete obsolete private wrappers and dead compatibility scaffolding created during extraction.
- Keep argument parsing, budget loading, dispatch, and final rendering only.
- Compare workload-name coverage and pass/fail decisions before and after.
- Verify focused contracts, workflow-only JSON, full discovery, and diff cleanliness.
