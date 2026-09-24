# Nautical Navigator Package Extraction Checklist

This is a future structural project. Nautical Navigator remains part of the
supported hook/scheduling tooling while this checklist is incomplete.

## 1. Establish the package boundary

- [ ] Create `nautical_core/navigator/` as the owned application package.
- [ ] Define package-level interfaces for snapshot loading, scheduling,
  calendar analysis, chain analysis, and presentation.
- [ ] Keep `nautical_navigator.py` as a thin executable launcher and temporary
  compatibility import surface.
- [ ] Add an architecture test preventing navigator implementation code from
  moving back into the root launcher.

## 2. Extract immutable models and analysis

- [ ] Move navigator view models and metadata types from
  `nautical_navigator.py` into focused package modules.
- [ ] Move `analyze_navigator_snapshot()` behind an explicit snapshot and
  collaborator contract.
- [ ] Preserve deterministic serialization and Unicode behavior.
- [ ] Update `tests/test_navigator_analysis.py` with direct package-owner tests.

## 3. Extract presentation and trace services

- [ ] Move anchor preview and presentation result construction into a focused
  presentation module.
- [ ] Move trace aggregation, summaries, and explanation rendering into a
  trace/presentation owner.
- [ ] Ensure presentation code receives data and ports explicitly, without
  importing the root launcher or mutable facade state.
- [ ] Preserve operator text, diagnostics, and terminal output contracts.

## 4. Extract TaskAnalyzer responsibilities

- [ ] Separate Taskwarrior discovery/loading from analysis and projection.
- [ ] Move chain/calendar analysis into dedicated navigator services.
- [ ] Move rendering and interactive navigation into presentation adapters.
- [ ] Keep scheduling behavior delegated to the existing scheduler service;
  do not duplicate recurrence logic.

## 5. Rewire the launcher

- [ ] Replace root-level implementations with imports from
  `nautical_core.navigator`.
- [ ] Preserve executable invocation and documented command examples.
- [ ] Preserve any intentionally supported compatibility imports during the
  transition, with a removal deadline recorded in this checklist.
- [ ] Verify no package module imports implementation symbols back from the
  root launcher.

## 6. Verification and release gate

- [ ] Run `tests/test_navigator_analysis.py`.
- [ ] Run `tests/test_navigator_budget.py`.
- [ ] Run navigator CLI smoke tests in installed and source layouts.
- [ ] Run the complete unit suite and golden suites.
- [ ] Run compilation, mypy, deployment sanity, and black-box checks.
- [ ] Run Desloppify scan/review after the extraction is complete.
- [ ] Remove the navigator compatibility layer only after all callers and
  documented imports have migrated.

## Completion criteria

The root launcher contains only startup, argument handling, and compatibility
exports; navigator analysis, scheduling coordination, and presentation owners
live under `nautical_core/navigator/`; all existing behavior and verification
gates remain green.
