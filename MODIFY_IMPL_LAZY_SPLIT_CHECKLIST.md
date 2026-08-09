# Lazy Modify Lifecycle Split

Optional follow-up performance work for slow devices. The current lazy hook
loader already defers the core package and queue adapters; this checklist
covers the deeper split of `nautical_core/hooks/modify_impl.py`.

## Goals

- Reduce cold-start parsing for ordinary, completion, and expiration edits.
- Preserve strict hook JSON behavior and fail-closed mutation decisions.
- Preserve shared runtime state, cache invalidation, diagnostics, and locking.
- Preserve public compatibility aliases and existing test monkeypatch points.

## Passes

- [~] Measure current cold imports and lifecycle timings on Linux and Termux.
- [x] Provide a focused lifecycle router (`hook_engine.handle_on_modify`) that parses the route and selects one lifecycle.
- [x] Move ordinary/non-completion modification handling into a focused module.
- [x] Move completion and chain-generation orchestration into focused modules.
- [x] Move expiration, deletion, and recovery handling into a focused module.
- [x] Keep `modify_impl.py` as an explicit compatibility facade during migration.
- [x] Make shared runtime state and service construction request-scoped and lazy.
- [x] Verify lifecycle imports remain lazy; the import probe confirms ordinary,
  expiration, completion, chain-generation, and queue modules stay unloaded.
- [~] Add cold-import, ordinary-edit, completion, expiration, and queue benchmarks.
- [~] Run the full golden suite, mypy, hook replay, and deployment sanity checks.
- [ ] Compare Termux results against the baseline before deciding whether to remove facade code.

`[~]` means the work is in progress. The existing hook engine now provides the
router role, and completion, expiration, and chain generation use focused
service modules. Ordinary modification handling and the final measurement
pass are still pending. The compatibility delegates are isolated in
`nautical_core/modify_generation_compat.py` and remain lazy-loaded so existing
imports and monkeypatch points continue to work.

### Current verification

- Full golden suite: 890/890 passing after the compatibility extraction.
- Hook import regression: full core facade remains deferred during hook-module
  loading.
- The Linux/Termux baseline and lifecycle-specific benchmark matrix are not
  yet fully recorded; the Linux baseline is now captured below, while the
  Termux values still need a rerun after the latest commit.

### Baseline (Linux, Python 3.11)

Seven cold subprocess samples gave these medians:

- `nautical_core` import: 57.2 ms
- `modify_impl.py` import: 35.3 ms
- CP completion: 0.567 s
- Anchor completion: 0.562 s
- Queue drain: 0.287 s
- Reconcile: 0.193 s

All measured workflows remained within their configured budgets. The existing
Termux reference run should be repeated after this split; its slowest measured
paths were `build_hints` at 5.43 s and `seasonal_build_hints` at 23.29 s.

## Completion Criteria

- No lifecycle behavior changes in golden and replay tests.
- No new stdout output from hooks; diagnostics remain opt-in on stderr.
- Existing compatibility imports and monkeypatch points continue to work.
- Measurable cold-start improvement on Termux without exceeding current reliability budgets.
