# Nautical Architecture Hardening Checklist

This checklist follows the architecture audit completed after the previous
hardening work. Work through the sections in priority order and commit each
completed pass independently.

## Baseline

Status: baseline capture and protocol verification are tracked in
`LIFECYCLE_ENGINE_READINESS_CHECKLIST.md` section 2.

- [ ] Record the current golden, black-box, deployment, mypy, and performance results before editing. *(readiness section 2)*
- [ ] Preserve strict hook JSON on stdout and emit diagnostics to stderr only when `NAUTICAL_DIAG=1`. *(readiness section 2)*
- [ ] Keep every mutation-sensitive Taskwarrior read fail-closed. *(readiness sections 3 and 4)*

## 1. Version Persistent Hint Caches

- [x] Add the parser, cache schema, scheduler, and installed release fingerprint to hint-cache keys.
- [x] Reuse the central semantic fingerprint rather than maintaining manual `parser:N` or `cache:N` markers.
- [x] Ensure a cache hit cannot bypass validation performed by the current parser and scheduler.
- [x] Add an upgrade regression test proving that an older valid cache entry becomes a miss after a semantic fingerprint change.
- [x] Add coverage for default, configured shared, source-checkout, and managed-release cache locations.
- [x] Verify malformed and obsolete entries remain safely quarantined.

Completion criteria:

- [x] No persistent hint entry can survive a scheduling-semantic change without revalidation.
- [x] Warm cache behavior and existing cache-corruption recovery remain intact.

## 2. Make Hint Performance Measurements Trustworthy

- [x] Isolate hint benchmarks from the user and repository cache directories.
- [x] Split `build_hints` and seasonal hint measurements into explicit cold-miss and warm-hit cases.
- [x] Use fresh cache keys or fresh cache directories for every cold sample.
- [x] Assert whether each sample was a cache hit or miss so the benchmark cannot measure the wrong path.
- [x] Add separate desktop and slow-device budgets for cold and warm construction.
- [x] Re-run the reduced benchmark on Termux and preserve the report outside version control.

Completion criteria:

- [x] Repeated benchmark runs produce comparable results regardless of pre-existing cache files.
- [x] CI can fail on a cold hint-generation regression.

## 3. Reduce Cold Hint-Generation Cost

- [x] Profile normal and seasonal cold hint generation on Termux after fixing the benchmark.
- [x] Identify whether the dominant cost is occurrence scanning, selection evaluation, astronomy, cache locking, or file I/O.
  - Profile result: repeated scheduler occurrence lookup dominates; cache locking and file I/O are not material on the current workloads.
- [x] Avoid computing annual statistics that are not consumed by the active panel mode or hook decision.
- [x] Reuse scheduler/evaluator state within one hint build without sharing mutable task state.
- [x] Preserve exact dates, natural text, limits, random determinism, and terminal exhaustion evidence.

Completion criteria:

- [x] Cold normal and seasonal hint generation satisfy the agreed slow-device budgets.
- [x] Golden output and scheduling behavior remain unchanged.

## 4. Unify the Hook Command Boundary

Status: this is a lifecycle prerequisite and is tracked in
`LIFECYCLE_ENGINE_READINESS_CHECKLIST.md` section 3. The existing
`TaskCommandResult` implementation is not yet end-to-end because legacy tuple
adapters remain in hook query paths.

- [ ] Make `TaskCommandResult` the end-to-end result type for add, modify, and exit Taskwarrior subprocesses. *(readiness section 3)*
- [ ] Preserve return code, failure kind, attempt count, timeout, stdout, and stderr without reconstructing them from text. *(readiness section 3)*
- [ ] Retry only failures explicitly classified as retryable. *(readiness section 3)*
- [ ] Convert mutation-sensitive reads directly into found, absent, or unavailable results. *(readiness section 3)*
- [ ] Remove duplicate UUID reads caused by the tuple compatibility path. *(readiness section 3)*
- [ ] Remove obsolete tuple adapters only after all hook callers use the typed boundary. *(readiness section 3)*
- [ ] Add tests for missing binaries, timeouts, locks, nonzero exits, empty output, malformed JSON, and noisy stderr. *(readiness section 3)*

Completion criteria:

- [ ] No mutation decision depends on matching words such as `lock` or `timeout` in untyped command output. *(readiness section 3)*
- [ ] Unavailable reads always reject or defer mutation. *(readiness section 3)*

## 5. Reduce Full Nautical Hook Startup Cost

Status: import deferral that is already implemented is marked below. Remaining
startup measurement and configuration-safety work is split between the
lifecycle readiness checklist and deferred post-lifecycle work.

- [x] Capture import-time profiles for `nautical_core`, add, modify, and exit on desktop and Termux.
- [ ] Reproduce the slower device result with enforced, repeated source, staged, and managed-hook measurements. *(deferred until lifecycle boundaries stabilize)*
- [x] Record cold-import module counts alongside timings so filesystem/process variance is distinguishable from dependency growth.
- [x] Defer automatic configuration discovery and `_CONF` construction until a scheduling API requires them.
- [ ] Defer `core_config` import and `_CONF` construction until an API actually requires scheduling configuration. *(deferred until lifecycle boundaries stabilize)*
- [ ] Ensure add, modify, completion, reconcile, doctor, and navigator explicitly resolve validated configuration before scheduling or mutation. *(moved to `LIFECYCLE_ENGINE_READINESS_CHECKLIST.md` section 4)*
- [ ] Preserve fail-closed handling for malformed TOML, invalid timezones, astronomy profiles, calendars, and recurrence presets after deferral. *(moved to `LIFECYCLE_ENGINE_READINESS_CHECKLIST.md` section 4)*
- [x] Make parser, scheduler, expansion, and linting APIs own their standard-library dependencies explicitly.
- [x] Make ACF and precompute APIs own their standard-library dependencies explicitly.
- [x] Make cache and ACF codec dependencies local while preserving injectable test and monkeypatch overrides.
- [x] Remove facade imports with no runtime or compatibility consumers.
- [x] Remove datetime and calendar facade imports after explicit API ownership was established.
- [x] Lazy-resolve the diagnostic model while preserving its public export and hook behavior.
- [x] Lazy-resolve parser and scheduler model modules while preserving public exception/type names.
- [ ] Move heavyweight standard-library imports out of `nautical_core/__init__.py` and into their owning modules or lazy call sites. *(deferred until lifecycle boundaries stabilize)*
- [x] Lazy-resolve public parser, scheduler, and diagnostic model exports without breaking public names, monkeypatch points, or type checking.
- [x] Make remaining business-calendar, presentation, cache, and lifecycle-specific imports lazy where measurement shows a benefit.
  - [x] Defer `business_calendar_config` while preserving its public error export.
  - [x] Defer `season_support` until hemisphere configuration or seasonal evaluation is requested.
  - [x] Defer `business_calendar` until its default calendar or displacement helpers are requested.
  - [x] Defer `tokenutil` tables until a parser, expansion, linting, or token API is resolved.
  - [x] Defer `runtime` diagnostics/context helpers behind forwarding facade functions.
  - [x] Defer `cp_parser` behind forwarding facade functions with preserved signatures.
  - [x] Defer `dates` and `timeutil` adapters until their public helpers are used.
  - [x] Defer `recurrence_metadata` behind signature-preserving metadata helpers.
  - [x] Defer `compat_api` export metadata through a tuple-like lazy `__all__`.
- [ ] Measure each extraction independently and retain it only when cold startup or module count improves without shifting cost into ordinary hooks. *(deferred until lifecycle module boundaries stabilize)*
- [ ] Profile the managed launcher separately for path resolution, environment setup, runtime discovery, and hook dispatch overhead. *(deferred until lifecycle module boundaries stabilize)*
- [ ] Keep the managed-layout ratio check stable against warmup and filesystem jitter while still detecting real deployment regressions. *(deferred until lifecycle module boundaries stabilize)*
- [x] Retain deployment-manifest validation for every newly lazy module.
- [x] Re-run full CP completion, anchor completion, queue drain, and reconcile benchmarks after each split.
- [ ] Re-run golden, black-box, deployment, mypy, hook-protocol, and workflow performance checks after the final split. *(handoff verification is tracked in `LIFECYCLE_ENGINE_READINESS_CHECKLIST.md` section 8)*
- [ ] Re-run the enforced benchmark on both Termux devices and preserve both reports outside version control. *(handoff verification is tracked in `LIFECYCLE_ENGINE_READINESS_CHECKLIST.md` section 8)*

Completion criteria:

- [ ] Cold core and full-hook imports meet the agreed budgets on repeated runs from both Termux devices. *(deferred)*
- [ ] Configuration is loaded only when required and every scheduling entry point still fails closed on unavailable configuration. *(configuration safety is readiness section 4; import timing is deferred)*
- [ ] Cold-import module counts do not regress and no deferred module is missing from staged releases. *(deferred)*
- [ ] Thin ordinary-task routing and full Nautical behavior remain unchanged. *(readiness handoff section 8)*

## 6. Isolate the Golden Regression Suite

Status: lifecycle-specific isolation is tracked in
`LIFECYCLE_ENGINE_READINESS_CHECKLIST.md` section 5. General suite isolation
remains deferred until after lifecycle consolidation.

- [ ] Make tests restore modified environment variables, module globals, caches, and configuration state. *(lifecycle subset: readiness section 5; broader suite deferred)*
- [ ] Make tests initialize the lazy APIs they use rather than relying on an earlier test. *(lifecycle subset: readiness section 5; broader suite deferred)*
- [ ] Split process-global hook tests into isolated groups where restoration is impractical. *(deferred outside lifecycle tests)*
- [ ] Add a deterministic shuffled-order CI run or an equivalent state-leak detector. *(lifecycle subset: readiness section 5; broader suite deferred)*
- [ ] Treat unexpected warnings from deleted temporary Taskdata or configuration paths as failures. *(lifecycle subset: readiness section 5; broader suite deferred)*
- [x] Keep registry enforcement for every top-level golden test.

Completion criteria:

- [ ] Registered and deterministic shuffled orders both pass without state-dependent warnings. *(lifecycle subset: readiness section 5)*
- [ ] Running a focused test produces the same result as running it in the complete suite. *(lifecycle subset: readiness section 5)*

## 7. Expand Strict Type Boundaries

Status: only command, lookup, configuration, and child-generation typing is a
pre-lifecycle gate; it is tracked in
`LIFECYCLE_ENGINE_READINESS_CHECKLIST.md` section 7. Broad parser and preview
typing remains deferred.

- [ ] Enable consequential mypy checks for `hook_support` and `runtime_command`. *(readiness section 7)*
- [ ] Then enable them for `occurrence_provider`, `anchor_inclusion`, and `add_anchor_preview` after lifecycle boundaries stabilize. *(deferred)*
- [ ] Type fatal callbacks as `NoReturn` so successful paths do not remain spuriously nullable. *(deferred)*
- [ ] Replace heterogeneous occurrence unions with explicit result models where needed. *(lifecycle prerequisite only: readiness section 7)*
- [ ] Reduce `Callable[..., Any]` at scheduler and preview orchestration boundaries. *(deferred)*
- [ ] Remove global error-code suppressions only after covered modules pass independently. *(deferred)*

Completion criteria:

- [ ] Strict command, configuration, lookup, and child-generation modules pass `arg-type`, `return-value`, `assignment`, `union-attr`, `attr-defined`, and `operator` checks. *(readiness section 7)*
- [ ] No typing-only cast or assertion weakens runtime validation. *(readiness section 7)*

## 8. Preserve Scheduler Terminal Evidence

Status: the mutation-relevant subset is tracked in
`LIFECYCLE_ENGINE_READINESS_CHECKLIST.md` section 6. Preview- and timeline-only
propagation remains part of the scheduler follow-up.

- [x] Annotate evaluator collection APIs as returning `OccurrenceBatch` rather than a plain list.
- [ ] Preserve `OccurrenceSearchExhausted` evidence through chainMax and chainUntil mutation decisions. *(readiness section 6; preview/timeline propagation deferred)*
- [ ] Define which presentation consumers may intentionally discard terminal evidence. *(deferred)*
- [x] Add regressions for a valid prefix followed by date-limit exhaustion and search-limit exhaustion.

Completion criteria:

- [ ] A valid occurrence prefix cannot be mistaken for an ordinary complete result in mutation paths. *(readiness section 6)*
- [ ] Search exhaustion never becomes silent absence or a fabricated occurrence in mutation paths. *(readiness section 6)*

## 9. Expand Runtime Compatibility CI

Status: deferred until lifecycle consolidation; the readiness checklist records
the current runtime baseline but does not block the lifecycle planner.

- [ ] Add lightweight Python 3.13 and 3.14 jobs for core parsing, golden tests, hook protocol, and deployment sanity. *(deferred)*
- [ ] Keep the supported Astral floor and current Astral release covered on newer Python versions. *(deferred)*
- [ ] Add a reduced slow-device performance profile as a non-enforced trend artifact until stable budgets are established. *(deferred)*
- [ ] Document the minimum supported Python and Taskwarrior versions in one authoritative location used by CI and doctor. *(deferred)*

Completion criteria:

- [ ] The Python version used by current Termux installations is represented in CI. *(deferred)*
- [ ] Runtime, dependency, and deployment matrices cannot drift silently. *(deferred)*

## Final Verification

Status: the pre-lifecycle subset is tracked in
`LIFECYCLE_ENGINE_READINESS_CHECKLIST.md` section 8. The remaining compatibility
and broad startup checks resume after lifecycle module ownership is stable.

- [x] Run `python3 dev_tools/nautical_golden_tests.py`. *(2026-08-11: 936/936 passed; readiness section 8)*
- [x] Run `python3 dev_tools/nautical_black_box_test.py --json` with Taskwarrior available. *(2026-08-11: passed CP, preset, anchor-file, modify, navigator, duplicate guard, and queue drain scenarios; readiness section 8)*
- [x] Run `python3 dev_tools/nautical_deploy_sanity.py`. *(2026-08-11: passed; readiness section 8)*
- [x] Run `python3 -m mypy --config-file mypy.ini`. *(2026-08-11: no issues in 136 source files; readiness section 8)*
- [x] Run `python3 dev_tools/nautical_perf_budget.py --json --enforce`. *(2026-08-11: 35 checks passed with no failed checks; readiness section 8)*
- [ ] Run extended workflow and slow-device performance profiles. *(deferred)*
- [x] Run doctor and reconcile dry-run smoke tests against isolated Taskdata. *(2026-08-11: isolated reliability/queue and reconcile suites passed; readiness section 8)*
- [x] Confirm hook stdout remains strict JSON for valid, malformed, and failing inputs. *(2026-08-11: hook protocol, golden, and black-box checks passed; readiness section 8)*
- [ ] Remove this checklist from the repository after every item and completion criterion is satisfied. *(after readiness and lifecycle checklists are complete)*
