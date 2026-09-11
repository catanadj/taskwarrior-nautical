# Test Strategy, API Coherence, and Architecture Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Nautical's Test strategy, API coherence, Cross-module
architecture, and the adjacent Test health, elegance, abstraction, contract,
error, and security dimensions with evidence-backed changes that increase
reliability without gaming coverage or removing compatibility.

**Architecture:** First promote high-risk behavior into directly discoverable
tests. Then standardize shared result and datetime contracts, replace facade
dictionaries and hook-host reach-through with typed subsystem dependencies,
validate lifecycle capabilities at composition, and narrow presentation
contexts. Retain compatibility only at explicit external boundaries and
adjudicate security signals from code evidence before the final blind review.

**Tech Stack:** Python 3.11+, `unittest`, typed dataclasses and protocols,
Taskwarrior subprocess integration, the existing golden/black-box/stress/soak
tools, mypy, and desloppify.

**Spec:** `.desloppify/query.json`,
`.desloppify/subagents/runs/20260909_173534/holistic_issues_merged.json`,
`checklists/DESLOPPIFY_REVIEW_REMEDIATION_CHECKLIST.md`, and the repository
`AGENTS.md`.

## Global Constraints

- Keep Taskwarrior hook output strict JSON on stdout; send diagnostics to stderr
  only when `NAUTICAL_DIAG=1`.
- Be defensive with malformed hook input and never leak a traceback through a
  hook protocol boundary.
- Preserve `ensure_ascii=False` for all hook and operator JSON output.
- Preserve recurrence semantics, omission behavior, scheduler terminal
  evidence, lifecycle idempotency, mutation guards, and postcondition checks.
- Work on the existing dedicated remediation branch or another explicitly
  dedicated branch. Keep commits local until the final integration decision.
- Add a failing characterization or contract test before each behavioral or
  architectural change.
- Use direct behavioral tests; an import-only smoke test does not close a
  direct-coverage finding.
- Migrate tests rather than duplicating them. Remove each migrated golden test
  from the custom registry in the same logical change.
- Keep compatibility adapters at explicit external boundaries until installed
  hooks, operator tools, and documented public imports pass.
- Do not resolve desloppify findings or claim score improvement until a fresh
  scan or blind review confirms the result.
- Avoid unrelated cleanup. Every changed line must support a checklist item.

---

## Audit Baseline and Score Constraints

- [ ] Record the branch, revision, Python version, worktree state, and current
  desloppify scores before implementation.
- [ ] Preserve the audit baseline in the implementation notes:
  - Overall: `84.7/100`
  - Objective: `91.1/100`
  - Strict: `84.3/100`
  - Test strategy: `67.0%`
  - API coherence: `76.0%`
  - Cross-module architecture: `77.0%`
  - Test health: `70.8%`, strict `65.8%`
  - Mid-level elegance: `78.0%` (configured weight `22`)
  - High-level elegance: `84.0%` (configured weight `22`)
  - Low-level elegance: `76.5%` (configured weight `12`)
  - Contract coherence: `83.0%` (configured weight `12`)
  - Abstraction fitness: `78.0%` (configured weight `8`)
  - Package organization: `80.0%` (configured weight `5`)
  - Error consistency: `78.0%` (configured weight `3`)
  - Security: `98.0%`, strict `97.8%`
  - Direct/transitive coverage findings: `149` (`25` Tier 2, `124` Tier 3)
  - Standard unittest discovery baseline: `518` passing tests
  - Golden registry inventory: `995` registered tests in a `37,646`-line file
- [ ] Record that the three requested subjective dimensions have weight `1`
  each in a subjective pool whose total configured weight is `123`.
- [ ] Do not promise that these three dimensions alone can reach strict `85.0`:
  raising all three to `100` has a theoretical maximum overall gain of about
  `0.49` points.
- [ ] Use direct-test remediation to improve Test health as well as Test
  strategy. Desloppify estimates that resolving all 149 coverage findings is
  worth about `1.2` overall points.
- [ ] Treat the following as theoretical prioritization ceilings, not promised
  score gains: Mid-level elegance about `2.95` overall points, high-level
  elegance `2.15`, low-level elegance `1.72`, contract coherence `1.24`,
  abstraction fitness `1.07`, package organization `0.61`, and error
  consistency `0.40` if each dimension independently reached `100`.
- [ ] Record that most of these subjective assessments are stale after the
  current remediation work. Do not infer remaining defects from a stale score;
  require current code evidence or a fresh blind review.

```bash
git branch --show-current
git rev-parse HEAD
git status --short
python3 --version
/home/pooK/venv/test_1/bin/desloppify status
/home/pooK/venv/test_1/bin/desloppify show test_coverage --status open --top 200 --no-budget
/home/pooK/venv/test_1/bin/desloppify show review --status open --top 200 --no-budget
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest discover -s tests -q
```

### Findings already implemented or stale

- [ ] Do not reimplement the retired `holiday_region`, deferred configuration
  loading, optional Astral, structured outbox failure, panel diagnostics,
  canonical `OmitState`, normalization naming, or lifecycle drain-limit work
  unless a current focused regression demonstrates a defect.
- [ ] Keep those findings visible in audit notes until a fresh blind review
  replaces the stale assessment; distinguish stale review state from missing
  production behavior.
- [ ] Preserve focused evidence for already-implemented adjacent-dimension
  fixes: explicit `_in_place` business-calendar normalization, canonical
  `OmitState`, deferred configuration loading, optional Astral installation,
  retired `holiday_region` cache removal, and structured maintenance and panel
  source failures.

### Adjacent-dimension regression evidence

- [ ] Verify `normalize_task_business_calendar_in_place(...)` both returns the
  selected calendar and canonicalizes a present `bc` value, while the legacy
  `normalize_task_business_calendar` alias has identical compatibility
  behavior.
- [ ] Verify `combine_omit_state(...)` returns `OmitState | None` for every
  primary caller and legacy raw/dictionary decoding occurs only in the named
  compatibility split boundary.
- [ ] Verify importing `nautical_core` with `NAUTICAL_CONFIG` set performs no
  configuration-file read; first supported configuration access performs the
  read and surfaces validation failure through the documented diagnostic path.
- [ ] Verify the base requirements omit Astral, the astronomy requirements pin
  it, the offline kit includes both install modes, and astronomy-disabled
  operation never imports Astral.
- [ ] Verify changing retired `holiday_region` cannot change the scheduler or
  cache fingerprint while deprecated-key recognition remains available at the
  configuration boundary.
- [ ] Verify outbox permission/setup failures retain the stable
  `filesystem_security_failure` classification and always close an opened
  connection.
- [ ] Verify missing optional panel sources remain non-fatal, while permission,
  parsing, and unexpected loader failures are surfaced with source context and
  never silently converted into an empty warning list.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_cache_legacy_region -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_runtime_initialization_boundary -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_offline_kit -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_structured_failure_boundaries -v
```

---

## Work Package 1 — Direct Natural-Language Contract Tests

**Status: IMPLEMENTATION COMPLETE.** Direct behavioral coverage and the
natural-language migration are complete. Fresh desloppify confirmation remains
an audit gate, not an implementation task.

**Files:**

- Create: `tests/test_natural_language_contract.py`
- Inspect and test directly: `nautical_core/natural_language.py`
- Use public composition only for parity: `nautical_core/natural_language_api.py`
- Migrate relevant cases from: `dev_tools/nautical_golden_tests.py`

**Interfaces:**

- Consumes: production parser and natural-language bindings.
- Produces: directly discoverable behavioral coverage for
  `describe_anchor_expr(...)`, `describe_anchor_dnf(...)`, and shared-tail
  compression.

- [x] Add a table-driven test importing `nautical_core.natural_language`
  directly and exercising weekly, monthly, yearly, interval, roll, time-window,
  random, and malformed inputs through the production binding.
- [x] Include exact expectations for at least these public expressions:

```python
cases = {
    "w:mon": "Mondays",
    "m:1": "the 1st day of each month",
    "w:mon|w:fri": "either Mondays or Fridays",
    "w/2:mon": "every 2 weeks: Mondays",
    "malformed": "",
}
```

- [x] Add mode-tail assertions for `skip`, `flex`, and `all` using
  `describe_anchor_dnf(...)`.
- [x] Promote the existing golden cases for interval OR phrasing, repeated
  `within` clauses, repeated `that fall on` clauses, previous/next weekday
  rolls, multiple times, and bounded time windows.
- [x] Assert the direct implementation and the public facade return identical
  text for each supported case.
- [ ] Run the focused test and confirm it fails for missing direct coverage or
  any behavioral mismatch before changing production code.
- [ ] Remove only the migrated golden registry entries after the direct tests
  pass in normal and shuffled order.
- [ ] Resolve
  `review::.::holistic::test_strategy::natural_language_formatter_untested`
  only after a fresh scan recognizes direct behavioral coverage.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_natural_language_contract -v
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py --only natural --verbose
```

**Done when:** User-visible recurrence explanations have direct, table-driven
coverage outside the golden monolith, including malformed and compression
branches.

---

## Work Package 2 — Executable On-Add Route Matrix

**Status: IMPLEMENTATION COMPLETE.** The route matrix, strict JSON isolation,
failure cases, and golden migration are complete. Fresh desloppify confirmation
remains an audit gate.

**Files:**

- Create: `tests/test_on_add_hook_routes.py`
- Reuse protocol helpers from: `tests/test_hook_input_contract.py`
- Exercise: `on-add.nautical`, `nautical_core/hooks/add_impl.py`
- Use temporary anchor files and configuration only.

**Interfaces:**

- Consumes: executable hook stdin/stdout/stderr contract.
- Produces: deterministic route coverage for ordinary, CP, anchor, and
  anchor-file additions.

- [x] Extract a shared subprocess fixture that creates temporary Taskdata,
  writes configuration and recurrence files, sets trusted core paths, and
  captures text-mode stdout/stderr with a 15-second timeout.
- [x] Add a valid-route matrix containing:
  - ordinary task passthrough;
  - `cp=1d` with implicit due;
  - `cp=1d` with explicit due;
  - `anchor=w:mon`;
  - an `anchor_file` backed by a temporary CSV or text file;
  - a scheduled-only recurrence;
  - a recurrence with `chainMax` and `chainUntil`;
  - Unicode description and recurrence-file description values.
- [x] Require exit status `0` for every valid route. Replace the permissive
  `assertIn(returncode, (0, 1))` assertion in the existing anchor test.
- [x] Parse stdout with `json.loads` and assert exact preservation or mutation
  of `chain`, `chainID`, `link`, recurrence fields, `due`, and `scheduled` for
  each route.
- [x] Assert preview/panel text never contaminates stdout. Treat operator UI on
  stderr separately from opt-in diagnostic messages.
- [x] Add invalid-route cases for malformed CP, malformed anchor, missing
  anchor file, invalid limit, and conflicting recurrence kinds. Require
  non-zero status, empty stdout, actionable stderr, and no traceback.
- [x] Run every route once without `NAUTICAL_DIAG` and once where applicable
  with `NAUTICAL_DIAG=1`; diagnostic lines must remain on stderr.
- [x] Migrate equivalent executable on-add cases from the golden registry and
  remove their old registrations in the same change.
- [ ] Resolve `review::.::holistic::test_strategy::add_hook_branches_untested`
  only after the executable matrix passes and the fresh scan sees it.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_hook_input_contract tests.test_on_add_hook_routes -v
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py --only hook_on_add --verbose
```

**Done when:** Every supported on-add route has one success case, one relevant
failure case, and strict JSON/diagnostic isolation assertions.

---

## Work Package 3 — Risk-Ordered Direct Test Migration

**Status: IMPLEMENTATION SLICES COMPLETE; WAVE GATE PENDING.** Cache, parser,
scheduler, calendar, add/preview, chain recovery, configuration/time, operator,
deployment, and stress slices have been migrated and verified. The aggregate
wave gate and fresh scan remain outstanding.

**Files:** Create focused `tests/test_<owner>.py` modules and migrate matching
tests from `dev_tools/nautical_golden_tests.py`.

**Interfaces:**

- Consumes: public functions/classes from the production module named by each
  test file.
- Produces: direct behavioral coverage recognized by unittest, coverage tools,
  and desloppify.

### Wave A — Pure and deterministic owners

- [ ] Add direct tests for `natural_language_api.py`, `quarter_api.py`,
  `acf_api.py`, `expansion_api.py`, and `business_calendar_api.py`.
- [x] Add direct tests for `parsing/parser_support_api.py` and
  `parsing/parser_frontend.py`.
- [x] Add direct tests for `scheduler_atom.py` and `scheduler_api.py`.
- [x] Add direct cache contract tests for `cache_api.py`: miss/hit, corrupt
  payload, quarantine, lock refusal, Unicode, configuration fingerprint, and
  isolated per-instance state.
- [ ] Add deterministic boundary matrices rather than import-only assertions.
  Every test module must call at least one behavior owned by its production
  module.

### Wave B — Application and composition owners

- [ ] Add direct tests for `add_anchor_compute.py`, `add_anchor_preview.py`,
  `add_preview_composition.py`, and `chain_generation.py`.
- [ ] Cover valid result, malformed input, exhausted provider, omitted
  occurrence, date limit, and injected dependency failure for applicable
  services.
- [ ] Add direct tests for `business_calendar_config.py`, `timeutil.py`, and
  `chain_integrity_recovery.py` using temporary files and immutable fixtures.

### Wave C — Operator and operational entry points

- [ ] Add direct command-contract tests for
  `nautical_core/tools/nautical_query.py`, `nautical_doctor.py`, and
  `nautical_reconcile.py` using temporary Taskdata and injected/stubbed process
  boundaries where available.
- [ ] Test `dev_tools/nautical_deploy_sanity.py` as a callable report producer,
  including one passing inventory and one deliberately broken temporary
  inventory.
- [ ] Test `dev_tools/nautical_reliability_smoke.py` argument validation and
  result classification without operating on live Taskdata.
- [ ] Test the stress campaign's profile selection, validation, JSON envelope,
  and non-zero enforcement result without running a long campaign.

### Wave gates

- [ ] After each wave, run unittest discovery and the relevant golden slice.
- [ ] Run desloppify after the complete wave, not after every file; record the
  exact reduction in Tier-2 and Tier-3 coverage findings.
- [ ] Do not mark the wave complete if the direct-test count improved but the
  exercised branches did not include failures and edge conditions.

**Done when:** All 25 Tier-2 coverage findings are either closed by direct
behavioral tests or documented with evidence that the detector cannot represent
the executable/shell boundary; no item is closed with an import-only test.

---

## Work Package 4 — Decompose the Golden Test Monolith

**Status: COMPLETE.** Package discovery, registry integrity coverage, inventory
documentation, and the retained golden boundary are implemented. The known
full-suite recurrence-terminal failure is pre-existing and tracked separately.

**Files:**

- Modify: `dev_tools/nautical_golden_tests.py`
- Create packages under: `tests/recurrence/`, `tests/hooks/`,
  `tests/lifecycle/`, `tests/operators/`, and `tests/cache/`
- Create shared test-only support under: `tests/support/`

**Interfaces:**

- Consumes: existing golden functions and their fixture dependencies.
- Produces: normally discoverable domain tests plus a smaller acceptance-only
  golden runner.

- [x] Add `__init__.py` files so standard unittest discovery descends into each
  test package.
- [ ] Move reusable builders—not assertions or production behavior—into focused
  test-only support modules for Taskdata, tasks, clocks, recurrence files,
  subprocesses, and lifecycle outboxes.
- [ ] Migrate one domain at a time in this order: parser/natural language,
  scheduler/occurrences, cache, add hook, modify hook, lifecycle/outbox,
  operators/reconcile, installation/backup/restore.
- [ ] Preserve each test's deterministic input and expected outcome. Replace
  the custom `expect(...)` helper with the corresponding `unittest.TestCase`
  assertion.
- [ ] Remove migrated functions from `TESTS`/`DEEP_TESTS` immediately so CI does
  not execute duplicate tests.
- [ ] Keep only end-to-end acceptance, installed-layout, cross-process race,
  and long-running compatibility scenarios in the golden runner.
- [ ] Preserve normal and seeded shuffled execution until no shared mutable
  state remains in the migrated domains.
- [x] Add a registry-integrity check proving every function left in the golden
  file is registered exactly once.
- [ ] Record per-domain unit count, golden count, runtime, and any intentionally
  retained golden scenarios.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest discover -s tests -q
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py --shuffle-seed 20260811
```

**Done when:** Unit and contract behavior is discoverable under `tests/`; the
golden runner contains only scenarios that genuinely require its acceptance or
cross-process harness.

---

## Work Package 5 — Coverage and Test-Quality Ratchets

**Files:**

- Create: `.coveragerc`
- Create or modify: an explicit development-test requirements file
- Modify: `.github/workflows/type-check.yml`
- Create: `tests/test_architecture_contract.py`

**Interfaces:**

- Consumes: standard unittest discovery and production package imports.
- Produces: branch-coverage evidence, a non-decreasing coverage floor, and
  architecture regression enforcement.

- [ ] Add `coverage.py` only as a development/CI dependency; do not add it to
  Nautical's runtime installation requirements.
- [ ] Configure branch coverage for `nautical_core`, excluding generated,
  archived, and test-only compatibility material already outside production
  scope.
- [ ] Capture the actual initial branch-coverage result and set the CI floor to
  that measured integer value. Do not invent a target above the baseline.
- [ ] Fail CI when total branch coverage falls below the recorded floor and
  upload the text/XML report for diagnosis.
- [ ] Increase the floor only after a merged batch produces a stable higher
  baseline.
- [ ] Add deterministic parser/scheduler invariants using the standard library:
  canonical round-trip, strictly advancing occurrences, stable seeded random
  selection, and no result beyond an explicit date limit.
- [ ] Keep stress and soak tests outside the fast unit gate; retain their
  existing enforced CI/nightly roles.

```bash
PYTHONDONTWRITEBYTECODE=1 coverage run --branch --source=nautical_core -m unittest discover -s tests -q
coverage report --show-missing
```

**Done when:** CI detects lost branch coverage, coverage can only ratchet upward,
and deterministic recurrence invariants fail on semantic regressions.

---

## Work Package 6 — One Operator Result Contract

**Status: COMPLETE.** Query, Doctor, and reconcile now share the typed operator
result boundary; focused process, presentation, and reconcile evidence passed.

**Files:**

- Modify: `nautical_core/query_report.py`
- Inspect/update: `nautical_core/doctor_report.py`,
  `nautical_core/reconcile_report.py`
- Modify callers: `nautical_core/tools/nautical_query.py`
- Test: `tests/test_operator_process_contract.py`

**Interfaces:**

- Produces:
  `to_operator_result(payload: Mapping[str, Any]) -> OperatorV2Result` in all
  three report modules.
- Serialization remains owned by `operator_presentation.render_result(...)` or
  `render_json_document(...)` at transport boundaries.

- [x] Add a failing contract test importing all three converters and asserting
  that each returns `OperatorV2Result`, never an already serialized dictionary.
- [x] Change `query_report.to_operator_result` to return the constructed
  `OperatorV2Result` without calling `.to_dict()`.
- [x] Update query emission and budget attachment to operate on the typed
  result. Use immutable replacement for extensions rather than dictionary
  merging into a result object.
- [x] Keep `.to_dict()` and JSON encoding solely in presentation code.
- [ ] If an external mapping compatibility path is proven necessary, expose it
  as `to_operator_document(...)`; do not overload `to_operator_result` with two
  return shapes.
- [x] Add success, invalid, unavailable, Unicode, and budget-extension tests
  for query, Doctor, and reconcile envelopes.
- [x] Run process-level JSON schema tests for all three operators.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_operator_process_contract -v
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_deploy_sanity.py --json
```

**Done when:** A caller can infer the return type of `to_operator_result` without
knowing which report module supplied it.

---

## Work Package 7 — One Datetime Parsing Port

**Status: COMPLETE.** Add, modify, generation, and reconcile use the injected
datetime parser port with one error taxonomy; compatibility wrappers were
removed from production paths.

**Files:**

- Create: `nautical_core/task_datetime.py`
- Modify: `nautical_core/add_validation.py`
- Modify: `nautical_core/modify_datetime_effects.py`
- Modify: `nautical_core/chain_generation.py`
- Update composition callers and focused tests.

**Interfaces:**

- Produces:

```python
class TaskDatetimeParser(Protocol):
    def parse(self, value: object) -> tuple[datetime | None, str | None]: ...
```

- Field-aware validation decorates the stable parse result; it does not define
  another parser signature.

- [x] Add contract tests proving add, modify, generation, and reconcile paths
  receive identical `(datetime | None, str | None)` semantics for empty, valid,
  malformed, wrong-type, and parser-exception inputs.
- [x] Implement one parser adapter around the configured `parse_dt_any`
  dependency and diagnostic sink.
- [x] Rename field-aware behavior to `validate_datetime_field(...)`; accept the
  canonical parser and add the field name only to user-facing validation text.
- [x] Inject `TaskDatetimeParser` into `ChainGenerationService`; remove its
  independent parsing implementation.
- [x] Inject the same parser into modify effects rather than passing a hook host
  to `safe_parse_datetime`.
- [x] Update lambdas at composition roots to named adapters where field
  decoration is necessary.
- [x] Retain compatibility wrappers only for documented or installed-layout
  callers, with tests identifying their removal conditions.

**Done when:** Core parsing behavior has one signature and one error taxonomy;
field-specific presentation no longer creates competing parser APIs.

---

## Work Package 8 — Typed API Bindings and Public Surface

**Status: COMPLETE.** All 13 factories return frozen typed `ApiBinding`
objects, the public surface is pinned at 130 names, and canonical parser/cache
fingerprints are covered by tests.

**Files:**

- Modify incrementally: `nautical_core/*_api.py`
- Modify: `nautical_core/__init__.py`, `nautical_core/compat_api.py`
- Test: public API and direct-owner contract tests.

**Interfaces:**

- Consumes: typed subsystem dependencies created in later architecture packages.
- Produces: frozen, annotated API binding objects instead of untyped
  `SimpleNamespace` bundles.

- [x] Inventory the 13 `for_core(...)` factories and their returned attributes.
- [x] Replace each `SimpleNamespace` return with a frozen typed dataclass or a
  focused service class. Migrate parser, time/datetime, operator, scheduler, and
  cache surfaces first.
- [x] Give mutators explicit names ending in `_in_place`; keep deprecated names
  as thin boundary aliases with direct compatibility tests.
- [x] Define the supported public surface explicitly and snapshot it in a
  compatibility test. Record the current 133-name facade as compatibility
  evidence, not as a mandate that every private alias remain public forever.
- [x] Ensure internal production modules import owners directly and do not use
  root-facade aliases merely for convenience.
- [x] Keep installed-layout lazy loading functional and retain the documented
  public recurrence API throughout migration.

**Done when:** API bundles are statically discoverable, shared operation names
have shared types, and compatibility aliases are distinguishable from primary
interfaces.

---

## Work Package 9 — Executable Architecture Contract

**Status: COMPLETE (commit `7dc93ac`).** The seven-layer AST contract, facade
direction rules, deployment-sanity integration, CI gate, invalid-fixture tests,
and removal of internal root-facade imports are implemented. The current scan
reports zero architecture violations.

**Files:**

- Create: `nautical_core/architecture_contract.py`
- Modify: `nautical_core/runtime_manifest.py`
- Modify: `dev_tools/nautical_deploy_sanity.py`
- Create: `tests/test_architecture_contract.py`

**Interfaces:**

- Produces an explicit module-layer map and deterministic import validation.

- [x] Define layers for domain/models, pure recurrence, application services,
  integration adapters, presentation, hook/tool entry points, and compatibility.
- [x] Encode the permitted direction: domain and pure recurrence cannot import
  hooks, tools, Taskwarrior adapters, SQLite, or Rich; application services may
  depend on domain ports; adapters may depend inward; compatibility may depend
  on owners but owners may not depend back on compatibility.
- [x] Extend the existing operator forbidden-import checks rather than creating
  a competing AST scanner.
- [x] Add explicit rules forbidding internal imports from the root
  `nautical_core` facade except in listed compatibility and CLI boundaries.
- [x] Add a test fixture containing one deliberately invalid import and prove
  the validator reports the importing file, forbidden dependency, and layer.
- [x] Run the validator in deployment sanity and the fast CI gate.

**Done when:** A dependency-direction regression fails CI before it becomes a
new facade or hook-global coupling pattern.

---

## Work Package 10 — Replace `CoreContext` Service-Locator State

**Status: COMPLETE (commits `4dbb278`–`7e8221c`).** Parser, scheduler, and cache factories now snapshot
their inputs into immutable dependency objects. Cache directory and lock state
are isolated per binding. The parser, scheduler, and cache lookup migrations
are complete; all three modules
report zero primary `core[...]`/`core.get(...)` lookups.

**Polish complete.** Regression coverage now enforces the no-primary-lookup
rule and verifies scheduler/cache snapshots remain immutable after factory
creation. Replacing remaining `Any` annotations with focused protocols is
optional future hardening, not part of this migration gate.

**Files:**

- Modify: `nautical_core/core_context.py`
- Modify: `nautical_core/parser_api.py`
- Modify: `nautical_core/scheduler_api.py`
- Modify: `nautical_core/cache_api.py`
- Modify affected composition and compatibility factories.

**Interfaces:**

- Produces typed `ParserDependencies`, `SchedulerDependencies`,
  `CacheDependencies`, and an explicitly mutable `CacheState`.
- Retains `for_core(...)` only as a compatibility constructor during migration.

### Parser migration

- [ ] Characterize every parser facade entry point and preset/configuration
  dependency before changing construction.
- [ ] Define a frozen parser dependency object containing only parser-owned
  collaborators, presets, and configuration values.
- [ ] Direct-import pure parser owners from `nautical_core.parsing`; remove
  string-keyed `core[...]` lookups from the primary parser implementation.
- [ ] Make the compatibility factory translate a legacy `CoreContext` into
  `ParserDependencies` once at the boundary.

### Scheduler migration

- [ ] Define frozen scheduler configuration and dependency objects for clock,
  calendar, randomness, limits, tracing, and occurrence owners.
- [ ] Replace facade lookups and `_with_business_calendar` callback wrapping
  with explicit service construction.
- [ ] Preserve deterministic random namespaces, terminal evidence, date limits,
  and business-calendar displacement behavior in direct tests.

### Cache migration

- [ ] Define immutable cache configuration separately from mutable `CacheState`.
- [ ] Move cache directory selection, memory entries, and lock state out of the
  facade namespace; no cache function may write `core["_CACHE_DIR"]`.
- [ ] Inject filesystem, clock, randomness, locking, serialization, and
  diagnostics explicitly.
- [ ] Preserve atomic replacement, quarantine, bounded allocation, lock
  behavior, semantic fingerprints, and per-loader isolation.

### Cutover gates

- [ ] After each subsystem, confirm direct tests, facade compatibility tests,
  installed-layout checks, and the architecture validator pass.
- [ ] Confirm `rg -n 'core\[|core\.get\('` reports no primary dependency lookup
  in the migrated subsystem; any remaining occurrence must be documented as a
  compatibility adapter.

**Done when:** Internal parser, scheduler, and cache behavior can be constructed
without a mutable root-facade dictionary.

---

## Work Package 11 — Remove Hook-Host Reach-Through

**Status: IN PROGRESS.** Added an isolation characterization test proving
`modify_read_effects` imports without bootstrapping `hooks.modify_impl` and
that the composition capability set remains an explicit frozen boundary.
Lifecycle-read repository rebinding is removed, and `ModifyRuntimeServices` no
longer stores a live hook host. Route functions still have legacy `host`
parameters and require staged capability migration.

**Files:**

- Modify: `nautical_core/modify_composition.py`
- Modify: `nautical_core/modify_effects.py`
- Modify: `nautical_core/modify_read_effects.py`
- Modify affected validation, presentation, completion, and expiration adapters.
- Keep executable bootstrap ownership in `nautical_core/hooks/modify_impl.py`.

**Interfaces:**

- Produces narrow typed read, execution, presentation, and diagnostic services.
- Effect functions consume request/context plus services; they do not consume a
  live hook module or `_HookHost`.

- [x] Add isolation tests constructing each effect service without importing
  `nautical_core.hooks.modify_impl`.
- [x] Build `LifecycleReadService` once at the composition root with its final
  repository, cache store, query port, limits, codec, and diagnostics.
- [x] Remove post-construction repository rebinding and writes through
  `_modify_runtime_state()` from `modify_read_effects.py`.
- [x] Replace the 22-module `Any` capability bag with narrow typed capabilities
  grouped by one operation, not by the entire hook.
- [x] Remove `host` from `ModifyRuntimeServices` and the migrated route helpers;
  remaining host adapters are tracked for staged extraction.
- [ ] Replace `host._module(...)`, `host.core...`, `_read_query_get`, and
  `_READ_QUERY_MISSING` reads in effects with explicit ports.
- [ ] Keep the hook module responsible only for input protocol, composition,
  response emission, and process exit.
- [ ] Retain `_HookHost` only at an explicitly named compatibility/test adapter
  if an installed-layout test proves it is still required; production effects
  must never receive it.
- [ ] Run ordinary edit, recurrence activation, completion, deletion,
  expiration, lifecycle failure, malformed input, and strict JSON tests after
  each route migration.

**Done when:** Extracted modify modules are executable and testable without hook
globals, dynamic module lookup, or shared runtime-state mutation.

---

## Work Package 12 — Validate Lifecycle Execution Capabilities at Composition

**Files:**

- Modify: `nautical_core/lifecycle_application.py`
- Modify: `nautical_core/lifecycle_operator_owner.py`
- Inspect/update production composition in `nautical_core/hooks/exit_impl.py`,
  `nautical_core/modify_spawn_effects.py`, and
  `nautical_core/tools/nautical_reconcile.py`.
- Test: `tests/test_lifecycle_execution_capabilities.py`
- Migrate focused cases from: `dev_tools/nautical_golden_tests.py`

**Interfaces:**

- Produce one structural execution port whose required operations match the
  concrete `TaskwarriorMutationService` methods:

```python
class LifecycleExecutionPort(Protocol):
    def apply_lifecycle_unverified(
        self, request: MutationRequest
    ) -> MutationOutcome: ...

    def apply_lifecycle_children_unverified(
        self, requests: Sequence[MutationRequest]
    ) -> dict[str, MutationOutcome]: ...

    def verify_lifecycle_children(
        self, requests: Sequence[MutationRequest]
    ) -> dict[str, MutationOutcome]: ...

    def verify_lifecycle_parents(
        self, requests: Sequence[MutationRequest]
    ) -> dict[str, MutationOutcome]: ...

    def preflight_lifecycle_batch(
        self,
        payloads: Sequence[ChildImportPayload],
        *,
        parent_expectations: Sequence[tuple[str, str]] = (),
    ) -> None: ...
```

- A stage-only service may omit the execution port, but calling `drain()`,
  `drain_claimed()`, or an immediate mutation without it must raise
  `LifecycleApplicationError("lifecycle execution capability is unavailable")`
  before claiming work.

- [ ] Add failing construction and execution tests for a complete provider, an
  incomplete provider, and the supported stage-only service shape. Reject an
  incomplete provider with `LifecycleApplicationError` listing its missing
  capability names in sorted order.
- [ ] Add a production-shape integration test proving
  `LifecycleOperatorOwner.apply()` supplies `limit=1` and uses the same
  configuration and schedule fingerprints for stage and drain.
- [ ] Replace `LifecycleExecutionCapabilities.from_dependencies()` and its
  optional `getattr(...)` discovery with the explicit `LifecycleExecutionPort`.
- [ ] Pass the concrete execution port from each production composition root;
  validate it once, before a lifecycle intent is claimed or mutated.
- [ ] Replace partial golden-test mutation doubles with a dedicated fixture
  implementing the complete port. Keep deliberately incomplete doubles only in
  the contract rejection test.
- [ ] If an installed-layout compatibility caller genuinely supplies a legacy
  object, isolate dynamic discovery in a named
  `LegacyLifecycleExecutionAdapter`; primary lifecycle processing must not
  probe for methods dynamically.
- [ ] Preserve guarded mutation ordering, batched verification, retryability,
  crash recovery, drain limits, and authoritative postconditions.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_lifecycle_execution_capabilities -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_lifecycle_failure_injection -v
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py --only lifecycle
```

**Done when:** Lifecycle execution never discovers collaborators during a drain;
composition either supplies the complete mutation/verification capability or
fails deterministically before external state changes.

---

## Work Package 13 — Narrow Hook Presentation Contexts

**Files:**

- Modify: `nautical_core/add_anchor_preview.py`
- Modify: `nautical_core/add_preview_composition.py`
- Modify: `nautical_core/modify_timeline.py`
- Modify: `nautical_core/modify_presentation_effects.py`
- Modify only affected entry points in `nautical_core/modify_feedback.py`.
- Test: renderer and hook composition contract tests.

**Interfaces:**

- Separate validation, recurrence projection, and rendering dependencies. A
  public operation receives only the focused context it uses; no focused
  context contains `core`, a module loader, a hook host, or dependencies used
  solely by another operation.
- Use the primary names `AnchorExpressionPreviewServices`,
  `AnchorFilePreviewServices`, `TimelineProjectionServices`, and
  `TimelineFormattingServices`; keep an old name only in an explicit
  compatibility adapter proven necessary by an installed-layout test.
- Keep concrete callbacks at the composition root. Do not replace one wide bag
  with a hierarchy of pass-through wrappers.

- [ ] Add a usage-matrix test that constructs anchor-expression preview,
  anchor-file preview, CP timeline, and anchor timeline independently with
  sentinels that fail if an unrelated dependency is accessed.
- [ ] Preserve exact renderer output with contract cases for normal preview,
  malformed expression, omitted occurrence, exhausted provider, timezone
  fallback, compact output, and Unicode text.
- [ ] Replace `AnchorPreviewServices` with
  `AnchorExpressionPreviewServices`; keep `AnchorFilePreviewServices` separate
  so anchor-file callers do not construct expression-only validators or
  expiration rendering dependencies.
- [ ] Split `TimelineServices` into focused projection and formatting
  collaborators; inject configured evaluator/scheduler ports directly and
  remove `core` and `module_loader` from the timeline boundary.
- [ ] Replace the callback construction in `add_preview_composition.py` and
  `modify_presentation_effects.py` with the focused contexts. Construct each
  context once per hook invocation.
- [ ] Remove old wide service bags immediately after their callers and tests
  migrate; retain a compatibility adapter only when an installed-layout test
  identifies a real external caller.
- [ ] Confirm presentation continues to be side-effect free except for its
  explicit renderer sink and that hook stdout remains strict JSON.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_chain_summary_renderer_contract -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_effect_boundary -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_hook_input_contract -v
```

**Done when:** Preview and timeline operations can be tested with only their
actual collaborators, and no presentation context acts as a hook-shaped
service locator.

---

## Work Package 14 — Adjudicate Security Signals at Trust Boundaries

**Files:**

- Inspect production findings first, especially
  `nautical_core/taskwarrior_mutations.py` and filesystem/subprocess boundaries.
- Inspect test-only findings in `dev_tools/nautical_golden_tests.py` separately.
- Modify production code only for a reproduced security weakness.

- [ ] Export the complete security finding list without the display noise
  budget and classify each production finding as genuine, false positive,
  exaggerated, or not worth changing, with file-and-line evidence.
- [ ] Record that `import_child(request, verify=False)` and
  `link_parent(request, verify=False)` disable immediate Taskwarrior
  postcondition reads for later batch verification; they do not disable TLS.
  Treat the current `weak_crypto_tls` reports on those calls as false positives
  unless code evidence shows an actual network/TLS path.
- [ ] Review insecure-random findings by purpose. Deterministic recurrence and
  randomized test order are not cryptographic contexts; identity, nonce,
  credential, or untrusted-token generation must use a cryptographically
  suitable source.
- [ ] Replace hard-coded shared temporary paths only where concurrent or
  untrusted users could race, replace, or read the artifact. Keep deterministic
  fixture paths scoped inside a securely created temporary directory.
- [ ] Manually verify the real trust boundaries: Taskwarrior subprocess
  argument construction, JSON import/export, configured file paths, cache and
  outbox permissions, symlink handling, SQLite state, and diagnostic
  redaction.
- [ ] Do not apply broad suppressions or semantic changes to make the 504 raw
  signals disappear. Suppress or document only the exact reviewed instance,
  and retain the evidence for the next scan.
- [ ] Run the strict JSON, structured-failure, mutation, offline-kit, and
  deployment tests after any genuine security fix.

```bash
/home/pooK/venv/test_1/bin/desloppify show security --status open --top 1000 --no-budget
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_structured_failure_boundaries -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_operator_process_contract -v
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_deploy_sanity.py --json
```

**Done when:** Every production security signal has code-backed adjudication,
every genuine trust-boundary defect has a regression test and fix, and known
false positives have not weakened recurrence or postcondition verification.

---

## Work Package 15 — Bound the Root Compatibility Facade

**Files:**

- Modify: `nautical_core/__init__.py`
- Modify: `nautical_core/compat_api.py`
- Update internal callers and installed-layout manifests.

**Interfaces:**

- Root package remains a documented external compatibility surface.
- Internal production code imports typed owners and never depends on lazy alias
  registration order.

- [ ] Inventory every internal `from nautical_core import ...` and
  `import nautical_core as ...` occurrence.
- [ ] Convert internal imports to explicit relative owner modules, one domain at
  a time, with direct tests before each conversion.
- [ ] Move legacy aliases, lazy resolution, and deprecation behavior behind
  `compat_api.py`; do not let primary modules import that compatibility owner.
- [ ] Remove facade write-back of resolved functions after typed subsystem
  factories no longer require it.
- [ ] Preserve documented public names and installed hooks through a public API
  snapshot and deployment sanity test.
- [ ] Enforce the resulting direction with the architecture contract.

**Done when:** The root facade can be replaced or deprecated independently of
parser, scheduler, cache, lifecycle, and modify implementations.

---

## Work Package 16 — Type-Check the New Boundaries

**Files:**

- Modify: `mypy.ini`
- Modify: `.github/workflows/type-check.yml`
- Annotate only the new or migrated boundary modules.

- [ ] Require complete definitions and relevant strict error codes for the new
  datetime, API binding, architecture, parser/scheduler/cache dependency, and
  modify, lifecycle execution, and presentation service modules.
- [ ] Remove `Any` from dependency object fields where a protocol or concrete
  type is known.
- [ ] Test a normal-import configuration with `follow_imports=normal` for the
  migrated modules before changing the whole repository default.
- [ ] Expand normal import following domain by domain; do not silence new errors
  with blanket ignores.
- [ ] Keep heterogeneous Taskwarrior payload values appropriately open rather
  than forcing false precision into arbitrary UDA mappings.

```bash
python3 -m mypy --config-file mypy.ini nautical_core/task_datetime.py
python3 -m mypy --config-file mypy.ini nautical_core/parser_api.py nautical_core/scheduler_api.py nautical_core/cache_api.py
python3 -m mypy --config-file mypy.ini nautical_core/modify_composition.py nautical_core/modify_effects.py nautical_core/modify_read_effects.py
python3 -m mypy --config-file mypy.ini nautical_core/lifecycle_application.py nautical_core/lifecycle_operator_owner.py
python3 -m mypy --config-file mypy.ini nautical_core/add_anchor_preview.py nautical_core/modify_timeline.py
```

**Done when:** Static checking crosses the migrated boundaries and rejects a
missing or wrongly typed collaborator without relying on runtime dictionary
lookups.

---

## Work Package 17 — Final Reliability, Compatibility, and Score Gate

- [ ] Stop implementation and review the full diff for unnecessary changes,
  compatibility loss, unbounded work, or weakened error handling.
- [ ] Run Python compilation, full unittest discovery, complete and shuffled
  golden suites, mypy, deployment sanity, and black-box integration.
- [ ] Run the enforced CI stress profile and a short disposable soak. Use only
  temporary Taskdata and state.
- [ ] Confirm hook stdout remains one Unicode-preserving JSON document for valid
  input and empty for rejected malformed input.
- [ ] Confirm lifecycle mutations remain guarded, idempotent, postcondition
  verified, and recoverable after interruption.
- [ ] Re-run focused regression evidence for adjacent dimensions already fixed:
  explicit mutator naming, canonical omission state, deferred configuration,
  optional Astral, retired holiday cache identity, structured outbox failures,
  and panel source failure propagation.
- [ ] Confirm the security adjudication contains no unresolved genuine
  production finding and that false-positive classification is supported by
  file-and-line evidence rather than score pressure.
- [ ] Run `git diff --check` and inspect `git status --short`; do not include
  `.desloppify/`, temporary audit reports, caches, backups, or unrelated local
  files in product commits.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile nautical_core/*.py nautical_core/hooks/*.py nautical_core/tools/*.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest discover -s tests -q
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py --shuffle-seed 20260811
python3 -m mypy --config-file mypy.ini
python3 dev_tools/nautical_deploy_sanity.py --json
python3 dev_tools/nautical_black_box_test.py --json
python3 dev_tools/nautical_stress_campaign.py --profile ci --json --enforce
python3 dev_tools/nautical_soak_test.py --seconds 30 --json --enforce
git diff --check
git status --short
```

### Desloppify evidence gate

- [ ] Run a fresh scan only after the implementation queue and verification
  gates are complete.
- [ ] Record the new strict and objective scores plus Test health, Test
  strategy, API coherence, Cross-module architecture, Mid-level elegance,
  High-level elegance, Low-level elegance, Contract coherence, Abstraction
  fitness, Package organization, Error consistency, and Security.
- [ ] Run a new blind review because `desloppify scan` alone does not refresh
  stale subjective assessments.
- [ ] Inspect every newly imported finding against current code before resolving
  or skipping it.
- [ ] Do not alter review evidence to hit `85.0`; accept the blind score and
  continue only when a concrete finding remains.

```bash
/home/pooK/venv/test_1/bin/desloppify scan --path .
/home/pooK/venv/test_1/bin/desloppify review --run-batches --runner codex --parallel --scan-after-import
/home/pooK/venv/test_1/bin/desloppify status
/home/pooK/venv/test_1/bin/desloppify next
```

## Completion Criteria

- [ ] The two Test strategy review findings have direct, meaningful behavioral
  coverage and no permissive success assertions.
- [ ] All Tier-2 test-coverage findings are closed or backed by documented
  executable-boundary evidence; the Tier-3 backlog is materially reduced.
- [ ] `to_operator_result` has one return type across query, Doctor, and
  reconcile.
- [ ] Datetime parsing has one injected port and one error taxonomy.
- [ ] Parser, scheduler, and cache implementations no longer use the mutable
  root facade as their primary service locator.
- [ ] Modify effects no longer receive or reconstruct a live hook host.
- [ ] Lifecycle drain and immediate application receive one complete execution
  port validated before claims or mutations; workflow code performs no dynamic
  capability discovery.
- [ ] Anchor preview and timeline presentation use focused collaborators with no
  hook host, facade namespace, or module loader in their primary interfaces.
- [ ] The architecture validator prevents internal facade and hook reverse
  dependencies from returning.
- [ ] Every production security signal has a code-backed verdict, and every
  genuine security fix has focused regression coverage.
- [ ] Standard unit, golden, shuffled, black-box, deployment, mypy, stress, and
  soak gates pass.
- [ ] A fresh blind review replaces the stale subjective scores, including the
  adjacent elegance, contract, abstraction, organization, and error dimensions.
- [ ] The strict score improvement is reported honestly with the remaining
  mechanical and subjective debt; reaching the numeric target is not accepted
  as a substitute for the behavioral completion criteria above.
