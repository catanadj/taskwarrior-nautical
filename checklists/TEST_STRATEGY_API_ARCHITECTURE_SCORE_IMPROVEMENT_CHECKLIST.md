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

## Reconciliation — 2026-09-13

This reconciliation does not begin Work Package 17. It distinguishes completed
implementation from scan-dependent acceptance evidence.

- Active branch: `desloppify/review-remediation`; HEAD at reconciliation:
  `1c7b9ef`; Python `3.11.2`; shared checkout is dirty and is not a linked
  worktree. Existing edits are preserved.
- Standard discovery after the current batch: `712` tests passed with `47%`
  branch coverage against the `44%` CI floor.
- Current focused golden slices: natural-language name filter `0` remaining;
  scheduler `27/27`; on-add `44/44`. Direct natural-language, cache,
  precompute, scheduler-exhaustion, and chain-graph/planner tests also pass in
  the full suite.
- Branch coverage: original measured baseline `44%` with branch measurement;
  current suite measures `47%`. CI floor remains `44%` until a higher batch is
  merged, as required by the ratchet policy.
- Current cached desloppify summary is overall `85.6/100`, strict `85.2/100`
  (target `85.0`), with the
  scan dated `2026-09-13 14:16 UTC`; it predates these checklist/test changes
  and is not fresh acceptance evidence.
- Packages 1, 2, and 6 have implementation-complete status; their unresolved
  fresh-scan items remain open for a later evidence pass. Package 3's direct
  test slices are implemented and the current unittest gate passes; scan-based
  finding reduction remains open. Package 4 was mislabeled complete: only its
  registry-integrity slice was complete, and domain migration remains active.
  Package 5's measured branch-coverage gate is now implemented. Package 10 is
  complete by the recorded cutover evidence and current no-primary-lookup scan.
- The attempted rescan was rejected because the backlog was not drained; forcing
  it would regenerate issue IDs and disrupt triage. Current status still shows
  four live subjective queue items, 149 test-health findings, and seven open
  review issues. `desloppify next` was inspected instead; no forced reset or
  finding resolution was used. The last scan remains the 14:16 UTC snapshot.
- Packages 11–16 are already complete. Package 17 remains untouched.

---

## Audit Baseline and Score Constraints

- [x] Record the branch, revision, Python version, checkout state, and last
  recorded desloppify score in the reconciliation note above. The score is
  explicitly labeled stale rather than represented as a fresh measurement.
- [x] Preserve the historical audit baseline in the implementation notes:
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
- [x] Record that the three requested subjective dimensions have weight `1`
  each in a subjective pool whose total configured weight is `123`.
- [x] Do not promise that these three dimensions alone can reach strict `85.0`:
  raising all three to `100` has a theoretical maximum overall gain of about
  `0.49` points.
- [x] Use direct-test remediation to improve Test health as well as Test
  strategy. Desloppify estimates that resolving all 149 coverage findings is
  worth about `1.2` overall points.
- [x] Treat the following as theoretical prioritization ceilings, not promised
  score gains: Mid-level elegance about `2.95` overall points, high-level
  elegance `2.15`, low-level elegance `1.72`, contract coherence `1.24`,
  abstraction fitness `1.07`, package organization `0.61`, and error
  consistency `0.40` if each dimension independently reached `100`.
- [x] Record that most of these subjective assessments are stale after the
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

- [x] Do not reimplement the retired `holiday_region`, deferred configuration
  loading, optional Astral, structured outbox failure, panel diagnostics,
  canonical `OmitState`, normalization naming, or lifecycle drain-limit work
  unless a current focused regression demonstrates a defect.
- [x] Keep those findings visible in audit notes until a fresh blind review
  replaces the stale assessment; distinguish stale review state from missing
  production behavior.
- [x] Preserve focused evidence for already-implemented adjacent-dimension
  fixes: explicit `_in_place` business-calendar normalization, canonical
  `OmitState`, deferred configuration loading, optional Astral installation,
  retired `holiday_region` cache removal, and structured maintenance and panel
  source failures.

### Adjacent-dimension regression evidence

- [x] Verify `normalize_task_business_calendar_in_place(...)` both returns the
  selected calendar and canonicalizes a present `bc` value, while the legacy
  `normalize_task_business_calendar` alias has identical compatibility
  behavior.
- [x] Verify `combine_omit_state(...)` returns `OmitState | None` for every
  primary caller and legacy raw/dictionary decoding occurs only in the named
  compatibility split boundary.
- [x] Verify importing `nautical_core` with `NAUTICAL_CONFIG` set performs no
  configuration-file read; first supported configuration access performs the
  read and surfaces validation failure through the documented diagnostic path.
- [x] Verify the base requirements omit Astral, the astronomy requirements pin
  it, the offline kit includes both install modes, and astronomy-disabled
  operation never imports Astral.
- [x] Verify changing retired `holiday_region` cannot change the scheduler or
  cache fingerprint while deprecated-key recognition remains available at the
  configuration boundary.
- [x] Verify outbox permission/setup failures retain the stable
  `filesystem_security_failure` classification and always close an opened
  connection.
- [x] Verify missing optional panel sources remain non-fatal, while permission,
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

**Status: IMPLEMENTATION COMPLETE; FRESH SCAN GATE OPEN.** Direct behavioral
coverage and the natural-language migration are complete. The named natural
golden slice is empty. The finding stays unresolved until a fresh desloppify
scan can safely reassess it after the queue is drained.

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
- [x] Run the focused test and confirm it fails for missing direct coverage or
  any behavioral mismatch before changing production code.
- [x] Remove only the migrated golden registry entries after the direct tests
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

**Status: IMPLEMENTATION COMPLETE; FRESH SCAN GATE OPEN.** The route matrix,
strict JSON isolation, failure cases, and golden migration are complete. The
full hook-on-add golden slice passes `44/44`; fresh desloppify confirmation
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

**Status: DIRECT-TEST SLICES COMPLETE; SCAN GATE PENDING.** Cache, parser,
scheduler, calendar, add/preview, chain recovery, configuration/time, operator,
deployment, stress, and parser-owner adapter contracts are directly tested.
Direct graph construction and repair planning contracts now cover deterministic
ordering, reference ambiguity/out-of-coverage, topology, guarded reciprocal and
missing-link plans, and refusal under partial coverage. Fresh finding-reduction
evidence remains outstanding.

**Files:** Create focused `tests/test_<owner>.py` modules and migrate matching
tests from `dev_tools/nautical_golden_tests.py`.

**Interfaces:**

- Consumes: public functions/classes from the production module named by each
  test file.
- Produces: direct behavioral coverage recognized by unittest, coverage tools,
  and desloppify.

### Wave A — Pure and deterministic owners

- [x] Add direct tests for `natural_language_api.py`, `quarter_api.py`,
  `acf_api.py`, `expansion_api.py`, and `business_calendar_api.py`.
- [x] Add direct tests for `parsing/parser_support_api.py` and
  `parsing/parser_frontend.py`.
- [x] Add direct tests for `scheduler_atom.py` and `scheduler_api.py`.
- [x] Add direct cache contract tests for `cache_api.py`: miss/hit, corrupt
  payload, quarantine, lock refusal, Unicode, configuration fingerprint, and
  isolated per-instance state.
- [x] Add deterministic boundary matrices rather than import-only assertions.
  Every test module must call at least one behavior owned by its production
  module.

### Wave B — Application and composition owners

- [x] Add direct tests for `add_anchor_compute.py`, `add_anchor_preview.py`,
  `add_preview_composition.py`, and `chain_generation.py`.
- [x] Cover valid result, malformed input, exhausted provider, omitted
  occurrence, date limit, and injected dependency failure for applicable
  services.
- [x] Add direct tests for `business_calendar_config.py`, `timeutil.py`, and
  `chain_integrity_recovery.py` using temporary files and immutable fixtures.

### Wave C — Operator and operational entry points

- [x] Add direct command-contract tests for
  `nautical_core/tools/nautical_query.py`, `nautical_doctor.py`, and
  `nautical_reconcile.py` using temporary Taskdata and injected/stubbed process
  boundaries where available.
- [x] Test `dev_tools/nautical_deploy_sanity.py` as a callable report producer,
  including one passing inventory and one deliberately broken temporary
  inventory.
- [x] Test `dev_tools/nautical_reliability_smoke.py` argument validation and
  result classification without operating on live Taskdata.
- [x] Test the stress campaign's profile selection, validation, JSON envelope,
  and non-zero enforcement result without running a long campaign.

### Wave gates

- [x] After each wave, run unittest discovery and the relevant golden slice.
- [ ] Run desloppify after the complete wave, not after every file; record the
  exact reduction in Tier-2 and Tier-3 coverage findings.
- [x] Do not mark the wave complete if the direct-test count improved but the
  exercised branches did not include failures and edge conditions.

**Done when:** All 25 Tier-2 coverage findings are either closed by direct
behavioral tests or documented with evidence that the detector cannot represent
the executable/shell boundary; no item is closed with an import-only test.

Current evidence: unittest discovery passes `712` tests at `47%` branch coverage.
A fresh coverage review and exact issue-count reconciliation remain
open; the last recorded scan predates these direct tests.

---

## Work Package 4 — Decompose the Golden Test Monolith

**Status: COMPLETE.** Direct-contract migrations, retained-acceptance
classification, exclusive domain inventory, and verification gates are complete.
The current registry integrity contract asserts 416 top-level golden definitions,
404 registered cases, 12 explicitly retired helpers, 464 migrated direct-contract
names, three removed ineffective definitions, and no duplicate registrations.
Fresh gates on 2026-09-14 passed standard unittest discovery (1,219 tests, 3
optional skips) and the golden runner in normal and seeded-shuffle order (404/404
each, seed `20260811`). These passing gates prove the current suites execute; they
are paired with the exclusive, digest-pinned case classification and measured
per-domain inventory in `docs/design/GOLDEN_REGISTRY_INVENTORY.md`. Historical
checkpoints below are not current measurements.

Registry integrity and inventory are maintained. Three hundred twenty-eight golden
functions across recurrence, cache, hook, chain-integrity, lifecycle, operator,
query, Navigator-view, CP scheduling, astronomy, yearly-ordinal, file-backed,
seasonal calendar/selector, business-calendar, occurrence-provider, time-window,
parser/scheduler expression, core-utility, diagnostic-warning, add-validation,
config-path security, scheduler runtime, 24 individually reviewed
anchor-file/provider/inclusion/consumer-parity cases, and 17 positional-selection
owner contracts now run under standard unittest discovery,
with duplicate golden registrations removed. Parser fuzz,
normalization, characterization, and direct modified-atom scheduling contracts
also run under direct discovery. After the cross-path parity batch, complete
gates passed at 996 unit tests and 637/637 normal and shuffled golden tests.
After the recurrence-identity migration, complete gates passed: unittest
discovery 999/999 in 67.423 seconds; normal golden 634/634 and seeded shuffled
golden 634/634 (seed `20260811`). The current registry is 644 top-level / 632
registered / 12 explicitly retired / 239 newly migrated in this WP4 cycle / 0
duplicates. Focused direct time-window, parser, file-backed, and registry suites
pass. Latest complete gates pass: unittest discovery 1002/1002 in 67.185
seconds; normal golden 632/632 and seeded shuffled golden 632/632
(seed `20260811`). The positional-selection migration then moved eleven pure
owner cases and six public parser/scheduler cases; focused direct and registry
checks pass 21/21. Combined gates pass: unittest discovery 1016/1016, normal
golden 615/615, and seeded shuffled golden 615/615 (seed `20260811`). A further
six omission contracts and one file-source parser contract now have direct
tests; focused suites pass 25/25. Consolidated gates pass: unittest discovery
1023/1023, normal golden 608/608, and seeded shuffled golden 608/608 (seed
`20260811`). Current registry: 620 top-level / 608 registered / 12 retired /
263 WP4 migrations. Three native-until owner contracts have since moved into
direct tests; focused native-until and registry checks pass 7/7. Current
registry is 617 top-level / 605 registered / 12 retired / 266 WP4 migrations;
three shared-time contracts have since moved into `tests/test_config_time_contract.py`.
Focused time, native-until, and registry tests pass 14/14. Current registry:
614 top-level / 602 registered / 12 retired / 269 WP4 migrations. Consolidated
gates at that checkpoint passed: unittest discovery 1029/1029, normal golden
602/602, and seeded shuffled golden 602/602 (seed `20260811`). Three lifecycle/
configuration contracts have since moved into direct suites; focused checks
pass 25/25. Current registry is 611 top-level / 599 registered / 12 retired /
272 WP4 migrations. Two more direct tests now cover recurrence-spec and
hook-bootstrap helper contracts; focused suites pass 21/21. Current registry
is 609 top-level / 597 registered / 12 retired / 274 WP4 migrations; full gates
pass: unittest discovery 1034/1034, normal golden 597/597, and seeded shuffled
golden 597/597 (seed `20260811`). These gates do not establish that every
remaining golden scenario is acceptance-only. Since that gate, the deterministic
last-Friday scheduler check and missing-explicit-config diagnostic moved into
direct tests; focused owner/registry tests pass 20/20. Current registry is 607
top-level / 595 registered / 12 retired / 276 WP4 migrations. Full gates pass:
unittest discovery 1036/1036, normal golden 595/595, and seeded shuffled golden
595/595 (seed `20260811`). Two runtime-manifest ownership contracts have since
moved into `tests/test_architecture_contract.py`; focused architecture/registry
checks pass 17/17. Current registry is 605 top-level / 593 registered / 12
retired / 278 WP4 migrations. Full gates pass: unittest discovery 1038/1038,
normal golden 593/593, and seeded shuffled golden 593/593 (seed `20260811`).
Two more direct tests now cover preview timezone-fallback and panel file-source
diagnostic policies; focused preview/failure-boundary/registry checks pass
29/29. Current registry is 603 top-level / 591 registered / 12 retired / 280
WP4 migrations. Full gates pass: unittest discovery 1040/1040, normal golden
591/591, and seeded shuffled golden 591/591 (seed `20260811`). The latest full
gates before the current batch passed at unittest discovery 1,051/1,051 in
75.070s and golden 579/579 in normal and seeded-shuffled order (seed
`20260811`). Since then, direct suites gained thirteen contracts: operator-context
discovery and configuration failure-stage coverage, an architecture guard on
presentation imports, bounded/full lifecycle query selection, four
completion-analytics cases, facade export/signature behavior, query error and
serialization contracts, integrity report parity, and TaskCommand failure/retry
wrappers. Sixteen more owner-level contracts moved into direct suites: four
omit-file modifier cases, TaskCodec sanitization, Navigator's authoritative-
empty snapshot behavior, runtime command metadata/input/timeout/retry behavior,
roll convergence, cached-hint isolation, moon-phase parser contradictions,
shared config exposure, compact-preview occurrence limits, and hook command
runner/result contracts. Two stale "fallback when core load fails" cases were
retired after inspection showed they never induced that failure or called the
hook runner. Shipped-config schema/layout, scheduler no-progress, ACF input
bounds, panel line routing, hook import failure detail, season-mode choices,
authoritative-empty lifecycle reads, completion fail-closed preflight, and live
render routing also moved to direct suites. Three real Astral provider checks
now run from the astronomy owner suite; they skip explicitly only where Astral
is optional, and fail when the astronomy CI requirement flag is set. The latest
registry is 479 top-level / 467 registered / 12 explicitly retired
characterization helpers, plus 2 removed ineffective test definitions / 402
WP4 migrations / 0 duplicates; cumulative direct migrations are 515. The
latest batches moved TaskCommand observation privacy, recurrence activation
identity/failure handling, moon-window/filter scheduling, occurrence-prefix
terminal evidence, omission timeline warnings, anchor-file tie metadata,
DST-fold until validation, live-render generator fallback, completion terminal
exhaustion, exact/over-limit guards, captured-stderr/dumb-terminal UI behavior,
cache-clear environment semantics, canonical chain identity, add-side exhaustion
identity, Navigator projection failure evidence, dumb-terminal rendering, and
reconcile's child-local-time evidence formatting, Navigator terminal projection
metadata, astronomy-provider failure handling, random scheduling behavior,
completion-cap selection, diagnostic stdout/stderr, timeline warning/terminal
rows, preview included-event limits, static panel layout/theme, and typed
anchor-file omission scans and static Rich delegation to the shared builder
into direct owner suites. Another 16 direct contracts now cover seasonal
calculation and rollover, Navigator business-calendar and symbolic-time
projection, repeated-hour ordering, hook lifecycle-result retention, modify
schedule progress/cap/provider reuse, recovery fail-closed evidence, and typed
anchor-file metadata/context/cache behavior. Focused migrated and adjacent
suite checks passed 87/87 at that historical checkpoint. Shared
subprocess/Taskdata, recurrence-file, and lifecycle execution fixtures live
under `tests/support/`. This migration recap records an earlier checkpoint;
the final 1,219/404 gates and completed domain inventory are recorded at the
start of this section and in the golden inventory.

**Files:**

- Modify: `dev_tools/nautical_golden_tests.py`
- Extend nested packages only where a domain suite is actually organized as a
  nested directory; top-level hook/lifecycle/operator modules remain deliberate
  single-file suites.
- Create shared test-only support under: `tests/support/`

**Interfaces:**

- Consumes: existing golden functions and their fixture dependencies.
- Produces: normally discoverable domain tests plus a smaller acceptance-only
  golden runner.

- [x] Add `tests/recurrence/__init__.py` so standard unittest discovery descends
  into the migrated recurrence package.
- [x] Add initializers for `tests/cache/` as the cache-domain migration begins.
- [x] Keep existing top-level hook/lifecycle/operator suites as ordinary test
  modules; only nested recurrence/cache/support suites need package initializers.
- [x] Move genuinely shared test fixtures—not assertions or production
  behavior—into test-only support modules: Taskdata/subprocess execution,
  recurrence files, and lifecycle execution.
- [x] Keep task builders and clocks local where the fixtures have distinct
  contracts; do not force unrelated shapes into a generic shared helper.
- [x] Classify the remaining golden scenarios by direct-contract versus
  acceptance need. Every retained case is assigned exactly once to one of
  eight acceptance domains, with counts and case-name digests enforced by
  `tests/test_golden_registry_integrity.py`. Direct contracts and retired
  characterizations remain separately allowlisted.
- [x] Preserve migrated tests' deterministic inputs and expected outcomes. Replace
  the custom `expect(...)` helper with the corresponding `unittest.TestCase`
  assertion.
- [x] Remove migrated functions from `TESTS`/`DEEP_TESTS` immediately so CI does
  not execute duplicate tests.
- [x] Keep only scenarios with an evidence-backed process, installed-layout,
  cross-process, Taskwarrior, or long-running compatibility boundary in the
  golden runner; classify every remaining case before the acceptance-only
  cutover.
- [x] Preserve normal and seeded shuffled execution until no shared mutable
  state remains in the migrated domains.
- [x] Move the hook-protocol and TaskDocument pure contracts into
  `tests/test_hook_protocol.py` and `tests/test_taskwarrior_io.py`; retain
  isolated-load, executable, output, and permission acceptance coverage in the
  golden runner.
- [x] Move the lifecycle planner's recurrence-candidate policy, shared
  completion/reconcile plan, scheduled-expiration basis, recurrence boundary
  matrix, and idempotent terminal patch contracts into
  `tests/test_lifecycle_terminal_plans.py` with explicit child-field and
  identity assertions.
- [x] Move lifecycle read-service index/merge behavior, safe full-snapshot
  filtering, and request-cache isolation into
  `tests/test_lifecycle_read_service.py`.
- [x] Move chain-integrity model validation, authoritative snapshot/cache
  fail-closed contracts, pure engine/report parity, bounded hydration, invariant
  behavior, outbox/graph provenance, acknowledged postconditions, and safe
  application-boundary decisions into direct chain-integrity unittest modules.
- [x] Move typed Taskwarrior read outcomes, TaskObservation/NauticalTask/TaskView,
  task codec framing/serialization, and TaskDraft/TaskPatch semantics into
  `tests/test_task_domain_models.py`.
- [x] Move authoritative read snapshot/index/ambiguity and bounded UUID/slot
  set-read fail-closed contracts into `tests/test_task_read_repository_contracts.py`.
- [x] Move scripted repository cache/fallback, malformed-output, mutation-epoch,
  and typed domain-read contracts into `tests/test_task_read_repository_contracts.py`;
  retain real Taskwarrior process acceptance separately.
- [x] Move in-memory Taskwarrior unit-of-work cache scope, broad coverage, and
  invocation-isolation contracts into `tests/test_taskwarrior_uow_contracts.py`;
  retain process/retry/budget diagnostics in golden acceptance.
- [x] Move Navigator metadata/query identity parity and renderer-neutral view
  serialization into `tests/test_navigator_view_models.py`; retain rendering,
  import/layout, repository integration, and scale cases pending case-level
  classification.
- [x] Move CP duration/sequence parsing, link-boundary selection,
  deterministic random/jitter bounds and scope, and DST-safe whole-day stepping
  into `tests/recurrence/test_cp_sequence_contracts.py`; retain on-add/on-modify
  cross-path agreement in the golden runner.
- [x] Move deterministic moon-phase grammar/configuration/math contracts to
  `tests/test_astronomy_contracts.py`, and year-day/ISO-week validation,
  expansion, scheduling, omission, and round-trip contracts to
  `tests/recurrence/test_yearly_token_migration.py`.
- [x] Move file-name safety, omit-file CSV header/deduplication/description,
  anchor-file parsing/expansion, and deterministic file-provider contracts into
  `tests/test_file_backed_contracts.py`; move typed-provider values, metadata,
  progress, DST cursor, and bounded-collection guarantees into
  `tests/test_occurrence_provider_contracts.py`. Retain hook, cross-provider,
  installed-layout, and Taskwarrior-facing acceptance coverage in golden.
- [x] Re-run complete unittest, normal golden, seeded-shuffle golden, and
  compilation gates after the latest file/provider migration batches. Current
  full unittest and both golden gates pass; `git diff --check` is clean.
- [x] Migrate the evaluator-versus-chain-generation time-form parity matrix to
  `tests/recurrence/test_scheduler_cross_path_conformance.py`; initialize the
  lazy timezone configuration before capturing the evaluator context.
- [x] Split evaluator shadow-parity characterization for DST gaps and
  business-calendar policy into two explicit direct cross-path tests.
- [x] Move modify-schedule, modify-timeline, and add-preview recurrence
  identity contracts into `tests/test_recurrence_identity_contracts.py`.
- [x] Move deterministic random-window DST projection, random-time composition
  rejection, and anchor-file random-window canonicalization into direct owner
  suites; retain cross-process random-seed stability in golden acceptance.
- [x] Move eleven pure positional-selection parser/evaluator/cache contracts
  and six public parser/scheduler cases into
  `tests/recurrence/test_position_selection_contracts.py`; retain hook,
  completion, and timeline behavior in golden acceptance.
- [x] Move date-only omission parsing, expression and loaded-date-state
  scheduling, grouped evaluation, and modifier behavior into
  `tests/recurrence/test_omit_contracts.py`; move file-source grouping and
  safety parsing into `tests/test_file_backed_contracts.py`.
- [x] Move pure native-until presentation, fold-aware validation, and exact
  carry/fail-closed converter contracts into
  `tests/test_native_until_contracts.py`; retain hook preview and modify-flow
  policy scenarios in golden acceptance.
- [x] Move shared consumer comparator identity, DST-fold comparison, and
  date-line-gap timezone resolution into `tests/test_config_time_contract.py`.
- [x] Move recurrence-fingerprint stability, effective-config snapshot
  isolation/provenance, and warm config-fingerprint no-stat behavior to direct
  lifecycle/configuration owner tests.
- [x] Move recurrence-spec normalization/context-identity and bounded
  hook-bootstrap numeric parsing to the owning recurrence-runtime and bootstrap
  test modules; keep malformed-environment subprocess behavior in golden tests.
- [x] Split the last-Friday example's owner-level scheduling assertion from its
  already-direct phrase assertion; migrate missing explicit-config diagnostics
  to structured failure boundary tests.
- [x] Move lazy panel-colour manifest inclusion and removed legacy exit-flow
  ownership checks into direct architecture contracts.
- [x] Move the final owner-level candidates from this pass into direct suites:
  UOW command-budget behavior, hook deletion routing, Navigator terminal/scale
  projections, cache location selection, Taskwarrior retry classification,
  renderer layout/fallback policy, completion-finalize analytics, outbox
  connection/session cleanup, repository timing summaries, modify lifecycle
  promotion, performance-manifest coverage, chain-generation adapters and
  identity guards, CSV anchor-source metadata, exit-probe conservatism, and
  scoped business-calendar policy. Remove their golden copies and record them
  in the migration allowlist.
- [x] Add a registry-integrity check proving every function left in the golden
  file is registered exactly once.
- [x] Re-run complete unittest, normal golden, and seeded-shuffle gates after
  the latest migrations: 1,219 unittest tests (3 optional skips), golden
  404/404 in normal and seeded order (seed `20260811`). `git diff --check`
  passes.
- [x] Record per-domain unit count, golden count, runtime, and intentionally
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

**Status: IMPLEMENTED AT THE MEASURED BASELINE.** `coverage.py` is isolated in
`requirements-test.txt`; the initial branch result was `44%`, which is the CI
floor. The latest measured suite was `47%`, but the floor is intentionally not
raised until the change is merged.

**Files:**

- Create: `.coveragerc`
- Create or modify: an explicit development-test requirements file
- Modify: `.github/workflows/type-check.yml`
- Create: `tests/test_architecture_contract.py`

**Interfaces:**

- Consumes: standard unittest discovery and production package imports.
- Produces: branch-coverage evidence, a non-decreasing coverage floor, and
  architecture regression enforcement.

- [x] Add `coverage.py` only as a development/CI dependency; do not add it to
  Nautical's runtime installation requirements.
- [x] Configure branch coverage for `nautical_core`, excluding generated,
  archived, and test-only compatibility material already outside production
  scope.
- [x] Capture the actual initial branch-coverage result and set the CI floor to
  that measured integer value. Do not invent a target above the baseline.
- [x] Fail CI when total branch coverage falls below the recorded floor and
  upload the text/XML report for diagnosis.
- [x] Increase the floor only after a merged batch produces a stable higher
  baseline.
- [x] Add deterministic parser/scheduler invariants using the standard library:
  canonical round-trip, strictly advancing occurrences, stable seeded random
  selection, and no result beyond an explicit date limit.
- [x] Keep stress and soak tests outside the fast unit gate; retain their
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
- [x] No external mapping compatibility caller exists in the repository, so
  none was added; `to_operator_result` has one typed return shape and mapping
  serialization remains in presentation code.
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

- [x] Characterize every parser facade entry point and preset/configuration
  dependency before changing construction.
- [x] Define a frozen parser dependency object containing only parser-owned
  collaborators, presets, and configuration values.
- [x] Direct-import pure parser owners from `nautical_core.parsing`; remove
  string-keyed `core[...]` lookups from the primary parser implementation.
- [x] Make the compatibility factory translate a legacy `CoreContext` into
  `ParserDependencies` once at the boundary.

### Scheduler migration

- [x] Define frozen scheduler configuration and dependency objects for clock,
  calendar, randomness, limits, tracing, and occurrence owners.
- [x] Replace facade lookups and `_with_business_calendar` callback wrapping
  with explicit service construction.
- [x] Preserve deterministic random namespaces, terminal evidence, date limits,
  and business-calendar displacement behavior in direct tests.

### Cache migration

- [x] Define immutable cache configuration separately from mutable `CacheState`.
- [x] Move cache directory selection, memory entries, and lock state out of the
  facade namespace; no cache function may write `core["_CACHE_DIR"]`.
- [x] Inject filesystem, clock, randomness, locking, serialization, and
  diagnostics explicitly.
- [x] Preserve atomic replacement, quarantine, bounded allocation, lock
  behavior, semantic fingerprints, and per-loader isolation.

### Cutover gates

- [x] After each subsystem, confirm direct tests, facade compatibility tests,
  installed-layout checks, and the architecture validator pass.
- [x] Confirm `rg -n 'core\[|core\.get\('` reports no primary dependency lookup
  in the migrated subsystem; any remaining occurrence must be documented as a
  compatibility adapter.

**Done when:** Internal parser, scheduler, and cache behavior can be constructed
without a mutable root-facade dictionary.

---

## Work Package 11 — Remove Hook-Host Reach-Through

**Status: COMPLETE.** Route and renderer operations no longer accept or reach
through `_HookHost`. Hook-aware assembly is centralized in the composition
adapter and composition root; effect modules retain only explicitly named
`*_port_for`, `*_ports_for`, and `*_services_for` constructors at that seam.
`LifecycleReadService` is built once at the composition root, the former
`modify_effects.py` route wrapper was removed, and presentation effects now
consume typed ports. `ModifyRuntimeServices` retains no live hook host.

Verification: the full on-modify golden slice passes 143/143; the focused
architecture/hook/lifecycle unit slice passes 34/34; deployment sanity passes;
compilation and `git diff --check` pass. A hardening pass removed the generic
hook-module loader from `TimelineServices`, replacing it with typed omission
callbacks and a regression test. Full unittest discovery now passes 653/653;
the cursor-terminal evidence failure was fixed by preserving cursor metadata
and terminal evidence through the evaluator collection boundary.

**Files:**

- Modify: `nautical_core/modify_composition.py`
- Add: `nautical_core/modify_composition_adapters.py`
- Remove: `nautical_core/modify_effects.py` (route adapters consolidated)
- Modify: `nautical_core/modify_read_effects.py`
- Modify: `nautical_core/modify_timeline.py` and its focused architecture test.
- Modify: `nautical_core/runtime_manifest.py` and the hook module loader map.
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
- [x] Remove `host` from `ModifyRuntimeServices` and all route/presentation
  operations; place hook-dependent route assembly in the named composition
  adapter. Only explicit port/service constructor factories adapt the host.
- [x] Ensure effect operations use their supplied ports and contain no
  operational `host._module(...)`, `host.core...`, `_read_query_get`, or
  `_READ_QUERY_MISSING` reach-through; timeline services use explicit omission
  callbacks rather than retaining the generic hook module loader.
- [x] Keep the hook module responsible for input protocol, composition,
  response emission, diagnostics, and process exit. `_HookHost` remains only as
  the composition-root view needed by the import-by-file hook layout; production
  effect operations never receive it.
- [x] Run ordinary edit, recurrence activation, completion, deletion,
  expiration, lifecycle failure, malformed input, and strict JSON tests after
  each route migration; final on-modify golden gate passes 143/143.

**Done when:** Extracted modify operations are executable and testable without
hook globals, dynamic module lookup, or shared runtime-state mutation. The only
host-aware functions are explicitly named constructors at the composition
seam; effect operations themselves receive frozen ports/services.

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

- Composition also validates the direct mutation gateway operations used by
  the service outside the batch port (`apply` and
  `compensate_imported_child`), so neither sequential mutation nor guarded
  compensation can fail from a missing method after an intent is claimed.

- An execution-capable service requires the complete outbox surface it calls:
  single-intent `enqueue`; session handling; atomic wave APIs (`enqueue_many`,
  `claim_intents`, `renew_leases`, `advance_stages`, and `acknowledge_many`);
  and the single-item claim, transition, retry, and review operations used by
  bounded one-record drains and recovery. Stage-only services need only the
  single-intent enqueue path. Bulk transaction failures must remain failures;
  do not silently downgrade to per-record storage operations.

- A stage-only service may omit the execution port, but calling `drain()`,
  `drain_claimed()`, or an immediate mutation without it must raise
  `LifecycleApplicationError("lifecycle execution capability is unavailable")`
  before claiming work.

- [x] Add failing construction and execution tests for a complete provider, an
  incomplete provider, and the supported stage-only service shape. Reject an
  incomplete provider with `LifecycleApplicationError` listing its missing
  capability names in sorted order.
- [x] Add a production-shape integration test proving
  `LifecycleOperatorOwner.apply()` supplies `limit=1` and uses the same
  configuration and schedule fingerprints for stage and drain.
- [x] Replace `LifecycleExecutionCapabilities.from_dependencies()` and its
  optional `getattr(...)` discovery with the explicit `LifecycleExecutionPort`.
- [x] Pass the concrete execution port from each production execution root;
  on-modify intentionally remains stage-only while Taskwarrior holds its lock;
  validate it once, before a lifecycle intent is claimed or mutated.
- [x] Validate the direct mutation gateway methods (`apply` and
  `compensate_imported_child`) at composition as well; these remain used by
  sequential mutation and compensation paths outside the batch port.
- [x] Validate every outbox method used by execution at composition, remove
  runtime session/bulk capability fallbacks, and preserve batch-level storage
  failures without falling back to per-intent writes.
- [x] Replace partial golden-test mutation doubles with a dedicated fixture
  implementing the complete port. Keep deliberately incomplete doubles only in
  the contract rejection test.
- [x] If an installed-layout compatibility caller genuinely supplies a legacy
  object, isolate dynamic discovery in a named
  `LegacyLifecycleExecutionAdapter`; repository caller audit found no such
  caller, so no dynamic compatibility adapter was added.
- [x] Preserve guarded mutation ordering, batched verification, retryability,
  crash recovery, drain limits, and authoritative postconditions.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_lifecycle_execution_capabilities -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_lifecycle_failure_injection -v
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py --only lifecycle
```

**Done when:** Lifecycle mutation/verification collaborators are validated at
composition and are never discovered dynamically during a drain; an incomplete
mutation gateway, execution port, or execution outbox fails deterministically
before external state changes. Atomic bulk-storage failures remain visible and
do not trigger per-record downgrade behavior.

---

## Work Package 13 — Narrow Hook Presentation Contexts

**Files:**

- Modify: `nautical_core/add_anchor_preview.py`
- Modify: `nautical_core/add_preview_composition.py`
- Modify: `nautical_core/modify_timeline.py`
- Modify: `nautical_core/modify_composition_adapters.py`
- Review: `nautical_core/modify_presentation_effects.py` (keep its existing
  focused chain-style port; no timeline-context factory belongs there)
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

- [x] Add a usage-matrix test that constructs anchor-expression preview,
  anchor-file preview, CP timeline, and anchor timeline independently with
  sentinels that fail if an unrelated dependency is accessed.
- [x] Preserve exact renderer output with contract cases for normal preview,
  malformed expression, omitted occurrence, exhausted provider, timezone
  fallback, compact output, and Unicode text.
- [x] Replace `AnchorPreviewServices` with
  `AnchorExpressionPreviewServices`; keep `AnchorFilePreviewServices` separate
  so anchor-file callers do not construct expression-only validators or
  expiration rendering dependencies.
- [x] Split `TimelineServices` into focused projection and formatting
  collaborators; inject configured evaluator/scheduler ports directly and
  remove `core` and `module_loader` from the timeline boundary.
- [x] Replace callback construction in `add_preview_composition.py` and
  `modify_composition_adapters.py` with focused contexts. Keep
  `modify_presentation_effects.py` limited to chain-style ports. Construct each
  context once per hook invocation.
- [x] Remove old wide service bags immediately after their callers and tests
  migrate; retain a compatibility adapter only when an installed-layout test
  identifies a real external caller.
- [x] Confirm presentation continues to be side-effect free except for its
  explicit renderer sink and that hook stdout remains strict JSON.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_chain_summary_renderer_contract -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_presentation_context_contract -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_effect_boundary -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_hook_input_contract -v
```

**Done when:** Preview and timeline operations can be tested with only their
actual collaborators, and no presentation context acts as a hook-shaped
service locator. Verified: the focused usage matrix passes independently for
both preview contexts and both timeline kinds; all listed renderer edge cases
remain covered by unit/golden contracts.

---

## Work Package 14 — Adjudicate Security Signals at Trust Boundaries

**Files:**

- Inspect production findings first, especially
  `nautical_core/taskwarrior_mutations.py` and filesystem/subprocess boundaries.
- Inspect test-only findings in `dev_tools/nautical_golden_tests.py` separately.
- Modify production code only for a reproduced security weakness.

- [x] Export the complete security finding list without the display noise
  budget and classify each production finding as genuine, false positive,
  exaggerated, or not worth changing, with file-and-line evidence.
- [x] Record that `import_child(request, verify=False)` and
  `link_parent(request, verify=False)` disable immediate Taskwarrior
  postcondition reads for later batch verification; they do not disable TLS.
  Treat the current `weak_crypto_tls` reports on those calls as false positives
  unless code evidence shows an actual network/TLS path.
- [x] Review insecure-random findings by purpose. Deterministic recurrence and
  randomized test order are not cryptographic contexts; identity, nonce,
  credential, or untrusted-token generation must use a cryptographically
  suitable source.
- [x] Replace hard-coded shared temporary paths only where concurrent or
  untrusted users could race, replace, or read the artifact. Keep deterministic
  fixture paths scoped inside a securely created temporary directory.
- [x] Manually verify the real trust boundaries: Taskwarrior subprocess
  argument construction, JSON import/export, configured file paths, cache and
  outbox permissions, symlink handling, SQLite state, and diagnostic
  redaction.
- [x] Do not apply broad suppressions or semantic changes to make the 504 raw
  signals disappear. Suppress or document only the exact reviewed instance,
  and retain the evidence for the next scan.
- [x] Run the strict JSON, structured-failure, mutation, offline-kit, and
  deployment tests after any genuine security fix.

**Audit record — 2026-09-13**

- Complete open-finding export: `/tmp/nautical-security-findings-20260913.json`
  (504 signals across 61 files; 179 in `nautical_core/` and
  `nautical_navigator.py`, 325 in development/test tooling). The golden runner
  accounts for 240 tooling signals. The export retains each finding's ID,
  detector, file, line, and code context.
- Production verdicts by detector: `weak_crypto_tls` 2 — false positives at
  `nautical_core/taskwarrior_mutations.py:685,687`; `B603` 5 — exaggerated
  signals because calls use argument vectors without a shell and binaries are
  resolved or explicitly supplied at the integration boundary
  (`nautical_core/taskwarrior_client.py:117`,
  `nautical_core/integration_context.py:171-183`); `B404` 4 — false positives
  on importing `subprocess`; `B101` 23 and `B112` 8 — exaggerated signals on
  internal invariants or fail-closed best-effort iteration; `B110` 137 — not
  security defects by themselves, covering optional diagnostics, cache
  fallback/cleanup, and optimization-only preflight. The export provides the
  file-and-line evidence for every member of these groups.
- The two unverified lifecycle methods retain per-intent guards and mutation
  commands; lifecycle execution performs later authoritative batch verification
  in `nautical_core/taskwarrior_mutations.py:676-688,773-835`. No TLS/network
  client is involved.
- Test/tooling-only verdicts: `B311` uses pseudorandomness only for load/stress
  task generation; `B607` occurs in the performance harness's Taskwarrior
  invocations; `B108` mostly identifies inert path-resolution/rendering fixtures
  or retired golden cases. The active `_test_operator_uow()` helper now allocates
  isolated temporary Taskdata instead of the shared
  `/tmp/nautical-test-taskdata` path.
- Genuine issue reproduced and fixed: outbox initialization followed symlinks
  for `.nautical-state`, the SQLite database, and WAL/SHM files; permission
  hardening could affect external targets. It now rejects symlinks and
  non-regular files, verifies opened-file identity, creates the database
  privately, enforces directory/file modes, and closes the SQLite connection
  if connection setup fails in `nautical_core/lifecycle_outbox.py:320-427`.
  WAL/SHM sidecars remain protected by the mode-0700 state directory rather
  than being chmodded while concurrent SQLite users may hold them. Regression
  cases are in `tests/test_structured_failure_boundaries.py:21-94`.
- Trust-boundary review found argv-based subprocess use without shell execution,
  shape validation around Taskwarrior JSON, traversal/ownership checks for
  configured directories, private cache/outbox directories and state files,
  symlink-rejecting backup verification, and opt-in/content-redacted diagnostics.
  No broad suppressions were added.
- Final gates: 671 unit tests passed, all 984 registered golden tests passed,
  deployment sanity returned `status: ok`, and `git diff --check` passed. The
  cross-owner queue/reconcile claim test also passed five consecutive isolated
  runs after avoiding concurrent sidecar chmod operations.
- Revalidation on 2026-09-13: structured failure-boundary tests passed 9/9,
  operator process contracts passed 26/26, deployment sanity returned
  `status: ok`, and `git diff --check` passed. No product code changed during
  this revalidation.

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

- [x] Inventory every internal `from nautical_core import ...` and
  `import nautical_core as ...` occurrence.
- [x] Convert internal imports to explicit relative owner modules, one domain at
  a time, with direct tests before each conversion.
- [x] Move legacy aliases, lazy resolution, and deprecation behavior behind
  `compat_api.py`; do not let primary modules import that compatibility owner.
- [x] Remove facade write-back of resolved functions after typed subsystem
  factories no longer require it.
- [x] Preserve documented public names and installed hooks through a public API
  snapshot and deployment sanity test.
- [x] Enforce the resulting direction with the architecture contract.

**Progress record — 2026-09-13**

- Inventory found facade imports in the operator query, doctor, and reconcile
  command roots; these are now confined to tool composition boundaries. Module
  owners use their explicit modules, including reconcile's lock owner and the
  Doctor cache-maintenance owner, rather than importing individual root aliases.
- `compat_api.py` now owns lazy sibling/API binding, public model resolution,
  export enumeration support, and the legacy business-calendar alias mapping.
  The root package delegates those compatibility mechanics while retaining
  its documented names and lazy behavior.
- The architecture contract now rejects direct imports of
  `nautical_core.compat_api` from primary production layers; the synthetic
  violation test and full architecture tests pass. The deployment sanity
  check passes with the stable 130-name public API snapshot.
- Query and integrity-query services now receive immutable runtime ports instead
  of a live root-facade module. Integration startup and operator UOW construction
  likewise receive an explicit `IntegrationRuntime`; the facade adapter is kept
  at the CLI composition edge. A typed-parser migration regression exposed that
  decoded temporal values are `datetime` instances, so query parsing now uses
  the observation's original wire value.
- Lazy facade alias resolution no longer writes bound functions back over the
  registered wrappers. A contract test proves aliases remain stable while still
  resolving and invoking the typed binding. Wrappers retain their documented
  public signatures and expose lazy cache controls, preserving introspection,
  cache metrics, and invalidation without facade namespace replacement.
- Verification: unit discovery passed (675 tests), complete golden suite passed
  (984/984), focused operator process contracts passed (26), architecture and
  typed API tests passed, and deployment sanity reported `status: ok`. The
  remaining `nautical_core` imports are confined to the CLI composition roots;
  domain and primary application owners do not depend on the root facade.

**Done when:** The root facade can be replaced or deprecated independently of
parser, scheduler, cache, lifecycle, and modify implementations.

---

## Work Package 16 — Type-Check the New Boundaries

**Files:**

- Modify: `mypy.ini`
- Modify: `.github/workflows/type-check.yml`
- Annotate only the new or migrated boundary modules.

- [x] Require complete definitions and relevant strict error codes for the new
  datetime, API binding, architecture, parser/scheduler/cache dependency, and
  modify, lifecycle execution, and presentation service modules.
- [x] Remove `Any` from dependency object fields where a protocol or concrete
  type is known.
- [x] Test a normal-import configuration with `follow_imports=normal` for the
  migrated modules before changing the whole repository default.
- [x] Expand normal import following domain by domain; do not silence new errors
  with blanket ignores.
- [x] Keep heterogeneous Taskwarrior payload values appropriately open rather
  than forcing false precision into arbitrary UDA mappings.

**Progress record — 2026-09-13**

- Strict complete-definition and error-code checks now cover the migrated
  datetime parser, API binding and compatibility boundary, architecture
  contract, integration/runtime context, recurrence context, and occurrence
  query service. Existing modify, lifecycle, scheduler, and presentation
  boundaries remain under their targeted strict CI groups; the full package
  strict error-code gate remains enabled.
- Typed dependency records now use callable contracts where they carry
  callbacks; recurrence timezone/calendar and cache-memory state use concrete
  types. Heterogeneous namespace/configuration and Taskwarrior UDA mappings
  remain open where their runtime shape is intentionally dynamic.
- A dedicated `follow_imports=normal` CI pass covers 13 migrated and adjacent
  dependency modules. Initial findings in the followed modules were corrected
  directly. The normal-import pass and full strict package check now both pass;
  no blanket type ignores were added.

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
