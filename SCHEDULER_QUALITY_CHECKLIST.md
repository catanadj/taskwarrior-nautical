# Nautical Scheduler Engine Checklist

Consolidate Nautical scheduling around one typed recurrence engine. Improve the
existing grammar and provider implementations through a controlled cutover,
not a ground-up recurrence rewrite.

## Scope And Working Model

- [x] Develop exclusively on `scheduler-engine-v6`; keep `main` operational
  until the new scheduler passes every completion gate.
- [x] Treat Nautical as offline while this branch is under construction. An
  intermediate commit does not need to remain usable as an installed release.
- [x] Do not build old/new scheduler bridges, dual execution, silent fallback,
  or permanent compatibility callbacks.
- [ ] Move each operational consumer to the new contract and delete its old
  path in the same pass once focused parity tests pass.
- [x] Keep Taskwarrior as the durable owner of task and chain data. Scheduler
  caches, hints, fingerprints, and other derived state may be discarded and
  rebuilt at cutover.
- [ ] Preserve Nautical grammar and task UDAs unless a checklist item explicitly
  identifies an invalid or contradictory behavior.
- [ ] Preserve strict hook JSON on stdout, `ensure_ascii=False`, diagnostics on
  stderr only when `NAUTICAL_DIAG=1`, and fail-closed mutation decisions.
- [x] Keep this checklist local. Push implementation commits to the branch and
  merge only after the scheduler is operational.

Cutover policy:

- [ ] Stop hooks and Nautical workers before installing the completed branch.
- [ ] Discard obsolete scheduler caches rather than migrating internal formats.
- [ ] Run doctor, navigator explain, reconcile dry-run, and isolated lifecycle
  smoke tests before re-enabling hooks.
- [ ] Roll back by restoring the previous release, not by retaining two
  scheduler engines in production.

## Baseline

- [x] Record the current golden, black-box, deployment, mypy, and performance results.
  Existing main-branch lifecycle, black-box, deployment, and mypy results are
  retained as the pre-scheduler baseline.
- [x] Record cold scheduler import, parse/validate, next-occurrence, hints,
  completion, navigator, and reconcile timings on desktop.
- [x] Record the reduced enforced performance profile on both Termux devices.
  `perf.termux.baseline_flagship` (Android/Python 3.14) and
  `perf.termux.baseline_normal` (Linux/Python 3.12) are the reference reports.
  Flagship/normal `next_after` medians are 0.503s/0.506s; warm hint medians
  are 0.455s/0.476s; queue-drain medians are 1.899s/4.002s with 11 calls.
  Current budget failures are recorded as pressure points, not relaxed.
- [x] Preserve strict hook JSON on stdout and diagnostic-only stderr behavior.
- [x] Preserve deterministic random identities, timezone handling, terminal evidence, and fail-closed mutation behavior.
- [x] Inventory every direct call to expression, atom, provider, and projection
  scheduling APIs before removing any path.
  The initial inventory is retained in the scheduler call-site audit; direct
  calls are concentrated in `add_anchor_compute.py`, `recurrence_candidates.py`,
  `anchor_omit.py`, and the core facade and will be removed as consumers move.

Completion criteria:

- [x] The baseline is reproducible from the checked-in performance budget
  command and the two preserved Termux reports.
- [x] Every operational scheduler consumer has a named characterization test.

## 1. Compile One Canonical Schedule Model

- [x] Introduce an immutable, versioned `CompiledSchedule` produced from a
  validated recurrence specification and validated scheduling configuration.
- [x] Compile normalized AND/OR clauses, selection modifiers, date providers,
  time projection, omissions, business-calendar policy, limits, timezone,
  recurrence identity, and deterministic-random inputs into that model.
- [x] Keep raw Taskwarrior task dictionaries, UI callbacks, subprocesses, and
  mutable caches outside the compiled representation.
- [x] Define a canonical serialization and semantic fingerprint independent of
  input formatting, dictionary ordering, and equivalent duration spellings.
- [x] Reject contradictory or incomplete schedules during compilation; an
  invalid compiled schedule must be impossible to construct silently.
- [x] Make provider and projection instructions typed so evaluators do not
  reinterpret raw parser atoms or task UDA values.
- [x] Compile once per task/configuration fingerprint and reuse the same object
  for preview, completion, reconcile, navigator, hints, and feedback.
- [x] Version derived cache entries with the compiler/schema/release
  fingerprint; discard stale compiled state rather than translating it.
- [x] Add round-trip diagnostic serialization without making deserialization a
  runtime compatibility requirement.

Completion criteria:

- [x] Semantically equal recurrence definitions compile to equal canonical
  schedules and fingerprints.
- [x] Invalid or contradictory definitions cannot produce a `CompiledSchedule`.
- [x] Operational evaluators never inspect raw task recurrence fields or parser
  DNF after compilation.
- [x] Evaluating a compiled schedule never invokes the parser again.
- [x] Compilation is deterministic, side-effect free, and independently
  testable without Taskwarrior or UI state.

## 2. Define One Cursor Contract

- [x] Make the authoritative public lookup strictly-after an instant; represent
  inclusive range starts explicitly rather than by subtracting time in callers.
- [x] Introduce a typed cursor carrying local instant, inclusivity, timezone,
  and date-limit context where those distinctions are required.
- [x] Audit cursor advancement for previews, hints, completion, reconcile, navigator, and range collection.
  Existing consumers route through evaluator/provider boundaries; remaining `timedelta(days=1)` uses are internal
  calendar probes or date-window arithmetic, not caller compensation.
- [x] Remove caller-side `+ 1 day` compensation where the scheduler already implements strict-after semantics.
  The audit found no removable caller compensation; internal scheduler probes remain intentionally date-based.
- [x] Reject non-advancing provider results at the engine boundary.
- [x] Add adjacent-valid-day regressions, including `w:mon..fri` and multi-time schedules.
- [x] Add boundary tests for month, year, DST, business-day, astronomy, and date-limit transitions.
  Existing golden coverage exercises each boundary; cursor-specific adjacent-weekday coverage was added here.

Completion criteria:

- [x] Every occurrence lookup documents and enforces the same cursor semantics.
  Evaluator and provider collection expose the typed cursor while preserving strict-after datetime defaults.
- [x] Adjacent valid occurrences cannot be skipped or duplicated.

## 3. Return Typed Occurrence Outcomes

- [x] Introduce immutable result models for found, exhausted, unavailable, and
  invalid evaluations.
- [x] Include local and UTC occurrence time, source/provider evidence,
  projection evidence, selected expression term, and terminal evidence.
- [x] Replace ambiguous date/tuple/`None` combinations at scheduler boundaries.
- [x] Preserve `OccurrenceSearchExhausted` evidence without converting it to ordinary absence.
- [x] Make date-limit exhaustion distinct from bounded-search exhaustion.
- [x] Require mutation paths to reject or defer unavailable/invalid results.
  `mutation_candidate()` is the explicit fail-closed boundary for new mutation consumers.
- [x] Define which presentation-only consumers may intentionally summarize or
  omit detailed evidence.
  UI panels may use `presentation_summary()`; mutation and operator decisions retain the typed outcome.

Completion criteria:

- [x] Callers cannot confuse exhaustion, invalid input, unavailable dependencies, or ordinary absence.
  Typed `OccurrenceOutcome` statuses and `next_outcome()` keep these states distinct at the evaluator boundary.
- [x] Type checking covers all public scheduler-result boundaries.
  `OccurrenceOutcome` and its immutable variants are the typed evaluator boundary; focused tests cover each state.

## 4. Introduce a Task-Scoped Evaluation Session

- [x] Let one task-scoped session own a `CompiledSchedule`, provider bindings,
  validated runtime dependencies, and bounded evaluation caches.
- [x] Cache provider bindings and repeated cursor lookups within one task evaluation.
- [x] Keep mutable caches task-scoped and prevent cross-task state leakage.
- [x] Replace broad callback assembly with narrow service dependencies.
  New scheduling consumers can depend on `EvaluationSession.next_outcome()` and `collect_after_cursor()`.
- [x] Reuse the same session for planning, feedback, preview, and timeline work
  within one hook or operator request.
- [x] Invalidate or rebuild a session when scheduling-affecting task fields or
  configuration fingerprints change.

Completion criteria:

- [x] One task evaluation does not rebuild configuration or provider state unnecessarily.
  The session owns one evaluator and bounded task-local cache for its request lifetime.
- [x] Repeated evaluation remains deterministic and isolated across tasks and processes.
  Fingerprint matching rejects cross-task reuse and refresh clears stale state.

## 5. Use One Scheduling Entry Point

- [x] Define one scheduler service accepting a `CompiledSchedule`, typed
  cursor, and task-scoped evaluation session.
- [x] Expose explicit operations for next occurrence, bounded range collection,
  and finite preview collection; all return typed occurrence outcomes.
- [ ] Route preview, hint, completion, reconcile, navigator, natural previews,
  file-backed scheduling, and range collection through that service.
  The service boundary exists; operational consumer migration remains below.
- [ ] Remove direct scheduling calls from orchestration and presentation modules.
- [ ] Remove `next_for_or`, `next_for_and`, atom-level, and alternate expression
  entry points from operational consumers as each is migrated.
- [ ] Remove obsolete public compatibility exports. External tooling must use
  the new scheduler service after cutover.
- [ ] Delete fallback branches that resume through a different scheduler after
  configuration, timezone, astronomy, or provider failure.

### Behavioral Migration Sequence

Migration policy:

- [x] Keep old/new comparison in tests only. Do not add dual runtime execution,
  fallback scheduling, or production compatibility bridges.
- [ ] Replace each consumer and remove its direct scheduler route in the same
  pass once focused parity tests pass.
- [ ] Push each independently verified pass to `scheduler-engine-v6`; merge only
  after the final operational and performance gates pass.

Passes:

- [x] Complete the service contract. Replace plain collection/preview lists
  with an immutable typed result carrying occurrences, cursor/source evidence,
  terminal exhaustion, and an explicit empty-result reason. Validate that the
  service, session, compiled schedule, and cursor share one context.
- [x] Add a test-only parity harness covering local/UTC timestamps, ordering,
  monotonicity, omissions, exhaustion, deterministic random selection, DST,
  astronomy, business calendars, and file-backed schedules.
  The harness is callback-injected and runtime-independent; its matrix expands as each consumer migrates.
- [ ] Migrate read-only presentation paths first: on-add preview, natural
  preview, navigator explain, and timeline preview. Remove their scheduler
  callback parameters after parity is established. Navigator and timeline
  evaluator ownership now use `SchedulerService`; on-add collection remains on
  the verified evaluator path until omission/event parity is complete.
- [ ] Migrate hints and range collection in `precompute.py`,
  `precompute_api.py`, and `recurrence_candidates.py` to one shared session and
  typed collection result. This is the next architectural boundary: these APIs
  currently accept raw DNF and callback bundles, so completion requires changing
  their contract to accept `CompiledSchedule`/`SchedulerService`, not adding a
  runtime adapter around the old path.
- [ ] Consolidate expression/file inclusion and omission scheduling behind the
  service. Keep `anchor_inclusion.py`, `anchor_files.py`, and `anchor_omit.py`
  as internal engine components rather than consumer entry points.
- [ ] Migrate completion to one service per task operation. Reuse it for child
  selection, limits, timeline, and feedback; invalid or unavailable outcomes
  must abort or defer mutation.
- [ ] Migrate reconcile to one service per task with validated Taskdata
  configuration. Preserve terminal evidence and fail closed on unavailable
  configuration or providers.
- [ ] Remove remaining operational calls to low-level expression, AND/OR,
  atom, range, and alternate scheduling APIs. Remove obsolete facade and
  compatibility exports while retaining only private engine primitives.
- [ ] Run the full golden, black-box, deployment, mypy, desktop performance,
  and both Termux performance gates. Confirm the direct operational call-site
  inventory is empty and no mutation accepts a non-found outcome.

#### Hints And Range Collection Design

Ownership model:

- [x] Keep `SchedulerService` as the only component allowed to interpret a
  `CompiledSchedule` or resolve occurrences.
- [x] Make hint generation a pure consumer of typed scheduler results. It must
  not inspect raw Taskwarrior fields, parser DNF, provider internals, or accept
  scheduling callbacks.
- [x] Treat preview, hints, and large-range enumeration as consumers of one
  bounded collection operation rather than separate scheduling engines.

Typed contract:

- [x] Introduce an immutable `OccurrenceRangeRequest` carrying an
  `OccurrenceCursor`, optional end boundary, result limit, and omission policy.
- [ ] Complete collection outcomes so callers can distinguish found, empty,
  date-limit exhaustion, search-limit exhaustion, unavailable dependencies,
  and invalid schedules.
- [x] Preserve occurrence ordering, local and UTC evidence, source/provider
  evidence, and terminal evidence in every bounded collection result.
- [x] Keep preview formatting outside `SchedulerService`; preview should submit
  a range request and render the returned typed result.

Hint generation and caching:

- [x] Replace `precompute.py` scheduling logic with a pure `HintBuilder` that
  aggregates typed collection results from one `SchedulerService`.
- [x] Key derived hint data by compiled schedule cache key, hint-request schema,
  query bounds, and file/calendar/astronomy resource fingerprints.
- [x] Cache only derived serializable hint output. Never cache runtime service
  objects, callbacks, mutable evaluator state, or parser structures.
- [x] Treat unavailable or invalid scheduling as an actionable typed failure,
  never as a valid zero-occurrence hint result.

Implementation passes:

- [x] Pass 1: add `OccurrenceRangeRequest` and the complete typed collection
  outcome family; validate cursor, context, end boundary, limit, and monotonicity.
- [x] Pass 2: rewrite hint generation to consume
  `SchedulerService.collect(range_request)` and add test-only parity coverage.
- [x] Pass 3: rewrite large-range candidate enumeration through the same
  request/result contract, preserving finite terminal evidence and limits.
- [x] Pass 4: remove raw-DNF and callback parameters from hint/range APIs;
  delete `precompute_api.py` and obsolete `recurrence_candidates.py` adapters
  once the operational call-site inventory is empty. The old public adapters
  and `core.anchors_between_expr` export are intentionally removed; range
  consumers now use `SchedulerService.collect_request`.
- [x] Pass 5: extract the pure `HintBuilder`, avoid service construction on
  cache hits, and include request schema/start bounds in hint cache keys.

Completion criteria:

- [x] Hints, previews, and range enumeration cannot resolve an occurrence
  outside `SchedulerService`.
- [x] Hint and range modules are independently testable as pure consumers of
  typed scheduler results.
- [x] No runtime compatibility adapter remains around the old callback-based
  precompute or candidate APIs.

Section 5 completion criteria:

- [ ] All operational paths resolve occurrences through one authoritative service.
- [ ] No feature can silently fall back to a different scheduler implementation.

## 6. Separate Date Selection from Time Projection

- [ ] Make calendar-date selection an explicit scheduler phase.
- [ ] Make `@t`, time windows, astronomy events, DST resolution, and time offsets a separate projection phase.
- [ ] Model projection as typed success, unavailable-on-date, invalid, or
  terminal rather than mutating/advancing the date implicitly.
- [ ] Define how unavailable astronomical events advance within a matching date window.
- [ ] Preserve local-date semantics and UTC serialization across DST boundaries.
- [ ] Cover ordinary times, lists, equal partitions, interval windows,
  overnight windows, random windows, and astronomical offsets.
- [ ] Ensure a projection cannot return an instant that violates the cursor,
  selected date, omission policy, or chain limit.

Completion criteria:

- [ ] Date constraints and time projection can be tested independently.
- [ ] Time projection cannot alter the selected recurrence date without explicit evidence.

## 7. Formalize Occurrence Providers

- [ ] Define one narrow provider protocol for weekly, monthly, yearly,
  positional selection, business-calendar, astronomy, moon-phase, seasonal,
  and file-backed sources.
- [ ] Require providers to declare cursor semantics, terminal evidence, and date bounds.
- [ ] Require monotonic results and explicit exhaustion; fabricated guard dates
  are forbidden.
- [ ] Standardize provider errors and unavailable-dependency behavior.
- [ ] Remove provider-specific orchestration from scheduler consumers.
- [ ] Keep deterministic random selection keyed by the central recurrence
  identity and provider period.
- [ ] Let file-backed providers retain one-pass cursors and hot-read indexes
  without exposing their cache implementation to the scheduler.

Completion criteria:

- [ ] Providers are substitutable behind one documented protocol.
- [ ] Adding a provider does not require changes across hook and operator-tool paths.

## 8. Add Cross-Path Conformance Coverage

- [ ] Assert identical occurrences for preview, completion, reconcile, navigator, hints, and range collection.
- [ ] Cover adjacent dates, sparse rules, intervals, AND/OR terms, omissions, selections, astronomy, files, and random schedules.
- [ ] Add generated recurrence matrices for cursor monotonicity, determinism,
  timezone preservation, and finite termination.
- [ ] Verify a valid prefix followed by exhaustion retains terminal evidence in every consumer.
- [ ] Add differential tests between each provider's reference and optimized
  paths before enabling an optimization.
- [ ] Run deterministic shuffled-order tests to expose leaked session or cache
  state.

Completion criteria:

- [ ] Operational paths cannot disagree about the next occurrence for the same context.
- [ ] Generated tests detect skipped, duplicated, fabricated, and non-monotonic occurrences.

## 9. Add Optional Scheduler Tracing

- [ ] Add a structured diagnostic trace disabled by default.
- [ ] Record proposed candidates, rejected constraints, selected terms, provider identity, and terminal reason.
- [ ] Emit traces only to diagnostic channels and never contaminate hook JSON stdout.
- [ ] Expose concise trace summaries through doctor or navigator diagnostics.
- [ ] Bound trace size and redact task/configuration values that are not needed
  to explain scheduling decisions.

Completion criteria:

- [ ] Complex scheduling failures can be explained without instrumenting production code manually.
- [ ] Tracing has negligible cost when disabled.

## 10. Add Provider-Certified Optimizations

- [ ] Let providers advertise safe batch generation, arithmetic counting, or cursor reuse capabilities.
- [ ] Keep optimized paths inside the provider that owns the recurrence semantics.
- [ ] Differentially test every optimized result against the authoritative scheduler path.
- [ ] Retain an optimization only when dates, evidence, and exhaustion are identical.
- [ ] Benchmark desktop and Termux performance before and after each optimization.
- [ ] Measure cache hit rate and allocation cost for large omission and
  anchor-file schedules.
- [ ] Reject an optimization if its complexity is not justified by measured
  operational latency.

Completion criteria:

- [ ] No optimization independently reimplements recurrence semantics.
- [ ] Fast and reference paths are behaviorally identical across the conformance matrix.

## 11. Remove Old Scheduler Ownership

- [ ] Remove superseded scheduling callbacks, facades, fallback imports, and
  shadow implementations after all operational consumers use the service.
- [ ] Keep parser, provider, projection, session, result, and service ownership
  in focused modules; do not create another scheduler monolith.
- [ ] Reduce `scheduler_api.py`, `recurrence_evaluator.py`, and core facade
  exports to explicit ownership boundaries.
- [ ] Update runtime manifests, installer validation, doctor diagnostics, and
  mypy scopes for every new or removed module.
- [ ] Search hooks, tools, tests, and documentation for direct legacy entry
  points and remove remaining references.
- [ ] Delete obsolete scheduler caches and bump cache/schema fingerprints where
  derived formats changed.

Completion criteria:

- [ ] There is one production scheduler service and one provider protocol.
- [ ] No operational path imports or invokes a removed scheduler entry point.
- [ ] Hooks and operator tools depend only on public scheduler contracts.

## Final Verification

- [ ] Run the full golden, black-box, deployment, mypy, hook-protocol, and
  deterministic shuffled-order suites.
- [ ] Run add preview, CP/anchor completion, expiration, queue, reconcile,
  navigator, anchor-file, astronomy, and hint workflow benchmarks.
- [ ] Run enforced performance profiles on both Termux devices.
- [ ] Confirm hook stdout remains strict JSON for successful and failing inputs.
- [ ] Confirm preview, completion, reconcile, navigator, and hints return the
  same occurrence and terminal evidence from equal inputs.
- [ ] Confirm malformed configuration, unavailable astronomy, malformed files,
  exhausted searches, and invalid grammar fail closed with actionable output.
- [ ] Run doctor and reconcile against isolated Taskdata after deleting all
  derived scheduler caches.
- [ ] Merge `scheduler-engine-v6` into `main` only when every criterion passes.
- [ ] Remove this checklist after every item and completion criterion is satisfied.
