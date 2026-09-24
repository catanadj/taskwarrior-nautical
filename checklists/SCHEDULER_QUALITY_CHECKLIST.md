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
- [x] Move each operational consumer to the new contract and delete its old
  path in the same pass once focused parity tests pass. Navigator, hints,
  evaluator, anchor helpers, and inclusion now use the typed service/private
  engine; callback-only hint and range implementations were removed.
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
- [x] Route preview, hint, completion, reconcile, navigator, natural previews,
  file-backed scheduling, and range collection through that service.
- [x] Remove direct scheduling calls from orchestration and presentation modules.
  Remaining low-level calls are confined to internal engine/provider modules and
  test-only characterization helpers.
- [x] Remove `next_for_or`, `next_for_and`, atom-level, and alternate expression
  entry points from operational consumers as each is migrated.
- [x] Remove obsolete public compatibility exports. External tooling must use
  the new scheduler service after cutover.
- [x] Delete operational fallback branches that resume through a different
  scheduler after configuration, timezone, astronomy, or provider failure.
  Test-only helper fallbacks remain available for isolated engine tests.

### Behavioral Migration Sequence

Migration policy:

- [x] Keep old/new comparison in tests only. Do not add dual runtime execution,
  fallback scheduling, or production compatibility bridges.
- [x] Replace each consumer and remove its direct scheduler route in the same
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
- [x] Migrate read-only presentation paths first: on-add preview, natural
  preview, navigator explain, and timeline preview. Remove their scheduler
  callback parameters after parity is established. Navigator explain/date/
  calendar projection, on-add first/collection paths, and timeline event
  rendering now use `SchedulerService`; formatting remains outside the service.
- [x] Migrate hints and range collection to one shared session and typed
  collection result. `HintBuilder` owns hint aggregation; the obsolete
  `precompute_api.py`, `recurrence_candidates.py`, and raw public exports were
  removed, with omission and failure provenance covered by range tests.
- [x] Consolidate expression/file inclusion and omission scheduling behind the
  service for operational consumers. On-add, completion, timeline, navigator,
  and range/hint paths now use task-scoped service collections; keep
  `anchor_inclusion.py`, `anchor_files.py`, and `anchor_omit.py` as internal
  engine components rather than consumer entry points.
- [x] Migrate completion to one service per task operation. Child selection,
  limits, included occurrence collection, timeline event collection, and
  feedback now share the service; callback-shaped parameters remain only as
  test-only helper seams. Invalid or unavailable outcomes abort or defer
  mutation.
- [x] Migrate reconcile to one shared chain-generation/scheduler service per
  task with validated Taskdata configuration. Preserve terminal evidence and
  fail closed on unavailable configuration or providers.
- [x] Remove remaining operational calls to low-level expression, AND/OR,
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
- [x] Complete collection outcomes so callers can distinguish found, empty,
  date-limit exhaustion, search-limit exhaustion, unavailable dependencies,
  and invalid schedules. `include` returns marked events and `report` returns
  omitted evidence separately. Unavailable and invalid collection failures are
  returned as typed result variants rather than empty success results.
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
- [x] Pass 6: expose omission provenance through bounded `exclude`, `include`,
  and `report` range policies without changing normal scheduling semantics.
- [x] Pass 7: wrap unavailable, invalid, and exhaustion failures in the typed
  collection result while preserving request-validation errors at the boundary.

Completion criteria:

- [x] Hints, previews, and range enumeration cannot resolve an occurrence
  outside `SchedulerService`.
- [x] Hint and range modules are independently testable as pure consumers of
  typed scheduler results.
- [x] No runtime compatibility adapter remains around the old callback-based
  precompute or candidate APIs.

Section 5 completion criteria:

- [x] All operational paths resolve occurrences through one authoritative service.
- [x] No operational feature can silently fall back to a different scheduler
  implementation; only test-only direct helper adapters remain.

## 6. Separate Date Selection from Time Projection

- [x] Make calendar-date selection an explicit scheduler phase. The evaluator
  selects the recurrence date first and passes it unchanged to projection.
- [x] Make `@t`, time windows, astronomy events, DST resolution, and time offsets a separate projection phase.
- [x] Model projection as typed success, unavailable-on-date, invalid, or
  terminal rather than mutating/advancing the date implicitly.
- [x] Define how unavailable astronomical events advance within a matching date window.
- [x] Preserve local-date semantics and UTC serialization across DST boundaries.
- [x] Cover ordinary times, lists, equal partitions, interval windows,
  overnight windows, random windows, and astronomical offsets.
- [x] Ensure a projection cannot return an instant that violates the cursor,
  selected date, omission policy, or chain limit.

Completion criteria:

- [x] Date constraints and time projection can be tested independently.
- [x] Time projection cannot alter the selected recurrence date without explicit evidence.

## 7. Formalize Occurrence Providers

- [x] Define one narrow provider protocol for weekly, monthly, yearly,
  positional selection, business-calendar, astronomy, moon-phase, seasonal,
-  and file-backed sources. `ProviderContract` now carries source identity,
  cursor semantics, finiteness, and optional date bounds for all adapters.
- [x] Require providers to declare cursor semantics, terminal evidence, and date bounds
  on authoritative evaluator collection; bounded collectors validate declared dates
  and preserve `OccurrenceBatch.terminal` evidence.
- [x] Require monotonic results and explicit exhaustion; fabricated guard dates
  are forbidden. The bounded collector enforces instant-forward progress and
  retains date-limit terminal evidence.
- [x] Standardize provider errors and unavailable-dependency behavior.
  Collector failures now use `OccurrenceProviderUnavailable` or
  `OccurrenceProviderInvalid`; scheduler exhaustion remains terminal evidence.
- [x] Remove provider-specific orchestration from scheduler consumers.
  Evaluator event collection now uses the shared event-provider adapter and
  bounded collector rather than a second hand-written stream loop.
- [x] Keep deterministic random selection keyed by the central recurrence
  identity and provider period. Existing random providers consume the
  task-scoped chain identity and period key.
- [x] Let file-backed providers retain one-pass cursors and hot-read indexes
  without exposing their cache implementation to the scheduler.

Completion criteria:

- [x] Providers are substitutable behind one documented protocol.
- [x] Adding a provider does not require changes across hook and operator-tool paths.

## 8. Add Cross-Path Conformance Coverage

- [x] Assert identical occurrences for preview, completion, reconcile, navigator, hints, and range collection.
  The service matrix covers preview/range directly and the existing operational
  path tests exercise completion, reconcile, navigator, and hint consumers.
- [x] Cover adjacent dates, sparse rules, intervals, AND/OR terms, omissions, selections, astronomy, files, and random schedules.
- [x] Add generated recurrence matrices for cursor monotonicity, determinism,
  timezone preservation, and finite termination.
- [x] Verify a valid prefix followed by exhaustion retains terminal evidence in every consumer.
- [x] Add differential tests between each provider's reference and optimized
  paths before enabling an optimization.
- [x] Run deterministic shuffled-order tests to expose leaked session or cache
  state.

Completion criteria:

- [x] Operational paths cannot disagree about the next occurrence for the same context.
- [x] Generated tests detect skipped, duplicated, fabricated, and non-monotonic occurrences.

## 9. Add Optional Scheduler Tracing

- [x] Add a structured diagnostic trace disabled by default. `SchedulerTrace` is
  opt-in through `NAUTICAL_SCHEDULER_TRACE` plus the diagnostic gate.
- [x] Record proposed candidates, rejected constraints, selected terms, provider identity, and terminal reason.
- [x] Emit traces only to diagnostic channels and never contaminate hook JSON stdout.
- [x] Expose concise trace summaries through navigator diagnostics with
  `nautical_navigator.py --explain '<anchor>' --trace`.
- [x] Bound trace size and redact task/configuration values that are not needed
  to explain scheduling decisions.

Completion criteria:

- [x] Complex scheduling failures can be explained without instrumenting production code manually.
- [x] Tracing has negligible cost when disabled; disabled traces do not allocate
  events and the regression test covers this contract.

## 10. Add Provider-Certified Optimizations

- [x] Let providers advertise safe batch generation, arithmetic counting, or cursor reuse capabilities.
  `ProviderCapabilities` is explicit; arithmetic remains uncertified and
  therefore disabled for all current providers.
- [x] Keep optimized paths inside the provider that owns the recurrence semantics.
  Anchor-file batching and cursor reuse stay inside `AnchorFileOccurrenceProvider`.
- [x] Differentially test every optimized result against the authoritative scheduler path.
- [x] Retain an optimization only when dates, evidence, and exhaustion are identical.
  Batch and cursor tests compare fresh-provider reference results.
- [x] Benchmark desktop and Termux performance before and after each optimization.
  `perf.termux.flagship_section10` and `perf.termux.normal_section10` both
  include the dedicated `anchor_file_batch_provider` check and pass it. Hot,
  non-monotonic, and business-day anchor-file paths pass on both devices. The
  cold 5,000-row load remains above the generic budget and is tracked as a
  separate future cold-start optimization, not a batch-path regression.
- [x] Measure cache hit rate and allocation cost for large omission and
  anchor-file schedules. Anchor-file cache statistics expose lookups, builds,
  records, hits, and hit ratio for diagnostics/benchmarks.
- [x] Reject an optimization if its complexity is not justified by measured
  operational latency. No arithmetic/counting fast path is enabled without a
  provider-owned implementation, differential coverage, and a budget.

Completion criteria:

- [x] No optimization independently reimplements recurrence semantics.
- [x] Fast and reference paths are behaviorally identical across the conformance matrix.

## 11. Remove Old Scheduler Ownership

- [x] Remove superseded scheduling callbacks, facades, fallback imports, and
  shadow implementations after all operational consumers use the service.
  Public low-level aliases were removed from the export contract; callback-only
  hint and range implementations were deleted.
- [x] Keep parser, provider, projection, session, result, and service ownership
  in focused modules; do not create another scheduler monolith. The service,
  evaluator, providers, and projections remain separate modules.
- [x] Reduce `scheduler_api.py`, `recurrence_evaluator.py`, and core facade
  exports to explicit ownership boundaries. The facade exposes the typed
  service contract; low-level engine bindings remain private implementation
  seams for the evaluator and provider code.
- [x] Update runtime manifests, installer validation, doctor diagnostics, and
  mypy scopes for every new or removed module. Deployment sanity checks the
  scheduler ownership boundary and strict typing includes the service modules.
- [x] Search hooks, tools, tests, and documentation for direct legacy entry
  points and remove remaining operational references. Test-only characterization
  helpers may still exercise private engine behavior deliberately.
- [x] Delete obsolete scheduler caches and bump cache/schema fingerprints where
  derived formats changed. No obsolete scheduler cache or changed derived
  format remained after the cutover, so no fingerprint bump was needed.

Completion criteria:

- [x] There is one production scheduler service and one provider protocol.
  `SchedulerService` is the public occurrence boundary and `ProviderContract`
  is the provider boundary; low-level scheduler aliases are no longer public.
- [x] No operational path imports or invokes a removed scheduler entry point.
  Deployment sanity now performs an AST ownership check for direct public
  scheduler calls.
- [x] Hooks and operator tools depend only on public scheduler contracts.
  Navigator, hints, completion, reconcile, and anchor helpers use the service
  or the private core-bound engine; test-only characterization helpers remain
  isolated.

## Final Verification

- [x] Run the full golden, black-box, deployment, mypy, hook-protocol, and
  deterministic shuffled-order suites.
- [x] Run add preview, CP/anchor completion, expiration, queue, reconcile,
  navigator, anchor-file, astronomy, and hint workflow benchmarks.
- [x] Run enforced performance profiles on both Termux devices.
- [x] Confirm hook stdout remains strict JSON for successful and failing inputs.
- [x] Confirm preview, completion, reconcile, navigator, and hints return the
  same occurrence and terminal evidence from equal inputs.
- [x] Confirm malformed configuration, unavailable astronomy, malformed files,
  exhausted searches, and invalid grammar fail closed with actionable output.
- [x] Run doctor and reconcile against isolated Taskdata after deleting all
  derived scheduler caches.
- [ ] Merge `scheduler-engine-v6` into `main` only when every criterion passes.
- [x] Retain this checklist locally for the merge handoff; do not stage or push it.
