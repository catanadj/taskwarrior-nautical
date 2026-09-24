# Nautical Local Query API Checklist

Expose Nautical recurrence knowledge to external tools through a stable,
versioned, read-only local CLI contract. External consumers must not import
`nautical_core`, export and reinterpret Taskwarrior data, or reproduce Nautical
scheduling behavior.

Version 1 intentionally answers one question well: which schedule occurrences
match a real Nautical task within an explicitly bounded range. Later operations
may expose lifecycle projections, chain inspection, and streaming queries
without changing the version 1 occurrence contract.

## Scope And Design Rules

- [x] Expose the contract as `nautical query`; do not introduce a daemon,
  socket, HTTP server, or background service.
- [x] Make strict JSON the public transport so shell scripts and tools written
  in any language can consume the API.
- [x] Keep `SchedulerService` as the sole authority for recurrence evaluation.
- [x] Keep the read-only Taskwarrior unit of work as the sole authority for
  resolving real tasks and Taskdata configuration.
- [x] Do not make internal Python imports part of the supported public API.
- [x] Do not add a second parser, scheduler, time projector, omission engine,
  astronomy resolver, calendar resolver, or random-time implementation.
- [x] Do not silently fall back to expression-only evaluation when a real task
  or validated scheduling dependency cannot be loaded.
- [x] Require every occurrence request to be bounded by an end instant or a
  count, with a hard safety cap.
- [x] Keep version 1 read-only. No query operation may stage lifecycle work,
  modify Taskwarrior, update links, repair metadata, or write task state.
- [x] Preserve Unicode with `ensure_ascii=False`.
- [x] Keep stdout to exactly one JSON document. Emit diagnostics to stderr only
  when `NAUTICAL_DIAG=1`.

The target flow is:

```text
external tool
     |
nautical query
     |
versioned query models
     |
read-only query service
    / \
task read repository   SchedulerService
    \ /
versioned JSON response
```

## Baseline And Inventory

- [x] Record current desktop behavior for Navigator occurrence projection and
  direct `SchedulerService.collect_request()` use.
- [x] Record current cold-process and warm-process timings for one task and a
  representative task batch.
- [x] Record Taskwarrior read counts for UUID, chainID, and broad active-task
  reads; the query performance baseline records one authoritative repository
  read for each selector shape, including the batched UUID path.
- [x] Add characterization cases for a fixed time list, an equally partitioned
  time window, a duration-stepped window, an overnight window, and random
  times.
- [x] Inventory the task lookup and recurrence projection logic currently used
  by Navigator so the query API can reuse its read boundary without copying
  presentation code.
- [x] Confirm the installed runtime contains every scheduler, configuration,
  provider, astronomy, calendar, and read-service module required by queries.

Completion criteria:

- [x] Baseline tests and timings run without the user's live Taskdata.
- [x] Every existing component that can answer part of the query has a named
  owner; no implementation begins by copying a consumer-specific helper.

## 1. Define The Versioned Query Contract

- [x] Add a focused `query_models.py` containing immutable request and response
  models with no Taskwarrior subprocess or scheduling behavior.
- [x] Define an integer API version and a stable schema identifier, initially
  `nautical.query.occurrences` version `1`.
- [x] Define a top-level request envelope containing version, operation,
  selector, range, omission policy, and safety limits.
- [x] Define a top-level response envelope containing version, operation,
  status, authoritative timezone, query echo, results, and structured failure.
- [x] Define typed task selectors for exactly one of:
  - one or more full UUIDs or unambiguous UUID prefixes;
  - one chainID;
  - all active Nautical tasks.
- [x] Do not expose unstable Taskwarrior numeric IDs as durable task identity in
  version 1.
- [x] Define result statuses that cannot be confused: `found`, `empty`,
  `exhausted`, `absent`, `unavailable`, and `invalid`.
- [x] Define structured failures with a stable code, human-readable message,
  retryability, and optional task identity. Do not expose tracebacks or depend
  on exception class names as the public contract.
- [x] Reject unknown request versions and operations explicitly.
- [x] Define the compatibility rule: fields may only be added within version 1
  when old consumers can safely ignore them; changed meaning or removal
  requires a new version.
- [x] Validate every model at construction and reject contradictory selector,
  range, and limit combinations before reading Taskwarrior.

Completion criteria:

- [x] Contract models round-trip through JSON without loss of Unicode or
  timestamp precision.
- [x] Invalid states cannot be constructed silently.
- [x] An external consumer can branch entirely on stable version, operation,
  status, and failure code fields.

## 2. Specify Occurrence Semantics Precisely

- [x] Set version 1 `basis` to `schedule`: results are recurrence calendar
  matches, not promises that lifecycle tasks will be created.
- [x] State explicitly that `anchor_mode`, `chainMax`, `chainUntil`, current
  completion state, and queued lifecycle work do not turn schedule matches into
  projected chain links in this operation.
- [x] Reserve actual lifecycle projection for a future, separately named
  operation such as `nautical query next`.
- [x] Support CP, anchor, and anchor-file schedules through their authoritative
  scheduler/provider paths.
- [x] Define an inclusive `from` boundary and an inclusive `to` boundary using
  `OccurrenceCursor` and `OccurrenceRangeRequest`; do not subtract arbitrary
  time in the CLI adapter.
- [x] Permit a strict-after cursor as an explicit alternative for count-based
  queries. Reject requests that specify both inclusive and exclusive starts.
- [x] Accept RFC 3339 timestamps with an explicit offset. Permit date-only
  boundaries as local calendar dates in Nautical's configured timezone.
- [x] Reject ambiguous offset-free timestamps rather than guessing during DST
  folds or timezone changes.
- [x] Use Nautical's validated timezone for evaluation. Do not allow a query to
  override recurrence timezone semantics; consumers can convert returned UTC.
- [x] Require either `to` or `count`. If both are present, stop at whichever
  bound is reached first.
- [x] Define hard caps for requested tasks, returned occurrences per task, and
  total returned occurrences.
- [x] Support omission policies `exclude`, `include`, and `report` with the same
  meaning as `OccurrenceRangeRequest`.
- [x] Preserve deterministic random behavior by evaluating the complete real
  task identity, including chainID and link context. Never seed task queries
  from an expression string or filesystem path alone.
- [x] Return every time projected by multi-time expressions, including time
  lists, composed windows, equal partitions, duration steps, overnight
  windows, astronomical times, and random windows.
- [x] Preserve strict chronological ordering by instant and deterministic
  tie-breaking by task UUID and provider evidence.

Completion criteria:

- [x] The same task, configuration, range, and API version always produce the
  same ordered response for deterministic schedules.
- [x] Query results match the scheduler used by add, completion, reconcile, and
  Navigator for the same cursor and occurrence basis.
- [x] Documentation makes it impossible to mistake schedule matches for future
  Taskwarrior child tasks.

## 3. Build The Read-Only Task Query Boundary

- [x] Resolve validated Taskdata, configuration, timezone, calendars,
  astronomy profiles, presets, and task binary once per query invocation.
- [x] Construct the integration unit of work with `IntegrationAccess.READ_ONLY`.
  The service consumes an already validated operator UOW; CLI construction is a later pass.
- [x] Reject execution if the context is not demonstrably read-only.
- [x] Resolve UUID selectors through typed repository reads and require exactly
  one match per UUID or prefix.
- [x] Resolve a chainID with one authoritative chain snapshot.
- [x] Resolve `all` with one broad snapshot containing only tasks that have a
  complete Nautical recurrence identity.
- [x] Treat a successful authoritative empty read as `absent` or `empty` only
  where the query contract permits it.
- [x] Preserve `Unavailable` for malformed JSON, command rejection, lock,
  timeout, missing binary, incomplete snapshot, ambiguous prefix, or unsafe
  configuration.
- [x] Never pass raw Taskwarrior filter expressions supplied by a caller into a
  command in version 1.
- [x] Normalize literal-null and empty optional UDAs through the same
  recurrence specification boundary used by operational scheduling.
- [x] Do not expose full raw task documents in query responses. Return only
  documented identity and recurrence metadata.

Completion criteria:

- [x] Query code performs no direct subprocess invocation.
- [x] A failed task read cannot become an empty occurrence result.
- [x] Batch selectors reuse one authoritative snapshot when its scope is
  sufficient.

## 4. Add One Occurrence Query Service

- [x] Add `query_service.py` as the sole orchestration boundary between query
  models, read-only task resolution, and scheduling.
- [x] Build one `SchedulerService` task session for each resolved task and reuse
  it for the complete request.
- [x] Translate the public range model into one validated
  `OccurrenceRangeRequest` without reinterpreting grammar or cursor semantics.
- [x] Call `SchedulerService.collect_request()` for authoritative occurrence
  collection.
- [x] Map `OccurrenceCollectionResult` states to public query statuses without
  collapsing failure, exhaustion, absence, or ordinary empty results.
- [x] Keep task-specific failures isolated inside batch results when the broad
  Taskwarrior snapshot and configuration remain authoritative.
- [x] Fail the complete request when shared configuration, timezone, Taskdata,
  or broad snapshot state is unavailable.
- [x] Include the opaque compiled schedule fingerprint and scheduling
  configuration fingerprint so consumers can invalidate their own caches.
- [x] Do not persist query results or introduce a cross-process task cache.
- [x] Reuse existing file-backed and compiled-schedule caches only through
  their current validated APIs.
- [x] Keep presentation text and Rich rendering out of the service.

Completion criteria:

- [x] The service can be tested without argument parsing or terminal output.
- [x] The service contains no parser, provider, time projection, or recurrence
  calculations of its own.
- [x] One task failure cannot corrupt or reorder another task's result.

## 5. Define The Public JSON Response

- [x] Include these top-level fields: `schema`, `version`, `operation`,
  `status`, `basis`, `timezone`, `query`, `results`, and `failure`.
- [x] Include a compact task identity per result: full UUID, chainID, link,
  description, recurrence kind, and opaque schedule fingerprint.
- [x] Echo normalized range boundaries, inclusivity, count, omission policy,
  and applied safety caps.
- [x] Represent each occurrence with:
  - local RFC 3339 timestamp and timezone name;
  - UTC RFC 3339 timestamp;
  - UTC offset and DST fold where relevant;
  - source/provider identifier;
  - source description when available;
  - omission state and omission evidence when requested.
- [x] Preserve provider terminal evidence in a documented structured form
  instead of serializing an exception string.
- [x] Use `null` only for fields whose absence is explicitly documented.
- [x] Keep object keys stable and occurrence arrays deterministically ordered.
- [x] Avoid presentation-only natural text in the machine contract. Consumers
  may render their own labels from structured data.
- [x] Keep internal paths, raw configuration, command argv, task annotations,
  and unrelated task fields out of ordinary responses.

Completion criteria:

- [x] Response fixtures are stable across repeated runs and dictionary input
  ordering.
- [x] Local and UTC timestamps identify the same instant across DST folds,
  gaps, and timezone offset changes.
- [x] A consumer never needs to parse human-readable messages to understand a
  successful response.

## 6. Add The `nautical query` CLI

- [x] Add `nautical_core/tools/nautical_query.py` as a thin transport adapter.
- [x] Register `query` in the top-level `nautical` launcher and installed
  runtime manifest.
- [x] Provide the initial command surface:

```text
nautical query capabilities
nautical query occurrences --uuid UUID --from START (--to END | --count N)
nautical query next --uuid UUID --from START --count 1
nautical query occurrences --chain-id CHAIN --from START (--to END | --count N)
nautical query occurrences --all --from START (--to END | --count N)
```

- [x] Permit repeated `--uuid` arguments for a bounded batch.
- [x] Add `--after` for explicit exclusive count-based queries and make it
  mutually exclusive with `--from`.
- [x] Add `--omissions=exclude|include|report` with `exclude` as the default.
- [x] Accept the same versioned request envelope from stdin through an explicit
  `--request -` mode; CLI flags and stdin must construct the same model.
- [x] Make JSON the only output format for `nautical query` version 1.
- [x] Emit exactly one JSON document followed by one newline on stdout.
- [x] Use `json.dumps(..., ensure_ascii=False)` and never emit Rich control
  sequences, spinners, panels, progress, or explanatory text.
- [x] Define stable exit behavior:
  - `0` for a valid response, including `empty` or `exhausted` results;
  - `2` for invalid CLI or request input;
  - a documented nonzero code for unavailable shared dependencies;
  - a separate documented nonzero code for unexpected internal failure.
- [x] Return a structured JSON failure document even when the process exits
  nonzero after argument parsing has established query mode.
- [x] Keep diagnostics silent by default and route diagnostic output only to
  stderr under `NAUTICAL_DIAG=1`.

Completion criteria:

- [x] Shell, Python, and another language can consume the output without
  importing Nautical.
- [x] Process-level tests prove stdout always contains exactly one valid JSON
  document and no diagnostic contamination.
- [x] CLI flag requests and equivalent stdin requests return equal payloads.

## 7. Add Capability Discovery And Extension Rules

- [x] Implement `nautical query capabilities` without reading Taskwarrior task
  data.
- [x] Report supported API versions, operations, selector kinds, omission
  policies, timestamp rules, and hard limits.
- [x] Report availability of optional providers such as astronomy without
  exposing secrets or full configuration.
- [x] Keep capability discovery read-only and fail clearly when core runtime or
  validated configuration cannot be loaded.
- [x] Reserve operation names for later `inspect` and `chains`; `next` is now
  implemented as a read-only projected successor operation.
  contracts without returning placeholder data in version 1.
- [x] Require every future operation to define its own basis and result model;
  do not add mode switches that change the meaning of `occurrences`.
- [x] Treat a long-running JSON-lines mode as a future transport optimization,
  not part of version 1 semantics.
- [x] Do not add a daemon merely to avoid Python startup; optimize imports or
  add explicit stdio streaming only after measurement.

Completion criteria:

- [x] A tool can detect compatibility before issuing a task query.
- [x] Future operations can be added without changing occurrence semantics or
  requiring consumers to import implementation modules.

## 8. Make Batch Queries Efficient And Bounded

- [x] Resolve repeated UUIDs, one chain, or all active tasks from the smallest
  authoritative Taskwarrior snapshot.
- [x] Reuse one validated configuration and integration context for the batch.
- [x] Reuse task-local scheduler sessions only within their owning task.
- [x] Deduplicate repeated task selectors while preserving deterministic
  response ordering.
- [x] Bound task count, occurrence count per task, total occurrence count,
  provider iterations, file skips, subprocess attempts, and command timeout.
- [x] Return an explicit truncation or exhaustion marker whenever a safety cap
  stops collection.
- [x] Never return an apparently complete response after truncation.
- [x] Make one unavailable task a per-task failure when the shared snapshot is
  authoritative; do not discard valid sibling results.
- [x] Record Taskwarrior call count and scheduler timing by purpose under
  diagnostics without logging task contents.
- [x] Add a benchmark for one task with many daily times and a batch of active
  Nautical tasks.

Completion criteria:

- [x] A multi-task query avoids one Taskwarrior subprocess per task.
- [x] Batch execution remains deterministic and cannot exceed documented
  memory, result-size, or iteration bounds.
- [x] Termux measurements show acceptable cold and batch latency without a
  background service.

## 9. Harden Failure And Security Boundaries

- [x] Fail closed on unsafe or malformed configuration, invalid timezone,
  missing astronomy profile, unreadable anchor files, invalid business
  calendar, malformed task JSON, and ambiguous task identity.
- [x] Keep `absent`, `empty`, `exhausted`, `unavailable`, and `invalid`
  distinguishable in both models and serialized output.
- [x] Verify the integration context cannot expose mutation services to query
  orchestration.
- [x] Prevent arbitrary Taskwarrior filters, shell fragments, filesystem paths,
  or command options from crossing the typed selector boundary.
- [x] Bound stdin size and JSON nesting before decoding a request.
- [x] Reject trailing non-whitespace after the stdin JSON document.
- [x] Avoid echoing sensitive environment variables, full Taskdata paths, or
  raw task documents in failures and diagnostics.
- [x] Preserve actionable provider and configuration errors without exposing
  tracebacks by default.
- [ ] Add interruption tests for timeout, SIGINT, broken pipe, and unavailable
  Taskwarrior state.

Completion criteria:

- [x] No failure path emits a successful empty result.
- [x] No query path can mutate Taskwarrior or Nautical lifecycle state.
- [x] Malformed or hostile input cannot escape the typed request boundary.

## 10. Integrate Installation, Doctor, And Deployment

- [x] Add query modules and tool entry points to the single authoritative
  runtime manifest used by installation and deployment validation.
- [x] Ensure install and upgrade place the query command and every lazy
  dependency in the managed release.
- [x] Add an installed-layout smoke test for `nautical query capabilities`.
- [ ] Add an isolated Taskdata smoke test for one fixed multi-time occurrence
  query.
- [ ] Extend doctor with a compact query API check covering runtime presence,
  version support, read-only context construction, and optional-provider
  availability.
- [x] Keep doctor diagnostics human-facing while its `--json` output remains
  structured and independent from query response schemas.
- [x] Make deployment sanity detect launcher commands or lazy modules missing
  from the runtime manifest.

Completion criteria:

- [x] A clean first install and an upgrade both expose the same query contract.
- [x] Installed-layout tests do not rely on the repository being importable.
- [ ] Doctor identifies missing query dependencies before an external tool
  encounters them.

## 11. Add Contract, Conformance, And Process Tests

- [x] Add construction and JSON round-trip tests for every request, response,
  selector, occurrence, terminal, and failure model.
- [ ] Add unknown-version, unknown-operation, contradictory-bound, excessive-
  limit, malformed-JSON, trailing-data, and oversized-input tests.
- [ ] Add scheduler conformance for:
  - fixed single and multiple times;
  - hour-only and minute-precision times;
  - duration-stepped and equal-partition windows;
  - composed and overnight windows;
  - deterministic random windows;
  - CP schedules;
  - anchor and anchor-file schedules;
  - omissions and business calendars;
  - astronomy and moon phases;
  - DST gaps, folds, and timezone offset changes;
  - sparse schedules and representable-date exhaustion.
- [x] Compare query occurrences with direct `SchedulerService` results for the
  same task, context, cursor, end, count, and omission policy.
- [ ] Add Taskwarrior read tests for found, absent, unavailable, ambiguous UUID
  prefix, empty chain, mixed statuses, and malformed export.
- [x] Add process-level tests proving strict stdout JSON, conditional stderr
  diagnostics, Unicode preservation, stable exit codes, and broken-pipe
  handling.
- [x] Add installed-runtime black-box tests that execute the managed launcher
  from outside the repository.
- [x] Add deterministic shuffled golden coverage and register every new test in
  the enforced test registry.
- [x] Add mypy coverage for query models, service, and CLI adapter without
  `Any` callback bundles.

Completion criteria:

- [x] All query behavior is tested through both the service and real process
  boundaries.
- [x] Query results cannot drift from the operational scheduler unnoticed.
- [ ] Full golden, black-box, deployment, mypy, and workflow checks pass.

## 12. Document The Public Contract

- [x] Document the local CLI contract in `Manual.md`, including versioning,
  selectors, bounds, statuses, timestamps, omissions, and exit behavior.
- [x] Include concise examples for one task, a chain, all active tasks, stdin
  requests, Unicode output, and omission reporting.
- [x] Include examples using `jq`, Python, and plain shell without importing
  Nautical internals.
- [x] Explain that occurrence results are schedule matches and not guaranteed
  future child tasks.
- [x] Explain that actual lifecycle projection will use a separate future
  operation rather than changing occurrence semantics.
- [x] Document hard caps, truncation evidence, optional-provider failures, and
  retryable unavailable results.
- [ ] Keep the README light; add at most a short pointer to the Manual if public
  query support belongs in the introductory feature list.
- [x] State explicitly that `nautical_core` is private and may change without
  preserving an external Python import contract.

Completion criteria:

- [x] An external tool author can implement a correct consumer using only CLI
  help, the Manual, and capability discovery.
- [x] Examples never require parsing human-readable output or Taskwarrior raw
  recurrence UDAs.

## 13. Performance And Final Completion Gates

- [x] Add desktop budgets for cold capability discovery, one UUID occurrence
  query, one multi-time query, one chain batch, and all-active batch.
- [x] Record the same reduced profile on both Termux devices.
- [x] Record and enforce Taskwarrior call counts for single and batch queries.
- [x] Confirm no query work runs during ordinary Taskwarrior hooks.
- [ ] Confirm the query implementation adds no eager imports to thin hook
  routing.
- [ ] Run full golden, deterministic shuffled golden, process protocol,
  black-box, deployment, mypy, and performance suites.
- [x] Run doctor and query smoke tests against an installed runtime outside the
  checkout.
- [ ] Verify all query commands are read-only with Taskwarrior command
  observation enabled.
- [ ] Freeze the version 1 request/response fixtures only after every semantic
  and process-level gate passes.
- [ ] Remove any temporary adapters, duplicate projections, debugging output,
  and test-only production branches introduced during implementation.

Final completion criteria:

- [ ] External tools can query all bounded occurrences of real Nautical tasks
  without exporting Taskwarrior data or reproducing Nautical behavior.
- [ ] Multi-time, random, file-backed, business-calendar, omission, astronomy,
  CP, DST, sparse, empty, and exhausted cases match the operational scheduler.
- [x] Invalid or unavailable inputs fail closed with structured, actionable
  JSON.
- [x] Single and batch queries remain bounded and practical on both Termux
  devices.
- [x] The installed CLI contract is versioned, documented, discoverable, and
  independent of Nautical's private module layout.
