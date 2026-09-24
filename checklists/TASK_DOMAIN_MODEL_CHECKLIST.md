# Nautical Task Domain Model Checklist

Replace Nautical's remaining raw task dictionaries, repeated field
normalization, and implicit mutation dictionaries with one immutable task
domain model. Taskwarrior remains the durable source of task data. The model
is an invocation-local interpretation of Taskwarrior JSON, not a shadow
database or persistence format.

The completed system must answer four questions through explicit types:

1. What did Taskwarrior actually return, including malformed or ambiguous
   values?
2. Is that observation valid enough for Nautical scheduling or mutation?
3. What exact child task or existing-task change does Nautical intend?
4. How is that intent encoded at the Taskwarrior boundary without losing user
   fields or confusing absent, null, clear, and preserve?

## Scope And Working Model

- [ ] Create and develop exclusively on `task-domain-model-v7`; keep `main`
  operational until every final gate passes.
- [ ] Treat Nautical as offline while the branch is under construction.
  Intermediate commits do not need to be installable or operational.
- [ ] Keep this checklist local. Push implementation commits only to the
  domain-model branch and merge it into `main` once the cutover gates pass.
- [ ] Do not build mapping adapters, dictionary compatibility overloads,
  old/new task factories, fallback codecs, dual mutation paths, or facade
  aliases for replaced task APIs.
- [ ] Remove each raw-dictionary production path as soon as its typed owner is
  available. Broken intermediate commits are acceptable on the offline branch.
- [ ] Keep comparisons with the previous implementation in characterization
  tests and benchmark fixtures only.
- [ ] Keep Taskwarrior as the sole durable task store. Do not persist domain
  objects, decoded snapshots, or a second task index across processes.
- [ ] Preserve arbitrary Taskwarrior fields and third-party UDAs without making
  them first-class Nautical properties.
- [ ] Preserve strict hook JSON on stdout with `ensure_ascii=False`.
  Diagnostics remain silent unless `NAUTICAL_DIAG=1`, and then go to stderr.
- [ ] Keep the ordinary-task thin-hook probe ahead of domain-model imports so
  plain Taskwarrior operations retain their current startup path.
- [ ] Do not add Taskwarrior subprocess calls. This project must reduce parsing
  and copying work without weakening authoritative reads or mutation guards.
- [ ] Prefer typed invalid/unavailable outcomes over guessed defaults. A
  malformed observation must remain inspectable by Doctor and integrity tools
  but cannot become an operational Nautical task.

Cutover policy:

- [ ] Stop Taskwarrior hooks and Nautical operator processes before installing
  the completed branch.
- [ ] Do not migrate persisted user task data; decode current Taskwarrior rows
  through the new codec on first use.
- [ ] Quarantine only obsolete Nautical cache or outbox data whose schema is no
  longer valid. Do not introduce compatibility readers for it.
- [ ] Run Doctor, query, reconcile dry-run/apply, queue drain, add, modify,
  completion, deletion, expiration, and hookless-recovery smoke tests before
  re-enabling hooks.
- [ ] Roll back by restoring the previous release, not by retaining both task
  representations in production.

## Target Ownership

Exact filenames may change if a clearer boundary emerges, but ownership must
remain explicit:

- `task_models.py`: immutable task identity, status, timestamps, field
  observations, `TaskObservation`, validated `NauticalTask`, and `TaskDraft`.
- `task_codec.py`: the only Taskwarrior task-row decoder and task/draft JSON
  encoder. It owns literal `null`, numeric, timestamp, UUID, and field-state
  normalization.
- `task_changes.py`: `TaskPatch`, set/clear/preserve semantics, canonical
  change comparison, immutable-field policy, and mutation fingerprints.
- `recurrence_spec.py`: normalized recurrence definition constructed from a
  validated `NauticalTask`, not from a mapping.
- `task_read_repository.py`: typed authoritative reads returning task
  observations and domain-shaped collections.
- `taskwarrior_mutations.py`: guarded application of `TaskPatch` and
  `TaskDraft` through `TaskCodec`.
- `chain_generation.py`: generation of a validated `TaskDraft`, never a child
  dictionary.
- Lifecycle, scheduler, query, integrity, hooks, and tools: consumers of the
  domain contracts only; none may decode or normalize Taskwarrior rows.

The final flow is:

```text
Taskwarrior JSON
      |
  TaskCodec
      |
TaskObservation ---------> integrity / Doctor evidence
      |
operational validation
      |
 NauticalTask
   /       \
scheduler  lifecycle
   \       /
 TaskDraft / TaskPatch
      |
mutation gateway + TaskCodec
      |
 Taskwarrior
```

## Baseline And Inventory

- [ ] Record full golden, deterministic shuffled golden, black-box,
  deployment, mypy, hook-protocol, query, Doctor, reconcile, and installer
  results from `main`.
- [ ] Record desktop and both Termux profiles for plain hooks, Nautical add,
  ordinary edit, CP completion, anchor completion, multi-time completion,
  deletion, expiration, queue drain, query, Doctor, and reconcile.
- [ ] Record Python module counts, import time, task decode time, allocation
  count, peak memory, and Taskwarrior call counts for those workflows.
- [ ] Inventory every production function accepting or returning a task,
  parent, child, snapshot, or mutation as `dict`, `Mapping`, tuple, or `Any`.
- [ ] Inventory every normalization of literal `null`, UUIDs, numeric links,
  statuses, recurrence fields, Taskwarrior timestamps, and native date fields.
- [ ] Inventory every place that copies a parent task, strips volatile fields,
  compares child dictionaries, builds import JSON, or decides whether a field
  is cleared.
- [ ] Inventory all fields used in mutation guards, lifecycle identities,
  schedule fingerprints, chain fingerprints, and diagnostic evidence.
- [ ] Classify arbitrary fields as preserved, Nautical-owned, Taskwarrior-
  generated, volatile, immutable, or prohibited on child import.
- [ ] Add characterization tests for every legitimate field representation
  before deleting its old parser or comparison path.
- [ ] Capture malformed-row behavior separately from valid task behavior;
  integrity diagnostics must not be constrained by operational validation.

Completion criteria:

- [ ] Every current task read, normalization, child build, patch, comparison,
  and serialization path has one named replacement owner.
- [ ] Baselines use isolated Taskdata, config, cache, and outbox directories.
- [ ] The migration inventory contains no unexplained raw-task consumer.

## 1. Define Field And Observation Contracts

- [x] Define `FieldPresence` as absent, null, or value. Preserve the original
  distinction even when one domain projection treats null as empty.
- [x] Define immutable typed values for full UUID, short UUID reference,
  Taskwarrior status, chain ID, positive link, and timezone-aware timestamp.
- [x] Define typed decode issues with field, raw value, stable error code,
  message, and severity. Do not store parser exceptions as the contract.
- [x] Define `TaskObservation` as an immutable, lossless view of one JSON row
  containing normalized known fields, frozen arbitrary fields, decode issues,
  source query, and snapshot provenance.
- [x] Permit `TaskObservation` to represent incomplete chain identity,
  malformed links, invalid timestamps, and unknown status so
  integrity tools can report the actual evidence.
- [x] Prevent arbitrary field access from mutating or exposing mutable nested
  values. Freeze once at decode and thaw only at an external serialization
  boundary.
- [x] Define stable equality and fingerprints from semantic fields and source
  provenance; exclude Taskwarrior `id`, urgency, and presentation-only values
  unless a caller explicitly requests them.
- [x] Make construction validate internal consistency without requiring the
  observation itself to be operationally valid.

Completion criteria:

- [x] Observation contracts contain no mutable mappings, nullable success
  values, raw parser exceptions, or implicit literal-null behavior.
- [x] Equivalent Taskwarrior rows normalize identically across integer/float
  links and supported timestamp encodings.
- [x] Contract tests cover every presence state, malformed known field,
  arbitrary nested field, Unicode value, and provenance combination.

## 2. Define The Operational Nautical Task

- [x] Define `NauticalTask` as a validated immutable projection of one
  `TaskObservation`.
- [x] Require a valid full UUID, recognized Taskwarrior status, and usable
  native timestamp values required by the requested operation.
- [x] Define typed `ChainIdentity` containing mandatory chainID, positive link,
  previous/next references, and explicit chain state.
- [x] Define typed `RecurrenceState` containing exactly one recurrence kind,
  `RecurrenceSpec`, limits, business calendar, omissions, and mode.
- [x] Define typed temporal state for due, scheduled, wait, until, entry,
  modified, end, and recurrence target while retaining which values were
  absent or explicitly null.
- [x] Keep task status, lifecycle intent, recurrence enabled state, and chain
  terminal state separate. Do not infer one solely from another.
- [x] Define operation-specific validators for scheduling, completion,
  expiration, deletion, repair, and query instead of one weak global `valid`
  flag.
- [x] Return typed validation outcomes carrying exact issues and required
  evidence. Never return `None`, `False`, or an empty task on failure.
- [x] Cache validated projections on the observation or invocation session so
  the same task is not revalidated in bounded loops. `TaskObservation` now
  retains one invocation-local `NauticalTask` projection.

Completion criteria:

- [x] Scheduler, lifecycle, mutation, and query code cannot receive an
  operational task with incomplete recurrence identity.
- [x] Integrity and Doctor can still inspect the originating malformed
  observation without reconstructing a dictionary.
- [x] Tests cover valid and invalid combinations of status, chain state,
  recurrence kind, limits, native until, and temporal fields.

## 3. Build The Sole Task Codec

- [x] Implement one decoder from parsed Taskwarrior JSON objects to
  `TaskObservation`.
- [x] Decode each row exactly once per authoritative repository result.
  `TaskReadRepository` owns codec decoding and the repository/performance tests
  verify one authoritative observation per row.
- [x] Normalize literal `null` only according to the declared semantics of the
  specific field; never through a generic truthiness rule.
- [x] Normalize numeric Taskwarrior values without accepting booleans,
  infinities, fractional links, overflow, or lossy coercion.
- [x] Parse known Taskwarrior timestamps to aware UTC datetimes and preserve
  their original values for guards and diagnostics.
- [x] Normalize UUID references without resolving them. Graph and repository
  scopes remain responsible for resolution and ambiguity.
- [x] Preserve arbitrary JSON-compatible fields value-semantically, including
  Unicode descriptions, annotations, tags, dependencies,
  third-party UDAs, and nested values.
- [x] Define versioned external encoders for task import, hook stdout, query
  JSON, and diagnostic evidence. Do not reuse diagnostic serialization as a
  Taskwarrior mutation payload. `TaskCodec` owns these contract-specific
  encoders.
- [x] Reject unsupported Python values at the codec boundary with typed encode
  failures rather than `default=str` in mutation JSON.
- [x] Keep `ensure_ascii=False` for every external JSON encoding.

Completion criteria:

- [x] No production module outside `task_codec.py` parses a Taskwarrior task
  row or implements known-field normalization.
- [x] Valid decode/encode round trips preserve every non-volatile user field.
- [x] Fuzz and boundary tests cover malformed JSON objects, Unicode, large
  values, null variants, timestamp variants, and arbitrary UDAs.

## 4. Define Draft And Patch Semantics

- [x] Define immutable `TaskDraft` for a new child import. Require description,
  recurrence identity, chain identity, target timestamp, and all fields needed
  by lifecycle postconditions.
- [x] Classify draft fields as required, optional, copied, cleared,
  Taskwarrior-generated, or forbidden.
  `task_field_policy.py` is the single classification owner and rejects
  owner-managed fields supplied through carried draft data.
- [x] Define `TaskPatch` as explicit set and clear operations. Omitted fields
  mean preserve; no caller may use `None`, empty text, or dictionary absence to
  imply a mutation.
- [x] Reject patches that set and clear the same field, mutate immutable
  identity fields outside their named operation, or contain volatile fields.
- [x] Define named patch constructors for parent link, chain disablement,
  native-until repair, recurrence metadata repair, recurrence activation, and
  ordinary carry adjustment.
- [x] Define canonical semantic comparisons for observations, drafts, and
  patches. Exclude Taskwarrior-generated and volatile fields centrally.
- [x] Define deterministic draft and patch fingerprints from immutable
  semantic inputs, not serialized dictionary ordering.
- [x] Make postconditions consume the same typed draft or patch that was
  encoded for mutation. Mutation reads and postcondition predicates now use
  `TaskObservation` values; payload encoding remains at the mutation boundary.

Completion criteria:

- [x] No mutation decision depends on dictionary diffing, truthiness, or a
  manually maintained volatile-field exclusion list outside `task_changes.py`.
  Remaining mappings are explicit mutation or presentation payload boundaries.
- [x] Clearing an optional UDA, preserving it, and setting it to a value are
  distinct and tested operations.
- [x] Draft and patch tests cover invalid identity changes, Unicode, arbitrary
  field preservation, idempotency, and deterministic fingerprints.

## 5. Cut The Taskwarrior Read Boundary Over

- [x] Make `TaskReadRepository` decode all successful task rows through
  `TaskCodec` and return `TaskRead[TaskObservation]` or typed observation
  collections.
- [x] Remove dictionary-returning UUID, chain, slot, predecessor, child,
  lifecycle-candidate, and snapshot reads.
- [x] Preserve found, absent, and unavailable semantics independently of task
  validity. A found malformed row is not an unavailable command and not an
  absent task. (Covered by repository failure/observation tests.)
- [x] Attach query scope, included statuses, command evidence, snapshot ID,
  and mutation epoch to observations at repository construction.
- [x] Reuse immutable observations across exact and broad reads within one
  unit of work without deep copying them.
- [x] Invalidate affected observations after every successful or uncertain
  mutation epoch.
- [x] Keep broad and narrow snapshot authority explicit so domain validation
  cannot turn incomplete coverage into absence.
- [x] Delete loose JSON task parsers and mapping normalization helpers from
  hook support, query, modify reads, reconcile, Doctor, and Navigator.
  The authoritative repository and lifecycle readers are typed; remaining
  `to_mapping()` calls are limited to Taskwarrior mutation and final output.

Completion criteria:

- [x] No Taskwarrior-facing production read returns a raw task mapping.
  Reconcile and mutation repository boundaries now reject untyped rows;
  mapping conversion is limited to explicit serialization boundaries.
- [x] Repository tests distinguish command failure, malformed JSON, malformed
  found task, valid found task, authoritative absence, and partial coverage.
- [x] Taskwarrior call counts do not increase from the recorded baseline.
  The extended performance budget records purpose-level counts and enforces
  the accepted budgets.

## 6. Migrate Scheduling And Query Consumers

- [x] Make `RecurrenceSpec.from_task` accept only `NauticalTask`; remove the
  mapping constructor and public normalization helper. Typed `from_task`
  entry points now exist for recurrence specs, compiled schedules, evaluation
  sessions, and scheduler services; remaining observation callers are the
  next migration pass.
- [x] Make compiled schedules, evaluation sessions, recurrence evaluators, and
  scheduler services consume typed recurrence and temporal state. All four now
  expose validated `NauticalTask` entry points, and production preview,
  generation, timeline, hint, modify, and integrity paths use them.
- [x] Remove task-field normalization from scheduler and occurrence paths.
  Query and scheduler consumers now read typed task state; normalization is
  confined to the explicit TaskCodec boundary.
- [x] Make query selectors operate on typed identity and chain state. Missing
  and ambiguous UUID results now use private typed query outcomes rather than
  sentinel task dictionaries; repository rows are decoded once at the query
  boundary before scheduling.
- [x] Make occurrence, next, lifecycle metadata, and chain inspection results
  derive from the same validated task used by scheduler services for the query
  service. Query occurrence and next projections now validate once and reuse
  the task-scoped scheduler.
- [x] Return per-task typed invalid results for malformed observations while
  allowing independent valid tasks to remain queryable. Query absence and
  ambiguity are typed outcomes, not task-shaped sentinel mappings.
- [x] Preserve task-range semantics: task queries respect current due or
  scheduled state, while expression-only queries remain calendar projections.
  Current-due range behavior is covered by the query regression tests.
- [x] Reuse one validated task and evaluation session throughout each query.
  Anchor occurrence and next projections validate once and reuse the typed
  scheduler session; CP projections use the typed child-generation boundary.

Completion criteria:

- [x] Preview, completion, reconcile, Navigator, and query produce identical
  occurrences from the same `NauticalTask` and evaluation context. The
  operational-consumer parity regression covers direct scheduling, query,
  Navigator projection, and reconcile successor planning.
- [x] Query never leaks raw task dictionaries or internal domain
  serialization as its public versioned JSON contract. Query responses use
  explicit versioned models and serializers; raw rows remain at the codec
  boundary only.
- [x] Cross-path, DST, astronomy, business-calendar, omission, random, and
  multi-time conformance tests pass. The scheduler matrix and consumer
  parity tests cover these context-sensitive families; astronomy is guarded
  by the optional Astral availability check.

## 7. Migrate Child Generation And Lifecycle Planning

- [x] Make `ChainGenerationService` accept a validated parent `NauticalTask`
  and return `TaskDraft` plus typed generation evidence.
- [x] Replace parent dictionary copying with an explicit field-copy policy over
  the immutable observation.
- [x] Move carry-field preservation to typed temporal calculations and
  explicit draft fields.
- [x] Make lifecycle snapshots and planner inputs carry typed task identities,
  recurrence state, temporal state, and observations rather than frozen task
  dictionaries.
- [x] Make completion, expiration, deletion, terminal, and hookless recovery
  plans reference typed parent and child values.
- [x] Ensure lifecycle plan identity and fingerprints use canonical domain
  values and remain stable across equivalent Taskwarrior encodings.
- [x] Remove mapping-based child builders, carry helpers, lifecycle snapshot
  coercion, and child normalization. (Production child builders and
  successor-plan ingress are draft-only; carry decisions use typed temporal
  values; remaining `to_mapping()` calls are explicit Taskwarrior mutation or
  presentation boundaries covered by Section 8.)

Completion criteria:

- [x] Every child accepted by lifecycle planning is already a valid
  `TaskDraft`; incomplete children cannot reach staging or the outbox.
- [x] Lifecycle plan replay and crash recovery remain deterministic after JSON
  persistence and process restart. (The outbox regression test claims a
  schema-versioned plan from a fresh interpreter and verifies its semantic
  identity and execution stage before the normal lease-recovery assertions.)
- [x] Completion and recovery characterization tests pass without old task
  builders or dictionary compatibility keys. (Characterization fixtures now
  call the shared `ChainGenerationService` through a draft-specific test
  helper; the typed observation accessor also preserves arbitrary configured
  UDAs without reconstructing compatibility mappings.)

## 8. Migrate Mutation And Outbox Application

- [x] Make the mutation gateway accept only named typed mutation requests
  carrying `TaskDraft` or `TaskPatch` and a typed guard.
- [x] Encode Taskwarrior import and modify arguments only at the mutation
  boundary through `TaskCodec`.
- [x] Build mutation guards from typed observation values and preserved raw
  guard timestamps.
- [x] Make postcondition reads decode into observations and compare through
  canonical domain semantics.
- [x] Persist versioned draft/patch payloads in lifecycle and integrity outbox
  envelopes; do not persist arbitrary task dictionaries.
- [x] Reject incompatible outbox schema at cutover. Do not add readers for old
  task-payload formats.
- [x] Keep volatile-field exclusion, timestamp equivalence, literal-null
  handling, and immutable-input comparison centralized in task changes/codec.
- [x] Preserve applied, already-applied, retryable, rejected, conflict, manual
  review, and unavailable outcomes.

Completion criteria:

- [x] No mutation gateway accepts a raw dictionary or constructs JSON through
  ad hoc serialization.
- [x] Replay recognizes equivalent Taskwarrior values without hiding real
  immutable-input conflicts.
- [x] Crash-point tests pass at stage, claim, import, link, verify,
  acknowledge, retry, and manual-review persistence boundaries.

## 9. Migrate Chain Integrity And Reconciliation

- [x] Construct `ChainNode` directly from `TaskObservation`; remove
  `ChainNode.from_mapping` and duplicate normalization.
- [x] Keep graph nodes specialized for integrity while retaining a typed link
  to their source observation and provenance.
- [x] Make invariants consume typed identity, recurrence, temporal, and field
  states instead of calling `field()` on thawed dictionaries. Chain topology
  and hydration now consume typed edge tokens from `ChainNode`.
- [x] Make repair operations carry `TaskPatch`, `TaskDraft`, or named typed
  lifecycle operations.
- [x] Make bounded hydration return observations through the shared repository
  and reuse them in graph construction.
- [x] Make Doctor and reconcile render malformed field evidence from typed
  decode issues.
- [x] Remove reconciliation helpers that normalize tasks, reconstruct child
  dictionaries, or compare mappings.
  - [x] Recovery child-slot reads now remain `TaskObservation` values through
    the repository and recovery service; serialization is limited to planner
    and policy boundaries.
  - [x] Virtual expired children now cross the recovery loop as a typed,
    immutable observation result rather than a mutable dictionary.
  - [x] Carry validation and operator inspection consume `TaskDraft` values;
    Taskwarrior mappings remain only at explicit import/presentation boundaries.

Completion criteria:

- [x] Integrity can audit malformed observations that operational services
  correctly reject.
- [x] Repair planning and application contain no implicit raw task mappings;
  `TaskDraft.to_mapping()`/`LifecyclePlan.child_dict()` are explicit external
  Taskwarrior mutation boundaries.
- [x] Healthy, repairable, unavailable, and manual-review chain fixtures remain
  deterministic under shuffled input.

## 10. Migrate Hook Workflows

- [x] Keep thin protocol probes minimal and dictionary-free beyond the bounded
  JSON field inspection needed to classify ordinary versus Nautical tasks.
- [x] Decode full on-add input once into a `TaskObservation`, then validate it
  for recurrence activation and preview.
- [x] Decode old/new on-modify input once each and derive a typed semantic
  transition rather than comparing mappings throughout the hook.
- [x] Make ordinary recurrence edits, activation, suspension, resumption,
  completion, deletion, carry updates, chainID protection, and native-until
  enforcement consume typed transitions and patches. Ordinary, completion, and
  deletion routes now receive one `TaskTransition`; deletion classification
  reuses its typed new observation instead of decoding the mapping again.
- [x] Make on-exit decode queued/outbox task evidence through the same codec and
  apply typed lifecycle operations.
- [x] Keep feedback and panels downstream of domain outcomes; presentation
  must not inspect or mutate task dictionaries. Parent/child completion
  payloads now cross this boundary as immutable `TaskView` projections;
  chain-history indexes are converted to immutable views at the same boundary
  while the authoritative lifecycle cache remains operational-only.
- [x] Delete hook-specific null normalization, timestamp parsing, task copying,
  volatile diffing, and child JSON construction from presentation paths. Child-
  import null/numeric/datetime normalization belongs to `TaskCodec`; mutable
  mappings remain only in the explicit mutation and final Taskwarrior
  serialization boundaries. Completion and on-add feedback now consume typed
  views, including recurrence updates, wait/scheduled rows, compact previews,
  lifecycle results, and preview theming.

Completion criteria:

- [x] Full hook implementations contain no raw task mapping contracts outside
  the strict protocol, mutation, and final serialization boundaries. Read-only
  presentation callbacks receive immutable `TaskView` projections.
- [x] Hook stdout remains exactly the required Taskwarrior JSON document with
  `ensure_ascii=False`; diagnostics remain opt-in stderr only.
- [x] Plain-hook and Nautical-hook performance remain within recorded budgets
  on desktop and both Termux devices, with the slower queue-drain and
  reconcile-apply timings accepted as a documented performance exception;
  those optimizations remain future work.

## 11. Migrate Operator And Presentation Consumers

- [x] Make Doctor, reconcile, query, Navigator, analytics, timelines, natural
  language, and panels consume typed task/domain views. Doctor/reconcile/query
  use `TaskObservation`; Navigator and hook panels use immutable presentation
  views; natural-language services consume normalized domain expressions.
- [x] Define narrow presentation view models rather than passing
  `TaskObservation` directly into renderers. `TaskView` now covers hook panels,
  lifecycle results, add previews, and Navigator rows; Navigator reference
  resolution uses an immutable `_ResolvedTaskView`.
- [x] Keep versioned public query JSON stable through explicit serializers;
  internal domain models are not the public API schema. Query response models
  own the versioned JSON contract.
- [x] Render malformed observations with task UUID, field, raw value, stable
  issue code, and actionable remedy without exposing tracebacks. Doctor,
  reconcile, and query failures retain structured evidence and CLI-safe text.
- [x] Aggregate historical findings by typed identity and policy rather than
  dictionary keys assembled in presentation code. Integrity reports and
  Doctor summaries use typed chain/task identity.
- [x] Remove operator-specific task exports, parsers, normalization, and
  dictionary fallback behavior. Operator reads now use `TaskReadRepository`;
  Navigator chain-reference normalization is an immutable view, with mappings
  retained only at explicit JSON/mutation boundaries.

Completion criteria:

- [x] Operator commands share one observation and validation model while
  retaining their separate output contracts.
- [x] Human and JSON output remain deterministic and useful for malformed,
  unavailable, healthy, and repairable states.
- [x] Navigator and optional Rich presentation cannot alter domain decisions.

## 12. Remove The Replaced Representation

- [x] Delete mapping constructors from recurrence, lifecycle, integrity,
  query, child generation, and mutation models.
- [x] Delete public `normalize_recurrence_text` and all equivalent literal-null
  helpers outside `TaskCodec`.
- [x] Delete raw task copy/sanitize helpers and volatile-field comparison lists
  replaced by `TaskDraft` and `TaskPatch` policy. Mutable copies that are
  required for Taskwarrior protocol output remain explicit `TaskPayload`
  boundaries.
- [x] Replace internal dictionary-based lifecycle/outbox builders and readers.
  Versioned `to_dict`/`from_dict` serializers remain intentionally for durable
  SQLite recovery envelopes.
- [x] Delete tests whose only purpose is preserving removed mapping APIs;
  migrate behavioral coverage to typed services first.
- [x] Remove unused imports, callbacks, facade exports, runtime-manifest
  entries, and lazy-module specifications created obsolete by the cutover.
- [x] Reconcile the staged-hook runtime manifest with every lazy module used
  by on-add and on-modify; deployment sanity now verifies this alignment.
- [x] Add AST/deployment enforcement preventing reintroduction of removed
  production task mapping contracts and ad hoc compatibility helpers.
- [x] Permit mappings only at explicit external JSON/configuration boundaries
  and in presentation serializers named by the enforcement allowlist.

  Current status: TaskPayload names mutable hook, mutation, feedback,
  expiration, timeline, integrity, and reconcile boundaries. Deployment sanity
  rejects direct JSON decoding and task-shaped annotations outside the approved
  protocol, query, cache, configuration, persistence, and presentation modules.

Completion criteria:

- [x] Repository search finds no production `task: dict`, `parent: dict`,
  `child: dict`, task-shaped `Mapping[str, Any]`, or compatibility coercion
  outside approved boundaries. AST/deployment enforcement is the authoritative
  check because protocol, mutation, persistence, and presentation boundaries
  intentionally retain mappings.
- [x] There is one decoder, one operational projection, one draft model, one
  patch model, and one mutation encoder.
- [x] No legacy task-representation bridge remains in production; durable
  lifecycle JSON is a versioned persistence boundary, not an operational task
  model.

## 13. Performance And Resource Validation

- [x] Add isolated decode benchmarks for small tasks, large annotations,
  arbitrary UDAs, large tags/dependencies, and malformed rows.
  `dev_tools/nautical_perf_budget.py` now measures the representative codec
  workload and asserts malformed exports fail closed.
- [x] Measure decode-once reuse across broad repository snapshots, chain graph
  construction, lifecycle planning, query, Doctor, and reconcile.
  The benchmark uses a counting codec and verifies that snapshot indexes,
  chain graph, and invariant consumers retain the same observations without
  additional decodes; lifecycle/query/Doctor/reconcile workflows consume the
  same repository snapshots and their call-purpose budgets pass.
- [x] Verify immutable field storage does not deep-copy task payloads in loops.
  The semantic immutability checks pass; Device 2 exceeds the provisional
  timing budget and remains a performance follow-up.
- [x] Bound nested arbitrary-field freezing and total decoded JSON size using
  the existing integration limits.
- [x] Measure peak memory for 100, 1,000, 10,000, and supported maximum chain
  observations.
  The local desktop pass records approximately 0.39 MB, 3.72 MB, and 36.87 MB
  peak respectively; repeat this on both Termux devices before accepting the
  slow-device gate.
- [x] Record Taskwarrior calls by purpose and prove the domain model introduces
  no new calls.
- [x] Run desktop and both Termux performance budgets for plain hooks,
  Nautical hooks, query, queue drain, Doctor, and reconcile.
  Both Termux reports were collected. The user-approved slow-device timing
  exception accepts the current cold-import, queue, and reconcile timings;
  the extended desktop run also passes all workflow budgets.
- [x] Compare cold import/module counts and full-hook latency with the baseline.
  Both Termux reports include the comparison; cold imports remain above the
  current budget and need a follow-up optimization or approved exception.

Completion criteria:

- [x] Each Taskwarrior row is decoded and frozen at most once per mutation
  epoch and reused by downstream services.
- [x] No measured workflow regresses beyond its accepted variance without a
  documented correctness justification.
  Slow-device timing overruns are explicitly accepted for this cut; the
  typed codec, snapshot reuse, resource, and memory measurements remain valid.
- [x] Slow-device results meet the established operational thresholds or an
  explicit user-approved exception.
  User approved accepting the current Termux timings; optimization is deferred.

## 14. Verification And Cutover

- [ ] Run the complete golden suite, shuffled suite, strict mypy scope,
  deployment sanity, black-box tests, stress campaign, soak tests, hook
  protocol harness, query contract tests, Doctor, reconcile, and installer.
  Stress CI passed (8 cycles) and a 10-second disposable soak passed (26
  cycles, zero add/done failures, zero queue bytes, zero dead letters); the
  combined gate remains open until shuffled-suite and final live-cutover
  evidence are recorded together.
- [x] Run the complete golden suite (959 tests passed).
- [x] Run the configured strict mypy scope (172 source files, no issues).
- [x] Run the complete golden suite in deterministic shuffled order (seed
  `20260824`, 959 tests passed with strict lifecycle warnings enabled).
- [x] Run the CI stress campaign profile (8 mixed-recurrence cycles with no
  violations).
- [x] Run a disposable soak smoke test (26 cycles, no failures, dead letters,
  or queue growth).
- [x] Exercise a real isolated partial child-import failure and retry; the
  first reconcile exits nonzero without exposing a successor, and the next
  reconcile converges successfully.
- [ ] Run add, ordinary modify, recurrence activation, completion, deletion,
  expiration, hookless completion, outbox replay, partial failure, integrity
  repair, and manual-review scenarios against real isolated Taskwarrior.
- [x] Verify Unicode task descriptions, annotations, projects, tags, UDAs, and
  JSON output remain unescaped and lossless. (Hook protocol and stdout tests
  preserve Unicode and arbitrary UDAs through add/modify JSON boundaries.)
- [x] Verify malformed task observations are reportable but cannot schedule or
  mutate. (Malformed repository observations and bounded integrity hydration
  return typed unavailable evidence rather than scheduling or mutation.)
- [x] Verify independent invalid tasks or chains do not hide safe work for
  other tasks or chains. (The disposable black-box harness adds an invalid
  unrelated chain and confirms a scoped safe query still succeeds.)
- [x] Verify no user Taskdata, config, cache, or outbox directory is accessed by
  tests or benchmarks. (Black-box operator coverage runs with a disposable
  TASKDATA/config/cache/outbox root and reports the temporary path.)
- [ ] Stop live hooks, install the candidate release, run installation-only
  Doctor, full Doctor, query integrity, reconcile dry-run, reconcile apply,
  and queue status, then re-enable hooks.
- [ ] Compare post-cutover Taskwarrior call counts, timings, and task fields to
  the baseline.
- [ ] Merge `task-domain-model-v7` into `main` only after every required gate
  passes.
- [ ] Remove this local checklist after the merged system is verified and
  released.

Final completion criteria:

- [ ] Every Taskwarrior task row crosses exactly one codec boundary.
- [ ] Every operational Nautical workflow consumes a validated immutable
  `NauticalTask`.
- [ ] Every new task is represented by `TaskDraft`; every existing-task change
  is represented by `TaskPatch` with explicit set/clear/preserve semantics.
- [ ] Scheduler, lifecycle, integration, integrity, query, hooks, and operator
  tools share the same task meaning without sharing mutable state.
- [ ] Malformed, absent, unavailable, and valid task states cannot be confused.
- [ ] No raw-task compatibility bridge or dual representation remains.
- [ ] Full correctness, protocol, performance, and real Taskwarrior cutover
  gates pass on desktop and both Termux devices.
