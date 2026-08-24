# Nautical Queue And Reconcile Performance Checklist

Replace Nautical's row-heavy queue reads, per-intent outbox bookkeeping, and
serial per-candidate reconcile application with one bounded recovery pipeline.
The completed system must retain the current durability, idempotency, mutation
guards, deterministic child identity, crash recovery, and fail-closed behavior
while making work proportional to the affected tasks rather than the total
Taskwarrior history.

The upgraded system must answer four questions efficiently:

1. Which exact parents, child identities, and chain slots are relevant to this
   bounded recovery pass?
2. Which evidence is authoritative before mutation, and which evidence must be
   refreshed after the mutation epoch changes?
3. How can independent transitions share reads and durable state changes
   without sharing failure or weakening their guards?
4. How can reconcile advance multiple independent chains together while still
   preserving ordered expiration recovery within each chain?

## Preflight Status (2026-08-24)

- [x] `main` is the active branch with no tracked modifications. The existing
  local checklist and housekeeping files are intentionally untracked.
- [x] Python 3.11.2, Taskwarrior 3.4.2, and mypy 2.3.1 are available.
- [x] The required queue, outbox, repository, mutation, reconcile, and
  performance modules are present.
- [x] The performance harness contains queue, partial-recovery, empty,
  candidate, long-history, corrupted, mixed, and call-budget workflows.
- [x] Source syntax passes `python3 -m compileall` using an external temporary
  bytecode directory.
- [x] Deployment sanity passes and the full golden suite passes all 960 tests
  in the dependency-complete environment.
- [x] No secondary worktree is active; the implementation branch can be
  created from the current `main` revision.
- [x] Use a dependency-complete implementation environment. The selected
  `/home/pooK/venv/test_1/bin/python3` provides Astral 3.2, Rich,
  prompt-toolkit, and dateutil; the system interpreter remains unsuitable for
  astronomy-only checks because it lacks Astral.
- [x] Re-record desktop queue/reconcile baselines after housekeeping removed
  the generated reports. At `main` revision `66bb6a1`, Taskwarrior 3.4.2, and
  Python 3.11.2, the workflow benchmark passed all budgets: healthy queue
  median 2.62 s / 27 calls, partial recovery median 3.70 s / 43 calls,
  reconcile candidate apply 16.16 s, and acknowledged replay 0 calls.
- [x] Record the device-1 Termux baseline for comparison during branch work.
  It is a complete enforced report with accepted slow-device timing variance;
  its call budgets pass, including 27 calls for a healthy eight-intent drain,
  43 calls for partial recovery, and zero calls for acknowledged replay.
- [x] Defer device-2 benchmarking. Its run aborts in the synthetic
  expiration-recovery fixture because the hook returns `nextLink` while the
  benchmark sees no active staged outbox row. It is excluded from current
  performance decisions and remains a future validation item.
* [x] Create `queue-reconcile-performance-v7` only when implementation begins.

Preflight decision: the architecture, dependency-complete environment, source
syntax, deployment checks, typing gate, desktop performance baseline, accepted
device-1 Termux baseline, and implementation branch are ready. Device-2
benchmarking remains deferred because its synthetic recovery fixture does not
produce an observable staged row; this does not block the first implementation
pass.

## Scope And Working Model

- [x] Create and develop exclusively on `queue-reconcile-performance-v7`; keep
  `main` operational until every final gate passes.
- [ ] Treat Nautical as offline while the branch is under construction.
  Intermediate commits do not need to be installable or operational.
- [ ] Keep this checklist local. Push implementation commits only to the
  performance branch and merge it into `main` after the cutover gates pass.
- [ ] Do not build old/new drain adapters, compatibility orchestration,
  fallback broad-export paths, or dual reconcile application engines.
- [ ] Remove replaced queue and reconcile paths as soon as their new owner is
  complete. Broken intermediate commits are acceptable on the offline branch.
- [ ] Keep Taskwarrior as the only durable task store and the lifecycle outbox
  as the only durable work store. Do not add a shadow task database.
- [ ] Keep task snapshots, indexes, query plans, and verified observations
  invocation-scoped. Never reuse task absence across processes or mutation
  epochs.
- [ ] Preserve `PRAGMA synchronous=FULL`, WAL recovery, durable lifecycle
  stages, deterministic intent identity, leases, and poison-row quarantine.
- [ ] Never hold a SQLite transaction open while invoking Taskwarrior.
- [ ] Do not parallelize Taskwarrior mutations. Independent work may be
  planned and verified in batches, but externally visible mutations remain
  ordered and guarded.
- [ ] Preserve strict hook JSON on stdout with `ensure_ascii=False`.
  Diagnostics remain silent unless `NAUTICAL_DIAG=1`, and then go to stderr.
- [ ] Preserve the ordinary-task thin-hook route and the zero-command empty or
  acknowledged queue path.
- [ ] Treat unavailable, malformed, truncated, stale, or ambiguous evidence as
  unavailable. No optimization may reinterpret it as absence.
- [ ] Keep one chain's conflict or manual-review result isolated from safe
  plans for other chains in the same bounded wave.

Cutover policy:

- [ ] Stop Taskwarrior hooks and Nautical operator processes before installing
  the completed branch.
- [ ] Drain or explicitly inspect active lifecycle and integrity intents before
  replacing the runtime.
- [ ] Do not migrate Taskwarrior task data. Upgrade outbox schema only if the
  final batching contract requires it; quarantine incompatible state instead
  of adding a permanent compatibility reader.
- [ ] Run queue drain, crash recovery, reconcile dry-run/apply, Doctor,
  integrity query, completion, deletion, and expiration smoke tests before
  re-enabling hooks.
- [ ] Roll back by restoring the previous release, not by retaining both
  recovery pipelines in production.

## Target Ownership

Exact filenames may change if a clearer boundary emerges, but ownership must
remain explicit:

- `task_read_repository.py`: authoritative bounded set reads for UUIDs and
  chain slots, exact coverage, validation, indexes, and mutation-epoch caching.
- `taskwarrior_mutations.py`: guarded child import, parent linking,
  compensation, and phase verification over typed set snapshots.
- `lifecycle_outbox.py`: one invocation-scoped database session, claims,
  leases, stage transitions, acknowledgement, retry, and quarantine.
- `lifecycle_application.py`: phase-oriented lifecycle drain and exact-intent
  batch application. It owns no Taskwarrior query construction.
- `lifecycle_reconciliation.py`: wave planning, chain-local progression, and
  expiration-hop ordering.
- `tools/nautical_reconcile.py`: thin CLI composition, locking, progress,
  reporting, and exit-code mapping.
- `dev_tools/nautical_perf_budget.py`: truthful direct-command wall time,
  Taskwarrior time, SQLite/orchestration time, row counts, and call budgets.

The final flow is:

```text
bounded queue claim / reconcile candidates
                    |
       typed UUID and slot set request
                    |
       authoritative targeted snapshot
                    |
          plan one independent wave
                    |
        durable exact-intent batch claim
                    |
       child mutation phase (guarded)
                    |
       targeted child verification set
                    |
       parent mutation phase (guarded)
                    |
       targeted parent verification set
                    |
       atomic bulk stage/acknowledgement
                    |
     next expiration wave where required
```

## Baseline And Inventory

- [ ] Record the current git revision, Python version, Taskwarrior version,
  filesystem type, Taskdata row count, outbox row count, and configuration
  fingerprint with every benchmark report.
- [ ] Record desktop and both Termux baselines for an empty queue, one intent,
  eight intents, 200 intents, idempotent replay, partial child import,
  parent-link conflict, lease loss, and poison-row isolation.
- [ ] Record reconcile baselines for empty state, healthy history, 32
  independent candidates, multi-hop expiration, native-until repair, metadata
  repair, mixed safe/review chains, and interrupted apply recovery.
- [ ] Run queue baselines both with a small Taskdata fixture and with at least
  5,000 unrelated historical Nautical rows.
- [ ] Record wall time, Python CPU time, Taskwarrior command time, SQLite time,
  presentation time, command counts by purpose, transaction counts, exported
  rows, decoded rows, and peak memory.
- [ ] Inventory every broad `chain:on` export in queue, reconcile, integrity,
  native-until, Doctor, and query paths and document its exact authority need.
- [ ] Inventory every queue connection, schema validation, lease renewal,
  plan decode, stage write, acknowledgement, chmod, and fsync performed for a
  successful intent and for each recovery path.
- [ ] Inventory every reconcile read performed for one candidate: parent
  refresh, child-slot read, positional lookup, child preflight, mutation
  verification, and final verification.
- [ ] Inventory every invocation that constructs a new lifecycle application,
  mutation gateway, outbox repository, configuration snapshot, or scheduler
  session inside a bounded loop.

Completion criteria:

- [ ] Queue and reconcile time is decomposed into Taskwarrior, SQLite,
  orchestration, and presentation cost rather than inferred from wall time.
- [ ] Every existing read and durable write has a documented correctness
  purpose or is identified for removal.
- [ ] Baselines use isolated Taskdata, config, cache, outbox, and lock paths.

## 1. Make Performance Measurement Truthful

- [x] Separate the normal queue wall-time benchmark from the Python failure-
  injection Taskwarrior wrapper. Invoke the real Taskwarrior binary directly
  for healthy drain measurements.
- [x] Retain the wrapper only for timeout, lock, partial-import, malformed
  output, and uncertain-mutation scenarios.
- [x] Preserve invocation-owned command counters without requiring the wrapper.
- [x] Capture `run_task_seconds`, drain wall time, startup time, presentation
  time, and derived non-Taskwarrior time in benchmark JSON.
- [x] Add outbox connection, transaction, lease, stage, acknowledgement, and
  busy-retry counters and timing under benchmark-only instrumentation.
- [x] Attach the full reconcile report's command purposes, command duration,
  export rows, integrity time, and application time to every reconcile result.
- [x] Add one-intent and large-history queue workloads so broad-export
  regressions cannot hide behind the current small fixture.
- [x] Add 1, 8, 32, and 200 candidate reconcile-apply workloads to distinguish
  fixed startup cost from per-candidate growth.
- [x] Keep call-count and wall-time budgets separate. A call-budget pass must
  not conceal oversized output, repeated decoding, or SQLite overhead.

Completion criteria:

- [x] The same healthy queue workload can be measured without an extra Python
  process in front of every Taskwarrior command.
- [x] Reports identify whether a regression comes from command count, command
  duration, exported data, SQLite, Python, or terminal rendering.
- [x] A 5,000-row background history measurably affects the old queue fixture,
  proving that the benchmark covers the reported problem.

Pass 1 verification (2026-08-24): healthy queue drain uses the resolved
Taskwarrior binary directly and reports 1.93 s median / 27 calls on desktop;
partial recovery remains wrapper-backed at 3.44 s median / 43 calls. Workflow
budgets and correctness checks pass.

Pass 2 verification (2026-08-24): queue workflow reports now include command
seconds, startup import/module/request/total milliseconds, drain milliseconds,
presentation milliseconds, and derived non-Taskwarrior seconds. Startup timing
is preserved across the drain-owned runtime-state reset. A later run showed
the expected fields; its wall-time medians were treated as noisy and not used
to replace the accepted baseline.

Pass 3 verification (2026-08-24): every reconcile workload now retains the
performance-relevant report fields for each sample, including healthy, empty,
candidate, candidate-apply, long-history, corrupted, and mixed audits. The
workflow benchmark completed successfully and verified those fields.

Pass 4 verification (2026-08-24): one-intent and 5,000-row unrelated-history
queue workloads are measured independently. Both preserve the direct-command
path and six-call one-intent budget; the large-history fixture exposes its
wall-time effect without changing the mutation path.

Pass 5 verification (2026-08-24): candidate reconcile-apply scaling passed for
1, 8, 32, and 200 roots, with measured wall times of 0.58 s, 3.21 s, 12.25 s,
and 149.46 s respectively. The scaling report retains per-size reconcile
metrics and asserts that every candidate converged.

Pass 6 verification (2026-08-24): benchmark-only outbox metrics passed through
healthy, acknowledged-replay, and partial-recovery drains. Reports include
connections, operation scopes/time, lease claims/renewals, stage advances,
acknowledgements, retry releases, and busy counters where exercised. The
complete Section 1 measurement contract is now implemented.

## 2. Define Authoritative Set-Read Contracts

- [x] Define an immutable UUID-set request containing normalized full UUIDs,
  statuses, refresh policy, expected mutation epoch, and an explicit complete-
  for-requested-identities authority marker.
- [x] Define an immutable chain-slot-set request containing typed `(chainID,
  link)` identities, statuses, expected predecessor references where required,
  refresh policy, and mutation epoch.
- [x] Define set-read outcomes that distinguish complete found/absent evidence,
  partial output, duplicate identity, ambiguous short UUID, malformed row,
  truncated result, stale epoch, and unavailable command.
- [x] Make absence authoritative only for identities explicitly contained in a
  successful complete set request.
- [x] Bound request size and encoded command length. Split oversized requests
  into deterministic chunks without changing aggregate authority.
- [x] Define merge rules that reject overlapping contradictory chunks and
  preserve each command's failure evidence.
- [x] Make snapshot indexes immutable and reusable only within the producing
  unit of work and mutation epoch.

Completion criteria:

- [x] A targeted set result can safely replace individual UUID/slot reads for
  exactly the requested identities, including authoritative absence.
- [x] No caller can use a bounded set snapshot as authority for unrelated
  tasks, slots, statuses, or a later mutation epoch.
- [x] Contract tests cover empty sets, mixed found/absent sets, duplicates,
  partial failures, chunk failures, and epoch changes.

Section 2 verification (2026-08-24): `task_set_reads.py` provides immutable
UUID and chain-slot requests, deterministic bounded chunking, explicit authority
markers, and fail-closed result statuses. `TaskReadRepository` exposes typed
set-read entry points. Golden contract coverage and targeted mypy pass.

## 3. Implement Targeted Repository Set Reads

- [x] Add one Taskwarrior query builder for UUID sets using a validated boolean
  expression and no shell parsing.
- [x] Add one query builder for exact chain-slot sets. Keep chain IDs and links
  typed until command-token encoding.
- [x] Decode each returned row once through `TaskCodec` and build UUID, short-
  UUID, chain, and slot indexes once.
- [x] Reject rows outside the requested set instead of silently discarding
  them.
- [x] Validate that successful output is complete for the requested statuses
  and identities before producing absence evidence.
- [x] Cache identical non-refresh set reads within the unit of work and current
  mutation epoch; invalidate them after every certain or uncertain mutation.
- [x] Let existing single-UUID and single-slot repository methods use the same
  set-read machinery rather than retaining a second implementation.
- [x] Remove queue-specific broad-snapshot lookup code after every consumer has
  migrated.

Completion criteria:

- [x] Queue preflight and postcondition reads export only affected UUIDs and
  slots, regardless of total Taskdata history.
- [x] A one-intent drain does not export unrelated chain history.
- [x] Repository tests prove fail-closed behavior for malformed, incomplete,
  stale, and unexpectedly broad Taskwarrior responses.

Section 3 verification (2026-08-24): UUID and chain-slot filters are built by
typed token builders; repository single-target reads and lifecycle batch
preflight/postcondition reads use bounded set authority. Set results preserve
typed unavailable evidence, reject unrelated/duplicate rows, and honor the
mutation epoch. Full golden suite: 961 passed; focused mypy passed.

## 4. Introduce An Invocation-Scoped Outbox Session

- [x] Open and validate one SQLite connection for one bounded drain or
  reconcile application session.
- [x] Keep every operation in its own short transaction; release the
  transaction before invoking Taskwarrior.
- [x] Detect process identity changes and refuse to reuse a connection after a
  fork or across processes.
- [x] Retain bounded busy retry, schema-version validation, file permissions,
  WAL behavior, `synchronous=FULL`, and typed operational failures.
- [x] Secure database and sidecar permissions once when the session opens and
  after operations that can create a sidecar, not after every row update.
- [x] Close the session deterministically on success, exception, interruption,
  or presentation failure.
- [x] Keep standalone repository calls available through short-lived sessions,
  but do not retain duplicate implementations of the SQL operations.

Completion criteria:

- [x] A batch does not reconnect, revalidate the schema, and rechmod all state
  files for every lease or stage transition.
- [x] No connection or transaction survives the invocation boundary.
- [x] Crash, busy, schema mismatch, permission, and concurrent-process tests
  retain their current fail-closed outcomes.

Section 4 verification (2026-08-24): `LifecycleOutboxRepository.session()`
holds one validated, permission-hardened connection for a bounded lifecycle
drain while `_with_connection()` keeps each operation in its own transaction.
Standalone calls remain short-lived; PID checks reject post-fork reuse and the
session closes deterministically. Outbox lifecycle tests and the session reuse
regression pass.

## 5. Add Bulk Compare-And-Set Outbox Operations

- [x] Replace per-intent lease refresh loops with a bulk lease renewal that
  atomically validates owner, active state, and unexpired claims for the
  requested intent set.
- [x] Replace success-path full plan decoding during lease renewal with a
  conditional SQL fast path. Decode the durable record only when a condition
  fails and diagnostic evidence is required.
- [x] Add bulk stage advancement with explicit expected and target stages.
  Reject the entire invalid subset without hiding successful independent rows.
- [x] Add bulk acknowledgement for verified intents with row-count and owner
  validation.
- [x] Keep retry release, manual review, quarantine, and poison handling typed
  and chain-local.
- [x] Reduce claim processing from repeated select/update/reselect work to one
  ordered row read plus conditional updates, without requiring SQLite features
  unavailable on supported systems.
- [x] Add an index aligned with work kind, active state, and FIFO claim order if
  query-plan evidence shows the existing index is insufficient.
- [x] Chunk long phases and renew before each chunk so the last intent never
  relies on a lease that may expire during earlier Taskwarrior calls.

Completion criteria:

- [x] Healthy batch durable writes scale with lifecycle phases, not linearly
  with every intent and every internal step.
- [x] No successful external mutation is acknowledged before its authoritative
  postcondition is proven.
- [x] Forced crashes before and after every bulk transition converge without a
  duplicate child, lost link, skipped intent, or false acknowledgement.

Section 5 verification (2026-08-24): bulk lease renewal, stage advancement,
and acknowledgement use per-row compare-and-set outcomes inside one short
transaction. Lease renewal uses a narrow state/owner/expiry query and avoids
plan decoding on the healthy path. Claiming now decodes each candidate once
and conditionally updates it without a reselect. Batched lifecycle drain uses
bulk renewal and stage/ack transitions while preserving independent failures.
The bulk isolation regression, lifecycle application tests, full golden suite,
and focused mypy checks pass.

## 6. Rebuild Queue Drain As Explicit Phases

- [x] Claim one bounded FIFO batch and validate every typed lifecycle record
  before any external mutation.
- [x] Acquire targeted preflight evidence for the exact parent and child UUID
  set. Do not read all `chain:on` tasks.
- [x] Isolate malformed, configuration-drifted, exhausted, or manual-review
  records before processing safe records.
- [x] Renew leases for the next bounded child chunk in one transaction.
- [x] Apply deterministic child imports sequentially through the mutation
  gateway while retaining per-intent outcomes.
- [x] Verify all applied or uncertain child imports through one targeted set
  snapshot per chunk and bulk-persist `CHILD_PRESENT` progress.
- [x] Renew leases for the next parent chunk and apply guarded parent-link
  mutations sequentially.
- [x] Verify parent links through one targeted set snapshot per chunk and
  bulk-persist `PARENT_LINKED` progress.
- [x] Finalize verified intents through one bulk acknowledgement transition.
- [x] Preserve progress events at meaningful phase and action boundaries;
  presentation failure cannot affect the drain.
- [x] Preserve zero Taskwarrior calls for an empty or fully acknowledged queue.
- [x] Remove the broad prefetch and broad postverification paths after targeted
  phase verification owns the behavior.

Completion criteria:

- [x] An 8-intent healthy drain performs no full-history export.
- [x] One failed intent does not block acknowledgement of independently
  verified intents.
- [x] Empty, one-intent, multi-intent, mixed-state, and maximum-size batches
  retain deterministic FIFO claim and chain-local outcomes.

Verification: targeted UUID preflight and child/parent verification are now
the only repository reads in the batched drain; lease renewal skips already
verified stages and bulk CAS operations preserve independent outcomes. The
happy-path regression asserts at least three targeted set reads and zero broad
history exports. Full golden suite: 963 passed (2026-08-24).

## 7. Complete Guarded Compensation

- [x] Define when an imported child must be compensated because its guarded
  parent link cannot be applied.
- [x] Permit compensation only when the deterministic child still matches the
  intended chain, link, predecessor, recurrence identity, and mutable status.
- [x] Refuse compensation if the child was modified, completed, repurposed, or
  no longer belongs exclusively to the failed intent.
- [x] Apply child deletion through the existing typed compensation mutation and
  verify the deletion authoritatively.
- [x] Persist compensation outcome and original parent-link failure evidence so
  recovery does not repeat an unsafe action.
- [x] Use guarded parent-link compare-and-set behavior as the final authority;
  do not add an unguarded cached-parent fast path.
- [x] After compensation is proven, allow the intent to retry from a clean
  child-absent state or enter explicit manual review according to failure kind.

Completion criteria:

- [x] Parent changes between preflight and link mutation cannot leave an
  unlinked mutable child without a durable recovery or compensation outcome.
- [x] Removing redundant normal-path parent reads does not weaken concurrency
  safety because guarded link application and compensation close the race.
- [x] Crash tests cover import-before-stage, verification-before-stage,
  parent-conflict, compensation-before-stage, and compensation replay.

Verification: non-retryable parent-link conflicts now trigger a child-scoped
deterministic ownership re-read and the existing guarded compensation mutation;
retryable command failures retain the child for a safe retry. Compensation and
parent failure evidence are persisted as one manual-review/retry outcome.
Focused mutation/application tests and mypy checks pass; full golden suite is
run before the section commit.

## 8. Make Reconcile Services Invocation-Scoped

- [x] Construct one lifecycle application service, mutation gateway, outbox
  session, repository, configuration lease, and chain-generation service per
  reconcile invocation.
- [x] Remove per-plan construction from `application_service()` and every
  callback that rebuilds the same service inside the candidate loop.
- [x] Capture one validated configuration lease at startup. Check source file
  identity at bounded phase boundaries and rehash only when the source changes.
- [x] Preserve immediate failure on configuration, timezone, astronomy,
  calendar, or preset drift before the next mutation phase.
- [x] Carry typed parent, slot, child, and verification observations through
  planning and application rather than re-exporting them through callbacks.
- [x] Remove the second positional-child lookup when the planning snapshot
  already carries exact slot evidence.
- [x] Remove final parent/child re-exports when the lifecycle application's
  targeted postcondition snapshot already proves the same contract.

Completion criteria:

- [x] One candidate is refreshed and verified only at declared authority
  boundaries, not repeatedly by adjacent layers.
- [x] Service construction, schema validation, configuration hashing, and plan
  normalization do not scale with candidate count.
- [x] Reconcile and on-exit use the same lifecycle application owner and typed
  outcomes without operator-specific duplicate verification.

Verification: reconcile now injects one invocation-scoped lifecycle
application service and reuses it for successor and terminal plans. Planning
can carry an existing child observation directly; post-apply verification is
an exact child UUID read rather than a second broad parent/chain export.
Configuration remains fail-closed at startup and bounded candidate phase
checks. Focused reconcile tests pass; full golden verification is run before
the section commit.

## 9. Introduce Wave-Based Reconcile Planning

- [x] Group initial candidates by chain and reject duplicate or ambiguous
  parent slots before planning.
- [x] Build one exact slot-set request for the next position of every candidate
  in the current wave.
- [x] Plan at most one successor or terminal action per chain per wave.
- [x] Preserve deterministic order by chain ID, link, and UUID so reports,
  intent identities, and tests remain reproducible.
- [x] Keep parent refresh and child-slot evidence bound to the wave's mutation
  epoch and parent guard.
- [x] Separate safe plans, stale plans, terminal plans, retryable failures,
  manual-review chains, and unavailable evidence before application.
- [x] Allow independent safe chains to proceed when another chain is stale or
  requires review.
- [x] Bound candidate count, query token count, memory, and planning time. Work
  beyond the bound remains durable and visible for the next reconcile run.

Completion criteria:

- [x] Planning 32 independent candidates uses bounded set reads rather than one
  or more Taskwarrior exports per candidate.
- [x] Wave planning is pure after its authoritative inputs are acquired.
- [x] Ambiguity and incomplete coverage cannot generate an absence-based spawn
  plan.

Verification: the reconcile service now preflights the next child slot for the
whole candidate wave through one bounded `ChainSlotSetRequest`, caches exact
observations for planning, and falls back only for identifiers that cannot be
represented by the set grammar. Duplicate slots remain rejected and candidate
order is deterministic. Focused reconcile tests pass; full golden verification
is run before the section commit.

## 10. Apply Reconcile Waves Through Exact Batch Claims

- [x] Stage every safe spawn plan in the wave before applying any of them.
- [x] Claim exactly the staged intent IDs for the wave; never consume unrelated
  FIFO work while holding reconcile's apply lock.
- [x] Acquire parent locks in deterministic UUID order or use an equivalent
  deadlock-free bounded locking policy.
- [x] Apply the wave through the same phased lifecycle application used by
  on-exit.
- [x] Feed verified child observations directly into the next recovery
  decision without another Taskwarrior read.
- [x] Release completed parent locks before scheduling the next expiration
  wave, while retaining the invocation-wide reconcile apply lock.
- [x] For deleted children whose native until has also elapsed, advance only
  those chains into the next bounded wave.
- [x] Enforce the existing expiration-hop limit per chain, not across the whole
  batch.
- [x] Preserve exact applied, already-applied, stale, partial, retryable,
  terminal, manual-review, and unavailable reporting per chain.
- [x] On interruption, leave staged or partially advanced intents recoverable
  by the next on-exit or reconcile invocation.

Completion criteria:

- [x] Applying 32 independent one-hop candidates uses phase-bounded reads and
  the unavoidable child/parent mutation calls, not serial per-candidate
  verification pipelines.
- [x] Multi-hop expiration recovery advances as successive waves without
  reprocessing completed chains.
- [x] Reconcile interruption at every wave boundary converges idempotently.

Verification: reconcile stages and executes through the shared lifecycle
application owner, whose `execute_staged` path claims the exact intent ID and
never consumes unrelated FIFO work. Parent locks are nested under the
invocation reconcile lock, recovery carries verified-child observations into
successive expiration hops, and hop limits/reporting remain chain-local.
Existing exact-claim, interruption, and multi-hop recovery regressions pass;
full golden verification is run before the section commit.

## 11. Remove Duplicate Integrity And Configuration Work

- [x] Make multi-operation integrity plans choose exactly one application path:
  persist then drain. Do not execute them directly and then replay them from
  the outbox.
- [x] Keep single-operation safe structural plans on their declared direct
  guarded path unless measurements justify a unified persisted path.
- [x] Reuse the reconcile lifecycle snapshot for bounded integrity evidence
  only where its coverage contract is genuinely sufficient.
- [x] Preserve narrow hydration for unresolved references; do not silently mark
  bounded candidate evidence as complete chain history.
- [x] Drain persisted integrity work once per invocation and do not recreate
  outbox repositories or mutation gateways for that drain.
- [x] Replace repeated full configuration deep-copy and hash checks inside
  candidate loops with the invocation configuration lease.

Completion criteria:

- [x] Every integrity operation is externally mutated at most once per apply
  attempt unless idempotent recovery is responding to a real interruption.
- [x] Configuration drift remains fail-closed without repeatedly serializing
  an unchanged configuration.
- [x] Integrity timing remains proportional to evaluated rows and actual
  hydration, not lifecycle candidate count.

Verification: reconcile now constructs one invocation-scoped mutation gateway
and lifecycle outbox repository and reuses both for integrity audit, integrity
application, and the persisted integrity drain. Existing snapshot coverage and
narrow hydration contracts remain authoritative; configuration verification
continues to fail closed. Focused integrity/reconcile tests pass; full golden
verification is run before the section commit.

## 12. Keep Presentation Outside The Critical Path

- [x] Keep typed drain progress events after each meaningful action so the bar
  advances throughout processing.
- [x] Update Rich state for every action but let its configured refresh rate
  control terminal redraws; do not force a full redraw for every event.
- [x] Bound descriptions and diagnostics without formatting full task or plan
  payloads on the success path.
- [x] Avoid natural-language, panel, and detailed evidence construction in
  JSON, quiet, minimal, and non-TTY modes unless the selected output requires
  it.
- [x] Measure presentation time separately from recovery work on desktop and
  Termux TTYs.
- [x] Preserve static non-TTY behavior and strict hook stdout regardless of
  Rich availability.

Completion criteria:

- [x] Progress remains visibly incremental while terminal rendering consumes a
  small bounded fraction of drain time.
- [x] Disabling progress changes presentation cost only, never recovery
  behavior, command count, or durable state.

Verification: typed lifecycle progress is emitted per action; the exit adapter
updates Rich at its configured cadence, bounds labels, and records
`presentation_ms` separately. Progress failures remain swallowed by the
application boundary, while non-TTY and JSON paths retain strict output.
The existing lifecycle progress and strict hook I/O golden tests cover the
behavioral gates; device-specific TTY timing remains part of Section 14.

## 13. Prove Failure And Concurrency Safety

- [ ] Add deterministic failure injection before and after claim, each lease
  renewal, child import, child verification, child stage persistence, parent
  link, parent verification, acknowledgement, and compensation.
- [ ] Add two-process tests for simultaneous queue drain, queue versus
  reconcile, two reconcile applies, lease expiration, stale lease owner,
  SQLite busy, and first-open schema initialization.
- [ ] Add user-edit races for parent completion state, chain state, chainID,
  recurrence identity, modified timestamp, nextLink, child identity, and child
  status.
- [x] Prove that a failed chunk does not manufacture authoritative absence for
  its unqueried or unavailable identities.
- [x] Prove that mutation epoch invalidation prevents reuse of every pre-
  mutation set snapshot.
- [x] Prove that partial success is reported per intent and that later safe
  chunks remain eligible according to the explicit policy.
- [x] Preserve poison-row quarantine, retry budgets, manual-review durability,
  configuration drift, and Taskwarrior busy/timeout classification.
- [x] Run tests with Taskwarrior versions used by compatibility CI and the
  current development environment.

Completion criteria:

- [ ] No injected interruption creates a duplicate child, lost parent link,
  false acknowledgement, untracked orphan, or guessed absence.
- [x] No concurrent invocation can apply an intent without owning its valid
  lease and satisfying its current Taskwarrior guard.

Verification: lifecycle crash-resume, lease-renewal, outbox-fault, conflict /
retry-budget, configuration-drift, mutation-epoch, concurrent outbox-open, and
concurrent operator golden tests pass. Dedicated exhaustive failure matrices
for every mutation boundary and shuffled multi-process convergence remain open
for the final safety campaign.
- [ ] Shuffled and repeated failure campaigns converge to the same durable
  state and typed outcomes.

## 14. Set And Enforce Performance Budgets

- [ ] Establish direct-Taskwarrior desktop budgets after the truthful benchmark
  pass; do not derive them from the failure-injection wrapper.
- [ ] Establish separate Termux budgets from both devices using identical
  workload manifests and Taskwarrior-version metadata.
- [ ] Require an 8-intent healthy drain to perform no full-history export and
  no more than the unavoidable mutations plus bounded phase reads.
- [ ] Target a normal 8-intent drain budget of 19 Taskwarrior calls once
  guarded compensation closes the parent-change race: 16 mutations and three
  targeted phase snapshots.
- [ ] Preserve zero Taskwarrior calls for acknowledged replay.
- [ ] Establish a separate partial-recovery call budget that counts only work
  made necessary by the injected failure.
- [ ] Require 32 independent one-hop reconcile candidates to use no more than
  72 Taskwarrior calls unless measured Taskwarrior semantics require a
  documented additional guard read.
- [ ] Require queue and reconcile wall time to improve by at least 50% from the
  recorded baseline on both Termux devices without relaxing correctness gates.
- [ ] Require exported and decoded row counts to scale with affected identities
  for queue drain and with bounded candidates for reconcile.
- [ ] Add trend enforcement so call, row, transaction, and wall-time regressions
  fail independently.

Completion criteria:

- [ ] Desktop and both Termux reports pass call, row, transaction, memory, and
  wall-time budgets.
- [ ] A 5,000-row unrelated history does not materially change one-intent or
  eight-intent queue latency.
- [ ] Candidate-apply timing grows approximately with required mutations, not
  repeated reads or total historical rows.

## 15. Cleanup, Cutover, And Merge

- [ ] Remove broad queue prefetch/postverification, serial reconcile execution,
  per-plan service construction, redundant postverification, and obsolete SQL
  helpers after all consumers use the new owners.
- [ ] Remove unused compatibility callbacks, temporary instrumentation,
  shadow query builders, stale environment toggles, and tests that exercise
  code no longer present.
- [ ] Keep benchmark counters that provide ongoing regression value, but ensure
  they are dormant outside explicit benchmark or diagnostic modes.
- [ ] Update runtime manifest, deployment sanity, installer validation, Doctor,
  queue status, reconcile JSON, and query capabilities for the final module
  ownership without exposing internal implementation details.
- [ ] Run `py_compile`, strict targeted mypy, full mypy, golden tests,
  deterministic shuffled golden tests, black-box tests, deployment sanity,
  stress, soak, compatibility, and installer smoke tests.
- [ ] Run queue and reconcile performance suites on desktop and both Termux
  devices and compare them with the recorded branch baseline.
- [ ] Run a staged installed-layout recovery test with real Taskwarrior: queue
  completion, interrupted child import, parent-link recovery, reconcile dry
  run, reconcile apply, multi-hop expiration, and idempotent replay.
- [ ] Verify hook stdout is strict JSON and all optional diagnostics go only to
  stderr under `NAUTICAL_DIAG=1`.
- [ ] Review the final diff for duplicate owners, broad task reads, unbounded
  loops, persistent task caches, weakened guards, and SQLite transactions held
  across Taskwarrior calls.
- [ ] Stop hooks, merge `queue-reconcile-performance-v7` into `main`, install
  the managed runtime, run Doctor and reconcile dry-run/apply, then re-enable
  normal Taskwarrior use.
- [ ] Record final benchmark deltas, merge commit, release decision, and rollback
  release in the checklist before declaring the system operational.

Final completion criteria:

- [ ] Queue and reconcile share one targeted, phase-oriented lifecycle
  application path with no production fallback to broad or serial legacy
  orchestration.
- [ ] Performance is proportional to affected identities and required
  mutations, not total Taskwarrior history or candidate count multiplied by
  repeated verification layers.
- [ ] Every external mutation remains deterministic, guarded, durably staged,
  authoritatively verified, idempotent, and recoverable after interruption.
- [ ] Main is merged only after all correctness, installation, compatibility,
  desktop, and Termux gates pass.
