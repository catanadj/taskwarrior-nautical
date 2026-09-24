# Performance, Resilience, and Reliability Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` when implementation is requested. Complete one
> numbered work package at a time, with its own regression and review gate.

**Goal:** Resolve the seven findings from the 2026-09-07 project audit through
small, independently verifiable changes.

**Architecture:** Retain the existing task codec, guarded mutation gateway,
SQLite lifecycle outbox, cache, and backup/restore services. Harden their
existing boundaries without changing recurrence semantics or adding another
execution framework.

**Tech stack:** Python standard library, `unittest`, SQLite, POSIX subprocesses,
and the project's existing golden tests and performance tools.

**Spec:** The seven audited failures/opportunities and acceptance criteria in
this document, together with the repository's `AGENTS.md`. This is a companion
to the older `PERFORMANCE_RELIABILITY_AUDIT_REMEDIATION_CHECKLIST.md`; it does not
replace that checklist or claim its unfinished work is complete.

## Global Constraints

- Keep Taskwarrior hook output strict JSON on stdout; send diagnostics to
  stderr only when `NAUTICAL_DIAG=1`.
- Be defensive with hook input parsing; avoid crashing hooks on malformed input.
- Preserve `ensure_ascii=False` for JSON output so Unicode is not escaped.
- Prefer small, targeted edits; avoid network access or heavyweight tooling.
- Preserve user changes, including the existing untracked
  `tests/test_hook_input_contract.py`. Do not overwrite or silently drop them.
- Use temporary Taskdata, databases, caches, restore destinations, and process
  fixtures. Do not run tests against live Taskwarrior data.
- Preserve WAL, `synchronous=FULL`, deterministic intent identity, mutation
  guards, postcondition verification, and serial Taskwarrior mutations.
- Never hold a SQLite transaction open while invoking Taskwarrior.
- Treat cache contents as disposable and lifecycle state as durable.
- Implement only the seven packages below. Deployment, installation, dependency
  upgrades, recurrence changes, and unrelated refactoring are outside scope.

## Audit Evidence And Execution Order

The audit's local baseline was 439 passing unit tests and 17 passing targeted
golden tests. Those results predate implementation; rerun the applicable checks
after changes. Full stress, soak, and platform matrices were not run.

| Package | Priority | Reproduced evidence | Dependency |
| --- | --- | --- | --- |
| 1. JSON container fidelity | High | Both hooks changed empty arrays into objects and returned success | None |
| 2. Restore inventory coverage | High | Empty manifest inventory validated; an unlisted hook was restored | None |
| 3. Lease timestamp correctness | Medium | Renewal succeeded with an expired lease; another owner immediately claimed it | None |
| 4. Process timeout cleanup | Medium | A 0.1 s timeout took 1.23 s with a descendant holding pipes | None |
| 5. Cache size limits | Medium | About 10 KB of encoded cache caused about 25 MB of allocation before rejection | None |
| 6. SQLite lock timeout | Medium | A configured 0.1 s lock timeout waited about 0.2 s | Package 3 first: shared outbox code |
| 7. Absent-field allocation | Low | Prototype reduced component runtime by about 5% to 38%, depending on workload | Package 1 first: shared task representation |

The numbered order is the recommended order. Each package is a separate
reviewable change; packages 3/6 and 1/7 must not be edited concurrently.

## 0. Establish The Implementation Baseline

- [x] Record the starting revision, branch, Python version, and worktree status
  in the implementation evidence. Preserve all existing user-owned changes.
- [x] Run unit discovery and the audit's golden slice using the commands below.
- [x] For each correctness package, add the regression first and confirm that
  it fails for the audited reason before changing production code.
- [x] Record each package's changed files, test result, and any remaining
  limitation before marking it complete. Do not mark an item complete based
  solely on a code change.

```bash
git status --short
git rev-parse HEAD
git branch --show-current
python3 --version
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -q
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py --only task_codec --only hook_protocol --only taskwarrior_client --only lifecycle_outbox --only cache_payload
```

## 1. Preserve JSON Arrays And Objects Through The Task Codec

**Files:** `nautical_core/task_models.py`; inspect consumers in
`nautical_core/task_codec.py` and `nautical_core/hook_protocol.py`.
Create `tests/test_task_codec_roundtrip.py`; extend the existing
`tests/test_hook_input_contract.py`.

**Interface:** Preserve `TaskCodec.decode_row(...) -> TaskObservation`,
`TaskObservation.to_mapping() -> dict`, and `TaskDraft.to_mapping() -> dict`.
Change only the internal immutable representation needed to preserve types.

- [x] Add a table-driven round-trip regression covering `[]`, `{}`, `[[]]`,
  `[{}]`, `[["key", "value"]]`, duplicate-key pairs, nested annotations, and
  Unicode. Include empty `tags`, `annotations`, and `depends` fields.

  ```python
  row = {"description": "café", "tags": [], "annotations": [],
         "depends": [], "custom": [["key", "value"]]}
  observed = DEFAULT_TASK_CODEC.decode_row(row, source_query="test:roundtrip")
  self.assertEqual(observed.to_mapping(), row)
  self.assertEqual(json.loads(DEFAULT_TASK_CODEC.encode_task_import(observed)), row)
  ```

- [x] Give mappings and sequences distinguishable immutable representations.
  For example, use separate frozen dataclasses with `items: tuple` for mapping
  and sequence values. Dispatch `_thaw` on the representation's type; never
  infer a mapping from the shape of its contents.
- [x] Make `_freeze` idempotent for already-frozen values. Both
  `TaskObservation.from_mapping` and `TaskObservation.__post_init__` currently
  freeze arbitrary fields; repeated freezing must not change their meaning.
- [x] Verify `TaskDraft`, arbitrary-field access, diagnostic encoding, and
  semantic fingerprints still work. With identical provenance, an array and
  an object must have distinct semantic fingerprints; ordinary unaffected
  rows must retain their existing behavior.
- [x] Update `_semantic_value` and any direct frozen-value consumers to handle
  the new representation. Do not rewrite existing durable outbox rows to
  compensate for previously lost container information.
- [x] Verify mutating the input or a returned mapping cannot mutate the
  observation/draft. Preserve immutability while correcting serialization.
- [x] Exercise both executable hooks with the regression payload: exit status
  zero, unchanged JSON types, literal Unicode, and empty diagnostic output.
- [x] Run the commands below, review the diff, and record the package result.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_task_codec_roundtrip.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_hook_input_contract.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py --only task_codec --only task_domain --only task_draft
```

**Done when:** Task input, observation, draft, import output, and hook output
preserve JSON container identity without exposing mutable internal state.

**Package 1 evidence (2026-09-08):** Added `tests/test_task_codec_roundtrip.py`
and explicit `_FrozenMapping`/`_FrozenSequence` representations in
`nautical_core/task_models.py`. Unit tests: 442 passed; Package 1 golden slice:
9 passed; executable on-add/on-modify container probe passed with zero stderr.

## 2. Require Checksum Coverage For Every Restored Artifact

**Files:** `nautical_core/restore_service.py`; use existing validators from
`nautical_core/backup_service.py`. Extend `tests/test_restore_service.py` and
`tests/test_restore_cli.py`; document the restore contract in
`docs/operations/offline-readiness.md`.

**Interface:** Preserve `validate_backup(...) -> RestoreReport` and
`restore_backup(..., apply=False) -> RestoreReport`. Put mandatory backup-file
rules in restore validation; generic manifests also serve partial inventories
and must not suddenly require Taskwarrior files.

- [x] Reproduce acceptance of an empty inventory using the existing fixture:

  ```python
  source = self._backup(root)
  publish_manifest(source / "manifest.json", create_manifest(source, files=()))
  self.assertEqual(validate_backup(source).status, "rejected")
  target = root / "restored"
  self.assertEqual(restore_backup(source, target, apply=True).status, "rejected")
  self.assertFalse(target.exists())
  ```

- [x] Add separate regressions for a missing mandatory manifest entry and for
  an unlisted file added under each of `hooks/`, `runtime/`, and `resources/`.
  A changed listed file must continue to fail checksum validation.
- [x] Derive the restore inventory from the two mandatory files plus all files
  under those three optional directories. Require
  `{"taskwarrior-export.json", "lifecycle-outbox.db"}` to be a subset of the
  validated manifest paths and every file to be copied to have a matching
  manifest record. Do not reject unrelated files outside the restore surface
  merely because a generic fixture uses an explicit partial inventory.
- [x] Exclude only the root backup `manifest.json` from self-checksumming.
  Nested runtime manifests are ordinary restored artifacts and need coverage.
- [x] Copy only validated inventory entries, retaining the existing source to
  destination mapping: `runtime/` becomes `.nautical-runtime/`; the outbox
  becomes `.nautical-state/.nautical_lifecycle_outbox.db`.
- [x] Recheck each staged artifact's size and digest against its original
  inventory entry before atomic publication. A source change between
  validation and copying must reject the restore and clean only its staging
  directory. Preserve symlink rejection and the existing empty-target rollback.
- [x] Cover validation-only mode, a valid complete backup, Unicode resources,
  runtime pointer reconstruction, and a source mutation during copying.
- [x] Run the commands below and document that incomplete restore coverage is
  rejected without overwriting live data. Record the package result.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_restore*.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_backup*.py' -q
```

**Done when:** Every restored artifact is matched to a verified record, an
empty/incomplete restore inventory fails, and valid existing backups restore.

**Package 2 evidence (2026-09-08):** Restore validation now requires the two
canonical state artifacts and rejects unlisted files under hooks, runtime, and
resources. Restore copies only validated records, verifies staged size/digest,
and rechecks the source before publication. Restore tests: 25 passed; backup
tests: 48 passed; `py_compile` passed. Contract documented in
`docs/operations/offline-readiness.md`.

## 3. Evaluate Lease Ownership Using The Transaction's Current Time

**Files:** `nautical_core/lifecycle_outbox.py`; inspect the pre-mutation renewal
checks in `nautical_core/lifecycle_application.py`. Extend
`tests/test_lifecycle_failure_injection.py`.

**Interface:** Preserve the existing single/bulk claim and renewal signatures,
result kinds, lease owner checks, and retry semantics.

- [x] Add a deterministic regression using the repository's injected `clock`
  and a transaction wrapper that advances that clock only after the real
  `BEGIN IMMEDIATE` succeeds. Start a valid five-second claim; advance by two
  seconds during acquisition and request a one-second renewal.
- [x] Assert an `APPLIED` renewal has a fresh future expiry and cannot be
  immediately claimed by a second owner. If the old lease expires during lock
  acquisition, renewal must instead return `CONFLICT` without extending it.
- [x] Add one real contention regression with a temporary WAL database and a
  competing writer. Coordinate lock acquisition with an event/pipe handshake;
  use bounded waits and guaranteed worker cleanup, not guessed startup sleeps.
- [x] Move clock sampling inside acquired write transactions before expiry
  comparisons and before calculating the new deadline:

  ```python
  with self._transaction(conn):
      now = self._clock()
      expires = now + float(lease_seconds)
      # Existing ownership comparisons and updates use these fresh values.
  ```

- [x] Apply this rule to `claim_batch`, `claim_intents`, `claim_intent`,
  `claim_integrity_batch`, `_claimed_update`, and `renew_leases`. Check the
  corresponding ownership-sensitive bulk stage/acknowledgement methods and
  `_integrity_transition`; do not change unrelated record-age timestamps.
- [x] Test missing intent, wrong owner, expired owner, terminal state, single
  and bulk renewal, and second-owner reclamation. Verify lifecycle application
  never reaches a Taskwarrior mutation after renewal reports lease loss.
- [x] Retain short configured lease support and the default exit-hook lease;
  do not hide stale-time errors by increasing lease durations.
- [x] Run the commands below, review transaction boundaries, and record the
  package result.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_lifecycle_failure_injection.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py --only lifecycle_outbox --only lease --only shared_outbox
```

**Done when:** Lock-acquisition delay cannot make a successful renewal grant an
already-expired lease, and expired ownership never authorizes a new mutation.

**Package 3 progress (2026-09-08):** Single, bulk, lifecycle, and integrity
claim paths now sample the clock inside their write transaction. Deterministic
renewal tests cover fresh post-lock expiry and expiry during acquisition;
22 lifecycle failure-injection tests pass, including bounded WAL contention
and missing/wrong/expired ownership cases.

**Package 3 evidence (2026-09-08):** Transaction-local time is also used by
bulk renewal, bulk stage/acknowledgement, and integrity transitions. The full
claim-path audit is complete; 22 focused failure-injection tests and 13 golden
lease/outbox tests pass, with mypy clean for `lifecycle_outbox.py`.

## 4. Bound Subprocess Cleanup And Terminate Owned Descendants

**Files:** `nautical_core/taskwarrior_client.py` and
`tests/test_operator_process_contract.py`.

**Interface:** Preserve `TaskwarriorClient.execute(...) -> TaskCommandResult`,
typed `TIMEOUT`, exit code 124, captured evidence, and existing attempt policy.

- [x] Add a POSIX regression whose executable spawns a descendant retaining
  stdout/stderr. The descendant must outlive the configured timeout. Test both
  piped output and `use_tempfiles=True` with an external watchdog and cleanup
  that runs even if the assertion fails.
- [x] Add a descendant that ignores SIGTERM and a parent that exits while its
  descendant keeps the pipes open. Verify the fixture processes are terminated,
  not merely that the caller returns quickly.
- [x] Start POSIX attempts with `start_new_session=True` so cleanup owns a
  separate process group. Signal only that group; never the caller's group.
- [x] Send SIGTERM to the group, then SIGKILL after bounded grace. Do not skip
  group cleanup merely because the immediate child has already exited.
- [x] Replace unbounded `proc.communicate()` after timeout with a bounded final
  drain. Retain the existing 0.2 s termination grace and budget another 0.2 s
  for forced cleanup/output draining. Start the shared 0.4 s cleanup deadline
  when the timeout is detected; retain captured evidence if draining fails.

  ```python
  cleanup_deadline = time.monotonic() + 0.4
  remaining = max(0.0, cleanup_deadline - time.monotonic())
  # Pass remaining to the final communicate/wait; never omit its timeout.
  ```

- [x] Keep direct-child termination as the fallback where process groups are
  unavailable. Preserve `attempts=1` on mutation paths; this fix must not add
  automatic mutation retries.
- [x] Verify normal success, missing executable, leaf timeout, descendant
  timeout, tempfile output, descriptor closure, and child reaping.
- [x] Run the commands below and record observed timeout/cleanup durations and
  the package result. Timing assertions must include reasonable CI scheduling
  tolerance while remaining well below the fixture descendant's lifespan.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_operator_process_contract.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py --only taskwarrior_client
```

**Done when:** A stuck descendant cannot extend a timeout into an unbounded
wait or survive ordinary timeout cleanup within the owned process group.

**Package 4 evidence (2026-09-08):** POSIX attempts now use a dedicated
process group. Timeout handling applies a 0.4-second cleanup deadline, bounded
SIGTERM/SIGKILL group handling, and bounded output draining while retaining
partial evidence. Process-contract tests: 24 passed, including piped and
SIGTERM-ignoring descendants plus tempfile output; mypy passed for
`taskwarrior_client.py`.

## 5. Bound Cache File Reads And Decompression

**Files:** `nautical_core/cache_payload.py`; inspect wiring in
`nautical_core/cache_api.py`. Create `tests/test_cache_resource_limits.py` and
retain existing golden cache corruption/atomic-replacement coverage.

**Interface:** Preserve cache-load miss as `None` and cache-save refusal as
`False`; oversized cache entries must not fail the enclosing task operation.

- [x] Add regressions for an oversized encoded file, a small compressed entry
  exceeding the decoded limit, truncated/corrupt streams, and exact-limit
  payloads. Patch test limits downward so the unit tests need little memory.
- [x] Introduce fixed internal encoded-file and decoded-JSON byte ceilings.
  Start by validating 2 MiB encoded and 8 MiB decoded against existing generated
  hint/cache fixtures; record the adopted limits and fixture sizes. Change
  those initial values only if measured legitimate payloads justify it.
- [x] Enforce the encoded ceiling using both metadata and a bounded
  `read(MAX_CACHE_FILE_BYTES + 1)`, including the existing replacement retry.
- [x] Use a bounded decompressor, not `zlib.decompress` followed by a size test:

  ```python
  decoder = zlib_mod.decompressobj()
  data = decoder.decompress(compressed, MAX_CACHE_JSON_BYTES + 1)
  if (len(data) > MAX_CACHE_JSON_BYTES or decoder.unconsumed_tail
          or not decoder.eof or decoder.unused_data):
      raise ValueError("cache payload exceeds limits or is incomplete")
  ```

- [x] Reject oversized data before UTF-8 decoding, JSON parsing, shape checks,
  or insertion into the in-memory cache. Use the existing quiet miss and
  quarantine handling; diagnostics remain opt-in.
- [x] Have `cache_save` decline entries above the same ceilings so it does not
  repeatedly publish entries that `cache_load` refuses. Preserve atomic writes
  and existing lock behavior; do not change durable outbox settings.
- [x] Verify Unicode, valid boundary-size entries, stale/versioned entries,
  cache replacement during reads, and recurrence results after a cache miss.
- [x] Run the commands below, record a bounded-allocation reproduction, and
  record the package result.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_cache_resource_limits.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py --only cache_load --only cache_save --only cache_schema --only cache_consistency
```

**Done when:** Input size and expansion have explicit bounds, rejected entries
do not enter memory caches, and tasks remain correct when caching is skipped.

**Package 5 evidence (2026-09-08):** Cache reads enforce a 2 MiB encoded
ceiling with bounded file reads and an 8 MiB decoded ceiling with bounded zlib
decompression. Cache saves decline payloads above either limit; truncated,
corrupt, and expansion-limit entries remain quiet cache misses. Resource-limit
tests: 5 passed; targeted cache golden tests: 8 passed; mypy passed.

## 6. Make SQLite's Busy Timeout Match Its Configured Units

**Files:** `nautical_core/lifecycle_outbox.py` and
`tests/test_lifecycle_failure_injection.py`. Inspect hook staging in
`nautical_core/modify_spawn_effects.py`.

**Interface:** `connect_timeout` continues to be specified in seconds;
lock contention continues to produce typed retryable evidence.

- [x] After package 3, add a regression checking that `connect_timeout=0.1`
  produces `PRAGMA busy_timeout` of 100 milliseconds on the created connection.
- [x] Correct the conversion consistently in write and read-only connections:

  ```python
  conn.execute(f"PRAGMA busy_timeout={int(self.connect_timeout * 1000)}")
  ```

- [x] Add a temporary-database writer-contention test that confirms bounded
  return and `RETRYABLE` with `lock_busy=True`. Use the exact PRAGMA assertion
  for unit correctness; wall-clock assertions need scheduling tolerance.
- [x] Verify hook staging never reports a successful enqueue after contention
  prevented persistence; preserve its existing failure/recovery behavior.
- [x] Measure the corrected default hook staging wait before adding a separate
  policy. Retain the existing two-second default for this fix; a shorter
  hook-specific value needs contention/recovery evidence and a separate
  explicit policy decision. Do not silently add new configuration knobs.
- [x] Confirm `synchronous=FULL`, WAL, leases, and transaction scope remain
  intact. Record the package result after the checks below.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_lifecycle_failure_injection.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py --only lifecycle_outbox --only staged_plan
```

**Done when:** The configured seconds agree with SQLite milliseconds and lock
failure remains visible to callers without weakening durability.

**Package 6 evidence (2026-09-08):** Busy-timeout conversion is now
`connect_timeout * 1000` for write and read-only connections. A direct
`connect_timeout=0.1` PRAGMA regression asserts 100 ms; contention asserts
bounded `RETRYABLE` with `lock_busy=True`. Lifecycle failure-injection tests:
23 passed; staged-plan/outbox golden tests: 8 passed; mypy passed.

## 7. Reduce Absent-Field Allocations Without Changing Semantics

**Files:** `nautical_core/task_models.py`; existing benchmarks in
`dev_tools/nautical_perf_budget.py`. Add benchmark coverage there only if the
current workloads do not expose the change; reuse package 1's correctness tests.

**Interface:** Preserve `FieldState.absent()`, `TaskObservation.field(...)`,
field presence semantics, fingerprints, and serialization.

- [x] After package 1, record a fresh baseline with five alternating/repeated
  samples of the existing codec benchmark and a small-task decode/lookup
  workload. Include raw samples, medians, Python version, and workload size.
- [x] Reuse a module-level immutable absent `FieldState` or avoid constructing
  an unused default. Remove eager allocations from both the known-field
  initialization loop and `field()` lookup. No new mutable shared state.

  ```python
  _ABSENT_FIELD_STATE = FieldState(FieldPresence.ABSENT)
  # FieldState.absent() returns this immutable instance.
  # Use the same instance in setdefault/get defaults where appropriate.
  ```

- [x] Rerun package 1's round-trip and immutability tests plus existing domain
  tests. Do not add tests that require object identity merely to mirror the
  implementation; externally observable behavior is the contract.
- [x] Rerun both component workloads against the new baseline on the same
  machine and report the improvement separately. The audit's 5%/38% figures
  are observations, not guaranteed targets or CI timing thresholds.
- [x] Check a representative hook or snapshot workflow for regressions and
  run the existing performance budget tool as part of final integration.
  Do not change budget thresholds to make the optimization pass.
- [x] Retain the optimization only if repeatable benefit is demonstrated
  without semantic changes. Record the package result and measured scope.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_task_codec_roundtrip.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py --only task_codec --only task_domain --only task_draft
```

**Done when:** Equivalent results require fewer allocations and measured
component runtime improves; no whole-command speedup is inferred from a
component-only benchmark.

**Package 7 progress (2026-09-08):** `_ABSENT_FIELD_STATE` is shared by
`FieldState.absent()`, known-field initialization, and lookup defaults without
changing presence or serialization semantics. Full unit discovery: 460 passed;
codec/domain tests and representative workflows pass. Five post-change codec
samples (5 rounds each) were 0.002986, 0.002550, 0.002774, 0.002551, and
0.002782 seconds (median 0.002774 s). A paired pre-change workload showed
allocation optimization with neutral wall time; no whole-command speedup is
claimed.

**Package 7 evidence (2026-09-08):** Identical isolated workload (Python 3.11,
1,000 task decodes plus six field lookups each, five samples) measured baseline
median 0.047752 s (`0.048271, 0.049121, 0.047567, 0.047517, 0.047752`) and
optimized median 0.047871 s (`0.050297, 0.048251, 0.047723, 0.047871,
0.047737`). Wall time is neutral within noise; the optimization is retained
for eliminating repeated immutable absent-state allocations. Full budget
comparison remains part of Package 8 integration.

## 8. Final Integration And Handoff

- [x] All seven packages have recorded acceptance evidence; no incomplete
  package is presented as fixed.
- [x] Run full unit discovery and the full golden suite on isolated fixtures.
- [x] Run the existing performance budget tool after all changes, preserving
  its thresholds. Run the repository's configured type checks for modified
  typed modules using already-available tooling.
- [x] Review strict JSON/Unicode output, mutation guards, postcondition checks,
  WAL/FULL durability, retry behavior, and backup atomic publication in the
  final diff. No tests or benchmarks should touch live user Taskdata.
- [x] Confirm that every changed line belongs to one of the seven packages and
  that unrelated tracked/untracked files were preserved.
- [x] Record checks that could not run, including device-specific Termux
  verification. Do not describe an untested platform as verified.
- [x] Hand off a concise result with each package's status, verification,
  performance measurements, and any migration/compatibility implications.
Installation and release remain separate work.

**Package 8 evidence (2026-09-08):** Full unit discovery: 460 passed. Full
golden suite: 991 passed. Performance budget completed with `ok: true` and no
failed checks. Full-package mypy passed for 236 source files; diff checks pass.
Final review covered strict JSON/Unicode output, mutation guards, lifecycle
postconditions, WAL/FULL durability, retry behavior, and backup atomic
publication. Termux/device-specific verification remains open and is not
represented as verified here.

**Termux verification (2026-09-08):** `termux-15.json` completed on Android/
Python 3.14.6. All staged-hook, lifecycle-staging, queue-drain, and reconcile
apply workflows passed. The enforced run reports two cold-import budget
variances: `cold_core_import` (0.473 s median; 135 modules versus a 120-module
budget) and `cold_modify_impl_import` (0.333 s median; wall-time budget miss,
97 modules within its module budget). These are recorded as device-specific
performance variance; no functional failure was observed.

**Package 6 measurement (2026-09-08):** With the existing default
`connect_timeout=2.0`, a controlled WAL writer contention sample returned
`retryable` with `lock_busy=true` after 2.004 seconds and the worker terminated
within the five-second bound. The two-second default remains unchanged; no new
hook-specific policy was introduced.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -q
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py
PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_perf_budget.py --json --enforce --budget-file dev_tools/perf_budget.json
git diff --check
git diff --stat
git status --short
```
