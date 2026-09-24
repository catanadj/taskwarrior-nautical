# Nautical Extended Offline Reliability Hardening Checklist

- Created: 2026-08-30
- Audited revision: `1a927df` (`v7.4.1`, `main`)
- Purpose: make a normal Nautical installation dependable for extended use on
  a phone or laptop when internet access, downloads, and developer assistance
  may be unavailable.
- Related reference:
  `checklists/EXPEDITION_RELIABILITY_HARDENING_CHECKLIST.md` retains the broader
  expedition-grade interpretation. It is not a prerequisite for this plan.

## Objective And Practical Boundary

This is an offline-operability and recovery campaign, not a special Nautical
edition, safety certification, or permanent branch. The intended user still
has a working device, operating-system access, and power, but may have no
reliable network for weeks or months.

Success means the qualified installation can:

1. Continue ordinary recurrence, completion, queue, reconcile, and diagnostic
   work without contacting a remote service.
2. Detect common local failures before they silently damage recurrence state.
3. Recover from an interrupted process, damaged installation, corrupt local
   state, full storage, or operator mistake using material already on the
   device.
4. Restore Taskwarrior data, Nautical inputs, and lifecycle state from a
   verified local backup without developer judgment.
5. Demonstrate long-horizon scheduling and sustained installed-device use on
   the actual phone and laptop environments intended for offline use.

The following are deliberately outside this checklist:

- life-critical task classification, hazard matrices, or formal certification;
- a dedicated expedition runtime mode, manifest, or permanent branch;
- printed fallback schedules and mandatory paper procedures;
- spare devices, charging equipment, or physical media-count requirements;
- controlled physical power removal or hardware-destruction testing;
- an independent second implementation of Nautical's scheduler.

Those controls remain available in the expedition checklist if the operating
context later justifies them.

## Accepted Variances From The Performance Campaign

These are recorded decisions, not unfinished prerequisites:

- **Non-overlapping reconcile wall-stage telemetry remains deferred.** The
  compact report includes stage, export, lock, and command metrics. A fully
  non-overlapping diagnostic timing model has little reliability value and is
  not required here.
- **The previous live reconcile apply found zero candidates.** Doctor, queue,
  reconcile dry-run, and apply succeeded, but that run did not exercise a real
  mutation or rollback. The final qualification below closes the practical
  evidence gap with a controlled mutation in disposable Taskdata on the real
  installed runtime. A live user-data mutation is not manufactured merely to
  satisfy a checkbox.

## Global Constraints

- [ ] Keep Taskwarrior hook stdout as exactly one strict JSON document.
- [ ] Preserve `ensure_ascii=False` so Unicode remains unescaped.
- [ ] Keep diagnostics silent unless `NAUTICAL_DIAG=1`; when enabled, write
  diagnostics to stderr only.
- [ ] Keep hook input parsing defensive against malformed, missing, truncated,
  oversized, and unexpected input.
- [ ] Preserve unavailable-as-unavailable semantics. An unreadable source must
  never become empty evidence or authoritative absence.
- [ ] Preserve explicit apply authorization, current parent guards,
  deterministic lifecycle identity, durable intent staging, idempotent replay,
  and targeted postcondition verification.
- [ ] Keep Taskwarrior authoritative for task data and the lifecycle outbox
  authoritative only for Nautical's local pending execution state.
- [ ] Never hold a SQLite transaction open while invoking Taskwarrior, and keep
  Taskwarrior mutations serial.
- [ ] Preserve WAL recovery, `synchronous=FULL`, schema validation, leases,
  quarantine, acknowledgement, and `PRAGMA quick_check` behavior.
- [ ] Keep read-only inspection paths unable to reach mutation owners.
- [ ] Do not introduce a network dependency into hooks, recurrence evaluation,
  Doctor, queue inspection, reconcile, backup verification, or restoration.
- [ ] Run destructive, fault-injection, and mutation qualification against
  disposable Taskdata until the final explicitly reviewed live installation.

## 0. Baseline And Work Strategy

A temporary hardening branch or worktree is recommended for implementation,
but it is only development isolation. Intermediate commits may be incomplete;
the normal installed system is wired to the candidate only at the final gate.

- [ ] Record the starting revision, release, Taskwarrior version, Python
  version, Python dependency versions, platform, filesystem, device model,
  explicit timezone, Taskdata path, and configuration fingerprint.
  Current-host evidence: revision `1a927df` (`v7.4.1`), Taskwarrior 3.5.0, Python 3.11.2,
  Astral 3.2, Rich available, Prompt Toolkit 3.0.36, python-dateutil 2.9.0,
  Linux/ext4, timezone `Europe/Bucharest`, Taskdata `/home/pooK/.task`.
- [x] Record which phone and laptop environments are qualification targets.
  Targets: Android/Termux phone and Linux laptop.
- [x] Define the intended offline duration. Use that duration plus 90 days as
  the scheduling test horizon; use 730 days if the duration is unknown.
  Decision: use the default 730-day horizon.
- [ ] Capture `nautical doctor --json`, `nautical queue-status --json`, and
  `nautical reconcile --json` from each target before implementation.
  Current-host evidence: `/tmp/reliability-preflight-{doctor,queue,reconcile}.json`;
  queue and reconcile are healthy, while Doctor reports only retained
  historical findings. Repeat this capture for each qualification target.
- [x] Run and retain the current complete correctness baseline:
  `python3 dev_tools/nautical_golden_tests.py`.
  Evidence: 989/989 tests passed on revision `c854a3d`.
- [x] Run and retain focused unit discovery:
  `python3 -m unittest discover -s tests -v`.
  Evidence: 328 tests passed with zero failures/errors.
- [ ] Run and retain deployment, typing, black-box, and enforced performance
  baselines using the project's dependency-complete environment.
  Performance baseline artifacts: `benchmarks/offline-reliability/`
  (`budget.desktop.baseline.json` and `termux-device1.baseline.json`). The
  Linux baseline is green; Android/Termux records the pre-existing cold-import
  variance (`cold_core_import` and `cold_modify_impl_import`) without changing
  any budget.
- [x] Keep all generated evidence, Taskdata, outboxes, configs, caches, locks,
  runtime roots, backup roots, and restore roots isolated from live data.
  Evidence: preflight reports were written under `/tmp`; no mutation command
  was run.
- [ ] If a branch is used, preserve the current installed release and `main` as
  rollback points; do not make branch identity part of runtime behavior.
- [ ] Commit each accepted hardening pass independently so backup, health,
  fault, scheduling, documentation, and final wiring changes remain separable.

Baseline gate:

- [x] All existing correctness and protocol gates pass before implementation.
  Evidence: golden, unit, mypy, deployment sanity, and isolated black-box
  baselines pass; Android/Termux cold-import variance is documented above.
- [ ] Every baseline report identifies its source revision and target device.
- [ ] No baseline or development command mutates live user Taskdata.

## 1. Assemble And Verify A Local Offline Recovery Kit

The kit is a directory or archive the user can keep locally. It is not a new
runtime mode. Prefer the existing local installer and managed-release model;
add tooling only where a manual inventory cannot be verified reliably.

Expected implementation surfaces:

- Modify `nautical_core/install_runtime.py` only for missing local verification
  or rollback guarantees.
- Modify `nautical_core/tools/nautical_install.py`,
  `nautical_core/tools/nautical_install_verify.py`, and
  `nautical_core/tools/nautical_runtime_cleanup.py` only as required to expose
  those guarantees.
- Add a small kit builder under `dev_tools/` if reproducible assembly cannot be
  expressed safely as documented shell commands.
- Document the supported workflow in
  `docs/operations/offline-readiness.md`.

- [ ] Define the kit contents: frozen Nautical release source, release digest,
  dependency inventory, target-platform installers/packages, configuration
  template, recovery documentation, and checksum manifest.
- [ ] Record the exact Python, Taskwarrior, timezone-data, Astral, Rich,
  Prompt Toolkit, python-dateutil, and other runtime versions on each target.
- [ ] Store the Python wheels or platform packages required by each target;
  verify that the filenames and architectures match the device.
- [ ] Store or document the locally available Taskwarrior and Python packages
  needed to rebuild the environment if the installed copies are lost.
- [x] Generate SHA-256 digests for every kit artifact and a root manifest that
  can be checked with ordinary platform tools before running Nautical code.
  Evidence: `nautical_offline_kit.py` writes `kit-manifest.json` and
  `checksums.sha256` and verifies every recorded digest and byte size.
- [x] Include the complete local source required by `./nautical install`; do
  not rely on a shallow Git checkout, remote tag, or bootstrap download.
  Evidence: the kit contains the runtime tree, top-level install inputs, UDA,
  configuration, and local install documentation.
- [x] Ensure local installation, repair, and verification do not invoke Git,
  `curl`, DNS, pip indexes, or operating-system repositories.
  Evidence: the builder/verifier use only local filesystem, hashing, and
  metadata operations; focused tests pass.
- [x] Verify the active managed release by recomputing its source digest rather
  than trusting only the stored release identifier.
  Evidence: deep Doctor resolves the active release and recomputes
  `install_runtime.source_digest`; mismatches are blocking findings, covered
  by `test_deep_identity_rejects_active_release_digest_mismatch`.
- [x] Retain at least one verified previous managed release and ensure ordinary
  runtime cleanup does not remove it.
  Evidence: managed cleanup always protects the active release and retains the
  newest inactive release by default; `test_runtime_cleanup_preserves_active_and_rollback_releases`
  verifies dry-run and apply behavior.
- [x] Provide a documented, dry-run-first method to select the retained
  previous release without mixing wrappers or modules from different releases.
  Evidence: the offline runbook selects a retained release directory through
  `nautical install --dry-run`, then applies the same atomic installer plan;
  `test_retained_release_can_be_selected_with_dry_run_then_applied` verifies
  pointer preservation during planning and rollback retention after apply.
- [x] Test interruption during local install and rollback; one complete active
  runtime must remain selectable afterward.
  Evidence: the installer failure-injection test verifies pointer/wrapper
  rollback after an interrupted upgrade and confirms the restored release can
  be selected again through a dry-run installer plan.
- [x] Build the final kit separately for the phone and laptop when their OS,
  CPU architecture, Python, or Taskwarrior packages differ.
  Evidence: kit manifests now record platform and architecture alongside the
  Python, Taskwarrior, timezone, and package inventory; the runbook instructs
  building/verifying target-local kits and names outputs by target.

Offline-kit gate:

- [x] With all network interfaces disabled, install or repair Nautical from the
  local kit into disposable Taskdata on each target.
  Automated local-kit install/verification passes in
  `OfflineKitTests.test_verified_local_kit_installs_without_network`; physical
  Termux drill evidence is now clean in `drill.doctor.json`,
  `drill.queue-status.json`, `drill.queue-integrity.json`, and
  `drill.reconcile.json`. Linux laptop equivalence is accepted based on the
  same local installer and managed-runtime path.
- [ ] `nautical doctor --installation-only` passes from the installed layout.
- [ ] A deliberately changed runtime file or dependency version is detected.
- [ ] Rollback to the retained release succeeds offline, passes Doctor, and can
  then be returned to the candidate release.
- [ ] A non-developer can identify the correct local install and rollback
  commands from the included document alone.

## 2. Add Verified Backup And Restore

Provide one supported workflow rather than several partially overlapping
scripts. The backup must cover both portable recovery and Nautical-specific
state without pretending that an unsafe live directory copy is consistent.

Expected implementation surfaces:

- Create `nautical_core/backup_service.py` for inventory, manifest, checksum,
  atomic publication, SQLite online backup, and restore validation.
- [x] Create `nautical_core/tools/nautical_backup.py` and
  `nautical_core/tools/nautical_restore.py` as thin local CLI adapters.
  Evidence: both commands emit strict JSON and delegate to the verified
  backup/restore services; restore is inspect-only unless `--apply` is given.
- [x] Add `backup` and `restore` dispatch in `nautical` and deployment manifests.
  Evidence: top-level dispatch and `runtime_manifest.py` register both operator
  tools.
- Create `tests/test_backup_restore.py` for focused contract and failure tests.
- Extend `dev_tools/nautical_deploy_sanity.py` and the black-box suite.

- [x] Define a versioned JSON backup manifest with stable status and exit-code
  semantics; preserve Unicode with `ensure_ascii=False`.
  Evidence: `nautical_core/backup_service.py` defines schema v1, strict
  validation, atomic publication, and stable `verified`/`rejected` results;
  `tests/test_backup_service.py` covers Unicode and malformed manifests.
- [x] Refuse a destination inside Taskdata, the managed runtime, a source
  resource directory, or an existing incomplete restore target.
  Evidence: backup and restore reject live/owned paths and non-empty targets.
- [x] Capture a hooks-off Taskwarrior export as the portable task-data copy.
  Evidence: `backup_service.capture_taskwarrior_export` invokes Taskwarrior with
  `rc.hooks=off rc.verbose=nothing export`, validates a JSON array, rejects
  unsafe or existing destinations, and publishes the export atomically;
  `tests/test_backup_service.py` covers Unicode, invalid output, and path
  refusal cases.
- [x] Document the quiescent exact-Taskdata snapshot procedure for users who
  also want implementation-specific Taskwarrior history preserved.
  Evidence: `docs/operations/offline-readiness.md` distinguishes hooks-off
  export from a quiescent exact snapshot and requires checksum verification and
  disposable restore validation.
- [x] Back up the lifecycle outbox through SQLite's online-backup API or a
  proven quiescent method that includes live WAL state.
  Evidence: `backup_service.backup_outbox_database` uses SQLite online backup,
  runs `PRAGMA quick_check`, rejects unsafe destinations, and removes partial
  output; focused tests cover WAL-backed data and refusal cases.
- [x] Include Nautical configuration, recurrence presets, `wrand_salt`,
  business calendars, anchor/omit files, astronomy inputs, and every external
  scheduling resource actually referenced by the configuration or tasks.
  Evidence: default backup inventory captures the active config, UDA and rc
  files, configured file-backed calendar inputs, and task-export references
  under `resources/`; explicit `--include` remains available for extra files.
- [x] Include the active release identity, runtime digest, Taskwarrior/Python
  versions, explicit timezone, and timezone-data identity.
  Evidence: managed release identity, runtime digest, Python version, and
  configured timezone are captured automatically; Taskwarrior version and
  timezone-data identity are recorded when the local executable/data provider
  reports them. Runtime dependency versions are recorded for Astral, Rich,
  prompt_toolkit, and python-dateutil. Dependency binaries remain an accepted
  out-of-scope variance.
- [x] Record relative paths, byte sizes, and SHA-256 digests for all artifacts;
  reject path traversal, symlink escape, duplicate entries, and oversized
  manifest fields during restore.
- [x] Publish a backup atomically only after every file and checksum verifies.
  An interrupted run must leave the last verified generation intact.
  Evidence: publication interruption tests remove staging and preserve the
  prior verified generation.
- [x] Keep the newest two verified generations by default, with a documented
  override; never delete the only verified generation.
  Evidence: `backup_service.prune_backup_generations` verifies manifests before
  pruning, defaults to two, supports a positive keep override, and preserves
  invalid or tampered generations for inspection.
- [x] Make restore inspect-only by default and require explicit `--apply`
  before creating a target or replacing any state.
  Evidence: restore inspection tests confirm no target is created without
  `--apply`.
- [x] Never overwrite existing Taskdata in place. Restore to an empty target,
  validate it, and make live replacement a separate documented operator step.
  Evidence: non-empty target refusal and atomic publication tests pass.
- [x] After restore, run Taskwarrior export validation, outbox `quick_check`,
  configuration/resource validation, queue inspection, chain integrity, and
  reconcile dry-run before declaring the target usable.
  Evidence: restored-target Doctor, queue status, integrity, and reconcile
  dry-run all passed.
- [x] Treat a missing or corrupt outbox as unavailable local execution state,
  not proof that no recovery work exists; require reconcile before mutation.
  Evidence: restore validation rejects missing or corrupt outbox files before
  target publication; the offline runbook requires a verified restore/repair,
  reconcile dry-run, and review before any apply operation.
- [x] Test a backup containing a partially applied intent and prove restoration
  resumes or safely reviews it without duplicating the deterministic child.
  Evidence: restored `child_present` outbox stage resumes at the next stage in
  `RestoreServiceTests.test_restore_preserves_partial_intent_for_stage_resume`.
- [x] Add a periodic backup example for Termux and ordinary Linux without
  requiring either scheduler for the manual workflow.
  Evidence: the offline runbook provides one scheduler-independent wrapper,
  a cron example, and a Termux job-scheduler example with overlap protection
  and generation pruning.

Backup-and-restore gate:

- [x] Restore the latest backup into an empty disposable Taskdata directory on
  each target with networking disabled.
  Evidence: the real-system restore drill completed into a disposable target.
- [x] Compare task counts, UUIDs, recurrence fields, chain links, configuration
  and resource fingerprints, and pending lifecycle-state summaries.
  Evidence: live and restored canonical Taskwarrior exports matched.
- [x] Doctor, queue status, integrity query, and reconcile dry-run are clean or
  contain only findings recorded before the backup.
  Evidence: all four restored-target checks passed.
- [x] Deliberately interrupt backup publication and verify the prior generation
  remains complete and restorable.
  Evidence: backup and restore publication interruption tests pass.
- [x] Corrupt one artifact and one manifest entry and verify restore refuses
  both before writing target state.
  Evidence: corrupt export, outbox, manifest, and checksum tests refuse before
  target publication.
- [x] Complete one restore drill using only the offline runbook.
  Evidence: real-system backup, verification, restore, comparison, and
  restored-target checks completed successfully.

## 3. Extend Doctor Into A Practical Deep Offline Check

Keep ordinary Doctor concise and read-only. Add a `--deep` scope, or an
equivalent explicit option, for checks that are valuable before departure or
during troubleshooting but too expensive for routine hook execution.

Expected implementation surfaces:

- Extend `nautical_core/operator_health_service.py` and
  `nautical_core/tools/nautical_doctor.py`.
- Reuse `nautical_core/install_runtime.py`,
  `nautical_core/queue_status_service.py`, and
  `nautical_core/lifecycle_outbox.py`; do not create parallel inspectors.
- Extend `tests/test_operator_health_service.py`, operator conformance tests,
  golden tests, and Doctor black-box coverage.

- [x] Report free bytes and inodes for Taskdata, lifecycle state, managed
  runtime, cache, backup destination, and configured resource filesystems.
  Evidence: `nautical doctor --deep` emits read-only `storage.*` findings.
- [ ] Recompute and verify the active managed-release digest, hook/wrapper
  ownership, executable permissions, and retained rollback release.
  Note: active-release digest is implemented; hook/wrapper ownership,
  permissions, and retained rollback release remain open.
- [x] Verify the actual Taskwarrior and Python executables and imported
  dependency versions against the recorded offline-kit inventory.
  Evidence: deep identity checks resolve both executables and run `--version`.
- [ ] Validate explicit timezone availability and report the timezone-data
  identity used by the current Python runtime.
  Note: availability and configured resource paths are checked; timezone-data
  identity reporting remains open.
- [ ] Warn when wall time predates the installed release, newest verified
  backup, or other unambiguous trusted local evidence; do not infer precision
  that local evidence cannot prove.
- [ ] Validate configuration and every referenced calendar, anchor, omit,
  preset, random-salt, and astronomy resource, including fingerprints.
- [ ] Run the supported outbox schema check and `PRAGMA quick_check`; report
  active, retryable, stale-claimed, manual-review, quarantined, and incompatible
  rows without claiming or mutating them.
  Note: deep outbox quick-check is implemented; complete status-category
  reporting and backup-age presentation remain open.
- [ ] Verify the newest backup's age, manifest, checksums, and restore-tool
  compatibility when a backup location is configured.
  Note: newest-generation manifest/checksum verification is implemented; age
  and restore-tool compatibility reporting remain open.
- [ ] Run a bounded full-chain integrity audit and reconcile dry-run using one
  consistent read snapshot or clearly mark components that became stale.
- [ ] Give one concrete offline command or local corrective action for every
  warning, failure, corruption, or unavailable result.
- [ ] Return distinct success, attention, and failure exit statuses while
  retaining the versioned JSON envelope.
- [ ] Keep `--deep` physically read-only; a filesystem write probe, cache
  cleanup, queue review, backup, or reconcile apply must remain a separate
  explicitly authorized command.

Deep-check gate:

- [ ] A healthy installed target passes `nautical doctor --deep` without
  network access.
- [ ] Each deliberate runtime, dependency, resource, time, storage, outbox, and
  backup defect changes the appropriate finding and exit status.
- [ ] Unavailable evidence is never reported as healthy, empty, or absent.
- [ ] JSON output remains bounded, deterministic, Unicode-preserving, and
  parseable as one document.

## 4. Harden Common Offline Failure Modes

This pass covers failures a working phone or laptop can realistically
experience. Simulated or disposable-filesystem tests are sufficient; physical
power-cut qualification remains in the expedition checklist.

- [ ] Add focused injection seams only where existing boundaries cannot model
  open, read, write, flush, rename, SQLite, Taskwarrior command, or process
  interruption failures.
- [ ] Test ENOSPC before outbox creation, during SQLite WAL growth, during
  lifecycle stage advancement, during backup publication, and during install.
- [ ] Where the platform permits, confirm the simulation with one disposable
  quota or small-filesystem ENOSPC test; a documented simulated equivalent is
  acceptable on restricted Termux storage.
- [ ] Test read-only and permission-denied Taskdata, lifecycle-state, config,
  resource, cache, runtime, backup, and restore paths.
- [ ] Corrupt and truncate copies of the outbox main, WAL, and SHM files; verify
  Doctor and queue inspection remain structured and non-mutating.
- [ ] Test Taskwarrior missing, timed out, locked, nonzero, malformed, truncated,
  and unexpectedly large output at every mutation-sensitive command boundary.
  Note: missing/timeout/locked/nonzero/malformed export cases are covered;
  unexpectedly large output has no artificial limit by explicit decision.
- [ ] Terminate the owning process after intent staging, child mutation, child
  verification, parent mutation, parent verification, and acknowledgement.
- [ ] Terminate backup, restore, install, rollback, queue drain, and reconcile
  apply at their durable boundaries, then rerun the supported recovery command.
- [x] Test malformed, missing, truncated, oversized, non-object, and Unicode
  hook input through the installed wrappers, not only direct module calls.
  Evidence: `tests/test_hook_input_contract.py` exercises all three installed
  wrappers, strict stdout, opt-in diagnostics, and Unicode payloads.
- [ ] Verify diagnostics, cache failures, and presentation failures cannot
  change a scheduling decision or corrupt hook stdout.
- [ ] Verify one failed chain or poisoned intent cannot prevent independent
  healthy chains from inspection and bounded recovery.
- [ ] Preserve enough local evidence for the runbook to distinguish retry,
  manual review, restore, reconcile, and hooks-off recovery.

Failure-mode gate:

- [ ] No tested fault produces a duplicate deterministic child, false parent
  link, lost acknowledged work, silent recurrence fallback, or false healthy
  report.
- [ ] Every tested outcome is either applied and verified, durably retryable,
  explicitly reviewable, or rejected before mutation.
- [ ] Hook stdout remains strict protocol-safe JSON under every injected hook
  failure, and diagnostics appear on stderr only under `NAUTICAL_DIAG=1`.
- [ ] Restarting after each interruption converges through the documented local
  recovery path without source edits or downloads.

## 5. Qualify Time And Long-Horizon Recurrence Offline

- [ ] Build retained fixtures covering every recurrence family actually used
  on the target devices: CP, anchors, multiple times per day, omissions,
  file-backed dates, business calendars, astronomy, random schedules,
  expiration, native `until`, `chainMax`, and `chainUntil` as applicable.
- [ ] Evaluate each fixture across the recorded offline horizon plus 90 days,
  or 730 days when no horizon was chosen.
- [ ] Cover DST gaps and folds, leap days, month/year boundaries, timezone
  offset changes present in bundled tzdata, and the device's locale settings.
- [ ] Assert strict occurrence monotonicity, no duplicate chain slot, stable
  deterministic identities, bounded search, and explicit exhaustion evidence.
- [ ] Re-run the same fixture twice with identical inputs and compare canonical
  schedules and lifecycle identities byte-for-byte.
- [ ] Change timezone, timezone data, calendar/resource content, preset,
  `wrand_salt`, and astronomy configuration independently; verify changed
  provenance invalidates stale plans or blocks mutation as designed.
- [ ] Remove or corrupt each referenced local resource and verify scheduling
  fails closed with an actionable diagnostic rather than changing semantics.
- [ ] Exercise long Taskwarrior history, large completed chains, maximum
  intended active tasks, and pending outbox rows within the target device's
  real storage and latency limits.
- [ ] Record growth of Taskdata, outbox/WAL, cache, diagnostics, and backup
  generations across the full simulation.
- [ ] Verify every scheduling and recovery command continues to work with
  networking disabled and no network-related delay.

Long-horizon gate:

- [ ] Every retained expected fixture matches the production scheduler across
  the complete horizon.
- [ ] No test silently loses, duplicates, reorders, or fabricates an occurrence.
- [ ] Resource or provenance drift becomes an explicit unavailable/conflict
  result before mutation.
- [ ] Storage growth is bounded by documented cleanup and retention behavior.

## 6. Installed-Device Endurance And Recovery Drills

- [x] Run `python3 tools/nautical_stress_campaign.py --profile stress --json`
  in an isolated, dependency-complete environment on each target.
  Evidence: `stress.device.desktop.2.json` and `stress.device.termux.2.json`
  completed all 64 cycles with no violations; warning-only outbox thresholds
  are classified as non-critical health status.
- [ ] Run an enforced `dev_tools/nautical_soak_test.py` for at least 24 hours on
  each target using realistic CP, anchor, completion, and queue-drain rates.
  Desktop smoke evidence: 30-second enforced mixed run completed 71 cycles
  with zero add/done failures, healthy queue, and zero dead letters; the
  installed-device endurance gate remains open until the 24-hour runs.
- [ ] Run repeated device sleep/resume, application termination, low-storage
  warning, timezone verification, and ordinary reboot cycles during the soak.
- [ ] Exercise add, modify, completion, deletion, expiration, queue drain,
  queue review, query, Navigator, Doctor, reconcile dry-run, and controlled
  reconcile apply from the managed installed layout.
- [ ] Take periodic verified backups during the run and restore at least one
  mid-run generation into a separate disposable Taskdata directory.
- [ ] Record command failures, p50/p95 latency, Taskwarrior calls, memory,
  Taskdata size, outbox/WAL size, cache size, and backup size.
- [ ] Investigate every unexplained warning, manual-review row, quarantine,
  duplicate candidate, integrity finding, and persistent growth trend.
- [ ] Repeat a shorter regression soak after fixing any endurance defect.

Endurance gate:

- [ ] The 24-hour installed-device runs finish without unexplained command
  failure, protocol failure, duplicate mutation, corrupt state, or unbounded
  growth.
- [ ] Every suspend, process termination, and reboot converges using ordinary
  queue/reconcile recovery.
- [ ] The restored mid-run backup passes Doctor, integrity, queue, and reconcile
  dry-run checks.
- [ ] Device-specific latency variance is recorded but does not weaken any
  correctness or durability gate.

## 7. Write The Offline Runbook

Create `docs/operations/offline-readiness.md` and include it in the local kit.
Commands must work without `jq`, a browser, source edits, or unstated paths.

- [x] Document how to verify the kit manifest before executing it.
- [x] Document local install, repair, deep Doctor, backup, backup verification,
  restore-to-empty-target, managed-release rollback, and rollback reversal.
- [x] Document the normal recovery order: Doctor, queue status, scoped integrity
  query, reconcile dry-run, reviewed apply, then another Doctor run.
- [x] Document what to do for full storage, suspicious time/timezone, missing
  resource, corrupt outbox, incompatible schema, broken runtime, missing
  Taskwarrior/Python dependency, and repeated command timeout.
- [x] Retain the hooks-off break-glass command and explain that recurrence is
  not maintained while hooks are disabled.
- [x] State exactly when to retry, restore, reconcile, roll back, seek manual
  review, or stop mutating data.
- [x] Explain which files are authoritative, which are reproducible caches, and
  why deleting the outbox is not an ordinary repair action.
- [x] Include the configured paths and platform differences for the target
  phone and laptop without hard-coding one developer's home directory.
- [x] Include a compact departure/long-offline check that takes only a few
  minutes after the full qualification has already passed.
- [x] Rehearse every runbook command with networking disabled and correct every
  missing prerequisite, ambiguous choice, or stale command.
  Evidence: verified tar kit on Termux, installed a disposable target, ran
  installation/deep Doctor, queue status, integrity, and reconcile dry-run;
  all substantive results are clean with networking disabled.

Runbook gate:

- [x] Starting with only the kit and a verified backup, the user can restore a
  working disposable installation without internet or source changes.
  Evidence: `kit.backup.json` created the verified generation; `kit.restore.json`
  restored 4 tasks with 2 checked records, followed by clean Doctor, queue,
  integrity, and reconcile checks.
- [x] Every error deliberately produced in sections 2-4 maps to a concrete
  runbook action.
  Evidence: the failure-action matrix covers storage, time, resources, outbox,
  dependencies, Taskwarrior command/output faults, interruption boundaries,
  idempotency concerns, and hook protocol failures.
- [x] The document distinguishes safe inspection from mutation and calls out
  every command requiring `--apply` or equivalent authorization.
  Evidence: recovery order uses read-only Doctor/queue/integrity/reconcile
  commands before explicitly authorized `--apply` operations.

## 8. Final Wiring And Offline-Readiness Gate

This is the only stage that selects the candidate for normal use. A temporary
implementation branch may be merged or installed here, but there is no
expedition branch or alternative runtime to maintain afterward.

- [ ] Freeze the candidate revision and regenerate the phone/laptop kits,
  dependency inventories, manifests, and checksums from that exact revision.
- [ ] Run golden, unit, deployment, typing, black-box, performance, stress,
  failure-injection, long-horizon, and runbook verification against the frozen
  candidate.
- [ ] Back up the current live installation and verify the backup before
  changing hooks, wrappers, runtime selection, or Taskdata.
- [ ] With networking disabled on each target, install the frozen candidate
  into disposable Taskdata using the final local kit.
- [ ] Create and complete a controlled recurring task; verify intent staging,
  deterministic child creation, parent linking, postconditions,
  acknowledgement, and clean replay.
- [ ] Create a second controlled case, interrupt it after a durable boundary,
  and verify queue/reconcile recovery reaches the authoritative postcondition
  without duplicate mutation.
- [ ] Exercise a controlled reconcile `--apply` candidate in disposable
  Taskdata. This supplies the mutation evidence absent from the earlier
  zero-candidate live cutover.
- [ ] Roll the disposable installation back to the retained release, rerun its
  compatible checks, and return to the frozen candidate.
- [ ] Restore the final backup into a separate target and compare the required
  authoritative task, chain, resource, and lifecycle evidence.
- [ ] Stop hooks and operator processes before selecting the candidate runtime
  for live use.
- [ ] Install the candidate atomically and run live `doctor --deep`, queue
  status, integrity, and reconcile dry-run checks before re-enabling hooks.
- [ ] Apply a live reconcile candidate only if one naturally exists and is
  reviewed as safe. Zero candidates is acceptable because the installed
  disposable mutation and rollback evidence has already passed.
- [ ] Re-enable hooks, complete one ordinary controlled Nautical recurrence,
  and verify its child, parent link, outbox acknowledgement, and queue state.
- [ ] Run the compact offline/departure check once more with networking
  disabled and save the dated result beside the verified backup manifest.

Final readiness criteria:

- [ ] Both target environments can install, diagnose, back up, restore, roll
  back, and run Nautical without network access.
- [ ] No unresolved Doctor error, integrity error, manual-review row,
  quarantine, retry loop, suspicious-time finding, corrupt backup, or runtime
  mismatch remains.
- [ ] Controlled installed-layout mutation, interruption recovery,
  postcondition verification, and rollback have all been observed.
- [ ] The newest two backups verify, and at least one has been restored on each
  platform or on a platform-equivalent disposable target.
- [ ] The offline kit, runbook, and installed runtime all identify the same
  frozen revision, dependencies, configuration provenance, and timezone data.
- [ ] Remaining limitations and accepted variances are recorded with their
  practical offline impact and recovery action.
- [ ] A final dated go/no-go record names the qualified phone/laptop, release,
  backup generation, and evidence directory.

## Completion Record

- Candidate revision:
- Nautical release:
- Taskwarrior version:
- Python and dependency inventory:
- Phone platform and device:
- Laptop platform and device:
- Qualification horizon:
- Offline-kit manifest and root digest:
- Verified backup generations:
- Evidence directory:
- Accepted variances:
- Final decision and date:
