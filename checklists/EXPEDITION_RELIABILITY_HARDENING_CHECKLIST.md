# Nautical Expedition Reliability Hardening Checklist

- Created: 2026-08-30
- Audited revision: `1a927df` (`v7.4.1`, `main`)
- Purpose: qualify Nautical for prolonged offline operation where updates,
  remote support, replacement downloads, and developer intervention may be
  unavailable.

## Reliability Objective And Boundary

Nautical already has strong logical recovery: deterministic lifecycle identity,
durable SQLite intent staging, guarded Taskwarrior mutation, targeted
postcondition verification, idempotent replay, chain-integrity analysis, and
reconcile recovery. This campaign must preserve those guarantees while adding
protection against physical storage failure, data loss, invalid system time,
runtime corruption, unavailable dependencies, and operator error.

This checklist is an engineering hardening and operational-readiness program,
not a safety certification. Software on one device cannot be the only control
for a life-critical activity. Final readiness requires a spare device,
independent power and storage, and a human-readable critical schedule that does
not depend on Nautical or Taskwarrior being operational.

Success means the expedition installation can:

1. Detect unsafe time, configuration, resource, runtime, storage, and task-data
   conditions without falsely reporting health.
2. Fail in the explicitly chosen safe direction for every critical recurrence.
3. Recover deterministically after process death, power loss, outbox loss,
   storage exhaustion, or restoration onto a replacement device.
4. Be installed, verified, rolled back, backed up, and restored without network
   access or developer judgment.
5. Preserve a separately readable schedule when the complete software stack is
   unavailable.

## Accepted Variances From The Performance Checklist

The following items were deliberately accepted during the previous performance
campaign. They are recorded as closed variances, not unfinished prerequisites
for this checklist:

- **Fully non-overlapping reconcile wall-stage telemetry was deferred.** Compact
  reports retain stage, export, lock-contention, and command-purpose evidence,
  but do not divide the entire workflow into mutually exclusive wall stages.
  This is diagnostic-only and has little expedition reliability value. Do not
  implement it merely to close a checkbox.
- **The live cutover apply found zero candidates.** Live Doctor, queue,
  reconcile dry-run, and apply completed successfully, but no Taskwarrior
  mutation occurred, so that run could not prove mutation postconditions or a
  mutation-sensitive rollback. The variance remains accepted for that release.
  Section 10 adds a new expedition qualification drill using controlled,
  disposable Taskdata and the actual expedition installation; it does not
  retroactively redefine the earlier cutover result.

## Global Safety Constraints

- [ ] Keep Taskwarrior hook stdout as one strict JSON document.
- [ ] Preserve `ensure_ascii=False` for JSON output.
- [ ] Keep diagnostics silent unless `NAUTICAL_DIAG=1`, then send them to
  stderr only.
- [ ] Keep hook input parsing defensive against malformed, missing, truncated,
  oversized, and unexpected input.
- [ ] Preserve unavailable-as-unavailable semantics; never turn an unreadable
  source into empty evidence or authoritative absence.
- [ ] Preserve explicit apply authorization, current parent guards,
  deterministic lifecycle identity, durable intent staging, idempotent replay,
  and targeted postcondition verification.
- [ ] Keep read-only operations physically unable to reach mutation owners.
- [ ] Keep Taskwarrior mutations serial and never hold a SQLite transaction
  open while invoking Taskwarrior.
- [ ] Preserve SQLite WAL, `synchronous=FULL`, schema validation, leases,
  poison-row quarantine, acknowledged replay, and `PRAGMA quick_check`.
- [ ] Do not add a shadow task store. Taskwarrior remains authoritative task
  data; the Nautical outbox remains local durable execution state.
- [ ] Do not require network access for normal use, diagnosis, backup,
  restoration, installation, or rollback.
- [ ] Do not weaken correctness, durability, validation, or evidence coverage
  to satisfy a timing, storage, or convenience target.
- [ ] Keep every test, fault campaign, restore drill, and mutation rehearsal off
  live user Taskdata until the final explicitly authorized cutover checks.

## 0. Isolate The Campaign And Capture A Baseline

This work should use a dedicated, intentionally non-operational branch because
backup, rollback, health, and expedition-profile surfaces may be incomplete
until the final wiring stage.

- [ ] Create a dedicated branch or worktree from an explicit green `main`
  revision, for example `expedition-reliability-hardening-v1`.
- [ ] Record the starting revision, Nautical release, Taskwarrior version,
  Python version, dependency versions, tzdata identity, platform, filesystem,
  device model, and configuration fingerprint.
- [ ] Preserve `main` and the current managed release as known-good rollback
  points while the branch is under construction.
- [ ] Keep the development branch uninstalled and non-operational until the
  final wiring gate.
- [ ] Use disposable Taskdata, config, calendar, cache, lock, outbox, runtime,
  backup, and restore paths for all automated work.
- [ ] Record normal and shuffled golden results, unit discovery, configured and
  strict mypy, compilation, deployment sanity, black-box integration, stress,
  soak, Doctor, queue, reconcile, and performance results.
- [ ] Capture a complete read-only baseline report from each intended
  expedition device before changing production code.
- [ ] Commit each accepted hardening pass independently; keep benchmark,
  documentation, and production changes attributable.

Baseline gate:

- [ ] The starting revision is green under the complete current verification
  matrix.
- [ ] No baseline or campaign command read or mutated live user Taskdata.
- [ ] The exact source, binaries, dependencies, configuration, and device used
  for the baseline can be identified from retained evidence.

## 1. Define The Expedition Safety Profile

Reliability cannot be proven until unsafe outcomes and the required failure
direction are explicit.

- [ ] Inventory every recurrence intended for expedition use and classify it as
  critical or ordinary.
- [ ] Record which recurrence families each critical task uses: CP, anchor,
  multiple times per day, omissions, file-backed dates, business calendars,
  astronomy, random selection, expiration, native `until`, `chainMax`, and
  `chainUntil`.
- [ ] Define the mission horizon and the maximum acceptable early, late,
  missing, and duplicate occurrence behavior for each critical task.
- [ ] Build a hazard matrix covering at least missed occurrence, duplicate
  occurrence, wrong local time, wrong date, stale configuration, missing
  resource, invalid clock, full storage, corrupt task data, corrupt outbox,
  unavailable Taskwarrior, broken runtime, false healthy status, and operator
  misuse.
- [ ] Choose the safe disposition for each hazard: reject the operation, retain
  durable retry work, require manual review, allow recorded hookless completion
  for reconcile, enter hooks-off recovery, or require manual fallback.
- [ ] Decide whether critical recurrence is selected by an explicit task field,
  tag, project, configuration allow-list, or an external expedition manifest.
- [ ] Define a versioned expedition manifest containing the approved critical
  tasks, recurrence families, configuration and resource identities, exact
  runtime versions, mission time bounds, backup policy, and health thresholds.
- [ ] Either qualify every feature used by critical tasks or make expedition
  validation reject that feature for critical use. Do not silently reduce or
  reinterpret an existing recurrence.
- [ ] Define whether a critical completion may proceed when lifecycle intent
  staging is unavailable. Make the availability-versus-fail-closed tradeoff
  explicit and test both the chosen path and recovery path.
- [ ] Ensure ordinary non-Nautical tasks remain usable when expedition
  scheduling is safely blocked.

Profile completion criteria:

- [ ] Every critical task has an explicit hazard policy and fallback action.
- [ ] One command can validate the current system against the exact expedition
  manifest without mutation.
- [ ] An unqualified recurrence feature, changed critical task, or changed
  scheduling input produces a blocking, actionable finding rather than an
  implicit fallback.

## 2. Build Atomic Backup And Bare-Device Restore

Provide both an exact recovery image and a portable salvage format. A backup is
not accepted until restoration has been demonstrated.

- [ ] Define one supported backup command or scripted workflow with versioned
  JSON results and stable exit codes.
- [ ] Require the backup destination to be outside Taskdata, the managed
  runtime, and every included resource directory; reject recursive or
  self-containing backup layouts.
- [ ] Acquire a bounded maintenance/backup lease or otherwise prove Taskwarrior
  and Nautical state is quiescent while the exact snapshot is taken.
- [ ] Include a complete hooks-off Taskwarrior export as the portable salvage
  representation.
- [ ] Include an exact Taskdata snapshot needed to preserve Taskwarrior history,
  UUIDs, configuration, and implementation-specific data.
- [ ] Back up the lifecycle outbox with the SQLite online-backup API or an
  equivalently WAL-safe quiescent procedure; never copy only the main database
  while ignoring live `-wal` state.
- [ ] Include Nautical configuration, `wrand_salt`, presets, business-calendar
  definitions, anchor files, omit files, astronomy profiles, and every external
  scheduling resource referenced by critical tasks.
- [ ] Include the active release, protected rollback release, expedition
  manifest, dependency lock, operator runbook, and fallback schedule.
- [ ] Write a canonical manifest containing sizes and SHA-256 digests for every
  artifact, with Unicode preserved.
- [ ] Make backup publication atomic: a failed or interrupted backup must not
  replace the most recent verified backup.
- [ ] Support bounded retention while protecting at least the newest verified
  backup and one older known-good generation.
- [ ] Store at least three verified copies across at least two media types, with
  one copy physically separated from the primary device.
- [ ] Make checksum verification possible without importing Nautical modules or
  opening Taskwarrior data.
- [ ] Define an explicit restore workflow that starts with an empty target and
  never overwrites an existing installation without apply authorization and a
  preserved pre-restore snapshot.
- [ ] After restore, run runtime verification, Taskwarrior export, outbox
  integrity, full critical-chain integrity, queue inspection, and reconcile
  dry-run before permitting mutation.
- [ ] Test restoration without the outbox and prove authoritative Taskwarrior
  evidence plus reconcile converges without duplicate children or lost links.
- [ ] Test portable salvage restoration from the Taskwarrior export when the
  exact Taskdata image is unavailable or unreadable, and document any history
  or metadata that cannot be reconstructed by that path.
- [ ] Test restoration with partially applied lifecycle work and prove the
  durable stage resumes without repeating completed mutations.

Backup and restore completion criteria:

- [ ] Restore the latest backup onto an empty spare device with all network
  interfaces disabled.
- [ ] Task counts, UUIDs, recurrence fields, critical-chain topology, config and
  resource fingerprints, and expected pending lifecycle work match the source
  evidence.
- [ ] Doctor, queue status, full critical integrity, and reconcile dry-run are
  clean or report only pre-recorded accepted findings.
- [ ] A deliberately interrupted backup leaves the prior verified generation
  usable.
- [ ] A non-developer can complete the documented restore drill without source
  code edits or unstated decisions.

## 3. Produce A Self-Contained Offline Runtime Kit

- [ ] Pin the exact Taskwarrior, Python, Rich, Prompt Toolkit, python-dateutil,
  Astral, tzdata, and other runtime versions qualified for each expedition
  device.
- [ ] Produce a dependency lock with hashes; do not rely on open-ended
  `>=` requirements for disaster recovery.
- [ ] Carry verified installers, packages, wheels, or binaries for every target
  operating system and CPU architecture.
- [ ] Record and carry required operating-system packages, native shared
  libraries, certificates, timezone files, and executable permissions needed
  by Taskwarrior and Python on each target device.
- [ ] Include the complete Nautical source/release needed by the managed
  installer and at least two complete managed releases.
- [ ] Make clean installation possible without Git, `curl`, DNS, package
  indexes, or remote repositories.
- [ ] Extend deep installation verification to recompute the active release
  digest and compare it with the manifest instead of merely displaying the
  stored digest.
- [ ] Verify the Taskwarrior executable identity and exact supported version,
  not only that some `task` command can execute.
- [ ] Verify the Python executable and imported dependency versions against the
  expedition manifest.
- [ ] Record the tzdata source/version or a stable digest of the zone files used
  by critical schedules.
- [ ] Add a supported dry-run-first rollback operation that validates the target
  release, outbox schema compatibility, hook APIs, wrappers, and dependencies
  before atomically selecting it.
- [ ] Protect the designated expedition rollback release from ordinary runtime
  cleanup.
- [ ] Ensure interrupted install and rollback operations preserve one complete
  active runtime and do not mix wrappers or modules from different releases.
- [ ] Carry a standalone checksum verifier and printed expected root digests for
  both software media and backup media.

Offline kit completion criteria:

- [ ] Install onto a bare spare device with networking disabled.
- [ ] Run installed-layout hook, schema, dependency, Unicode JSON, Doctor,
  query, queue, Navigator, and reconcile smoke checks successfully.
- [ ] Roll back to the protected previous release and then restore the selected
  expedition release without modifying Taskwarrior task data.
- [ ] Detect a deliberately changed runtime file, dependency version, manifest,
  and damaged installation artifact before hooks are re-enabled.

## 4. Exercise Real Storage, SQLite, Process, And Power Failures

Keep deterministic unit-level fault seams, but also exercise actual filesystem
and hardware behavior. A mocked `OSError("disk full")` is not evidence that
SQLite, Taskwarrior, and the filesystem converge after real exhaustion.

- [ ] Add focused injection seams for directory creation, file open, write,
  flush, `fsync`, rename, permission change, SQLite begin, commit, checkpoint,
  close, and Taskwarrior process execution where an existing typed boundary can
  own them without broad abstraction.
- [ ] Test full storage before outbox creation, during WAL growth, during stage
  commit, during acknowledgement, during backup publication, and during
  diagnostic rotation.
- [ ] Run a real ENOSPC campaign on a small disposable filesystem or quota,
  covering both Taskwarrior and Nautical state.
- [ ] Test read-only Taskdata, state, config, resource, cache, backup, and runtime
  paths independently.
- [ ] Test permission loss and unavailable directories without running as the
  live user or changing live paths.
- [ ] Corrupt and truncate disposable outbox main, WAL, and SHM files in a
  matrix; confirm status and Doctor never report a corrupt database as healthy.
- [ ] Test a future outbox schema, malformed rows, fingerprint mismatch, and
  partially written resources as explicit unavailable/manual-review states.
- [ ] Kill the owning process before and after each durable lifecycle boundary:
  plan staging, exact claim, child import, child verification, parent mutation,
  parent verification, final stage, and acknowledgement.
- [ ] Kill reconcile between bulk staging and exact claiming and during each
  Taskwarrior mutation phase.
- [ ] Test Taskwarrior missing binary, timeout, lock contention, nonzero exit,
  malformed JSON, empty output, noisy stderr, and successful empty output at
  every relevant read or mutation boundary.
- [ ] On a spare device, perform controlled power removal during outbox staging,
  child creation, parent linking, backup, restore, install, and rollback.
- [ ] After every fault, reboot/restart, run SQLite integrity checks, inspect the
  queue, run scoped integrity and reconcile dry-run, then apply only an
  explicitly reviewed recovery plan.
- [ ] Verify hook output remains protocol-safe under every injected failure.
- [ ] Verify cache and diagnostic failures cannot change scheduling or lifecycle
  decisions.

Fault-campaign completion criteria:

- [ ] No tested failure produces a duplicate deterministic child, false parent
  link, false acknowledgement, lost durable safe plan, or false healthy status.
- [ ] Every outcome is applied and verified, durably retryable, manual review,
  quarantined, or explicitly unavailable with actionable evidence.
- [ ] Loss of non-authoritative caches, locks, diagnostics, or the local outbox
  does not prevent deterministic recovery from authoritative task evidence.
- [ ] An interrupted backup, restore, installation, or rollback always leaves a
  previously verified recovery path intact.

## 5. Harden Clock, Timezone, And Calendar Provenance

- [ ] Define acceptable wall-clock accuracy and jump thresholds for the mission
  and each critical task class.
- [ ] Persist a bounded last-known-valid UTC observation or explicit mission
  not-before/not-after bounds without putting a filesystem write on ordinary
  non-Nautical task paths.
- [ ] Detect RTC reset, time earlier than trusted durable evidence, and
  implausible backward or forward wall-clock jumps.
- [ ] Define how an acknowledged clock correction is recorded and how blocked
  lifecycle work is revalidated afterward.
- [ ] Keep one clock sample per invocation for scheduling decisions and use
  monotonic time for in-process budgets and elapsed durations.
- [ ] Audit durable lease and retention behavior under backward and forward
  wall-clock jumps; ensure a jump cannot grant two effective mutation owners or
  silently prune recent evidence.
- [ ] Include timezone name, tzdata identity, astronomy-library version,
  astronomy profile, business-calendar fingerprint, file-resource content
  fingerprints, and randomization salt in expedition provenance.
- [ ] Make devices sharing a chain reject incompatible time/calendar provenance
  rather than silently converging on different schedules.
- [ ] Test every critical recurrence across DST gaps and folds, leap days,
  month/year boundaries, timezone-rule boundaries, polar astronomy edge cases,
  and the mission start/end dates.
- [ ] Test boot with an invalid RTC, correction before Taskwarrior use,
  correction after work was staged, and reboot with a valid persisted clock
  anchor.
- [ ] Document an offline time-verification procedure using an independent
  trusted clock source.

Time completion criteria:

- [ ] An invalid or suspicious clock blocks critical scheduling with an
  actionable finding and does not silently use UTC or another fallback zone.
- [ ] Ordinary Taskwarrior access and the documented recovery path remain
  available when critical scheduling is blocked.
- [ ] Identical expedition manifests on the supported devices produce identical
  critical occurrence instants across the mission horizon.
- [ ] Changing tzdata, Astral, timezone, calendar/resource content, or
  `wrand_salt` invalidates affected evidence before mutation.

## 6. Add One Deep Expedition Health Check

Prefer extending the existing Doctor/control-plane ownership rather than
creating an independent health interpretation.

- [ ] Define `nautical doctor --expedition MANIFEST` or one equivalent
  read-only command with versioned JSON and stable text presentation.
- [ ] Reuse existing typed configuration, Taskwarrior, outbox, integrity,
  runtime, and operator findings rather than reimplementing their policy.
- [ ] Report free bytes and inodes for Taskdata, outbox, runtime, cache,
  resource, backup, and temporary paths with mission-appropriate warning and
  critical thresholds.
- [ ] Check directory identity, mount availability, permissions, and read-only
  state without weakening Doctor's default read-only contract.
- [ ] If an actual write probe is useful, make it a separate explicitly
  authorized scratch-path operation that cannot touch task or lifecycle data.
- [ ] Run `PRAGMA quick_check` through the supported outbox repository and
  distinguish absent, busy, incompatible, corrupt, and clean states.
- [ ] Verify Taskwarrior availability, exact version/binary identity, bounded
  export health, and critical-chain evidence completeness.
- [ ] Recompute the active managed-release digest and verify wrappers, hook
  implementations, operator files, manifest, and protected rollback release.
- [ ] Verify Python, dependency, tzdata, astronomy, configuration, calendar,
  resource, and expedition-manifest identities.
- [ ] Verify backup presence, age, manifest integrity, and the last successful
  restore-drill record without treating an unreadable backup as absent.
- [ ] Report current outbox states, stale claims, attempts, retention,
  quarantined/manual-review rows, WAL size, and integrity.
- [ ] Run a complete critical-chain integrity audit within declared task/chain
  limits and report limit exhaustion as incomplete, never healthy.
- [ ] Show clock sanity, mission time bounds, last valid clock anchor, and any
  detected jump.
- [ ] Provide one concrete local command or manual action for every blocking or
  actionable finding.
- [ ] Return nonzero for blocking, unavailable, corrupt, stale-backup,
  suspicious-clock, runtime-mismatch, or incomplete-critical-audit states.
- [ ] Keep the ordinary healthy output compact; retain full evidence in JSON.

Health-check completion criteria:

- [ ] The check is read-only by construction and cannot invoke Taskwarrior or
  Nautical mutation owners.
- [ ] A clean result proves every expedition-manifest requirement was checked,
  not merely that no exception was raised.
- [ ] Deliberate corruption or unavailability in each checked component changes
  the result to a non-healthy state with actionable evidence.
- [ ] The command runs successfully from the managed installed layout on every
  expedition device with networking disabled.

## 7. Qualify Long-Horizon Semantics And Endurance

- [ ] Define the mission horizon plus an explicit safety margin for schedule
  projection and backup/fallback generation.
- [ ] Build a retained fixture for every critical recurrence using its exact
  timezone, calendar, files, astronomy profile, randomization salt, limits, and
  omission behavior.
- [ ] Compare the production scheduler with a deliberately simple independent
  oracle for the critical subset; do not use a second wrapper around the same
  implementation as the sole oracle.
- [ ] Add property or metamorphic tests for strict monotonicity, deterministic
  replay, no occurrence before the cursor, omission behavior, deduplication,
  chain-limit termination, and local/UTC round trips.
- [ ] Sweep every critical recurrence across the full mission horizon and
  relevant DST, leap-year, month/year, seasonal, and astronomy boundaries.
- [ ] Test the maximum intended Taskwarrior history, active-task count, chain
  length, number of critical chains, outbox backlog, file-resource size, and
  backup size.
- [ ] Measure cache, diagnostics, locks, outbox/WAL, acknowledged retention,
  Taskwarrior data, and backup growth; define warning and critical storage
  budgets.
- [ ] Run at least a 48-72-hour installed-device soak on each supported device
  with ordinary use, recurrence churn, health checks, backup cycles, restore
  verification, controlled restarts, and storage-pressure phases.
- [ ] Run repeated boot, suspend/resume, battery-low, process-kill, and clock
  correction cycles on the actual expedition hardware.
- [ ] Preserve all failure, latency, call-count, exported-row, memory, SQLite,
  scheduler-iteration, and storage-growth evidence.
- [ ] Rerun the complete unit, normal/shuffled golden, type, compilation,
  deployment, black-box, failure-injection, stress, and performance matrices
  after the final endurance candidate is frozen.

Endurance completion criteria:

- [ ] No critical occurrence differs from the independent expected schedule.
- [ ] No unexplained memory, file, WAL, outbox, log, cache, task-history, or
  latency growth appears during the installed-device soak.
- [ ] Every controlled restart and fault converges without duplicate mutation,
  lost chain progression, false health, or unresolved lifecycle work.
- [ ] Device-specific latency variance may be recorded, but no correctness,
  durability, health, storage, or recovery threshold is waived as performance
  variance.

## 8. Produce An Independent Fallback Schedule And Operator Runbook

- [ ] Generate a deterministic plain-text and CSV schedule for every critical
  occurrence across the mission horizon where the recurrence semantics permit
  precomputation.
- [ ] For outcome-dependent recurrence that cannot be fully precomputed, record
  the decision rule, latest safe action, and manual continuation procedure.
- [ ] Include task identity, description, local date/time, UTC instant,
  timezone, recurrence source, generation time, manifest identity, and
  configuration/resource digests.
- [ ] Hash the fallback files and print the expected digest or short verification
  code on the paper copy.
- [ ] Store the fallback schedule on the primary device, spare device,
  independent removable media, and paper.
- [ ] Regenerate and review the schedule whenever a critical task, config,
  timezone, calendar, resource, dependency, or mission bound changes.
- [ ] Add a concise decision-tree runbook for suspicious time, hook failure,
  Taskwarrior failure, full storage, corrupt outbox, corrupt Taskdata, missing
  resource, broken runtime, failed health check, backup restore, rollback,
  hooks-off operation, reconcile, and manual recurrence continuation.
- [ ] State explicit stop conditions: when not to apply reconcile, when not to
  force a successor, when to preserve evidence, and when to move to the manual
  schedule.
- [ ] Include exact local commands that do not require `jq`, internet access,
  shell history, repository knowledge, or remembered paths.
- [ ] Create a one-page printed quick-reference containing daily health checks,
  backup cadence, clock verification, hooks-off break glass, restore, rollback,
  and spare-device activation.
- [ ] Rehearse the runbook with networking disabled and without consulting
  source code or developer notes.
- [ ] Establish a periodic verification reminder independent of Nautical so a
  Nautical failure cannot suppress its own health check.

Fallback completion criteria:

- [ ] A user can identify and perform the next critical action with the primary
  device completely unavailable.
- [ ] The printed and digital fallback schedules match the frozen expedition
  manifest and independent oracle.
- [ ] Every recovery command and decision point in the runbook has been
  rehearsed successfully on the target hardware.

## 9. Prepare Physical Redundancy

These items are operational rather than Nautical code, but they are mandatory
for the stated reliability objective.

- [ ] Prepare a fully installed and verified spare device with the same
  expedition manifest and compatible Taskwarrior/Nautical versions.
- [ ] Keep verified backups and the offline runtime kit on independent storage,
  not only on the primary device's filesystem.
- [ ] Provide independent charging/power, cables, adapters, and protected
  storage appropriate to the expedition environment.
- [ ] Verify the spare device battery, clock retention, storage health, display,
  input, and boot process before departure.
- [ ] Define how task changes move between devices without assuming an internet
  sync service; prevent both devices from mutating the same chain from divergent
  snapshots.
- [ ] Rehearse primary-device loss, spare activation, backup restoration,
  integrity inspection, and controlled resumption.
- [ ] Keep the printed runbook and critical schedule separately from both
  devices.

Physical-readiness completion criteria:

- [ ] Failure or loss of any one device, storage copy, charger, or software
  release does not remove both the schedule and the recovery path.
- [ ] Spare activation requires no download, package resolution, source edit,
  or undocumented decision.

## 10. Final Installed-Device Qualification And Atomic Cutover

Do not wire or install the hardened branch into live Taskwarrior until Sections
0-9 pass. Final qualification must use the exact expedition source, binaries,
dependencies, configuration, resources, devices, backup media, and mission
manifest.

- [ ] Freeze the candidate revision and regenerate the offline kit, dependency
  lock, expedition manifest, fallback schedule, backup, and checksums from that
  exact revision.
- [ ] Run the complete correctness, type, deployment, process, black-box,
  failure, performance, long-horizon, endurance, backup, restore, and rollback
  gates against the frozen candidate.
- [ ] Build a disposable installed layout and prove install, health, hooks,
  Taskwarrior integration, backup, restore, and rollback with networking
  disabled.
- [ ] On each actual expedition device, use disposable Taskdata with the real
  installed Taskwarrior binary, managed Nautical runtime, filesystem, timezone,
  and dependencies.
- [ ] Create one controlled recurring task, complete it, verify durable staging,
  drain/application, deterministic child creation, parent link, child link,
  outbox acknowledgement, and authoritative postconditions.
- [ ] Interrupt a second controlled transition after a durable boundary and
  prove restart/reconcile resumes without duplicate mutation.
- [ ] Roll the installed runtime back to the protected previous release,
  rerun read-only health/integrity checks, then restore the frozen expedition
  release and verify the same disposable task state.
- [ ] Restore the final backup onto the spare device and compare authoritative
  Taskwarrior, critical-chain, resource, runtime, and fallback-schedule evidence.
- [ ] Stop hooks and operator processes before live installation.
- [ ] Record and review active lifecycle intents, leases, manual-review and
  quarantined rows, chain findings, storage health, clock health, backup health,
  and rollback availability before cutover.
- [ ] Install the frozen managed release atomically while the system remains
  stopped.
- [ ] Run live read-only expedition health, Doctor, queue status/review, scoped
  and full critical integrity, query, Navigator, and reconcile dry-run before
  any live apply.
- [ ] Apply only a real, reviewed live candidate when one safely exists. If no
  candidate exists, record that fact without claiming live mutation evidence;
  retain the installed-device disposable mutation drill as the qualification
  proof.
- [ ] Verify every applied mutation's authoritative Taskwarrior and outbox
  postconditions before normal use resumes.
- [ ] Re-enable hooks only after the active and rollback releases, current
  backup, restore evidence, clock, runtime digest, dependencies, resources,
  outbox, and critical chains all pass.
- [ ] Run one final network-disabled expedition health check and compare it with
  the frozen expected manifest.

Final go/no-go criteria:

- [ ] No unresolved error, manual-review, quarantine, suspicious-clock,
  incomplete-audit, corrupt-state, stale-backup, runtime-drift, dependency-drift,
  or resource-drift finding affects a critical chain.
- [ ] Backup restoration, bare-device installation, protected rollback,
  forward restoration, real storage faults, process interruption, and spare
  activation have all been exercised on target hardware.
- [ ] Critical occurrences match the independent fallback schedule across the
  mission horizon.
- [ ] The primary device, spare device, offline kit, backup media, runbook, and
  paper schedule are mutually consistent and checksum-verified.
- [ ] Remaining accepted variances are documented with their safety impact and
  do not weaken critical correctness, durability, diagnosis, or recovery.
- [ ] A named reviewer records a final go/no-go decision only after examining
  the retained evidence below.

## Completion Record

- Starting revision:
- Hardening branch/worktree:
- Frozen candidate revision:
- Nautical release:
- Protected rollback release:
- Taskwarrior binary/version/digest:
- Python binary/version/digest:
- Dependency lock/digest:
- tzdata identity:
- Device models and operating systems:
- Filesystem and available-capacity baseline:
- Expedition manifest/digest:
- Configuration/resource digest set:
- Unit result:
- Golden normal result:
- Golden shuffled result:
- Mypy result:
- Compilation/deployment result:
- Black-box/process result:
- Storage/process/power fault result:
- Clock/timezone campaign result:
- Long-horizon/oracle result:
- Installed-device soak result:
- Backup artifact/digest:
- Backup interruption result:
- Bare-device restore result:
- Outbox-loss recovery result:
- Offline clean-install result:
- Controlled mutation/postcondition result:
- Controlled interruption/reconcile result:
- Rollback/forward-restore result:
- Spare-device activation result:
- Fallback schedule/digest:
- Printed runbook revision:
- Final expedition health report:
- Accepted variances:
- Reviewer/date:
- Go/no-go decision:
