# Offline Readiness

Nautical can be installed and repaired without network access when a local
offline kit is prepared in advance. A kit contains the exact runtime source,
configuration and UDA inputs, documentation, a dependency inventory, and
checksums.

## Build And Verify A Kit

From a Nautical checkout, choose a destination outside the checkout:

```bash
python3 dev_tools/nautical_offline_kit.py build /path/to/nautical-kit \
  --archive /path/to/nautical-kit.tar.gz
python3 dev_tools/nautical_offline_kit.py verify /path/to/nautical-kit
python3 dev_tools/nautical_offline_kit.py verify /path/to/nautical-kit.tar.gz
```

The builder does not contact Git, download packages, or modify Taskwarrior.
It refuses to overwrite an existing destination. Verify the kit with the
ordinary Python interpreter before taking the device offline.

## Local Install

After copying the verified kit to a device, verify it again and install from
its root:

```bash
python3 dev_tools/nautical_offline_kit.py verify /path/to/nautical-kit
./nautical install --source /path/to/nautical-kit --dry-run
./nautical install --source /path/to/nautical-kit
nautical doctor --installation-only
```

Prefer transferring the archive so hidden files and runtime modules cannot be
dropped by a file manager. On the target, verify the archive, extract it, then
verify the extracted directory once more:

```bash
python3 dev_tools/nautical_offline_kit.py verify /path/to/nautical-kit.tar.gz
mkdir /path/to/nautical-kit
tar -xzf /path/to/nautical-kit.tar.gz -C /path/to/nautical-kit
python3 dev_tools/nautical_offline_kit.py verify /path/to/nautical-kit
```

Use a disposable `TASKDATA` directory for repair drills. The kit does not
include user Taskwarrior data, lifecycle state, or private resource files;
back those up separately before an offline recovery operation.

## Snapshot Taskdata Before Recovery

For a verified Nautical backup generation, use the local backup command. It
captures the hooks-off export and the lifecycle outbox into a new directory:

Run this only against an initialized installation that has a lifecycle outbox;
an installer `--dry-run` target is intentionally not backup-ready.

```bash
python3 nautical_core/tools/nautical_backup.py --taskdata "$HOME/.task" \
  --destination "$HOME/nautical-backup-$(date +%Y%m%d-%H%M%S)" \
  --task-bin "$(command -v task)" --json
```

The destination must be outside Taskdata and must not already exist.
The backup automatically includes the active Nautical configuration, UDA
definition, Taskwarrior rc file, and every configured or task-referenced
anchor/omit resource. These are copied under `resources/` and checksummed in
the manifest. Additional files can still be selected with repeatable
`--include NAME=PATH` options:

```bash
python3 nautical_core/tools/nautical_backup.py --taskdata "$HOME/.task" \
  --destination "$HOME/nautical-backup-$(date +%Y%m%d-%H%M%S)" \
  --include calendar="$HOME/.local/share/nautical/calendar.json" --json
```

The manifest also records the versions of Nautical's Python runtime
distributions (`astral`, `rich`, `prompt_toolkit`, and `python-dateutil`).
Dependency binaries are intentionally outside the backup contract; install
the requirements separately when rebuilding a different device.

For an offline recovery kit, build and verify one kit on each qualification
target rather than copying a kit between unlike systems. The kit manifest
records platform, CPU architecture, Python executable/version, Taskwarrior
version, timezone, and Python package versions. Name the outputs by target so
the correct inventory is unambiguous:

```bash
python3 dev_tools/nautical_offline_kit.py build \
  "$HOME/nautical-kit-linux-$(uname -m)" \
  --archive "$HOME/nautical-kit-linux-$(uname -m).tar.gz"
python3 dev_tools/nautical_offline_kit.py verify \
  "$HOME/nautical-kit-linux-$(uname -m).tar.gz"
```

Run the same commands locally on the Termux device, using its own writable
storage and Python interpreter. A kit whose manifest does not match the
target's platform, architecture, interpreter, or Taskwarrior version must be
rejected or rebuilt for that target; the shared source may remain identical.

The backup also includes the active managed release and hook layout. Restore
stages these as `.nautical-runtime/` and `hooks/` in the disposable target and
recreates the managed `current` release pointer. The launcher is deliberately
not installed into the host PATH; use the local installer as the explicit
cutover step after validating the staged target.

## Periodic Backups

The manual backup command above is the supported workflow and does not depend
on a scheduler. If periodic copies are useful, place this small wrapper in a
writable local directory and adjust `BACKUP_ROOT`:

```bash
#!/usr/bin/env bash
set -euo pipefail
umask 077
TASKDATA="${TASKDATA:-$HOME/.task}"
BACKUP_ROOT="${BACKUP_ROOT:-$HOME/nautical-backups}"
LOCK="$BACKUP_ROOT/.backup.lock"
mkdir -p "$BACKUP_ROOT"
if ! mkdir "$LOCK" 2>/dev/null; then
  exit 0
fi
trap 'rmdir "$LOCK"' EXIT
stamp=$(date +%Y%m%d-%H%M%S)
nautical backup --taskdata "$TASKDATA" \
  --destination "$BACKUP_ROOT/$stamp" \
  --task-bin "$(command -v task)" --keep 2 --prune --json
```

On ordinary Linux, a daily cron entry can invoke the wrapper (use its absolute
path):

```cron
17 3 * * * /home/user/bin/nautical-periodic-backup.sh >>/home/user/.local/state/nautical-backup.log 2>&1
```

On Termux, the same wrapper can be scheduled when
`termux-job-scheduler` is installed:

```bash
termux-job-scheduler --script "$HOME/bin/nautical-periodic-backup.sh" \
  --period-ms 86400000 --persisted true
```

Keep the backup root on writable storage. The wrapper skips overlapping runs,
keeps the newest two verified generations, and returns the backup command's
failure status for scheduler logs; a manual invocation remains equivalent.

There are two different backups, and they serve different purposes. For a
portable Taskwarrior copy, first stop other Taskwarrior/Nautical processes and
export with hooks disabled:

```bash
TASKDATA="$HOME/.task"
task rc.hooks=off rc.verbose=nothing export > taskwarrior-export.json
```

This export preserves task records and UUIDs, but not Taskwarrior's local
history or Nautical's lifecycle outbox. It is the safer choice for moving
tasks to another installation.

For implementation-specific recovery, take an exact filesystem snapshot only
after all Taskwarrior and Nautical processes have stopped. This preserves the
Taskwarrior database/history and Nautical state, including the outbox WAL:

```bash
TASKDATA="$HOME/.task"
SNAPSHOT="$HOME/nautical-taskdata-snapshot-$(date +%Y%m%d-%H%M%S).tar"
tar -C "$TASKDATA" -cf "$SNAPSHOT" .
sha256sum "$SNAPSHOT" > "$SNAPSHOT.sha256"
```

Do not make an exact copy while a process may be writing Taskwarrior or SQLite
files; an interrupted or live copy is not a consistent backup. Restore an
exact snapshot only into an empty disposable directory first, verify its
checksum, and run `nautical doctor --installation-only`, `nautical queue-status
--json`, and a reconcile dry-run before considering any live replacement. Never
extract it over an existing Taskdata directory.

## Recovery Order

Use read-only inspection before any mutation:

```bash
nautical doctor --deep --json
nautical queue-status --json
nautical query integrity --all
nautical reconcile --dry-run --json --no-housekeeping
```

Review the scoped plan, then authorize only the intended repair:

```bash
nautical reconcile --chain-id CHAIN_ID --apply --json
nautical doctor --deep --json
```

Use `nautical queue-review --all` for manual-review intents. Resolve an intent
only when its evidence proves the child is already present or the operation is
otherwise safe to retry. Do not delete the outbox as a repair; it is the durable
record used for idempotent recovery.

## Failure Actions

- **Full or read-only storage:** stop mutation, free or remount storage, then
  rerun Doctor. Do not retry repeatedly while space is exhausted.
- **Suspicious time or timezone:** stop mutation, correct the system/configured
  timezone, and rerun Doctor before scheduling or reconcile.
- **Missing or corrupt resource:** restore the named resource from a verified
  copy. Do not substitute a different calendar, preset, or astronomy source.
- **Corrupt or incompatible outbox:** preserve Taskdata and state, take a
  verified backup, then restore a known-good generation or seek manual review.
- **Broken runtime or missing dependency:** repair from the verified local kit;
  do not download while offline. Re-run installation-only Doctor afterward.
- **Repeated timeout or locked Taskwarrior:** stop concurrent processes and
  retry once. If it repeats, keep hooks disabled and use backup/restore.
- **Malformed, truncated, or unexpectedly large Taskwarrior output:** stop
  mutation and retain the raw diagnostic; repair the Taskwarrior/runtime
  boundary before retrying. Never treat an undecodable export as empty data.
- **Interrupted install, backup, restore, queue drain, or reconcile:** rerun
  the corresponding read-only verification first, then resume through queue
  status/reconcile. Do not manually replay a child or parent link.
- **Duplicate-child or idempotency concern:** stop mutation, inspect the chain
  and deterministic intent with `nautical query integrity` and
  `nautical queue-review`; apply only a reviewed plan.
- **Hook protocol or diagnostic failure:** preserve stdout as the protocol
  record, enable `NAUTICAL_DIAG=1` only for stderr diagnostics, and keep hooks
  disabled until a strict-JSON smoke test passes.

## Hooks-Off Break Glass

For emergency Taskwarrior access only:

```bash
task rc.hooks=off <command>
```

Recurrence is not maintained while hooks are disabled. After resolving the
issue, run Doctor, queue status, a scoped integrity query, and a reconcile dry
run before re-enabling normal hooks.

## Restore And Rollback

Validate a generation, then stage it into a new empty directory:

```bash
python3 nautical_core/tools/nautical_restore.py --source BACKUP --json
python3 nautical_core/tools/nautical_restore.py --source BACKUP \
  --target "$HOME/.task-restore-check" --apply --json
```

Run installation-only Doctor, queue status, integrity, and reconcile dry-run
against the disposable target. For a managed runtime rollback, stop hooks and
operator processes first, select a retained release with the local installer,
verify it, and repeat those checks. Reverse a rollback by selecting the
previously verified release; never copy files into the active runtime.

A missing or corrupt lifecycle outbox makes the restore unavailable; it does
not prove that no recovery work exists. Restore a verified generation (or
repair the outbox), then run reconcile dry-run and review its result before
authorizing any mutation.

### Selecting A Retained Runtime Release

List retained releases first and choose a release directory under
`.nautical-runtime/releases/`. Always perform a dry run before switching:

```bash
TASKDATA="$HOME/.task"
RELEASE_ID=r-previous
SOURCE="$TASKDATA/.nautical-runtime/releases/$RELEASE_ID"
nautical install --source "$SOURCE" --taskdata "$TASKDATA" \
  --release-id "$RELEASE_ID" --dry-run --json
```

Inspect `previous_release`, `release_id`, `content_sha256`, and the planned
operation. If they identify the intended retained release, repeat the same
command without `--dry-run`:

```bash
nautical install --source "$SOURCE" --taskdata "$TASKDATA" \
  --release-id "$RELEASE_ID" --json
```

The installer validates the retained tree, publishes its managed pointer and
wrappers atomically, and leaves the previously active release available for
another rollback. Do not copy individual modules or hooks between releases.

## Authority And Departure Check

Taskwarrior records and export are authoritative for tasks. The lifecycle
outbox is authoritative for pending recovery work. Runtime manifests,
configuration, UDA definitions, and explicitly included resources are verified
inputs. Caches and diagnostics are reproducible; deleting them is not a data
repair.

Before leaving a device offline:

```bash
nautical doctor --installation-only --json
nautical queue-status --json
nautical query integrity --all
python3 nautical_core/tools/nautical_backup.py --taskdata "$HOME/.task" \
  --destination "$HOME/nautical-backup-$(date +%Y%m%d-%H%M%S)" \
  --task-bin "$(command -v task)" --json
```

All commands above are local-only and require no browser, `jq`, source edit, or
network access.

On Linux use the normal `python3`, `task`, and `sha256sum` commands. On Termux,
use the same commands from the active Python environment and use the absolute
Taskwarrior path if `command -v task` is empty; keep the kit and backup on a
writable Termux filesystem rather than shared read-only storage.
