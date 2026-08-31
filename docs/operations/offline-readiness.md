# Offline Readiness

Nautical can be installed and repaired without network access when a local
offline kit is prepared in advance. A kit contains the exact runtime source,
configuration and UDA inputs, documentation, a dependency inventory, and
checksums.

## Build And Verify A Kit

From a Nautical checkout, choose a destination outside the checkout:

```bash
python3 dev_tools/nautical_offline_kit.py build /path/to/nautical-kit
python3 dev_tools/nautical_offline_kit.py verify /path/to/nautical-kit
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

Use a disposable `TASKDATA` directory for repair drills. The kit does not
include user Taskwarrior data, lifecycle state, or private resource files;
back those up separately before an offline recovery operation.

## Snapshot Taskdata Before Recovery

For a verified Nautical backup generation, use the local backup command. It
captures the hooks-off export and the lifecycle outbox into a new directory:

```bash
python3 nautical_core/tools/nautical_backup.py --taskdata "$HOME/.task" \
  --destination "$HOME/nautical-backup-$(date +%Y%m%d-%H%M%S)" \
  --task-bin "$(command -v task)" --json
```

The destination must be outside Taskdata and must not already exist.
Additional files are included only when explicitly selected with repeatable
`--include NAME=PATH` options; Nautical does not scan configuration or resource
directories implicitly:

```bash
python3 nautical_core/tools/nautical_backup.py --taskdata "$HOME/.task" \
  --destination "$HOME/nautical-backup-$(date +%Y%m%d-%H%M%S)" \
  --include calendar="$HOME/.local/share/nautical/calendar.json" --json
```

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
nautical query integrity --all --json
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
nautical query integrity --all --json
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
