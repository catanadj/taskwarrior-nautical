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
