# Backup and Restore

Nautical backups are verified, local backup generations containing Taskwarrior
data, lifecycle state, the active managed runtime, configuration, UDA
definitions, and referenced calendar resources. Use them for a complete local
recovery or to stage a second device. For the broader offline-kit and
device-transfer procedure, see [Offline Readiness](../operations/offline-readiness.md).

## Create a backup

Choose a new destination outside `TASKDATA`. The destination must not already
exist:

```bash
nautical backup \
  --taskdata "$HOME/.task" \
  --destination "$HOME/nautical-backup-$(date +%Y%m%d-%H%M%S)" \
  --task-bin "$(command -v task)" \
  --json
```

The command validates the Taskwarrior export, copies the lifecycle outbox,
records runtime and platform provenance, and writes checksums to
`manifest.json`. It also includes the active configuration, UDA definitions,
Taskwarrior rc file, and configured or task-referenced anchor/omission
resources. Add explicitly selected regular files with repeatable options:

```bash
nautical backup \
  --taskdata "$HOME/.task" \
  --destination "$HOME/nautical-backup-calendar" \
  --include calendar="$HOME/.local/share/nautical/calendar.json" \
  --json
```

Use `--keep N --prune` when maintaining a backup directory of generations.
Pruning happens only after a new verified generation succeeds. Dependency
binaries are not embedded; rebuild the matching runtime from the offline kit
or install the bounded compatibility requirements separately.

## Validate a generation

Validation is read-only and should be performed after copying a backup to
another device:

```bash
nautical restore --source "$HOME/nautical-backup-20260917-120000" --json
```

Validation rejects missing or unlisted managed files, checksum changes,
symlinks, and source changes during inspection. A valid generation must include
both `taskwarrior-export.json` and `lifecycle-outbox.db`; a missing or corrupt
outbox is not treated as an empty queue.

## Stage a disposable restore

Never overwrite a live `TASKDATA` directory as the first restore step. Stage
into a new or empty directory only after validation:

```bash
nautical restore \
  --source "$HOME/nautical-backup-20260917-120000" \
  --target "$HOME/.task-restore-check" \
  --apply \
  --json
```

The restore publishes the staged Taskwarrior export, lifecycle outbox,
configuration, resources, hooks, and managed runtime together. It does not
install a launcher into the host `PATH`. After staging, run the read-only
checks against the disposable data directory before any cutover:

```bash
TASKDATA="$HOME/.task-restore-check" nautical doctor --installation-only
TASKDATA="$HOME/.task-restore-check" nautical queue-status --json
TASKDATA="$HOME/.task-restore-check" nautical query integrity --all
TASKDATA="$HOME/.task-restore-check" nautical reconcile --dry-run --json --no-housekeeping
```

Review the reconcile plan before applying any recovery mutation. Keep the
original backup generation unchanged until the restored data has passed these
checks.

## Portable task copy versus full recovery

A hooks-off Taskwarrior export is appropriate when moving task records to an
existing, independently installed Nautical device:

```bash
TASKDATA="$HOME/.task" \
  task rc.hooks=off rc.verbose=nothing export > taskwarrior-export.json
```

That export preserves task UUIDs but excludes local Taskwarrior history and
Nautical's lifecycle outbox. Use a verified Nautical backup generation when
you need lifecycle recovery evidence, runtime provenance, and referenced
resources as well. Do not copy individual runtime modules or hooks between
releases; restore and validate the complete generation instead.

For periodic copies, use the documented wrapper and retention workflow in
[Offline Readiness](../operations/offline-readiness.md#periodic-backups).
