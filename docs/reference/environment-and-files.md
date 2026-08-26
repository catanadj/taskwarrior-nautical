# Environment and Files

## Environment variables

| Variable | Purpose |
| --- | --- |
| `TASKDATA` | Explicit Taskwarrior data directory |
| `TASKRC` | Taskwarrior configuration path used during config discovery |
| `NAUTICAL_CONFIG` | Explicit Nautical TOML file |
| `NAUTICAL_TRUST_CONFIG_PATH=1` | Allow an otherwise rejected explicit config path after user review |
| `NAUTICAL_DIAG=1` | Current-command diagnostics on stderr |
| `NAUTICAL_DIAG_LOG=1` | Persist structured diagnostic JSONL |
| `NAUTICAL_DIAG_LOG_MAX_BYTES` | Diagnostic log rotation threshold |
| `NAUTICAL_PROFILE=1` | Lightweight timing on stderr |
| `NAUTICAL_EXIT_STRICT=1` | Nonzero on-exit status for drain errors/review |
| `NAUTICAL_DNF_DISK_CACHE=0` | Disable on-add parsed-expression disk cache |
| `NAUTICAL_CLEAR_CACHES=1` | Clear in-process parser caches after parsing |
| `NAUTICAL_SOURCE` | Checkout hint for first-install launcher recovery |
| `NAUTICAL_CORE_PATH` | Development/test core override; do not set normally |

Hooks also understand Taskwarrior Hooks v2 data-location arguments.

## Managed installation

| Path | Purpose |
| --- | --- |
| `TASKDATA/.nautical-runtime/` | Content-addressed releases and active manifest |
| `TASKDATA/hooks/on-add.nautical` | Add hook |
| `TASKDATA/hooks/on-modify.nautical` | Modify hook |
| `TASKDATA/hooks/on-exit.nautical` | Exit/drain hook |
| `~/.local/bin/nautical` | Default Linux command launcher |
| `$PREFIX/bin/nautical` | Default Termux command launcher |

## Runtime state

| Path | Purpose |
| --- | --- |
| `TASKDATA/.nautical-state/.nautical_lifecycle_outbox.db` | Durable lifecycle plans |
| database `-wal` and `-shm` sidecars | SQLite WAL state |
| `TASKDATA/.nautical-locks/.nautical_parent_nextlink.<uuid>.lock` | Parent mutation serialization |
| `TASKDATA/.nautical-locks/.nautical_reconcile.lock` | Reconcile application lock |
| `TASKDATA/.nautical_diag.jsonl` | Optional structured diagnostics |

Cache location is selected from explicit `anchor_cache_dir`, writable managed
locations, Taskdata, and platform cache directories.

Do not sync or edit the outbox database as task data. Taskwarrior tasks are the
shared record; the outbox is local durable execution state.
