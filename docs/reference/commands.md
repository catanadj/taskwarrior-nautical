# Command Reference

`nautical` is the operator entry point. Use the focused guide for the command
you need; commands that emit JSON are documented with their schema and exit
contract in [Command Status and Exit Codes](command-status.md).

| Command | Use it for | Detailed guide |
| --- | --- | --- |
| `nautical install` | Install, upgrade, or validate a managed runtime | [Installation](../getting-started/installation.md) |
| `nautical runtime-clean` | Inspect or remove inactive runtime releases | [Installation](../getting-started/installation.md#clean-inactive-runtime-releases) |
| `nautical doctor` | Read-only installation and chain health checks | [Doctor](../tools/doctor.md) |
| `nautical queue-status` | Inspect durable lifecycle intents and outbox health | [Lifecycle Outbox](../tools/lifecycle-outbox.md) |
| `nautical queue-review` | Inspect or explicitly resolve manual-review intents | [Lifecycle Outbox](../tools/lifecycle-outbox.md#review-intents) |
| `nautical backup` | Create a verified local backup generation | [Backup and Restore](../tools/backup-restore.md) |
| `nautical restore` | Validate or stage a backup into a disposable target | [Backup and Restore](../tools/backup-restore.md) |
| `nautical reconcile` | Preview or apply chain and lifecycle recovery | [Reconcile](../tools/reconcile.md) |
| `nautical query` | Read-only schedule and integrity projections | [Query API](../tools/query-api.md) |
| `nautical navigator` | Explain, validate, or inspect anchor expressions | [Navigator](../tools/navigator.md) |

The aliases `nautical queue` and `nautical nav` are supported for interactive
use. Scripts should use the canonical command names. Taskwarrior invokes the
installed `on-add`, `on-modify`, and `on-exit` hooks automatically; those hooks
are not replacements for the read-only operator commands above.

For the shortest recovery path, start with `nautical doctor`, inspect
`nautical queue-status --json`, and run a scoped `nautical reconcile` dry run
before applying any mutation.
