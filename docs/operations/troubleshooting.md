# Troubleshooting

Start with the narrowest supported command.

| Symptom | First action |
| --- | --- |
| Install, launcher, UDA, timezone, or dependency problem | `nautical doctor --installation-only` |
| Unsure what an anchor means | `nautical navigator --explain "EXPRESSION"` |
| Validate an expression | `nautical navigator --validate "EXPRESSION"` |
| Inspect future matches | `nautical query occurrences --uuid UUID --count 5` |
| Missing successor | `nautical reconcile --uuid UUID` |
| Inspect one chain | `nautical query integrity --chain-id CHAIN_ID` |
| Lifecycle retry or review | `nautical queue-status --json` |
| Startup/config search details | `NAUTICAL_DIAG=1 nautical doctor` |

## Hook error without feedback

Run the same operation with diagnostics:

```bash
NAUTICAL_DIAG=1 task <original arguments>
```

Strict hooks emit one task JSON object on stdout. Startup and scheduling detail
belongs on stderr, so diagnostics reveal the cause without breaking
Taskwarrior's protocol.

## No child after completion

```bash
nautical queue-status --json
nautical reconcile --uuid PARENT_UUID --json
```

If the dry-run action and target are correct:

```bash
nautical reconcile --uuid PARENT_UUID --apply
```

## Invalid configuration or timezone

Nautical fails closed when an automatically discovered scheduling config is
malformed, unsafe, or contains an invalid timezone. The diagnostic identifies
the rejected path and reason. Correct that file; do not rely on fallback
defaults.

```bash
NAUTICAL_CONFIG=/path/to/config-nautical.toml nautical doctor --installation-only
```

## Astronomy event unavailable

Rise and set events can be absent for a date and latitude. Use a wider date
expression when any eligible date is acceptable. For an exact date, choose a
different event/date or location. Confirm the active profile with Doctor.

## Query reports unavailable

Exit code 3 means required data could not be read or evaluated. Inspect the
structured `failure`, repair the Taskwarrior/configuration/provider issue, and
retry. Do not interpret unavailable as an empty schedule.

## Manual-review lifecycle intent

Use `queue-status --json` and a scoped integrity query. The result should name
the changed immutable fields or failed mutation stage. Volatile Taskwarrior
fields and equivalent timestamp encodings are normalized before comparison;
remaining differences require actual evidence review.

## Launcher is stale or absent

```bash
type -a nautical
command -v nautical
nautical doctor --installation-only
```

Re-run the bootstrap. On Termux, the launcher normally belongs in
`$PREFIX/bin`; on Linux it defaults to `~/.local/bin`.

## Report an issue

Open a [GitHub issue](https://github.com/catanadj/taskwarrior-nautical/issues)
with the smallest reproducible command and sanitized diagnostic evidence.
