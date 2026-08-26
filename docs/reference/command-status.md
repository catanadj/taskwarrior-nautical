# Command Status and Exit Codes

Machine-readable commands emit one JSON document with `ensure_ascii=False`.
Diagnostics are sent to stderr only when enabled.

## Query

| Exit | Meaning |
| ---: | --- |
| `0` | Valid request; inspect top-level and per-result status |
| `2` | Invalid request or schedule input |
| `3` | Required data/provider unavailable |

## Doctor and queue status

| Exit | Meaning |
| ---: | --- |
| `0` | Healthy |
| `1` | Warning/attention |
| `2` | Error or required evidence unavailable |

## Reconcile

| Exit | Meaning |
| ---: | --- |
| `0` | `ok` |
| `1` | `error` |
| `2` | `degraded` |

`degraded` can represent bounded deferral or review without an unsafe
mutation. Always inspect the versioned summary.

## Installer

| Exit | Meaning |
| ---: | --- |
| `0` | Staged release validated and installed, or dry run passed |
| `2` | Installation or validation failed |

## Hooks

Taskwarrior hooks follow Taskwarrior's protocol, not the operator CLI contract.
Successful add/modify hooks return the task JSON expected by Taskwarrior on
stdout. A failing hook returns nonzero and must provide valid user feedback;
diagnostic detail belongs on stderr.

On-exit returns zero by default when work is deferred, preserving the original
Taskwarrior command. Set `NAUTICAL_EXIT_STRICT=1` when automation must receive a
nonzero status for drain failure or manual review.

## Stable JSON consumption

- Check `schema` and version before fields.
- Treat `unavailable` differently from `empty` or `absent`.
- Use structured `failure.code`, retryability, and evidence.
- Do not parse human panel text or stderr diagnostics.
