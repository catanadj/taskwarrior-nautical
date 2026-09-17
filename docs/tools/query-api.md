# Query API

The query command is Nautical's stable read-only contract for external tools.
It emits exactly one versioned JSON document on stdout. Consumers should not
import private `nautical_core` modules or parse human-readable panels.

## Discover capabilities

```bash
nautical query capabilities | jq
```

The response advertises operations, selectors, timestamp semantics, omission
policies, hard limits, provider availability, and short usage examples.

## Occurrence schedules

```bash
nautical query occurrences \
  --uuid TASK_UUID \
  --from 2026-08-24 \
  --count 5

nautical query occurrences \
  --chain-id 176f5c68 \
  --from 2026-08-24 \
  --to 2026-08-31

nautical query occurrences \
  --all \
  --from 2026-08-24 \
  --to 2026-08-24
```

For task selectors, results never precede the current task's `due`, or its
`scheduled` value when due is absent. A task due in a future year therefore
does not appear as an active occurrence in an earlier range.

An occurrence contains configured local time, UTC, timezone, UTC offset,
daylight-saving fold, source, and omission details. This is a schedule
projection; it does not claim that lifecycle hooks have already created every
future Taskwarrior child.

## Paginate whole-system queries

Whole-system (`--all`) occurrence queries are returned in deterministic pages.
When more tasks remain, the response has `complete: false` and includes a
`cursor` object. Pass that cursor back unchanged to fetch the next page:

```bash
nautical query occurrences \
  --all \
  --from 2026-08-24 \
  --to 2026-08-31 > page.json

cursor=$(jq -c '.cursor // empty' page.json)
if [ -n "$cursor" ]; then
  nautical query occurrences \
    --all \
    --from 2026-08-24 \
    --to 2026-08-31 \
    --cursor "$cursor"
fi
```

Keep the selector, date boundaries, omission policy, and safety limits the
same on every continuation request. A cursor is opaque: do not construct it,
edit its fields, or treat its internal fields as a stable API. Cursors are
valid only for `--all` queries and are bound to the observed snapshot,
configuration fingerprint, and mutation epoch. If Taskwarrior data or the
configuration changes between pages, the response is `unavailable` with the
`cursor_unavailable` failure code; discard the cursor and restart the query.

The `--count` option limits occurrences per task, not the number of tasks in a
page. `--max-total-occurrences` caps occurrences returned by one response and
is independent of page continuation. Use the `complete` flag, rather than the
presence of results, to decide whether another page is required.

## Omission policies

```bash
nautical query occurrences --uuid TASK_UUID --count 10 --omissions exclude
nautical query occurrences --uuid TASK_UUID --count 10 --omissions include
nautical query occurrences --uuid TASK_UUID --count 10 --omissions report
```

- `exclude` returns only usable occurrences.
- `include` includes omitted occurrences in the main stream.
- `report` returns usable and omitted occurrences separately.

## Next lifecycle projection

`next` is read-only. It does not create a child or mutate Taskwarrior.

```bash
nautical query next --uuid TASK_UUID --at 2026-08-24T15:00:00+03:00
```

Anchor results include `lifecycle.daily_instances` with:

- `date`;
- `total`;
- `current_position`;
- `missed`;
- `upcoming`.

Skip mode also reports exact `missed_occurrences` and the selected next slot.
Supplying `--at` keeps mode-aware progress reproducible.

## Chain integrity

```bash
nautical query integrity --chain-id 176f5c68 | jq
nautical query integrity --uuid TASK_UUID | jq
nautical query integrity --all | jq
```

Integrity output includes snapshot coverage, findings, repair plans, refusals,
and per-chain status. It is read-only; use [Reconcile](reconcile.md) to preview
or apply recovery.

## JSON request transport

Flags and JSON requests produce the same validated request model:

```bash
printf '%s\n' '{
  "selector": {"chainID": "176f5c68"},
  "from": "2026-08-24T09:00:00+03:00",
  "count": 5,
  "omission_policy": "exclude"
}' | nautical query occurrences
```

Alternatively use `--request`, `--request -`, or `--request-file`. Do not mix a
JSON request with selector/range flags.

## Status and exit contract

| Exit | Meaning |
| ---: | --- |
| `0` | Request was valid; inspect response and per-task statuses |
| `2` | Invalid request or schedule input |
| `3` | Required Taskwarrior, configuration, provider, or snapshot data unavailable |

Structured statuses distinguish `found`, `empty`, `invalid`, `exhausted`,
`absent`, and `unavailable`. Code against schema/version, status, and failure
codes rather than human messages.
