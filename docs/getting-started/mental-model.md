# Mental Model

## A chain of ordinary tasks

Nautical recurrence is a linked chain, not Taskwarrior's native recurring-task
template model. Each occurrence is an ordinary task with its own UUID.

Core identity fields are:

| Field | Meaning |
| --- | --- |
| `chainID` | Stable identity shared by every link in one recurrence |
| `link` | One-based position in the chain |
| `prevLink` | Previous task UUID, when one exists |
| `nextLink` | Next task UUID, after the successor is created |
| `chain` | `on` while recurrence may continue |

Do not edit chain identity fields manually. They are guarded lifecycle data,
not user configuration.

## Planning and application are separate

When a Nautical task is completed:

1. The modify hook validates the transition and builds one deterministic plan.
2. The plan is persisted in the lifecycle outbox.
3. The exit hook applies it after Taskwarrior releases its datastore lock.
4. Nautical imports the child, links the parent, and verifies postconditions.

If the process stops between stages, the durable intent can be replayed without
creating a second child. Conflicting or incomplete evidence becomes retry or
manual review instead of an unsafe mutation.

## Schedule versus lifecycle

The scheduler answers **when occurrences exist**. The lifecycle engine answers
**what Taskwarrior mutation is valid now**.

This distinction matters for tools:

- `nautical query occurrences` returns schedule projections.
- `nautical query next` returns a read-only lifecycle projection.
- `nautical reconcile` plans and optionally applies recovery mutations.

## Time model

Calendar expressions are evaluated in the configured local timezone and stored
as UTC Taskwarrior timestamps. Day- and week-sized completion periods preserve
the seed wall-clock time; other periods are exact additions from completion.

Timezone, astronomy, calendar, preset, or file configuration that affects a
schedule is validated before mutation. Nautical fails closed when it cannot
establish that context.

## Limits are part of the recurrence

`chainMax` limits the number of links. `chainUntil` limits target datetimes.
Native Taskwarrior `until` controls per-occurrence expiration and is carried
according to the recurrence policy; it is not the same as `chainUntil`.

Read [Lifecycle and chain integrity](../operations/lifecycle-and-integrity.md)
for the full operational model.
