# Taskwarrior Nautical

<p class="nautical-lead">
Nautical is a recurrence engine for Taskwarrior. It creates ordinary
Taskwarrior tasks, while adding calendar anchors, completion-relative periods,
multiple times per day, deterministic random schedules, business calendars,
and guarded recovery.
</p>

Nautical has two recurrence styles:

- **Completion periods (`cp`)** schedule the next task relative to completion.
- **Calendar anchors (`anchor`)** select dates and times from a calendar rule.

```bash
# Twelve days after completion.
task add "Mow the lawn" cp:12d

# Every Monday, Wednesday, and Friday at 09:00.
task add "Workout" anchor:"w:mon,wed,fri@t=09:00"
```

Complete either task normally with `task <id> done`. Nautical plans the next
link during the modify hook and applies it after Taskwarrior releases its data
lock.

!!! tip "New to Nautical?"
    Follow [Installation](getting-started/installation.md), then create the two
    examples in [First recurrence](getting-started/first-recurrence.md).

<div class="grid cards" markdown>

-   **Build a routine**

    ---

    Learn completion periods, calendar rules, multiple daily times, and chain
    limits.

    [Open the guides](guides/completion-periods.md)

-   **Compose precise schedules**

    ---

    Combine logic, files, business calendars, seasonal selectors, astronomy,
    and reproducible random choices.

    [Explore advanced scheduling](advanced/grammar-and-composition.md)

-   **Inspect without guessing**

    ---

    Query occurrences through Nautical's versioned JSON API or explain an
    expression with Navigator.

    [Use the query API](tools/query-api.md)

-   **Recover safely**

    ---

    Doctor, the lifecycle outbox, chain integrity, and reconcile provide a
    fail-closed recovery path.

    [Read the recovery workflow](operations/sync-and-recovery.md)

</div>

## Design principles

- Taskwarrior remains the system of record.
- Calendar calculations happen in an explicit IANA timezone.
- Random choices are deterministic for the same chain and occurrence.
- Mutation decisions use durable plans and authoritative reads.
- Missing or conflicting evidence stops mutation instead of inventing state.

The supported operator and integration surface is the `nautical` command. Do
not import private `nautical_core` modules from external tools; use the
[query API](tools/query-api.md) for stable read-only access.
