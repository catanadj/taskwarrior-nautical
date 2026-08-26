# First Recurrence

## Completion-relative task

Add a task whose next link is due twelve days after completion:

```bash
task add "Mow the lawn" cp:12d
```

Complete it normally:

```bash
task <id> done
```

Nautical creates the next pending task in the same chain. The new task has a
new UUID and incremented `link`, while `chainID`, recurrence configuration,
and supported carry fields remain consistent.

Use `cp` when the important question is: **how long after I finish this should
it return?**

## Calendar task

Add a weekly review anchored to Monday at 09:00:

```bash
task add "Weekly review" anchor:"w:mon@t=09:00"
```

Nautical previews the first due date and upcoming matches before Taskwarrior
saves the root task. Complete the task normally to advance the chain.

Use `anchor` when the important question is: **which calendar dates and times
should contain this task?**

## Multiple times in one day

```bash
task add "Drink water" anchor:"w:mon..sun@t=09,12,18"
```

Bare hours and `HH:MM` values can be mixed. Every time is a separate occurrence
in the same recurrence stream.

## Choose missed-occurrence behavior

`anchor_mode` controls what happens when one or more calendar matches are
already in the past:

```bash
task add "Practice" anchor:"w:mon..fri@t=09,17" anchor_mode:skip
task add "Submit records" anchor:"m:1@t=09:00" anchor_mode:all
```

- `skip` jumps to the next future match.
- `all` creates every missed match in order.
- `flex` skips backlog once, then changes itself to `all`.

## Inspect the result

```bash
nautical navigator --explain "w:mon..fri@t=09,17"
nautical query occurrences --uuid TASK_UUID --count 5
```

Continue with [Completion periods](../guides/completion-periods.md) or
[Calendar anchors](../guides/calendar-anchors.md).
