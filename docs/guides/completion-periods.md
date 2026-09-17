# Completion Periods

Use `cp` when the next link is based on when the current task is completed.

## Fixed periods

```bash
task add "Trim the grass" cp:12d due:tomorrow+9h
task add "Equipment check" cp:28h due:today+12h
```

Accepted period forms include short durations such as `90m`, `12d`, and `2w`,
plus ISO 8601 durations such as `PT90M`, `P12D`, and `P2W`.

## Wall-clock and exact periods

Multiples of 24 hours preserve the seed task's local due time. Thus `cp:2d`
returns at the same local clock time across daylight-saving changes.

Other spans, such as `28h` or `33h`, are exact elapsed additions from the
completion timestamp. Add or subtract one second to force exact behavior near a
whole-day period:

```bash
task add "Exact follow-up" cp:24h+1s
```

## Keep `scheduled` and `wait` offsets when due moves

`due` is the root of the temporal relationship. When you move the due date of
an existing completion-period task, Nautical shifts each unchanged `scheduled`,
`wait`, and native `until` value by the same local-time amount. For example, if
`scheduled` is 10 minutes before `due`, `wait` is 20 minutes before `due`, and
`until` is one hour after `due`, moving `due` from July 10, 2026, at 09:00 to
July 15, 2026, at 09:00 moves those fields to July 15 at 08:50, 08:40, and
10:00. Editing `scheduled` or `wait` directly changes only that field; it does
not move `due`, the other temporal fields, or `until`. Editing `until` directly
also changes only `until`.

Nautical applies the offset in its configured timezone. If the move crosses a
daylight-saving transition, it preserves the local clock relationship; the
elapsed UTC difference can therefore change. If Nautical cannot parse a
timestamp needed for the carry, it rejects the update instead of guessing.

### How reconcile reconstructs the relationship

Reconcile does not infer offsets from task descriptions or from the current
clock. It builds an authoritative snapshot of the active chain, then resolves
the candidate's predecessor by `chainID` and the immediately lower `link`
(refreshing Taskwarrior data when necessary). For each available pair it parses
the predecessor's target (`due`, or `scheduled` for a scheduled-only chain),
its `scheduled`, `wait`, and `until` values, and the candidate's target. The
expected relationship is the predecessor's local-time offset from its target:

```text
offset(field) = predecessor[field] - predecessor[target]
expected     = candidate[target] + offset(field)
```

The comparison uses canonical timestamps. Ordinary `scheduled` and `wait`
values preserve their exact offsets. Native `until` also preserves whether the
predecessor used calendar carry or an exact second-level carry, so daylight-
saving transitions do not create false alarms. Mismatches are reported as
repairable or manual review according to the available evidence; they are
never silently overwritten.

If the predecessor is missing, malformed, changed since the snapshot, or the
candidate no longer matches the audited task, reconcile refuses to guess. It
reports the evidence gap, refreshes the relevant rows under mutation guards,
and requires a new dry-run before applying a change. For native `until`, a
documented local end-of-day fallback may be offered when predecessor evidence
cannot provide a safe policy.

When Nautical creates a successor, it carries `scheduled` and `wait` relative
to the recurrence target. For a task with `due`, that target is `due`. For a
scheduled-only task, `scheduled` is the target and the child remains
scheduled-only; `wait` keeps its offset from `scheduled`.

## Carry custom date UDAs

If a recurrence has another date-valued UDA that should follow the same local
time relationship, list its field name in the configuration:

```toml
recurrence_update_udas = ["review_at", "reminder_at"]
```

The fields must be registered with Taskwarrior and contain parseable date or
datetime values. For example, a task with `due:2026-07-10T09:00` and
`review_at:2026-07-10T08:30` carries `review_at` to the successor at 08:30
relative to that successor's target. The same rule applies to `cp` and anchor
recurrences.

For a task with `due`, Nautical measures the custom field from `due`. For a
scheduled-only task, it measures from `scheduled`. The calculation preserves
the configured local clock relationship across daylight-saving transitions,
just like `wait` and `scheduled`; it does not copy the parent's absolute UTC
timestamp.

Only configured fields present on the parent are carried. Field names are
matched case-insensitively, duplicate names are ignored, and lifecycle-owned
fields are never treated as custom carry fields. The generated child clears the
field first, so an absent parent value does not leave a stale inherited value.

If a configured value or its recurrence target is missing or malformed,
successor generation fails safely with a carry error instead of guessing. Fix
the date value or configuration, then retry through the normal lifecycle or
reconcile workflow. The nested compatibility form is also accepted:

```toml
[recurrence]
update_udas = ["review_at"]
```

## Period sequences

```bash
task add "Treatment cycle" cp:"3d,20d,7d,10d,3d"
```

Nautical chooses one sequence entry per completed link and repeats from the
beginning after the last entry. The active position is derived from `link`, so
there is no separate sequence cursor to drift.

Repeat-count syntax shortens repeated entries:

```bash
task add "Habit ramp" cp:"7d*3,14d"
```

This is equivalent to `7d,7d,7d,14d`.

!!! note
    Sequence and random forms require `uda.cp.type=string`. The installer
    registers the supported v7 UDA definition.

## Random and jittered periods

```bash
task add "Check trap" cp:"rand(3d..7d)"
task add "Routine inspection" cp:"14d~2d"
task add "Follow-up" cp:"3d,rand(10d..20d),7d"
```

`14d~2d` is the readable form of a deterministic selection from 12 through 16
days. Random values are derived from stable recurrence identity, so retries and
other Nautical paths resolve the same interval.

## Stop the sequence

Apply [chain limits](limits-and-expiration.md):

```bash
task add "Calibration" cp:33h chainMax:5 due:today+12h
task add "Daily focus" cp:1d chainUntil:2030-12-20T12:00 due:today+12h
```

Setting `chain:off` prevents another link after the current one. Intentional
deletion before native `until` also stops a chain; automatic expiration at
`until` advances it.
