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
