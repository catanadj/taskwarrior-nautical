# Deterministic Random Schedules

Nautical's random schedules vary without becoming unreproducible. A result is
derived from the algorithm version, `wrand_salt`, `chainID`, normalized rule,
and recurrence period.

## Random calendar dates

```bash
task add "Weekly variation" anchor:"w:rand"
task add "Two weekly checks" anchor:"w:2rand"
task add "Monthly weekday sample" anchor:"m:3rand + w:mon..fri"
task add "October surprise" anchor:"y:10-rand"
task add "Two annual checks" anchor:"y:2rand + y:apr,jul,oct"
```

Counted selection is without replacement and returned chronologically. A
period with fewer than the requested number of eligible candidates contributes
no occurrence.

Constraints and omissions are applied before the draw. If an omitted date
would reduce a counted result, Nautical selects from the remaining candidates.

## Candidate pools and branches

These expressions differ:

```text
m:rand + w:mon,sat
m:rand + (w:mon | w:sat)
```

The first chooses one date from a single Monday-and-Saturday pool. The second
has two OR branches and can produce one random Monday plus one random Saturday.

## Random times

```bash
task add "Flexible practice" anchor:"w:mon@t=rand(06..18)"
task add "Three checks" anchor:"w:mon@t=rand(06..18/3)"
task add "Night checks" anchor:"w:fri@t=rand(22:30..02:30/3)"
```

Random windows select minute-precision values. `/N` divides the inclusive
window into N buckets and selects once from each bucket. Overnight slots after
midnight still belong to the anchor date that opened the window.

Random time windows require a chain identity and cannot be mixed with fixed
times in the same `@t=` schedule. Use separate anchor branches when both are
needed.

## Random completion periods

```bash
task add "Inspection" cp:"rand(3d..7d)"
task add "Jittered reminder" cp:"14d~2d"
```

The choice is stable for one chain link and therefore survives previews,
retries, sync, query, reconcile, and lifecycle replay.

## Salt changes

Changing `wrand_salt` deliberately changes future selections. Keep it stable
when existing chains must retain their current schedule.

Random outcomes may repeat across periods. Nautical does not force balance or
maintain mutable shuffle state.
