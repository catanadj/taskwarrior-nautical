# Recipes

These are complete Taskwarrior commands and expression patterns. Change the
description, project, tags, times, and limits while keeping the recurrence
expression intact. Quote expressions that contain operators, parentheses, or
spaces.

## Completion periods

Use `cp` when the next task is relative to completion:

```bash
task add "Every 12 days at the seed time" due:tomorrow+9h cp:12d
task add "Every 28 elapsed hours" due:today+15h cp:28h
task add "Staged 3d, 20d, 7d, 10d, 3d intervals" cp:"3d,20d,7d,10d,3d" due:today
task add "Random interval of 3 to 7 days" cp:"rand(3d..7d)" due:today
task add "About every 14 days, plus or minus 2" cp:"14d~2d" due:today
task add "3d, random 10-20d, then 7d" cp:"3d,rand(10d..20d),7d" due:today
task add "Finite 1d, 3d, 7d, 14d, 30d follow-up" cp:"1d,3d,7d,14d,30d" chainMax:5 due:today
task add "Every 2 days, stop after 6 links" cp:2d chainMax:6 due:today+09:00
task add "Daily until 2030-12-20 at noon" cp:1d chainUntil:2030-12-20T12:00 due:today+12h
```

Whole-day periods preserve the seed wall-clock time. Other periods add exact
elapsed time from the completion or `end` timestamp. See [Completion
Periods](completion-periods.md) for carry fields, random periods, and
completion-time examples.

## Weekly and daily anchors

```bash
task add "Mondays, Wednesdays, and Fridays" anchor:"w:mon,wed,fri"
task add "Weekdays at 09:00 and 17:30" anchor:"w:mon..fri@t=09:00,17:30"
task add "Weekends at 11:00" anchor:"w:we@t=11:00"
task add "Every 2 hours in a 22:30-06:30 window" anchor:"w:mon..sun@t=22:30..06:30/2h"
task add "Every second week on Monday and Tuesday" anchor:"w/2:mon,tue"
task add "Two distinct random days each week" anchor:"w:2rand"
task add "Weekdays at 09:00 or weekends at 11:00" anchor:"w:wd@t=09:00 | w:we@t=11:00"
task add "Monday at 09:00 and Friday at 15:00" anchor:"w:mon@t=09:00,fri@t=15:00"
task add "Monday or Friday, both at 09:00" anchor:"(w:mon | w:fri)@t=09:00"
```

`w:wk` and `w:wd` mean weekdays, while `w:we` means weekends. Use `+` for
an intersection and `|` for alternatives.

## Daily time schedules with `@t=`

The `@t=` modifier turns a date rule into one or more daily time slots. Keep
the date rule and the time rule separate: the anchor chooses *which dates*,
while `@t=` chooses *when on those dates*.

```bash
task add "Every weekday at 09:00" anchor:"w:mon..fri@t=09:00"
task add "Every day at 09:00, 12:30, and 18:00" anchor:"w:mon..sun@t=9,12:30,18"
task add "Monday at 09:00 and Friday at 15:00" anchor:"w:mon@t=09:00,fri@t=15:00"
task add "Monday or Friday, both at 09:00" anchor:"(w:mon | w:fri)@t=09:00"
```

Use an interval when the schedule repeats inside a daily window. A duration
step (`/3h`) advances from the start time, while a unitless count (`/3`)
divides the window into that many slots, including both boundaries:

```bash
task add "Every 3h30 from 04:30 through 19:30" anchor:"w:mon..sun@t=04:30..19:30/3h30min"
task add "Three equally spaced times from 06:00 through 18:00" anchor:"w:mon..sun@t=06..18/3"
task add "Every 2h from 06:00-12:00 and 16:00-20:00, plus 22:00" anchor:"w:mon..fri@t=06..12/2h,16..20/2h,22"
task add "Every 2h from 22:30 through 06:30 overnight" anchor:"w:mon..sun@t=22:30..06:30/2h"
```

Windows may cross midnight. Slots after midnight remain owned by the date
that opened the window, so previews, completion, and reconcile agree about
which recurrence produced them. For deterministic variety, choose one or
more repeatable random times from a window:

```bash
task add "One deterministic random time from 06:00-18:00 on weekdays" anchor:"w:mon..fri@t=rand(06..18)"
task add "Three deterministic random times from 06:00-18:00 on weekdays" anchor:"w:mon..fri@t=rand(06..18/3)"
```

Time slots can also follow astronomy events when an astronomy profile is
configured. Offsets are applied to the event time:

```bash
task add "Every weekday at local sunrise" anchor:"w:mon..fri@t=sunrise"
task add "Every Friday 45 minutes after local sunset" anchor:"w:fri@t=sunset@+45m"
task add "At the moonrise accompanying each full moon" anchor:"moon:full@t=moonrise"
```

For the complete grammar, interval rules, midnight ownership, random slots,
and validation details, see [Multiple daily times](multiple-times.md) and the
[grammar reference](../reference/grammar.md). Use `|` when you mean
“either/or”; use `+` when the date conditions must intersect.

## Monthly anchors

### Calendar dates and business days

```bash
task add "First day of every month" anchor:"m:1"
task add "Last calendar day of every month" anchor:"m:-1"
task add "First, 15th, and last day of every month" anchor:"m:1,15,-1"
task add "A day in the first week of every month" anchor:"m:1..7"
task add "Fifth business day of every month" anchor:"m:5bd"
task add "Last business day of every month" anchor:"m:lbd"
task add "First open day of every month at 09:00" anchor:"m:1@nbd@t=09:00" anchor_mode:all
task add "Two business days before month end" anchor:"m:-1@pbd@-2bd"
task add "15th or nearest business day" anchor:"m:15@nw"
task add "Last day every second month" anchor:"m/2:-1"
```

`m:ld` is an alias for the last calendar day; `m:lbd` is the last business
day. Monthly weekday positions include `m:2sat`, `m:last-fri`, and
`m:1wed,3fri`.

```bash
task add "Second Saturday in the month" anchor:"m:2sat"
task add "Last Friday in the month" anchor:"m:last-fri"
task add "First Wednesday and third Friday" anchor:"m:1wed,3fri"
task add "Last Friday every fourth month" anchor:"m/4:last-fri"
```

### Monthly random selections

```bash
task add "One random day each month" anchor:"m:rand"
task add "Three distinct random days each month" anchor:"m:3rand"
task add "Three distinct random weekdays each month" anchor:"m:3rand + w:mon..fri"
task add "One random business day each month" anchor:"m:rand@bd"
task add "One random business day in days 1-7" anchor:"m:1..7 + m:rand@bd"
```

Random selections are deterministic per chain, so previews, completion, and
reconcile reproduce the same dates.

## Yearly and seasonal anchors

```bash
task add "May 20 every year" anchor:"y:05-20"
task add "Quarterly dates: Jan, Apr, Jul, Oct 15" anchor:"y:01-15,04-15,07-15,10-15"
task add "Every day from April 20 through May 15" anchor:"y:04-20..05-15"
task add "One random October date at noon" anchor:"y:10-rand@t=12:00"
task add "Two distinct random dates each year" anchor:"y:2rand"
task add "February 29 in leap years" anchor:"y:02-29"
task add "100th calendar day of every year" anchor:"y:d100"
task add "Last calendar day of every year" anchor:"y:d-1"
task add "Monday in ISO week 20" anchor:"y:w20 + w:mon"
task add "Every day in the first quarter" anchor:"y:q1"
task add "Every day of first month of the first quarter" anchor:"y:q1s"
task add "Every day of middle month of the first quarter" anchor:"y:q1m"
task add "Every day of last month of the first quarter" anchor:"y:q1e"
```

Ordinal yearly selectors use `dN` for calendar days and `wN` for ISO weeks.
Fixed yearly dates use `MM-DD`; `y:02-29` therefore occurs only in leap
years. Seasonal selectors require the configured season mode; see [Seasonal
selection](../advanced/seasons.md).

## Positional selectors

Positional selectors keep only a numbered or named occurrence inside a larger
period. They are useful when the rule is “the last Monday of the month” or
“the first and last Monday of spring,” rather than a fixed calendar date:

```bash
task add "Last Monday in each month" anchor:"(w:mon)@in-month=last"
task add "Last Monday or Friday in each month" anchor:"(w:mon | w:fri)@in-month=last"
task add "First and third Monday in each month" anchor:"(w:mon)@in-month=first,3rd"
task add "Second occurrence in each quarter" anchor:"(w:mon)@in-quarter=2nd"
task add "Second-to-last Monday in each quarter" anchor:"(w:mon)@in-quarter=2nd-last"
task add "Tenth occurrence in each year" anchor:"(w:mon)@in-year=10th"
task add "First and last Monday in spring" anchor:"(w:mon)@in-spring=first,last"
```

Supported scopes include `in-week`, `in-month`, `in-quarter`, `in-year`, and
season scopes such as `in-spring`. Selection happens before rolls and offsets;
validate complex combinations with Navigator.

## Astronomy schedules

Astronomy requires Astral and a configured location profile. Symbolic times
resolve against the local event for the schedule date, then apply any time
offset:

```bash
task add "Weekdays at sunrise" anchor:"w:mon..fri@t=sunrise"
task add "Thirty minutes before sunrise" anchor:"w:mon..fri@t=sunrise@-30m"
task add "One hour before dusk" anchor:"w:mon..fri@t=dusk@-1h"
task add "Friday sunset plus 45 minutes" anchor:"w:fri@t=sunset@+45m"
task add "First day of month at moonrise" anchor:"m:1@t=moonrise"
task add "Full moon at moonrise" anchor:"moon:full@t=moonrise"
task add "Friday full moon at 20:00" anchor:"(moon:full + w:fri)@t=20:00"
task add "July last quarter at moonrise" \
  anchor:"(moon:last-quarter + y:jul)@t=moonrise"
```

Supported events are `sunrise`, `sunset`, `dawn`, `dusk`, `moonrise`, and
`moonset`. If an event is unavailable for an exact date or location, Nautical
reports an actionable scheduling error rather than substituting a default
time. See [Astronomy](../advanced/astronomy.md) for location configuration,
polar-day behavior, and diagnostics.

## Random schedules

```bash
task add "One random day each ISO week" anchor:"w:rand"
task add "Two distinct random days each ISO week" anchor:"w:2rand"
task add "Wednesday plus one random weekday" anchor:"(w:rand | w:wed)"
task add "One random Saturday or Sunday each month" anchor:"m:rand + w:sat,sun"
task add "One random Saturday and one random Sunday each month" anchor:"m:rand + (w:sat | w:sun)"
task add "One random day each month in Apr 20-May 15" anchor:"m:rand + y:04-20..05-15"
task add "One random date in each half of the year" anchor:"(y:rand + y:01-01..06-30) | (y:rand + y:07-01..12-31)"
task add "One random date in Apr, Jul, or Oct" anchor:"y:rand + y:apr,jul,oct"
task add "Two random dates in Apr, Jul, or Oct" anchor:"y:2rand + y:apr,jul,oct"
```

Counted forms such as `m:3rand` select distinct dates without replacement.
Keep the same `wrand_salt` when chains are shared between devices.

## Rolls and offsets

Rolls and offsets are applied in this order: roll, calendar-day offset, then
business-day offset.

| Modifier | Effect |
| --- | --- |
| `@nbd` | Keep an open date; otherwise roll to the next business day |
| `@pbd` | Keep an open date; otherwise roll to the previous business day |
| `@nw` | Keep an open date; otherwise roll to the nearest business day |
| `@bd` | Keep business-day candidates only |
| `@+2d`, `@-2d` | Shift by calendar days |
| `@+2bd`, `@-2bd` | Shift by open business days |
| `@next-mon` | Roll forward to the next Monday |
| `@prev-mon` | Roll backward to the previous Monday |
| `@next-sat` | Roll forward to the next Saturday |

```bash
task add "First day or next open day" anchor:"m:1@nbd"
task add "Two calendar days before month end" anchor:"m:-1@-2d"
task add "Monday on or before the first" anchor:"m:1@prev-mon"
task add "Friday after the first open day" anchor:"m:1@nbd + w:fri"
```

## Combining conditions

Use `+` for AND, `|` for OR, and parentheses to control grouping. `+` binds
tighter than `|`.

```bash
task add "Monday that is the 1st or 15th" anchor:"w:mon + m:1,15"
task add "Monday-Wednesday or Friday" anchor:"w:mon..wed | w:fri"
task add "First Saturday or third Friday" anchor:"m:1sat | m:3fri"
task add "Random business day in days 1-7 or 8-14" anchor:"(m:1..7 + m:rand@bd) | (m:8..14 + m:rand@bd)"
task add "Monday in the first 21 days at 09:00" anchor:"w:mon@t=09:00 + m:1..21"
task add "Monday or last Friday, both at 09:00" anchor:"(w:mon | m:last-fri)@t=09:00"
```

## Business calendars and files

Configure trusted directories and named calendars before using these forms:

```bash
task add "First open work-calendar day at 09:00" anchor:"m:1@nbd@t=09:00" bc:work
task add "Dates from events.csv at 12:00" anchor_file:"events.csv@t=12:00"
task add "Weekdays except holidays.csv" anchor:"w:mon..fri" omit_file:holidays.csv
task add "Random monthly date except Dec 24-31" anchor:"m:rand" omit:"y:12-24..12-31"
```

See [Business calendars](../advanced/business-calendars.md) and [Files and
omissions](../advanced/files-and-omissions.md) for trusted paths, CSV formats,
and failure behavior.

## Missed occurrences and limits

```bash
task add "Weekdays at 09:00 and 17:00, skip backlog" anchor:"w:mon..fri@t=09,17" anchor_mode:skip
task add "First of month at 09:00, backfill all missed" anchor:"m:1@t=09:00" anchor_mode:all
task add "Mondays, stop after five links" anchor:"w:mon" chainMax:5
task add "Daily until 2030-12-20 at noon" cp:1d chainUntil:2030-12-20T12:00
task ID modify anchor_mode:flex <== jumps over all missed occurences and continues with 'all'
```

- `skip` jumps to the next future match.
- `all` creates missed matches in order.
- `flex` skips the backlog once, then behaves strictly.

See [Limits and expiration](limits-and-expiration.md) for the distinction
between chain limits and Taskwarrior's native `until`.

## Inspect a recipe

```bash
nautical navigator --validate "(m:1..7 + m:rand@bd) | (m:8..14 + m:rand@bd)"
nautical navigator --explain "w:mon..fri@t=09,17"
nautical query occurrences --uuid TASK_UUID --count 5
```

Use [Calendar anchors](calendar-anchors.md), [Grammar and composition](../advanced/grammar-and-composition.md),
and [Navigator](../tools/navigator.md) for the full syntax and diagnostics.
