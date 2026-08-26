# Calendar Anchors

An anchor describes calendar positions. Quote expressions so the shell cannot
interpret operators, parentheses, commas, or time schedules.

## Weekly

```bash
task add "Workout" anchor:"w:mon,wed,fri@t=09:00"
task add "Long weekend check" anchor:"w:fri..sun"
task add "Fortnightly report" anchor:"w/2:mon@t=10:00"
```

Use `mon` through `sun`, comma lists, inclusive `..` ranges, or `/N` stepped
cadence.

## Monthly dates

```bash
task add "Billing" anchor:"m:1,-1"
task add "Payroll" anchor:"m:1@nbd@t=09:00" anchor_mode:all
task add "Preparation" anchor:"m:-1@pbd@-2bd"
task add "Supplier payment" anchor:"m:5bd"
```

- `m:-1` or `m:ld` is the last calendar day.
- `m:lbd` is the last business day.
- `m:5bd` is the fifth business day.
- `m:1..7` is an inclusive day-of-month bucket.

## Monthly weekday positions

```bash
task add "Bake day" anchor:"m:2sat"
task add "Design review" anchor:"m:last-fri"
task add "Meetings" anchor:"m:1wed,3fri"
```

## Yearly

```bash
task add "Anniversary" anchor:"y:05-20"
task add "Quarterly review" anchor:"y:01-15,04-15,07-15,10-15"
task add "Year end" anchor:"y:d-1@t=17:00"
task add "ISO checkpoint" anchor:"y:w20 + w:mon"
```

Yearly dates use `MM-DD`. `dN` is a calendar-day ordinal; `wN` is an ISO-week
ordinal. Negative ordinals count backward from the relevant year.

## Combine conditions

`+` requires both conditions; `|` accepts either condition. `+` binds tighter.

```bash
task add "April training" anchor:"w:mon,wed,fri + y:apr"
task add "Weekend deadline" anchor:"w:fri | w:sun"
task add "Specific Mondays" anchor:"w:mon + m:1,15"
```

Use parentheses when the grouping matters:

```bash
task add "Shared time" anchor:"(w:mon | m:last-fri)@t=09:00"
```

## Transform a match

Rolls and offsets compose in a fixed order: roll, calendar-day offset, then
business-day offset.

| Modifier | Effect |
| --- | --- |
| `@nbd` | Keep an open date; otherwise next business day |
| `@pbd` | Keep an open date; otherwise previous business day |
| `@nw` | Nearest business day |
| `@bd` | Keep business-day candidates only |
| `@+2d`, `@-2d` | Shift by calendar days |
| `@+2bd`, `@-2bd` | Shift by open business days |
| `@next-mon`, `@prev-fri` | Roll to a named weekday |

Named [business calendars](../advanced/business-calendars.md) make these
operations aware of closures and exceptional open dates.
