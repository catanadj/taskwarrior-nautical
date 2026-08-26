# Grammar Reference

## Recurrence fields

| Need | Field | Example |
| --- | --- | --- |
| Completion-relative recurrence | `cp` | `cp:"7d*3,14d"` |
| Calendar recurrence | `anchor` | `anchor:"w:mon,wed"` |
| Explicit date files | `anchor_file` | `anchor_file:"events.csv@t=12:00"` |
| Calendar exclusions | `omit` | `omit:"y:12-24..12-31"` |
| Explicit exclusion files | `omit_file` | `omit_file:holidays.csv` |
| Missed-match policy | `anchor_mode` | `skip`, `all`, `flex` |
| Named business calendar | `bc` | `bc:work` |
| Link limit | `chainMax` | `chainMax:12` |
| Date/time limit | `chainUntil` | `chainUntil:2027-12-31` |

## Completion periods

| Form | Meaning |
| --- | --- |
| `3d`, `8h`, `90m`, `P2W` | Fixed duration |
| `3d,20d,7d` | Repeating sequence |
| `7d*3,14d` | Repeat-count sequence |
| `rand(3d..7d)` | Deterministic bounded duration |
| `14d~2d` | Deterministic base plus/minus spread |

## Weekly atoms

| Form | Meaning |
| --- | --- |
| `w:mon,fri` | Weekday list |
| `w:mon..wed` | Inclusive range |
| `w/2:mon` | Every second ISO week |
| `w:rand`, `w:2rand` | One or two random weekdays |

## Monthly atoms

| Form | Meaning |
| --- | --- |
| `m:1,15,-1` | Day list; `-1` is month end |
| `m:1..7` | Inclusive day bucket |
| `m:5bd`, `m:lbd` | Business-day ordinal or last business day |
| `m:2sat`, `m:last-fri` | Positional weekday |
| `m/3:1` | Every third month on day 1 |
| `m:rand`, `m:3rand` | Random date selection |

## Yearly atoms

| Form | Meaning |
| --- | --- |
| `y:05-20` | Month and day |
| `y:01-15,04-15` | Date list |
| `y:04-20..05-15` | Inclusive date range |
| `y:d100`, `y:d-1` | Calendar-day ordinal |
| `y:w20 + w:mon` | Monday of ISO week 20 |
| `y:10-rand`, `y:2rand` | Random date selection |
| `y:q1s`, `y:q2m`, `y:q4e` | Quarter start/middle/end month |
| `y/3:06-07` | Every third year |

## Logic

| Operator | Meaning |
| --- | --- |
| `+` | Same date must satisfy both sides |
| `\|` | Either branch may match |
| `( ... )` | Explicit grouping or shared modifiers |

`+` binds tighter than `|`.

## Time forms

| Form | Meaning |
| --- | --- |
| `@t=9` | 09:00 |
| `@t=09:00,17:30` | Exact time list |
| `@t=06..18/3` | Three equally spaced slots |
| `@t=04:30..19:30/3h30min` | Fixed interval inside a window |
| `@t=06..12/2h,16..20/2h,22` | Composed fixed schedule |
| `@t=rand(06..18)` | One deterministic random minute |
| `@t=rand(06..18/3)` | One random minute from each of three buckets |
| `@t=sunset@+45m` | Astronomy event plus offset |

Fixed and random windows may cross midnight.

## Date modifiers

| Modifier | Meaning |
| --- | --- |
| `@bd` | Keep business-day candidates |
| `@nbd`, `@pbd`, `@nw` | Next, previous, or nearest business-day roll |
| `@+Nd`, `@-Nd` | Calendar-day offset |
| `@+Nbd`, `@-Nbd` | Business-day offset |
| `@next-mon`, `@prev-fri` | Named weekday roll |

## Positional selectors

```text
(expression)@in-week=last
(expression)@in-month=first,3rd,last
(expression)@in-quarter=2nd-last
(expression)@in-year=10th
(expression)@in-season=1st
(expression)@in-spring=first,last
```

Selection happens before modifiers. See
[Grammar and composition](../advanced/grammar-and-composition.md) for the
validation rules.
