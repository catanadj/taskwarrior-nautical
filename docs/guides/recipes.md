# Recipes

These commands are starting points. Change descriptions, projects, tags, and
times without changing the recurrence structure.

## Completion periods

```bash
# Every 12 days at the seed wall-clock time.
task add "Mow lawn" cp:12d due:tomorrow+9h

# Exact elapsed cadence.
task add "Take vitamin" cp:28h

# Staged intervals that repeat.
task add "Treatment" cp:"3d,20d,7d"

# Three weekly links, then one fortnightly link.
task add "Habit ramp" cp:"7d*3,14d"

# Deterministic variation.
task add "Inspection" cp:"14d~2d"
```

## Weekly and daily anchors

```bash
task add "Workout" anchor:"w:mon,wed,fri@t=09:00"
task add "Hydrate" anchor:"w:mon..sun@t=09,12,18" anchor_mode:skip
task add "Night checks" anchor:"w:mon..sun@t=22:30..06:30/2h"
```

## Monthly operations

```bash
task add "Payroll" anchor:"m:1@nbd@t=09:00" anchor_mode:all
task add "Month-end prep" anchor:"m:-1@pbd@-2bd"
task add "Design review" anchor:"m:last-fri@t=15:00"
task add "Monthly sample" anchor:"m:3rand + w:mon..fri"
```

## Yearly and seasonal work

```bash
task add "Anniversary" anchor:"y:05-20@t=18:00"
task add "ISO review" anchor:"y:w20 + w:mon"
task add "Spring planning" anchor:"(w:mon)@in-spring=first,last"
task add "Quarter handoff" anchor:"(w:mon)@in-quarter=last@+1bd"
```

## Business calendars and files

```bash
task add "Company payroll" anchor:"m:1@nbd@t=09:00" bc:work
task add "Published events" anchor_file:"events.csv@t=12:00"
task add "Routine except holidays" anchor:"w:mon..fri" omit_file:holidays.csv
```

Configure trusted directories and named calendars before using file-backed or
calendar-aware rules. See [Business calendars](../advanced/business-calendars.md)
and [Files and omissions](../advanced/files-and-omissions.md).
