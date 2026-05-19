![Nautical Banner](./nautical-banner.svg)

# Taskwarrior Nautical

Taskwarrior is already a powerhouse; Nautical is the recurrence system I wanted on top of it.

It is not a separate task manager and it does not replace Taskwarrior. It is a small hook-based layer that teaches Taskwarrior how to carry recurring work forward in a predictable way.

When you complete a Nautical task, the hook asks one question:

> What should the next task be, and when should it happen?

Then it creates the next link, keeps the chain connected, and shows you what it did.

![demo_nautical](https://github.com/user-attachments/assets/8420c1a8-907b-483e-86ec-4385eec892e3)

## What Nautical Adds

Taskwarrior is excellent at storing and finding tasks. Nautical focuses on recurrence.

It gives you two ways to describe recurring work:

- `cp` for completion-based period chains
- `anchor` for calendar-based recurrence

It also gives anchors file-backed inclusion and exclusion:

- `anchor_file` adds dates from a trusted file
- `omit` removes dates using anchor grammar
- `omit_file` removes dates from a trusted file

For anchor recurrence, the mental model is:

```text
included dates from anchor and anchor_file
minus
blocked dates from omit and omit_file
```

In plain language: include the dates from the anchor expression and anchor file, then remove the dates blocked by omit rules and omit files.

## Chains: Period Recurrence

Use `cp` when the next task should be based on a duration.

```bash
# Every 12 days
task add "Mow lawn" cp:12d due:today

# Every 33 hours
task add "Take medication" cp:33h due:now

# Stop after 5 links
task add "Calibration" cp:3d chainMax:5
```

Chains are useful for work that depends on completion rhythm: maintenance, medication, reviews, chores, and anything where "next time" follows from this time.

## Anchors: Calendar Recurrence

Use `anchor` when the next task should land on calendar positions.

```bash
# Every Monday, Wednesday, and Friday
task add "Workout" anchor:w:mon,wed,fri due:today

# Last Friday of every month
task add "Monthly review" anchor:m:last-fri due:today

# A random Saturday each year
task add "Annual retreat" anchor:"y:rand + w:sat" due:today
```

Anchors are useful when the calendar matters more than the completion interval.

## Files and Omitted Dates

Real calendars have exceptions: holidays, blackout windows, travel, company shutdowns, and fixed external schedules.

Nautical handles those without forcing everything into one dense expression.

```bash
# Mondays, Wednesdays, Fridays, except Wednesdays in June
task add "Workout" anchor:w:mon,wed,fri omit:"w:wed + y:jun"

# Use a CSV file as an additional source of anchor dates
task add "Payroll prep" anchor_file:2026.csv

# Use a holiday file to block dates
task add "Date night" anchor:"m:rand + w:sat" omit_file:2026.csv

# Combine all four sources
task add "Hybrid schedule" \
  anchor:"w:tue,fri | y:05-05" \
  anchor_file:"2026.csv@-1d@t=12:00,18:00" \
  omit:"y:04-28..05-05" \
  omit_file:2026.csv
```

This split is intentional. Inclusion and exclusion stay separate, so the rule is easier to read, preview, validate, and debug.

## What You See

Nautical is deliberately visible.

On add and completion, it can show:

- the recurrence pattern
- the natural-language explanation
- the next task that was created
- the timeline of upcoming links
- omitted dates as explicit skipped rows
- chain caps and remaining links


## Sync and Duplicate Protection

Nautical uses chain identity to keep recurring tasks connected across devices.

The next task is spawned from the completed task state available to the device running the hook. If you annotate or adjust the current task and then complete it, that local state is what Nautical carries forward.

In a synced setup, sync order still matters. If another device changes a task after a child was already spawned elsewhere, Nautical cannot retroactively rewrite that child. The rule is simple and explicit: the completed task state at spawn time is the source.

Nautical also guards against duplicate children and handles batch completion through `on-exit`.

## Install

Nautical consists of:

- `nautical_core/`
- `on-add-nautical.py`
- `on-modify-nautical.py`
- `on-exit-nautical.py`
- the UDA definitions in `uda.conf`
- optional settings in `config-nautical.toml`

Quick install:

```bash
# 1. Drop the hooks in place
cd ~/.task/hooks
curl -LO https://github.com/catanadj/taskwarrior-nautical/raw/main/on-{add,modify,exit}-nautical.py
chmod +x on-*.py

# 2. Install the shared Nautical package next to your Taskwarrior data
cd ..
curl -L https://github.com/catanadj/taskwarrior-nautical/archive/refs/heads/main.tar.gz \
  | tar -xz --strip-components=1 taskwarrior-nautical-main/nautical_core

# 3. Add UDAs
curl -s https://raw.githubusercontent.com/catanadj/taskwarrior-nautical/main/uda.conf >> ~/.taskrc

# 4. Optional pretty panels
pip install rich

# 5. Test
task add "System test" anchor:"m:4mon"
```

For a full setup, including config keys and file-backed recurrence directories, read the manual.

## Current Boundaries

- `cp` and anchor recurrence are separate engines.
- `omit` is date-based and does not support timed omissions.
- `anchor_file` and `omit_file` resolve basenames inside trusted config directories.
- Nautical follows Taskwarrior hook and sync behavior; it does not run a daemon.
- The system still refuses to spawn tasks after `9999-12-31`, so long-term planning has limits. :)

## Documentation

- [PDF Manual](./TW-Nautical-Manual.pdf)
- [Releases](https://github.com/catanadj/taskwarrior-nautical/releases)
- [Issues](https://github.com/catanadj/taskwarrior-nautical/issues)

## Support

If Nautical is useful to you, support is appreciated:

[Buy me a coffee](https://buymeacoffee.com/catanadj)

