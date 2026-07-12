![Nautical Banner](./nautical-banner.svg)

![demo_nautical](https://github.com/user-attachments/assets/8420c1a8-907b-483e-86ec-4385eec892e3)

# Taskwarrior Nautical

Nautical is a recurrence engine for Taskwarrior.

It runs as normal Taskwarrior hooks. When you complete a Nautical task, it creates
the next task in the chain using either a period rule (`cp`) or a calendar rule
(`anchor`). The spawned task is still a normal Taskwarrior task: visible, editable,
syncable, and managed with the regular `task` command.

Use Nautical when Taskwarrior's built-in recurrence is too simple for the schedule
you actually need.

---

## Quick Examples

Period-based recurrence:

```bash
# Every 12 days from completion
task add "Mow the lawn" cp:12d

# Every 8 hours
task add "Take vitamin" cp:8h

# Pick a repeat period between 3 and 7 days
task add "Inspect field trap" cp:"rand(3d..7d)"

# Follow a repeating sequence of periods
task add "Insect lifecycle check" cp:4d,10d,7d,20d,3d
```

Calendar-based recurrence:

```bash
# Every Monday, Wednesday, and Friday
task add "Workout" anchor:"w:mon,wed,fri"

# One random Saturday each month
task add "Date night" anchor:"m:rand + w:sat"

# Every April 12
task add "Anniversary" anchor:"y:04-12"

# The 15th and the last business day of each month
task add "Pay bills" anchor:"m:15,-1bd"
```

Calendar rules with exclusions:

```bash
# M/W/F workouts, except Wednesdays in April
task add "Workout" anchor:"w:mon,wed,fri" omit:"w:wed + y:apr"

# M/W/F workouts, except the holiday window
task add "Workout" anchor:"w:mon,wed,fri" omit:"y:12-24..12-31"

# Random Saturday date night, excluding dates from a file
task add "Date night" anchor:"m:rand + w:sat" omit_file:"2026.csv"
```

File-backed dates:

```bash
# One day before each file date, at 12:00 and 18:00
task add "Company event prep" anchor_file:"2026.csv@-1d@t=12:00,18:00"
```

---

## The Two Recurrence Modes

### `cp`: period chains

Use `cp` when the next task should happen after a duration.

```bash
task add "Follow up" cp:3d
task add "Medication" cp:8h
task add "Variable cycle" cp:3d,20d,7d,10d,3d
task add "Loose weekly check" cp:"7d~2d"
task add "Random field inspection" cp:"rand(3d..7d)"
```

Day-based periods preserve the wall-clock routine. Sub-day periods use exact time.
Random ranges and jitter are deterministic per chain link, so retries and sync do
not create different schedules.

### `anchor`: calendar rules

Use `anchor` when the next task is tied to the calendar.

```bash
task add "Gym" anchor:"w:mon,wed,fri"
task add "Monthly report" anchor:"m:-1bd"
task add "Random weekday audit" anchor:"m:2rand + w:mon..fri"
task add "Quarter check" anchor:"y:2rand + y:apr,jul,oct"
task add "Office day" anchor:"(w:mon | m:last-fri)@t=09:00"
```

Useful syntax:

| Expression | Meaning |
|---|---|
| `w:mon,wed,fri` | Mondays, Wednesdays, Fridays |
| `m:1` | First day of each month |
| `m:-1bd` | Last business day of each month |
| `m:-1@pbd@-2bd` | Two business days before the rolled month end |
| `m:rand + w:sat` | One random Saturday each month |
| `w:2rand` | Two random days each week |
| `m:3rand + w:mon..fri` | Three random weekdays each month |
| `y:04-12` | April 12 every year |
| `w:tue,fri | y:05-05` | Tuesdays, Fridays, or May 5 |
| `(w:mon | m:last-fri)@t=09:00` | Apply the same time to a grouped expression |

For anchors, Nautical resolves dates as:

```text
(anchor ∪ anchor_file) − (omit ∪ omit_file)
```

That separation is deliberate: schedules and blackout dates are easier to reason
about when inclusion and exclusion stay separate.

---

## Presets

Reusable rules can live in `config-nautical.toml`:

```toml
[anchor_presets]
payday = "m:15,-1bd"
workout = "w:mon,wed,fri"

[omit_presets]
holidays = "y:12-24..12-31"
april = "y:apr"
```

Use them with `@name`:

```bash
task add "Pay bills" anchor:"@payday"
task add "April workouts" anchor:"@workout + y:apr"
task add "Workout except April" anchor:"@workout" omit:"@april"
```

Anchor presets and omit presets are separate namespaces.

---

## File-Backed Dates

Configure trusted folders:

```toml
anchor_file_dir = "/home/user/.task/nautical_anchors"
omit_file_dir = "/home/user/.task/nautical_omits"
```

Then reference CSV or plain date files:

```bash
task add "Company event prep" anchor_file:"2026.csv@-1d@t=12:00,18:00"
task add "Hybrid schedule" anchor:"w:tue,fri | y:05-05" anchor_file:"2026.csv@-1d"
task add "Workout" anchor:"w:mon,wed,fri" omit_file:"holidays.csv"
```

File modifiers are practical:

| Modifier | Meaning |
|---|---|
| `@-1d` | Shift each file date one day earlier |
| `@+2d` | Shift each file date two days later |
| `@-2bd` | Shift each file date two business days earlier |
| `@+2bd` | Shift each file date two business days later |
| `@nbd` | Roll to next business day |
| `@pbd` | Roll to previous business day |
| `@bd` | Keep only business days |
| `@t=09:00,17:00` | Create times on `anchor_file` dates |

`omit_file` is date-based only; it intentionally does not support `@t=`.

When combined, modifiers run in this order: roll, calendar-day offset, then
business-day offset. Business days currently mean Monday through Friday.

---

## Limits, Sync, And Safety

Stop chains by count or date:

```bash
task add "Short experiment" cp:1d chainMax:3 due:today
task add "Anniversary prep" anchor:"y:04-12" chainUntil:2028-04-12
```

Nautical is built for synced Taskwarrior setups. It tracks chain identity,
previous/next links, and equivalent children to avoid duplicate next tasks when
multiple devices are involved.

The completed task is the source for the next one. If you change or annotate a
task before completing it, those changes carry forward. Changes made on another
device after completion cannot be retroactively copied into an already spawned
child.

---

## Panels And Diagnostics

Nautical explains what it is doing in add, completion, delete, and end-of-chain
panels. Depending on the task, you may see:

- the next generated date
- the parsed recurrence rule
- omitted dates in the timeline
- file-backed sources
- chain limits
- chain integrity warnings
- timezone/config/file-source diagnostics when something is degraded

For a deeper read-only check:

```bash
nautical doctor
nautical doctor --json
```

`nautical doctor` checks hooks, UDAs, config, file directories, queue state,
duplicate chain slots, and broken lineage.

---

## Installation

Quick install:

```bash
# 1. Install hooks
cd ~/.task/hooks
curl -LO https://github.com/catanadj/taskwarrior-nautical/raw/main/on-{add,modify,exit}-nautical.py
chmod +x on-*.py

# 2. Install the shared Nautical package next to Taskwarrior data
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

Install the command helper from a checkout:

```bash
mkdir -p ~/.local/bin
ln -s "$PWD/nautical" ~/.local/bin/nautical
nautical doctor
```

See the [systems manual](./Taskwarrior-Nautical-v4-Systems-Manual.pdf) for full
installation and configuration details.

---

## Documentation

- [Systems Manual PDF](./Taskwarrior-Nautical-v4-Systems-Manual.pdf)
- [Cheatsheet PDF](./Taskwarrior-Nautical-v4-CheatSheet.pdf)
- [Releases](https://github.com/catanadj/taskwarrior-nautical/releases)
- [Issues](https://github.com/catanadj/taskwarrior-nautical/issues)

---

## Is Nautical For You?

If you only need a simple daily or weekly repeat, Taskwarrior's native recurrence
may be enough.

Nautical is useful when recurrence needs real calendar logic: business days,
random-but-repeatable choices, blackout dates, file-backed schedules, sync safety,
and clear timelines.

Hard limit: Nautical will not spawn tasks after `9999-12-31`.

---

## Feedback

Issues and real-world recurrence edge cases are welcome, especially around:

- syntax that feels awkward in daily use
- sync behavior across devices
- schedules that are hard to express cleanly

If Nautical is useful to you, support is appreciated:

[Buy me a coffee](https://buymeacoffee.com/catanadj)
