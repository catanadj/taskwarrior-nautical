![Nautical Banner](./nautical-banner.svg)

# Taskwarrior-Nautical ⚓︎⛓
**Real-world recurrence for Taskwarrior** - without cloud lock-in or drift.

``` bash
# Last Friday of every month
task add "Monthly review" anchor:"m:last-fri"

# Every 33 h, wall-clock preserved
task add "Take Vitamin" cp:33h due:now+12h

# 2nd Mon or first working day before 15th
task add "Board prep" anchor:"m:2mon | m:15@pbd"
```

<p align="center">
  <a href="https://asciinema.org/a/123456" target="_blank">
  <img src="https://asciinema.org/a/123456.svg" width="700">
  </a>
</p>

---

## What & Why

Traditional repeat syntax breaks the moment life gets messy:

- “Every 2nd Monday and 5th Friday”
- “Last working day **of the quarter**”
- “Random weekday every 3 months”
- "Monday at 09:00, Friday at 11:15 and 18:30, Saturday at 15:00"

Nautical gives you a **single line that just works**, keeps your data **local**, and never drifts because it stores **no floating-point time**.

---

## Install (≤ 60 s)

```bash
# 1. Drop the hooks in place
cd ~/.task/hooks
curl -LO https://github.com/catanadj/taskwarrior-nautical/raw/main/on-{add,modify,exit}-nautical.py
chmod +x on-*.py
cd ..
curl -LO https://github.com/catanadj/taskwarrior-nautical/raw/main/nautical_core.py

# 2. Add UDAs (copy/paste block)
curl -s https://raw.githubusercontent.com/catanadj/taskwarrior-nautical/main/uda.conf >> ~/.taskrc

# 3. Optional pretty panels
pip install rich

# 4. Test drive
task add "System test" anchor:"m:2mon" due:today
```

You should see a colour panel showing the next occurrences.
Done - you're ready for the next section.

---

## 30-Second Tutorial

| Pattern you need                                  | One line                             |
| ------------------------------------------------- | ------------------------------------ |
| Last Friday monthly                               | `anchor:"m:last-fri"`                |
| 1st & 15th (nearest weekday)                      | `anchor:"m:1@nw,m:15@nw"`            |
| Every 8 h (exact)                                 | `cp:8h`                              |
| Random weekday in period 20 and last day of month | `anchor:"m:rand@bd + m:20..-1"`      |
| Quarterly on 15-Jan,15-Apr…                       | `anchor:"y:01-15,04-15,07-15,10-15"` |

---
## What you get (and what you don’t)

Feature matrix (✓ = first-class, ◐ = partial, ✗ = missing)
| Capability                                                             | TW+Nautical | Todoist            | TickTick | Notion | OmniFocus       | Things  |
| ---------------------------------------------------------------------- | ----------- | ------------------ | -------- | ------ | --------------- | ------- |
| **Open source & self-host**                                            | ✓           | ✗                  | ✗        | ✗      | ✗               | ✗       |
| Plain-text storage                                                     | ✓           | ✗                  | ✗        | ✗      | ✗               | ✗       |
| Off-line first                                                         | ✓           | ◐                  | ◐        | ◐      | ✓               | ✓       |
| **Arbitrary date DNF** (last Mon of quarter, every 3rd Tue except Dec) | ✓           | ✗                  | ✗        | ✗      | ◐ (Defer)       | ✗       |
| **Multi-time per day** (`@t=09:00,17:30`)                              | ✓           | ✗                  | ✗        | ✗      | ✗               | ✗       |
| **Skip / Flex / Back-fill modes**                                      | ✓           | ✗                  | ✗        | ✗      | ✓ (FIFO)        | ✗       |
| **Chain caps** (max N or until-date)                                   | ✓           | ◐ (only “N times”) | ◐        | ◐      | ✓               | ✓       |
| **Sub-day periods** (`cp=6h30m`)                                       | ✓           | ✗                  | ✗        | ✗      | ✗               | ✗       |
| **Scriptable / CLI**                                                   | ✓           | ◐ (REST)           | ◐        | ◐      | ◐ (AppleScript) | ◐ (URL) |
| **Hook ecosystem** (user code on add/modify/done)                      | ✓           | ✗                  | ✗        | ✗      | ✗               | ✗       |
| **Speed** (add task ≤ 50 ms)                                           | ✓           | ✓                  | ✓        | ◐      | ✓               | ✓       |

---

## Anchors (calendar logic)

<details>
<summary>Click for mini-language cheat-sheet</summary>

| Token | Meaning |
|-------|---------|
| `w:mon,wed,fri` | Mon Wed Fri |
| `m:1,15` | 1st & 15th of month |
| `m:last-fri` | last Friday |
| `m:3wed` | 3rd Wednesday |
| `m:1@nw` | 1st, roll to nearest weekday |
| `m:1@pbd` | 1st, roll to **prev** business day |
| `m:1..7@bd` | bucket 1-7, business day only |
| `m:rand` | one random day/month (seeded) |
| `@t=09:00,17:00` | multi-time per day |
| `&` / `,` | AND / OR (parentheses allowed) |

Full grammar & more examples → [Nautical-Manual.pdf](https://github.com/catanadj/taskwarrior-nautical/blob/main/TW-Nautical-Manual.pdf)
</details>

---

## Chains (completion-based)

<details>
<summary>Click for examples</summary>

```bash
# 12 d after completion, keep 09:00 wall clock
task add "mow lawn" cp:12d due:tomorrow+9h

# 33 h exact (drift-free)
task add "meds" cp:33h due:now+12h

# stop after 5th occurrence
task add "calibrate" cp:3d chainMax:5 due:today
```
</details>

---

## Anchor Modes

- **skip** (default) - missed? move on
- **all** - back-fill every missed slot
- **flex** - skip past, anchor future

---

## Panels & Timeline

Every add/completion prints a compact panel:

```
⚓︎ Next anchor  #2  a4bf5egh → 8c31d2ss
Pattern: m:last-fri  SKIP
First due: Fri 2024-06-28 09:00 (in 3 d)
Upcoming: 02 ▸ Fri 2024-07-26 09:00
          03 ▸ Fri 2024-08-30 09:00
Links left: 8 left (cap #10)
```

---

## Performance & Safety

- **≤ 50 ms** on Termux / old phones
- No network, no cloud, 100 % local JSON
- Atomic import - never orphans a task
- DST-safe, leap-second-safe, 64-bit time
- Plain-text backup - grep your rules!

---

## Operational Knobs

Common knobs:

- `NAUTICAL_DNF_DISK_CACHE=0` disables the on-add JSONL cache (default: enabled).
- `NAUTICAL_EXIT_STRICT=1` makes on-exit return 1 when spawns are dead-lettered or errored (for scripting).
- `NAUTICAL_DIAG=1` prints diagnostics and config search paths.
- `NAUTICAL_DIAG_LOG=1` persists structured diagnostics to `TASKDATA/.nautical_diag.jsonl`.
- `NAUTICAL_DIAG_LOG_MAX_BYTES=262144` caps the diagnostic log before rotation.
- `NAUTICAL_DURABLE_QUEUE=1` enables fsync for queue/dead-letter writes (safer, slower).
- `NAUTICAL_PROFILE=1` emits lightweight timing (stderr).
- `panel_mode="fast"` forces plain panel rendering (skip Rich).
- `panel_mode="line"` shows a single summary line inside a compact panel.
- `fast_color=false` disables ANSI in fast panels.
- `spawn_queue_max_bytes` caps deferred spawn queue size.
- `spawn_queue_drain_max_items` caps items drained per batch.

Data directory resolution:

- Hooks resolve Taskwarrior data dir from `TASKDATA` or hook argv (`data:` / `data.location:` in Hooks v2).
- `rc.data.location=...` is only injected for hook-spawned `task` calls when that data dir is explicit.

Dead-letter recovery (manual):

1. Inspect `.nautical_dead_letter.jsonl` in `TASKDATA` and pick entries to retry.
2. Append those JSON lines to `.nautical_spawn_queue.jsonl` (one entry per line).
3. Optionally remove retried lines from the dead-letter file.
4. Run any Taskwarrior command to trigger on-exit drain.

Self-check:

```
python3 nautical_navigator.py --self-check
NAUTICAL_DIAG=1 python3 nautical_navigator.py --self-check
```

Production hardening rollout:

1. Stage 1: enable diagnostics only.
2. `NAUTICAL_DIAG_LOG=1` and optional `NAUTICAL_DURABLE_QUEUE=1`.
3. Monitor queue/dead-letter health with `python3 tools/nautical_health_check.py --json`.
4. Keep `NAUTICAL_EXIT_STRICT=0` during bake-in.
5. Stage 2: flip `NAUTICAL_EXIT_STRICT=1` after dead-letter/requeue-failure rates are stable.

Periodic alert example:

```
python3 tools/nautical_health_check.py --taskdata ~/.task --queue-crit-bytes 524288
```

Exit code semantics:

- `0` = healthy
- `1` = warning
- `2` = critical

Automation templates:

- `tools/ops/nautical-health-check.crontab`
- `tools/ops/nautical-health-check.service`
- `tools/ops/nautical-health-check.timer`
- `tools/ops/nautical_health_check_cron.sh`
- `tools/ops/README.md` (install steps)

---

## Performance Checklist

- Enable disk cache: default on (disable only if debugging).
- Use `panel_mode="fast"` on slow terminals or mobile.
- If you see slowdowns, run `NAUTICAL_PROFILE=1` for a short session.
- For heavy workloads, raise `spawn_queue_max_bytes` and `spawn_queue_drain_max_items`.

---

## Load Testing

Use `tools/load_test_nautical.py` to measure performance on your machine:

```
python3 tools/load_test_nautical.py --tasks 2000 --concurrency 4
python3 tools/load_test_nautical.py --ramp --ramp-start 200 --ramp-step 500 --ramp-max 10000 --concurrency 16
python3 tools/load_test_nautical.py --ramp --done-only --ramp-start 200 --ramp-step 500 --ramp-max 10000 --concurrency 16
python3 tools/load_test_nautical.py --rate-ramp --rate-secs 30 --rate-start 5 --rate-step 5 --rate-max 100
```

What each mode does:

- Batch: fixed number of adds (and optional dones), report latency stats.
- Ramp: increase task count per stage until thresholds are hit.
- Done-only: measure on-modify performance by completing tasks created in the stage.
- Rate-ramp: increase target ops/sec and report throughput and latency limits.

## Requirements

- Taskwarrior ≥ 2.6
- Python ≥ 3.9
- `rich` (optional, for colour panels)

---

## Links

| [Full Manual (PDF)](https://github.com/catanadj/taskwarrior-nautical/blob/main/TW-Nautical-Manual.pdf) | [Examples Gallery](https://github.com/catanadj/taskwarrior-nautical/wiki/Pattern-Gallery) | [Report Bug](https://github.com/catanadj/taskwarrior-nautical/issues) |
|---|---|---|

---
## Support

If you find this tool helpful, any support will be greatly apreciated.

You can do so [here](https://buymeacoffee.com/catanadj). Thank you.

---

**Stop thinking about scheduling - start doing.**
⚓︎ *Deus vult.*
