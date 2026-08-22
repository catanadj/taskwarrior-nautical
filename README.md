![Nautical Banner](./nautical-banner.svg)

![Nautical demo](https://github.com/user-attachments/assets/8420c1a8-907b-483e-86ec-4385eec892e3)

# Taskwarrior Nautical

Nautical is a full recurrence engine for Taskwarrior. Give it a compact routine
rule, then complete tasks as usual; Nautical creates each next occurrence as a
normal Taskwarrior task that remains visible, editable, and in sync with the
regular `task` command.

Simple routines stay simple. When a routine follows business days, exception
dates, multiple times, changing intervals, or a fixed end point, the same system
scales with it. If you can describe when something should happen, Nautical is
designed to express it.

## Install

Install the current stable release in one command:

```bash
curl -fsSL https://raw.githubusercontent.com/catanadj/taskwarrior-nautical/main/bootstrap.sh \
  | bash -s -- --version v6.5.2
```

To try the latest pushed fixes before the next stable release, use
`--version main` instead.

The bootstrap downloads a temporary pinned release, runs the validated
installer, checks the installed runtime with Doctor, and removes the checkout.
Use `--dry-run` to validate without changing anything. For Termux, pass the
launcher location explicitly:

```bash
bash bootstrap.sh --version v6.5.2 --launcher-path "$PREFIX/bin/nautical"
```

For an auditable install, download the bootstrap first, inspect it, then run
it. Set `TASKDATA` or pass `--taskdata` for a custom Taskwarrior data directory.
The lower-level `./nautical install` command remains available for local release
trees and repairs.

The installer also creates `uda-nautical.conf` and adds its include to your
Taskwarrior rc file when the fields are not already configured. Existing UDA
files are preserved. It creates `~/.local/bin/nautical` as the user-facing
launcher and runs a Doctor check after installation.

On Termux, use its executable directory explicitly when installing:

```bash
./nautical install --launcher-path "$PREFIX/bin/nautical"
```

For a custom data directory, set `$TASKDATA` and substitute that path for
`~/.task` above.

Install Navigator and the formatted-panel dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The installer writes an explicit local IANA timezone to a new installation's
`config-nautical.toml`. Existing configurations are preserved; Doctor reports
when a timezone is missing or unavailable.

Create a first task:

```bash
task add "Weekly review" anchor:"w:mon"
```

Complete it with `task <id> done`; Nautical queues the next Monday review.

When hooks are interrupted or a task is changed outside Nautical, the system
can audit the chain and safely recover missing links through `nautical doctor`
and `nautical reconcile`.

## Two Ways To Repeat

Use `cp` when the next task follows a period from completion:

```bash
task add "Mow lawn" cp:12d
task add "Take vitamin" cp:8h
```

Use `anchor` when it belongs on the calendar:

```bash
task add "Workout" anchor:"w:mon,wed,fri"
task add "Monthly report" anchor:"m:1"  # first day of the month
task add "Anniversary" anchor:"y:04-12"
```

That is enough to begin. `cp` is for “after this long”; `anchor` is for “on
these dates.” Together they are a small doorway into routines as precise and
individual as real life requires. The manual holds the deeper grammar and
ready-to-adapt examples whenever you need them.

## Learn More

- [Systems Manual PDF](./Taskwarrior-Nautical-v6-Systems-Manual.pdf): setup, configuration, grammar, examples, and recovery.
- [Cheatsheet PDF](./Taskwarrior-Nautical-v6-CheatSheet.pdf): quick anchor and period reference.
- [Releases](https://github.com/catanadj/taskwarrior-nautical/releases)
- [Issues](https://github.com/catanadj/taskwarrior-nautical/issues)

If Nautical is useful to you, support is appreciated:

[Buy me a coffee](https://buymeacoffee.com/catanadj)
