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

## Start Here

Download Nautical and run the installer:

```bash
git clone --depth 1 https://github.com/catanadj/taskwarrior-nautical.git
cd taskwarrior-nautical
./nautical install --dry-run
./nautical install
```

The same `./nautical install` command handles first installation, upgrades,
repairs, and safe re-runs. For later upgrades, update the checkout and run it
again:

```bash
git pull --ff-only
./nautical install
```

The installer uses `$TASKDATA` when set, otherwise `~/.task`. It validates the
complete runtime before switching releases, keeps the previous release
available during upgrades, and leaves an already-current installation
unchanged.

Register Nautical's Taskwarrior fields and expose its command launcher:

```bash
cp uda.conf ~/.task/uda-nautical.conf
grep -Fqx "include $HOME/.task/uda-nautical.conf" "$HOME/.taskrc" 2>/dev/null || \
  printf '\ninclude %s\n' "$HOME/.task/uda-nautical.conf" >> "$HOME/.taskrc"
mkdir -p ~/.local/bin
ln -sf ~/.task/nautical ~/.local/bin/nautical
~/.local/bin/nautical doctor
```

For a custom data directory, set `$TASKDATA` and substitute that path for
`~/.task` above.

Optional, for formatted panels:

```bash
pip install rich
```

Create a first task:

```bash
task add "Weekly review" anchor:"w:mon"
```

Complete it with `task <id> done`; Nautical queues the next Monday review.

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
task add "Spring planning" anchor:"(w:mon)@in-spring=first,last"
```

That is enough to begin. `cp` is for “after this long”; `anchor` is for “on
these dates.” Together they are a small doorway into routines as precise and
individual as real life requires. The manual holds the deeper grammar and
ready-to-adapt examples whenever you need them.

## Learn More

- [Systems Manual PDF](./Taskwarrior-Nautical-v5-Systems-Manual.pdf): setup, configuration, grammar, examples, and recovery.
- [Cheatsheet PDF](./Taskwarrior-Nautical-v5-CheatSheet.pdf): quick anchor and period reference.
- [Releases](https://github.com/catanadj/taskwarrior-nautical/releases)
- [Issues](https://github.com/catanadj/taskwarrior-nautical/issues)

If Nautical is useful to you, support is appreciated:

[Buy me a coffee](https://buymeacoffee.com/catanadj)
