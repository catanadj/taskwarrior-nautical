![Nautical Banner](./nautical-banner.svg)

![Nautical demo](https://github.com/user-attachments/assets/8420c1a8-907b-483e-86ec-4385eec892e3)

# Taskwarrior Nautical

Nautical gives Taskwarrior practical recurring tasks. Complete a Nautical task
as usual and it creates the next normal Taskwarrior task for you. The tasks
remain visible, editable, and syncable with the regular `task` command.

Use it for simple repeating work today, then reach for the manual when a
schedule needs calendar rules, business days, exclusions and all other advanced details.

## Start Here

Install the hooks and shared package beside your Taskwarrior data:

```bash
cd ~/.task/hooks
curl -LO https://github.com/catanadj/taskwarrior-nautical/raw/main/on-{add,modify,exit}-nautical.py
chmod +x on-*.py

cd ..
curl -L https://github.com/catanadj/taskwarrior-nautical/archive/refs/heads/main.tar.gz \
  | tar -xz --strip-components=1 taskwarrior-nautical-main/nautical_core

curl -s https://raw.githubusercontent.com/catanadj/taskwarrior-nautical/main/uda.conf >> ~/.taskrc

# Install the nautical command launcher
curl -Lo ~/.task/nautical https://raw.githubusercontent.com/catanadj/taskwarrior-nautical/main/nautical
chmod +x ~/.task/nautical
mkdir -p ~/.local/bin
ln -sf ~/.task/nautical ~/.local/bin/nautical
```

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
```

That is enough to begin. `cp` is for “after this long”; `anchor` is for “on
these dates.”

## Learn More

- [Systems Manual PDF](./Taskwarrior-Nautical-v5-Systems-Manual.pdf): setup, configuration, grammar, examples, and recovery.
- [Cheatsheet PDF](./Taskwarrior-Nautical-v5-CheatSheet.pdf): quick anchor and period reference.
- [Releases](https://github.com/catanadj/taskwarrior-nautical/releases)
- [Issues](https://github.com/catanadj/taskwarrior-nautical/issues)

If Nautical is useful to you, support is appreciated:

[Buy me a coffee](https://buymeacoffee.com/catanadj)
