# Navigator

Navigator explains anchor syntax and analyzes existing chains without changing
Taskwarrior.

## Explain an expression

```bash
nautical navigator --explain "m:last-fri@t=09:00"
nautical navigator --explain "moon:full@t=moonrise" --trace
```

The analysis shows normalized natural language, current reference, future
matches, local timezone, and optionally a concise scheduler decision trace.

Expression-only explanation uses a synthetic recurrence context where needed.
It does not require an existing Taskwarrior chain.

## Validate syntax

```bash
nautical navigator --validate "w:mon..fri@t=09:00,17:00"
```

Validation includes parsing, contradiction checks, configuration-dependent
calendar rules, and bounded satisfiability checks.

## Inspect a chain

Launch the interactive picker:

```bash
nautical navigator
```

Or start from one Taskwarrior task:

```bash
nautical navigator --id 42 --mode chain --count 20
nautical navigator --id 42 --mode task
```

Use `--vertical` for Termux or narrow terminals and `--horizontal` for a wide
timing chart.

## Self-check

```bash
nautical navigator --self-check
NAUTICAL_DIAG=1 nautical navigator --self-check
```

Self-check covers optional dependencies, configuration discovery, timezone,
and provider readiness. Diagnostics are written to stderr only when enabled.
