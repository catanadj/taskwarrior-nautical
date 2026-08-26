# Diagnostics and Performance

## One-command diagnostics

Set `NAUTICAL_DIAG=1` for configuration discovery and startup evidence:

```bash
NAUTICAL_DIAG=1 nautical doctor
NAUTICAL_DIAG=1 nautical navigator --self-check
```

Diagnostics go to stderr. Hook and query JSON remains unpolluted on stdout.

Persist bounded structured diagnostics only when needed:

```bash
NAUTICAL_DIAG_LOG=1 task 42 done
NAUTICAL_DIAG_LOG_MAX_BYTES=262144 task 42 done
```

The JSONL log is stored at `TASKDATA/.nautical_diag.jsonl` and rotates at the
configured size.

## Timing

```bash
NAUTICAL_PROFILE=1 task 42 done
```

Profiling writes lightweight timing to stderr. Use it for a short reproduction,
then disable it.

## Slow terminals and Termux

Choose a presentation mode in `config-nautical.toml`:

```toml
panel_mode = "fast"       # rich, live, fast, line, compact, minimal, or text
exit_progress = true
fast_color = true
```

- `rich` uses static Rich panels.
- `live` reveals eligible rows within one bounded duration.
- `fast` avoids Rich layout work.
- `line`, `compact`, and `minimal` reduce presentation work and output.
- Captured/non-interactive output automatically avoids animation.

Enable `enable_anchor_cache=true` for file-backed and expensive preview hints.
Use scoped query/reconcile commands instead of full history for routine work.

## Development benchmarks

The repository includes release-verification tools under `dev_tools/`:

```bash
python3 dev_tools/nautical_perf_budget.py --json
python3 dev_tools/nautical_perf_budget.py --json --enforce
```

These benchmarks cover parsing, scheduling, hooks, lifecycle application,
queue processing, and reconcile. They create isolated Taskwarrior data and are
not user maintenance commands.

## What to collect for a report

- Nautical release, platform, Python, and `task --version`;
- exact command and expected behavior;
- affected task export with private content removed;
- scoped Doctor, query, queue-status, or reconcile JSON;
- stderr produced with `NAUTICAL_DIAG=1`.
