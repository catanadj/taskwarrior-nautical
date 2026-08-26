# Doctor

Doctor is the first command to run when installation, scheduling, or chain
state looks wrong. It is read-only except when `--clean-cache` is requested.

## Installation check

```bash
nautical doctor --installation-only
```

This checks the platform, Taskwarrior command and data directory, managed
runtime, hooks, launcher, UDA registration, configuration, timezone, and
optional dependencies without auditing all task chains.

## Operational audit

```bash
nautical doctor
nautical doctor --json
```

Default human output prioritizes pending or otherwise actionable tasks.
Historical findings from completed/deleted links are summarized rather than
printed repeatedly. JSON retains full evidence for tooling and audits.

Findings include:

- stable identifier and severity;
- affected chain and UUIDs;
- observed and expected evidence;
- whether a safe repair exists;
- a concrete next action when user action is possible.

## Exit status

| Exit | Meaning |
| ---: | --- |
| `0` | Checks passed |
| `1` | Attention or warnings |
| `2` | Operational failure or unavailable required evidence |

Do not treat a warning as a mutation instruction. When a chain finding is
repairable, review the [reconcile](reconcile.md) dry run.

## Cache maintenance

```bash
nautical doctor --clean-cache
```

This prunes expired and orphaned anchor cache files. It does not repair task
chains or lifecycle intents.
