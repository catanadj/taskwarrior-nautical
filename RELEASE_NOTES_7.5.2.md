# Nautical 7.5.2

Nautical 7.5.2 is a reliability-focused maintenance release following 7.5.1.

## Fixed

- Detect malformed lifecycle outbox databases before enqueueing new work.
- Preserve corrupted SQLite databases and their WAL/SHM sidecars in a durable,
  timestamped quarantine instead of discarding them.
- Automatically retry the current lifecycle enqueue against a fresh outbox
  after successful quarantine, allowing task completion to continue.
- Publish quarantine manifests atomically with filesystem durability barriers.
- Recover stale quarantine lock markers left behind by interrupted processes
  while keeping active recovery fail-closed.
- Surface successful-but-recovered completions with a warning and guidance to
  run `nautical reconcile --apply`.
- Add diagnostics counters for integrity recovery, successful quarantine, and
  quarantine failures.

## Verification

- Focused lifecycle, structured-failure, and feedback tests pass (55 tests).
- Affected production modules and tests pass mypy.
- Python compilation and whitespace checks pass.

## Upgrade

Upgrade through the normal Nautical installer. Existing Taskwarrior data and
configuration are preserved; corrupted outbox state is retained under a
timestamped quarantine directory if automatic recovery is required.
