# Limits and Expiration

Nautical distinguishes whole-chain limits from Taskwarrior's native
per-occurrence expiration.

## Limit by link count

```bash
task add "Bootcamp" anchor:"w:mon,fri" chainMax:6
task add "Calibration" cp:33h chainMax:5 due:today+12h
```

`chainMax:N` includes the Nth link and then stops.

## Limit by target time

```bash
task add "Accounts payable" \
  anchor:"m:1@nbd" \
  chainUntil:2030-12-31T17:00
```

`chainUntil` is a local date or datetime. An occurrence is eligible only when
its recurrence target is at or before that boundary. Nautical rejects a root
whose first target is already beyond the limit. If both `chainMax` and
`chainUntil` exist, the earlier limit wins.

## Expire one occurrence

Taskwarrior's native `until` expires the current task; it does not end the
chain.

```bash
task add "Take trash out" due:today until:eow cp:7d chainUntil:eoy
```

By default, a child keeps the same calendar-day distance and local expiration
clock relative to its own due or scheduled-only target. Native `until` must be
strictly later than that target.

Add one second when exact elapsed carry is required:

```bash
task add "Short response" due:today+09:00 until:eod+1s cp:1d
```

Native `until` cannot be combined with `anchor_mode:all` or `flex`, because
those modes intentionally retain missed occurrences. Use `skip`, or use
`chainUntil` to stop the whole chain.

## Completion, deletion, and expiration

- Completing a pending link advances recurrence.
- Automatic deletion at its native `until` advances recurrence from the
  original target.
- Intentional deletion before `until` stops recurrence.
- `chain:off` prevents another successor.

If an expiration occurs on a client without current hooks, inspect recovery
with `nautical reconcile` before applying it.
