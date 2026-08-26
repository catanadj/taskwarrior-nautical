# Multiple Times

Attach `@t=` to an anchor atom or a parenthesized expression.

## Exact times

```bash
task add "Hydrate" anchor:"w:mon..sun@t=9,12:30,18"
task add "Different shifts" anchor:"w:mon@t=09:00,fri@t=15:00"
task add "Shared time" anchor:"(w:mon | m:last-fri)@t=09:00"
```

One-digit hours are accepted: `9` is canonicalized to `09:00`. Do not combine
a group-level time with inner `@t=` modifiers.

## Fixed-interval windows

```bash
task add "Daylight routine" anchor:"w:mon..sun@t=04:30..19:30/3h30min"
task add "Mixed windows" anchor:"w:mon..sun@t=06..12/2h,16..20/2h,22"
```

The interval can use minutes, hours plus minutes, or decimal hours when the
result is a whole number of minutes: `30m`, `3h30min`, or `3.5h`.

The end is a boundary, not a forced final slot. Starting at `04:30`, a `/3h`
window adds slots every three hours while they remain inside the window.

## Equally spaced slots

A unitless divisor means a total number of slots, including both boundaries:

```bash
task add "Three checks" anchor:"w:mon..sun@t=06..18/3"
```

This expands to `06:00`, `12:00`, and `18:00`. When the interval is not an
exact number of minutes, Nautical uses minute-precision approximations and
describes the interval with `~`.

## Windows across midnight

```bash
task add "Night watch" anchor:"w:mon..sun@t=22:30..06:30/2h"
```

Slots after midnight belong to the anchor date that opened the window. Nautical
checks the previous anchor date when finding an early-morning next occurrence,
so completion, previews, queries, Navigator, and reconcile retain the same
ownership.

## Random times

```bash
task add "Flexible practice" anchor:"w:mon..fri@t=rand(06..18)"
task add "Three random checks" anchor:"w:mon..fri@t=rand(06..18/3)"
```

The first form chooses one deterministic minute from the inclusive window. The
second divides the window into three buckets and chooses one minute from each.
See [Deterministic random schedules](../advanced/random-schedules.md).

## Astronomy event times

With an astronomy profile, `@t=` also accepts `sunrise`, `sunset`, `dawn`,
`dusk`, `moonrise`, and `moonset`, plus offsets such as:

```bash
task add "Evening walk" anchor:"w:fri@t=sunset@+45m"
```

See [Astronomy](../advanced/astronomy.md) for availability behavior.
