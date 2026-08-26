# Astronomy

Astronomy is optional. It supports solar and lunar event times plus moon-phase
calendar anchors. It requires Astral and an explicit location profile.

## Configure a location

```toml
tz = "Europe/Bucharest"

[astronomy]
default_location = "home"

[astronomy.locations.home]
latitude = 44.4268
longitude = 26.1025
elevation = 75
timezone = "Europe/Bucharest"
```

Coordinates do not determine civil timezone rules. Keep both the global and
location timezone explicit and consistent with the intended schedule.

## Solar and lunar event times

```bash
task add "Morning light" anchor:"w:mon..fri@t=sunrise"
task add "Evening walk" anchor:"w:fri@t=sunset@+45m"
task add "Moon observation" anchor:"y:jul@t=moonrise"
```

Supported symbolic times are `sunrise`, `sunset`, `dawn`, `dusk`, `moonrise`,
and `moonset`. Time offsets are applied after the event is resolved.

Rise and set events can be unavailable on a particular date and location.
When a broader date expression is used, Nautical continues searching eligible
dates within the scheduler bounds. An exact unavailable date produces an
actionable scheduling error.

## Moon phases

```bash
task add "Full moon" anchor:"moon:full@t=moonrise"
task add "Friday full moon" anchor:"(moon:full + w:fri)@t=20:00"
task add "July last quarter" \
  anchor:"(moon:last-quarter + y:jul)@t=moonrise"
```

Phases are `new`, `first-quarter`, `full`, and `last-quarter`; common aliases
are accepted. Multiple matching phase windows in one year or month remain
distinct occurrences.

## Seasonal boundaries

Astronomical seasonal mode uses Nautical's focused equinox and solstice
calculator and does not require a full ephemeris engine. See
[Seasonal selection](seasons.md).

## Failure behavior

Missing Astral, an invalid profile, timezone disagreement, polar event
unavailability, or computation failure is reported as unavailable/invalid.
Hooks and reconcile fail closed instead of substituting a default time.

Verify the active profile with:

```bash
nautical doctor --installation-only
NAUTICAL_DIAG=1 nautical navigator --self-check
```
