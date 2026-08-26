# Seasonal Selection

Seasonal selectors choose positions from deterministic candidate dates inside
a season.

## Select a position

```bash
task add "Spring planning" \
  anchor:"(w:mon)@in-spring=first,last"

task add "Winter review" \
  anchor:"(w:fri)@in-winter=last@t=09:00"

task add "Every season" \
  anchor:"(w:mon)@in-season=1st"
```

The candidate expression must be parenthesized. Nautical collects all matching
dates in the seasonal window, selects positions, and then applies modifiers.

## Boundary modes

Configure the hemisphere and boundary model:

```toml
season_hemisphere = "north"       # north or south
season_mode = "astronomical"      # fixed or astronomical
tz = "Europe/Bucharest"
```

`fixed` uses conventional three-month calendar windows. In the northern
hemisphere, spring is March through May, summer June through August, autumn
September through November, and winter December through February. The southern
hemisphere mapping is inverted.

`astronomical` calculates the year's equinox and solstice transition instants,
converts them to the configured timezone, and assigns the complete local
transition date to the new season. This date-level rule keeps windows
contiguous and avoids splitting one task day between seasons.

Winter is identified by its December start year and may end in the following
calendar year.

## Other positional scopes

The same mechanism supports week, month, quarter, and year:

```bash
task add "Last weekly workday" \
  anchor:"(w:mon..fri)@in-week=last"

task add "Quarter handoff" \
  anchor:"(w:mon)@in-quarter=last@+1bd"
```

A position that does not exist contributes no date. The scheduler advances to
the next period instead of fabricating a match.
