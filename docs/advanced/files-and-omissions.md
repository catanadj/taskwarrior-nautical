# Files and Omissions

File-backed schedules let an external calendar provide explicit dates while
Nautical retains timezone, omission, lifecycle, and recovery behavior.

## Trusted directories

```toml
anchor_file_dir = "/home/me/.config/nautical/calendars"
omit_file_dir = "/home/me/.config/nautical/calendars"
```

Task values are basename-only. Absolute paths, `..`, recursive `**`, and
character classes are rejected. `*` and `?` filename patterns are supported
inside the configured directory and must match at least one file.

## File formats

Plain text accepts one value per line:

```text
# Release dates
2027-01-15
2027-02-01..2027-02-03
```

CSV requires a `date` column. Omission CSV may also contain `description`:

```csv
date,description
2027-01-01,New Year closure
2027-12-25,Holiday closure
```

Blank lines and comments are ignored. Dates and ranges are deduplicated.

## Include dates

```bash
task add "Regional events" \
  anchor_file:"north.csv@t=09:00 | south.csv@t=15:00"

task add "Prepared events" \
  anchor_file:"(public.csv | company.txt)@-1d@t=12:00"
```

File sources accept time, roll, calendar-day, and business-day modifiers.
Multiple sources are merged with expression-based anchors and duplicate local
datetimes are removed.

## Omit dates

```bash
task add "Weekday routine" \
  anchor:"w:mon..fri@t=09:00" \
  omit_file:holidays.csv

task add "No year-end work" \
  anchor:"m:rand + w:sat" \
  omit:"y:12-24..12-31"
```

`omit` and `omit_file` remove whole local dates after all inclusion streams are
merged. Omitted dates are skipped, not rolled. Omission expressions reject
`@t=` because they cannot remove one timed slot while retaining another.

## Source identity and caches

Nautical records source provenance in compiled schedules and includes file
metadata and content fingerprints in cache identity. Hot reads use bounded
metadata-aware caching; changed files are revalidated before schedule decisions.

Unavailable, unsafe, or malformed files fail closed. A missing file is not an
empty calendar.
