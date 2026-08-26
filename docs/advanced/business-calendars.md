# Business Calendars

Without `bc`, Nautical treats Monday through Friday as open. A named business
calendar can add exceptional open dates and remove holidays or closures.

## Define a calendar

```toml
[business_calendar.work]
anchor = "w:mon..fri"
omit = ["y:01-01", "y:12-25"]
# anchor_file = ["extra-open-days.csv"]
# omit_file = ["holidays.csv", "company-closures-*.csv"]
```

Select it on a task:

```bash
task add "Submit payroll" anchor:"m:-1bd@t=16:00" bc:work
```

The calendar's open dates are:

```text
(anchor union anchor_file) - (omit union omit_file)
```

Calendar fields accept one string or an array. File sources use the trusted
directories configured for ordinary anchor and omit files.

## Operations affected

The selected calendar controls:

- monthly business-day ordinals such as `m:5bd` and `m:lbd`;
- `@bd`, `@nbd`, `@pbd`, and `@nw`;
- `@+Nbd` and `@-Nbd`;
- business-day file modifiers;
- previews, queries, completion, and recovery.

Calendar names are case-insensitive and normalized when stored. An unknown name
is rejected with the available configured names.

## Stable definitions

A calendar defines date membership, so its own rules cannot depend on random
selection, stepped `/N` cadence, times, business-day ordinals, or business-day
modifiers. Nautical rejects such circular definitions.

Configuration fingerprints include resolved rules and file data. Editing a
calendar invalidates affected schedule caches instead of reusing stale hints.

## Example with closures

```toml
anchor_file_dir = "/home/me/.config/nautical/calendars"
omit_file_dir = "/home/me/.config/nautical/calendars"

[business_calendar.company]
anchor = "w:mon..fri"
anchor_file = "weekend-openings.csv"
omit_file = ["public-holidays.csv", "office-closures-*.csv"]
```

```bash
task add "Company month end" anchor:"m:lbd@t=16:00" bc:company
```
