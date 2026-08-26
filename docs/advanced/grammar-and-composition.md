# Grammar and Composition

Nautical has separate period grammar for `cp` and calendar grammar for
`anchor` and `omit`.

## Completion periods

```text
cp = period
   | period,period,...
   | period*N
   | rand(period..period)
   | period~spread
```

Examples: `3d`, `7d*3,14d`, `rand(3d..7d)`, and `14d~2d`.

## Calendar expressions

```text
expression = branch ("|" branch)*
branch     = factor ("+" factor)*
factor     = atom
           | "(" expression ")" [positional-selector] ("@" modifier)*
atom       = family ["/N"] ":" spec ("@" modifier)*
family     = "w" | "m" | "y" | "moon"
```

- `|` means either branch may match.
- `+` means every factor must match the same date.
- `+` binds tighter than `|`.
- A comma is a list inside one atom, not a top-level OR.
- Whitespace is ignored inside expressions.

```bash
task add "April routine" anchor:"w:mon,wed,fri + y:apr"
task add "Either date" anchor:"m:1 | m:last-fri"
task add "Shared time" anchor:"(w:mon | m:last-fri)@t=09:00"
```

## Selection before transformation

Positional selectors operate on a complete deterministic candidate set:

```text
(candidate expression)@in-<scope>=<positions>
```

Supported scopes are `week`, `month`, `quarter`, `year`, `season`, `spring`,
`summer`, `autumn`, and `winter`. Positions include `first`, `last`, `3rd`,
`2nd-last`, and comma lists.

```bash
task add "Month review" \
  anchor:"(w:tue | w:thu)@in-month=first,last@t=09:00"
```

Evaluation order is:

```text
collect candidates -> select positions -> apply modifiers
```

Random candidate atoms, candidate-side modifiers, nested positional selectors,
and `@bd` after selection are rejected because the candidate set must remain
stable.

## Modifier order

Nautical applies date transforms by meaning, not textual position:

```text
business/weekday roll -> calendar-day offset -> business-day offset
```

Time selection is then resolved for the resulting local date. Date modifiers
on an AND group must stay on individual atoms; distributing them across an
intersection could create false matches.

## Omissions

`omit` uses calendar grammar, but it removes whole local dates. Time modifiers
are invalid in omissions. The combined recurrence is:

```text
(anchor union anchor_file) - (omit union omit_file)
```

## Validation

Nautical rejects malformed dates, contradictory intersections, invalid
positions, ambiguous modifier placement, resource-limit violations, and
search exhaustion. These are scheduling failures, not invitations to fabricate
an occurrence.

Use Navigator before adding a complex expression:

```bash
nautical navigator --validate "(w:mon | w:fri)@in-month=last@t=09:00"
nautical navigator --explain "(w:mon | w:fri)@in-month=last@t=09:00"
```
