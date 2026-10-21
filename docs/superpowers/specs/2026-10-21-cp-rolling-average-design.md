# Rolling-average completion periods

## Goal

Add a history-derived completion period expressed as `cp:avg(x)` without changing the behavior of fixed, random, jittered, or sequenced CP values.

## Semantics

- `cp:avg(x)` accepts a positive integer window `x`.
- A missing `due` on add is seeded to the current hook time plus one day only for `cp:avg(x)`.
- Fixed-period CP keeps its existing missing-due behavior (`entry + cp`).
- Before two completed links with valid completion timestamps exist, the average mode uses a one-day bootstrap interval.
- Once history exists, intervals are measured between adjacent completed links using their `end` timestamps, ordered by numeric `link`.
- The latest `x` valid adjacent intervals are averaged in exact seconds; the result is applied from the completing task's `end`.
- Non-adjacent links, malformed timestamps, incomplete history, and unavailable history fail closed and produce the existing manual-review path rather than guessing.
- The same resolver is used by completion and reconcile so both paths produce identical child due values.

## Boundary

Parsing identifies average mode and its window; chain generation owns history lookup and interval calculation. The authoritative chain repository is injected into generation rather than accessed through hook globals. Add-preview/validation render and validate the new form without attempting to predict a history-derived interval.

## Non-goals

No changes to fixed CP semantics, Taskwarrior recurrence fields, or the existing `cp` sequence grammar beyond recognizing the new `avg(x)` token.
