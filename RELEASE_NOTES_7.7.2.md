# Nautical 7.7.2

Nautical 7.7.2 adds named calendar boundaries for `chainUntil` and fixes adaptive
completion-period history handling.

## Highlights

- `chainUntil` now accepts named date markers such as `christmas`, `easter`,
  `summer-start`, and `summer-end`. Markers resolve once to the local end of the
  named date, and Taskwarrior stores the resolved timestamp as a string UDA.
- Doctor checks and fixtures recognize the `chainUntil` string UDA, and invalid
  values retain actionable validation guidance.
- Fixed `cp:median` history reconstruction so reconciliation does not count a
  link twice, and completion can use the pending parent as the adjacent
  predecessor when calculating the next interval.
- Recurrence timeline panels distinguish expired links from completed links and
  show the current adaptive `cp:avg` or `cp:median` interval.
- Updated the documentation with the new date markers and adaptive completion
  period behavior.
