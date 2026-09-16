# Nautical 7.5.1

Nautical 7.5.1 is a focused reliability bug-fix release following 7.5.0.

## Fixed

- Corrected anchor-file recurrence handling across spring-forward DST gaps.
  Equivalent wall-clock slots are now deduplicated by their resolved instant.
- Preserved distinct fall-back DST folds instead of collapsing legitimate
  occurrences.
- Kept retained anchor-file occurrences projected in the configured local
  timezone after UTC ordering and deduplication.
- Made interrupted on-exit drains user-friendly: Ctrl-C now produces a compact
  recovery panel instead of exposing a traceback, with guidance to run
  `nautical reconcile --apply`.
- Added recurrence-update panels for direct `until`, `wait`, and `scheduled`
  edits, while keeping `due` as the root for coordinated temporal shifts.
- Fixed astronomy anchor offsets so hour and minute modifiers are accepted and
  preserved in the natural-language recurrence description.

## Verification

- Full local unittest suite passes (1,243 tests, 3 skipped).
- Package and boundary mypy checks pass.
- Anchor-file DST gap, fold ordering, and fold description contracts pass.

## Upgrade

Upgrade through the normal Nautical installer. Existing Taskwarrior data and
configuration are preserved; no data migration is required.
