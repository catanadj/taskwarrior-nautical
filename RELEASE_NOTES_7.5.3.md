# Nautical 7.5.3

Nautical 7.5.3 is a focused astronomy and installation reliability release.

## Fixed

- Skip an unavailable astronomical event on a matching date and continue to
  the next valid date in the same phase/month recurrence window.
- Prevent an unavailable astronomical event on the previous calendar date from
  invalidating a valid current occurrence during overnight-slot inspection.
- Make the bootstrapper reject installed-but-incompatible dependency versions;
  configured astronomy installations now require the pinned `astral==3.2`
  runtime instead of accepting any Astral version.

## Documentation

- Astronomy guidance now explains that phase/month expressions search the full
  calendar window and that unavailable event dates are skipped safely.

## Verification

- Astronomy and on-add hook contracts pass, including the July last-quarter at
  moonrise regression scenario.
- Bootstrap shell syntax and dependency-gate contract checks pass.

## Upgrade

Upgrade through the normal Nautical installer. Existing Taskwarrior data and
configuration are preserved.
