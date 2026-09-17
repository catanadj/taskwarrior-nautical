# Nautical 7.5.4

Nautical 7.5.4 is a completion-safety and error-feedback patch release.

## Fixed

- Preserve the actionable recurrence-computation reason in the user-facing
  chain-error panel instead of replacing it with a generic message.
- Block completion when Nautical cannot compute the next recurrence timestamp,
  preventing an invalid recurring task from silently terminating its chain.
- Keep recovery-oriented lifecycle outcomes such as deferred outbox work
  available for their existing retry/manual-review handling.

## Verification

- Hook-engine and completion-computation contracts cover the completion veto and
  detailed scheduler feedback.
- Full test suite passes: 1,254 tests, with 3 optional skips.

## Upgrade

Upgrade through the normal Nautical installer. Existing Taskwarrior data and
configuration are preserved.
