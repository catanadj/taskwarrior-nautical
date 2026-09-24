# Guided Manual Review

Nautical keeps lifecycle and chain-integrity decisions inside its typed review
workflow. Use this workflow instead of inspecting raw Taskwarrior exports or
editing `nextLink`, `prevLink`, or `link` by hand.

## Select the next item

```bash
nautical review --next
```

In human mode this stays open as a guided loop. At each item, choose `s` to
leave it unresolved, `r` to attempt the guarded already-applied resolution,
`n` to move on, or `q` to quit. JSON mode remains single-shot:

```bash
nautical review --next --json
```

The command presents one unresolved review item with:

- the chain and affected link range;
- redacted parent and successor identities;
- the expected successor and any known occupants;
- the most relevant task date/time and description when the task still exists;
- the reason automatic processing stopped;
- a confirmation token and the actions currently supported for that item.

Use JSON when integrating the review flow with another tool:

```bash
nautical review --next --json
```

The JSON output is bounded to one item when `--next` is used. Full UUIDs are
not displayed in the review projection; Nautical retains the authoritative
values internally for guarded verification.

## Safe actions

Actions require the exact confirmation token shown by the current review
snapshot:

```bash
nautical review \
  --intent li1-... \
  --action skip \
  --confirm TOKEN
```

`skip` leaves the item unresolved and records no chain mutation.

`resolve-applied` is available only when Nautical has independently verified
that the expected successor and parent link already converge:

```bash
nautical review \
  --intent li1-... \
  --action resolve-applied \
  --confirm TOKEN
```

If the state changes after the review screen is displayed, the confirmation
token becomes invalid and the action is refused. This prevents decisions based
on stale synchronization data.

## Ambiguous chains

When multiple tasks occupy the same `(chainID, link)` slot, Nautical presents
the occupants and keeps the item in `manual_review`. It does not offer an
automatic branch-selection or deletion action. Re-run the review after the
underlying synchronization state has settled, or use the integrity query to
understand the chain-level findings:

```bash
nautical query integrity --chain-id CHAIN_ID
```

Do not modify chain fields manually. If no safe action is offered, leave the
item unresolved and preserve the review evidence for the next reconciliation
pass.

## Relationship to queue review

`nautical queue-review` remains available for compatibility and for listing
multiple intents. `nautical review --next` is the guided, one-item workflow.
Lifecycle outbox records are shown first. When the outbox has no review record,
the command also projects current authoritative integrity findings (for
example, duplicate chain slots) into the same bounded review format. These
projected IDs are read-only views; the integrity query remains the source of
truth and the outbox remains the durable source for lifecycle review state.
See [Lifecycle Outbox](../tools/lifecycle-outbox.md) for storage and retention.
