# Editing, Pausing, and Resuming Recurrence

Nautical treats recurrence edits as explicit lifecycle transitions. You can
change a task's recurrence rule, pause a chain, resume a paused chain, or
remove recurrence from a task. The modify hook validates the transition before
it stages any successor work.

## Activate an ordinary task

Adding `cp`, `anchor`, or `anchor_file` to an unlinked task promotes it to a
Nautical recurrence root. Nautical sets `chain:on`, derives `chainID` from the
task UUID when needed, and uses `link:1` for the root:

```bash
task 42 modify anchor:"w:mon..fri@t=09:00"
```

The task must have a UUID and must not already contain `prevLink` or
`nextLink`. `cp` cannot be combined with `anchor` or `anchor_file`.

## Change an existing recurrence

Edit the user-owned recurrence fields on a pending task:

```bash
task 42 modify cp:3d
task 42 modify anchor:"w:mon,wed,fri@t=09:00"
task 42 modify anchor_mode:flex
```

Nautical keeps the existing chain identity and validates the new schedule
before staging the change. The `chainID` must remain unchanged. Do not manually
change `chainID`, `link`, `prevLink`, or `nextLink`; those fields prove lineage
and are guarded by the lifecycle contract.

Temporal fields follow the separate due-rooted carry rules described in
[Completion periods](completion-periods.md). In particular, editing `due` may
move unchanged `scheduled`, `wait`, and native `until`; editing those fields
directly changes only the field that was edited.

## Pause a chain

Set `chain:off` to stop Nautical from creating another successor after the
current task:

```bash
task 42 modify chain:off
```

The recurrence fields and chain identity remain available for inspection. This
is a pause or terminal decision, not a request to delete the chain's history.

## Resume a paused chain

Restore `chain:on` on a task that still has its recurrence fields:

```bash
task 42 modify chain:on
```

Nautical resumes the existing chain and keeps its `chainID` and link history.
If the task no longer has a valid recurrence source, the transition remains
disabled and requires a corrected recurrence rule first.

## Remove recurrence from a task

Clear the recurrence inputs when the task should no longer be Nautical-managed:

```bash
task 42 modify anchor: cp:
```

When `anchor`, `anchor_file`, and `cp` are all absent, Nautical sets
`chain:off`. It does not rewrite or clear the existing lineage fields, so the
completed chain remains auditable. Removing recurrence from a linked task is a
terminal decision; it does not create a replacement successor.

## Transition summary

| Edit | Result |
| --- | --- |
| Add `cp`, `anchor`, or `anchor_file` to an unlinked task | Promote to a root with `chain:on` and `link:1` |
| Change recurrence fields on an existing chain | Validate and stage the same chain with its identity preserved |
| Set `chain:off` | Disable successor creation while preserving evidence |
| Set `chain:on` on a valid paused recurrence | Resume the existing chain |
| Remove all recurrence inputs | Disable the chain and preserve its lineage |
| Change `chainID`, `link`, `prevLink`, or `nextLink` manually | Reject the transition |

If a transition is rejected, keep the task unchanged, inspect the panel or
diagnostics, and correct the user-owned fields. Do not repair lineage by hand;
use [Doctor](../tools/doctor.md), [Query integrity](../tools/query-api.md), or
[Reconcile](../tools/reconcile.md) when the chain evidence itself is uncertain.
