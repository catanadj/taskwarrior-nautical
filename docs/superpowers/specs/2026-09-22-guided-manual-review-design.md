# Guided Manual-Review Workflow

## Goal

Provide a Nautical-owned workflow for understanding and resolving lifecycle and
chain-integrity manual-review items. Users should not need to inspect raw
Taskwarrior exports or edit chain fields by hand.

## User workflow

The primary command is:

```text
nautical review --next
```

It presents one review item at a time. Each item includes:

- chain ID and transition/intent identity;
- affected source and target links;
- current connected path and slot occupants;
- parent and child reference state;
- the invariant or lifecycle reason that blocked automation;
- confidence and evidence freshness;
- actions that are safe for this specific finding.

The command is read-only unless an explicit action is selected. JSON output
must expose the same evidence and action list for automation.

## Action model

Actions are typed, preconditioned operations:

- `accept-connected`: accept an already-connected branch when the planner can
  prove the alternative is historical or orphaned;
- `retry`: re-evaluate the current state and retry a stale lifecycle intent;
- `replan`: rebuild the integrity plan from a fresh authoritative snapshot;
- `resolve-applied`: acknowledge a transition already proven complete;
- `quarantine`: preserve the finding while preventing repeated automatic work;
- `skip`: leave the finding unresolved and move to the next item.

Only actions that apply to the current finding are offered. Ambiguous duplicate
slots must not expose an automatic branch-selection action.

Mutating actions require an explicit `--apply` mode and a confirmation token
derived from the displayed review item. The command re-reads the authoritative
state immediately before mutation and refuses if the evidence changed.

## Data flow

1. Read manual-review lifecycle records from the existing outbox repository.
2. Read the affected chain through the typed Nautical query/read services.
3. Build a `ChainGraph` and evaluate existing invariants.
4. Ask the repair planner for safe plans and refusal reasons.
5. Compose a review item containing evidence, explanation, and typed actions.
6. Apply only through the lifecycle application or integrity mutation gateway.
7. Persist the decision and resulting evidence in the existing review/outbox
   record; never mutate through raw shell commands from the review UI.

## Safety and output contracts

- Default mode performs no writes.
- Hook stdout remains strict JSON; diagnostics remain stderr-only and opt-in.
- Every mutation is guarded by the current parent/slot evidence.
- A review action is idempotent and safe to repeat after interruption.
- Ambiguous or stale evidence always returns to `manual_review`.
- Human output is concise but complete; `--json` is stable and machine-readable.

## Implementation slices

1. Introduce typed review-item and action models backed by current outbox and
   chain-integrity evidence.
2. Extend `QueueStatusService` to construct one review item with a fresh chain
   snapshot and planner explanation.
3. Add `nautical review --next` and JSON rendering, keeping the existing queue
   commands compatible.
4. Add guarded action dispatch for `resolve-applied`, `retry`, `replan`, and
   `quarantine`; defer branch acceptance until its planner proof is complete.
5. Add CLI, stale-evidence, duplicate-slot, interruption, and multi-device
   race tests.

## Non-goals

- No raw Taskwarrior editing instructions in user-facing output.
- No automatic deletion of duplicate or orphan tasks.
- No distributed lock protocol beyond existing sync-safe identity and guarded
  mutation behavior.
- No new required UDA solely for the review workflow.
