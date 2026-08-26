# Lifecycle and Chain Integrity

Nautical v7 separates pure scheduling, lifecycle planning, durable application,
and integrity analysis. Each layer has a narrow responsibility and a typed
boundary.

## Completion flow

```text
Taskwarrior modify
    -> validate task transition
    -> compile recurrence context
    -> plan deterministic successor
    -> persist lifecycle intent
Taskwarrior releases lock
    -> claim intent
    -> import or find deterministic child
    -> link parent
    -> verify postconditions
    -> acknowledge intent
```

The child UUID and plan fingerprint are deterministic. Repeating an
acknowledged operation remains `already_applied`; a crash between child import
and parent linking converges on the same child.

## Chain invariants

The integrity engine reasons from an authoritative Taskwarrior snapshot plus
outbox evidence. Core checks include:

- complete and immutable chain identity;
- one task per chain slot;
- reciprocal `prevLink` and `nextLink` relationships;
- increasing link positions and recurrence targets;
- valid terminal bounds;
- preserved carry-field relationships;
- lifecycle intent and Taskwarrior postcondition agreement.

Findings are categorized as healthy, repairable, manual review, or unavailable.
One invalid chain does not hide independent safe plans for another chain.

## Evidence coverage

A snapshot states whether it covers one task, one chain, bounded active
candidates, or complete history. A proof that requires missing predecessor or
successor data triggers bounded hydration. If required evidence cannot be
obtained within safety limits, the result is unavailable rather than absent.

## Mutation ownership

- On-modify owns transition validation and intent staging.
- On-exit owns the normal low-latency drain.
- Reconcile owns recovery drains, multi-operation repair, and expiration waves.
- Doctor and query integrity are read-only presentations of the same findings.

External tools should use [Query API](../tools/query-api.md) and operator
commands rather than reaching into lifecycle modules.

## Manual edits

Do not edit `chainID`, `link`, `prevLink`, or `nextLink`. Nautical rejects
identity changes because they invalidate deterministic child and recovery
proofs. Edit recurrence configuration only through supported Taskwarrior fields
while the current link is pending.
