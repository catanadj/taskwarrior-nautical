# Lifecycle Outbox

The lifecycle outbox stores deterministic mutation plans until they are fully
applied and acknowledged.

## Inspect status

```bash
nautical queue-status
nautical queue-status --json
```

The report includes schema compatibility, queued/claimed/retry/manual-review/
poison states, stale claims, attempts, retained acknowledgements, and a bounded
sample of intents.

## States

| State | Meaning |
| --- | --- |
| queued | Ready for a drain pass |
| claimed | Leased by one executor |
| retryable | Temporary failure; safe to attempt again |
| manual review | Evidence conflicts or requires a user decision |
| poison | Stored plan cannot be decoded or trusted |
| acknowledged | Postconditions verified; retained for idempotency |

On-exit normally owns the fast drain after Taskwarrior releases its lock.
Reconcile owns recovery drains and larger repair plans.

## Housekeeping

Acknowledged intents are retained for 90 days by default. Explicit pruning
removes only acknowledgements older than policy:

```bash
nautical queue-status --prune-acknowledged
nautical queue-status --prune-acknowledged --checkpoint
```

Use `--retention-seconds` and `--maintenance-limit` only for controlled
operations. Reconcile also runs bounded, rate-limited housekeeping when useful.

## Storage

The database is:

```text
TASKDATA/.nautical-state/.nautical_lifecycle_outbox.db
```

Do not edit SQLite rows by hand. The plan fingerprint, stage records, leases,
and acknowledgement history are part of replay safety.
