# Reconcile

Reconcile repairs chains after hookless completion, native expiration,
interrupted lifecycle application, or deletion on another client.

## Dry run first

Dry run is the default. `--dry-run` is accepted for explicit scripts:

```bash
nautical reconcile
nautical reconcile --dry-run
nautical reconcile --json
```

Review the chain, parent, next link, action, target timestamp, and reason. A dry
run never changes Taskwarrior.

## Narrow the scope

```bash
nautical reconcile --chain-id 176f5c68
nautical reconcile --uuid TASK_UUID
```

Use a chain or UUID scope while investigating. Default audits are bounded to
active tips and unresolved terminals. `--full-audit` exports complete history
for deep validation and can be substantially slower.

## Apply a reviewed plan

```bash
nautical reconcile --apply
nautical reconcile --apply --json
```

Reconcile can:

- drain retryable lifecycle intents;
- link a parent to an existing deterministic child;
- create a missing successor;
- recover multiple expired links within a bounded wave;
- identify a legitimate `chainMax` or `chainUntil` terminal;
- repair supported native `until` carry defects.

Every mutation is guarded and postconditions are verified. Snapshot drift,
conflicting children, missing configuration, unavailable reads, or ambiguous
history stop the affected mutation.

## Expiration waves

```bash
nautical reconcile --apply --max-expiration-hops 64
```

The default is 32 hops per chain; the hard maximum is 1000. Increase the limit
only after reviewing the dry run.

## Housekeeping

Apply mode may perform bounded outbox housekeeping at most once per eligible
day. Disable it for one run with:

```bash
nautical reconcile --apply --no-housekeeping
```

Reconcile is not a force command. Manual-review results require correcting the
reported data or environment before retrying.
