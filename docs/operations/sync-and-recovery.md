# Sync and Recovery

Taskwarrior remains the replicated data store. Nautical's outbox is local
operational state, while deterministic identities and chain-slot proofs allow
devices to converge on Taskwarrior data after interruptions.

## Keep devices aligned

- Install a current Nautical release on every device that completes or modifies
  Nautical tasks.
- Use the same scheduling timezone, business calendars, presets, file-backed
  data, astronomy profile, and `wrand_salt` where chains are shared.
- Sync Taskwarrior before and after repair operations.

Configuration fingerprints prevent a device from silently applying a plan
computed under different scheduling inputs.

## Recovery sequence

1. Inspect installation and active findings:

    ```bash
    nautical doctor
    ```

2. Inspect durable lifecycle work:

    ```bash
    nautical queue-status --json
    ```

3. Narrow the affected chain when possible:

    ```bash
    nautical query integrity --chain-id CHAIN_ID | jq
    nautical reconcile --chain-id CHAIN_ID --json | jq
    ```

4. Apply only a reviewed plan:

    ```bash
    nautical reconcile --chain-id CHAIN_ID --apply
    ```

5. Sync again and repeat the read-only checks.

## Common recoverable cases

- A completion occurred with hooks disabled.
- Taskwarrior expired a task on a client where Nautical did not run.
- The deterministic child exists but the parent lacks `nextLink`.
- Application stopped after importing a child but before recording the stage.
- A retryable command or lock failure interrupted an outbox drain.

## Cases that require review

- More than one plausible child occupies a chain slot.
- Chain identity was manually changed or removed.
- Task evidence changed after planning.
- A required configuration or file fingerprint differs.
- An outbox plan is malformed, poison, or from an unsupported schema.

Unavailable evidence is not absence. Correct the environment or synchronize
the missing data before retrying; do not force a successor.

## Hooks-off break glass

```bash
task rc.hooks=off <command>
```

Use this only to regain control when a hook itself blocks Taskwarrior. It does
not maintain recurrence. Run Doctor and reconcile after the underlying issue is
fixed.
