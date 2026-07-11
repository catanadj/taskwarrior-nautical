# Nautical Improvement Checklist

Review date: 2026-07-10

## Major Improvements

- [x] Update Navigator anchor support to use the current `anchor` and
  `anchor_file` UDAs instead of the obsolete `cp_anchor` field.
  - [x] Add regression coverage for anchor summaries, forecasts, calendars, and
    adherence analysis.

- [x] Fix direct Navigator task selection so it resolves the complete chain.
  - [x] Normalize short `prevLink` and `nextLink` values before graph traversal.
  - [x] Prefer `chainID` for targeted chain retrieval.
  - [x] Avoid exporting every Nautical task when a task ID or chain ID is known.

- [x] Reduce hook startup overhead for ordinary Taskwarrior commands.
  - [x] Add a shared, dependency-light hook protocol gate with focused tests.
  - [x] Add a dependency-light JSON passthrough path before importing
    `nautical_core` in `on-add` and `on-modify`.
  - [x] Add a cheap empty-queue guard before loading the full `on-exit` runtime.
  - [x] Preserve strict hook JSON output, malformed-input handling, Unicode output,
    and diagnostics-on-stderr behavior.
  - [x] Add normalized end-to-end latency budgets for plain add, modify, and
    empty exit hooks.
  - [x] Document the staged thin-wrapper follow-up in
    `THIN_HOOK_WRAPPER_PLAN.md`.

- [x] Repair and enable the new stress and soak CI workflows.
  - [x] Point the stress campaign at the existing `dev_tools` scripts.
  - [x] Fix the nightly soak-test script path.
  - [x] Add the golden correctness suite to CI.

- [ ] Make `nautical doctor` discover the effective Taskwarrior data directory.
  - Reuse the hook data-location resolver when `--taskdata` is not explicit.
  - Cover split configuration/data installations.

- [ ] Consolidate operational tools under the `nautical` command.
  - Expose queue inspection, reconciliation, chain repair, and navigation as
    supported subcommands.
  - Replace direct internal Python paths in diagnostic guidance and docs.

## Review Baseline

- [x] Golden tests pass: 442/442.
- [x] Mypy passes across 80 source files.
- [x] Deployment sanity and Python compilation checks pass.
- [x] Core performance budgets pass.
- [x] Plain-hook startup overhead measured locally:
  - `on-add`: median 76 ms
  - `on-modify`: median 90 ms
  - empty `on-exit`: median 73 ms
- [x] Add/modify fast-path overhead measured after protocol-gate integration:
  - `on-add`: median 31 ms
  - `on-modify`: median 48 ms
- [x] Empty-exit overhead measured after queue-probe integration:
  - fresh Taskdata: median 32 ms
  - existing empty SQLite queue: median 36 ms
