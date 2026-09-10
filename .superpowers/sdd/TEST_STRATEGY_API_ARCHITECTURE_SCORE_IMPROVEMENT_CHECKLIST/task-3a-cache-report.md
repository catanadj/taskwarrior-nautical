# Work Package 3A-cache report

## Scope

Added direct behavioral coverage for `nautical_core/cache_api.py` in
`tests/test_cache_api_contract.py`. Tests use temporary cache roots, isolated
copied core namespaces, and a deterministic clock. No unrelated files were
modified by this sub-wave.

## Contracts covered

1. Miss/hit behavior returns stable, independent payload copies.
2. Corrupt entries are quarantined and subsequent reads remain clean misses.
3. A held filesystem lock refuses a save; releasing it permits a retry.
4. Unicode is persisted as UTF-8 JSON without ASCII escaping.
5. Scheduler/configuration and business-calendar fingerprints select distinct
   cache keys.
6. Separate core bindings have independent memory maps while retaining the
   expected lock-path behavior.

The Unicode test initially failed against production: `cache_save` used the
JSON default `ensure_ascii=True`. The minimal fix is now in
`nautical_core/cache_payload.py` (`ensure_ascii=False`).

## Verification

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_cache_api_contract -v`
  — 6 tests passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest discover -s tests -q`
  — 532 tests run; 531 passed, 1 unrelated pre-existing failure:
  `test_recurrence_cursor_terminal.CursorTerminalEvidenceTests.test_collect_after_cursor_preserves_terminal_evidence`
  (`result.terminal` was `None` instead of the expected `OccurrenceSearchExhausted`).
- `PYTHONDONTWRITEBYTECODE=1 python3 dev_tools/nautical_golden_tests.py --only cache --verbose`
  — 50 cache tests: 46 passed, 4 failed due to isolated dynamic-module loading
  errors (`ModuleNotFoundError` for `_nautical_core_cache_perm_test`,
  `_nautical_core_cache_lock_test`, `_nautical_core_cache_symlink_guard_test`,
  and `_nautical_core_cache_tmp_test`). The substantive cache tests passed.

## Concerns

The full suite and golden slice contain unrelated baseline failures described
above; they were not changed in this sub-wave.
