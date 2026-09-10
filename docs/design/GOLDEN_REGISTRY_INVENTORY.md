# Golden registry inventory

Snapshot recorded for the registry-integrity wave on 2026-09-10:

| Item | Count |
| --- | ---: |
| Top-level `test_*` functions in `dev_tools/nautical_golden_tests.py` | 995 |
| Functions in the normal/deep golden registries | 983 |
| Explicitly retired characterization helpers | 12 |
| Duplicate registry entries | 0 |

The 12 retired helpers are kept as an explicit allowlist in
`tests/test_golden_registry_integrity.py`. They are reconcile characterization
helpers and natural-language cases migrated to direct contract tests; they are
not silently omitted. New golden functions must be registered exactly once or
added to that allowlist with a verified replacement.

The golden runner currently selects tests by name substring (`--only`) rather
than maintaining a typed domain registry. Consequently, domain counts are
not treated as authoritative coverage metrics; the counts above are the
authoritative registry inventory. Use the runner's `--only` filters for a
focused domain slice and the full unshuffled run for registry-wide evidence.
