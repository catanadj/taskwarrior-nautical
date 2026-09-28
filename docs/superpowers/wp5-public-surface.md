# WP5 Public Surface Snapshot

Captured on 2026-09-21 from `nautical_core.compat_api.PUBLIC_EXPORTS`.

- Export count: 130
- Ordered-name SHA-256: `d246ffa075edc62eca043d324d362443217883880f9473e55e7d3b6e27e0d96b`
- Wildcard export source: `nautical_core.__all__`
- Existing signature coverage: parser, duration, datetime, and timezone facade callables are asserted in `tests/test_typed_api_bindings.py`.
- Removed legacy facade alias: `normalize_task_business_calendar`; callers use the explicit `normalize_task_business_calendar_in_place` owner.

The existing `ApiBinding` and lazy-bundle tests are the baseline for owner migration. The legacy business-calendar alias is intentionally absent; no forwarding bridge is retained.

## Compatibility classification

`compat_api.PUBLIC_EXPORT_CATEGORIES` classifies every export as one of:

- `supported_public_api`: stable integrations may import it.
- `installed_runtime`: used by installed hooks/bootstrap/runtime wiring.
- `test_seam`: private helpers or explicitly exposed dependency seams retained for tests.

## Importer audit

The repository-wide audit covered `docs/`, hook entry points (`*.nautical` and
`nautical_core/hooks/`), tools, tests, `.github/`, and release/bootstrap files.
The removed compatibility alias had no repository consumers; the explicit
replacement is covered by business-calendar contract tests. The
installed hook path references task-data resolution through the bootstrap
adapter, so that export remains an installed-runtime contract. Cache, parser,
calendar, and panel exports have direct contract coverage in the test suite.

No alias bridge is retained for unsupported internal imports.
