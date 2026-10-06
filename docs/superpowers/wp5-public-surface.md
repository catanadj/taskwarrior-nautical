# WP5 Public Surface Snapshot

Captured on 2026-09-21 from `nautical_core.compat_api.PUBLIC_EXPORTS`.

- Export count: 130
- Ordered-name SHA-256: `d246ffa075edc62eca043d324d362443217883880f9473e55e7d3b6e27e0d96b`
- Wildcard export source: `nautical_core.__all__`
- Existing signature coverage: parser, duration, datetime, and timezone facade callables are asserted in `tests/test_typed_api_bindings.py`.
- Existing compatibility alias: `normalize_task_business_calendar` forwards to `normalize_task_business_calendar_in_place`.

The existing `ApiBinding` and lazy-bundle tests are the baseline for owner migration. No public name or alias is removed by WP5 inventory work.

## Compatibility classification

`compat_api.PUBLIC_EXPORT_CATEGORIES` classifies every export as one of:

- `supported_public_api`: stable integrations may import it.
- `installed_runtime`: used by installed hooks/bootstrap/runtime wiring.
- `test_seam`: private helpers or explicitly exposed dependency seams retained for tests.
- `legacy_compatibility_alias`: an older spelling retained as a forwarding alias.

Names in the last category require a release note and one full compatibility
window before removal. No alias is removed until importer and installed-layout
checks cover the replacement name.

## Importer audit

The repository-wide audit covered `docs/`, hook entry points (`*.nautical` and
`nautical_core/hooks/`), tools, tests, `.github/`, and release/bootstrap files.
The compatibility alias is referenced by the facade and this inventory only;
the explicit replacement is covered by business-calendar contract tests. The
installed hook path references task-data resolution through the bootstrap
adapter, so that export remains an installed-runtime contract. Cache, parser,
calendar, and panel exports have direct contract coverage in the test suite.

No public name is removed based solely on having no internal importer.
