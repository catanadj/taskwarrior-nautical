# Golden registry inventory

Snapshot after recurrence, lifecycle, operator, hook-protocol, TaskDocument,
query-model/command, query-service, lifecycle-planner/read-service,
chain-integrity, task-domain, Navigator-view, CP scheduling, file-backed,
season-calendar/selector, business-calendar, occurrence-provider, time-window,
parser/scheduler expression, core-utility, diagnostic-warning, scheduler
runtime, position-selection, omission, native-until, shared-time, runtime-
configuration, recurrence-spec/hook-bootstrap, weekday-scheduler, panel-
diagnostic, runtime-manifest ownership, preview-warning, Doctor UDA-alias,
configuration-schema, live-panel, and concise text-presentation contract
migrations on 2026-09-14:

| Item | Count |
| --- | ---: |
| Top-level `test_*` functions in `dev_tools/nautical_golden_tests.py` | 416 |
| Functions in the normal/deep golden registries | 404 |
| Explicitly retired characterization helpers | 12 |
| Removed ineffective test definitions | 3 |
| Additional direct-contract functions migrated and removed in this WP4 cycle | 464 |
| Duplicate registry entries | 0 |

Doctor configuration and presentation contracts now live in
`tests/test_doctor_configuration_contract.py` and
`tests/test_doctor_presentation_contract.py`. They directly verify the UDA
alias message/default, schema findings, live-panel fallback/dependency policy,
prominent timezone summary, compact large-history output, and historical
finding grouping. Six additional configuration contracts cover timezone data,
astronomy preflight, seasonal backend/event projection, active-source drift,
and missing Navigator dependencies. Long-lived drift detection remains golden
because it exercises process-lifetime behavior.

The 12 retired helpers are kept as an explicit allowlist in
`tests/test_golden_registry_integrity.py`. They are reconcile characterization
helpers and natural-language cases migrated to direct contract tests; they are
not silently omitted. New golden functions must be registered exactly once or
added to that allowlist with a verified replacement.

The current snapshot supersedes the older chronology below: registry integrity
asserts 416 top-level definitions, 404 registered cases, 12 retired helpers,
464 migrated direct-contract names, and three removed ineffective definitions.
Fresh gates on 2026-09-14 passed standard unittest discovery (1,219 tests, 3
optional skips) and the golden runner in normal and seeded-shuffle order
(404/404 each, seed `20260811`). Together with the exclusive domain table and
digest guard below, this is the current case-level classification and runtime
inventory. Historical checkpoints later in this file are retained for traceability.

## Exclusive domain inventory — 2026-09-14

Each unittest is assigned once by its discovered owner module. The unit runtime
is the sum of individual test durations from one complete discovered run. Each
golden case is assigned once by the ordered category rules pinned in
`tests/test_golden_registry_integrity.py`; those checks validate the exact
membership digest and count of every bucket. Golden runtimes are cumulative
case durations; the seven module-isolation cases use the parent-observed child
process wall time. These are attribution measurements, not benchmark gates.

### Standard unittest discovery

| Domain | Tests | Runtime |
| --- | ---: | ---: |
| Taskwarrior and task data | 51 | 0.427 s |
| Backup and restore | 73 | 4.350 s |
| Cache | 27 | 0.114 s |
| Configuration and architecture | 91 | 0.730 s |
| Hook and add integration | 98 | 10.414 s |
| Lifecycle, modify, and reconciliation | 190 | 9.769 s |
| Operator, query, Doctor, and queue | 212 | 8.008 s |
| Performance and deployment reliability | 68 | 15.519 s |
| Presentation and Navigator | 44 | 0.071 s |
| Recurrence and file providers | 365 | 26.486 s |
| **Total** | **1,219** | **75.901 s wall** |

The final unprofiled unittest gate also passed 1,219 tests in 90.371 s; elapsed
time varies with machine load and is recorded for context, not as a threshold.

### Retained golden acceptance runner

| Primary acceptance domain | Cases | Cumulative runtime |
| --- | ---: | ---: |
| Configuration and bootstrap | 37 | 2.740 s |
| Install and deployment | 15 | 33.766 s |
| Lifecycle and durable mutation | 37 | 24.854 s |
| Operator, query, Doctor, and Navigator | 30 | 6.018 s |
| Performance and soak | 7 | 9.780 s |
| Reconcile and recovery | 26 | 7.556 s |
| Recurrence and hook integration | 242 | 44.686 s |
| Storage and filesystem safety | 10 | 1.777 s |
| **Total** | **404** | **131.177 s cumulative** |

These are primary rather than exclusive product domains: for example,
recurrence-and-hook cases intentionally cover the composed on-add, on-modify,
completion, preview, timeline, and strict-JSON paths. Retained domains represent
configuration/bootstrap isolation, installed filesystem behavior, durable
outbox/mutation state, command/operator envelopes, Taskwarrior interaction,
cross-hook recurrence agreement, locks/permissions, or performance/soak
behavior. Owner-level model, parser, scheduler, repository, renderer, and planner
contracts migrated to unittest discovery remain the fast failure-localization
layer. The acceptance-domain digest gate prevents unreviewed registry drift.

## Case-level classification status

Every currently registered golden case belongs to exactly one of the eight
acceptance domains above. The registry test pins each domain's count and SHA-256
digest over sorted case names, so case addition, removal, or rename requires an
explicit classification refresh. The per-domain rationale above documents why
these cases remain in the acceptance runner; direct owner contracts are tracked
separately by the migrated-contract allowlist.

Two natural-language characterization functions were moved into
`tests/recurrence/test_natural_language_migration.py`; their direct tests are
now discovered by standard `unittest` and the old registry entries were
removed. The long-interval recurrence contract and on-add acceptance checks
were moved into `tests/recurrence/test_scheduler_long_interval_contract.py`
and `tests/test_on_add_hook_routes.py`; its duplicate golden entry was
removed. Yearly-token acceptance and malformed-range contracts were moved to
`tests/recurrence/test_yearly_token_migration.py`, removing two more golden
entries. Moon-phrase, per-atom time, grouped-filter, constrained-yearly-random,
year-ordinal, business-day-roll, and counted-random natural-language cases now
run in the recurrence unittest suite; seven duplicate golden registrations
were removed. Yearly-random month/weekday scheduler constraints also moved to
the direct recurrence suite. Three position-selection contracts (public
ACF/cache shape, post-selection modifiers, and public period scopes/hints) now
have direct tests in `tests/recurrence/test_position_selection_contracts.py`;
their golden registrations were removed. Seasonal selection language/advice is
also directly tested against the configured boundary profile, replacing a
golden assertion that incorrectly hard-coded fixed boundaries. In total, 21
parser/natural-language, scheduler, and position-selection golden functions
have moved into direct unittest coverage. The scheduler cross-path conformance
matrix now runs in `tests/recurrence/test_scheduler_cross_path_conformance.py`,
preserving agreement across the next, collect, preview, and range paths,
strict monotonicity, and deterministic repeat behavior. Four sparse-intersection and typed
exhaustion contracts were also moved to
`tests/recurrence/test_scheduler_exhaustion_contracts.py`; four chain-graph and
repair-planner contracts moved to `tests/test_chain_graph_and_repair_planner.py`.
Precompute cache-hit validation and parse-on-miss behavior now have direct
tests in `tests/test_precompute_contract.py`, replacing two more golden cases.
Cached parse DNF isolation moved to `tests/cache/test_recurrence_cache_contracts.py`.
Cache-key memoization and reset behavior now have direct coverage in
`tests/test_cache_api_contract.py`.
Cache file permissions, schema/shape quarantine, and atomic-replace failure
are also covered directly there; equivalent golden registrations were removed.
Three parser-front-end normalization and validation characterization cases now
have direct contracts in `tests/test_parser_api_contract.py`.
The parser atom helper characterization now directly exercises the owner in
`tests/test_parser_atom_contract.py` rather than reaching through the facade.
The parser validation matrix, yearly-token error surfaces, and yearly-format
owner helper now run in `tests/recurrence/test_yearly_token_migration.py`.
The natural-language golden case that exercised on-add omit-aware preview text
moved to `tests/test_on_add_hook_routes.py`, where JSON stdout and stderr
presentation are asserted together. Lifecycle model transition validation,
draft/identity/outcome preservation, planner purity, and terminal policy now
run in `tests/test_lifecycle_terminal_plans.py`. Operator v2 public envelope,
status, text presentation, round-trip behavior, and immutable presentation
contracts now run in `tests/test_operator_conformance.py` and
`tests/test_operator_presentation.py`; process, installed-layout, persistence,
and cross-process operator acceptance tests remain in their existing suites.
The hook protocol's add/modify classification, strict validation, stream, and
ordinary-edit contracts now run in `tests/test_hook_protocol.py`. Taskwarrior
payload/response shape and `TaskDocument` contracts now run in
`tests/test_taskwarrior_io.py`. The isolated-load and hook subprocess/output/
permissions acceptance cases remain in the golden runner because they exercise
the executable/bootstrap boundary.

Query request/response model serialization and validation now run in
`tests/test_query_models.py`; in-process query command parsing, flag-path
failure, and capability-document contracts run in
`tests/test_query_command_contracts.py`. The golden runner retains query
subprocess, installed-layout, and concurrent-Taskdata isolation scenarios
because they exercise process and Taskwarrior boundaries.

Read-only occurrence and `next` service projection contracts now run in
`tests/test_query_service_contracts.py`: scheduler parity, omission reporting,
read-state preservation, selector filtering/batching, malformed per-task
failures, due-bounded anchor and CP projections, chain bounds, and daily
skip-mode evidence. Query process, installed-layout, and concurrent-Taskdata
acceptance scenarios remain in the golden runner because they exercise process
or Taskwarrior boundaries.

Lifecycle planner ownership of recurrence candidate policy, completion/reconcile
plan parity, scheduled-expiration basis, recurrence-boundary child semantics,
and the idempotent terminal chain patch now runs in
`tests/test_lifecycle_terminal_plans.py`. The runner retains lifecycle runtime,
mutation, subprocess, and Taskwarrior acceptance scenarios.

Lifecycle read-service chain indexing/merge behavior, safe filtering of an
authoritative full snapshot without a repository fallback, and request-scoped
cache rows/indexes are directly covered in `tests/test_lifecycle_read_service.py`.
The golden runner continues to retain chain mutation and Taskwarrior-boundary
coverage.

Chain-integrity model validity, incomplete versus repairable identity,
dependency validation, immutable plans, and snapshot authority/cache/fail-closed
behavior now run in `tests/test_chain_integrity_models.py`. The engine's audit,
drain, hydration, and mutation-boundary scenarios remain separate because they
exercise service composition and mutation controls.

The empty-drain/audit lifecycle, audit-versus-snapshot report parity, and
bounded candidate hydration contracts now run in
`tests/test_chain_integrity_engine.py`; its busy-read case proves unavailable
hydration cannot be reported as authoritative.

The invariant owner registry and deterministic rule evaluation across identity,
duplicate slot, link reciprocity, temporal carry, deletion, and child continuity
cases also run directly in `tests/test_chain_integrity_engine.py`.

Outbox/graph provenance, acknowledged lifecycle postconditions, and the
integrity-application refusal/delegation boundary are also direct in-memory
contracts in that module. Golden coverage is retained for actual Taskwarrior,
cross-process, or installed-runtime behavior.

Task command/read outcomes, immutable lossless observations, operation-validated
task projections, TaskView temporal presence, strict contract-specific codec
serialization/framing, and explicit TaskDraft/TaskPatch mutation semantics now
run in `tests/test_task_domain_models.py`. Process execution and repository
boundary contracts remain in the golden runner.

Authoritative task snapshot scope/index/ambiguity and bounded UUID/slot set-read
contracts now run in `tests/test_task_read_repository_contracts.py`, including
mixed found/absent results, duplicate identities, contradictory slots, stale
epochs, partial chunks, and short-UUID rejection.

Repository cache reuse, narrow fallback, malformed-output handling,
mutation-epoch invalidation, malformed-found preservation, and all domain read
shapes are directly tested in the same module with a scripted command client.
The real Taskwarrior process boundary remains acceptance coverage.

Taskwarrior unit-of-work cache scope/invalidation, explicit broad-snapshot
coverage, and per-invocation isolation now run in
`tests/test_taskwarrior_uow_contracts.py`. Actual command process/retry/timeout
and budget-diagnostic behavior remains in the golden runner. Ten Navigator
metadata/query parity and renderer-neutral view serialization contracts now run
in `tests/test_navigator_view_models.py`; rendering, import/layout, repository
integration, and scale scenarios remain in the golden runner pending
case-level classification. CP duration/sequence parsing, sequence-link
boundaries, random/jitter determinism and bounds, chain scope, and local wall
time across DST now run in `tests/recurrence/test_cp_sequence_contracts.py`;
on-add/on-modify cross-path agreement remains in the golden runner. Pure
moon-phase grammar, timezone configuration, circular-distance, and phase-band
contracts now run in `tests/test_astronomy_contracts.py`; optional Astral
provider behavior remains golden. Year-day and ISO-week ordinal validation,
Gregorian/ISO-year expansion, expression composition, interval, random/omit,
positional ACF, and JSON round-trip contracts now run in
`tests/recurrence/test_yearly_token_migration.py`; hook/timeline integration
remains golden. File-name safety, omit-file CSV header/date-deduplication and
description mapping, and anchor-file time/offset/window parsing now run as
direct filesystem contracts in `tests/test_file_backed_contracts.py`. File
occurrence expansion, provider cursor/order behavior, and wider recurrence
integration remain golden.

Additional pure parser and scheduler contracts now run under standard unittest
discovery in `tests/test_parser_owner_api_contracts.py`,
`tests/test_scheduler_api_contract.py`,
`tests/recurrence/test_yearly_token_migration.py`, and
`tests/recurrence/test_counted_random_contracts.py`, and
`tests/recurrence/test_parser_fuzz_contracts.py`. Coverage includes parser
limits and diagnostics, yearly token parsing, grouped modifier distribution
and rejection, direct interval/date-boundary scheduling, business-day and
leap-day behavior, seeded random/counted-random guarantees, and typed search
exhaustion, modified-atom interval/roll/offset behavior, parser satisfiability
diagnostics, ISO-week boundaries, and bounded core coercions. Their former
golden wrappers have been removed. Hook,
Taskwarrior/process, Navigator, DST/provider integration, and statistical
acceptance scenarios remain in the golden runner pending case-by-case review.

Direct occurrence-runtime contracts now cover cursor inclusivity and timezone
agreement, typed found/absent/invalid/exhausted outcomes, fail-closed mutation
conversion, evaluation-session identity and cache scope, SchedulerService
construction and typed paths, range bounds and omission provenance, redacted
bounded tracing, evaluator-owned context/limits/CP/event streams, and terminal
evidence preservation. Generated scheduler streams and test-only parity-helper
contracts now run directly as well. The migrated cursor/outcome, session/service,
range, trace, shuffled-session, hint-failure, generated-matrix, parity-helper,
and terminal-evidence wrappers were
removed. The broader evaluator-owner and cross-consumer parity scenarios remain
in the golden runner pending case-by-case classification.

WP4 remains incomplete. Parser validation/front-end/atom, scheduler/occurrence,
cache, on-add route, lifecycle/operator, and backup/restore direct suites exist,
but their presence does not classify every related golden scenario as a direct
contract migration. The remaining domains require scenario classification,
per-domain unit/golden inventory and retained-acceptance rationale, and normal
plus shuffled golden-run evidence. The golden runner currently selects tests by name substring (`--only`) rather
than maintaining a typed domain registry. Consequently, domain counts are
not treated as authoritative coverage metrics; the counts above are the
authoritative registry inventory. Use the runner's `--only` filters for a
focused domain slice and the full unshuffled run for registry-wide evidence.

## Case-level decisions — anchor-file provider batch

| Former golden case | Decision | Evidence / rationale |
| --- | --- | --- |
| `test_anchor_file_occurrence_provider_supports_lazy_next_after` | Direct contract | Exercises `AnchorFileOccurrenceProvider.next_after` with an isolated temporary CSV and injected datetime conversion callbacks; no hook, process, Taskwarrior, or installation boundary is involved. Migrated to `tests/test_file_backed_contracts.py`. |
| `test_anchor_file_occurrence_provider_caches_expanded_specs` | Direct contract | Asserts provider-local reuse of occurrence-spec expansion across successive lookups, restoring the patched loader in `finally`. This is an owner-level cache contract, not an acceptance boundary. Migrated to `tests/test_file_backed_contracts.py`. |
| `test_anchor_file_occurrence_provider_advances_cached_lookup_cursor` | Direct contract | Checks stable repeat results, sequential cache reuse, and correct reset on a backward cursor directly on the provider. Migrated to `tests/test_anchor_files.py`. |
| `test_anchor_file_provider_uses_binary_search_for_nonmonotonic_cursor` | Direct contract | Guards indexed backward lookup over cached anchor records without invoking the linear comparison path. This is an internal performance regression contract, not an external acceptance check. Migrated to `tests/test_anchor_files.py`. |
| `test_anchor_file_occurrence_provider_sorts_dst_normalized_candidates` | Direct contract | Uses deterministic timezone conversion callbacks to verify successor order after a DST-gap wall time is normalized. Migrated to `tests/test_anchor_files.py`. |
| `test_anchor_file_provider_retries_after_failed_load` | Direct contract | Calls the provider directly and verifies a failed load propagates without poisoning its cache, then a later load succeeds. Migrated to `tests/test_anchor_files.py`. |
| `test_anchor_file_provider_keeps_description_for_overnight_slots` | Direct contract | Verifies typed provider output keeps source-date metadata across an overnight slot boundary. Migrated to `tests/test_anchor_files.py`. |
| `test_anchor_file_provider_merges_duplicate_source_descriptions` | Direct contract | Verifies the provider's deterministic duplicate-date merge rule when a later file supplies missing metadata. Migrated to `tests/test_anchor_files.py`. |
| `test_anchor_file_provider_preserves_dst_fold_descriptions` | Direct contract | Checks provider metadata association for the selected instant during a repeated local hour. Migrated to `tests/test_anchor_files.py`. |
| `test_anchor_file_provider_orders_dst_fallback_by_instant` | Direct contract | Confirms successor choice distinguishes the two local-time folds by instant. Migrated to `tests/test_anchor_files.py`. |
| `test_anchor_file_provider_rejects_incomparable_datetimes` | Direct contract | Supplies mixed naive/aware values directly to the provider and checks fail-closed comparison behavior. Migrated to `tests/test_anchor_files.py`. |
| `test_included_provider_bounds_anchor_file_omission_scan` | Direct contract | Exercises `next_included_occurrence` with an injected provider and omission decision, proving the configured skip bound fails closed. Migrated to `tests/test_anchor_inclusion_contracts.py`. |
| `test_anchor_inclusion_scheduler_propagates_internal_errors` | Direct contract | Injects a scheduler callback and proves an internal `TypeError` propagates after one call rather than triggering a fallback retry. Migrated to `tests/test_anchor_inclusion_contracts.py`. |
| `test_anchor_file_omit_evaluation_failures_propagate` | Direct contract | Tests the omission boundary's typed/wrapped failure directly, including preserving the original cause. Migrated to `tests/test_anchor_inclusion_contracts.py`. |
| `test_recurrence_evaluator_loads_omit_file_without_text_rule` | Direct contract | Injects an isolated omit-file directory into `RecurrenceEvaluator` and verifies file-only dates and descriptions form omission state. Migrated to `tests/recurrence/test_scheduler_runtime_contracts.py`. |
| `test_recurrence_evaluator_loads_omit_file_dates_and_descriptions_once` | Direct contract | Verifies the evaluator makes one combined dates/descriptions load, memoizes the immutable state, and combines it with the text omit rule. Migrated to `tests/recurrence/test_scheduler_runtime_contracts.py`. |
| `test_recurrence_evaluator_owns_context_spec_and_timezone_boundary` | Retired duplicate characterization | Its normalized context, DST projection, immutable DNF/CP data, limits, provider reuse, streams, and mode outcomes are directly asserted by `test_evaluator_owns_normalized_recurrence_context_without_io`, `test_evaluator_projects_cp_and_reuses_its_file_provider`, and `test_evaluator_stream_modes_and_guards_preserve_occurrence_semantics` in `tests/recurrence/test_scheduler_runtime_contracts.py`; retaining the broad monolithic wrapper would duplicate those contracts. It remains in the explicit retired allowlist and is not registered. |
| `test_domain_scheduler_parity_across_operational_consumers` | Direct cross-consumer contract | Drives the typed scheduler, query service, Navigator projection, and reconcile planner in-process over one deterministic task and requires identical next-occurrence evidence. No subprocess or installed-layout boundary is needed. Migrated to `tests/recurrence/test_scheduler_cross_path_conformance.py`. |

These decisions currently cover the anchor-file provider/inclusion and
cross-consumer cases; the rest remain unclassified until reviewed case by
case.

| `test_recurrence_evaluator_shadow_parity_time_matrix` | Direct cross-path contract | Compares `ChainGenerationService.compute_anchor_child_due` with evaluator mode selection across fixed, multiple, bounded, seeded-random, and overnight time forms. The fixture initializes lazy timezone configuration before constructing the evaluator context, ensuring both paths use the same local zone. Migrated to `tests/recurrence/test_scheduler_cross_path_conformance.py`. |
| `test_recurrence_evaluator_shadow_parity_dst_and_business_calendar` | Direct cross-path contracts | Split into DST-gap and explicit business-calendar parity tests. Both compare the shared chain-generation service with evaluator mode selection in-process, using isolated configured context and no hook subprocess or Taskwarrior boundary. Migrated to `tests/recurrence/test_scheduler_cross_path_conformance.py`. |
| `test_modify_timeline_uses_explicit_recurrence_identity` | Direct owner contract | Exercises chainID-first seed selection and the documented preview fallback directly in `modify_timeline`. Migrated to `tests/test_recurrence_identity_contracts.py`. |
| `test_modify_hook_uses_explicit_recurrence_identity` | Direct owner contract | Exercises chainID-first identity and fallback behavior at the schedule-effects owner without loading the hook host. Migrated to `tests/test_recurrence_identity_contracts.py`. |
| `test_add_preview_uses_explicit_recurrence_identity` | Direct owner contract | Verifies preview seed selection prefers chainID and uses the root UUID fallback directly in `add_anchor_preview`. Migrated to `tests/test_recurrence_identity_contracts.py`. |
| `test_random_time_window_dst_projection_is_deterministic` | Direct time-projection contract | Resolves deterministic random wall-clock slots and uses explicit `ZoneInfo` conversions to verify unique instants over a DST gap and fold. Migrated to `tests/recurrence/test_time_windows_contracts.py`; process-stability remains a separate golden acceptance case. |
| `test_random_time_window_composition_and_anchor_file_guidance` | Direct parser/file-spec contracts | Split into direct parser rejection guidance and anchor-file random-window canonicalization tests in `tests/test_parser_owner_api_contracts.py` and `tests/test_file_backed_contracts.py`. Neither case requires hook or process execution. |
| `test_position_selection_parses_arbitrary_ordinals` | Direct owner contract | Verifies accepted aliases, signed positions, scope maxima, and stable deduplication directly through `position_selection.parse_positions`. Migrated to `tests/recurrence/test_position_selection_contracts.py`. |
| `test_position_selection_rejects_invalid_tokens_and_bounds` | Direct owner contract | Preserves malformed ordinal, invalid suffix, zero, empty item, unknown-scope, and scope-limit diagnostics at the parsing owner. |
| `test_position_selection_candidate_capacity_bounds` | Direct owner contract | Checks exact conservative structural upper bounds across weekly, monthly, and yearly candidate forms. |
| `test_position_selection_candidate_capacity_bounds_are_sound` | Direct owner contract | Brute-checks capacity estimates against actual matching dates across four scopes and representative period samples. |
| `test_position_selection_rejects_only_fully_impossible_candidates` | Direct parser contract | Verifies only impossible positional candidates are rejected and mixed viable/dead positions remain parseable. |
| `test_position_selection_semantic_advice` | Direct owner/scheduler contract | Checks dead-position, redundancy, and ISO-week/calendar-year advice while verifying the advised expression remains schedulable. |
| `test_position_selection_period_boundaries` | Direct owner contract | Checks exact ISO-week, month, quarter, and year bounds plus wrong-type rejection. |
| `test_position_selection_internal_evaluator` | Direct owner contract | Covers signed positional selection, deduplication, and absent positions at `position_selection` itself. |
| `test_position_selection_internal_evaluator_validation` | Direct owner contract | Exercises malformed, empty, nested, and out-of-range internal selector nodes and rejects zero positions. |
| `test_position_selection_next_date_jumps_periods` | Direct owner contract | Verifies strict successor behavior, empty-period skipping, and the bounded scan work count. |
| `test_position_selection_candidate_cache_identity` | Direct cache contract | Ensures canonical expression, chain seed, and business-calendar fingerprint define the cache identity; cache state is cleared after the test. |
| `test_position_selection_public_monthly_parser_validation` | Direct public parser contract | Verifies monthly positional AST shape and actionable rejection for unsupported scopes, malformed candidate groups, modifier/filter placement, nesting, and multiple selectors. Migrated to `tests/recurrence/test_position_selection_contracts.py`. |
| `test_position_selection_public_monthly_scheduler` | Direct scheduler contract | Checks monthly first/last successor selection, candidate intersection, and skipping an empty bucket through `core.next_after_expr`; no hook/process boundary is needed. |
| `test_position_selection_post_modifiers_parser_and_scheduler` | Direct parser/scheduler contract | Checks modifier AST fields and selected-date transforms, collision deduplication, factor matching, and custom business-calendar propagation. |
| `test_position_selection_public_period_scopes_validation` | Direct public parser contract | Verifies week, quarter, and year scope normalization plus their independent limits and unsupported selector diagnostics. |
| `test_position_selection_public_period_scopes_scheduler` | Direct scheduler contract | Exercises week/quarter/year successor behavior, leap-day selection, post-selection shifts that retain source-bucket semantics, and custom business-calendar handling. |
| `test_position_selection_documented_examples_and_feedback` | Direct grammar contract | Keeps documented selector examples parseable and verifies actionable guidance for unparenthesized candidates, misplaced modifiers, and unsupported scopes. |
| `test_anchor_omit_rejects_time_modifiers` | Direct omission grammar contract | Verifies date-only omit validation rejects time modifiers with actionable guidance, directly at `anchor_omit`. Migrated to `tests/recurrence/test_omit_contracts.py`. |
| `test_anchor_omit_next_after_expr_skips_matching_dates` | Direct omission scheduler contract | Verifies `next_after_expr_with_omit` advances past a matching omit expression without involving a hook or file read. |
| `test_anchor_omit_next_after_expr_skips_omit_file_dates` | Direct omission-state contract | Uses an already-loaded immutable omit-date state to prove recurrence advancement skips that date; file loading itself remains covered by file-backed contracts. |
| `test_anchor_omit_grouped_list_plus_expr_applies_filter_to_all_items` | Direct omission evaluator contract | Verifies the grouped list-plus expression matches every expected April weekday and does not leak into May. |
| `test_anchor_omit_business_day_roll_matches_rolled_date` | Direct omission evaluator contract | Checks omit evaluation against the business-day-rolled date rather than only the unmodified base date. |
| `test_anchor_omit_positive_day_offset_matches_shifted_date` | Direct omission evaluator contract | Covers both positive calendar-day and business-day offsets directly at omission evaluation. |
| `test_file_source_expression_flattens_groups_and_rejects_unsafe_patterns` | Direct file-source parser contract | Checks inner-to-outer modifier-layer ordering, branch flattening, and rejection of empty, path-traversal, recursive, and unsupported-glob syntax. Migrated to `tests/test_file_backed_contracts.py`. |
| `test_native_until_carry_descriptions` | Direct presentation contract | Verifies calendar-day and exact expiration descriptions at the owning `add_validation` formatter. Migrated to `tests/test_native_until_contracts.py`. |
| `test_native_until_validation_orders_dst_fold_by_instant` | Direct temporal validation contract | Checks that the earlier instant in a repeated wall-clock hour is rejected as an expiration before its target. |
| `test_native_until_exact_carry_orders_dst_fold_by_instant` | Direct carry contract | Verifies elapsed-second preservation across a DST fold, exact carry wording, and fail-closed handling of a malformed converter. |
| `test_schedule_and_completion_use_shared_datetime_comparator` | Direct cross-owner contract | Proves schedule and completion effects expose the same `timeutil.compare_datetimes` implementation. |
| `test_local_datetime_full_day_gap_shifts_to_next_valid_wall_time` | Direct timezone contract | Verifies that the Pacific/Apia skipped calendar day resolves to the same wall time on the next valid date. |
| `test_public_datetime_comparator_preserves_dst_fold_and_provider_alias` | Direct time comparison contract | Checks repeated-hour instant ordering, the provider compatibility alias, and rejection of mixed naive/aware values. |
| `test_recurrence_fingerprint_is_canonical_and_mutation_sensitive` | Direct lifecycle identity contract | Verifies formatting/presentation-only edits preserve the recurrence fingerprint while a scheduling edit invalidates it. Migrated to `tests/test_lifecycle_terminal_plans.py`. |
| `test_effective_config_snapshot_isolated_and_provenanced` | Direct configuration API contract | Checks source provenance and that mutating returned values cannot mutate the active core configuration. Migrated to `tests/test_runtime_config_contracts.py`. |
| `test_hot_config_fingerprint_avoids_filesystem_stat` | Direct performance contract | Ensures a warm effective-config fingerprint lookup reuses cached state without metadata syscalls. |
| `test_recurrence_spec_normalizes_task_fields_and_context` | Direct typed-spec contract | Checks normalized recurrence fields, null UDA handling, typed-observation conversion, caller context preservation, and conflicting identity rejection. Migrated to `tests/recurrence/test_scheduler_runtime_contracts.py`. |
| `test_hook_bootstrap_numeric_env_parsing_is_bounded` | Direct bootstrap contract | Verifies malformed integer fallback, numeric bounds, and non-finite float fallback using the pure hook-bootstrap helpers. The hook-level malformed-environment process check remains golden. |
| `test_last_weekday` | Direct recurrence scheduling contract | Its user-facing phrase is already covered by the direct natural-language table; this migration moves the distinct deterministic scheduling check for five last-Friday occurrences to `tests/recurrence/test_scheduler_runtime_contracts.py`. |
| `test_panel_diagnostics_warns_for_missing_env_config` | Direct diagnostic contract | Verifies an explicit missing config path is surfaced as a user diagnostic using an isolated temporary path. Migrated to `tests/test_structured_failure_boundaries.py`. |
| `test_runtime_manifest_covers_lazy_panel_colour_module` | Direct runtime-manifest contract | Checks all three hook release manifests include the lazily loaded panel colour module. Migrated to `tests/test_architecture_contract.py`. |
| `test_legacy_exit_flow_modules_are_not_runtime_owned` | Direct repository-architecture contract | Checks removed legacy exit-flow modules remain absent from the source tree and every hook runtime manifest. |
| `test_on_add_preview_warns_when_anchor_uses_utc_fallback` | Direct preview-warning contract | Exercises timezone fallback warning policy directly; hook JSON/stdout behavior remains covered at the acceptance boundary. Migrated to `tests/test_add_preview_contract.py`. |
| `test_panel_diagnostics_warns_for_empty_file_sources` | Direct diagnostics contract | Uses isolated file providers to verify unusable-but-readable anchor/omit sources and unmatched wildcard diagnostics. Migrated to `tests/test_structured_failure_boundaries.py`. |

This is an additional case-level decision; the remaining registered scenarios
still require review and an explicit direct-contract or acceptance-boundary
rationale.

## Previous full WP4 gate checkpoint — 2026-09-14

- Standard unittest discovery: 993 passed in 69.113s before the latest three
  cross-path test additions.
- Normal and shuffled golden runs, seed `20260811`: 639/639 before the latest
  two wrapper removals.
- At this checkpoint, the registry had 651 top-level / 639 registered / 12
  explicitly retired / 232 newly migrated functions / 0 duplicate registrations.
- The cumulative WP4 direct-migration count was 345.
- Current registry after the latest migration: 649 top-level / 637 registered /
  12 explicitly retired / 234 migrated in this WP4 cycle / 0 duplicates; three
  focused chain-generation/evaluator parity methods pass. Full gates are pending.
- After the parity batch, full gates passed: standard discovery 996 tests in
  67.493s and normal plus shuffled golden 637/637 (seed `20260811`).
- Current registry after recurrence-identity migration: 646 top-level / 634
  registered / 12 explicitly retired / 237 migrated in this WP4 cycle / 0
  duplicates. Standard unittest discovery passes 999 tests in 67.423s; normal
  and shuffled golden runs pass 634/634 (seed `20260811`). Focused owner,
  cross-path, and registry tests pass 15/15.
- The cumulative WP4 direct-migration count is 350.
- The following random-time-window batch migrates two additional golden
  wrappers into direct timezone, parser, and file-spec contracts. Current
  registry: 644 top-level / 632 registered / 12 explicitly retired / 239
  migrated this WP4 cycle / 0 duplicates; cumulative direct migrations: 352.
  Focused recurrence time-window, parser-owner, file-backed, and registry
  suites pass. Complete gates pass: standard discovery 1002 tests in 67.185s;
  normal and shuffled golden runs pass 632/632 (seed `20260811`).
- Current positional-selection batch migrates eleven pure parser, evaluator,
  semantic-advice, period, candidate-capacity, and cache contracts. Registry is
  now 633 top-level / 621 registered / 12 explicitly retired / 250 migrated in
  this WP4 cycle / 0 duplicates; cumulative WP4 direct migrations: 363.
- The next six pure public-parser/scheduler position-selection cases have also
  moved to `tests/recurrence/test_position_selection_contracts.py`; hook,
  completion, and timeline paths remain golden. Registry is now 627 top-level /
  615 registered / 12 explicitly retired / 256 migrated in this WP4 cycle / 0
  duplicates; focused direct and registry tests pass 21/21. Combined full gates
  pass: unittest discovery 1016 tests; normal golden 615/615; seeded shuffle
  615/615 (seed `20260811`). The cumulative WP4 direct-migration count is 369.
- A further six direct omission contracts and one file-source parser contract
  have moved to `tests/recurrence/test_omit_contracts.py` and
  `tests/test_file_backed_contracts.py`. Registry is now 620 top-level / 608
  registered / 12 explicitly retired / 263 migrated this WP4 cycle / 0
  duplicates; focused recurrence, file-backed, and registry suites pass 25/25.
  Consolidated gates pass: unittest discovery 1023 tests in 75.517s; normal
  golden 608/608; seeded shuffled golden 608/608 (seed `20260811`). The
  cumulative WP4 direct-migration count is 376.
- Three native-until owner contracts now run in
  `tests/test_native_until_contracts.py`; hook preview and modify-flow carry
  scenarios remain golden. Current registry is 617 top-level / 605 registered
  / 12 explicitly retired / 266 migrated in this WP4 cycle / 0 duplicates;
  focused direct and registry checks pass 7/7. Full gates are pending.
- Three shared-time contracts now run under `tests/test_config_time_contract.py`:
  consumer comparator identity, DST-fold ordering/alias behavior, and a full-day
  timezone-gap resolution. Current registry: 614 top-level / 602 registered /
  12 explicitly retired / 269 migrated in this WP4 cycle / 0 duplicates;
  focused time, native-until, and registry tests pass 14/14. Consolidated full
  gates pass: unittest discovery 1029 tests in 83.740s; normal golden 602/602;
  seeded shuffled golden 602/602 (seed `20260811`). Cumulative WP4 direct
  migrations: 382.
- Three lifecycle/configuration contracts now have direct owner tests in
  `tests/test_lifecycle_terminal_plans.py` and
  `tests/test_runtime_config_contracts.py`. Registry is 611 top-level / 599
  registered / 12 explicitly retired / 272 migrated in this WP4 cycle / 0
  duplicates; focused lifecycle/config/registry tests pass 25/25. Full gates
  are pending.
- Two more direct contracts cover recurrence-spec normalization/context identity
  and bounded hook-bootstrap numeric parsing. Registry is now 609 top-level /
  597 registered / 12 explicitly retired / 274 migrated in this WP4 cycle / 0
  duplicates; focused hook-bootstrap, recurrence-runtime, and registry tests
  pass 21/21. Consolidated full gates pass: unittest discovery 1034 tests in
  77.052s; normal golden 597/597; seeded shuffled golden 597/597 (seed
  `20260811`). Cumulative WP4 direct migrations: 387.
- The last-Friday scheduler contract and missing-explicit-config diagnostic
  contract now run directly in recurrence-runtime and structured-failure test
  modules. Current registry: 607 top-level / 595 registered / 12 explicitly
  retired / 276 migrated this WP4 cycle / 0 duplicates; focused owner and
  registry checks pass 20/20. Full gates pass: unittest discovery 1036 tests in
  74.921s; normal golden 595/595; seeded shuffled golden 595/595 (seed
  `20260811`). Cumulative WP4 direct migrations: 389.
- Two runtime ownership assertions now live in `tests/test_architecture_contract.py`.
  Current registry: 605 top-level / 593 registered / 12 explicitly retired /
  278 migrated this WP4 cycle / 0 duplicates; focused architecture and registry
  tests pass 17/17. Full gates pass: unittest discovery 1038 tests in 74.692s;
  normal golden 593/593; seeded shuffled golden 593/593 (seed `20260811`).
- Two more owner contracts moved to direct tests: timezone fallback warning
  policy in `tests/test_add_preview_contract.py` and file-source diagnostic
  output in `tests/test_structured_failure_boundaries.py`. Current registry:
  603 top-level / 591 registered / 12 explicitly retired / 280 migrated this
  WP4 cycle / 0 duplicates; focused preview, failure-boundary, and registry
  tests pass 29/29. Full verification passes: unittest discovery 1040 tests in
  74.930s; normal golden 591/591; seeded shuffled golden 591/591 (seed
  `20260811`). The cumulative WP4 direct-migration count is 393.
- Eleven Doctor owner/configuration/presentation cases were migrated into
  direct `unittest` modules; two old timezone cases are consolidated into one
  contract. Current registry: 591 top-level / 579 registered / 12 retired / 292
  direct migrations this WP4 cycle / 0 duplicates; cumulative WP4 direct
  migrations: 405. Full gates pass: unittest discovery 1051/1051 in 75.070s,
  normal golden 579/579, and seeded shuffled golden 579/579 (seed `20260811`).
- Two more direct cases now cover the operator-context discovery/configuration
  boundary and assert that the presentation owner has no mutation imports.
  Registry: 589 top-level / 577 registered / 12 retired / 294 WP4 migrations /
  0 duplicates; full gates require refresh.
- Moved bounded-versus-full lifecycle candidate query selection beside the
  repository direct-contract tests. Its injected client verifies filter
  construction without an external Taskwarrior workflow. Registry: 588
  top-level / 576 registered / 12 retired / 295 WP4 migrations / 0 duplicates;
  cumulative WP4 direct migrations: 408. Verification pending.
- Moved four completion-analytics contracts into
  `tests/test_modify_analytics_contracts.py`: chain gaps and missing identity,
  healthy streak advice, low on-time guidance, and clinical drift/style
  normalization. The fixture uses the same 3,600-second tolerance supplied by
  the hook. Registry is 584 top-level / 572 registered / 12 retired / 299 WP4
  migrations / 0 duplicates; cumulative direct migrations: 412.
- Moved the public-facade export/signature contract to typed API tests and
  replaced the golden self-registration validator with standard registry
  integrity coverage. Query invalid-request output and compact Unicode/budget
  serialization now live in query command tests; cross-consumer integrity
  projection lives in the integrity report contract suite. Registry: 579
  top-level / 567 registered / 12 retired / 304 WP4 migrations / 0 duplicates;
  cumulative direct migrations: 417.
- TaskCommand failure classification and lock-retry opt-in now have direct
  subprocess-backed wrapper contracts in `tests/test_task_command_contracts.py`.
  Four omit-file modifier contracts, TaskCodec string sanitization, and
  Navigator's authoritative-empty snapshot behavior now run in direct owner
  suites. Ten more direct contracts now cover the TaskCommand runtime facade,
  roll convergence guard, cached-hint mutation isolation, moon-phase parser
  contradictions, shared config exposure, and compact-preview occurrence
  limits. Direct command-boundary contracts now cover hook result identity,
  TaskCommand fallback when tempfiles fail, shared runtime output/status, and
  on-modify runner success/failure. Two misleading "falls back when core load
  fails" cases were retired: they never simulated core-load failure or invoked
  the named hook runner, and merely repeated a direct core command call. They
  are tracked in a separate absent-test allowlist. Registry: 550 top-level / 538
  registered / 12 retired characterization helpers / 2 removed ineffective
  tests / 331 WP4 migrations / 0 duplicates; cumulative direct migrations: 444.
- Shipped-config schema/top-level layout, scheduler no-progress guarding, ACF
  cache input bounds, and panel line-mode routing now have direct owner tests.
- Hook import-failure detail, season-mode choices, authoritative-empty lifecycle
  read behavior, completion fail-closed preflight, and successful live-render
  routing now also have direct contracts. Registry: 545 top-level / 533
  registered / 12 retired characterization helpers / 2 removed ineffective
  tests / 336 WP4 migrations / 0 duplicates; cumulative direct migrations: 449.
- Real-provider Astral phase-boundary, timezone/DST, and unavailable-moonrise
  contracts moved into `tests/test_astronomy_contracts.py`. The tests now report
  an explicit unittest skip when Astral is optional and absent, but fail if the
  astronomy CI job sets `NAUTICAL_REQUIRE_ASTRAL=1` without the dependency.
  Registry: 542 top-level / 530 registered / 12 retired characterization helpers
  / 2 removed ineffective tests / 339 WP4 migrations / 0 duplicates; cumulative
  direct migrations: 452.
- Scheduler transient-stall recovery and Navigator's configured display timezone
  now have direct tests. Registry: 540 top-level / 528 registered / 12 retired
  characterization helpers / 2 removed ineffective tests / 341 WP4 migrations /
  0 duplicates; cumulative direct migrations: 454.
- Fail-closed omission scheduling, moon-intersection exhaustion, plain live
  timeline reveal, and stable terminal mutation-guard selectors now have direct
  contracts. Registry: 536 top-level / 524 registered / 12 retired
  characterization helpers / 2 removed ineffective tests / 345 WP4 migrations /
  0 duplicates; cumulative direct migrations: 458.
- Configuration diagnostics, hint-scheduler routing, scoped business-calendar
  selection, astronomy preflight/error boundaries, Navigator metadata and sparse
  calendar rendering, and force-rich panel routing now have direct owner tests.
  Registry: 528 top-level / 516 registered / 12 retired characterization
  helpers / 2 removed ineffective tests / 353 WP4 migrations / 0 duplicates;
  cumulative direct migrations: 466.
- Nine more owner contracts now directly cover TaskCommand observation privacy,
  complete recurrence activation identity, moon source/filter intersections and
  phase-window deduplication, typed occurrence-prefix exhaustion, omission
  warnings, anchor-file tie metadata, DST-fold until validation, and live-render
  generator fallback. Focused suites and registry integrity pass 61/61 (three
  optional-Astral skips). Registry: 519 top-level / 507 registered / 12 retired
  characterization helpers / 2 removed ineffective tests / 362 WP4 migrations /
  0 duplicates; cumulative direct migrations: 475. Full gates need refresh.
- Completion owner suites now cover date-versus-search exhaustion, exact and
  exceeded chainMax/chainUntil boundaries, and recurrence-activation failure
  without mutation. Focused owner/registry checks pass 17/17. Registry: 516
  top-level / 504 registered / 12 retired / 2 removed ineffective / 365 WP4
  migrations / 0 duplicates; cumulative direct migrations: 478. Full gates need
  refresh.
- The captured-stderr live-render fallback now has a direct renderer contract,
  checking that Live is not started, static rows remain visible, stdout stays
  untouched, and no terminal escapes leak. Focused renderer coverage passes 6/6.
  Registry: 515 top-level / 503 registered / 12 retired / 2 ineffective removed
  / 366 WP4 migrations / 0 duplicates; cumulative direct migrations: 479.
- Five more direct contracts now cover cache-clear environment semantics,
  canonical chain identity, add-side exhaustion identity, Navigator projection
  failure evidence, and the dumb-terminal live-render guard. Focused owner and
  registry suites pass 58/58. Registry: 510 top-level / 498 registered / 12
  retired / 2 ineffective removed / 371 WP4 migrations / 0 duplicates;
  cumulative direct migrations: 484.
- Reconcile child-local-time evidence formatting now has a direct report
  contract using typed task/lifecycle models. The focused direct and registry
  suites pass 73/73. Registry: 509 top-level / 497 registered / 12 retired /
  2 ineffective removed / 372 WP4 migrations / 0 duplicates; cumulative direct
  migrations: 485.
- Navigator now has a direct contract preserving scheduler terminal metadata
  alongside valid projected dates. Focused Navigator/registry suites pass
  19/19. Registry: 508 top-level / 496 registered / 12 retired / 2 ineffective
  removed / 373 WP4 migrations / 0 duplicates; cumulative direct migrations:
  486. The latest full gates predate this single-contract change.
- Direct owner coverage now also includes astronomy-provider failure propagation,
  yearly-random chain isolation/distribution, explicit weekday-OR scheduling,
  earliest completion-cap selection, diagnostics stdout/stderr behavior, typed
  timeline failure/terminal rows, completed-row formatting, included-event
  preview limits, and static Rich panel layout/theme policy. Focused owner and
  registry suites pass 37/37. Registry: 498 top-level / 486 registered / 12
  retired / 2 ineffective removed / 383 WP4 migrations / 0 duplicates;
  cumulative direct migrations: 496.
- Two compact anchor-file acceptance tests now use the typed scheduler service
  and no longer depend on `dev_tools.legacy_preview_adapter`: long omission
  scans remain uncapped at the old 512-probe boundary, and fully omitted finite
  files return no included dates. Focused fixture/registry coverage passes 7/7.
  Registry: 496 top-level / 484 registered / 12 retired / 2 ineffective removed
  / 385 WP4 migrations / 0 duplicates; cumulative direct migrations: 498.
- Static Rich rendering now has direct coverage that its public static route
  prints the shared builder's renderable with the requested title, rows, kind,
  and theme. Focused renderer/registry coverage passes 14/14. Registry: 479
  top-level / 467 registered / 12 retired / 2 ineffective removed / 402 WP4
  migrations / 0 duplicates; cumulative direct migrations: 515.
- Another sixteen direct contracts moved into normal owner suites, covering
  seasonal boundary/scheduler behavior; Navigator calendar and symbolic-time
  projections; DST-fold selection and file-gap deduplication; hook completion
  lifecycle-result retention; modify-side progress, iteration-cap and cached
  provider behavior; recovery fail-closed evidence; and typed anchor-file
  metadata, context propagation, fallback-specific provider caching, and DST
  ordering. Focused migrated and adjacent suites pass 87/87. The deleted golden
  cases are recorded in the migrated-contract allowlist.
- At the historical 467-case checkpoint, registered functions were not yet individually classified
  as direct-contract candidates or justified acceptance cases. Marker counts
  from the earlier scan are stale and overlapping; they must be recomputed
  before being used as per-domain inventory. These textual signals are not
  case-level verdicts.
- Navigator direct module now includes the business-calendar and deterministic
  symbolic-time projection contracts; the remaining name-filtered Navigator
  golden slice: 20 tests, 0.846s. The legacy substring filter makes this a
  focused signal only, not an authoritative exclusive domain partition.
- CP direct module: 4 tests, 0.012s; the remaining CP name-filtered golden
  slice passed 27/27.
- Astronomy/year-ordinal direct suites and registry integrity: 23/23 passed;
  selected remaining year-ordinal golden scenarios: 2/2 passed. These slices
  overlap broad name filters and are not exclusive per-domain measures.
- Python compilation passes for the golden module and migrated direct-test
  modules. `git diff --check` reports trailing whitespace in unrelated hunks
  already present in the modified golden file; this batch did not alter those
  lines, so the whole-worktree whitespace check is not clean yet.
- Full unittest discovery passes 1156 tests with 3 optional-dependency skips;
  the golden runner passes all 467 registered acceptance cases in both normal
  and seeded-shuffle (`20260811`) order.
- File-backed direct contracts plus registry integrity pass 8/8; selected file
  occurrence/CSV golden acceptance cases pass 2/2.
- Twelve direct season-calendar and seasonal-selection contracts now run in
  `tests/recurrence/test_season_calendar_contracts.py`: fixed and astronomical
  windows, timezone and hemisphere boundaries, parser limits, ACF/cache
  round-tripping, generic/period-specific scheduling, boundary modifiers,
  overflow, and semantic validation.
- Seven business-calendar contracts now run under standard unittest discovery:
  default weekday roll/offset/ordinal behavior, normalized immutable
  definitions, unstable-rule/unmatched-file rejection, custom calendar flow
  through scheduler/filter/random-pool behavior, rules/files resolution and
  omission precedence, injected-calendar anchor/omit modifiers, and precise
  displacement capture. These extend
  `tests/test_business_calendar_contract.py`; the seven golden copies were
  removed. Focused direct plus registry tests pass 16/16. Provider and
  hook/on-add/on-modify/reconcile integration remain golden acceptance.
- Registry integrity is now 778 top-level / 767 registered / 11 explicitly
  retired / 105 newly migrated in this WP4 cycle / 0 duplicates.
- Deterministic anchor-file expansion/provider contracts now run from
  `tests/test_file_backed_contracts.py`, including bounded and random windows,
  typed occurrences, overnight/composed schedules, date/description modifiers,
  and task-level-time precedence. Core typed-provider/collector behavior is
  covered in `tests/test_occurrence_provider_contracts.py`; remaining DST
  integration, provider reuse, and hook projections stay in the golden suite.
- Pure deterministic time-window/random-window parsing, slot-limit and
  composable-schedule contracts now run in
  `tests/recurrence/test_time_windows_contracts.py`; parser/ACF/hook integration
  remains golden. Quarter-selector/rewrite and description-alias owner
  contracts now run in `tests/test_parser_owner_api_contracts.py`; preview and
  user-facing grammar integration remains golden.
- Current registry after parser-owner migrations: 778 top-level / 767
  registered / 11 explicitly retired / 105 newly migrated in this WP4 cycle /
  0 duplicates.
- Python compilation passed at that historical checkpoint. Its
  `git diff --check` warning concerned unrelated existing hunks and is
  superseded by the current clean whitespace check.
