# Desloppify Review Remediation Checklist

**Goal:** Resolve the 39 imported holistic-review findings while preserving hook safety, recurrence semantics, lifecycle durability, and compatibility.

**Scope:** Functional code and tests only. `local-archive/` remains excluded from desloppify. Do not treat generated reports, golden-test data, or compatibility facades as dead code without checking their runtime callers.

**Execution model:** Work on a dedicated branch. Each item is a small, independently testable change. Add a regression test before changing behavior, run the focused test, then run the relevant full suite. Only after verification should the matching desloppify issue be resolved.

**Global safety gates:**

- Keep Taskwarrior hook stdout strict JSON; diagnostics go to stderr under the existing diagnostic controls.
- Preserve `ensure_ascii=False` and malformed-input fail-safe behavior.
- Do not remove public compatibility exports, lazy-loading entries, or legacy data handling until callers and installed-layout hooks are tested.
- For recurrence, cursor, omission, lifecycle, and cache changes, retain explicit terminal evidence and idempotent mutation behavior.
- Run `python3 -m unittest discover -s tests -q`, hook contract tests, and `git diff --check` before each logical commit.
- After the queue is complete, run `desloppify scan --path .` and confirm no regressions before resolving remaining structural cascades.

## Phase 0 — Baseline and branch

- [ ] Create a dedicated branch for this remediation batch.
- [ ] Record the current desloppify queue and the 39 review IDs with `desloppify show review --status open --no-budget`.
- [ ] Capture baseline results for the recurrence, lifecycle, hook-contract, and operator-process test groups.
- [ ] Confirm `local-archive` remains excluded and do not add generated `.desloppify/` state to product commits.

## Phase 1 — Correctness, safety, and failure contracts

### Hook trust and invocation isolation

- [x] **Bootstrap trust ordering** — `on-add.nautical`, `on-modify.nautical`, `on-exit.nautical`, `nautical_core/hook_bootstrap.py`.
  - Validate candidate ownership/permissions with `trusted_core_base` before executing any `NAUTICAL_CORE_PATH` override.
  - Remove raw untrusted bootstrap fallbacks; retain only installed or repository-local candidates.
  - Add tests for owner mismatch, world-writable paths, explicit trust override, and valid installed paths.
  - Issue: `review::.::holistic::authorization_consistency::bootstrap_executes_override_before_trust_validation`

- [x] **Refresh exit policy per invocation** — `nautical_core/hooks/exit_impl.py`.
  - Replace import-time snapshots of strict mode, export timeout, retry, and lease settings with invocation-time resolution.
  - Pass resolved values explicitly into drain and exit-policy functions.
  - Test two invocations in one interpreter with different environment values.
  - Issue: `review::.::holistic::initialization_coupling::exit_policy_frozen_at_import`

- [x] **Rebuild integration context per invocation** — `nautical_core/hooks/exit_impl.py`.
  - Separate reusable module/import caches from per-run `IntegrationContext` and derived Taskdata values.
  - Ensure changed hook arguments and environment cannot reuse the first invocation’s context.
  - Add a sequential multi-invocation regression test.
  - Issue: `review::.::holistic::initialization_coupling::exit_context_survives_new_invocation`

- [x] **Apply modify-hook diagnostic redaction consistently** — `nautical_core/hooks/add_impl.py`, `nautical_core/hooks/modify_impl.py`, `nautical_core/hooks/exit_impl.py`, `nautical_core/runtime.py`.
  - Redact before diagnostic-event construction and before fallback stderr output.
  - Match sibling hook behavior and verify both normal-core and bootstrap-failure paths.
  - Add tests proving secrets never appear in emitted diagnostics.
  - Issue: `review::.::holistic::convention_outlier::modify_diag_skips_sibling_redaction`

### Recurrence and scheduler semantics

- [x] **Preserve cursor terminal evidence** — `nautical_core/recurrence_evaluator.py`, `nautical_core/occurrence_provider.py`, `nautical_core/scheduler_service.py`.
  - Return the provider’s `OccurrenceBatch` unchanged from cursor collection.
  - Keep DATE_LIMIT and other terminal evidence through `SchedulerService.collect`.
  - Add a regression test with a valid prefix and terminal exhaustion.
  - Issue: `review::.::holistic::contract_coherence::cursor_collection_drops_terminal`

- [x] **Enforce cursor date limits** — `nautical_core/scheduler_cursor.py`, `nautical_core/occurrence_provider.py`, `nautical_core/recurrence_evaluator.py`.
  - Carry `date_limit` through lazy and batch collection paths.
  - Preserve inclusive/exclusive semantics and terminal reporting at the limit.
  - Test both boundary inclusivity modes and a provider that returns an out-of-range next occurrence.
  - Issue: `review::.::holistic::contract_coherence::collection_ignores_cursor_date_limit`

- [x] **Make scheduler collection arguments explicit** — `nautical_core/scheduler_service.py`, `nautical_core/recurrence_evaluator.py`.
  - Replace keyword-presence dispatch on `count_omitted` with an explicit omission policy/count parameter.
  - Document accepted keywords and ensure `False` does not silently select an omission-preserving stream.
  - Test omitted, `False`, and `True` policy values and update all callers.
  - Issue: `review::.::holistic::api_surface_coherence::scheduler_collection_keyword_presence_dispatch`

- [x] **Continue past exhausted ordinary OR terms** — `nautical_core/scheduler_expr.py`.
  - Catch and record `OccurrenceSearchExhausted` for ordinary terms just as for random terms.
  - Raise only when every candidate is exhausted.
  - Test both term orders, one valid sibling, and all terms exhausted.
  - Issue: `review::.::holistic::logic_clarity::ordinary_or_exhaustion_aborts_valid_siblings`

### Input, lifecycle, and persistence failure handling

- [x] **Sanitize every declared mutable mapping** — `nautical_core/task_codec.py`.
  - Accept `MutableMapping` rather than only `dict`.
  - Test `dict`, `collections.UserDict`, and malformed values without changing unrelated fields.
  - Issue: `review::.::holistic::contract_coherence::sanitizer_rejects_declared_mapping_inputs`

- [x] **Contain non-finite link parsing failures** — `nautical_core/task_read_repository.py`, `nautical_core/task_set_reads.py`.
  - Catch `OverflowError` alongside existing numeric parsing exceptions.
  - Preserve `None`/MALFORMED outcomes and prevent `inf`/`1e999` from escaping authoritative reads.
  - Add focused regression tests.
  - Issue: `review::.::holistic::error_consistency::link_overflow_escapes_read_results`

- [x] **Keep outbox cleanup errors structured** — `nautical_core/lifecycle_outbox.py`.
  - Move permission enforcement inside result-producing error boundaries.
  - Guarantee database closure independently in `finally` for bulk and manual-review operations.
  - Test `PermissionError` during cleanup and verify structured outcomes plus closed connections.
  - Issue: `review::.::holistic::error_consistency::outbox_cleanup_overrides_results`

- [x] **Classify manual-review database locks as retryable** — `nautical_core/lifecycle_outbox.py`.
  - Route manual-review resolution through `_with_connection` so `sqlite3.OperationalError` preserves `lock_busy`/RETRYABLE semantics.
  - Add a locked-database regression test and retain permanent rejection for non-transient failures.
  - Issue: `review::.::holistic::error_consistency::manual_review_lock_classification_drift`

- [x] **Remove the impossible datetime fallback** — `nautical_core/modify_format_effects.py`, `nautical_core/time_api.py`.
  - Call `humanize_delta` with its required three-argument signature.
  - Do not catch and replace unrelated internal `TypeError` exceptions.
  - Test month/day formatting and genuine formatter errors.
  - Issue: `review::.::holistic::ai_generated_debt::unsupported_human_delta_fallback`

- [x] **Simplify identical link branches** — `nautical_core/chain_integrity_models.py`.
  - Collapse missing and invalid link branches only if behavior and diagnostics remain identical.
  - Retain integer and boolean handling tests.
  - Issue: `review::.::holistic::logic_clarity::identical_missing_invalid_link_branches`

## Phase 2 — Contract and migration cleanup

- [x] **Preserve concrete completion types** — `nautical_core/modify_models.py`, `nautical_core/modify_completion_compute.py`, `nautical_core/modify_completion_spawn.py`.
  - Replace `Any` with `datetime | None` and `LifecyclePlan | None` across results, callbacks, and spawn logic.
  - Payload-shaped metadata and parsed DNF values remain intentionally open because their schemas are heterogeneous.
  - Compare `LifecycleAction.SPAWN_CHILD` directly; keep only genuinely heterogeneous task payload fields dynamic.
  - Run mypy and completion/lifecycle tests.
  - Issue: `review::.::holistic::type_safety::completion_handoffs_erase_known_types`

- [x] **Annotate nullable CP parser APIs** — `nautical_core/cp_parser.py`.
  - Add the exact nullable return types for duration, interval, sequence, and token functions.
  - Add type-check and invalid-input tests.
  - Issue: `review::.::holistic::type_safety::cp_parser_public_nullable_returns_untyped`

- [x] **Make configuration loading use the typed protocol** — `nautical_core/config_support.py`, `nautical_core/core_config.py`.
  - Require `read_toml_result`, remove the obsolete dictionary-selection branch, and update all callers/tests.
  - Preserve distinctions between absent, invalid, and valid-empty configuration.
  - Issue: `review::.::holistic::incomplete_migration::unused_dict_config_loading_protocol`

- [x] **Remove retired holiday-region cache identity** — `nautical_core/config_schema.py`, `nautical_core/core_config.py`, `nautical_core/cache_api.py`, `nautical_core/cache_support.py`, `nautical_core/hint_builder_api.py`, `nautical_core/precompute.py`.
  - Keep deprecated-key recognition and migration guidance at the boundary.
  - Remove the ineffective value from cache keys, hint parameters, and serialized metadata.
  - Test equal cache identity for differing retired values and unchanged business-calendar fingerprints.
  - Issue: `review::.::holistic::incomplete_migration::retired_holiday_region_cache_dependency`

- [x] **Unify queue and mutation guard comparison** — `nautical_core/queue_status_service.py`, `nautical_core/taskwarrior_mutations.py`.
  - Expose one guard-comparison service/predicate owned by the mutation boundary.
  - Include recurrence fingerprint and all guard timestamps.
  - Test queue review and mutation application against the same changed-field matrix.
  - Issue: `review::.::holistic::mid_level_elegance::queue_review_partial_guard`

- [x] **Rename the non-reserving UUID helper** — `nautical_core/modify_command_effects.py`, `nautical_core/modify_spawn_prep.py`, `nautical_core/modify_spawn_effects.py`.
  - Rename `reserve_child_uuid` to `generate_child_uuid_candidate` through exports, callbacks, and wiring.
  - Preserve behavior and document that availability checks are advisory, not reservations.
  - Issue: `review::.::holistic::naming_quality::uuid_candidate_named_reservation`

- [x] **Make add workflow plans observable or remove them** — `nautical_core/hook_engine.py`, `nautical_core/hook_context.py`, `nautical_core/hooks/add_impl.py`, `nautical_core/add_workflow.py`.
  - Choose one behavior: remove unused plan construction/recording, or add an explicit persisted/returned consumer.
  - Do not retain rich metadata with no receiving boundary.
  - Add a test for the selected contract and update response documentation.
  - Issue: `review::.::holistic::mid_level_elegance::unused_add_workflow_handoff`

## Phase 3 — Reduce facade and ownership coupling

- [x] **Route internal APIs directly to owning modules** — `nautical_core/__init__.py`, `nautical_core/parser_api.py`, `nautical_core/scheduler_api.py`, `nautical_core/cache_api.py`.
  - Progress: introduced explicit owned `CoreContext` and context-aware parser/cache factories; legacy module-bound entry points remain during migration.
  - Progress: `CoreContext` now owns an isolated mutable namespace (required for cache state), and scheduler factories accept the same explicit context.
  - Progress: parser-support and ACF factories also accept explicit contexts; the remaining factories can be migrated independently before changing the lazy loader.
  - Progress: business-calendar and expansion factories now accept explicit contexts as well.
  - Progress: hint-builder and linting factories now accept explicit contexts as well.
  - Progress: natural-language, quarter, time, and token factories now accept explicit contexts; all API factories are prepared for a later loader cutover.
  - Central lazy loading now passes one explicit `CoreContext` to every migrated API factory while retaining compatibility aliases.
  - `CoreContext` now provides legacy attribute reads, allowing parser and scheduler implementations to execute against the context rather than the module object; provenance remains available separately for cache diagnostics.
  - Preserve intentional public re-exports while replacing internal mutable-facade lookups with direct imports and explicit dependencies.
  - Remove private facade wiring only after `rg` confirms no callers and installed-layout imports pass.
  - Issue: `review::.::holistic::cross_module_architecture::internal_dependencies_route_through_mutable_facade`

- [x] **Remove reverse dependency on hook globals** — `nautical_core/hooks/modify_impl.py`, `nautical_core/modify_composition.py`, `nautical_core/modify_effects.py`.
  - Progress: centralized the live hook-host view in `modify_impl.py`; repeated `hook_host(globals(), ...)` reconstruction is removed while the dynamic loader boundary remains intact.
  - Construct route services at the composition root and pass bounded runtime state/services into effects.
  - Remove unrestricted host/module lookups from extracted effect handlers.
  - Run modify hook integration and malformed-input tests.
  - Issue: `review::.::holistic::cross_module_architecture::modify_effects_depend_on_entrypoint_namespace`

- [x] **Move lifecycle recovery policy into its service** — `nautical_core/tools/nautical_reconcile.py`, `nautical_core/reconcile_operator_service.py`, `nautical_core/lifecycle_reconciliation.py`.
  - Move terminal timing validation, virtual-expiration construction, and recovery classification into lifecycle reconciliation.
  - Keep the CLI limited to argument handling, construction, rendering, and actual external effects.
  - Test dry-run, partial, error, and recovered terminal outcomes.
  - Progress: terminal timing validation and virtual-expiration construction now live in `LifecycleRecoveryPolicy`; the CLI supplies only parsing, comparison, validation, and identifier adapters.
  - Issue: `review::.::holistic::high_level_elegance::reconcile_cli_retains_recovery_policy`

- [x] **Reduce duplicated cache-lock wiring** — `nautical_core/cache_api.py`, `nautical_core/cache_locking.py`, `nautical_core/__init__.py`.
  - Keep lock-helper composition in `cache_locking.py`; retain only required public facade bindings.
  - Verify per-core configuration and compatibility callers before deleting aliases.
  - Progress: `cache_locking.bind_locking()` now binds the complete lock port once per core instance; existing facade names remain compatibility aliases.
  - Issue: `review::.::holistic::design_coherence::cache_lock_internal_wiring_duplicated`

- [x] **Bound the dynamic package facade** — `nautical_core/__init__.py`, `nautical_core/compat_api.py`, `nautical_core/parser_api.py`, `nautical_core/runtime.py`.
  - Lazy API bundles now share an explicit per-loader `CoreContext`; compatibility aliases remain available while execution is context-bound.
  - Preserve required external names, but remove unnecessary internal alias registration and private facade traversal.
  - Document the intentionally supported compatibility surface.
  - Run import, hook-loading, and public API tests.
  - Issue: `review::.::holistic::convention_outlier::dynamic_facade_obscures_module_ownership`

- [x] **Reorganize core domains incrementally** — `nautical_core/parsing/`, `nautical_core/operator/`, `nautical_core/hooks/modify/`, plus `nautical_core/runtime_manifest.py` and affected callers/tests.
  - Move one domain at a time; update static imports, string-based lazy module names, manifests, and installed-layout hooks together.
  - Do not move genuinely shared task/lifecycle services into a hook-specific package.
  - Run the full unittest suite and installed-layout/hook-loading checks after each move.
  - Progress: parser implementations now have canonical ownership under `nautical_core/parsing/`; historical module paths remain compatibility shims, and lazy parser loading uses the canonical package.
  - Issue: `review::.::holistic::package_organization::core_directory_erases_domain_boundaries`

- [x] **Separate Navigator analysis from presentation and interaction** — `nautical_navigator.py`.
  - Make analysis consume explicit snapshots and return existing Navigator view types.
  - Leave Rich rendering, prompts, process exit, and terminal interaction in the entry layer.
  - Add tests for analysis without a terminal and preserve interactive behavior.
  - Issue: `review::.::holistic::high_level_elegance::navigator_analysis_and_presentation_share_owner`

## Phase 4 — Preview and presentation simplification

### Canonical occurrence collection

- [x] **Unify preview collectors behind one typed occurrence stream** — `nautical_core/add_anchor_preview.py`, `nautical_core/occurrence_provider.py`.
  - Define one canonical collector returning `OccurrenceBatch[Occurrence]` with omission and terminal evidence intact.
  - Keep `exclude`, `include`, and `report` as thin projections over that stream.
  - Share provider construction, cursor handling, date/file-skip limits, and scheduler/evaluator routing.
  - Migrate production callers and golden tests before deleting duplicate collector branches.
  - Gate each step with the full unittest suite and canonical omission-fixture tests.

- [x] **Extract the canonical anchor-preview pipeline** — `nautical_core/add_anchor_preview.py`, `nautical_core/add_preview_composition.py`, `nautical_core/add_workflow.py`.
  - Separate source/first-occurrence resolution, task mutation/validation, and row rendering.
  - Merge file-only and merged-source branches using a shared resolved-occurrence value with data-driven error labels.
  - Preserve due assignment, limits, diagnostics, and compact/rich output tests.
  - Issues: `review::.::holistic::design_coherence::anchor_preview_mixes_schedule_and_rendering`, `review::.::holistic::low_level_elegance::preview_orchestrator_branch_tree`

- [x] **Remove the stale first-due contract** — `nautical_core/add_anchor_preview.py`.
  - Remove unused `dnf`, `seed_base`, evaluator, and obsolete imports from `anchor_preview_first_due` if the scheduler contract remains authoritative.
  - Update its caller and focused tests.
  - Issue: `review::.::holistic::low_level_elegance::stale_first_due_contract`

- [x] **Delete the uncalled file-only preview handler after migration** — `nautical_core/add_anchor_preview.py`, `nautical_core/add_preview_composition.py`.
  - Confirm no external API or installed hook calls the handler, migrate meaningful tests to the generalized path, then remove the handler/helper.
  - Issue: `review::.::holistic::low_level_elegance::duplicate_file_preview_handler`

- [x] **Remove competing preview collection implementations** — `nautical_core/add_anchor_preview.py`, `nautical_core/recurrence_evaluator.py`.
  - Progress: migrated the remaining golden-test direct call to `_collect_included_with_provider` and removed `_anchor_file_occurrences_local`; the evaluator-less compatibility branch remains for legacy direct callers.
  - Evaluator-less production collection fails closed; all legacy golden callers now use the explicitly named `dev_tools.legacy_preview_adapter`.
  - Make scheduler/evaluator collection canonical and isolate any compatibility adapter under an explicitly named test boundary.
  - Delete legacy provider branches and unused parameters only after direct-call tests migrate.
  - Issue: `review::.::holistic::low_level_elegance::legacy_collection_fallbacks`

- [x] **Remove dead pre-evaluator compute loops** — `nautical_core/add_anchor_compute.py`, `nautical_core/add_anchor_preview.py`.
  - Require the evaluator in `anchor_until_summary` and `anchor_build_preview`; remove legacy date-stepping loops and callback parameters.
  - Preserve evaluator-backed limits and preview regression tests.
  - Issue: `review::.::holistic::low_level_elegance::dead_legacy_compute_paths`

- [x] **Replace oversized presentation callback bags** — `nautical_core/add_anchor_preview.py`, `nautical_core/modify_timeline.py`, `nautical_core/modify_feedback.py`, `nautical_core/modify_presentation_effects.py`.
  - Progress: anchor preview's primary handler now receives a typed `AnchorPreviewServices` bundle from the composition root; timeline, file-preview, and completion paths remain to be migrated.
  - Progress: timeline task rendering now receives a typed `TimelineServices` bundle; file-preview and completion paths remain to be migrated.
  - Progress: completion renderers already consume typed `AnchorFeedbackServices`/`CpFeedbackServices`; the remaining migration is the large orchestration request passed through `modify_completion_flow`.
  - Progress: `modify_completion_flow` now passes `AnchorCompletionFeedbackModel`/`CpCompletionFeedbackModel` request objects instead of expanding twenty-plus keyword arguments, and `modify_feedback` consumes those request objects directly. Only the environment dependency bundle remains separate.
  - Progress: the standalone anchor-file preview handler now accepts a typed `AnchorFilePreviewServices` bundle as well.
  - Introduce bounded request/context/service dataclasses at the composition root.
  - Pass those objects across boundaries instead of dozens of callbacks and host lookups.
  - Test anchor preview, timeline, and both completion-feedback paths through the new contracts.
  - Issue: `review::.::holistic::abstraction_fitness::hook_presentation_callback_bags`

- [x] **Remove chain-summary callback round trips** — `nautical_core/modify_chain_summary.py`, `nautical_core/modify_diagnostics_effects.py`.
  - Let the summary renderer call its local row helpers directly after receiving resolved facts.
  - Keep external reads and final panel emission at the effects boundary.
  - Issue: `review::.::holistic::design_coherence::chain_summary_callback_roundtrip`

## Phase 5 — Tests, dependencies, and final review quality

- [x] **Make timeout tests deterministic** — `tests/test_operator_process_contract.py`, `nautical_core/taskwarrior_client.py`.
  - Use an explicit child readiness signal before testing descendant termination.
  - Separate deterministic partial-output collection from the real timeout smoke test and use a realistic startup budget.
  - Assert descendant termination rather than merely bounded parent return.
  - Issue: `review::.::holistic::test_strategy::process_timeout_startup_races`

- [x] **Test renderer behavior instead of source text** — `tests/test_effect_boundary.py`, `nautical_core/modify_feedback.py`.
  - Replace `inspect.getsource`/call-count assertions with a recording renderer and assertions on delegated `PanelView` values for all three paths.
  - Issue: `review::.::holistic::test_strategy::renderer_contract_source_count`

- [x] **Add reproducible runtime dependency constraints** — `requirements.txt`, `requirements-constraints.txt` or lock file, `nautical_core/backup_service.py`, `dev_tools/nautical_offline_kit.py`.
  - Pin a tested compatible range/exact set for runtime dependencies and make bootstrap/offline recovery consume it.
  - Update versions only through compatibility-tested changes; retain recorded-version diagnostics.
  - Verify clean installation from the constraints file.
  - Issue: `review::.::holistic::dependency_health::unlocked_runtime_versions`

- [ ] **Resolve the remaining review findings in desloppify**.
  - For each completed checklist item, run its focused tests, record the commit, then execute `desloppify plan resolve <issue-id>` with a concise evidence note.
  - Use `desloppify show review --status open` to ensure no issue is skipped or accidentally resolved by a broad pattern.

- [ ] **Run final verification and rescan**.
  - Run the full Python test suite, hook contract suite, mypy/compile checks, shell syntax checks, and `git diff --check`.
  - Run `desloppify scan --path .` with `local-archive` excluded, then `desloppify status` and `desloppify next`.
  - Confirm functional findings are resolved or explicitly documented with evidence; do not claim completion while the queue contains unresolved correctness/security issues.
