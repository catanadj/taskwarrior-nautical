"""Integrity checks for the developer golden-test registry.

These checks verify golden registration, migration allowlists, and the reviewed
exclusive acceptance-domain inventory.
"""

import hashlib
import ast
import importlib
from pathlib import Path
import unittest


GOLDEN_MODULE = "dev_tools.nautical_golden_tests"
EXPECTED_GOLDEN_REGISTRY_ORDER_SHA256 = (
    "e8189832b71d9a0f598b4c52d4f314724629054d5a1863456b8c18f0f44371b7"
)
GOLDEN_ACCEPTANCE_DOMAIN_MARKERS = (
    (
        "lifecycle and durable mutation",
        ("lifecycle", "outbox", "mutation_service", "child_import", "integration_contract", "staged_plan", "carry_field"),
    ),
    ("reconcile and recovery", ("reconcile", "backfill")),
    (
        "install and deployment",
        ("installer", "runtime_cleanup", "retained_release", "deploy", "ops_templates", "layout", "package_core"),
    ),
    ("operator/query/Doctor/Navigator", ("operator", "doctor", "query", "queue", "health_check", "navigator")),
    ("performance and soak", ("perf", "benchmark", "soak", "replay", "mixed_recurrence_loop", "load_benchmark")),
    ("configuration and bootstrap", ("config", "taskdata", "core_import", "protocol", "unsafe", "business_calendar_toml")),
    (
        "storage and filesystem safety",
        ("backup", "restore", "safe_lock", "diag_log", "cache_dir"),
    ),
)
EXPECTED_GOLDEN_ACCEPTANCE_DOMAINS = {
    "configuration and bootstrap": (
        1,
        "2e6634db21ef0824b2069d5a122310169c6ea361592970b98c8491b2e3017fb6",
    ),
    "install and deployment": (
        9,
        "c37f811dc74ed2c50af824c98849b329be7772e87f71b36fa26a6aa3ed47679c",
    ),
    "lifecycle and durable mutation": (
        22,
        "2aa8a39d79f30a39d4cc41845b0f86d303f7cf28ee1b5b06b8afe2e1d6e397f7",
    ),
    "operator/query/Doctor/Navigator": (
        23,
        "5295decb4d3d885cb8271b8c901536e3396affcbb6ee162389c4ad0b77d5fc3f",
    ),
    "performance and soak": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "reconcile and recovery": (
        7,
        "3f231d13386fde51907b43f62cd02da910cccfd81438f6dda155f3db77cc59d2",
    ),
    "recurrence and hook integration": (
        75,
        "5b84614239c951daeb965d7d723f41233564c7eed09614408a5992ea6b1c0e10",
    ),
    "storage and filesystem safety": (
        1,
        "656cc935f9c5ca336334754710e073cd6dfedfedefcd574f2d3f72fc47f715d6",
    ),
}
RETIRED_CHARACTERIZATION_TESTS = frozenset()
REMOVED_INEFFECTIVE_TESTS = frozenset(
    {
        "test_core_import_deterministic",
        "test_hook_run_task_falls_back_when_core_load_fails",
        "test_on_add_run_task_falls_back_when_core_load_fails",
        "test_performance_large_expressions",
        "test_on_add_preview_hard_cap",
        "test_doctor_reports_reconcile_backfill_plans",
        "test_reconcile_expiration_plan_reuses_limits_and_deleted_slot_dedup",
    }
)
MIGRATED_DIRECT_CONTRACT_TESTS = frozenset(
    {
        "test_reconcile_apply_refuses_a_second_full_run",
        "test_reconcile_apply_lease_serializes_mutations",
        "test_reconcile_subprocess_output_contracts",
        "test_reconcile_repairs_invalid_native_until_from_previous_link",
        "test_recurrence_update_udas_use_canonical_key_and_filter_aliases",
        "test_live_panel_duration_defaults_and_clamps",
        "test_live_panel_footer_defaults_and_accepts_configured_text",
        "test_uda_aliases_remain_opt_in_through_configuration",
        "test_business_calendar_toml_resolves_through_public_core_api",
        "test_configuration_drift_detects_edit_and_removal",
        "test_scheduler_config_changes_cache_keys_but_ui_changes_do_not",
        "test_taskdata_reload_keeps_validated_fingerprints_consistent",
        "test_taskdata_discovery_rejects_malformed_configuration",
        "test_operator_context_rejects_malformed_toml_and_invalid_timezone",
        "test_invalid_timezone_uses_utc_fallback_and_blocks_scheduling",
        "test_world_writable_explicit_config_fails_closed",
        "test_random_salt_replays_and_namespaces_random_draws",
        "test_query_process_boundary_emits_one_json_document",
        "test_query_installed_layout_runs_outside_checkout",
        "test_shared_time_slot_resolver_keeps_hook_and_navigator_parity",
        "test_on_add_requires_integration_context_helper",
        "test_deploy_sanity_script_reports_ok",
        "test_deploy_sanity_enforces_removed_lifecycle_ownership",
        "test_deploy_sanity_rejects_missing_lazy_lifecycle_module",
        "test_deploy_sanity_rejects_missing_operator_runtime_tool",
        "test_deploy_sanity_rejects_unowned_taskwarrior_subprocess",
        "test_health_check_json_ok_empty_taskdata",
        "test_queue_status_json_ok_empty_taskdata",
        "test_queue_status_explicit_prune_reports_maintenance_result",
        "test_operator_queue_status_json_ok_empty_taskdata",
        "test_nautical_dispatches_supported_subcommands",
        "test_perf_cold_import_records_module_profile",
        "test_ops_templates_present_and_runner_executable",
        "test_perf_hint_benchmark_isolates_persistent_cache",
        "test_perf_hook_fast_path_ratio_enforcement",
        "test_hook_replay_harness_reports_ok",
        "test_mixed_recurrence_loop_harness_reports_ok",
        "test_soak_runner_reports_ok",
        "test_load_benchmark_installs_complete_hook_runtime",
        "test_load_benchmark_queue_and_lineage_verification",
        "test_chain_colour_uses_complete_root_identity",
        "test_cp_interval_helpers_agree_between_on_add_and_on_modify",
        "test_on_modify_compute_cp_sequence_selects_interval_by_link",
        "test_on_modify_compute_cp_random_selects_deterministic_interval",
        "test_on_modify_cp_sequence_estimates_chainmax_final_date",
        "test_on_modify_anchor_chainmax_forecast_is_bounded",
        "test_on_modify_compute_cp_child_due_uses_scheduled_when_due_missing",
        "test_on_modify_compute_anchor_child_due_uses_scheduled_seed_for_all_mode",
        "test_on_modify_compute_anchor_child_due_builds_timed_slots_in_configured_timezone",
        "test_on_modify_anchor_file_child_projection_reuses_provider",
        "test_on_modify_pure_anchor_file_projection_reuses_provider",
        "test_on_modify_anchor_dnf_accepts_configured_preset",
        "test_on_modify_omit_dnf_accepts_configured_preset",
        "test_on_modify_compute_anchor_child_due_accepts_scheduled_after_due",
        "test_on_modify_compute_anchor_child_due_skips_omit_date",
        "test_on_modify_compute_anchor_child_due_unsatisfiable_omit_fails",
        "test_on_modify_compute_counted_random_advances_within_period",
        "test_on_add_anchor_file_root_gets_chainid_stamp",
        "test_on_add_chainid_stamp_failure_rejects_recurring_root",
        "test_on_add_anchor_preview_auto_assigns_when_due_matches_entry",
        "test_on_add_due_context_treats_due_matching_entry_as_implicit",
        "test_hook_on_add_anchor_preview_skips_omit_date",
        "test_hook_on_add_anchor_preview_skips_omit_file_date",
        "test_hook_on_add_anchor_preview_marks_omitted_future_slots",
        "test_hook_on_add_anchor_preview_uses_omit_file_description_in_upcoming",
        "test_hook_on_add_anchor_preview_skips_omit_file_modifier_date",
        "test_safe_lock_fallback_stale_cleanup",
        "test_safe_lock_fallback_stale_pid_cleanup",
        "test_safe_lock_fallback_contention",
        "test_core_cache_dir_and_lock_permissions",
        "test_core_cache_lock_contention_matches_safe_lock",
        "test_core_cache_dir_rejects_symlink_override",
        "test_hook_files_are_private_permissions",
        "test_diag_log_rotation_bounds",
        "test_diag_log_redacts_sensitive_fields",
        "test_hook_diag_redact_msg_masks_sensitive_json_fields",
        "test_on_add_format_anchor_rows_numbers_upcoming_from_three_with_next_anchor",
        "test_on_add_format_anchor_rows_numbers_upcoming_from_two_without_next_anchor",
        "test_on_add_preview_and_completion_skip_choose_same_next_anchor",
        "test_on_modify_render_anchor_completion_feedback_wrapper",
        "test_on_modify_render_cp_completion_feedback_jitter_selected_interval",
        "test_on_modify_render_cp_completion_feedback_random_selected_interval",
        "test_on_modify_render_cp_completion_feedback_text_mode",
        "test_on_modify_render_anchor_file_completion_feedback_wrapper",
        "test_on_modify_render_cp_completion_feedback_wrapper",
        "test_on_modify_completion_panel_distinguishes_expiration_and_chain_boundaries",
        "test_on_add_anchor_and_anchor_file_can_coexist",
        "test_on_add_preview_uses_configured_chain_colour",
        "test_on_add_position_selection_renders_semantic_advice",
        "test_on_add_preview_fails_closed_when_evaluator_initialization_fails",
        "test_on_add_preview_reports_scheduler_exhaustion_actionably",
        "test_on_add_preview_uses_evaluator_for_first_due_and_upcoming_rows",
        "test_on_add_fail_and_exit_emits_json",
        "test_on_add_panic_passthrough_emits_valid_json",
        "test_on_add_rejects_oversized_stdin_early",
        "test_on_add_dnf_cache_uses_central_api_and_fingerprints_parser",
        "test_on_add_dnf_cache_quarantines_central_corruption",
        "test_on_add_no_explicit_taskdata_skips_rc_data_location",
        "test_on_add_reads_data_arg_from_hook_argv",
        "test_on_add_data_arg_overrides_taskdata_env",
        "test_on_modify_no_explicit_taskdata_skips_rc_data_location",
        "test_on_modify_reads_data_arg_from_hook_argv",
        "test_on_modify_data_arg_overrides_taskdata_env",
        "test_on_exit_reads_data_arg_from_hook_argv",
        "test_on_exit_data_arg_overrides_taskdata_env",
        "test_on_modify_missing_taskdata_uses_tw_dir",
        "test_on_modify_ignores_unsafe_core_path_override",
        "test_on_modify_requires_integration_context_helper",
        "test_on_exit_requires_integration_context_helper",
        "test_delete_chain_summary_span_uses_stop_time_without_last_end",
        "test_end_summary_history_marks_deleted_pending_tail",
        "test_delete_chain_summary_uses_stopped_title",
        "test_on_modify_manual_delete_persists_chain_off",
        "test_on_modify_expiration_wrapper_preserves_json_stdout",
        "test_hook_bootstrap_uses_symlink_path_and_core_path_rescue",
        "test_hooks_survive_malformed_numeric_environment",
        "test_full_hooks_receive_one_explicit_integration_context",
        "test_light_taskdata_resolution_matches_hook_precedence",
        "test_full_hook_modules_defer_core_import",
        "test_plain_hook_fast_paths_do_not_import_core_package",
        "test_full_hooks_reuse_wrapper_protocol_probe",
        "test_hooks_require_package_core_layout",
        "test_on_modify_panic_passthrough_uses_latest_task",
        "test_on_modify_invalid_anchor_has_no_stdout",
        "test_on_modify_rejects_oversized_stdin_early",
        "test_ui_live_test_term_guard_restores_environment",
        "test_core_import_defers_panel_colour_module",
        "test_core_import_defers_diagnostic_model",
        "test_core_import_defers_parser_scheduler_models",
        "test_core_import_defers_optional_stacks",
        "test_hook_stdout_strict_json_with_diag_on_add",
        "test_hook_stdout_strict_json_with_diag_on_modify",
        "test_hook_stdout_unicode_unescaped_on_add",
        "test_hook_stdout_unicode_unescaped_on_modify",
        "test_on_modify_invalid_json_passthrough",
        "test_hook_stdout_empty_on_exit",
        "test_on_modify_expiration_panel_explains_carry",
        "test_on_modify_expiration_delegates_to_extracted_orchestration",
        "test_on_modify_expiration_internal_failure_remains_recoverable",
        "test_on_add_rejects_oversized_stdin_early",
        "test_on_add_dnf_cache_uses_central_api_and_fingerprints_parser",
        "test_on_add_dnf_cache_quarantines_central_corruption",
        "test_on_add_no_explicit_taskdata_skips_rc_data_location",
        "test_on_add_reads_data_arg_from_hook_argv",
        "test_position_selection_on_add_and_modify_completion",
        "test_position_selection_modify_timeline_projects_future_dates",
        "test_position_selection_post_modifiers_modify_completion",
        "test_position_selection_public_period_scopes_hooks",
        "test_on_add_seasonal_selection_feedback",
        "test_hook_on_add_uses_and_normalizes_business_calendar",
        "test_hook_on_add_reports_business_calendar_displacement_only_when_shifted",
        "test_hook_on_add_rejects_unknown_business_calendar_cleanly",
        "test_hook_on_add_rejects_invalid_timezone_for_nautical_task",
        "test_hook_on_modify_rejects_unknown_business_calendar_cleanly",
        "test_on_modify_spawned_child_preserves_business_calendar",
        "test_hook_on_modify_rejects_invalid_timezone_for_nautical_task",
        "test_hook_on_add_multitime_preview_emits_all_slots",
        "test_hook_on_add_time_window_preview_emits_bounded_slots",
        "test_on_modify_time_window_completion_advances_within_same_day",
        "test_on_modify_partitioned_window_completion_rolls_to_next_day",
        "test_hook_on_add_overnight_window_keeps_json_and_next_day_preview",
        "test_hook_on_add_random_time_window_keeps_json_and_preview",
        "test_hook_on_add_anchor_preset_resolves_from_config",
        "test_on_modify_overnight_window_completion_uses_next_day_slots",
        "test_on_modify_random_time_window_completion_reuses_stable_slots",
        "test_time_window_dst_gap_deduplicates_shifted_local_slot",
        "test_partitioned_time_window_dst_gap_deduplicates_shifted_local_slot",
        "test_overnight_time_window_dst_fallback_deduplicates_repeated_local_slot",
        "test_chain_until_overnight_window_survives_dst_fallback",
        "test_hook_on_add_live_panel_mode_preserves_captured_protocol",
        "test_hook_on_add_counted_random_preview_uses_group_time",
        "test_hook_on_add_accepts_group_date_modifiers",
        "test_hook_on_add_anchor_unknown_preset_fails_cleanly",
        "test_hook_on_add_anchor_composed_preset_resolves_from_config",
        "test_hook_on_add_anchor_recursive_preset_fails_cleanly",
        "test_hook_on_add_omit_preset_resolves_from_config",
        "test_hook_on_add_omit_unknown_preset_fails_cleanly",
        "test_hook_on_add_omit_recursive_preset_fails_cleanly",
        "test_hook_on_add_anchor_preview_rolled_business_day_uses_timed_slot",
        "test_hook_on_add_anchor_preview_positive_day_offset_uses_timed_slot",
        "test_hook_on_add_anchor_preview_negative_day_offset_uses_timed_slot",
        "test_hook_on_add_timed_omit_rejected",
        "test_hook_on_add_invalid_omit_file_rejected",
        "test_hook_on_add_cp_sequence_preview_accepts_string_periods",
        "test_hook_on_add_cp_random_preview_shows_selected_periods",
        "test_hook_on_add_cp_random_preview_uses_stamped_chain_id",
        "test_hook_on_add_cp_random_malformed_fails_with_guidance",
        "test_hook_on_add_cp_jitter_preview_shows_selected_periods",
        "test_on_add_native_until_requires_strictly_later_target",
        "test_on_add_native_until_checks_generated_cp_due",
        "test_on_add_native_until_checks_generated_anchor_due",
        "test_on_add_native_until_guard_ignores_ordinary_tasks",
        "test_on_add_native_until_rejects_strict_anchor_modes",
        "test_on_add_preview_distinguishes_expiration_from_chain_end_point",
        "test_on_add_chain_until_rejects_before_first_anchor_occurrence",
        "test_hook_on_add_omit_timed_preset_rejected",
        "test_hook_on_add_unsatisfiable_omit_fails_cleanly",
        "test_hook_on_add_cp_scheduled_only_preserves_no_due",
        "test_hook_on_add_anchor_scheduled_only_preserves_no_due",
        "test_hook_on_add_anchor_file_preview_auto_assigns_first_match",
        "test_hook_on_add_anchor_and_anchor_file_preview_uses_earliest_union_match",
        "test_hook_on_add_anchor_file_time_padding_hint",
        "test_hook_on_add_rejects_invalid_chain_max_for_cp_and_anchor",
        "test_on_add_expands_enabled_description_uda_aliases",
        "test_hook_on_add_uda_aliases_emit_canonical_json_and_reject_conflicts",
        "test_hook_on_add_disabled_uda_aliases_leave_description_untouched",
        "test_on_add_ignores_unsafe_core_path_override",
        "test_hook_on_add_cp_malformed_inputs_fail_with_parser_guidance",
        "test_on_add_profiler_lazy_init",
        "test_on_add_flushes_stdout",
        "test_on_modify_chain_cache_thread_safety_smoke",
        "test_on_modify_chain_cache_reads_through_typed_repository",
        "test_on_modify_chain_cache_preserves_repository_unavailability",
        "test_on_modify_predecessor_read_preserves_repository_unavailability",
        "test_on_modify_collect_prev_two_prefers_live_statuses",
        "test_on_modify_get_chain_export_filters_cached_chain_in_memory",
        "test_tw_export_chain_extra_validation",
        "test_tw_export_chain_extra_rejects_dash_prefixed_tokens",
        "test_on_modify_diag_blocks_pretty_print",
        "test_on_exit_diag_blocks_pretty_print",
        "test_on_modify_lifecycle_diagnostics_are_gated_to_stderr",
        "test_on_modify_run_task_diag_bucket_stats",
        "test_on_exit_outcome_diagnostics_are_bounded",
        "test_on_exit_emit_exit_feedback_reaches_stdout_contract",
        "test_on_modify_state_files_use_dedicated_dir",
        "test_on_modify_read_two_uuid_mismatch_without_nautical_fields_is_ignored",
        "test_on_add_lowercase_chainid_does_not_mark_nautical",
        "test_on_modify_read_two_fuzz_inputs",
        "test_on_add_read_one_fuzz_inputs",
        "test_on_modify_read_two_invalid_trailing",
        "test_on_modify_read_two_array_uuid_mismatch_fails",
        "test_on_modify_read_two_array_single_missing_uuid_fails",
        "test_on_modify_read_two_single_plain_delete_without_uuid_is_ignored",
        "test_on_add_run_task_timeout",
        "test_on_modify_run_task_timeout",
        "test_config_schema_reports_retired_unknown_and_ineffective_values",
        "test_build_and_cache_hints_routes_scheduler_through_service",
        "test_task_business_calendar_context_selects_and_restores_policy",
        "test_astronomy_preflight_reports_configuration_and_provider_health",
        "test_astronomy_none_event_is_actionable",
        "test_navigator_snapshot_metadata_preserves_typed_chain_identity",
        "test_navigator_sparse_calendar_renders_only_active_months",
        "test_core_render_panel_line_force_rich_kind_skips_panel_line",
        "test_moon_phase_intersection_fails_closed_without_synthetic_date",
        "test_omit_scheduler_failures_do_not_fail_open",
        "test_ui_live_renderer_reveals_timeline_without_highlight",
        "test_ui_live_renderer_reveals_cumulative_row_frames",
        "test_ui_live_renderer_reveals_multiline_values_progressively",
        "test_ui_live_animation_policy_caps_motion_and_prioritizes_urgent_panels",
        "test_ui_live_mid_animation_failure_settles_without_static_duplicate",
        "test_ui_live_oversized_panel_settles_without_starting_animation",
        "test_cache_load_quarantines_corrupt_entries_and_gc_removes_them",
        "test_compiled_schedule_is_canonical_and_reusable",
        "test_weekday_weekend_single_time",
        "test_hint_cache_keys_include_semantic_fingerprint",
        "test_core_domain_configuration_validation_fails_closed",
        "test_completion_parent_guard_uses_persisted_terminal_timestamp",
        "test_taskwarrior_client_preserves_evidence_and_redacts_observation",
        "test_hook_protocol_loads_without_core_package",
        "test_hooks_no_direct_subprocess_run",
        "test_lifecycle_outbox_prunes_only_expired_acknowledged_rows",
        "test_lifecycle_outbox_bulk_compare_and_set_operations_isolate_rows",
        "test_lifecycle_outbox_claims_quarantine_exhausted_and_inconsistent_rows",
        "test_child_import_rejects_incomplete_existing_rows",
        "test_lifecycle_child_prefetch_reuses_authoritative_uuid_set_read",
        "test_lifecycle_batch_prefetch_uses_one_union_uuid_set_read",
        "test_batch_postverification_fails_closed_for_untrusted_snapshots",
        "test_modify_lifecycle_activation_requires_complete_root_identity",
        "test_moon_phase_source_and_filter_compose_with_weekday",
        "test_moon_phase_source_emits_once_per_phase_window",
        "test_occurrence_collection_preserves_prefix_before_date_terminal",
        "test_modify_timeline_marks_omit_evaluation_failures",
        "test_anchor_and_file_tie_preserves_file_description",
        "test_modify_until_past_guard_orders_dst_fold_by_instant",
        "test_ui_live_failure_preserves_rows_for_static_fallback",
        "test_modify_ordinary_transition_failure_rejects_instead_of_noop",
        "test_completion_scheduler_terminal_outcomes_are_not_spawned",
        "test_chain_cap_guards_are_inclusive_at_boundary",
        "test_ui_live_mode_non_tty_falls_back_without_live_control_codes",
        "test_clear_all_caches_env",
        "test_chainid_legacy_reads_do_not_drive_chain_identity",
        "test_anchor_step_preserves_scheduler_exhaustion",
        "test_navigator_surfaces_anchor_projection_failures",
        "test_ui_live_renderer_rejects_dumb_terminal",
        "test_reconcile_evidence_includes_local_child_time_when_formatter_available",
        "test_navigator_projection_preserves_scheduler_terminal_evidence",
        "test_moon_phase_operational_errors_are_actionable",
        "test_yearly_rand_uses_independent_chain_scoped_draws",
        "test_random_weekday_explicit_or_keeps_separate_draws",
        "test_completion_caps_earliest_limit_wins",
        "test_diagnostic_event_renders_to_stderr_and_has_stable_record",
        "test_modify_timeline_marks_projection_failures_instead_of_silent_truncation",
        "test_modify_timeline_preserves_typed_terminal_projection_evidence",
        "test_add_preview_event_collection_counts_only_included_occurrences",
        "test_timeline_completed_rows_place_uuid_before_delta",
        "test_ui_build_rich_panel_preserves_static_layout_and_theme",
        "test_compact_anchor_file_lookup_scans_past_legacy_probe_limit",
        "test_compact_anchor_file_lookup_reports_cursor_exhaustion",
        "test_ui_static_rich_renderer_delegates_to_shared_builder",
        "test_astronomical_season_calculator_contract",
        "test_seasonal_selection_scheduler_windows_and_rollover",
        "test_navigator_uses_task_business_calendar_for_anchor_projection",
        "test_modify_anchor_file_mode_orders_dst_fold_by_instant",
        "test_hook_engine_retains_completion_lifecycle_result_on_runtime_context",
        "test_modify_inclusion_collection_uses_shared_progress_guard",
        "test_modify_until_projection_fails_closed_at_iteration_limit",
        "test_modify_until_projection_reuses_anchor_file_provider",
        "test_integrity_recovery_fault_matrix_fails_closed",
        "test_navigator_resolves_symbolic_anchor_time_offsets",
        "test_add_anchor_file_local_projection_deduplicates_dst_gap",
        "test_merged_anchor_file_provider_carries_context_and_reuses_specs",
        "test_event_provider_preserves_anchor_file_source_description",
        "test_included_provider_reuses_shared_anchor_file_provider",
        "test_included_provider_rebuilds_shared_provider_when_fallback_changes",
        "test_anchor_file_next_occurrence_after_uses_shared_dst_ordering",
        "test_anchor_file_expression_merges_sources_and_applies_group_modifiers",
        "test_file_source_wildcards_are_deterministic_and_star_dot_star_means_all",
        "test_anchor_file_expression_preserves_per_source_times_and_dedupes_matches",
        "test_omit_file_expression_merges_sources_atomically_and_rejects_group_times",
        "test_file_source_symlink_must_remain_inside_configured_directory",
        "test_file_backed_csv_missing_date_column_reports_columns",
        "test_file_backed_empty_or_no_usable_dates_fail_cleanly",
        "test_file_backed_cache_detects_same_size_content_replacement",
        "test_file_backed_cache_uses_metadata_for_hot_reads_and_bounds_lru",
        "test_file_backed_cache_reuses_digest_matches_after_metadata_changes",
        "test_file_backed_resource_limits_reject_oversized_inputs",
        "test_file_source_resource_limits_bound_wildcard_fanout",
        "test_file_date_aggregate_limits_cover_anchor_omit_and_business_calendars",
        "test_seasonal_selection_business_calendar_and_cache_identity",
        "test_astronomical_time_skips_unavailable_candidate_dates",
        "test_cache_load_retries_when_file_is_replaced_during_read",
        "test_core_anchor_preset_unknown_lists_available_names",
        "test_core_omit_preset_unknown_lists_available_names",
        "test_core_preset_recursion_chain_is_deterministic",
        "test_core_nested_preset_display_shows_resolved_leaf",
        "test_anchor_cache_cleans_stale_tmp_files",
        "test_anchor_cache_garbage_collection_prunes_expired_and_overflow",
        "test_cache_metrics_emits_when_enabled",
        "test_diag_log_structured_fields",
        "test_warn_rate_limited_any",
        "test_business_calendar_fingerprint_invalidates_rule_file_and_hint_caches",
        "test_build_local_datetime_dst_gap_and_ambiguous",
        "test_dst_round_trip_noon_preserves_local_date",
        "test_config_support_reports_automatically_discovered_toml_parse_errors",
        "test_config_support_rejects_unsafe_toml_and_reports_reason",
        "test_config_support_distinguishes_empty_missing_and_invalid_candidates",
        "test_integration_mutation_models_enforce_guards_and_postconditions",
        "test_integration_mutation_requests_use_named_typed_payloads",
        "test_integration_outbox_models_enforce_deterministic_identity_and_progress",
        "test_integration_context_resolves_and_validates_invocation_once",
        "test_next_for_and_transient_stall_recovers",
        "test_navigator_uses_nautical_configured_timezone",
        "test_moon_phase_real_astral_boundary_smoke",
        "test_moon_astral_events_preserve_timezone_and_dst",
        "test_moonrise_unavailable_location_fails_closed",
        "test_hook_runtime_retains_module_import_failure_details",
        "test_season_mode_configuration_contract",
        "test_on_modify_loaded_empty_snapshot_prevents_full_timeline_export",
        "test_ui_render_panel_routes_live_mode_without_static_duplicate",
        "test_completion_preflight_stops_on_unavailable_next_lookup",
        "test_on_modify_completion_defers_chain_export_until_after_preflight",
        "test_on_modify_completion_snapshot_malformed_json_is_unavailable",
        "test_next_for_and_no_progress_fails_fast",
        "test_normalize_spec_for_acf_cache_guards",
        "test_shipped_config_keeps_hook_toggles_top_level",
        "test_shipped_config_matches_authoritative_schema",
        "test_core_render_panel_line_mode_uses_panel_line",
        "test_hook_task_runner_handles_nonzero",
        "test_shared_hook_subprocess_runner_preserves_output_and_status",
        "test_hook_task_result_preserves_typed_runner_result",
        "test_core_run_task_tempfiles_fallback_handles_bytes_input",
        "test_roll_apply_has_guard",
        "test_build_and_cache_hints_returns_isolated_cached_payload",
        "test_moon_phase_contradictions_are_rejected",
        "test_core_run_task_tempfiles_accepts_text_input",
        "test_core_run_task_timeout_reports_timeout_with_tempfiles",
        "test_core_run_task_result_exposes_typed_metadata",
        "test_core_run_task_nonzero_retries_use_expected_backoff",
        "test_core_run_task_does_not_retry_ordinary_nonzero",
        "test_config_exposes_anchor_file_dir",
        "test_on_add_compact_anchor_preview_requests_one_occurrence",
        "test_omit_file_modifiers_roll_dates_and_carry_descriptions",
        "test_omit_file_modifiers_support_negative_day_offsets",
        "test_omit_file_modifiers_apply_even_when_base_file_is_cached",
        "test_omit_file_modifiers_reject_time_modifiers",
        "test_sanitize_task_strings_removes_controls",
        "test_navigator_empty_snapshot_is_a_valid_empty_chain",
        "test_navigator_and_query_share_task_chain_facts",
        "test_navigator_snapshot_exposes_deterministic_typed_view",
        "test_navigator_calendar_view_is_immutable_and_serializable",
        "test_navigator_chain_summary_is_immutable_and_serializable",
        "test_navigator_chain_choice_is_immutable_and_serializable",
        "test_navigator_change_row_is_immutable_and_serializable",
        "test_navigator_task_detail_view_is_immutable_and_serializable",
        "test_navigator_projection_view_is_immutable_and_serializable",
        "test_navigator_trace_view_is_immutable_and_serializable",
        "test_navigator_analysis_view_aggregates_typed_sections",
        "test_moon_phase_anchor_grammar_normalizes_canonical_names",
        "test_astronomy_profile_requires_explicit_timezone",
        "test_moon_phase_resolver_uses_circular_phase_distance",
        "test_moon_phase_resolver_uses_documented_phase_bands",
        "test_cp_duration_parser_and_dst_preserve_whole_days",
        "test_cp_sequence_link_boundary_contract",
        "test_cp_random_and_jitter_are_deterministic_per_link_and_bounded",
        "test_cp_random_seed_is_chain_scoped_and_normalized",
        "test_year_day_ordinals_validate_strictly",
        "test_year_day_ordinals_expand_and_schedule",
        "test_iso_week_ordinals_expand_across_year_boundaries",
        "test_iso_week_ordinals_validate_strictly",
        "test_year_ordinals_compose_with_weekdays_or_and_modifiers",
        "test_iso_week_interval_uses_iso_year_buckets",
        "test_year_ordinals_filter_random_and_omit_candidates",
        "test_year_ordinals_positional_acf_and_cache_round_trip",
        "test_year_ordinals_documented_examples",
        "test_omit_file_name_rejects_paths",
        "test_omit_file_csv_header_parsing_is_order_independent_and_dedupes",
        "test_omit_file_csv_description_mapping_is_order_independent",
        "test_anchor_file_name_rejects_paths",
        "test_anchor_file_spec_parses_time_and_negative_offset",
        "test_anchor_file_spec_parses_bounded_time_window",
        "test_anchor_file_spec_rejects_unpadded_times",
        "test_anchor_file_composable_schedule_rejects_empty_members",
        "test_anchor_file_occurrence_provider_supports_lazy_next_after",
        "test_anchor_file_occurrence_provider_caches_expanded_specs",
        "test_anchor_file_occurrence_provider_advances_cached_lookup_cursor",
        "test_anchor_file_provider_uses_binary_search_for_nonmonotonic_cursor",
        "test_anchor_file_occurrence_provider_sorts_dst_normalized_candidates",
        "test_anchor_file_provider_retries_after_failed_load",
        "test_anchor_file_provider_keeps_description_for_overnight_slots",
        "test_anchor_file_provider_merges_duplicate_source_descriptions",
        "test_anchor_file_provider_preserves_dst_fold_descriptions",
        "test_anchor_file_provider_orders_dst_fallback_by_instant",
        "test_anchor_file_provider_rejects_incomparable_datetimes",
        "test_included_provider_bounds_anchor_file_omission_scan",
        "test_anchor_inclusion_scheduler_propagates_internal_errors",
        "test_anchor_file_omit_evaluation_failures_propagate",
        "test_recurrence_evaluator_loads_omit_file_without_text_rule",
        "test_recurrence_evaluator_loads_omit_file_dates_and_descriptions_once",
        "test_domain_scheduler_parity_across_operational_consumers",
        "test_recurrence_evaluator_shadow_parity_time_matrix",
        "test_recurrence_evaluator_shadow_parity_dst_and_business_calendar",
        "test_modify_timeline_uses_explicit_recurrence_identity",
        "test_modify_hook_uses_explicit_recurrence_identity",
        "test_add_preview_uses_explicit_recurrence_identity",
        "test_random_time_window_dst_projection_is_deterministic",
        "test_random_time_window_composition_and_anchor_file_guidance",
        "test_position_selection_parses_arbitrary_ordinals",
        "test_position_selection_rejects_invalid_tokens_and_bounds",
        "test_reconcile_expired_pending_child_is_resumable_partial",
        "test_reconcile_evidence_prefers_due_over_carried_scheduled",
        "test_reconcile_expiration_anchor_advances_from_recurrence_target",
        "test_reconcile_expiration_cp_advances_from_recurrence_target",
        "test_reconcile_expiration_candidate_requires_expiry_evidence",
        "test_reconcile_tool_computes_year_ordinal_anchor",
        "test_reconcile_parent_identity_errors_are_actionable",
        "test_reconcile_native_until_manual_review_is_not_a_hard_error",
        "test_reconcile_startup_config_failure_is_structured",
        "test_reconcile_configuration_verification_fails_closed",
        "test_shared_outbox_persists_integrity_work_without_lifecycle_claiming",
        "test_reconcile_tool_print_plan_includes_evidence",
        "test_reconcile_tool_defaults_core_path_to_install_base",
        "test_outbox_drain_limit_config_and_env_override",
        "test_queue_status_does_not_initialize_missing_outbox",
        "test_health_check_critical_outbox_bytes",
        "test_health_check_critical_outbox_rows",
        "test_reconcile_hookless_completion_verifies_scheduled_and_wait_carry",
        "test_shared_outbox_persists_integrity_work_without_lifecycle_claiming",
        "test_reconcile_startup_config_failure_is_structured",
        "test_reconcile_native_until_manual_review_is_not_a_hard_error",
        "test_reconcile_parent_identity_errors_are_actionable",
        "test_position_selection_candidate_capacity_bounds",
        "test_position_selection_candidate_capacity_bounds_are_sound",
        "test_position_selection_rejects_only_fully_impossible_candidates",
        "test_position_selection_semantic_advice",
        "test_position_selection_period_boundaries",
        "test_position_selection_internal_evaluator",
        "test_position_selection_internal_evaluator_validation",
        "test_position_selection_next_date_jumps_periods",
        "test_position_selection_candidate_cache_identity",
        "test_position_selection_public_monthly_parser_validation",
        "test_position_selection_public_monthly_scheduler",
        "test_position_selection_post_modifiers_parser_and_scheduler",
        "test_position_selection_public_period_scopes_validation",
        "test_position_selection_public_period_scopes_scheduler",
        "test_position_selection_documented_examples_and_feedback",
        "test_anchor_omit_rejects_time_modifiers",
        "test_anchor_omit_next_after_expr_skips_matching_dates",
        "test_anchor_omit_next_after_expr_skips_omit_file_dates",
        "test_anchor_omit_grouped_list_plus_expr_applies_filter_to_all_items",
        "test_anchor_omit_business_day_roll_matches_rolled_date",
        "test_anchor_omit_positive_day_offset_matches_shifted_date",
        "test_file_source_expression_flattens_groups_and_rejects_unsafe_patterns",
        "test_native_until_carry_descriptions",
        "test_native_until_validation_orders_dst_fold_by_instant",
        "test_native_until_exact_carry_orders_dst_fold_by_instant",
        "test_schedule_and_completion_use_shared_datetime_comparator",
        "test_local_datetime_full_day_gap_shifts_to_next_valid_wall_time",
        "test_public_datetime_comparator_preserves_dst_fold_and_provider_alias",
        "test_recurrence_fingerprint_is_canonical_and_mutation_sensitive",
        "test_effective_config_snapshot_isolated_and_provenanced",
        "test_hot_config_fingerprint_avoids_filesystem_stat",
        "test_hook_bootstrap_numeric_env_parsing_is_bounded",
        "test_recurrence_spec_normalizes_task_fields_and_context",
        "test_last_weekday",
        "test_panel_diagnostics_warns_for_missing_env_config",
        "test_runtime_manifest_covers_lazy_panel_colour_module",
        "test_legacy_exit_flow_modules_are_not_runtime_owned",
        "test_on_add_preview_warns_when_anchor_uses_utc_fallback",
        "test_panel_diagnostics_warns_for_empty_file_sources",
        "test_doctor_reports_uda_alias_configuration",
        "test_doctor_reports_live_panel_configuration_health",
        "test_doctor_reports_authoritative_config_schema_findings",
        "test_doctor_text_timezone_summary",
        "test_doctor_text_large_history_is_actionable_and_compact",
        "test_doctor_text_groups_historical_findings_across_chains",
        "test_doctor_reports_missing_timezone_data",
        "test_doctor_reports_missing_timezone_configuration",
        "test_doctor_reports_astronomy_preflight_health",
        "test_doctor_reports_season_backend_and_astronomical_events",
        "test_doctor_reports_matching_config_drift",
        "test_doctor_reports_missing_navigator_dependencies",
        "test_operator_context_discovers_taskdata_once",
        "test_operator_presentation_has_no_mutation_dependencies",
        "test_lifecycle_candidate_reads_support_bounded_and_full_audit_modes",
        "test_chain_integrity_warnings_detects_issues",
        "test_chain_health_advice_coach_healthy_streak",
        "test_chain_health_advice_coach_low_ontime_issue",
        "test_chain_health_advice_clinical_drift_and_style_normalization",
        "test_core_explicit_facade_all_contains_supported_symbols",
        "test_all_golden_tests_are_registered",
        "test_query_cli_emits_one_json_document_for_invalid_request",
        "test_query_emit_compact_json_preserves_unicode_and_budget",
        "test_integrity_consumers_share_report_components",
        "test_task_command_classifies_boundary_failures",
        "test_task_command_retries_only_opted_in_locks",
        "test_fixed_season_calendar_boundaries",
        "test_fixed_season_calendar_finds_active_or_next_window",
        "test_fixed_season_calendar_rejects_invalid_contract_values",
        "test_fixed_season_calendar_supports_southern_hemisphere_profile",
        "test_seasonal_selection_parser_contract",
        "test_seasonal_selection_acf_round_trip",
        "test_astronomical_season_support_boundaries_are_mode_and_hemisphere_aware",
        "test_astronomical_season_local_date_and_overflow_contract",
        "test_generic_seasonal_selection_scheduler_and_round_trip",
        "test_seasonal_selection_scheduler_post_modifiers",
        "test_seasonal_selection_boundary_and_overflow_contract",
        "test_seasonal_selection_semantic_guard",
        "test_default_business_calendar_operations_characterization",
        "test_business_calendar_config_normalizes_immutable_definitions",
        "test_business_calendar_config_rejects_ambiguous_or_unstable_rules",
        "test_business_calendar_policy_flows_through_scheduler_paths",
        "test_business_calendar_displacement_capture_is_shift_only",
        "test_business_calendar_policy_flows_through_file_modifiers",
        "test_business_calendar_config_resolves_rules_files_and_omissions",
        "test_anchor_file_occurrences_expand_bounded_time_window",
        "test_anchor_file_occurrences_expand_random_time_window_with_context",
        "test_anchor_file_occurrence_provider_exposes_typed_values",
        "test_anchor_file_occurrences_expand_overnight_time_window",
        "test_anchor_file_occurrences_expand_composable_time_schedule",
        "test_anchor_file_loader_transforms_dates_and_carries_descriptions",
        "test_anchor_file_next_occurrence_after_uses_task_level_time",
        "test_provider_contract_advertises_only_certified_capabilities",
        "test_occurrence_provider_adapters_preserve_stream_metadata",
        "test_occurrence_provider_rejects_malformed_callback_payloads",
        "test_occurrence_providers_reject_non_advancing_values",
        "test_occurrence_values_reject_inconsistent_fields",
        "test_occurrence_event_provider_requires_boolean_omitted_flag",
        "test_occurrence_collection_inclusive_cursor_steps_back_by_instant",
        "test_occurrence_collection_fails_closed_on_invalid_values_and_exhaustion",
        "test_occurrence_collection_enforces_cursor_progress_and_timezone_consistency",
        "test_anchor_occurrence_provider_exposes_typed_values_and_lazy_lookup",
        "test_anchor_file_cursor_reuse_matches_fresh_provider_reference",
        "test_anchor_file_batch_generation_matches_repeated_reference_lookups",
        "test_occurrence_provider_rejects_dst_fallback_backward_progress",
        "test_modify_completion_advances_past_second_dst_fold",
        "test_modify_overnight_window_advances_past_second_dst_fold",
        "test_time_window_parser_expands_inclusive_exact_boundary",
        "test_time_window_parser_rejects_unsafe_or_ambiguous_ranges",
        "test_time_window_slot_limit_uses_shared_resource_policy",
        "test_time_window_parser_accepts_compound_minute_intervals",
        "test_time_window_parser_accepts_hour_only_and_mixed_endpoints",
        "test_random_time_window_parser_selects_deterministic_bucketed_slots",
        "test_composable_time_schedule_deduplicates_overlaps_and_boundaries",
        "test_time_window_parser_rejects_unpadded_or_out_of_range_hour_endpoints",
        "test_time_window_partition_rounding_preserves_boundaries",
        "test_composable_time_schedule_enforces_aggregate_slot_limit",
        "test_description_alias_parser_extracts_short_udas",
        "test_description_alias_parser_avoids_prose_and_rejects_duplicates",
        "test_quarter_selector_mode_characterization",
        "test_quarter_selector_mode_rejections",
        "test_term_quarter_rewrite_mode_characterization",
        "test_quarter_spec_rewrite_characterization",
        "test_rewrite_quarters_in_context_characterization",
        "test_time_window_parser_accepts_even_partition_counts",
        "test_random_time_window_flows_through_anchor_parser_and_resolver",
        "test_hour_only_time_lists_normalize_across_anchor_and_anchor_file",
        "test_composable_time_schedule_unions_windows_and_clock_slots",
        "test_composable_time_schedule_rejects_non_numeric_members",
        "test_composable_time_schedule_rejects_empty_members",
        "test_time_window_grammar_expands_and_round_trips_through_acf",
        "test_grouped_time_window_metadata_distributes_to_each_branch",
        "test_composable_schedule_preserves_offsets_and_group_validation",
        "test_time_window_natural_language_uses_bounded_interval",
        "test_random_time_metadata_rejects_contradictory_cached_shapes",
        "test_cached_time_window_metadata_rejects_slot_drift",
        "test_cached_random_time_metadata_rejects_invalid_specs",
        "test_cached_time_schedule_metadata_rejects_slot_drift",
        "test_anchor_parse_term_explosion_guard",
        "test_parser_satisfiability_agrees_with_scheduler",
        "test_quarters_window",
        "test_quarter_alias_unambiguous_month_selectors",
        "test_astronomical_event_vocabulary_is_shared_by_parser_and_runtime",
        "test_weekly_and_unsat",
        "test_expansion_helpers_characterization",
        "test_nth_weekday_range",
        "test_lint_anchor_expr_characterization",
        "test_monthly_support_helpers_characterization",
        "test_leap_year_29feb",
        "test_anchor_expr_length_limit",
        "test_lint_formats",
        "test_anchor_grouped_list_plus_expr_applies_filter_to_all_items",
        "test_lint_grouped_list_plus_expr_matches_current_grammar",
        "test_unsat_hint_uses_yearly_alias_for_month_name_examples",
        "test_inline_time_mods_split_ok",
        "test_weekly_trailing_time_modifier_applies_to_whole_list",
        "test_same_day_next_weekday_roll_moves_forward_one_week",
        "test_same_day_prev_weekday_roll_moves_back_one_week",
        "test_next_weekday_roll_cross_year_date_still_matches_expression",
        "test_weekly_multi_days_every_2weeks_spacing_and_days",
        "test_monthly_valid_months_m2_5th_mon",
        "test_monthly_valid_months_m2_5th_mon_upcoming_within_valid_months",
        "test_leap_year_29feb_upcoming_only_on_leap_year",
        "test_rand_with_year_window_filtering",
        "test_weekly_rand_N_gate_spacing",
        "test_deterministic_randomness",
        "test_business_day_modifiers",
        "test_weekly_rand_is_chain_scoped_and_deterministic",
        "test_monthly_rand_year_intersection_is_chain_scoped",
        "test_random_weekday_list_is_one_grouped_draw",
        "test_group_time_modifier_distributes_to_all_branches",
        "test_group_time_modifier_supports_multiple_times",
        "test_group_astronomical_time_offset_distributes_to_all_branches",
        "test_group_date_modifiers_distribute_across_or_branches",
        "test_group_modifiers_reject_ambiguous_combinations",
        "test_counted_random_selects_unique_dates_per_period",
        "test_counted_random_is_deterministic_chain_scoped_and_constrained",
        "test_counted_random_omit_redraws_from_remaining_pool",
        "test_counted_random_cadence_time_and_canonical_round_trip",
        "test_edge_cases",
        "test_anchor_date_calculations",
        "test_heads_with_slashN_parse_ok",
        "test_random_anchor_cross_chain_matrix",
        "test_validate_year_tokens_in_dnf_characterization",
        "test_parse_y_token_characterization",
        "test_complex_dnf_expressions",
        "test_interval_patterns",
        "test_yearly_month_aliases_and_ranges",
        "test_business_day_bd_skip_semantics",
        "test_guard_commas_between_atoms_after_mods_fatal",
        "test_heads_with_slashN_parse_ok_again",
        "test_monthname_and_numeric_equivalence",
        "test_yearly_month_names",
        "test_rand_with_year_window",
        "test_weekly_rand_N_gate",
        "test_monthly_and_yearly_random_intervals_scale_and_exhaust",
        "test_weekly_random_intervals_scale_and_exhaust",
        "test_time_splitting_per_atom",
        "test_weekly_multi_days_and_every_2weeks",
        "test_symbolic_anchor_time_modifiers_accept_supported_events",
        "test_rand_bucket_signature_characterization",
        "test_pick_hhmm_from_dnf_for_positive_day_offset_shifted_date",
        "test_atom_matches_on_positive_day_offset_shifted_date",
        "test_parse_anchor_expr_fuzz_inputs",
        "test_anchor_parse_validate_fuzz_no_unexpected_exceptions",
        "test_anchor_validate_roundtrip_preserves_next_occurrence",
        "test_anchor_parse_deep_nesting_guard",
        "test_anchor_validate_rejects_legacy_tuple_error_payload",
        "test_rand_determinism_with_seed",
        "test_next_after_expr_branch_characterization",
        "test_anchor_normalization_is_semantically_idempotent",
        "test_anchor_expression_characterization_matrix",
        "test_weeks_between_iso_boundary",
        "test_short_uuid_invalid_inputs",
        "test_coerce_int_bounds",
        "test_next_after_atom_with_mods_characterization",
        "test_satisfiability_helpers_characterization",
        "test_month_alias_in_monthly_anchor_suggests_yearly_anchor",
        "test_warn_once_per_day_stamp_written",
        "test_warn_once_per_day_no_diag_silent",
        "test_warn_once_per_day_any_no_diag_silent",
        "test_cache_consistency",
        "test_chain_max_parser_requires_positive_integer",
        "test_core_resolve_task_data_context_rejects_parent_traversal_segments",
        "test_core_config_paths_rejects_parent_traversal_in_env",
        "test_core_config_paths_trust_override_allows_parent_traversal_in_env",
        "test_core_resolve_task_data_context_precedence",
        "test_core_resolve_task_data_context_rejects_unsafe_world_writable_dir",
        "test_core_resolve_task_data_context_trust_override_allows_explicit_dir",
        "test_occurrence_cursor_makes_lookup_semantics_explicit",
        "test_occurrence_cursor_keeps_adjacent_weekday_occurrences",
        "test_typed_occurrence_outcomes_preserve_found_invalid_and_absent_states",
        "test_typed_occurrence_outcomes_preserve_terminal_evidence",
        "test_typed_occurrence_outcomes_fail_closed_for_mutation",
        "test_typed_occurrence_outcomes_define_compact_presentation_summary",
        "test_evaluation_session_is_task_scoped_and_fingerprint_bound",
        "test_scheduler_service_is_one_typed_occurrence_entry_point",
        "test_scheduler_trace_is_disabled_bounded_and_redacted",
        "test_scheduler_cross_path_preserves_terminal_evidence",
        "test_occurrence_range_request_validates_context_bounds_and_policy",
        "test_occurrence_range_request_exposes_omission_provenance",
        "test_occurrence_range_request_wraps_unavailable_and_invalid_failures",
        "test_scheduler_conformance_isolated_under_shuffled_session_order",
        "test_hint_builder_does_not_convert_typed_failure_to_empty_hints",
        "test_recurrence_evaluator_events_between_preserves_terminal_evidence",
        "test_scheduler_generated_recurrence_matrix_is_monotonic_and_deterministic",
        "test_scheduler_parity_harness_compares_legacy_callback_only_in_tests",
        "test_scheduler_parity_matrix_covers_context_sensitive_rules",
        "test_taskwarrior_uow_observes_budget_without_blocking_commands",
        "test_taskwarrior_client_retries_only_transient_failures",
        "test_cache_location_selection_covers_install_layouts",
        "test_ui_live_panel_has_nautical_branding_without_changing_static_panels",
        "test_on_modify_completion_finalize_skips_analytics_when_hidden",
        "test_lifecycle_outbox_session_reuses_connection_and_closes_at_boundary",
        "test_reconcile_export_diagnostics_include_elapsed_time",
        "test_modify_lifecycle_routes_and_promotes_new_nautical_tasks",
        "test_chain_generation_hook_adapter_does_not_capture_modify_helpers",
        "test_perf_budget_config_covers_cache_io_checks",
        "test_included_provider_preserves_anchor_file_source_description",
        "test_exit_probe_is_conservative_across_queue_states",
        "test_reconcile_plan_uses_task_business_calendar_context",
        "test_chain_generation_rejects_missing_chain_id",
        "test_hook_engine_reports_pending_nautical_delete_without_spawning",
        "test_navigator_narrow_terminal_uses_vertical_mode_without_rich_probe",
        "test_navigator_shared_graph_scales_to_large_chain",
        "test_native_until_shared_policy_covers_recurrence_kinds_and_conflicts",
    }
)


class GoldenRegistryIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = importlib.import_module(GOLDEN_MODULE)

    def test_registered_functions_are_unique_and_callable(self):
        registered = [*self.golden.TESTS, *self.golden.DEEP_TESTS]
        names = [fn.__name__ for fn in registered]
        self.assertEqual(len(names), len(set(names)), "golden registry contains duplicates")
        self.assertTrue(all(callable(fn) for fn in registered))
        self.assertTrue(all(name.startswith("test_") for name in names))

    def test_registered_order_matches_reviewed_inventory(self):
        names = [fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)]
        digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
        self.assertEqual(digest, EXPECTED_GOLDEN_REGISTRY_ORDER_SHA256)

    def test_live_top_level_tests_are_registered_or_explicitly_retired(self):
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertEqual(
            top_level - registered,
            set(RETIRED_CHARACTERIZATION_TESTS),
            "unregistered golden tests must be migrated or explicitly retired",
        )
        self.assertFalse(RETIRED_CHARACTERIZATION_TESTS & registered)

    def test_retired_characterization_bodies_are_removed(self):
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertFalse(RETIRED_CHARACTERIZATION_TESTS & top_level)

    def test_runner_has_no_orphaned_legacy_test_support(self):
        source = Path(self.golden.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        definitions = {
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        legacy_helpers = {
            "_BoundCompletionEffects",
            "_BoundTransitionEffects",
            "_BoundPresentationEffects",
            "_BoundDiagnosticsEffects",
            "_test_modify_engine_services",
            "_legacy_test_on_modify_staged_plan_carries_parent_guard_and_stable_intent_id",
        }
        self.assertFalse(legacy_helpers & definitions)

    def test_runner_does_not_define_golden_case_bodies(self):
        source = Path(self.golden.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level_cases = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ]
        self.assertEqual(top_level_cases, [])

    def test_navigator_cases_are_owned_by_their_golden_domain(self):
        navigator = importlib.import_module("dev_tools.golden_tests.navigator")
        expected = (
            "test_navigator_surfaces_configuration_drift_warning",
            "test_navigator_reloads_validated_taskdata_configuration",
            "test_navigator_fallback_export_uses_empty_filter",
            "test_navigator_projects_all_slots_in_a_time_window",
            "test_navigator_reads_through_read_only_invocation_repository",
            "test_navigator_uses_anchor_and_anchor_file_sources",
        )
        self.assertEqual(tuple(test.__name__ for test in navigator.TESTS), expected)

    def test_golden_domains_do_not_import_the_runner_as_a_helper_library(self):
        domain_dir = Path(self.golden.__file__).parent / "golden_tests"
        offenders = []
        for path in sorted(domain_dir.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == GOLDEN_MODULE
                ):
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_migrated_direct_contracts_are_absent_from_golden_runner(self):
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertFalse(MIGRATED_DIRECT_CONTRACT_TESTS & registered)
        self.assertFalse(MIGRATED_DIRECT_CONTRACT_TESTS & top_level)
        self.assertFalse(MIGRATED_DIRECT_CONTRACT_TESTS & RETIRED_CHARACTERIZATION_TESTS)

    def test_reconcile_evidence_contract_is_owned_by_direct_suite(self):
        direct = importlib.import_module("tests.test_reconcile_error_contracts")
        self.assertTrue(
            callable(
                getattr(
                    direct.ReconcileErrorContracts,
                    "test_reconcile_evidence_prefers_due_over_carried_scheduled",
                    None,
                )
            )
        )
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        self.assertNotIn(
            "test_reconcile_evidence_prefers_due_over_carried_scheduled",
            registered,
        )

    def test_year_ordinal_anchor_contract_is_owned_by_generation_suite(self):
        direct = importlib.import_module("tests.test_chain_generation_recovery_contract")
        self.assertTrue(
            callable(
                getattr(
                    direct.ChainGenerationContractTests,
                    "test_year_ordinal_anchor_reconcile_calculation",
                    None,
                )
            )
        )
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        self.assertNotIn("test_reconcile_tool_computes_year_ordinal_anchor", registered)

    def test_parent_identity_diagnostics_are_owned_by_reconcile_suite(self):
        direct = importlib.import_module("tests.test_reconcile_error_contracts")
        self.assertTrue(
            callable(
                getattr(
                    direct.ReconcileErrorContracts,
                    "test_parent_identity_errors_are_actionable",
                    None,
                )
            )
        )
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        self.assertNotIn("test_reconcile_parent_identity_errors_are_actionable", registered)

    def test_apply_lease_rejection_is_owned_by_reconcile_error_suite(self):
        direct = importlib.import_module("tests.test_reconcile_error_contracts")
        self.assertTrue(
            callable(
                getattr(
                    direct.ReconcileErrorContracts,
                    "test_apply_lease_conflict_returns_before_session_build",
                    None,
                )
            )
        )
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        self.assertNotIn("test_reconcile_apply_refuses_a_second_full_run", registered)
        self.assertTrue(
            callable(
                getattr(
                    direct.ReconcileErrorContracts,
                    "test_apply_lease_is_exclusive_and_released",
                    None,
                )
            )
        )
        self.assertNotIn("test_reconcile_apply_lease_serializes_mutations", registered)

    def test_reconcile_startup_output_is_owned_by_operator_suite(self):
        direct = importlib.import_module("tests.test_operator_command_contracts")
        self.assertTrue(
            callable(
                getattr(
                    direct.OperatorCommandContractTests,
                    "test_reconcile_startup_failures_keep_mode_specific_output_streams",
                    None,
                )
            )
        )
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        self.assertNotIn("test_reconcile_subprocess_output_contracts", registered)

    def test_native_until_carry_is_owned_by_reconcile_error_suite(self):
        direct = importlib.import_module("tests.test_reconcile_error_contracts")
        self.assertTrue(
            callable(
                getattr(
                    direct.ReconcileErrorContracts,
                    "test_native_until_carry_fallback_and_verification_contract",
                    None,
                )
            )
        )
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        self.assertNotIn(
            "test_reconcile_repairs_invalid_native_until_from_previous_link",
            registered,
        )

    def test_native_until_manual_review_is_owned_by_reconcile_suite(self):
        direct = importlib.import_module("tests.test_reconcile_error_contracts")
        self.assertTrue(
            callable(
                getattr(
                    direct.ReconcileErrorContracts,
                    "test_native_until_manual_review_is_not_a_hard_error",
                    None,
                )
            )
        )
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        self.assertNotIn(
            "test_reconcile_native_until_manual_review_is_not_a_hard_error",
            registered,
        )

    def test_startup_configuration_failure_is_owned_by_reconcile_suite(self):
        direct = importlib.import_module("tests.test_reconcile_error_contracts")
        self.assertTrue(
            callable(
                getattr(
                    direct.ReconcileErrorContracts,
                    "test_startup_config_failure_is_structured",
                    None,
                )
            )
        )
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        self.assertNotIn("test_reconcile_startup_config_failure_is_structured", registered)

    def test_configuration_verification_is_owned_by_reconcile_suite(self):
        direct = importlib.import_module("tests.test_reconcile_error_contracts")
        self.assertTrue(
            callable(
                getattr(
                    direct.ReconcileErrorContracts,
                    "test_configuration_verification_fails_closed_on_unexpected_fault",
                    None,
                )
            )
        )
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        self.assertNotIn(
            "test_reconcile_configuration_verification_fails_closed", registered
        )

    def test_shared_integrity_outbox_contract_is_owned_by_outbox_suite(self):
        direct = importlib.import_module("tests.test_lifecycle_outbox_contract")
        self.assertTrue(
            callable(
                getattr(
                    direct.LifecycleOutboxContractTests,
                    "test_integrity_work_shares_storage_without_lifecycle_claiming",
                    None,
                )
            )
        )
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        self.assertNotIn(
            "test_shared_outbox_persists_integrity_work_without_lifecycle_claiming",
            registered,
        )

    def test_reconcile_plan_output_contract_is_owned_by_error_suite(self):
        direct = importlib.import_module("tests.test_reconcile_error_contracts")
        self.assertTrue(
            callable(
                getattr(
                    direct.ReconcileErrorContracts,
                    "test_reconcile_plan_output_includes_safety_evidence",
                    None,
                )
            )
        )
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        self.assertNotIn("test_reconcile_tool_print_plan_includes_evidence", registered)

    def test_reconcile_tool_install_base_contract_is_owned_by_deployment_suite(self):
        direct = importlib.import_module("tests.test_deployment_reliability_contract")
        self.assertTrue(
            callable(
                getattr(
                    direct.DeploymentSanityContractTests,
                    "test_reconcile_tool_defaults_core_path_to_install_base",
                    None,
                )
            )
        )
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        self.assertNotIn(
            "test_reconcile_tool_defaults_core_path_to_install_base", registered
        )

    def test_outbox_drain_configuration_is_owned_by_runtime_config_suite(self):
        direct = importlib.import_module("tests.test_runtime_config_contracts")
        self.assertTrue(
            callable(
                getattr(
                    direct.RuntimeConfigContracts,
                    "test_outbox_drain_limit_config_and_env_override",
                    None,
                )
            )
        )
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        self.assertNotIn("test_outbox_drain_limit_config_and_env_override", registered)

    def test_read_only_queue_contract_is_owned_by_operator_process_suite(self):
        direct = importlib.import_module("tests.test_operator_process_contract")
        self.assertTrue(
            callable(
                getattr(
                    direct.OperatorProcessContractTests,
                    "test_core_queue_status_does_not_create_missing_outbox",
                    None,
                )
            )
        )
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        self.assertNotIn("test_queue_status_does_not_initialize_missing_outbox", registered)

    def test_health_check_budget_contracts_are_owned_by_operator_process_suite(self):
        direct = importlib.import_module("tests.test_operator_process_contract")
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        for name in (
            "test_health_check_critical_outbox_bytes",
            "test_health_check_critical_outbox_rows",
        ):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(direct.OperatorProcessContractTests, name, None)))
                self.assertNotIn(name, registered)

    def test_hookless_recovery_carry_contract_is_owned_by_generation_suite(self):
        direct = importlib.import_module("tests.test_chain_generation_recovery_contract")
        self.assertTrue(
            callable(
                getattr(
                    direct.IntegrityRecoveryContractTests,
                    "test_hookless_recovery_preserves_scheduled_and_wait_offsets",
                    None,
                )
            )
        )
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        self.assertNotIn(
            "test_reconcile_hookless_completion_verifies_scheduled_and_wait_carry",
            registered,
        )

    def test_unit_tests_do_not_import_the_golden_runner_directly(self):
        tests_dir = Path(__file__).resolve().parent
        offenders = []
        for path in sorted(tests_dir.glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                if "dev_tools.nautical_golden_tests" in modules:
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_navigator_anchor_source_golden_restores_shared_core_state(self):
        navigator = importlib.import_module("dev_tools.golden_tests.navigator")
        core = navigator.core
        previous = (
            core._FACADE_CONFIG_SYNCED,
            core.ANCHOR_FILE_DIR,
            core._core_config.ANCHOR_FILE_DIR,
        )
        try:
            navigator.test_navigator_uses_anchor_and_anchor_file_sources()
            self.assertEqual(
                (
                    core._FACADE_CONFIG_SYNCED,
                    core.ANCHOR_FILE_DIR,
                    core._core_config.ANCHOR_FILE_DIR,
                ),
                previous,
            )
        finally:
            core._FACADE_CONFIG_SYNCED = previous[0]
            core.ANCHOR_FILE_DIR = previous[1]
            core._core_config.ANCHOR_FILE_DIR = previous[2]

    def test_removed_ineffective_tests_stay_absent(self):
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertFalse(REMOVED_INEFFECTIVE_TESTS & registered)
        self.assertFalse(REMOVED_INEFFECTIVE_TESTS & top_level)
        self.assertEqual(len(REMOVED_INEFFECTIVE_TESTS), 7)

    def test_timeline_golden_cases_are_owned_by_timeline_domain(self):
        timeline = importlib.import_module("dev_tools.golden_tests.timeline")
        expected = (
            "test_hook_on_modify_timeline_multitime_includes_all_slots",
            "test_hook_on_modify_timeline_cp_sequence_labels_future_intervals",
            "test_hook_on_modify_timeline_cp_random_labels_selected_intervals",
            "test_hook_on_modify_timeline_marks_omitted_anchor_slots",
            "test_hook_on_modify_merged_timeline_marks_projection_failures",
            "test_hook_on_modify_timeline_uses_omit_file_description_label",
            "test_hook_on_modify_timeline_keeps_anchor_match_after_shifted_anchor_file_child",
            "test_hook_on_modify_timeline_omits_shifted_anchor_file_dates_in_merged_stream",
            "test_hook_on_modify_timeline_shows_anchor_side_omit_file_dates_in_merged_stream",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }

        self.assertEqual(tuple(test.__name__ for test in timeline.TESTS), expected)
        self.assertTrue(set(expected) <= registered)
        self.assertFalse(set(expected) & top_level)

    def test_operator_diagnostics_cases_are_owned_by_operator_domain(self):
        operator = importlib.import_module("dev_tools.golden_tests.operator")
        expected = (
            "test_queue_status_and_doctor_report_schema_health",
            "test_doctor_installation_json_and_verifier_contract",
            "test_queue_status_warns_on_stale_processing_and_dead_letters",
            "test_doctor_reports_healthy_installation",
            "test_doctor_hook_inventory_allows_third_party_and_symlink_install",
            "test_doctor_hook_inventory_rejects_duplicates_without_counting_backups",
            "test_doctor_hook_inventory_reports_incomplete_core_and_api_mismatch",
            "test_doctor_reports_retired_queue_state_without_migrating_it",
            "test_doctor_discovers_effective_taskdata_directory",
            "test_doctor_reports_actionable_broken_installation",
            "test_doctor_reports_chain_repair_plan_findings",
            "test_operator_doctor_loads_colocated_queue_helper",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }

        self.assertTrue(set(expected) <= registered)
        self.assertTrue(set(expected) <= {test.__name__ for test in operator.TESTS})
        self.assertFalse(set(expected) & top_level)

    def test_configuration_cases_are_owned_by_configuration_domain(self):
        configuration = importlib.import_module("dev_tools.golden_tests.configuration")
        expected = (
            "test_hook_on_modify_uda_aliases_route_through_thin_wrapper",
            "test_hook_on_modify_uda_alias_anchor_change_emits_ack_panel",
            "test_hook_on_modify_empty_uda_alias_clears_through_thin_wrapper",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertTrue(set(expected) <= registered)
        self.assertEqual(tuple(test.__name__ for test in configuration.TESTS), expected)
        self.assertFalse(set(expected) & top_level)

    def test_deployment_cases_are_owned_by_performance_domain(self):
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertNotIn("test_ops_templates_present_and_runner_executable", registered)
        deployment_tests = importlib.import_module("tests.test_deployment_reliability_contract")
        self.assertTrue(
            callable(
                getattr(
                    deployment_tests.DeploymentSanityContractTests,
                    "test_ops_templates_present_and_health_runner_executable",
                    None,
                )
            )
        )
        self.assertFalse(
            {
                "test_perf_cold_import_records_module_profile",
                "test_perf_hint_benchmark_isolates_persistent_cache",
                "test_perf_hook_fast_path_ratio_enforcement",
                "test_ops_templates_present_and_runner_executable",
            }
            & top_level
        )

    def test_runtime_performance_cases_are_owned_by_performance_domain(self):
        direct = importlib.import_module("tests.test_performance_harness_contracts")
        expected = (
            "test_hook_replay_harness_reports_ok",
            "test_mixed_recurrence_loop_harness_reports_ok",
            "test_soak_runner_reports_ok",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        direct_tests = tuple(
            name
            for name, value in vars(direct.PerformanceHarnessContractTests).items()
            if name.startswith("test_") and callable(value)
        )
        self.assertTrue(set(expected) <= set(direct_tests))
        self.assertFalse(set(expected) & registered)

    def test_modify_feedback_cases_are_owned_by_modify_domain(self):
        modify = importlib.import_module("dev_tools.golden_tests.modify")
        expected = (
            "test_on_modify_reports_business_calendar_displacement",
            "test_on_modify_promotes_chain_emits_upgrade_panel",
            "test_on_modify_promotes_cp_emits_period_explanation",
            "test_on_modify_disables_chain_emits_disabled_panel",
            "test_on_modify_resumes_chain_emits_resumed_panel",
            "test_on_modify_resume_wrapper_preserves_json_and_emits_panel",
            "test_on_modify_recurrence_update_emits_ack_panel",
            "test_on_modify_recurrence_update_groups_and_flattens_changes",
            "test_on_modify_native_until_update_explains_carry",
            "test_on_modify_limit_update_emits_effective_boundaries",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertTrue(set(expected) <= registered)
        self.assertEqual(tuple(test.__name__ for test in modify.TESTS[:10]), expected)
        self.assertFalse(set(expected) & top_level)

    def test_native_until_temporal_cases_are_owned_by_modify_domain(self):
        modify = importlib.import_module("dev_tools.golden_tests.modify")
        expected = (
            "test_on_modify_carry_wall_clock_across_dst",
            "test_on_modify_build_child_carries_until_across_dst",
            "test_on_modify_native_until_calendar_and_exact_carry_policy",
            "test_on_modify_native_until_exact_carry_preserves_elapsed_time_across_dst",
            "test_native_until_calendar_slot_guard_rejects_impossible_anchor_expirations",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertTrue(set(expected) <= registered)
        self.assertEqual(tuple(test.__name__ for test in modify.TESTS[10:15]), expected)
        self.assertFalse(set(expected) & top_level)

    def test_completion_planning_cases_are_owned_by_modify_domain(self):
        modify = importlib.import_module("dev_tools.golden_tests.modify")
        expected = (
            "test_on_modify_completion_preflight_context_happy_path",
            "test_on_modify_completion_compute_next_and_limits_happy_path",
            "test_cap_from_until_cp_includes_exact_deadline",
            "test_hook_on_modify_rejects_invalid_chain_max_for_cp_and_anchor",
            "test_on_modify_validates_chain_until_only_when_recurrence_or_caps_change",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertTrue(set(expected) <= registered)
        self.assertEqual(tuple(test.__name__ for test in modify.TESTS[15:20]), expected)
        self.assertFalse(set(expected) & top_level)

    def test_native_until_hook_cases_are_owned_by_modify_domain(self):
        modify = importlib.import_module("dev_tools.golden_tests.modify")
        expected = (
            "test_on_modify_native_until_rejects_invalid_window_changes",
            "test_on_modify_native_until_follows_recurrence_target_move",
            "test_on_modify_native_until_rejects_uncarryable_anchor_target_move",
            "test_on_modify_completion_reschedule_carries_native_until",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertTrue(set(expected) <= registered)
        self.assertEqual(
            tuple(test.__name__ for test in modify.TESTS[20:24]),
            expected,
        )
        self.assertFalse(set(expected) & top_level)

    def test_native_until_validation_cases_are_owned_by_modify_domain(self):
        modify = importlib.import_module("dev_tools.golden_tests.modify")
        expected = (
            "test_on_modify_native_until_accepts_valid_window_change",
            "test_on_modify_native_until_validates_recurrence_promotion",
            "test_on_modify_native_until_validates_simultaneous_completion",
            "test_on_modify_native_until_rejects_strict_anchor_mode_changes",
            "test_on_modify_native_until_rejects_legacy_all_completion",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertTrue(set(expected) <= registered)
        self.assertEqual(tuple(test.__name__ for test in modify.TESTS[24:29]), expected)
        self.assertFalse(set(expected) & top_level)

    def test_modify_timing_and_child_cases_are_owned_by_modify_domain(self):
        modify = importlib.import_module("dev_tools.golden_tests.modify")
        expected = (
            "test_on_modify_build_child_transitions_flex_to_all",
            "test_on_modify_cp_due_edit_preserves_relative_offsets",
            "test_on_modify_explicit_timing_edits_warn_on_invalid_order",
            "test_on_modify_timing_warning_wrapper_preserves_json_stdout",
            "test_on_modify_build_child_carries_configured_uda_datetime",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertTrue(set(expected) <= registered)
        self.assertEqual(tuple(test.__name__ for test in modify.TESTS[29:34]), expected)
        self.assertFalse(set(expected) & top_level)

    def test_modify_feedback_and_identity_cases_are_owned_by_modify_domain(self):
        modify = importlib.import_module("dev_tools.golden_tests.modify")
        expected = (
            "test_on_modify_anchor_feedback_warns_when_timed_anchor_uses_utc_fallback",
            "test_on_modify_promotes_chain_when_task_becomes_nautical",
            "test_on_modify_link_limit",
            "test_on_modify_stable_child_uuid_is_slot_deterministic",
            "test_on_modify_expands_and_clears_description_uda_aliases",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertTrue(set(expected) <= registered)
        self.assertEqual(tuple(test.__name__ for test in modify.TESTS[34:39]), expected)
        self.assertFalse(set(expected) & top_level)

    def test_hook_panel_cases_are_owned_by_modify_domain(self):
        modify = importlib.import_module("dev_tools.golden_tests.modify")
        expected = (
            "test_on_modify_panel_fallback",
            "test_on_modify_panel_forwards_live_duration",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertTrue(set(expected) <= registered)
        self.assertEqual(tuple(test.__name__ for test in modify.TESTS[39:41]), expected)
        self.assertFalse(set(expected) & top_level)

    def test_malformed_cp_guidance_case_is_owned_by_modify_domain(self):
        modify = importlib.import_module("dev_tools.golden_tests.modify")
        expected = ("test_hook_on_modify_cp_malformed_inputs_fail_with_parser_guidance",)
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertEqual(tuple(test.__name__ for test in modify.TESTS[41:42]), expected)
        self.assertTrue(set(expected) <= registered)
        self.assertFalse(set(expected) & top_level)

    def test_on_add_context_contract_is_owned_by_direct_unittest(self):
        support = importlib.import_module("dev_tools.golden_tests.support")
        self.assertFalse(hasattr(support, "assert_hook_requires_integration_context"))
        hook_context_tests = importlib.import_module("tests.test_hook_context_requirements")
        self.assertTrue(
            callable(
                getattr(
                    hook_context_tests.HookContextRequirementTests,
                    "test_on_add_requires_integration_context_helper",
                    None,
                )
            )
        )
        source = Path(self.golden.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        local_definitions = {
            node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertNotIn("test_on_add_requires_integration_context_helper", local_definitions)

    def test_completion_lifecycle_export_reuse_is_owned_by_lifecycle_domain(self):
        lifecycle = importlib.import_module("dev_tools.golden_tests.lifecycle")
        expected = ("test_on_modify_lifecycle_export_reuses_completion_chain_snapshot",)
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertTrue(set(expected) <= registered)
        self.assertEqual(tuple(test.__name__ for test in lifecycle.TESTS[30:31]), expected)
        self.assertFalse(set(expected) & top_level)

    def test_load_benchmarks_are_owned_by_performance_domain(self):
        direct = importlib.import_module("tests.test_load_test_nautical")
        expected = (
            "test_load_benchmark_installs_complete_hook_runtime",
            "test_load_benchmark_queue_and_lineage_verification",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        direct_tests = tuple(
            name
            for name, value in vars(direct.LoadTestSupportTests).items()
            if name.startswith("test_") and callable(value)
        )
        self.assertTrue(set(expected) <= set(direct_tests))
        self.assertFalse(set(expected) & registered)

    def test_completion_snapshot_cases_are_owned_by_lifecycle_domain(self):
        lifecycle = importlib.import_module("dev_tools.golden_tests.lifecycle")
        expected = (
            "test_on_modify_completion_chain_snapshot_modes_and_query",
            "test_on_modify_recompleted_task_with_nextlink_skips_spawn",
            "test_on_modify_recompleted_task_with_existing_link_skips_spawn",
            "test_on_modify_completion_reuses_single_chain_export_when_chain_needed",
            "test_on_modify_completion_snapshot_reuses_full_chain_read",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertTrue(set(expected) <= registered)
        self.assertEqual(tuple(test.__name__ for test in lifecycle.TESTS[25:30]), expected)
        self.assertFalse(set(expected) & top_level)

    def test_outbox_and_mutation_cases_are_owned_by_lifecycle_domain(self):
        lifecycle = importlib.import_module("dev_tools.golden_tests.lifecycle")
        expected = (
            "test_taskwarrior_mutation_service_is_guarded_idempotent_and_fail_closed",
            "test_lifecycle_outbox_persists_typed_plans_and_recovers_claims",
            "test_lifecycle_outbox_initialization_is_concurrent_and_rejects_unknown_schema",
            "test_queue_claim_quarantines_poison_rows_and_queue_status_reports_them",
            "test_on_modify_spawn_intent_queue_failure_is_reported",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertTrue(set(expected) <= registered)
        self.assertEqual(tuple(test.__name__ for test in lifecycle.TESTS[15:20]), expected)
        self.assertFalse(set(expected) & top_level)

    def test_completion_spawn_cases_are_owned_by_lifecycle_domain(self):
        lifecycle = importlib.import_module("dev_tools.golden_tests.lifecycle")
        expected = (
            "test_on_modify_completion_build_and_spawn_child_happy_path",
            "test_on_modify_completion_spawn_exception_is_retryable_with_reason",
            "test_on_modify_build_child_scheduled_only_keeps_due_unset_and_carries_wait",
            "test_on_modify_cp_completion_spawns_next_link",
            "test_on_modify_completion_helper_returns_finalized_lifecycle_result",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertTrue(set(expected) <= registered)
        self.assertEqual(tuple(test.__name__ for test in lifecycle.TESTS[20:25]), expected)
        self.assertFalse(set(expected) & top_level)

    def test_temporal_cases_are_owned_by_scheduling_domain(self):
        scheduling = importlib.import_module("dev_tools.golden_tests.scheduling")
        expected = (
            "test_year_ordinals_hooks_modes_calendar_and_timeline",
            "test_local_datetime_non_hour_dst_gap_is_shared_by_modify",
            "test_anchor_preview_explains_nonexistent_wall_time_adjustment",
            "test_random_anchor_and_omit_presets_keep_chain_scope",
            "test_on_modify_reuses_task_scoped_evaluator_and_scheduler_binding",
            "test_random_time_window_is_stable_across_processes",
            "test_astronomical_season_selection_scheduler_uses_transition_dates",
            "test_seasonal_selection_modify_modes_times_and_timeline",
            "test_on_modify_compute_anchor_child_due_from_anchor_file",
            "test_on_modify_compute_anchor_child_due_from_random_anchor_file",
            "test_on_modify_compute_anchor_child_due_from_multiple_file_times",
            "test_on_modify_compute_anchor_child_due_from_combined_anchor_sources",
            "test_on_modify_compute_combined_overnight_sources_in_time_order",
            "test_modifier_boundary_paths_agree_and_advance_strictly",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertTrue(set(expected) <= registered)
        self.assertEqual(tuple(test.__name__ for test in scheduling.TESTS), expected)
        self.assertFalse(set(expected) & top_level)

    def test_installer_runtime_cases_are_owned_by_installer_domain(self):
        installer = importlib.import_module("dev_tools.golden_tests.installer")
        expected = (
            "test_installer_dry_run_fresh_install_and_idempotent_reinstall",
            "test_installer_navigator_dependency_failure_is_actionable",
            "test_installer_upgrade_rollback_restores_active_runtime",
            "test_installer_migrates_legacy_core_and_rolls_back_first_switch",
            "test_installer_lock_and_duplicate_hook_guards",
            "test_installer_cli_and_doctor_managed_runtime_diagnostics",
            "test_runtime_cleanup_preserves_active_and_rollback_releases",
            "test_retained_release_can_be_selected_with_dry_run_then_applied",
        )
        registered = {test.__name__ for test in (*self.golden.TESTS, *self.golden.DEEP_TESTS)}
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }

        self.assertTrue(set(expected) <= registered)
        self.assertTrue(set(expected) <= {test.__name__ for test in installer.TESTS})
        self.assertFalse(set(expected) & top_level)

    def test_registry_inventory_counts_match_documented_snapshot(self):
        registered = [*self.golden.TESTS, *self.golden.DEEP_TESTS]
        top_level = [
            value
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        ]
        operator = importlib.import_module("dev_tools.golden_tests.operator")
        configuration = importlib.import_module("dev_tools.golden_tests.configuration")
        installer = importlib.import_module("dev_tools.golden_tests.installer")
        lifecycle = importlib.import_module("dev_tools.golden_tests.lifecycle")
        reconcile = importlib.import_module("dev_tools.golden_tests.reconcile")
        storage = importlib.import_module("dev_tools.golden_tests.storage")
        timeline = importlib.import_module("dev_tools.golden_tests.timeline")
        scheduling = importlib.import_module("dev_tools.golden_tests.scheduling")
        self.assertEqual(len(top_level), 0)
        self.assertEqual(len(registered), 138)
        self.assertEqual(len(operator.TESTS), 14)
        self.assertEqual(len(configuration.TESTS), 3)
        self.assertEqual(len(installer.TESTS), 9)
        modify = importlib.import_module("dev_tools.golden_tests.modify")
        self.assertEqual(len(modify.TESTS), 42)
        self.assertEqual(len(lifecycle.TESTS), 31)
        self.assertEqual(len(reconcile.TESTS), 9)
        self.assertEqual(len(storage.TESTS), 1)
        self.assertEqual(len(timeline.TESTS), 9)
        self.assertEqual(len(scheduling.TESTS), 14)
        self.assertEqual(len(RETIRED_CHARACTERIZATION_TESTS), 0)
        self.assertEqual(len(MIGRATED_DIRECT_CONTRACT_TESTS), 726)

    def test_cross_process_lock_golden_is_owned_by_storage_domain(self):
        storage = importlib.import_module("dev_tools.golden_tests.storage")
        self.assertEqual(
            tuple(test.__name__ for test in storage.TESTS),
            ("test_safe_lock_fcntl_contention",),
        )

    def test_retained_cases_have_a_stable_exclusive_acceptance_inventory(self):
        registered = [fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)]
        domains = {name: [] for name in EXPECTED_GOLDEN_ACCEPTANCE_DOMAINS}
        for name in registered:
            lowered = name.lower()
            for domain, markers in GOLDEN_ACCEPTANCE_DOMAIN_MARKERS:
                if any(marker in lowered for marker in markers):
                    domains[domain].append(name)
                    break
            else:
                domains["recurrence and hook integration"].append(name)

        self.assertEqual(
            set(domains), set(EXPECTED_GOLDEN_ACCEPTANCE_DOMAINS),
            "acceptance domain definitions and expected inventory diverged",
        )
        actual = {
            domain: (
                len(names),
                hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest(),
            )
            for domain, names in domains.items()
        }
        self.assertEqual(actual, EXPECTED_GOLDEN_ACCEPTANCE_DOMAINS)


if __name__ == "__main__":
    unittest.main()
