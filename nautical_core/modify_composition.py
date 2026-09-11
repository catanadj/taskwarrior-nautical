"""Composition services for the on-modify hook."""

from __future__ import annotations

import re
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, Callable
from .task_datetime import parser_for_core


class _HookHost:
    """Attribute view over a dynamically loaded hook module's globals."""

    def __init__(self, values: dict[str, Any], name: str) -> None:
        self._values = values
        self.__name__ = name

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class ModifyHookCapabilities:
    """Modules and facilities resolved once at the hook composition root.

    Effects receive this explicit capability set instead of repeatedly reaching
    through the dynamic host module loader.  The loader remains the boundary
    that constructs the set, preserving import-by-file compatibility.
    """

    modify_ordinary: Any
    modify_effects: Any
    hook_results: Any
    hook_context: Any
    hook_engine: Any
    modify_lifecycle: Any
    modify_transition_effects: Any
    modify_presentation_effects: Any
    modify_diagnostics_effects: Any
    modify_validation_effects: Any
    modify_ui_effects: Any
    modify_task_fields: Any
    modify_completion_effects: Any
    modify_read_effects: Any
    modify_queries: Any
    modify_expiration: Any
    modify_generation_effects: Any
    task_codec: Any
    task_models: Any
    chain_integrity_lifecycle: Any
    modify_datetime_effects: Any
    modify_spawn_effects: Any

    @classmethod
    def from_host(cls, host: Any) -> "ModifyHookCapabilities":
        load = host._module
        return cls(
            modify_ordinary=load("modify_ordinary"),
            modify_effects=load("modify_effects"),
            hook_results=load("hook_results"),
            hook_context=load("hook_context"),
            hook_engine=load("hook_engine"),
            modify_lifecycle=load("modify_lifecycle"),
            modify_transition_effects=load("modify_transition_effects"),
            modify_presentation_effects=load("modify_presentation_effects"),
            modify_diagnostics_effects=load("modify_diagnostics_effects"),
            modify_validation_effects=load("modify_validation_effects"),
            modify_ui_effects=load("modify_ui_effects"),
            modify_task_fields=load("modify_task_fields"),
            modify_completion_effects=load("modify_completion_effects"),
            modify_read_effects=load("modify_read_effects"),
            modify_queries=load("modify_queries"),
            modify_expiration=load("modify_expiration", required=False),
            modify_generation_effects=load("modify_generation_effects"),
            task_codec=load("task_codec"),
            task_models=load("task_models"),
            chain_integrity_lifecycle=load("chain_integrity_lifecycle"),
            modify_datetime_effects=load("modify_datetime_effects"),
            modify_spawn_effects=load("modify_spawn_effects"),
        )


@dataclass(frozen=True)
class ModifyRuntimeServices:
    """Explicit runtime services supplied to route effects at composition time."""

    capabilities: ModifyHookCapabilities
    runtime_state: Callable[..., Any]
    import_module: Callable[..., Any]
    diag_summary: Callable[..., Any]
    diagnostic: Callable[..., Any]
    show_analytics: bool
    check_integrity: bool
    analytics_style: str
    seed_runtime_lookup_tasks: Callable[..., Any]
    lifecycle_read_service: Callable[[], Any]
    chain_health_advice: Callable[..., Any]
    chain_integrity_warnings: Callable[..., Any]
    render_anchor_completion_feedback: Callable[..., Any]
    render_cp_completion_feedback: Callable[..., Any]
    render_lifecycle_result: Callable[..., Any]
    print_task: Callable[..., Any]
    prepare_recurrence: Callable[..., Any]
    preserve_cp_relative_offsets: Callable[..., Any]
    preserve_native_until: Callable[..., Any]
    validate_native_until: Callable[..., Any]
    validate_native_until_slots: Callable[..., Any]
    now_utc: Callable[[], Any]
    compute_next_and_limits: Callable[..., Any]

    @classmethod
    def from_host(cls, host: Any, capabilities: ModifyHookCapabilities | None = None):
        capabilities = capabilities or capabilities_for(host)
        return cls(
            capabilities=capabilities,
            runtime_state=host._modify_runtime_state,
            import_module=host.importlib.import_module,
            diag_summary=host._diag_summary,
            diagnostic=host._diag,
            show_analytics=host._SHOW_ANALYTICS,
            check_integrity=host._CHECK_CHAIN_INTEGRITY,
            analytics_style=host._ANALYTICS_STYLE,
            seed_runtime_lookup_tasks=lambda *tasks: capabilities.modify_read_effects.seed_runtime_lookup_tasks(
                capabilities.modify_read_effects.SeedLookupPorts(
                    service=capabilities.modify_read_effects.lifecycle_read_service(host),
                    decode_row=capabilities.task_codec.DEFAULT_TASK_CODEC.decode_row,
                    cache_set=host._query_ctx_set,
                ), *tasks
            ),
            lifecycle_read_service=lambda: capabilities.modify_read_effects.lifecycle_read_service(host),
            chain_health_advice=lambda *args, **kwargs: capabilities.modify_diagnostics_effects.chain_health_advice(
                capabilities.modify_diagnostics_effects.analytics_ports_for(host), *args, **kwargs
            ),
            chain_integrity_warnings=lambda *args, **kwargs: capabilities.modify_diagnostics_effects.chain_integrity_warnings(
                capabilities.modify_diagnostics_effects.analytics_ports_for(host), *args, **kwargs
            ),
            render_anchor_completion_feedback=lambda **kwargs: capabilities.modify_presentation_effects.render_anchor_completion_feedback(host, **kwargs),
            render_cp_completion_feedback=lambda **kwargs: capabilities.modify_presentation_effects.render_cp_completion_feedback(host, **kwargs),
            render_lifecycle_result=lambda result, task: capabilities.modify_presentation_effects.render_lifecycle_result(host, result, task),
            print_task=lambda task: capabilities.modify_ui_effects.print_task(host, task),
            prepare_recurrence=lambda old, new, **kwargs: capabilities.modify_transition_effects.validate_completion_cp_and_anchor(host, old, new, **kwargs),
            preserve_cp_relative_offsets=lambda old, new, cp, **kwargs: capabilities.modify_transition_effects.preserve_cp_relative_offsets_on_due_change(host, old, new, cp, **kwargs),
            preserve_native_until=lambda old, new, kind, **kwargs: capabilities.modify_transition_effects.preserve_native_until_on_target_change(host, old, new, kind, **kwargs),
            validate_native_until=lambda task: capabilities.modify_validation_effects.validate_native_until(
                capabilities.modify_validation_effects.native_until_ports_for(host), task
            ),
            validate_native_until_slots=lambda task: capabilities.modify_validation_effects.validate_native_until_slots(
                capabilities.modify_validation_effects.native_until_slot_ports_for(host), task
            ),
            now_utc=host.core.now_utc,
            compute_next_and_limits=lambda *args, **kwargs: capabilities.modify_completion_effects.compute_next_and_limits(host, *args, **kwargs),
        )


def capabilities_for(host: Any) -> ModifyHookCapabilities:
    """Return the composition-root capability set for ``host``."""
    cached = getattr(host, "_MODIFY_CAPABILITIES", None)
    if cached is None:
        cached = ModifyHookCapabilities.from_host(host)
        try:
            setattr(host, "_MODIFY_CAPABILITIES", cached)
        except Exception:
            pass
    # Construct the datetime port once at the composition root.  Effects may
    # consume it through their narrow parser adapter without rediscovering the
    # live hook/core namespace on every field.
    if getattr(host, "_TASK_DATETIME_PARSER", None) is None:
        try:
            setattr(
                host,
                "_TASK_DATETIME_PARSER",
                parser_for_core(host.core, diagnostic=getattr(host, "_diag", None)),
            )
        except Exception:
            pass
    return cached


class ModifyCompositionServices:
    """Bind on-modify effects to the hook's validated composition root."""

    def __init__(self, host: Any, result_cls: Callable[..., Any]) -> None:
        self._host = host
        self._result_cls = result_cls
        self._capabilities = capabilities_for(host)
        self._runtime = ModifyRuntimeServices.from_host(host, self._capabilities)

    def result(self, task, *, sanitize: bool):
        return self._result_cls(task=task, sanitize=sanitize)

    def has_nautical_fields(self, task):
        return self._capabilities.modify_lifecycle.task_has_nautical_fields(task)

    def load_core(self):
        self._host._load_core()

    def diag(self, message: str):
        self._host._diag(message)

    def fail_and_exit(self, title: str, message: str):
        self._host._fail_and_exit(title, message)

    def handle_non_completion(self, old, new, unit_of_work, transition=None):
        self._capabilities.modify_effects.handle_non_completion(
            self._host, old, new, unit_of_work, transition=transition,
            runtime=self._runtime,
        )

    def handle_completion(self, old, new, unit_of_work, transition=None):
        return self._capabilities.modify_effects.handle_completion(
            self._host, old, new, unit_of_work, transition=transition,
            runtime=self._runtime,
        )

    def handle_deleted(
        self, old, new, unit_of_work, transition=None, terminal_decision=None
    ):
        return self._capabilities.modify_effects.handle_deleted(
            self._host,
            old,
            new,
            unit_of_work,
            transition=transition,
            terminal_decision=terminal_decision,
            runtime=self._runtime,
        )


def hook_host(values: dict[str, Any], name: str) -> Any:
    """Return a live attribute view for import-by-file test harnesses."""
    return _HookHost(values, name)


def run_on_modify(host: Any) -> None:
    """Run the validated on-modify composition root for ``host``."""
    host._reset_modify_runtime_state()
    state = host._modify_runtime_state()
    startup_t0 = host._ptime.perf_counter()
    module_t0 = host._ptime.perf_counter()
    capabilities = capabilities_for(host)
    hook_results = capabilities.hook_results
    state.diag_stats["startup_module_ms"] = round(
        (host._ptime.perf_counter() - module_t0) * 1000.0, 3
    )
    read_t0 = host._ptime.perf_counter()
    old, new = host._read_two()
    alias_candidate = re.search(
        r"(?:^|\s)(?:a|af|am|o|of|cm|cu):",
        str(new.get("description") or ""),
    ) is not None
    lifecycle = capabilities.modify_lifecycle
    has_nautical_fields = lifecycle.task_has_nautical_fields(old) or lifecycle.task_has_nautical_fields(new)
    if not has_nautical_fields and not alias_candidate:
        hook_results.emit_passthrough_json(new)
        host._write_bench_stats()
        return
    host._load_core()
    hook_context = capabilities.hook_context
    hook_engine = capabilities.hook_engine
    host._apply_description_uda_aliases(old, new)
    validation = host.core._import_sibling("hook_validation_pipeline")
    # Alias expansion mutates the canonical task mapping. Refresh the typed
    # observation so transition diffs and recurrence feedback include aliases.
    if host._PARSED_NEW_OBSERVATION is not None:
        task_models = host.core._import_sibling("task_models")
        host._PARSED_NEW_OBSERVATION = task_models.TaskObservation.from_mapping(
            new,
            source_query="on-modify alias-normalized task",
        )
    _validated_observation, validation_report = validation.validate_task_mapping(
        new,
        route=validation.WorkflowRoute.RECURRING_EDIT,
        source_query="on-modify validation",
    )
    if validation_report.status is not validation.ValidationStatus.VALID:
        finding = validation_report.findings[0]
        title = "Invalid chainMax" if finding.code == "chain_max_invalid" else "Invalid Nautical task"
        host._fail_and_exit(title, f"{finding.reason} {finding.correction}")
    if host._PARSED_OLD_OBSERVATION is not None and host._PARSED_NEW_OBSERVATION is not None:
        transition_report = validation.validate_task_transition(
            host._PARSED_OLD_OBSERVATION,
            host._PARSED_NEW_OBSERVATION,
            route=validation.WorkflowRoute.RECURRING_EDIT,
            source_query="on-modify transition validation",
        )
        if transition_report.status is not validation.ValidationStatus.VALID:
            finding = transition_report.findings[0]
            title = "Invalid chainMax" if finding.code == "chain_max_invalid" else "Invalid recurrence transition"
            host._fail_and_exit(title, f"{finding.reason} {finding.correction}")
    config_error = str(getattr(host.core, "scheduling_configuration_error", lambda: "")() or "")
    if config_error and has_nautical_fields:
        host._fail_and_exit(
            "Invalid Nautical configuration",
            f"{config_error}. Fix Nautical configuration before modifying a recurring task.",
        )
    state.diag_stats["startup_read_input_ms"] = round(
        (host._ptime.perf_counter() - read_t0) * 1000.0, 3
    )
    try:
        calendar_context = host.core.use_task_business_calendar(new)
    except Exception as exc:
        host._fail_and_exit("Invalid business calendar", str(exc))
        return
    request_t0 = host._ptime.perf_counter()
    capabilities.modify_read_effects.seed_runtime_lookup_tasks(host, old, new)
    runtime = host._build_hook_runtime_context(new)
    host._modify_runtime_state().workflow_context = runtime.workflow
    request = hook_context.build_on_modify_request(
        runtime=runtime,
        old=old,
        new=new,
        old_observation=host._PARSED_OLD_OBSERVATION,
        new_observation=host._PARSED_NEW_OBSERVATION,
    )
    if host._IMPORT_MS is not None:
        state.diag_stats["startup_import_ms"] = round(float(host._IMPORT_MS), 3)
    state.diag_stats["startup_request_ms"] = round(
        (host._ptime.perf_counter() - request_t0) * 1000.0, 3
    )
    state.diag_stats["startup_total_ms"] = round(
        (host._ptime.perf_counter() - startup_t0) * 1000.0, 3
    )
    displacement_context = (
        host.core.capture_business_calendar_displacements()
        if str(new.get("bc") or "").strip()
        else nullcontext()
    )
    try:
        with calendar_context, displacement_context:
            result = hook_engine.handle_on_modify(
                request,
                services=ModifyCompositionServices(
                    host, hook_results.TaskHookResponse
                ),
            )
        if result is not None:
            hook_results.emit_json_result(result, core=host.core)
    finally:
        runtime.close()
        host._write_bench_stats()


__all__ = ("ModifyCompositionServices",)
