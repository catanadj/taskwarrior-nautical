"""Composition services for the on-modify hook."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable


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


class ModifyCompositionServices:
    """Bind on-modify effects to the hook's validated composition root."""

    def __init__(self, host: Any, result_cls: Callable[..., Any]) -> None:
        self._host = host
        self._result_cls = result_cls

    def result(self, task, *, sanitize: bool):
        return self._result_cls(task=task, sanitize=sanitize)

    def has_nautical_fields(self, task):
        return self._host._task_has_nautical_fields(task, task)

    def load_core(self):
        self._host._load_core()

    def diag(self, message: str):
        self._host._diag(message)

    def fail_and_exit(self, title: str, message: str):
        self._host._fail_and_exit(title, message)

    def handle_non_completion(self, old, new, unit_of_work, transition=None):
        self._host._module("modify_effects").handle_non_completion(
            self._host,
            old, new, unit_of_work, transition=transition
        )

    def handle_completion(self, old, new, unit_of_work, transition=None):
        return self._host._module("modify_effects").handle_completion(
            self._host,
            old, new, unit_of_work, transition=transition
        )

    def handle_deleted(
        self, old, new, unit_of_work, transition=None, terminal_decision=None
    ):
        return self._host._module("modify_effects").handle_deleted(
            self._host,
            old,
            new,
            unit_of_work,
            transition=transition,
            terminal_decision=terminal_decision,
        )


def hook_host(values: dict[str, Any], name: str) -> Any:
    """Return a live attribute view for import-by-file test harnesses."""
    return _HookHost(values, name)


def run_on_modify(host: Any) -> None:
    """Run the validated on-modify composition root for ``host``."""
    host._load_core()
    host._reset_modify_runtime_state()
    state = host._modify_runtime_state()
    startup_t0 = host._ptime.perf_counter()
    module_t0 = host._ptime.perf_counter()
    hook_context = host._module("hook_context")
    hook_results = host._module("hook_results")
    hook_engine = host._module("hook_engine")
    composition = host._module("modify_composition")
    state.diag_stats["startup_module_ms"] = round(
        (host._ptime.perf_counter() - module_t0) * 1000.0, 3
    )
    read_t0 = host._ptime.perf_counter()
    old, new = host._read_two()
    host._apply_description_uda_aliases(old, new)
    validation = host.core._import_sibling("hook_validation_pipeline")
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
    if config_error and host._task_has_nautical_fields(old, new):
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
    host._seed_runtime_lookup_tasks(old, new)
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
