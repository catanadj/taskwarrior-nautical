"""Completion preflight and occurrence-limit effects for on-modify."""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import Any, Protocol

from .task_models import TaskPayload
from .task_datetime import datetime_value, parser_for_host
from .timeutil import compare_datetimes
from .callback_ports import CallbackPort


class CompletionPreflightService(Protocol):
    """Validated preflight service used by completion effects."""

    def completion_link_numbers_or_fail(self, task: TaskPayload, **kwargs: Any) -> Any: ...
    def completion_kind_or_stop(self, task: TaskPayload, now_utc: datetime, **kwargs: Any) -> Any: ...
    def completion_chain_id_or_fail(self, task: TaskPayload, **kwargs: Any) -> str | None: ...
    def completion_existing_next_or_fail(self, task: TaskPayload, next_no: int, **kwargs: Any) -> bool: ...
    def completion_preflight_context(self, task: TaskPayload, now_utc: datetime, *, services: Any) -> Any: ...


class CompletionComputeService(Protocol):
    """Validated compute service used by completion effects."""

    def completion_compute_child_due(self, task: TaskPayload, kind: str, **kwargs: Any) -> Any: ...
    def completion_until_or_fail(self, task: TaskPayload, now_utc: datetime, **kwargs: Any) -> Any: ...
    def completion_until_guard_or_stop(self, task: TaskPayload, child_due: Any, until_dt: Any, now_utc: datetime, **kwargs: Any) -> bool: ...
    def completion_require_child_due_or_fail(self, task: TaskPayload, child_due: Any, **kwargs: Any) -> bool: ...
    def completion_warn_unreasonable_duration(self, task: TaskPayload, child_due: Any, until_dt: Any, now_utc: datetime, **kwargs: Any) -> None: ...
    def completion_caps(self, kind: str, task: TaskPayload, child_due: Any, dnf: Any, **kwargs: Any) -> Any: ...
    def completion_cap_guard_or_stop(self, task: TaskPayload, next_no: int, cap_no: int | None, now_utc: datetime, **kwargs: Any) -> bool: ...
    def completion_compute_next_and_limits(self, task: TaskPayload, kind: str, next_no: int, now_utc: datetime, *, services: Any) -> Any: ...
    def attach_lifecycle_plan(self, task: TaskPayload, computed: Any, next_no: int, now_utc: datetime, **kwargs: Any) -> Any: ...


class CompletionSpawnService(Protocol):
    """Validated child-spawn service used by completion effects."""

    def completion_build_and_spawn_child(self, task: TaskPayload, *, services: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class SnapshotPorts:
    repository: Any
    mode: Any
    models: Any
    task_observation: Any


@dataclass(frozen=True, slots=True)
class CompletionPreflightPorts:
    preflight: CompletionPreflightService
    coerce_int: CallbackPort
    max_link_number: int
    short_uuid: Any
    panel: Any
    print_task: Any
    end_chain_summary: Any
    existing_next_lookup: Any


@dataclass(frozen=True, slots=True)
class CompletionFeedbackPorts:
    compute: CompletionComputeService
    panel: Any
    print_task: Any
    end_chain_summary: Any


@dataclass(frozen=True, slots=True)
class UntilCompletionPorts:
    compute: CompletionComputeService
    parse_datetime: Any
    validate_until_not_past: Any
    panel: Any
    print_task: Any


@dataclass(frozen=True, slots=True)
class CompletionCapsPorts:
    compute: CompletionComputeService
    coerce_int: Any
    parse_datetime: Any
    estimate_cp: Any
    estimate_anchor: Any
    cap_cp: Any
    cap_anchor: Any


@dataclass(frozen=True, slots=True)
class ChildDuePorts:
    compute: CompletionComputeService
    generation: Any
    decode_task: Any
    task_model: Any
    exhaustion_message: Any
    ensure_terminal: Any
    end_summary: Any
    now_utc: Any
    panel: Any
    print_task: Any
    diag: Any


@dataclass(frozen=True, slots=True)
class DurationWarningPorts:
    compute: CompletionComputeService
    validate_duration: Any
    panel: Any


@dataclass(frozen=True, slots=True)
class CompletionLifecyclePlanPorts:
    generation: Any
    scheduler_fingerprint: Any
    compare_datetimes: Any
    invalid_relative_carry_reason: Any
    lifecycle_planner: Any
    lifecycle_models: Any
    modify_models: Any
    end_chain_summary: Any
    ensure_terminal_chain_off: Any
    panel: Any
    print_task: Any
    diagnostic: Any


@dataclass(frozen=True, slots=True)
class CompletionComputePorts:
    compute: CompletionComputeService
    services_type: Any
    compute_child_due: Any
    until_or_fail: Any
    until_guard_or_stop: Any
    require_child_due_or_fail: Any
    warn_unreasonable_duration: Any
    caps: Any
    cap_guard_or_stop: Any
    lifecycle_result_type: Any
    lifecycle_plan: CompletionLifecyclePlanPorts


@dataclass(frozen=True, slots=True)
class CompletionPreflightContextPorts:
    preflight: CompletionPreflightService
    models: Any
    task_observation: Any
    snapshot_mode: Any
    coerce_int: Any
    max_link_number: int
    short_uuid: Any
    panel: Any
    print_task: Any
    end_chain_summary: Any


@dataclass(frozen=True, slots=True)
class CompletionSpawnPorts:
    spawn: CompletionSpawnService
    services_type: Any
    build_child_draft: Any
    spawn_child_atomic: Any
    panel: Any
    print_task: Any
    diagnostic: Any


def _ui_ports_for(host: Any):
    ui = host._module("modify_ui_effects")
    return ui, ui.ui_ports_for(host)


def _print_task_port_for(host: Any):
    ui, ports = _ui_ports_for(host)
    return lambda task: ui.print_task(ports, task)


def _end_summary_port_for(host: Any):
    diagnostics = host._module("modify_diagnostics_effects")
    ports = diagnostics.end_chain_summary_ports_for(host)
    return lambda task, reason, now, current_task=None: diagnostics.end_chain_summary(
        ports, task, reason, now, current_task
    )


def _feedback_ports_for(host: Any, compute: Any, *, summarize: bool = True) -> CompletionFeedbackPorts:
    return CompletionFeedbackPorts(
        compute=compute,
        panel=_panel_port_for(host),
        print_task=_print_task_port_for(host),
        end_chain_summary=_end_summary_port_for(host) if summarize else (lambda *args, **kwargs: None),
    )


def _panel_port_for(host: Any):
    ui, ports = _ui_ports_for(host)
    return lambda title, rows, **kwargs: ui.panel(ports, title, rows, **kwargs)


def link_numbers_or_fail(ports: CompletionPreflightPorts, new: TaskPayload):
    return ports.preflight.completion_link_numbers_or_fail(
        new,
        coerce_int=ports.coerce_int, max_link_number=ports.max_link_number,
        panel=ports.panel, print_task=ports.print_task,
    )


def kind_or_stop(ports: CompletionPreflightPorts, new: TaskPayload, now_utc: datetime):
    return ports.preflight.completion_kind_or_stop(
        new,
        now_utc,
        panel=ports.panel,
        print_task=ports.print_task,
        end_chain_summary=ports.end_chain_summary,
    )


def chain_id_or_fail(ports: CompletionPreflightPorts, new: TaskPayload) -> str | None:
    return ports.preflight.completion_chain_id_or_fail(
        new,
        panel=ports.panel, print_task=ports.print_task,
    )


def existing_next_or_fail(ports: CompletionPreflightPorts, new: TaskPayload, next_no: int, chain_snapshot) -> bool:
    return ports.preflight.completion_existing_next_or_fail(
        new,
        next_no,
        existing_next_lookup=ports.existing_next_lookup,
        short=ports.short_uuid, panel=ports.panel, print_task=ports.print_task,
    )


def _snapshot_mode(ports: SnapshotPorts) -> str:
    return ports.mode()


def chain_snapshot(ports: SnapshotPorts, chain_id: str, base_no: int, next_no: int):
    del base_no, next_no
    from .integration_models import Absent, Found, Unavailable

    snapshot = ports.repository.chain_snapshot(chain_id)
    if isinstance(snapshot, Found):
        value = getattr(snapshot.value, "rows", snapshot.value)
        if not isinstance(value, (list, tuple)):
            return ports.models.CompletionChainSnapshot(
                mode=_snapshot_mode(ports), rows=[], loaded=False,
                chain_id=chain_id, error="typed chain snapshot rows are unavailable"
            )
        rows = [
            row if hasattr(row, "to_mapping") else ports.task_observation.from_mapping(
                row, source_query=f"chain:{chain_id}:completion"
            )
            for row in value
        ]
        loaded, error = True, ""
    elif isinstance(snapshot, Absent):
        rows, loaded, error = [], True, ""
    elif isinstance(snapshot, Unavailable):
        rows, loaded = [], False
        error = snapshot.evidence.detail or snapshot.evidence.kind.value
    else:
        rows, loaded, error = [], False, "typed chain read returned an unsupported result"
    return ports.models.CompletionChainSnapshot(
        mode=_snapshot_mode(ports), rows=rows, loaded=loaded, chain_id=chain_id, error=error
    )


def completion_preflight_context_ports_for(host: Any) -> CompletionPreflightContextPorts:
    preflight = host._module("modify_completion_preflight")
    models = host._module("modify_models")
    return CompletionPreflightContextPorts(
        preflight=preflight,
        models=models,
        task_observation=host._module("task_models").TaskObservation,
        snapshot_mode=lambda: (
            "full" if host._SHOW_ANALYTICS or host._CHECK_CHAIN_INTEGRITY else
            "next" if str(getattr(host.core, "PANEL_MODE", "rich") or "rich").strip().lower()
            in {"line", "minimal", "quiet", "text"} else "recent"
        ),
        coerce_int=host.core.coerce_int,
        max_link_number=host.core.MAX_LINK_NUMBER,
        short_uuid=host.core.short_uuid,
        panel=_panel_port_for(host),
        print_task=_print_task_port_for(host),
        end_chain_summary=_end_summary_port_for(host),
    )


def preflight_context(
    ports: CompletionPreflightContextPorts,
    new: TaskPayload,
    now_utc: datetime,
    repository,
):
    preflight = ports.preflight
    models = ports.models
    snapshot_ports = SnapshotPorts(
        repository=repository,
        mode=ports.snapshot_mode,
        models=models,
        task_observation=ports.task_observation,
    )
    preflight_ports = CompletionPreflightPorts(
        preflight=preflight,
        coerce_int=ports.coerce_int,
        max_link_number=ports.max_link_number,
        short_uuid=ports.short_uuid,
        panel=ports.panel,
        print_task=ports.print_task,
        end_chain_summary=ports.end_chain_summary,
        existing_next_lookup=lambda task, link: repository.exact_child_slot(str(task.get("chainID") or ""), link),
    )
    services = models.CompletionPreflightServices(
        short=ports.short_uuid,
        completion_link_numbers_or_fail=lambda task: link_numbers_or_fail(preflight_ports, task),
        completion_kind_or_stop=lambda task, clock: kind_or_stop(
            preflight_ports, task, clock
        ),
        completion_chain_id_or_fail=lambda task: chain_id_or_fail(preflight_ports, task),
        completion_chain_snapshot=lambda chain_id, base_no, next_no: chain_snapshot(snapshot_ports, chain_id, base_no, next_no),
        completion_existing_next_or_fail=lambda task, next_no, snapshot: existing_next_or_fail(preflight_ports, task, next_no, snapshot),
    )
    return preflight.completion_preflight_context(new, now_utc, services=services)


def compute_child_due(ports: ChildDuePorts, new: TaskPayload, kind: str):
    compute = ports.compute

    def typed_task(task):
        return ports.task_model.NauticalTask.from_observation(ports.decode_task(task, source_query="on-modify completion"))

    def handle_terminal(exc) -> bool:
        message = ports.exhaustion_message(exc)
        if exc.is_date_limit:
            ports.ensure_terminal(new, "complete")
            try:
                ports.end_summary(new, message, ports.now_utc(), current_task=new)
            except Exception as summary_exc:
                ports.diag(f"terminal chain summary failed: {summary_exc}")
                ports.panel("⛔ Nautical chain stopped", [("Reason", message), ("Task", str(new.get("uuid") or "")[:8] or "–")], kind="summary")
            ports.print_task(new)
            return True
        ports.panel("⛔ Chain error", [("Scheduler", message), ("Fix", "Use a less sparse rule or adjust its search limits.")], kind="error")
        ports.print_task(new)
        return True

    return compute.completion_compute_child_due(
        new,
        kind,
        compute_anchor_child_due=lambda task: ports.generation.compute_anchor_child_due(typed_task(task)),
        compute_cp_child_due=lambda task: ports.generation.compute_cp_child_due(typed_task(task)),
        panel=ports.panel, print_task=ports.print_task, diag=ports.diag,
        on_terminal=handle_terminal,
    )


def until_or_fail(ports: UntilCompletionPorts, new: TaskPayload, now_utc: datetime):
    return ports.compute.completion_until_or_fail(
        new, now_utc,
        safe_parse_datetime=ports.parse_datetime,
        validate_until_not_past=ports.validate_until_not_past,
        panel=ports.panel, print_task=ports.print_task,
    )


def until_guard_or_stop(ports: CompletionFeedbackPorts, new: TaskPayload, child_due, until_dt, now_utc: datetime) -> bool:
    return ports.compute.completion_until_guard_or_stop(
        new, child_due, until_dt, now_utc,
        end_chain_summary=ports.end_chain_summary, print_task=ports.print_task,
    )


def require_child_due_or_fail(ports: CompletionFeedbackPorts, new: TaskPayload, child_due) -> bool:
    return ports.compute.completion_require_child_due_or_fail(
        new, child_due, panel=ports.panel, print_task=ports.print_task
    )


def warn_unreasonable_duration(ports: DurationWarningPorts, new: TaskPayload, child_due, until_dt, now_utc: datetime) -> None:
    ports.compute.completion_warn_unreasonable_duration(
        new, child_due, until_dt, now_utc,
        validate_chain_duration_reasonable=ports.validate_duration,
        panel=ports.panel,
    )


def caps(ports: CompletionCapsPorts, kind: str, new: TaskPayload, child_due, dnf):
    return ports.compute.completion_caps(
        kind, new, child_due, dnf,
        coerce_int=ports.coerce_int, dtparse=ports.parse_datetime,
        estimate_cp_final_by_max=ports.estimate_cp,
        estimate_anchor_final_by_max=ports.estimate_anchor,
        cap_from_until_cp=ports.cap_cp,
        cap_from_until_anchor=ports.cap_anchor,
    )


def cap_guard_or_stop(ports: CompletionFeedbackPorts, new: TaskPayload, next_no: int, cap_no: int | None, now_utc: datetime) -> bool:
    return ports.compute.completion_cap_guard_or_stop(
        new, next_no, cap_no, now_utc,
        end_chain_summary=ports.end_chain_summary, print_task=ports.print_task
    )


def completion_compute_ports_for(host: Any) -> CompletionComputePorts:
    compute = host._module("modify_completion_compute")
    models = host._module("modify_models")
    schedule = host._module("modify_schedule_effects")
    validation = host._module("modify_validation_effects")
    generation_module = host._module("modify_generation_effects")
    generation = generation_module.chain_generation_service(
        generation_module.generation_ports_for(host)
    )
    cp_schedule_ports = schedule.cp_completion_ports_for(host)
    anchor_schedule_ports = schedule.anchor_completion_ports_for(host)
    feedback = _feedback_ports_for(host, compute)
    feedback_without_summary = _feedback_ports_for(host, compute, summarize=False)
    child_due_ports = ChildDuePorts(
        compute=compute,
        generation=generation,
        decode_task=host._module("task_codec").DEFAULT_TASK_CODEC.decode_row,
        task_model=host._module("task_models"),
        exhaustion_message=host.core._import_sibling("scheduler_models").occurrence_exhaustion_message,
        ensure_terminal=lambda task, event=None: host._module(
            "modify_composition_adapters"
        ).ensure_terminal_chain_off_for(host, task, event),
        end_summary=_end_summary_port_for(host),
        now_utc=host._workflow_now_utc,
        panel=_panel_port_for(host),
        print_task=_print_task_port_for(host),
        diag=host._diag,
    )
    until_ports = UntilCompletionPorts(
        compute=compute,
        parse_datetime=host._TASK_DATETIME_PARSER.parse,
        validate_until_not_past=lambda until_dt, now: validation.until_not_past(
            validation.UntilPorts(
                lambda _now: host.timedelta(minutes=1),
                compare_datetimes,
                host.core.humanize_delta,
            ),
            until_dt,
            now,
        ),
        panel=_panel_port_for(host),
        print_task=_print_task_port_for(host),
    )
    duration_ports = DurationWarningPorts(
        compute=compute,
        validate_duration=lambda child_due, until_dt, now: validation.chain_duration_reasonable(
            validation.DurationPorts(host._MIN_FUTURE_WARN, host.core.fmt_dt_local),
            child_due,
            until_dt,
            now,
        ),
        panel=_panel_port_for(host),
    )
    caps_ports = CompletionCapsPorts(
        compute=compute,
        coerce_int=host.core.coerce_int,
        parse_datetime=lambda value: datetime_value(parser_for_host(host), value),
        estimate_cp=lambda task, due: schedule.estimate_cp_final_by_max(
            cp_schedule_ports, task, due
        ),
        estimate_anchor=lambda task, due, expression: schedule.estimate_anchor_final_by_max(
            anchor_schedule_ports, task, due, expression
        ),
        cap_cp=lambda task, due: schedule.cap_from_until_cp(cp_schedule_ports, task, due),
        cap_anchor=lambda task, due, expression: schedule.cap_from_until_anchor(
            anchor_schedule_ports, task, due, expression
        ),
    )
    fingerprint = getattr(host.core, "scheduler_config_fingerprint", None)
    plan_ports = CompletionLifecyclePlanPorts(
        generation=generation,
        scheduler_fingerprint=fingerprint if callable(fingerprint) else (lambda: ""),
        compare_datetimes=lambda left, right: host._module(
            "modify_value_effects"
        ).compare_datetimes(
            host._module("modify_value_effects").DatetimePorts(
                compare_datetimes
            ),
            left,
            right,
        ),
        invalid_relative_carry_reason=host._module(
            "chain_integrity_lifecycle"
        ).invalid_relative_carry_reason,
        lifecycle_planner=host._module("lifecycle_planner"),
        lifecycle_models=host._module("lifecycle_models"),
        modify_models=models,
        end_chain_summary=_end_summary_port_for(host),
        ensure_terminal_chain_off=lambda task, event=None: host._module(
            "modify_composition_adapters"
        ).ensure_terminal_chain_off_for(host, task, event),
        panel=_panel_port_for(host),
        print_task=_print_task_port_for(host),
        diagnostic=host._diag,
    )
    return CompletionComputePorts(
        compute=compute,
        services_type=models.CompletionComputeServices,
        compute_child_due=lambda value, value_kind: compute_child_due(
            child_due_ports, value, value_kind
        ),
        until_or_fail=lambda value, clock: until_or_fail(until_ports, value, clock),
        until_guard_or_stop=lambda value, due, until, clock: until_guard_or_stop(
            feedback, value, due, until, clock
        ),
        require_child_due_or_fail=lambda value, due: require_child_due_or_fail(
            feedback_without_summary, value, due
        ),
        warn_unreasonable_duration=lambda value, due, until, clock: warn_unreasonable_duration(
            duration_ports, value, due, until, clock
        ),
        caps=lambda value_kind, value, due, dnf: caps(
            caps_ports, value_kind, value, due, dnf
        ),
        cap_guard_or_stop=lambda value, number, cap, clock: cap_guard_or_stop(
            feedback, value, number, cap, clock
        ),
        lifecycle_result_type=models.CompletionLifecycleResult,
        lifecycle_plan=plan_ports,
    )


def compute_next_and_limits(
    ports: CompletionComputePorts,
    new: TaskPayload,
    kind: str,
    next_no: int,
    now_utc: datetime,
    *,
    preflight=None,
):
    services = ports.services_type(
        completion_compute_child_due=ports.compute_child_due,
        completion_until_or_fail=ports.until_or_fail,
        completion_until_guard_or_stop=ports.until_guard_or_stop,
        completion_require_child_due_or_fail=ports.require_child_due_or_fail,
        completion_warn_unreasonable_duration=ports.warn_unreasonable_duration,
        completion_caps=ports.caps,
        completion_cap_guard_or_stop=ports.cap_guard_or_stop,
    )
    computed = ports.compute.completion_compute_next_and_limits(
        new, kind, next_no, now_utc, services=services
    )
    if computed is None:
        return None
    if isinstance(computed, ports.lifecycle_result_type):
        return computed
    if not str(new.get("uuid") or "").strip() or not str(new.get("chainID") or "").strip():
        return computed
    plan = ports.lifecycle_plan
    return ports.compute.attach_lifecycle_plan(
        new,
        computed,
        next_no,
        now_utc,
        preflight=preflight,
        generation=plan.generation,
        scheduler_fingerprint=plan.scheduler_fingerprint(),
        compare_datetimes=plan.compare_datetimes,
        invalid_relative_carry_reason=plan.invalid_relative_carry_reason,
        lifecycle_planner=plan.lifecycle_planner,
        lifecycle_models=plan.lifecycle_models,
        modify_models=plan.modify_models,
        end_chain_summary=plan.end_chain_summary,
        ensure_terminal_chain_off=plan.ensure_terminal_chain_off,
        panel=plan.panel,
        print_task=plan.print_task,
        diag=plan.diagnostic,
    )


def completion_spawn_ports_for(host: Any) -> CompletionSpawnPorts:
    spawn = host._module("modify_completion_spawn")
    generation_module = host._module("modify_generation_effects")
    generation = generation_module.chain_generation_service(generation_module.generation_ports_for(host))
    codec = host._module("task_codec")
    task_models = host._module("task_models")
    models = host._module("modify_models")

    def build_child_draft(task, *args, **inner_kwargs):
        typed_task = task_models.NauticalTask.from_observation(
            codec.DEFAULT_TASK_CODEC.decode_row(task, source_query="on-modify completion")
        )
        return generation.build_child_draft(typed_task, *args, **inner_kwargs)

    spawn_effects = host._module("modify_spawn_effects")
    spawn_ports = spawn_effects.spawn_child_ports_for(host)
    return CompletionSpawnPorts(
        spawn=spawn,
        services_type=models.CompletionSpawnServices,
        build_child_draft=build_child_draft,
        spawn_child_atomic=lambda child, parent, *, lifecycle_plan=None: spawn_effects.spawn_child_atomic(
            spawn_ports, child, parent, lifecycle_plan=lifecycle_plan
        ),
        panel=_panel_port_for(host),
        print_task=_print_task_port_for(host),
        diagnostic=host._diag,
    )


def build_and_spawn_child(ports: CompletionSpawnPorts, new: TaskPayload, **kwargs):
    services = ports.services_type(
        build_child_draft=ports.build_child_draft,
        spawn_child_atomic=ports.spawn_child_atomic,
        panel=ports.panel,
        print_task=ports.print_task,
        diag=ports.diagnostic,
    )
    return ports.spawn.completion_build_and_spawn_child(new, services=services, **kwargs)


__all__ = (
    "CompletionComputePorts", "CompletionLifecyclePlanPorts",
    "CompletionPreflightContextPorts", "CompletionSpawnPorts",
    "completion_compute_ports_for", "completion_preflight_context_ports_for",
    "completion_spawn_ports_for",
    "link_numbers_or_fail", "kind_or_stop", "chain_id_or_fail", "existing_next_or_fail", "chain_snapshot", "preflight_context",
    "compute_child_due", "until_or_fail", "until_guard_or_stop", "require_child_due_or_fail",
    "warn_unreasonable_duration", "caps", "cap_guard_or_stop",
    "compute_next_and_limits", "build_and_spawn_child",
)
