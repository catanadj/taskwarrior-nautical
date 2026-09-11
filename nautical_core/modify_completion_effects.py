"""Completion preflight and occurrence-limit effects for on-modify."""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import Any

from .task_models import TaskPayload
from .task_datetime import datetime_value, parser_for_host


@dataclass(frozen=True, slots=True)
class SnapshotPorts:
    repository: Any
    mode: Any
    models: Any
    task_observation: Any


@dataclass(frozen=True, slots=True)
class CompletionPreflightPorts:
    preflight: Any
    coerce_int: Any
    max_link_number: int
    short_uuid: Any
    panel: Any
    print_task: Any
    end_chain_summary: Any
    existing_next_lookup: Any


@dataclass(frozen=True, slots=True)
class CompletionFeedbackPorts:
    compute: Any
    panel: Any
    print_task: Any
    end_chain_summary: Any


@dataclass(frozen=True, slots=True)
class UntilCompletionPorts:
    compute: Any
    parse_datetime: Any
    validate_until_not_past: Any
    panel: Any
    print_task: Any


@dataclass(frozen=True, slots=True)
class CompletionCapsPorts:
    compute: Any
    coerce_int: Any
    parse_datetime: Any
    estimate_cp: Any
    estimate_anchor: Any
    cap_cp: Any
    cap_anchor: Any


@dataclass(frozen=True, slots=True)
class ChildDuePorts:
    compute: Any
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
    compute: Any
    validate_duration: Any
    panel: Any


def _feedback_ports(host: Any, compute: Any, *, summarize: bool = True) -> CompletionFeedbackPorts:
    return CompletionFeedbackPorts(
        compute=compute,
        panel=_panel_callback(host),
        print_task=lambda task: host._module("modify_ui_effects").print_task(host, task),
        end_chain_summary=(
            lambda task, reason, now, current_task=None: host._module("modify_diagnostics_effects").end_chain_summary(
                host, task, reason, now, current_task
            )
            if summarize else (lambda *args, **kwargs: None)
        ),
    )


def _panel(host: Any, title, rows, **kwargs):
    return host._module("modify_ui_effects").panel(host, title, rows, **kwargs)


def _panel_callback(host: Any):
    return lambda title, rows, **kwargs: _panel(host, title, rows, **kwargs)


def link_numbers_or_fail(ports: CompletionPreflightPorts, new: TaskPayload):
    return ports.preflight.completion_link_numbers_or_fail(
        new,
        coerce_int=ports.coerce_int, max_link_number=ports.max_link_number,
        panel=ports.panel, print_task=ports.print_task,
    )


def kind_or_stop(host: Any, new: TaskPayload, now_utc: datetime):
    return host._module("modify_completion_preflight").completion_kind_or_stop(
        new,
        now_utc,
        panel=_panel_callback(host),
        print_task=lambda task: host._module("modify_ui_effects").print_task(host, task),
        end_chain_summary=lambda task, reason, now, current_task=None: host._module("modify_diagnostics_effects").end_chain_summary(host, task, reason, now, current_task),
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


def _snapshot_mode_for_host(host: Any) -> str:
    if host._SHOW_ANALYTICS or host._CHECK_CHAIN_INTEGRITY:
        return "full"
    mode = str(getattr(host.core, "PANEL_MODE", "rich") or "rich").strip().lower()
    return "next" if mode in {"line", "minimal", "quiet", "text"} else "recent"


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


def preflight_context(host: Any, new: TaskPayload, now_utc: datetime, repository):
    preflight = host._module("modify_completion_preflight")
    models = host._module("modify_models")
    snapshot_ports = SnapshotPorts(
        repository=repository,
        mode=lambda: _snapshot_mode_for_host(host),
        models=models,
        task_observation=host._module("task_models").TaskObservation,
    )
    preflight_ports = CompletionPreflightPorts(
        preflight=preflight,
        coerce_int=host.core.coerce_int,
        max_link_number=host.core.MAX_LINK_NUMBER,
        short_uuid=host.core.short_uuid,
        panel=_panel_callback(host),
        print_task=lambda task: host._module("modify_ui_effects").print_task(host, task),
        end_chain_summary=lambda task, reason, now, current_task=None: host._module("modify_diagnostics_effects").end_chain_summary(host, task, reason, now, current_task),
        existing_next_lookup=lambda task, link: repository.exact_child_slot(str(task.get("chainID") or ""), link),
    )
    services = models.CompletionPreflightServices(
        short=host.core.short_uuid,
        completion_link_numbers_or_fail=lambda task: link_numbers_or_fail(preflight_ports, task),
        completion_kind_or_stop=lambda task, clock: preflight.completion_kind_or_stop(task, clock, panel=preflight_ports.panel, print_task=preflight_ports.print_task, end_chain_summary=preflight_ports.end_chain_summary),
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


def compute_next_and_limits(host: Any, new: TaskPayload, kind: str, next_no: int, now_utc: datetime, *, preflight=None):
    compute = host._module("modify_completion_compute")
    models = host._module("modify_models")
    services = models.CompletionComputeServices(
        completion_compute_child_due=lambda value, value_kind: compute_child_due(
            ChildDuePorts(
                compute=compute,
                generation=host._module("modify_generation_effects").chain_generation_service(
                    host._module("modify_generation_effects").generation_ports_for(host)
                ),
                decode_task=host._module("task_codec").DEFAULT_TASK_CODEC.decode_row,
                task_model=host._module("task_models"),
                exhaustion_message=host.core._import_sibling("scheduler_models").occurrence_exhaustion_message,
                ensure_terminal=lambda task, event=None: host._module("modify_presentation_effects").ensure_terminal_chain_off(host, task, event),
                end_summary=lambda task, reason, now, current_task=None: host._module("modify_diagnostics_effects").end_chain_summary(host, task, reason, now, current_task),
                now_utc=host._workflow_now_utc,
                panel=_panel_callback(host),
                print_task=lambda task: host._module("modify_ui_effects").print_task(host, task),
                diag=host._diag,
            ), value, value_kind
        ),
        completion_until_or_fail=lambda value, clock: until_or_fail(
            UntilCompletionPorts(
                compute=compute,
                parse_datetime=host._TASK_DATETIME_PARSER.parse,
                validate_until_not_past=lambda until_dt, now: host._module("modify_validation_effects").until_not_past(
                    host._module("modify_validation_effects").UntilPorts(
                        lambda _now: host.timedelta(minutes=1),
                        host._module("timeutil").compare_datetimes,
                        host.core.humanize_delta,
                    ), until_dt, now,
                ),
                panel=_panel_callback(host),
                print_task=lambda task: host._module("modify_ui_effects").print_task(host, task),
            ), value, clock
        ),
        completion_until_guard_or_stop=lambda value, due, until, clock: until_guard_or_stop(_feedback_ports(host, compute), value, due, until, clock),
        completion_require_child_due_or_fail=lambda value, due: require_child_due_or_fail(_feedback_ports(host, compute, summarize=False), value, due),
        completion_warn_unreasonable_duration=lambda value, due, until, clock: warn_unreasonable_duration(
            DurationWarningPorts(
                compute=compute,
                validate_duration=lambda child_due, until_dt, now: host._module("modify_validation_effects").chain_duration_reasonable(
                    host._module("modify_validation_effects").DurationPorts(host._MIN_FUTURE_WARN, host.core.fmt_dt_local), child_due, until_dt, now
                ),
                panel=_panel_callback(host),
            ), value, due, until, clock
        ),
        completion_caps=lambda value_kind, value, due, dnf: caps(
            CompletionCapsPorts(
                compute=compute,
                coerce_int=host.core.coerce_int,
                parse_datetime=lambda value: datetime_value(parser_for_host(host), value),
                estimate_cp=lambda task, due: host._module("modify_schedule_effects").estimate_cp_final_by_max(host, task, due),
                estimate_anchor=lambda task, due, expression: host._module("modify_schedule_effects").estimate_anchor_final_by_max(host, task, due, expression),
                cap_cp=lambda task, due: host._module("modify_schedule_effects").cap_from_until_cp(host, task, due),
                cap_anchor=lambda task, due, expression: host._module("modify_schedule_effects").cap_from_until_anchor(host, task, due, expression),
            ), value_kind, value, due, dnf
        ),
        completion_cap_guard_or_stop=lambda value, number, cap, clock: cap_guard_or_stop(_feedback_ports(host, compute), value, number, cap, clock),
    )
    computed = compute.completion_compute_next_and_limits(new, kind, next_no, now_utc, services=services)
    if computed is None:
        return None
    if isinstance(computed, host._module("modify_models").CompletionLifecycleResult):
        return computed
    if not str(new.get("uuid") or "").strip() or not str(new.get("chainID") or "").strip():
        return computed
    fingerprint_fn = getattr(host.core, "scheduler_config_fingerprint", None)
    return compute.attach_lifecycle_plan(
        new, computed, next_no, now_utc,
        preflight=preflight,
        generation=host._module("modify_generation_effects").chain_generation_service(
            host._module("modify_generation_effects").generation_ports_for(host)
        ),
        scheduler_fingerprint=fingerprint_fn() if callable(fingerprint_fn) else "",
        compare_datetimes=lambda left, right: host._module("modify_value_effects").compare_datetimes(
            host._module("modify_value_effects").DatetimePorts(host._module("timeutil").compare_datetimes), left, right
        ),
        invalid_relative_carry_reason=host._module("chain_integrity_lifecycle").invalid_relative_carry_reason,
        lifecycle_planner=host._module("lifecycle_planner"),
        lifecycle_models=host._module("lifecycle_models"),
        modify_models=host._module("modify_models"),
        end_chain_summary=lambda task, reason, now, current_task=None: host._module("modify_diagnostics_effects").end_chain_summary(host, task, reason, now, current_task),
        ensure_terminal_chain_off=lambda task, event=None: host._module("modify_presentation_effects").ensure_terminal_chain_off(host, task, event),
        panel=_panel_callback(host),
        print_task=lambda task: host._module("modify_ui_effects").print_task(host, task),
        diag=host._diag,
    )


def build_and_spawn_child(host: Any, new: TaskPayload, **kwargs):
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
    services = models.CompletionSpawnServices(
        build_child_draft=build_child_draft,
        spawn_child_atomic=lambda child, parent, *, lifecycle_plan=None: spawn_effects.spawn_child_atomic(
            host, child, parent, lifecycle_plan=lifecycle_plan
        ),
        panel=_panel_callback(host),
        print_task=lambda task: host._module("modify_ui_effects").print_task(host, task),
        diag=host._diag,
    )
    return spawn.completion_build_and_spawn_child(new, services=services, **kwargs)


__all__ = (
    "link_numbers_or_fail", "kind_or_stop", "chain_id_or_fail", "existing_next_or_fail", "chain_snapshot", "preflight_context",
    "compute_child_due", "until_or_fail", "until_guard_or_stop", "require_child_due_or_fail",
    "warn_unreasonable_duration", "caps", "cap_guard_or_stop",
    "compute_next_and_limits", "build_and_spawn_child",
)
