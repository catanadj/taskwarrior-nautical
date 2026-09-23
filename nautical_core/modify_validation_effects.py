"""Validation effects for ordinary on-modify recurrence edits."""

from __future__ import annotations

import re
from datetime import timedelta
from dataclasses import dataclass
from typing import Any
from .callback_ports import CallbackPort
from .task_datetime import datetime_value, parser_for_host
from .timeutil import compare_datetimes


@dataclass(frozen=True, slots=True)
class DurationPorts:
    min_future_warn: int
    format_local: CallbackPort


@dataclass(frozen=True, slots=True)
class UntilPorts:
    minute_delta: CallbackPort
    compare: CallbackPort
    humanize: CallbackPort


@dataclass(frozen=True, slots=True)
class AnchorModePorts:
    panel: CallbackPort


@dataclass(frozen=True, slots=True)
class SharedValidationPorts:
    pipeline: Any
    parse_anchor: Any
    validate_anchor: Any
    validate_omit: Any


@dataclass(frozen=True, slots=True)
class CPValidationPorts:
    validate: Any
    parse_cp_sequence: Any
    cp_sequence_error: Any
    parse_chain_max: Any
    parse_datetime: Any


@dataclass(frozen=True, slots=True)
class ChainLimitPorts:
    pipeline: Any
    validate_limits: Any
    parse_cp_sequence: Any
    cp_sequence_error: Any
    parse_chain_max: Any
    parse_datetime: Any
    validate_until_not_past: Any
    now_utc: Any
    fail: Any


@dataclass(frozen=True, slots=True)
class NativeUntilPorts:
    validate: Any
    validate_anchor_mode: Any
    parse_datetime: Any
    validate_after_target: Any
    format_local: Any
    panel: Any
    fail: Any
    abort: Any


@dataclass(frozen=True, slots=True)
class NativeUntilSlotPorts:
    validate: Any
    parse_datetime: Any
    validate_anchor: Any
    collect_time_slots: Any
    validate_time_slots: Any
    normalize_time_slots: Any
    anchor_file_dir: str
    recurrence_context: Any
    to_local: Any
    format_local: Any
    astronomy_is_error: Any
    astronomy_error_message: Any
    panel: Any
    abort: Any


@dataclass(frozen=True, slots=True)
class AnchorValidationPorts:
    lint: Any
    validate_strict: Any
    panel: Any
    is_astronomy_error: Any
    astronomy_error_message: Any
    fail: Any


@dataclass(frozen=True, slots=True)
class OmitValidationPorts:
    pipeline: Any
    parse_anchor: Any
    validate_anchor: Any
    validate_omit: Any
    validate_files: Any
    load_anchor_file: Any
    load_omit_file: Any
    fail: Any


def anchor_error_message(anchor_expr: str, default_msg: str) -> str:
    if re.search(r"(?:^|[^A-Za-z])(w|m|y)(?:/\d+)?:", anchor_expr, re.IGNORECASE):
        return default_msg
    return f"{default_msg} (expected an anchor such as w:mon, m:15, or y:jul)"


def anchor_mode(ports: AnchorModePorts, old: Any, new: Any) -> str:
    raw = str(new.get("anchor_mode") or old.get("anchor_mode") or "skip").strip()
    mode = raw.lower()
    aliases = {"all": "all", "skip": "skip", "flex": "flex"}
    normalized = aliases.get(mode)
    if normalized is None:
        ports.panel("⚠ Anchor mode", [("Warning", f"Unknown anchor mode {raw!r}; using skip.")], kind="warning")
        normalized = "skip"
        new["anchor_mode"] = normalized
    elif new.get("anchor_mode"):
        new["anchor_mode"] = normalized
    return normalized.upper()


def validate_anchor(ports: AnchorValidationPorts, old: Any, new: Any, anchor_expr: str) -> None:
    try:
        _, warns = ports.lint(anchor_expr)
        if warns:
            ports.panel("ℹ️  Lint", [("Hint", warning) for warning in warns], kind="note")
        anchor_mode(AnchorModePorts(ports.panel), old, new)
        # Validation must remain decision-only. Hint persistence has no
        # synchronous consumer and would repeat scheduler work on every edit.
        ports.validate_strict(anchor_expr)
    except TypeError:
        ports.validate_strict(anchor_expr)
    except Exception as exc:
        if ports.is_astronomy_error(exc):
            ports.fail("Invalid anchor", ports.astronomy_error_message(exc))
        ports.fail("Invalid anchor", anchor_error_message(anchor_expr, str(exc)))


def validate_omit(ports: OmitValidationPorts, anchor_expr: str, anchor_file_expr: str, omit_expr: str, omit_file: str) -> None:
    try:
        validate_shared_omit(
            SharedValidationPorts(
                ports.pipeline, ports.parse_anchor, ports.validate_anchor, ports.validate_omit,
            ), omit_expr,
        )
        findings = ports.validate_files(
            anchor_expr,
            anchor_file_expr,
            omit_expr,
            omit_file,
            load_anchor_file=ports.load_anchor_file,
            load_omit_file=ports.load_omit_file,
        )
    except Exception as exc:
        ports.fail("Invalid omit", str(exc))
        return
    if findings:
        finding = findings[0]
        ports.fail(f"Invalid {finding.field}", finding.reason)


def anchor_validation_ports_for(host: Any) -> AnchorValidationPorts:
    astronomy = host.core._import_sibling("astronomy")
    ui = host._module("modify_ui_effects")
    ui_ports = ui.ui_ports_for(host)
    return AnchorValidationPorts(
        lint=host.core.lint_anchor_expr,
        validate_strict=host.core._parser_api.validate_anchor_expr_strict,
        panel=lambda title, rows, **kwargs: ui.panel(ui_ports, title, rows, **kwargs),
        is_astronomy_error=astronomy.is_astronomy_error,
        astronomy_error_message=astronomy.scheduling_error_message,
        fail=host._fail_and_exit,
    )


def omit_validation_ports_for(host: Any) -> OmitValidationPorts:
    pipeline = host.core._import_sibling("hook_validation_pipeline")
    return OmitValidationPorts(
        pipeline=pipeline,
        parse_anchor=host.core._parser_api.parse_anchor_expr_to_dnf,
        validate_anchor=host._validate_anchor_expr_cached,
        validate_omit=host._validate_omit_expr_cached,
        validate_files=pipeline.validate_recurrence_files,
        load_anchor_file=host._load_anchor_file_dates,
        load_omit_file=host._load_omit_file_dates,
        fail=host._fail_and_exit,
    )


def validate_shared_anchor(ports: SharedValidationPorts, expr: str) -> None:
    ports.pipeline.validate_anchor_expression(
        expr,
        parse_anchor_expr=ports.parse_anchor,
        validate_anchor_expr=ports.validate_anchor,
    )


def validate_shared_omit(ports: SharedValidationPorts, expr: str) -> None:
    ports.pipeline.validate_omit_expression(
        expr,
        validate_omit_expr=ports.validate_omit,
    )


def validate_cp(ports: CPValidationPorts, cp_value: str, chain_max_value: Any, chain_until_value: Any) -> None:
    ports.validate(
        cp_value,
        chain_max_value,
        chain_until_value,
        parse_cp_sequence=ports.parse_cp_sequence,
        cp_sequence_parse_error=ports.cp_sequence_error,
        parse_chain_max=ports.parse_chain_max,
        parse_datetime=ports.parse_datetime,
    )


def validate_chain_limits(ports: ChainLimitPorts, task: dict) -> None:
    cpmax, _until_dt, findings = ports.pipeline.validate_recurrence_limits(
        task.get("cp"), task.get("chainMax"), task.get("chainUntil"),
        parse_cp_sequence=ports.parse_cp_sequence,
        cp_sequence_parse_error=ports.cp_sequence_error,
        parse_chain_max=ports.parse_chain_max,
        parse_datetime=ports.parse_datetime,
    )

    if findings:
        finding = findings[0]
        ports.fail(f"Invalid {finding.field}", finding.reason)
    if cpmax is not None:
        task["chainMax"] = cpmax
    return ports.validate_limits(
        task,
        parse_chain_max=ports.parse_chain_max,
        parse_datetime=ports.parse_datetime,
        validate_until_not_past=ports.validate_until_not_past,
        now_utc=ports.now_utc,
        fail=ports.fail,
    )


def chain_limit_ports_for(host: Any) -> ChainLimitPorts:
    add_validation = host.core._import_sibling("add_validation")
    return ChainLimitPorts(
        pipeline=host.core._import_sibling("hook_validation_pipeline"),
        validate_limits=host._module("modify_validation").validate_chain_limits_on_modify,
        parse_cp_sequence=host.core.parse_cp_sequence,
        cp_sequence_error=host.core.cp_sequence_parse_error,
        parse_chain_max=add_validation.parse_chain_max,
        parse_datetime=lambda value: datetime_value(parser_for_host(host), value),
        validate_until_not_past=lambda until_dt, now: until_not_past(
            UntilPorts(lambda _now: timedelta(minutes=1), compare_datetimes, host.core.humanize_delta), until_dt, now
        ),
        now_utc=host.core.now_utc,
        fail=host._fail_and_exit,
    )


def validate_native_until(ports: NativeUntilPorts, task: dict) -> None:
    ports.validate(
        task,
        validate_anchor_mode=ports.validate_anchor_mode,
        safe_parse_datetime=ports.parse_datetime,
        validate_after_target=ports.validate_after_target,
        format_local=ports.format_local,
        panel=ports.panel,
        fail=ports.fail,
        abort=ports.abort,
    )


def native_until_ports_for(host: Any) -> NativeUntilPorts:
    add_validation = host.core._import_sibling("add_validation")
    ui = host._module("modify_ui_effects")
    ui_ports = ui.ui_ports_for(host)
    return NativeUntilPorts(
        validate=host._module("modify_validation").validate_native_until_after_target_or_fail,
        validate_anchor_mode=add_validation.validate_native_until_anchor_mode,
        parse_datetime=host._TASK_DATETIME_PARSER.parse,
        validate_after_target=add_validation.validate_native_until_after_target,
        format_local=host.core.fmt_dt_local,
        panel=lambda title, rows, **kwargs: ui.panel(ui_ports, title, rows, **kwargs),
        fail=host._fail_and_exit,
        abort=host.sys.exit,
    )


def validate_native_until_slots(ports: NativeUntilSlotPorts, task: dict) -> None:
    ports.validate(
        task,
        safe_parse_datetime=ports.parse_datetime,
        validate_anchor=ports.validate_anchor,
        collect_time_slots=ports.collect_time_slots,
        validate_time_slots=ports.validate_time_slots,
        normalize_time_slots=ports.normalize_time_slots,
        anchor_file_dir=ports.anchor_file_dir,
        recurrence_context=ports.recurrence_context,
        to_local=ports.to_local,
        format_local=ports.format_local,
        astronomy_is_error=ports.astronomy_is_error,
        astronomy_error_message=ports.astronomy_error_message,
        panel=ports.panel,
        abort=ports.abort,
    )


def native_until_slot_ports_for(host: Any) -> NativeUntilSlotPorts:
    add_validation = host.core._import_sibling("add_validation")
    astronomy = host.core._import_sibling("astronomy")
    native_until = host.core._import_sibling("native_until")
    recurrence_context = host.core._import_sibling("recurrence_context").RecurrenceContext
    ui = host._module("modify_ui_effects")
    ui_ports = ui.ui_ports_for(host)
    return NativeUntilSlotPorts(
        validate=host._module("modify_validation").validate_native_until_anchor_slots_or_fail,
        parse_datetime=host._TASK_DATETIME_PARSER.parse,
        validate_anchor=host._validate_anchor_expr_cached,
        collect_time_slots=add_validation.collect_anchor_time_slots,
        validate_time_slots=native_until.validate_calendar_slots,
        normalize_time_slots=lambda value, target_date=None: host._module("modify_time_effects").normalize_hhmm_list(
            host._module("modify_time_effects").time_slot_ports_for(host), value, target_date
        ),
        anchor_file_dir=getattr(host.core, "ANCHOR_FILE_DIR", ""),
        recurrence_context=recurrence_context.from_task,
        to_local=host._tolocal,
        format_local=host.core.fmt_dt_local,
        astronomy_is_error=astronomy.is_astronomy_error,
        astronomy_error_message=astronomy.scheduling_error_message,
        panel=lambda title, rows, **kwargs: ui.panel(ui_ports, title, rows, **kwargs),
        abort=host.sys.exit,
    )


def until_not_past(ports: UntilPorts, until_dt: Any, now_utc: Any) -> tuple[bool, str | None]:
    if not until_dt:
        return True, None
    grace = ports.minute_delta(now_utc)
    if ports.compare(until_dt, now_utc - grace) < 0:
        past_s = ports.humanize(until_dt, now_utc, use_months_days=False)
        return False, f"chainUntil is in the past (was {past_s} ago)"
    return True, None


def chain_duration_reasonable(ports: DurationPorts, child_due: Any, until_dt: Any, now_utc: Any) -> tuple[bool, str | None]:
    if not until_dt:
        return True, None
    days = (until_dt - now_utc).days
    if days > ports.min_future_warn:
        years = days / 365.25
        return True, f"Chain extends {years:.1f} years into future (until {ports.format_local(until_dt)})"
    return True, None


def semantic_diff_value(old_text: str, new_text: str) -> str:
    return f"[dim]{old_text}[/] [cyan]→[/] [bold]{new_text}[/]"


__all__ = (
    "AnchorValidationPorts", "OmitValidationPorts", "anchor_validation_ports_for",
    "omit_validation_ports_for", "validate_anchor", "validate_omit", "validate_shared_anchor",
    "validate_shared_omit", "validate_cp", "validate_chain_limits",
    "validate_native_until", "validate_native_until_slots", "native_until_slot_ports_for", "until_not_past",
    "chain_duration_reasonable", "semantic_diff_value",
)
