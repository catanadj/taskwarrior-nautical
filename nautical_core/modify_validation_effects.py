"""Validation effects for ordinary on-modify recurrence edits."""

from __future__ import annotations

import re
from typing import Any


def anchor_error_message(anchor_expr: str, default_msg: str) -> str:
    if re.search(r"(?:^|[^A-Za-z])(w|m|y)(?:/\d+)?:", anchor_expr, re.IGNORECASE):
        return default_msg
    return f"{default_msg} (expected an anchor such as w:mon, m:15, or y:jul)"


def anchor_mode(host: Any, old: Any, new: Any) -> str:
    raw = str(new.get("anchor_mode") or old.get("anchor_mode") or "skip").strip()
    mode = raw.lower()
    aliases = {"all": "all", "skip": "skip", "flex": "flex"}
    normalized = aliases.get(mode)
    if normalized is None:
        host._module("modify_ui_effects").panel(
            host, "⚠ Anchor mode", [("Warning", f"Unknown anchor mode {raw!r}; using skip.")], kind="warning"
        )
        normalized = "skip"
        new["anchor_mode"] = normalized
    elif new.get("anchor_mode"):
        new["anchor_mode"] = normalized
    return normalized.upper()


def validate_anchor(host: Any, old: Any, new: Any, anchor_expr: str) -> None:
    try:
        _, warns = host.core.lint_anchor_expr(anchor_expr)
        if warns:
            host._module("modify_ui_effects").panel(
                host, "ℹ️  Lint", [("Hint", warning) for warning in warns], kind="note"
            )
        anchor_mode(host, old, new)
        # Validation must remain decision-only. Hint persistence has no
        # synchronous consumer and would repeat scheduler work on every edit.
        host.core.validate_anchor_expr_strict(anchor_expr)
    except TypeError:
        host.core.validate_anchor_expr_strict(anchor_expr)
    except Exception as exc:
        astronomy = host.core._import_sibling("astronomy")
        if astronomy.is_astronomy_error(exc):
            host._fail_and_exit("Invalid anchor", astronomy.scheduling_error_message(exc))
        host._fail_and_exit("Invalid anchor", anchor_error_message(anchor_expr, str(exc)))


def validate_omit(host: Any, anchor_expr: str, anchor_file_expr: str, omit_expr: str, omit_file: str) -> None:
    try:
        validate_shared_omit(host, omit_expr)
        findings = host.core._import_sibling("hook_validation_pipeline").validate_recurrence_files(
            anchor_expr,
            anchor_file_expr,
            omit_expr,
            omit_file,
            load_anchor_file=host._load_anchor_file_dates,
            load_omit_file=host._load_omit_file_dates,
        )
    except Exception as exc:
        host._fail_and_exit("Invalid omit", str(exc))
        return
    if findings:
        finding = findings[0]
        host._fail_and_exit(f"Invalid {finding.field}", finding.reason)


def validate_shared_anchor(host: Any, expr: str) -> None:
    pipeline = host.core._import_sibling("hook_validation_pipeline")
    pipeline.validate_anchor_expression(
        expr,
        parse_anchor_expr=host.core.parse_anchor_expr_to_dnf,
        validate_anchor_expr=host._validate_anchor_expr_cached,
    )


def validate_shared_omit(host: Any, expr: str) -> None:
    pipeline = host.core._import_sibling("hook_validation_pipeline")
    pipeline.validate_omit_expression(
        expr,
        validate_omit_expr=host._validate_omit_expr_cached,
    )


def validate_cp(host: Any, cp_value: str, chain_max_value: Any, chain_until_value: Any) -> None:
    add_validation = host.core._import_sibling("add_validation")
    host._module("modify_validation").validate_cp_on_modify(
        cp_value,
        chain_max_value,
        chain_until_value,
        parse_cp_sequence=host.core.parse_cp_sequence,
        cp_sequence_parse_error=host.core.cp_sequence_parse_error,
        parse_chain_max=add_validation.parse_chain_max,
        parse_datetime=host.core.parse_dt_any,
    )


def validate_chain_limits(host: Any, task: dict) -> None:
    add_validation = host.core._import_sibling("add_validation")
    pipeline = host.core._import_sibling("hook_validation_pipeline")
    cpmax, _until_dt, findings = pipeline.validate_recurrence_limits(
        task.get("cp"), task.get("chainMax"), task.get("chainUntil"),
        parse_cp_sequence=host.core.parse_cp_sequence,
        cp_sequence_parse_error=host.core.cp_sequence_parse_error,
        parse_chain_max=add_validation.parse_chain_max,
        parse_datetime=host.core.parse_dt_any,
    )
    if findings:
        finding = findings[0]
        host._fail_and_exit(f"Invalid {finding.field}", finding.reason)
    if cpmax is not None:
        task["chainMax"] = cpmax
    return host._module("modify_validation").validate_chain_limits_on_modify(
        task,
        parse_chain_max=add_validation.parse_chain_max,
        parse_datetime=host.core.parse_dt_any,
        validate_until_not_past=lambda until_dt, now: until_not_past(host, until_dt, now),
        now_utc=host.core.now_utc,
        fail=host._fail_and_exit,
    )


def validate_native_until(host: Any, task: dict) -> None:
    add_validation = host.core._import_sibling("add_validation")
    host._module("modify_validation").validate_native_until_after_target_or_fail(
        task,
        validate_anchor_mode=add_validation.validate_native_until_anchor_mode,
        safe_parse_datetime=lambda value: host._module("modify_datetime_effects").safe_parse_datetime(host, value),
        validate_after_target=add_validation.validate_native_until_after_target,
        format_local=host.core.fmt_dt_local,
        panel=lambda title, rows, **kwargs: host._module("modify_ui_effects").panel(host, title, rows, **kwargs),
        fail=host._fail_and_exit,
        abort=host.sys.exit,
    )


def validate_native_until_slots(host: Any, task: dict) -> None:
    add_validation = host.core._import_sibling("add_validation")
    astronomy = host.core._import_sibling("astronomy")
    native_until = host.core._import_sibling("native_until")
    recurrence_context = host.core._import_sibling("recurrence_context").RecurrenceContext
    host._module("modify_validation").validate_native_until_anchor_slots_or_fail(
        task,
        safe_parse_datetime=lambda value: host._module("modify_datetime_effects").safe_parse_datetime(host, value),
        validate_anchor=host._validate_anchor_expr_cached,
        collect_time_slots=add_validation.collect_anchor_time_slots,
        validate_time_slots=native_until.validate_calendar_slots,
        normalize_time_slots=lambda value, target_date=None: host._module("modify_time_effects").normalize_hhmm_list(host, value, target_date),
        anchor_file_dir=getattr(host.core, "ANCHOR_FILE_DIR", ""),
        recurrence_context=recurrence_context.from_task,
        to_local=host._tolocal,
        format_local=host.core.fmt_dt_local,
        astronomy_is_error=astronomy.is_astronomy_error,
        astronomy_error_message=astronomy.scheduling_error_message,
        panel=lambda title, rows, **kwargs: host._module("modify_ui_effects").panel(host, title, rows, **kwargs),
        abort=host.sys.exit,
    )


def until_not_past(host: Any, until_dt, now_utc) -> tuple[bool, str | None]:
    if not until_dt:
        return True, None
    grace = host.timedelta(minutes=1)
    if host._module("modify_value_effects").compare_datetimes(host, until_dt, now_utc - grace) < 0:
        past_s = host.core.humanize_delta(until_dt, now_utc, use_months_days=False)
        return False, f"chainUntil is in the past (was {past_s} ago)"
    return True, None


def chain_duration_reasonable(host: Any, child_due, until_dt, now_utc) -> tuple[bool, str | None]:
    if not until_dt:
        return True, None
    days = (until_dt - now_utc).days
    if days > host._MIN_FUTURE_WARN:
        years = days / 365.25
        return True, f"Chain extends {years:.1f} years into future (until {host.core.fmt_dt_local(until_dt)})"
    return True, None


def semantic_diff_value(old_text: str, new_text: str) -> str:
    return f"[dim]{old_text}[/] [cyan]→[/] [bold]{new_text}[/]"


__all__ = (
    "validate_anchor", "validate_omit", "validate_shared_anchor",
    "validate_shared_omit", "validate_cp", "validate_chain_limits",
    "validate_native_until", "validate_native_until_slots", "until_not_past",
    "chain_duration_reasonable", "semantic_diff_value",
)
