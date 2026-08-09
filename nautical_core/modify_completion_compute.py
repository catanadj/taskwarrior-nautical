from __future__ import annotations

from datetime import datetime
from typing import Any

from nautical_core.modify_models import (
    CompletionComputeResult,
    CompletionComputeServices,
    ServiceCallback,
)
from nautical_core.timeutil import compare_datetimes


def completion_compute_child_due(
    new: dict[str, Any],
    kind: str,
    *,
    compute_anchor_child_due: ServiceCallback,
    compute_cp_child_due: ServiceCallback,
    panel: ServiceCallback,
    print_task: ServiceCallback,
    diag: ServiceCallback | None = None,
) -> tuple[Any, Any, Any] | None:
    try:
        if kind in {"anchor", "anchor_file"}:
            child_due, meta, dnf = compute_anchor_child_due(new)
            if kind == "anchor_file" and dnf is None:
                dnf = []
        else:
            child_due, meta = compute_cp_child_due(new)
            dnf = None
        return child_due, meta, dnf
    except ValueError as exc:
        panel(
            "⛔ Chain error",
            [("Reason", f"Invalid task field: {str(exc)}")],
            kind="error",
        )
        print_task(new)
        return None
    except Exception as exc:
        if callable(diag):
            diag(f"compute next due failed: {exc}")
        panel(
            "⛔ Chain error",
            [("Reason", "Could not compute next recurrence timestamp")],
            kind="error",
        )
        print_task(new)
        return None


def completion_until_or_fail(
    new: dict[str, Any],
    now_utc: Any,
    *,
    safe_parse_datetime: ServiceCallback,
    validate_until_not_past: ServiceCallback,
    panel: ServiceCallback,
    print_task: ServiceCallback,
) -> datetime | None | bool:
    until_dt, err = safe_parse_datetime(new.get("chainUntil"))
    if err:
        panel("⛔ Chain error", [("Reason", f"Invalid chainUntil: {err}")], kind="error")
        print_task(new)
        return False

    if until_dt:
        is_valid, err_msg = validate_until_not_past(until_dt, now_utc)
        if not is_valid:
            panel(
                "⛔ Chain error",
                [("Reason", f"Invalid chainUntil: {err_msg}")],
                kind="error",
            )
            print_task(new)
            return False
    return until_dt


def completion_until_guard_or_stop(
    new: dict[str, Any],
    child_due: Any,
    until_dt: Any,
    now_utc: Any,
    *,
    end_chain_summary: ServiceCallback,
    print_task: ServiceCallback,
) -> bool:
    if until_dt and compare_datetimes(child_due, until_dt) > 0:
        end_chain_summary(new, "Reached 'until' limit", now_utc)
        new["chain"] = "off"
        print_task(new)
        return False
    return True


def completion_require_child_due_or_fail(
    new: dict[str, Any],
    child_due: Any,
    *,
    panel: ServiceCallback,
    print_task: ServiceCallback,
) -> bool:
    if child_due:
        return True
    panel(
        "⛔ Chain error",
        [("Reason", "Could not compute next recurrence timestamp (no end date on parent)")],
        kind="error",
    )
    print_task(new)
    return False


def completion_warn_unreasonable_duration(
    new: dict[str, Any],
    child_due: Any,
    until_dt: Any,
    now_utc: Any,
    *,
    validate_chain_duration_reasonable: ServiceCallback,
    panel: ServiceCallback,
) -> None:
    if not until_dt:
        return
    is_reasonable, warn_msg = validate_chain_duration_reasonable(child_due, until_dt, now_utc)
    if warn_msg and not is_reasonable:
        panel("⚠ Chain duration warning", [("Warning", warn_msg)], kind="warning")


def completion_caps(
    kind: str,
    new: dict[str, Any],
    child_due: Any,
    dnf: Any,
    *,
    coerce_int: ServiceCallback,
    dtparse: ServiceCallback,
    estimate_cp_final_by_max: ServiceCallback,
    estimate_anchor_final_by_max: ServiceCallback,
    cap_from_until_cp: ServiceCallback,
    cap_from_until_anchor: ServiceCallback,
) -> tuple[int, datetime | None, int | None, list[tuple[str, Any]], int | None]:
    cpmax = coerce_int(new.get("chainMax"), 0)
    until_dt = dtparse(new.get("chainUntil"))
    cap_no = cpmax if cpmax else None
    finals = []

    if kind == "cp" and cpmax:
        try:
            fmax = estimate_cp_final_by_max(new, child_due)
            if fmax:
                finals.append(("max", fmax))
        except Exception:
            pass
    if kind in {"anchor", "anchor_file"} and cpmax:
        try:
            fmax = estimate_anchor_final_by_max(new, child_due, dnf)
            if fmax:
                finals.append(("max", fmax))
        except Exception:
            pass

    until_cap_no = None
    if until_dt:
        if kind == "cp":
            u_no, u_dt = cap_from_until_cp(new, child_due)
        else:
            u_no, u_dt = cap_from_until_anchor(new, child_due, dnf)
        if u_no:
            until_cap_no = u_no
            cap_no = min(cap_no, u_no) if cap_no else u_no
        if u_dt:
            finals.append(("until", u_dt))
    return cpmax, until_dt, cap_no, finals, until_cap_no


def completion_cap_guard_or_stop(
    new: dict[str, Any],
    next_no: int,
    cap_no: int | None,
    now_utc: Any,
    *,
    end_chain_summary: ServiceCallback,
    print_task: ServiceCallback,
) -> bool:
    if cap_no and next_no > cap_no:
        end_chain_summary(new, f"Reached cap #{cap_no}", now_utc, current_task=new)
        new["chain"] = "off"
        print_task(new)
        return False
    return True


def completion_compute_next_and_limits(
    new: dict[str, Any],
    kind: str,
    next_no: int,
    now_utc: Any,
    *,
    services: CompletionComputeServices,
) -> CompletionComputeResult | None:
    completion_compute_child_due = services.completion_compute_child_due
    completion_until_or_fail = services.completion_until_or_fail
    completion_until_guard_or_stop = services.completion_until_guard_or_stop
    completion_require_child_due_or_fail = services.completion_require_child_due_or_fail
    completion_warn_unreasonable_duration = services.completion_warn_unreasonable_duration
    completion_caps = services.completion_caps
    completion_cap_guard_or_stop = services.completion_cap_guard_or_stop
    computed = completion_compute_child_due(new, kind)
    if computed is None:
        return None
    child_due, meta, dnf = computed

    until_dt = completion_until_or_fail(new, now_utc)
    if until_dt is False:
        return None

    if not completion_until_guard_or_stop(new, child_due, until_dt, now_utc):
        return None

    if not completion_require_child_due_or_fail(new, child_due):
        return None

    completion_warn_unreasonable_duration(new, child_due, until_dt, now_utc)
    cpmax, until_dt, cap_no, finals, until_cap_no = completion_caps(kind, new, child_due, dnf)

    if not completion_cap_guard_or_stop(new, next_no, cap_no, now_utc):
        return None

    return CompletionComputeResult(
        child_due=child_due,
        meta=meta,
        dnf=dnf,
        until_dt=until_dt,
        cpmax=cpmax,
        cap_no=cap_no,
        finals=finals,
        until_cap_no=until_cap_no,
    )
