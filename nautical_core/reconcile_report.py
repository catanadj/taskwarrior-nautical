"""Stable reconcile report serialization and terminal rendering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .operator_presentation import bounded_text
from .operator_models import OperatorFailure, OperatorV2Result, OperatorV2Status


_JSON_SCHEMA = "nautical.reconcile"


def to_operator_result(summary: Mapping[str, Any]) -> OperatorV2Result:
    """Convert one reconcile summary into the shared operator envelope."""
    status = OperatorV2Status(str(summary.get("status") or "error"))
    failure = None
    if status in {OperatorV2Status.ERROR, OperatorV2Status.UNAVAILABLE, OperatorV2Status.INVALID}:
        reason = str(summary.get("configuration_drift") or "")
        if not reason:
            errors = summary.get("errors") or summary.get("native_until_errors") or ()
            reason = str(errors[0] if isinstance(errors, (list, tuple)) and errors else "reconcile reported an error")
        failure = OperatorFailure(code="reconcile_error", message=reason)
    return OperatorV2Result(
        schema=_JSON_SCHEMA,
        operation="reconcile",
        status=status,
        payload={key: value for key, value in summary.items() if key not in {"schema", "status"}},
        failure=failure,
    )


def _count(summary: Mapping[str, Any], key: str) -> int:
    try:
        return max(0, int(summary.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _style(style: Callable[[str, str], str], text: str, color: str) -> str:
    try:
        return str(style(text, color))
    except Exception:
        return text


def render_human(summary: Mapping[str, Any], style: Callable[[str, str], str]) -> tuple[str, str]:
    """Return the stable summary and diagnostics lines for terminal output."""
    summary_line = (
        "summary: "
        f"{bounded_text(summary.get('mode') or 'unknown')}; candidates={_count(summary, 'candidates')} "
        f"spawn={_count(summary, 'spawn')} backfill={_count(summary, 'backfill_nextlink')} "
        f"expiration_hops={_count(summary, 'expiration_hops')} recovered={_count(summary, 'recovered_chains')} "
        f"final={_count(summary, 'legitimate_final')} manual={_count(summary, 'manual_stop')} "
        f"terminal={_count(summary, 'terminal')} stale={_count(summary, 'stale')} partial={_count(summary, 'partial')} errors={_count(summary, 'errors')}"
        f" plan_errors={_count(summary, 'plan_errors')}"
        f" native_until_errors={_count(summary, 'native_until_error_count')}"
        f" native_until={len(summary.get('native_until_repairs') or ())}"
        f" manual_review={_count(summary, 'native_until_manual_review')}"
        f" audit_skipped={_count(summary, 'native_until_audit_skipped')}"
        f" config={bounded_text(summary.get('configuration_status') or 'unknown')}"
        f" housekeeping={(summary.get('housekeeping') or {}).get('status', 'unknown')}"
    )
    degraded = summary.get("status") == "degraded"
    has_errors = summary.get("status") == "error"
    summary_color = "red" if has_errors else "yellow" if degraded else "green"
    diagnostics_line = (
        "diagnostics: "
        f"exports={_count(summary, 'export_calls')} rows={_count(summary, 'export_rows')} "
        f"export_s={float(summary.get('export_seconds', 0.0) or 0.0):.4f} "
        f"slowest_export_s={float(summary.get('slowest_export_seconds', 0.0) or 0.0):.4f} "
        f"snapshot_hits={_count(summary, 'snapshot_hits')} "
        f"lock_busy={sum((summary.get('lock_contention') or {}).values())}"
    )
    return _style(style, summary_line, summary_color), _style(style, diagnostics_line, "dim")


def exit_code(summary: Mapping[str, Any]) -> int:
    """Map the stable report status to the reconcile process contract."""
    status = str(summary.get("status") or "error")
    if status == "error":
        return 1
    if status == "degraded":
        return 2
    return 0


__all__ = ["exit_code", "render_human", "to_operator_result"]
