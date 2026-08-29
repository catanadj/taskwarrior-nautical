"""Stable reconcile report serialization and terminal rendering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .operator_presentation import bounded_text
from .operator_models import OperatorFailure, OperatorV2Result, OperatorV2Status
from .lifecycle_models import LifecycleAction, LifecycleEvent
from .lifecycle_recovery_models import RecoveryRefusal, RecoveryResult


_JSON_SCHEMA = "nautical.reconcile"
_JSON_SCHEMA_VERSION = 1


def _short_uuid(value: object) -> str:
    text = str(value or "").strip()
    return text[:8] if text else "?"


def _int_or_default(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _recurrence_kind(parent: object) -> str:
    values = parent.to_mapping() if hasattr(parent, "to_mapping") else {}
    if str(values.get("anchor") or "").strip():
        return "anchor"
    if str(values.get("anchor_file") or "").strip():
        return "anchor_file"
    return "cp"


def format_parent(parent: Mapping[str, Any]) -> str:
    """Format a parent task reference for reconcile's human renderer."""
    uuid = str(parent.get("uuid") or "").strip()[:8] or "????????"
    chain_id = str(parent.get("chainID") or "?")
    try:
        link = int(parent.get("link") or 0)
    except (TypeError, ValueError):
        link = 0
    description = str(parent.get("description") or "").strip()
    return f"{uuid} chain {chain_id} link {link}" + (f" · {description}" if description else "")


def describe_recovery_result(result: RecoveryResult, *, fmt_dt_local: Any = None) -> dict[str, Any]:
    """Render typed recovery evidence for reconcile output."""
    parent = result.parent.to_mapping()
    if isinstance(result, RecoveryRefusal):
        return {
            "parent": _short_uuid(parent.get("uuid")),
            "chainID": str(parent.get("chainID") or ""),
            "parent_link": _int_or_default(parent.get("link")),
            "kind": _recurrence_kind(result.parent),
            "reason": result.reason,
            "status": result.status.value,
            **dict(result.evidence),
        }
    plan = result.plan
    evidence: dict[str, Any] = {
        "parent": _short_uuid(parent.get("uuid")),
        "chainID": str(parent.get("chainID") or ""),
        "parent_link": _int_or_default(parent.get("link")),
        "next_link": plan.identity.target_link,
        "kind": _recurrence_kind(result.parent),
        "trigger": "expiration" if plan.identity.event is LifecycleEvent.EXPIRE else "completion",
        "reason": result.reason,
        "action": plan.action.value,
    }
    if result.terminal_kind:
        evidence["terminal"] = True
        evidence["terminal_kind"] = result.terminal_kind
    if result.child_due is not None:
        evidence["child_due"] = str(result.child_due)
        if callable(fmt_dt_local):
            try:
                evidence["child_local"] = str(fmt_dt_local(result.child_due))
            except Exception:
                pass
    if result.child_short:
        evidence["existing_child"] = result.child_short
    if plan.action is LifecycleAction.SPAWN_CHILD:
        child = plan.child_dict()
        field = "scheduled" if child.get("scheduled") and not child.get("due") else "due"
        evidence["child_field"] = field
        evidence["child_target"] = str(child.get(field) or "")
    return evidence


def recovery_action(result: RecoveryResult) -> str:
    """Project a typed recovery result to the reconcile action vocabulary."""
    if isinstance(result, RecoveryRefusal):
        return result.status.value
    return {
        LifecycleAction.SPAWN_CHILD: "spawn",
        LifecycleAction.UPDATE_PARENT: "backfill_nextlink",
        LifecycleAction.FINALIZE_CHAIN: "legitimate_final",
        LifecycleAction.DISABLE_CHAIN: "manual_stop",
    }.get(result.plan.action, result.plan.action.value)


class ReconcileReport(dict[str, Any]):
    """Validated public reconcile document, including startup reports."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        if not isinstance(document, Mapping):
            raise ValueError("reconcile report must be an object")
        if document.get("schema") != _JSON_SCHEMA:
            raise ValueError("invalid reconcile report schema")
        if document.get("schema_version") != _JSON_SCHEMA_VERSION:
            raise ValueError("unsupported reconcile report version")
        if str(document.get("status") or "") not in {"ok", "degraded", "error"}:
            raise ValueError("invalid reconcile report status")
        if str(document.get("mode") or "") not in {"dry-run", "apply"}:
            raise ValueError("invalid reconcile report mode")
        super().__init__(document)

    @classmethod
    def from_mapping(cls, value: object) -> "ReconcileReport":
        if not isinstance(value, Mapping):
            raise ValueError("reconcile report must be an object")
        return cls(value)

    def to_dict(self) -> dict[str, Any]:
        return dict(self)


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


__all__ = ["ReconcileReport", "describe_recovery_result", "exit_code", "format_parent", "recovery_action", "render_human", "to_operator_result"]
