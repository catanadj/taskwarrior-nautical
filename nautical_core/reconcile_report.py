"""Stable reconcile report serialization and terminal rendering."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import date, datetime, timezone
from typing import Any


def _json_default(value: object) -> str:
    """Encode temporal evidence without weakening the JSON report contract."""
    if isinstance(value, datetime):
        encoded = value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return encoded
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def render_json(summary: Mapping[str, Any]) -> str:
    """Serialize the versioned report without presentation-side mutations."""
    return json.dumps(dict(summary), ensure_ascii=False, indent=2, default=_json_default)


def render_human(summary: Mapping[str, Any], style: Callable[[str, str], str]) -> tuple[str, str]:
    """Return the stable summary and diagnostics lines for terminal output."""
    summary_line = (
        "summary: "
        f"{summary['mode']}; candidates={summary['candidates']} "
        f"spawn={summary['spawn']} backfill={summary['backfill_nextlink']} "
        f"expiration_hops={summary['expiration_hops']} recovered={summary['recovered_chains']} "
        f"final={summary['legitimate_final']} manual={summary['manual_stop']} "
        f"terminal={summary['terminal']} "
        f"stale={summary['stale']} partial={summary['partial']} errors={summary['errors']}"
        f" plan_errors={summary['plan_errors']}"
        f" native_until_errors={summary['native_until_error_count']}"
        f" native_until={len(summary['native_until_repairs'])}"
        f" manual_review={summary['native_until_manual_review']}"
        f" audit_skipped={summary['native_until_audit_skipped']}"
        f" config={summary['configuration_status']}"
        f" housekeeping={summary['housekeeping'].get('status', 'unknown')}"
    )
    degraded = summary.get("status") == "degraded"
    has_errors = summary.get("status") == "error"
    summary_color = "red" if has_errors else "yellow" if degraded else "green"
    diagnostics_line = (
        "diagnostics: "
        f"exports={summary['export_calls']} rows={summary['export_rows']} "
        f"export_s={summary['export_seconds']:.4f} "
        f"slowest_export_s={summary['slowest_export_seconds']:.4f} "
        f"snapshot_hits={summary['snapshot_hits']} "
        f"lock_busy={sum(summary['lock_contention'].values())}"
    )
    return style(summary_line, summary_color), style(diagnostics_line, "dim")


def exit_code(summary: Mapping[str, Any]) -> int:
    """Map the stable report status to the reconcile process contract."""
    status = str(summary.get("status") or "error")
    if status == "error":
        return 1
    if status == "degraded":
        return 2
    return 0


__all__ = ["exit_code", "render_human", "render_json"]
