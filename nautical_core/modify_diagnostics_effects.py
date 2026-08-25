"""Typed diagnostics and chain-summary effects for on-modify."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def chain_health_advice(host: Any, chain, kind: str, task, tol_secs: int, style: str):
    return host._module("modify_analytics").chain_health_advice(
        chain,
        kind,
        task,
        core=host.core,
        parse_datetime=host._dtparse,
        format_delta=host._fmt_td_dd_hhmm,
        coerce_int=host.core.coerce_int,
        tol_secs=tol_secs,
        style=style,
    )


def chain_integrity_warnings(host: Any, chain, expected_chain_id: str | None = None) -> list[str]:
    return host._module("modify_analytics").chain_integrity_warnings(
        chain,
        expected_chain_id=expected_chain_id,
        coerce_int=host.core.coerce_int,
        short=host._short,
    )


def lateness_stats(host: Any, chain, tol_secs: int = 60) -> dict:
    return host._module("modify_analytics").lateness_stats(
        chain, parse_datetime=host._dtparse, tol_secs=tol_secs
    )


def sort_chain_for_analytics(host: Any, chain):
    return host._module("modify_analytics").sort_chain_for_analytics(
        chain, coerce_int=host.core.coerce_int, parse_datetime=host._dtparse
    )


def last_n_timeline(host: Any, chain, n: int = 6) -> list[str]:
    return host._module("modify_chain_summary").last_n_timeline(
        chain,
        n,
        coerce_int=host.core.coerce_int,
        parse_datetime=host._dtparse,
        format_local=host._fmtlocal,
        format_on_time_delta=host._fmt_on_time_delta,
        short_uuid=host._short,
    )


def _fmt_secs_delta(host: Any, secs: float | None) -> str:
    if secs is None:
        return "—"
    base = datetime(2000, 1, 1, tzinfo=timezone.utc)
    target = base + timedelta(seconds=secs)
    value = (
        host.core.humanize_delta(base, target, use_months_days=False)
        .replace("in ", "")
        .replace("overdue by ", "")
    )
    if secs > 0:
        return f"[yellow]+{value}[/]"
    if secs < 0:
        return f"[cyan]-{value}[/]"
    return "[green]±0[/]"


def end_chain_summary(host: Any, current: dict, reason: str, now_utc, current_task: dict | None = None) -> None:
    summary = host._module("modify_chain_summary")

    def export_sorted_chain(chain_id: str, actual_current: dict) -> list:
        chain = host.tw_export_chain_required(actual_current)
        if actual_current and chain:
            for index, task in enumerate(chain):
                if task.get("uuid") == actual_current.get("uuid"):
                    chain[index] = host._module("task_models").TaskObservation.from_mapping(
                        actual_current, source_query=f"chain:{chain_id}:current"
                    )
                    break
        try:
            return sort_chain_for_analytics(host, chain)
        except Exception:
            return chain

    def span_fields(chain_id: str, chain: list[dict], *, stop_at=None, stopped_by_delete: bool = False):
        return summary.span_fields(
            chain_id,
            chain,
            stop_at=stop_at,
            stopped_by_delete=stopped_by_delete,
            export_endpoint=host._export_chain_endpoint,
            parse_datetime=host._dtparse,
            human_delta=host._human_delta,
        )

    def kind_rows(rows, kind: str, task: Any) -> None:
        summary.kind_rows(
            rows,
            kind,
            task,
            anchor_preset_display=host.core.anchor_preset_display,
            validate_anchor=host._validate_anchor_expr_cached,
            describe_anchor=host.core.describe_anchor_dnf,
        )

    def stats_rows(rows, chain, clock) -> None:
        summary.stats_rows(
            rows,
            chain,
            clock,
            lateness_stats=lambda value: lateness_stats(host, value),
            format_seconds_delta=lambda value: _fmt_secs_delta(host, value),
        )

    def limits_row(rows, task) -> None:
        summary.limits_row(
            rows,
            task,
            coerce_int=host.core.coerce_int,
            parse_datetime=host._dtparse,
            format_local=host.core.fmt_dt_local,
        )

    summary.render_chain_summary(
        current,
        reason,
        now_utc,
        current_task,
        export_sorted_chain=export_sorted_chain,
        root_uuid_from=host._root_uuid_from,
        short_uuid=host._short,
        format_root_and_age=host._format_root_and_age,
        kind_rows=kind_rows,
        span_fields=span_fields,
        stats_rows=stats_rows,
        limits_row=limits_row,
        last_n_timeline_rows=lambda chain: last_n_timeline(host, chain),
        format_rows=host._format_chain_summary_rows,
        coerce_int=host.core.coerce_int,
        format_local=host.core.fmt_dt_local,
        max_chain_walk=host._MAX_CHAIN_WALK,
        panel=host._panel,
        diagnostic=host._diag,
    )


__all__ = ("chain_health_advice", "chain_integrity_warnings", "end_chain_summary")
