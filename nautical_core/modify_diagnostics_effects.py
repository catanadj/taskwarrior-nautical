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


def export_chain_endpoint(host: Any, chain_id: str, direction: str):
    """Return a chain endpoint from the invocation's authoritative snapshot."""
    rows = host._module("modify_read_effects").lifecycle_read_service(host).get_chain_export(chain_id)
    if rows is None:
        raise RuntimeError(f"Chain export unavailable for chainID {chain_id}")
    with_links = [
        (host.core.coerce_int(row.get("link"), None), row)
        for row in rows
    ]
    with_links = [(link, row) for link, row in with_links if link is not None]
    if not with_links:
        return None
    with_links.sort(key=lambda item: item[0])
    return with_links[0 if direction == "first" else -1][1]


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


def span_fields(host: Any, chain_id: str, chain, *, stop_at=None, stopped_by_delete: bool = False):
    return host._module("modify_chain_summary").span_fields(
        chain_id, chain, stop_at=stop_at, stopped_by_delete=stopped_by_delete,
        export_endpoint=lambda chain_id, direction: export_chain_endpoint(host, chain_id, direction),
        parse_datetime=host._dtparse,
        human_delta=host._human_delta,
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
            export_endpoint=lambda chain_id, direction: export_chain_endpoint(host, chain_id, direction),
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
        format_root_and_age=lambda task, now: host._module("modify_queries").cached_format_root_and_age(host, task, now),
        kind_rows=kind_rows,
        span_fields=span_fields,
        stats_rows=stats_rows,
        limits_row=limits_row,
        last_n_timeline_rows=lambda chain: last_n_timeline(host, chain),
        format_rows=host._module("modify_feedback").format_chain_summary_rows,
        coerce_int=host.core.coerce_int,
        format_local=host.core.fmt_dt_local,
        max_chain_walk=host._MAX_CHAIN_WALK,
        panel=host._panel,
        diagnostic=host._diag,
    )


__all__ = ("chain_health_advice", "chain_integrity_warnings", "span_fields", "end_chain_summary")
