"""Typed diagnostics and chain-summary effects for on-modify."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from .task_datetime import datetime_value, parser_for_host
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatetimeValuePort:
    parser: Any


@dataclass(frozen=True, slots=True)
class AnalyticsPorts:
    core: Any
    parse_datetime: Any
    format_delta: Any
    coerce_int: Any
    short_uuid: Any


@dataclass(frozen=True, slots=True)
class ChainExportPorts:
    service: Any
    coerce_int: Any


@dataclass(frozen=True, slots=True)
class TimelineSummaryPorts:
    summary: Any
    coerce_int: Any
    parse_datetime: Any
    format_local: Any
    format_on_time_delta: Any
    short_uuid: Any


@dataclass(frozen=True, slots=True)
class SpanFieldsPorts:
    summary: Any
    export_endpoint: Any
    parse_datetime: Any
    human_delta: Any


@dataclass(frozen=True, slots=True)
class SecondsDeltaPort:
    humanize: Any


@dataclass(frozen=True, slots=True)
class EndChainSummaryPorts:
    summary: Any
    services: Any


def _parse_datetime_value(port: DatetimeValuePort, value: object):
    return datetime_value(port.parser, value)


def chain_health_advice(ports: AnalyticsPorts, chain, kind: str, task, tol_secs: int, style: str):
    return ports.core._import_sibling("modify_analytics").chain_health_advice(
        chain,
        kind,
        task,
        core=ports.core,
        parse_datetime=ports.parse_datetime,
        format_delta=ports.format_delta,
        coerce_int=ports.coerce_int,
        tol_secs=tol_secs,
        style=style,
    )


def chain_integrity_warnings(ports: AnalyticsPorts, chain, expected_chain_id: str | None = None) -> list[str]:
    return ports.core._import_sibling("modify_analytics").chain_integrity_warnings(
        chain,
        expected_chain_id=expected_chain_id,
        coerce_int=ports.coerce_int,
        short=ports.short_uuid,
    )


def analytics_ports_for(host: Any) -> AnalyticsPorts:
    return AnalyticsPorts(
        core=host.core,
        parse_datetime=lambda value: _parse_datetime_value(DatetimeValuePort(parser_for_host(host)), value),
        format_delta=host._module("modify_value_effects").format_delta,
        coerce_int=host.core.coerce_int,
        short_uuid=host.core.short_uuid,
    )


def lateness_stats(ports: AnalyticsPorts, chain, tol_secs: int = 60) -> dict:
    return ports.core._import_sibling("modify_analytics").lateness_stats(
        chain, parse_datetime=ports.parse_datetime, tol_secs=tol_secs
    )


def sort_chain_for_analytics(ports: AnalyticsPorts, chain):
    return ports.core._import_sibling("modify_analytics").sort_chain_for_analytics(
        chain, coerce_int=ports.coerce_int, parse_datetime=ports.parse_datetime
    )


def chain_export_ports_for(host: Any) -> ChainExportPorts:
    return ChainExportPorts(
        service=host._module("modify_composition").lifecycle_read_service_for(host),
        coerce_int=host.core.coerce_int,
    )


def export_chain_endpoint(ports: ChainExportPorts, chain_id: str, direction: str):
    """Return a chain endpoint from the invocation's authoritative snapshot."""
    rows = ports.service.get_chain_export(chain_id)
    if rows is None:
        raise RuntimeError(f"Chain export unavailable for chainID {chain_id}")
    with_links = [
        (ports.coerce_int(row.get("link"), None), row)
        for row in rows
    ]
    with_links = [(link, row) for link, row in with_links if link is not None]
    if not with_links:
        return None
    with_links.sort(key=lambda item: item[0])
    return with_links[0 if direction == "first" else -1][1]


def timeline_summary_ports_for(host: Any) -> TimelineSummaryPorts:
    formatting = host._module("modify_format_effects")
    return TimelineSummaryPorts(
        summary=host._module("modify_chain_summary"),
        coerce_int=host.core.coerce_int,
        parse_datetime=lambda value: _parse_datetime_value(DatetimeValuePort(parser_for_host(host)), value),
        format_local=host._fmtlocal,
        format_on_time_delta=lambda due, end, tol=60: formatting.on_time_delta(
            formatting.HumanDeltaPort(host.core.humanize_delta), due, end, tol
        ),
        short_uuid=host.core.short_uuid,
    )


def last_n_timeline(ports: TimelineSummaryPorts, chain, n: int = 6) -> list[str]:
    return ports.summary.last_n_timeline(
        chain,
        n,
        coerce_int=ports.coerce_int,
        parse_datetime=ports.parse_datetime,
        format_local=ports.format_local,
        format_on_time_delta=ports.format_on_time_delta,
        short_uuid=ports.short_uuid,
    )


def span_fields_ports_for(host: Any) -> SpanFieldsPorts:
    formatting = host._module("modify_format_effects")
    export_ports = chain_export_ports_for(host)
    return SpanFieldsPorts(
        summary=host._module("modify_chain_summary"),
        export_endpoint=lambda chain_id, direction: export_chain_endpoint(export_ports, chain_id, direction),
        parse_datetime=lambda value: _parse_datetime_value(DatetimeValuePort(parser_for_host(host)), value),
        human_delta=lambda start, end, prefer=True, *, prefer_months=None: formatting.human_delta(
            formatting.HumanDeltaPort(host.core.humanize_delta),
            start, end, prefer if prefer_months is None else prefer_months,
        ),
    )


def span_fields(ports: SpanFieldsPorts, chain_id: str, chain, *, stop_at=None, stopped_by_delete: bool = False):
    return ports.summary.span_fields(
        chain_id, chain, stop_at=stop_at, stopped_by_delete=stopped_by_delete,
        export_endpoint=ports.export_endpoint,
        parse_datetime=ports.parse_datetime,
        human_delta=ports.human_delta,
    )


def seconds_delta_port_for(host: Any) -> SecondsDeltaPort:
    return SecondsDeltaPort(host.core.humanize_delta)


def format_seconds_delta(port: SecondsDeltaPort, secs: float | None) -> str:
    if secs is None:
        return "—"
    base = datetime(2000, 1, 1, tzinfo=timezone.utc)
    target = base + timedelta(seconds=secs)
    value = (
        port.humanize(base, target, use_months_days=False)
        .replace("in ", "")
        .replace("overdue by ", "")
    )
    if secs > 0:
        return f"[yellow]+{value}[/]"
    if secs < 0:
        return f"[cyan]-{value}[/]"
    return "[green]±0[/]"


def end_chain_summary_ports_for(host: Any) -> EndChainSummaryPorts:
    summary = host._module("modify_chain_summary")
    analytics_ports = analytics_ports_for(host)
    span_ports = span_fields_ports_for(host)
    timeline_ports = timeline_summary_ports_for(host)
    seconds_port = seconds_delta_port_for(host)
    read_effects = host._module("modify_read_effects")
    chain_export_port = read_effects.ChainExportPort(
        host._module("modify_composition").lifecycle_read_service_for(host)
    )
    task_observation = host._module("task_models").TaskObservation
    anchor_preset_display = host.core.anchor_preset_display
    validate_anchor = host._validate_anchor_expr_cached
    describe_anchor = host.core.describe_anchor_dnf
    coerce_int = host.core.coerce_int
    parse_datetime = lambda value: _parse_datetime_value(DatetimeValuePort(parser_for_host(host)), value)
    format_local = host.core.fmt_dt_local
    root_uuid_from = host._module("modify_task_fields").root_uuid
    queries = host._module("modify_queries")
    query_ports = queries.query_ports_for(host)
    format_rows = host._module("modify_feedback").format_chain_summary_rows
    short_uuid = host.core.short_uuid
    max_chain_walk = host._MAX_CHAIN_WALK
    diagnostic = host._diag

    def export_sorted_chain(chain_id: str, actual_current: dict) -> list:
        chain = read_effects.export_chain_required(chain_export_port, actual_current)
        if actual_current and chain:
            for index, task in enumerate(chain):
                if task.get("uuid") == actual_current.get("uuid"):
                    chain[index] = task_observation.from_mapping(
                        actual_current, source_query=f"chain:{chain_id}:current"
                    )
                    break
        try:
            return sort_chain_for_analytics(analytics_ports, chain)
        except Exception:
            return chain

    def render_span_fields(chain_id: str, chain: list[dict], *, stop_at=None, stopped_by_delete: bool = False):
        return span_fields(
            span_ports, chain_id, chain, stop_at=stop_at, stopped_by_delete=stopped_by_delete
        )

    def kind_rows(rows, kind: str, task: Any) -> None:
        summary.kind_rows(
            rows,
            kind,
            task,
            anchor_preset_display=anchor_preset_display,
            validate_anchor=validate_anchor,
            describe_anchor=describe_anchor,
        )

    def stats_rows(rows, chain, clock) -> None:
        summary.stats_rows(
            rows,
            chain,
            clock,
            lateness_stats=lambda value: lateness_stats(analytics_ports, value),
            format_seconds_delta=lambda _now, value: format_seconds_delta(seconds_port, value),
        )

    def limits_row(rows, task) -> None:
        summary.limits_row(
            rows,
            task,
            coerce_int=coerce_int,
            parse_datetime=parse_datetime,
            format_local=format_local,
        )

    ui = host._module("modify_ui_effects")
    ui_ports = ui.ui_ports_for(host)
    render_services = summary.ChainSummaryRenderServices(
        export_sorted_chain=export_sorted_chain,
        root_uuid_from=root_uuid_from,
        short_uuid=short_uuid,
        format_root_and_age=lambda task, now: queries.cached_format_root_and_age(query_ports, task, now),
        kind_rows=kind_rows,
        span_fields=render_span_fields,
        stats_rows=stats_rows,
        limits_row=limits_row,
        last_n_timeline_rows=lambda chain, n=6: last_n_timeline(timeline_ports, chain, n),
        format_rows=format_rows,
        coerce_int=coerce_int,
        format_local=format_local,
        max_chain_walk=max_chain_walk,
        panel=lambda title, rows, **kwargs: ui.panel(ui_ports, title, rows, **kwargs),
        diagnostic=diagnostic,
    )
    return EndChainSummaryPorts(summary=summary, services=render_services)


def end_chain_summary(ports: EndChainSummaryPorts, current: dict, reason: str, now_utc, current_task: dict | None = None) -> None:
    ports.summary.render_chain_summary(
        current,
        reason,
        now_utc,
        current_task,
        services=ports.services,
    )


__all__ = (
    "AnalyticsPorts", "analytics_ports_for", "lateness_stats", "sort_chain_for_analytics",
    "ChainExportPorts", "chain_export_ports_for", "export_chain_endpoint",
    "TimelineSummaryPorts", "timeline_summary_ports_for", "last_n_timeline",
    "SpanFieldsPorts", "span_fields_ports_for", "span_fields",
    "SecondsDeltaPort", "seconds_delta_port_for", "format_seconds_delta",
    "EndChainSummaryPorts", "end_chain_summary_ports_for", "end_chain_summary",
    "chain_health_advice", "chain_integrity_warnings",
)
