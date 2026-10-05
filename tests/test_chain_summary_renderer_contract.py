from __future__ import annotations

import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any, get_type_hints

from nautical_core.modify_chain_summary import (
    ChainSummaryRenderServices,
    SpanFieldsCallback,
    SummaryKindRows,
    SummaryLimitsRow,
    SummaryRowsFormatter,
    SummaryStatsRows,
    SummaryTimelineRows,
    kind_rows,
    render_chain_summary_with_services,
    render_chain_summary,
)
from nautical_core.parsing.parser_models import AnchorDNF, ParseError
from nautical_core.task_models import TaskObservation, TaskPayload


class ChainSummaryRendererContractTests(unittest.TestCase):
    def test_summary_timeline_callbacks_use_validated_value_types(self) -> None:
        from nautical_core.modify_chain_summary import last_n_timeline

        hints = get_type_hints(last_n_timeline)
        self.assertEqual(
            {
                name: hints[name]
                for name in (
                    "coerce_int",
                    "parse_datetime",
                    "format_local",
                    "format_on_time_delta",
                    "short_uuid",
                )
            },
            {
                "coerce_int": Callable[[object, int], int | None],
                "parse_datetime": Callable[[object], datetime | None],
                "format_local": Callable[[datetime], str],
                "format_on_time_delta": Callable[
                    [datetime | None, datetime | None], str
                ],
                "short_uuid": Callable[[str | None], str],
            },
        )

    def test_kind_rows_keeps_pattern_when_anchor_syntax_is_invalid(self) -> None:
        rows: list[tuple[str, str]] = []

        def reject_invalid(_expression: str) -> object:
            raise ParseError("invalid anchor")

        kind_rows(
            rows,
            "anchor",
            {"anchor": "malformed", "anchor_mode": "skip"},
            anchor_preset_display=lambda _expression: None,
            validate_anchor=reject_invalid,
            describe_anchor=lambda _dnf, _task: self.fail("invalid anchors have no natural row"),
        )

        self.assertEqual(rows, [("Pattern", "malformed  [cyan]SKIP[/]")])

    def test_kind_rows_surfaces_unexpected_preset_display_failure(self) -> None:
        def broken_preset(_expression: str) -> tuple[str, str] | None:
            raise RuntimeError("preset renderer defect")

        with self.assertRaisesRegex(RuntimeError, "preset renderer defect"):
            kind_rows(
                [],
                "anchor",
                {"anchor": "@daily"},
                anchor_preset_display=broken_preset,
                validate_anchor=lambda _expression: [],
                describe_anchor=lambda _dnf, _task: "daily",
            )

    def test_kind_rows_surfaces_unexpected_natural_description_failure(self) -> None:
        def broken_description(_dnf: object, _task: dict) -> str:
            raise RuntimeError("anchor description defect")

        with self.assertRaisesRegex(RuntimeError, "anchor description defect"):
            kind_rows(
                [],
                "anchor",
                {"anchor": "w:mon"},
                anchor_preset_display=lambda _expression: None,
                validate_anchor=lambda _expression: [[{"type": "weekday", "value": "mon"}]],
                describe_anchor=broken_description,
            )

    def test_summary_retains_primary_rows_when_optional_chain_read_fails(self) -> None:
        rendered: list[tuple[str, str]] = []
        diagnostics: list[str] = []

        def fail_chain_export(_chain_id: str, _task: TaskPayload) -> list[TaskObservation]:
            raise RuntimeError("repository offline")

        services = ChainSummaryRenderServices(
            export_sorted_chain=fail_chain_export,
            root_uuid_from=lambda task: str(task.get("chainID") or ""),
            short_uuid=lambda value: str(value or "")[:8],
            format_root_and_age=lambda _task, _now: "root",
            kind_rows=lambda _rows, _kind, _task: None,
            span_fields=lambda _chain_id, _chain, **_options: (None, None, "–"),
            stats_rows=lambda _rows, _chain: None,
            limits_row=lambda _rows, _task: None,
            last_n_timeline_rows=lambda _chain, _count=6: [],
            format_rows=lambda rows: rows,
            coerce_int=lambda value, default: int(value) if value is not None else default,
            format_local=lambda value: value.isoformat(),
            max_chain_walk=10,
            panel=lambda _title, rows, **_options: rendered.extend(rows),
            diagnostic=diagnostics.append,
        )

        render_chain_summary(
            {"uuid": "task-1234", "chainID": "chain-1234", "link": 1},
            "Task completed.",
            datetime(2026, 10, 4, tzinfo=timezone.utc),
            services=services,
        )

        self.assertIn(("Reason", "Task completed."), rendered)
        self.assertIn(("Chain read", "Unavailable: repository offline"), rendered)
        self.assertEqual(
            diagnostics,
            ["chain summary export unavailable (chainID=chain-1234): repository offline"],
        )

    def test_render_service_span_callback_has_explicit_contract(self) -> None:
        self.assertIs(
            get_type_hints(ChainSummaryRenderServices)["span_fields"],
            SpanFieldsCallback,
        )

    def test_render_service_row_callbacks_have_explicit_contracts(self) -> None:
        annotations = get_type_hints(ChainSummaryRenderServices)
        self.assertEqual(
            {name: annotations[name] for name in (
                "kind_rows", "stats_rows", "limits_row",
                "last_n_timeline_rows", "format_rows",
            )},
            {
                "kind_rows": SummaryKindRows,
                "stats_rows": SummaryStatsRows,
                "limits_row": SummaryLimitsRow,
                "last_n_timeline_rows": SummaryTimelineRows,
                "format_rows": SummaryRowsFormatter,
            },
        )

    def test_render_service_task_and_time_ports_use_domain_types(self) -> None:
        annotations = get_type_hints(ChainSummaryRenderServices)
        self.assertEqual(
            {name: annotations[name] for name in (
                "export_sorted_chain", "root_uuid_from", "short_uuid",
                "format_root_and_age", "format_local",
            )},
            {
                "export_sorted_chain": Callable[[str, TaskPayload], list[TaskObservation]],
                "root_uuid_from": Callable[[TaskPayload], str],
                "short_uuid": Callable[[str | None], str],
                "format_root_and_age": Callable[[TaskPayload, datetime], str],
                "format_local": Callable[[datetime], str],
            },
        )

    def test_kind_rows_uses_parsed_anchor_contract(self) -> None:
        annotations = get_type_hints(kind_rows)
        self.assertEqual(
            annotations["validate_anchor"],
            Callable[[str], AnchorDNF],
        )
        self.assertEqual(
            annotations["describe_anchor"],
            Callable[[AnchorDNF, dict[str, Any]], str],
        )

    def test_summary_renderers_accept_domain_task_and_time_types(self) -> None:
        expected = {
            "current": TaskPayload,
            "now_utc": datetime,
            "current_task": TaskPayload | None,
        }
        for renderer in (render_chain_summary, render_chain_summary_with_services):
            with self.subTest(renderer=renderer.__name__):
                annotations = get_type_hints(renderer)
                self.assertEqual(
                    {name: annotations[name] for name in expected},
                    expected,
                )

    def test_stats_rows_format_lateness_without_an_unused_clock_argument(self) -> None:
        from nautical_core.modify_analytics import LatenessStats
        from nautical_core.modify_chain_summary import stats_rows

        rows: list[tuple[str, str]] = []
        stats: LatenessStats = {
            "early": 1,
            "on_time": 2,
            "late": 3,
            "avg": 4.0,
            "median": 5.0,
            "best_early": -6.0,
            "worst_late": 7.0,
            "count": 6,
        }
        stats_rows(
            rows,
            [],
            lateness_stats=lambda _chain: stats,
            format_seconds_delta=lambda seconds: f"{seconds}s",
        )

        self.assertEqual(
            rows,
            [
                ("Performance", "early 1, on-time 2, late 3"),
                ("Avg lateness", "4.0s"),
                ("Median lateness", "5.0s"),
                ("Best early", "-6.0s"),
                ("Worst late", "7.0s"),
            ],
        )

    def test_delete_chain_summary_span_uses_stop_time_without_last_end(self) -> None:
        from nautical_core.modify_chain_summary import span_fields

        first, last, span = span_fields(
            "cid",
            [{"uuid": "root", "due": "20260101T000000Z"}, {"uuid": "tail", "status": "deleted"}],
            stop_at=datetime(2026, 1, 11, tzinfo=timezone.utc),
            stopped_by_delete=True,
            export_endpoint=lambda *_: None,
            parse_datetime=lambda value: (
                datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                if value else None
            ),
            human_delta=lambda start, end, **_kwargs: f"{(end - start).days} days",
        )

        self.assertIsNotNone(first)
        self.assertIsNone(last)
        self.assertEqual(span, "Active for 10 days before deletion")

    def test_end_summary_history_marks_deleted_pending_tail(self) -> None:
        from nautical_core.modify_chain_summary import last_n_timeline

        lines = last_n_timeline(
            [
                {"uuid": "00000000-0000-4000-8000-000000000111", "status": "completed", "link": 1,
                 "due": "20260101T000000Z", "end": "20260101T000000Z"},
                {"uuid": "00000000-0000-4000-8000-000000000222", "status": "deleted", "link": 2,
                 "due": "20260102T000000Z"},
            ],
            n=6,
            coerce_int=lambda value, default: int(value) if value is not None else default,
            parse_datetime=lambda value: (
                datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                if value else None
            ),
            format_local=lambda value: value.isoformat(),
            format_on_time_delta=lambda _due, _end: "on time",
            short_uuid=lambda value: str(value)[:8],
        )
        rendered = "\n".join(lines)

        self.assertIn("#2", rendered)
        self.assertIn("×", rendered)
        self.assertIn("deleted", rendered)
        self.assertNotIn("#2  ✓", rendered)

    def test_delete_chain_summary_uses_stopped_title(self) -> None:
        captured = []
        services = ChainSummaryRenderServices(
            export_sorted_chain=lambda *_: [],
            root_uuid_from=lambda task: task.get("uuid"),
            short_uuid=lambda value: str(value or "")[:8],
            format_root_and_age=lambda *_: "root",
            kind_rows=lambda *_: None,
            span_fields=lambda *_args, **_kwargs: (None, None, "–"),
            stats_rows=lambda *_: None,
            limits_row=lambda *_: None,
            last_n_timeline_rows=lambda *_: [],
            format_rows=lambda rows: rows,
            coerce_int=lambda value, default: int(value) if value is not None else default,
            format_local=lambda value: str(value),
            max_chain_walk=10,
            panel=lambda title, _rows, **_kwargs: captured.append(title),
            diagnostic=lambda _message: None,
        )

        render_chain_summary_with_services(
            {"uuid": "00000000-0000-4000-8000-000000000222", "chainID": "00000000", "link": 2},
            "Pending task deleted.",
            datetime(2026, 1, 3, tzinfo=timezone.utc),
            None,
            services=services,
        )

        self.assertEqual(captured, ["⛔ Chain stopped – summary"])

    def test_service_bundle_delegates_to_renderer(self) -> None:
        calls = []
        services = ChainSummaryRenderServices(
            export_sorted_chain=lambda *_: [], root_uuid_from=lambda value: value.get("uuid"),
            short_uuid=lambda value: str(value or "")[:4], format_root_and_age=lambda *_: "root",
            kind_rows=lambda *_: None, span_fields=lambda *_args, **_kwargs: (None, None, "–"),
            stats_rows=lambda *_: None, limits_row=lambda *_: None,
            last_n_timeline_rows=lambda *_: [], format_rows=lambda rows: rows,
            coerce_int=lambda value, default: int(value or default), format_local=lambda value: str(value),
            max_chain_walk=10, panel=lambda *args, **kwargs: calls.append((args, kwargs)), diagnostic=lambda _: None,
        )
        with patch("nautical_core.modify_chain_summary.render_chain_summary") as renderer:
            render_chain_summary_with_services(
                {"uuid": "u", "chainID": "c"},
                "done",
                datetime(2026, 1, 3, tzinfo=timezone.utc),
                None,
                services=services,
            )
        renderer.assert_called_once()
        self.assertIs(renderer.call_args.kwargs["services"], services)


if __name__ == "__main__":
    unittest.main()
