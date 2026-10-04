from __future__ import annotations

import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from collections.abc import Callable
from typing import get_type_hints

from nautical_core.modify_chain_summary import (
    ChainSummaryRenderServices,
    SpanFieldsCallback,
    SummaryKindRows,
    SummaryLimitsRow,
    SummaryRowsFormatter,
    SummaryStatsRows,
    SummaryTimelineRows,
    render_chain_summary_with_services,
)
from nautical_core.task_models import TaskObservation, TaskPayload


class ChainSummaryRendererContractTests(unittest.TestCase):
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
            render_chain_summary_with_services({"uuid": "u", "chainID": "c"}, "done", None, None, services=services)
        renderer.assert_called_once()
        self.assertIs(renderer.call_args.kwargs["services"], services)


if __name__ == "__main__":
    unittest.main()
