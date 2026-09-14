"""Direct contracts for Navigator's immutable, renderer-neutral views."""

import unittest
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo
from rich.console import Console

import nautical_core as core
import nautical_navigator as navigator
from nautical_core import astronomy
from nautical_core.occurrence_outcomes import ExhaustedOccurrence
from nautical_core.occurrence_provider import Occurrence
from nautical_core.scheduler_models import OccurrenceSearchExhausted
from nautical_core.query_models import TaskIdentity


class NavigatorViewModelTests(unittest.TestCase):
    def test_symbolic_anchor_time_applies_event_offset_in_navigator_zone(self) -> None:
        zone = ZoneInfo("America/New_York")
        event = datetime(2026, 7, 6, 23, 5, tzinfo=timezone.utc)
        config = {
            "default_location": "test",
            "locations": {
                "test": {
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "timezone": "America/New_York",
                }
            },
        }
        task = {
            "uuid": "00000000-0000-4000-8000-000000000902",
            "description": "symbolic navigator time",
            "status": "pending",
            "link": 1,
            "anchor": "w:mon@t=sunset@+45m",
            "chainID": "navigator-symbolic",
        }

        with (
            patch.object(navigator, "LOCAL_ZONE", zone),
            patch.object(core, "ASTRONOMY_CONFIG", config),
            patch.object(astronomy, "resolve_event", return_value=event) as resolve_event,
        ):
            projected = navigator.TaskAnalyzer()._project_anchor_dates(
                task,
                limit=1,
                start_from_date=date(2026, 7, 1),
            )

        expected = (event.astimezone(zone) + timedelta(minutes=45)).replace(
            second=0,
            microsecond=0,
        )
        self.assertEqual(projected[0], expected)
        resolve_event.assert_called_once_with(
            "sunset",
            date(2026, 7, 6),
            config=config,
        )

    def test_anchor_projection_and_due_check_use_task_business_calendar(self) -> None:
        calendars = core.resolve_business_calendar_config(
            {"weekend": {"anchor": "w:sat,sun"}}
        )
        task = {
            "uuid": "00000000-0000-4000-8000-000000000901",
            "status": "pending",
            "link": 1,
            "chainID": "navigator-business-calendar",
            "description": "weekend navigator anchor",
            "anchor": "m:1bd@t=09:00",
            "bc": "weekend",
        }
        with patch.object(core, "configured_business_calendars", return_value=calendars):
            analyzer = navigator.TaskAnalyzer()
            projected = analyzer._project_anchor_dates(
                task,
                limit=1,
                start_from_date=date(2026, 6, 30),
            )
            due = core.fmt_isoz(
                core.build_local_datetime(date(2026, 7, 4), (9, 0))
            )
            due_is_anchor_day = analyzer._due_is_anchor_day(due, task)

        self.assertEqual([item.date() for item in projected], [date(2026, 7, 4)])
        self.assertTrue(due_is_anchor_day)

    def test_anchor_projection_preserves_scheduler_terminal_evidence(self) -> None:
        context = SimpleNamespace(timezone=timezone.utc)
        result = SimpleNamespace(
            occurrences=[
                Occurrence(
                    date(2026, 8, 24),
                    9,
                    0,
                    local_datetime=datetime(2026, 8, 24, 9, 0, tzinfo=timezone.utc),
                )
            ],
            terminal=ExhaustedOccurrence(
                OccurrenceSearchExhausted("navigator test", limit=1)
            ),
        )

        class FakeService:
            session = SimpleNamespace(evaluator=SimpleNamespace(context=context))

            def collect(self, *_args, **_kwargs):
                return result

        with patch.object(
            navigator,
            "_build_navigator_scheduler",
            return_value=(FakeService(), context),
        ):
            projected = navigator.TaskAnalyzer()._project_anchor_dates(
                {
                    "anchor": "w:mon",
                    "uuid": "00000000-0000-4000-8000-000000000903",
                    "chainID": "terminal-test",
                },
                limit=1,
                start_from_date=date(2026, 8, 1),
            )

        self.assertEqual([item.date() for item in projected], [date(2026, 8, 24)])
        self.assertIs(projected.terminal, result.terminal)

    def test_metadata_keeps_chain_links_and_lifecycle_flags(self) -> None:
        row = {
            "uuid": "aaaaaaaa-0000-4000-8000-000000000001",
            "chainID": "chain-test",
            "link": 4,
            "prevLink": "bbbbbbbb",
            "nextLink": "cccccccc",
            "status": "pending",
            "chain": "on",
            "anchor": "w:mon",
        }

        metadata = navigator.NavigatorTaskMetadata.from_mapping(row)

        self.assertEqual(metadata.chain_id, "chain-test")
        self.assertEqual(metadata.link, 4)
        self.assertEqual(metadata.previous, "bbbbbbbb")
        self.assertEqual(metadata.following, "cccccccc")
        self.assertTrue(metadata.is_active)
        self.assertTrue(metadata.has_anchor)

    def test_sparse_calendar_renders_only_months_with_activity(self) -> None:
        analyzer = navigator.TaskAnalyzer()
        panel = analyzer.create_enhanced_calendar(
            completed_dates=[date(2026, 7, 1)],
            upcoming_dates=[date(2027, 7, 1), date(2035, 7, 1)],
            pending_due_dates=[],
        )
        output = StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        console.print(panel)
        rendered = output.getvalue()

        self.assertIn("July 2026", rendered)
        self.assertIn("July 2027", rendered)
        self.assertIn("July 2035", rendered)
        self.assertNotIn("August 2026", rendered)
        self.assertNotIn("January 2035", rendered)

    def test_anchor_projection_failure_is_retained_without_fabricating_dates(self) -> None:
        analyzer = navigator.TaskAnalyzer()
        with patch.object(
            core,
            "business_calendar_for_task",
            side_effect=ValueError("calendar is invalid"),
        ):
            projected = analyzer._project_anchor_dates(
                {"anchor": "w:mon", "uuid": "navigator-warning-test"},
                limit=1,
                start_from_date=date(2026, 8, 2),
            )

        self.assertEqual(projected, [])
        self.assertEqual(
            analyzer._projection_warnings,
            ["Business calendar: calendar is invalid"],
        )
    def test_metadata_matches_query_task_identity(self) -> None:
        row = {
            "uuid": "00000000-0000-4000-8000-000000000123",
            "chainID": "parity-chain",
            "link": 7,
            "description": "Parity task",
            "anchor": "w:mon@t=09:00",
            "due": "20260824T060000Z",
            "scheduled": "20260824T053000Z",
            "status": "pending",
        }
        metadata = navigator.NavigatorTaskMetadata.from_mapping(row)
        identity = TaskIdentity(
            uuid=row["uuid"],
            chain_id=row["chainID"],
            link=row["link"],
            description=row["description"],
            recurrence_kind="anchor",
            expression=row["anchor"],
            current_due=row["due"],
            current_scheduled=row["scheduled"],
        )

        self.assertEqual(metadata.uuid, identity.uuid)
        self.assertEqual(metadata.chain_id, identity.chain_id)
        self.assertEqual(metadata.link, identity.link)
        self.assertEqual(metadata.due, identity.current_due)
        self.assertEqual(metadata.scheduled, identity.current_scheduled)

    def test_task_metadata_serialization_is_deterministic(self) -> None:
        metadata = navigator.NavigatorTaskMetadata(
            uuid="u1", chain_id="c1", link=2, previous="u0", following=None,
            status="pending", chain_enabled=True, anchor="w:mon",
        )

        serialized = metadata.to_dict()
        self.assertEqual(serialized["chainID"], "c1")
        self.assertEqual(serialized["anchor"], "w:mon")
        self.assertEqual(metadata.to_dict(), serialized)

    def test_calendar_view_serializes_dates_by_category(self) -> None:
        view = navigator.NavigatorCalendarView(
            completed=(date(2026, 8, 1),),
            upcoming=(date(2026, 8, 8),),
            pending_due=(date(2026, 8, 3),),
        )

        self.assertEqual(view.to_dict(), {
            "completed": ["2026-08-01"],
            "upcoming": ["2026-08-08"],
            "pending_due": ["2026-08-03"],
        })

    def test_chain_summary_serializes_dates_and_completion_facts(self) -> None:
        summary = navigator.NavigatorChainSummary(
            3, 3, date(2026, 8, 1), date(2026, 8, 3), 2, 1.0, 1.0,
            "100% on-anchor",
        )

        serialized = summary.to_dict()
        self.assertEqual(serialized["completed_links"], 3)
        self.assertEqual(serialized["first_end"], "2026-08-01")

    def test_chain_choice_serializes_stable_identifiers(self) -> None:
        choice = navigator.NavigatorChainChoice(
            0, "c1", "[pending] Demo", "u1", "2026-08-01"
        )

        self.assertEqual(choice.to_dict()["chainID"], "c1")
        self.assertEqual(choice.to_dict()["tail_uuid"], "u1")

    def test_change_row_serializes_typed_changes(self) -> None:
        row = navigator.NavigatorChangeRow(
            "u1", "2026-08-01",
            (navigator.TaskChange("due", "changed", "a", "b"),),
        )

        self.assertEqual(row.to_dict()["changes"][0]["field"], "due")

    def test_task_detail_serializes_fields_and_truncation(self) -> None:
        view = navigator.NavigatorTaskDetailView(
            (("UUID", "u1"), ("Status", "Pending")), truncated=True
        )

        self.assertEqual(view.to_dict(), {
            "fields": [
                {"label": "UUID", "value": "u1"},
                {"label": "Status", "value": "Pending"},
            ],
            "truncated": True,
        })

    def test_projection_view_serializes_warnings(self) -> None:
        view = navigator.NavigatorProjectionView(("anchor unavailable",))

        self.assertEqual(view.to_dict(), {"warnings": ["anchor unavailable"]})

    def test_trace_view_preserves_event_summary(self) -> None:
        view = navigator.NavigatorTraceView(
            event_count=2, providers=("anchor",), phases=(("selected", 1),),
            selected="2026-08-01",
        )

        self.assertEqual(view.to_dict()["events"], 2)
        self.assertIn("selected=2026-08-01", view.summary())

    def test_analysis_view_aggregates_typed_sections(self) -> None:
        view = navigator.NavigatorAnalysisView(
            chain_size=2,
            calendar=navigator.NavigatorCalendarView(
                completed=(date(2026, 8, 1),)
            ),
            projection=navigator.NavigatorProjectionView(("warning",)),
        )

        document = view.to_dict()
        self.assertEqual(document["chain_size"], 2)
        self.assertEqual(document["calendar"]["completed"], ["2026-08-01"])


if __name__ == "__main__":
    unittest.main()
