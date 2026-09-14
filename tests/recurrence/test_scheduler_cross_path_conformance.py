"""Cross-path scheduler conformance matrix migrated from golden tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
from zoneinfo import ZoneInfo

import nautical_core as core
from nautical_core.occurrence_outcomes import FoundOccurrence
from nautical_core.occurrence_provider import Occurrence
from nautical_core.recurrence_context import RecurrenceContext
from nautical_core.scheduler_cursor import OccurrenceCursor, OccurrenceRangeRequest
from nautical_core.scheduler_service import SchedulerService
from nautical_core.task_models import TaskObservation


def _scheduler_for_fixture(
    task: dict[str, str], *, context: RecurrenceContext
) -> SchedulerService:
    values = dict(task)
    values.setdefault("uuid", "00000000-0000-4000-8000-000000000001")
    values.setdefault("description", "typed fixture task")
    values.setdefault("status", "pending")
    values.setdefault("link", 1)
    return SchedulerService.from_observation(
        TaskObservation.from_mapping(values, source_query="scheduler cross-path fixture"),
        context=context,
    )


def _occurrence_signature(item: Occurrence) -> tuple[datetime, bool, str, str]:
    if item.local_datetime is None:
        raise AssertionError("scheduler returned an occurrence without a local datetime")
    instant = item.local_datetime.astimezone(timezone.utc) if item.local_datetime.tzinfo else item.local_datetime
    return instant, item.omitted, item.source, item.description


def _astral_test_available() -> bool:
    try:
        from astral import Observer, moon, sun  # noqa: F401
    except ImportError:
        if os.environ.get("NAUTICAL_REQUIRE_ASTRAL") == "1":
            raise AssertionError("Astral is required for this test job; install requirements.txt")
        return False
    return True


class SchedulerCrossPathConformanceTests(unittest.TestCase):
    def assert_monotonic(self, items: tuple[Occurrence, ...], name: str) -> None:
        signatures = tuple(_occurrence_signature(item) for item in items)
        self.assertTrue(
            all(left[0] < right[0] for left, right in zip(signatures, signatures[1:])),
            f"{name} occurrence stream is not strictly monotonic: {signatures!r}",
        )

    def test_scheduler_cross_path_conformance_matrix(self) -> None:
        """Next, preview, collection, and range paths agree across recurrence families."""
        zone = ZoneInfo("Europe/Sofia")
        cases = (
            ("ordinary", {"anchor": "w:mon,wed,fri@t=09:00"}, "2026-03-01T09:00:00", 3),
            ("sparse", {"anchor": "y:02-29@t=09:00"}, "2026-01-01T09:00:00", 2),
            ("interval", {"anchor": "m/2:15@t=09:00"}, "2026-01-01T09:00:00", 3),
            ("and-or", {"anchor": "(w:mon + m:1) | w:fri@t=09:00"}, "2026-03-01T09:00:00", 3),
            ("omit", {"anchor": "w:mon | w:wed@t=09:00", "omit": "w:wed"}, "2026-03-01T09:00:00", 3),
            ("random", {"anchor": "m:rand@t=09:00"}, "2026-04-01T09:00:00", 3),
            ("overnight", {"anchor": "w:mon@t=22:30..06:30/2"}, "2026-08-03T23:00:00", 3),
        )
        for name, task, stamp, limit in cases:
            with self.subTest(name=name):
                task = dict(task, chainID=f"conformance-{name}")
                context = RecurrenceContext(chain_id=task["chainID"], timezone=zone)
                service = _scheduler_for_fixture(task, context=context)
                cursor = OccurrenceCursor.strict_after(
                    datetime.fromisoformat(stamp).replace(tzinfo=zone), timezone=zone
                )
                first = service.next(cursor)
                collected = service.collect(cursor, limit=limit)
                preview = service.preview(cursor.local_datetime, limit=1, timezone=zone)
                ranged = service.collect_request(
                    OccurrenceRangeRequest(
                        cursor,
                        end_local=cursor.local_datetime + timedelta(days=2400),
                        limit=limit,
                    )
                )
                self.assert_monotonic(tuple(collected.occurrences), name)
                self.assert_monotonic(tuple(ranged.occurrences), name)
                self.assertIsInstance(first, FoundOccurrence, f"{name} next path did not find an occurrence: {first!r}")
                self.assertTrue(collected.occurrences, f"{name} collection returned no occurrence")
                self.assertTrue(preview.occurrences, f"{name} preview returned no occurrence")
                self.assertTrue(ranged.occurrences, f"{name} range returned no occurrence")
                first_signature = _occurrence_signature(first.occurrence)
                collected_signatures = tuple(_occurrence_signature(item) for item in collected.occurrences)
                ranged_signatures = tuple(_occurrence_signature(item) for item in ranged.occurrences)
                self.assertEqual(first_signature, _occurrence_signature(collected.occurrences[0]), f"{name} next/collection diverged")
                self.assertEqual(first_signature, _occurrence_signature(preview.occurrences[0]), f"{name} next/preview diverged")
                self.assertEqual(first_signature, _occurrence_signature(ranged.occurrences[0]), f"{name} next/range diverged")
                self.assertEqual(collected_signatures, ranged_signatures, f"{name} collection/range stream diverged")
                repeat = service.collect(cursor, limit=limit)
                self.assertEqual(
                    collected_signatures,
                    tuple(_occurrence_signature(item) for item in repeat.occurrences),
                    f"{name} collection was not deterministic",
                )

        if _astral_test_available() and getattr(core, "ASTRONOMY_CONFIG", {}):
            task = {"anchor": "w:mon@t=sunrise", "chainID": "conformance-astronomy"}
            context = RecurrenceContext(
                chain_id=task["chainID"], timezone=zone,
                astronomy_config=core.ASTRONOMY_CONFIG,
            )
            service = _scheduler_for_fixture(task, context=context)
            cursor = OccurrenceCursor.strict_after(
                datetime(2026, 6, 1, 9, tzinfo=zone), timezone=zone
            )
            self.assertIsInstance(service.next(cursor), FoundOccurrence, "astronomy next path did not find an occurrence")

        with TemporaryDirectory() as directory:
            Path(directory, "calendar.csv").write_text(
                "date,description\n2026-08-03,first\n2026-08-10,second\n", encoding="utf-8"
            )
            task = {"anchor_file": "calendar.csv@t=09:00", "chainID": "conformance-file"}
            context = RecurrenceContext(chain_id=task["chainID"], timezone=zone, anchor_file_dir=directory)
            service = _scheduler_for_fixture(task, context=context)
            cursor = OccurrenceCursor.strict_after(datetime(2026, 8, 1, 9, tzinfo=zone), timezone=zone)
            first = service.next(cursor)
            ranged = service.collect_request(OccurrenceRangeRequest(cursor, limit=2))
            self.assertIsInstance(first, FoundOccurrence, "file next path did not find an occurrence")
            self.assertTrue(ranged.occurrences and ranged.occurrences[0].description == "first", "file range lost description")
            self.assertEqual(_occurrence_signature(first.occurrence), _occurrence_signature(ranged.occurrences[0]), "file next/range diverged")

    def test_generated_recurrence_matrix_is_monotonic_timezone_aware_and_repeatable(self) -> None:
        import random

        rng = random.Random(20260812)
        anchors = [
            "w:mon", "w/2:tue", "m:1", "m/2:15", "y:01-01", "y:02-29",
            "w:mon + m:1", "w:mon | w:fri", "m:rand", "w:sun@t=01:30",
        ]
        rng.shuffle(anchors)
        zone = ZoneInfo("Europe/Sofia")
        for index, anchor in enumerate(anchors):
            with self.subTest(anchor=anchor):
                chain_id = f"generated-matrix-{index}"
                task = {"anchor": anchor, "chainID": chain_id}
                service = _scheduler_for_fixture(
                    task,
                    context=RecurrenceContext(chain_id=chain_id, timezone=zone),
                )
                cursor = OccurrenceCursor.strict_after(
                    datetime(2026, 1, 1, 0, 0, tzinfo=zone), timezone=zone
                )
                collected = service.collect(cursor, limit=3, max_iterations=2048)
                self.assertTrue(collected.occurrences)
                self.assert_monotonic(tuple(collected.occurrences), anchor)
                self.assertTrue(
                    all(
                        item.local_datetime is not None
                        and item.local_datetime.tzinfo is not None
                        for item in collected
                    )
                )
                replay = service.collect(cursor, limit=3, max_iterations=2048)
                self.assertEqual(
                    [item.local_datetime for item in collected],
                    [item.local_datetime for item in replay],
                )

                step = cursor
                for _ in range(3):
                    outcome = service.next(step, max_file_skips=2048)
                    self.assertIsInstance(outcome, FoundOccurrence)
                    self.assertGreater(outcome.local_datetime, step.local_datetime)
                    step = OccurrenceCursor.strict_after(
                        outcome.local_datetime, timezone=zone
                    )

    def test_parity_harness_compares_next_and_collection_with_reference_callbacks(self) -> None:
        from dev_tools.nautical_scheduler_parity import compare_collection, compare_next

        zone = ZoneInfo("Europe/Sofia")
        service = _scheduler_for_fixture(
            {"chainID": "parity-chain", "anchor": "w:mon@t=09:00"},
            context=RecurrenceContext(chain_id="parity-chain", timezone=zone),
        )
        cursor = OccurrenceCursor.strict_after(
            datetime(2026, 8, 2, 9, 0, tzinfo=zone), timezone=zone
        )
        evaluator = service.session.evaluator

        compare_next(service, cursor, lambda: evaluator.next_after_cursor(cursor))
        compare_collection(
            service,
            cursor,
            lambda: evaluator.collect_after_cursor(cursor, limit=2),
            limit=2,
        )

    def test_parity_matrix_covers_omission_dst_and_random_contexts(self) -> None:
        from dev_tools.nautical_scheduler_parity import compare_collection

        cases = (
            ("w:mon,wed,fri@t=09:00", "w:wed", "2026-03-01T09:00:00"),
            ("w:sun@t=01:30", "", "2026-10-24T01:00:00"),
            ("m:rand", "", "2026-04-01T09:00:00"),
        )
        for index, (anchor, omit, stamp) in enumerate(cases):
            with self.subTest(anchor=anchor):
                zone = ZoneInfo("Europe/Sofia")
                task = {"chainID": f"parity-matrix-{index}", "anchor": anchor}
                if omit:
                    task["omit"] = omit
                context = RecurrenceContext(chain_id=task["chainID"], timezone=zone)
                service = _scheduler_for_fixture(task, context=context)
                cursor = OccurrenceCursor.strict_after(
                    datetime.fromisoformat(stamp).replace(tzinfo=zone), timezone=zone
                )
                evaluator = service.session.evaluator
                compare_collection(
                    service,
                    cursor,
                    lambda evaluator=evaluator, cursor=cursor: evaluator.collect_after_cursor(
                        cursor, limit=2
                    ),
                    limit=2,
                )

    def test_query_navigator_and_reconcile_share_scheduler_projection(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        import nautical_navigator as navigator
        from nautical_core.chain_integrity_lifecycle import plan_recovery_decision
        from nautical_core.integration_context import IntegrationAccess
        from nautical_core.integration_models import Found
        from nautical_core.query_models import OccurrenceQueryRequest
        from nautical_core.query_service import OccurrenceQueryRuntime, OccurrenceQueryService
        from nautical_core.task_codec import DEFAULT_TASK_CODEC
        from nautical_core.task_models import NauticalTask

        zone = ZoneInfo("Europe/Sofia")
        task = {
            "uuid": "00000000-0000-4000-8000-000000000721",
            "description": "domain parity task",
            "status": "completed",
            "chain": "on",
            "chainID": "domain-parity-consumers",
            "link": 7,
            "anchor": "w:mon@t=09:00",
            "anchor_mode": "skip",
            "due": "20260817T060000Z",
            "end": "20260817T060000Z",
        }
        observation = DEFAULT_TASK_CODEC.decode_row(task, source_query="domain parity")
        typed_task = NauticalTask.from_observation(observation)
        context = RecurrenceContext(chain_id=typed_task.identity.chain_id, timezone=zone)
        scheduler = SchedulerService.from_task(typed_task, context=context)
        cursor = OccurrenceCursor.strict_after(
            datetime(2026, 8, 17, 9, tzinfo=zone), timezone=zone
        )
        direct = scheduler.next(cursor)
        self.assertIsInstance(direct, FoundOccurrence)
        expected_local = direct.occurrence.local_datetime
        self.assertIsNotNone(expected_local)

        class Repository:
            def by_uuid(self, value, **_kwargs):
                return Found(task, f"uuid:{value}")

        uow = SimpleNamespace(
            context=SimpleNamespace(
                access=IntegrationAccess.READ_ONLY,
                local_timezone=zone,
                configuration=SimpleNamespace(fingerprint="domain-parity"),
            ),
            repository=Repository(),
        )
        query_request = OccurrenceQueryRequest.from_mapping(
            {
                "selector": {"uuids": [task["uuid"]]},
                "from": "2026-08-24",
                "to": "2026-08-24",
            }
        )
        query_result = OccurrenceQueryService(
            uow, runtime=OccurrenceQueryRuntime.from_compatibility_facade(core)
        ).query(query_request).results[0]
        self.assertEqual(query_result.status, "found")
        self.assertEqual(query_result.occurrences[0].local, expected_local)

        previous_zone = navigator.LOCAL_ZONE
        navigator.LOCAL_ZONE = zone
        try:
            projected = navigator.TaskAnalyzer()._project_anchor_dates(
                task, limit=1, start_from_date=date(2026, 8, 17)
            )
        finally:
            navigator.LOCAL_ZONE = previous_zone
        self.assertTrue(projected)
        self.assertEqual(projected[0], expected_local)

        with patch.object(core, "LOCAL_TZ_NAME", "Europe/Sofia"), patch.object(
            core, "_LOCAL_TZ", zone
        ):
            recovery = plan_recovery_decision(observation, existing_children=[], hook=None)
        self.assertEqual(recovery.plan.action.value, "spawn_child")
        self.assertEqual(recovery.child_due, expected_local.astimezone(timezone.utc))

    def test_chain_generation_matches_evaluator_across_time_form_modes(self) -> None:
        from nautical_core.chain_generation import ChainGenerationService
        from nautical_core.task_codec import DEFAULT_TASK_CODEC
        from nautical_core.task_models import NauticalTask

        cases = (
            "w:mon@t=09:00",
            "w:mon@t=09:00,13:00",
            "w:mon@t=06..18/3",
            "w:mon@t=rand(06..18/3)",
            "w:mon@t=22:30..06:30/7",
        )

        # Force lazy timezone configuration before capturing the context.
        due_utc = core.build_local_datetime(date(2026, 8, 3), (9, 0))
        end_utc = core.build_local_datetime(date(2026, 8, 3), (10, 0))
        local_timezone = core._LOCAL_TZ
        due_local = core.to_local(due_utc)
        end_local = core.to_local(end_utc)
        for index, expression in enumerate(cases):
            with self.subTest(expression=expression):
                chain_id = f"time-mode-parity-{index}"
                parent = {
                    "uuid": f"00000000-0000-4000-8000-{index:012d}",
                    "status": "pending",
                    "link": 1,
                    "chain": "on",
                    "chainID": chain_id,
                    "anchor": expression,
                    "anchor_mode": "skip",
                    "due": core.fmt_isoz(due_utc),
                    "end": core.fmt_isoz(end_utc),
                }
                observation = DEFAULT_TASK_CODEC.decode_row(
                    parent, source_query="time-form parity"
                )
                typed_parent = NauticalTask.from_observation(observation)
                generated_due, metadata, _dnf = ChainGenerationService.from_core(
                    core
                ).compute_anchor_child_due(typed_parent)

                evaluator = _scheduler_for_fixture(
                    parent,
                    context=RecurrenceContext(chain_id=chain_id, timezone=local_timezone),
                ).session.evaluator
                selected = evaluator.select_mode(
                    "skip",
                    due_local=due_local,
                    end_local=end_local,
                    fallback_hhmm=(9, 0),
                )

                self.assertEqual(
                    selected.selected_occurrence.astimezone(timezone.utc),
                    generated_due,
                )
                self.assertEqual(selected.basis, metadata.get("basis"))
                self.assertEqual(selected.source, metadata.get("source"))

    def test_chain_generation_matches_evaluator_across_dst_gap(self) -> None:
        from unittest.mock import patch

        from nautical_core.chain_generation import ChainGenerationService
        from nautical_core.task_codec import DEFAULT_TASK_CODEC
        from nautical_core.task_models import NauticalTask

        core.to_local(datetime(2026, 1, 1, tzinfo=timezone.utc))
        local_timezone = ZoneInfo("America/New_York")
        with patch.object(core, "_LOCAL_TZ", local_timezone):
            due_utc = core.build_local_datetime(date(2025, 3, 9), (1, 30))
            end_utc = core.build_local_datetime(date(2025, 3, 9), (2, 0))
            due_local = core.to_local(due_utc)
            end_local = core.to_local(end_utc)
            task = {
                "uuid": "00000000-0000-4000-8000-000000000731",
                "status": "pending",
                "link": 1,
                "chainID": "dst-gap-parity",
                "anchor": "w:sun@t=02:30",
                "anchor_mode": "skip",
                "due": core.fmt_isoz(due_utc),
                "end": core.fmt_isoz(end_utc),
            }
            typed_task = NauticalTask.from_observation(
                DEFAULT_TASK_CODEC.decode_row(task, source_query="DST gap parity")
            )
            generated_due, metadata, _dnf = ChainGenerationService.from_core(
                core
            ).compute_anchor_child_due(typed_task)
            evaluator = _scheduler_for_fixture(
                task,
                context=RecurrenceContext(
                    chain_id="dst-gap-parity", timezone=local_timezone
                ),
            ).session.evaluator
            selected = evaluator.select_mode(
                "skip",
                due_local=due_local,
                end_local=end_local,
                fallback_hhmm=(9, 0),
            )

        self.assertEqual(
            selected.selected_occurrence.astimezone(timezone.utc), generated_due
        )
        self.assertEqual(selected.basis, metadata.get("basis"))
        self.assertEqual(selected.source, metadata.get("source"))

    def test_chain_generation_matches_evaluator_with_business_calendar(self) -> None:
        from unittest.mock import patch

        from nautical_core.chain_generation import ChainGenerationService
        from nautical_core.task_codec import DEFAULT_TASK_CODEC
        from nautical_core.task_models import NauticalTask

        class SetCalendar:
            name = "shadow-business"
            fingerprint = "shadow-business-v1"

            def is_business_day(self, value: date) -> bool:
                return value in {date(2026, 1, 2), date(2026, 1, 7)}

        core.to_local(datetime(2026, 1, 1, tzinfo=timezone.utc))
        local_timezone = core._LOCAL_TZ
        policy = SetCalendar()
        due_utc = core.build_local_datetime(date(2026, 1, 1), (9, 0))
        end_utc = core.build_local_datetime(date(2026, 1, 1), (10, 0))
        due_local = core.to_local(due_utc)
        end_local = core.to_local(end_utc)
        task = {
            "uuid": "00000000-0000-4000-8000-000000000732",
            "status": "pending",
            "link": 1,
            "chainID": "business-calendar-parity",
            "anchor": "m:1bd",
            "anchor_mode": "skip",
            "bc": "shadow-business",
            "due": core.fmt_isoz(due_utc),
            "end": core.fmt_isoz(end_utc),
        }
        typed_task = NauticalTask.from_observation(
            DEFAULT_TASK_CODEC.decode_row(task, source_query="business calendar parity")
        )
        with patch.object(core, "business_calendar_for_task", return_value=policy):
            generated_due, metadata, _dnf = ChainGenerationService.from_core(
                core
            ).compute_anchor_child_due(typed_task)
        evaluator = _scheduler_for_fixture(
            task,
            context=RecurrenceContext(
                chain_id="business-calendar-parity",
                timezone=local_timezone,
                business_calendar=policy,
            ),
        ).session.evaluator
        selected = evaluator.select_mode(
            "skip",
            due_local=due_local,
            end_local=end_local,
            fallback_hhmm=(9, 0),
        )

        self.assertEqual(
            selected.selected_occurrence.astimezone(timezone.utc), generated_due
        )
        self.assertEqual(selected.basis, metadata.get("basis"))
        self.assertEqual(selected.source, metadata.get("source"))


if __name__ == "__main__":
    unittest.main()
