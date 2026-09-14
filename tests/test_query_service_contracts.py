"""Direct read-only query-service contracts migrated from golden coverage."""

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import nautical_core as core
from nautical_core.integration_context import IntegrationAccess
from nautical_core.integration_models import (
    Absent, CommandFailureKind, FailureEvidence, Found, TaskCommand, Unavailable,
)
from nautical_core.query_models import OccurrenceQueryRequest
from nautical_core.query_service import OccurrenceQueryRuntime, OccurrenceQueryService
from nautical_core.recurrence_context import RecurrenceContext
from nautical_core.scheduler_cursor import OccurrenceCursor, OccurrenceRangeRequest
from nautical_core.scheduler_service import SchedulerService
from nautical_core.task_models import TaskObservation


def _runtime() -> OccurrenceQueryRuntime:
    return OccurrenceQueryRuntime.from_compatibility_facade(core)


def _uow(repository: object, local_timezone: timezone = timezone.utc) -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(
            access=IntegrationAccess.READ_ONLY,
            local_timezone=local_timezone,
            configuration=SimpleNamespace(fingerprint="query-config"),
        ),
        repository=repository,
    )


def _scheduler(task: dict[str, object], context: RecurrenceContext) -> SchedulerService:
    return SchedulerService.from_observation(
        TaskObservation.from_mapping(task, source_query="query service contract fixture"),
        context=context,
    )


class QueryServiceContractsTests(unittest.TestCase):
    def test_projects_schedule_read_only_with_omissions_cap_and_scheduler_parity(self) -> None:
        task = {
            "uuid": "00000000-0000-4000-8000-000000000002", "chainID": "query-chain",
            "link": 1, "description": "Two daily slots", "anchor": "w:mon..sun@t=04:30,12:30",
            "anchor_mode": "skip", "status": "pending",
        }

        class Repository:
            def by_uuid(self, value, **kwargs):
                del kwargs
                return Found(task, f"uuid:{value}")

        uow = _uow(Repository(), timezone(timedelta(hours=3)))
        service = OccurrenceQueryService(uow, runtime=_runtime())
        request = OccurrenceQueryRequest.from_mapping({
            "selector": {"uuids": [task["uuid"]]}, "from": "2026-08-24", "to": "2026-08-24",
            "omission_policy": "exclude",
        })
        response = service.query(request)
        self.assertEqual(response.status, "found")
        self.assertEqual(len(response.results), 1)
        result = response.results[0]
        self.assertTrue(result.task and result.task.schedule_fingerprint)
        self.assertEqual([item.local.hour for item in result.occurrences], [4, 12])
        self.assertIsNotNone(result.occurrences[0].utc.tzinfo)
        context = RecurrenceContext(
            chain_id=task["chainID"], timezone=uow.context.local_timezone,
            business_calendar=core.business_calendar_for_task(dict(task)),
            astronomy_config=core.ASTRONOMY_CONFIG, anchor_file_dir=core.ANCHOR_FILE_DIR,
        )
        direct = _scheduler(task, context).collect_request(OccurrenceRangeRequest(
            OccurrenceCursor(datetime(2026, 8, 24, tzinfo=uow.context.local_timezone), inclusive=True,
                             timezone=uow.context.local_timezone),
            end_local=datetime(2026, 8, 24, 23, 59, 59, tzinfo=uow.context.local_timezone), limit=10,
        ))
        self.assertEqual(tuple(item.local_datetime for item in direct.occurrences),
                         tuple(item.local for item in result.occurrences))
        omitted_task = dict(task, omit="y:08-24")

        class OmitRepository:
            def by_uuid(self, value, **kwargs):
                del kwargs
                return Found(omitted_task, f"uuid:{value}")

        reported = OccurrenceQueryService(_uow(OmitRepository(), uow.context.local_timezone), runtime=_runtime()).query(
            OccurrenceQueryRequest.from_mapping({
                "selector": {"uuids": [task["uuid"]]}, "from": "2026-08-24", "to": "2026-08-24",
                "omission_policy": "report",
            })
        ).results[0]
        self.assertFalse(reported.occurrences)
        self.assertTrue(reported.omitted_occurrences)
        limited = service.query(OccurrenceQueryRequest.from_mapping({
            "selector": {"uuids": [task["uuid"]]}, "from": "2026-08-24", "to": "2026-08-24",
            "max_total_occurrences": 1,
        })).results[0]
        self.assertEqual(limited.status, "exhausted")
        self.assertEqual(len(limited.occurrences), 1)

    def test_preserves_absent_and_retryable_unavailable_reads_without_identity(self) -> None:
        class Repository:
            def __init__(self, read): self.read = read
            def by_uuid(self, value, **kwargs):
                del kwargs
                return Absent(f"uuid:{value}", self.read.reason) if isinstance(self.read, Absent) else self.read

        request = OccurrenceQueryRequest.from_mapping({
            "selector": {"uuids": ["00000000-0000-4000-8000-000000000003"]}, "from": "2026-08-24", "count": 1,
        })
        absent = OccurrenceQueryService(_uow(Repository(Absent("uuid:missing", "no exact match"))), runtime=_runtime()).query(request)
        self.assertEqual(absent.status, "absent")
        self.assertIsNone(absent.results[0].task)
        evidence = FailureEvidence(TaskCommand(("task", "export"), "query test", 1.0), CommandFailureKind.BUSY,
                                   2, 1, 0.01, True, "Taskwarrior lock active")
        unavailable = OccurrenceQueryService(_uow(Repository(Unavailable("uuid:busy", evidence))), runtime=_runtime()).query(request)
        self.assertEqual(unavailable.status, "unavailable")
        self.assertTrue(unavailable.failure and unavailable.failure.retryable)

    def test_all_selector_excludes_ordinary_rows_and_keeps_recurrence_rows(self) -> None:
        recurrence = {"uuid": "00000000-0000-4000-8000-000000000004", "chainID": "query-all", "link": 1,
                      "description": "Recurring", "anchor": "w:mon..sun@t=08:00", "anchor_mode": "skip", "status": "pending"}
        ordinary = {"uuid": "00000000-0000-4000-8000-000000000005", "description": "Ordinary task", "status": "pending"}
        future = {"uuid": "00000000-0000-4000-8000-000000000011", "chainID": "query-future", "link": 1,
                  "description": "Far future", "anchor": "(m:1:15 + m:rand)", "due": "20300107T090000Z", "status": "pending"}
        empty = {"uuid": "00000000-0000-4000-8000-000000000012", "chainID": "query-empty", "link": 1,
                 "description": "No match", "anchor": "y:12-31", "due": "20260101T090000Z", "status": "pending"}
        class Repository:
            def broad_snapshot(self, **kwargs):
                del kwargs
                return Found(SimpleNamespace(rows=(recurrence, ordinary, future, empty)), "broad:query:all-active")
        response = OccurrenceQueryService(_uow(Repository()), runtime=_runtime()).query(OccurrenceQueryRequest.from_mapping({
            "selector": {"all_tasks": True}, "from": "2026-08-24", "to": "2026-08-31", "count": 1,
        }))
        uuids = {item.task.uuid for item in response.results if item.task}
        self.assertLessEqual(uuids, {recurrence["uuid"], future["uuid"], empty["uuid"]})
        self.assertIn(recurrence["uuid"], uuids)

    def test_batches_uuid_reads_from_one_snapshot_and_preserves_ambiguity_and_absence(self) -> None:
        class Snapshot:
            def uuid_matches(self, value):
                return ({"uuid": "ambiguous-1"}, {"uuid": "ambiguous-2"}) if value == "ambiguous" else ()
        class Repository:
            def __init__(self):
                self.broad_snapshot_calls = 0
            def broad_snapshot(self, **kwargs):
                del kwargs
                self.broad_snapshot_calls += 1
                return Found(Snapshot(), "broad:query:uuids")
            def by_uuid(self, *args, **kwargs):
                raise AssertionError("batched UUID query must not perform per-UUID reads")
        repository = Repository()
        response = OccurrenceQueryService(_uow(repository), runtime=_runtime()).query(OccurrenceQueryRequest.from_mapping({
            "selector": {"uuids": ["ambiguous", "missing"]}, "from": "2026-08-24", "count": 1,
        }))
        self.assertEqual(response.status, "invalid")
        self.assertEqual(response.results[0].failure.code, "ambiguous_uuid")
        self.assertEqual(response.results[1].status, "absent")
        self.assertEqual(repository.broad_snapshot_calls, 1)

    def test_invalid_task_schedule_becomes_per_task_failure(self) -> None:
        broken = {"uuid": "00000000-0000-4000-8000-000000000008", "chainID": "query-broken", "link": 1,
                  "description": "Broken recurrence", "anchor": "(m:1:15 + m:rand)", "anchor_mode": "skip", "status": "pending"}
        class Repository:
            def broad_snapshot(self, **kwargs):
                del kwargs
                return Found(SimpleNamespace(rows=(broken,)), "broad:query:all-active")
        result = OccurrenceQueryService(_uow(Repository()), runtime=_runtime()).query(OccurrenceQueryRequest.from_mapping({
            "selector": {"all_tasks": True}, "from": "2026-08-24", "count": 1,
        })).results[0]
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.failure.code, "task_invalid")

    def test_current_due_bounds_anchor_projections(self) -> None:
        task = {"uuid": "00000000-0000-4000-8000-000000000009", "chainID": "query-reference", "link": 4,
                "description": "Future task", "anchor": "w:mon", "anchor_mode": "all", "due": "2030-01-07T09:00:00+00:00", "status": "pending"}
        class Repository:
            def by_uuid(self, value, **kwargs):
                del kwargs
                return Found(task, f"uuid:{value}")
        result = OccurrenceQueryService(_uow(Repository()), runtime=_runtime()).query(OccurrenceQueryRequest.from_mapping({
            "selector": {"uuids": [task["uuid"]]}, "from": "2026-08-24", "to": "2026-08-31", "count": 20,
        })).results[0]
        self.assertEqual(result.status, "empty")
        self.assertEqual(result.task.current_due, task["due"])

    def test_cp_occurrences_advance_from_current_due(self) -> None:
        task = {"uuid": "00000000-0000-4000-8000-000000000010", "chainID": "query-cp", "link": 4,
                "description": "CP task", "cp": "P3D", "due": "20260821T043500Z", "status": "pending"}
        class Repository:
            def by_uuid(self, value, **kwargs):
                del kwargs
                return Found(task, f"uuid:{value}")
        result = OccurrenceQueryService(_uow(Repository()), runtime=_runtime()).query(OccurrenceQueryRequest.from_mapping({
            "selector": {"uuids": [task["uuid"]]}, "from": "2026-08-24", "to": "2026-08-31", "count": 20,
        })).results[0]
        self.assertEqual(result.status, "found")
        self.assertEqual([item.utc.date().isoformat() for item in result.occurrences], ["2026-08-24", "2026-08-27", "2026-08-30"])

    def test_next_projects_anchor_and_cp_without_mutation_and_respects_chain_bounds(self) -> None:
        anchor = {"uuid": "00000000-0000-4000-8000-000000000006", "chainID": "query-next-anchor", "link": 1,
                  "description": "Next anchor", "anchor": "w:mon..sun@t=04:30", "anchor_mode": "skip", "due": "20260824T013000Z", "status": "pending"}
        cp = {"uuid": "00000000-0000-4000-8000-000000000007", "chainID": "query-next-cp", "link": 1,
              "description": "Next cp", "cp": "1d", "due": "20260824T013000Z", "status": "pending"}
        class Repository:
            def broad_snapshot(self, **kwargs):
                del kwargs
                class Snapshot:
                    def uuid_matches(self, value): return (anchor,) if value.endswith("006") else (cp,)
                return Found(Snapshot(), "broad:query:uuids")
            def by_uuid(self, value, **kwargs):
                del kwargs
                return Found(anchor if value.endswith("006") else cp, f"uuid:{value}")
        uow = _uow(Repository(), timezone(timedelta(hours=3)))
        request = OccurrenceQueryRequest.from_mapping({"operation": "next", "selector": {"uuids": [anchor["uuid"], cp["uuid"]]}, "from": "2026-08-24", "count": 1})
        anchor_before = dict(anchor)
        cp_before = dict(cp)
        response = OccurrenceQueryService(uow, runtime=_runtime()).query_next(request)
        self.assertEqual(response.schema, "nautical.query.next")
        self.assertEqual(response.status, "found")
        self.assertEqual([item.occurrences[0].source for item in response.results], ["anchor", "cp"])
        self.assertEqual(response.results[0].chain["chainID"], anchor["chainID"])
        self.assertFalse(response.results[0].lifecycle["child_created"])
        self.assertEqual(response.results[1].lifecycle["reference_field"], "due")
        self.assertEqual(anchor, anchor_before)
        self.assertEqual(cp, cp_before)
        for field, value in (("chainMax", "1"), ("chainUntil", "2026-08-24T01:00:00Z")):
            bounded = dict(anchor, **{field: value})
            class BoundedRepository:
                def by_uuid(self, selected, **kwargs):
                    del kwargs
                    return Found(bounded, f"uuid:{selected}")
            bounded_result = OccurrenceQueryService(_uow(BoundedRepository(), uow.context.local_timezone), runtime=_runtime()).query_next(
                OccurrenceQueryRequest.from_mapping({"operation": "next", "selector": {"uuids": [anchor["uuid"]]}, "from": "2026-08-24", "count": 1})
            ).results[0]
            self.assertEqual(bounded_result.status, "empty", field)

    def test_next_daily_skip_mode_reports_selected_slot_and_progress(self) -> None:
        task = {"uuid": "00000000-0000-4000-8000-000000000013", "chainID": "query-daily-progress", "link": 1,
                "description": "Three daily slots", "anchor": "w:mon..sun@t=09:00,12:00,18:00", "anchor_mode": "skip", "due": "20260824T060000Z", "status": "pending"}
        class Repository:
            def by_uuid(self, value, **kwargs):
                del kwargs
                return Found(task, f"uuid:{value}")
        result = OccurrenceQueryService(_uow(Repository(), timezone(timedelta(hours=3))), runtime=_runtime()).query_next(
            OccurrenceQueryRequest.from_mapping({"operation": "next", "selector": {"uuids": [task["uuid"]]}, "at": "2026-08-24T15:00:00+03:00"})
        ).results[0]
        self.assertEqual(result.status, "found")
        self.assertEqual(result.occurrences[0].local.hour, 18)
        self.assertEqual(result.lifecycle["daily_instances"], {"date": "2026-08-24", "total": 3, "current_position": 1, "missed": 1, "upcoming": 1})
        self.assertEqual(result.lifecycle["missed_occurrences"], ["2026-08-24T12:00:00+03:00"])


if __name__ == "__main__":
    unittest.main()
