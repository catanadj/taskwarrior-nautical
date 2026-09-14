"""Public query model contract tests."""

import json
import unittest
from datetime import datetime, timedelta, timezone

from nautical_core.query_models import (
    OccurrenceQueryRequest,
    OccurrenceQueryResponse,
    OccurrenceRecord,
    QueryContractError,
    QueryFailure,
    QuerySelector,
    TaskIdentity,
    TaskOccurrenceResult,
)


class QueryModelContractsTests(unittest.TestCase):
    def test_public_models_round_trip_and_reject_invalid_shapes(self) -> None:
        request = OccurrenceQueryRequest.from_mapping(
            {
                "version": 1,
                "operation": "occurrences",
                "selector": {"uuids": ["00000000-0000-4000-8000-000000000001"]},
                "from": "2026-08-21",
                "to": "2026-08-22",
                "omission_policy": "report",
                "max_occurrences": "20",
            }
        )
        self.assertEqual(OccurrenceQueryRequest.from_mapping(request.to_dict()), request)
        self.assertEqual(request.max_occurrences, 20)

        duplicate_selector = OccurrenceQueryRequest.from_mapping(
            {
                "selector": {"uuids": ["SAME", "same"]},
                "from": "2026-08-21",
                "count": 1,
                "max_total_occurrences": 2,
            }
        )
        self.assertEqual(duplicate_selector.selector.uuids, ("same",))
        self.assertEqual(duplicate_selector.max_total_occurrences, 2)

        local = datetime(2026, 8, 21, 4, 30, tzinfo=timezone(timedelta(hours=3)))
        identity = TaskIdentity(
            uuid="00000000-0000-4000-8000-000000000001",
            chain_id="query-chain",
            link=4,
            description="Morning task \N{SNOWMAN}",
            recurrence_kind="anchor",
            expression="w:mon..sun@t=04:30",
            schedule_fingerprint="schedule-v1",
        )
        occurrence = OccurrenceRecord(
            local=local,
            utc=local.astimezone(timezone.utc),
            timezone="Europe/Bucharest",
            source="anchor",
        )
        response = OccurrenceQueryResponse(
            request=request,
            timezone="Europe/Bucharest",
            results=(TaskOccurrenceResult(identity, "found", (occurrence,)),),
            status="found",
            configuration_fingerprint="config-v1",
        )
        encoded = json.dumps(response.to_dict(), ensure_ascii=False, sort_keys=True)
        self.assertNotIn("\\u2603", encoded)
        self.assertNotIn("snowman", encoded.lower())
        self.assertIn("\N{SNOWMAN}", encoded)

        absent = TaskOccurrenceResult(
            None,
            "absent",
            failure=QueryFailure("task_absent", "task was not found", task_uuid=identity.uuid),
        )
        self.assertIsNone(absent.to_dict()["task"])

        next_request = OccurrenceQueryRequest.from_mapping(
            {
                "operation": "next",
                "selector": {"all_tasks": True},
                "from": "2026-08-21",
                "count": 1,
            }
        )
        self.assertEqual(
            OccurrenceQueryResponse(next_request, "Europe/Bucharest").to_dict()["schema"],
            "nautical.query.next",
        )
        evaluated_request = OccurrenceQueryRequest.from_mapping(
            {
                "operation": "next",
                "selector": {"all_tasks": True},
                "at": "2026-08-21T15:00:00+03:00",
            }
        )
        self.assertEqual(
            OccurrenceQueryRequest.from_mapping(evaluated_request.to_dict()),
            evaluated_request,
        )

        invalid_cases = (
            lambda: QuerySelector.from_mapping({"all_tasks": True, "chain_id": "query-chain"}),
            lambda: OccurrenceQueryRequest.from_mapping(
                {
                    "selector": {"all_tasks": True},
                    "from": "2026-08-21T04:30:00",
                    "count": 1,
                }
            ),
            lambda: OccurrenceQueryRequest.from_mapping(
                {"selector": {"all_tasks": True}, "from": "2026-08-22", "to": "2026-08-21"}
            ),
            lambda: OccurrenceQueryRequest.from_mapping(
                {"operation": "next", "selector": {"all_tasks": True}, "at": "2026-08-21"}
            ),
            lambda: OccurrenceQueryRequest.from_mapping(
                {
                    "operation": "next",
                    "selector": {"all_tasks": True},
                    "at": "2026-08-21T15:00:00+03:00",
                    "from": "2026-08-21",
                }
            ),
            lambda: QueryFailure("bad", "bad", details=()),
        )
        for make_invalid in invalid_cases:
            with self.subTest(make_invalid=make_invalid), self.assertRaises(QueryContractError):
                make_invalid()


if __name__ == "__main__":
    unittest.main()
