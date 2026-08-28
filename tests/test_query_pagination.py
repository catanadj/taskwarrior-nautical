import unittest
from datetime import timezone
from types import SimpleNamespace

from nautical_core.query_models import OccurrenceQueryRequest, OccurrenceQueryResponse
from nautical_core.operator_models import OperatorCursor
from nautical_core.query_service import OccurrenceQueryService, QueryServiceError


class QueryPaginationTests(unittest.TestCase):
    def _service(self) -> OccurrenceQueryService:
        service = object.__new__(OccurrenceQueryService)
        service._timezone = timezone.utc
        service._uow = SimpleNamespace(
            mutation_epoch=0,
            context=SimpleNamespace(configuration=SimpleNamespace(fingerprint="config-1")),
        )
        return service

    @staticmethod
    def _request(*, cursor=None, max_tasks=2) -> OccurrenceQueryRequest:
        value = {
            "selector": {"all_tasks": True},
            "from": "2026-08-24",
            "count": 1,
            "max_tasks": max_tasks,
        }
        if cursor is not None:
            value["cursor"] = cursor.to_dict()
        return OccurrenceQueryRequest.from_mapping(value)

    def test_pages_are_complete_and_resume_without_overlap(self) -> None:
        service = self._service()
        rows = tuple(SimpleNamespace(uuid=f"task-{index}") for index in range(5))
        first, cursor, complete = service._page_rows(rows, self._request())
        self.assertEqual(tuple(row.uuid for row in first), ("task-0", "task-1"))
        self.assertFalse(complete)
        self.assertIsNotNone(cursor)

        second, next_cursor, complete = service._page_rows(rows, self._request(cursor=cursor))
        self.assertEqual(tuple(row.uuid for row in second), ("task-2", "task-3"))
        self.assertFalse(complete)
        self.assertIsNotNone(next_cursor)

        final, next_cursor, complete = service._page_rows(rows, self._request(cursor=next_cursor))
        self.assertEqual(tuple(row.uuid for row in final), ("task-4",))
        self.assertTrue(complete)
        self.assertIsNone(next_cursor)

    def test_cursor_rejects_changed_snapshot(self) -> None:
        service = self._service()
        rows = tuple(SimpleNamespace(uuid=f"task-{index}") for index in range(3))
        _, cursor, _ = service._page_rows(rows, self._request())
        changed = rows + (SimpleNamespace(uuid="task-new"),)
        with self.assertRaises(QueryServiceError):
            service._page_rows(changed, self._request(cursor=cursor))

    def test_single_page_has_no_cursor(self) -> None:
        service = self._service()
        rows = (SimpleNamespace(uuid="task-0"),)
        page, cursor, complete = service._page_rows(rows, self._request())
        self.assertEqual(len(page), 1)
        self.assertTrue(complete)
        self.assertIsNone(cursor)

    def test_empty_scope_is_a_complete_page(self) -> None:
        service = self._service()
        page, cursor, complete = service._page_rows((), self._request())
        self.assertEqual(page, ())
        self.assertTrue(complete)
        self.assertIsNone(cursor)

    def test_exact_maximum_has_no_continuation(self) -> None:
        service = self._service()
        rows = tuple(SimpleNamespace(uuid=f"task-{index}") for index in range(2))
        page, cursor, complete = service._page_rows(rows, self._request(max_tasks=2))
        self.assertEqual(len(page), 2)
        self.assertTrue(complete)
        self.assertIsNone(cursor)

    def test_maximum_plus_one_emits_one_continuation(self) -> None:
        service = self._service()
        rows = tuple(SimpleNamespace(uuid=f"task-{index}") for index in range(3))
        page, cursor, complete = service._page_rows(rows, self._request(max_tasks=2))
        self.assertEqual(tuple(row.uuid for row in page), ("task-0", "task-1"))
        self.assertFalse(complete)
        self.assertIsNotNone(cursor)
        tail, tail_cursor, tail_complete = service._page_rows(rows, self._request(cursor=cursor))
        self.assertEqual(tuple(row.uuid for row in tail), ("task-2",))
        self.assertTrue(tail_complete)
        self.assertIsNone(tail_cursor)

    def test_response_envelope_exposes_incomplete_cursor(self) -> None:
        request = self._request(max_tasks=2)
        cursor = OperatorCursor("snapshot", "config-1", "0", position=2, page_size=2)
        response = OccurrenceQueryResponse(
            request=request,
            timezone="UTC",
            status="empty",
            cursor=cursor,
            complete=False,
            coverage={"kind": "bounded", "observed": ["task-0"], "omitted_count": 1},
        )
        payload = response.to_dict()
        self.assertFalse(payload["pagination"]["complete"])
        self.assertEqual(payload["pagination"]["cursor"]["position"], 2)
        self.assertEqual(payload["coverage"]["kind"], "bounded")
        with self.assertRaises(ValueError):
            OccurrenceQueryResponse(request=request, timezone="UTC", cursor=cursor, complete=True)

    def test_many_chain_page_preserves_all_identity_rows(self) -> None:
        service = self._service()
        rows = tuple(
            SimpleNamespace(uuid=uuid)
            for uuid in ("chain-b-task", "chain-a-task", "chain-c-task")
        )
        page, cursor, complete = service._page_rows(rows, self._request(max_tasks=10))
        self.assertEqual(tuple(row.uuid for row in page), ("chain-b-task", "chain-a-task", "chain-c-task"))
        self.assertTrue(complete)
        self.assertIsNone(cursor)


if __name__ == "__main__":
    unittest.main()
