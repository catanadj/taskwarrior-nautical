import unittest
from datetime import timezone
from types import SimpleNamespace

from nautical_core.query_models import OccurrenceQueryRequest
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


if __name__ == "__main__":
    unittest.main()
