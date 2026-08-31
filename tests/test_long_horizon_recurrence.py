"""Retained long-horizon recurrence qualification fixtures."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

from nautical_core.scheduler_cursor import OccurrenceCursor
from nautical_core.scheduler_service import SchedulerService
from nautical_core.task_models import TaskObservation


FIXTURES = (
    ("anchor-weekly", {"anchor": "w:mon"}, 200),
    ("anchor-multi-time", {"anchor": "w:mon..sun@t=09:00,18:00"}, 1600),
)


def _signatures(service: SchedulerService, *, limit: int) -> tuple[str, ...]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2028, 1, 1, tzinfo=timezone.utc)
    result = service.collect(
        OccurrenceCursor.inclusive_at(start, timezone=timezone.utc, date_limit=end.date()),
        limit=limit,
        max_iterations=max(512, limit * 3),
    )
    values = tuple(item.local_datetime.astimezone(timezone.utc).isoformat() for item in result.occurrences)
    assert all(left < right for left, right in zip(values, values[1:]))
    return values


class LongHorizonRecurrenceTests(unittest.TestCase):
    def test_retained_fixtures_are_monotonic_and_repeatable(self) -> None:
        for index, (name, recurrence, limit) in enumerate(FIXTURES, start=1):
            base = {
                "uuid": f"00000000-0000-4000-8000-{index:012d}",
                "chainID": f"horizon-{name}",
                "link": 1,
                "status": "pending",
                "description": name,
            }
            base.update(recurrence)
            first = SchedulerService.from_observation(
                TaskObservation.from_mapping(base, source_query="long-horizon-fixture")
            )
            second = SchedulerService.from_observation(
                TaskObservation.from_mapping(base, source_query="long-horizon-fixture")
            )
            first_values = _signatures(first, limit=limit)
            second_values = _signatures(second, limit=limit)
            self.assertEqual(first_values, second_values, name)
            self.assertTrue(first_values, name)
            self.assertLessEqual(len(first_values), limit, name)

    def test_cp_fixture_projects_a_730_day_horizon_deterministically(self) -> None:
        base = datetime(2026, 1, 1, 9, tzinfo=timezone.utc)
        values = []
        previous = base
        for link in range(1, 731):
            observation = TaskObservation.from_mapping(
                {
                    "uuid": f"00000000-0000-4000-8000-{link:012d}",
                    "chainID": "horizon-cp-daily",
                    "link": link,
                    "status": "pending",
                    "cp": "P1D",
                },
                source_query="long-horizon-fixture",
            )
            service = SchedulerService.from_observation(observation)
            previous = service.session.evaluator.project_cp(previous, 1)
            values.append(previous)
        self.assertEqual(values[-1], datetime(2028, 1, 1, 9, tzinfo=timezone.utc))
        self.assertTrue(all(left < right for left, right in zip(values, values[1:])))


if __name__ == "__main__":
    unittest.main()
