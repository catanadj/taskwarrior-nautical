from __future__ import annotations

import unittest
from datetime import date

from nautical_core.scheduler_models import OccurrenceSearchExhausted
from nautical_core.time_projection import (
    ProjectedTime,
    ProjectionInvalid,
    ProjectionTerminal,
    ProjectionUnavailable,
    TimeProjectionService,
)


class TimeProjectionContractTests(unittest.TestCase):
    selected = date(2026, 10, 6)

    def test_projects_two_and_three_component_slots_without_changing_date(self) -> None:
        service = TimeProjectionService(resolver=lambda *_args, **_kwargs: [(9, 30), (1, 10, 45)])

        result = service.project({"t": "09:30"}, self.selected)

        self.assertIsInstance(result, ProjectedTime)
        assert isinstance(result, ProjectedTime)
        self.assertEqual(result.selected_date, self.selected)
        self.assertEqual(result.slots, ((0, 9, 30), (1, 10, 45)))

    def test_missing_time_modifier_uses_default_projection(self) -> None:
        result = TimeProjectionService(resolver=lambda *_args, **_kwargs: []).project(
            {}, self.selected
        )

        self.assertIsInstance(result, ProjectedTime)
        assert isinstance(result, ProjectedTime)
        self.assertEqual(result.source, "default")
        self.assertEqual(result.slots, ())

    def test_empty_or_malformed_resolver_output_is_invalid(self) -> None:
        empty = TimeProjectionService(resolver=lambda *_args, **_kwargs: []).project(
            {"t": "09:30"}, self.selected
        )
        malformed = TimeProjectionService(resolver=lambda *_args, **_kwargs: [("bad", 30)]).project(
            {"t": "09:30"}, self.selected
        )

        self.assertIsInstance(empty, ProjectionInvalid)
        self.assertEqual(empty.error_type, "EmptyProjection")
        self.assertIsInstance(malformed, ProjectionInvalid)
        self.assertEqual(malformed.error_type, "ValueError")

    def test_lookup_and_io_failures_are_unavailable(self) -> None:
        for error in (LookupError("calendar missing"), OSError("calendar unreadable")):
            with self.subTest(error=type(error).__name__):
                result = TimeProjectionService(
                    resolver=lambda *_args, error=error, **_kwargs: (_ for _ in ()).throw(error)
                ).project({"t": "sunrise"}, self.selected)
                self.assertIsInstance(result, ProjectionUnavailable)
                assert isinstance(result, ProjectionUnavailable)
                self.assertEqual(result.error_type, type(error).__name__)

    def test_scheduler_exhaustion_is_terminal(self) -> None:
        exhausted = OccurrenceSearchExhausted("no occurrence")
        result = TimeProjectionService(
            resolver=lambda *_args, **_kwargs: (_ for _ in ()).throw(exhausted)
        ).project({"t": "sunrise"}, self.selected)

        self.assertIsInstance(result, ProjectionTerminal)
        assert isinstance(result, ProjectionTerminal)
        self.assertIs(result.error, exhausted)


if __name__ == "__main__":
    unittest.main()
