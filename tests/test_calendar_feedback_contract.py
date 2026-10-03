"""Contracts for optional business-calendar feedback."""

from __future__ import annotations

from datetime import datetime
import unittest

from nautical_core.calendar_feedback import render_business_calendar_displacement


class CalendarFeedbackContractTests(unittest.TestCase):
    def test_expected_calendar_feedback_conversion_failure_is_omitted(self) -> None:
        class Core:
            @staticmethod
            def to_local(_value: datetime) -> datetime:
                raise ValueError("datetime is outside local timezone range")

        self.assertFalse(
            render_business_calendar_displacement(
                {"bc": "custom"},
                datetime(2026, 10, 3),
                core=Core(),
                panel=lambda *_args, **_kwargs: None,
            )
        )

    def test_unexpected_calendar_feedback_failure_propagates(self) -> None:
        class Core:
            @staticmethod
            def to_local(_value: datetime) -> datetime:
                raise RuntimeError("timezone adapter failed")

        with self.assertRaisesRegex(RuntimeError, "timezone adapter failed"):
            render_business_calendar_displacement(
                {"bc": "custom"},
                datetime(2026, 10, 3),
                core=Core(),
                panel=lambda *_args, **_kwargs: None,
            )


if __name__ == "__main__":
    unittest.main()
