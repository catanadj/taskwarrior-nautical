"""Contracts for optional business-calendar feedback."""

from __future__ import annotations

from datetime import datetime
from typing import get_type_hints
import unittest

from nautical_core.calendar_feedback import render_business_calendar_displacement
from nautical_core.modify_models import PanelCallback


class CalendarFeedbackContractTests(unittest.TestCase):
    def test_add_preview_ports_use_calendar_feedback_callback_contract(self) -> None:
        from typing import get_type_hints

        from nautical_core.add_anchor_preview import (
            AnchorExpressionPreviewServices,
            AnchorFilePreviewServices,
        )
        from nautical_core.add_anchor_preview import AddCalendarFeedbackCallback

        self.assertIs(
            get_type_hints(AnchorExpressionPreviewServices)[
                "render_business_calendar_displacement"
            ],
            AddCalendarFeedbackCallback,
        )
        self.assertIs(
            get_type_hints(AnchorFilePreviewServices)[
                "render_business_calendar_displacement"
            ],
            AddCalendarFeedbackCallback,
        )

    def test_calendar_feedback_uses_shared_panel_callback_contract(self) -> None:
        self.assertIs(
            get_type_hints(render_business_calendar_displacement)["panel"],
            PanelCallback,
        )

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
