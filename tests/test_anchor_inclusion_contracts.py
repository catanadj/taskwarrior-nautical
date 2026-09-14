"""Direct contracts for merging recurrence and anchor-file occurrences."""

import unittest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import nautical_core as core
from nautical_core import anchor_inclusion, anchor_omit
from nautical_core.occurrence_provider import Occurrence


class AnchorInclusionContractTests(unittest.TestCase):
    def test_same_instant_anchor_file_occurrence_keeps_file_description(self) -> None:
        target = datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc)

        class Provider:
            def next_after(self, *_args, **_kwargs):
                return Occurrence(
                    target.date(), 9, 0, source="anchor_file",
                    description="watering", local_datetime=target,
                )

        result = anchor_inclusion.next_included_occurrence(
            dnf=object(),
            anchor_file_str="calendar.csv@t=09:00",
            after_local_dt=target - timedelta(minutes=1),
            inclusive=False,
            fallback_hhmm=(9, 0),
            default_seed_date=target.date(),
            seed_base="tie-test",
            omit_dnf=None,
            core=core,
            next_occurrence_after_local_dt=lambda *_args, **_kwargs: target,
            anchor_file_provider=Provider(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.source, "anchor_file")
        self.assertEqual(result.description, "watering")

    def test_omission_scheduler_failure_does_not_fail_open(self) -> None:
        with patch.object(
            core,
            "next_after_expr",
            side_effect=RuntimeError("scheduler unavailable"),
        ):
            with self.assertRaisesRegex(ValueError, "Unable to evaluate omit rule"):
                anchor_omit.omit_expr_fires_on_date(
                    [[{"kind": "w", "value": "mon", "mods": {}}]],
                    date(2026, 8, 3),
                    date(2026, 8, 1),
                    "omit-fail-closed",
                    core=core,
                )

    def test_scheduler_callback_failure_is_not_retried_or_hidden(self) -> None:
        calls = []

        def scheduler(_dnf, after, **_kwargs):
            calls.append(after)
            raise TypeError("internal scheduler defect")

        with self.assertRaisesRegex(TypeError, "internal scheduler defect"):
            anchor_inclusion.next_included_occurrence(
                dnf=[[]],
                anchor_file_str="",
                after_local_dt=datetime(2026, 8, 3, 9),
                inclusive=False,
                fallback_hhmm=(9, 0),
                default_seed_date=date(2026, 8, 3),
                seed_base="dispatch-contract",
                omit_dnf=None,
                core=core,
                next_occurrence_after_local_dt=scheduler,
            )

        self.assertEqual(calls, [datetime(2026, 8, 3, 9)])

    def test_omission_scan_fails_closed_at_configured_bound(self) -> None:
        after = datetime(2026, 8, 2, 9)

        class DailyProvider:
            def next_after(self, cursor, **_kwargs):
                value = cursor + timedelta(days=1)
                return Occurrence(
                    value.date(), value.hour, value.minute, local_datetime=value
                )

        with patch.object(anchor_inclusion, "_anchor_file_occurrence_is_omitted", return_value=True):
            with self.assertRaisesRegex(ValueError, "omission scan exceeded 2 occurrences"):
                anchor_inclusion.next_included_occurrence(
                    dnf=None,
                    anchor_file_str="calendar.csv",
                    after_local_dt=after,
                    inclusive=False,
                    fallback_hhmm=(9, 0),
                    default_seed_date=after.date(),
                    seed_base="omission-bound-contract",
                    omit_dnf=None,
                    core=core,
                    next_occurrence_after_local_dt=lambda *_args, **_kwargs: None,
                    anchor_file_provider=DailyProvider(),
                    max_file_skips=2,
                )

    def test_omit_evaluation_failure_is_reported_as_unavailable(self) -> None:
        with patch.object(
            anchor_omit,
            "omit_expr_fires_on_date",
            side_effect=RuntimeError("broken omit evaluator"),
        ):
            with self.assertRaisesRegex(ValueError, "Unable to evaluate omit rule") as raised:
                anchor_inclusion._anchor_file_occurrence_is_omitted(
                    datetime(2026, 8, 3, 9),
                    omit_dnf=[["omit"]],
                    default_seed_date=date(2026, 8, 3),
                    seed_base="omit-failure-contract",
                    core=core,
                )

        self.assertIsInstance(raised.exception.__cause__, RuntimeError)


if __name__ == "__main__":
    unittest.main()
