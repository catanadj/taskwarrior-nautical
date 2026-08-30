import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from nautical_core.hint_builder import HintBuilder
from nautical_core.occurrence_outcomes import OccurrenceCollectionResult
from nautical_core.occurrence_provider import Occurrence
from nautical_core.scheduler_cursor import OccurrenceCursor


class HintBuilderTests(unittest.TestCase):
    def _result(self, days: list[int], *, limit: int = 384) -> OccurrenceCollectionResult:
        values = tuple(
            Occurrence(
                date(2026, 1, 1) + timedelta(days=day - 1),
                9,
                0,
                local_datetime=datetime(2026, 1, 1, 9, tzinfo=timezone.utc) + timedelta(days=day - 1),
            )
            for day in days
        )
        cursor = OccurrenceCursor.strict_after(datetime(2026, 1, 1, tzinfo=timezone.utc), timezone=timezone.utc)
        return OccurrenceCollectionResult(values, cursor)

    def test_complete_preview_is_reused_for_annual_estimate(self) -> None:
        builder = HintBuilder(SimpleNamespace(session=SimpleNamespace(evaluator=SimpleNamespace(context=SimpleNamespace(timezone=timezone.utc)))))
        preview = self._result([2, 3])
        with patch.object(HintBuilder, "_collect_dates", return_value=preview) as collect:
            hints = builder.build(
                start_dt=None,
                k_next=2,
                sample_days_for_year=366,
                now_local=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        self.assertEqual(collect.call_count, 1)
        self.assertEqual(hints["per_year"]["est"], 2)

    def test_capped_preview_keeps_independent_annual_query(self) -> None:
        builder = HintBuilder(SimpleNamespace(session=SimpleNamespace(evaluator=SimpleNamespace(context=SimpleNamespace(timezone=timezone.utc)))))
        capped = self._result(list(range(1, 385)))
        annual = self._result([2])
        with patch.object(HintBuilder, "_collect_dates", side_effect=(capped, annual)) as collect:
            hints = builder.build(
                start_dt=None,
                k_next=24,
                sample_days_for_year=366,
                now_local=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        self.assertEqual(collect.call_count, 2)
        self.assertEqual(hints["per_year"]["est"], 1)


if __name__ == "__main__":
    unittest.main()
