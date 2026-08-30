import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from nautical_core.hint_builder import HintBuilder
from nautical_core.occurrence_outcomes import OccurrenceCollectionResult, UnavailableOccurrence
from nautical_core.occurrence_provider import Occurrence
from nautical_core.scheduler_cursor import OccurrenceCursor
from nautical_core.scheduler_models import OccurrenceSearchExhausted


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

    def test_next_only_uses_bounded_pages_and_strict_cursor(self) -> None:
        timezone_utc = timezone.utc
        values = [
            self._result([2]).occurrences[0],
            Occurrence(date(2026, 1, 2), 18, 0, local_datetime=datetime(2026, 1, 2, 18, tzinfo=timezone_utc)),
            Occurrence(date(2026, 1, 3), 9, 0, local_datetime=datetime(2026, 1, 3, 9, tzinfo=timezone_utc)),
        ]
        calls = []

        class Service:
            session = SimpleNamespace(
                evaluator=SimpleNamespace(context=SimpleNamespace(timezone=timezone_utc), kind="anchor")
            )

            def collect_request(self, request):
                calls.append(request)
                offset = 0 if len(calls) == 1 else 2
                batch = tuple(values[offset : offset + (2 if len(calls) == 1 else 1)])
                terminal = (
                    OccurrenceSearchExhausted("test", reference=request.cursor.local_datetime, limit=24)
                    if len(calls) == 2 else None
                )
                return OccurrenceCollectionResult(batch, request.cursor, terminal=terminal)

        builder = HintBuilder(Service())
        hints = builder.build(
            start_dt=None,
            k_next=2,
            sample_days_for_year=366,
            now_local=lambda: datetime(2026, 1, 1, tzinfo=timezone_utc),
            include_per_year=False,
        )
        self.assertEqual(hints["next_dates"], ["2026-01-02T00:00", "2026-01-03T00:00"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].limit, 24)
        self.assertGreater(calls[1].cursor.local_datetime, calls[0].cursor.local_datetime)

    def test_next_only_propagates_typed_scheduler_failure(self) -> None:
        timezone_utc = timezone.utc

        class Service:
            session = SimpleNamespace(
                evaluator=SimpleNamespace(context=SimpleNamespace(timezone=timezone_utc), kind="anchor")
            )

            def collect_request(self, request):
                return OccurrenceCollectionResult(
                    (), request.cursor, failure=UnavailableOccurrence("astronomy unavailable", "LookupError")
                )

        builder = HintBuilder(Service())
        with self.assertRaisesRegex(RuntimeError, "astronomy unavailable"):
            builder.build(
                start_dt=None,
                k_next=1,
                sample_days_for_year=366,
                now_local=lambda: datetime(2026, 1, 1, tzinfo=timezone_utc),
                include_per_year=False,
            )

    def test_next_only_enforces_occurrence_cap(self) -> None:
        timezone_utc = timezone.utc
        occurrence = Occurrence(
            date(2026, 1, 2),
            9,
            0,
            local_datetime=datetime(2026, 1, 2, 9, tzinfo=timezone_utc),
        )
        calls = []

        class Service:
            session = SimpleNamespace(
                evaluator=SimpleNamespace(context=SimpleNamespace(timezone=timezone_utc), kind="anchor")
            )

            def collect_request(self, request):
                calls.append(request)
                return OccurrenceCollectionResult((occurrence,), request.cursor)

        builder = HintBuilder(Service())
        result = builder._collect_next_only(date(2026, 1, 1), date(2031, 1, 1))
        self.assertEqual(len(result.occurrences), 384)
        self.assertEqual(len(calls), 384)
        self.assertEqual(calls[-1].limit, 1)


if __name__ == "__main__":
    unittest.main()
