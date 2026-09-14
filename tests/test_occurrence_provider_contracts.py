"""Direct contracts for typed occurrence values and provider adapters."""

from datetime import date, datetime, timedelta, timezone
import unittest

from nautical_core.anchor_files import AnchorFileOccurrenceProvider
from nautical_core.occurrence_provider import (
    AnchorEventOccurrenceProvider,
    AnchorOccurrenceProvider,
    Occurrence,
    ProviderCapabilities,
    ProviderContract,
    collect_after,
)
from nautical_core.scheduler_models import OccurrenceSearchExhausted


class OccurrenceProviderContractTests(unittest.TestCase):
    def test_collection_preserves_valid_prefix_when_date_limit_ends_stream(self) -> None:
        first = datetime(2026, 1, 5, 9, 0)
        terminal = OccurrenceSearchExhausted(
            "test stream", reference=date(9999, 1, 1), limit=1
        )
        calls = 0

        def next_after(_cursor):
            nonlocal calls
            calls += 1
            if calls == 1:
                return Occurrence(first.date(), 9, 0, local_datetime=first)
            raise terminal

        collected = collect_after(
            AnchorOccurrenceProvider(next_after),
            datetime(2026, 1, 1, 9, 0),
            limit=3,
            build_local_datetime=lambda day, hhmm: datetime.combine(day, hhmm),
            to_local=lambda value: value,
        )

        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0].local_datetime, first)
        self.assertIs(collected.terminal, terminal)
    def test_provider_contract_advertises_only_certified_capabilities(self) -> None:
        ordinary = ProviderContract(source="anchor")
        self.assertEqual(ordinary.capabilities, ProviderCapabilities())

        file_provider = AnchorFileOccurrenceProvider(None, None, (9, 0))
        self.assertTrue(file_provider.contract.capabilities.cursor_reuse)
        self.assertTrue(file_provider.contract.capabilities.batch_generation)
        self.assertFalse(file_provider.contract.capabilities.arithmetic_counting)

    def test_adapters_preserve_source_and_description_metadata(self) -> None:
        after = datetime(2026, 8, 3, 9, 0)
        ordinary = AnchorOccurrenceProvider(
            lambda value: value + timedelta(hours=1),
            source="anchor+anchor_file",
            description="merged source",
        )
        event = AnchorEventOccurrenceProvider(
            lambda value: (value + timedelta(hours=1), False),
            source="anchor_file",
            description="calendar entry",
        )
        typed = AnchorOccurrenceProvider(
            lambda value: Occurrence(
                value.date(),
                value.hour + 1,
                value.minute,
                source="astronomy",
                description="sunrise",
                local_datetime=value + timedelta(hours=1),
            )
        )

        actual = [
            provider.next_after(
                after,
                build_local_datetime=lambda day, hhmm: datetime.combine(day, hhmm),
                to_local=lambda value: value,
            )
            for provider in (ordinary, event, typed)
        ]

        self.assertEqual(
            [(item.source, item.description) for item in actual if item is not None],
            [
                ("anchor+anchor_file", "merged source"),
                ("anchor_file", "calendar entry"),
                ("astronomy", "sunrise"),
            ],
        )

    def test_adapters_reject_malformed_callback_payloads(self) -> None:
        after = datetime(2026, 8, 3, 9, 0)
        kwargs = {
            "build_local_datetime": lambda day, hhmm: datetime.combine(day, hhmm),
            "to_local": lambda value: value,
        }
        for payload, expected in (
            ((after, False, "extra"), "tuple"),
            ([after, False], "tuple"),
            (("not-a-datetime", False), "non-datetime value"),
        ):
            with self.subTest(payload=payload), self.assertRaisesRegex(TypeError, expected):
                AnchorEventOccurrenceProvider(
                    lambda _value, payload=payload: payload
                ).next_after(after, **kwargs)

        with self.assertRaisesRegex(TypeError, "non-datetime value"):
            AnchorOccurrenceProvider(lambda _value: "not-a-datetime").next_after(
                after, **kwargs
            )

    def test_adapters_reject_non_advancing_values(self) -> None:
        after = datetime(2026, 8, 3, 9, 0)
        for provider in (
            AnchorOccurrenceProvider(lambda value: value),
            AnchorEventOccurrenceProvider(lambda value: (value, False)),
        ):
            with self.subTest(provider=provider), self.assertRaisesRegex(
                ValueError, "non-advancing"
            ):
                provider.next_after(
                    after,
                    build_local_datetime=lambda day, hhmm: datetime.combine(day, hhmm),
                    to_local=lambda value: value,
                )

    def test_occurrence_rejects_invalid_or_inconsistent_fields(self) -> None:
        invalid_values = (
            lambda: Occurrence(date(2026, 8, 3), 24, 0),
            lambda: Occurrence(date(2026, 8, 3), 9, 60),
            lambda: Occurrence(
                date(2026, 8, 3), 9, 0, local_datetime=datetime(2026, 8, 3, 10)
            ),
            lambda: Occurrence(date(2026, 8, 3), 9, 0, source=object()),
            lambda: Occurrence(date(2026, 8, 3), 9, 0, description=object()),
        )
        for factory in invalid_values:
            with self.subTest(factory=factory), self.assertRaises((TypeError, ValueError)):
                factory()

    def test_event_adapter_requires_boolean_omitted_flag(self) -> None:
        provider = AnchorEventOccurrenceProvider(
            lambda value: (value + timedelta(hours=1), "false")
        )

        with self.assertRaisesRegex(TypeError, "non-boolean"):
            provider.next_after(
                datetime(2026, 8, 3, 9, 0),
                build_local_datetime=lambda day, hhmm: datetime.combine(day, hhmm),
                to_local=lambda value: value,
            )

    def test_inclusive_collection_moves_cursor_back_by_instant(self) -> None:
        from zoneinfo import ZoneInfo

        zone = ZoneInfo("Europe/Bucharest")
        after = datetime(2026, 10, 25, 3, 15, tzinfo=zone, fold=1)
        seen = []

        class Echo:
            def next_after(self, cursor, **_kwargs):
                seen.append(cursor)
                return Occurrence(
                    after.date(), after.hour, after.minute, local_datetime=after
                )

        collect_after(
            Echo(),
            after,
            limit=1,
            inclusive=True,
            build_local_datetime=lambda day, hhmm: datetime.combine(day, hhmm),
            to_local=lambda value: value,
        )

        self.assertEqual(
            seen[0].astimezone(timezone.utc),
            after.astimezone(timezone.utc) - timedelta(microseconds=1),
        )

    def test_collection_fails_closed_on_invalid_lazy_values_and_exhaustion(self) -> None:
        after = datetime(2026, 8, 3, 9, 0)

        class MissingLocal:
            def next_after(self, *args, **kwargs):
                return Occurrence(date(2026, 8, 3), 10, 0)

        with self.assertRaisesRegex(ValueError, "no local datetime"):
            collect_after(
                MissingLocal(),
                after,
                limit=1,
                max_iterations=2,
                build_local_datetime=lambda day, hhmm: datetime.combine(day, hhmm),
                to_local=lambda value: value,
            )

        class NeverIncluded:
            def __init__(self):
                self.value = after

            def next_after(self, *args, **kwargs):
                self.value += timedelta(hours=1)
                return Occurrence(
                    self.value.date(),
                    self.value.hour,
                    self.value.minute,
                    local_datetime=self.value,
                    omitted=True,
                )

        with self.assertRaisesRegex(ValueError, "iteration limit"):
            collect_after(
                NeverIncluded(),
                after,
                limit=1,
                max_iterations=2,
                build_local_datetime=lambda day, hhmm: datetime.combine(day, hhmm),
                to_local=lambda value: value,
            )

    def test_collection_rejects_duplicate_timestamps_and_timezone_drift(self) -> None:
        after = datetime(2026, 8, 3, 9, 0)

        class Duplicate:
            def next_after(self, *args, **kwargs):
                return Occurrence(date(2026, 8, 3), 9, 0, local_datetime=after)

        with self.assertRaisesRegex(ValueError, "non-advancing"):
            collect_after(
                Duplicate(),
                after,
                limit=1,
                build_local_datetime=lambda day, hhmm: datetime.combine(day, hhmm),
                to_local=lambda value: value,
            )

        class Aware:
            def next_after(self, *args, **kwargs):
                value = after.replace(tzinfo=timezone.utc)
                return Occurrence(
                    value.date(), value.hour, value.minute, local_datetime=value
                )

        with self.assertRaisesRegex(ValueError, "incomparable"):
            collect_after(
                Aware(),
                after,
                limit=1,
                build_local_datetime=lambda day, hhmm: datetime.combine(day, hhmm),
                to_local=lambda value: value,
            )

    def test_ordinary_anchor_provider_exposes_typed_lazy_occurrence(self) -> None:
        provider = AnchorOccurrenceProvider(
            lambda after: (
                datetime(2026, 8, 4, 9, 0)
                if after < datetime(2026, 8, 4, 9, 0)
                else None
            )
        )

        occurrence = provider.next_after(
            datetime(2026, 8, 3, 9, 0),
            build_local_datetime=lambda day, hhmm: datetime(
                day.year, day.month, day.day, *hhmm
            ),
            to_local=lambda value: value,
        )

        self.assertEqual(occurrence, Occurrence(date(2026, 8, 4), 9, 0))

    def test_anchor_provider_rejects_dst_fallback_backward_progress(self) -> None:
        from zoneinfo import ZoneInfo

        zone = ZoneInfo("Europe/Bucharest")
        after = datetime(2026, 10, 25, 3, 15, tzinfo=zone, fold=1)
        backward = datetime(2026, 10, 25, 3, 30, tzinfo=zone, fold=0)
        provider = AnchorOccurrenceProvider(lambda _value: backward)

        with self.assertRaisesRegex(ValueError, "non-advancing"):
            provider.next_after(
                after,
                build_local_datetime=lambda day, hhmm: datetime(
                    day.year, day.month, day.day, *hhmm, tzinfo=zone
                ),
                to_local=lambda value: value,
            )

    def test_anchor_file_cursor_reuse_matches_fresh_provider_lookups(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from zoneinfo import ZoneInfo

        zone = ZoneInfo("UTC")
        with TemporaryDirectory() as directory:
            Path(directory, "calendar.csv").write_text(
                "date,description\n2026-08-03,first\n2026-08-10,second\n2026-08-17,third\n",
                encoding="utf-8",
            )

            def build(day, hhmm):
                return datetime(
                    day.year, day.month, day.day, hhmm[0], hhmm[1], tzinfo=zone
                )

            cursors = (
                datetime(2026, 8, 1, 9, tzinfo=zone),
                datetime(2026, 8, 9, 9, tzinfo=zone),
                datetime(2026, 8, 4, 9, tzinfo=zone),
            )
            identity = lambda value: value
            cached = AnchorFileOccurrenceProvider("calendar.csv", directory, (9, 0))
            for cursor in cursors:
                optimized = cached.next_after(
                    cursor, build_local_datetime=build, to_local=identity
                )
                reference = AnchorFileOccurrenceProvider(
                    "calendar.csv", directory, (9, 0)
                ).next_after(
                    cursor, build_local_datetime=build, to_local=identity
                )
                self.assertIsNotNone(optimized)
                self.assertIsNotNone(reference)
                self.assertEqual(optimized.local_datetime, reference.local_datetime)
                self.assertEqual(optimized.description, reference.description)

            stats = cached.cache_stats

        self.assertEqual(stats["lookups"], 3)
        self.assertEqual(stats["builds"], 1)
        self.assertEqual(stats["records"], 3)
        self.assertGreater(stats["hit_ratio"], 0.6)

    def test_anchor_file_batch_generation_matches_repeated_lazy_lookups(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from zoneinfo import ZoneInfo

        zone = ZoneInfo("UTC")
        with TemporaryDirectory() as directory:
            Path(directory, "calendar.csv").write_text(
                "date,description\n2026-08-03,first\n2026-08-10,second\n2026-08-17,third\n",
                encoding="utf-8",
            )

            def build(day, hhmm):
                return datetime(
                    day.year, day.month, day.day, hhmm[0], hhmm[1], tzinfo=zone
                )

            start = datetime(2026, 8, 1, 9, tzinfo=zone)
            batch = collect_after(
                AnchorFileOccurrenceProvider("calendar.csv", directory, (9, 0)),
                start,
                limit=3,
                build_local_datetime=build,
                to_local=lambda value: value,
                require_contract=True,
            )
            reference_provider = AnchorFileOccurrenceProvider(
                "calendar.csv", directory, (9, 0)
            )
            repeated = []
            cursor = start
            for _ in range(3):
                occurrence = reference_provider.next_after(
                    cursor,
                    build_local_datetime=build,
                    to_local=lambda value: value,
                )
                self.assertIsNotNone(occurrence)
                repeated.append(occurrence)
                cursor = occurrence.local_datetime

        self.assertEqual(
            [(item.local_datetime, item.description) for item in batch],
            [(item.local_datetime, item.description) for item in repeated],
        )

if __name__ == "__main__":
    unittest.main()
