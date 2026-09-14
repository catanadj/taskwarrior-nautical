"""Direct contracts for native-until validation, carry, and descriptions."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import unittest

from nautical_core import add_validation, native_until


class NativeUntilContracts(unittest.TestCase):
    def test_carry_descriptions_distinguish_calendar_and_exact_policies(self) -> None:
        due = datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc)
        cases = (
            (
                datetime(2026, 8, 3, 18, 0, tzinfo=timezone.utc),
                "Same day at 18:00",
            ),
            (
                datetime(2026, 8, 4, 9, 0, tzinfo=timezone.utc),
                "1 calendar day later at 09:00",
            ),
            (
                datetime(2026, 8, 4, 0, 0, 1, tzinfo=timezone.utc),
                "Exact · 14h 00m 01s after occurrence",
            ),
        )
        for until, expected in cases:
            with self.subTest(until=until):
                self.assertEqual(
                    add_validation.describe_native_until_carry(
                        until, due, to_local=lambda value: value
                    ),
                    expected,
                )

    def test_validation_orders_repeated_wall_times_by_instant(self) -> None:
        zone = ZoneInfo("Europe/Bucharest")
        target = datetime(2026, 10, 25, 3, 20, tzinfo=zone, fold=1)
        earlier = datetime(2026, 10, 25, 3, 20, tzinfo=zone, fold=0)

        valid, message = native_until.validate_after_target(earlier, target, "due")

        self.assertFalse(valid)
        self.assertEqual(message, "until must be later than due")

    def test_exact_carry_preserves_elapsed_seconds_across_repeated_hour(self) -> None:
        zone = ZoneInfo("Europe/Bucharest")
        parent_target = datetime(2026, 10, 25, 3, 20, tzinfo=zone, fold=0)
        parent_until = datetime(2026, 10, 25, 3, 20, 1, tzinfo=zone, fold=1)
        child_target = datetime(2026, 10, 26, 3, 20, tzinfo=zone)

        result = native_until.carry(
            parent_target,
            parent_until,
            child_target,
            "cp",
            utc_to_local_naive=lambda value: value.astimezone(zone).replace(tzinfo=None),
            local_naive_to_utc=lambda value: value.replace(tzinfo=zone).astimezone(
                timezone.utc
            ),
        )
        expected = (
            child_target.astimezone(timezone.utc) + timedelta(seconds=3601)
        ).astimezone(zone)
        self.assertEqual(
            result.astimezone(timezone.utc), expected.astimezone(timezone.utc)
        )
        self.assertEqual(
            native_until.describe_carry(
                parent_until, parent_target, to_local=lambda value: value
            ),
            "Exact · 01h 00m 01s after occurrence",
        )

        malformed_until = datetime(2026, 10, 25, 23, 0, tzinfo=zone)
        with self.assertRaises(native_until.NativeUntilCarryError) as raised:
            native_until.carry(
                parent_target,
                malformed_until,
                child_target,
                "cp",
                utc_to_local_naive=lambda value: value.astimezone(zone).replace(
                    tzinfo=None
                ),
                local_naive_to_utc=lambda value: value.replace(tzinfo=None),
            )
        self.assertEqual(raised.exception.code, native_until.CARRY_FAILED)


if __name__ == "__main__":
    unittest.main()
