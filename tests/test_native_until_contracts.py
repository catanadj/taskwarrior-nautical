"""Direct contracts for native-until validation, carry, and descriptions."""

from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo
import unittest

import nautical_core.add_validation as add_validation
import nautical_core.native_until as native_until


class NativeUntilContracts(unittest.TestCase):
    def test_calendar_carry_is_shared_across_recurrence_kinds_and_conflicts(self) -> None:
        parent_target = datetime(2026, 8, 1, 9)
        parent_until = datetime(2026, 8, 1, 23)
        child_target = datetime(2026, 8, 2, 9)
        expected = datetime(2026, 8, 2, 23)

        for kind in ("cp", "anchor", "anchor_file"):
            with self.subTest(kind=kind):
                self.assertEqual(
                    native_until.carry(
                        parent_target,
                        parent_until,
                        child_target,
                        kind,
                        utc_to_local_naive=lambda value: value,
                        local_naive_to_utc=lambda value: value,
                    ),
                    expected,
                )

        with self.assertRaises(native_until.NativeUntilCarryError) as raised:
            native_until.carry(
                parent_target,
                parent_until,
                datetime(2026, 8, 1, 23, 30),
                "anchor",
                utc_to_local_naive=lambda value: value,
                local_naive_to_utc=lambda value: value,
            )

        self.assertEqual(raised.exception.code, native_until.CARRY_CONFLICT)

    def test_carry_description_omits_only_the_optional_summary_on_adapter_failure(self) -> None:
        def fail_timezone(_value):
            raise RuntimeError("timezone formatter failed")

        description = native_until.describe_carry(
            datetime(2026, 8, 3, 18, tzinfo=timezone.utc),
            datetime(2026, 8, 3, 9, tzinfo=timezone.utc),
            to_local=fail_timezone,
        )

        self.assertIsNone(description)

    def test_carry_wraps_conversion_failure_as_typed_failure_with_cause(self) -> None:
        timestamp = datetime(2026, 8, 3, 9)

        def fail_timezone(_value):
            raise RuntimeError("timezone conversion failed")

        with self.assertRaises(native_until.NativeUntilCarryError) as raised:
            native_until.carry(
                timestamp,
                timestamp.replace(hour=18),
                timestamp.replace(day=4),
                "anchor",
                utc_to_local_naive=fail_timezone,
                local_naive_to_utc=lambda value: value,
            )

        self.assertEqual(raised.exception.code, native_until.CARRY_FAILED)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    def test_carry_wraps_postcondition_comparison_failure_with_cause(self) -> None:
        class BrokenTimezone(tzinfo):
            def utcoffset(self, _value):
                raise RuntimeError("comparison timezone failed")

        parent_target = datetime(2026, 8, 3, 9)
        parent_until = parent_target.replace(hour=18)
        child_target = parent_target.replace(day=4)

        with self.assertRaises(native_until.NativeUntilCarryError) as raised:
            native_until.carry(
                parent_target,
                parent_until,
                child_target,
                "anchor",
                utc_to_local_naive=lambda value: value,
                local_naive_to_utc=lambda value: value.replace(tzinfo=BrokenTimezone()),
            )

        self.assertEqual(raised.exception.code, native_until.CARRY_FAILED)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    def test_until_validation_does_not_hide_timezone_adapter_failures(self) -> None:
        class BrokenTimezone(tzinfo):
            def utcoffset(self, _value):
                raise RuntimeError("timezone adapter failed")

        until = datetime(2026, 8, 3, 18, tzinfo=BrokenTimezone())
        target = datetime(2026, 8, 3, 9, tzinfo=timezone.utc)

        with self.assertRaisesRegex(RuntimeError, "timezone adapter failed"):
            native_until.validate_after_target(until, target, "due")

    def test_until_validation_classifies_malformed_values_as_uncomparable(self) -> None:
        valid, reason = native_until.validate_after_target(
            "not-a-datetime",
            datetime(2026, 8, 3, 9, tzinfo=timezone.utc),
            "scheduled",
        )

        self.assertFalse(valid)
        self.assertEqual(reason, "until and scheduled could not be compared")

    def test_calendar_slot_validation_propagates_unexpected_timezone_failures(self) -> None:
        def fail_timezone(_value):
            raise RuntimeError("timezone resolver failed")

        with self.assertRaisesRegex(RuntimeError, "timezone resolver failed"):
            native_until.validate_calendar_slots(
                datetime(2026, 8, 3, 18, tzinfo=timezone.utc),
                datetime(2026, 8, 3, 9, tzinfo=timezone.utc),
                ((9, 0),),
                to_local=fail_timezone,
            )

    def test_calendar_slot_validation_classifies_malformed_slots(self) -> None:
        valid, reason = native_until.validate_calendar_slots(
            datetime(2026, 8, 3, 18, tzinfo=timezone.utc),
            datetime(2026, 8, 3, 9, tzinfo=timezone.utc),
            (("not-an-hour", 0),),
            to_local=lambda value: value,
        )

        self.assertFalse(valid)
        self.assertEqual(reason, "could not compare calendar expiration with anchor times")

    def test_exact_carry_detection_does_not_hide_timestamp_adapter_failures(self) -> None:
        class BrokenTimestamp:
            @property
            def second(self) -> int:
                raise RuntimeError("timestamp adapter failed")

        with self.assertRaisesRegex(RuntimeError, "timestamp adapter failed"):
            native_until.uses_exact_carry(BrokenTimestamp())

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
