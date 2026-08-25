from __future__ import annotations

import unittest
from datetime import datetime, timezone

from nautical_core.astronomy_validation import validate_native_until_slots


class AstronomyValidationTests(unittest.TestCase):
    def test_shared_validator_collects_and_normalizes_effective_slots(self) -> None:
        calls: list[dict[str, object]] = []

        def collect(*args: object, **kwargs: object) -> list[tuple[int, int]]:
            calls.append({"args": args, **kwargs})
            return [(9, 0), (18, 30)]

        def validate(until: datetime, target: datetime, slots: object, **_: object) -> tuple[bool, str | None]:
            self.assertEqual(slots, [(9, 0), (18, 30)])
            return False, "expiration is before the final slot"

        target = datetime(2026, 8, 25, 9, tzinfo=timezone.utc)
        valid, reason, slots = validate_native_until_slots(
            until_dt=datetime(2026, 8, 25, 17, tzinfo=timezone.utc),
            target_dt=target,
            dnf=object(),
            anchor_file_value="",
            fallback_hhmm=(0, 0),
            collect_time_slots=collect,
            normalize_time_slots=lambda value: value,
            resolve_time_slots=None,
            anchor_file_dir="",
            recurrence_context=object(),
            to_local=lambda value: value,
            validate_time_slots=validate,
        )

        self.assertFalse(valid)
        self.assertEqual(reason, "expiration is before the final slot")
        self.assertEqual(slots, ((9, 0), (18, 30)))
        self.assertEqual(calls[0]["target_date"], target.date())

    def test_shared_validator_skips_tasks_without_recurrence_slots(self) -> None:
        def unexpected_collect(*_: object, **__: object) -> object:
            self.fail("collector should not run without an anchor or anchor file")

        self.assertEqual(
            validate_native_until_slots(
                until_dt=datetime.now(timezone.utc),
                target_dt=datetime.now(timezone.utc),
                dnf=None,
                anchor_file_value="",
                fallback_hhmm=(0, 0),
                collect_time_slots=unexpected_collect,
                normalize_time_slots=lambda value: value,
                resolve_time_slots=None,
                anchor_file_dir="",
                recurrence_context=object(),
                to_local=lambda value: value,
                validate_time_slots=lambda *args, **kwargs: (True, None),
            ),
            (True, None, ()),
        )


if __name__ == "__main__":
    unittest.main()
