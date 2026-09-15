import unittest
from datetime import datetime, timezone

from nautical_core.modify_datetime_effects import (
    DatetimeEffectPorts,
    local_naive_to_utc,
    safe_dt,
    utc_to_local_naive,
)


class ModifyDatetimeEffectsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ports = DatetimeEffectPorts(
            parse_datetime=lambda value: datetime(2026, 1, 2, 3, 4),
            utc_to_local=lambda value: value.astimezone(timezone.utc).replace(tzinfo=None),
            local_to_utc=lambda value: value.replace(tzinfo=timezone.utc),
        )

    def test_safe_dt_preserves_datetime_and_parses_other_values(self) -> None:
        value = datetime(2026, 1, 1)
        self.assertIs(safe_dt(self.ports, value), value)
        self.assertEqual(safe_dt(self.ports, "tomorrow"), datetime(2026, 1, 2, 3, 4))

    def test_safe_dt_returns_none_for_parser_failure(self) -> None:
        ports = DatetimeEffectPorts(lambda _value: (_ for _ in ()).throw(ValueError("bad")), lambda value: value, lambda value: value)
        self.assertIsNone(safe_dt(ports, "bad"))

    def test_timezone_adapters_validate_and_normalize_values(self) -> None:
        aware = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(utc_to_local_naive(self.ports, aware), datetime(2026, 1, 1, 12, 0))
        local = datetime(2026, 1, 1, 12, 0, 0, 123456)
        self.assertEqual(local_naive_to_utc(self.ports, local), datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))

    def test_timezone_adapters_reject_non_datetime_values(self) -> None:
        with self.assertRaises(TypeError):
            utc_to_local_naive(self.ports, "not a datetime")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            local_naive_to_utc(self.ports, "not a datetime")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
