from __future__ import annotations

import unittest
from unittest.mock import patch

import nautical_core.timezone_facade as timezone_facade


class TimezoneFacadeContracts(unittest.TestCase):
    def test_unexpected_zoneinfo_failure_is_not_reported_as_bad_configuration(self) -> None:
        class BrokenZoneInfo:
            @staticmethod
            def ZoneInfo(_name: str) -> None:
                raise RuntimeError("timezone resolver implementation failed")

        warnings = []
        with (
            patch.object(timezone_facade, "_local_timezone", None),
            patch.object(timezone_facade, "_configuration_error", ""),
            self.assertRaisesRegex(RuntimeError, "timezone resolver implementation failed"),
        ):
            timezone_facade.resolve("UTC", BrokenZoneInfo, lambda *warning: warnings.append(warning))

        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
