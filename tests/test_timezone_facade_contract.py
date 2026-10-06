from __future__ import annotations

import zoneinfo
from unittest import TestCase
from unittest.mock import patch
from zoneinfo import ZoneInfo

import nautical_core.timezone_facade as timezone_facade


class TimezoneFacadeContractTests(TestCase):
    def test_timezone_resolution_updates_owner_state_for_consumers(self) -> None:
        with (
            patch.object(timezone_facade, "_local_timezone", None, create=True),
            patch.object(timezone_facade, "_configuration_error", "", create=True),
        ):
            resolved, error = timezone_facade.resolve(
                "UTC",
                zoneinfo,
                lambda *_args: None,
            )

            self.assertIsInstance(resolved, ZoneInfo)
            self.assertEqual(error, "")
            self.assertIs(timezone_facade.current_timezone(), resolved)
            self.assertEqual(timezone_facade.configuration_error(), "")
