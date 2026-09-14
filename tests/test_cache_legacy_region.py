from __future__ import annotations

import unittest

from nautical_core.cache_support import cache_key


class LegacyHolidayRegionCacheTests(unittest.TestCase):
    def test_cache_identity_is_defined_without_retired_holiday_region(self) -> None:
        common = dict(
            acf="m:1",
            anchor_mode="next",
            business_calendar_fingerprint="bc-v1",
            anchor_year_fmt="%Y",
            wrand_salt="salt",
            local_tz_name="UTC",
        )
        self.assertEqual(cache_key(**common), cache_key(**common))


if __name__ == "__main__":
    unittest.main()
