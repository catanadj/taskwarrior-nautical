from __future__ import annotations

import unittest

from dev_tools import nautical_stress_campaign as campaign


class StressCampaignTests(unittest.TestCase):
    def test_profiles_have_explicit_cycle_and_timeout_budgets(self) -> None:
        self.assertEqual(campaign.PROFILE_BUDGETS["ci"], (8, 300.0))
        self.assertEqual(campaign.PROFILE_BUDGETS["nightly"], (24, 300.0))
        self.assertEqual(campaign.PROFILE_BUDGETS["stress"], (64, 900.0))


if __name__ == "__main__":
    unittest.main()
