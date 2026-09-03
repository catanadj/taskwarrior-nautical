from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dev_tools import nautical_stress_campaign as campaign
from dev_tools import nautical_mixed_recurrence_loop as mixed


class StressCampaignTests(unittest.TestCase):
    def test_profiles_have_explicit_cycle_and_timeout_budgets(self) -> None:
        self.assertEqual(campaign.PROFILE_BUDGETS["ci"], (8, 300.0))
        self.assertEqual(campaign.PROFILE_BUDGETS["nightly"], (24, 300.0))
        self.assertEqual(campaign.PROFILE_BUDGETS["stress"], (64, 900.0))

    def test_health_warning_is_retained_without_failing_campaign(self) -> None:
        payload = '{"status":"warn","checks":[]}'
        process = SimpleNamespace(returncode=1, stdout=payload, stderr="")
        with patch.object(campaign.subprocess, "run", return_value=process):
            result = mixed._health_snapshot({}, None)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "warn")


if __name__ == "__main__":
    unittest.main()
