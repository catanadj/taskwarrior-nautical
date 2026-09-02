import unittest

from nautical_core.chain_integrity_engine import IntegrityEngineResult
from nautical_core.chain_integrity_models import IntegrityReportStatus
from nautical_core.integrity_report import components


class IntegrityReportContractTests(unittest.TestCase):
    def test_healthy_internal_status_maps_to_public_ok(self):
        payload = components(IntegrityEngineResult(IntegrityReportStatus.HEALTHY))
        self.assertEqual(payload["status"], "ok")
        self.assertIsNone(payload["failure"])


if __name__ == "__main__":
    unittest.main()
