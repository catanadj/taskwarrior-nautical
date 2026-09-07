from __future__ import annotations

import unittest

from nautical_core.support_policy import policy_document


class SupportPolicyTests(unittest.TestCase):
    def test_policy_document_is_json_safe_and_authoritative(self) -> None:
        self.assertEqual(policy_document()["minimum_python"], "3.11")
        self.assertEqual(policy_document()["minimum_taskwarrior"], "3.4.2")
        self.assertEqual(policy_document()["tested_taskwarrior"], ["3.4.2", "3.5.0"])


if __name__ == "__main__":
    unittest.main()
