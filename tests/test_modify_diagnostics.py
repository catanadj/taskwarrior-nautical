from __future__ import annotations

import json
import unittest

from nautical_core.hooks import modify_impl


class ModifyDiagnosticRedactionTests(unittest.TestCase):
    def test_sensitive_fields_are_redacted_before_event_creation(self) -> None:
        rendered = modify_impl._diag_redact_msg(
            json.dumps({"description": "private text", "uuid": "safe"}, ensure_ascii=False)
        )
        payload = json.loads(rendered)
        self.assertEqual(payload["description"], "[redacted]")
        self.assertEqual(payload["uuid"], "safe")


if __name__ == "__main__":
    unittest.main()
