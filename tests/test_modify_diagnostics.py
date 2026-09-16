from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import nautical_core
from nautical_core.hooks import modify_impl


class ModifyDiagnosticRedactionTests(unittest.TestCase):
    def test_sensitive_fields_are_redacted_before_event_creation(self) -> None:
        rendered = modify_impl._diag_redact_msg(
            json.dumps({"description": "private text", "uuid": "safe"}, ensure_ascii=False)
        )
        payload = json.loads(rendered)
        self.assertEqual(payload["description"], "[redacted]")
        self.assertEqual(payload["uuid"], "safe")

    def test_wait_schedule_feedback_uses_time_comparator_port(self) -> None:
        rows = []
        with patch.object(modify_impl, "core", nautical_core):
            modify_impl._append_next_wait_sched_rows(
                rows,
                {"scheduled": "2026-01-02T00:00:00Z"},
                datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        self.assertTrue(any(label == "⚠ Wait/Sched" for label, _value in rows))


if __name__ == "__main__":
    unittest.main()
