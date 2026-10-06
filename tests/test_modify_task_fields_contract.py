"""Task-field normalization contracts for modify workflows."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import nautical_core.modify_task_fields as modify_task_fields


class ModifyTaskFieldsContractTests(unittest.TestCase):
    def test_field_comparison_keeps_fallback_for_non_json_task_values(self) -> None:
        self.assertFalse(modify_task_fields.field_changed({"value": {1}}, {"value": {1}}, "value"))
        self.assertTrue(modify_task_fields.field_changed({"value": {1}}, {"value": {2}}, "value"))

    def test_field_comparison_does_not_hide_unexpected_json_serializer_defects(self) -> None:
        with patch.object(modify_task_fields.json, "dumps", side_effect=RuntimeError("serializer defect")):
            with self.assertRaisesRegex(RuntimeError, "serializer defect"):
                modify_task_fields.field_changed({"value": {"nested": 1}}, {}, "value")


if __name__ == "__main__":
    unittest.main()
