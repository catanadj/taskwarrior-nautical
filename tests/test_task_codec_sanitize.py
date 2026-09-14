from __future__ import annotations

from collections import UserDict
import unittest

from nautical_core.task_codec import TaskCodec


class TaskCodecSanitizeTests(unittest.TestCase):
    def test_sanitizes_mutable_mapping_implementations(self) -> None:
        task = UserDict({"description": "ok\x00" + "x" * 10, "count": 3})
        TaskCodec.sanitize_task_mapping(task, max_len=4)
        self.assertEqual(task["description"], "okxx")
        self.assertEqual(task["count"], 3)


if __name__ == "__main__":
    unittest.main()
