from __future__ import annotations

import os
import unittest

from nautical_core.hooks import exit_impl


class ExitPolicyRefreshTests(unittest.TestCase):
    def test_policy_is_resolved_for_each_invocation(self) -> None:
        names = ("NAUTICAL_EXIT_STRICT", "NAUTICAL_TASK_TIMEOUT_EXPORT")
        previous = {name: os.environ.get(name) for name in names}
        try:
            os.environ["NAUTICAL_EXIT_STRICT"] = "1"
            os.environ["NAUTICAL_TASK_TIMEOUT_EXPORT"] = "11"
            exit_impl._refresh_exit_policy()
            self.assertTrue(exit_impl._EXIT_STRICT)
            self.assertEqual(exit_impl._TASK_TIMEOUT_EXPORT, 11.0)

            os.environ["NAUTICAL_EXIT_STRICT"] = "0"
            os.environ["NAUTICAL_TASK_TIMEOUT_EXPORT"] = "2"
            exit_impl._refresh_exit_policy()
            self.assertFalse(exit_impl._EXIT_STRICT)
            self.assertEqual(exit_impl._TASK_TIMEOUT_EXPORT, 2.0)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            exit_impl._refresh_exit_policy()

    def test_invocation_context_is_not_reused(self) -> None:
        previous = (
            exit_impl._INTEGRATION_CONTEXT,
            exit_impl._TASKDATA_RAW,
            exit_impl._USE_RC_DATA_LOCATION,
            exit_impl._CORE_READY,
        )
        try:
            exit_impl._INTEGRATION_CONTEXT = object()
            exit_impl._TASKDATA_RAW = "stale"
            exit_impl._USE_RC_DATA_LOCATION = True
            exit_impl._CORE_READY = True
            exit_impl._reset_integration_context()
            self.assertIsNone(exit_impl._INTEGRATION_CONTEXT)
            self.assertEqual(exit_impl._TASKDATA_RAW, "")
            self.assertFalse(exit_impl._USE_RC_DATA_LOCATION)
            self.assertFalse(exit_impl._CORE_READY)
        finally:
            (
                exit_impl._INTEGRATION_CONTEXT,
                exit_impl._TASKDATA_RAW,
                exit_impl._USE_RC_DATA_LOCATION,
                exit_impl._CORE_READY,
            ) = previous


if __name__ == "__main__":
    unittest.main()
