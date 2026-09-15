import contextlib
import io
import sys
import unittest
from unittest.mock import patch

from nautical_core.tools import nautical_runtime_cleanup


class RuntimeCleanupCliTests(unittest.TestCase):
    def test_text_mode_reports_singular_cleanup_error(self) -> None:
        with patch.object(
            nautical_runtime_cleanup.install_runtime,
            "cleanup_runtime",
            return_value={"status": "error", "error": "permission denied"},
        ), patch.object(sys, "argv", ["nautical_runtime_cleanup"]):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = nautical_runtime_cleanup.main()

        self.assertEqual(exit_code, 2)
        self.assertIn("permission denied", output.getvalue())


if __name__ == "__main__":
    unittest.main()
