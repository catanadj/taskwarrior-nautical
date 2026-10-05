from __future__ import annotations

from contextlib import redirect_stderr
import io
import os
from types import SimpleNamespace
from typing import Any, get_type_hints
import unittest
from unittest.mock import patch

import nautical_core.panel_diagnostics as panel_diagnostics


class PanelDiagnosticsContractTests(unittest.TestCase):
    def test_file_warning_core_uses_narrow_diagnostic_protocol(self) -> None:
        core_type = get_type_hints(panel_diagnostics.file_source_warnings)["core"]

        self.assertIs(core_type, panel_diagnostics.PanelDiagnosticsCore)
        self.assertIsNot(core_type, Any)

    def test_optional_file_source_failures_are_gated_and_redacted(self) -> None:
        def missing_owner(name: str) -> object:
            if name == "anchor_files":
                raise FileNotFoundError("secret anchor path")
            raise ImportError("secret omit path")

        core = SimpleNamespace(
            ANCHOR_FILE_DIR="/secret/anchor",
            OMIT_FILE_DIR="/secret/omit",
            _import_sibling=missing_owner,
        )
        stderr = io.StringIO()
        with patch.dict(os.environ, {"NAUTICAL_DIAG": "1"}), redirect_stderr(stderr):
            warnings = panel_diagnostics.file_source_warnings(
                core,
                {"anchor_file": "anchor.csv", "omit_file": "omit.csv"},
            )

        self.assertEqual(warnings, [])
        output = stderr.getvalue()
        self.assertIn("anchor_file diagnostic unavailable (FileNotFoundError)", output)
        self.assertIn("omit_file diagnostic unavailable (ImportError)", output)
        self.assertNotIn("secret", output)

    def test_optional_file_source_failures_are_silent_without_diagnostic_mode(self) -> None:
        def missing_owner(_name: str) -> object:
            raise ImportError("secret path")

        core = SimpleNamespace(_import_sibling=missing_owner)
        stderr = io.StringIO()
        with patch.dict(os.environ, {"NAUTICAL_DIAG": "0"}), redirect_stderr(stderr):
            warnings = panel_diagnostics.file_source_warnings(
                core,
                {"anchor_file": "anchor.csv", "omit_file": "omit.csv"},
            )

        self.assertEqual(warnings, [])
        self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
