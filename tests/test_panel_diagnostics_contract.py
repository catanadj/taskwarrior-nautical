from __future__ import annotations

from typing import Any, get_type_hints
import unittest

import nautical_core.panel_diagnostics as panel_diagnostics


class PanelDiagnosticsContractTests(unittest.TestCase):
    def test_file_warning_core_uses_narrow_diagnostic_protocol(self) -> None:
        core_type = get_type_hints(panel_diagnostics.file_source_warnings)["core"]

        self.assertIs(core_type, panel_diagnostics.PanelDiagnosticsCore)
        self.assertIsNot(core_type, Any)


if __name__ == "__main__":
    unittest.main()
