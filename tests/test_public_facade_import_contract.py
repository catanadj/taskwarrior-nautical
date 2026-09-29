from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublicFacadeImportContractTests(unittest.TestCase):
    def _run_probe(self, source: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(ROOT), environment.get("PYTHONPATH", "")) if part
        )
        return subprocess.run(
            [sys.executable, "-c", source],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def test_core_import_defers_panel_colour_module(self) -> None:
        process = self._run_probe(
            "import sys, nautical_core; "
            "assert 'nautical_core.panel_colours' not in sys.modules; "
            "from nautical_core.panel_colours import chain_colour_root; "
            "chain_colour_root('chain', 'root'); "
            "assert 'nautical_core.panel_colours' in sys.modules"
        )

        self.assertEqual(process.returncode, 0, process.stderr)

    def test_core_import_defers_diagnostic_model(self) -> None:
        process = self._run_probe(
            "import sys, nautical_core; "
            "assert 'nautical_core.diagnostic_models' not in sys.modules; "
            "assert nautical_core.DiagnosticEvent.__name__ == 'DiagnosticEvent'; "
            "assert 'nautical_core.diagnostic_models' in sys.modules"
        )

        self.assertEqual(process.returncode, 0, process.stderr)

    def test_core_import_defers_parser_scheduler_models(self) -> None:
        process = self._run_probe(
            "import sys, nautical_core; "
            "assert 'nautical_core.parsing.parser_models' not in sys.modules; "
            "assert 'nautical_core.scheduler_models' not in sys.modules; "
            "assert nautical_core.ParseError.__name__ == 'ParseError'; "
            "assert 'nautical_core.parsing.parser_models' in sys.modules; "
            "assert 'nautical_core.scheduler_models' in sys.modules"
        )

        self.assertEqual(process.returncode, 0, process.stderr)

    def test_core_import_defers_optional_and_parser_stacks(self) -> None:
        process = self._run_probe(
            "import sys, nautical_core; "
            "names=('rich','nautical_core.ui','nautical_core.astronomy',"
            "'nautical_core.natural_language','nautical_core.linting',"
            "'nautical_core.parser_api','nautical_core.parser_support_api',"
            "'nautical_core.acf_api','nautical_core.expansion_api',"
            "'nautical_core.quarter_api','nautical_core.scheduler_api',"
            "'nautical_core.cached_expansion','nautical_core.monthly_support',"
            "'nautical_core.recurrence_evaluator','nautical_core.token_api',"
            "'nautical_core.time_api','nautical_core.business_calendar_api',"
            "'nautical_core.cache_api','nautical_core.hint_builder_api',"
            "'nautical_core.natural_language_api','nautical_core.linting_api'); "
            "loaded=sorted(name for name in names if name in sys.modules); "
            "count=sum(name.startswith('nautical_core') for name in sys.modules); "
            "assert not loaded, loaded; assert 'subprocess' not in sys.modules; "
            "assert 'tempfile' not in sys.modules; assert count <= 30, count"
        )

        self.assertEqual(process.returncode, 0, process.stderr or process.stdout)


if __name__ == "__main__":
    unittest.main()
