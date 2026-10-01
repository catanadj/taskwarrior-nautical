from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from tests.support.hook_process import HookSubprocessFixture

ROOT = Path(__file__).resolve().parents[1]


class RuntimeInitializationBoundaryTests(HookSubprocessFixture):
    def test_explicit_config_is_loaded_on_runtime_access_not_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config-nautical.toml"
            config.write_text('tz = "Europe/Bucharest"\n', encoding="utf-8")
            code = (
                "import nautical_core; "
                "print(nautical_core.LOCAL_TZ_NAME); "
                "nautical_core.effective_config_snapshot(); "
                "print(nautical_core.LOCAL_TZ_NAME)"
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "NAUTICAL_CONFIG": str(config),
                    "PYTHONPATH": str(ROOT),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            result = self.run_python_code(
                code,
                cwd=ROOT,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), ["UTC", "Europe/Bucharest"])


if __name__ == "__main__":
    unittest.main()
