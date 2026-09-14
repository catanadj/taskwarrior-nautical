from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RuntimeInitializationBoundaryTests(unittest.TestCase):
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
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), ["UTC", "Europe/Bucharest"])


if __name__ == "__main__":
    unittest.main()
