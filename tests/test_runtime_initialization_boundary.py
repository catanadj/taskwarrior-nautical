from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from tests.support.hook_process import HookSubprocessFixture

ROOT = Path(__file__).resolve().parents[1]


class RuntimeInitializationBoundaryTests(HookSubprocessFixture):
    def _read_configured_core_value(self, config_text: str, expression: str) -> object:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config-nautical.toml"
            config.write_text(config_text, encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "NAUTICAL_CONFIG": str(config),
                    "NAUTICAL_TRUST_CONFIG_PATH": "1",
                    "PYTHONPATH": str(ROOT),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            result = self.run_python_code(
                "import json\nfrom nautical_core import core_config\n"
                "core_config.ensure_loaded()\n"
                f"print(json.dumps({expression}))\n",
                cwd=ROOT,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

    def test_recurrence_update_udas_use_canonical_key_and_filter_aliases(self) -> None:
        canonical = self._read_configured_core_value(
            'recurrence_update_udas = ["rappel", "next_review"]\n'
            '[recurrence]\nupdate_udas = "ignored_alias"\n',
            "core_config.RECURRENCE_UPDATE_UDAS",
        )
        aliases = self._read_configured_core_value(
            '[recurrence]\nupdate_udas = "rappel, next_review, bad-name, 9x"\n',
            "core_config.RECURRENCE_UPDATE_UDAS",
        )
        self.assertEqual(canonical, ["rappel", "next_review"])
        self.assertEqual(aliases, ["rappel", "next_review"])

    def test_live_panel_duration_defaults_and_clamps(self) -> None:
        cases = (
            ("", 160),
            ("live_panel_duration_ms = -20\n", 0),
            ("live_panel_duration_ms = 275\n", 275),
            ("live_panel_duration_ms = 5000\n", 1000),
            ('live_panel_duration_ms = "bad"\n', 160),
        )
        for config_text, expected in cases:
            with self.subTest(config_text=config_text):
                self.assertEqual(
                    self._read_configured_core_value(
                        config_text, "core_config.LIVE_PANEL_DURATION_MS"
                    ),
                    expected,
                )

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
