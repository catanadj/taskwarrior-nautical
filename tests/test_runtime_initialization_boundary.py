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

    def test_live_panel_footer_defaults_and_accepts_configured_text(self) -> None:
        self.assertEqual(
            self._read_configured_core_value("", "core_config.LIVE_PANEL_FOOTER"),
            "NAUTICAL",
        )
        self.assertEqual(
            self._read_configured_core_value(
                'live_panel_footer = "STATUS"\n', "core_config.LIVE_PANEL_FOOTER"
            ),
            "STATUS",
        )

    def test_uda_aliases_remain_opt_in_through_configuration(self) -> None:
        cases = (("", False), ("enable_uda_aliases = true\n", True), ("enable_uda_aliases = false\n", False))
        for config_text, expected in cases:
            with self.subTest(config_text=config_text):
                self.assertEqual(
                    self._read_configured_core_value(
                        config_text, "core_config.ENABLE_UDA_ALIASES"
                    ),
                    expected,
                )

    def test_business_calendar_toml_resolves_through_public_core_api(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config-nautical.toml"
            config.write_text(
                '[business_calendar.work]\n'
                'anchor = "w:mon..fri"\n'
                'omit = "y:04-20"\n',
                encoding="utf-8",
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
                "import json\nfrom datetime import date\n"
                "import nautical_core as core\n"
                "policy = core.get_configured_business_calendar('WORK')\n"
                "print(json.dumps({'names': sorted(core.business_calendar_definitions()), "
                "'open': policy.is_business_day(date(2026, 4, 21)), "
                "'closed': policy.is_business_day(date(2026, 4, 20))}))\n",
                cwd=ROOT,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(result.stdout),
                {"names": ["work"], "open": True, "closed": False},
            )

    def test_configuration_drift_detects_edit_and_removal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config-nautical.toml"
            config.write_text('tz = "UTC"\n', encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "NAUTICAL_CONFIG": str(config),
                    "NAUTICAL_TRUST_CONFIG_PATH": "1",
                    "PYTHONPATH": str(ROOT),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            code = (
                "import json, os\nfrom pathlib import Path\n"
                "import nautical_core as core\n"
                "config = Path(os.environ['NAUTICAL_CONFIG'])\n"
                "before = core.configuration_drift()\n"
                "config.write_text('tz = \\\"Europe/Bucharest\\\"\\n', encoding='utf-8')\n"
                "edited = core.configuration_drift()\n"
                "config.unlink()\n"
                "removed = core.configuration_drift()\n"
                "print(json.dumps({'before': before, 'edited': edited, 'removed': removed}))\n"
            )
            result = self.run_python_code(code, cwd=ROOT, env=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["before"]["status"], "ok")
            self.assertEqual(payload["edited"]["status"], "changed")
            self.assertEqual(payload["removed"]["status"], "changed")

    def test_scheduler_config_changes_cache_keys_but_ui_changes_do_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config-nautical.toml"
            environment = os.environ.copy()
            environment.update(
                {
                    "NAUTICAL_CONFIG": str(config),
                    "NAUTICAL_TRUST_CONFIG_PATH": "1",
                    "PYTHONPATH": str(ROOT),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )

            def cache_key(config_text: str) -> str:
                config.write_text(config_text, encoding="utf-8")
                result = self.run_python_code(
                    "import nautical_core as core\n"
                    "print(core.cache_key_for_task('w:mon', 'skip'))\n",
                    cwd=ROOT,
                    env=environment,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout.strip()

            footer_one = cache_key('tz = "UTC"\nlive_panel_footer = "ONE"\n')
            footer_two = cache_key('tz = "UTC"\nlive_panel_footer = "TWO"\n')
            timezone_utc = cache_key('tz = "UTC"\nlive_panel_footer = "TWO"\n')
            timezone_bucharest = cache_key(
                'tz = "Europe/Bucharest"\nlive_panel_footer = "TWO"\n'
            )
            self.assertEqual(footer_one, footer_two)
            self.assertNotEqual(timezone_utc, timezone_bucharest)

    def test_taskdata_reload_keeps_validated_fingerprints_consistent(self) -> None:
        config = Path(self.taskdata) / "config-nautical.toml"
        config.write_text(
            'tz = "Europe/Athens"\nseason_hemisphere = "north"\n',
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment.pop("NAUTICAL_CONFIG", None)
        environment.update(
            {
                "TASKDATA": self.taskdata,
                "PYTHONPATH": str(ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        code = (
            "import json, os, nautical_core as core\n"
            "first = core.reload_taskdata_config(os.environ['TASKDATA'])\n"
            "drift = core.configuration_drift()\n"
            "second = core.reload_taskdata_config(os.environ['TASKDATA'])\n"
            "print(json.dumps({'first': first, 'second': second, 'drift': drift, "
            "'effective': core.effective_config_fingerprint(), "
            "'scheduler': core.scheduler_config_fingerprint()}))\n"
        )
        result = self.run_python_code(
            code,
            cwd=ROOT,
            env=environment,
            clear_environment=("NAUTICAL_CONFIG",),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        first = payload["first"]
        second = payload["second"]
        self.assertTrue(first["ok"] and second["ok"], payload)
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(
            first["scheduler_fingerprint"], second["scheduler_fingerprint"]
        )
        self.assertEqual(first["fingerprint"], payload["effective"])
        self.assertEqual(first["scheduler_fingerprint"], payload["scheduler"])
        self.assertEqual(payload["drift"]["status"], "ok")

    def test_taskdata_discovery_rejects_malformed_configuration(self) -> None:
        config = Path(self.taskdata) / "config-nautical.toml"
        config.write_text('tz = "Europe/Athens"\n[broken\n', encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "TASKDATA": self.taskdata,
                "TASKRC": str(Path(self.taskdata) / "taskrc"),
                "PYTHONPATH": str(ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        result = self.run_python_code(
            "import os, nautical_core as core\n"
            "core.reload_taskdata_config(os.environ['TASKDATA'])\n",
            cwd=ROOT,
            env=environment,
            clear_environment=("NAUTICAL_CONFIG",),
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("config parse failed", f"{result.stdout}\n{result.stderr}".lower())

    def test_operator_context_rejects_malformed_toml_and_invalid_timezone(self) -> None:
        environment = os.environ.copy()
        environment.pop("NAUTICAL_CONFIG", None)
        environment.pop("TASKDATA", None)
        environment.update(
            {
                "PYTHONPATH": str(ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        code = (
            "import sys\nimport nautical_core\n"
            "from nautical_core.integration_context import IntegrationRuntime, build_operator_context\n"
            "try:\n"
            "    build_operator_context(runtime=IntegrationRuntime.from_compatibility_facade(nautical_core), "
            "task_binary=sys.executable, taskdata=sys.argv[1])\n"
            "except Exception as exc:\n"
            "    print(type(exc).__name__ + ': ' + str(exc))\n"
            "else:\n"
            "    raise SystemExit('reload unexpectedly succeeded')\n"
        )
        cases = (
            ("tz = [\n", "config parse failed"),
            ('tz = "Invalid/Timezone"\n', "invalid or unavailable"),
        )
        for config_text, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "config-nautical.toml").write_text(
                    config_text, encoding="utf-8"
                )
                result = self.run_python_code(
                    code,
                    arguments=(directory,),
                    cwd=ROOT,
                    env=environment,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected, result.stdout)

    def test_invalid_timezone_uses_utc_fallback_and_blocks_scheduling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config-nautical.toml"
            config.write_text('tz = "Invalid/Timezone"\n', encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "NAUTICAL_CONFIG": str(config),
                    "NAUTICAL_DIAG": "1",
                    "PYTHONPATH": str(ROOT),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "XDG_CACHE_HOME": directory,
                }
            )
            result = self.run_python_code(
                "import json, zoneinfo\nimport nautical_core as core\n"
                "from nautical_core import core_config, timezone_facade\n"
                "core_config.ensure_loaded()\n"
                "timezone_facade.resolve(core_config.LOCAL_TZ_NAME, zoneinfo, "
                "core_config.warn_once_per_day)\n"
                "print(json.dumps({'timezone_missing': timezone_facade.current_timezone() is None, "
                "'error': core.scheduling_configuration_error()}))\n",
                cwd=ROOT,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["timezone_missing"])
            self.assertIn("invalid or unavailable", payload["error"])
            self.assertIn("utc fallback", result.stderr.lower())

    def test_world_writable_explicit_config_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config-nautical.toml"
            config.write_text('tz = "Pacific/Auckland"\n', encoding="utf-8")
            try:
                config.chmod(0o666)
            except OSError as error:
                self.skipTest(f"could not set world-writable mode: {error}")
            environment = os.environ.copy()
            environment.update(
                {
                    "NAUTICAL_CONFIG": str(config),
                    "PYTHONPATH": str(ROOT),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            result = self.run_python_code(
                "import nautical_core as core\n"
                "print(core.scheduling_configuration_error())\n",
                cwd=ROOT,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(config), result.stdout)
            self.assertIn("world-writable", result.stdout)

    def test_random_salt_replays_and_namespaces_random_draws(self) -> None:
        start = "2026-06-07"
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config-nautical.toml"
            environment = os.environ.copy()
            environment.update(
                {
                    "NAUTICAL_CONFIG": str(config),
                    "NAUTICAL_TRUST_CONFIG_PATH": "1",
                    "PYTHONPATH": str(ROOT),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )

            def sequence(salt: str) -> list[str]:
                config.write_text(f'wrand_salt = "{salt}"\n', encoding="utf-8")
                code = (
                    "import json\nfrom datetime import date\n"
                    "import nautical_core as core\n"
                    "core.effective_config_snapshot()\n"
                    f"start = date.fromisoformat({start!r})\n"
                    "dnf = core.parse_anchor_expr_to_dnf_cached('w:rand')\n"
                    "current = start\noutput = []\n"
                    "for _ in range(12):\n"
                    "    current, _metadata = core.next_after_expr("
                    "dnf, current, default_seed=start, seed_base='salt-test-chain')\n"
                    "    output.append(current.isoformat())\n"
                    "print(json.dumps(output))\n"
                )
                result = self.run_python_code(code, cwd=ROOT, env=environment)
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(result.stdout)

            first = sequence("salt-a")
            replay = sequence("salt-a")
            changed_salt = sequence("salt-b")
            self.assertEqual(first, replay)
            self.assertNotEqual(first, changed_salt)

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
