"""Configuration and Taskdata discovery golden tests."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import nautical_core as core
from dev_tools.golden_tests.support import expect, load_core_module


ROOT = Path(__file__).resolve().parents[2]


def test_core_invalid_timezone_warns_and_falls_back_to_utc():
    """Invalid timezone config falls back to UTC and emits a diagnostic warning."""
    core_path = ROOT / "nautical_core" / "__init__.py"
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "nautical.toml"
        config.write_text('tz = "Invalid/Timezone"\n', encoding="utf-8")

        previous_diag = os.environ.get("NAUTICAL_DIAG")
        previous_cache = os.environ.get("XDG_CACHE_HOME")
        os.environ["NAUTICAL_DIAG"] = "1"
        os.environ["XDG_CACHE_HOME"] = td
        try:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                module = load_core_module(str(core_path), "_nautical_core_bad_tz_fallback_test", str(config))
            expect(getattr(module, "_LOCAL_TZ", None) is None, "invalid timezone should use UTC fallback")
            expect(
                "invalid or unavailable" in module.scheduling_configuration_error(),
                "invalid timezone should block Nautical scheduling",
            )
            expect("utc fallback" in stderr.getvalue().lower(), f"expected timezone fallback warning: {stderr.getvalue()!r}")
        finally:
            if previous_diag is None:
                os.environ.pop("NAUTICAL_DIAG", None)
            else:
                os.environ["NAUTICAL_DIAG"] = previous_diag
            if previous_cache is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = previous_cache


def test_explicit_unsafe_config_blocks_scheduling_with_actionable_error():
    """An explicit world-writable config must not silently fall back to UTC."""
    script = "import nautical_core\nprint(nautical_core.scheduling_configuration_error())\n"
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "nautical.toml"
        config.write_text('tz = "Pacific/Auckland"\n', encoding="utf-8")
        try:
            config.chmod(0o666)
        except OSError:
            return
        env = os.environ.copy()
        env.update({"NAUTICAL_CONFIG": str(config), "PYTHONPATH": str(ROOT)})
        process = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT),
        )
        expect(process.returncode == 0, f"unsafe config verification failed: {process.stderr[:500]!r}")
        expect(str(config) in process.stdout, f"rejected config path missing: {process.stdout!r}")
        expect("world-writable" in process.stdout, f"rejected config reason missing: {process.stdout!r}")


def test_taskdata_config_reload_fails_closed_for_malformed_toml_and_timezone():
    """Taskdata reload rejects malformed config and invalid timezones."""
    script = (
        "import sys\n"
        "import nautical_core\n"
        "from nautical_core.integration_context import IntegrationRuntime, build_operator_context\n"
        "try:\n"
        "    build_operator_context(runtime=IntegrationRuntime.from_compatibility_facade(nautical_core), task_binary=sys.executable, taskdata=sys.argv[1])\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__ + ': ' + str(exc))\n"
        "else:\n"
        "    raise SystemExit('reload unexpectedly succeeded')\n"
    )
    env = os.environ.copy()
    env.pop("NAUTICAL_CONFIG", None)
    env.pop("TASKDATA", None)
    env["PYTHONPATH"] = str(ROOT)
    cases = (("tz = [\n", "config parse failed"), ('tz = "Invalid/Timezone"\n', "invalid or unavailable"))
    for contents, expected in cases:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "config-nautical.toml").write_text(contents, encoding="utf-8")
            process = subprocess.run(
                [sys.executable, "-c", script, td],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(ROOT),
            )
            expect(process.returncode == 0, f"Taskdata config reload process failed: {process.stderr[:500]!r}")
            expect(expected in process.stdout, f"reload error was not actionable: {process.stdout!r}")


def test_config_fingerprint_invalidates_persistent_cache_keys():
    """Config changes update scheduling fingerprints but ignore UI-only edits."""
    previous = os.environ.get("NAUTICAL_CONFIG")
    try:
        with tempfile.TemporaryDirectory() as td:
            config = Path(td) / "config-nautical.toml"
            config.write_text('tz = "UTC"\nlive_panel_footer = "ONE"\n', encoding="utf-8")
            os.environ["NAUTICAL_CONFIG"] = str(config)
            first = core.effective_config_snapshot()
            config.write_text('tz = "Europe/Bucharest"\nlive_panel_footer = "TWO"\n', encoding="utf-8")
            second = core.effective_config_snapshot()
            expect(first.get("fingerprint") != second.get("fingerprint"), "config edits did not change fingerprint")

            def key_in_fresh_process(config_text):
                config.write_text(config_text, encoding="utf-8")
                env = os.environ.copy()
                env.update(
                    {
                        "NAUTICAL_CONFIG": str(config),
                        "NAUTICAL_TRUST_CONFIG_PATH": "1",
                        "PYTHONPATH": str(ROOT) + (os.pathsep + env.get("PYTHONPATH", "")),
                    }
                )
                process = subprocess.run(
                    [sys.executable, "-c", "import nautical_core; print(nautical_core.cache_key_for_task('w:mon', 'skip'))"],
                    text=True,
                    capture_output=True,
                    env=env,
                    timeout=8.0,
                )
                expect(process.returncode == 0, f"fresh cache-key process failed: {process.stderr!r}")
                return (process.stdout or "").strip().splitlines()[-1]

            footer_one = key_in_fresh_process('tz = "UTC"\nlive_panel_footer = "ONE"\n')
            footer_two = key_in_fresh_process('tz = "UTC"\nlive_panel_footer = "TWO"\n')
            expect(footer_one == footer_two, "UI-only config edits unnecessarily invalidated cache key")
            tz_one = key_in_fresh_process('tz = "UTC"\nlive_panel_footer = "TWO"\n')
            tz_two = key_in_fresh_process('tz = "Europe/Bucharest"\nlive_panel_footer = "TWO"\n')
            expect(tz_one != tz_two, "scheduler config edits did not invalidate cache key")
    finally:
        if previous is None:
            os.environ.pop("NAUTICAL_CONFIG", None)
        else:
            os.environ["NAUTICAL_CONFIG"] = previous


def test_configuration_drift_detects_edit_and_removal():
    """A long-lived core process detects config edits and removal."""
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "config-nautical.toml"
        config.write_text('tz = "UTC"\n', encoding="utf-8")
        script = (
            "import json, os\n"
            "from pathlib import Path\n"
            "import nautical_core as core\n"
            "p = Path(os.environ['NAUTICAL_CONFIG'])\n"
            "before = core.configuration_drift()\n"
            "p.write_text('tz = \\\"Europe/Bucharest\\\"\\n', encoding='utf-8')\n"
            "edited = core.configuration_drift()\n"
            "p.unlink()\n"
            "removed = core.configuration_drift()\n"
            "print(json.dumps({'before': before, 'edited': edited, 'removed': removed}))\n"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env.get("PYTHONPATH", ""))
        env["NAUTICAL_CONFIG"] = str(config)
        env["NAUTICAL_TRUST_CONFIG_PATH"] = "1"
        process = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, env=env, timeout=10)
        expect(process.returncode == 0, f"configuration drift probe failed: {process.stderr}")
        payload = json.loads(process.stdout)
        expect(payload["before"]["status"] == "ok", f"fresh config reported drift: {payload}")
        expect(payload["edited"]["status"] == "changed", f"edited config drift missing: {payload}")
        expect(payload["removed"]["status"] == "changed", f"removed config drift missing: {payload}")


TESTS = (
    test_core_invalid_timezone_warns_and_falls_back_to_utc,
    test_explicit_unsafe_config_blocks_scheduling_with_actionable_error,
    test_taskdata_config_reload_fails_closed_for_malformed_toml_and_timezone,
    test_config_fingerprint_invalidates_persistent_cache_keys,
    test_configuration_drift_detects_edit_and_removal,
)
