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
from dev_tools.golden_tests.support import (
    assert_stdout_json_only,
    expect,
    extract_last_json,
    find_hook_file,
    load_core_module,
    run_hook_script_raw,
)


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
                module = load_core_module(
                    str(core_path), "_nautical_core_bad_tz_fallback_test", str(config)
                )
            expect(
                module.timezone_facade.current_timezone() is None,
                "invalid timezone should use UTC fallback",
            )
            expect(
                "invalid or unavailable" in module.scheduling_configuration_error(),
                "invalid timezone should block Nautical scheduling",
            )
            expect(
                "utc fallback" in stderr.getvalue().lower(),
                f"expected timezone fallback warning: {stderr.getvalue()!r}",
            )
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
    script = (
        "import nautical_core\nprint(nautical_core.scheduling_configuration_error())\n"
    )
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
        expect(
            process.returncode == 0,
            f"unsafe config verification failed: {process.stderr[:500]!r}",
        )
        expect(
            str(config) in process.stdout,
            f"rejected config path missing: {process.stdout!r}",
        )
        expect(
            "world-writable" in process.stdout,
            f"rejected config reason missing: {process.stdout!r}",
        )


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
    cases = (
        ("tz = [\n", "config parse failed"),
        ('tz = "Invalid/Timezone"\n', "invalid or unavailable"),
    )
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
            expect(
                process.returncode == 0,
                f"Taskdata config reload process failed: {process.stderr[:500]!r}",
            )
            expect(
                expected in process.stdout,
                f"reload error was not actionable: {process.stdout!r}",
            )


def test_config_fingerprint_invalidates_persistent_cache_keys():
    """Config changes update scheduling fingerprints but ignore UI-only edits."""
    previous = os.environ.get("NAUTICAL_CONFIG")
    try:
        with tempfile.TemporaryDirectory() as td:
            config = Path(td) / "config-nautical.toml"
            config.write_text(
                'tz = "UTC"\nlive_panel_footer = "ONE"\n', encoding="utf-8"
            )
            os.environ["NAUTICAL_CONFIG"] = str(config)
            first = core.effective_config_snapshot()
            config.write_text(
                'tz = "Europe/Bucharest"\nlive_panel_footer = "TWO"\n', encoding="utf-8"
            )
            second = core.effective_config_snapshot()
            expect(
                first.get("fingerprint") != second.get("fingerprint"),
                "config edits did not change fingerprint",
            )

            def key_in_fresh_process(config_text):
                config.write_text(config_text, encoding="utf-8")
                env = os.environ.copy()
                env.update(
                    {
                        "NAUTICAL_CONFIG": str(config),
                        "NAUTICAL_TRUST_CONFIG_PATH": "1",
                        "PYTHONPATH": str(ROOT)
                        + (os.pathsep + env.get("PYTHONPATH", "")),
                    }
                )
                process = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import nautical_core; print(nautical_core.cache_key_for_task('w:mon', 'skip'))",
                    ],
                    text=True,
                    capture_output=True,
                    env=env,
                    timeout=8.0,
                )
                expect(
                    process.returncode == 0,
                    f"fresh cache-key process failed: {process.stderr!r}",
                )
                return (process.stdout or "").strip().splitlines()[-1]

            footer_one = key_in_fresh_process('tz = "UTC"\nlive_panel_footer = "ONE"\n')
            footer_two = key_in_fresh_process('tz = "UTC"\nlive_panel_footer = "TWO"\n')
            expect(
                footer_one == footer_two,
                "UI-only config edits unnecessarily invalidated cache key",
            )
            tz_one = key_in_fresh_process('tz = "UTC"\nlive_panel_footer = "TWO"\n')
            tz_two = key_in_fresh_process(
                'tz = "Europe/Bucharest"\nlive_panel_footer = "TWO"\n'
            )
            expect(
                tz_one != tz_two, "scheduler config edits did not invalidate cache key"
            )
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
        process = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            timeout=10,
        )
        expect(
            process.returncode == 0,
            f"configuration drift probe failed: {process.stderr}",
        )
        payload = json.loads(process.stdout)
        expect(
            payload["before"]["status"] == "ok",
            f"fresh config reported drift: {payload}",
        )
        expect(
            payload["edited"]["status"] == "changed",
            f"edited config drift missing: {payload}",
        )
        expect(
            payload["removed"]["status"] == "changed",
            f"removed config drift missing: {payload}",
        )


def test_business_calendar_toml_section_resolves_lazily():
    """A real business-calendar TOML section resolves through the public core facade."""
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config-nautical.toml"
        config_path.write_text(
            '[business_calendar.work]\nanchor = "w:mon..fri"\nomit = "y:04-20"\n',
            encoding="utf-8",
        )
        script = (
            "import json\nfrom datetime import date\nimport nautical_core as core\n"
            "policy = core.get_configured_business_calendar('WORK')\n"
            "print(json.dumps({'names': sorted(core.business_calendar_definitions()), "
            "'open': policy.is_business_day(date(2026, 4, 21)), "
            "'closed': policy.is_business_day(date(2026, 4, 20))}))\n"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env.get("PYTHONPATH", ""))
        env["NAUTICAL_CONFIG"] = str(config_path)
        process = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            timeout=10,
        )
        expect(
            process.returncode == 0,
            f"calendar TOML subprocess failed: {process.stderr}",
        )
        payload = json.loads(process.stdout)
        expect(
            payload == {"names": ["work"], "open": True, "closed": False},
            f"unexpected TOML result: {payload!r}",
        )


def test_core_live_panel_footer_config_defaults_and_customizes():
    """Live panel footer defaults to Nautical and accepts configured text."""
    core_path = ROOT / "nautical_core" / "__init__.py"
    for index, (config_text, expected) in enumerate(
        (("", "NAUTICAL"), ('live_panel_footer = "STATUS"\n', "STATUS"))
    ):
        with tempfile.TemporaryDirectory() as td:
            config = Path(td) / "nautical.toml"
            config.write_text(config_text, encoding="utf-8")
            module = load_core_module(
                str(core_path), f"_nautical_core_live_footer_{index}", str(config)
            )
            expect(
                module.LIVE_PANEL_FOOTER == expected,
                f"unexpected live footer: {module.LIVE_PANEL_FOOTER!r}",
            )


def test_core_uda_aliases_config_defaults_disabled_and_can_enable():
    """Description-based UDA aliases remain opt-in through config."""
    core_path = ROOT / "nautical_core" / "__init__.py"
    cases = (
        ("", False),
        ("enable_uda_aliases = true\n", True),
        ("enable_uda_aliases = false\n", False),
    )
    for index, (config_text, expected) in enumerate(cases):
        with tempfile.TemporaryDirectory() as td:
            config = Path(td) / "nautical.toml"
            config.write_text(config_text, encoding="utf-8")
            module = load_core_module(
                str(core_path), f"_nautical_core_uda_aliases_{index}", str(config)
            )
            expect(
                module.ENABLE_UDA_ALIASES is expected,
                f"unexpected UDA alias setting for {config_text!r}: {module.ENABLE_UDA_ALIASES!r}",
            )


def test_discovered_malformed_config_blocks_taskdata_reload():
    """A malformed Taskdata-discovered config must not silently select defaults."""
    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td)
        (taskdata / "config-nautical.toml").write_text(
            'tz = "Europe/Athens"\n[broken\n', encoding="utf-8"
        )
        env = os.environ.copy()
        env["TASKDATA"] = str(taskdata)
        env["TASKRC"] = str(taskdata / "taskrc")
        env.pop("NAUTICAL_CONFIG", None)
        env["PYTHONPATH"] = str(ROOT)
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os, nautical_core as c; c.reload_taskdata_config(os.environ['TASKDATA'])",
            ],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
        )
        expect(process.returncode != 0, "malformed discovered config was accepted")
        detail = f"{process.stdout}\n{process.stderr}".lower()
        expect("config parse failed" in detail, f"parse failure detail missing: {detail[:800]!r}")


def test_taskdata_reload_exposes_consistent_validated_fingerprints():
    """Lifecycle tools receive one consistent effective configuration identity."""
    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td)
        (taskdata / "config-nautical.toml").write_text(
            'tz = "Europe/Athens"\nseason_hemisphere = "north"\n', encoding="utf-8"
        )
        env = os.environ.copy()
        env.pop("NAUTICAL_CONFIG", None)
        env["PYTHONPATH"] = str(ROOT)
        env["TASKDATA"] = str(taskdata)
        script = (
            "import json, os, nautical_core as c\n"
            "a = c.reload_taskdata_config(os.environ['TASKDATA'])\n"
            "drift = c.configuration_drift()\n"
            "b = c.reload_taskdata_config(os.environ['TASKDATA'])\n"
            "print(json.dumps({'a': a, 'b': b, 'drift': drift,"
            " 'effective': c.effective_config_fingerprint(),"
            " 'scheduler': c.scheduler_config_fingerprint()}))\n"
        )
        process = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
        )
        expect(process.returncode == 0, f"validated reload process failed: {process.stderr[:500]!r}")
        payload = json.loads(process.stdout.strip().splitlines()[-1])
        first, second = payload["a"], payload["b"]
        expect(first["ok"] and second["ok"], f"reload did not report success: {payload!r}")
        expect(first["fingerprint"] == second["fingerprint"], "effective fingerprint changed on identical reload")
        expect(
            first["scheduler_fingerprint"] == second["scheduler_fingerprint"],
            "scheduler fingerprint changed on identical reload",
        )
        expect(first["fingerprint"] == payload["effective"], "reload and core effective fingerprints differ")
        expect(
            first["scheduler_fingerprint"] == payload["scheduler"],
            "reload and core scheduler fingerprints differ",
        )
        expect(payload["drift"]["status"] == "ok", f"identical reload left config drifted: {payload!r}")


def test_hook_on_modify_uda_aliases_route_through_thin_wrapper():
    """Alias-bearing plain modifies must not be swallowed by the thin fast path."""
    hook = find_hook_file("on-modify.nautical")
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "nautical.toml"
        config.write_text('enable_uda_aliases = true\ntz = "UTC"\n', encoding="utf-8")
        old = {
            "uuid": "00000000-0000-4000-8000-000000000115",
            "description": "plain",
            "status": "pending",
        }
        new = dict(old, description="plain a:w:mon")
        env = {
            "NAUTICAL_CONFIG": str(config),
            "NAUTICAL_TRUST_CONFIG_PATH": "1",
            "TASKDATA": td,
            "NO_COLOR": "1",
        }
        process = run_hook_script_raw(hook, json.dumps(old) + "\n" + json.dumps(new), env_extra=env)
        expect(process.returncode == 0, f"enabled alias modify failed: {process.stderr[:600]!r}")
        assert_stdout_json_only(process.stdout)
        normalized = extract_last_json(process.stdout)
        expect(normalized.get("description") == "plain", f"modify alias remained in description: {normalized!r}")
        expect(normalized.get("anchor") == "w:mon", f"modify alias did not reach canonical UDA: {normalized!r}")

        alias_only = dict(old, description="a:w:tue")
        process = run_hook_script_raw(
            hook, json.dumps(old) + "\n" + json.dumps(alias_only), env_extra=env
        )
        expect(process.returncode == 0, f"alias-only modify failed: {process.stderr[:600]!r}")
        assert_stdout_json_only(process.stdout)
        normalized = extract_last_json(process.stdout)
        expect(normalized.get("description") == "plain", f"alias-only modify erased description: {normalized!r}")
        expect(normalized.get("anchor") == "w:tue", f"alias-only modify did not update canonical UDA: {normalized!r}")


def test_hook_on_modify_uda_alias_anchor_change_emits_ack_panel():
    """A description alias changing an existing anchor must acknowledge the edit."""
    hook = find_hook_file("on-modify.nautical")
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "nautical.toml"
        config.write_text('enable_uda_aliases = true\ntz = "UTC"\n', encoding="utf-8")
        old = {
            "uuid": "00000000-0000-4000-8000-000000000117",
            "description": "plain",
            "status": "pending",
            "anchor": "w:mon",
            "chain": "on",
            "chainID": "abcd1234",
            "link": 1,
        }
        new = dict(old, description="plain a:w:tue")
        env = {
            "NAUTICAL_CONFIG": str(config),
            "NAUTICAL_TRUST_CONFIG_PATH": "1",
            "TASKDATA": td,
            "NO_COLOR": "1",
        }
        process = run_hook_script_raw(
            hook, json.dumps(old) + "\n" + json.dumps(new), env_extra=env
        )

    expect(process.returncode == 0, f"alias anchor modify failed: {process.stderr[:600]!r}")
    assert_stdout_json_only(process.stdout)
    normalized = extract_last_json(process.stdout)
    expect(normalized.get("anchor") == "w:tue", f"alias anchor was not normalized: {normalized!r}")
    expect("Nautical recurrence updated" in process.stderr, f"alias anchor acknowledgement missing: {process.stderr!r}")
    expect("Anchor: w:mon" in process.stderr and "w:tue" in process.stderr, f"alias anchor diff missing: {process.stderr!r}")


def test_hook_on_modify_empty_uda_alias_clears_through_thin_wrapper():
    """The native empty-value clearing form survives the wrapper boundary."""
    hook = find_hook_file("on-modify.nautical")
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "nautical.toml"
        config.write_text('enable_uda_aliases = true\ntz = "UTC"\n', encoding="utf-8")
        old = {
            "uuid": "00000000-0000-4000-8000-000000000116",
            "description": "plain",
            "status": "pending",
            "anchor": "w:mon",
            "anchor_mode": "skip",
            "chain": "on",
        }
        new = dict(old, description="plain a:")
        env = {
            "NAUTICAL_CONFIG": str(config),
            "NAUTICAL_TRUST_CONFIG_PATH": "1",
            "TASKDATA": td,
            "NO_COLOR": "1",
        }
        process = run_hook_script_raw(
            hook, json.dumps(old) + "\n" + json.dumps(new), env_extra=env
        )
        expect(process.returncode == 0, f"empty alias clear failed: {process.stderr[:600]!r}")
        assert_stdout_json_only(process.stdout)
        normalized = extract_last_json(process.stdout)
        expect(normalized.get("description") == "plain", f"empty alias remained in description: {normalized!r}")
        expect("anchor" not in normalized, f"empty alias did not clear anchor: {normalized!r}")


TESTS = (
    test_core_invalid_timezone_warns_and_falls_back_to_utc,
    test_explicit_unsafe_config_blocks_scheduling_with_actionable_error,
    test_taskdata_config_reload_fails_closed_for_malformed_toml_and_timezone,
    test_config_fingerprint_invalidates_persistent_cache_keys,
    test_configuration_drift_detects_edit_and_removal,
    test_business_calendar_toml_section_resolves_lazily,
    test_core_live_panel_footer_config_defaults_and_customizes,
    test_core_uda_aliases_config_defaults_disabled_and_can_enable,
    test_discovered_malformed_config_blocks_taskdata_reload,
    test_taskdata_reload_exposes_consistent_validated_fingerprints,
    test_hook_on_modify_uda_aliases_route_through_thin_wrapper,
    test_hook_on_modify_uda_alias_anchor_change_emits_ack_panel,
    test_hook_on_modify_empty_uda_alias_clears_through_thin_wrapper,
)
