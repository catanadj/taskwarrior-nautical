"""Configuration and Taskdata discovery golden tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from dev_tools.golden_tests.support import (
    assert_stdout_json_only,
    expect,
    extract_last_json,
    find_hook_file,
    run_hook_script_raw,
)


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
    test_hook_on_modify_uda_aliases_route_through_thin_wrapper,
    test_hook_on_modify_uda_alias_anchor_change_emits_ack_panel,
    test_hook_on_modify_empty_uda_alias_clears_through_thin_wrapper,
)
