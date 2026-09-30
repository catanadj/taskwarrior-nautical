"""On-modify recurrence-state feedback golden tests."""

from __future__ import annotations

import json
import tempfile

from dev_tools.golden_tests.support import (
    assert_stdout_json_only,
    expect,
    find_hook_file,
    load_hook_module,
    modify_effect,
    run_hook_script_raw,
    test_operator_uow,
)


def test_on_modify_promotes_chain_emits_upgrade_panel():
    """Promotion to Nautical should show a small informative panel."""
    hook = find_hook_file("on-modify.nautical")
    module = load_hook_module(hook, "_nautical_on_modify_chain_upgrade_panel_test")
    old = {"uuid": "00000000-0000-4000-8000-000000000446", "description": "plain task", "status": "pending"}
    new = {
        **old,
        "anchor": "w:mon",
        "due": "20260727T090000Z",
        "chain": "off",
    }
    captured = {}
    original_panel = module._panel
    original_print_task = module._print_task
    try:
        def fake_panel(title, rows, *, kind=None):
            captured["title"] = title
            captured["rows"] = list(rows)
            captured["kind"] = kind

        module._panel = fake_panel
        module._print_task = lambda task: captured.setdefault("task", dict(task))
        modify_effect(module, "handle_non_completion", old, new, test_operator_uow())
    finally:
        module._panel = original_panel
        module._print_task = original_print_task

    expect(captured.get("title") == "⚓ Nautical enabled", f"expected upgrade panel, got {captured!r}")
    expect(captured.get("kind") == "note", f"expected note panel, got {captured!r}")
    rows = captured.get("rows") or []
    expect(not any(key == "Chain" for key, _value in rows), f"enabled panel should omit chain:on row: {rows!r}")
    expect(any(key == "Source" and value == "anchor" for key, value in rows), f"expected anchor source row: {rows!r}")
    expect(any(key == "Anchor" and value == "w:mon" for key, value in rows), f"expected added anchor row: {rows!r}")
    expect(any(key == "Natural" and "Monday" in value for key, value in rows), f"expected natural anchor explanation: {rows!r}")
    expect(any(key == "Mode" and value.startswith("SKIP —") for key, value in rows), f"expected anchor mode explanation: {rows!r}")
    expect(any(key == "First next" for key, _value in rows), f"expected first calculated occurrence: {rows!r}")
    expect(new.get("chain") == "on", f"promotion should set chain:on: {new!r}")
    expect(bool((new.get("chainID") or "").strip()), f"promotion should stamp chainID: {new!r}")


def test_on_modify_promotes_cp_emits_period_explanation():
    """Promotion by cp should show the configured period and readable meaning."""
    hook = find_hook_file("on-modify.nautical")
    module = load_hook_module(hook, "_nautical_on_modify_cp_upgrade_panel_test")
    old = {"uuid": "00000000-0000-4000-8000-000000000448", "description": "plain task", "status": "pending"}
    new = {**old, "cp": "7d", "due": "20260727T090000Z", "chain": "off"}
    captured = {}
    original_panel = module._panel
    original_print_task = module._print_task
    try:
        module._panel = lambda title, rows, *, kind=None: captured.update(title=title, rows=list(rows), kind=kind)
        module._print_task = lambda _task: None
        modify_effect(module, "handle_non_completion", old, new, test_operator_uow())
    finally:
        module._panel = original_panel
        module._print_task = original_print_task
    rows = captured.get("rows") or []
    expect(captured.get("title") == "⚓ Nautical enabled", f"expected upgrade panel, got {captured!r}")
    expect(any(key == "Period" and value == "7d" for key, value in rows), f"expected added period row: {rows!r}")
    expect(any(key == "Natural" and value == "Every 7d" for key, value in rows), f"expected natural period explanation: {rows!r}")
    expect(any(key == "First next" for key, _value in rows), f"expected first calculated occurrence: {rows!r}")


def test_on_modify_disables_chain_emits_disabled_panel():
    """Disabling Nautical recurrence should show a small informative panel."""
    hook = find_hook_file("on-modify.nautical")
    module = load_hook_module(hook, "_nautical_on_modify_chain_disabled_panel_test")
    base_old = {
        "uuid": "00000000-0000-4000-8000-000000000447",
        "description": "nautical task",
        "status": "pending",
        "anchor": "w:mon",
        "chain": "on",
        "chainID": "abcd1234",
    }
    cases = (
        {"label": "chain_off", "new": {**base_old, "chain": "off"}, "reason": "disabled because chain:off", "source": "anchor"},
        {
            "label": "fields_cleared",
            "new": {"uuid": base_old["uuid"], "description": base_old["description"], "status": base_old["status"], "chain": "on", "chainID": "abcd1234", "anchor_mode": "skip"},
            "reason": "no longer has Nautical recurrence fields",
            "source": None,
        },
    )
    for case in cases:
        old = dict(base_old)
        new = dict(case["new"])
        captured = {"panels": []}
        original_panel = module._panel
        original_print_task = module._print_task
        try:
            def fake_panel(title, rows, *, kind=None):
                captured["panels"].append((title, list(rows), kind))

            module._panel = fake_panel
            module._print_task = lambda task: captured.setdefault("task", dict(task))
            modify_effect(module, "handle_non_completion", old, new, test_operator_uow())
        finally:
            module._panel = original_panel
            module._print_task = original_print_task
        disabled = [panel for panel in captured["panels"] if panel[0] == "⚓ Nautical disabled"]
        expect(disabled, f"{case['label']} expected disabled panel: {captured!r}")
        _title, rows, kind = disabled[-1]
        expect(kind == "disabled", f"{case['label']} expected disabled panel kind: {captured!r}")
        expect(any(key == "Reason" and case["reason"] in str(value) for key, value in rows), f"{case['label']} expected reason: {rows!r}")
        if case["source"] is None:
            expect(not any(key == "Source" for key, _value in rows), f"{case['label']} should omit source: {rows!r}")
        else:
            expect(any(key == "Source" and value == case["source"] for key, value in rows), f"{case['label']} expected source: {rows!r}")
        expect(any(key == "Chain" and value == "off" for key, value in rows), f"{case['label']} expected chain:off: {rows!r}")
        if case["label"] == "fields_cleared":
            expect(any(title == "⛔ Nautical chain stopped" and panel_kind == "summary" for title, _rows, panel_kind in captured["panels"]), f"{case['label']} should include finished-chain summary: {captured!r}")
            summary = next(rows for title, rows, _kind in captured["panels"] if title == "⛔ Nautical chain stopped")
            expect(any(label == "Reason" and "removed" in str(value) for label, value in summary), f"summary should explain removed recurrence: {captured!r}")
        elif case["label"] == "chain_off":
            expect(any(title == "⛔ Nautical chain stopped" and panel_kind == "summary" for title, _rows, panel_kind in captured["panels"]), f"{case['label']} should include finished-chain summary: {captured!r}")
        expect(new.get("chain") == "off", f"{case['label']} should set chain:off: {new!r}")


def test_on_modify_resumes_chain_emits_resumed_panel():
    """Explicitly resuming an existing recurrence should acknowledge its effect."""
    hook = find_hook_file("on-modify.nautical")
    module = load_hook_module(hook, "_nautical_on_modify_chain_resumed_panel_test")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000451",
        "description": "paused nautical task",
        "status": "pending",
        "anchor": "w:mon",
        "due": "20260727T090000Z",
        "chain": "off",
        "chainID": "abcd1234",
        "chainMax": 8,
    }
    new = {**old, "chain": "on"}
    captured = {}
    original_panel = module._panel
    original_print_task = module._print_task
    try:
        def fake_panel(title, rows, *, kind=None):
            captured.update(title=title, rows=list(rows), kind=kind)

        module._panel = fake_panel
        module._print_task = lambda task: captured.setdefault("task", dict(task))
        modify_effect(module, "handle_non_completion", old, new, test_operator_uow())
    finally:
        module._panel = original_panel
        module._print_task = original_print_task
    expect(captured.get("title") == "⚓ Nautical resumed", f"expected resumed panel, got {captured!r}")
    expect(captured.get("kind") == "note", f"expected note panel, got {captured!r}")
    rows = captured.get("rows") or []
    expect(("Source", "anchor") in rows, f"expected anchor source row, got {rows!r}")
    expect(("Chain", "[dim]off[/] [cyan]→[/] [bold]on[/]") in rows, f"expected styled transition row: {rows!r}")
    expect(captured.get("task") == new, f"modified task should still be printed: {captured!r}")


def test_on_modify_resume_wrapper_preserves_json_and_emits_panel():
    """The thin wrapper routes chain resume without polluting stdout."""
    hook = find_hook_file("on-modify.nautical")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000452",
        "description": "paused nautical task",
        "status": "pending",
        "cp": "1d",
        "chain": "off",
        "chainID": "abcd1234",
        "link": 3,
    }
    new = {**old, "chain": "on"}
    with tempfile.TemporaryDirectory() as td:
        process = run_hook_script_raw(hook, json.dumps(old) + "\n" + json.dumps(new), env_extra={"TASKDATA": td})
    expect(process.returncode == 0, f"resume hook failed: {process.stderr!r}")
    expect(assert_stdout_json_only(process.stdout) == new, f"resume hook changed task JSON: {process.stdout!r}")
    expect("Nautical resumed" in process.stderr, f"resume panel missing from stderr: {process.stderr!r}")
    expect("off → on" in process.stderr, f"chain transition missing from panel: {process.stderr!r}")


TESTS = (
    test_on_modify_promotes_chain_emits_upgrade_panel,
    test_on_modify_promotes_cp_emits_period_explanation,
    test_on_modify_disables_chain_emits_disabled_panel,
    test_on_modify_resumes_chain_emits_resumed_panel,
    test_on_modify_resume_wrapper_preserves_json_and_emits_panel,
)
