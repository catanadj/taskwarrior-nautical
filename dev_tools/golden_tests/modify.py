"""On-modify recurrence-state feedback golden tests."""

from __future__ import annotations

import json
from datetime import date, timedelta, timezone
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


def test_on_modify_recurrence_update_emits_ack_panel():
    """Changing recurrence settings on an existing Nautical task is acknowledged."""
    module = load_hook_module(find_hook_file("on-modify.nautical"), "_nautical_on_modify_recurrence_update_panel_test")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000449",
        "description": "nautical task",
        "status": "pending",
        "anchor": "w:mon",
        "due": "20260727T090000Z",
        "chain": "on",
        "chainID": "abcd1234",
        "link": 1,
    }
    new = {**old, "anchor": "w:tue,thu"}
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
    rows = captured.get("rows") or []
    expect(captured.get("title") == "⚓ Nautical recurrence updated", f"expected recurrence update panel: {captured!r}")
    expect(captured.get("kind") == "note", f"expected note panel: {captured!r}")
    expect(("Changed", "Anchor: [dim]w:mon[/] [cyan]→[/] [bold]w:tue,thu[/]") in rows, f"expected styled anchor change: {rows!r}")
    expect(any(key == "Natural" and "Tuesday" in str(value) and "Thursday" in str(value) for key, value in rows), f"expected natural row: {rows!r}")
    expect(any(key == "First next" for key, _value in rows), f"expected recalculated first occurrence: {rows!r}")
    expect(not any(key == "Chain" for key, _value in rows), f"recurrence panel should omit redundant chain:on: {rows!r}")
    expect(captured.get("task") == new, f"modified task should still be printed: {captured!r}")


def test_on_modify_recurrence_update_groups_and_flattens_changes():
    """Multi-field recurrence updates stay grouped and readable in one-line modes."""
    module = load_hook_module(find_hook_file("on-modify.nautical"), "_nautical_recurrence_update_layout_test")
    changes = [("anchor", "w:mon", "w:tue"), ("chainMax", "5", "8")]
    rich_rows = [
        ("Changed", "Anchor: [dim]w:mon[/] [cyan]→[/] [bold]w:tue[/]"),
        ("Changed", "Max links: [dim]5[/] [cyan]→[/] [bold]8[/]"),
    ]
    feedback = module.core._import_sibling("modify_feedback")
    grouped = feedback._recurrence_update_panel_rows(
        changes,
        rich_rows,
        panel_mode=module.core.PANEL_MODE,
        strip_markup=module.core.strip_rich_markup,
    )
    expect(grouped[1][0] is None, f"expected spacing between recurrence and limits: {grouped!r}")
    previous_mode = module.core.PANEL_MODE
    try:
        module.core.PANEL_MODE = "text"
        flattened = feedback._recurrence_update_panel_rows(
            changes,
            rich_rows,
            panel_mode=module.core.PANEL_MODE,
            strip_markup=module.core.strip_rich_markup,
        )
    finally:
        module.core.PANEL_MODE = previous_mode
    expect(flattened[0][0] == "Changes", f"expected one-line change summary: {flattened!r}")
    expect("Anchor:" in flattened[0][1] and "Max links:" in flattened[0][1], f"summary omitted a change: {flattened!r}")


def test_on_modify_native_until_update_explains_carry():
    """Changing native until acknowledges its exact or calendar carry policy."""
    module = load_hook_module(find_hook_file("on-modify.nautical"), "_nautical_on_modify_until_update_panel_test")
    due = module.core.build_local_datetime(date(2026, 8, 3), (10, 0)).astimezone(timezone.utc)
    old_until = module.core.build_local_datetime(date(2026, 8, 3), (18, 0)).astimezone(timezone.utc)
    new_until = module.core.build_local_datetime(date(2026, 8, 4), (0, 0)).astimezone(timezone.utc) + timedelta(seconds=1)
    old = {
        "uuid": "00000000-0000-4000-8000-000000000446",
        "description": "nautical expiration update",
        "status": "pending",
        "cp": "1d",
        "due": module.core.fmt_isoz(due),
        "until": module.core.fmt_isoz(old_until),
        "chain": "on",
        "chainID": "abcd1234",
    }
    new = {**old, "until": module.core.fmt_isoz(new_until)}
    captured = {}
    original_panel = module._panel
    original_print_task = module._print_task
    try:
        module._panel = lambda title, rows, *, kind=None: captured.update(title=title, rows=list(rows), kind=kind)
        module._print_task = lambda task: captured.setdefault("task", dict(task))
        modify_effect(module, "handle_non_completion", old, new, test_operator_uow())
    finally:
        module._panel = original_panel
        module._print_task = original_print_task
    rows = captured.get("rows") or []
    expect(captured.get("title") == "⚓ Nautical recurrence updated", f"unexpected expiration panel: {captured!r}")
    expect(captured.get("kind") == "note", f"unexpected expiration panel style: {captured!r}")
    expect(any(label == "Changed" and "Expiration:" in str(value) and "2026-08-03" in str(value) and "2026-08-04" in str(value) for label, value in rows), f"missing expiration diff: {rows!r}")
    expect(("Carry", "Exact · 14h 00m 01s after occurrence") in rows, f"missing exact carry explanation: {rows!r}")
    expect(captured.get("task") == new, f"modified task should still be printed: {captured!r}")


def test_on_modify_limit_update_emits_effective_boundaries():
    """Changing chain limits acknowledges both boundaries without speculative dates."""
    module = load_hook_module(find_hook_file("on-modify.nautical"), "_nautical_on_modify_limit_update_panel_test")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000450",
        "description": "limited nautical task",
        "status": "pending",
        "cp": "1d",
        "chain": "on",
        "chainID": "abcd1234",
        "link": 2,
        "chainMax": 5,
        "chainUntil": "20990810T070000Z",
    }
    new = {**old, "chainMax": 8, "chainUntil": "20990820T070000Z"}
    cleared = dict(new)
    cleared.pop("chainMax")
    cleared.pop("chainUntil")
    panels = []
    original_panel = module._panel
    original_print_task = module._print_task
    try:
        module._panel = lambda title, rows, *, kind=None: panels.append((title, list(rows), kind))
        module._print_task = lambda _task: None
        modify_effect(module, "handle_non_completion", old, new, test_operator_uow())
        modify_effect(module, "handle_non_completion", new, cleared, test_operator_uow())
    finally:
        module._panel = original_panel
        module._print_task = original_print_task
    expect(len(panels) == 2, f"each limit update should emit one panel: {panels!r}")
    title, rows, kind = panels[0]
    expect(title == "⚓ Nautical recurrence updated" and kind == "note", f"unexpected limit panel: {panels!r}")
    expect(("Changed", "Max links: [dim]5[/] [cyan]→[/] [bold]8[/]") in rows, f"missing chainMax update: {rows!r}")
    expect(
        any(label == "Changed" and "Chain end point:" in value and "2099-08-10" in value and "2099-08-20" in value for label, value in rows),
        f"missing localized chainUntil update: {rows!r}",
    )
    expect(("Final link", "#8") in rows, f"missing final link boundary: {rows!r}")
    expect(
        sum(label == "Changed" and "Chain end point:" in str(value) for label, value in rows) == 1,
        f"chain end point should not be repeated: {rows!r}",
    )
    expect(("Effective", "Whichever boundary is reached first") in rows, f"missing effective limit rule: {rows!r}")
    expect(panels[1][0] == "⚓ Nautical recurrence updated", f"cleared limits should be acknowledged: {panels!r}")
    expect(("Chain limits", "None") in panels[1][1], f"clearing both limits should be explicit: {panels[1]!r}")
    expect(("Removed", "Max links: [dim]8[/]") in panels[1][1], f"cleared max link should be marked removed: {panels[1]!r}")
    expect(
        any(label == "Removed" and "Chain end point:" in str(value) for label, value in panels[1][1]),
        f"cleared chain end should be marked removed: {panels[1]!r}",
    )


def test_on_modify_reports_business_calendar_displacement():
    """Completion feedback reports captured calendar rolls in every panel mode."""
    module = load_hook_module(find_hook_file("on-modify.nautical"), "_nautical_on_modify_calendar_displacement_test")
    module._SHOW_TIMELINE_GAPS = False
    module._CHAIN_COLOR_PER_CHAIN = False
    module._append_next_wait_sched_rows = lambda *_args, **_kwargs: None
    module._format_root_and_age = lambda *_args, **_kwargs: "abcd1234"
    module._timeline_lines = lambda *_args, **_kwargs: []
    module._panel_line = lambda *_args, **_kwargs: None
    panels = []
    module._panel = lambda title, rows, **_kwargs: panels.append((title, list(rows)))
    policy = module.core.resolve_business_calendar_config({"work": {"anchor": "w:mon..fri", "omit": "y:04-24"}})["work"]
    dnf = module.core.validate_anchor_expr_strict("y:04-24@nbd@t=09:00")
    previous_mode = module.core.PANEL_MODE
    try:
        module.core.PANEL_MODE = "minimal"
        with module.core.use_business_calendar(policy), module.core.capture_business_calendar_displacements():
            child_date, _meta = module.core.next_after_expr(dnf, date(2026, 4, 20), date(2026, 4, 20))
            child_due = module.core.build_local_datetime(child_date, (9, 0))
            module._presentation_effects.render_anchor_completion_feedback(
                new={"anchor": "y:04-24@nbd@t=09:00", "anchor_mode": "skip", "bc": "work", "uuid": "00000000-0000-4000-8000-000000000127", "chainID": "calendar-chain"},
                child={"uuid": "00000000-0000-4000-8000-000000000128"},
                child_due=child_due,
                child_short="beeswax",
                next_no=2,
                parent_short="00000000",
                cap_no=None,
                finals=[],
                now_utc=module.core.now_utc(),
                until_dt=None,
                until_cap_no=None,
                dnf=dnf,
                meta={"mode": "skip"},
                stripped_attrs=[],
                deferred_spawn=False,
                spawn_intent_id=None,
                chain_by_short=None,
                analytics_advice=None,
                integrity_warnings=None,
                base_no=1,
            )
    finally:
        module.core.PANEL_MODE = previous_mode
    calendar_panels = [rows for title, rows in panels if title == "⚓ Business calendar adjusted"]
    expect(len(calendar_panels) == 1, f"completion should emit one displacement panel: {panels!r}")
    rows = calendar_panels[0]
    expect(("Calendar", "work") in rows, f"calendar name missing: {rows!r}")
    expect(("Original", "Fri 2026-04-24") in rows, f"original occurrence missing: {rows!r}")
    expect(("Adjusted", "Mon 2026-04-27 (+3d)") in rows, f"adjusted occurrence missing: {rows!r}")


TESTS = (
    test_on_modify_reports_business_calendar_displacement,
    test_on_modify_promotes_chain_emits_upgrade_panel,
    test_on_modify_promotes_cp_emits_period_explanation,
    test_on_modify_disables_chain_emits_disabled_panel,
    test_on_modify_resumes_chain_emits_resumed_panel,
    test_on_modify_resume_wrapper_preserves_json_and_emits_panel,
    test_on_modify_recurrence_update_emits_ack_panel,
    test_on_modify_recurrence_update_groups_and_flattens_changes,
    test_on_modify_native_until_update_explains_carry,
    test_on_modify_limit_update_emits_effective_boundaries,
)
