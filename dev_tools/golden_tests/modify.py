"""On-modify recurrence-state feedback golden tests."""

from __future__ import annotations

import json
from datetime import date, timedelta, timezone
import tempfile
from pathlib import Path

from dev_tools.golden_tests.support import (
    assert_stdout_json_only,
    build_child_draft_for_test,
    carry_relative_datetime,
    expect,
    find_hook_file,
    load_hook_module,
    modify_effect,
    run_hook_script,
    run_hook_script_raw,
    strip_markup,
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


def test_on_modify_carry_wall_clock_across_dst():
    """carry-forward should preserve local wall-clock offset across DST."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_carry_dst_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    try:
        from zoneinfo import ZoneInfo
    except Exception:
        return

    previous_tz_name = mod.core.LOCAL_TZ_NAME
    previous_tz = mod.core._LOCAL_TZ
    try:
        mod.core.LOCAL_TZ_NAME = "America/New_York"
        mod.core._LOCAL_TZ = ZoneInfo("America/New_York")

        due_local = date(2025, 3, 9)
        due_utc = mod.core.build_local_datetime(due_local, (1, 30))
        wait_utc = mod.core.build_local_datetime(due_local, (3, 30))

        child_due_utc = mod.core.build_local_datetime(date(2025, 3, 10), (1, 30))

        parent = {
            "due": mod.core.fmt_isoz(due_utc),
            "wait": mod.core.fmt_isoz(wait_utc),
        }
        child = {"due": mod.core.fmt_isoz(child_due_utc)}

        carry_relative_datetime(mod, parent, child, child_due_utc, "wait")
        wait_child = mod.core.parse_dt_any(child.get("wait"))
        wait_local = mod.core.to_local(wait_child)

        expect(wait_local.hour == 3 and wait_local.minute == 30, f"unexpected local wait: {wait_local}")
    finally:
        mod.core.LOCAL_TZ_NAME = previous_tz_name
        mod.core._LOCAL_TZ = previous_tz



def test_on_modify_build_child_carries_until_across_dst():
    """native until should retain its local wall-clock offset from the recurrence due."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_carry_until_dst_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    try:
        from zoneinfo import ZoneInfo
    except Exception:
        return

    previous_tz_name = mod.core.LOCAL_TZ_NAME
    previous_tz = mod.core._LOCAL_TZ
    try:
        mod.core.LOCAL_TZ_NAME = "America/New_York"
        mod.core._LOCAL_TZ = ZoneInfo("America/New_York")

        parent_due = mod.core.build_local_datetime(date(2025, 3, 8), (9, 0))
        parent_until = mod.core.build_local_datetime(date(2025, 3, 9), (17, 0))
        child_due = mod.core.build_local_datetime(date(2025, 3, 15), (9, 0))
        parent = {
            "uuid": "00000000-0000-4000-8000-000000000994",
            "status": "completed",
            "link": 1,
            "due": mod.core.fmt_isoz(parent_due),
            "until": mod.core.fmt_isoz(parent_until),
            "cp": "7d",
            "chainID": "cid_until",
        }

        child = build_child_draft_for_test(mod,
            parent,
            child_due,
            "due",
            2,
            "beef",
            "cp",
            0,
            None,
        )
        child_until_local = mod.core.to_local(mod.core.parse_dt_any(child.get("until")))
        expect(
            child_until_local.date() == date(2025, 3, 16)
            and (child_until_local.hour, child_until_local.minute) == (17, 0),
            f"unexpected carried until: {child_until_local}",
        )

        pending_parent = dict(parent, status="pending")
        moved_parent = dict(pending_parent, due=mod.core.fmt_isoz(child_due))
        expect(
            mod._transition_effects.preserve_native_until_on_target_change(pending_parent, moved_parent, "cp"),
            "ordinary target move skipped native-until carry across DST",
        )
        moved_until_local = mod.core.to_local(mod.core.parse_dt_any(moved_parent.get("until")))
        expect(
            moved_until_local.date() == date(2025, 3, 16)
            and (moved_until_local.hour, moved_until_local.minute) == (17, 0),
            f"ordinary target move changed calendar expiration across DST: {moved_until_local}",
        )
    finally:
        mod.core.LOCAL_TZ_NAME = previous_tz_name
        mod.core._LOCAL_TZ = previous_tz



def test_on_modify_native_until_calendar_and_exact_carry_policy():
    """native until should use calendar carry by default and exact carry with the +1s marker."""
    import nautical_core.chain_integrity_lifecycle as reconcile
    from nautical_core.task_codec import DEFAULT_TASK_CODEC

    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_native_until_carry_policy_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    due_0900 = mod.core.build_local_datetime(date(2026, 7, 20), (9, 0))
    due_1300 = mod.core.build_local_datetime(date(2026, 7, 20), (13, 0))
    due_1800 = mod.core.build_local_datetime(date(2026, 7, 20), (18, 0))
    until_2300 = mod.core.build_local_datetime(date(2026, 7, 20), (23, 0))

    def build(kind, child_due, until_value):
        parent = {
            "uuid": "00000000-0000-4000-8000-000000000995",
            "description": "native until carry test",
            "status": "completed",
            "link": 1,
            "due": mod.core.fmt_isoz(due_0900),
            "until": mod.core.fmt_isoz(until_value),
            "chainID": "cid_until_policy",
        }
        if kind == "cp":
            parent["cp"] = "8h"
        elif kind == "anchor":
            parent.update({"anchor": "d:*@t=09:00,13:00", "anchor_mode": "skip"})
        else:
            parent.update({"anchor_file": "calendar.csv", "anchor_mode": "skip"})
        return build_child_draft_for_test(mod,
            parent,
            child_due,
            "due",
            2,
            "beef",
            kind,
            0,
            None,
        )

    for kind in ("cp", "anchor"):
        child = build(kind, due_1300, until_2300)
        carried = mod.core.to_local(mod.core.parse_dt_any(child.get("until")))
        expect(
            carried.date() == date(2026, 7, 20)
            and (carried.hour, carried.minute, carried.second) == (23, 0, 0),
            f"{kind} should keep a same-day calendar expiration: {carried}",
        )

    until_1700 = mod.core.build_local_datetime(date(2026, 7, 20), (17, 0))
    cp_rollover = build("cp", due_1800, until_1700)
    carried_rollover = mod.core.to_local(mod.core.parse_dt_any(cp_rollover.get("until")))
    expect(
        carried_rollover.date() == date(2026, 7, 21)
        and (carried_rollover.hour, carried_rollover.minute) == (17, 0),
        f"CP should roll an elapsed calendar expiration to the next local day: {carried_rollover}",
    )

    until_eod = mod.core.build_local_datetime(date(2026, 7, 20), (23, 59)) + timedelta(seconds=59)
    eod_child = build("anchor", due_1300, until_eod)
    carried_eod = mod.core.to_local(mod.core.parse_dt_any(eod_child.get("until")))
    expect(
        carried_eod.date() == date(2026, 7, 20)
        and (carried_eod.hour, carried_eod.minute, carried_eod.second) == (23, 59, 59),
        f"end-of-day expiration should retain calendar carry: {carried_eod}",
    )

    until_exact = until_2300 + timedelta(seconds=1)
    for kind in ("cp", "anchor"):
        exact_child = build(kind, due_1300, until_exact)
        carried_exact = mod.core.to_local(mod.core.parse_dt_any(exact_child.get("until")))
        expect(
            carried_exact.date() == date(2026, 7, 21)
            and (carried_exact.hour, carried_exact.minute, carried_exact.second) == (3, 0, 1),
            f"{kind} +1s expiration should retain the exact elapsed window: {carried_exact}",
        )

    expired_parent = {
        "uuid": "00000000-0000-4000-8000-000000000996",
        "description": "native until reconcile test",
        "status": "deleted",
        "anchor": "w:mon@t=09:00,13:00",
        "anchor_mode": "skip",
        "chain": "on",
        "chainID": "cid_until_reconcile",
        "link": 1,
        "due": mod.core.fmt_isoz(due_0900),
        "until": mod.core.fmt_isoz(until_2300),
        "end": mod.core.fmt_isoz(until_2300),
    }
    plan = reconcile.plan_recovery_decision(
        DEFAULT_TASK_CODEC.decode_row(expired_parent, source_query="golden recovery"),
        existing_children=[], hook=mod,
    )
    child = (
        plan.child_observation.to_mapping()
        if getattr(plan, "child_observation", None) is not None
        else plan.plan.child_dict()
        if getattr(plan, "plan", None) is not None
        else {}
    )
    reconciled_until = mod.core.to_local(mod.core.parse_dt_any(child.get("until")))
    expect(getattr(getattr(plan, "plan", None), "action", None).value == "spawn_child", f"expired anchor should produce a child plan: {plan}")
    expect(
        reconciled_until.date() == date(2026, 7, 20)
        and (reconciled_until.hour, reconciled_until.minute) == (23, 0),
        f"reconciled child should use the same calendar expiration policy: {reconciled_until}",
    )

    early_until_parent = {
        "uuid": "00000000-0000-4000-8000-000000000997",
        "description": "native until end-of-day fallback test",
        "status": "deleted",
        "anchor": "w:mon@t=09:00,13:00",
        "anchor_mode": "skip",
        "chain": "on",
        "chainID": "cid_until_reconcile_eod",
        "link": 1,
        "due": mod.core.fmt_isoz(due_0900),
        "until": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2026, 7, 20), (9, 10))),
        "end": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2026, 7, 20), (9, 10))),
    }
    from nautical_core.chain_generation import ChainGenerationService

    class FailingBuildGeneration(ChainGenerationService):
        def build_child_from_parent(self, *_args, **_kwargs):
            raise ValueError("native until must be later than the child recurrence target")

    failing_generation = FailingBuildGeneration.from_core(mod.core)
    untyped_plan = reconcile.plan_recovery_decision(
        DEFAULT_TASK_CODEC.decode_row(early_until_parent, source_query="golden recovery"),
        existing_children=[],
        hook=mod,
        generation=failing_generation,
    )
    expect(
        getattr(getattr(untyped_plan, "plan", None), "action", None).value == "spawn_child",
        f"typed reconcile planning should not depend on the removed builder seam: {untyped_plan}",
    )

    early_plan = reconcile.plan_recovery_decision(
        DEFAULT_TASK_CODEC.decode_row(early_until_parent, source_query="golden recovery"),
        existing_children=[], hook=mod,
    )
    early_child = (
        early_plan.child_observation.to_mapping()
        if getattr(early_plan, "child_observation", None) is not None
        else early_plan.plan.child_dict()
        if getattr(early_plan, "plan", None) is not None
        else {}
    )
    early_until = mod.core.to_local(mod.core.parse_dt_any(early_child.get("until")))
    expect(getattr(getattr(early_plan, "plan", None), "action", None).value == "spawn_child", f"expired anchor should still produce a child plan: {early_plan}")
    expect(
        early_until.date() == date(2026, 7, 20)
        and (early_until.hour, early_until.minute, early_until.second) == (23, 59, 59),
        f"reconcile should fall back to end of day for expired anchor carry: {early_until}",
    )



def test_on_modify_native_until_exact_carry_preserves_elapsed_time_across_dst():
    """the +1s expiration marker should preserve elapsed seconds instead of local clock offset."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_native_until_exact_dst_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    try:
        from zoneinfo import ZoneInfo
    except Exception:
        return

    previous_tz_name = mod.core.LOCAL_TZ_NAME
    previous_tz = mod.core._LOCAL_TZ
    try:
        mod.core.LOCAL_TZ_NAME = "America/New_York"
        mod.core._LOCAL_TZ = ZoneInfo("America/New_York")
        parent_due = mod.core.build_local_datetime(date(2025, 3, 8), (9, 0))
        parent_until = mod.core.build_local_datetime(date(2025, 3, 9), (17, 0)) + timedelta(seconds=1)
        child_due = mod.core.build_local_datetime(date(2025, 3, 15), (9, 0))
        parent = {
            "uuid": "00000000-0000-4000-8000-000000000997",
            "status": "completed",
            "due": mod.core.fmt_isoz(parent_due),
            "until": mod.core.fmt_isoz(parent_until),
            "cp": "7d",
            "chainID": "cid_until_exact_dst",
        }

        child = build_child_draft_for_test(mod,
            parent,
            child_due,
            "due",
            2,
            "beef",
            "cp",
            0,
            None,
        )
        carried = mod.core.parse_dt_any(child.get("until"))
        expect(
            carried - child_due == parent_until - parent_due,
            f"exact expiration should preserve UTC elapsed time: {carried} from {child_due}",
        )
        carried_local = mod.core.to_local(carried)
        expect(
            carried_local.date() == date(2025, 3, 16)
            and (carried_local.hour, carried_local.minute, carried_local.second) == (16, 0, 1),
            f"exact DST carry should not preserve the old local clock offset: {carried_local}",
        )
    finally:
        mod.core.LOCAL_TZ_NAME = previous_tz_name
        mod.core._LOCAL_TZ = previous_tz



def test_native_until_calendar_slot_guard_rejects_impossible_anchor_expirations():
    """calendar expiration should reject fixed anchor slots at or after its clock time."""
    add_hook = find_hook_file("on-add.nautical")
    modify_hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(modify_hook, "_nautical_native_until_slot_guard_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    anchor_day = date(2030, 7, 1)  # Monday, deliberately beyond the test clock.
    due = mod.core.build_local_datetime(anchor_day, (9, 0))
    until_1900 = mod.core.build_local_datetime(anchor_day, (19, 0))
    base = {
        "uuid": "00000000-0000-4000-8000-000000000998",
        "description": "invalid anchor expiration slot",
        "status": "pending",
        "entry": "20300630T080000Z",
        "anchor": "w:mon@t=09:00,18:00,20:00",
        "anchor_mode": "skip",
        "due": "20300701T090000Z",
        "until": "20300701T190000Z",
    }

    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "config-nautical.toml"
        config.write_text('tz = "UTC"\n', encoding="utf-8")
        env = {"NO_COLOR": "1", "NAUTICAL_CONFIG": str(config)}
        added = run_hook_script(add_hook, dict(base), env_extra=env)
        expect(added.returncode != 0, "on-add accepted a same-day expiration before an anchor slot")
        expect(not added.stdout.strip(), f"rejected on-add leaked stdout: {added.stdout!r}")
        added_stderr = strip_markup(added.stderr)
        expect("Invalid expiration window" in added_stderr, f"missing on-add expiration panel: {added_stderr!r}")
        expect("20:00" in added_stderr, f"missing conflicting anchor slot: {added_stderr!r}")

        exact = run_hook_script(
            add_hook,
            dict(base, until="20300701T190001Z"),
            env_extra=env,
        )
        expect(exact.returncode == 0, f"+1s exact expiration should bypass calendar slot rejection: {exact.stderr!r}")

        old = dict(base, chain="on", chainID="cid_until_slots", link=1, until="20300701T210000Z")
        new = dict(old, until="20300701T190000Z")
        modified = run_hook_script_raw(
            modify_hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra=dict(env, TASKDATA=td),
        )
        anchor_old = dict(
            base,
            anchor="w:mon@t=09:00",
            chain="on",
            chainID="cid_until_anchor_edit",
            link=1,
        )
        anchor_invalid = run_hook_script_raw(
            modify_hook,
            json.dumps(anchor_old) + "\n" + json.dumps(dict(anchor_old, anchor="w:mon@t=09:00,20:00")),
            env_extra=dict(env, TASKDATA=td),
        )
        anchor_valid = run_hook_script_raw(
            modify_hook,
            json.dumps(anchor_old) + "\n" + json.dumps(dict(anchor_old, anchor="w:mon@t=09:00,18:00")),
            env_extra=dict(env, TASKDATA=td),
        )
        cp_old = dict(
            anchor_old,
            anchor=None,
            anchor_mode=None,
            cp="7d",
            chainID="cid_until_cp_to_anchor",
        )
        cp_to_anchor = dict(
            cp_old,
            cp=None,
            anchor="w:mon@t=09:00,20:00",
            anchor_mode="skip",
        )
        converted = run_hook_script_raw(
            modify_hook,
            json.dumps(cp_old) + "\n" + json.dumps(cp_to_anchor),
            env_extra=dict(env, TASKDATA=td),
        )
    expect(modified.returncode != 0, "on-modify accepted a same-day expiration before an anchor slot")
    expect(not modified.stdout.strip(), f"rejected on-modify leaked stdout: {modified.stdout!r}")
    expect("Invalid expiration window" in strip_markup(modified.stderr), f"missing modify expiration panel: {modified.stderr!r}")
    expect(anchor_invalid.returncode != 0, "on-modify accepted an anchor edit adding a slot after expiration")
    expect(not anchor_invalid.stdout.strip(), f"rejected anchor edit leaked stdout: {anchor_invalid.stdout!r}")
    expect("20:00" in strip_markup(anchor_invalid.stderr), f"missing edited anchor slot: {anchor_invalid.stderr!r}")
    expect(anchor_valid.returncode == 0, f"valid anchor slot edit was rejected: {anchor_valid.stderr!r}")
    expect(
        assert_stdout_json_only(anchor_valid.stdout).get("anchor") == "w:mon@t=09:00,18:00",
        f"valid anchor edit changed unexpectedly: {anchor_valid.stdout!r}",
    )
    expect(converted.returncode != 0, "CP-to-anchor conversion bypassed expiration slot validation")
    expect(not converted.stdout.strip(), f"rejected CP-to-anchor conversion leaked stdout: {converted.stdout!r}")

    with tempfile.TemporaryDirectory() as td:
        anchor_dir = Path(td) / "anchor"
        anchor_dir.mkdir()
        (anchor_dir / "events.csv").write_text(f"date\n{anchor_day.isoformat()}\n", encoding="utf-8")
        config = Path(td) / "config-nautical.toml"
        config.write_text(f'tz = "UTC"\nanchor_file_dir = "{anchor_dir}"\n', encoding="utf-8")
        file_task = dict(base, anchor=None, anchor_file="events.csv@t=09:00,18:00,20:00")
        from_file = run_hook_script(
            add_hook,
            file_task,
            env_extra={"NO_COLOR": "1", "NAUTICAL_CONFIG": str(config)},
        )
        file_old = dict(
            file_task,
            anchor_file="events.csv@t=09:00",
            chain="on",
            chainID="cid_until_file_edit",
            link=1,
        )
        file_new = dict(file_old, anchor_file="events.csv@t=09:00,20:00")
        modified_file = run_hook_script_raw(
            modify_hook,
            json.dumps(file_old) + "\n" + json.dumps(file_new),
            env_extra={"NO_COLOR": "1", "NAUTICAL_CONFIG": str(config), "TASKDATA": td},
        )
    expect(from_file.returncode != 0, "anchor_file accepted a same-day expiration before a file slot")
    expect(not from_file.stdout.strip(), f"rejected anchor_file add leaked stdout: {from_file.stdout!r}")
    expect("20:00" in strip_markup(from_file.stderr), f"missing anchor_file slot evidence: {from_file.stderr!r}")
    expect(modified_file.returncode != 0, "on-modify accepted an anchor_file edit adding a slot after expiration")
    expect(not modified_file.stdout.strip(), f"rejected anchor_file edit leaked stdout: {modified_file.stdout!r}")
    expect("20:00" in strip_markup(modified_file.stderr), f"missing edited anchor_file slot: {modified_file.stderr!r}")

    invalid_parent = {
        "uuid": "00000000-0000-4000-8000-000000000998",
        "status": "completed",
        "anchor": "w:mon@t=09:00,18:00,20:00",
        "anchor_mode": "skip",
        "due": mod.core.fmt_isoz(due),
        "until": mod.core.fmt_isoz(until_1900),
        "chainID": "cid_until_slots",
    }
    try:
        build_child_draft_for_test(mod,
            invalid_parent,
            mod.core.build_local_datetime(anchor_day, (20, 0)),
            "due",
            2,
            "beef",
            "anchor",
            0,
            None,
        )
    except ValueError as exc:
        expect("until" in str(exc), f"unexpected child guard error: {exc!r}")
    else:
        raise AssertionError("child builder accepted an expiration at or before the next anchor slot")

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
    test_on_modify_carry_wall_clock_across_dst,
    test_on_modify_build_child_carries_until_across_dst,
    test_on_modify_native_until_calendar_and_exact_carry_policy,
    test_on_modify_native_until_exact_carry_preserves_elapsed_time_across_dst,
    test_native_until_calendar_slot_guard_rejects_impossible_anchor_expirations,
)
