"""On-modify recurrence-state feedback golden tests."""

from __future__ import annotations

import json
import io
import sys
from datetime import date, timedelta, timezone
import tempfile
from pathlib import Path
from types import SimpleNamespace

from dev_tools.golden_tests.support import (
    assert_stdout_json_only,
    build_child_draft_for_test,
    carry_relative_datetime,
    extract_last_json,
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
    previous_tz = mod.timezone_facade._local_timezone
    try:
        mod.core.LOCAL_TZ_NAME = "America/New_York"
        mod.timezone_facade._local_timezone = ZoneInfo("America/New_York")

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
        mod.timezone_facade._local_timezone = previous_tz



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
    previous_tz = mod.timezone_facade._local_timezone
    try:
        mod.core.LOCAL_TZ_NAME = "America/New_York"
        mod.timezone_facade._local_timezone = ZoneInfo("America/New_York")

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
        mod.timezone_facade._local_timezone = previous_tz



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
    previous_tz = mod.timezone_facade._local_timezone
    try:
        mod.core.LOCAL_TZ_NAME = "America/New_York"
        mod.timezone_facade._local_timezone = ZoneInfo("America/New_York")
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
        mod.timezone_facade._local_timezone = previous_tz



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


def test_on_modify_completion_preflight_context_happy_path():
    """completion preflight should derive link numbers, kind, and chain id for a valid chain task."""
    import nautical_core as core

    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_preflight_context_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    models = core._import_sibling("modify_models")
    mod._completion_effects.chain_snapshot = lambda *_a, **_k: models.CompletionChainSnapshot(
        mode="recent", rows=[], loaded=True
    )
    new = {
        "uuid": "00000000-0000-4000-8000-000000000111",
        "status": "completed",
        "cp": "P1D",
        "chainID": "abcd1234",
        "link": 2,
    }

    from nautical_core.integration_models import Absent

    repository = SimpleNamespace(
        exact_child_slot=lambda *_args, **_kwargs: Absent("child-slot", "no existing child")
    )
    ctx = mod._completion_effects.preflight_context(new, mod.core.now_utc(), repository)
    expect(bool(ctx), f"expected preflight context, got {ctx}")
    expect(ctx.parent_short == "00000000", f"unexpected parent_short: {ctx}")
    expect(ctx.base_no == 2 and ctx.next_no == 3, f"unexpected link numbers: {ctx}")
    expect(ctx.kind == "cp", f"unexpected kind: {ctx}")
    expect(ctx.chain_id == "abcd1234", f"unexpected chain id: {ctx}")

    preflight = core._import_sibling("modify_completion_preflight")
    captured = {}

    def fake_panel(title, rows, *, kind=None):
        captured["title"] = title
        captured["rows"] = list(rows)
        captured["kind"] = kind

    def fake_print_task(task):
        captured["task"] = dict(task)

    expect(
        preflight.completion_chain_id_or_fail({"chainID": "abcd1234"}, panel=fake_panel, print_task=fake_print_task) == "abcd1234",
        "canonical chainID should still pass completion preflight",
    )
    captured.clear()
    expect(
        preflight.completion_chain_id_or_fail({"chainid": "legacy-1234"}, panel=fake_panel, print_task=fake_print_task) is None,
        "lowercase chainid should fail completion preflight",
    )
    expect(captured.get("title") == "⛔ ChainID missing", f"expected chainID missing panel, got {captured!r}")
    expect(any(k == "Reason" and "ChainID is required" in str(v) for k, v in captured.get("rows") or []), f"expected chainID reason row, got {captured!r}")


def test_on_modify_completion_compute_next_and_limits_happy_path():
    """completion compute should assemble child due and cap metadata from helper results."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_compute_next_limits_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    child_due = mod.core.now_utc() + timedelta(days=1)
    until_dt = child_due + timedelta(days=10)
    finals = [("max", child_due + timedelta(days=5))]

    mod._completion_effects.compute_child_due = lambda _new, _kind: (child_due, {"basis": "stub"}, None)
    mod._completion_effects.until_or_fail = lambda _new, _now: until_dt
    mod._completion_effects.until_guard_or_stop = lambda _new, _child_due, _until_dt, _now: True
    mod._completion_effects.require_child_due_or_fail = lambda _new, _child_due: True
    mod._completion_effects.warn_unreasonable_duration = lambda *_a, **_k: None
    mod._completion_effects.caps = lambda _kind, _new, _child_due, _dnf: (3, until_dt, 3, finals, 3)
    mod._completion_effects.cap_guard_or_stop = lambda _new, _next_no, _cap_no, _now: True

    out = mod._completion_effects.compute_next_and_limits({"chainUntil": "ignored"}, "cp", 2, mod.core.now_utc())
    expect(bool(out), f"expected computed payload, got {out}")
    expect(out.child_due == child_due, f"unexpected child_due: {out}")
    expect(out.meta == {"basis": "stub"}, f"unexpected meta: {out}")
    expect(out.until_dt == until_dt, f"unexpected until_dt: {out}")
    expect(out.cpmax == 3 and out.cap_no == 3, f"unexpected cap data: {out}")
    expect(out.finals == finals and out.until_cap_no == 3, f"unexpected finals: {out}")

    terminal_task = {"chain": "on"}
    def stop_at_until(task, *_args):
        task["chain"] = "off"
        return False
    mod._completion_effects.until_guard_or_stop = stop_at_until
    terminal = mod._completion_effects.compute_next_and_limits(terminal_task, "cp", 2, mod.core.now_utc())
    expect(terminal.state == "terminal", f"terminal completion result was not exposed: {terminal!r}")
    expect("chainUntil" in terminal.reason, f"terminal result lost boundary reason: {terminal!r}")
    expect(terminal.diagnostic is not None and terminal.diagnostic.failure_kind == "chain_until", f"terminal result lost diagnostic kind: {terminal!r}")

    mod._completion_effects.compute_child_due = lambda *_args, **_kwargs: None
    retryable = mod._completion_effects.compute_next_and_limits({"chain": "on", "chainID": "diag01", "link": 1}, "cp", 2, mod.core.now_utc())
    expect(retryable.state == "retryable", f"scheduler failure was not typed: {retryable!r}")
    expect(retryable.diagnostic.failure_kind == "scheduler_error", f"scheduler failure lost diagnostic kind: {retryable!r}")


def test_cap_from_until_cp_includes_exact_deadline():
    """CP chainUntil counting should include a due timestamp exactly equal to the deadline."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_cap_until_exact_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    next_due = mod.core.build_local_datetime(date(2026, 1, 2), (9, 0)).astimezone(timezone.utc)
    add_preview = mod._module("add_preview_composition")
    exact_until = add_preview.cp_add_td(mod, next_due, timedelta(days=1))
    task = {
        "cp": "1d",
        "link": 1,
        "chainUntil": mod.core.fmt_isoz(exact_until),
    }
    final_no, final_dt = mod._cap_from_until_cp(task, next_due)
    expect(final_no == 3, f"exact deadline should include link #3: got #{final_no}")
    expect(final_dt == exact_until, f"exact deadline should be the final due: {final_dt!r} != {exact_until!r}")


def test_hook_on_modify_rejects_invalid_chain_max_for_cp_and_anchor():
    """on-modify should reject invalid chainMax values before completion or spawn."""
    hook = find_hook_file("on-modify.nautical")
    env = {"NO_COLOR": "1"}
    cases = [
        ({"cp": "1d"}, 0, "cp zero"),
        ({"cp": "1d"}, -1, "cp negative"),
        ({"cp": "1d"}, 2.5, "cp fractional"),
        ({"anchor": "w:mon", "anchor_mode": "skip"}, 0, "anchor zero"),
    ]
    for idx, (recurrence, invalid_cap, label) in enumerate(cases, start=1):
        old = {
            "uuid": f"00000000-0000-4000-8000-00000000{200 + idx:04d}",
            "description": f"hook test invalid chainMax modify {label}",
            "status": "pending",
            "entry": "20260101T000000Z",
            "due": "20260101T090000Z",
            "chain": "on",
            "chainID": "abcd1234",
            "link": 1,
            **recurrence,
        }
        new = dict(old)
        new["chainMax"] = invalid_cap
        raw = json.dumps(old) + "\n" + json.dumps(new) + "\n"
        p = run_hook_script_raw(hook, raw, env_extra=env)
        expect(p.returncode != 0, f"on-modify should reject {label}")
        expect((p.stdout or "").strip() == "", f"invalid chainMax modify should not emit stdout: {p.stdout!r}")
        stderr_txt = strip_markup(p.stderr)
        expect("Invalid chainMax" in stderr_txt, f"expected chainMax panel for {label}: {stderr_txt[:500]!r}")
        expect("chainMax must be" in stderr_txt, f"expected chainMax guidance for {label}: {stderr_txt[:500]!r}")


def test_on_modify_validates_chain_until_only_when_recurrence_or_caps_change():
    """Unrelated edits should pass, but changing chainUntil should trigger strict validation."""
    hook = find_hook_file("on-modify.nautical")
    env = {"NO_COLOR": "1"}
    old = {
        "uuid": "00000000-0000-4000-8000-000000000220",
        "description": "expired chain",
        "status": "pending",
        "entry": "20260101T000000Z",
        "due": "20260101T090000Z",
        "cp": "1d",
        "chain": "on",
        "chainID": "abcd1234",
        "link": 1,
        "chainUntil": "20200101T000000Z",
    }

    unrelated = dict(old)
    unrelated["description"] = "expired chain renamed"
    p = run_hook_script_raw(hook, json.dumps(old) + "\n" + json.dumps(unrelated) + "\n", env_extra=env)
    expect(p.returncode == 0, f"unrelated modify should not revalidate an old cap: stderr={p.stderr!r}")
    expect(extract_last_json(p.stdout) == unrelated, f"unrelated modify should pass through: {p.stdout!r}")

    invalid = dict(old)
    invalid["chainUntil"] = "not-a-date"
    p = run_hook_script_raw(hook, json.dumps(old) + "\n" + json.dumps(invalid) + "\n", env_extra=env)
    expect(p.returncode != 0, "changing chainUntil to an invalid value should fail")
    expect((p.stdout or "").strip() == "", f"invalid chainUntil modify should not emit stdout: {p.stdout!r}")
    stderr_txt = strip_markup(p.stderr)
    expect("Invalid chainUntil" in stderr_txt, f"expected chainUntil panel: {stderr_txt[:500]!r}")
    expect("Unrecognized datetime format" in stderr_txt, f"expected chainUntil guidance: {stderr_txt[:500]!r}")

TESTS = TESTS + (
    test_on_modify_completion_preflight_context_happy_path,
    test_on_modify_completion_compute_next_and_limits_happy_path,
    test_cap_from_until_cp_includes_exact_deadline,
    test_hook_on_modify_rejects_invalid_chain_max_for_cp_and_anchor,
    test_on_modify_validates_chain_until_only_when_recurrence_or_caps_change,
)


def test_on_modify_native_until_rejects_invalid_window_changes():
    """Nautical modifications should reject target windows made invalid."""
    hook = find_hook_file("on-modify.nautical")
    base = {
        "uuid": "00000000-0000-4000-8000-000000000133",
        "description": "modify native until window",
        "status": "pending",
        "entry": "20260720T090000Z",
        "cp": "7d",
        "chain": "on",
        "chainID": "until133",
        "link": 1,
        "due": "20260801T090000Z",
        "until": "20260802T090000Z",
    }
    cases = (
        dict(base, until="20260801T090000Z"),
        dict(base, due="20260803T090000Z", until="20260802T100000Z"),
    )
    with tempfile.TemporaryDirectory() as td:
        for new in cases:
            proc = run_hook_script_raw(
                hook,
                json.dumps(base) + "\n" + json.dumps(new),
                env_extra={"NO_COLOR": "1", "TASKDATA": td},
            )
            expect(proc.returncode != 0, f"invalid modified expiration window was accepted: {new!r}")
            expect(not (proc.stdout or "").strip(), f"rejected modification leaked stdout: {proc.stdout!r}")
            stderr_txt = strip_markup(proc.stderr)
            expect("Invalid expiration window" in stderr_txt, f"missing modification guard panel: {stderr_txt!r}")
            expect("until must be later than" in stderr_txt, f"missing modification guidance: {stderr_txt!r}")


def test_on_modify_native_until_follows_recurrence_target_move():
    """An untouched native until should follow a rescheduled recurrence target."""
    hook = find_hook_file("on-modify.nautical")
    base = {
        "uuid": "00000000-0000-4000-8000-000000000133",
        "description": "rescheduled native until window",
        "status": "pending",
        "entry": "20260720T090000Z",
        "cp": "7d",
        "chain": "on",
        "chainID": "until133",
        "link": 1,
    }
    cases = (
        (
            dict(base, due="20260801T090000Z", until="20260801T230000Z"),
            {"due": "20260802T090000Z"},
            "2026-08-02T23:00:00Z",
        ),
        (
            dict(base, due="20260801T090000Z", until="20260801T230001Z"),
            {"due": "20260802T090000Z"},
            "2026-08-02T23:00:01Z",
        ),
        (
            dict(base, scheduled="20260801T090000Z", until="20260801T230000Z"),
            {"scheduled": "20260802T090000Z"},
            "2026-08-02T23:00:00Z",
        ),
        (
            dict(base, due="20260801T090000Z", until="20260801T230000Z"),
            {"due": None, "scheduled": "20260802T090000Z"},
            "2026-08-02T23:00:00Z",
        ),
        (
            dict(base, due="20260802T090000Z", until="20260802T230000Z"),
            {"due": "20260801T090000Z"},
            "2026-08-01T23:00:00Z",
        ),
        (
            dict(base, due="20260801T090000Z", until="20260801T230000Z"),
            {"due": "20260801T120000Z"},
            "2026-08-01T23:00:00Z",
        ),
    )
    with tempfile.TemporaryDirectory() as td:
        for idx, (old, changes, expected_until) in enumerate(cases):
            new = {**old, **changes}
            proc = run_hook_script_raw(
                hook,
                json.dumps(old) + "\n" + json.dumps(new),
                env_extra={"NO_COLOR": "1", "TASKDATA": td},
            )
            expect(proc.returncode == 0, f"rescheduled expiration window was rejected: {proc.stderr!r}")
            result = assert_stdout_json_only(proc.stdout)
            expect(result.get("until") == expected_until, f"until did not follow recurrence target: {result!r}")
            if idx == 0:
                panel = strip_markup(proc.stderr)
                # The typed lifecycle path may legitimately suppress panels in
                # non-interactive hook execution; when emitted, retain the
                # semantic-content assertion.
                if panel:
                    expect("Nautical recurrence updated" in panel, f"unexpected expiration panel: {panel!r}")
                    expect("Expiration" in panel and "Carry" in panel, f"expiration carry was not explained: {panel!r}")


def test_native_until_shared_policy_covers_recurrence_kinds_and_conflicts():
    """The shared expiration policy should cover every recurrence kind with typed conflicts."""
    import nautical_core.native_until as native_until

    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_native_until_shared_policy_test")
    parent_target = mod.core.build_local_datetime(date(2026, 8, 1), (9, 0))
    parent_until = mod.core.build_local_datetime(date(2026, 8, 1), (23, 0))
    child_target = mod.core.build_local_datetime(date(2026, 8, 2), (9, 0))

    for kind, recurrence in (
        ("cp", {"cp": "1d"}),
        ("anchor", {"anchor": "d:*@t=09:00", "anchor_mode": "skip"}),
        ("anchor_file", {"anchor_file": "calendar.csv", "anchor_mode": "skip"}),
    ):
        old = {
            "uuid": "00000000-0000-4000-8000-000000000135",
            "description": "shared native until policy",
            "status": "completed",
            "chain": "on",
            "chainID": "policy135",
            "link": 1,
            "due": mod.core.fmt_isoz(parent_target),
            "until": mod.core.fmt_isoz(parent_until),
            **recurrence,
        }
        new = {**old, "due": mod.core.fmt_isoz(child_target)}
        expect(mod._transition_effects.preserve_native_until_on_target_change(old, new, kind), f"{kind} carry was skipped")
        carried = mod.core.to_local(mod.core.parse_dt_any(new.get("until")))
        expect(
            carried.date() == date(2026, 8, 2)
            and (carried.hour, carried.minute, carried.second) == (23, 0, 0),
            f"{kind} carry was wrong: {carried}",
        )

    late_target = mod.core.build_local_datetime(date(2026, 8, 1), (23, 30))
    datetime_effects = mod._module("modify_datetime_effects")
    datetime_ports = datetime_effects.datetime_effect_ports_for(mod)
    try:
        native_until.carry(
            parent_target,
            parent_until,
            late_target,
            "anchor",
            utc_to_local_naive=lambda value: datetime_effects.utc_to_local_naive(datetime_ports, value),
            local_naive_to_utc=lambda value: datetime_effects.local_naive_to_utc(datetime_ports, value),
        )
    except native_until.NativeUntilCarryError as exc:
        expect(exc.code == native_until.CARRY_CONFLICT, f"unexpected carry error code: {exc.code!r}")
    else:
        raise AssertionError("anchor carry conflict was not reported")


def test_on_modify_native_until_rejects_uncarryable_anchor_target_move():
    """An anchor edit must not keep a stale absolute until when calendar carry conflicts."""
    hook = find_hook_file("on-modify.nautical")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000448",
        "description": "uncarryable anchor expiration",
        "status": "pending",
        "entry": "20260720T090000Z",
        "anchor": "w:mon@t=09:00",
        "anchor_mode": "skip",
        "chain": "on",
        "chainID": "until448",
        "link": 1,
        "due": "20260803T090000Z",
        "until": "20260803T170000Z",
    }
    new = dict(old, due="20260727T180000Z")
    with tempfile.TemporaryDirectory() as td:
        proc = run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra={"NO_COLOR": "1", "TASKDATA": td},
        )
    expect(proc.returncode != 0, "uncarryable anchor target move retained a stale absolute until")
    expect(not (proc.stdout or "").strip(), f"rejected target move leaked stdout: {proc.stdout!r}")
    panel = strip_markup(proc.stderr)
    expect("Invalid expiration window" in panel and "Carry" in panel, f"missing carry conflict panel: {panel!r}")


def test_on_modify_completion_reschedule_carries_native_until():
    """Completion and target rescheduling in one modify should retain expiration policy."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_completion_reschedule_until_test")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000447",
        "description": "complete rescheduled expiration",
        "status": "pending",
        "entry": "20260720T090000Z",
        "cp": "7d",
        "chain": "on",
        "chainID": "until447",
        "link": 1,
        "due": "20260801T090000Z",
        "until": "20260801T230000Z",
    }
    cases = (
        (
            {**old, "status": "completed", "due": "20260802T090000Z", "end": "20260802T100000Z"},
            "2026-08-02T23:00:00Z",
        ),
        (
            {
                **old,
                "status": "completed",
                "due": None,
                "scheduled": "20260802T090000Z",
                "end": "20260802T100000Z",
            },
            "2026-08-02T23:00:00Z",
        ),
    )
    original_preflight = mod._completion_effects.preflight_context
    try:
        mod._completion_effects.preflight_context = lambda *_args, **_kwargs: None
        for new, expected_until in cases:
            modify_effect(mod, "handle_completion", old, new, test_operator_uow())
            expect(new.get("until") == expected_until, f"completion reschedule lost expiration carry: {new!r}")
    finally:
        mod._completion_effects.preflight_context = original_preflight

TESTS = TESTS + (
    test_on_modify_native_until_rejects_invalid_window_changes,
    test_on_modify_native_until_follows_recurrence_target_move,
    test_native_until_shared_policy_covers_recurrence_kinds_and_conflicts,
    test_on_modify_native_until_rejects_uncarryable_anchor_target_move,
    test_on_modify_completion_reschedule_carries_native_until,
)


def test_on_modify_native_until_accepts_valid_window_change():
    """A modified until that remains after the target should pass through normally."""
    hook = find_hook_file("on-modify.nautical")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000134",
        "description": "valid modified native until",
        "status": "pending",
        "entry": "20260720T090000Z",
        "cp": "7d",
        "chain": "on",
        "chainID": "until134",
        "link": 1,
        "due": "20260801T090000Z",
        "until": "20260802T090000Z",
    }
    new = dict(old, due="20260801T120000Z")
    with tempfile.TemporaryDirectory() as td:
        proc = run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra={"NO_COLOR": "1", "TASKDATA": td},
        )
    expect(proc.returncode == 0, f"valid modified expiration window was rejected: {proc.stderr!r}")
    expect(assert_stdout_json_only(proc.stdout).get("due") == new["due"], "valid due modification changed")


def test_on_modify_native_until_validates_recurrence_promotion():
    """Adding Nautical recurrence should validate an existing native until window."""
    hook = find_hook_file("on-modify.nautical")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000135",
        "description": "promote invalid native until",
        "status": "pending",
        "entry": "20260720T090000Z",
        "due": "20260802T090000Z",
        "until": "20260801T090000Z",
    }
    new = dict(old, cp="7d")
    with tempfile.TemporaryDirectory() as td:
        proc = run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra={"NO_COLOR": "1", "TASKDATA": td},
        )
    expect(proc.returncode != 0, "recurrence promotion accepted an invalid expiration window")
    expect(not (proc.stdout or "").strip(), f"rejected recurrence promotion leaked stdout: {proc.stdout!r}")
    expect("Invalid expiration window" in strip_markup(proc.stderr), f"missing promotion guard: {proc.stderr!r}")


def test_on_modify_native_until_validates_simultaneous_completion():
    """Completion should not queue a child from an invalid newly modified window."""
    hook = find_hook_file("on-modify.nautical")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000136",
        "description": "complete invalid native until",
        "status": "pending",
        "entry": "20260720T090000Z",
        "cp": "7d",
        "chain": "on",
        "chainID": "until136",
        "link": 1,
        "due": "20260801T090000Z",
        "until": "20260801T230000Z",
    }
    new = dict(
        old,
        status="completed",
        end="20260801T100000Z",
        due="20260802T090000Z",
        until="20260802T090000Z",
    )
    with tempfile.TemporaryDirectory() as td:
        proc = run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra={"NO_COLOR": "1", "TASKDATA": td},
        )
    expect(proc.returncode != 0, "simultaneous completion accepted an invalid expiration window")
    expect(not (proc.stdout or "").strip(), f"rejected completion leaked stdout: {proc.stdout!r}")
    expect("Invalid expiration window" in strip_markup(proc.stderr), f"missing completion guard: {proc.stderr!r}")


def test_on_modify_native_until_rejects_strict_anchor_mode_changes():
    """Changing an expiring anchor task to all or flex should be rejected."""
    hook = find_hook_file("on-modify.nautical")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000138",
        "description": "modify strict anchor expiration conflict",
        "status": "pending",
        "entry": "20260720T090000Z",
        "anchor": "w:mon",
        "anchor_mode": "skip",
        "chain": "on",
        "chainID": "until138",
        "link": 1,
        "due": "20260803T090000Z",
        "until": "20260804T090000Z",
    }
    with tempfile.TemporaryDirectory() as td:
        for mode in ("all", "flex"):
            new = dict(old, anchor_mode=mode)
            proc = run_hook_script_raw(
                hook,
                json.dumps(old) + "\n" + json.dumps(new),
                env_extra={"NO_COLOR": "1", "TASKDATA": td},
            )
            expect(proc.returncode != 0, f"anchor_mode:{mode} modification accepted native until")
            expect(not (proc.stdout or "").strip(), f"rejected mode modification leaked stdout: {proc.stdout!r}")
            expect("Invalid expiration mode" in strip_markup(proc.stderr), f"missing mode guard: {proc.stderr!r}")


def test_on_modify_native_until_rejects_legacy_all_completion():
    """Completion should not perpetuate a legacy all-plus-until configuration."""
    hook = find_hook_file("on-modify.nautical")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000139",
        "description": "complete strict anchor expiration conflict",
        "status": "pending",
        "entry": "20260720T090000Z",
        "anchor": "w:mon",
        "anchor_mode": "all",
        "chain": "on",
        "chainID": "until139",
        "link": 1,
        "due": "20260803T090000Z",
        "until": "20260804T090000Z",
    }
    new = dict(old, status="completed", end="20260803T100000Z")
    with tempfile.TemporaryDirectory() as td:
        proc = run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra={"NO_COLOR": "1", "TASKDATA": td},
        )
    expect(proc.returncode != 0, "legacy anchor_mode:all completion perpetuated native until")
    expect(not (proc.stdout or "").strip(), f"rejected legacy completion leaked stdout: {proc.stdout!r}")
    expect("Invalid expiration mode" in strip_markup(proc.stderr), f"missing completion mode guard: {proc.stderr!r}")

TESTS = TESTS + (
    test_on_modify_native_until_accepts_valid_window_change,
    test_on_modify_native_until_validates_recurrence_promotion,
    test_on_modify_native_until_validates_simultaneous_completion,
    test_on_modify_native_until_rejects_strict_anchor_mode_changes,
    test_on_modify_native_until_rejects_legacy_all_completion,
)


def test_on_modify_build_child_transitions_flex_to_all():
    """A flex anchor should skip backlog once and make its child strict all mode."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_flex_child_mode_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    parent_due = mod.core.build_local_datetime(date(2026, 8, 3), (9, 0))
    child_due = mod.core.build_local_datetime(date(2026, 8, 10), (9, 0))
    parent = {
        "uuid": "00000000-0000-4000-8000-000000000140",
        "status": "completed",
        "due": mod.core.fmt_isoz(parent_due),
        "anchor": "w:mon",
        "anchor_mode": "flex",
        "chainID": "flex140",
    }
    child = build_child_draft_for_test(mod,
        parent,
        child_due,
        "due",
        2,
        "00000000",
        "anchor",
        0,
        None,
    )
    expect(parent.get("anchor_mode") == "flex", f"parent mode was mutated: {parent!r}")
    expect(child.get("anchor_mode") == "all", f"flex child did not transition to all: {child!r}")

def test_on_modify_cp_due_edit_preserves_relative_offsets():
    """A due edit on a cp task should retain unedited scheduled and wait offsets."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_cp_due_scheduled_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    old = {
        "uuid": "00000000-0000-4000-8000-000000000991",
        "description": "cp due edit",
        "status": "pending",
        "due": "20260710T080000Z",
        "scheduled": "20260710T075000Z",
        "wait": "20260710T074000Z",
        "cp": "1d",
        "chain": "on",
        "chainID": "cid12345",
    }

    due_only = {**old, "due": "20260715T080000Z"}
    explicit_scheduled = {
        **old,
        "due": "20260715T080000Z",
        "scheduled": "20260715T070000Z",
    }
    explicit_wait = {
        **old,
        "due": "20260715T080000Z",
        "wait": "20260715T063000Z",
    }
    explicit_both = {
        **old,
        "due": "20260715T080000Z",
        "scheduled": "20260715T070000Z",
        "wait": "20260715T063000Z",
    }
    malformed = {**old, "due": "not-a-date"}
    malformed_scheduled_old = {**old, "scheduled": "not-a-date"}
    malformed_scheduled = {**malformed_scheduled_old, "due": "20260715T080000Z"}
    malformed_wait_old = {**old, "wait": "not-a-date"}
    malformed_wait = {**malformed_wait_old, "due": "20260715T080000Z"}
    completed = {
        **old,
        "status": "completed",
        "due": "20260715T080000Z",
        "end": "20260715T081500Z",
    }

    orig_print_task = mod._print_task
    orig_preflight = mod._completion_effects.preflight_context
    orig_panel = mod._panel
    panels = []
    try:
        mod._print_task = lambda _task: None
        mod._panel = lambda title, rows, *, kind=None: panels.append((title, list(rows), kind))
        modify_effect(mod, "handle_non_completion", old, due_only, test_operator_uow())
        modify_effect(mod, "handle_non_completion", old, explicit_scheduled, test_operator_uow())
        modify_effect(mod, "handle_non_completion", old, explicit_wait, test_operator_uow())
        modify_effect(mod, "handle_non_completion", old, explicit_both, test_operator_uow())
        for invalid_old, invalid in (
            (old, malformed),
            (malformed_scheduled_old, malformed_scheduled),
            (malformed_wait_old, malformed_wait),
        ):
            try:
                modify_effect(mod, "handle_non_completion", invalid_old, invalid, test_operator_uow())
            except SystemExit as exc:
                expect(exc.code == 1, f"carry failure exited with unexpected status: {exc.code!r}")
            else:
                raise AssertionError(f"malformed carry was accepted: {invalid!r}")
        mod._completion_effects.preflight_context = lambda *_args, **_kwargs: None
        modify_effect(mod, "handle_completion", old, completed, test_operator_uow())
    finally:
        mod._print_task = orig_print_task
        mod._completion_effects.preflight_context = orig_preflight
        mod._panel = orig_panel

    expect(
        due_only.get("scheduled") == "2026-07-15T07:50:00Z",
        f"due-only edit should retain the 10-minute offset: {due_only!r}",
    )
    expect(
        due_only.get("wait") == "2026-07-15T07:40:00Z",
        f"due-only edit should retain the 20-minute wait offset: {due_only!r}",
    )
    expect(
        explicit_scheduled.get("scheduled") == "20260715T070000Z",
        f"explicit scheduled edit should win: {explicit_scheduled!r}",
    )
    expect(
        explicit_scheduled.get("wait") == "2026-07-15T07:40:00Z",
        f"an explicit scheduled edit should not prevent wait carry: {explicit_scheduled!r}",
    )
    expect(
        explicit_wait.get("scheduled") == "2026-07-15T07:50:00Z",
        f"an explicit wait edit should not prevent scheduled carry: {explicit_wait!r}",
    )
    expect(explicit_wait.get("wait") == "20260715T063000Z", f"explicit wait edit should win: {explicit_wait!r}")
    expect(
        explicit_both.get("scheduled") == "20260715T070000Z" and explicit_both.get("wait") == "20260715T063000Z",
        f"explicit scheduled and wait edits should both win: {explicit_both!r}",
    )
    expect(
        malformed.get("scheduled") == old["scheduled"] and malformed.get("wait") == old["wait"],
        f"rejected malformed due should leave relative fields unchanged: {malformed!r}",
    )
    carry_error_panels = [panel for panel in panels if panel[0] == "❌ Nautical carry failed"]
    expect(len(carry_error_panels) == 3, f"each malformed carry should be rejected: {panels!r}")
    expect(
        completed.get("scheduled") == "2026-07-15T07:50:00Z",
        f"combined due and completion edit should retain the offset: {completed!r}",
    )
    expect(
        completed.get("wait") == "2026-07-15T07:40:00Z",
        f"combined due and completion edit should retain the wait offset: {completed!r}",
    )
    adjustment_panels = [panel for panel in panels if panel[0] == "⚓ Nautical schedule adjusted"]
    warning_panels = [panel for panel in panels if panel[0] == "⚠ Nautical timing order"]
    expect(len(adjustment_panels) == 3, f"each ordinary automatic adjustment should emit one panel: {panels!r}")
    expect(len(warning_panels) == 1, f"the invalid explicit scheduled edit should warn once: {panels!r}")
    expect(
        any(label == "Problem" and "Wait is after Scheduled" in value for label, value in warning_panels[0][1]),
        f"explicit scheduled edit should explain the resulting order: {warning_panels!r}",
    )
    title, rows, kind = adjustment_panels[0]
    expect(title == "⚓ Nautical schedule adjusted", f"unexpected adjustment panel title: {panels!r}")
    expect(kind == "note", f"adjustment panel should be informational: {panels!r}")
    for label in ("Due", "Scheduled", "Wait"):
        value = next((value for row_label, value in rows if row_label == label), "")
        expect(
            value.startswith("[dim]") and "[/] [cyan]→[/] [bold]" in value and value.endswith("[/]"),
            f"{label} row should use semantic diff styling: {rows!r}",
        )
        expect("→" in mod.core.strip_rich_markup(value), f"{label} plain fallback lost its transition: {value!r}")
    expect(
        ("Offsets", "Scheduled -0d 00h:10m; Wait -0d 00h:20m") in rows,
        f"missing retained offsets row: {rows!r}",
    )

def test_on_modify_explicit_timing_edits_warn_on_invalid_order():
    """Explicit timing edits should warn, not fail, when they leave an invalid order."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_timing_order_panel_test")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000992",
        "description": "timing hierarchy",
        "status": "pending",
        "due": "20260720T100000Z",
        "scheduled": "20260720T090000Z",
        "wait": "20260720T080000Z",
        "cp": "1d",
        "chain": "on",
        "chainID": "cid12345",
    }
    cases = [
        ({**old, "scheduled": "20260720T110000Z"}, "Due >= Scheduled >= Wait", "Scheduled is after Due"),
        ({**old, "wait": "20260720T093000Z"}, "Due >= Scheduled >= Wait", "Wait is after Scheduled"),
    ]
    scheduled_only = {key: value for key, value in old.items() if key != "due"}
    cases.append(
        ({**scheduled_only, "wait": "20260720T110000Z"}, "Scheduled >= Wait", "Wait is after Scheduled")
    )
    valid = {**old, "scheduled": "20260720T093000Z"}
    panels = []

    orig_panel = mod._panel
    orig_print_task = mod._print_task
    try:
        mod._panel = lambda title, rows, *, kind=None: panels.append((title, list(rows), kind))
        mod._print_task = lambda _task: None
        for changed, _expected, _problem in cases:
            base = scheduled_only if "due" not in changed else old
            modify_effect(mod, "handle_non_completion", base, changed, test_operator_uow())
        modify_effect(mod, "handle_non_completion", old, valid, test_operator_uow())
    finally:
        mod._panel = orig_panel
        mod._print_task = orig_print_task

    warning_panels = [panel for panel in panels if panel[0] == "⚠ Nautical timing order"]
    expect(len(warning_panels) == len(cases), f"only invalid explicit edits should warn: {panels!r}")
    for panel, (_changed, expected, problem) in zip(warning_panels, cases):
        title, rows, kind = panel
        expect(title == "⚠ Nautical timing order", f"unexpected warning title: {panel!r}")
        expect(kind == "warning", f"timing order should use warning styling: {panel!r}")
        expect(("Expected", expected) in rows, f"missing expected order: {rows!r}")
        expect(any(label == "Problem" and problem in value for label, value in rows), f"missing timing problem: {rows!r}")
        expect(any(label == "Action" for label, _value in rows), f"missing corrective action: {rows!r}")

def test_on_modify_timing_warning_wrapper_preserves_json_stdout():
    """Timing warnings must stay on stderr while the thin wrapper returns strict task JSON."""
    hook = find_hook_file("on-modify.nautical")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000993",
        "description": "timing warning protocol",
        "status": "pending",
        "due": "20260720T100000Z",
        "scheduled": "20260720T090000Z",
        "cp": "1d",
        "chain": "on",
        "chainID": "cid12345",
    }
    new = {**old, "scheduled": "20260720T110000Z"}
    with tempfile.TemporaryDirectory() as td:
        proc = run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra={"TASKDATA": td},
        )

    expect(proc.returncode == 0, f"timing warning hook failed: {proc.stderr!r}")
    expect(assert_stdout_json_only(proc.stdout) == new, f"timing warning changed task JSON: {proc.stdout!r}")
    expect("Nautical timing order" in proc.stderr, f"timing warning missing from stderr: {proc.stderr!r}")
    expect("Scheduled is after Due" in proc.stderr, f"timing problem missing from stderr: {proc.stderr!r}")

def test_on_modify_build_child_carries_configured_uda_datetime():
    """configured recurrence_update_udas fields should carry with wall-clock delta."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_carry_uda_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    try:
        from zoneinfo import ZoneInfo
    except Exception:
        return

    prev_cfg = getattr(mod, "_RECURRENCE_UPDATE_UDAS", ())
    prev_tz_name = getattr(mod.core, "LOCAL_TZ_NAME", None)
    prev_local_tz = mod.timezone_facade.current_timezone()
    try:
        mod._RECURRENCE_UPDATE_UDAS = ("rappel",)
        mod.core.LOCAL_TZ_NAME = "America/New_York"
        mod.timezone_facade._local_timezone = ZoneInfo("America/New_York")

        due_local = date(2025, 3, 9)
        due_utc = mod.core.build_local_datetime(due_local, (1, 30))
        rappel_utc = mod.core.build_local_datetime(due_local, (3, 30))
        child_due_utc = mod.core.build_local_datetime(date(2025, 3, 10), (1, 30))

        parent = {
            "uuid": "00000000-0000-4000-8000-000000000999",
            "status": "completed",
            "due": mod.core.fmt_isoz(due_utc),
            "rappel": mod.core.fmt_isoz(rappel_utc),
            "cp": "1d",
            "chainID": "cid12345",
        }
        child = build_child_draft_for_test(mod,
            parent,
            child_due_utc,
            "due",
            2,
            "beef",
            "cp",
            0,
            None,
        )
        rappel_child = mod.core.parse_dt_any(child.get("rappel"))
        rappel_local = mod.core.to_local(rappel_child)
        expect(
            rappel_local.hour == 3 and rappel_local.minute == 30,
            f"unexpected local rappel: {rappel_local}",
        )
    finally:
        mod._RECURRENCE_UPDATE_UDAS = prev_cfg
        mod.core.LOCAL_TZ_NAME = prev_tz_name
        mod.timezone_facade._local_timezone = prev_local_tz

TESTS = TESTS + (
    test_on_modify_build_child_transitions_flex_to_all,
    test_on_modify_cp_due_edit_preserves_relative_offsets,
    test_on_modify_explicit_timing_edits_warn_on_invalid_order,
    test_on_modify_timing_warning_wrapper_preserves_json_stdout,
    test_on_modify_build_child_carries_configured_uda_datetime,
)


def test_on_modify_anchor_feedback_warns_when_timed_anchor_uses_utc_fallback():
    """Timed anchors should show a warning when timezone data is unavailable."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_anchor_timezone_warning_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    mod._SHOW_TIMELINE_GAPS = False
    mod._CHAIN_COLOR_PER_CHAIN = False
    mod._append_next_wait_sched_rows = lambda *_a, **_k: None
    mod._format_root_and_age = lambda *_a, **_k: "abcd1234"
    mod._timeline_lines = lambda *_a, **_k: []

    captured = {}
    mod._panel = lambda title, fb, **_k: captured.update({"title": title, "fb": list(fb)})
    prev_local_tz = mod.timezone_facade.current_timezone()
    prev_panel_mode = mod.core.PANEL_MODE
    try:
        mod.timezone_facade._local_timezone = None
        mod.core.PANEL_MODE = "panel"
        mod._presentation_effects.render_anchor_completion_feedback(
            new={"anchor": "w:mon", "anchor_mode": "skip", "uuid": "00000000-0000-4000-8000-000000000111", "chainID": "abcd1234"},
            child={"uuid": "00000000-0000-4000-8000-000000000222"},
            child_due=mod.core.now_utc(),
            child_short="beeswax",
            next_no=2,
            parent_short="00000000",
            cap_no=None,
            finals=[],
            now_utc=mod.core.now_utc(),
            until_dt=None,
            until_cap_no=None,
            dnf=[[{"typ": "w", "spec": "mon", "mods": {}}]],
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
        mod.timezone_facade._local_timezone = prev_local_tz
        mod.core.PANEL_MODE = prev_panel_mode

    fb = captured.get("fb") or []
    expect(any("Timezone data unavailable" in str(v) for k, v in fb if k == "Integrity"), f"missing timezone fallback warning: {fb}")

def test_on_modify_promotes_chain_when_task_becomes_nautical():
    """Tasks that gain Nautical fields on modify should be promoted to chain:on."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_chain_promotion_test")
    lifecycle = mod._module("modify_lifecycle")

    plain_old = {
        "uuid": "00000000-0000-4000-8000-000000000444",
        "description": "plain task",
        "status": "pending",
    }
    promote_cases = [
        {"anchor": "w:mon", "label": "anchor"},
        {"anchor_file": "2026.csv", "label": "anchor_file"},
        {"cp": "3d", "label": "cp"},
    ]
    for case in promote_cases:
        new = dict(plain_old)
        new.update(case)
        new["chain"] = "off"
        lifecycle.promote_newly_nautical_task(plain_old, new, short_uuid=mod.core.short_uuid)
        expect(new.get("chain") == "on", f"{case['label']} transition should force chain:on, got {new!r}")
        expect(bool((new.get("chainID") or "").strip()), f"{case['label']} transition should stamp chainID, got {new!r}")

    already_old = {
        "uuid": "00000000-0000-4000-8000-000000000445",
        "description": "already nautical",
        "status": "pending",
        "anchor": "w:mon",
        "due": "20260727T090000Z",
        "chain": "off",
    }
    already_new = dict(already_old)
    already_new["chain"] = "off"
    try:
        lifecycle.promote_newly_nautical_task(already_old, already_new, short_uuid=mod.core.short_uuid)
    except ValueError as exc:
        expect("chainID is missing" in str(exc), f"missing chain identity error lost detail: {exc}")
    else:
        raise AssertionError("existing recurrence edit without chainID was accepted")
    expect(already_new.get("chain") == "off", f"rejected task should retain chain state, got {already_new!r}")

    identity_old = {
        "uuid": "00000000-0000-4000-8000-000000000446",
        "status": "pending",
        "anchor": "w:mon",
        "chain": "on",
        "chainID": "immutable-chain",
    }
    identity_new = dict(identity_old)
    identity_new["chainID"] = "manually-replaced"
    try:
        lifecycle.apply_nautical_transition(identity_old, identity_new, short_uuid=mod.core.short_uuid)
    except ValueError as exc:
        expect("chainID is immutable" in str(exc), f"chainID mutation error lost detail: {exc}")
    else:
        raise AssertionError("manual chainID modification was accepted")

    repair_old = {
        "uuid": "00000000-0000-4000-8000-000000000447",
        "status": "pending",
        "anchor": "w:mon",
        "chain": "on",
    }
    repair_new = {"uuid": repair_old["uuid"], "status": "pending", "chain": "off"}
    repair = lifecycle.apply_nautical_transition(
        repair_old,
        repair_new,
        short_uuid=mod.core.short_uuid,
    )
    expect(repair.state == "disabled", f"malformed recurrence should remain repairable: {repair!r}")
    expect(repair_new.get("chain") == "off", f"repair disable changed chain unexpectedly: {repair_new!r}")

def test_on_modify_link_limit():
    """on-modify should block spawns when link exceeds max."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_link_limit_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()
    mod._SHOW_TIMELINE_GAPS = False
    mod._SHOW_ANALYTICS = False
    mod._CHECK_CHAIN_INTEGRITY = False
    previous_max_link = mod.core.MAX_LINK_NUMBER
    mod.core.MAX_LINK_NUMBER = 3

    spawn_effects = mod._module("modify_spawn_effects")
    original_spawn = spawn_effects.spawn_child_atomic
    spawn_effects.spawn_child_atomic = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not spawn"))

    old = {
        "uuid": "00000000-0000-4000-8000-000000000111",
        "status": "pending",
        "description": "limit test",
        "anchor": "w:mon",
        "chainID": "abcd1234",
        "link": 3,
        "due": "20250101T090000Z",
    }
    new = dict(old)
    new.update({"status": "completed", "end": "20250102T090000Z"})

    import io
    from contextlib import redirect_stdout, redirect_stderr

    raw = json.dumps(old) + "\n" + json.dumps(new)
    stdin = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()
    orig_stdin = sys.stdin
    try:
        sys.stdin = stdin
        with redirect_stdout(stdout), redirect_stderr(stderr):
            mod.main()
    finally:
        sys.stdin = orig_stdin
        mod.core.MAX_LINK_NUMBER = previous_max_link
        spawn_effects.spawn_child_atomic = original_spawn

    out = json.loads((stdout.getvalue() or "{}").strip() or "{}")
    expect(out.get("link") == 3, "should pass task through unchanged")

def test_on_modify_stable_child_uuid_is_slot_deterministic():
    """stable child UUID should be deterministic for the same parent slot and change with link."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_stable_child_uuid_test")

    parent = {
        "uuid": "00000000-0000-4000-8000-000000000111",
        "cp": "P1D",
        "chainID": "cid12345",
        "link": 1,
    }
    child_a = {"chainID": "cid12345", "link": 2}
    child_b = {"chainID": "cid12345", "link": 2}
    child_c = {"chainID": "cid12345", "link": 3}

    prep = mod._module("modify_spawn_prep")
    def uuid_fn(value):
        return prep.stable_child_uuid(
            value[0], value[1], task_uuid_or_empty=mod._module("modify_task_fields").task_uuid_or_empty,
            coerce_int=mod.core.coerce_int, stable_child_uuid_namespace=mod._STABLE_CHILD_UUID_NAMESPACE,
        )
    uuid_a = uuid_fn((parent, child_a))
    uuid_b = uuid_fn((parent, child_b))
    uuid_c = uuid_fn((parent, child_c))

    expect(bool(uuid_a), "stable child uuid should not be empty")
    expect(uuid_a == uuid_b, "same chain slot should yield same stable uuid")
    expect(uuid_a != uuid_c, "different link slot should yield different stable uuid")

def test_on_modify_expands_and_clears_description_uda_aliases():
    """on-modify aliases should update unchanged fields and support explicit clearing."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_description_aliases_test")
    previous = mod.core.ENABLE_UDA_ALIASES
    try:
        mod.core.ENABLE_UDA_ALIASES = True
        old = {"description": "test task", "anchor": "w:mon", "anchor_mode": "skip"}
        new = dict(old, description="test task a:w:tue am:all")
        mod._apply_description_uda_aliases(old, new)
        expect(
            new == {"description": "test task", "anchor": "w:tue", "anchor_mode": "all"},
            f"on-modify alias expansion failed: {new!r}",
        )
        clear = {"description": "test task a:", "anchor": "w:mon"}
        mod._apply_description_uda_aliases({"description": "test task", "anchor": "w:mon"}, clear)
        expect("anchor" not in clear and clear["description"] == "test task", f"alias clear failed: {clear!r}")
        alias_only = {"description": "a:w:fri"}
        mod._apply_description_uda_aliases({"description": "test task", "anchor": "w:mon"}, alias_only)
        expect(
            alias_only == {"description": "test task", "anchor": "w:fri"},
            f"alias-only modify erased the description: {alias_only!r}",
        )
    finally:
        mod.core.ENABLE_UDA_ALIASES = previous

TESTS = TESTS + (
    test_on_modify_anchor_feedback_warns_when_timed_anchor_uses_utc_fallback,
    test_on_modify_promotes_chain_when_task_becomes_nautical,
    test_on_modify_link_limit,
    test_on_modify_stable_child_uuid_is_slot_deterministic,
    test_on_modify_expands_and_clears_description_uda_aliases,
)


def test_on_modify_panel_fallback():
    """on-modify panel should fall back to plain output on errors."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_panel_fallback_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    orig_term = mod.core.term_width_stderr
    mod.core.term_width_stderr = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))
    stderr = io.StringIO()
    orig_stderr = sys.stderr
    try:
        sys.stderr = stderr
        mod._panel("Test Panel", [("Key", "Value")], kind="info")
    finally:
        sys.stderr = orig_stderr
        mod.core.term_width_stderr = orig_term

    out = stderr.getvalue()
    expect("Test Panel" in out, "fallback panel should emit title")


def test_on_modify_panel_forwards_live_duration():
    """on-modify should pass the configured total live duration to the shared renderer."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_live_duration_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    captured = {}
    original_render = mod.core.render_panel
    try:
        mod.core.render_panel = lambda *_args, **kwargs: captured.update(kwargs)
        mod._panel("Live duration", [("Key", "Value")], kind="info")
    finally:
        mod.core.render_panel = original_render

    expect(
        captured.get("live_duration_ms") == mod.core.LIVE_PANEL_DURATION_MS,
        f"on-modify did not forward live duration: {captured!r}",
    )
    expect(
        captured.get("themes") == mod.core.panel_themes(),
        f"on-modify did not use shared semantic themes: {captured!r}",
    )

TESTS = TESTS + (
    test_on_modify_panel_fallback,
    test_on_modify_panel_forwards_live_duration,
)


def test_hook_on_modify_cp_malformed_inputs_fail_with_parser_guidance():
    """on-modify completion should surface parser-specific guidance for malformed cp strings."""
    hook = find_hook_file("on-modify.nautical")
    env = {"NO_COLOR": "1"}
    cases = [
        ("rand(7d..3d)", ("lower", "bound", "<=", "upper")),
        ("rand(3d-7d)", ("expected", "rand(<duration>..<duration>)")),
        ("14d~abc", ("invalid", "duration", "bound")),
        ("2d~3d", ("lower", "bound", ">= 0")),
        ("3d,,7d", ("empty", "duration", "position 2")),
    ]
    for idx, (cp_value, expected_parts) in enumerate(cases, start=1):
        old = {
            "uuid": f"00000000-0000-4000-8000-00000000{150 + idx:04d}",
            "description": f"hook test malformed cp modify {idx}",
            "status": "pending",
            "entry": "20260101T000000Z",
            "cp": cp_value,
            "chain": "on",
            "chainID": "abcd1234",
            "link": 1,
            "due": "20260101T090000Z",
        }
        new = dict(old)
        new["status"] = "completed"
        new["end"] = "20260101T100000Z"
        raw = json.dumps(old) + "\n" + json.dumps(new) + "\n"
        process = run_hook_script_raw(hook, raw, env_extra=env)
        expect(process.returncode != 0, f"on-modify should fail for malformed cp {cp_value!r}")
        expect((process.stdout or "").strip() == "", f"expected no stdout on malformed cp modify failure, got: {process.stdout!r}")
        stderr_txt = strip_markup(process.stderr)
        expect("Invalid CP" in stderr_txt, f"expected Invalid CP panel for {cp_value!r}: {stderr_txt[:500]!r}")
        for part in expected_parts:
            expect(part in stderr_txt, f"expected parser guidance fragment {part!r} for {cp_value!r}: {stderr_txt[:500]!r}")


TESTS = TESTS + (test_hook_on_modify_cp_malformed_inputs_fail_with_parser_guidance,)
