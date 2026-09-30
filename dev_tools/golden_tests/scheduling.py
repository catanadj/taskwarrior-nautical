"""Golden contracts for temporal recurrence and hook scheduling behavior."""

from __future__ import annotations

from datetime import date, datetime
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

from dev_tools.golden_tests.support import (
    assert_stdout_json_only,
    call_with_supported_kwargs,
    compute_anchor_child_due,
    expect,
    find_hook_file,
    load_hook_module,
    run_hook_script,
    strip_markup,
)


def test_year_ordinals_hooks_modes_calendar_and_timeline():
    """Ordinal selectors work through add, completion modes, calendars, and timelines."""
    add_hook = find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config-nautical.toml"
        config_path.write_text(
            'tz = "UTC"\n'
            "[business_calendar.work]\n"
            'anchor = "w:mon..fri"\n'
            'omit = "y:12-31"\n',
            encoding="utf-8",
        )
        task = {
            "uuid": "00000000-0000-4000-8000-000000000781",
            "description": "ordinal calendar integration",
            "status": "pending",
            "entry": "20260101T000000Z",
            "anchor": "y:d-1@pbd@t=09:00",
            "anchor_mode": "skip",
            "bc": "WORK",
        }
        process = run_hook_script(
            add_hook,
            task,
            env_extra={"NO_COLOR": "1", "NAUTICAL_CONFIG": str(config_path)},
        )
        expect(
            process.returncode == 0,
            f"on-add rejected year-day calendar anchor: {process.stderr}",
        )
        out_task = assert_stdout_json_only(process.stdout)
        expect(
            out_task.get("anchor") == task["anchor"],
            f"on-add changed ordinal anchor: {out_task}",
        )
        expect(
            out_task.get("bc") == "work",
            f"on-add did not normalize calendar: {out_task}",
        )
        due = datetime.fromisoformat(str(out_task.get("due")))
        expect(
            due.date() == date(2026, 12, 30),
            f"calendar did not roll closed d-1 backward: {due}",
        )
        expect(
            "last day of each year" in strip_markup(process.stderr),
            f"add preview omitted ordinal natural text: {process.stderr}",
        )

    modify_hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(modify_hook, "_nautical_year_ordinal_hook_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    def stamp(day, hhmm):
        return mod.core.fmt_isoz(mod.core.build_local_datetime(day, hhmm))

    common = {
        "anchor": "y:w20@t=09:00",
        "due": stamp(date(2026, 5, 11), (9, 0)),
        "end": stamp(date(2026, 5, 14), (10, 0)),
        "chainID": "abcd1234",
    }
    all_due, all_meta, _all_dnf = compute_anchor_child_due(
        mod, dict(common, anchor_mode="all", scheduled=stamp(date(2026, 5, 11), (9, 0)))
    )
    skip_due, skip_meta, _skip_dnf = compute_anchor_child_due(
        mod, dict(common, anchor_mode="skip")
    )
    expect(
        mod.core.to_local(all_due).date() == date(2026, 5, 12),
        f"all mode did not backfill ISO week: {all_due}",
    )
    expect(all_meta.get("basis") == "missed", f"all mode metadata drifted: {all_meta}")
    expect(
        mod.core.to_local(skip_due).date() == date(2026, 5, 15),
        f"skip mode did not advance after completion: {skip_due}",
    )
    expect(
        skip_meta.get("basis") == "after_end",
        f"skip mode metadata drifted: {skip_meta}",
    )

    monday_expr = "y:w20 + w:mon@t=09:00"
    child_due, _meta, child_dnf = compute_anchor_child_due(
        mod,
        {
            "anchor": monday_expr,
            "anchor_mode": "skip",
            "due": stamp(date(2026, 5, 11), (9, 0)),
            "end": stamp(date(2026, 5, 11), (10, 0)),
            "chainID": "abcd1234",
        },
    )
    expect(
        mod.core.to_local(child_due).date() == date(2027, 5, 17),
        f"completion lost ISO-week weekday: {child_due}",
    )

    saved_collect = getattr(mod, "_collect_prev_two", None)
    mod._collect_prev_two = lambda _task: []
    try:
        lines = call_with_supported_kwargs(
            mod._timeline_lines,
            kind="anchor",
            task={
                "anchor": monday_expr,
                "anchor_mode": "skip",
                "link": 2,
                "due": stamp(date(2026, 5, 11), (9, 0)),
                "end": stamp(date(2026, 5, 11), (10, 0)),
                "chainID": "abcd1234",
            },
            child_due_utc=child_due,
            child_short="0000abcd",
            dnf=child_dnf,
            next_count=3,
            cap_no=None,
            cur_no=2,
        )
    finally:
        if saved_collect is not None:
            mod._collect_prev_two = saved_collect
        else:
            delattr(mod, "_collect_prev_two")
    timeline = strip_markup("\n".join(lines))
    expect("2027-05-17" in timeline, f"timeline omitted ordinal child: {timeline}")
    expect(
        "2028-05-15" in timeline,
        f"timeline omitted future ordinal occurrence: {timeline}",
    )


def test_local_datetime_non_hour_dst_gap_is_shared_by_modify():
    """A 30-minute DST gap shifts by its actual transition size everywhere."""
    from nautical_core.timeutil import build_local_datetime

    zone = ZoneInfo("Australia/Lord_Howe")
    scheduled = build_local_datetime(date(2026, 10, 4), (2, 15), zone)
    scheduled_local = scheduled.astimezone(zone)
    expect(
        (scheduled_local.hour, scheduled_local.minute) == (2, 45),
        f"non-hour DST shift changed: {scheduled_local}",
    )

    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_modify_non_hour_dst_gap_test")
    old_name, old_tz = mod.core.LOCAL_TZ_NAME, mod.core._LOCAL_TZ
    try:
        mod.core.LOCAL_TZ_NAME = "Australia/Lord_Howe"
        mod.core._LOCAL_TZ = zone
        effects = mod._module("modify_datetime_effects")
        carried = effects.local_naive_to_utc(
            effects.datetime_effect_ports_for(mod), datetime(2026, 10, 4, 2, 15)
        )
    finally:
        mod.core.LOCAL_TZ_NAME, mod.core._LOCAL_TZ = old_name, old_tz
    carried_local = carried.astimezone(zone)
    expect(
        (carried_local.hour, carried_local.minute) == (2, 45),
        f"modify DST resolver diverged: {carried_local}",
    )


def test_modify_completion_advances_past_second_dst_fold():
    """A second-fold completion cannot select an earlier first-fold slot."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_modify_second_fold_completion_test")
    zone = ZoneInfo("Europe/Bucharest")
    old_name, old_tz = mod.core.LOCAL_TZ_NAME, mod.core._LOCAL_TZ
    try:
        mod.core.LOCAL_TZ_NAME = "Europe/Bucharest"
        mod.core._LOCAL_TZ = zone
        due = datetime(2026, 10, 25, 3, 0, tzinfo=zone, fold=0)
        completed = datetime(2026, 10, 25, 3, 15, tzinfo=zone, fold=1)
        child_due, _meta, _dnf = compute_anchor_child_due(
            mod,
            {
                "anchor": "w:sun@t=03:20",
                "anchor_mode": "skip",
                "chainID": "dst-second-fold",
                "link": 1,
                "due": mod.core.fmt_isoz(due),
                "end": mod.core.fmt_isoz(completed),
            },
        )
    finally:
        mod.core.LOCAL_TZ_NAME, mod.core._LOCAL_TZ = old_name, old_tz
    child_local = child_due.astimezone(zone)
    expect(
        child_local.date() == date(2026, 11, 1)
        and (child_local.hour, child_local.minute) == (3, 20),
        f"second-fold completion selected a backward occurrence: {child_local}",
    )


def test_modify_overnight_window_advances_past_second_dst_fold():
    """An overnight window rejects its first-fold slot after a second-fold cursor."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_modify_overnight_second_fold_test")
    zone = ZoneInfo("Europe/Bucharest")
    old_name, old_tz = mod.core.LOCAL_TZ_NAME, mod.core._LOCAL_TZ
    try:
        mod.core.LOCAL_TZ_NAME = "Europe/Bucharest"
        mod.core._LOCAL_TZ = zone
        dnf = mod.core.validate_anchor_expr_strict("w:sat@t=22:20..03:20/6")
        cursor = datetime(2026, 10, 25, 3, 15, tzinfo=zone, fold=1)
        schedule = mod._module("modify_schedule_effects")
        from nautical_core.add_anchor_compute import (
            anchor_next_occurrence_after_local_dt,
        )

        result = schedule.next_occurrence_after_local_dt(
            schedule.OccurrencePorts(
                lambda expression,
                after,
                **kwargs: anchor_next_occurrence_after_local_dt(
                    expression, after, core=mod.core, **kwargs
                )
            ),
            dnf,
            cursor,
            default_seed_date=date(2026, 10, 24),
            seed_base="dst-overnight-second-fold",
            fallback_hhmm=(22, 20),
        )
    finally:
        mod.core.LOCAL_TZ_NAME, mod.core._LOCAL_TZ = old_name, old_tz
    expect(
        result.date() == date(2026, 10, 31)
        and (result.hour, result.minute) == (22, 20),
        f"overnight second-fold cursor selected a backward occurrence: {result}",
    )


def test_anchor_preview_explains_nonexistent_wall_time_adjustment():
    """The add panel identifies a fixed anchor time shifted by DST."""
    hook = find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "nautical.toml"
        config.write_text('tz = "Australia/Lord_Howe"\n', encoding="utf-8")
        task = {
            "uuid": "00000000-0000-4000-8000-000000000141",
            "description": "non-hour DST preview",
            "status": "pending",
            "entry": "20260801T000000Z",
            "due": "20261003T143000Z",
            "anchor": "y:10-04@t=02:15,02:45",
            "anchor_mode": "skip",
        }
        process = run_hook_script(
            hook, task, env_extra={"NAUTICAL_CONFIG": str(config), "NO_COLOR": "1"}
        )
    expect(process.returncode == 0, f"non-hour DST preview failed: {process.stderr!r}")
    panel = strip_markup(process.stderr)
    expect("DST adjusted" in panel, f"DST adjustment row is missing: {panel!r}")
    expect("02:15 -> 02:45" in panel, f"DST adjustment clocks are missing: {panel!r}")


TESTS = (
    test_year_ordinals_hooks_modes_calendar_and_timeline,
    test_local_datetime_non_hour_dst_gap_is_shared_by_modify,
    test_modify_completion_advances_past_second_dst_fold,
    test_modify_overnight_window_advances_past_second_dst_fold,
    test_anchor_preview_explains_nonexistent_wall_time_adjustment,
)
