"""Golden contracts for temporal recurrence and hook scheduling behavior."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

from dev_tools.golden_tests.support import (
    assert_stdout_json_only,
    build_child_draft_for_test,
    call_with_supported_kwargs,
    compute_anchor_child_due,
    expect,
    find_hook_file,
    evaluator_for_fixture,
    load_core_module,
    load_hook_module,
    run_hook_script,
    strip_markup,
)

ROOT = Path(__file__).resolve().parents[2]


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


def test_random_anchor_and_omit_presets_keep_chain_scope():
    """Random presets should preserve the same chain-scoped draw contract."""
    def verify(mod):
        anchor_omit = mod._import_sibling("anchor_omit")
        start = date(2026, 6, 1)
        preset_dnf = mod.validate_anchor_expr_strict("@random-workday")
        direct_dnf = mod.validate_anchor_expr_strict("m:rand@bd")
        preset_picks = []
        for idx in range(48):
            seed = f"random-preset-{idx}"
            preset_pick, _meta = mod.next_after_expr(
                preset_dnf,
                start,
                default_seed=start,
                seed_base=seed,
            )
            direct_pick, _meta = mod.next_after_expr(
                direct_dnf,
                start,
                default_seed=start,
                seed_base=seed,
            )
            expect(preset_pick == direct_pick, f"anchor preset changed the random draw for {seed}")
            preset_picks.append(preset_pick)
        expect(len(set(preset_picks)) >= 12, f"random anchor preset lacked chain diversity: {preset_picks}")

        omit_dnf = anchor_omit.validate_omit_expr_strict(
            "@random-weekday",
            validate_anchor_expr_cached=mod.validate_anchor_expr_strict,
            resolve_omit_presets=mod.resolve_omit_presets,
        )
        omit_picks = []
        weekly_dnf = mod.validate_anchor_expr_strict("w:rand")
        week_start = date(2026, 6, 7)
        for idx in range(48):
            seed = f"random-omit-{idx}"
            selected, _meta = mod.next_after_expr(
                weekly_dnf,
                week_start,
                default_seed=week_start,
                seed_base=seed,
            )
            expect(
                anchor_omit.omit_expr_fires_on_date(
                    omit_dnf,
                    selected,
                    week_start,
                    seed,
                    core=mod,
                ),
                f"random omit preset did not recognize its selected date for {seed}",
            )
            omit_picks.append(selected.weekday())
        expect(len(set(omit_picks)) >= 5, f"random omit preset lacked chain diversity: {omit_picks}")

    core_path = ROOT / "nautical_core" / "__init__.py"
    with tempfile.TemporaryDirectory() as td:
        cfg = Path(td) / "nautical.toml"
        cfg.write_text(
            '[anchor_presets]\nrandom-workday = "m:rand@bd"\n\n'
            '[omit_presets]\nrandom-weekday = "w:rand"\n',
            encoding="utf-8",
        )
        mod = load_core_module(str(core_path), "_nautical_core_random_preset_test", str(cfg))
        verify(mod)

def test_on_modify_reuses_task_scoped_evaluator_and_scheduler_binding():
    """One completion task should build its evaluator and scheduler binding once."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_evaluator_session_test")
    task = {
        "uuid": "00000000-0000-4000-8000-000000000111",
        "chainID": "session-chain",
        "status": "pending",
        "link": 1,
        "anchor": "w:mon@t=09:00",
        "anchor_mode": "skip",
        "due": "20250106T090000Z",
        "end": "20250106T100000Z",
    }
    mod._reset_modify_runtime_state()
    try:
        schedule = mod._module("modify_schedule_effects")
        evaluator, _service = schedule.scheduler_callbacks(schedule.scheduler_ports_for(mod))
        first = evaluator(task)
        second = evaluator(dict(task))
        expect(first is second, "equivalent task copies rebuilt the evaluator within one hook session")
        binding_a = first._get_cached("scheduler_binding", first._build_scheduler_binding)
        binding_b = first._get_cached("scheduler_binding", first._build_scheduler_binding)
        expect(binding_a is binding_b, "scheduler binding was rebuilt within one evaluator session")
    finally:
        mod._reset_modify_runtime_state()


def test_random_time_window_is_stable_across_processes():
    """The random-time seed must not depend on interpreter-local state."""
    script = (
        "import json; from nautical_core.time_windows import parse_random_time_window_spec; "
        "w=parse_random_time_window_spec('rand(22:30..02:30/3)'); "
        "print(json.dumps(w.slots_with_offsets('cross-process/2026-08-05')))"
    )
    outputs = [
        subprocess.check_output([sys.executable, "-c", script], cwd=str(ROOT), text=True).strip()
        for _ in range(2)
    ]
    expect(outputs[0] == outputs[1], f"random slots changed across processes: {outputs!r}")


def test_astronomical_season_selection_scheduler_uses_transition_dates():
    """Public seasonal scheduling should consume astronomical local-date windows."""
    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td)
        (taskdata / "config-nautical.toml").write_text(
            'tz = "UTC"\nseason_mode = "astronomical"\nseason_hemisphere = "north"\n',
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["TASKDATA"] = str(taskdata)
        env.pop("NAUTICAL_CONFIG", None)
        env["PYTHONPATH"] = str(ROOT)
        script = (
            "import json, os\n"
            "from datetime import date\n"
            "import nautical_core as c\n"
            "c.reload_taskdata_config(os.environ['TASKDATA'])\n"
            "import nautical_core.position_selection as position_selection\n"
            "import nautical_core.season_support as season_support\n"
            "dnf = c.validate_anchor_expr_strict('(w:mon)@in-season=1st')\n"
            "refs = [date(2026, 1, 1), date(2026, 3, 23), date(2026, 6, 22), date(2026, 9, 28), date(2026, 12, 21)]\n"
            "dates = [c.next_after_expr(dnf, ref, default_seed=date(2026, 1, 1))[0].isoformat() for ref in refs]\n"
            "advice = position_selection.selection_advice(dnf[0][0])\n"
            "print(json.dumps({'mode': season_support.active_mode(), 'dates': dates, 'bounds': tuple(x.isoformat() for x in position_selection.period_bounds('spring', date(2026, 4, 1))), 'advice': advice}))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
        )
        expect(proc.returncode == 0, f"astronomical scheduler process failed: {proc.stderr[:800]!r}")
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        expect(payload["mode"] == "astronomical", f"configured season mode was not applied: {payload!r}")
        expect(
            payload["dates"] == ["2026-03-23", "2026-06-22", "2026-09-28", "2026-12-21", "2027-03-22"],
            f"astronomical season date drifted: {payload!r}",
        )
        expect(payload["bounds"] == ["2026-03-20", "2026-06-20"], f"astronomical bounds drifted: {payload!r}")
        expect(any("astronomical" in line for line in payload["advice"]), f"astronomical advice missing: {payload!r}")


def test_seasonal_selection_modify_modes_times_and_timeline():
    """Completion modes should preserve seasonal slots, local times, and future projections."""
    modify_hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(modify_hook, "_nautical_seasonal_modify_modes_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()
    season_support = mod.core._import_sibling("season_support")
    previous_hemisphere = season_support.active_hemisphere()
    season_support.configure_hemisphere("north")
    mod.core.SEASON_HEMISPHERE = "north"

    previous_tz = mod.core._LOCAL_TZ
    mod.core._LOCAL_TZ = ZoneInfo("Europe/Helsinki")
    expression = "(w:mon)@in-spring=first,last@t=09:00,17:00"

    def stamp(day, hhmm):
        return mod.core.fmt_isoz(mod.core.build_local_datetime(day, hhmm))

    try:
        same_day_due, _same_meta, _same_dnf = compute_anchor_child_due(mod,
            {
                "anchor": expression,
                "anchor_mode": "skip",
                "due": stamp(date(2026, 3, 2), (9, 0)),
                "end": stamp(date(2026, 3, 2), (10, 0)),
                "chainID": "season123",
            }
        )
        same_day_local = mod.core.to_local(same_day_due)
        expect(
            same_day_local.date() == date(2026, 3, 2)
            and (same_day_local.hour, same_day_local.minute) == (17, 0),
            f"completion skipped the second same-day seasonal slot: {same_day_local}",
        )

        common = {
            "anchor": expression,
            "due": stamp(date(2026, 3, 2), (17, 0)),
            "end": stamp(date(2026, 7, 1), (10, 0)),
            "chainID": "season123",
        }
        all_due, all_meta, _all_dnf = compute_anchor_child_due(mod, dict(common, anchor_mode="all"))
        skip_due, skip_meta, skip_dnf = compute_anchor_child_due(mod, dict(common, anchor_mode="skip"))
        flex_due, flex_meta, _flex_dnf = compute_anchor_child_due(mod, dict(common, anchor_mode="flex"))
        all_local = mod.core.to_local(all_due)
        skip_local = mod.core.to_local(skip_due)
        flex_local = mod.core.to_local(flex_due)
        expect(
            all_local.date() == date(2026, 5, 25)
            and (all_local.hour, all_local.minute) == (9, 0),
            f"all mode did not backfill the missed spring slot: {all_local}",
        )
        expect(all_meta.get("basis") == "missed", f"all mode metadata drifted: {all_meta}")
        expect(all_meta.get("source") == "anchor", f"all mode source drifted: {all_meta}")
        expect(
            skip_local.date() == date(2027, 3, 1)
            and (skip_local.hour, skip_local.minute) == (9, 0),
            f"skip mode did not advance to the next spring: {skip_local}",
        )
        expect(skip_meta.get("basis") == "after_end", f"skip metadata drifted: {skip_meta}")
        expect(skip_meta.get("source") == "anchor", f"skip mode source drifted: {skip_meta}")
        expect(flex_local == skip_local, f"flex mode did not skip the seasonal backlog: {flex_local}")
        expect(flex_meta.get("basis") == "flex", f"flex metadata drifted: {flex_meta}")
        expect(flex_meta.get("source") == "anchor", f"flex mode source drifted: {flex_meta}")

        evaluator = evaluator_for_fixture(common, timezone=mod.core._LOCAL_TZ)
        for mode, hook_due, hook_meta in (
            ("all", all_due, all_meta),
            ("skip", skip_due, skip_meta),
            ("flex", flex_due, flex_meta),
        ):
            evaluator_result = evaluator.select_mode(
                mode,
                due_local=mod.core.to_local(mod.core.parse_dt_any(common["due"])),
                end_local=mod.core.to_local(mod.core.parse_dt_any(common["end"])),
                fallback_hhmm=(17, 0),
            )
            expect(
                evaluator_result.selected_occurrence is not None
                and evaluator_result.selected_occurrence.astimezone(timezone.utc) == hook_due,
                f"{mode} evaluator timestamp drifted from hook: {evaluator_result!r} vs {hook_due!r}",
            )
            expect(
                evaluator_result.basis == hook_meta.get("basis")
                and evaluator_result.source == hook_meta.get("source"),
                f"{mode} evaluator evidence drifted from hook: {evaluator_result!r} vs {hook_meta!r}",
            )
        expect(all_local.utcoffset() == timedelta(hours=3), f"summer offset drifted: {all_local}")
        expect(skip_local.utcoffset() == timedelta(hours=2), f"winter offset drifted: {skip_local}")

        parent = {
            **common,
            "uuid": "00000000-0000-4000-8000-000000000784",
            "status": "completed",
            "anchor_mode": "flex",
            "link": 1,
        }
        child = build_child_draft_for_test(
            mod, parent, flex_due, "due", 2, "00000000", "anchor", 0, None
        )
        expect(child.get("anchor") == expression, f"child lost seasonal anchor: {child}")
        expect(child.get("anchor_mode") == "all", f"flex child did not become all mode: {child}")
        expect(child.get("chainID") == "season123", f"child lost chain identity: {child}")

        saved_collect = getattr(mod, "_collect_prev_two", None)
        mod._collect_prev_two = lambda _task: []
        try:
            lines = call_with_supported_kwargs(
                mod._timeline_lines,
                kind="anchor",
                task={**parent, "anchor_mode": "skip"},
                child_due_utc=skip_due,
                child_short="0000abcd",
                dnf=skip_dnf,
                next_count=4,
                cap_no=None,
                cur_no=1,
            )
        finally:
            if saved_collect is not None:
                mod._collect_prev_two = saved_collect
            else:
                delattr(mod, "_collect_prev_two")
        timeline = strip_markup("\n".join(lines))
        expect("2027-03-01" in timeline, f"timeline omitted seasonal child: {timeline}")
        expect("2027-05-31" in timeline, f"timeline omitted later spring slot: {timeline}")
    finally:
        mod.core._LOCAL_TZ = previous_tz
        mod.core.SEASON_HEMISPHERE = previous_hemisphere
        season_support.configure_hemisphere(previous_hemisphere)


TESTS = (
    test_year_ordinals_hooks_modes_calendar_and_timeline,
    test_local_datetime_non_hour_dst_gap_is_shared_by_modify,
    test_modify_completion_advances_past_second_dst_fold,
    test_modify_overnight_window_advances_past_second_dst_fold,
    test_anchor_preview_explains_nonexistent_wall_time_adjustment,
    test_random_anchor_and_omit_presets_keep_chain_scope,
    test_on_modify_reuses_task_scoped_evaluator_and_scheduler_binding,
    test_random_time_window_is_stable_across_processes,
    test_astronomical_season_selection_scheduler_uses_transition_dates,
    test_seasonal_selection_modify_modes_times_and_timeline,
)
