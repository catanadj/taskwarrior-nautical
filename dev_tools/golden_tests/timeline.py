"""Golden contracts for hook-generated recurrence timelines."""

from datetime import datetime, timezone
import importlib
import re
import tempfile
from pathlib import Path

from dev_tools.golden_tests.support import (
    call_with_supported_kwargs,
    compute_anchor_child_due,
    expect,
    find_hook_file,
    load_hook_module,
    strip_markup,
)

core = importlib.import_module("nautical_core")


def test_hook_on_modify_timeline_multitime_includes_all_slots():
    """on-modify timeline generator must step occurrences (date+time), not only dates."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_test")
    if not hasattr(mod, "_timeline_lines"):
        raise AssertionError("on-modify hook does not expose _timeline_lines; cannot validate timeline stepping.")
    setattr(mod, "_collect_prev_two", lambda _task: [])
    expr = "w:mon..sun@t=06:00,12:00,22:00"
    dnf = core.validate_anchor_expr_strict(expr)
    child_due_utc = datetime(2025, 12, 20, 22, 0, tzinfo=timezone.utc)
    task = {
        "uuid": "00000000-0000-4000-8000-000000000222",
        "description": "hook test on-modify multitime",
        "anchor": expr,
        "anchor_mode": "skip",
        "link": 1,
        "end": "20251220T090000Z",
        "due": "20251220T120000Z",
    }
    lines = call_with_supported_kwargs(
        mod._timeline_lines,
        kind="anchor",
        task=task,
        child_due_utc=child_due_utc,
        child_short="0000abcd",
        dnf=dnf,
        next_count=8,
        cap_no=None,
        cur_no=1,
    )
    txt = strip_markup("\n".join(lines))
    times = sorted(set(re.findall(r"\b\d{2}:\d{2}\b", txt)))
    for value in ("06:00", "12:00", "22:00"):
        if value not in times:
            raise AssertionError(f"on-modify timeline missing time {value}. found={times}. text={txt[:500]!r}")
    if len(times) < 3:
        raise AssertionError(f"on-modify timeline collapsed times unexpectedly: {times}. text={txt[:500]!r}")


def test_hook_on_modify_timeline_cp_sequence_labels_future_intervals():
    """cp sequence timelines should show the interval used for future rows."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_cp_sequence_timeline_test")
    if not hasattr(mod, "_timeline_lines"):
        raise AssertionError("on-modify hook does not expose _timeline_lines; cannot validate cp sequence timeline.")
    setattr(mod, "_collect_prev_two", lambda _task: [])
    evaluator_calls = {"count": 0}
    schedule = mod._module("modify_schedule_effects")
    original_callbacks = schedule.scheduler_callbacks
    original_evaluator = original_callbacks(schedule.scheduler_ports_for(mod))[0]

    def _shared_evaluator(task):
        evaluator_calls["count"] += 1
        return original_evaluator(task)

    def _callbacks(ports):
        _evaluator, service = original_callbacks(ports)
        return _shared_evaluator, service

    schedule.scheduler_callbacks = _callbacks
    child_due_utc = datetime(2026, 1, 4, 9, 0, tzinfo=timezone.utc)
    task = {
        "uuid": "00000000-0000-4000-8000-000000000224",
        "description": "hook test on-modify cp sequence timeline",
        "cp": "3d,20d,7d",
        "link": 1,
        "end": "20260101T100000Z",
        "due": "20260101T090000Z",
    }
    lines = call_with_supported_kwargs(
        mod._timeline_lines,
        kind="cp",
        task=task,
        child_due_utc=child_due_utc,
        child_short="0000abcd",
        dnf=None,
        next_count=3,
        cap_no=None,
        cur_no=1,
    )
    txt = strip_markup("\n".join(lines))
    for token in ("(20d)", "(7d)", "(3d)"):
        expect(token in txt, f"cp sequence timeline missing {token}: {txt}")
    expect(
        evaluator_calls["count"] == 1,
        f"CP timeline rebuilt the task evaluator instead of reusing one session: {evaluator_calls}",
    )


def test_hook_on_modify_timeline_cp_random_labels_selected_intervals():
    """cp random timelines should display the selected interval for each future row."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_cp_random_timeline_test")
    if not hasattr(mod, "_timeline_lines"):
        raise AssertionError("on-modify hook does not expose _timeline_lines; cannot validate cp random timeline.")
    setattr(mod, "_collect_prev_two", lambda _task: [])
    cp = "rand(11d..14d)"
    chain_id = "chain-a"
    first_td = mod.core.cp_sequence_interval_for_link(cp, 1, chain_id)
    child_due_utc = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc) + first_td
    task = {
        "uuid": "00000000-0000-4000-8000-000000000225",
        "description": "hook test on-modify cp random timeline",
        "cp": cp,
        "chainID": chain_id,
        "link": 1,
        "end": "20260101T100000Z",
        "due": "20260101T090000Z",
    }
    lines = call_with_supported_kwargs(
        mod._timeline_lines,
        kind="cp",
        task=task,
        child_due_utc=child_due_utc,
        child_short="0000abcd",
        dnf=None,
        next_count=3,
        cap_no=None,
        cur_no=1,
    )
    txt = strip_markup("\n".join(lines))
    expect("(rand(" not in txt, f"cp random timeline should not show raw rand tokens: {txt}")
    expected_days = {
        int(mod.core.cp_sequence_interval_for_link(cp, link_no, chain_id).total_seconds() // 86400)
        for link_no in (2, 3, 4)
    }
    for selected_days in expected_days:
        expect(
            f"({selected_days}d)" in txt,
            f"cp random timeline omitted chain-scoped interval {selected_days}d: {txt}",
        )


def test_hook_on_modify_timeline_marks_omitted_anchor_slots():
    """anchor timelines should mark omitted future slots instead of showing them as normal links."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_omit_timeline_test")
    if not hasattr(mod, "_timeline_lines"):
        raise AssertionError("on-modify hook does not expose _timeline_lines; cannot validate omit timeline handling.")
    setattr(mod, "_collect_prev_two", lambda _task: [])
    expr = "w:mon,wed,fri"
    dnf = core.validate_anchor_expr_strict(expr)
    child_due_utc = datetime(2025, 1, 10, 9, 0, tzinfo=timezone.utc)
    task = {
        "uuid": "00000000-0000-4000-8000-000000000333",
        "description": "hook test on-modify omit timeline",
        "anchor": expr,
        "omit": "w:wed",
        "anchor_mode": "skip",
        "link": 1,
        "end": "20250106T090000Z",
        "due": "20250106T090000Z",
        "chainID": "abcd1234",
    }
    lines = call_with_supported_kwargs(
        mod._timeline_lines,
        kind="anchor",
        task=task,
        child_due_utc=child_due_utc,
        child_short="0000abcd",
        dnf=dnf,
        next_count=3,
        cap_no=None,
        cur_no=1,
    )
    txt = strip_markup("\n".join(lines))
    expect("(omitted)" in txt, f"expected omitted marker in anchor timeline: {txt!r}")
    expect(
        "2025-01-08" in txt or "2025-01-08 09:00" in txt or "Wed 2025-01-08" in txt,
        f"expected omitted Wednesday slot to remain visible: {txt!r}",
    )


def test_hook_on_modify_merged_timeline_marks_projection_failures():
    """Merged anchor/anchor-file timelines should expose provider failures as warning rows."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_merged_timeline_warning_test")
    from nautical_core.recurrence_evaluator import RecurrenceEvaluator

    previous_next = RecurrenceEvaluator._default_next_occurrence_after_local_dt
    previous_prev = getattr(mod, "_collect_prev_two", None)

    def broken(*args, **kwargs):
        raise ValueError("merged provider contract broken")

    previous_anchor_dir = getattr(mod.core, "ANCHOR_FILE_DIR", "")
    try:
        RecurrenceEvaluator._default_next_occurrence_after_local_dt = broken
        if previous_prev is not None:
            mod._collect_prev_two = lambda _task: []
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "2026.csv").write_text("date\n2026-08-10\n", encoding="utf-8")
            mod.core.ANCHOR_FILE_DIR = td
            task = {
                "uuid": "00000000-0000-4000-8000-000000000558",
                "description": "merged timeline warning",
                "anchor": "w:mon",
                "anchor_file": "2026.csv",
                "due": "20260803T090000Z",
                "end": "20260803T090000Z",
                "link": 1,
                "chainID": "timeline-merged-warning",
            }
            lines = call_with_supported_kwargs(
                mod._timeline_lines,
                kind="anchor",
                task=task,
                child_due_utc=datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc),
                child_short="f17ca92b",
                dnf=core.validate_anchor_expr_strict("w:mon"),
                next_count=2,
                cap_no=None,
                cur_no=1,
            )
    finally:
        RecurrenceEvaluator._default_next_occurrence_after_local_dt = previous_next
        if previous_prev is not None:
            mod._collect_prev_two = previous_prev
        mod.core.ANCHOR_FILE_DIR = previous_anchor_dir

    text = strip_markup("\n".join(lines))
    expect("Projection unavailable" in text, f"merged projection failure was hidden: {text!r}")
    expect("merged provider contract broken" in text, f"merged warning lost failure detail: {text!r}")


def test_hook_on_modify_timeline_uses_omit_file_description_label():
    """anchor timelines should use omit_file description text for omitted markers when available."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_omit_file_desc_timeline_test")
    if not hasattr(mod, "_timeline_lines"):
        raise AssertionError("on-modify hook does not expose _timeline_lines; cannot validate omit timeline handling.")
    if hasattr(mod, "_collect_prev_two"):
        setattr(mod, "_collect_prev_two", lambda _task: [])
    expr = "w:mon,wed,fri"
    dnf = core.validate_anchor_expr_strict(expr)
    child_due_utc = datetime(2025, 1, 10, 9, 0, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as td:
        omit_dir = Path(td)
        (omit_dir / "holidays.csv").write_text(
            "date,description\n"
            "2025-01-08,Company holiday shutdown\n",
            encoding="utf-8",
        )
        prev_dir = getattr(mod.core, "OMIT_FILE_DIR", "")
        mod.core.OMIT_FILE_DIR = str(omit_dir)
        try:
            task = {
                "uuid": "00000000-0000-4000-8000-000000000334",
                "description": "hook test on-modify omit_file label timeline",
                "anchor": expr,
                "omit_file": "holidays.csv",
                "anchor_mode": "skip",
                "link": 1,
                "end": "20250106T090000Z",
                "due": "20250106T090000Z",
                "chainID": "abcd1234",
            }
            lines = call_with_supported_kwargs(
                mod._timeline_lines,
                kind="anchor",
                task=task,
                child_due_utc=child_due_utc,
                child_short="0000abcd",
                dnf=dnf,
                next_count=3,
                cap_no=None,
                cur_no=1,
            )
        finally:
            mod.core.OMIT_FILE_DIR = prev_dir
    txt = strip_markup("\n".join(lines))
    expect("(Company holida...)" in txt, f"expected truncated omit_file description marker in anchor timeline: {txt!r}")
    expect("(omitted)" not in txt, f"expected omit_file description to replace default omitted marker: {txt!r}")


def test_hook_on_modify_timeline_keeps_anchor_match_after_shifted_anchor_file_child():
    """when anchor_file is shifted and anchor matches the original file date, timeline should still show the original date as the next future anchor."""
    from zoneinfo import ZoneInfo

    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_shifted_anchor_file_timeline_test")
    setattr(mod, "_collect_prev_two", lambda _task: [])
    previous_tz_name = mod.core.LOCAL_TZ_NAME
    previous_tz = mod.timezone_facade._local_timezone
    mod.core.LOCAL_TZ_NAME = "Europe/Bucharest"
    mod.timezone_facade._local_timezone = ZoneInfo("Europe/Bucharest")
    try:
        with tempfile.TemporaryDirectory() as td:
            anchor_dir = Path(td)
            (anchor_dir / "2026.csv").write_text("date\n2026-04-25\n", encoding="utf-8")
            old_dir = getattr(mod.core, "ANCHOR_FILE_DIR", "")
            mod.core.ANCHOR_FILE_DIR = str(anchor_dir)
            try:
                parent = {
                    "uuid": "00000000-0000-4000-8000-000000000555",
                    "description": "shifted anchor_file timeline",
                    "anchor": "y:04-25@t=12:00",
                    "anchor_file": "2026.csv@-1d@t=12:00",
                    "anchor_mode": "skip",
                    "link": 1,
                    "chainID": "abcd1234",
                    "due": "2026-04-23T12:00:00Z",
                    "end": "2026-04-23T13:00:00Z",
                }
                child_due, _meta, dnf = compute_anchor_child_due(mod, parent)
                expect(mod.core.fmt_isoz(child_due) == "2026-04-24T09:00:00Z", f"unexpected shifted child due: {mod.core.fmt_isoz(child_due)}")
                lines = call_with_supported_kwargs(
                    mod._timeline_lines,
                    kind="anchor",
                    task=parent,
                    child_due_utc=child_due,
                    child_short="beeswax",
                    dnf=dnf,
                    _collect_prev_two_override=lambda _task: [],
                    next_count=4,
                    cap_no=None,
                    cur_no=1,
                )
            finally:
                mod.core.ANCHOR_FILE_DIR = old_dir
    finally:
        mod.core.LOCAL_TZ_NAME = previous_tz_name
        mod.timezone_facade._local_timezone = previous_tz
    txt = strip_markup("\n".join(lines))
    expect("Fri 2026-04-24 12:00" in txt, f"expected shifted anchor_file child in timeline: {txt!r}")
    expect("Sat 2026-04-25 12:00" in txt, f"expected original file date preserved via anchor match: {txt!r}")


def test_hook_on_modify_timeline_omits_shifted_anchor_file_dates_in_merged_stream():
    """merged anchor timelines should still omit shifted anchor_file dates when omit matches their shifted local date."""
    from zoneinfo import ZoneInfo

    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_shifted_anchor_file_omit_timeline_test")
    setattr(mod, "_collect_prev_two", lambda _task: [])
    previous_tz_name = mod.core.LOCAL_TZ_NAME
    previous_tz = mod.timezone_facade._local_timezone
    mod.core.LOCAL_TZ_NAME = "Europe/Bucharest"
    mod.timezone_facade._local_timezone = ZoneInfo("Europe/Bucharest")
    try:
        with tempfile.TemporaryDirectory() as td:
            anchor_dir = Path(td)
            (anchor_dir / "2026.csv").write_text("date\n2026-05-01\n2026-05-05\n", encoding="utf-8")
            old_dir = getattr(mod.core, "ANCHOR_FILE_DIR", "")
            mod.core.ANCHOR_FILE_DIR = str(anchor_dir)
            try:
                parent = {
                    "uuid": "00000000-0000-4000-8000-000000000556",
                    "description": "shifted anchor_file omit timeline",
                    "anchor": "w:tue,fri | y:05-05",
                    "anchor_file": "2026.csv@-1d@t=12:00,18:00",
                    "omit": "y:04-28..05-05",
                    "anchor_mode": "skip",
                    "link": 4,
                    "chainID": "abcd1234",
                    "due": "2026-04-24T09:00:00Z",
                    "end": "2026-04-24T09:00:00Z",
                }
                child_due = mod.core.parse_dt_any("2026-04-24T09:00:00Z")
                dnf = mod.core.validate_anchor_expr_strict(parent["anchor"])
                lines = call_with_supported_kwargs(
                    mod._timeline_lines,
                    kind="anchor",
                    task=parent,
                    child_due_utc=child_due,
                    child_short="f17ca92b",
                    dnf=dnf,
                    next_count=6,
                    cap_no=None,
                    cur_no=4,
                )
            finally:
                mod.core.ANCHOR_FILE_DIR = old_dir
    finally:
        mod.core.LOCAL_TZ_NAME = previous_tz_name
        mod.timezone_facade._local_timezone = previous_tz
    txt = strip_markup("\n".join(lines))
    expect("Thu 2026-04-30 12:00" in txt and "(omitted)" in txt, f"shifted omitted anchor_file date was not marked: {txt!r}")
    expect("Thu 2026-04-30 18:00" in txt and "(omitted)" in txt, f"shifted omitted anchor_file date was not marked: {txt!r}")
    expect("Mon 2026-05-04 12:00" in txt and "(omitted)" in txt, f"shifted omitted anchor_file date was not marked: {txt!r}")


def test_hook_on_modify_timeline_shows_anchor_side_omit_file_dates_in_merged_stream():
    """merged timelines should still show omitted anchor-side dates when omit_file blocks them."""
    from zoneinfo import ZoneInfo

    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_anchor_side_omit_file_timeline_test")
    if hasattr(mod, "_collect_prev_two"):
        setattr(mod, "_collect_prev_two", lambda _task: [])
    previous_tz_name = mod.core.LOCAL_TZ_NAME
    previous_tz = mod.timezone_facade._local_timezone
    mod.core.LOCAL_TZ_NAME = "Europe/Bucharest"
    mod.timezone_facade._local_timezone = ZoneInfo("Europe/Bucharest")
    try:
        with tempfile.TemporaryDirectory() as td:
            anchor_dir = Path(td)
            omit_dir = Path(td)
            (anchor_dir / "2026.csv").write_text("date\n2026-05-01\n2026-05-05\n", encoding="utf-8")
            old_anchor_dir = getattr(mod.core, "ANCHOR_FILE_DIR", "")
            old_omit_dir = getattr(mod.core, "OMIT_FILE_DIR", "")
            mod.core.ANCHOR_FILE_DIR = str(anchor_dir)
            mod.core.OMIT_FILE_DIR = str(omit_dir)
            try:
                parent = {
                    "uuid": "00000000-0000-4000-8000-000000000557",
                    "description": "anchor side omit_file timeline",
                    "anchor": "w:tue,fri | y:05-05",
                    "anchor_file": "2026.csv@-1d@t=12:00,18:00",
                    "omit_file": "2026.csv",
                    "anchor_mode": "skip",
                    "link": 7,
                    "chainID": "abcd1234",
                    "due": "2026-04-30T15:00:00Z",
                    "end": "2026-04-30T15:00:00Z",
                }
                child_due = mod.core.parse_dt_any("2026-04-30T15:00:00Z")
                dnf = mod.core.validate_anchor_expr_strict(parent["anchor"])
                lines = call_with_supported_kwargs(
                    mod._timeline_lines,
                    kind="anchor",
                    task=parent,
                    child_due_utc=child_due,
                    child_short="ba5b8228",
                    dnf=dnf,
                    _collect_prev_two_override=lambda _task: [],
                    next_count=4,
                    cap_no=None,
                    cur_no=7,
                )
            finally:
                mod.core.ANCHOR_FILE_DIR = old_anchor_dir
                mod.core.OMIT_FILE_DIR = old_omit_dir
    finally:
        mod.core.LOCAL_TZ_NAME = previous_tz_name
        mod.timezone_facade._local_timezone = previous_tz
    txt = strip_markup("\n".join(lines))
    expect("Tue 2026-05-05" in txt, f"expected omitted anchor-side date to remain visible: {txt!r}")
    expect("(omitted)" in txt, f"expected merged timeline omitted marker for anchor-side omit_file date: {txt!r}")


TESTS = (
    test_hook_on_modify_timeline_multitime_includes_all_slots,
    test_hook_on_modify_timeline_cp_sequence_labels_future_intervals,
    test_hook_on_modify_timeline_cp_random_labels_selected_intervals,
    test_hook_on_modify_timeline_marks_omitted_anchor_slots,
    test_hook_on_modify_merged_timeline_marks_projection_failures,
    test_hook_on_modify_timeline_uses_omit_file_description_label,
    test_hook_on_modify_timeline_keeps_anchor_match_after_shifted_anchor_file_child,
    test_hook_on_modify_timeline_omits_shifted_anchor_file_dates_in_merged_stream,
    test_hook_on_modify_timeline_shows_anchor_side_omit_file_dates_in_merged_stream,
)
