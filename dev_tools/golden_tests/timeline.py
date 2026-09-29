"""Golden contracts for hook-generated recurrence timelines."""

from datetime import datetime, timezone
import importlib
import re
import tempfile
from pathlib import Path

from dev_tools.golden_tests.support import (
    call_with_supported_kwargs,
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


TESTS = (
    test_hook_on_modify_timeline_multitime_includes_all_slots,
    test_hook_on_modify_timeline_cp_sequence_labels_future_intervals,
    test_hook_on_modify_timeline_cp_random_labels_selected_intervals,
    test_hook_on_modify_timeline_marks_omitted_anchor_slots,
    test_hook_on_modify_merged_timeline_marks_projection_failures,
)
