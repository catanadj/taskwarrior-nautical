#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nautical Golden Tests
 - Imports local nautical_core/__init__.py
 - Verifies parsing, lint, natural, and next-occurrence properties
 - Covers prior regressions: leap day, quarters, /N monthly valid-month gating, last-<dow>,
   weekly AND unsatisfiable, @bd/@nbd/@nw natural text + date effects, rand with yearly window, etc.

Run:
  python3 nautical_golden_tests.py
Optional:
  python3 nautical_golden_tests.py --only leap --verbose
  python3 nautical_golden_tests.py --shuffle-seed 20260811
  python3 nautical_golden_tests.py --only reconcile --strict-lifecycle-warnings
"""

import importlib
import sys, os, re, json, io, contextlib
import random
import time
import sqlite3
import tempfile
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEV_TOOLS = HERE
CORE_TOOLS = os.path.join(ROOT, "nautical_core", "tools")
_TEST_OPERATOR_TASKDATA = []
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ.setdefault("NAUTICAL_CORE_PATH", ROOT)

from tests.support.lifecycle_execution import LifecycleExecutionFixture
from nautical_core.query_service import OccurrenceQueryRuntime
from nautical_core.panel_colours import chain_colour_root
from dev_tools.golden_tests.recurrence import TESTS as RECURRENCE_TESTS
from dev_tools.golden_tests.operator import TESTS as OPERATOR_TESTS
from dev_tools.golden_tests.installer import TESTS as INSTALLER_TESTS
from dev_tools.golden_tests.performance import TESTS as PERFORMANCE_TESTS
from dev_tools.golden_tests.storage import TESTS as STORAGE_TESTS
from dev_tools.golden_tests.timeline import TESTS as TIMELINE_TESTS
from dev_tools.golden_tests.lifecycle import TESTS as LIFECYCLE_TESTS
from dev_tools.golden_tests.reconcile import TESTS as RECONCILE_TESTS
from dev_tools.golden_tests.configuration import TESTS as CONFIGURATION_TESTS
from dev_tools.golden_tests.modify import TESTS as MODIFY_TESTS
from dev_tools.golden_tests.scheduling import TESTS as SCHEDULING_TESTS
from dev_tools.golden_tests.support import (
    astral_test_available as _astral_test_available,
    absent_task as _absent_task,
    chain_node as _chain_node,
    expect,
    iso,
    parse_due,
    scheduler_for_fixture as _scheduler_for_fixture,
    evaluator_for_fixture as _evaluator_for_fixture,
    seed_sqlite_queue as _seed_sqlite_queue,
    build_preview,
    doctor_findings as _doctor_findings,
    doctor_hook_installation as _doctor_hook_installation,
    doctor_obsolete_queue_state as _doctor_obsolete_queue_state,
    install_doctor_hook_wrappers as _install_doctor_hook_wrappers,
    write_fake_task_for_doctor as _write_fake_task_for_doctor,
    test_operator_uow as _test_operator_uow,
    load_core_module as _load_core_module,
    load_hook_module as _load_hook_module,
    load_hook_protocol_module as _load_hook_protocol_module,
    load_exit_probe_module as _load_exit_probe_module,
    must_preview as _must_preview,
    must_natural as _must_natural,
    run_hook_script as _run_hook_script,
    run_hook_script_raw as _run_hook_script_raw,
    modify_effect as _modify_effect,
    strip_markup as _strip_markup,
    found_task as _found_task,
    fixture_observation as _fixture_observation,
    fixture_task as _fixture_task,
    find_hook_file as _find_hook_file,
    generation_service as _generation_service,
    compute_anchor_child_due as _compute_anchor_child_due,
    compute_cp_child_due as _compute_cp_child_due,
    carry_relative_datetime as _carry_relative_datetime,
    carry_native_until as _carry_native_until,
    build_child_draft_for_test as _build_child_draft_for_test,
    force_tz_utc as _force_tz_utc,
    extract_last_json as _extract_last_json,
    assert_stdout_json_only as _assert_stdout_json_only,
    call_with_supported_kwargs as _call_with_supported_kwargs,
    has_function,
    child_payload_from_values as _child_payload_from_values,
    metadata_payload_from_values as _metadata_payload_from_values,
    must_parse as _must_parse,
    new_lifecycle_read_service as _new_lifecycle_read_service,
    plan_from_values as _plan_from_values,
    recovery_action as _recovery_action,
    recovery_child as _recovery_child,
    recovery_plan as _recovery_plan,
    task_draft as _task_draft,
    task_observation as _task_observation,
    task_observations as _task_observations,
    task_snapshot as _task_snapshot,
    test_term as _test_term,
    typed_command_result as _typed_command_result,
    unavailable_task as _unavailable_task,
)

core = importlib.import_module("nautical_core")
reconcile_report = importlib.import_module("nautical_core.reconcile_report")
_hook = importlib.import_module("nautical_core.hooks.modify_impl")

# -------- Helpers -------------------------------------------------------------

# -------- Test cases ----------------------------------------------------------
# -------- Hook checks ---------------------------------------------------------
# These tests validate the shipped hook scripts (on-add / on-modify) at a high
# level, to catch regressions that can slip through core-only tests.

import subprocess
import shutil
import importlib.util
import importlib.machinery
import inspect
import time as _time

class _BoundCompletionEffects:
    """Test-only bound view of the extracted completion-effects module."""

    _PORT_FACTORIES = {
        "preflight_context": "completion_preflight_context_ports_for",
        "compute_next_and_limits": "completion_compute_ports_for",
        "build_and_spawn_child": "completion_spawn_ports_for",
    }

    def __init__(self, hook):
        object.__setattr__(self, "_hook", hook)
        object.__setattr__(self, "_module", importlib.import_module("nautical_core.modify_completion_effects"))
        originals = getattr(type(self), "_originals", None)
        if originals is None:
            originals = {
                name: getattr(self._module, name)
                for name in (
                    "chain_snapshot", "existing_next_or_fail", "preflight_context",
                    "compute_child_due", "until_or_fail", "until_guard_or_stop",
                    "require_child_due_or_fail", "warn_unreasonable_duration", "caps",
                    "cap_guard_or_stop", "compute_next_and_limits", "build_and_spawn_child",
                )
            }
            setattr(type(self), "_originals", originals)
        else:
            for name, fn in originals.items():
                setattr(self._module, name, fn)

    def __getattr__(self, name):
        fn = getattr(self._module, name)
        factory_name = self._PORT_FACTORIES.get(name)
        if factory_name:
            return lambda *args, **kwargs: fn(
                getattr(self._module, factory_name)(self._hook), *args, **kwargs
            )
        if name == "chain_snapshot":
            def bound_chain_snapshot(chain_id, base_no, next_no, repository):
                context_ports = self._module.completion_preflight_context_ports_for(self._hook)
                ports = self._module.SnapshotPorts(
                    repository=repository,
                    mode=context_ports.snapshot_mode,
                    models=context_ports.models,
                    task_observation=context_ports.task_observation,
                )
                return fn(ports, chain_id, base_no, next_no)
            return bound_chain_snapshot
        if name == "existing_next_or_fail":
            def bound_existing_next(new, next_no, snapshot, repository):
                context = self._module.completion_preflight_context_ports_for(self._hook)
                ports = self._module.CompletionPreflightPorts(
                    preflight=context.preflight,
                    coerce_int=context.coerce_int,
                    max_link_number=context.max_link_number,
                    short_uuid=context.short_uuid,
                    panel=context.panel,
                    print_task=context.print_task,
                    end_chain_summary=context.end_chain_summary,
                    existing_next_lookup=lambda task, link: repository.exact_child_slot(
                        str(task.get("chainID") or ""), link
                    ),
                )
                return fn(ports, new, next_no, snapshot)
            return bound_existing_next
        return lambda *args, **kwargs: fn(self._hook, *args, **kwargs)

    def __setattr__(self, name, value):
        setattr(self._module, name, lambda _host, *args, **kwargs: value(*args, **kwargs))


class _BoundTransitionEffects:
    """Test-only bound view of the extracted transition-effects module."""

    def __init__(self, hook):
        object.__setattr__(self, "_hook", hook)
        object.__setattr__(self, "_module", importlib.import_module("nautical_core.modify_transition_effects"))

    def __getattr__(self, name):
        fn = getattr(self._module, name)
        if name in {
            "preserve_cp_relative_offsets_on_due_change",
            "preserve_native_until_on_target_change",
            "validate_completion_cp_and_anchor",
        }:
            def bound(*args, **kwargs):
                composition = self._hook._module("modify_composition")
                capabilities = composition.capabilities_for(self._hook)
                ports_for = {
                    "preserve_cp_relative_offsets_on_due_change": composition._cp_carry_ports,
                    "preserve_native_until_on_target_change": composition._native_preserve_ports,
                    "validate_completion_cp_and_anchor": composition._completion_validation_ports,
                }[name]
                return fn(ports_for(self._hook, capabilities), *args, **kwargs)
            return bound
        return lambda *args, **kwargs: fn(self._hook, *args, **kwargs)

    def __setattr__(self, name, value):
        setattr(self._module, name, lambda _host, *args, **kwargs: value(*args, **kwargs))


class _BoundPresentationEffects:
    """Test-only bound view of the extracted presentation-effects module."""

    _RENAMED = {
        "render_anchor_completion_feedback": "render_anchor_completion_feedback_for",
        "render_cp_completion_feedback": "render_cp_completion_feedback_for",
        "render_recurrence_updated_panel": "render_recurrence_updated_panel_for",
        "first_recurrence_target": "first_recurrence_target_for",
        "recurrence_enabled_rows": "recurrence_enabled_rows_for",
        "render_cp_schedule_adjusted_panel": "render_cp_schedule_adjusted_panel_for",
        "render_explicit_timing_order_warning": "render_explicit_timing_order_warning_for",
        "render_disabled_chain_summary": "render_disabled_chain_summary_for",
        "ensure_terminal_chain_off": "ensure_terminal_chain_off_for",
        "timeline_lines": "timeline_lines_for",
    }

    def __init__(self, hook):
        object.__setattr__(self, "_hook", hook)
        module = importlib.import_module("nautical_core.modify_composition_adapters")
        object.__setattr__(self, "_module", module)
        originals = getattr(type(self), "_originals", None)
        if originals is None:
            originals = {
                current_name: getattr(module, current_name)
                for current_name in (
                    "render_anchor_completion_feedback_for",
                    "render_cp_completion_feedback_for",
                    "render_recurrence_updated_panel_for",
                )
            }
            setattr(type(self), "_originals", originals)
        else:
            for name, fn in originals.items():
                setattr(module, name, fn)

    def __getattr__(self, name):
        fn = getattr(self._module, self._RENAMED.get(name, name))
        if name in {"render_anchor_completion_feedback", "render_cp_completion_feedback"}:
            def bound_feedback(*args, **kwargs):
                kwargs.setdefault("lifecycle_result", None)
                return fn(self._hook, request=SimpleNamespace(**kwargs))
            return bound_feedback
        return lambda *args, **kwargs: fn(self._hook, *args, **kwargs)

    def __setattr__(self, name, value):
        setattr(
            self._module,
            self._RENAMED.get(name, name),
            lambda _host, *args, **kwargs: value(*args, **kwargs),
        )


class _BoundDiagnosticsEffects:
    """Test-only bound view of the extracted diagnostics-effects module."""

    _PORT_FACTORIES = {
        "last_n_timeline": "timeline_summary_ports_for",
        "span_fields": "span_fields_ports_for",
        "end_chain_summary": "end_chain_summary_ports_for",
    }

    def __init__(self, hook):
        object.__setattr__(self, "_hook", hook)
        object.__setattr__(self, "_module", importlib.import_module("nautical_core.modify_diagnostics_effects"))

    def __getattr__(self, name):
        fn = getattr(self._module, name)
        factory = self._PORT_FACTORIES.get(name)
        if factory:
            ports = getattr(self._module, factory)(self._hook)
            return lambda *args, **kwargs: fn(ports, *args, **kwargs)
        return lambda *args, **kwargs: fn(self._hook, *args, **kwargs)

    def __setattr__(self, name, value):
        setattr(self._module, name, lambda _host, *args, **kwargs: value(*args, **kwargs))


def _assert_hook_requires_integration_context(hook_name: str, module_name: str):
    hook = _find_hook_file(hook_name)
    prev_core = os.environ.get("NAUTICAL_CORE_PATH")
    prev_argv = list(sys.argv)
    with tempfile.TemporaryDirectory() as td:
        fake_core = Path(td) / "nautical_core/__init__.py"
        fake_core.parent.mkdir(parents=True, exist_ok=True)
        fake_core.write_text(
            "def _warn_once_per_day_any(*_args, **_kwargs):\n"
            "    return None\n",
            encoding="utf-8",
        )
        os.environ["NAUTICAL_CORE_PATH"] = td
        sys.argv = [hook_name]
        try:
            try:
                _load_hook_module(hook, module_name)
                raise AssertionError("expected hook import to fail without core data resolver")
            except Exception as exc:
                expect(
                    "integration_context.py is required" in str(exc)
                    or "core resolver is unavailable" in str(exc)
                    or "required runtime port is unavailable" in str(exc),
                    f"unexpected error when context module is missing: {exc!r}",
                )
        finally:
            sys.argv = prev_argv
            if prev_core is None:
                os.environ.pop("NAUTICAL_CORE_PATH", None)
            else:
                os.environ["NAUTICAL_CORE_PATH"] = prev_core


def test_on_add_requires_integration_context_helper():
    """on-add should fail closed when the integration context is unavailable."""
    _assert_hook_requires_integration_context("on-add.nautical", "_nautical_on_add_requires_context_test")



def _test_modify_engine_services(
    result_cls,
    *,
    has_nautical_fields,
    load_core,
    diag,
    fail_and_exit,
    is_non_completion,
    handle_non_completion,
    handle_completion,
    handle_deleted,
):
    return SimpleNamespace(
        result=lambda task, *, sanitize: result_cls(task=task, sanitize=sanitize),
        has_nautical_fields=has_nautical_fields,
        load_core=load_core,
        diag=diag,
        fail_and_exit=fail_and_exit,
        is_non_completion=is_non_completion,
        handle_non_completion=handle_non_completion,
        handle_completion=handle_completion,
        handle_deleted=handle_deleted,
    )


def test_load_benchmark_installs_complete_hook_runtime():
    """The end-to-end benchmark must install on-exit and the Nautical UDAs."""
    load_test = _load_hook_module(
        os.path.join(DEV_TOOLS, "load_test_nautical.py"),
        "_nautical_load_test_runtime_test",
    )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_dir = root / "taskdata"
        hooks_dir = data_dir / "hooks"
        taskrc = root / "taskrc"
        config = root / "config-nautical.toml"
        data_dir.mkdir()
        load_test._install_hooks(hooks_dir)
        load_test._write_taskrc(taskrc, data_dir, hooks_dir)
        load_test._write_nautical_config(config)

        for hook_name in ("on-add", "on-modify", "on-exit"):
            hook = hooks_dir / hook_name
            expect(hook.is_file(), f"load benchmark did not install {hook_name}")
            expect(os.access(hook, os.X_OK), f"load benchmark hook is not executable: {hook_name}")
        taskrc_text = taskrc.read_text(encoding="utf-8")
        expect(f"hooks.location={hooks_dir}" in taskrc_text, f"missing hooks.location: {taskrc_text!r}")
        expect(f"include {Path(ROOT) / 'uda.conf'}" in taskrc_text, f"missing UDA include: {taskrc_text!r}")
        expect("verbose=nothing" not in taskrc_text, "benchmark must preserve task IDs in command output")
        expect('tz = "UTC"' in config.read_text(encoding="utf-8"), "benchmark config should be deterministic")


def test_load_benchmark_queue_and_lineage_verification():
    """Benchmark validation should detect active SQLite work and broken parent-child links."""
    load_test = _load_hook_module(
        os.path.join(DEV_TOOLS, "load_test_nautical.py"),
        "_nautical_load_test_validation_test",
    )
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        state_dir = data_dir / ".nautical-state"
        state_dir.mkdir()
        db_path = state_dir / ".nautical_queue.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "CREATE TABLE queue_entries (id INTEGER PRIMARY KEY, payload TEXT NOT NULL, state TEXT NOT NULL)"
            )
            conn.execute("INSERT INTO queue_entries(payload, state) VALUES (?, ?)", ('{"child":1}', "queued"))
            conn.execute("INSERT INTO queue_entries(payload, state) VALUES (?, ?)", ('{"child":2}', "done"))
        metrics = load_test._queue_metrics(data_dir)
        expect(metrics == {"items": 1, "bytes": len('{"child":1}')}, f"bad active queue metrics: {metrics!r}")

    parent_uuid = "11111111-0000-0000-0000-000000000001"
    child_uuid = "22222222-0000-0000-0000-000000000002"
    rows = [
        {
            "uuid": parent_uuid,
            "status": "completed",
            "chainID": "chain-a",
            "link": 1,
            "nextLink": "22222222",
        },
        {
            "uuid": child_uuid,
            "status": "pending",
            "chainID": "chain-a",
            "link": 2,
            "prevLink": "11111111",
        },
    ]
    valid = load_test._verify_link_rows(rows, [parent_uuid])
    expect(valid == {"expected": 1, "verified": 1, "failures": []}, f"valid lineage was rejected: {valid!r}")
    rows[1]["prevLink"] = "wrong"
    invalid = load_test._verify_link_rows(rows, [parent_uuid])
    expect(invalid.get("verified") == 0 and invalid.get("failures"), f"broken lineage was accepted: {invalid!r}")




def test_natural_interval_or_branches_keep_cadence_with_subject():
    """Interval OR branches should not begin with an awkward nested prefix."""
    expr = "(w/2:2rand@t=18:00 | w/3:thu@t=12:00)"
    expected = "either 2 random days every 2 weeks at 18:00 or Thursdays every 3 weeks at 12:00"
    expect(core.describe_anchor_expr(expr) == expected, f"unexpected interval OR natural text")
    expect(
        core.describe_anchor_dnf(core.validate_anchor_expr_strict(expr), {"anchor_mode": "skip"})
        == f"{expected}; skip missed anchors",
        "mode suffix should follow the polished interval OR text",
    )
    expect(
        core.describe_anchor_expr("w/2:mon") == "every 2 weeks: Mondays",
        "standalone interval wording should remain backward-compatible",
    )


def test_natural_compresses_repeated_within_variants():
    """describe_anchor_dnf should compact repeated OR terms that only vary by yearly 'within' token."""
    expr = (
        "(w:mon..wed) + (m:1..10) + "
        "(y:01-01|y:02-01|y:03-01|y:04-01|y:05-01|y:06-01|y:07-01|y:08-01|y:09-01|y:10-01)"
    )
    dnf = core.validate_anchor_expr_strict(expr)
    nat = core.describe_anchor_dnf(dnf, {"anchor_mode": "skip"})
    low = (nat or "").lower()

    assert "and within either " in low, f"Natural should compact yearly variants: {nat!r}"
    assert low.count("mondays through wednesdays") == 1, f"Natural repeats shared prefix: {nat!r}"
    assert "jan 1" in low and "oct 1" in low, f"Natural lost yearly endpoints: {nat!r}"
    assert "skip missed anchors" in low, f"Natural should include mode tail: {nat!r}"

def test_natural_compresses_repeated_fall_on_variants():
    """describe_anchor_dnf should compact repeated OR terms that only vary by monthly 'that fall on' token."""
    expr = "(w:mon) + (m:1|m:2|m:3)"
    dnf = core.validate_anchor_expr_strict(expr)
    nat = core.describe_anchor_dnf(dnf, {"anchor_mode": "skip"})
    low = (nat or "").lower()

    assert "that fall on either " in low, f"Natural should compact monthly variants: {nat!r}"
    assert low.count("mondays") == 1, f"Natural repeats shared prefix: {nat!r}"
    assert "the 1st" in low and "the 3rd day of each month" in low, f"Natural lost monthly endpoints: {nat!r}"
    assert "skip missed anchors" in low, f"Natural should include mode tail: {nat!r}"

def test_prev_weekday_natural_text():
    """
    Natural for 'm:-1@prev-fri' should mention 'previous Friday before the last day of the month'
    """
    nat = _must_natural("m:-1@prev-fri")
    want_any = [
        "previous Friday before the last day of the month",
        "previous Friday before the last day",
        "previous Friday before month end",
    ]
    assert any(w in nat for w in want_any), f"Natural missing expected phrasing: {nat!r}"


def test_modifier_boundary_paths_agree_and_advance_strictly():
    """Rolled and shifted anchors should agree across preview, completion, timeline, and omit."""
    import nautical_core.anchor_omit as anchor_omit
    import nautical_core.modify_timeline as modify_timeline

    add_mod = _load_hook_module(_find_hook_file("on-add.nautical"), "_nautical_modifier_boundary_add_test")
    modify_mod = _load_hook_module(_find_hook_file("on-modify.nautical"), "_nautical_modifier_boundary_modify_test")
    if hasattr(add_mod, "_load_core"):
        add_mod._load_core()
    if hasattr(modify_mod, "_load_core"):
        modify_mod._load_core()

    cases = [
        ("y:04-25@pbd@t=09:00", "y:04-25@pbd", date(2026, 4, 24), date(2027, 4, 23)),
        ("y:04-25@nbd@t=09:00", "y:04-25@nbd", date(2026, 4, 27), date(2027, 4, 26)),
        ("y:04-25@nbd@t=12:00,17:00", "y:04-25@nbd", date(2026, 4, 27), date(2026, 4, 27)),
        ("y:01-31@+1d@t=09:00", "y:01-31@+1d", date(2026, 2, 1), date(2027, 2, 1)),
        ("y:03-01@-1d@t=09:00", "y:03-01@-1d", date(2026, 2, 28), date(2027, 2, 28)),
        ("y:02-29@+1d@t=09:00", "y:02-29@+1d", date(2028, 3, 1), date(2032, 3, 1)),
        ("y:04-24@+1bd@t=09:00", "y:04-24@+1bd", date(2026, 4, 27), date(2027, 4, 26)),
        ("w:sun@t=09:00", "w:sun", date(2026, 3, 22), date(2026, 3, 29)),
    ]
    chain_id = "modifier-boundary"
    def next_preview(dnf, current_local, fallback_hhmm, interval_seed):
        return add_mod._module("add_anchor_compute").anchor_next_occurrence_after_local_dt(
            dnf,
            current_local,
            fallback_hhmm,
            interval_seed,
            chain_id,
            core=add_mod.core,
            norm_t_mod=add_mod._norm_t_mod,
            resolve_time_slots=add_mod._resolve_time_slots,
        )

    for expr, omit_expr, current_day, expected_next_day in cases:
        dnf = add_mod.core.validate_anchor_expr_strict(expr)
        omit_dnf = anchor_omit.validate_omit_expr_strict(
            omit_expr,
            validate_anchor_expr_cached=add_mod.core.validate_anchor_expr_strict,
        )
        first_slot = (12, 0) if "12:00,17:00" in expr else (9, 0)
        current_utc = add_mod.core.build_local_datetime(current_day, first_slot).astimezone(timezone.utc)
        current_local = add_mod.core.to_local(current_utc)

        preview_next = next_preview(dnf, current_local, first_slot, current_day)
        expect(preview_next is not None, f"{expr}: preview did not find the next occurrence")
        expect(preview_next > current_local, f"{expr}: preview did not advance strictly: {preview_next}")
        expect(preview_next.date() == expected_next_day, f"{expr}: unexpected preview date {preview_next.date()}")
        expected_hhmm = (17, 0) if "12:00,17:00" in expr else first_slot
        expect((preview_next.hour, preview_next.minute) == expected_hhmm, f"{expr}: preview lost wall-clock time")

        parent = {
            "uuid": "00000000-0000-4000-8000-000000000901",
            "description": "modifier boundary fixture",
            "status": "completed",
            "anchor": expr,
            "anchor_mode": "skip",
            "due": modify_mod.core.fmt_isoz(current_utc),
            "end": modify_mod.core.fmt_isoz(current_utc),
            "chainID": chain_id,
            "link": 1,
        }
        child_due, _meta, completion_dnf = _compute_anchor_child_due(modify_mod, parent)
        completion_next = modify_mod.core.to_local(child_due)
        expect(completion_next == preview_next, f"{expr}: completion {completion_next} != preview {preview_next}")

        preview_after_child = next_preview(dnf, preview_next, expected_hhmm, current_day)
        _evaluator_callback, scheduler_service_for_task = modify_mod._module("modify_schedule_effects").scheduler_callbacks(
            modify_mod._module("modify_schedule_effects").scheduler_ports_for(modify_mod)
        )
        timeline_items = modify_timeline._timeline_future_anchor_items(
            parent,
            completion_dnf,
            child_due,
            start_no=2,
            allowed_future=1,
            cap_no=None,
            to_local_cached=modify_mod._to_local_cached,
            safe_parse_datetime=modify_mod._TASK_DATETIME_PARSER.parse,
            scheduler_service=scheduler_service_for_task(parent),
            omit_dnf=None,
            omit_description_for_date=None,
            max_iterations=32,
        )
        expect(len(timeline_items) == 1, f"{expr}: timeline did not produce one future occurrence")
        timeline_next = modify_mod.core.to_local(timeline_items[0][1])
        expect(timeline_next == preview_after_child, f"{expr}: timeline {timeline_next} != preview {preview_after_child}")

        expect(
            anchor_omit.omit_expr_fires_on_date(
                omit_dnf,
                current_day,
                current_day - timedelta(days=10),
                chain_id,
                core=add_mod.core,
            ),
            f"{omit_expr}: omit did not recognize the effective rolled/shifted date",
        )

    dst_dnf = add_mod.core.validate_anchor_expr_strict("w:sun@t=09:00")
    before_dst = add_mod.core.to_local(add_mod.core.build_local_datetime(date(2026, 3, 22), (9, 0)))
    after_dst = next_preview(dst_dnf, before_dst, (9, 0), date(2026, 3, 22))
    expect((after_dst.hour, after_dst.minute) == (9, 0), f"DST transition changed anchor wall clock: {after_dst}")


# -------- Runner --------------------------------------------------------------

def test_hook_on_modify_cp_malformed_inputs_fail_with_parser_guidance():
    """on-modify completion should surface parser-specific guidance for malformed cp strings."""
    hook = _find_hook_file("on-modify.nautical")
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
        p = _run_hook_script_raw(hook, raw, env_extra=env)
        expect(p.returncode != 0, f"on-modify should fail for malformed cp {cp_value!r}")
        expect((p.stdout or "").strip() == "", f"expected no stdout on malformed cp modify failure, got: {p.stdout!r}")
        stderr_txt = _strip_markup(p.stderr)
        expect("Invalid CP" in stderr_txt, f"expected Invalid CP panel for {cp_value!r}: {stderr_txt[:500]!r}")
        for part in expected_parts:
            expect(part in stderr_txt, f"expected parser guidance fragment {part!r} for {cp_value!r}: {stderr_txt[:500]!r}")


















def test_navigator_uses_anchor_and_anchor_file_sources():
    """Navigator anchor helpers should summarize and merge anchor sources from anchor + anchor_file."""
    module_name = "_nautical_navigator_anchor_sources_test"
    loader = importlib.machinery.SourceFileLoader(module_name, os.path.join(ROOT, "nautical_navigator.py"))
    spec = importlib.util.spec_from_loader(module_name, loader)
    navigator = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = navigator
    try:
        loader.exec_module(navigator)
    finally:
        sys.modules.pop(module_name, None)

    with tempfile.TemporaryDirectory() as td:
        anchor_dir = Path(td)
        (anchor_dir / "calendar.csv").write_text(
            "date,description\n"
            "2026-04-14,Midweek check\n"
            "2026-04-25,Month-end check\n",
            encoding="utf-8",
        )
        old_dir = getattr(navigator.core, "ANCHOR_FILE_DIR", "")
        old_core_dir = navigator.core._core_config.ANCHOR_FILE_DIR
        old_facade_config_synced = navigator.core._FACADE_CONFIG_SYNCED
        navigator.core.ANCHOR_FILE_DIR = str(anchor_dir)
        # This test intentionally overrides the lazy facade's configured
        # source; mark the override as synchronized before scheduler access.
        navigator.core._FACADE_CONFIG_SYNCED = True
        try:
            analyzer = navigator.TaskAnalyzer()
            navigator.core.ANCHOR_FILE_DIR = str(anchor_dir)
            navigator.core._core_config.ANCHOR_FILE_DIR = str(anchor_dir)
            task = {
                "uuid": "00000000-0000-4000-8000-000000000900",
                "chainID": "navigator-anchor-sources",
                "status": "pending",
                "link": 1,
                "description": "combined navigator anchor",
                "anchor": "w:fri@t=09:00",
                "anchor_file": "calendar.csv@t=12:00",
                "due": "2026-04-11T09:00:00Z",
            }

            expect(
                analyzer._anchor_summary(task) == ("Sources", "anchor + anchor_file"),
                f"unexpected anchor summary: {analyzer._anchor_summary(task)!r}",
            )

            projected = analyzer._project_anchor_dates(task, limit=4, start_from_date=date(2026, 4, 11))
            projected_txt = [(item.date().isoformat(), item.strftime("%H:%M")) for item in projected]
            expect(
                projected_txt[:3] == [("2026-04-14", "12:00"), ("2026-04-17", "09:00"), ("2026-04-24", "09:00")],
                f"unexpected merged projection order: {projected_txt!r}",
            )
            expect(
                analyzer._due_is_anchor_day("2026-04-14T12:00:00Z", task) is True,
                "anchor_file due date should count as an anchor day",
            )
        finally:
            navigator.core.ANCHOR_FILE_DIR = old_dir
            navigator.core._core_config.ANCHOR_FILE_DIR = old_core_dir
            navigator.core._FACADE_CONFIG_SYNCED = old_facade_config_synced


def test_navigator_surfaces_configuration_drift_warning():
    """Navigator should visibly warn about stale configuration before forecasting."""
    module_name = "_nautical_navigator_config_drift_test"
    loader = importlib.machinery.SourceFileLoader(module_name, os.path.join(ROOT, "nautical_navigator.py"))
    spec = importlib.util.spec_from_loader(module_name, loader)
    navigator = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = navigator
    loader.exec_module(navigator)
    original_drift = navigator.core.configuration_drift
    original_print = navigator.console.print
    printed = []
    try:
        navigator.TaskAnalyzer.convert_to_local("20260731T090000Z")
        analyzer = navigator.TaskAnalyzer()
        analyzer._task_cache[1] = {"uuid": "stale"}
        analyzer._uuid_cache["stale"] = analyzer._task_cache[1]
        analyzer._children["stale"] = [analyzer._task_cache[1]]
        expect(
            navigator.TaskAnalyzer.convert_to_local.cache_info().currsize > 0,
            "test did not seed Navigator's conversion cache",
        )
        navigator.core.configuration_drift = lambda: {"changed": True, "source": "/tmp/config-nautical.toml"}
        navigator.console.print = printed.append
        expect(navigator._show_config_drift_warning(), "drift warning was not emitted")
        expect(
            navigator.TaskAnalyzer.convert_to_local.cache_info().currsize == 0,
            "configuration drift left stale conversion cache entries",
        )
        expect(not analyzer._task_cache and not analyzer._uuid_cache and not analyzer._children,
               "configuration drift left stale analyzer indexes")
        expect(
            printed and getattr(printed[0], "title", "") == "⚠ Configuration changed",
            f"unexpected drift warning: {printed!r}",
        )
        printed.clear()
        navigator.core.configuration_drift = lambda: {"changed": False, "status": "ok"}
        expect(not navigator._show_config_drift_warning(), "clean config should not emit a warning")
        expect(not printed, f"clean config emitted output: {printed!r}")
    finally:
        navigator.core.configuration_drift = original_drift
        navigator.console.print = original_print
        sys.modules.pop(module_name, None)


def test_navigator_reloads_validated_taskdata_configuration():
    """Navigator must construct and retain the shared validated context."""
    module_name = "_nautical_navigator_config_reload_test"
    loader = importlib.machinery.SourceFileLoader(module_name, os.path.join(ROOT, "nautical_navigator.py"))
    spec = importlib.util.spec_from_loader(module_name, loader)
    navigator = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = navigator
    loader.exec_module(navigator)
    old_builder = navigator.build_operator_uow
    old_taskdata = os.environ.get("TASKDATA")
    calls: list[dict] = []
    try:
        with tempfile.TemporaryDirectory() as td:
            os.environ["TASKDATA"] = td
            expected_context = SimpleNamespace(
                taskdata=Path(td).resolve(),
                local_timezone=timezone.utc,
                command_prefix=("task",),
            )
            expected_uow = SimpleNamespace(context=expected_context)

            def build_uow(**kwargs):
                calls.append(kwargs)
                return expected_uow

            navigator.build_operator_uow = build_uow
            navigator._reload_navigator_configuration()
            expect(len(calls) == 1, f"Navigator built its unit of work repeatedly: {calls!r}")
            expect(navigator._UNIT_OF_WORK is expected_uow, "Navigator discarded its unit of work")
            expect(navigator.LOCAL_ZONE is timezone.utc, "Navigator ignored the context timezone")

            navigator.build_operator_uow = lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("invalid discovered TOML")
            )
            try:
                navigator._reload_navigator_configuration()
            except RuntimeError as exc:
                expect("invalid discovered TOML" in str(exc), f"reload detail was lost: {exc}")
            else:
                raise AssertionError("Navigator accepted a failed configuration reload")
    finally:
        if old_taskdata is None:
            os.environ.pop("TASKDATA", None)
        else:
            os.environ["TASKDATA"] = old_taskdata
        navigator.build_operator_uow = old_builder
        sys.modules.pop(module_name, None)


def test_navigator_fallback_export_uses_empty_filter():
    """Navigator's broad fallback must use Taskwarrior's valid empty filter."""
    from dataclasses import replace
    from nautical_core.integration_context import IntegrationAccess

    module_name = "_nautical_navigator_export_fallback_test"
    loader = importlib.machinery.SourceFileLoader(module_name, os.path.join(ROOT, "nautical_navigator.py"))
    spec = importlib.util.spec_from_loader(module_name, loader)
    navigator = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = navigator
    calls = []
    try:
        loader.exec_module(navigator)
        uow = _test_operator_uow()
        uow.context = replace(uow.context, access=IntegrationAccess.READ_ONLY)

        class Client:
            def execute(self, args, *, purpose, timeout, **_kwargs):
                calls.append(list(args))
                rows = []
                if "chainID.not:" not in args:
                    rows = [{"id": 1, "uuid": "u1", "chainID": "c1", "link": 1, "status": "pending"}]
                return _typed_command_result(("task", *args), True, json.dumps(rows))

        uow.client = Client()
        navigator._UNIT_OF_WORK = uow

        tasks = navigator.TaskAnalyzer().get_all_chained_tasks()
        expect(len(tasks) == 1 and tasks[0].get("chainID") == "c1", f"fallback export failed: {tasks!r}")
        expect(any("chainID.not:" not in call for call in calls), f"missing empty-filter export: {calls!r}")
        expect(not any("all" in call for call in calls), f"invalid Taskwarrior 'all' filter remains: {calls!r}")
    finally:
        sys.modules.pop(module_name, None)


def test_shared_time_slot_resolver_keeps_hook_and_navigator_parity():
    """add, modify, and Navigator should resolve the same symbolic slot and offset."""
    import nautical_core.time_slots as time_slots

    add_mod = _load_hook_module(_find_hook_file("on-add.nautical"), "_nautical_add_time_slots_parity_test")
    modify_mod = _load_hook_module(_find_hook_file("on-modify.nautical"), "_nautical_modify_time_slots_parity_test")
    original_event = time_slots.astronomy.resolve_event
    original_to_local = core.to_local
    original_add_to_local = add_mod.core.to_local
    original_modify_to_local = modify_mod.core.to_local
    try:
        time_slots.astronomy.resolve_event = lambda *_args, **_kwargs: datetime(2026, 7, 6, 18, 0, tzinfo=timezone.utc)
        core.to_local = lambda value: value
        add_mod.core.to_local = lambda value: value
        modify_mod.core.to_local = lambda value: value
        value = {"t": "sunset", "time_offset_minutes": 45}
        expected = [(18, 45)]
        expect(time_slots.resolve_time_slots(value, date(2026, 7, 6), to_local=core.to_local) == expected, "shared resolver drifted")
        expect(add_mod._resolve_time_slots(value, date(2026, 7, 6)) == expected, "on-add resolver drifted")
        modify_time = modify_mod._module("modify_time_effects")
        expect(
            modify_time.normalize_hhmm_list(
                modify_time.time_slot_ports_for(modify_mod), value, date(2026, 7, 6)
            ) == expected,
            "on-modify resolver drifted",
        )
    finally:
        time_slots.astronomy.resolve_event = original_event
        core.to_local = original_to_local
        add_mod.core.to_local = original_add_to_local
        modify_mod.core.to_local = original_modify_to_local


def test_navigator_projects_all_slots_in_a_time_window():
    """Navigator analysis should retain every same-day slot in a bounded window."""
    module_name = "_nautical_navigator_time_window_projection_test"
    loader = importlib.machinery.SourceFileLoader(module_name, os.path.join(ROOT, "nautical_navigator.py"))
    spec = importlib.util.spec_from_loader(module_name, loader)
    navigator = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = navigator
    old_tz_name = core.LOCAL_TZ_NAME
    old_tz = core._LOCAL_TZ
    try:
        loader.exec_module(navigator)
        core.LOCAL_TZ_NAME = "Etc/GMT-3"
        core._LOCAL_TZ = timezone(timedelta(hours=3))
        navigator.LOCAL_ZONE = core._LOCAL_TZ
        analyzer = navigator.TaskAnalyzer()
        dates = analyzer._project_anchor_dates(
            {"anchor": "w:mon..sun@t=04:30..19:30/3h30min", "uuid": "00000000-0000-4000-8000-000000000910", "chainID": "navigator-window", "status": "pending", "link": 1},
            limit=5,
            start_from_date=date(2026, 8, 2),
        )
        expect(
            [item.strftime("%H:%M") for item in dates] == ["04:30", "08:00", "11:30", "15:00", "18:30"],
            f"Navigator collapsed same-day time-window slots: {dates!r}",
        )
        same_day_task = {
            "anchor": "w:mon@t=06:00,12:00,18:00",
            "uuid": "00000000-0000-4000-8000-000000000911",
            "chainID": "navigator-same-day",
            "status": "pending",
            "link": 1,
            "due": "20260803T080000Z",
        }
        same_day = analyzer._project_anchor_dates(same_day_task, limit=2, start_from_date=date(2026, 8, 3))
        expect(
            [item.strftime("%H:%M") for item in same_day] == ["12:00", "18:00"],
            f"Navigator skipped later same-day slots: {same_day!r}",
        )
        repeated_a = analyzer._project_anchor_dates(same_day_task, limit=2, start_from_date=date(2026, 8, 3))
        repeated_b = analyzer._project_anchor_dates(same_day_task, limit=2, start_from_date=date(2026, 8, 3))
        expect(repeated_a == repeated_b, "Navigator projection changed across identical queries")
        partitioned = analyzer._project_anchor_dates(
            {"anchor": "w:mon..sun@t=04:30..19:30/3", "uuid": "00000000-0000-4000-8000-000000000912", "chainID": "navigator-partition", "status": "pending", "link": 1},
            limit=3,
            start_from_date=date(2026, 8, 2),
        )
        expect(
            [item.strftime("%H:%M") for item in partitioned] == ["04:30", "12:00", "19:30"],
            f"Navigator did not retain evenly partitioned slots: {partitioned!r}",
        )
        random_uuid = "00000000-0000-4000-8000-000000000913"
        random_dates = analyzer._project_anchor_dates(
            {"anchor": "w:mon@t=rand(06..18/3)", "uuid": random_uuid, "chainID": random_uuid, "status": "pending", "link": 1},
            limit=3,
            start_from_date=date(2026, 8, 2),
        )
        random_window = core._import_sibling("time_windows").parse_random_time_window_spec("rand(06..18/3)")
        expected_random = random_window.slots_with_offsets(f"{random_uuid}/2026-08-03")
        expect(
            [(item.hour, item.minute) for item in random_dates]
            == [(slot[1], slot[2]) for slot in expected_random],
            f"Navigator did not reuse deterministic random slots: {random_dates!r}",
        )
        overnight = analyzer._project_anchor_dates(
            {"anchor": "w:mon@t=22:30..06:30/7", "uuid": "00000000-0000-4000-8000-000000000914", "chainID": "navigator-overnight", "status": "pending", "link": 1},
            limit=7,
            start_from_date=date(2026, 8, 2),
        )
        expect(
            [(item.date().isoformat(), item.strftime("%H:%M")) for item in overnight] == [
                ("2026-08-03", "22:30"),
                ("2026-08-03", "23:50"),
                ("2026-08-04", "01:10"),
                ("2026-08-04", "02:30"),
                ("2026-08-04", "03:50"),
                ("2026-08-04", "05:10"),
                ("2026-08-04", "06:30"),
            ],
            f"Navigator misplaced overnight slots: {overnight!r}",
        )
        _natural, preview = navigator._anchor_preview("w:mon..sun@t=04:30..19:30/3h30min", count=5)
        expect(
            [item.rsplit(" ", 2)[-2] for item in preview] == ["04:30", "08:00", "11:30", "15:00", "18:30"],
            f"Navigator preview did not resolve all time-window slots: {preview!r}",
        )
        overnight_natural, _overnight_preview = navigator._anchor_preview("w:mon@t=22:30..06:30/7", count=3)
        expect("next day" in (overnight_natural or "").lower(), f"Navigator omitted overnight ownership wording: {overnight_natural!r}")
    finally:
        core.LOCAL_TZ_NAME = old_tz_name
        core._LOCAL_TZ = old_tz
        sys.modules.pop(module_name, None)




def test_recurrence_evaluator_owns_context_spec_and_timezone_boundary():
    """The evaluator should normalize recurrence state without performing I/O."""
    from zoneinfo import ZoneInfo
    from nautical_core.recurrence_context import RecurrenceContext

    task = {
        "chainID": "evaluator-chain",
        "anchor": "w:mon@t=02:30",
        "anchor_mode": "SKIP",
        "chainMax": "4",
    }
    context = RecurrenceContext(
        chain_id="evaluator-chain",
        timezone=ZoneInfo("America/New_York"),
        anchor_file_dir="/tmp/evaluator-anchor-files",
    )
    evaluator = _evaluator_for_fixture(task, context=context)
    expect(evaluator.chain_id == "evaluator-chain", "evaluator lost chain identity")
    expect(evaluator.seed_base == "evaluator-chain", "evaluator seed identity changed")
    expect(evaluator.kind == "anchor" and evaluator.enabled, "evaluator kind was not normalized")


    expect(evaluator.spec.anchor_mode == "skip" and evaluator.spec.chain_max == 4, "evaluator spec was not normalized")

    shifted = evaluator.build_local_datetime(date(2025, 3, 9), (2, 30))
    shifted_local = evaluator.to_local(shifted)
    expect(
        (shifted_local.hour, shifted_local.minute) == (3, 30),
        f"evaluator bypassed the shared DST policy: {shifted_local}",
    )
    expect(
        evaluator.utc_to_local_naive(shifted) == datetime(2025, 3, 9, 3, 30),
        "evaluator local-naive conversion disagreed with its timezone",
    )

    parsed = _evaluator_for_fixture(
        {
            "chainID": "parsed-chain",
            "anchor": "w:mon@t=02:30",
            "omit": "w:sun",
            "anchor_mode": "ALL",
            "chainMax": "4",
            "chainUntil": "2025-12-31T23:00:00Z",
        },
        timezone=timezone.utc,
    )
    expect(parsed.anchor_mode == "all", "evaluator did not normalize anchor mode")
    parsed_anchor = parsed.anchor_dnf
    parsed_omit = parsed.omit_dnf
    expect(len(parsed_anchor) == 1 and len(parsed_anchor[0]) == 1, "anchor parsing was not owned by evaluator")
    expect(len(parsed_omit) == 1, "omit parsing was not owned by evaluator")
    expect(parsed.anchor_dnf is parsed_anchor, "anchor DNF was copied again after evaluation")
    expect(parsed.omit_dnf is parsed_omit, "omit DNF was copied again after evaluation")
    try:
        parsed_anchor[0][0]["spec"] = "corrupted"
    except TypeError:
        pass
    else:
        raise AssertionError("evaluator anchor DNF remained mutable")
    try:
        parsed_anchor.clear()
    except TypeError:
        pass
    else:
        raise AssertionError("evaluator anchor DNF list remained mutable")
    expect(parsed.limits.chain_max == 4, "chainMax limit was not normalized")
    expect(
        parsed.limits.chain_until == datetime(2025, 12, 31, 23, 0, tzinfo=timezone.utc),
        "chainUntil limit was not parsed as UTC",
    )
    cp_evaluator = _evaluator_for_fixture(
        {"chainID": "cp-chain", "cp": "1d,rand(2d..3d)"}
    )
    cp_tokens = cp_evaluator.cp_tokens
    expect(len(cp_tokens or []) == 2, "CP token parsing was not owned by evaluator")
    try:
        if cp_tokens is not None:
            cp_tokens[0]["kind"] = "corrupted"
    except TypeError:
        pass
    else:
        raise AssertionError("evaluator CP tokens remained mutable")
    expect(
        cp_evaluator.cp_interval_for_link(1) == timedelta(days=1),
        "fixed CP interval projection changed",
    )
    random_interval = cp_evaluator.cp_interval_for_link(2)
    expect(
        random_interval is not None and timedelta(days=2) <= random_interval <= timedelta(days=3),
        f"random CP interval was not projected through chain identity: {random_interval!r}",
    )
    expect(
        cp_evaluator.project_cp(datetime(2025, 1, 1, tzinfo=timezone.utc), 1)
        == datetime(2025, 1, 2, tzinfo=timezone.utc),
        "CP due-date projection changed",
    )
    file_evaluator = _evaluator_for_fixture(
        {"chainID": "file-chain", "anchor_file": "events.csv"}
    )
    expect(
        file_evaluator._anchor_file_provider_for((9, 0))
        is file_evaluator._anchor_file_provider_for((9, 0)),
        "evaluator rebuilt the anchor-file provider within one session",
    )
    try:
        cp_evaluator.cp_interval_for_link(0)
    except ValueError as exc:
        expect("positive link" in str(exc), f"invalid CP link error was not actionable: {exc}")
    else:
        raise AssertionError("invalid CP link number was silently accepted")
    try:
        cp_evaluator.project_cp(datetime(2025, 1, 1), 1)
    except ValueError as exc:
        expect("timezone-aware" in str(exc), f"naive CP projection error was not actionable: {exc}")
    else:
        raise AssertionError("naive CP projection was silently accepted")
    expect(parsed.limits_allow(datetime(2025, 12, 31, 22, 0), 4), "valid chain limit was rejected")
    expect(not parsed.limits_allow(datetime(2026, 1, 1, 0, 0), 4), "chainUntil limit was ignored")
    expect(not parsed.limits_allow(datetime(2025, 12, 31, 22, 0), 5), "chainMax limit was ignored")

    def next_day(_dnf, after_local, **_kwargs):
        return after_local + timedelta(days=1)

    stream = parsed.collect_after(
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        limit=2,
    )
    expect(
        [item.local_datetime for item in stream]
        == [
            datetime(2025, 1, 6, 2, 30, tzinfo=timezone.utc),
            datetime(2025, 1, 13, 2, 30, tzinfo=timezone.utc),
        ],
        "evaluator did not expose a merged typed occurrence stream",
    )

    event_evaluator = _evaluator_for_fixture(
        {"chainID": "event-chain", "anchor": "w:mon..tue", "omit": "w:mon"},
        timezone=timezone.utc,
    )
    omitted_event = event_evaluator.next_event_after(
        datetime(2025, 1, 5, tzinfo=timezone.utc),
        include_omitted=True,
    )
    expect(
        omitted_event is not None and omitted_event.omitted
        and omitted_event.local_datetime == datetime(2025, 1, 6, 9, 0, tzinfo=timezone.utc),
        f"event stream did not retain omitted occurrence: {omitted_event!r}",
    )
    included_event = event_evaluator.next_event_after(
        datetime(2025, 1, 5, tzinfo=timezone.utc),
    )
    expect(
        included_event is not None and not included_event.omitted
        and included_event.local_datetime == datetime(2025, 1, 7, 9, 0, tzinfo=timezone.utc),
        f"event stream did not skip omitted occurrence by default: {included_event!r}",
    )
    ranged_events = event_evaluator.events_between(
        datetime(2025, 1, 5, tzinfo=timezone.utc),
        datetime(2025, 1, 20, tzinfo=timezone.utc),
        limit=1,
        inclusive=False,
        include_omitted=True,
    )
    expect(
        [item.local_datetime for item in ranged_events]
        == [
            datetime(2025, 1, 6, 9, 0, tzinfo=timezone.utc),
            datetime(2025, 1, 7, 9, 0, tzinfo=timezone.utc),
        ],
        f"bounded event stream did not retain omitted events while counting included ones: {ranged_events!r}",
    )

    mode_evaluator = _evaluator_for_fixture(
        {"chainID": "mode-chain", "anchor": "w:mon"},
        timezone=timezone.utc,
    )
    mode_common = {
        "due_local": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "end_local": datetime(2025, 1, 3, tzinfo=timezone.utc),
    }
    all_result = mode_evaluator.select_mode("all", **mode_common)
    skip_result = mode_evaluator.select_mode("skip", **mode_common)
    flex_result = mode_evaluator.select_mode("flex", **mode_common)
    expect(
        all_result.selected_occurrence == datetime(2025, 1, 6, 9, 0, tzinfo=timezone.utc)
        and all_result.basis == "after_due"
        and all_result.source == "anchor",
        f"all mode policy selected the wrong typed result: {all_result!r}",
    )
    expect(
        skip_result.selected_occurrence == datetime(2025, 1, 6, 9, 0, tzinfo=timezone.utc)
        and skip_result.basis == "after_end",
        f"skip mode policy selected the wrong typed result: {skip_result!r}",
    )
    expect(
        flex_result.selected_occurrence == datetime(2025, 1, 6, 9, 0, tzinfo=timezone.utc)
        and flex_result.basis == "flex"
        and flex_result.missed_count == 0,
        f"flex mode policy lost missed evidence: {flex_result!r}",
    )
    try:
        event_evaluator.events_between(
            datetime(2025, 1, 5, tzinfo=timezone.utc),
            datetime(2025, 1, 10, tzinfo=timezone.utc),
            limit=2,
            max_iterations=1,
        )
    except ValueError as exc:
        expect("range iteration limit" in str(exc), f"unexpected range guard error: {exc}")
    else:
        raise AssertionError("evaluator silently truncated an iteration-bounded range")

    limited = _evaluator_for_fixture(
        {
            "chainID": "limited-chain",
            "anchor": "w:mon",
            "chainMax": "2",
            "chainUntil": "2025-01-10T00:00:00Z",
        }
    )
    expect(limited.limits_allow(datetime(2025, 1, 9), 2), "valid evaluator chain limit was rejected")
    expect(not limited.limits_allow(datetime(2025, 1, 11), 2), "chainUntil limit was ignored by evaluator")
    expect(not limited.limits_allow(datetime(2025, 1, 9), 3), "chainMax limit was ignored by evaluator")

    astronomy_evaluator = _evaluator_for_fixture(
        {"chainID": "astronomy-chain", "anchor": "moon:full"}
    )
    try:
        parsed.collect_after(
            datetime(2025, 1, 1),
            limit=1,
            fallback_hhmm=(24, 0),
        )
    except ValueError as exc:
        expect("Fallback occurrence time" in str(exc), f"invalid fallback time error was not actionable: {exc}")
    else:
        raise AssertionError("invalid fallback occurrence time was silently accepted")

    astronomy = core._import_sibling("astronomy")
    original_resolve_event = astronomy.resolve_event
    astronomy.resolve_event = lambda _event, day, config=None: datetime(
        day.year, day.month, day.day, 6, 30, tzinfo=timezone.utc
    )
    try:
        astronomical_time = _evaluator_for_fixture(
            {
                "chainID": "evaluator-astronomy-time",
                "anchor": "w:mon@t=sunrise",
            },
            timezone=timezone.utc,
            astronomy_config={"default_location": "test"},
        )
        next_event = astronomical_time.next_after(
            datetime(2025, 1, 6, 7, 0, tzinfo=timezone.utc)
        )
        expect(
            next_event is not None
            and next_event.local_datetime is not None
            and next_event.local_datetime.date() == date(2025, 1, 13)
            and (next_event.local_datetime.hour, next_event.local_datetime.minute) == (6, 30),
            f"evaluator scheduler did not preserve astronomical time context: {next_event!r}",
        )
    finally:
        astronomy.resolve_event = original_resolve_event

    try:
        _evaluator_for_fixture(
            {"chainID": "invalid-mode", "anchor": "w:mon", "anchor_mode": "bad"}
        )
    except ValueError as exc:
        expect("anchor_mode" in str(exc), f"invalid mode error was not actionable: {exc}")
    else:
        raise AssertionError("invalid anchor mode was silently accepted")

    try:
        _evaluator_for_fixture({"anchor": "w:mon"})
    except ValueError as exc:
        expect("chain ID" in str(exc), f"missing-chain failure was not actionable: {exc}")
    else:
        raise AssertionError("evaluator silently invented a chain identity")


def test_navigator_reads_through_read_only_invocation_repository():
    """Navigator must use the typed read snapshot and reject mutation-capable UOWs."""
    from dataclasses import replace
    from nautical_core.integration_context import IntegrationAccess

    module_name = "_nautical_navigator_read_boundary_test"
    loader = importlib.machinery.SourceFileLoader(module_name, os.path.join(ROOT, "nautical_navigator.py"))
    spec = importlib.util.spec_from_loader(module_name, loader)
    navigator = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = navigator
    try:
        loader.exec_module(navigator)
        uow = _test_operator_uow()
        uow.context = replace(uow.context, access=IntegrationAccess.READ_ONLY)

        uow.context = replace(uow.context, access=IntegrationAccess.MUTATION)
        navigator._UNIT_OF_WORK = uow
        try:
            navigator._run_chain_snapshot()
        except RuntimeError as exc:
            expect("read-only" in str(exc), f"mutation boundary error was unclear: {exc}")
        else:
            raise AssertionError("Navigator accepted a mutation-capable integration context")

        class Client:
            def execute(self, args, *, purpose, timeout, **kwargs):
                del purpose, timeout, kwargs
                return _typed_command_result(("task", *args), True, "")

        uow.client = Client()
        uow.context = replace(uow.context, access=IntegrationAccess.READ_ONLY)
        snapshot = navigator._run_chain_snapshot()
        expect(
            isinstance(snapshot, navigator.NavigatorSnapshot) and not snapshot.rows,
            f"empty shared snapshot was not preserved: {snapshot!r}",
        )
    finally:
        navigator._UNIT_OF_WORK = None
        sys.modules.pop(module_name, None)



def test_on_modify_panel_fallback():
    """on-modify panel should fall back to plain output on errors."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_panel_fallback_test")
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
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_live_duration_test")
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


def test_ui_live_test_term_guard_restores_environment():
    """Terminal-sensitive tests must not leak TERM changes into later cases."""
    original = os.environ.get("TERM")
    try:
        os.environ["TERM"] = "before-test"
        with _test_term("xterm"):
            expect(os.environ.get("TERM") == "xterm", "test TERM guard did not set the requested value")
        expect(os.environ.get("TERM") == "before-test", "test TERM guard did not restore an existing value")

        os.environ.pop("TERM", None)
        with _test_term("xterm"):
            expect(os.environ.get("TERM") == "xterm", "test TERM guard did not set an absent value")
        expect("TERM" not in os.environ, "test TERM guard recreated an absent value")
    finally:
        if original is None:
            os.environ.pop("TERM", None)
        else:
            os.environ["TERM"] = original








def test_reconcile_delayed_expiration_dry_run_converges_to_live_slot():
    """Dry-run should preview every elapsed expiration hop through the first live slot."""
    path = Path(ROOT) / "nautical_core" / "tools" / "nautical_reconcile.py"
    tool = _load_hook_module(str(path), "_nautical_reconcile_delayed_dry_run_test")
    hook_path = _find_hook_file("on-modify.nautical")
    hook = _load_hook_module(hook_path, "_nautical_reconcile_delayed_dry_run_hook")
    recovery_at = hook.core.build_local_datetime(date(2026, 7, 23), (9, 30))
    parent = {
        "uuid": "11111111-0000-4000-8000-000000000001",
        "status": "deleted",
        "description": "delayed daily occurrence",
        "cp": "1d",
        "chain": "on",
        "chainID": "delayed1",
        "link": 1,
        "due": hook.core.fmt_isoz(hook.core.build_local_datetime(date(2026, 7, 20), (9, 0))),
        "until": hook.core.fmt_isoz(hook.core.build_local_datetime(date(2026, 7, 20), (10, 0))),
        "end": hook.core.fmt_isoz(hook.core.build_local_datetime(date(2026, 7, 20), (10, 0))),
    }
    original = tool._existing_children_for_plan
    try:
        tool._existing_children_for_plan = lambda _task_bin, _parent, _hook: []
        outcomes = tool._reconcile_candidate(
            "task",
            hook,
            parent,
            taskdata=None,
            apply=False,
            max_expiration_hops=8,
            recovery_at=recovery_at,
            reconciliation_service=tool._reconcile_runtime_state().lifecycle_service,
        )
        limited = tool._reconcile_candidate(
            "task",
            hook,
            parent,
            taskdata=None,
            apply=False,
            max_expiration_hops=2,
            recovery_at=recovery_at,
            reconciliation_service=tool._reconcile_runtime_state().lifecycle_service,
        )
        capped = tool._reconcile_candidate(
            "task",
            hook,
            dict(parent, chainMax=2),
            taskdata=None,
            apply=False,
            max_expiration_hops=8,
            recovery_at=recovery_at,
            reconciliation_service=tool._reconcile_runtime_state().lifecycle_service,
        )
        anchor_parent = {
            **parent,
            "anchor": "w:mon..sun@t=09:00,13:00",
            "anchor_mode": "skip",
            "chainID": "delayed-anchor",
            "until": hook.core.fmt_isoz(hook.core.build_local_datetime(date(2026, 7, 20), (14, 0))),
            "end": hook.core.fmt_isoz(hook.core.build_local_datetime(date(2026, 7, 20), (14, 0))),
        }
        anchor_parent.pop("cp")
        anchor_outcomes = tool._reconcile_candidate(
            "task",
            hook,
            anchor_parent,
            taskdata=None,
            apply=False,
            max_expiration_hops=8,
            recovery_at=hook.core.build_local_datetime(date(2026, 7, 21), (14, 30)),
            reconciliation_service=tool._reconcile_runtime_state().lifecycle_service,
        )
    finally:
        tool._existing_children_for_plan = original

    plans = [plan for plan, _applied in outcomes]
    expect([plan.action for plan in plans] == ["spawn", "spawn", "spawn"], f"unexpected recovery path: {plans}")
    expect([plan.parent.get("link") for plan in plans] == [1, 2, 3], f"recovery skipped chain links: {plans}")
    final_until, final_until_err = hook._TASK_DATETIME_PARSER.parse((plans[-1].child or {}).get("until"))
    expect(not final_until_err and final_until is not None and final_until > recovery_at, f"final slot is already expired: {plans[-1]}")
    expect(
        [plan.action for plan, _applied in limited] == ["spawn", "spawn", "partial"],
        f"hop limit should leave an actionable resumable partial result: {limited}",
    )
    expect("hop limit" in limited[-1][0].reason, f"missing hop-limit guidance: {limited[-1][0]}")
    expect(
        [plan.action for plan, _applied in capped] == ["spawn", "legitimate_final"],
        f"chainMax was not enforced during delayed recovery: {capped}",
    )
    expect(
        [plan.next_link for plan, _applied in anchor_outcomes] == [2, 3, 4, 5],
        f"multi-time anchor recovery skipped slots: {anchor_outcomes}",
    )


def test_reconcile_reuses_verified_live_recovery_child():
    """A freshly verified live child should not require a second export before recovery stops."""
    path = Path(ROOT) / "nautical_core" / "tools" / "nautical_reconcile.py"
    tool = _load_hook_module(str(path), "_nautical_reconcile_verified_child_reuse_test")
    parent = {
        "uuid": "11111111-0000-4000-8000-000000000001",
        "status": "deleted",
        "chain": "on",
        "chainID": "verified1",
        "link": 1,
        "cp": "1d",
    }
    child = {
        "uuid": "22222222-0000-0000-0000-000000000002",
        "status": "pending",
        "chain": "on",
        "chainID": "verified1",
        "link": 2,
        "prevLink": "00000000",
        "due": "20260723T090000Z",
        "until": "20260723T100000Z",
    }

    def parse(value):
        try:
            return datetime.strptime(str(value), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc), None
        except Exception:
            return None, "invalid datetime"

    hook = SimpleNamespace(datetime_parser=SimpleNamespace(parse=parse))
    original_apply = tool._apply_parent_atomic
    original_lookup = tool._next_recovery_child
    try:
        def apply_parent(_task_bin, _hook, current, *, taskdata, lease_held=False, verified_children=None):
            expect(verified_children is not None, "recovery did not provide verified-child cache")
            verified_children["22222222"] = dict(child)
            from nautical_core.lifecycle_models import (
                LifecycleAction,
                LifecycleEvent,
                LifecycleIdentity,
                LifecyclePlan,
                ParentGuard,
                recurrence_fingerprint,
            )
            from nautical_core.lifecycle_recovery_models import RecoveryPlanResult
            parent_observation = _fixture_observation(current)
            parent_values = parent_observation.to_mapping()
            guard = ParentGuard(
                status="deleted",
                chain="on",
                chain_id=str(parent_values["chainID"]),
                link=1,
                recurrence_fingerprint=recurrence_fingerprint(parent_values),
                modified="",
            )
            identity = LifecycleIdentity(
                chain_id=str(parent_values["chainID"]),
                parent_uuid=str(parent_values["uuid"]),
                source_link=1,
                target_link=2,
                event=LifecycleEvent.EXPIRE,
            )
            typed_plan = LifecyclePlan(
                identity=identity,
                action=LifecycleAction.SPAWN_CHILD,
                parent_guard=guard,
                child_payload=tuple(sorted(child.items())),
            )
            return RecoveryPlanResult(
                parent_observation,
                typed_plan,
                reason="expired link missing next link",
                child_short="22222222",
                child_observation=_fixture_observation(child),
            ), "22222222"

        tool._apply_parent_atomic = apply_parent
        tool._next_recovery_child = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("live verified child was exported again")
        )
        outcomes = tool._reconcile_candidate(
            "task",
            hook,
            parent,
            taskdata=Path("/tmp/nautical-reconcile-verified-child-test"),
            apply=True,
            max_expiration_hops=8,
            recovery_at=datetime(2026, 7, 22, 9, 30, tzinfo=timezone.utc),
            reconciliation_service=tool._reconcile_runtime_state().lifecycle_service,
        )
    finally:
        tool._apply_parent_atomic = original_apply
        tool._next_recovery_child = original_lookup
    expect([plan.action for plan, _applied in outcomes] == ["spawn"], f"unexpected recovery outcomes: {outcomes!r}")


def test_reconcile_candidate_discovery_is_narrow_and_deterministic():
    """Repository candidates exclude linked parents and retain stable order."""
    path = Path(ROOT) / "nautical_core" / "tools" / "nautical_reconcile.py"
    tool = _load_hook_module(str(path), "_nautical_reconcile_candidate_order_test")
    rows = [
        {
            "uuid": "33333333-0000-0000-0000-000000000003",
            "status": "deleted",
            "cp": "1d",
            "chain": "on",
            "chainID": "z-chain",
            "link": 3,
            "until": "20260720T100000Z",
            "end": "20260720T100000Z",
        },
        {
            "uuid": "11111111-0000-0000-0000-000000000001",
            "status": "completed",
            "cp": "1d",
            "chain": "on",
            "chainID": "a-chain",
            "link": 2,
        },
    ]
    class Repository:
        def lifecycle_candidates(self, **_kwargs):
            return _found_task(tuple(reversed(rows)))

    snapshot = tool._ReconcileSnapshot(Repository())
    found = tool.LifecycleReconciliationService(
        snapshot, snapshot.repository, configuration_fingerprint="test", schedule_fingerprint="test"
    ).candidates()
    expect(
        [(row["chainID"], row["link"]) for row in found] == [("a-chain", 2), ("z-chain", 3)],
        f"candidate order was not deterministic: {found}",
    )
    duplicate = {**rows[1], "uuid": "22222222-0000-0000-0000-000000000002"}
    conflicts = tool.IntegrityRecoveryService.ambiguous_candidate_slots([rows[1], duplicate])
    expect(("a-chain", 2) in conflicts and "2 distinct parent tasks" in conflicts[("a-chain", 2)],
           f"duplicate candidate slot was not rejected: {conflicts!r}")


def test_reconcile_snapshot_reuses_initial_chain_export():
    """Active and candidate scans reuse one authoritative repository snapshot."""
    path = Path(ROOT) / "nautical_core" / "tools" / "nautical_reconcile.py"
    tool = _load_hook_module(str(path), "_nautical_reconcile_snapshot_reuse_test")
    rows = [
        {
            "uuid": "11111111-0000-0000-0000-000000000001",
            "status": "completed",
            "cp": "1d",
            "chain": "on",
            "chainID": "chain-1",
            "link": 1,
        },
        {
            "uuid": "22222222-0000-0000-0000-000000000002",
            "status": "pending",
            "chain": "on",
            "chainID": "chain-1",
            "link": 2,
        },
    ]
    calls = []

    class Repository:
        def lifecycle_candidates(self, **kwargs):
            calls.append(kwargs)
            return _found_task(tuple(rows))

    snapshot = tool._ReconcileSnapshot(Repository())
    candidates = tool.LifecycleReconciliationService(
        snapshot, snapshot.repository, configuration_fingerprint="test", schedule_fingerprint="test"
    ).candidates()
    snapshot.active_rows()
    snapshot.active_rows()
    expect(candidates == [rows[0]], f"candidate view included non-candidates: {candidates!r}")
    expect(len(calls) == 1, f"snapshot views repeated the lifecycle export: {calls!r}")
    expect(tool._EXPORT_STATS["snapshot_hits"] == 2, f"snapshot hits were not recorded: {tool._EXPORT_STATS!r}")


def test_reconcile_empty_snapshot_is_authoritative():
    """Successful empty active and candidate views remain authoritative."""
    path = Path(ROOT) / "nautical_core" / "tools" / "nautical_reconcile.py"
    tool = _load_hook_module(str(path), "_nautical_reconcile_empty_snapshot_test")
    calls = []

    class Repository:
        def lifecycle_candidates(self, **kwargs):
            calls.append(kwargs)
            return _absent_task("no lifecycle candidates")

    snapshot = tool._ReconcileSnapshot(Repository())
    candidates = tool.LifecycleReconciliationService(
        snapshot, snapshot.repository, configuration_fingerprint="test", schedule_fingerprint="test"
    ).candidates()
    active = snapshot.active_rows()
    expect(candidates == [] and active == [], f"empty snapshot produced unexpected rows: {candidates!r}, {active!r}")
    expect(len(calls) == 1, f"empty snapshot triggered repeated exports: {calls!r}")




def test_reconcile_lifecycle_outcomes_preserve_retry_and_manual_review():
    """Typed lifecycle outcomes remain actionable instead of becoming generic errors."""
    path = Path(ROOT) / "nautical_core" / "tools" / "nautical_reconcile.py"
    tool = _load_hook_module(str(path), "_nautical_reconcile_lifecycle_outcomes_test")
    parent = {
        "uuid": "11111111-0000-0000-0000-000000000001",
        "status": "completed",
        "chain": "on",
        "chainID": "outcome01",
        "link": 1,
    }
    original_apply = tool._apply_parent_atomic
    try:
        for exception_type, expected_action in (
            (tool._LifecycleRetryable, "partial"),
            (tool._LifecycleManualReview, "manual_review"),
        ):
            def fail_apply(*_args, _exception_type=exception_type, **_kwargs):
                raise _exception_type("typed lifecycle outcome")

            tool._apply_parent_atomic = fail_apply
            outcomes = tool._reconcile_candidate(
                "task",
                SimpleNamespace(),
                parent,
                taskdata=Path("/tmp/nautical-reconcile-lifecycle-outcomes-test"),
                apply=True,
                max_expiration_hops=4,
                recovery_at=datetime.now(timezone.utc),
                reconciliation_service=tool._reconcile_runtime_state().lifecycle_service,
            )
            expect(len(outcomes) == 1, f"typed outcome produced extra reconcile work: {outcomes!r}")
            plan, applied = outcomes[0]
            expect(plan.action == expected_action, f"typed outcome became {plan.action!r}: {plan!r}")
            expect("typed lifecycle outcome" in plan.reason, f"typed reason was lost: {plan!r}")
            expect(not applied, f"typed outcome reported a mutation: {applied!r}")
    finally:
        tool._apply_parent_atomic = original_apply


def test_reconcile_planning_configuration_drift_is_partial():
    """A configuration change at the planning boundary must block preview mutations."""
    path = Path(ROOT) / "nautical_core" / "tools" / "nautical_reconcile.py"
    tool = _load_hook_module(str(path), "_nautical_reconcile_planning_drift_test")
    parent = {
        "uuid": "11111111-0000-0000-0000-000000000001",
        "status": "completed",
        "chain": "on",
        "chainID": "drift01",
        "link": 1,
        "cp": "1d",
    }
    hook = SimpleNamespace(
        core=SimpleNamespace(
            configuration_drift=lambda: {"changed": True, "source": "test-config"},
        )
    )
    outcomes = tool._reconcile_candidate(
        "task",
        hook,
        parent,
        taskdata=None,
        apply=False,
        max_expiration_hops=4,
        recovery_at=datetime.now(timezone.utc),
        generation=object(),
        reconciliation_service=tool._reconcile_runtime_state().lifecycle_service,
    )
    expect(len(outcomes) == 1, f"configuration drift produced extra work: {outcomes!r}")
    plan, applied = outcomes[0]
    expect(plan.action == "partial", f"configuration drift was not partial: {plan!r}")
    expect("configuration changed" in plan.reason, f"configuration reason was lost: {plan!r}")
    expect(not applied, f"configuration drift reported a mutation: {applied!r}")












def test_on_modify_lifecycle_export_reuses_completion_chain_snapshot():
    """Lifecycle filtering and completion presentation share one chain export."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_lifecycle_export_reuse_test")
    mod._reset_modify_runtime_state()
    saved_analytics = mod._SHOW_ANALYTICS
    mod._SHOW_ANALYTICS = True
    from nautical_core.integration_models import CommandFailureKind, TaskCommand, TaskCommandResult

    uow = _test_operator_uow()
    calls = {"count": 0}

    class Client:
        def execute(self, args, *, purpose, timeout, **_kwargs):
            calls["count"] += 1
            rows = [{
                "uuid": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "chainID": "reuse02",
                "link": 2,
                "chain": "on",
                "status": "pending",
            }]
            command = TaskCommand(("task", *args), purpose, timeout)
            return TaskCommandResult(command, 0, json.dumps(rows), "", CommandFailureKind.SUCCESS, 1, 0.001)

    uow.client = Client()
    mod._modify_runtime_state().task_repository = uow.repository
    try:
        rows = mod._module("modify_composition").lifecycle_read_service_for(mod).get_chain_export("reuse02")
        expect(len(rows) == 1, f"lifecycle chain export returned unexpected rows: {rows!r}")
        snapshot = mod._completion_effects.chain_snapshot("reuse02", 1, 2, uow.repository)
        expect(snapshot.loaded and snapshot.rows, f"completion snapshot did not reuse chain rows: {snapshot!r}")
        expect(calls["count"] == 1, f"lifecycle and completion repeated chain export: {calls}")
    finally:
        mod._SHOW_ANALYTICS = saved_analytics
        mod._reset_modify_runtime_state()


TESTS = [
    *SCHEDULING_TESTS[:1],
    *SCHEDULING_TESTS[9:10],
    *CONFIGURATION_TESTS[5:6],
    *CONFIGURATION_TESTS[10:12],
    test_modifier_boundary_paths_agree_and_advance_strictly,
    *RECURRENCE_TESTS,
    *RECONCILE_TESTS,
    *SCHEDULING_TESTS[5:6],
    test_hook_on_modify_cp_malformed_inputs_fail_with_parser_guidance,
    *MODIFY_TESTS[20:25],
    *MODIFY_TESTS[25:30],
    *TIMELINE_TESTS[:3],
    *LIFECYCLE_TESTS[15:18],
    *STORAGE_TESTS,
    *MODIFY_TESTS[:1],
    *MODIFY_TESTS[35:36],
    *OPERATOR_TESTS[:2],
    *LIFECYCLE_TESTS[18:19],
    *OPERATOR_TESTS[2:5],
    *OPERATOR_TESTS[5:10],
    *OPERATOR_TESTS[10:11],
    *INSTALLER_TESTS[:5],
    *INSTALLER_TESTS[5:6],
    *OPERATOR_TESTS[11:12],
    *INSTALLER_TESTS[6:8],
    *OPERATOR_TESTS[12:13],
    *OPERATOR_TESTS[13:17],
    *PERFORMANCE_TESTS[7:8],
    *PERFORMANCE_TESTS[:3],
    *PERFORMANCE_TESTS[8:9],
    test_load_benchmark_installs_complete_hook_runtime,
    test_load_benchmark_queue_and_lineage_verification,
    *PERFORMANCE_TESTS[3:6],
    *PERFORMANCE_TESTS[9:],
    *PERFORMANCE_TESTS[6:7],
    *SCHEDULING_TESTS[1:5],
    *MODIFY_TESTS[36:37],
    *MODIFY_TESTS[1:6],
    *MODIFY_TESTS[6:10],
    test_on_add_requires_integration_context_helper,
    *MODIFY_TESTS[10:15],
    *MODIFY_TESTS[30:34],
    *MODIFY_TESTS[37:38],
    *MODIFY_TESTS[15:20],
    *LIFECYCLE_TESTS[25:26],
    *SCHEDULING_TESTS[10:],
    *TIMELINE_TESTS[3:5],
    *TIMELINE_TESTS[5:6],
    *LIFECYCLE_TESTS[20:23],
    test_on_modify_panel_fallback,
    test_on_modify_panel_forwards_live_duration,
    test_ui_live_test_term_guard_restores_environment,
    *LIFECYCLE_TESTS[26:30],
    test_on_modify_lifecycle_export_reuses_completion_chain_snapshot,
    *LIFECYCLE_TESTS[23:24],
    *LIFECYCLE_TESTS[19:20],
    *MODIFY_TESTS[38:39],
    *CONFIGURATION_TESTS[:3],
    *CONFIGURATION_TESTS[6:10],
    *CONFIGURATION_TESTS[12:15],
    *MODIFY_TESTS[39:40],
    *SCHEDULING_TESTS[8:9],
    *MODIFY_TESTS[34:35],

]

# This characterization deliberately mutates the compatibility facade's
# synchronization state while exercising Navigator. Keep its lazy API
# bindings from leaking into later hook cases in shuffled runs.
ISOLATED_GOLDEN_TESTS = frozenset({
    "test_navigator_uses_anchor_and_anchor_file_sources",
})

DEEP_TESTS = [

]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        action="append",
        help="substring filter for test names (repeatable; filters are ORed)",
    )
    ap.add_argument("--verbose", action="store_true", help="show detailed test information")
    ap.add_argument(
        "--shuffle-seed",
        type=int,
        help="run the selected tests in a deterministic shuffled order",
    )
    ap.add_argument(
        "--strict-lifecycle-warnings",
        action="store_true",
        help="fail selected tests on leaked temporary-Taskdata or stale-config warnings",
    )
    ap.add_argument("--isolated-child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    selected = TESTS
    if args.only:
        filters = tuple(value.lower() for value in args.only)
        selected = [
            fn for fn in TESTS
            if any(value in fn.__name__.lower() for value in filters)
        ]
    if args.shuffle_seed is not None:
        selected = list(selected)
        random.Random(args.shuffle_seed).shuffle(selected)

    fails = 0
    total_tests = 0
    
    for fn in selected:
        total_tests += 1
        captured_stderr = io.StringIO()
        try:
            if not args.isolated_child and (
                "_load_core_module" in fn.__code__.co_names
                or fn.__name__ in ISOLATED_GOLDEN_TESTS
            ):
                child_env = dict(os.environ)
                child_env["NAUTICAL_GOLDEN_ISOLATED"] = "1"
                child = subprocess.run(
                    [sys.executable, __file__, "--only", fn.__name__, "--isolated-child"],
                    cwd=ROOT,
                    env=child_env,
                    capture_output=True,
                    text=True,
                    timeout=60.0,
                )
                if child.returncode != 0:
                    detail = (child.stdout or "") + (child.stderr or "")
                    raise AssertionError(f"isolated test failed: {detail.strip()[-1200:]}")
                if args.verbose:
                    docstring = fn.__doc__ or "No description available"
                    description = docstring.strip().split('\n')[0] if docstring else fn.__name__
                    print(f"✓ {fn.__name__}: {description}")
                continue
            # Reset process-wide presentation/season knobs before every case;
            # a shuffled run must not inherit state from tests that exercise
            # alternate seasonal profiles or panel modes.
            try:
                season = core._import_sibling("season_support")
                season.configure_mode("fixed")
                season.configure_hemisphere("north")
                season.configure_timezone(core.LOCAL_TZ_NAME)
                core.PANEL_MODE = "rich"
            except Exception:
                pass
            # Some isolation tests intentionally import a disposable package;
            # restore the canonical facade before the next test so shuffled
            # runs do not inherit that temporary module.
            if sys.modules.get("nautical_core") is not core:
                sys.modules["nautical_core"] = core
            if args.strict_lifecycle_warnings:
                with contextlib.redirect_stderr(captured_stderr):
                    fn()
            else:
                fn()
            if args.strict_lifecycle_warnings:
                warning_text = captured_stderr.getvalue()
                leaked = [
                    line.strip()
                    for line in warning_text.splitlines()
                    if "Taskwarrior data directory does not exist" in line
                    or "stale configuration" in line.lower()
                    or "temporary Taskdata" in line
                ]
                if leaked:
                    raise AssertionError("lifecycle state warning leaked: " + " | ".join(leaked))
            if args.verbose:
                # Extract docstring and print test description
                docstring = fn.__doc__ or "No description available"
                # Get first line of docstring
                description = docstring.strip().split('\n')[0] if docstring else fn.__name__
                print(f"✓ {fn.__name__}: {description}")
        except AssertionError as e:
            fails += 1
            if args.verbose:
                docstring = fn.__doc__ or "No description available"
                description = docstring.strip().split('\n')[0] if docstring else fn.__name__
                print(f"✗ {fn.__name__}: {description}")
                print(f"  ERROR: {e}")
            else:
                print(f"✗ {fn.__name__}: {e}")
        except Exception as e:
            fails += 1
            if args.verbose:
                docstring = fn.__doc__ or "No description available"
                description = docstring.strip().split('\n')[0] if docstring else fn.__name__
                print(f"✗ {fn.__name__}: {description}")
                print(f"  UNEXPECTED ERROR: {e}")
                import traceback
                traceback.print_exc()
            else:
                print(f"✗ {fn.__name__}: unexpected error {e}")
        except SystemExit as e:
            fails += 1
            message = f"unexpected SystemExit({e.code!r})"
            if args.verbose:
                docstring = fn.__doc__ or "No description available"
                description = docstring.strip().split('\n')[0] if docstring else fn.__name__
                print(f"✗ {fn.__name__}: {description}")
                print(f"  ERROR: {message}")
            else:
                print(f"✗ {fn.__name__}: {message}")

    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests run: {total_tests}")
    print(f"Passed: {total_tests - fails}")
    print(f"Failed: {fails}")
    success_rate = ((total_tests - fails) / total_tests * 100) if total_tests else 100.0
    print(f"Success rate: {success_rate:.1f}%")
    print(f"{'='*60}")
    
    sys.exit(1 if fails else 0)

TESTS.extend([
    *OPERATOR_TESTS[17:],
    *TIMELINE_TESTS[6:],
    test_navigator_surfaces_configuration_drift_warning,
    test_navigator_reloads_validated_taskdata_configuration,
    test_navigator_fallback_export_uses_empty_filter,
    test_shared_time_slot_resolver_keeps_hook_and_navigator_parity,
    test_navigator_projects_all_slots_in_a_time_window,
    *SCHEDULING_TESTS[6:8],
    test_navigator_reads_through_read_only_invocation_repository,
    test_navigator_uses_anchor_and_anchor_file_sources,
    *CONFIGURATION_TESTS[3:5],
    *INSTALLER_TESTS[8:],
])

TESTS.extend(LIFECYCLE_TESTS[24:25])
# =============================================================================
# Section 12: Failure, Concurrency, and Recovery Verification
# Tests for lifecycle_application.LifecycleApplicationService
# Written against the new architecture. Replaces white-box tests that targeted
# deleted legacy lifecycle internals.
# =============================================================================















def _legacy_test_on_modify_staged_plan_carries_parent_guard_and_stable_intent_id():
    """Staging via on-modify's _enqueue_spawn_intent must persist the parent
    guard that authorized the spawn, and the intent_id must be stable across
    repeated calls for the same transition."""
    import tempfile
    from pathlib import Path
    from nautical_core.lifecycle_models import (
        LifecycleAction, LifecycleEvent, LifecycleIdentity, ParentGuard,
        recurrence_fingerprint,
    )
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository

    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_staged_guard_test")

    parent = {
        "uuid": "00000000-0000-4000-8000-000000000111",
        "status": "completed",
        "chain": "on",
        "chainID": "abcd1234",
        "link": 4,
        "nextLink": "",
        "modified": "20260101T000000Z",
        "cp": "1d",
    }
    child_uuid = "00000000-0000-4000-8000-00000000abcd"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mod._INTEGRATION_CONTEXT = type("FakeCtx", (), {
            "configuration": type("FakeCfg", (), {
                "fingerprint": "cfg-guard-test",
                "scheduler_fingerprint": "sch-guard-test",
            })(),
        })()
        mod.TW_DATA_DIR = root

        # Build and stage a typed lifecycle plan directly via _enqueue_spawn_intent,
        # which is the unit under test (no need to involve _spawn_child_atomic internals).
        rf = recurrence_fingerprint(parent)
        guard = ParentGuard(
            status=parent["status"],
            chain=parent["chain"],
            chain_id=parent["chainID"],
            link=int(parent["link"]),
            recurrence_fingerprint=rf,
            modified=parent["modified"],
        )
        identity = LifecycleIdentity(
            chain_id=parent["chainID"],
            parent_uuid=parent["uuid"],
            source_link=int(parent["link"]),
            target_link=int(parent["link"]) + 1,
            event=LifecycleEvent.COMPLETE,
        )
        plan = _plan_from_values(
            identity=identity,
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=guard,
            child_payload={"uuid": child_uuid, "chainID": parent["chainID"], "link": 5, "prevLink": parent["uuid"][:8]},
            parent_patch={"nextLink": child_uuid[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
        )

        spawn_effects = mod._module("modify_spawn_effects")
        ports = spawn_effects.spawn_intent_ports_for(mod)
        ok, reason = spawn_effects.enqueue_spawn_intent(ports, plan)
        expect(ok, f"_enqueue_spawn_intent failed: {reason}")

        outbox = _LifecycleOutboxRepository(root)
        _, status = outbox.status()
        expect(len(status["records"]) == 1, f"expected 1 staged record: {status}")
        record = status["records"][0]

        # The staged plan must carry the parent guard with the recurrence fingerprint
        claim = outbox.claim_intent(owner="test-guard", lease_seconds=30, intent_id=record["intent_id"])
        expect(claim.ok, f"could not claim the staged intent: {claim}")
        staged_plan = claim.record.plan
        expect(staged_plan.parent_guard.status == parent["status"],
               f"parent guard status wrong: {staged_plan.parent_guard}")
        expect(staged_plan.parent_guard.chain_id == parent["chainID"],
               f"parent guard chainID wrong: {staged_plan.parent_guard}")
        expect(staged_plan.parent_guard.link == int(parent["link"]),
               f"parent guard link wrong: {staged_plan.parent_guard}")
        expect(
            str(staged_plan.parent_guard.recurrence_fingerprint or "").startswith("rf1-"),
            f"staged plan did not carry a recurrence fingerprint: {staged_plan.parent_guard}",
        )

        # Staging the same plan again must be idempotent (same intent_id, no second record)
        ok2, reason2 = spawn_effects.enqueue_spawn_intent(ports, plan)
        expect(ok2, f"second _enqueue_spawn_intent failed: {reason2}")
        _, status2 = outbox.status()
        expect(len(status2["records"]) == 1,
               f"duplicate staging created a second intent: {status2}")
        expect(status2["records"][0]["intent_id"] == record["intent_id"],
               f"second staging produced a different intent_id: {status2}")


TESTS.extend([
    *LIFECYCLE_TESTS[:15],
])

if __name__ == "__main__":
    main()
