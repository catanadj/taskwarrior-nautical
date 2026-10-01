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
import sys, os, json, io, contextlib
import random
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

from dev_tools.golden_tests.recurrence import TESTS as RECURRENCE_TESTS
from dev_tools.golden_tests.operator import TESTS as OPERATOR_TESTS
from dev_tools.golden_tests.installer import TESTS as INSTALLER_TESTS
from dev_tools.golden_tests.storage import TESTS as STORAGE_TESTS
from dev_tools.golden_tests.timeline import TESTS as TIMELINE_TESTS
from dev_tools.golden_tests.lifecycle import TESTS as LIFECYCLE_TESTS
from dev_tools.golden_tests.reconcile import TESTS as RECONCILE_TESTS
from dev_tools.golden_tests.configuration import TESTS as CONFIGURATION_TESTS
from dev_tools.golden_tests.modify import TESTS as MODIFY_TESTS
from dev_tools.golden_tests.scheduling import TESTS as SCHEDULING_TESTS
from dev_tools.golden_tests.support import (
    expect,
    test_operator_uow as _test_operator_uow,
    load_hook_module as _load_hook_module,
    find_hook_file as _find_hook_file,
    typed_command_result as _typed_command_result,
)

core = importlib.import_module("nautical_core")
timezone_facade = importlib.import_module("nautical_core.timezone_facade")
reconcile_report = importlib.import_module("nautical_core.reconcile_report")
_hook = importlib.import_module("nautical_core.hooks.modify_impl")

# -------- Helpers -------------------------------------------------------------

# -------- Test cases ----------------------------------------------------------
# -------- Hook checks ---------------------------------------------------------
# These tests validate the shipped hook scripts (on-add / on-modify) at a high
# level, to catch regressions that can slip through core-only tests.

import subprocess
import importlib.util
import importlib.machinery

# -------- Runner --------------------------------------------------------------

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
    old_tz = timezone_facade.current_timezone()
    try:
        loader.exec_module(navigator)
        core.LOCAL_TZ_NAME = "Etc/GMT-3"
        timezone_facade._local_timezone = timezone(timedelta(hours=3))
        navigator.LOCAL_ZONE = timezone_facade.current_timezone()
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
        timezone_facade._local_timezone = old_tz
        sys.modules.pop(module_name, None)




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













TESTS = [
    *SCHEDULING_TESTS[:1],
    *SCHEDULING_TESTS[9:10],
    *CONFIGURATION_TESTS[3:5],
    *SCHEDULING_TESTS[15:16],
    *RECURRENCE_TESTS,
    *RECONCILE_TESTS,
    *SCHEDULING_TESTS[5:6],
    *MODIFY_TESTS[42:43],
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
    *SCHEDULING_TESTS[1:5],
    *MODIFY_TESTS[36:37],
    *MODIFY_TESTS[1:6],
    *MODIFY_TESTS[6:10],
    *MODIFY_TESTS[10:15],
    *MODIFY_TESTS[30:34],
    *MODIFY_TESTS[37:38],
    *MODIFY_TESTS[15:20],
    *LIFECYCLE_TESTS[25:26],
    *SCHEDULING_TESTS[10:15],
    *TIMELINE_TESTS[3:5],
    *TIMELINE_TESTS[5:6],
    *LIFECYCLE_TESTS[20:23],
    *MODIFY_TESTS[40:42],
    *LIFECYCLE_TESTS[26:30],
    *LIFECYCLE_TESTS[30:31],
    *LIFECYCLE_TESTS[23:24],
    *LIFECYCLE_TESTS[19:20],
    *MODIFY_TESTS[38:39],
    *CONFIGURATION_TESTS[:3],
    *CONFIGURATION_TESTS[5:8],
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
    print("TEST SUMMARY")
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
    *INSTALLER_TESTS[8:],
])

TESTS.extend(LIFECYCLE_TESTS[24:25])
# =============================================================================
# Section 12: Failure, Concurrency, and Recovery Verification
# Tests for lifecycle_application.LifecycleApplicationService
# Written against the new architecture. Replaces white-box tests that targeted
# deleted legacy lifecycle internals.
# =============================================================================















TESTS.extend([
    *LIFECYCLE_TESTS[:15],
])

if __name__ == "__main__":
    main()
