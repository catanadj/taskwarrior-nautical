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
import sys, os, io, contextlib
import random
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ.setdefault("NAUTICAL_CORE_PATH", ROOT)

from dev_tools.golden_tests.operator import TESTS as OPERATOR_TESTS
from dev_tools.golden_tests.installer import TESTS as INSTALLER_TESTS
from dev_tools.golden_tests.storage import TESTS as STORAGE_TESTS
from dev_tools.golden_tests.timeline import TESTS as TIMELINE_TESTS
from dev_tools.golden_tests.lifecycle import TESTS as LIFECYCLE_TESTS
from dev_tools.golden_tests.reconcile import TESTS as RECONCILE_TESTS
from dev_tools.golden_tests.configuration import TESTS as CONFIGURATION_TESTS
from dev_tools.golden_tests.modify import TESTS as MODIFY_TESTS
from dev_tools.golden_tests.scheduling import TESTS as SCHEDULING_TESTS
from dev_tools.golden_tests.navigator import (
    ISOLATED_GOLDEN_TESTS,
    TESTS as NAVIGATOR_TESTS,
)

core = importlib.import_module("nautical_core")

# -------- Helpers -------------------------------------------------------------

# -------- Runner --------------------------------------------------------------


























TESTS = [
    *SCHEDULING_TESTS[:1],
    *SCHEDULING_TESTS[8:9],
    *SCHEDULING_TESTS[4:5],
    *SCHEDULING_TESTS[15:16],
    *RECONCILE_TESTS,
    *MODIFY_TESTS[38:39],
    *MODIFY_TESTS[20:24],
    *MODIFY_TESTS[24:29],
    *TIMELINE_TESTS[:3],
    *LIFECYCLE_TESTS[15:18],
    *STORAGE_TESTS,
    *MODIFY_TESTS[:1],
    *MODIFY_TESTS[34:35],
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
    *SCHEDULING_TESTS[1:4],
    *MODIFY_TESTS[35:36],
    *MODIFY_TESTS[1:6],
    *MODIFY_TESTS[6:10],
    *MODIFY_TESTS[10:15],
    *MODIFY_TESTS[29:33],
    *MODIFY_TESTS[36:37],
    *MODIFY_TESTS[15:20],
    *LIFECYCLE_TESTS[25:26],
    *SCHEDULING_TESTS[9:13],
    *TIMELINE_TESTS[3:5],
    *TIMELINE_TESTS[5:6],
    *LIFECYCLE_TESTS[20:23],
    *LIFECYCLE_TESTS[26:30],
    *LIFECYCLE_TESTS[30:31],
    *LIFECYCLE_TESTS[23:24],
    *LIFECYCLE_TESTS[19:20],
    *MODIFY_TESTS[37:38],
    *CONFIGURATION_TESTS[:3],
    *SCHEDULING_TESTS[7:8],
    *MODIFY_TESTS[33:34],

]

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
    *NAVIGATOR_TESTS[:4],
    *SCHEDULING_TESTS[5:7],
    *NAVIGATOR_TESTS[4:],
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
