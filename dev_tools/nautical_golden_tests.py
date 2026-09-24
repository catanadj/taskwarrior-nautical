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
import sys, os, re, json, io, contextlib, stat
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
from dev_tools.golden_tests.recurrence import TESTS as RECURRENCE_TESTS
from dev_tools.golden_tests.hooks import TESTS as HOOK_TESTS
from dev_tools.golden_tests.operator import TESTS as OPERATOR_TESTS
from dev_tools.golden_tests.installer import TESTS as INSTALLER_TESTS
from dev_tools.golden_tests.performance import TESTS as PERFORMANCE_TESTS
from dev_tools.golden_tests.lifecycle import TESTS as LIFECYCLE_TESTS
from dev_tools.golden_tests.reconcile import TESTS as RECONCILE_TESTS
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


def test_on_add_fail_and_exit_emits_json():
    """_fail_and_exit should fail-closed without emitting task JSON."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_fail_test")
    task = {"uuid": "00000000-0000-4000-8000-000000000abc", "description": "fail test"}
    mod._PARSED_TASK = dict(task)
    mod._RAW_INPUT_TEXT = json.dumps(task, ensure_ascii=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            mod._fail_and_exit("Invalid anchor", "anchor syntax error: bad")
        except SystemExit as e:
            expect(e.code == 1, f"unexpected exit code: {e.code}")
        else:
            raise AssertionError("_fail_and_exit did not exit")
    out = buf.getvalue().strip()
    expect(out == "", f"expected no stdout on failure, got: {out!r}")


def test_on_add_panic_passthrough_emits_valid_json():
    """on-add panic passthrough should always emit a valid JSON object."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_panic_passthrough_test")
    mod._PARSED_TASK = {"uuid": "00000000-0000-4000-8000-000000000111", "description": "panic-add"}
    mod._RAW_INPUT_TEXT = "{not-json"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod._panic_passthrough()
    out = buf.getvalue().strip()
    obj = json.loads(out or "{}")
    expect(isinstance(obj, dict), f"panic passthrough must emit JSON object, got: {out!r}")
    expect(obj.get("uuid") == "00000000-0000-4000-8000-000000000111", "parsed task should be preserved")


def test_on_modify_panic_passthrough_uses_latest_task():
    """on-modify panic passthrough should emit the latest task object."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_panic_passthrough_test")
    mod._PARSED_NEW = None
    old = {"uuid": "00000000-0000-4000-8000-000000000111", "status": "pending"}
    new = {"uuid": "00000000-0000-4000-8000-000000000111", "status": "completed"}
    mod._RAW_INPUT_TEXT = json.dumps(old) + "\n" + json.dumps(new)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod._panic_passthrough()
    out = buf.getvalue().strip()
    obj = json.loads(out or "{}")
    expect(isinstance(obj, dict), f"panic passthrough must emit JSON object, got: {out!r}")
    expect(obj.get("status") == "completed", f"expected latest task, got: {obj}")


def test_on_add_ignores_unsafe_core_path_override():
    """on-add should ignore unsafe NAUTICAL_CORE_PATH overrides by default."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_unsafe_core_path_test")
    prev = os.environ.get("NAUTICAL_CORE_PATH")
    prev_trust = os.environ.get("NAUTICAL_TRUST_CORE_PATH")
    try:
        with tempfile.TemporaryDirectory() as td:
            try:
                os.chmod(td, 0o777)
            except Exception:
                pass
            os.environ["NAUTICAL_CORE_PATH"] = td
            os.environ.pop("NAUTICAL_TRUST_CORE_PATH", None)
            got = mod._trusted_core_base(Path(mod.TW_DIR))
            expect(Path(got).resolve() == Path(mod.TW_DIR).resolve(),
                   f"unsafe core path should fall back to TW_DIR, got {got}")
    finally:
        if prev is None:
            os.environ.pop("NAUTICAL_CORE_PATH", None)
        else:
            os.environ["NAUTICAL_CORE_PATH"] = prev
        if prev_trust is None:
            os.environ.pop("NAUTICAL_TRUST_CORE_PATH", None)
        else:
            os.environ["NAUTICAL_TRUST_CORE_PATH"] = prev_trust


def test_hook_bootstrap_uses_symlink_path_and_core_path_rescue():
    """Symlinked hook installs should boot from the symlink path and NAUTICAL_CORE_PATH rescue."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        staging = base / "staging"
        hooks = base / "hooks"
        taskdata = base / "taskdata"
        staging.mkdir(parents=True, exist_ok=True)
        hooks.mkdir(parents=True, exist_ok=True)
        taskdata.mkdir(parents=True, exist_ok=True)

        hook_names = ("on-add.nautical", "on-modify.nautical", "on-exit.nautical")
        for name in hook_names:
            src = Path(ROOT) / name
            dst = staging / name
            shutil.copy2(src, dst)
            link = hooks / name
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(dst, link)

        env = os.environ.copy()
        env["NAUTICAL_CORE_PATH"] = ROOT
        env["NAUTICAL_TRUST_CORE_PATH"] = "1"
        env["TASKDATA"] = str(taskdata)
        env["TZ"] = "UTC"
        env.pop("NAUTICAL_DIAG", None)
        env.pop("NAUTICAL_DIAG_LOG", None)

        add_task = {
            "uuid": "11111111-1111-1111-1111-111111111111",
            "description": "symlink bootstrap add",
            "status": "pending",
            "entry": "20260101T000000Z",
            "modified": "20260101T000000Z",
        }
        modify_old = {
            "uuid": "22222222-2222-2222-2222-222222222222",
            "description": "symlink bootstrap modify",
            "status": "pending",
            "entry": "20260101T000000Z",
            "modified": "20260101T000000Z",
        }
        modify_new = dict(modify_old)
        modify_new["modified"] = "20260101T000001Z"

        cases = [
            (
                "on-add",
                hooks / "on-add.nautical",
                json.dumps(add_task, ensure_ascii=False),
                True,
            ),
            (
                "on-modify",
                hooks / "on-modify.nautical",
                json.dumps(modify_old, ensure_ascii=False) + "\n" + json.dumps(modify_new, ensure_ascii=False),
                True,
            ),
            (
                "on-exit",
                hooks / "on-exit.nautical",
                "",
                False,
            ),
        ]

        for name, path, raw_input, expect_json in cases:
            p = subprocess.run(
                [sys.executable, str(path)],
                input=raw_input,
                text=True,
                capture_output=True,
                env=env,
                timeout=10.0,
            )
            expect(p.returncode == 0, f"{name} failed via symlink install: rc={p.returncode}, stderr={p.stderr!r}")
            if expect_json:
                _assert_stdout_json_only(p.stdout or "")
            else:
                expect((p.stdout or "").strip() == "", f"{name} should keep stdout empty, got {p.stdout!r}")


def test_hooks_survive_malformed_numeric_environment():
    """Malformed numeric overrides must not break hook output or disable the full implementations."""
    malformed_names = (
        "NAUTICAL_PROFILE",
        "NAUTICAL_OUTBOX_DRAIN_MAX_ITEMS",
        "NAUTICAL_OUTBOX_DIAG_MAX_ITEMS",
        "NAUTICAL_OUTBOX_RETRY_MAX",
        "NAUTICAL_TASK_TIMEOUT_EXPORT",
        "NAUTICAL_TASK_TIMEOUT_IMPORT",
        "NAUTICAL_TASK_TIMEOUT_MODIFY",
        "NAUTICAL_TASK_RETRIES_EXPORT",
        "NAUTICAL_TASK_RETRIES_MODIFY",
        "NAUTICAL_TASK_RETRY_DELAY",
        "NAUTICAL_PARENT_LOCK_RETRIES",
        "NAUTICAL_PARENT_LOCK_SLEEP_BASE",
        "NAUTICAL_PARENT_LOCK_STALE_AFTER",
        "NAUTICAL_LOCK_STORM_THRESHOLD",
        "NAUTICAL_LOCK_BACKOFF_BASE",
        "NAUTICAL_LOCK_BACKOFF_MAX",
        "NAUTICAL_OUTBOX_LEASE_SECONDS",
        "NAUTICAL_CHAIN_EXPORT_TIMEOUT_BASE",
        "NAUTICAL_CHAIN_EXPORT_TIMEOUT_PER_100",
        "NAUTICAL_CHAIN_EXPORT_TIMEOUT_MAX",
        "NAUTICAL_DIAG_LOG_MAX_BYTES",
    )

    with tempfile.TemporaryDirectory() as td:
        env = {name: "not-a-number" for name in malformed_names}
        env.update({"TASKDATA": td, "NAUTICAL_BENCH_FORCE_FULL": "1"})
        task = {
            "uuid": "00000000-0000-4000-8000-000000000706",
            "status": "pending",
            "description": "Malformed env ăîșț",
        }

        add = _run_hook_script_raw(
            _find_hook_file("on-add.nautical"),
            json.dumps(task, ensure_ascii=False),
            env_extra=env,
        )
        expect(add.returncode == 0, f"on-add failed with malformed numeric env: {add.stderr!r}")
        expect(json.loads(add.stdout) == task, f"on-add stdout was not strict passthrough JSON: {add.stdout!r}")
        expect(add.stderr == "", f"on-add emitted diagnostics without opt-in: {add.stderr!r}")

        modified = dict(task, description="Modified malformed env ăîșț")
        modify = _run_hook_script_raw(
            _find_hook_file("on-modify.nautical"),
            json.dumps(task, ensure_ascii=False) + "\n" + json.dumps(modified, ensure_ascii=False),
            env_extra=env,
        )
        expect(modify.returncode == 0, f"on-modify failed with malformed numeric env: {modify.stderr!r}")
        expect(json.loads(modify.stdout) == modified, f"on-modify stdout was not strict JSON: {modify.stdout!r}")
        expect(modify.stderr == "", f"on-modify emitted diagnostics without opt-in: {modify.stderr!r}")

        exit_hook = _run_hook_script_raw(
            _find_hook_file("on-exit.nautical"),
            "",
            env_extra=env,
        )
        expect(exit_hook.returncode == 0, f"on-exit failed with malformed numeric env: {exit_hook.stderr!r}")
        expect(exit_hook.stdout == "", f"on-exit wrote to stdout: {exit_hook.stdout!r}")
        expect(exit_hook.stderr == "", f"on-exit emitted diagnostics without opt-in: {exit_hook.stderr!r}")

        runtime = importlib.import_module("nautical_core.runtime")
        saved_diag_log = os.environ.get("NAUTICAL_DIAG_LOG")
        saved_diag_max = os.environ.get("NAUTICAL_DIAG_LOG_MAX_BYTES")
        try:
            os.environ["NAUTICAL_DIAG_LOG"] = "1"
            os.environ["NAUTICAL_DIAG_LOG_MAX_BYTES"] = "not-a-number"
            runtime.diag_log("malformed env diagnostic", "golden", td)
            diag_path = Path(td) / ".nautical_diag.jsonl"
            expect(diag_path.is_file(), "malformed diagnostic size override prevented logging")
            payload = json.loads(diag_path.read_text(encoding="utf-8").splitlines()[-1])
            expect(payload.get("msg") == "malformed env diagnostic", f"unexpected diagnostic payload: {payload}")
        finally:
            if saved_diag_log is None:
                os.environ.pop("NAUTICAL_DIAG_LOG", None)
            else:
                os.environ["NAUTICAL_DIAG_LOG"] = saved_diag_log
            if saved_diag_max is None:
                os.environ.pop("NAUTICAL_DIAG_LOG_MAX_BYTES", None)
            else:
                os.environ["NAUTICAL_DIAG_LOG_MAX_BYTES"] = saved_diag_max


def test_on_modify_ignores_unsafe_core_path_override():
    """on-modify should ignore unsafe NAUTICAL_CORE_PATH overrides by default."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_unsafe_core_path_test")
    prev = os.environ.get("NAUTICAL_CORE_PATH")
    prev_trust = os.environ.get("NAUTICAL_TRUST_CORE_PATH")
    try:
        with tempfile.TemporaryDirectory() as td:
            try:
                os.chmod(td, 0o777)
            except Exception:
                pass
            os.environ["NAUTICAL_CORE_PATH"] = td
            os.environ.pop("NAUTICAL_TRUST_CORE_PATH", None)
            got = mod._trusted_core_base(Path(mod.TW_DIR))
            expect(Path(got).resolve() == Path(mod.TW_DIR).resolve(),
                   f"unsafe core path should fall back to TW_DIR, got {got}")
    finally:
        if prev is None:
            os.environ.pop("NAUTICAL_CORE_PATH", None)
        else:
            os.environ["NAUTICAL_CORE_PATH"] = prev
        if prev_trust is None:
            os.environ.pop("NAUTICAL_TRUST_CORE_PATH", None)
        else:
            os.environ["NAUTICAL_TRUST_CORE_PATH"] = prev_trust


def test_hook_protocol_loads_without_core_package():
    """The lightweight protocol gate must not import the nautical_core package."""
    path = os.path.join(ROOT, "nautical_core", "hook_protocol.py")
    code = (
        "import importlib.util,sys;"
        "spec=importlib.util.spec_from_file_location('_nautical_protocol_isolated',sys.argv[1]);"
        "mod=importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(mod);"
        "assert 'nautical_core' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, path],
        text=True,
        capture_output=True,
        timeout=5.0,
    )
    expect(proc.returncode == 0, f"protocol gate imported core: {proc.stderr!r}")


def test_taskwarrior_mutation_service_is_guarded_idempotent_and_fail_closed():
    """Named mutations re-read, verify, classify replay, and preserve failures."""
    from nautical_core.integration_models import (
        Absent,
        ChainDisablePayload,
        ChildCompensationPayload,
        CommandFailureKind,
        FailureEvidence,
        Found,
        GuardTimestamp,
        GuardTimestampField,
        MutationGuard,
        MutationOperation,
        MutationOutcomeKind,
        MutationRequest,
        NativeUntilRepairPayload,
        ParentLinkClearPayload,
        ParentLinkPayload,
        TaskCommand,
        TaskCommandResult,
        Unavailable,
    )
    from nautical_core.lifecycle_models import recurrence_fingerprint
    from nautical_core.task_codec import DEFAULT_TASK_CODEC
    from nautical_core.task_codec import DEFAULT_TASK_CODEC
    from nautical_core.taskwarrior_mutations import TaskwarriorMutationService

    parent_uuid = "00000000-0000-4000-8000-000000000924"
    child_uuid = "00000000-0000-4000-8000-000000000925"
    parent = {
        "uuid": parent_uuid,
        "status": "completed",
        "chain": "on",
        "chainID": "chain-service",
        "link": 7,
        "modified": "20260813T100000Z",
        "anchor": "w:mon",
        "cp": "1d",
    }

    class Repo:
        def __init__(self):
            self.rows = {parent_uuid: parent}
            self.unavailable = False

        def by_uuid(self, uuid_value, *, refresh=False):
            del refresh
            command = TaskCommand(("task", "export"), "test read", 1.0)
            if self.unavailable:
                evidence = FailureEvidence(command, CommandFailureKind.BUSY, 1, 1, 0.01, True, "lock active")
                return Unavailable(f"uuid:{uuid_value}", evidence)
            row = self.rows.get(str(uuid_value).lower())
            if row is None:
                return Absent(f"uuid:{uuid_value}", "not present")
            return Found(
                DEFAULT_TASK_CODEC.decode_row(row, source_query=f"uuid:{uuid_value}"),
                f"uuid:{uuid_value}",
            )

        def exact_child_slot(
            self,
            chain_id,
            link,
            *,
            statuses=(),
            expected_prev_link="",
            complete_chain_history=False,
            refresh=False,
        ):
            del complete_chain_history, refresh
            wanted_statuses = {str(value).lower() for value in statuses}
            for row in self.rows.values():
                if str(row.get("chainID") or "") != str(chain_id):
                    continue
                try:
                    if int(float(row.get("link"))) != int(link):
                        continue
                except (TypeError, ValueError):
                    continue
                if wanted_statuses and str(row.get("status") or "").lower() not in wanted_statuses:
                    continue
                if expected_prev_link and str(row.get("prevLink") or "") != str(expected_prev_link):
                    continue
                return Found(
                    DEFAULT_TASK_CODEC.decode_row(row, source_query=f"chainID:{chain_id} link:{link}"),
                    f"chainID:{chain_id} link:{link}",
                )
            return Absent(f"chainID:{chain_id} link:{link}", "not present")

    class Client:
        def __init__(self, repo):
            self.repo = repo
            self.calls = []

        def execute(self, args, *, purpose, timeout, input_text=None, attempts=1):
            del attempts
            args = list(args)
            self.calls.append((args, purpose))
            command = TaskCommand(("task", *args), purpose, timeout, input_text)
            if "import" in args:
                for line in (input_text or "{}").splitlines():
                    row = json.loads(line)
                    self.repo.rows[str(row["uuid"]).lower()] = row
            elif "delete" in args:
                uuid_token = next((item for item in args if item.startswith("uuid:")), "")
                self.repo.rows.pop(uuid_token.split(":", 1)[1].lower(), None)
            else:
                update_at = args.index("modify") + 1
                uuid_token = next((item for item in args if item.startswith("uuid:")), "")
                target_uuid = uuid_token.split(":", 1)[1].lower() if uuid_token else parent_uuid
                target = self.repo.rows[target_uuid]
                for token in args[update_at:]:
                    key, value = token.split(":", 1)
                    if key == "until" and "-" in value:
                        value = datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y%m%dT%H%M%SZ")
                    target[key] = value
            return TaskCommandResult(command, 0, "", "", CommandFailureKind.SUCCESS, 1, 0.01)

    class Uow:
        def __init__(self):
            self.repository = Repo()
            self.client = Client(self.repository)
            self.mutation_epoch = 0
            self.context = type("Context", (), {"mutation_capable": True})()

        def record_mutation(self, *, uncertain=False):
            del uncertain
            self.mutation_epoch += 1
            return self.mutation_epoch

    uow = Uow()

    def request(operation, payload, epoch, *, chain="on"):
        return MutationRequest(
            operation,
            MutationGuard(
                parent_uuid,
                "completed",
                "chain-service",
                7,
                recurrence_fingerprint(parent),
                (GuardTimestamp(GuardTimestampField.MODIFIED, parent["modified"]),),
                epoch,
                chain,
            ),
            payload,
        )

    service = TaskwarriorMutationService(uow)
    snapshot_parent = dict(parent)
    stale_request = request(MutationOperation.CHAIN_DISABLE, ChainDisablePayload(parent_uuid), 0)
    parent["modified"] = "20260813T100001Z"
    stale = service.apply(stale_request)
    expect(stale.kind is MutationOutcomeKind.CONFLICT, f"modified parent was not rejected: {stale}")
    expect(not uow.client.calls, "stale parent guard reached the mutation command")
    parent.clear()
    parent.update(snapshot_parent)
    uow.repository.rows.pop(parent_uuid)
    deleted = service.apply(request(MutationOperation.CHAIN_DISABLE, ChainDisablePayload(parent_uuid), 0))
    expect(deleted.kind in {MutationOutcomeKind.CONFLICT, MutationOutcomeKind.RETRYABLE}, f"deleted parent was applied: {deleted}")
    expect(not uow.client.calls, "deleted parent reached the mutation command")
    uow.repository.rows[parent_uuid] = parent
    parent["status"] = "pending"
    completion_changed = service.apply(stale_request)
    expect(completion_changed.kind is MutationOutcomeKind.CONFLICT, f"changed completion state was not rejected: {completion_changed}")
    expect(not uow.client.calls, "changed completion state reached the mutation command")
    parent["status"] = "completed"

    child = _child_payload_from_values(
        {
            "uuid": child_uuid,
            "chainID": "chain-service",
            "link": 8,
            "prevLink": parent_uuid[:8],
            "description": "service child",
            "status": "pending",
            "chain": "on",
            "modified": "20260813T100000Z",
            "anchor": "w:mon",
            "cp": "1d",
        },
        parent_uuid=parent_uuid,
    )
    imported = service.apply(request(MutationOperation.CHILD_IMPORT, child, 0))
    expect(imported.kind is MutationOutcomeKind.APPLIED, f"child import was not applied: {imported}")
    uow.repository.rows[child_uuid]["link"] = 8.0
    replay = service.apply(request(MutationOperation.CHILD_IMPORT, child, 1))
    expect(replay.kind is MutationOutcomeKind.ALREADY_APPLIED, f"numeric child link replay was not normalized: {replay}")
    link_payload = ParentLinkPayload(parent_uuid, child_uuid[:8])
    baseline_parent_link = request(MutationOperation.PARENT_LINK, link_payload, 1)

    # Child identity replacement race: an existing UUID with changed chain
    # identity or UUID payload is not treated as the requested child.
    for field, value in (("uuid", "00000000-0000-4000-8000-000000000926"), ("chainID", "user-child-chain"), ("link", 99), ("prevLink", "user-edit")):
        original = uow.repository.rows[child_uuid].get(field)
        uow.repository.rows[child_uuid][field] = value
        calls_before = len(uow.client.calls)
        raced_child_identity = service.apply(request(MutationOperation.CHILD_IMPORT, child, 1))
        expect(raced_child_identity.kind in {
            MutationOutcomeKind.CONFLICT,
            MutationOutcomeKind.RETRYABLE,
            MutationOutcomeKind.MANUAL_REVIEW,
        },
               f"user edit of child identity {field} was not rejected: {raced_child_identity}")
        expect(len(uow.client.calls) == calls_before, f"child identity {field} race reached Taskwarrior")
        if original is None:
            uow.repository.rows[child_uuid].pop(field, None)
        else:
            uow.repository.rows[child_uuid][field] = original

    # User-edit race matrix: every guarded parent identity change must stop a
    # stale mutation before it reaches Taskwarrior.
    parent_guard_fields = (
        ("status", "pending"),
        ("chain", "off"),
        ("chainID", "user-chain"),
        ("link", 8),
        ("anchor", "w:tue"),
        ("cp", "2d"),
        ("modified", "20260813T100002Z"),
    )
    for field, value in parent_guard_fields:
        original = parent.get(field)
        parent[field] = value
        calls_before = len(uow.client.calls)
        raced = service.apply(baseline_parent_link)
        expect(raced.kind in {MutationOutcomeKind.CONFLICT, MutationOutcomeKind.RETRYABLE},
               f"user edit of parent {field} was not rejected: {raced}")
        expect(len(uow.client.calls) == calls_before, f"parent {field} race reached Taskwarrior")
        if original is None:
            parent.pop(field, None)
        else:
            parent[field] = original

    parent["nextLink"] = "user-edit"
    calls_before = len(uow.client.calls)
    raced_link = service.apply(baseline_parent_link)
    expect(raced_link.kind in {MutationOutcomeKind.CONFLICT, MutationOutcomeKind.RETRYABLE},
           f"user edit of parent nextLink was not rejected: {raced_link}")
    expect(len(uow.client.calls) == calls_before, "parent nextLink race reached Taskwarrior")
    parent.pop("nextLink", None)

    linked = service.apply(request(MutationOperation.PARENT_LINK, link_payload, 1))
    expect(linked.kind is MutationOutcomeKind.APPLIED, f"parent link was not applied: {linked}")
    # Taskwarrior updates ``modified`` when the link succeeds.  Recovery can
    # therefore see the desired nextLink with a newer timestamp before the
    # outbox stage was persisted; that state must converge idempotently.
    parent["modified"] = "20260813T100001Z"
    recovered = service.apply(
        MutationRequest(
            MutationOperation.PARENT_LINK,
            MutationGuard(
                parent_uuid,
                "completed",
                "chain-service",
                7,
                recurrence_fingerprint(parent),
                (GuardTimestamp(GuardTimestampField.MODIFIED, "20260813T100000Z"),),
                2,
                "on",
            ),
            link_payload,
        )
    )
    expect(
        recovered.kind is MutationOutcomeKind.ALREADY_APPLIED,
        f"parent link recovery did not converge after modified changed: {recovered}",
    )
    cleared = service.apply(
        MutationRequest(
            MutationOperation.PARENT_LINK_CLEAR,
            MutationGuard(
                parent_uuid,
                "completed",
                "chain-service",
                7,
                recurrence_fingerprint(parent),
                (GuardTimestamp(GuardTimestampField.MODIFIED, parent["modified"]),),
                2,
                "on",
            ),
            ParentLinkClearPayload(parent_uuid, child_uuid[:8]),
        )
    )
    expect(cleared.kind is MutationOutcomeKind.APPLIED, f"parent link clear was not applied: {cleared}")
    disabled = service.apply(request(MutationOperation.CHAIN_DISABLE, ChainDisablePayload(parent_uuid), 3))
    expect(disabled.kind is MutationOutcomeKind.APPLIED, f"chain disablement was not applied: {disabled}")
    replay = service.apply(request(MutationOperation.CHAIN_DISABLE, ChainDisablePayload(parent_uuid), 4))
    expect(replay.kind is MutationOutcomeKind.ALREADY_APPLIED, f"chain replay was not idempotent: {replay}")
    parent["until"] = "20260813T200000Z"
    native = service.apply(
        request(
            MutationOperation.NATIVE_UNTIL_REPAIR,
            NativeUntilRepairPayload(parent_uuid, parent["until"], "20260814T200000Z"),
            4,
            chain="off",
        )
    )
    expect(native.kind is MutationOutcomeKind.APPLIED, f"native-until repair was not applied: {native}")
    metadata = service.apply(
        request(
            MutationOperation.METADATA_REPAIR,
            _metadata_payload_from_values(parent_uuid, {"chainMax": "5"}, expected={"chainMax": ""}),
            5,
            chain="off",
        )
    )
    expect(metadata.kind is MutationOutcomeKind.APPLIED, f"metadata repair was not applied: {metadata}")
    child_row = uow.repository.rows[child_uuid]
    child_guard = MutationGuard(
        child_uuid,
        "pending",
        "chain-service",
        8,
        recurrence_fingerprint(child_row),
        (GuardTimestamp(GuardTimestampField.MODIFIED, child_row["modified"]),),
        6,
        "on",
    )
    for field, value in (("status", "completed"), ("modified", "20260813T100002Z")):
        original = child_row.get(field)
        child_row[field] = value
        calls_before = len(uow.client.calls)
        raced_child = service.apply(
            MutationRequest(MutationOperation.CHILD_COMPENSATION, child_guard, ChildCompensationPayload(child_uuid))
        )
        expect(raced_child.kind in {MutationOutcomeKind.CONFLICT, MutationOutcomeKind.RETRYABLE},
               f"user edit of child {field} was not rejected: {raced_child}")
        expect(len(uow.client.calls) == calls_before, f"child {field} race reached Taskwarrior")
        if original is None:
            child_row.pop(field, None)
        else:
            child_row[field] = original
    compensated = service.apply(
        MutationRequest(
            MutationOperation.CHILD_COMPENSATION,
            child_guard,
            ChildCompensationPayload(child_uuid),
        )
    )
    expect(compensated.kind is MutationOutcomeKind.APPLIED, f"child compensation was not applied: {compensated}")
    replay_compensation = service.apply(
        MutationRequest(
            MutationOperation.CHILD_COMPENSATION,
            MutationGuard(
                child_uuid,
                "pending",
                "chain-service",
                8,
                child_guard.recurrence_identity,
                child_guard.timestamps,
                7,
                "on",
            ),
            ChildCompensationPayload(child_uuid),
        )
    )
    expect(
        replay_compensation.kind is MutationOutcomeKind.ALREADY_APPLIED,
        f"child compensation replay was not idempotent: {replay_compensation}",
    )
    modify_calls = [args for args, purpose in uow.client.calls if "modify" in args]
    expect(modify_calls, "lifecycle mutation test did not exercise a modify command")
    expect(
        all(args and args[0] == "rc.hooks=off" for args in modify_calls),
        f"lifecycle modify commands must disable hooks: {modify_calls}",
    )
    uow.repository.unavailable = True
    unavailable = service.apply(request(MutationOperation.CHAIN_DISABLE, ChainDisablePayload(parent_uuid), 7))
    expect(unavailable.kind is MutationOutcomeKind.RETRYABLE, f"unavailable guard was not retryable: {unavailable}")
    expect(len(uow.client.calls) == 7, f"unexpected Taskwarrior mutation count: {uow.client.calls}")


def test_child_import_rejects_incomplete_existing_rows():
    """A matching UUID is not enough to acknowledge a malformed child row."""
    from nautical_core.integration_models import (
        Absent, Found, GuardTimestamp, GuardTimestampField,
        MutationGuard, MutationOperation, MutationOutcomeKind, MutationRequest,
    )
    from nautical_core.lifecycle_models import recurrence_fingerprint
    from nautical_core.task_codec import DEFAULT_TASK_CODEC
    from nautical_core.taskwarrior_mutations import TaskwarriorMutationService

    parent_uuid = "00000000-0000-4000-8000-000000000926"
    child_uuid = "00000000-0000-4000-8000-000000000927"
    parent = {
        "uuid": parent_uuid,
        "status": "completed",
        "chain": "on",
        "chainID": "chain-child-check",
        "link": 4,
        "modified": "20260813T110000Z",
    }
    payload_map = {
        "uuid": child_uuid,
        "chainID": "chain-child-check",
        "link": 5,
        "prevLink": parent_uuid[:8],
        "status": "pending",
        "chain": "on",
        "cp": "1d",
    }

    class Repo:
        def __init__(self, rows):
            self.rows = rows

        def by_uuid(self, uuid_value, *, refresh=False):
            del refresh
            row = self.rows.get(str(uuid_value).lower())
            if row is None:
                return Absent(f"uuid:{uuid_value}", "not present")
            return Found(
                DEFAULT_TASK_CODEC.decode_row(row, source_query=f"uuid:{uuid_value}"),
                f"uuid:{uuid_value}",
            )

        def exact_child_slot(
            self,
            chain_id,
            link,
            *,
            statuses=(),
            expected_prev_link="",
            complete_chain_history=False,
            refresh=False,
        ):
            del complete_chain_history, refresh
            wanted_statuses = {str(value).lower() for value in statuses}
            for row in self.rows.values():
                if str(row.get("chainID") or "") != str(chain_id):
                    continue
                try:
                    if int(float(row.get("link"))) != int(link):
                        continue
                except (TypeError, ValueError):
                    continue
                if wanted_statuses and str(row.get("status") or "").lower() not in wanted_statuses:
                    continue
                if expected_prev_link and str(row.get("prevLink") or "") != str(expected_prev_link):
                    continue
                return Found(
                    DEFAULT_TASK_CODEC.decode_row(row, source_query=f"chainID:{chain_id} link:{link}"),
                    f"chainID:{chain_id} link:{link}",
                )
            return Absent(f"chainID:{chain_id} link:{link}", "not present")

    class Uow:
        def __init__(self, rows):
            self.repository = Repo(rows)
            self.mutation_epoch = 0
            self.context = type("Context", (), {"mutation_capable": True})()

        def record_mutation(self, *, uncertain=False):
            del uncertain
            self.mutation_epoch += 1
            return self.mutation_epoch

    for label, mutate in (
        ("missing prevLink", lambda row: row.pop("prevLink")),
        ("wrong status", lambda row: row.update(status="completed")),
        ("disabled chain", lambda row: row.update(chain="off")),
        ("changed recurrence metadata", lambda row: row.update(cp="2d")),
        ("sync replacement", lambda row: row.update(chainID="replacement-chain", link=99)),
    ):
        child = dict(payload_map)
        mutate(child)
        uow = Uow({parent_uuid: dict(parent), child_uuid: child})
        service = TaskwarriorMutationService(uow)
        payload = _child_payload_from_values(payload_map, parent_uuid=parent_uuid)
        guard = MutationGuard(
            parent_uuid,
            "completed",
            "chain-child-check",
            4,
            recurrence_fingerprint(parent),
            (GuardTimestamp(GuardTimestampField.MODIFIED, parent["modified"]),),
            0,
            "on",
        )
        outcome = service.apply(MutationRequest(MutationOperation.CHILD_IMPORT, guard, payload))
        expect(
            outcome.kind is MutationOutcomeKind.CONFLICT and not outcome.postconditions,
            f"{label} was incorrectly accepted as imported: {outcome}",
        )

    expired_payload_map = dict(payload_map, status="deleted", until="20200101T000000Z")
    expired_child = dict(expired_payload_map)
    expired_uow = Uow({parent_uuid: dict(parent), child_uuid: expired_child})
    expired_payload = _child_payload_from_values(expired_payload_map, parent_uuid=parent_uuid)
    expired_outcome = TaskwarriorMutationService(expired_uow).apply(
        MutationRequest(MutationOperation.CHILD_IMPORT, guard, expired_payload)
    )
    expect(
        expired_outcome.kind is MutationOutcomeKind.ALREADY_APPLIED,
        f"already-expired imported child was not accepted: {expired_outcome}",
    )

    future_payload_map = dict(payload_map, status="deleted", until="29990101T000000Z")
    future_child = dict(future_payload_map)
    future_uow = Uow({parent_uuid: dict(parent), child_uuid: future_child})
    future_payload = _child_payload_from_values(future_payload_map, parent_uuid=parent_uuid)
    future_outcome = TaskwarriorMutationService(future_uow).apply(
        MutationRequest(MutationOperation.CHILD_IMPORT, guard, future_payload)
    )
    expect(
        future_outcome.kind is MutationOutcomeKind.CONFLICT,
        f"future-dated deleted child was incorrectly accepted: {future_outcome}",
    )

    existing_uow = Uow({parent_uuid: dict(parent), child_uuid: dict(payload_map)})
    existing_outcome = TaskwarriorMutationService(existing_uow).apply(
        MutationRequest(MutationOperation.CHILD_IMPORT, guard, _child_payload_from_values(payload_map, parent_uuid=parent_uuid))
    )
    expect(
        existing_outcome.kind is MutationOutcomeKind.ALREADY_APPLIED,
        f"child created between snapshot and apply was not acknowledged idempotently: {existing_outcome}",
    )


def test_lifecycle_child_prefetch_reuses_one_authoritative_snapshot():
    """Batch child-absence checks avoid duplicate pre-import UUID exports safely."""
    from nautical_core.integration_models import (
        Absent, Found, GuardTimestamp, GuardTimestampField,
        MutationGuard,
    )
    from nautical_core.lifecycle_models import recurrence_fingerprint
    from nautical_core.taskwarrior_mutations import TaskwarriorMutationService

    parent_uuid = "00000000-0000-4000-8000-000000000928"
    child_uuid = "00000000-0000-4000-8000-000000000929"
    parent = {
        "uuid": parent_uuid,
        "status": "completed",
        "chain": "on",
        "chainID": "prefetch-chain",
        "link": 1,
        "due": "20260703T110000Z",
        "modified": "20260813T120000Z",
    }
    child = {
        "uuid": child_uuid,
        "chainID": "prefetch-chain",
        "link": 2,
        "prevLink": parent_uuid[:8],
        "status": "pending",
        "chain": "on",
        "cp": "1d",
    }
    payload = _child_payload_from_values(child, parent_uuid=parent_uuid)

    class Snapshot:
        def uuid_matches(self, uuid_value):
            return (parent,) if str(uuid_value).lower() == parent_uuid else ()

    class Repo:
        def __init__(self):
            self.uuid_calls = []
            self.broad_calls = 0
            self.set_calls = 0

        def by_uuid(self, uuid_value, *, refresh=False):
            del refresh
            self.uuid_calls.append(str(uuid_value))
            row = parent if str(uuid_value).lower() == parent_uuid else None
            return Found(row, f"uuid:{uuid_value}") if row is not None else Absent(f"uuid:{uuid_value}", "not present")

        def broad_snapshot(self, **kwargs):
            self.broad_calls += 1
            del kwargs
            return Found(Snapshot(), "broad:lifecycle-child-prefetch")

        def read_uuid_set(self, request):
            from nautical_core.task_set_reads import SetReadResult, SetReadStatus

            self.set_calls += 1
            return SetReadResult(
                SetReadStatus.COMPLETE,
                request.uuids,
                found={parent_uuid: parent},
                absent=tuple(identity for identity in request.uuids if identity != parent_uuid),
                complete_for_requested_identities=True,
            )

    class Uow:
        def __init__(self):
            self.repository = Repo()
            self.mutation_epoch = 0
            self.client = None

        def record_mutation(self, *, uncertain=False):
            del uncertain
            self.mutation_epoch += 1
            return self.mutation_epoch

    uow = Uow()
    service = TaskwarriorMutationService(uow)
    service.preflight_lifecycle_batch((payload,), parent_expectations=((parent_uuid, child_uuid[:8]),))
    guard = MutationGuard(
        parent_uuid,
        "completed",
        "prefetch-chain",
        1,
        recurrence_fingerprint(parent),
        (GuardTimestamp(GuardTimestampField.MODIFIED, parent["modified"]),),
        0,
        "on",
    )
    expect(
        child_uuid.lower() in service._prefetched_children,
        "authoritative absent child was not retained for the import decision",
    )
    expect(
        service._prefetched_parents.get(parent_uuid) == parent,
        "pre-mutation parent row was not retained for the guarded link decision",
    )
    expect(uow.repository.set_calls == 1, f"prefetch used {uow.repository.set_calls} targeted set reads")
    expect(uow.repository.broad_calls == 0, f"prefetch used {uow.repository.broad_calls} broad reads")
    expect(uow.repository.uuid_calls == [], f"child UUID was redundantly exported: {uow.repository.uuid_calls}")


def test_lifecycle_batch_prefetch_uses_one_union_set_read():
    """All child slots and parent guards share one bounded set read."""
    from nautical_core.integration_models import ChildImportPayload
    from nautical_core.taskwarrior_mutations import TaskwarriorMutationService
    from nautical_core.task_set_reads import SetReadResult, SetReadStatus

    parent_uuids = tuple(f"00000000-0000-4000-8000-00000000093{i}" for i in range(0, 3))
    child_uuids = tuple(f"00000000-0000-4000-8000-00000000094{i}" for i in range(0, 3))
    parents = {
        uuid: {
            "uuid": uuid,
            "status": "completed",
            "chain": "on",
            "chainID": "prefetch-batch",
            "link": index + 1,
            "modified": "20260813T120000Z",
        }
        for index, uuid in enumerate(parent_uuids)
    }
    payloads = tuple(
        ChildImportPayload(
            parent_uuid=parent_uuid,
            child_uuid=child_uuid,
            chain_id="prefetch-batch",
            target_link=index + 2,
            fields=(
                ("uuid", child_uuid),
                ("chainID", "prefetch-batch"),
                ("link", index + 2),
                ("prevLink", parent_uuid[:8]),
            ),
        )
        for index, (parent_uuid, child_uuid) in enumerate(zip(parent_uuids, child_uuids))
    )

    class Repo:
        def __init__(self):
            self.requests = []

        def read_uuid_set(self, request):
            self.requests.append(request)
            return SetReadResult(
                SetReadStatus.COMPLETE,
                request.uuids,
                found=dict(parents),
                absent=tuple(identity for identity in request.uuids if identity not in parents),
                complete_for_requested_identities=True,
            )

    class Uow:
        mutation_epoch = 0

        def __init__(self):
            self.repository = Repo()

    uow = Uow()
    TaskwarriorMutationService(uow).preflight_lifecycle_batch(
        payloads,
        parent_expectations=tuple((uuid, f"{index + 2:08x}") for index, uuid in enumerate(parent_uuids)),
    )
    expect(len(uow.repository.requests) == 1, f"batch preflight used {len(uow.repository.requests)} set reads")
    expect(
        set(uow.repository.requests[0].uuids) == set(parent_uuids + child_uuids),
        f"batch preflight requested the wrong identities: {uow.repository.requests[0].uuids}",
    )


def test_lifecycle_batch_postverification_fails_closed_on_unavailable_snapshot():
    """A failed phase snapshot cannot be mistaken for a verified mutation."""
    from nautical_core.integration_models import (
        CommandFailureKind, FailureEvidence, Found, GuardTimestamp,
        GuardTimestampField, MutationGuard, MutationOperation, MutationOutcomeKind,
        MutationRequest, ParentLinkPayload, TaskCommand, Unavailable,
    )
    from nautical_core.lifecycle_models import recurrence_fingerprint
    from nautical_core.taskwarrior_mutations import TaskwarriorMutationService

    parent_uuid = "00000000-0000-4000-8000-000000000930"
    child_uuid = "00000000-0000-4000-8000-000000000931"
    parent = {
        "uuid": parent_uuid,
        "status": "completed",
        "chain": "on",
        "chainID": "batch-fail-closed",
        "link": 1,
        "modified": "20260813T120000Z",
        "cp": "1d",
    }
    child = _child_payload_from_values(
        {
            "uuid": child_uuid,
            "chainID": "batch-fail-closed",
            "link": 2,
            "prevLink": parent_uuid[:8],
            "status": "pending",
            "chain": "on",
            "cp": "1d",
        },
        parent_uuid=parent_uuid,
    )
    from nautical_core.taskwarrior_mutations import _child_import_matches
    null_child = _child_payload_from_values(
        {
            "uuid": child_uuid,
            "chainID": "batch-fail-closed",
            "link": 2,
            "prevLink": parent_uuid[:8],
            "status": "pending",
            "chain": "on",
            "cp": "1d",
            "anchor_file": "null",
        },
        parent_uuid=parent_uuid,
    )
    null_row = null_child.to_dict()
    null_row.pop("anchor_file", None)
    expect(
        _child_import_matches(null_row, null_child, parent_uuid),
        "literal null recurrence UDA should not fail child postcondition verification",
    )

    class Repo:
        def __init__(self, mode):
            self.mode = mode

        def broad_snapshot(self, **kwargs):
            del kwargs
            command = TaskCommand(("task", "export"), "batch verification", 1.0)
            if self.mode == "unavailable":
                evidence = FailureEvidence(command, CommandFailureKind.BUSY, 1, 1, 0.01, True, "lock active")
                return Unavailable("broad:lifecycle-postverify", evidence)
            if self.mode == "malformed":
                return Found(object(), "broad:malformed")
            child_row = child.to_dict()
            parent_row = dict(parent)
            if self.mode == "stale":
                child_row["link"] = 99
                parent_row["nextLink"] = "stale00"
            rows = {
                "child": (child_row, child_row),
                "parent": (parent_row, parent_row),
            }

            class Snapshot:
                def uuid_matches(self, uuid_value):
                    if str(uuid_value).lower() == child_uuid:
                        return rows["child"] if self_mode == "conflict" else (rows["child"][0],)
                    if str(uuid_value).lower() == parent_uuid:
                        return rows["parent"] if self_mode == "conflict" else (rows["parent"][0],)
                    return ()

            self_mode = self.mode
            return Found(Snapshot(), f"broad:{self.mode}")

        def read_uuid_set(self, request):
            from nautical_core.task_set_reads import SetReadResult, SetReadStatus

            command = TaskCommand(("task", "export"), "batch verification", 1.0)
            if self.mode == "unavailable":
                evidence = FailureEvidence(command, CommandFailureKind.BUSY, 1, 1, 0.01, True, "lock active")
                return SetReadResult(SetReadStatus.UNAVAILABLE, request.uuids, failures=(evidence,))
            if self.mode == "malformed":
                return SetReadResult(SetReadStatus.MALFORMED, request.uuids, evidence=("malformed set",))
            child_row = child.to_dict()
            parent_row = dict(parent)
            if self.mode == "stale":
                child_row["link"] = 99
                parent_row["nextLink"] = "stale00"
            if self.mode == "conflict":
                return SetReadResult(
                    SetReadStatus.DUPLICATE,
                    request.uuids,
                    found={child_uuid: child_row, parent_uuid: parent_row},
                    evidence=("duplicate identity",),
                )
            return SetReadResult(
                SetReadStatus.COMPLETE,
                request.uuids,
                found={child_uuid: child_row, parent_uuid: parent_row},
                complete_for_requested_identities=True,
            )

    class Uow:
        def __init__(self, mode):
            self.repository = Repo(mode)
        mutation_epoch = 0
        client = None

    guard = MutationGuard(
        parent_uuid,
        "completed",
        "batch-fail-closed",
        1,
        recurrence_fingerprint(parent),
        (GuardTimestamp(GuardTimestampField.MODIFIED, parent["modified"]),),
        0,
        "on",
    )
    link = ParentLinkPayload(parent_uuid, child_uuid[:8])
    for mode, expected in (("unavailable", MutationOutcomeKind.RETRYABLE), ("malformed", MutationOutcomeKind.MANUAL_REVIEW), ("stale", MutationOutcomeKind.MANUAL_REVIEW), ("conflict", MutationOutcomeKind.MANUAL_REVIEW)):
        service = TaskwarriorMutationService(Uow(mode))
        child_result = service.verify_lifecycle_children((MutationRequest(MutationOperation.CHILD_IMPORT, guard, child),))
        expect(child_result[child_uuid].kind is expected, f"{mode} child snapshot was misclassified: {child_result}")
        parent_result = service.verify_lifecycle_parents((MutationRequest(MutationOperation.PARENT_LINK, guard, link),))
        expect(parent_result[parent_uuid].kind is expected, f"{mode} parent snapshot was misclassified: {parent_result}")


def test_lifecycle_outbox_persists_typed_plans_and_recovers_claims():
    """The durable outbox owns immutable plans, leases, stages, and poison rows."""
    import threading

    from nautical_core.lifecycle_models import (
        LifecycleAction,
        LifecycleEvent,
        LifecycleIdentity,
        LifecyclePlan,
        ParentGuard,
    )
    from nautical_core.lifecycle_outbox import (
        _LifecycleOutboxRepository,
        OutboxFailure,
        OutboxProcessingState,
        OutboxResultKind,
    )

    now = [1000.0]

    def clock():
        return now[0]

    def plan_for(
        link: int,
        *,
        legacy_null_anchor_file: bool = False,
        child_entry: str = "",
        child_description: str = "",
        numeric_variant: bool = False,
    ) -> LifecyclePlan:
        parent_uuid = f"00000000-0000-4000-8000-{link:012d}"
        child_uuid = f"10000000-0000-4000-8000-{link:012d}"
        child_payload = {
            "uuid": child_uuid,
            "description": child_description or "outbox child",
            "status": "pending",
            "chain": "on",
            "chainID": "outbox-chain",
            "link": link + 1,
            "prevLink": parent_uuid[:8],
            "cp": "1d",
            "due": "20260824T090000Z",
            "numeric_metadata": {"slot": link + 1},
        }
        if legacy_null_anchor_file:
            child_payload["anchor_file"] = None
        if numeric_variant:
            child_payload["link"] = float(link + 1)
            child_payload["numeric_metadata"] = {"slot": float(link + 1)}
        if child_entry:
            child_payload["entry"] = child_entry
        if child_description:
            child_payload["description"] = child_description
        return LifecyclePlan.from_draft(
            identity=LifecycleIdentity("outbox-chain", parent_uuid, link, link + 1, LifecycleEvent.COMPLETE),
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=ParentGuard("completed", "on", "outbox-chain", link, "rf1-test"),
            draft=_task_draft(child_payload),
            parent_patch={"nextLink": child_uuid[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
        )

    with tempfile.TemporaryDirectory() as td:
        repo = _LifecycleOutboxRepository(Path(td), clock=clock)
        plan = plan_for(1)
        first = repo.enqueue(plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1")
        expect(first.kind is OutboxResultKind.APPLIED, f"outbox enqueue failed: {first}")
        # Verify durable plan decoding and claiming from a fresh interpreter,
        # rather than only reopening the repository in-process.
        restart_plan = plan_for(30)
        restart = repo.enqueue(restart_plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1")
        expect(restart.ok, "process-restart lifecycle intent could not be staged")
        restart_script = """
import json
import sys
from pathlib import Path
from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository

repository = _LifecycleOutboxRepository(Path(sys.argv[1]))
result = repository.claim_intent(
    owner="fresh-process",
    lease_seconds=5,
    intent_id=sys.argv[2],
)
if not result.ok or result.record is None:
    raise SystemExit(f"fresh process could not claim lifecycle intent: {result!r}")
record = result.record
print(json.dumps({"semantic_key": record.plan.semantic_key(), "stage": record.stage.value}))
"""
        restart_process = subprocess.run(
            [sys.executable, "-c", restart_script, td, restart_plan.identity.idempotency_key],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": ROOT},
            capture_output=True,
            text=True,
            check=False,
        )
        expect(
            restart_process.returncode == 0,
            f"fresh lifecycle process failed: {restart_process.stderr!r}",
        )
        try:
            restart_evidence = json.loads(restart_process.stdout)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"fresh lifecycle process returned invalid evidence: {restart_process.stdout!r}"
            ) from exc
        expect(
            restart_evidence == {"semantic_key": restart_plan.semantic_key(), "stage": "planned"},
            f"fresh lifecycle process changed the persisted plan: {restart_evidence!r}",
        )
        duplicate = repo.enqueue(plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1")
        expect(duplicate.kind is OutboxResultKind.ALREADY_APPLIED, "outbox duplicate enqueue was not idempotent")
        fingerprint_drift = repo.enqueue(plan, configuration_fingerprint="cf2", schedule_fingerprint="sf1")
        expect(fingerprint_drift.kind is OutboxResultKind.ALREADY_APPLIED, "queued intent was blocked by fingerprint drift")
        conflict = repo.enqueue(
            plan_for(1, child_description="different immutable child"),
            configuration_fingerprint="cf1",
            schedule_fingerprint="sf1",
        )
        expect(conflict.kind is OutboxResultKind.CONFLICT, "outbox accepted divergent immutable intent")

        claimed, records = repo.claim_batch(owner="first-worker", lease_seconds=5, limit=10)
        expect(claimed.ok and len(records) == 1, f"outbox claim failed: {claimed}, {records}")
        intent_id = records[0].intent_id
        child = repo.advance_stage(intent_id=intent_id, owner="first-worker", stage="child_present")
        expect(child.ok, f"outbox child stage failed: {child}")
        released = repo.release_retry(
            intent_id=intent_id,
            owner="first-worker",
            failure=OutboxFailure("task_busy", "Taskwarrior lock active"),
        )
        expect(released.ok, f"outbox retry release failed: {released}")
        now[0] += 1
        reclaimed, records = repo.claim_batch(owner="second-worker", lease_seconds=5, limit=10)
        expect(reclaimed.ok and len(records) == 1, f"outbox retry claim failed: {reclaimed}, {records}")
        record = records[0]
        expect(record.stage.value == "child_present", "outbox retry lost verified child progress")
        expect(record.attempts == 2, "outbox retry did not increment attempts")
        expect(repo.advance_stage(intent_id=intent_id, owner="second-worker", stage="parent_linked").ok, "outbox parent stage failed")
        expect(repo.advance_stage(intent_id=intent_id, owner="second-worker", stage="verified").ok, "outbox verification stage failed")
        expect(repo.acknowledge(intent_id=intent_id, owner="second-worker").ok, "outbox acknowledgement failed")
        replay = repo.enqueue(plan, configuration_fingerprint="new-config", schedule_fingerprint="new-schedule")
        expect(replay.kind is OutboxResultKind.ALREADY_APPLIED, "acknowledged intent was blocked by fingerprint drift")

        legacy = plan_for(2, legacy_null_anchor_file=True)
        expect(
            repo.enqueue(legacy, configuration_fingerprint="cf1", schedule_fingerprint="sf1").kind
            is OutboxResultKind.APPLIED,
            "legacy null lifecycle intent could not be staged",
        )
        converged = repo.enqueue(plan_for(2), configuration_fingerprint="cf1", schedule_fingerprint="sf1")
        expect(
            converged.kind is OutboxResultKind.ALREADY_APPLIED,
            "legacy null lifecycle intent was treated as an immutable conflict",
        )
        old_entry = repo.enqueue(
            plan_for(3, child_entry="20260820T200000Z"),
            configuration_fingerprint="cf1",
            schedule_fingerprint="sf1",
        )
        expect(old_entry.kind is OutboxResultKind.APPLIED, "volatile-entry lifecycle intent could not be staged")
        new_entry = repo.enqueue(
            plan_for(3, child_entry="20260820T210000Z"),
            configuration_fingerprint="cf1",
            schedule_fingerprint="sf1",
        )
        expect(new_entry.kind is OutboxResultKind.ALREADY_APPLIED, "entry timestamp caused a lifecycle conflict")
        numeric = repo.enqueue(
            plan_for(11),
            configuration_fingerprint="cf1",
            schedule_fingerprint="sf1",
        )
        expect(numeric.kind is OutboxResultKind.APPLIED, "numeric lifecycle intent could not be staged")
        numeric_variant = repo.enqueue(
            plan_for(11, numeric_variant=True),
            configuration_fingerprint="cf1",
            schedule_fingerprint="sf1",
        )
        expect(
            numeric_variant.kind is OutboxResultKind.ALREADY_APPLIED,
            "numeric JSON representation caused a lifecycle conflict",
        )
        expect(numeric.record is not None, "numeric lifecycle intent lost its durable record")
        numeric_claim = repo.claim_intent(
            owner="numeric-cleanup",
            lease_seconds=5,
            intent_id=numeric.record.intent_id,
        )
        expect(numeric_claim.ok, "numeric lifecycle intent cleanup claim failed")
        expect(
            repo.manual_review(
                intent_id=numeric.record.intent_id,
                owner="numeric-cleanup",
                failure=OutboxFailure("test_cleanup", "numeric representation test complete"),
            ).ok,
            "numeric lifecycle intent cleanup failed",
        )
        manual_plan = plan_for(20)
        manual_staged = repo.enqueue(manual_plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1")
        expect(manual_staged.ok and manual_staged.record is not None, "manual intent enqueue failed")
        manual_claim = repo.claim_intent(
            owner="manual-owner", lease_seconds=5, intent_id=manual_staged.record.intent_id
        )
        expect(manual_claim.ok and manual_claim.record is not None, "manual intent claim failed")
        expect(
            repo.manual_review(
                intent_id=manual_staged.record.intent_id,
                owner="manual-owner",
                failure=OutboxFailure("mutation_conflict", "postcondition does not match"),
            ).ok,
            "manual intent setup failed",
        )
        reopened = repo.enqueue(manual_plan, configuration_fingerprint="new-config", schedule_fingerprint="sf1")
        expect(reopened.kind is OutboxResultKind.APPLIED, "known stale postcondition review was not reopened")
        expect(reopened.record is not None and reopened.record.state is OutboxProcessingState.RETRY, "reopened intent was not retryable")
        cleanup_claim = repo.claim_intent(owner="cleanup-owner", lease_seconds=5, intent_id=manual_staged.record.intent_id)
        expect(cleanup_claim.ok, "reopened intent cleanup claim failed")
        for stage in ("child_present", "parent_linked", "verified"):
            expect(
                repo.advance_stage(intent_id=manual_staged.record.intent_id, owner="cleanup-owner", stage=stage).ok,
                f"reopened intent cleanup could not reach {stage}",
            )
        expect(repo.acknowledge(intent_id=manual_staged.record.intent_id, owner="cleanup-owner").ok, "reopened intent cleanup failed")

        rejected_plan = plan_for(22)
        rejected_staged = repo.enqueue(rejected_plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1")
        expect(rejected_staged.ok and rejected_staged.record is not None, "rejected intent enqueue failed")
        rejected_claim = repo.claim_intent(
            owner="rejected-owner",
            lease_seconds=5,
            intent_id=rejected_staged.record.intent_id,
        )
        expect(rejected_claim.ok, "rejected intent claim failed")
        expect(
            repo.manual_review(
                intent_id=rejected_staged.record.intent_id,
                owner="rejected-owner",
                failure=OutboxFailure("mutation_rejected", "parent link command failed"),
            ).ok,
            "rejected intent review setup failed",
        )
        reopened_rejected = repo.enqueue(
            rejected_plan,
            configuration_fingerprint="cf1",
            schedule_fingerprint="sf1",
        )
        expect(reopened_rejected.kind is OutboxResultKind.APPLIED, "rejected mutation intent was not reopened")
        expect(reopened_rejected.record is not None, "reopened rejected intent lost its durable record")
        rejected_cleanup = repo.claim_intent(
            owner="rejected-cleanup",
            lease_seconds=5,
            intent_id=rejected_staged.record.intent_id,
        )
        expect(rejected_cleanup.ok, "reopened rejected intent cleanup claim failed")
        expect(
            repo.manual_review(
                intent_id=rejected_staged.record.intent_id,
                owner="rejected-cleanup",
                failure=OutboxFailure("test_cleanup", "rejected mutation test complete"),
            ).ok,
            "reopened rejected intent cleanup failed",
        )

        stage_sequences = (
            (),
            ("child_present",),
            ("child_present", "parent_linked"),
            ("child_present", "parent_linked", "verified"),
        )
        # Use IDs that do not overlap the legacy/numeric fixtures above.
        for link, prior_stages in enumerate(stage_sequences, start=30):
            staged_plan = plan_for(link)
            expect(
                repo.enqueue(staged_plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1").ok,
                f"stage recovery enqueue failed for link {link}",
            )
            claimed, records = repo.claim_batch(owner=f"stalled-{link}", lease_seconds=5, limit=1)
            expect(claimed.ok and len(records) == 1, f"stage recovery claim failed for link {link}: {claimed}")
            for stage in prior_stages:
                expect(
                    repo.advance_stage(
                        intent_id=records[0].intent_id,
                        owner=f"stalled-{link}",
                        stage=stage,
                    ).ok,
                    f"stage recovery could not persist {stage}",
                )
            now[0] += 6
            recovered, records = repo.claim_batch(owner=f"recovered-{link}", lease_seconds=5, limit=1)
            expect(recovered.ok and len(records) == 1, f"expired lease was not recovered for link {link}")
            expected_stage = prior_stages[-1] if prior_stages else "planned"
            expect(records[0].stage.value == expected_stage, f"recovery lost {expected_stage} progress")
            expect(
                repo.manual_review(
                    intent_id=records[0].intent_id,
                    owner=f"recovered-{link}",
                    failure=OutboxFailure("test_cleanup", "stage recovery test complete"),
                ).ok,
                f"stage recovery cleanup failed for link {link}",
            )

        concurrent_plan = plan_for(7)
        expect(
            repo.enqueue(concurrent_plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1").ok,
            "concurrent claim enqueue failed",
        )
        claim_barrier = threading.Barrier(2)
        concurrent_claims: list[tuple[bool, tuple[object, ...]]] = []
        claims_lock = threading.Lock()

        def claim_once(worker: str) -> None:
            claim_barrier.wait()
            result = _LifecycleOutboxRepository(Path(td), clock=clock).claim_intent(
                owner=worker,
                lease_seconds=5,
                intent_id=concurrent_plan.identity.idempotency_key,
            )
            with claims_lock:
                concurrent_claims.append((result.ok, (result.record,) if result.record is not None else ()))

        workers = [threading.Thread(target=claim_once, args=(f"concurrent-{index}",)) for index in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5)
        claimed_records = [record for ok, records in concurrent_claims if ok for record in records]
        expect(len(concurrent_claims) == 2 and len(claimed_records) == 1, "concurrent outbox claim was not exclusive")

        poison_plan = plan_for(9)
        expect(repo.enqueue(poison_plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1").ok, "poison test enqueue failed")
        with sqlite3.connect(str(repo.path)) as conn:
            conn.execute("UPDATE lifecycle_outbox SET plan_json='{' WHERE intent_id=?", (poison_plan.identity.idempotency_key,))
        now[0] += 1
        claimed = repo.claim_intent(
            owner="poison-worker", lease_seconds=5,
            intent_id=poison_plan.identity.idempotency_key,
        )
        expect(not claimed.ok and claimed.record is None, f"poison outbox row was claimed: {claimed}")
        status_result, status = repo.status()
        expect(status_result.ok, f"outbox status failed: {status_result}")
        expect(
            status.get("states", {}).get(OutboxProcessingState.QUARANTINED.value) == 1,
            f"poison row was not quarantined: {status}",
        )

        tampered_plan = plan_for(10)
        expect(repo.enqueue(tampered_plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1").ok, "guard test enqueue failed")
        with sqlite3.connect(str(repo.path)) as conn:
            conn.execute(
                "UPDATE lifecycle_outbox SET parent_guard_json=? WHERE intent_id=?",
                ('{"chain":"off"}', tampered_plan.identity.idempotency_key),
            )
        claimed = repo.claim_intent(
            owner="integrity-worker", lease_seconds=5,
            intent_id=tampered_plan.identity.idempotency_key,
        )
        expect(not claimed.ok and claimed.record is None, f"tampered immutable row was claimed: {claimed}")
        status_result, status = repo.status()
        expect(status_result.ok, f"outbox integrity status failed: {status_result}")
        expect(
            status.get("states", {}).get(OutboxProcessingState.QUARANTINED.value) == 2,
            f"tampered immutable row was not quarantined: {status}",
        )
        reasons = [record.get("reason", "") for record in status.get("records", [])]
        expect(any("parent guard differs" in reason for reason in reasons), f"integrity reason was not preserved: {status}")


def test_lifecycle_outbox_prunes_only_expired_acknowledged_rows():
    """Explicit retention removes old acknowledgements and preserves live evidence."""
    from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, LifecyclePlan, ParentGuard
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository, OutboxFailure, OutboxProcessingState

    now = [1000.0]

    def plan_for(link: int) -> LifecyclePlan:
        parent_uuid = f"00000000-0000-4000-8000-{link:012d}"
        child_uuid = f"10000000-0000-4000-8000-{link:012d}"
        return _plan_from_values(
            identity=LifecycleIdentity("retention-chain", parent_uuid, link, link + 1, LifecycleEvent.COMPLETE),
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=ParentGuard("completed", "on", "retention-chain", link, f"rf-{link}"),
            child_payload={"uuid": child_uuid, "chainID": "retention-chain", "link": link + 1, "prevLink": parent_uuid[:8]},
            parent_patch={"nextLink": child_uuid[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
        )

    with tempfile.TemporaryDirectory() as td:
        repo = _LifecycleOutboxRepository(Path(td), clock=lambda: now[0])
        acknowledged = plan_for(1)
        expect(repo.enqueue(acknowledged, configuration_fingerprint="cfg", schedule_fingerprint="sch").ok, "ack enqueue failed")
        claimed, records = repo.claim_batch(owner="ack-worker", lease_seconds=30, limit=1)
        expect(claimed.ok and records, "ack claim failed")
        intent_id = records[0].intent_id
        for stage in ("child_present", "parent_linked", "verified"):
            expect(repo.advance_stage(intent_id=intent_id, owner="ack-worker", stage=stage).ok, f"stage {stage} failed")
        expect(repo.acknowledge(intent_id=intent_id, owner="ack-worker").ok, "acknowledgement failed")

        retry_plan = plan_for(2)
        expect(repo.enqueue(retry_plan, configuration_fingerprint="cfg", schedule_fingerprint="sch").ok, "retry enqueue failed")
        _, retry_records = repo.claim_batch(owner="retry-worker", lease_seconds=30, limit=1)
        expect(retry_records, "retry claim failed")
        expect(repo.release_retry(intent_id=retry_records[0].intent_id, owner="retry-worker", failure=OutboxFailure("busy", "lock")).ok, "retry release failed")

        claimed_plan = plan_for(3)
        expect(repo.enqueue(claimed_plan, configuration_fingerprint="cfg", schedule_fingerprint="sch").ok, "claimed enqueue failed")
        _, claimed_records = repo.claim_batch(owner="live-worker", lease_seconds=300, limit=1)
        expect(claimed_records, "live claim failed")

        review_plan = plan_for(4)
        expect(repo.enqueue(review_plan, configuration_fingerprint="cfg", schedule_fingerprint="sch").ok, "review enqueue failed")
        _, review_records = repo.claim_batch(owner="review-worker", lease_seconds=30, limit=1)
        expect(review_records, "review claim failed")
        expect(repo.manual_review(intent_id=review_records[0].intent_id, owner="review-worker", failure=OutboxFailure("review", "inspect")).ok, "manual review failed")

        now[0] += 100.0
        status_result, status = repo.status(limit=20, retention_seconds=50.0)
        expect(status_result.ok, "retention status failed before cleanup")
        retention = status.get("retention") or {}
        expect(retention.get("acknowledged") == 1, f"acknowledged retention count was wrong: {status}")
        expect(retention.get("eligible") == 1, f"eligible retention count was wrong: {status}")
        expect(retention.get("oldest_age_s") == 100, f"oldest retention age was wrong: {status}")
        cleaned = repo.prune_acknowledged(retention_seconds=50.0, limit=10)
        expect(cleaned.ok and cleaned.removed == 1, f"unexpected retention result: {cleaned}")
        status_result, status = repo.status(limit=20)
        expect(status_result.ok, "status after retention failed")
        expect((status.get("retention") or {}).get("eligible") == 0, f"expired retention remained eligible: {status}")
        states = status.get("states", {})
        expect(states.get(OutboxProcessingState.RETRY.value) == 1, f"retry evidence was pruned: {states}")
        expect(states.get(OutboxProcessingState.CLAIMED.value) == 1, f"claimed evidence was pruned: {states}")
        expect(states.get(OutboxProcessingState.MANUAL_REVIEW.value) == 1, f"manual review evidence was pruned: {states}")
        expect(states.get(OutboxProcessingState.ACKNOWLEDGED.value, 0) == 0, f"old acknowledgement remained: {states}")

        # The retention boundary is inclusive: an acknowledgement exactly at
        # the cutoff is eligible, while all non-terminal evidence remains.
        boundary_plan = plan_for(5)
        expect(repo.enqueue(boundary_plan, configuration_fingerprint="cfg", schedule_fingerprint="sch").ok, "boundary enqueue failed")
        _, boundary_records = repo.claim_batch(owner="boundary-worker", lease_seconds=30, limit=1)
        expect(boundary_records, "boundary claim failed")
        boundary_id = boundary_records[0].intent_id
        for stage in ("child_present", "parent_linked", "verified"):
            expect(repo.advance_stage(intent_id=boundary_id, owner="boundary-worker", stage=stage).ok, f"boundary stage {stage} failed")
        expect(repo.acknowledge(intent_id=boundary_id, owner="boundary-worker").ok, "boundary acknowledgement failed")
        with sqlite3.connect(str(repo.path)) as conn:
            conn.execute("UPDATE lifecycle_outbox SET acknowledged_at=? WHERE intent_id=?", (1050.0, boundary_id))
        status_result, status = repo.status(retention_seconds=50.0)
        expect(status_result.ok and (status.get("retention") or {}).get("eligible") == 1, f"cutoff boundary was not eligible: {status}")

        # A read-only status call may race an explicit cleanup without
        # observing a malformed or partially deleted row set.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            status_future = pool.submit(repo.status, retention_seconds=50.0)
            prune_future = pool.submit(repo.prune_acknowledged, retention_seconds=50.0, limit=10)
            concurrent_status, concurrent_data = status_future.result(timeout=5)
            concurrent_prune = prune_future.result(timeout=5)
        expect(concurrent_status.ok and isinstance(concurrent_data, dict), "concurrent retention status failed")
        expect(concurrent_prune.ok and concurrent_prune.removed == 1, f"concurrent retention cleanup failed: {concurrent_prune}")

        # An interrupted delete must roll back atomically and leave the
        # acknowledgement available for a later maintenance attempt.
        interrupted_plan = plan_for(6)
        expect(repo.enqueue(interrupted_plan, configuration_fingerprint="cfg", schedule_fingerprint="sch").ok, "interrupted enqueue failed")
        _, interrupted_records = repo.claim_batch(owner="interrupted-worker", lease_seconds=30, limit=1)
        expect(interrupted_records, "interrupted claim failed")
        interrupted_id = interrupted_records[0].intent_id
        for stage in ("child_present", "parent_linked", "verified"):
            expect(repo.advance_stage(intent_id=interrupted_id, owner="interrupted-worker", stage=stage).ok, f"interrupted stage {stage} failed")
        expect(repo.acknowledge(intent_id=interrupted_id, owner="interrupted-worker").ok, "interrupted acknowledgement failed")
        with sqlite3.connect(str(repo.path)) as conn:
            conn.execute("UPDATE lifecycle_outbox SET acknowledged_at=? WHERE intent_id=?", (1000.0, interrupted_id))
            conn.execute(
                "CREATE TRIGGER reject_outbox_delete BEFORE DELETE ON lifecycle_outbox "
                "BEGIN SELECT RAISE(ABORT, 'simulated interrupted cleanup'); END"
            )
        interrupted = repo.prune_acknowledged(retention_seconds=50.0, limit=10)
        expect(not interrupted.ok, "interrupted cleanup unexpectedly succeeded")
        status_result, status = repo.status(retention_seconds=50.0)
        expect(status_result.ok and (status.get("retention") or {}).get("acknowledged") == 1, f"interrupted cleanup lost evidence: {status}")
        with sqlite3.connect(str(repo.path)) as conn:
            conn.execute("DROP TRIGGER reject_outbox_delete")
        automatic = repo.opportunistic_housekeeping(
            retention_seconds=50.0,
            interval_seconds=0.0,
            limit=1,
            checkpoint=False,
        )
        expect(automatic.ok and automatic.removed == 1, f"automatic housekeeping did not prune: {automatic}")
        deferred = repo.opportunistic_housekeeping(retention_seconds=50.0, checkpoint=False)
        expect(deferred.ok and deferred.skipped and deferred.reason == "cooldown", f"housekeeping cooldown was ignored: {deferred}")


def test_lifecycle_outbox_initialization_is_concurrent_and_rejects_unknown_schema():
    """First-open races are bounded, WAL-backed, and never silently downgrade schema."""
    import threading

    from nautical_core.lifecycle_outbox import (
        _LifecycleOutboxRepository,
        OUTBOX_LEGACY_SCHEMA_VERSION,
        OUTBOX_SCHEMA_VERSION,
        OutboxResultKind,
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        worker = (
            "import sys; from pathlib import Path; "
            "from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository; "
            "result = _LifecycleOutboxRepository(Path(sys.argv[1]), connect_timeout=0.5).open(); "
            "print(result.kind.value, flush=True); raise SystemExit(0 if result.ok else 1)"
        )
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", worker, str(root)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(2)
        ]
        process_results = [process.communicate(timeout=10) for process in processes]
        expect(
            all(process.returncode == 0 and stdout.strip() == "applied" for process, (stdout, _stderr) in zip(processes, process_results)),
            f"concurrent process outbox initialization failed: {process_results}",
        )
        barrier = threading.Barrier(4)
        outcomes = []
        outcomes_lock = threading.Lock()

        def open_repository() -> None:
            barrier.wait()
            result = _LifecycleOutboxRepository(root, connect_timeout=0.1).open()
            with outcomes_lock:
                outcomes.append(result)

        threads = [threading.Thread(target=open_repository) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        expect(len(outcomes) == 4 and all(result.ok for result in outcomes), f"concurrent outbox initialization failed: {outcomes}")

        repo = _LifecycleOutboxRepository(root)
        traced_sql: list[str] = []
        original_connect = repo._connect

        def traced_connect():
            conn = original_connect()
            conn.set_trace_callback(traced_sql.append)
            return conn

        repo._connect = traced_connect
        reopened = repo.open()
        expect(reopened.ok, f"reopening an adopted outbox failed: {reopened}")
        expect(
            not any("PRAGMA journal_mode=WAL" in statement for statement in traced_sql),
            f"reopening an adopted outbox renegotiated WAL: {traced_sql!r}",
        )
        with sqlite3.connect(str(repo.path)) as conn:
            journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            expect(journal_mode == "wal", f"outbox did not retain WAL journal mode: {journal_mode}")
            conn.execute("ALTER TABLE lifecycle_outbox DROP COLUMN work_kind")
            conn.execute(f"PRAGMA user_version={OUTBOX_LEGACY_SCHEMA_VERSION}")
        migrated = repo.open()
        expect(migrated.ok, f"legacy outbox schema did not migrate: {migrated}")
        with sqlite3.connect(str(repo.path)) as conn:
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(lifecycle_outbox)")}
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        expect("work_kind" in columns and version == OUTBOX_SCHEMA_VERSION, "outbox schema v2 migration incomplete")
        with sqlite3.connect(str(repo.path)) as conn:
            conn.execute(f"PRAGMA user_version={OUTBOX_SCHEMA_VERSION + 1}")
        rejected = repo.open()
        expect(rejected.kind is OutboxResultKind.REJECTED, f"future outbox schema was accepted: {rejected}")
        expect("newer than supported" in rejected.reason, f"future schema rejection was not actionable: {rejected}")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        path = root / ".nautical-state" / ".nautical_lifecycle_outbox.db"
        path.parent.mkdir()
        path.write_bytes(b"not a sqlite database")
        rejected = _LifecycleOutboxRepository(root).open()
        expect(rejected.kind is OutboxResultKind.REJECTED, f"corrupt outbox database was accepted: {rejected}")


def test_lifecycle_outbox_bulk_compare_and_set_operations_isolate_rows():
    """Bulk lease, stage, and acknowledgement CAS operations retain row isolation."""
    from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, ParentGuard, ExecutionStage
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository, OutboxResultKind

    with tempfile.TemporaryDirectory() as td:
        repo = _LifecycleOutboxRepository(Path(td))
        plans = []
        for index in (1, 2):
            parent_uuid = f"00000000-0000-4000-8000-0000000007{index:02d}"
            child_uuid = f"00000000-0000-4000-8000-0000000008{index:02d}"
            guard = ParentGuard("completed", "on", "bulk-cas", index, f"rf-{index}", "20260101T000000Z")
            identity = LifecycleIdentity("bulk-cas", parent_uuid, index, index + 1, LifecycleEvent.COMPLETE)
            plans.append(_plan_from_values(
                identity=identity,
                action=LifecycleAction.SPAWN_CHILD,
                parent_guard=guard,
                child_payload={"uuid": child_uuid, "chainID": "bulk-cas", "link": index + 1, "prevLink": parent_uuid[:8]},
                parent_patch={"nextLink": child_uuid[:8]},
                expected_postconditions=("child_present", "parent_linked", "verified"),
            ))
        for plan in plans:
            expect(repo.enqueue(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch").ok, "bulk CAS enqueue failed")
        claimed, records = repo.claim_batch(owner="bulk-owner", lease_seconds=30, limit=10)
        expect(claimed.ok and len(records) == 2, f"bulk CAS claim failed: {claimed} {records}")
        ids = tuple(record.intent_id for record in records)
        with repo.session():
            renewed, renewal_rows = repo.renew_leases(intent_ids=ids, owner="bulk-owner", lease_seconds=30)
            expect(renewed.ok and all(item.kind is OutboxResultKind.APPLIED for item in renewal_rows.values()), "bulk renewal failed")
            advanced, stage_rows = repo.advance_stages(
                stages={intent_id: ExecutionStage.CHILD_PRESENT for intent_id in ids}, owner="bulk-owner"
            )
            expect(advanced.ok and all(item.kind is OutboxResultKind.APPLIED for item in stage_rows.values()), "bulk stage advance failed")
            # One invalid owner must be isolated from the valid row.
            isolated, isolated_rows = repo.renew_leases(intent_ids=(ids[0], "missing-intent"), owner="wrong-owner", lease_seconds=30)
            expect(isolated.ok and isolated_rows[ids[0]].kind is OutboxResultKind.CONFLICT, "bulk CAS did not isolate ownership conflict")
            advanced, _ = repo.advance_stages(
                stages={intent_id: ExecutionStage.PARENT_LINKED for intent_id in ids}, owner="bulk-owner"
            )
            expect(advanced.ok, "bulk parent stage advance failed")
            advanced, _ = repo.advance_stages(
                stages={intent_id: ExecutionStage.VERIFIED for intent_id in ids}, owner="bulk-owner"
            )
            expect(advanced.ok, "bulk verification stage advance failed")
            acknowledged, ack_rows = repo.acknowledge_many(intent_ids=ids, owner="bulk-owner")
            expect(acknowledged.ok and all(item.kind is OutboxResultKind.APPLIED for item in ack_rows.values()), "bulk acknowledgement failed")




def test_lifecycle_outbox_claims_quarantine_exhausted_and_inconsistent_rows():
    """Claiming must not execute exhausted or stage/state-inconsistent rows."""
    from nautical_core.lifecycle_models import (
        LifecycleAction,
        LifecycleEvent,
        LifecycleIdentity,
        LifecyclePlan,
        ParentGuard,
    )
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository

    now = [1000.0]

    def plan_for(link: int, *, max_attempts: int = 3) -> LifecyclePlan:
        parent_uuid = f"00000000-0000-4000-8000-{link:012d}"
        child_uuid = f"10000000-0000-4000-8000-{link:012d}"
        return LifecyclePlan.from_draft(
            identity=LifecycleIdentity("claim-guards", parent_uuid, link, link + 1, LifecycleEvent.COMPLETE),
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=ParentGuard("completed", "on", "claim-guards", link, "rf1-claim"),
            draft=_task_draft({
                "uuid": child_uuid,
                "description": "claim guard child",
                "status": "pending",
                "chain": "on",
                "chainID": "claim-guards",
                "link": link + 1,
                "prevLink": parent_uuid[:8],
                "cp": "1d",
                "due": "20260102T000000Z",
            }),
            parent_patch={"nextLink": child_uuid[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
            max_attempts=max_attempts,
        )

    with tempfile.TemporaryDirectory() as td:
        repo = _LifecycleOutboxRepository(Path(td), clock=lambda: now[0])
        exhausted = plan_for(1, max_attempts=1)
        expect(repo.enqueue(exhausted, configuration_fingerprint="cfg", schedule_fingerprint="sch").ok, "exhaustion enqueue failed")
        first, records = repo.claim_batch(owner="crashed-worker", lease_seconds=5, limit=1)
        expect(first.ok and len(records) == 1 and records[0].attempts == 1, f"initial claim failed: {first}, {records}")
        now[0] += 6
        recovered, records = repo.claim_batch(owner="recovery-worker", lease_seconds=5, limit=1)
        expect(recovered.ok and not records, f"exhausted intent was reclaimed: {recovered}, {records}")
        _, status = repo.status()
        expect(status["states"].get("quarantined") == 1, f"exhausted intent was not quarantined: {status}")
        failure = next(item for item in status["records"] if item["intent_id"] == exhausted.identity.idempotency_key)
        expect(failure.get("failure", {}).get("code") == "retry_exhausted", f"wrong exhaustion evidence: {failure}")

    with tempfile.TemporaryDirectory() as td:
        repo = _LifecycleOutboxRepository(Path(td), clock=lambda: now[0])
        inconsistent = plan_for(2)
        expect(repo.enqueue(inconsistent, configuration_fingerprint="cfg", schedule_fingerprint="sch").ok, "inconsistent enqueue failed")
        with sqlite3.connect(str(repo.path)) as conn:
            conn.execute(
                "UPDATE lifecycle_outbox SET processing_state='retry', lifecycle_stage='manual_review' WHERE intent_id=?",
                (inconsistent.identity.idempotency_key,),
            )
        claimed, records = repo.claim_batch(owner="poison-worker", lease_seconds=5, limit=1)
        expect(claimed.ok and not records, f"inconsistent active row was claimed: {claimed}, {records}")
        _, status = repo.status()
        expect(status["states"].get("quarantined") == 1, f"inconsistent row was not quarantined: {status}")
        failure = next(item for item in status["records"] if item["intent_id"] == inconsistent.identity.idempotency_key)
        expect(
            "active outbox state" in str(failure.get("failure", {}).get("message") or ""),
            f"stage/state evidence was lost: {failure}",
        )

    with tempfile.TemporaryDirectory() as td:
        repo = _LifecycleOutboxRepository(Path(td), clock=lambda: now[0])
        exhausted = plan_for(3, max_attempts=1)
        expect(repo.enqueue(exhausted, configuration_fingerprint="cfg", schedule_fingerprint="sch").ok, "single-intent enqueue failed")
        first = repo.claim_intent(owner="single-worker", lease_seconds=5, intent_id=exhausted.identity.idempotency_key)
        expect(first.ok and first.record is not None and first.record.attempts == 1, f"single-intent claim failed: {first}")
        now[0] += 6
        recovered = repo.claim_intent(owner="single-recovery", lease_seconds=5, intent_id=exhausted.identity.idempotency_key)
        expect(recovered.kind.value == "rejected" and "retry budget exhausted" in recovered.reason, f"single intent was reclaimed: {recovered}")




def test_full_hooks_receive_one_explicit_integration_context():
    """Full hooks share one context and on-modify cannot acquire mutation access."""
    from pathlib import Path

    from nautical_core.integration_context import IntegrationAccess

    cases = (
        ("on-add.nautical", "_nautical_context_add", IntegrationAccess.READ_ONLY),
        ("on-modify.nautical", "_nautical_context_modify", IntegrationAccess.READ_ONLY),
        ("on-exit.nautical", "_nautical_context_exit", IntegrationAccess.MUTATION),
    )
    previous_argv = list(sys.argv)
    previous_taskdata = os.environ.get("TASKDATA")
    try:
        os.environ.pop("TASKDATA", None)
        with tempfile.TemporaryDirectory(prefix="nautical_hook_context_") as td:
            taskdata = Path(td).resolve()
            for hook_name, module_name, expected_access in cases:
                sys.argv = [hook_name, f"data:{taskdata}"]
                module = _load_hook_module(_find_hook_file(hook_name), module_name)
                context = module._INTEGRATION_CONTEXT
                expect(context is not None, f"{hook_name} did not construct an integration context")
                expect(context.taskdata == taskdata, f"{hook_name} changed its Taskdata")
                expect(context.access is expected_access, f"{hook_name} acquired {context.access.value}")
                request_context = module._build_hook_runtime_context()
                expect(
                    request_context.integration is context,
                    f"{hook_name} built a second request context",
                )
                expect(request_context.uow.context is context, f"{hook_name} UOW changed its context")
                expect(request_context.uow.reads.size == 0, f"{hook_name} UOW did not start empty")
                second_context = module._build_hook_runtime_context()
                expect(second_context.uow is not request_context.uow, f"{hook_name} reused its UOW")
                expect(
                    second_context.uow.reads is not request_context.uow.reads,
                    f"{hook_name} shared invocation read state",
                )
    finally:
        sys.argv = previous_argv
        if previous_taskdata is None:
            os.environ.pop("TASKDATA", None)
        else:
            os.environ["TASKDATA"] = previous_taskdata



def test_light_taskdata_resolution_matches_hook_precedence():
    """The early exit resolver should preserve argv, environment, and fallback precedence."""
    bootstrap = _load_hook_module(
        os.path.join(ROOT, "nautical_core", "hook_bootstrap.py"),
        "_nautical_bootstrap_light_taskdata_test",
    )
    path_support = _load_hook_module(
        os.path.join(ROOT, "nautical_core", "config_support.py"),
        "_nautical_config_support_light_taskdata_test",
    )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        env_dir = root / "env-data"
        arg_dir = root / "arg-data"
        env_dir.mkdir()
        arg_dir.mkdir()
        env = {"TASKDATA": str(env_dir)}

        from_env = bootstrap.resolve_task_data_context_light(
            path_support=path_support,
            argv=[],
            env=env,
            tw_dir=str(root),
        )
        expect(from_env == (str(env_dir), True, "env"), f"unexpected environment resolution: {from_env!r}")

        from_argv = bootstrap.resolve_task_data_context_light(
            path_support=path_support,
            argv=[f"data.location:{arg_dir}"],
            env=env,
            tw_dir=str(root),
        )
        expect(from_argv == (str(arg_dir), True, "argv"), f"unexpected argv resolution: {from_argv!r}")

        fallback = bootstrap.resolve_task_data_context_light(
            path_support=path_support,
            argv=[],
            env={},
            tw_dir=str(root),
        )
        expect(fallback == (str(root), False, "fallback"), f"unexpected fallback resolution: {fallback!r}")


def test_plain_hook_fast_paths_do_not_import_core_package():
    """Plain add/modify and empty exit should pass through without importing the core package."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        hooks_dir = root / "hooks"
        core_dir = root / "nautical_core"
        hooks_dir.mkdir()
        core_dir.mkdir()
        for hook_name in ("on-add.nautical", "on-modify.nautical", "on-exit.nautical"):
            shutil.copy2(_find_hook_file(hook_name), hooks_dir / hook_name)
        shutil.copy2(Path(ROOT) / "nautical_core" / "hook_bootstrap.py", core_dir / "hook_bootstrap.py")
        shutil.copy2(Path(ROOT) / "nautical_core" / "hook_protocol.py", core_dir / "hook_protocol.py")
        shutil.copy2(Path(ROOT) / "nautical_core" / "task_codec.py", core_dir / "task_codec.py")
        shutil.copy2(Path(ROOT) / "nautical_core" / "task_models.py", core_dir / "task_models.py")
        shutil.copy2(Path(ROOT) / "nautical_core" / "exit_probe.py", core_dir / "exit_probe.py")
        shutil.copy2(Path(ROOT) / "nautical_core" / "config_support.py", core_dir / "config_support.py")
        (core_dir / "__init__.py").write_text("raise RuntimeError('core must not load on plain fast path')\n", encoding="utf-8")

        env = os.environ.copy()
        env["TASKDATA"] = str(root)
        env.pop("NAUTICAL_CORE_PATH", None)
        env.pop("NAUTICAL_TRUST_CORE_PATH", None)
        env.pop("NAUTICAL_PROFILE", None)
        env.pop("NAUTICAL_BENCH_FORCE_FULL", None)
        plain = {
            "uuid": "00000000-0000-4000-8000-000000000706",
            "status": "pending",
            "description": "Cafe ăîșț ✅",
        }

        add = subprocess.run(
            [sys.executable, str(hooks_dir / "on-add.nautical")],
            input=json.dumps(plain, ensure_ascii=False),
            text=True,
            capture_output=True,
            env=env,
            timeout=5.0,
        )
        expect(add.returncode == 0, f"plain add fast path failed: {add.stderr!r}")
        expect(json.loads(add.stdout) == plain, f"plain add fast path changed task: {add.stdout!r}")
        expect("ăîșț ✅" in add.stdout and "\\u" not in add.stdout, f"plain add escaped Unicode: {add.stdout!r}")

        modified = dict(plain, description="Modified ăîșț ✅")
        modify = subprocess.run(
            [sys.executable, str(hooks_dir / "on-modify.nautical")],
            input=json.dumps(plain, ensure_ascii=False) + "\n" + json.dumps(modified, ensure_ascii=False),
            text=True,
            capture_output=True,
            env=env,
            timeout=5.0,
        )
        expect(modify.returncode == 0, f"plain modify fast path failed: {modify.stderr!r}")
        expect(json.loads(modify.stdout) == modified, f"plain modify fast path changed task: {modify.stdout!r}")
        expect("ăîșț ✅" in modify.stdout and "\\u" not in modify.stdout, f"plain modify escaped Unicode: {modify.stdout!r}")

        nautical_old = dict(
            plain,
            cp="P1D",
            chain="on",
            chainID="abcd1234",
            link=3,
            due="20270101T090000Z",
        )
        nautical_new = dict(nautical_old, description="Modified nautical ăîșț ✅")
        nautical_modify = subprocess.run(
            [sys.executable, str(hooks_dir / "on-modify.nautical")],
            input=json.dumps(nautical_old, ensure_ascii=False) + "\n" + json.dumps(nautical_new, ensure_ascii=False),
            text=True,
            capture_output=True,
            env=env,
            timeout=5.0,
        )
        expect(nautical_modify.returncode == 0, f"ordinary Nautical edit loaded the broken core: {nautical_modify.stderr!r}")
        expect(json.loads(nautical_modify.stdout) == nautical_new, f"ordinary Nautical edit changed task: {nautical_modify.stdout!r}")

        exit_hook = subprocess.run(
            [sys.executable, str(hooks_dir / "on-exit.nautical")],
            text=True,
            capture_output=True,
            env=env,
            timeout=5.0,
        )
        expect(exit_hook.returncode == 0, f"empty exit fast path failed: {exit_hook.stderr!r}")
        expect(exit_hook.stdout == "", f"empty exit fast path wrote stdout: {exit_hook.stdout!r}")
        expect(not (root / ".nautical-state").exists(), "empty exit fast path should not create queue state")

        forced_env = dict(env)
        forced_env["NAUTICAL_BENCH_FORCE_FULL"] = "1"
        forced_cases = (
            (hooks_dir / "on-add.nautical", json.dumps(plain, ensure_ascii=False)),
            (
                hooks_dir / "on-modify.nautical",
                json.dumps(plain, ensure_ascii=False) + "\n" + json.dumps(modified, ensure_ascii=False),
            ),
            (
                hooks_dir / "on-modify.nautical",
                json.dumps(nautical_old, ensure_ascii=False) + "\n" + json.dumps(nautical_new, ensure_ascii=False),
            ),
            (hooks_dir / "on-exit.nautical", ""),
        )
        for hook_path, input_text in forced_cases:
            forced = subprocess.run(
                [sys.executable, str(hook_path)],
                input=input_text,
                text=True,
                capture_output=True,
                env=forced_env,
                timeout=5.0,
            )
            expect(forced.returncode != 0, f"force-full switch did not reach the broken core for {hook_path.name}")

        impl_dir = core_dir / "hooks"
        impl_dir.mkdir()
        (impl_dir / "add_impl.py").write_text(
            "HOOK_IMPL_API = 999\n"
            "def run_hook(**_kwargs):\n"
            "    raise AssertionError('mismatched implementation must not run')\n",
            encoding="utf-8",
        )
        nautical = dict(plain, cp="P1D")
        mismatch = subprocess.run(
            [sys.executable, str(hooks_dir / "on-add.nautical")],
            input=json.dumps(nautical, ensure_ascii=False),
            text=True,
            capture_output=True,
            env=env,
            timeout=5.0,
        )
        expect(mismatch.returncode != 0, "on-add accepted an incompatible implementation API")
        expect(json.loads(mismatch.stdout) == nautical, "on-add API mismatch did not preserve the input task")
        expect(mismatch.stderr == "", f"on-add API mismatch wrote diagnostics without opt-in: {mismatch.stderr!r}")

        diag_env = dict(env)
        diag_env["NAUTICAL_DIAG"] = "1"
        mismatch_diag = subprocess.run(
            [sys.executable, str(hooks_dir / "on-add.nautical")],
            input=json.dumps(nautical, ensure_ascii=False),
            text=True,
            capture_output=True,
            env=diag_env,
            timeout=5.0,
        )
        expect("API mismatch" in mismatch_diag.stderr, "on-add API mismatch diagnostic was not actionable")

        (impl_dir / "modify_impl.py").write_text(
            "HOOK_IMPL_API = 999\n"
            "def run_hook(**_kwargs):\n"
            "    raise AssertionError('mismatched implementation must not run')\n",
            encoding="utf-8",
        )
        nautical_old = dict(plain, cp="P1D")
        nautical_changed = dict(nautical_old, cp="P2D")
        modify_input = json.dumps(nautical_old, ensure_ascii=False) + "\n" + json.dumps(
            nautical_changed,
            ensure_ascii=False,
        )
        modify_mismatch = subprocess.run(
            [sys.executable, str(hooks_dir / "on-modify.nautical")],
            input=modify_input,
            text=True,
            capture_output=True,
            env=env,
            timeout=5.0,
        )
        expect(modify_mismatch.returncode != 0, "on-modify accepted an incompatible implementation API")
        expect(
            json.loads(modify_mismatch.stdout) == nautical_changed,
            "on-modify API mismatch did not preserve the latest task",
        )
        expect(
            modify_mismatch.stderr == "",
            f"on-modify API mismatch wrote diagnostics without opt-in: {modify_mismatch.stderr!r}",
        )

        modify_mismatch_diag = subprocess.run(
            [sys.executable, str(hooks_dir / "on-modify.nautical")],
            input=modify_input,
            text=True,
            capture_output=True,
            env=diag_env,
            timeout=5.0,
        )
        expect(
            "API mismatch" in modify_mismatch_diag.stderr,
            "on-modify API mismatch diagnostic was not actionable",
        )

        (impl_dir / "exit_impl.py").write_text(
            "HOOK_IMPL_API = 999\n"
            "def run_hook(**_kwargs):\n"
            "    raise AssertionError('mismatched implementation must not run')\n",
            encoding="utf-8",
        )
        state_dir = root / ".nautical-state"
        state_dir.mkdir(exist_ok=True)
        with sqlite3.connect(str(state_dir / ".nautical_lifecycle_outbox.db")) as conn:
            conn.execute("CREATE TABLE lifecycle_outbox (intent_id TEXT PRIMARY KEY, processing_state TEXT NOT NULL)")
            conn.execute("INSERT INTO lifecycle_outbox VALUES ('intent-1', 'ready')")
            conn.commit()
        exit_mismatch = subprocess.run(
            [sys.executable, str(hooks_dir / "on-exit.nautical")],
            text=True,
            capture_output=True,
            env=env,
            timeout=5.0,
        )
        expect(exit_mismatch.returncode != 0, "on-exit accepted an incompatible implementation API")
        expect(exit_mismatch.stdout == "", f"on-exit API mismatch wrote stdout: {exit_mismatch.stdout!r}")
        expect(
            exit_mismatch.stderr == "",
            f"on-exit API mismatch wrote diagnostics without opt-in: {exit_mismatch.stderr!r}",
        )

        exit_mismatch_diag = subprocess.run(
            [sys.executable, str(hooks_dir / "on-exit.nautical")],
            text=True,
            capture_output=True,
            env=diag_env,
            timeout=5.0,
        )
        expect("API mismatch" in exit_mismatch_diag.stderr, "on-exit API mismatch diagnostic was not actionable")


def test_full_hook_modules_defer_core_import():
    """Loading lifecycle implementations must not parse the full core package."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        core_dir = root / "nautical_core"
        hooks_dir = core_dir / "hooks"
        hooks_dir.mkdir(parents=True)
        shutil.copy2(Path(ROOT) / "nautical_core" / "hook_bootstrap.py", core_dir / "hook_bootstrap.py")
        shutil.copy2(Path(ROOT) / "nautical_core" / "config_support.py", core_dir / "config_support.py")
        (core_dir / "__init__.py").write_text(
            "raise AssertionError('full core must not load while importing hook implementations')\n",
            encoding="utf-8",
        )
        for name in ("add_impl.py", "modify_impl.py", "exit_impl.py"):
            shutil.copy2(Path(ROOT) / "nautical_core" / "hooks" / name, hooks_dir / name)

        probe = root / "probe.py"
        probe.write_text(
            "import importlib.util, sys\n"
            "from pathlib import Path\n"
            "root = Path(__file__).parent\n"
            "for index, name in enumerate(('add_impl.py', 'modify_impl.py', 'exit_impl.py')):\n"
            "    spec = importlib.util.spec_from_file_location(f'probe_hook_{index}', root / 'nautical_core' / 'hooks' / name)\n"
            "    module = importlib.util.module_from_spec(spec)\n"
            "    spec.loader.exec_module(module)\n"
            "    assert module.core is None, f'{name} imported core eagerly'\n"
            "    if name == 'modify_impl.py':\n"
            "        for attr in ('_MODIFY_ORDINARY', '_MODIFY_EXPIRATION', '_MODIFY_GENERATION_COMPAT', '_CHAIN_GENERATION', '_QUEUE_STORE'):\n"
            "            assert getattr(module, attr, None) is None, f'{attr} loaded during modify import'\n"
            "assert 'nautical_core' not in sys.modules, 'hook import populated the full package'\n"
            "print('ok')\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env.pop("NAUTICAL_CORE_PATH", None)
        env.pop("NAUTICAL_TRUST_CORE_PATH", None)
        result = subprocess.run(
            [sys.executable, str(probe)],
            text=True,
            capture_output=True,
            env=env,
            timeout=5.0,
        )
        expect(result.returncode == 0, f"full hook import was not lazy: {result.stderr!r}")
        expect(result.stdout.strip() == "ok", f"unexpected lazy import probe output: {result.stdout!r}")


def test_full_hooks_reuse_wrapper_protocol_probe():
    """The full implementation must consume the wrapper's validated probe once."""
    cases = (
        ("on-add.nautical", "_nautical_probe_reuse_add", "probe_on_add"),
        ("on-modify.nautical", "_nautical_probe_reuse_modify", "probe_on_modify"),
    )
    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td) / "taskdata"
        taskdata.mkdir()
        for hook_name, module_name, probe_name in cases:
            mod = _load_hook_module(_find_hook_file(hook_name), module_name)
            calls = {"main": 0, "probe": 0}
            probe = object()

            def unexpected_probe(*_args, **_kwargs):
                calls["probe"] += 1
                raise AssertionError("full implementation reparsed wrapper input")

            protocol = SimpleNamespace(**{probe_name: unexpected_probe})
            previous_main = mod.main
            previous_resolve = mod._resolve_task_data_context
            previous_early = getattr(mod, "_EARLY_PROTOCOL_RESULT", None)
            try:
                mod.main = lambda: calls.__setitem__("main", calls["main"] + 1)
                mod._resolve_task_data_context = lambda: (str(taskdata), False)
                result = mod.run_hook(
                    raw_input=b"{\"uuid\":\"probe-reuse\"}",
                    argv=(),
                    hook_dir=str(taskdata / "hooks"),
                    core_base=str(Path(ROOT) / "nautical_core"),
                    protocol=protocol,
                    probe=probe,
                    protocol_error=None,
                )
                retained_probe = mod._EARLY_PROTOCOL_RESULT
            finally:
                mod.main = previous_main
                mod._resolve_task_data_context = previous_resolve
                mod._EARLY_PROTOCOL_RESULT = previous_early

            expect(result == 0, f"{hook_name} run_hook returned {result}")
            expect(calls["main"] == 1, f"{hook_name} implementation main was not called once")
            expect(calls["probe"] == 0, f"{hook_name} reparsed the wrapper input")
            expect(mod._PROTOCOL is protocol, f"{hook_name} did not retain wrapper protocol module")
            expect(retained_probe is probe, f"{hook_name} did not retain wrapper probe result")


def test_hook_files_are_private_permissions():
    """Lifecycle outbox and lock files should not be group/world-readable."""
    lock_path = None
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        os.environ["TASKDATA"] = td
        try:
            core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
            mod_core = _load_hook_module(core_path, "_nautical_core_perm_test")
            lock_path = os.path.join(td, ".nautical_perm_test.lock")
            with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0) as ok:
                expect(ok, "safe_lock did not acquire")
                mode = stat.S_IMODE(os.stat(lock_path).st_mode)
                expect((mode & 0o077) == 0, f"lock file has group/other perms: {oct(mode)}")

            from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository

            db_path = _LifecycleOutboxRepository(Path(td)).path
            opened = _LifecycleOutboxRepository(Path(td)).open()
            expect(opened.ok, f"lifecycle outbox did not open: {opened.reason}")
            expect(db_path.exists(), f"lifecycle outbox not created: {db_path}")
            mode = stat.S_IMODE(db_path.stat().st_mode)
            expect((mode & 0o077) == 0, f"lifecycle outbox has group/other perms: {oct(mode)}")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata

def test_safe_lock_fcntl_contention():
    """safe_lock should fail to acquire when another process holds the lock."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    mod_core = _load_hook_module(core_path, "_nautical_core_lock_fcntl_test")
    if getattr(mod_core, "fcntl", None) is None:
        return
    with tempfile.TemporaryDirectory() as td:
        lock_path = os.path.join(td, ".nautical_fcntl.lock")
        ready_path = os.path.join(td, ".nautical_fcntl.ready")
        script = (
            "import os, time\n"
            "import fcntl\n"
            "lp = os.environ['LOCK_PATH']\n"
            "rp = os.environ['READY_PATH']\n"
            "fd = os.open(lp, os.O_CREAT | os.O_RDWR, 0o600)\n"
            "f = os.fdopen(fd, 'a', encoding='utf-8')\n"
            "fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
            "with open(rp, 'w', encoding='utf-8') as r:\n"
            "    r.write('ready')\n"
            "time.sleep(1.0)\n"
        )
        p = subprocess.Popen(
            [sys.executable, "-c", script],
            env={**os.environ, "LOCK_PATH": lock_path, "READY_PATH": ready_path},
        )
        try:
            for _ in range(50):
                if os.path.exists(ready_path):
                    break
                _time.sleep(0.02)
            expect(os.path.exists(ready_path), "lock holder did not start")
            with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0) as ok:
                expect(not ok, "safe_lock should not acquire while locked")
        finally:
            p.wait(timeout=3.0)
        with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0) as ok:
            expect(ok, "safe_lock should acquire after lock release")

def test_safe_lock_fallback_contention():
    """safe_lock should fail to acquire when fallback lockfile exists."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    mod_core = _load_hook_module(core_path, "_nautical_core_lock_fallback_test")
    prev_fcntl = getattr(mod_core, "fcntl", None)
    mod_core.fcntl = None
    try:
        with tempfile.TemporaryDirectory() as td:
            lock_path = os.path.join(td, ".nautical_fallback.lock")
            with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0) as ok:
                expect(ok, "fallback safe_lock did not acquire")
                with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0) as ok2:
                    expect(not ok2, "fallback safe_lock should not acquire when locked")
            with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0) as ok3:
                expect(ok3, "fallback safe_lock should acquire after release")
    finally:
        mod_core.fcntl = prev_fcntl

def test_safe_lock_fallback_stale_cleanup():
    """safe_lock fallback should clear stale lockfiles."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    mod_core = _load_hook_module(core_path, "_nautical_core_lock_stale_test")
    prev_fcntl = getattr(mod_core, "fcntl", None)
    mod_core.fcntl = None
    try:
        with tempfile.TemporaryDirectory() as td:
            lock_path = os.path.join(td, ".nautical_stale.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                f.write("999999 0\n")
            os.utime(lock_path, (1, 1))
            with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0, stale_after=1.0) as ok:
                expect(ok, "stale fallback lock was not cleared")
    finally:
        mod_core.fcntl = prev_fcntl

def test_safe_lock_fallback_stale_pid_cleanup():
    """safe_lock fallback should clear lockfiles with dead PIDs."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    mod_core = _load_hook_module(core_path, "_nautical_core_lock_pid_stale_test")
    prev_fcntl = getattr(mod_core, "fcntl", None)
    mod_core.fcntl = None
    try:
        with tempfile.TemporaryDirectory() as td:
            lock_path = os.path.join(td, ".nautical_pid.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                f.write("999999 0\n")
            os.utime(lock_path, (1, 1))
            with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0, stale_after=1.0) as ok:
                expect(ok, "stale PID lock was not cleared")
    finally:
        mod_core.fcntl = prev_fcntl

def test_diag_log_rotation_bounds():
    """Persistent diag log should rotate when exceeding max size."""
    hook = _find_hook_file("on-modify.nautical")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        prev_diag_log = os.environ.get("NAUTICAL_DIAG_LOG")
        prev_diag_max = os.environ.get("NAUTICAL_DIAG_LOG_MAX_BYTES")
        os.environ["TASKDATA"] = td
        os.environ["NAUTICAL_DIAG_LOG"] = "1"
        os.environ["NAUTICAL_DIAG_LOG_MAX_BYTES"] = "20"
        try:
            mod = _load_hook_module(hook, "_nautical_diag_log_rotation_test")
            log_path = Path(td) / ".nautical_diag.jsonl"
            log_path.write_text("x" * 64, encoding="utf-8")
            mod._diag("rotate me")
            overflow = list(Path(td).glob(".nautical_diag.overflow.*.jsonl"))
            expect(overflow, "diag log did not rotate")
            expect(log_path.exists(), "diag log missing after rotation")
            content = log_path.read_text(encoding="utf-8").strip()
            expect(content, "diag log not written after rotation")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata
            if prev_diag_log is None:
                os.environ.pop("NAUTICAL_DIAG_LOG", None)
            else:
                os.environ["NAUTICAL_DIAG_LOG"] = prev_diag_log
            if prev_diag_max is None:
                os.environ.pop("NAUTICAL_DIAG_LOG_MAX_BYTES", None)
            else:
                os.environ["NAUTICAL_DIAG_LOG_MAX_BYTES"] = prev_diag_max

def test_diag_log_redacts_sensitive_fields():
    """Persistent diag log should redact sensitive fields."""
    hook = _find_hook_file("on-modify.nautical")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        prev_diag_log = os.environ.get("NAUTICAL_DIAG_LOG")
        os.environ["TASKDATA"] = td
        os.environ["NAUTICAL_DIAG_LOG"] = "1"
        try:
            mod = _load_hook_module(hook, "_nautical_diag_log_redact_test")
            msg = json.dumps({"description": "secret", "notes": "hidden", "ok": "keep"})
            mod._diag(msg)
            log_path = Path(td) / ".nautical_diag.jsonl"
            content = log_path.read_text(encoding="utf-8")
            expect("secret" not in content and "hidden" not in content, "diag log did not redact sensitive fields")
            expect("[redacted]" in content, "diag log missing redaction marker")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata
            if prev_diag_log is None:
                os.environ.pop("NAUTICAL_DIAG_LOG", None)
            else:
                os.environ["NAUTICAL_DIAG_LOG"] = prev_diag_log


def test_hook_diag_redact_msg_masks_sensitive_json_fields():
    """Hook-level diag redaction helper should mask sensitive JSON fields."""
    hook_add = _find_hook_file("on-add.nautical")
    hook_exit = _find_hook_file("on-exit.nautical")
    mod_add = _load_hook_module(hook_add, "_nautical_on_add_diag_redact_msg_test")
    mod_exit = _load_hook_module(hook_exit, "_nautical_on_exit_diag_redact_msg_test")
    raw = json.dumps(
        {
            "description": "sensitive text",
            "annotations": "top secret",
            "note": "private",
            "safe": "ok",
        },
        ensure_ascii=False,
    )
    red_add = mod_add._diag_redact_msg(raw)
    red_exit = mod_exit._diag_redact_msg(raw)
    obj_add = json.loads(red_add)
    obj_exit = json.loads(red_exit)
    expect(obj_add.get("description") == "[redacted]", f"on-add description not redacted: {obj_add}")
    expect(obj_add.get("annotations") == "[redacted]", f"on-add annotations not redacted: {obj_add}")
    expect(obj_add.get("note") == "[redacted]", f"on-add note not redacted: {obj_add}")
    expect(obj_add.get("safe") == "ok", f"on-add non-sensitive key changed: {obj_add}")
    expect(obj_exit.get("description") == "[redacted]", f"on-exit description not redacted: {obj_exit}")
    expect(obj_exit.get("annotations") == "[redacted]", f"on-exit annotations not redacted: {obj_exit}")
    expect(obj_exit.get("note") == "[redacted]", f"on-exit note not redacted: {obj_exit}")
    expect(obj_exit.get("safe") == "ok", f"on-exit non-sensitive key changed: {obj_exit}")


def test_core_cache_dir_and_lock_permissions():
    """Core cache dir and lock files should have restricted permissions."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    with tempfile.TemporaryDirectory() as td:
        cache_dir = os.path.join(td, "cache")
        mod = _load_hook_module(core_path, "_nautical_core_cache_perm_test")
        mod._refresh_facade_config_exports()
        mod.ANCHOR_CACHE_DIR_OVERRIDE = cache_dir
        mod._cache_api._resolve()
        previous_trust = os.environ.get("NAUTICAL_TRUST_CACHE_PATH")
        os.environ["NAUTICAL_TRUST_CACHE_PATH"] = "1"
        try:
            path = mod._cache_dir()
        finally:
            if previous_trust is None:
                os.environ.pop("NAUTICAL_TRUST_CACHE_PATH", None)
            else:
                os.environ["NAUTICAL_TRUST_CACHE_PATH"] = previous_trust
        expect(path == cache_dir, f"cache dir mismatch: {path}")
        mode = stat.S_IMODE(os.stat(path).st_mode)
        expect((mode & 0o077) == 0, f"cache dir has group/other perms: {oct(mode)}")
        lock_path = mod._cache_lock_path("permtest")
        with mod._cache_lock("permtest") as ok:
            expect(ok, "cache lock did not acquire")
            lmode = stat.S_IMODE(os.stat(lock_path).st_mode)
            expect((lmode & 0o077) == 0, f"cache lock has group/other perms: {oct(lmode)}")

def test_core_cache_lock_contention_matches_safe_lock():
    """_cache_lock should block contention similarly to safe_lock."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    with tempfile.TemporaryDirectory() as td:
        cache_dir = os.path.join(td, "cache")
        mod = _load_hook_module(core_path, "_nautical_core_cache_lock_test")
        mod._CACHE_DIR = None
        mod.ANCHOR_CACHE_DIR_OVERRIDE = cache_dir
        mod._cache_dir()
        with mod._cache_lock("contend") as ok:
            expect(ok, "cache lock did not acquire")
            with mod._cache_lock("contend") as ok2:
                expect(not ok2, "cache lock should not acquire when already locked")
        prev = getattr(mod, "fcntl", None)
        mod.fcntl = None
        try:
            with mod._cache_lock("contend2") as ok3:
                expect(ok3, "fallback cache lock did not acquire")
                with mod._cache_lock("contend2") as ok4:
                    expect(not ok4, "fallback cache lock should not acquire when locked")
        finally:
            mod.fcntl = prev


def test_core_cache_dir_rejects_symlink_override():
    """_cache_dir should reject symlink override paths and choose a real directory."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    with tempfile.TemporaryDirectory() as td:
        target = os.path.join(td, "real-cache")
        symlink = os.path.join(td, "cache-link")
        os.makedirs(target, exist_ok=True)
        os.symlink(target, symlink)

        prev_xdg = os.environ.get("XDG_CACHE_HOME")
        prev_tmp = os.environ.get("NAUTICAL_ALLOW_TMP_CACHE")
        os.environ["XDG_CACHE_HOME"] = td
        os.environ["NAUTICAL_ALLOW_TMP_CACHE"] = "1"
        try:
            mod = _load_hook_module(core_path, "_nautical_core_cache_symlink_guard_test")
            mod._CACHE_DIR = None
            mod.ANCHOR_CACHE_DIR_OVERRIDE = symlink
            chosen = mod._cache_dir()
            expect(chosen != symlink, f"symlink override should be rejected, got {chosen}")
            expect(chosen and os.path.isdir(chosen), f"cache dir should fall back to valid dir, got {chosen!r}")
            expect(not os.path.islink(chosen), f"cache dir should not be symlink, got {chosen}")
        finally:
            if prev_xdg is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = prev_xdg
            if prev_tmp is None:
                os.environ.pop("NAUTICAL_ALLOW_TMP_CACHE", None)
            else:
                os.environ["NAUTICAL_ALLOW_TMP_CACHE"] = prev_tmp


def test_on_exit_reads_data_arg_from_hook_argv():
    """on-exit should resolve TW_DATA_DIR from hook argv data: token."""
    hook = _find_hook_file("on-exit.nautical")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    if "TASKDATA" in os.environ:
        del os.environ["TASKDATA"]
    with tempfile.TemporaryDirectory(prefix="nautical_data_arg_exit_") as data_dir:
        sys.argv = ["on-exit.nautical", "api:2", "command:modify", f"data:{data_dir}"]
        try:
            mod = _load_hook_module(hook, "_nautical_on_exit_data_arg_test")
        finally:
            sys.argv = prev_argv
            if prev_taskdata is not None:
                os.environ["TASKDATA"] = prev_taskdata
        expect(Path(mod.TW_DATA_DIR) == Path(data_dir), f"unexpected TW_DATA_DIR: {mod.TW_DATA_DIR}")
    expect(bool(getattr(mod, "_USE_RC_DATA_LOCATION", False)), "rc.data.location should be enabled when data arg is present")

def test_on_modify_no_explicit_taskdata_skips_rc_data_location():
    """on-modify should not force rc.data.location when data dir is not explicit."""
    hook = _find_hook_file("on-modify.nautical")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    if "TASKDATA" in os.environ:
        del os.environ["TASKDATA"]
    sys.argv = ["on-modify.nautical"]
    try:
        mod = _load_hook_module(hook, "_nautical_on_modify_no_data_override_test")
    finally:
        sys.argv = prev_argv
        if prev_taskdata is not None:
            os.environ["TASKDATA"] = prev_taskdata
    command_prefix = mod._task_cmd_prefix()
    expect(
        all(not str(part).startswith("rc.data.location=") for part in command_prefix),
        f"should not force rc.data.location without explicit data dir: {command_prefix!r}",
    )

def test_on_modify_reads_data_arg_from_hook_argv():
    """on-modify should resolve TW_DATA_DIR from hook argv data: token."""
    hook = _find_hook_file("on-modify.nautical")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    if "TASKDATA" in os.environ:
        del os.environ["TASKDATA"]
    with tempfile.TemporaryDirectory(prefix="nautical_data_arg_modify_") as data_dir:
        sys.argv = ["on-modify.nautical", "api:2", "command:modify", f"data:{data_dir}"]
        try:
            mod = _load_hook_module(hook, "_nautical_on_modify_data_arg_test")
        finally:
            sys.argv = prev_argv
            if prev_taskdata is not None:
                os.environ["TASKDATA"] = prev_taskdata
        expect(Path(mod.TW_DATA_DIR) == Path(data_dir), f"unexpected TW_DATA_DIR: {mod.TW_DATA_DIR}")
    expect(bool(getattr(mod, "_USE_RC_DATA_LOCATION", False)), "rc.data.location should be enabled when data arg is present")

def test_on_add_no_explicit_taskdata_skips_rc_data_location():
    """on-add should not force rc.data.location when data dir is not explicit."""
    hook = _find_hook_file("on-add.nautical")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    if "TASKDATA" in os.environ:
        del os.environ["TASKDATA"]
    sys.argv = ["on-add.nautical"]
    try:
        mod = _load_hook_module(hook, "_nautical_on_add_no_data_override_test")
    finally:
        sys.argv = prev_argv
        if prev_taskdata is not None:
            os.environ["TASKDATA"] = prev_taskdata
    command_prefix = mod._task_cmd_prefix()
    expect(
        all(not str(part).startswith("rc.data.location=") for part in command_prefix),
        f"should not force rc.data.location without explicit data dir: {command_prefix!r}",
    )

def test_on_add_reads_data_arg_from_hook_argv():
    """on-add should resolve TW_DATA_DIR from hook argv data: token."""
    hook = _find_hook_file("on-add.nautical")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    if "TASKDATA" in os.environ:
        del os.environ["TASKDATA"]
    with tempfile.TemporaryDirectory(prefix="nautical_data_arg_add_") as data_dir:
        sys.argv = ["on-add.nautical", "api:2", "command:add", f"data:{data_dir}"]
        try:
            mod = _load_hook_module(hook, "_nautical_on_add_data_arg_test")
        finally:
            sys.argv = prev_argv
            if prev_taskdata is not None:
                os.environ["TASKDATA"] = prev_taskdata
        expect(Path(mod.TW_DATA_DIR) == Path(data_dir), f"unexpected TW_DATA_DIR: {mod.TW_DATA_DIR}")
    expect(bool(getattr(mod, "_USE_RC_DATA_LOCATION", False)), "rc.data.location should be enabled when data arg is present")

def test_on_exit_data_arg_overrides_taskdata_env():
    """on-exit should prefer hook argv data: over TASKDATA env when both are present."""
    hook = _find_hook_file("on-exit.nautical")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    with tempfile.TemporaryDirectory(prefix="nautical_env_exit_") as env_dir, tempfile.TemporaryDirectory(prefix="nautical_arg_exit_") as arg_dir:
        os.environ["TASKDATA"] = env_dir
        sys.argv = ["on-exit.nautical", "api:2", "command:modify", f"data:{arg_dir}"]
        try:
            mod = _load_hook_module(hook, "_nautical_on_exit_data_arg_precedence_test")
        finally:
            sys.argv = prev_argv
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata
        expect(Path(mod.TW_DATA_DIR) == Path(arg_dir), f"expected argv data dir, got: {mod.TW_DATA_DIR}")

def test_on_modify_data_arg_overrides_taskdata_env():
    """on-modify should prefer hook argv data: over TASKDATA env when both are present."""
    hook = _find_hook_file("on-modify.nautical")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    with tempfile.TemporaryDirectory(prefix="nautical_env_modify_") as env_dir, tempfile.TemporaryDirectory(prefix="nautical_arg_modify_") as arg_dir:
        os.environ["TASKDATA"] = env_dir
        sys.argv = ["on-modify.nautical", "api:2", "command:modify", f"data:{arg_dir}"]
        try:
            mod = _load_hook_module(hook, "_nautical_on_modify_data_arg_precedence_test")
        finally:
            sys.argv = prev_argv
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata
        expect(Path(mod.TW_DATA_DIR) == Path(arg_dir), f"expected argv data dir, got: {mod.TW_DATA_DIR}")

def test_on_add_data_arg_overrides_taskdata_env():
    """on-add should prefer hook argv data: over TASKDATA env when both are present."""
    hook = _find_hook_file("on-add.nautical")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    with tempfile.TemporaryDirectory(prefix="nautical_env_add_") as env_dir, tempfile.TemporaryDirectory(prefix="nautical_arg_add_") as arg_dir:
        os.environ["TASKDATA"] = env_dir
        sys.argv = ["on-add.nautical", "api:2", "command:add", f"data:{arg_dir}"]
        try:
            mod = _load_hook_module(hook, "_nautical_on_add_data_arg_precedence_test")
        finally:
            sys.argv = prev_argv
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata
        expect(Path(mod.TW_DATA_DIR) == Path(arg_dir), f"expected argv data dir, got: {mod.TW_DATA_DIR}")

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

def test_on_modify_requires_integration_context_helper():
    """on-modify should fail closed when the integration context is unavailable."""
    _assert_hook_requires_integration_context("on-modify.nautical", "_nautical_on_modify_requires_context_test")

def test_on_exit_requires_integration_context_helper():
    """on-exit should fail closed when the integration context is unavailable."""
    _assert_hook_requires_integration_context("on-exit.nautical", "_nautical_on_exit_requires_context_test")

def test_on_modify_promotes_chain_when_task_becomes_nautical():
    """Tasks that gain Nautical fields on modify should be promoted to chain:on."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_chain_promotion_test")
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


def test_on_modify_promotes_chain_emits_upgrade_panel():
    """Promotion to Nautical should show a small informative panel."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_chain_upgrade_panel_test")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000446",
        "description": "plain task",
        "status": "pending",
    }
    new = {
        "uuid": "00000000-0000-4000-8000-000000000446",
        "description": "plain task",
        "status": "pending",
        "anchor": "w:mon",
        "due": "20260727T090000Z",
        "chain": "off",
    }
    captured = {}

    orig_panel = mod._panel
    orig_print_task = mod._print_task
    try:
        def fake_panel(title, rows, *, kind=None):
            captured["title"] = title
            captured["rows"] = list(rows)
            captured["kind"] = kind

        mod._panel = fake_panel
        mod._print_task = lambda task: captured.setdefault("task", dict(task))
        _modify_effect(mod, "handle_non_completion", old, new, _test_operator_uow())
    finally:
        mod._panel = orig_panel
        mod._print_task = orig_print_task

    expect(captured.get("title") == "⚓ Nautical enabled", f"expected upgrade panel, got {captured!r}")
    expect(captured.get("kind") == "note", f"expected note panel, got {captured!r}")
    rows = captured.get("rows") or []
    expect(not any(k == "Chain" for k, _v in rows), f"enabled panel should omit redundant chain:on row, got {rows!r}")
    expect(any(k == "Source" and v == "anchor" for k, v in rows), f"expected anchor source row, got {rows!r}")
    expect(any(k == "Anchor" and v == "w:mon" for k, v in rows), f"expected added anchor row, got {rows!r}")
    expect(any(k == "Natural" and "Monday" in v for k, v in rows), f"expected natural anchor explanation, got {rows!r}")
    expect(any(k == "Mode" and v.startswith("SKIP —") for k, v in rows), f"expected anchor mode explanation, got {rows!r}")
    expect(any(k == "First next" for k, _v in rows), f"expected first calculated occurrence, got {rows!r}")
    expect(new.get("chain") == "on", f"promotion should set chain:on, got {new!r}")
    expect(bool((new.get("chainID") or "").strip()), f"promotion should stamp chainID, got {new!r}")


def test_on_modify_promotes_cp_emits_period_explanation():
    """Promotion by cp should show the configured period and readable meaning."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_cp_upgrade_panel_test")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000448",
        "description": "plain task",
        "status": "pending",
    }
    new = {**old, "cp": "7d", "due": "20260727T090000Z", "chain": "off"}
    captured = {}
    original_panel = mod._panel
    original_print_task = mod._print_task
    try:
        mod._panel = lambda title, rows, *, kind=None: captured.update(
            title=title, rows=list(rows), kind=kind
        )
        mod._print_task = lambda task: None
        _modify_effect(mod, "handle_non_completion", old, new, _test_operator_uow())
    finally:
        mod._panel = original_panel
        mod._print_task = original_print_task

    rows = captured.get("rows") or []
    expect(captured.get("title") == "⚓ Nautical enabled", f"expected upgrade panel, got {captured!r}")
    expect(any(k == "Period" and v == "7d" for k, v in rows), f"expected added period row, got {rows!r}")
    expect(any(k == "Natural" and v == "Every 7d" for k, v in rows), f"expected natural period explanation, got {rows!r}")
    expect(any(k == "First next" for k, _v in rows), f"expected first calculated occurrence, got {rows!r}")


def test_on_modify_disables_chain_emits_disabled_panel():
    """Disabling Nautical recurrence should show a small informative panel."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_chain_disabled_panel_test")

    base_old = {
        "uuid": "00000000-0000-4000-8000-000000000447",
        "description": "nautical task",
        "status": "pending",
        "anchor": "w:mon",
        "chain": "on",
        "chainID": "abcd1234",
    }
    disable_cases = [
        {
            "label": "chain_off",
            "new": {**base_old, "chain": "off"},
            "expect_reason": "disabled because chain:off",
            "expect_source": "anchor",
        },
        {
            "label": "fields_cleared",
            "new": {
                "uuid": base_old["uuid"],
                "description": base_old["description"],
                "status": base_old["status"],
                "chain": "on",
                "chainID": "abcd1234",
                "anchor_mode": "skip",
            },
            "expect_reason": "no longer has Nautical recurrence fields",
            "expect_source": None,
        },
    ]

    for case in disable_cases:
        old = dict(base_old)
        new = dict(case["new"])
        captured = {"panels": []}

        orig_panel = mod._panel
        orig_print_task = mod._print_task
        try:
            def fake_panel(title, rows, *, kind=None):
                captured["panels"].append((title, list(rows), kind))

            mod._panel = fake_panel
            mod._print_task = lambda task: captured.setdefault("task", dict(task))
            _modify_effect(mod, "handle_non_completion", old, new, _test_operator_uow())
        finally:
            mod._panel = orig_panel
            mod._print_task = orig_print_task

        disabled_panels = [panel for panel in captured["panels"] if panel[0] == "⚓ Nautical disabled"]
        expect(disabled_panels, f"{case['label']} expected disabled panel, got {captured!r}")
        _title, rows, kind = disabled_panels[-1]
        expect(kind == "disabled", f"{case['label']} expected disabled panel kind, got {captured!r}")
        expect(any(k == "Reason" and case["expect_reason"] in str(v) for k, v in rows), f"{case['label']} expected reason row, got {rows!r}")
        if case["expect_source"] is None:
            expect(not any(k == "Source" for k, _v in rows), f"{case['label']} should not include source row, got {rows!r}")
        else:
            expect(any(k == "Source" and v == case["expect_source"] for k, v in rows), f"{case['label']} expected source row, got {rows!r}")
        expect(any(k == "Chain" and v == "off" for k, v in rows), f"{case['label']} expected chain:off row, got {rows!r}")
        if case["label"] == "fields_cleared":
            expect(
                any(title == "⛔ Nautical chain stopped" and panel_kind == "summary" for title, _rows, panel_kind in captured["panels"]),
                f"{case['label']} should also show the finished-chain summary: {captured!r}",
            )
            summary_rows = next(rows for title, rows, _kind in captured["panels"] if title == "⛔ Nautical chain stopped")
            expect(any(label == "Reason" and "removed" in str(value) for label, value in summary_rows), f"summary should explain removed recurrence: {captured!r}")
        elif case["label"] == "chain_off":
            expect(
                any(title == "⛔ Nautical chain stopped" and panel_kind == "summary" for title, _rows, panel_kind in captured["panels"]),
                f"{case['label']} should also show the finished-chain summary: {captured!r}",
            )
        expect(new.get("chain") == "off", f"{case['label']} should set chain:off, got {new!r}")


def test_on_modify_resumes_chain_emits_resumed_panel():
    """Explicitly resuming an existing recurrence should acknowledge its effect."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_chain_resumed_panel_test")
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

    orig_panel = mod._panel
    orig_print_task = mod._print_task
    try:
        def fake_panel(title, rows, *, kind=None):
            captured["title"] = title
            captured["rows"] = list(rows)
            captured["kind"] = kind

        mod._panel = fake_panel
        mod._print_task = lambda task: captured.setdefault("task", dict(task))
        _modify_effect(mod, "handle_non_completion", old, new, _test_operator_uow())
    finally:
        mod._panel = orig_panel
        mod._print_task = orig_print_task

    expect(captured.get("title") == "⚓ Nautical resumed", f"expected resumed panel, got {captured!r}")
    expect(captured.get("kind") == "note", f"expected note panel, got {captured!r}")
    rows = captured.get("rows") or []
    expect(("Source", "anchor") in rows, f"expected anchor source row, got {rows!r}")
    expect(
        ("Chain", "[dim]off[/] [cyan]→[/] [bold]on[/]") in rows,
        f"expected styled chain transition row, got {rows!r}",
    )
    # Resume feedback is intentionally compact; callers can obtain the next
    # occurrence through the query API without hook-side recomputation.
    expect(captured.get("task") == new, f"modified task should still be printed: {captured!r}")


def test_on_modify_resume_wrapper_preserves_json_and_emits_panel():
    """The thin wrapper must route chain resume through feedback without polluting stdout."""
    hook = _find_hook_file("on-modify.nautical")
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
        proc = _run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra={"TASKDATA": td},
        )

    expect(proc.returncode == 0, f"resume hook failed: {proc.stderr!r}")
    expect(_assert_stdout_json_only(proc.stdout) == new, f"resume hook changed task JSON: {proc.stdout!r}")
    expect("Nautical resumed" in proc.stderr, f"resume panel missing from stderr: {proc.stderr!r}")
    expect("off → on" in proc.stderr, f"chain transition missing from panel: {proc.stderr!r}")


def test_on_modify_recurrence_update_emits_ack_panel():
    """Changing recurrence settings on an existing Nautical task should be acknowledged."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_recurrence_update_panel_test")
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

    orig_panel = mod._panel
    orig_print_task = mod._print_task
    try:
        def fake_panel(title, rows, *, kind=None):
            captured["title"] = title
            captured["rows"] = list(rows)
            captured["kind"] = kind

        mod._panel = fake_panel
        mod._print_task = lambda task: captured.setdefault("task", dict(task))
        _modify_effect(mod, "handle_non_completion", old, new, _test_operator_uow())
    finally:
        mod._panel = orig_panel
        mod._print_task = orig_print_task

    expect(captured.get("title") == "⚓ Nautical recurrence updated", f"expected recurrence update panel, got {captured!r}")
    expect(captured.get("kind") == "note", f"expected note panel, got {captured!r}")
    rows = captured.get("rows") or []
    expect(
        ("Changed", "Anchor: [dim]w:mon[/] [cyan]→[/] [bold]w:tue,thu[/]") in rows,
        f"expected styled anchor change row, got {rows!r}",
    )
    expect(any(k == "Natural" and "Tuesday" in str(v) and "Thursday" in str(v) for k, v in rows), f"expected natural row, got {rows!r}")
    expect(any(k == "First next" for k, _v in rows), f"expected recalculated first occurrence, got {rows!r}")
    expect(not any(k == "Chain" for k, _v in rows), f"recurrence panel should omit redundant chain:on row, got {rows!r}")
    expect(captured.get("task") == new, f"modified task should still be printed: {captured!r}")


def test_on_modify_recurrence_update_groups_and_flattens_changes():
    """Multi-field recurrence updates stay grouped and readable in one-line modes."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_recurrence_update_layout_test")
    changes = [
        ("anchor", "w:mon", "w:tue"),
        ("chainMax", "5", "8"),
    ]
    rich_rows = [
        ("Changed", "Anchor: [dim]w:mon[/] [cyan]→[/] [bold]w:tue[/]"),
        ("Changed", "Max links: [dim]5[/] [cyan]→[/] [bold]8[/]"),
    ]
    feedback = mod.core._import_sibling("modify_feedback")
    grouped = feedback._recurrence_update_panel_rows(
        changes,
        rich_rows,
        panel_mode=mod.core.PANEL_MODE,
        strip_markup=mod.core.strip_rich_markup,
    )
    expect(grouped[1][0] is None, f"expected spacing between recurrence and limits: {grouped!r}")

    previous_mode = mod.core.PANEL_MODE
    try:
        mod.core.PANEL_MODE = "text"
        flattened = feedback._recurrence_update_panel_rows(
            changes,
            rich_rows,
            panel_mode=mod.core.PANEL_MODE,
            strip_markup=mod.core.strip_rich_markup,
        )
    finally:
        mod.core.PANEL_MODE = previous_mode
    expect(flattened[0][0] == "Changes", f"expected one-line change summary: {flattened!r}")
    expect("Anchor:" in flattened[0][1] and "Max links:" in flattened[0][1], f"one-line summary omitted a change: {flattened!r}")


def test_on_modify_native_until_update_explains_carry():
    """Changing native until should acknowledge its exact or calendar carry policy."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_until_update_panel_test")
    due = mod.core.build_local_datetime(date(2026, 8, 3), (10, 0)).astimezone(timezone.utc)
    old_until = mod.core.build_local_datetime(date(2026, 8, 3), (18, 0)).astimezone(timezone.utc)
    new_until = mod.core.build_local_datetime(date(2026, 8, 4), (0, 0)).astimezone(timezone.utc) + timedelta(seconds=1)
    old = {
        "uuid": "00000000-0000-4000-8000-000000000446",
        "description": "nautical expiration update",
        "status": "pending",
        "cp": "1d",
        "due": mod.core.fmt_isoz(due),
        "until": mod.core.fmt_isoz(old_until),
        "chain": "on",
        "chainID": "abcd1234",
    }
    new = {**old, "until": mod.core.fmt_isoz(new_until)}
    captured = {}

    orig_panel = mod._panel
    orig_print_task = mod._print_task
    try:
        mod._panel = lambda title, rows, *, kind=None: captured.update(title=title, rows=list(rows), kind=kind)
        mod._print_task = lambda task: captured.setdefault("task", dict(task))
        _modify_effect(mod, "handle_non_completion", old, new, _test_operator_uow())
    finally:
        mod._panel = orig_panel
        mod._print_task = orig_print_task

    rows = captured.get("rows") or []
    expect(captured.get("title") == "⚓ Nautical recurrence updated", f"unexpected expiration panel: {captured!r}")
    expect(captured.get("kind") == "note", f"unexpected expiration panel style: {captured!r}")
    expect(any(label == "Changed" and "Expiration:" in str(value) and "2026-08-03" in str(value) and "2026-08-04" in str(value) for label, value in rows), f"missing expiration diff: {rows!r}")
    expect(("Carry", "Exact · 14h 00m 01s after occurrence") in rows, f"missing exact carry explanation: {rows!r}")
    expect(captured.get("task") == new, f"modified task should still be printed: {captured!r}")


def test_on_modify_limit_update_emits_effective_boundaries():
    """Changing chain limits should acknowledge both boundaries without speculative dates."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_limit_update_panel_test")
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

    orig_panel = mod._panel
    orig_print_task = mod._print_task
    try:
        mod._panel = lambda title, rows, *, kind=None: panels.append((title, list(rows), kind))
        mod._print_task = lambda _task: None
        _modify_effect(mod, "handle_non_completion", old, new, _test_operator_uow())
        _modify_effect(mod, "handle_non_completion", new, cleared, _test_operator_uow())
    finally:
        mod._panel = orig_panel
        mod._print_task = orig_print_task

    expect(len(panels) == 2, f"each limit update should emit one panel: {panels!r}")
    title, rows, kind = panels[0]
    expect(title == "⚓ Nautical recurrence updated" and kind == "note", f"unexpected limit panel: {panels!r}")
    expect(
        ("Changed", "Max links: [dim]5[/] [cyan]→[/] [bold]8[/]") in rows,
        f"missing styled chainMax update: {rows!r}",
    )
    expect(
        any(
            label == "Changed" and "Chain end point:" in value and "2099-08-10" in value and "2099-08-20" in value
            for label, value in rows
        ),
        f"missing localized chainUntil update: {rows!r}",
    )
    expect(("Final link", "#8") in rows, f"missing final link boundary: {rows!r}")
    expect(
        sum(label == "Changed" and "Chain end point:" in str(value) for label, value in rows) == 1,
        f"chain end point should not be repeated: {rows!r}",
    )
    expect(("Effective", "Whichever boundary is reached first") in rows, f"missing effective limit rule: {rows!r}")
    expect(("Chain limits", "None") in panels[1][1], f"clearing both limits should be explicit: {panels[1]!r}")
    expect(("Removed", "Max links: [dim]8[/]") in panels[1][1], f"cleared max link should be marked removed: {panels[1]!r}")
    expect(any(label == "Removed" and "Chain end point:" in str(value) for label, value in panels[1][1]), f"cleared chain end should be marked removed: {panels[1]!r}")


def test_on_add_lowercase_chainid_does_not_mark_nautical():
    """on-add should ignore lowercase chainid when deciding whether a task is Nautical."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_chainid_alias_test")
    expect(
        not mod._module("modify_lifecycle").task_has_nautical_fields({"chainid": "legacy-1234"}),
        "lowercase chainid should not mark a task Nautical on add",
    )
    expect(
        mod._module("modify_lifecycle").task_has_nautical_fields({"chainID": "abcd1234"}),
        "canonical chainID should still mark a task Nautical on add",
    )

def test_on_modify_read_two_fuzz_inputs():
    """on-modify input parsing should be strict and return JSON errors on bad input."""
    hook = _find_hook_file("on-modify.nautical")
    cases = [
        ("", "empty"),
        ("{not-json}", "invalid"),
        (json.dumps({"status": "pending", "anchor": "w:mon"}), "json"),
        (json.dumps({"status": "pending", "anchor": "w:mon"}) + "\n" + json.dumps({"status": "pending", "anchor": "w:mon"}), "json"),
        ("  \n" + json.dumps({"status": "pending", "anchor": "w:mon"}) + "\n", "json"),
    ]
    for raw, mode in cases:
        p = _run_hook_script_raw(hook, raw)
        if mode in {"empty", "invalid", "json"}:
            expect(p.returncode != 0, f"on-modify should fail for case {mode}")
            expect((p.stdout or "").strip() == "", f"expected no stdout on failure, got: {p.stdout!r}")
        else:
            expect(p.returncode == 0, f"on-modify returned {p.returncode} for case {mode}")
            _assert_stdout_json_only(p.stdout)

def test_on_add_read_one_fuzz_inputs():
    """on-add input parsing should reject malformed JSON and empty input."""
    hook = _find_hook_file("on-add.nautical")
    cases = [
        ("", "empty"),
        ("{not-json}", "invalid"),
        (json.dumps({"status": "pending"}) + "\n" + json.dumps({"status": "pending"}), "multi"),
        ("  \n" + json.dumps({"status": "pending"}) + "\n{bad", "trailing"),
    ]
    for raw, mode in cases:
        p = _run_hook_script_raw(hook, raw)
        expect(p.returncode != 0, f"on-add should fail for case {mode}")
        expect((p.stdout or "").strip() == "", f"expected no stdout on failure, got: {p.stdout!r}")

def test_on_modify_read_two_invalid_trailing():
    """on-modify should fail on extra garbage after JSON objects."""
    hook = _find_hook_file("on-modify.nautical")
    raw = json.dumps({"status": "pending"}) + "\n" + json.dumps({"status": "pending"}) + "\n" + "{bad"
    p = _run_hook_script_raw(hook, raw)
    expect(p.returncode != 0, "on-modify should fail on trailing garbage")
    expect((p.stdout or "").strip() == "", f"expected no stdout on failure, got: {p.stdout!r}")

def test_on_modify_read_two_array_uuid_mismatch_fails():
    """on-modify array input should reject old/new UUID mismatches for Nautical tasks."""
    hook = _find_hook_file("on-modify.nautical")
    raw = json.dumps(
        [
            {"uuid": "00000000-0000-4000-8000-000000000111", "status": "pending", "anchor": "w:mon"},
            {"uuid": "00000000-0000-4000-8000-000000000222", "status": "completed", "anchor": "w:mon"},
        ]
    )
    p = _run_hook_script_raw(hook, raw)
    expect(p.returncode != 0, "on-modify should fail for mismatched UUIDs in array input")
    expect((p.stdout or "").strip() == "", f"expected no stdout on failure, got: {p.stdout!r}")

def test_on_modify_read_two_array_single_missing_uuid_fails():
    """on-modify array input with one dict and no Nautical fields should be ignored."""
    hook = _find_hook_file("on-modify.nautical")
    raw = json.dumps([{"status": "deleted"}])
    p = _run_hook_script_raw(hook, raw)
    expect(p.returncode == 0, "on-modify should ignore array input with one plain non-nautical task lacking UUID")
    _assert_stdout_json_only(p.stdout)


def test_on_modify_read_two_single_plain_delete_without_uuid_is_ignored():
    """on-modify single-task plain deletes without Nautical fields should not fail on missing UUID."""
    hook = _find_hook_file("on-modify.nautical")
    raw = json.dumps({"status": "deleted", "description": "plain taskwarrior recurrence delete"})
    p = _run_hook_script_raw(hook, raw)
    expect(p.returncode == 0, f"expected plain delete without uuid to be ignored, got rc={p.returncode}, stderr={p.stderr!r}")
    _assert_stdout_json_only(p.stdout)


def test_on_modify_read_two_uuid_mismatch_without_nautical_fields_is_ignored():
    """on-modify should not fail UUID mismatch for plain Taskwarrior deletes without Nautical fields."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_uuid_mismatch_non_nautical_test")
    old = {"uuid": "00000000-0000-4000-8000-000000000111", "status": "pending"}
    new = {"uuid": "00000000-0000-4000-8000-000000000222", "status": "deleted"}
    got_old, got_new = mod._validate_modify_pair(old, new)
    expect(got_old is old and got_new is new, f"expected UUID mismatch to be ignored for non-nautical delete: {(got_old, got_new)!r}")


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


def test_delete_chain_summary_span_uses_stop_time_without_last_end():
    """Deletion summaries should show active chain span even when the deleted pending task has no end."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_delete_chain_summary_span_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    first, last, span = mod._diagnostics_effects.span_fields(
        "cid",
        [{"uuid": "root", "due": "20260101T000000Z"}, {"uuid": "tail", "status": "deleted"}],
        stop_at=datetime(2026, 1, 11, tzinfo=timezone.utc),
        stopped_by_delete=True,
    )
    expect(first is not None, "first due should parse")
    expect(last is None, f"deleted pending task should not create last end: {last!r}")
    expect(span.startswith("Active for "), f"delete span should describe active duration: {span!r}")
    expect("before deletion" in span, f"delete span should mention deletion: {span!r}")


def test_end_summary_history_marks_deleted_pending_tail():
    """End-summary history should not mark deleted pending tasks as completed."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_delete_chain_summary_history_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    lines = mod._diagnostics_effects.last_n_timeline(
        [
            {
                "uuid": "00000000-0000-4000-8000-000000000111",
                "status": "completed",
                "link": 1,
                "due": "20260101T000000Z",
                "end": "20260101T000000Z",
            },
            {
                "uuid": "00000000-0000-4000-8000-000000000222",
                "status": "deleted",
                "link": 2,
                "due": "20260102T000000Z",
            },
        ],
        n=6,
    )
    got = "\n".join(core.strip_rich_markup(line) for line in lines)
    expect("#2" in got and "×" in got and "deleted" in got, f"deleted tail should be marked as deleted: {got!r}")
    expect("#2  ✓" not in got and "(no end)" not in got.split("#1", 1)[0], f"deleted tail should not look completed/no-end: {got!r}")


def test_delete_chain_summary_uses_stopped_title():
    """Deletion-stopped summaries should use stopped wording in the panel title."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_delete_chain_summary_title_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    captured = {}
    task = {
        "uuid": "00000000-0000-4000-8000-000000000222",
        "status": "deleted",
        "chainID": "00000000",
        "link": 2,
        "cp": "1d",
        "due": "20260102T000000Z",
    }
    chain = [
        {
            "uuid": "00000000-0000-4000-8000-000000000111",
            "status": "completed",
            "chainID": "00000000",
            "link": 1,
            "cp": "1d",
            "due": "20260101T000000Z",
            "end": "20260101T000000Z",
        },
        task,
    ]

    ui_effects = mod._module("modify_ui_effects")
    prev_panel = ui_effects.panel
    read_effects = mod._module("modify_read_effects")
    prev_export = read_effects.export_chain_required
    try:
        ui_effects.panel = lambda _host, title, rows, **kwargs: captured.update(
            {"title": title, "rows": rows, "kind": kwargs.get("kind")}
        )
        read_effects.export_chain_required = lambda _host, _task: list(chain)
        mod._diagnostics_effects.end_chain_summary(
            task,
            "Pending task deleted.",
            datetime(2026, 1, 3, tzinfo=timezone.utc),
            current_task=task,
        )
    finally:
        ui_effects.panel = prev_panel
        read_effects.export_chain_required = prev_export

    expect(captured.get("title") == "⛔ Chain stopped – summary", f"unexpected delete summary title: {captured!r}")


def test_on_modify_expiration_panel_explains_carry():
    """The immediate expiration panel should explain the child's carry policy."""
    from nautical_core.lifecycle_models import LifecycleAction
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_expiration_carry_panel_test")
    expiration = mod._module("modify_expiration")
    child_due = mod.core.build_local_datetime(date(2026, 7, 27), (9, 0)).astimezone(timezone.utc)
    child_until = mod.core.build_local_datetime(date(2026, 8, 2), (23, 59)).astimezone(timezone.utc)
    parent = {
        "description": "Take the trash out",
        "link": 1,
    }
    child_draft = _task_draft(
        {
            "uuid": "22222222-0000-4000-8000-000000000001",
            "description": "Take the trash out",
            "status": "pending",
            "chain": "on",
            "chainID": "expiration-panel",
            "link": 2,
            "cp": "1d",
            "due": child_due,
            "until": mod.core.fmt_isoz(child_until),
        }
    )
    plan = SimpleNamespace(
        plan=SimpleNamespace(
            action=LifecycleAction.SPAWN_CHILD,
            child_dict=lambda: {"until": mod.core.fmt_isoz(child_until)},
        ),
        child_due=child_due,
        child_draft=child_draft,
        next_link=2,
        reason="expired link missing next link",
    )
    captured = {}
    original_panel = mod._panel
    try:
        mod._panel = lambda title, rows, **kwargs: captured.update(title=title, rows=list(rows), kwargs=dict(kwargs))
        expiration._render_recovery_panel(
            parent,
            plan,
            services=_modify_effect(mod, "expiration_services"),
            result="[green]Next occurrence created[/]",
            child_short="22222222",
        )
    finally:
        mod._panel = original_panel

    add_validation = mod.core._import_sibling("add_validation")
    expected = add_validation.describe_native_until_carry(child_until, child_due, to_local=mod.core.to_local)
    expect(captured.get("title") == "⌛ Nautical occurrence expired", f"unexpected expiration panel: {captured!r}")
    expect(("Expiration", expected) in (captured.get("rows") or []), f"missing carry policy: {captured!r}")
    expect(any(label == "Next expires" for label, _value in (captured.get("rows") or [])), f"missing next expiration: {captured!r}")


def test_on_modify_expiration_delegates_to_extracted_orchestration():
    """The hook should leave expiration decisions to the focused orchestration module."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_expiration_extraction_test")
    expiration = mod._module("modify_expiration")
    captured = {}
    original = expiration.handle_expired_deleted_modify
    try:
        def handle(task, *, services):
            captured["task"] = task
            captured["services"] = services
            return True

        expiration.handle_expired_deleted_modify = handle
        task = {"uuid": "00000000-0000-4000-8000-000000000411"}
        expect(_modify_effect(mod, "handle_expired_deleted", task), "extracted expiration handler result was lost")
    finally:
        expiration.handle_expired_deleted_modify = original

    expect(captured.get("task") is task, f"expiration task was not delegated unchanged: {captured!r}")
    services = captured.get("services")
    expect(
        services is not None and callable(services.stage_recovery_plan),
        "expiration services were not wired to lifecycle staging",
    )


def test_on_modify_expiration_internal_failure_remains_recoverable():
    """An internal expiration-path failure should warn without stopping or crashing the chain."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_expiration_failure_test")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000416",
        "status": "pending",
        "description": "Recoverable expiration",
        "cp": "7d",
        "chain": "on",
        "chainID": "expire16",
        "link": 1,
        "due": "20260720T090000Z",
        "until": "20260726T235900Z",
    }
    new = dict(old, status="deleted", end="20260727T000000Z")
    panels = []
    stopped = []
    expiration = mod._module("modify_expiration")
    original = (expiration.handle_expired_deleted_modify, mod._panel, mod._diagnostics_effects.end_chain_summary)
    try:
        expiration.handle_expired_deleted_modify = (
            lambda _task, *, services: (_ for _ in ()).throw(RuntimeError("missing module"))
        )
        mod._panel = lambda title, rows, *, kind=None: panels.append((title, list(rows), kind))
        mod._diagnostics_effects.end_chain_summary = lambda *_args, **_kwargs: stopped.append(True)
        _modify_effect(mod, "handle_deleted", old, new, _test_operator_uow())
    finally:
        expiration.handle_expired_deleted_modify, mod._panel, mod._diagnostics_effects.end_chain_summary = original

    expect(not stopped, "internal expiration failure must not be treated as an intentional deletion")
    expect(new.get("chain") == "on" and not new.get("nextLink"), f"recovery evidence was lost: {new!r}")
    expect(
        panels and panels[0][2] == "warning"
        and any(label == "Action" and "nautical reconcile --apply" in value for label, value in panels[0][1]),
        f"missing recovery warning: {panels!r}",
    )


def test_on_modify_expiration_wrapper_preserves_json_stdout():
    """Expiration feedback must stay on stderr while the hook returns one strict task object."""
    hook = _find_hook_file("on-modify.nautical")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000421",
        "status": "pending",
        "description": "Expiration protocol",
        "cp": "7d",
        "chain": "on",
        "chainID": "expire21",
        "link": 1,
        "due": "20260720T090000Z",
        "until": "20260726T235900Z",
    }
    new = dict(old, status="deleted", end="20260727T000000Z")
    with tempfile.TemporaryDirectory() as td:
        proc = _run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra={"TASKDATA": td, "NO_COLOR": "1"},
        )

    expect(proc.returncode == 0, f"expiration hook failed: {proc.stderr!r}")
    output = _assert_stdout_json_only(proc.stdout)
    expect(output.get("status") == "deleted" and output.get("chain") == "on", f"unexpected hook task: {output!r}")
    expect("Nautical occurrence expired" not in proc.stderr, f"deferred recovery should stay quiet: {proc.stderr!r}")


def test_on_modify_manual_delete_persists_chain_off():
    """The real hook should distinguish an intentional early deletion from expiration."""
    hook = _find_hook_file("on-modify.nautical")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000422",
        "status": "pending",
        "description": "Manual delete protocol",
        "cp": "7d",
        "chain": "on",
        "chainID": "delete22",
        "link": 1,
        "due": "20260720T090000Z",
        "until": "20260726T235900Z",
    }
    new = dict(old, status="deleted", end="20260725T000000Z")
    with tempfile.TemporaryDirectory() as td:
        proc = _run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra={"TASKDATA": td, "NO_COLOR": "1"},
        )

    expect(proc.returncode == 0, f"manual-delete hook failed: {proc.stderr!r}")
    output = _assert_stdout_json_only(proc.stdout)
    expect(output.get("status") == "deleted", f"manual deletion status changed: {output!r}")
    expect(output.get("chain") == "off", f"manual deletion did not stop the chain: {output!r}")


def test_on_modify_invalid_anchor_has_no_stdout():
    """on-modify should keep stdout empty on semantic validation failures."""
    hook = _find_hook_file("on-modify.nautical")
    old = {
        "uuid": "00000000-0000-4000-8000-000000000611",
        "status": "pending",
        "description": "invalid anchor test",
    }
    new = dict(old)
    new["anchor"] = "bad"
    raw = json.dumps(old) + "\n" + json.dumps(new)
    p = _run_hook_script_raw(hook, raw)
    expect(p.returncode != 0, "on-modify should fail on invalid anchor")
    expect((p.stdout or "").strip() == "", f"expected no stdout on failure, got: {p.stdout!r}")

def test_on_add_rejects_oversized_stdin_early():
    """on-add should reject stdin over _MAX_JSON_BYTES before JSON parsing."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_oversized_input_test")
    mod._MAX_JSON_BYTES = 32
    raw = json.dumps({"uuid": "u", "status": "pending", "description": "x" * 256})

    stdin = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    out = io.StringIO()
    err = io.StringIO()
    orig_stdin, orig_stdout, orig_stderr = sys.stdin, sys.stdout, sys.stderr
    try:
        sys.stdin, sys.stdout, sys.stderr = stdin, out, err
        try:
            mod.main()
            raise AssertionError("on-add should fail on oversized stdin")
        except SystemExit as e:
            expect(e.code == 1, f"unexpected exit code: {e.code}")
    finally:
        sys.stdin, sys.stdout, sys.stderr = orig_stdin, orig_stdout, orig_stderr
    expect((out.getvalue() or "").strip() == "", f"expected no stdout on oversized input, got: {out.getvalue()!r}")

def test_on_modify_rejects_oversized_stdin_early():
    """on-modify should reject stdin over _MAX_JSON_BYTES before object parsing."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_oversized_input_test")
    mod._MAX_JSON_BYTES = 32
    raw = json.dumps({"uuid": "u", "status": "pending", "description": "x" * 256})

    stdin = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    out = io.StringIO()
    err = io.StringIO()
    orig_stdin, orig_stdout, orig_stderr = sys.stdin, sys.stdout, sys.stderr
    try:
        sys.stdin, sys.stdout, sys.stderr = stdin, out, err
        try:
            mod._read_two()
            raise AssertionError("on-modify should fail on oversized stdin")
        except SystemExit as e:
            expect(e.code == 1, f"unexpected exit code: {e.code}")
    finally:
        sys.stdin, sys.stdout, sys.stderr = orig_stdin, orig_stdout, orig_stderr
    expect((out.getvalue() or "").strip() == "", f"expected no stdout on oversized input, got: {out.getvalue()!r}")

def test_health_check_json_ok_empty_taskdata():
    """health check should report ok for empty taskdata."""
    path = os.path.join(DEV_TOOLS, "nautical_health_check.py")
    with tempfile.TemporaryDirectory() as td:
        p = subprocess.run(
            [sys.executable, path, "--taskdata", td, "--json"],
            text=True,
            capture_output=True,
            timeout=8.0,
        )
        expect(p.returncode == 0, f"health check returned {p.returncode}: {p.stderr!r}")
        obj = json.loads((p.stdout or "").strip() or "{}")
        expect(obj.get("status") == "ok", f"unexpected status: {obj}")

def test_queue_status_and_doctor_report_schema_health():
    """Operator diagnostics should distinguish healthy and incompatible outboxes."""
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository

    status_path = os.path.join(CORE_TOOLS, "nautical_queue_status.py")
    doctor = _load_hook_module(
        os.path.join(CORE_TOOLS, "nautical_doctor.py"),
        "_nautical_doctor_queue_schema_test",
    )
    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td)
        repo = _LifecycleOutboxRepository(taskdata)
        expect(repo.open().ok, "lifecycle outbox did not initialize")
        db_path = repo.path

        proc = subprocess.run(
            [sys.executable, status_path, "--taskdata", td, "--json"],
            text=True,
            capture_output=True,
            timeout=8.0,
        )
        expect(proc.returncode == 0, f"healthy schema status failed: {proc.stderr!r}")
        payload = json.loads(proc.stdout)
        expect(payload.get("schema") == "nautical.lifecycle_outbox_status", f"outbox status schema missing: {payload!r}")
        expect(payload.get("schema_version") == 1, f"queue status schema version missing: {payload!r}")
        schema = (payload.get("outbox") or {}).get("schema") or {}
        expect(schema.get("status") == "ok", f"healthy schema was not reported: {payload!r}")
        expect((payload.get("outbox") or {}).get("integrity") == "ok", f"integrity was not checked: {payload!r}")

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("PRAGMA user_version = 3")
        proc = subprocess.run(
            [sys.executable, status_path, "--taskdata", td, "--json"],
            text=True,
            capture_output=True,
            timeout=8.0,
        )
        expect(proc.returncode == 2, f"future schema should be an operator error: {proc.stdout!r}")
        payload = json.loads(proc.stdout)
        expect(payload.get("status") == "error", f"future schema status was not error: {payload!r}")

        findings = []
        doctor._check_lifecycle_outbox(findings, taskdata, 300.0)
        schema_finding = next(item for item in findings if item.get("id") == "outbox.schema")
        expect(schema_finding.get("severity") == "error", f"Doctor missed future schema: {findings!r}")


def test_queue_status_json_ok_empty_taskdata():
    """Lifecycle outbox status should report ok for empty taskdata."""
    path = os.path.join(DEV_TOOLS, "nautical_queue_status.py")
    with tempfile.TemporaryDirectory() as td:
        p = subprocess.run(
            [sys.executable, path, "--taskdata", td, "--json"],
            text=True,
            capture_output=True,
            timeout=8.0,
        )
        expect(p.returncode == 0, f"queue status returned {p.returncode}: {p.stderr!r}")
        obj = json.loads((p.stdout or "").strip() or "{}")
        expect(obj.get("status") == "ok", f"unexpected queue status: {obj}")
        outbox = obj.get("outbox") or {}
        expect(outbox.get("states") == {}, f"unexpected lifecycle states: {obj}")
        expect((outbox.get("schema") or {}).get("status") == "absent", f"unexpected outbox schema: {obj}")


def test_queue_status_explicit_prune_reports_maintenance_result():
    """Retention cleanup is explicit and returns a structured maintenance result."""
    path = os.path.join(DEV_TOOLS, "nautical_queue_status.py")
    with tempfile.TemporaryDirectory() as td:
        proc = subprocess.run(
            [sys.executable, path, "--taskdata", td, "--prune-acknowledged", "--json"],
            text=True,
            capture_output=True,
            timeout=8,
        )
        expect(proc.returncode == 0, f"explicit queue maintenance failed: {proc.stderr!r}")
        payload = json.loads(proc.stdout)
        maintenance = payload.get("maintenance") or {}
        expect(maintenance.get("ok") is True, f"maintenance result was not successful: {payload!r}")
        expect(maintenance.get("removed") == 0, f"unexpected maintenance removal: {payload!r}")


def test_doctor_installation_json_and_verifier_contract():
    """Installation checks remain bounded and produce a concise report."""
    path = os.path.join(CORE_TOOLS, "nautical_doctor.py")
    with tempfile.TemporaryDirectory() as td:
        env = os.environ.copy()
        env.update(
            {
                "HOME": td,
                "TASKRC": os.path.join(td, ".taskrc"),
                "NAUTICAL_CONFIG": os.path.join(td, "missing-nautical.toml"),
                "TASKDATA": td,
            }
        )
        proc = subprocess.run(
            [sys.executable, path, "--taskdata", td, "--json", "--installation-only"],
            text=True,
            capture_output=True,
            env=env,
            timeout=10.0,
        )
        expect(proc.returncode in (0, 1, 2), f"doctor returned an invalid status: {proc.returncode}: {proc.stderr!r}")
        payload = json.loads((proc.stdout or "").strip() or "{}")
        expect(payload.get("schema") == "nautical.doctor", f"doctor schema missing: {payload!r}")
        expect(payload.get("schema_version") == 1, f"doctor schema version missing: {payload!r}")
        expect(payload.get("scope") == "installation", f"doctor installation scope missing: {payload!r}")
        expect(payload.get("counts") == {"tasks": 0, "nautical_tasks": 0, "chains": 0}, "installation check audited tasks")
        expect(payload.get("outbox") == {}, "installation check audited the lifecycle outbox")

        from nautical_core.tools.nautical_install_verify import build_report, render

        launcher = Path(td) / "nautical"
        launcher.write_text("#!/bin/sh\n", encoding="utf-8")
        launcher.chmod(0o700)
        verifier_payload = {
            "taskdata": td,
            "operator_findings": [
                {"code": "taskwarrior.version", "domain": "taskwarrior", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "taskdata.access", "domain": "taskdata", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "install.runtime", "domain": "install", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "hook.add", "domain": "hook", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "hook.modify", "domain": "hook", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "hook.exit", "domain": "hook", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "uda.anchor", "domain": "uda", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "config.timezone", "domain": "config", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "chains.carry.child_relative_offset", "domain": "chains", "severity": "error", "actionability": "actionable", "message": "historical", "guidance": "ignore history"},
            ],
        }
        report = build_report(verifier_payload, platform="Termux", launcher=launcher)
        expect(report.get("status") == "passed", f"operational findings leaked into installation status: {report!r}")
        expect(not report.get("manual_actions"), f"operational findings leaked into install actions: {report!r}")
        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            render(report)
        expect("\x1b[" not in rendered.getvalue(), "redirected installation report contains terminal styling")

        canonical_payload = dict(verifier_payload)
        canonical_payload["operator_findings"] = [
            item for item in verifier_payload["operator_findings"] if not str(item.get("code") or "").startswith("uda.")
        ]
        canonical_report = build_report(canonical_payload, platform="Linux", launcher=launcher)
        expect(canonical_report.get("status") == "passed", f"healthy canonical evidence was rejected: {canonical_report!r}")


def test_operator_queue_status_json_ok_empty_taskdata():
    """installed queue status should work from nautical_core/tools."""
    path = os.path.join(CORE_TOOLS, "nautical_queue_status.py")
    with tempfile.TemporaryDirectory() as td:
        p = subprocess.run(
            [sys.executable, path, "--taskdata", td, "--json"],
            text=True,
            capture_output=True,
            timeout=8.0,
        )
        expect(p.returncode == 0, f"operator queue status returned {p.returncode}: {p.stderr!r}")
        obj = json.loads((p.stdout or "").strip() or "{}")
        expect(obj.get("status") == "ok", f"unexpected operator queue status: {obj}")


def test_queue_status_warns_on_stale_processing_and_dead_letters():
    """Lifecycle outbox status should report expired leases and retry work."""
    path = os.path.join(DEV_TOOLS, "nautical_queue_status.py")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        state_dir = td_path / ".nautical-state"
        state_dir.mkdir(parents=True, exist_ok=True)

        db = state_dir / ".nautical_lifecycle_outbox.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("PRAGMA user_version = 2")
            conn.execute(
                """
                CREATE TABLE lifecycle_outbox (
                    intent_id TEXT PRIMARY KEY,
                    work_kind TEXT NOT NULL DEFAULT 'lifecycle',
                    plan_json TEXT NOT NULL,
                    plan_fingerprint TEXT NOT NULL,
                    parent_guard_json TEXT NOT NULL,
                    configuration_fingerprint TEXT NOT NULL,
                    schedule_fingerprint TEXT NOT NULL,
                    lifecycle_stage TEXT NOT NULL,
                    processing_state TEXT NOT NULL,
                    lease_owner TEXT NOT NULL DEFAULT '',
                    lease_expires_at REAL NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    failure_json TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    acknowledged_at REAL NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                "INSERT INTO lifecycle_outbox "
                "(intent_id, work_kind, plan_json, plan_fingerprint, parent_guard_json, configuration_fingerprint, "
                "schedule_fingerprint, lifecycle_stage, processing_state, lease_owner, lease_expires_at, "
                "attempts, failure_json, created_at, updated_at) "
                "VALUES (?, 'lifecycle', '{}', 'pf', '{}', 'cf', 'sf', 'planned', 'claimed', 'old-worker', 1.0, 3, '', 1.0, 1.0)",
                ("outbox-stale",),
            )
            conn.commit()

        p = subprocess.run(
            [
                sys.executable,
                path,
                "--taskdata",
                td,
                "--stale-after-seconds",
                "10",
                "--limit",
                "3",
                "--json",
            ],
            text=True,
            capture_output=True,
            timeout=8.0,
        )
        expect(p.returncode == 1, f"expected warn exit 1, got {p.returncode}: {p.stderr!r}")
        obj = json.loads((p.stdout or "").strip() or "{}")
        expect(obj.get("status") in {"warn", "attention"}, f"unexpected queue status: {obj}")
        outbox = obj.get("outbox") or {}
        states = outbox.get("states") or {}
        expect(int(states.get("claimed") or 0) == 1, f"unexpected outbox states: {outbox}")
        expect(int(outbox.get("stale_claims") or 0) == 1, f"unexpected stale count: {outbox}")
        expect(int(outbox.get("max_attempts") or 0) == 3, f"unexpected max attempts: {outbox}")
        expect(len(outbox.get("sample") or []) >= 1, f"expected sample rows: {outbox}")


def _write_fake_task_for_doctor(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
if args and args[-1] == "--version":
    print("3.4.2")
    raise SystemExit(0)
if len(args) >= 2 and args[-2] == "_get":
    key = args[-1]
    if key == "rc.hooks.location":
        print(os.environ.get("FAKE_HOOKS", ""))
        raise SystemExit(0)
    if key == "rc.data.location":
        print(os.environ.get("FAKE_DATA_DIR", ""))
        raise SystemExit(0)
    if key.startswith("rc.uda.") and key.endswith(".type"):
        name = key[len("rc.uda."):-len(".type")]
        expected = {
            "cp": "string", "chain": "string", "anchor": "string", "bc": "string",
            "anchor_file": "string", "anchor_mode": "string",
            "omit": "string", "omit_file": "string",
            "chainMax": "numeric", "chainUntil": "date",
            "prevLink": "string", "nextLink": "string",
            "link": "numeric", "chainID": "string",
        }
        if name == os.environ.get("FAKE_WRONG_UDA"):
            print("date")
        else:
            print(expected.get(name, ""))
        raise SystemExit(0)
if "export" in args:
    print(os.environ.get("FAKE_EXPORT", "[]"))
    raise SystemExit(0)
print("unsupported", file=sys.stderr)
raise SystemExit(2)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _install_doctor_hook_wrappers(hooks_dir: Path) -> None:
    for name in ("on-add.nautical", "on-modify.nautical", "on-exit.nautical"):
        shutil.copy2(Path(ROOT) / name, hooks_dir / name)


def test_doctor_reports_healthy_installation():
    """doctor should report ok for a complete installation with clean chain state."""
    path = os.path.join(DEV_TOOLS, "nautical_doctor.py")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        hooks = td_path / "hooks"
        hooks.mkdir()
        _install_doctor_hook_wrappers(hooks)
        (td_path / "config-nautical.toml").write_text('tz = "UTC"\n', encoding="utf-8")
        fake_task = td_path / "task"
        _write_fake_task_for_doctor(fake_task)
        rows = [
            {
                "uuid": "aaaaaaaa-0000-4000-8000-000000000901",
                "status": "completed",
                "chain": "on",
                "cp": "1d",
                "chain": "on",
                "status": "completed",
                "chain": "on",
                "status": "completed",
                "chainID": "cid",
                "link": 1,
                "nextLink": "bbbbbbbb",
            },
            {
                "uuid": "bbbbbbbb-0000-4000-8000-000000000902",
                "status": "pending",
                "chain": "on",
                "cp": "1d",
                "chain": "on",
                "status": "pending",
                "chain": "on",
                "status": "pending",
                "chainID": "cid",
                "link": 2,
                "prevLink": "aaaaaaaa",
            },
        ]
        env = os.environ.copy()
        env["NAUTICAL_CORE_PATH"] = ROOT
        env["NAUTICAL_TRUST_CORE_PATH"] = "1"
        env["FAKE_HOOKS"] = str(hooks)
        env["FAKE_EXPORT"] = json.dumps(rows)
        p = subprocess.run(
            [sys.executable, path, "--taskdata", td, "--task-bin", str(fake_task), "--json"],
            text=True,
            capture_output=True,
            env=env,
            timeout=8.0,
        )
        expect(p.returncode == 0, f"doctor returned {p.returncode}: {p.stderr!r} {p.stdout!r}")
        obj = json.loads((p.stdout or "").strip() or "{}")
        expect(obj.get("status") == "ok", f"unexpected doctor status: {obj}")
        expect((obj.get("counts") or {}).get("chains") == 1, f"unexpected doctor counts: {obj}")
        findings = _doctor_findings(obj)
        expect(
            any(item.get("id") == "uda.registration" and item.get("severity") == "ok" for item in findings),
            f"healthy UDA registration evidence is missing: {obj}",
        )


def test_doctor_hook_inventory_allows_third_party_and_symlink_install():
    """Doctor should validate symlinked Nautical hooks without rejecting unrelated hooks."""
    path = os.path.join(CORE_TOOLS, "nautical_doctor.py")
    mod = _load_hook_module(path, "_nautical_doctor_hook_symlink_test")
    with tempfile.TemporaryDirectory() as td:
        hooks = Path(td) / "hooks"
        hooks.mkdir()
        (hooks / "on-add").symlink_to(Path(ROOT) / "on-add.nautical")
        for name in ("on-modify.nautical", "on-exit.nautical"):
            (hooks / name).symlink_to(Path(ROOT) / name)
        third_party = hooks / "on-add-third-party"
        third_party.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        third_party.chmod(0o755)

        findings = []
        runtimes = _doctor_hook_installation(mod,
            findings,
            hooks_dir=hooks,
            env={"NAUTICAL_CORE_PATH": ROOT, "NAUTICAL_TRUST_CORE_PATH": "1"},
        )

    expect(set(runtimes) == {"on-add", "on-modify", "on-exit"}, f"missing validated hooks: {findings!r}")
    expect(not any(item.get("severity") == "error" for item in findings), f"valid symlink install failed: {findings!r}")
    expect(
        Path(runtimes["on-modify"]["implementation"]) == Path(ROOT) / "nautical_core/hooks/modify_impl.py",
        f"wrong on-modify implementation selected: {runtimes!r}",
    )


def test_doctor_hook_inventory_rejects_duplicates_without_counting_backups():
    """Doctor should reject duplicate active Nautical hooks but ignore non-executable backups."""
    path = os.path.join(CORE_TOOLS, "nautical_doctor.py")
    mod = _load_hook_module(path, "_nautical_doctor_hook_duplicate_test")
    with tempfile.TemporaryDirectory() as td:
        hooks = Path(td) / "hooks"
        hooks.mkdir()
        _install_doctor_hook_wrappers(hooks)
        backup = hooks / "on-add-nautical-old.py"
        shutil.copy2(Path(ROOT) / "on-add.nautical", backup)
        env = {"NAUTICAL_CORE_PATH": ROOT, "NAUTICAL_TRUST_CORE_PATH": "1"}

        findings = []
        runtimes = _doctor_hook_installation(mod, findings, hooks_dir=hooks, env=env)
        ids = {item.get("id") for item in findings}
        expect("hook.on-add.duplicate" in ids, f"active duplicate was not detected: {findings!r}")
        expect("on-add" not in runtimes, f"ambiguous on-add runtime should not be selected: {runtimes!r}")

        backup.chmod(0o644)
        findings = []
        runtimes = _doctor_hook_installation(mod, findings, hooks_dir=hooks, env=env)
        ids = {item.get("id") for item in findings}
        expect("hook.on-add.duplicate" not in ids, f"inactive backup counted as active: {findings!r}")
        expect("on-add" in runtimes, f"active on-add wrapper was not selected: {findings!r}")

    with tempfile.TemporaryDirectory() as td:
        hooks = Path(td) / "hooks"
        hooks.mkdir()
        third_party = hooks / "on-add-third-party"
        third_party.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        third_party.chmod(0o755)
        findings = []
        _doctor_hook_installation(mod, findings, hooks_dir=hooks, env={})
        ids = {item.get("id") for item in findings}
        expect("hook.on-add.missing" in ids, f"third-party hook falsely satisfied Nautical: {findings!r}")


def test_doctor_hook_inventory_reports_incomplete_core_and_api_mismatch():
    """Doctor should diagnose partial core installs and wrapper/core API skew."""
    path = os.path.join(CORE_TOOLS, "nautical_doctor.py")
    mod = _load_hook_module(path, "_nautical_doctor_hook_compatibility_test")
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        hooks = base / "hooks"
        core_dir = base / "nautical_core"
        hooks.mkdir()
        core_dir.mkdir()
        (core_dir / "__init__.py").write_text("", encoding="utf-8")
        _install_doctor_hook_wrappers(hooks)

        findings = []
        runtimes = _doctor_hook_installation(mod, findings,
            hooks_dir=hooks,
            env={"NAUTICAL_CORE_PATH": str(base), "NAUTICAL_TRUST_CORE_PATH": "1"},
        )
        incompatible = [item for item in findings if str(item.get("id") or "").endswith(".incompatible")]
        expect(not runtimes, f"incomplete core should not produce validated runtimes: {runtimes!r}")
        expect(len(incompatible) == 3, f"incomplete runtime findings missing: {findings!r}")
        expect(
            all(((item.get("details") or {}).get("observed") or {}).get("missing") for item in incompatible),
            f"missing runtime files were not identified: {findings!r}",
        )

    with tempfile.TemporaryDirectory() as td:
        hooks = Path(td) / "hooks"
        hooks.mkdir()
        _install_doctor_hook_wrappers(hooks)
        add_hook = hooks / "on-add.nautical"
        add_hook.write_text(
            add_hook.read_text(encoding="utf-8").replace("_EXPECTED_IMPL_API = 1", "_EXPECTED_IMPL_API = 999"),
            encoding="utf-8",
        )
        add_hook.chmod(0o755)

        findings = []
        runtimes = _doctor_hook_installation(mod, findings,
            hooks_dir=hooks,
            env={"NAUTICAL_CORE_PATH": ROOT, "NAUTICAL_TRUST_CORE_PATH": "1"},
        )
        mismatch = next(item for item in findings if item.get("id") == "hook.on-add.incompatible")
        details = mismatch.get("details") or {}
        expect("on-add" not in runtimes, f"mismatched on-add runtime should not be selected: {runtimes!r}")
        expect((details.get("observed") or {}).get("expected_api") == 999, f"wrapper API missing from mismatch: {findings!r}")
        expect((details.get("observed") or {}).get("actual_api") == 1, f"implementation API missing from mismatch: {findings!r}")


def test_installer_dry_run_fresh_install_and_idempotent_reinstall():
    """Local installs should validate before mutation and safely reuse identical releases."""
    from nautical_core import install_runtime

    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td) / "taskdata"
        dry = install_runtime.install_release(
            source=Path(ROOT),
            taskdata=taskdata,
            release_id="dry-run",
            dry_run=True,
            smoke=False,
        )
        expect(dry.get("status") == "dry-run", f"unexpected dry-run result: {dry!r}")
        expect(dry.get("operation") == "install", f"fresh dry-run did not plan an install: {dry!r}")
        expect(dry.get("changed") is False, f"dry-run reported target changes: {dry!r}")
        expect(not taskdata.exists(), f"dry-run mutated the target: {taskdata}")

        installed = install_runtime.install_release(
            source=Path(ROOT),
            taskdata=taskdata,
            release_id="release-one",
            smoke=False,
        )
        expect(installed.get("status") == "installed", f"fresh install failed: {installed!r}")
        expect(installed.get("operation") == "install", f"fresh install action was unclear: {installed!r}")
        expect(installed.get("changed") is True, f"fresh install did not report its change: {installed!r}")
        expect(installed.get("active_release") == "release-one", f"active release was not reported: {installed!r}")
        expect((taskdata / ".nautical-runtime/current").is_symlink(), "current pointer was not installed")
        expect(os.readlink(taskdata / ".nautical-runtime/current") == "releases/release-one", "wrong active release")
        expect((taskdata / "nautical_core").is_symlink(), "stable core path was not installed")
        expect(
            install_runtime.validate_installed(taskdata, taskdata / "hooks", smoke=True)
            == {"on-add": 1, "on-modify": 1, "on-exit": 1},
            "installed hook layout did not pass post-install validation",
        )
        status = install_runtime.runtime_status(taskdata)
        expect(status.get("active_release") == "release-one", f"runtime status is wrong: {status!r}")
        expect(not status.get("errors"), f"fresh runtime status has errors: {status!r}")

        current = taskdata / ".nautical-runtime/current"
        wrapper = taskdata / "hooks/on-modify.nautical"
        current_inode = os.lstat(current).st_ino
        wrapper_stamp = wrapper.stat().st_mtime_ns
        repeated = install_runtime.install_release(
            source=Path(ROOT),
            taskdata=taskdata,
            release_id="release-one",
            smoke=False,
        )
        expect(repeated.get("reused_release") is True, f"identical reinstall was not reused: {repeated!r}")
        expect(repeated.get("operation") == "reuse", f"identical reinstall was not a no-op: {repeated!r}")
        expect(repeated.get("changed") is False, f"identical reinstall reported changes: {repeated!r}")
        expect(os.lstat(current).st_ino == current_inode, "same-release install rewrote the active pointer")
        expect(wrapper.stat().st_mtime_ns == wrapper_stamp, "same-release install rewrote a valid wrapper")

        wrapper.unlink()
        repaired = install_runtime.install_release(
            source=Path(ROOT),
            taskdata=taskdata,
            release_id="release-one",
            smoke=False,
        )
        expect(repaired.get("operation") == "repair", f"damaged active release was not repaired: {repaired!r}")
        expect(repaired.get("changed") is True and wrapper.is_file(), f"repair did not restore the wrapper: {repaired!r}")

        launcher = taskdata / "nautical"
        launcher.write_text("#!/usr/bin/env python3\n# stale launcher\n", encoding="utf-8")
        launcher.chmod(0o755)
        repaired = install_runtime.install_release(
            source=Path(ROOT),
            taskdata=taskdata,
            release_id="release-one",
            smoke=False,
        )
        expect(repaired.get("operation") == "repair", f"stale launcher was not repaired: {repaired!r}")
        expect(launcher.read_bytes() == (Path(ROOT) / "nautical").read_bytes(), "repair retained a stale launcher")

        command_path = Path(td) / "bin" / "nautical"
        command_install = install_runtime.install_release(
            source=Path(ROOT),
            taskdata=Path(td) / "command-taskdata",
            release_id="command-release",
            smoke=False,
            launcher_path=command_path,
        )
        expect(command_install.get("operation") == "install", f"command install failed: {command_install!r}")
        expect(command_path.is_symlink(), "user command launcher was not created as a symlink")
        expect(command_path.resolve() == (Path(td) / "command-taskdata" / "nautical").resolve(), "command launcher points to the wrong target")

        command_path.unlink()
        command_repair = install_runtime.install_release(
            source=Path(ROOT),
            taskdata=Path(td) / "command-taskdata",
            release_id="command-release",
            smoke=False,
            launcher_path=command_path,
        )
        expect(command_repair.get("operation") == "repair", f"missing command launcher was not repaired: {command_repair!r}")
        expect(command_path.is_symlink(), "repair did not recreate the user command launcher")


def test_installer_navigator_dependency_failure_is_actionable():
    """Navigator smoke failures should identify missing requirements without a traceback."""
    from nautical_core import install_runtime

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "nautical_navigator.py").write_text("# test navigator\n", encoding="utf-8")
        original_run = install_runtime.subprocess.run
        install_runtime.subprocess.run = lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=(
                "Traceback (most recent call last):\n"
                "  File 'nautical_navigator.py', line 1, in <module>\n"
                "ModuleNotFoundError: No module named 'rich'\n"
            ),
        )
        try:
            install_runtime._smoke_navigator(root)
        except install_runtime.InstallError as exc:
            message = str(exc)
            expect("missing Python module 'rich'" in message, f"missing dependency was not identified: {message}")
            expect("requirements.txt" in message, f"dependency remedy is not actionable: {message}")
            expect("Traceback" not in message, f"dependency failure leaked a traceback: {message}")
        else:
            raise AssertionError("missing Navigator dependency did not fail validation")
        finally:
            install_runtime.subprocess.run = original_run


def test_installer_upgrade_rollback_restores_active_runtime():
    """A failed upgrade should restore its pointer and every managed wrapper."""
    from nautical_core import install_runtime

    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td) / "taskdata"
        install_runtime.install_release(
            source=Path(ROOT),
            taskdata=taskdata,
            release_id="release-one",
            smoke=False,
        )
        current = taskdata / ".nautical-runtime/current"
        wrapper = taskdata / "hooks/on-modify.nautical"
        pointer_before = os.readlink(current)
        wrapper_before = wrapper.read_bytes()

        try:
            install_runtime.install_release(
                source=Path(ROOT),
                taskdata=taskdata,
                release_id="release-two",
                smoke=False,
                _fail_after="after_wrappers",
            )
            raise AssertionError("injected upgrade failure should have raised")
        except install_runtime.InstallError as exc:
            expect("injected failure" in str(exc), f"unexpected rollback error: {exc}")

        expect(os.readlink(current) == pointer_before, "failed upgrade did not restore current")
        expect(wrapper.read_bytes() == wrapper_before, "failed upgrade did not restore wrapper")
        expect(not install_runtime.runtime_status(taskdata).get("errors"), "rollback left a broken managed runtime")
        retained = taskdata / ".nautical-runtime/releases/release-one"
        retained_plan = install_runtime.install_release(
            source=retained, taskdata=taskdata, release_id="release-one", dry_run=True, smoke=False,
        )
        expect(retained_plan.get("status") == "dry-run", f"retained release was not selectable: {retained_plan!r}")
        expect(retained_plan.get("previous_release") == "release-one", f"rollback selected an unexpected release: {retained_plan!r}")

        planned = install_runtime.install_release(
            source=Path(ROOT),
            taskdata=taskdata,
            release_id="release-two",
            dry_run=True,
            smoke=False,
        )
        expect(planned.get("operation") == "upgrade", f"upgrade dry-run was not identified: {planned!r}")
        expect(planned.get("previous_release") == "release-one", f"upgrade plan lost prior release: {planned!r}")
        expect(os.readlink(current) == pointer_before, "upgrade dry-run changed the active release")

        upgraded = install_runtime.install_release(
            source=Path(ROOT),
            taskdata=taskdata,
            release_id="release-two",
            smoke=False,
        )
        expect(upgraded.get("status") == "installed", f"upgrade failed: {upgraded!r}")
        expect(upgraded.get("operation") == "upgrade", f"upgrade action was unclear: {upgraded!r}")
        expect(upgraded.get("previous_release") == "release-one", f"upgrade lost prior release: {upgraded!r}")
        expect(upgraded.get("active_release") == "release-two", f"upgrade lost active release: {upgraded!r}")
        expect(os.readlink(current) == "releases/release-two", "upgrade did not atomically select release two")
        expect((taskdata / ".nautical-runtime/releases/release-one").is_dir(), "previous release was removed")


def test_installer_migrates_legacy_core_and_rolls_back_first_switch():
    """Legacy migration should preserve config and restore the directory on failure."""
    from nautical_core import install_runtime

    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td) / "taskdata"
        legacy = taskdata / "nautical_core"
        legacy.mkdir(parents=True)
        (legacy / "legacy-marker").write_text("keep", encoding="utf-8")
        (legacy / "config-nautical.toml").write_text('tz = "UTC"\n', encoding="utf-8")
        result = install_runtime.install_release(
            source=Path(ROOT),
            taskdata=taskdata,
            release_id="migrated",
            smoke=False,
        )
        backup = Path(str(result.get("legacy_backup") or ""))
        expect(result.get("migrated_legacy_core") is True, f"legacy migration was not reported: {result!r}")
        expect((backup / "legacy-marker").read_text(encoding="utf-8") == "keep", "legacy backup lost data")
        expect((taskdata / "config-nautical.toml").read_text(encoding="utf-8") == 'tz = "UTC"\n', "config was not preserved")

    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td) / "taskdata"
        legacy = taskdata / "nautical_core"
        legacy.mkdir(parents=True)
        (legacy / "legacy-marker").write_text("restore", encoding="utf-8")
        (legacy / "config-nautical.toml").write_text('tz = "UTC"\n', encoding="utf-8")
        try:
            install_runtime.install_release(
                source=Path(ROOT),
                taskdata=taskdata,
                release_id="rollback",
                smoke=False,
                _fail_after="after_pointer",
            )
            raise AssertionError("injected migration failure should have raised")
        except install_runtime.InstallError:
            pass
        expect(legacy.is_dir() and not legacy.is_symlink(), "rollback did not restore legacy core directory")
        expect((legacy / "legacy-marker").read_text(encoding="utf-8") == "restore", "rollback lost legacy data")
        expect(not (taskdata / "config-nautical.toml").exists(), "rollback left a migrated config copy")
        expect(install_runtime.runtime_status(taskdata).get("managed") is False, "failed first install looks active")


def test_installer_lock_and_duplicate_hook_guards():
    """Concurrent installs and pre-existing duplicate Nautical hooks should fail closed."""
    from nautical_core import install_runtime

    with tempfile.TemporaryDirectory() as td:
        lock_path = Path(td) / "install.lock"
        with install_runtime._InstallLock(lock_path):
            try:
                with install_runtime._InstallLock(lock_path):
                    raise AssertionError("second installer acquired an active lock")
            except install_runtime.InstallError as exc:
                expect("already running" in str(exc), f"unexpected lock error: {exc}")

    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td) / "taskdata"
        hooks = taskdata / "hooks"
        hooks.mkdir(parents=True)
        duplicate = hooks / "on-add-custom"
        shutil.copy2(Path(ROOT) / "on-add.nautical", duplicate)
        duplicate.chmod(0o755)
        try:
            install_runtime.install_release(
                source=Path(ROOT),
                taskdata=taskdata,
                release_id="blocked",
                smoke=False,
            )
            raise AssertionError("duplicate active hook should block installation")
        except install_runtime.InstallError as exc:
            expect("duplicate execution" in str(exc), f"unexpected duplicate-hook error: {exc}")


def test_installer_cli_and_doctor_managed_runtime_diagnostics():
    """Installer JSON and Doctor should expose active, abandoned, and broken runtime state."""
    from nautical_core import install_runtime

    install_tool = Path(ROOT) / "nautical_core/tools/nautical_install.py"
    doctor_path = os.path.join(CORE_TOOLS, "nautical_doctor.py")
    doctor = _load_hook_module(doctor_path, "_nautical_doctor_managed_runtime_test")
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "dry-target"
        proc = subprocess.run(
            [
                sys.executable,
                str(install_tool),
                "--source",
                ROOT,
                "--taskdata",
                str(target),
                "--release-id",
                "cli-dry",
                "--dry-run",
                "--json",
            ],
            text=True,
            capture_output=True,
            timeout=20.0,
        )
        expect(proc.returncode == 0, f"installer CLI dry-run failed: {proc.stderr!r}")
        cli_payload = json.loads(proc.stdout)
        expect(cli_payload.get("status") == "dry-run", f"bad installer JSON: {proc.stdout!r}")
        expect(cli_payload.get("operation") == "install", f"installer JSON omitted its plan: {proc.stdout!r}")
        expect(not target.exists(), "installer CLI dry-run mutated its target")

        proc = subprocess.run(
            [
                sys.executable,
                str(install_tool),
                "--source",
                ROOT,
                "--taskdata",
                str(target),
                "--release-id",
                "cli-dry",
                "--dry-run",
            ],
            text=True,
            capture_output=True,
            timeout=20.0,
        )
        expect(proc.returncode == 0, f"installer text dry-run failed: {proc.stderr!r}")
        expect("Plan: Install" in proc.stdout, f"installer text omitted its plan: {proc.stdout!r}")
        expect("Changes: none (dry run)" in proc.stdout, f"installer text obscured dry-run behavior: {proc.stdout!r}")
        expect(not target.exists(), "installer text dry-run mutated its target")

        taskdata = Path(td) / "managed"
        install_runtime.install_release(
            source=Path(ROOT),
            taskdata=taskdata,
            release_id="doctor-active",
            smoke=False,
        )
        findings = []
        doctor._check_managed_runtime(findings, taskdata / "hooks")
        active = next(item for item in findings if item.get("id") == "install.runtime")
        expect(active.get("severity") == "info", f"Doctor did not recognize active runtime: {findings!r}")

        abandoned = taskdata / ".nautical-runtime/.staging-abandoned"
        abandoned.mkdir()
        findings = []
        doctor._check_managed_runtime(findings, taskdata / "hooks")
        expect(
            any(item.get("id") == "install.runtime_abandoned" for item in findings),
            f"Doctor missed abandoned staging: {findings!r}",
        )

        current = taskdata / ".nautical-runtime/current"
        current.unlink()
        current.symlink_to("releases/missing")
        findings = []
        doctor._check_managed_runtime(findings, taskdata / "hooks")
        broken = next(item for item in findings if item.get("id") == "install.runtime")
        expect(broken.get("severity") == "error", f"Doctor missed broken runtime pointer: {findings!r}")


def test_doctor_reports_retired_queue_state_without_migrating_it():
    """Doctor should identify retired queue artifacts and leave them untouched."""
    from nautical_core.tools import nautical_doctor

    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td) / "taskdata"
        state = taskdata / ".nautical-state"
        state.mkdir(parents=True)
        retired = [
            taskdata / ".nautical_spawn_queue.jsonl",
            state / ".nautical_queue.db",
            state / ".nautical_queue.db-wal",
        ]
        for path in retired:
            path.write_text("retired\n", encoding="utf-8")
        findings: list[dict[str, object]] = []
        found = _doctor_obsolete_queue_state(nautical_doctor, findings, taskdata)
        expect(set(found) == {str(path) for path in retired}, f"retired queue paths were not reported: {found!r}")
        issue = next(item for item in findings if item.get("id") == "outbox.obsolete_state")
        expect(issue.get("severity") == "warning", f"retired queue state had the wrong severity: {issue!r}")
        expect("quarantine" in str(issue.get("fix") or "").lower(), f"missing quarantine guidance: {issue!r}")
        expect(all(path.read_text(encoding="utf-8") == "retired\n" for path in retired), "doctor modified retired state")


def test_runtime_cleanup_preserves_active_and_rollback_releases():
    """Runtime cleanup must retain the active release and newest rollback."""
    from nautical_core import install_runtime

    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td) / "taskdata"
        for release_id in ("oldest", "middle", "active"):
            install_runtime.install_release(source=Path(ROOT), taskdata=taskdata, release_id=release_id, smoke=False)
        planned = install_runtime.cleanup_runtime(taskdata, keep_releases=1, apply=False)
        expect(planned.get("active_release") == "active", f"wrong active release: {planned}")
        expect(set(planned.get("kept_releases") or {}) == {"active", "middle"}, f"rollback retention failed: {planned}")
        expect(any(path.endswith("oldest") for path in planned.get("remove_releases") or []), f"old release not planned: {planned}")
        applied = install_runtime.cleanup_runtime(taskdata, keep_releases=1, apply=True)
        expect(applied.get("removed"), f"cleanup did not remove old release: {applied}")
        remaining = {path.name for path in (taskdata / ".nautical-runtime" / "releases").iterdir() if path.is_dir()}
        expect(remaining == {"active", "middle"}, f"cleanup removed a protected release: {remaining}")


def test_retained_release_can_be_selected_with_dry_run_then_applied():
    """Rollback selection must validate the retained tree before switching."""
    from nautical_core import install_runtime

    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td) / "taskdata"
        install_runtime.install_release(source=Path(ROOT), taskdata=taskdata, release_id="release-one", smoke=False)
        install_runtime.install_release(source=Path(ROOT), taskdata=taskdata, release_id="release-two", smoke=False)
        current = taskdata / ".nautical-runtime/current"
        pointer_before = os.readlink(current)
        retained = taskdata / ".nautical-runtime/releases/release-one"
        planned = install_runtime.install_release(
            source=retained, taskdata=taskdata, release_id="release-one", dry_run=True, smoke=False,
        )
        expect(planned.get("status") == "dry-run", f"rollback was not dry-run: {planned!r}")
        expect(planned.get("previous_release") == "release-two", f"wrong rollback source state: {planned!r}")
        expect(os.readlink(current) == pointer_before, "rollback dry-run changed the active release")
        applied = install_runtime.install_release(
            source=retained, taskdata=taskdata, release_id="release-one", smoke=False,
        )
        expect(applied.get("active_release") == "release-one", f"rollback did not select retained release: {applied!r}")
        expect(os.readlink(current) == "releases/release-one", "rollback selected the wrong pointer")
        expect((taskdata / ".nautical-runtime/releases/release-two").is_dir(), "rollback removed newer release")


def test_doctor_discovers_effective_taskdata_directory():
    """doctor should discover the effective data dir when --taskdata is omitted."""
    path = os.path.join(DEV_TOOLS, "nautical_doctor.py")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        config_dir = td_path / "config"
        data_dir = td_path / "taskdata"
        config_dir.mkdir()
        hooks = data_dir / "hooks"
        hooks.mkdir(parents=True)
        _install_doctor_hook_wrappers(hooks)
        config = config_dir / "config-nautical.toml"
        config.write_text('tz = "UTC"\n', encoding="utf-8")
        fake_task = td_path / "task"
        _write_fake_task_for_doctor(fake_task)
        rows = [
            {
                "uuid": "aaaaaaaa-0000-4000-8000-000000000903",
                "status": "completed",
                "chain": "on",
                "cp": "1d",
                "chainID": "cid",
                "link": 1,
                "nextLink": "bbbbbbbb",
            },
            {
                "uuid": "bbbbbbbb-0000-4000-8000-000000000904",
                "status": "pending",
                "chain": "on",
                "cp": "1d",
                "chainID": "cid",
                "link": 2,
                "prevLink": "aaaaaaaa",
            },
        ]
        env = os.environ.copy()
        env["NAUTICAL_CORE_PATH"] = ROOT
        env["NAUTICAL_TRUST_CORE_PATH"] = "1"
        env["FAKE_HOOKS"] = str(hooks)
        env["FAKE_DATA_DIR"] = str(data_dir)
        env["FAKE_EXPORT"] = json.dumps(rows)
        env["NAUTICAL_CONFIG"] = str(config)
        p = subprocess.run(
            [sys.executable, path, "--task-bin", str(fake_task), "--json"],
            text=True,
            capture_output=True,
            env=env,
            timeout=8.0,
        )
        expect(p.returncode == 0, f"doctor returned {p.returncode}: {p.stderr!r} {p.stdout!r}")
        obj = json.loads((p.stdout or "").strip() or "{}")
        expect(obj.get("status") == "ok", f"unexpected doctor status: {obj}")
        expect(obj.get("taskdata") == str(data_dir), f"doctor did not discover the effective taskdata dir: {obj}")


def test_operator_doctor_loads_colocated_queue_helper():
    """installed doctor should load lifecycle outbox status from nautical_core/tools."""
    path = os.path.join(CORE_TOOLS, "nautical_doctor.py")
    with tempfile.TemporaryDirectory() as td:
        p = subprocess.run(
            [sys.executable, path, "--taskdata", td, "--task-bin", "/bin/false", "--json"],
            text=True,
            capture_output=True,
            timeout=8.0,
        )
        expect(p.returncode == 2, f"operator doctor returned {p.returncode}: {p.stderr!r}")
        obj = json.loads((p.stdout or "").strip() or "{}")
        ids = {item.get("id") for item in _doctor_findings(obj)}
        expect("outbox.state" in ids, f"operator doctor did not inspect outbox state: {obj}")
        expect("outbox.unreadable" not in ids, f"operator doctor could not load outbox helper: {obj}")


def test_nautical_dispatches_supported_subcommands():
    """nautical should dispatch supported subcommands to the matching scripts."""
    path = os.path.join(ROOT, "nautical")
    mod = _load_hook_module(path, "_nautical_entrypoint_dispatch_test")
    prev_argv = list(sys.argv)
    prev_run_path = mod.runpy.run_path
    calls = []
    targets = {
        "install": os.path.join(ROOT, "nautical_core", "tools", "nautical_install.py"),
        "doctor": os.path.join(ROOT, "nautical_core", "tools", "nautical_doctor.py"),
        "queue-status": os.path.join(ROOT, "nautical_core", "tools", "nautical_queue_status.py"),
        "reconcile": os.path.join(ROOT, "nautical_core", "tools", "nautical_reconcile.py"),
        "navigator": os.path.join(ROOT, "nautical_navigator.py"),
    }

    def _fake_run_path(target, run_name=None):
        calls.append((target, run_name, list(sys.argv)))
        return {}

    try:
        mod.runpy.run_path = _fake_run_path
        for command, expected_target in targets.items():
            sys.argv = ["nautical", command, "--json"]
            expect(mod.main() == 0, f"nautical returned non-zero for {command}")
        sys.argv = ["nautical", "unknown"]
        expect(mod.main() == 2, "nautical should reject unknown commands")
    finally:
        mod.runpy.run_path = prev_run_path
        sys.argv = prev_argv

    expect(len(calls) == len(targets), f"unexpected dispatch count: {calls!r}")
    for (target, run_name, argv), (command, expected_target) in zip(calls, targets.items()):
        expect(target == expected_target, f"wrong target for {command}: {target!r}")
        expect(run_name == "__main__", f"wrong run_name for {command}: {run_name!r}")
        expect(argv[0] == expected_target, f"argv not rewritten for {command}: {argv!r}")
        if command == "install":
            expect(
                argv[1:3] == ["--source", ROOT],
                f"install did not select the checkout as its source: {argv!r}",
            )

    previous_install_target = mod.COMMANDS["install"]
    previous_source = os.environ.get("NAUTICAL_SOURCE")
    try:
        mod.COMMANDS["install"] = Path("/tmp/nautical-missing-install.py")
        os.environ["NAUTICAL_SOURCE"] = str(ROOT)
        mod.runpy.run_path = _fake_run_path
        calls.clear()
        sys.argv = ["nautical", "install"]
        expect(mod.main() == 0, "missing install target did not recover through checkout")
        expect(
            calls and calls[0][0] == targets["install"],
            f"checkout recovery selected the wrong install target: {calls!r}",
        )
    finally:
        mod.COMMANDS["install"] = previous_install_target
        if previous_source is None:
            os.environ.pop("NAUTICAL_SOURCE", None)
        else:
            os.environ["NAUTICAL_SOURCE"] = previous_source


def test_config_fingerprint_invalidates_persistent_cache_keys():
    """Changing the selected config file must produce a new cache fingerprint and key."""
    previous = os.environ.get("NAUTICAL_CONFIG")
    try:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config-nautical.toml"
            path.write_text('tz = "UTC"\nlive_panel_footer = "ONE"\n', encoding="utf-8")
            os.environ["NAUTICAL_CONFIG"] = str(path)
            first = core.effective_config_snapshot()
            path.write_text('tz = "Europe/Bucharest"\nlive_panel_footer = "TWO"\n', encoding="utf-8")
            second = core.effective_config_snapshot()
            expect(first.get("fingerprint") != second.get("fingerprint"), "config edits did not change fingerprint")

            def key_in_fresh_process(config_text):
                path.write_text(config_text, encoding="utf-8")
                env = os.environ.copy()
                env.update(
                    {
                        "NAUTICAL_CONFIG": str(path),
                        "NAUTICAL_TRUST_CONFIG_PATH": "1",
                        "PYTHONPATH": ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
                    }
                )
                proc = subprocess.run(
                    [sys.executable, "-c", "import nautical_core; print(nautical_core.cache_key_for_task('w:mon', 'skip'))"],
                    text=True,
                    capture_output=True,
                    env=env,
                    timeout=8.0,
                )
                expect(proc.returncode == 0, f"fresh cache-key process failed: {proc.stderr!r}")
                return (proc.stdout or "").strip().splitlines()[-1]

            footer_one_key = key_in_fresh_process('tz = "UTC"\nlive_panel_footer = "ONE"\n')
            footer_two_key = key_in_fresh_process('tz = "UTC"\nlive_panel_footer = "TWO"\n')
            expect(footer_one_key == footer_two_key, "UI-only config edits unnecessarily invalidated cache key")
            tz_one_key = key_in_fresh_process('tz = "UTC"\nlive_panel_footer = "TWO"\n')
            tz_two_key = key_in_fresh_process('tz = "Europe/Bucharest"\nlive_panel_footer = "TWO"\n')
            expect(tz_one_key != tz_two_key, "scheduler config edits did not invalidate cache key")
    finally:
        if previous is None:
            os.environ.pop("NAUTICAL_CONFIG", None)
        else:
            os.environ["NAUTICAL_CONFIG"] = previous




def test_configuration_drift_detects_edit_and_removal():
    """A long-lived core process should report config edits and removal without reloading partially."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "config-nautical.toml"
        path.write_text('tz = "UTC"\n', encoding="utf-8")
        script = (
            "import json, os\n"
            "from pathlib import Path\n"
            "import nautical_core as core\n"
            "p = Path(os.environ['NAUTICAL_CONFIG'])\n"
            "before = core.configuration_drift()\n"
            "p.write_text('tz = \\\"Europe/Bucharest\\\"\\n', encoding='utf-8')\n"
            "edited = core.configuration_drift()\n"
            "p.unlink()\n"
            "removed = core.configuration_drift()\n"
            "print(json.dumps({'before': before, 'edited': edited, 'removed': removed}))\n"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["NAUTICAL_CONFIG"] = str(path)
        env["NAUTICAL_TRUST_CONFIG_PATH"] = "1"
        proc = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, env=env, timeout=10)
        expect(proc.returncode == 0, f"configuration drift probe failed: {proc.stderr}")
        payload = json.loads(proc.stdout)
        expect(payload["before"]["status"] == "ok", f"fresh config reported drift: {payload}")
        expect(payload["edited"]["status"] == "changed", f"edited config drift missing: {payload}")
        expect(payload["removed"]["status"] == "changed", f"removed config drift missing: {payload}")


def test_doctor_reports_actionable_broken_installation():
    """doctor should identify installation, queue, and chain failures with stable IDs."""
    path = os.path.join(DEV_TOOLS, "nautical_doctor.py")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        hooks = td_path / "hooks"
        hooks.mkdir()
        fake_task = td_path / "task"
        _write_fake_task_for_doctor(fake_task)
        (td_path / "config-nautical.toml").write_text("broken = [\n", encoding="utf-8")
        state_dir = td_path / ".nautical-state"
        state_dir.mkdir()
        with sqlite3.connect(str(state_dir / ".nautical_queue.db")) as conn:
            conn.execute(
                """
                CREATE TABLE queue_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    spawn_intent_id TEXT,
                    payload TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'queued',
                    claim_token TEXT,
                    claimed_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO queue_entries (spawn_intent_id, payload, state, created_at, updated_at) "
                "VALUES ('si_doctor', '{}', 'queued', 1.0, 1.0)"
            )
            conn.commit()
        rows = [
            {
                "uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "description": "Missing chain identity",
                "status": "pending",
                "anchor": "w:mon",
                "link": 1,
                "nextLink": "missing1",
            },
            {
                "uuid": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "description": "Duplicate slot first",
                "status": "pending",
                "cp": "1d",
                "chainID": "cid",
                "link": 2,
            },
            {
                "uuid": "cccccccc-cccc-cccc-cccc-cccccccccccc",
                "description": "Duplicate slot second",
                "status": "completed",
                "cp": "1d",
                "chainID": "cid",
                "link": 2,
            },
        ]
        env = os.environ.copy()
        env["NAUTICAL_CORE_PATH"] = ROOT
        env["NAUTICAL_TRUST_CORE_PATH"] = "1"
        env["FAKE_HOOKS"] = str(hooks)
        env["FAKE_EXPORT"] = json.dumps(rows)
        env["FAKE_WRONG_UDA"] = "cp"
        p = subprocess.run(
            [sys.executable, path, "--taskdata", td, "--task-bin", str(fake_task), "--json"],
            text=True,
            capture_output=True,
            env=env,
            timeout=8.0,
        )
        expect(p.returncode == 2, f"expected doctor error exit 2, got {p.returncode}: {p.stderr!r}")
        obj = json.loads((p.stdout or "").strip() or "{}")
        ids = {item.get("id") for item in _doctor_findings(obj)}
        expected = {
            "hook.on-add.missing",
            "hook.on-modify.missing",
            "hook.on-exit.missing",
            "uda.cp.type",
            "config.invalid",
            "outbox.schema",
            "outbox.state",
            "chains.export",
        }
        expect(expected <= ids, f"doctor findings missing {expected - ids}: {obj}")

        text = subprocess.run(
            [sys.executable, path, "--taskdata", td, "--task-bin", str(fake_task)],
            text=True,
            capture_output=True,
            env=env,
            timeout=8.0,
        )
        expect(text.returncode == 2, f"expected text doctor error exit 2, got {text.returncode}")
        report = text.stdout or ""
        expect(
            "Task data could not be exported for chain inspection" in report,
            f"missing fail-closed chain export finding from doctor text: {report!r}",
        )


def test_doctor_reports_chain_repair_plan_findings():
    """doctor should surface safe chain repairs and unresolved repair reasons."""
    path = os.path.join(DEV_TOOLS, "nautical_doctor.py")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        hooks = td_path / "hooks"
        hooks.mkdir()
        _install_doctor_hook_wrappers(hooks)
        (td_path / "config-nautical.toml").write_text('tz = "UTC"\n', encoding="utf-8")
        fake_task = td_path / "task"
        _write_fake_task_for_doctor(fake_task)
        rows = [
            {
                "uuid": "11111111-0000-4000-8000-000000000001",
                "status": "completed",
                "cp": "1d",
                "chain": "on",
                "chainID": "safe0001",
                "link": 1,
            },
            {
                "uuid": "22222222-0000-4000-8000-000000000002",
                "status": "pending",
                "cp": "1d",
                "chain": "on",
                "chainID": "safe0001",
                "link": 2,
                "prevLink": "wrong",
            },
            {
                "uuid": "33333333-0000-4000-8000-000000000003",
                "status": "pending",
                "cp": "1d",
                "chain": "on",
                "chainID": "review01",
                "prevLink": "missing1",
            },
        ]
        env = os.environ.copy()
        env["NAUTICAL_CORE_PATH"] = ROOT
        env["NAUTICAL_TRUST_CORE_PATH"] = "1"
        env["FAKE_HOOKS"] = str(hooks)
        env["FAKE_EXPORT"] = json.dumps(rows)
        p = subprocess.run(
            [sys.executable, path, "--taskdata", td, "--task-bin", str(fake_task), "--json"],
            text=True,
            capture_output=True,
            env=env,
            timeout=8.0,
        )
        expect(p.returncode == 1, f"expected doctor warn exit 1, got {p.returncode}: {p.stderr!r} {p.stdout!r}")
        obj = json.loads((p.stdout or "").strip() or "{}")
        findings = {item.get("id"): item for item in _doctor_findings(obj)}
        expect(any(item_id.startswith("chains.") for item_id in findings), f"missing integrity findings: {obj}")
        review_details = findings.get("chains.repair_review", {}).get("details") or {}
        if review_details:
            expect(review_details.get("reasons"), f"bad review reasons: {review_details}")

        text = subprocess.run(
            [sys.executable, path, "--taskdata", td, "--task-bin", str(fake_task)],
            text=True,
            capture_output=True,
            env=env,
            timeout=8.0,
        )
        report = text.stdout or ""
        expect("Issue:" in report, f"missing integrity review text: {report!r}")
        expect("Reason:" in report, f"missing integrity reason text: {report!r}")




def test_perf_hint_benchmark_isolates_persistent_cache():
    """Hint timing must use a temporary cache and restore production settings."""
    perf = _load_hook_module(
        os.path.join(DEV_TOOLS, "nautical_perf_budget.py"),
        "_nautical_perf_cache_isolation_test",
    )
    original_build = perf.core.build_and_cache_hints
    original_override = getattr(perf.core, "ANCHOR_CACHE_DIR_OVERRIDE", "")
    seen = []
    try:
        def fake_build(*_args, **_kwargs):
            seen.append(str(getattr(perf.core, "ANCHOR_CACHE_DIR_OVERRIDE", "")))
            key = "perf-isolation"
            payload = perf.core.cache_load(key)
            if payload is None:
                payload = {"dnf": []}
                perf.core.cache_save(key, payload)
            return payload

        perf.core.build_and_cache_hints = fake_build
        perf._bench_build_hints(["w:mon"], 1, mode="warm")
        expect(seen and "nautical-perf-cache-" in seen[0], f"benchmark used a non-isolated cache: {seen!r}")
        expect(
            getattr(perf.core, "ANCHOR_CACHE_DIR_OVERRIDE", "") == original_override,
            "benchmark did not restore cache configuration",
        )
    finally:
        perf.core.build_and_cache_hints = original_build


def test_core_import_defers_panel_colour_module():
    """Core import should not load presentation colour helpers before use."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(part for part in (ROOT, env.get("PYTHONPATH", "")) if part)
    probe = (
        "import sys, nautical_core; "
        "assert 'nautical_core.panel_colours' not in sys.modules; "
        "nautical_core.chain_colour_root('chain', 'root'); "
        "assert 'nautical_core.panel_colours' in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env, capture_output=True, text=True)
    expect(result.returncode == 0, f"panel colour helper was eager or failed lazily: {result.stderr!r}")


def test_core_import_defers_diagnostic_model():
    """Core import should not load diagnostic models before diagnostics are used."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(part for part in (ROOT, env.get("PYTHONPATH", "")) if part)
    probe = (
        "import sys, nautical_core; "
        "assert 'nautical_core.diagnostic_models' not in sys.modules; "
        "assert nautical_core.DiagnosticEvent.__name__ == 'DiagnosticEvent'; "
        "assert 'nautical_core.diagnostic_models' in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env, capture_output=True, text=True)
    expect(result.returncode == 0, f"diagnostic model was eager or failed lazily: {result.stderr!r}")


def test_core_import_defers_parser_scheduler_models():
    """Core import should defer parser and scheduler model modules until their names are used."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(part for part in (ROOT, env.get("PYTHONPATH", "")) if part)
    probe = (
        "import sys, nautical_core; "
        "assert 'nautical_core.parsing.parser_models' not in sys.modules; "
        "assert 'nautical_core.scheduler_models' not in sys.modules; "
        "assert nautical_core.ParseError.__name__ == 'ParseError'; "
        "assert 'nautical_core.parsing.parser_models' in sys.modules; "
        "assert 'nautical_core.scheduler_models' in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env, capture_output=True, text=True)
    expect(result.returncode == 0, f"parser/scheduler models were eager or failed lazily: {result.stderr!r}")


def test_deploy_sanity_enforces_removed_lifecycle_ownership():
    """Deployment checks must reject reintroduced exit modules or reconcile seams."""
    import shutil
    import tempfile

    path = Path(ROOT) / "dev_tools" / "nautical_deploy_sanity.py"
    module = _load_hook_module(str(path), "_nautical_removed_ownership_deploy_test")
    results = module._check_removed_ownership(Path(ROOT))
    failures = [item for item in results if not item.get("ok")]
    expect(not failures, f"removed lifecycle ownership checks failed: {failures!r}")

    with tempfile.TemporaryDirectory() as td:
        staged = Path(td)
        (staged / "nautical_core" / "tools").mkdir(parents=True)
        shutil.copy2(Path(ROOT) / "nautical_core" / "runtime_manifest.py", staged / "nautical_core" / "runtime_manifest.py")
        (staged / "nautical_core" / "exit_models.py").write_text("# stale module\n", encoding="utf-8")
        (staged / "nautical_core" / "tools" / "nautical_reconcile.py").write_text(
            "_validate_hook_protocol = object()\n", encoding="utf-8"
        )
        failures = [item for item in module._check_removed_ownership(staged) if not item.get("ok")]
        expect(len(failures) >= 2, f"reintroduced ownership paths were not rejected: {failures!r}")

    with tempfile.TemporaryDirectory() as td:
        staged = Path(td)
        (staged / "nautical_core" / "tools").mkdir(parents=True)
        shutil.copy2(Path(ROOT) / "nautical_core" / "runtime_manifest.py", staged / "nautical_core" / "runtime_manifest.py")
        (staged / "nautical_core" / "tools" / "nautical_reconcile.py").write_text(
            "from nautical_core.hooks import modify_impl\n", encoding="utf-8"
        )
        failures = [item for item in module._check_removed_ownership(staged) if not item.get("ok")]
        expect(
            any(item.get("name") == "operator-hook-imports:nautical_core/tools/nautical_reconcile.py" for item in failures),
            f"operator hook import was not rejected: {failures!r}",
        )

    results = module._check_removed_ownership(Path(ROOT))
    expect(
        any(item.get("name") == "pure-integrity:nautical_core/chain_graph.py" for item in results),
        f"pure integrity import checks were not reported: {results!r}",
    )


def test_perf_hook_fast_path_ratio_enforcement():
    """Hook latency checks should enforce the normalized fast/full median ratio."""
    perf = _load_hook_module(
        os.path.join(DEV_TOOLS, "nautical_perf_budget.py"),
        "_nautical_hook_perf_ratio_test",
    )

    def clearly_faster(_hook_path, *, input_text, env, expected_task):
        _ = (input_text, expected_task)
        return 0.100 if env.get("NAUTICAL_BENCH_FORCE_FULL") == "1" else 0.050

    perf._run_hook_timed = clearly_faster
    passing = perf._measure_hook_fast_path(
        "hook_test",
        Path("unused-hook"),
        input_text="{}",
        expected_task={},
        base_env={},
        repeats=3,
        max_ratio=0.8,
    )
    expect(passing.get("pass") is True, f"clear fast-path improvement should pass: {passing}")
    expect(abs(float(passing.get("fast_to_full_ratio")) - 0.5) < 0.001, f"unexpected ratio: {passing}")

    def insufficient_gain(_hook_path, *, input_text, env, expected_task):
        _ = (input_text, expected_task)
        return 0.100 if env.get("NAUTICAL_BENCH_FORCE_FULL") == "1" else 0.090

    perf._run_hook_timed = insufficient_gain
    failing = perf._measure_hook_fast_path(
        "hook_test",
        Path("unused-hook"),
        input_text="{}",
        expected_task={},
        base_env={},
        repeats=3,
        max_ratio=0.8,
    )
    expect(failing.get("pass") is False, f"insufficient fast-path improvement should fail: {failing}")

    perf._run_hook_timed = lambda *_args, **_kwargs: 0.060
    managed = perf._measure_managed_hook_latency(
        "managed_hook_test",
        Path("unused-hook"),
        input_text="{}",
        expected_task={},
        base_env={"NAUTICAL_CORE_PATH": "/source", "NAUTICAL_TRUST_CORE_PATH": "1"},
        repeats=3,
        baseline_median_s=0.050,
        max_ratio=1.5,
    )
    expect(managed.get("pass") is True, f"reasonable managed-layout overhead should pass: {managed}")
    expect(
        abs(float(managed.get("managed_to_source_ratio")) - 1.2) < 0.001,
        f"unexpected managed/source ratio: {managed}",
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


def test_deploy_sanity_rejects_missing_lazy_lifecycle_module():
    """Deployment sanity must fail when a declared lazy module is absent."""
    path = os.path.join(DEV_TOOLS, "nautical_deploy_sanity.py")
    with tempfile.TemporaryDirectory() as td:
        candidate = Path(td) / "candidate"
        shutil.copytree(
            ROOT,
            candidate,
            ignore=shutil.ignore_patterns(".git", "__pycache__", ".nautical-cache", ".nautical_cache"),
        )
        missing = candidate / "nautical_core" / "modify_completion_compute.py"
        missing.unlink()
        proc = subprocess.run(
            [sys.executable, path, "--root", str(candidate), "--no-require-exec", "--json"],
            text=True,
            capture_output=True,
            timeout=20.0,
        )
        expect(proc.returncode != 0, "deploy sanity accepted a release missing a lazy module")
        payload = json.loads((proc.stdout or "{}").strip() or "{}")
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        expect(
            any(
                item.get("path") == "nautical_core/modify_completion_compute.py" and not item.get("ok")
                for item in results
                if isinstance(item, dict)
            ),
            f"missing lazy module was not reported by required-file checks: {results}",
        )
        expect(
            any(
                item.get("kind") == "lazy-modules"
                and item.get("name") == "on-modify"
                and not item.get("ok")
                for item in results
                if isinstance(item, dict)
            ),
            f"modify lazy import smoke did not fail: {results}",
        )


def test_deploy_sanity_rejects_missing_operator_runtime_tool():
    """Deployment sanity must cover every command dispatched by nautical."""
    path = os.path.join(DEV_TOOLS, "nautical_deploy_sanity.py")
    with tempfile.TemporaryDirectory() as td:
        candidate = Path(td) / "candidate"
        shutil.copytree(
            ROOT,
            candidate,
            ignore=shutil.ignore_patterns(".git", "__pycache__", ".nautical-cache", ".nautical_cache"),
        )
        missing = candidate / "nautical_core" / "tools" / "nautical_doctor.py"
        missing.unlink()
        proc = subprocess.run(
            [sys.executable, path, "--root", str(candidate), "--no-require-exec", "--json"],
            text=True,
            capture_output=True,
            timeout=20.0,
        )
        expect(proc.returncode != 0, "deploy sanity accepted a release missing an operator tool")
        payload = json.loads((proc.stdout or "{}").strip() or "{}")
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        expect(
            any(
                item.get("path") == "nautical_core/tools/nautical_doctor.py" and not item.get("ok")
                for item in results
                if isinstance(item, dict)
            ),
            f"missing operator tool was not reported: {results}",
        )


def test_deploy_sanity_rejects_unowned_taskwarrior_subprocess():
    """Deployment checks keep Taskwarrior process ownership in one client."""
    module = _load_hook_module(
        os.path.join(DEV_TOOLS, "nautical_deploy_sanity.py"),
        "_nautical_deploy_process_ownership_test",
    )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        core_dir = root / "nautical_core"
        core_dir.mkdir()
        (core_dir / "bad_runner.py").write_text(
            "import subprocess\nsubprocess.run(['task', 'export'])\n",
            encoding="utf-8",
        )
        result = module._check_taskwarrior_process_ownership(root)
        expect(result and not result[0]["ok"], f"unowned subprocess was accepted: {result}")
        expect("bad_runner.py:2" in result[0]["message"], f"violation location was lost: {result}")


def test_hook_replay_harness_reports_ok():
    """Replay harness should pass the seeded hook corpus."""
    path = os.path.join(DEV_TOOLS, "nautical_hook_replay.py")
    corpus = os.path.join(DEV_TOOLS, "nautical_hook_replay_corpus.jsonl")
    p = subprocess.run(
        [sys.executable, path, "--json", "--corpus", corpus],
        text=True,
        capture_output=True,
        timeout=12.0,
    )
    expect(p.returncode == 0, f"replay harness returned {p.returncode}: stderr={p.stderr!r}")
    obj = json.loads((p.stdout or "").strip() or "{}")
    expect(obj.get("status") == "ok", f"unexpected replay harness status: {obj}")
    results = obj.get("results") if isinstance(obj.get("results"), list) else []
    expect(results, "replay harness should report per-case results")
    expect(all(bool(r.get("ok")) for r in results if isinstance(r, dict)), f"failing replay result: {results}")


def test_mixed_recurrence_loop_harness_reports_ok():
    """Mixed recurrence loop harness should complete a small deterministic cycle run."""
    path = os.path.join(DEV_TOOLS, "nautical_mixed_recurrence_loop.py")
    p = subprocess.run(
        [sys.executable, path, "--cycles", "3", "--json"],
        text=True,
        capture_output=True,
        timeout=30.0,
    )
    expect(p.returncode == 0, f"mixed recurrence loop returned {p.returncode}: stderr={p.stderr!r}")
    obj = json.loads((p.stdout or "").strip() or "{}")
    expect(obj.get("ok") is True, f"unexpected mixed loop status: {obj}")
    expect(int(obj.get("cycles_completed") or 0) >= 1, f"expected loop progress: {obj}")
    expect(not obj.get("violations"), f"mixed loop reported violations: {obj}")


def test_soak_runner_reports_ok():
    """Short soak runner should complete without violations."""
    path = os.path.join(DEV_TOOLS, "nautical_soak_test.py")
    p = subprocess.run(
        [
            sys.executable,
            path,
            "--seconds",
            "2",
            "--batch-size",
            "4",
            "--anchor-rate",
            "0.5",
            "--cp-rate",
            "0.5",
            "--done-rate",
            "0.5",
            "--progress-every-seconds",
            "0",
            "--json",
            "--enforce",
        ],
        text=True,
        capture_output=True,
        timeout=240,
    )
    expect(p.returncode == 0, f"soak runner returned {p.returncode}: stderr={p.stderr!r}")
    obj = json.loads((p.stdout or "").strip() or "{}")
    expect(obj.get("ok") is True, f"unexpected soak status: {obj}")
    expect(not obj.get("violations"), f"soak runner reported violations: {obj}")


def test_ops_templates_present_and_runner_executable():
    """ops templates should exist and runner script should be executable."""
    ops = os.path.join(DEV_TOOLS, "ops")
    files = [
        "README.md",
        "nautical-health-check.crontab",
        "nautical-health-check.service",
        "nautical-health-check.timer",
        "nautical_health_check_cron.sh",
    ]
    for name in files:
        p = os.path.join(ops, name)
        expect(os.path.isfile(p), f"missing ops template: {p}")
    runner = os.path.join(ops, "nautical_health_check_cron.sh")
    expect(os.access(runner, os.X_OK), f"runner should be executable: {runner}")

def test_tw_export_chain_extra_validation():
    """Chain snapshot filters should reject shell-like extra arguments."""
    from nautical_core.hook_support import parse_extra_tokens

    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_chain_export_extra_test")
    effects = mod._module("modify_read_effects")
    port = effects.ExtraTokenPort(parse_extra_tokens)
    expect(effects.parse_extra_tokens(port, "status:pending; rm -rf /") is None, "unsafe filter was accepted")


def test_tw_export_chain_extra_rejects_dash_prefixed_tokens():
    """tw_export_chain extra parser should reject dash-prefixed tokens."""
    from nautical_core.hook_support import parse_extra_tokens

    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_chain_export_extra_dash_test")
    effects = mod._module("modify_read_effects")
    port = effects.ExtraTokenPort(parse_extra_tokens)
    expect(effects.parse_extra_tokens(port, "status:pending -rc.hooks=on") is None, "dash-prefixed token was accepted")


def test_on_modify_diag_blocks_pretty_print():
    """on-modify diag output should emit indented multi-line blocks."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_diag_pretty_test")

    previous = os.environ.get("NAUTICAL_DIAG")
    try:
        os.environ["NAUTICAL_DIAG"] = "1"
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            mod._emit_diag_block("diag stats", [("a", 1), ("b", 2), ("c", 3), ("d", 4)], columns=2)
    finally:
        if previous is None:
            os.environ.pop("NAUTICAL_DIAG", None)
        else:
            os.environ["NAUTICAL_DIAG"] = previous

    out = buf.getvalue()
    expect("[nautical] diag stats:\n" in out, f"missing diag title: {out!r}")
    expect("[nautical]   a=1  b=2\n" in out, f"missing first wrapped diag line: {out!r}")
    expect("[nautical]   c=3  d=4\n" in out, f"missing second wrapped diag line: {out!r}")


def test_on_modify_lifecycle_diagnostics_are_gated_to_stderr():
    """Structured lifecycle diagnostics must never leak into hook stdout."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_lifecycle_diag_channel_test")
    models = core._import_sibling("modify_models")
    result = models.CompletionLifecycleResult(
        state="retryable",
        reason="Taskwarrior lock busy",
        diagnostic=models.CompletionLifecycleDiagnostic(
            transition_id="chain01:1->2",
            chain_id="chain01",
            parent_link=1,
            child_link=2,
            stage="spawn",
            attempts=1,
            failure_kind="command_error",
        ),
    )
    previous = os.environ.get("NAUTICAL_DIAG")
    try:
        os.environ["NAUTICAL_DIAG"] = "1"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            mod._diag_lifecycle_result(result)
        text = stderr.getvalue()
        expect(stdout.getvalue() == "", f"lifecycle diagnostics leaked to stdout: {stdout.getvalue()!r}")
        expect("completion lifecycle" in text and "failure_kind=command_error" in text, f"structured lifecycle diagnostics missing: {text!r}")

        os.environ.pop("NAUTICAL_DIAG", None)
        silent = io.StringIO()
        with contextlib.redirect_stderr(silent):
            mod._diag_lifecycle_result(result)
        expect(silent.getvalue() == "", f"lifecycle diagnostics ignored NAUTICAL_DIAG gate: {silent.getvalue()!r}")
    finally:
        if previous is None:
            os.environ.pop("NAUTICAL_DIAG", None)
        else:
            os.environ["NAUTICAL_DIAG"] = previous


def test_on_modify_run_task_diag_bucket_stats():
    """on-modify should classify Taskwarrior calls into stable diagnostic buckets."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_diag_bucket_test")

    mod._reset_modify_runtime_state()
    expect(mod._run_task_diag_bucket(["task", "rc.hooks=off", "rc.verbose=nothing", "_get", "beeswax.entry"]) == "get", "_get should classify as get")
    expect(mod._run_task_diag_bucket(["task", "rc.hooks=off", "uuid:beeswax", "export"]) == "other", "repository-owned UUID reads should not have a hook bucket")
    expect(mod._run_task_diag_bucket(["task", "rc.hooks=off", "rc.json.array=on", "chainID:cid", "export"]) == "export_chain", "chain export should classify correctly")
    expect(mod._run_task_diag_bucket(["task", "rc.hooks=off", "import", "-"]) == "import", "import should classify correctly")

    mod._diag_record_run_task(["task", "_get", "beeswax.entry"], ok=True, elapsed=0.25)
    mod._diag_record_run_task(["task", "rc.json.array=off", "uuid:beeswax", "export"], ok=False, elapsed=0.5)
    mod._diag_record_run_task(["task", "rc.json.array=on", "chainID:cid", "export"], ok=True, elapsed=0.75)

    stats = mod._modify_runtime_state().diag_stats
    expect(stats.get("run_task_calls_get") == 1, f"unexpected get call stats: {stats}")
    expect(stats.get("run_task_calls_export_chain") == 1, f"unexpected chain export call stats: {stats}")
    expect(stats.get("run_task_failures_other") == 1, f"unexpected repository-owned read fallback stats: {stats}")
    expect(abs(float(stats.get("run_task_seconds_get", 0.0)) - 0.25) < 1e-9, f"unexpected get seconds: {stats}")
    expect(abs(float(stats.get("run_task_seconds_other", 0.0)) - 0.5) < 1e-9, f"unexpected fallback command seconds: {stats}")
    expect(abs(float(stats.get("run_task_seconds_export_chain", 0.0)) - 0.75) < 1e-9, f"unexpected chain export seconds: {stats}")


def test_on_exit_diag_blocks_pretty_print():
    """on-exit diag output should emit indented multi-line blocks."""
    hook = _find_hook_file("on-exit.nautical")
    mod = _load_hook_module(hook, "_nautical_on_exit_diag_pretty_test")

    prev = os.environ.get("NAUTICAL_DIAG")
    os.environ["NAUTICAL_DIAG"] = "1"
    mod.core = None
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            mod._diag_block("on-exit task stats", [("a", 1), ("b", 2), ("c", 3), ("d", 4)], columns=2)
    finally:
        if prev is None:
            os.environ.pop("NAUTICAL_DIAG", None)
        else:
            os.environ["NAUTICAL_DIAG"] = prev

    out = buf.getvalue()
    expect("[nautical] on-exit task stats:\n" in out, f"missing exit diag title: {out!r}")
    expect("[nautical]   a=1  b=2\n" in out, f"missing first wrapped exit diag line: {out!r}")
    expect("[nautical]   c=3  d=4\n" in out, f"missing second wrapped exit diag line: {out!r}")


def test_on_exit_outcome_diagnostics_are_bounded():
    """Large drains summarize excess intent diagnostics instead of flooding stderr."""
    hook = _find_hook_file("on-exit.nautical")
    mod = _load_hook_module(hook, "_nautical_on_exit_diag_bound_test")
    from types import SimpleNamespace

    messages = []
    previous_limit = mod._OUTBOX_DIAG_MAX_ITEMS
    previous_diag = mod._diag
    mod._OUTBOX_DIAG_MAX_ITEMS = 2
    mod._diag = messages.append
    outcomes = [
        SimpleNamespace(intent_id=f"intent-{index}", kind=SimpleNamespace(value="retryable"), reason="busy")
        for index in range(5)
    ]
    try:
        diagnostics = importlib.import_module("nautical_core.exit_diagnostics")
        suppressed = diagnostics.emit_outcome_diagnostics(
            outcomes,
            diagnostic=messages.append,
            limit=mod._OUTBOX_DIAG_MAX_ITEMS,
        )
    finally:
        mod._OUTBOX_DIAG_MAX_ITEMS = previous_limit
        mod._diag = previous_diag
    expect(suppressed == 3, f"unexpected suppressed diagnostic count: {suppressed}")
    expect(len(messages) == 3, f"bounded diagnostics emitted unexpected lines: {messages!r}")
    expect("intent-0" in messages[0] and "intent-1" in messages[1], f"first diagnostics were lost: {messages!r}")
    expect("suppressed 3 additional" in messages[2], f"suppression summary was not actionable: {messages!r}")


def test_on_modify_chain_cache_thread_safety_smoke():
    """Concurrent chain cache set/read paths should not crash or return invalid shapes."""
    import threading

    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_chain_cache_thread_safety_test")

    full_uuid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    mod._module("modify_composition").lifecycle_read_service_for(mod).replace_chain_cache(
        "cid-a",
        [{"uuid": full_uuid, "link": 1, "entry": "2026-01-01T00:00:00Z"}],
    )
    mod._task = lambda *_args, **_kwargs: "[]"

    errs = []
    hits = {"short": 0}

    def _writer(chain_id: str):
        try:
            for i in range(300):
                mod._module("modify_composition").lifecycle_read_service_for(mod).replace_chain_cache(
                    chain_id,
                    [{"uuid": full_uuid, "link": 1, "entry": f"2026-01-01T00:00:{i % 60:02d}Z"}],
                )
        except Exception as e:
            errs.append(f"writer:{e}")

    def _reader():
        try:
            for _ in range(600):
                s, _chain_id = mod._module("modify_composition").lifecycle_read_service_for(mod).lookup_short("aaaaaaaa")
                if s is not None:
                    from nautical_core.task_models import TaskObservation
                    expect(isinstance(s, TaskObservation), f"short cache read should return observation, got {type(s)}")
                    hits["short"] += 1
        except Exception as e:
            errs.append(f"reader:{e}")

    threads = [
        threading.Thread(target=_writer, args=("cid-a",)),
        threading.Thread(target=_writer, args=("cid-b",)),
        threading.Thread(target=_reader),
        threading.Thread(target=_reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expect(not errs, f"concurrent chain cache access raised errors: {errs}")
    expect(hits["short"] > 0, f"expected cache hits, got {hits}")

def test_on_modify_get_chain_export_filters_cached_chain_in_memory():
    """Filtered chain reads should use the in-memory chain cache before falling back to Taskwarrior export."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_get_chain_export_cached_filter_test")
    mod._reset_modify_runtime_state()

    mod._module("modify_composition").lifecycle_read_service_for(mod).replace_chain_cache(
        "cid-1",
        [
            {"uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "link": 1, "status": "completed", "entry": "2026-01-01T00:00:00Z"},
            {"uuid": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "link": 2, "status": "pending", "entry": "2026-01-02T00:00:00Z"},
            {"uuid": "cccccccc-cccc-cccc-cccc-cccccccccccc", "link": 2, "status": "deleted", "entry": "2026-01-03T00:00:00Z"},
        ],
    )

    rows = mod._module("modify_composition").lifecycle_read_service_for(mod).get_chain_export(
        "cid-1", extra="link:2 status.not:deleted"
    )
    expect(len(rows) == 1, f"expected exactly one filtered cached row, got {rows}")
    expect(rows[0].get("uuid") == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", f"unexpected filtered row: {rows}")


def test_on_modify_chain_cache_reads_through_typed_repository():
    """Modify chain caches must filter one authoritative repository snapshot in memory."""
    from nautical_core.integration_models import Found

    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_modify_repository_chain_cache_test")
    mod._reset_modify_runtime_state()
    rows = (
        {"uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "chainID": "cid", "link": 1, "status": "completed", "modified": "20250101T090000Z"},
        {"uuid": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "chainID": "cid", "link": 2, "status": "pending", "modified": "20250102T090000Z"},
    )
    calls = []

    class Repository:
        def chain_snapshot(self, chain_id, **_kwargs):
            calls.append(chain_id)
            return Found(_task_observations(rows), "chain:cid")

    mod._modify_runtime_state().task_repository = Repository()
    selected = mod._module("modify_composition").lifecycle_read_service_for(mod).get_chain_export("cid", extra="status:pending")
    expect(calls == ["cid"], f"expected one repository read, got {calls!r}")
    expect([row.get("link") for row in selected] == [2], f"repository rows were not filtered: {selected!r}")


def test_on_modify_chain_cache_preserves_repository_unavailability():
    """An unavailable repository chain read must never become an empty chain."""
    from nautical_core.integration_models import CommandFailureKind, FailureEvidence, TaskCommand, Unavailable

    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_modify_repository_chain_failure_test")
    command = TaskCommand(("task", "export"), "test chain read", 1.0)
    evidence = FailureEvidence(command, CommandFailureKind.INVALID_RESPONSE, 0, 1, 0.0, False, "malformed JSON")

    class Repository:
        def chain_snapshot(self, _chain_id, **_kwargs):
            return Unavailable("chain:cid", evidence)

    mod._modify_runtime_state().task_repository = Repository()
    try:
        mod._module("modify_composition").lifecycle_read_service_for(mod).get_chain_export("cid")
    except RuntimeError as exc:
        expect("malformed JSON" in str(exc), f"unavailable detail was lost: {exc}")
    else:
        raise AssertionError("unavailable repository read became an empty chain")


def test_on_modify_predecessor_read_preserves_repository_unavailability():
    """Predecessor presentation must not turn an unavailable chain into no predecessors."""
    from nautical_core.integration_models import CommandFailureKind, FailureEvidence, TaskCommand, Unavailable

    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_modify_predecessor_chain_failure_test")
    command = TaskCommand(("task", "export"), "test predecessor read", 1.0)
    evidence = FailureEvidence(command, CommandFailureKind.INVALID_RESPONSE, 0, 1, 0.0, False, "malformed JSON")

    class Repository:
        def chain_snapshot(self, _chain_id, **_kwargs):
            return Unavailable("chain:cid", evidence)

    mod._modify_runtime_state().task_repository = Repository()
    try:
        reads = mod._module("modify_read_effects")
        state = mod._modify_runtime_state()
        ports = reads.PreviousChainPorts(
            service=mod._module("modify_composition").lifecycle_read_service_for(mod),
            panel_chain_by_link=state.panel_chain_by_link,
            panel_chain_snapshot_loaded=state.panel_chain_snapshot_loaded,
        )
        reads.collect_prev_two(ports, {"chainID": "cid", "link": 3})
    except RuntimeError as exc:
        expect("malformed JSON" in str(exc), f"predecessor failure detail was lost: {exc}")
    else:
        raise AssertionError("unavailable predecessor read became an empty list")


def test_core_invalid_timezone_warns_and_falls_back_to_utc():
    """Invalid timezone config should fall back to UTC and emit diagnostic warning when enabled."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "nautical.toml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write('tz = "Invalid/Timezone"\n')

        prev_diag = os.environ.get("NAUTICAL_DIAG")
        prev_xdg = os.environ.get("XDG_CACHE_HOME")
        os.environ["NAUTICAL_DIAG"] = "1"
        os.environ["XDG_CACHE_HOME"] = td
        try:
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                mod = _load_core_module(core_path, "_nautical_core_bad_tz_fallback_test", cfg)
            expect(getattr(mod, "_LOCAL_TZ", None) is None, "invalid timezone should use UTC fallback")
            expect(
                "invalid or unavailable" in mod.scheduling_configuration_error(),
                "invalid timezone should block Nautical scheduling",
            )
            stderr_text = buf.getvalue().lower()
            expect("utc fallback" in stderr_text, f"expected timezone fallback warning in stderr, got: {stderr_text!r}")
        finally:
            if prev_diag is None:
                os.environ.pop("NAUTICAL_DIAG", None)
            else:
                os.environ["NAUTICAL_DIAG"] = prev_diag
            if prev_xdg is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = prev_xdg


def test_explicit_unsafe_config_blocks_scheduling_with_actionable_error():
    """An explicit world-writable config must not silently fall back to UTC."""
    script = (
        "import nautical_core\n"
        "print(nautical_core.scheduling_configuration_error())\n"
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "nautical.toml"
        path.write_text('tz = "Pacific/Auckland"\n', encoding="utf-8")
        try:
            path.chmod(0o666)
        except OSError:
            return
        env = os.environ.copy()
        env.update({"NAUTICAL_CONFIG": str(path), "PYTHONPATH": str(ROOT)})
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT),
        )
        expect(proc.returncode == 0, f"unsafe config verification process failed: {proc.stderr[:500]!r}")
        expect(str(path) in proc.stdout, f"rejected config path missing: {proc.stdout!r}")
        expect("world-writable" in proc.stdout, f"rejected config reason missing: {proc.stdout!r}")


def test_taskdata_config_reload_fails_closed_for_malformed_toml_and_timezone():
    """The shared Taskdata reload must reject malformed files and invalid timezones."""
    script = (
        "import sys\n"
        "import nautical_core\n"
        "from nautical_core.integration_context import IntegrationRuntime, build_operator_context\n"
        "try:\n"
        "    build_operator_context(runtime=IntegrationRuntime.from_compatibility_facade(nautical_core), task_binary=sys.executable, taskdata=sys.argv[1])\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__ + ': ' + str(exc))\n"
        "else:\n"
        "    raise SystemExit('reload unexpectedly succeeded')\n"
    )
    env = os.environ.copy()
    env.pop("NAUTICAL_CONFIG", None)
    env.pop("TASKDATA", None)
    env["PYTHONPATH"] = str(ROOT)
    cases = (("tz = [\n", "config parse failed"), ("tz = \"Invalid/Timezone\"\n", "invalid or unavailable"))
    for contents, expected in cases:
        with tempfile.TemporaryDirectory() as td:
            Path(td, "config-nautical.toml").write_text(contents, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-c", script, td],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(ROOT),
            )
            expect(proc.returncode == 0, f"Taskdata config reload process failed: {proc.stderr[:500]!r}")
            expect(expected in proc.stdout, f"reload error was not actionable: {proc.stdout!r}")


def test_core_recurrence_update_udas_config_aliases():
    """recurrence UDA carry config should accept top-level and [recurrence] alias forms."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))

    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "nautical.toml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write('recurrence_update_udas = ["rappel", "next_review"]\n')
            f.write("[recurrence]\n")
            f.write('update_udas = "ignored_alias"\n')
        mod = _load_core_module(core_path, "_nautical_core_recur_udas_top_test", cfg)
        expect(
            mod.RECURRENCE_UPDATE_UDAS == ("rappel", "next_review"),
            f"unexpected top-level recurrence_update_udas: {mod.RECURRENCE_UPDATE_UDAS}",
        )

    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "nautical.toml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write("[recurrence]\n")
            f.write('update_udas = "rappel, next_review, bad-name, 9x"\n')
        mod = _load_core_module(core_path, "_nautical_core_recur_udas_alias_test", cfg)
        expect(
            mod.RECURRENCE_UPDATE_UDAS == ("rappel", "next_review"),
            f"unexpected alias recurrence.update_udas parse: {mod.RECURRENCE_UPDATE_UDAS}",
        )


def test_core_live_panel_duration_config_defaults_and_clamps():
    """Live panel duration should default to 160 ms and stay within its safe range."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    cases = [
        ("", 160),
        ("live_panel_duration_ms = -20\n", 0),
        ("live_panel_duration_ms = 275\n", 275),
        ("live_panel_duration_ms = 5000\n", 1000),
        ('live_panel_duration_ms = "bad"\n', 160),
    ]
    for index, (config_text, expected) in enumerate(cases):
        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, "nautical.toml")
            with open(cfg, "w", encoding="utf-8") as f:
                f.write(config_text)
            mod = _load_core_module(core_path, f"_nautical_core_live_duration_{index}", cfg)
            expect(
                mod.LIVE_PANEL_DURATION_MS == expected,
                f"unexpected live duration for {config_text!r}: {mod.LIVE_PANEL_DURATION_MS!r}",
            )


def test_core_live_panel_footer_config_defaults_and_customizes():
    """Live panel footer should default to Nautical and accept custom text."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    for index, (config_text, expected) in enumerate((("", "NAUTICAL"), ('live_panel_footer = "STATUS"\n', "STATUS"))):
        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, "nautical.toml")
            Path(cfg).write_text(config_text, encoding="utf-8")
            mod = _load_core_module(core_path, f"_nautical_core_live_footer_{index}", cfg)
            expect(mod.LIVE_PANEL_FOOTER == expected, f"unexpected live footer: {mod.LIVE_PANEL_FOOTER!r}")


def test_core_uda_aliases_config_defaults_disabled_and_can_enable():
    """Description-based UDA aliases should be opt-in through config."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    cases = [("", False), ("enable_uda_aliases = true\n", True), ("enable_uda_aliases = false\n", False)]
    for index, (config_text, expected) in enumerate(cases):
        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, "nautical.toml")
            with open(cfg, "w", encoding="utf-8") as f:
                f.write(config_text)
            mod = _load_core_module(core_path, f"_nautical_core_uda_aliases_{index}", cfg)
            expect(
                mod.ENABLE_UDA_ALIASES is expected,
                f"unexpected UDA alias setting for {config_text!r}: {mod.ENABLE_UDA_ALIASES!r}",
            )


def test_on_add_expands_enabled_description_uda_aliases():
    """on-add should expand enabled aliases before recurrence classification."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_description_aliases_test")
    previous = mod.core.ENABLE_UDA_ALIASES
    try:
        mod.core.ENABLE_UDA_ALIASES = True
        task = {"description": "test task a:w:mon am:all"}
        mod._apply_description_uda_aliases(task)
    finally:
        mod.core.ENABLE_UDA_ALIASES = previous
    expect(task == {"description": "test task", "anchor": "w:mon", "anchor_mode": "all"}, f"on-add alias expansion failed: {task!r}")


def test_hook_on_add_uda_aliases_emit_canonical_json_and_reject_conflicts():
    """The real on-add boundary should normalize aliases without contaminating stdout."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "nautical.toml"
        config.write_text("enable_uda_aliases = true\ntz = \"UTC\"\n", encoding="utf-8")
        task = {
            "uuid": "00000000-0000-4000-8000-000000000114",
            "description": "hook alias test a:w:mon am:all",
            "status": "pending",
            "project": "testing",
            "entry": "20260803T000000Z",
            "due": "20260810T090000Z",
        }
        env = {"NAUTICAL_CONFIG": str(config), "NAUTICAL_TRUST_CONFIG_PATH": "1", "TASKDATA": td, "NO_COLOR": "1"}
        proc = _run_hook_script(hook, task, env_extra=env)
        expect(proc.returncode == 0, f"enabled alias hook failed: {proc.stderr[:600]!r}")
        _assert_stdout_json_only(proc.stdout)
        normalized = _extract_last_json(proc.stdout)
        expect(normalized.get("description") == "hook alias test", f"alias text remained in description: {normalized!r}; stderr={proc.stderr[:500]!r}")
        expect(normalized.get("anchor") == "w:mon" and normalized.get("anchor_mode") == "all", f"canonical aliases missing: {normalized!r}")

        conflict = dict(task, description="hook alias conflict a:w:tue", anchor="w:mon", anchor_mode="skip")
        rejected = _run_hook_script(hook, conflict, env_extra=env)
        expect(rejected.returncode != 0, "conflicting canonical and alias values were accepted")
        expect("different value" in (rejected.stderr or ""), f"conflict error was not actionable: {rejected.stderr[:600]!r}")


def test_hook_on_modify_uda_aliases_route_through_thin_wrapper():
    """Alias-bearing plain modifies must not be swallowed by the thin fast path."""
    hook = _find_hook_file("on-modify.nautical")
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "nautical.toml"
        config.write_text("enable_uda_aliases = true\ntz = \"UTC\"\n", encoding="utf-8")
        old = {
            "uuid": "00000000-0000-4000-8000-000000000115",
            "description": "plain",
            "status": "pending",
        }
        new = dict(old, description="plain a:w:mon")
        raw = json.dumps(old) + "\n" + json.dumps(new)
        env = {"NAUTICAL_CONFIG": str(config), "NAUTICAL_TRUST_CONFIG_PATH": "1", "TASKDATA": td, "NO_COLOR": "1"}
        proc = _run_hook_script_raw(hook, raw, env_extra=env)
        expect(proc.returncode == 0, f"enabled alias modify hook failed: {proc.stderr[:600]!r}")
        _assert_stdout_json_only(proc.stdout)
        normalized = _extract_last_json(proc.stdout)
        expect(normalized.get("description") == "plain", f"modify alias remained in description: {normalized!r}")
        expect(normalized.get("anchor") == "w:mon", f"modify alias did not reach canonical UDA: {normalized!r}")

        alias_only = dict(old, description="a:w:tue")
        proc = _run_hook_script_raw(hook, json.dumps(old) + "\n" + json.dumps(alias_only), env_extra=env)
        expect(proc.returncode == 0, f"alias-only modify failed: {proc.stderr[:600]!r}")
        _assert_stdout_json_only(proc.stdout)
        normalized = _extract_last_json(proc.stdout)
        expect(normalized.get("description") == "plain", f"alias-only modify erased description: {normalized!r}")
        expect(normalized.get("anchor") == "w:tue", f"alias-only modify did not update canonical UDA: {normalized!r}")


def test_hook_on_modify_uda_alias_anchor_change_emits_ack_panel():
    """A description alias changing an existing anchor must still acknowledge the edit."""
    hook = _find_hook_file("on-modify.nautical")
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "nautical.toml"
        config.write_text("enable_uda_aliases = true\ntz = \"UTC\"\n", encoding="utf-8")
        old = {
            "uuid": "00000000-0000-4000-8000-000000000117",
            "description": "plain",
            "status": "pending",
            "anchor": "w:mon",
            "chain": "on",
            "chainID": "abcd1234",
            "link": 1,
        }
        new = dict(old, description="plain a:w:tue")
        env = {"NAUTICAL_CONFIG": str(config), "NAUTICAL_TRUST_CONFIG_PATH": "1", "TASKDATA": td, "NO_COLOR": "1"}
        proc = _run_hook_script_raw(hook, json.dumps(old) + "\n" + json.dumps(new), env_extra=env)

    expect(proc.returncode == 0, f"alias anchor modify failed: {proc.stderr[:600]!r}")
    _assert_stdout_json_only(proc.stdout)
    normalized = _extract_last_json(proc.stdout)
    expect(normalized.get("anchor") == "w:tue", f"alias anchor was not normalized: {normalized!r}")
    expect("Nautical recurrence updated" in proc.stderr, f"alias anchor acknowledgement missing: {proc.stderr!r}")
    expect("Anchor: w:mon" in proc.stderr and "w:tue" in proc.stderr, f"alias anchor diff missing: {proc.stderr!r}")


def test_hook_on_modify_empty_uda_alias_clears_through_thin_wrapper():
    """The native empty-value clearing form must survive the wrapper boundary."""
    hook = _find_hook_file("on-modify.nautical")
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "nautical.toml"
        config.write_text("enable_uda_aliases = true\ntz = \"UTC\"\n", encoding="utf-8")
        old = {
            "uuid": "00000000-0000-4000-8000-000000000116",
            "description": "plain",
            "status": "pending",
            "anchor": "w:mon",
            "anchor_mode": "skip",
            "chain": "on",
        }
        new = dict(old, description="plain a:")
        raw = json.dumps(old) + "\n" + json.dumps(new)
        env = {"NAUTICAL_CONFIG": str(config), "NAUTICAL_TRUST_CONFIG_PATH": "1", "TASKDATA": td, "NO_COLOR": "1"}
        proc = _run_hook_script_raw(hook, raw, env_extra=env)
        expect(proc.returncode == 0, f"empty alias clear failed: {proc.stderr[:600]!r}")
        _assert_stdout_json_only(proc.stdout)
        normalized = _extract_last_json(proc.stdout)
        expect(normalized.get("description") == "plain", f"empty alias remained in description: {normalized!r}")
        expect("anchor" not in normalized, f"empty alias did not clear anchor: {normalized!r}")


def test_hook_on_add_disabled_uda_aliases_leave_description_untouched():
    """Disabling aliases must preserve alias-looking text as ordinary description content."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "nautical.toml"
        config.write_text("enable_uda_aliases = false\ntz = \"UTC\"\n", encoding="utf-8")
        task = {
            "uuid": "00000000-0000-4000-8000-000000000117",
            "description": "ordinary prose a:book",
            "status": "pending",
            "entry": "20260803T000000Z",
        }
        env = {"NAUTICAL_CONFIG": str(config), "NAUTICAL_TRUST_CONFIG_PATH": "1", "TASKDATA": td, "NO_COLOR": "1"}
        proc = _run_hook_script(hook, task, env_extra=env)
        expect(proc.returncode == 0, f"disabled alias hook failed: {proc.stderr[:600]!r}")
        _assert_stdout_json_only(proc.stdout)
        normalized = _extract_last_json(proc.stdout)
        expect(normalized.get("description") == task["description"], f"disabled alias changed description: {normalized!r}")
        expect("anchor" not in normalized, f"disabled alias created a canonical UDA: {normalized!r}")


def test_on_modify_expands_and_clears_description_uda_aliases():
    """on-modify aliases should update unchanged fields and support explicit clearing."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_description_aliases_test")
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


def test_on_modify_invalid_json_passthrough():
    """Malformed JSON should fail fast without stdout JSON."""
    path = _find_hook_file("on-modify.nautical")
    raw = "{not-json}"
    p = _run_hook_script_raw(path, raw)
    expect(p.returncode != 0, "on-modify should fail on invalid JSON input")
    expect((p.stdout or "").strip() == "", f"expected no stdout on failure, got: {p.stdout!r}")

def test_local_datetime_non_hour_dst_gap_is_shared_by_modify():
    """A 30-minute DST gap must shift by its actual transition size everywhere."""
    from zoneinfo import ZoneInfo
    from nautical_core.timeutil import build_local_datetime

    zone = ZoneInfo("Australia/Lord_Howe")
    scheduled = build_local_datetime(date(2026, 10, 4), (2, 15), zone)
    scheduled_local = scheduled.astimezone(zone)
    expect(
        (scheduled_local.hour, scheduled_local.minute) == (2, 45),
        f"non-hour DST gap should preserve the 30-minute shift: {scheduled_local}",
    )

    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_modify_non_hour_dst_gap_test")
    old_name = mod.core.LOCAL_TZ_NAME
    old_tz = mod.core._LOCAL_TZ
    try:
        mod.core.LOCAL_TZ_NAME = "Australia/Lord_Howe"
        mod.core._LOCAL_TZ = zone
        effects = mod._module("modify_datetime_effects")
        carried = effects.local_naive_to_utc(
            effects.datetime_effect_ports_for(mod), datetime(2026, 10, 4, 2, 15)
        )
    finally:
        mod.core.LOCAL_TZ_NAME = old_name
        mod.core._LOCAL_TZ = old_tz
    carried_local = carried.astimezone(zone)
    expect(
        (carried_local.hour, carried_local.minute) == (2, 45),
        f"modify DST resolver diverged from scheduling: {carried_local}",
    )


def test_modify_completion_advances_past_second_dst_fold():
    """A completion in the second repeated hour must not select a first-fold slot."""
    from zoneinfo import ZoneInfo

    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_modify_second_fold_completion_test")
    zone = ZoneInfo("Europe/Bucharest")
    old_name = mod.core.LOCAL_TZ_NAME
    old_tz = mod.core._LOCAL_TZ
    try:
        mod.core.LOCAL_TZ_NAME = "Europe/Bucharest"
        mod.core._LOCAL_TZ = zone
        due = datetime(2026, 10, 25, 3, 0, tzinfo=zone, fold=0)
        completed = datetime(2026, 10, 25, 3, 15, tzinfo=zone, fold=1)
        child_due, _meta, _dnf = _compute_anchor_child_due(mod, {
            "anchor": "w:sun@t=03:20",
            "anchor_mode": "skip",
            "chainID": "dst-second-fold",
            "link": 1,
            "due": mod.core.fmt_isoz(due),
            "end": mod.core.fmt_isoz(completed),
        })
    finally:
        mod.core.LOCAL_TZ_NAME = old_name
        mod.core._LOCAL_TZ = old_tz
    child_local = child_due.astimezone(zone)
    expect(
        child_local.date() == date(2026, 11, 1)
        and (child_local.hour, child_local.minute) == (3, 20),
        f"second-fold completion selected a backward occurrence: {child_local}",
    )


def test_modify_overnight_window_advances_past_second_dst_fold():
    """An overnight window must reject its first-fold slot after a second-fold cursor."""
    from zoneinfo import ZoneInfo

    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_modify_overnight_second_fold_test")
    zone = ZoneInfo("Europe/Bucharest")
    old_name = mod.core.LOCAL_TZ_NAME
    old_tz = mod.core._LOCAL_TZ
    try:
        mod.core.LOCAL_TZ_NAME = "Europe/Bucharest"
        mod.core._LOCAL_TZ = zone
        dnf = mod.core.validate_anchor_expr_strict("w:sat@t=22:20..03:20/6")
        cursor = datetime(2026, 10, 25, 3, 15, tzinfo=zone, fold=1)
        schedule = mod._module("modify_schedule_effects")
        from nautical_core.add_anchor_compute import anchor_next_occurrence_after_local_dt

        result = schedule.next_occurrence_after_local_dt(schedule.OccurrencePorts(
            lambda dnf, after, **kwargs: anchor_next_occurrence_after_local_dt(
                dnf, after, core=mod.core, **kwargs
            )
        ),
            dnf,
            cursor,
            default_seed_date=date(2026, 10, 24),
            seed_base="dst-overnight-second-fold",
            fallback_hhmm=(22, 20),
        )
    finally:
        mod.core.LOCAL_TZ_NAME = old_name
        mod.core._LOCAL_TZ = old_tz
    expect(
        result.date() == date(2026, 10, 31)
        and (result.hour, result.minute) == (22, 20),
        f"overnight second-fold cursor selected a backward occurrence: {result}",
    )




def test_anchor_preview_explains_nonexistent_wall_time_adjustment():
    """The add panel should identify a fixed anchor time shifted by DST."""
    hook = _find_hook_file("on-add.nautical")
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
        proc = _run_hook_script(
            hook,
            task,
            env_extra={"NAUTICAL_CONFIG": str(config), "NO_COLOR": "1"},
        )
    expect(proc.returncode == 0, f"non-hour DST preview failed: {proc.stderr!r}")
    panel = _strip_markup(proc.stderr)
    expect("DST adjusted" in panel, f"DST adjustment row is missing: {panel!r}")
    expect("02:15 -> 02:45" in panel, f"DST adjustment clocks are missing: {panel!r}")


def test_on_modify_collect_prev_two_prefers_live_statuses():
    """Previous-link lookup should prefer live tasks over deleted duplicates."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_collect_prev_two_test")
    current = {"chainID": "abcd1234", "link": 4}
    chain_by_link = {
        2: [
            {"uuid": "deleted-2", "status": "deleted", "link": 2},
            {"uuid": "pending-2", "status": "pending", "link": 2},
        ],
        3: [
            {"uuid": "deleted-3", "status": "deleted", "link": 3},
            {"uuid": "completed-3", "status": "completed", "link": 3},
        ],
    }

    reads = mod._module("modify_read_effects")
    state = mod._modify_runtime_state()
    ports = reads.PreviousChainPorts(
        service=mod._module("modify_composition").lifecycle_read_service_for(mod),
        panel_chain_by_link=state.panel_chain_by_link,
        panel_chain_snapshot_loaded=state.panel_chain_snapshot_loaded,
    )
    prevs = reads.collect_prev_two(ports, current, chain_by_link=chain_by_link)
    expect([t.get("uuid") for t in prevs] == ["pending-2", "completed-3"], f"unexpected prevs: {prevs}")

def test_year_ordinals_hooks_modes_calendar_and_timeline():
    """Ordinal selectors should work through add, completion modes, named calendars, and timelines."""
    add_hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config-nautical.toml"
        config_path.write_text(
            'tz = "UTC"\n'
            '[business_calendar.work]\n'
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
        proc = _run_hook_script(
            add_hook,
            task,
            env_extra={"NO_COLOR": "1", "NAUTICAL_CONFIG": str(config_path)},
        )
        expect(proc.returncode == 0, f"on-add rejected year-day calendar anchor: {proc.stderr}")
        out_task = _assert_stdout_json_only(proc.stdout)
        expect(out_task.get("anchor") == task["anchor"], f"on-add changed ordinal anchor: {out_task}")
        expect(out_task.get("bc") == "work", f"on-add did not normalize calendar: {out_task}")
        due = datetime.fromisoformat(str(out_task.get("due")))
        expect(due.date() == date(2026, 12, 30), f"calendar did not roll closed d-1 backward: {due}")
        expect("last day of each year" in _strip_markup(proc.stderr), f"add preview omitted ordinal natural text: {proc.stderr}")

    modify_hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(modify_hook, "_nautical_year_ordinal_hook_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    def _stamp(day, hhmm):
        return mod.core.fmt_isoz(mod.core.build_local_datetime(day, hhmm))

    common = {
        "anchor": "y:w20@t=09:00",
        "due": _stamp(date(2026, 5, 11), (9, 0)),
        "end": _stamp(date(2026, 5, 14), (10, 0)),
        "chainID": "abcd1234",
    }
    all_due, all_meta, _all_dnf = _compute_anchor_child_due(mod,
        dict(common, anchor_mode="all", scheduled=_stamp(date(2026, 5, 11), (9, 0)))
    )
    skip_due, skip_meta, _skip_dnf = _compute_anchor_child_due(mod, dict(common, anchor_mode="skip"))
    expect(mod.core.to_local(all_due).date() == date(2026, 5, 12), f"all mode did not backfill ISO week: {all_due}")
    expect(all_meta.get("basis") == "missed", f"all mode metadata drifted: {all_meta}")
    expect(mod.core.to_local(skip_due).date() == date(2026, 5, 15), f"skip mode did not advance after completion: {skip_due}")
    expect(skip_meta.get("basis") == "after_end", f"skip mode metadata drifted: {skip_meta}")

    monday_expr = "y:w20 + w:mon@t=09:00"
    child_due, _meta, child_dnf = _compute_anchor_child_due(mod,
        {
            "anchor": monday_expr,
            "anchor_mode": "skip",
            "due": _stamp(date(2026, 5, 11), (9, 0)),
            "end": _stamp(date(2026, 5, 11), (10, 0)),
            "chainID": "abcd1234",
        }
    )
    expect(mod.core.to_local(child_due).date() == date(2027, 5, 17), f"completion lost ISO-week weekday: {child_due}")

    saved_collect = getattr(mod, "_collect_prev_two", None)
    mod._collect_prev_two = lambda _task: []
    try:
        lines = _call_with_supported_kwargs(
            mod._timeline_lines,
            kind="anchor",
            task={
                "anchor": monday_expr,
                "anchor_mode": "skip",
                "link": 2,
                "due": _stamp(date(2026, 5, 11), (9, 0)),
                "end": _stamp(date(2026, 5, 11), (10, 0)),
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
    timeline = _strip_markup("\n".join(lines))
    expect("2027-05-17" in timeline, f"timeline omitted ordinal child: {timeline}")
    expect("2028-05-15" in timeline, f"timeline omitted future ordinal occurrence: {timeline}")


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


def test_business_calendar_toml_section_resolves_lazily():
    """a real [business_calendar.<name>] TOML section should load through the core facade."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        config_path = base / 'config-nautical.toml'
        config_path.write_text(
            '[business_calendar.work]\n'
            'anchor = "w:mon..fri"\n'
            'omit = "y:04-20"\n',
            encoding='utf-8',
        )
        script = (
            'import json\n'
            'from datetime import date\n'
            'import nautical_core as core\n'
            'policy = core.get_configured_business_calendar("WORK")\n'
            'print(json.dumps({'
            '"names": sorted(core.business_calendar_definitions()), '
            '"open": policy.is_business_day(date(2026, 4, 21)), '
            '"closed": policy.is_business_day(date(2026, 4, 20))}))\n'
        )
        env = os.environ.copy()
        env['PYTHONPATH'] = ROOT + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env['NAUTICAL_CONFIG'] = str(config_path)
        proc = subprocess.run(
            [sys.executable, '-c', script],
            text=True,
            capture_output=True,
            env=env,
            timeout=10,
        )
        expect(proc.returncode == 0, f'calendar TOML subprocess failed: {proc.stderr}')
        payload = json.loads(proc.stdout)
        expect(payload == {'names': ['work'], 'open': True, 'closed': False}, f'unexpected TOML result: {payload!r}')


def test_hook_on_add_uses_and_normalizes_business_calendar():
    """on-add should use bc for recurrence calculation and emit its canonical name."""
    hook = _find_hook_file('on-add.nautical')
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / 'config-nautical.toml'
        config_path.write_text(
            '[business_calendar.weekend]\n'
            'anchor = "w:sat,sun"\n',
            encoding='utf-8',
        )
        task = {
            'uuid': '00000000-0000-4000-8000-000000000121',
            'description': 'weekend business calendar',
            'status': 'pending',
            'entry': '20260714T000000Z',
            'anchor': 'm:1bd@t=09:00',
            'anchor_mode': 'skip',
            'bc': 'WEEKEND',
        }
        proc = _run_hook_script(
            hook,
            task,
            env_extra={'NO_COLOR': '1', 'NAUTICAL_CONFIG': str(config_path)},
        )
        expect(proc.returncode == 0, f'on-add named calendar failed: {proc.stderr[:500]!r}')
        out_task = _assert_stdout_json_only(proc.stdout)
        due = datetime.fromisoformat(str(out_task.get('due')))
        expect(due.weekday() in {5, 6}, f'named weekend calendar was ignored: {due}')
        expect(out_task.get('bc') == 'weekend', f'bc was not normalized: {out_task!r}')


def test_hook_on_add_reports_business_calendar_displacement_only_when_shifted():
    """On-add should explain a named-calendar roll while leaving unchanged anchors quiet."""
    hook = _find_hook_file('on-add.nautical')
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / 'config-nautical.toml'
        config_path.write_text(
            '[business_calendar.work]\n'
            'anchor = "w:mon..fri"\n'
            'omit = "y:04-24"\n',
            encoding='utf-8',
        )
        base = {
            'uuid': '00000000-0000-4000-8000-000000000125',
            'description': 'calendar displacement',
            'status': 'pending',
            'due': '20260420T060000Z',
            'anchor_mode': 'skip',
            'bc': 'work',
        }
        shifted = _run_hook_script(
            hook,
            {**base, 'anchor': 'y:04-24@nbd@t=09:00'},
            env_extra={'NO_COLOR': '1', 'NAUTICAL_CONFIG': str(config_path)},
        )
        unchanged = _run_hook_script(
            hook,
            {**base, 'uuid': '00000000-0000-4000-8000-000000000126', 'anchor': 'y:04-23@nbd@t=09:00'},
            env_extra={'NO_COLOR': '1', 'NAUTICAL_CONFIG': str(config_path)},
        )

    expect(shifted.returncode == 0, f'shifted calendar add failed: {shifted.stderr[:500]!r}')
    _assert_stdout_json_only(shifted.stdout)
    shifted_err = _strip_markup(shifted.stderr)
    expect('Business calendar adjusted' in shifted_err, f'displacement panel missing: {shifted_err[:800]!r}')
    expect('Calendar work' in shifted_err, f'calendar name missing: {shifted_err[:800]!r}')
    expect('Fri 2026-04-24' in shifted_err and 'Mon 2026-04-27' in shifted_err, f'displacement dates missing: {shifted_err[:800]!r}')
    expect(unchanged.returncode == 0, f'unchanged calendar add failed: {unchanged.stderr[:500]!r}')
    _assert_stdout_json_only(unchanged.stdout)
    expect('Business calendar adjusted' not in _strip_markup(unchanged.stderr), f'unchanged anchor should stay quiet: {unchanged.stderr[:800]!r}')


def test_hook_on_add_rejects_unknown_business_calendar_cleanly():
    """unknown bc values should fail before recurrence scheduling with an actionable error."""
    hook = _find_hook_file('on-add.nautical')
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / 'config-nautical.toml'
        config_path.write_text(
            '[business_calendar.work]\n'
            'anchor = "w:mon..fri"\n',
            encoding='utf-8',
        )
        task = {
            'uuid': '00000000-0000-4000-8000-000000000122',
            'description': 'unknown business calendar',
            'status': 'pending',
            'anchor': 'w:mon',
            'bc': 'missing',
        }
        proc = _run_hook_script(
            hook,
            task,
            env_extra={'NO_COLOR': '1', 'NAUTICAL_CONFIG': str(config_path)},
        )
        expect(proc.returncode != 0, 'on-add should reject an unknown business calendar')
        stderr_text = _strip_markup(proc.stderr)
        expect('Invalid business calendar' in stderr_text, f'missing error title: {stderr_text[:500]!r}')
        expect(
            'configured calendars:' in stderr_text and 'work.' in stderr_text,
            f'missing available-calendar hint: {stderr_text[:500]!r}',
        )


def test_hook_on_add_rejects_invalid_timezone_for_nautical_task():
    """A bad timezone must block recurrence scheduling instead of using UTC silently."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config-nautical.toml"
        config_path.write_text('tz = "Invalid/Timezone"\n', encoding="utf-8")
        proc = _run_hook_script(
            hook,
            {
                "uuid": "00000000-0000-4000-8000-000000000127",
                "description": "invalid timezone recurrence",
                "status": "pending",
                "anchor": "w:mon",
            },
            env_extra={"NO_COLOR": "1", "NAUTICAL_CONFIG": str(config_path)},
        )
    expect(proc.returncode != 0, "invalid timezone should block Nautical on-add")
    stderr_text = _strip_markup(proc.stderr)
    expect("Invalid Nautical configuration" in stderr_text, f"missing config error title: {stderr_text[:800]!r}")
    expect("timezone" in stderr_text.lower(), f"timezone cause missing: {stderr_text[:800]!r}")




def test_discovered_malformed_config_blocks_taskdata_reload():
    """A malformed Taskdata-discovered config must not silently select defaults."""
    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td)
        (taskdata / "config-nautical.toml").write_text(
            "tz = \"Europe/Athens\"\n[broken\n", encoding="utf-8"
        )
        env = os.environ.copy()
        env["TASKDATA"] = str(taskdata)
        env["TASKRC"] = str(taskdata / "taskrc")
        env.pop("NAUTICAL_CONFIG", None)
        env["PYTHONPATH"] = str(ROOT)
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os, nautical_core as c; c.reload_taskdata_config(os.environ['TASKDATA'])",
            ],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
        )
        expect(proc.returncode != 0, "malformed discovered config was accepted")
        detail = f"{proc.stdout}\n{proc.stderr}".lower()
        expect("config parse failed" in detail, f"parse failure detail missing: {detail[:800]!r}")


def test_taskdata_reload_exposes_consistent_validated_fingerprints():
    """All lifecycle tools should receive one effective configuration identity."""
    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td)
        (taskdata / "config-nautical.toml").write_text(
            'tz = "Europe/Athens"\nseason_hemisphere = "north"\n', encoding="utf-8"
        )
        env = os.environ.copy()
        env.pop("NAUTICAL_CONFIG", None)
        env["PYTHONPATH"] = str(ROOT)
        script = (
            "import json, os, nautical_core as c\n"
            "a = c.reload_taskdata_config(os.environ['TASKDATA'])\n"
            "drift = c.configuration_drift()\n"
            "b = c.reload_taskdata_config(os.environ['TASKDATA'])\n"
            "print(json.dumps({'a': a, 'b': b, 'drift': drift,"
            " 'effective': c.effective_config_fingerprint(),"
            " 'scheduler': c.scheduler_config_fingerprint()}))\n"
        )
        env["TASKDATA"] = str(taskdata)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
        )
        expect(proc.returncode == 0, f"validated reload process failed: {proc.stderr[:500]!r}")
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        first = payload["a"]
        second = payload["b"]
        expect(first["ok"] and second["ok"], f"reload did not report success: {payload!r}")
        expect(first["fingerprint"] == second["fingerprint"], "effective fingerprint changed on identical reload")
        expect(
            first["scheduler_fingerprint"] == second["scheduler_fingerprint"],
            "scheduler fingerprint changed on identical reload",
        )
        expect(first["fingerprint"] == payload["effective"], "reload and core effective fingerprints differ")
        expect(
            first["scheduler_fingerprint"] == payload["scheduler"],
            "reload and core scheduler fingerprints differ",
        )
        expect(payload["drift"]["status"] == "ok", f"identical reload left configuration drifted: {payload!r}")


def test_hook_on_modify_rejects_unknown_business_calendar_cleanly():
    """changing bc to an unknown name should fail before recurrence is evaluated."""
    hook = _find_hook_file('on-modify.nautical')
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / 'config-nautical.toml'
        config_path.write_text(
            '[business_calendar.work]\n'
            'anchor = "w:mon..fri"\n',
            encoding='utf-8',
        )
        old = {
            'uuid': '00000000-0000-4000-8000-000000000123',
            'description': 'change business calendar',
            'status': 'pending',
            'anchor': 'w:mon',
            'bc': 'work',
        }
        new = dict(old, bc='missing')
        proc = _run_hook_script_raw(
            hook,
            json.dumps(old) + '\n' + json.dumps(new) + '\n',
            env_extra={'NO_COLOR': '1', 'NAUTICAL_CONFIG': str(config_path)},
        )
        expect(proc.returncode != 0, 'on-modify should reject an unknown business calendar')
        expect(not proc.stdout.strip(), f'failing hook should not emit stdout: {proc.stdout[:500]!r}')
        stderr_text = _strip_markup(proc.stderr)
        expect('Invalid business calendar' in stderr_text, f'missing error title: {stderr_text[:500]!r}')
        expect(
            'configured calendars:' in stderr_text and 'work.' in stderr_text,
            f'missing available-calendar hint: {stderr_text[:500]!r}',
        )


def test_hook_on_modify_rejects_invalid_timezone_for_nautical_task():
    """A bad timezone must block recurrence mutation instead of using UTC silently."""
    hook = _find_hook_file("on-modify.nautical")
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config-nautical.toml"
        config_path.write_text('tz = "Invalid/Timezone"\n', encoding="utf-8")
        old = {
            "uuid": "00000000-0000-4000-8000-000000000128",
            "description": "invalid timezone modify",
            "status": "pending",
            "anchor": "w:mon",
        }
        new = dict(old, status="completed", end="20260808T120000Z", modified="20260808T120000Z")
        proc = _run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new) + "\n",
            env_extra={"NO_COLOR": "1", "NAUTICAL_CONFIG": str(config_path)},
        )
    expect(proc.returncode != 0, "invalid timezone should block Nautical on-modify")
    stderr_text = _strip_markup(proc.stderr)
    expect("Invalid Nautical configuration" in stderr_text, f"missing config error title: {stderr_text[:800]!r}")
    expect("timezone" in stderr_text.lower(), f"timezone cause missing: {stderr_text[:800]!r}")


def test_on_modify_spawned_child_preserves_business_calendar():
    """completion spawning should copy the parent's canonical bc value unchanged."""
    hook = _find_hook_file('on-modify.nautical')
    mod = _load_hook_module(hook, '_nautical_on_modify_business_calendar_child_test')
    child_due = mod.core.build_local_datetime(date(2026, 7, 18), (9, 0))
    parent = {
        'uuid': '00000000-0000-4000-8000-000000000124',
        'description': 'weekend chain',
        'status': 'completed',
        'due': mod.core.fmt_isoz(mod.core.build_local_datetime(date(2026, 7, 12), (9, 0))),
        'anchor': 'm:1bd',
        'anchor_mode': 'skip',
        'bc': 'weekend',
        'chainID': 'calendar-chain',
        'link': 1,
    }
    child = _build_child_draft_for_test(mod,
        parent,
        child_due,
        'due',
        2,
        '00000000',
        'anchor',
        0,
        None,
    )
    expect(child.get('bc') == 'weekend', f'child lost its business calendar: {child!r}')


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

    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "nautical.toml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write('[anchor_presets]\nrandom-workday = "m:rand@bd"\n\n')
            f.write('[omit_presets]\nrandom-weekday = "w:rand"\n')
        mod = _load_core_module(core_path, "_nautical_core_random_preset_test", cfg)
        verify(mod)


def test_chain_colour_uses_complete_root_identity():
    """Chain colours should hash the full root instead of its final UUID suffix."""
    hook = _load_hook_module(
        _find_hook_file("on-modify.nautical"),
        "_nautical_chain_colour_full_identity_test",
    )

    root = "12345678-1234-4234-8234-00000000abcd"
    expect(
        hook.core.chain_colour_root("anchor", root) == hook.core.chain_colour_root("anchor", root),
        "chain colour should replay deterministically",
    )
    expect(
        hook.core.chain_colour_root("anchor", root.upper()) == hook.core.chain_colour_root("anchor", root),
        "UUID case should not change the chain colour",
    )
    expect(
        hook.core.chain_colour_root("anchor", "") == "bright_cyan",
        "empty anchor roots should retain the existing fallback colour",
    )
    expect(
        hook.core.chain_colour_root("cp", "") == "orange_red1",
        "empty cp roots should retain the existing fallback colour",
    )

    same_suffix_roots = [
        f"{idx:08x}-1234-4234-8234-00000000abcd"
        for idx in range(256)
    ]
    anchor_colours = {
        hook.core.chain_colour_root("anchor", candidate)
        for candidate in same_suffix_roots
    }
    cp_colours = {
        hook.core.chain_colour_root("cp", candidate)
        for candidate in same_suffix_roots
    }
    expect(
        len(anchor_colours) >= 16,
        f"anchor colours still collapse roots sharing one suffix: {anchor_colours}",
    )
    expect(
        len(cp_colours) >= 15,
        f"cp colours still collapse roots sharing one suffix: {cp_colours}",
    )

    legacy = "legacy/root identifier"
    expect(
        hook.core.chain_colour_root("anchor", legacy) == hook.core.chain_colour_root("anchor", legacy),
        "non-UUID legacy roots should remain deterministic",
    )
    expect(
        hook.core.chain_colour_root("anchor", root) != hook.core.chain_colour_root("cp", root),
        "anchor and cp colour domains should remain separated",
    )


def test_on_add_preview_uses_configured_chain_colour():
    """on-add should use the same chain colour as on-modify when enabled."""
    hook = _load_hook_module(
        _find_hook_file("on-add.nautical"),
        "_nautical_on_add_chain_colour_test",
    )
    task = {"chainID": "12345678", "anchor": "w:mon"}
    captured = {}
    original_render = hook.core.render_panel
    original_setting = hook.core.CHAIN_COLOR_PER_CHAIN
    try:
        hook.core.render_panel = lambda *_args, **kwargs: captured.update(kwargs)
        hook.core.CHAIN_COLOR_PER_CHAIN = True
        hook._panel("Preview", [("Pattern", "w:mon")], kind="preview_anchor", task=task)
        expected = hook.core.chain_colour_root("anchor", "12345678")
        theme = captured.get("themes", {}).get("preview_anchor", {})
        expect(theme.get("border") == expected, f"unexpected on-add border colour: {theme!r}")
        expect(theme.get("title") == expected, f"unexpected on-add title colour: {theme!r}")
        expect(
            captured.get("live_duration_ms") == hook.core.LIVE_PANEL_DURATION_MS,
            f"on-add did not forward live duration: {captured!r}",
        )

        captured.clear()
        hook.core.CHAIN_COLOR_PER_CHAIN = False
        hook._panel("Preview", [("Pattern", "w:mon")], kind="preview_anchor", task=task)
        theme = captured.get("themes", {}).get("preview_anchor", {})
        expect(theme.get("border") == "turquoise2", f"disabled setting changed static border: {theme!r}")
        expect(captured.get("themes") == hook.core.panel_themes(), f"on-add did not use shared themes: {captured!r}")
    finally:
        hook.core.render_panel = original_render
        hook.core.CHAIN_COLOR_PER_CHAIN = original_setting


def test_cp_interval_helpers_agree_between_on_add_and_on_modify():
    """on-add preview and on-modify completion should select the same cp interval for the same link."""
    add_mod = _load_hook_module(_find_hook_file("on-add.nautical"), "_nautical_on_add_cp_interval_agreement_test")
    modify_mod = _load_hook_module(_find_hook_file("on-modify.nautical"), "_nautical_on_modify_cp_interval_agreement_test")
    if hasattr(add_mod, "_load_core"):
        add_mod._load_core()
    if hasattr(modify_mod, "_load_core"):
        modify_mod._load_core()

    chain_id = "abcd1234"
    for cp, link_no in (
        ("3d,20d,7d", 1),
        ("3d,20d,7d", 3),
        ("3d,20d,7d", 4),
        ("3d,rand(10d..20d),7d", 2),
        ("3d,14d~2d,7d", 2),
        ("rand(12h..36h)", 5),
    ):
        tokens = core.parse_cp_sequence_tokens(cp)
        add_preview = add_mod._module("add_preview_composition")
        add_td = add_preview.cp_sequence_period_for_link(add_mod, tokens, cp, link_no, chain_id)
        schedule = modify_mod._module("modify_schedule_effects")
        modify_td = schedule.sequence_period_for_link(
            schedule.SequencePorts(core.cp_sequence_interval_for_token), tokens, cp, link_no, chain_id
        )
        core_td = core.cp_sequence_interval_for_link(cp, link_no, chain_id)
        expect(add_td == modify_td == core_td, f"cp interval mismatch for {cp!r} link {link_no}: add={add_td}, modify={modify_td}, core={core_td}")

# -------- Runner --------------------------------------------------------------

def test_hook_on_add_multitime_preview_emits_all_slots():
    """on-add must accept @t=HH:MM list and preview intra-day slots when due is explicit."""
    hook = _find_hook_file("on-add.nautical")
    # Disable ANSI colors for deterministic output.
    env = {"NO_COLOR": "1"}
    expr = "w:wed@t=06:00,12:00,22:00"
    task = {
        "uuid": "00000000-0000-4000-8000-000000000111",
        "description": "hook test on-add multitime",
        "status": "pending",
        "project": "testing",
        "entry": "20251217T000000Z",
        "anchor": expr,
        "anchor_mode": "skip",
        # Explicit due so the preview is deterministic independent of 'now'
        "due": "20251217T060000Z",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    if p.returncode != 0:
        raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:400]!r}")
    out_task = _extract_last_json(p.stdout)
    # The hook should not override an explicit due.
    if out_task.get("due") != task["due"]:
        raise AssertionError(f"on-add changed explicit due: got {out_task.get('due')!r}, want {task['due']!r}")
    # Preview should show other intra-day slots (12:00 and 22:00) on the same date.
    stderr_txt = _strip_markup(p.stderr)
    if "12:00" not in stderr_txt or "22:00" not in stderr_txt:
        raise AssertionError(f"on-add preview missing expected intra-day times. stderr={stderr_txt[:500]!r}")


def test_hook_on_add_time_window_preview_emits_bounded_slots():
    """on-add should preview generated window slots without forcing its end bound."""
    hook = _find_hook_file("on-add.nautical")
    task = {
        "uuid": "00000000-0000-4000-8000-000000000112",
        "description": "hook test time window",
        "status": "pending",
        "project": "testing",
        "entry": "20251217T000000Z",
        "anchor": "w:wed@t=06..17/3h",
        "anchor_mode": "skip",
        "due": "20251217T060000Z",
    }
    proc = _run_hook_script(hook, task, env_extra={"NO_COLOR": "1", "NAUTICAL_CONFIG": ""})
    expect(proc.returncode == 0, f"on-add window hook failed: {proc.stderr[:500]!r}")
    stderr_txt = _strip_markup(proc.stderr)
    expect("09:00" in stderr_txt and "15:00" in stderr_txt, f"window preview omitted generated slots: {stderr_txt[:700]!r}")
    expect("17:00 EET" not in stderr_txt, f"non-divisible window bound was shown as an occurrence: {stderr_txt[:700]!r}")


def test_hook_on_add_overnight_window_keeps_json_and_next_day_preview():
    """The real on-add hook should accept overnight anchors without polluting JSON stdout."""
    hook = _find_hook_file("on-add.nautical")
    task = {
        "uuid": "00000000-0000-4000-8000-000000000119",
        "description": "hook overnight window",
        "status": "pending",
        "project": "testing",
        "entry": "20260804T000000Z",
        "anchor": "w:mon@t=22:30..06:30/7",
        "anchor_mode": "skip",
        "due": "20260810T193000Z",
    }
    proc = _run_hook_script(hook, task, env_extra={"NO_COLOR": "1", "NAUTICAL_CONFIG": ""})
    expect(proc.returncode == 0, f"on-add overnight hook failed: {proc.stderr[:700]!r}")
    out_task = _extract_last_json(proc.stdout)
    expect(out_task.get("uuid") == task["uuid"], f"on-add overnight stdout lost task JSON: {out_task!r}")
    stderr_txt = _strip_markup(proc.stderr)
    expect("23:50" in stderr_txt and "01:10" in stderr_txt, f"overnight preview omitted next-day slots: {stderr_txt[:1000]!r}")


def test_hook_on_add_random_time_window_keeps_json_and_preview():
    """The add hook should resolve deterministic random slots without corrupting stdout JSON."""
    hook = _find_hook_file("on-add.nautical")
    task = {
        "uuid": "00000000-0000-4000-8000-000000000120",
        "description": "hook random window",
        "status": "pending",
        "project": "testing",
        "entry": "20260803T000000Z",
        "anchor": "w:mon@t=rand(06..18/3)",
        "anchor_mode": "skip",
        "chainID": "randomhook1",
        "due": "20260803T060000Z",
    }
    proc = _run_hook_script(hook, task, env_extra={"NO_COLOR": "1", "NAUTICAL_CONFIG": ""})
    expect(proc.returncode == 0, f"on-add random hook failed: {proc.stderr[:700]!r}")
    out_task = _extract_last_json(proc.stdout)
    expect(out_task.get("anchor") == task["anchor"], f"random anchor was lost from stdout JSON: {out_task!r}")
    stderr_txt = _strip_markup(proc.stderr)
    expect("Upcoming" in stderr_txt, f"random preview did not render upcoming slots: {stderr_txt[:1000]!r}")


def test_on_modify_time_window_completion_advances_within_same_day():
    """Completion should advance to the next generated slot before moving to a new date."""
    mod = _load_hook_module(_find_hook_file("on-modify.nautical"), "_nautical_modify_time_window_runtime_test")
    local_due = mod.core.build_local_datetime(date(2025, 12, 17), (6, 0))
    local_end = mod.core.build_local_datetime(date(2025, 12, 17), (6, 30))
    parent = {
        "uuid": "00000000-0000-4000-8000-000000000113",
        "description": "window completion",
        "anchor": "w:mon..sun@t=06..18/3h",
        "anchor_mode": "skip",
        "chain": "on",
        "chainID": "window1234",
        "link": 1,
        "due": mod.core.fmt_isoz(local_due.astimezone(timezone.utc)),
        "end": mod.core.fmt_isoz(local_end.astimezone(timezone.utc)),
    }
    child_due, meta, _dnf = _compute_anchor_child_due(mod, parent)
    child_local = mod.core.to_local(child_due)
    expect((child_local.date(), child_local.hour, child_local.minute) == (date(2025, 12, 17), 9, 0), f"window did not advance within day: {child_local}")
    expect(meta.get("basis") == "after_end", f"unexpected window completion basis: {meta!r}")


def test_on_modify_partitioned_window_completion_rolls_to_next_day():
    """A partitioned window should use each slot once, then roll to the next day."""
    mod = _load_hook_module(_find_hook_file("on-modify.nautical"), "_nautical_modify_partitioned_window_runtime_test")

    def completed_parent(local_due, local_end):
        return {
            "uuid": "00000000-0000-4000-8000-000000000115",
            "description": "partitioned completion",
            "anchor": "w:mon..sun@t=04:30..19:30/3",
            "anchor_mode": "skip",
            "chain": "on",
            "chainID": "partition123",
            "link": 1,
            "due": mod.core.fmt_isoz(local_due.astimezone(timezone.utc)),
            "end": mod.core.fmt_isoz(local_end.astimezone(timezone.utc)),
        }

    first = mod.core.build_local_datetime(date(2025, 12, 17), (4, 30))
    next_slot, _meta, _dnf = _compute_anchor_child_due(mod, completed_parent(first, first + timedelta(minutes=10)))
    expect(mod.core.to_local(next_slot).strftime("%Y-%m-%d %H:%M") == "2025-12-17 12:00", "partitioned window skipped its middle slot")

    last = mod.core.build_local_datetime(date(2025, 12, 17), (19, 30))
    next_day, _meta, _dnf = _compute_anchor_child_due(mod, completed_parent(last, last + timedelta(minutes=10)))
    expect(mod.core.to_local(next_day).strftime("%Y-%m-%d %H:%M") == "2025-12-18 04:30", "partitioned window did not roll to the next day")


def test_on_modify_overnight_window_completion_uses_next_day_slots():
    """Completion across midnight should advance within the owning overnight window."""
    mod = _load_hook_module(_find_hook_file("on-modify.nautical"), "_nautical_modify_overnight_window_runtime_test")

    def parent(local_due, local_end):
        return {
            "uuid": "00000000-0000-4000-8000-000000000117",
            "description": "overnight completion",
            "status": "completed",
            "anchor": "w:mon@t=22:30..06:30/7",
            "anchor_mode": "skip",
            "chain": "on",
            "chainID": "overnight123",
            "link": 1,
            "due": mod.core.fmt_isoz(local_due.astimezone(timezone.utc)),
            "end": mod.core.fmt_isoz(local_end.astimezone(timezone.utc)),
        }

    due = mod.core.build_local_datetime(date(2025, 12, 15), (22, 30))
    child_due, _meta, _dnf = _compute_anchor_child_due(mod, parent(due, due + timedelta(minutes=10)))
    expect(mod.core.to_local(child_due).strftime("%Y-%m-%d %H:%M") == "2025-12-15 23:50", "overnight completion skipped the same-night slot")

    after_midnight = mod.core.build_local_datetime(date(2025, 12, 16), (6, 30))
    child_due, _meta, _dnf = _compute_anchor_child_due(mod, parent(due, after_midnight + timedelta(minutes=10)))
    expect(mod.core.to_local(child_due).strftime("%Y-%m-%d %H:%M") == "2025-12-22 22:30", "overnight completion did not advance to the next Monday window")

    capped = parent(due, due + timedelta(minutes=10))
    capped["chainUntil"] = "20251216T063000Z"
    capped_dnf = mod.core.validate_anchor_expr_strict(capped["anchor"])
    final_no, final_dt = mod._cap_from_until_anchor(capped, due.astimezone(timezone.utc), capped_dnf)
    expect(final_no == 8, f"overnight chainUntil counted the wrong number of links: {final_no!r}, final={final_dt!r}")
    expect(
        final_dt is not None and mod.core.to_local(final_dt).strftime("%Y-%m-%d %H:%M") == "2025-12-16 06:30",
        f"overnight chainUntil stopped before the final morning slot: {final_dt!r}",
    )


def test_on_modify_random_time_window_completion_reuses_stable_slots():
    """Completion should select the next deterministic random slot, not redraw it."""
    mod = _load_hook_module(_find_hook_file("on-modify.nautical"), "_nautical_modify_random_window_runtime_test")
    chain_id = "randommodify1"
    target_date = date(2025, 12, 15)
    mods = {"time_random": "rand(06:00..18:00/3)", "t": []}
    slots = mod.core._import_sibling("time_slots").resolve_time_slots_with_offsets(mods, target_date, seed_base=chain_id)
    def local_slot(slot):
        day_offset, hour, minute = slot
        return mod.core.to_local(
            mod.core.build_local_datetime(target_date + timedelta(days=day_offset), (hour, minute))
        )

    first = local_slot(slots[0])
    expected = local_slot(slots[1])
    parent = {
        "uuid": "00000000-0000-4000-8000-000000000121",
        "description": "random completion",
        "anchor": "w:mon@t=rand(06..18/3)",
        "anchor_mode": "skip",
        "chain": "on",
        "chainID": chain_id,
        "link": 1,
        "due": mod.core.fmt_isoz(first.astimezone(timezone.utc)),
        "end": mod.core.fmt_isoz((first + timedelta(minutes=10)).astimezone(timezone.utc)),
    }
    child_due, _meta, _dnf = _compute_anchor_child_due(mod, parent)
    expect(mod.core.to_local(child_due) == expected, "random completion redrew or skipped the next stable slot")


def test_time_window_dst_gap_deduplicates_shifted_local_slot():
    """A spring-forward slot shifted onto the next slot must not duplicate the occurrence."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "nautical.toml"
        config.write_text('tz = "America/New_York"\n', encoding="utf-8")
        task = {
            "uuid": "00000000-0000-4000-8000-000000000114",
            "description": "DST window",
            "status": "pending",
            "entry": "20250301T000000Z",
            "anchor": "w:sun@t=01..04/1h",
            "anchor_mode": "skip",
            "due": "20250309T060000Z",
        }
        proc = _run_hook_script(
            hook,
            task,
            env_extra={"NAUTICAL_CONFIG": str(config), "NO_COLOR": "1"},
        )
    expect(proc.returncode == 0, f"DST window hook failed: {proc.stderr[:500]!r}")
    text = _strip_markup(proc.stderr)
    expect(text.count("Sun 2025-03-09 03:00 EDT") == 1, f"DST-shifted slot was duplicated: {text[:1200]!r}")


def test_partitioned_time_window_dst_gap_deduplicates_shifted_local_slot():
    """A partition slot shifted across a spring-forward gap must not duplicate a later slot."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "nautical.toml"
        config.write_text('tz = "America/New_York"\n', encoding="utf-8")
        task = {
            "uuid": "00000000-0000-4000-8000-000000000116",
            "description": "DST partitioned window",
            "status": "pending",
            "entry": "20250301T000000Z",
            "anchor": "w:sun@t=01..05/5",
            "anchor_mode": "skip",
            "due": "20250309T060000Z",
        }
        proc = _run_hook_script(
            hook,
            task,
            env_extra={"NAUTICAL_CONFIG": str(config), "NO_COLOR": "1"},
        )
    expect(proc.returncode == 0, f"DST partitioned window hook failed: {proc.stderr[:500]!r}")
    text = _strip_markup(proc.stderr)
    expect(text.count("Sun 2025-03-09 03:00 EDT") == 1, f"partitioned DST slot was duplicated: {text[:1200]!r}")


def test_overnight_time_window_dst_fallback_deduplicates_repeated_local_slot():
    """A fall-back repeated hour in an overnight window must remain one occurrence."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "nautical.toml"
        config.write_text('tz = "America/New_York"\n', encoding="utf-8")
        task = {
            "uuid": "00000000-0000-4000-8000-000000000118",
            "description": "DST overnight fallback",
            "status": "pending",
            "entry": "20261020T000000Z",
            "anchor": "w:sat@t=22:30..02:30/5",
            "anchor_mode": "skip",
            "due": "20261101T023000Z",
        }
        proc = _run_hook_script(
            hook,
            task,
            env_extra={"NAUTICAL_CONFIG": str(config), "NO_COLOR": "1"},
        )
    expect(proc.returncode == 0, f"DST fallback overnight hook failed: {proc.stderr[:500]!r}")
    text = _strip_markup(proc.stderr)
    expect(text.count("Sun 2026-11-01 01:30") == 1, f"repeated fallback slot was duplicated or omitted: {text[:1400]!r}")


def test_chain_until_overnight_window_survives_dst_fallback():
    """chainUntil should include the final overnight slot across a repeated local hour."""
    from zoneinfo import ZoneInfo

    mod = _load_hook_module(_find_hook_file("on-modify.nautical"), "_nautical_modify_dst_fallback_until_test")
    old_name = mod.core.LOCAL_TZ_NAME
    old_tz = mod.core._LOCAL_TZ
    mod.core.LOCAL_TZ_NAME = "America/New_York"
    mod.core._LOCAL_TZ = ZoneInfo("America/New_York")
    try:
        due = mod.core.build_local_datetime(date(2026, 10, 31), (22, 30))
        until = mod.core.build_local_datetime(date(2026, 11, 1), (2, 30))
        parent = {
            "uuid": "00000000-0000-4000-8000-000000000134",
            "description": "DST fallback chain end",
            "status": "completed",
            "anchor": "w:sat@t=22:30..02:30/5",
            "anchor_mode": "skip",
            "chain": "on",
            "chainID": "dstuntil",
            "link": 1,
            "chainUntil": mod.core.fmt_isoz(until.astimezone(timezone.utc)),
            "due": mod.core.fmt_isoz(due.astimezone(timezone.utc)),
            "end": mod.core.fmt_isoz(due.astimezone(timezone.utc)),
        }
        dnf = mod.core.validate_anchor_expr_strict(parent["anchor"])
        final_no, final_dt = mod._cap_from_until_anchor(parent, due.astimezone(timezone.utc), dnf)
        expect(final_no == 6, f"DST fallback chainUntil counted the wrong final link: {final_no!r}")
        expect(final_dt is not None and mod.core.to_local(final_dt).strftime("%Y-%m-%d %H:%M") == "2026-11-01 02:30", f"DST fallback chainUntil stopped early: {final_dt!r}")
    finally:
        mod.core.LOCAL_TZ_NAME = old_name
        mod.core._LOCAL_TZ = old_tz


def test_hook_on_add_live_panel_mode_preserves_captured_protocol():
    """Configured live panels should fall back cleanly when a hook's stderr is captured."""
    hook = _find_hook_file("on-add.nautical")
    task = {
        "uuid": "00000000-0000-4000-8000-000000000121",
        "description": "live panel protocol test",
        "status": "pending",
        "entry": "20260101T000000Z",
        "cp": "1d",
        "due": "20260102T090000Z",
    }
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "nautical.toml"
        config_path.write_text('tz = "UTC"\npanel_mode = "live"\n', encoding="utf-8")
        proc = _run_hook_script(
            hook,
            task,
            env_extra={"NAUTICAL_CONFIG": str(config_path), "NO_COLOR": "1"},
        )

    expect(proc.returncode == 0, f"on-add live mode failed: {proc.stderr[:500]!r}")
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    expect(len(stdout_lines) == 1, f"live mode emitted non-protocol stdout: {proc.stdout!r}")
    output_task = json.loads(stdout_lines[0])
    expect(output_task.get("uuid") == task["uuid"], f"live mode changed hook protocol output: {output_task!r}")
    stderr_text = _strip_markup(proc.stderr)
    expect(
        "Recurring Chain Preview" in stderr_text and "Period" in stderr_text,
        f"captured live mode lost the static panel fallback: {stderr_text[:500]!r}",
    )
    expect("\x1b[" not in proc.stderr, f"captured live hook emitted terminal controls: {proc.stderr!r}")


def test_hook_on_add_counted_random_preview_uses_group_time():
    """on-add should schedule and explain a constrained counted-random anchor."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    expr = "(m:2rand + w:mon..fri)@t=09:00"
    task = {
        "uuid": "00000000-0000-4000-8000-000000000119",
        "description": "hook test counted random",
        "status": "pending",
        "project": "testing",
        "entry": "20260101T000000Z",
        "anchor": expr,
        "anchor_mode": "skip",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    expect(p.returncode == 0, f"on-add counted random failed: {p.stderr[:500]!r}")
    out_task = _extract_last_json(p.stdout)
    due = datetime.fromisoformat(str(out_task.get("due")))
    expect((due.hour, due.minute) == (9, 0), f"group time was not used for first due: {due}")
    expect(due.weekday() < 5, f"weekday constraint was ignored for first due: {due}")
    stderr_txt = _strip_markup(p.stderr)
    expect(
        "2 random days each" in stderr_txt and "month at 09:00" in stderr_txt,
        f"counted-random natural text missing: {stderr_txt[:500]!r}",
    )


def test_hook_on_add_accepts_group_date_modifiers():
    """on-add should accept and schedule date modifiers shared by OR branches."""
    hook = _find_hook_file("on-add.nautical")
    task = {
        "uuid": "00000000-0000-4000-8000-000000000120",
        "description": "hook test grouped date modifiers",
        "status": "pending",
        "project": "testing",
        "entry": "20260101T000000Z",
        "anchor": "(y:04-24 | y:04-30)@pbd@-1bd@t=09:00",
        "anchor_mode": "skip",
    }
    proc = _run_hook_script(hook, task, env_extra={"NO_COLOR": "1"})
    expect(proc.returncode == 0, f"on-add grouped date modifiers failed: {proc.stderr[:500]!r}")
    out_task = _extract_last_json(proc.stdout)
    due = datetime.fromisoformat(str(out_task.get("due")))
    today = core.to_local(core.now_utc()).date()
    expected_date, _meta = core.next_after_expr(core.validate_anchor_expr_strict(task["anchor"]), today)
    expect(due.date() == expected_date, f"grouped date modifiers produced wrong due date: {due}, expected {expected_date}")
    expect((due.hour, due.minute) == (9, 0), f"grouped date modifiers lost shared time: {due}")


def test_hook_on_add_cp_scheduled_only_preserves_no_due():
    """scheduled-only recurring cp tasks should remain scheduled-only on add."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    task = {
        "uuid": "00000000-0000-4000-8000-000000000112",
        "description": "hook test on-add cp scheduled-only",
        "status": "pending",
        "project": "testing",
        "entry": "20251217T000000Z",
        "cp": "P7D",
        "scheduled": "20251217T090000Z",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    if p.returncode != 0:
        raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:400]!r}")
    out_task = _extract_last_json(p.stdout)
    expect(not out_task.get("due"), f"scheduled-only cp add should not set due: {out_task}")
    expect(out_task.get("scheduled") == task["scheduled"], f"scheduled changed unexpectedly: {out_task}")
    stderr_txt = _strip_markup(p.stderr)
    expect("First scheduled" in stderr_txt, f"preview should label scheduled anchor. stderr={stderr_txt[:500]!r}")


def test_hook_on_add_anchor_preset_resolves_from_config():
    """on-add should resolve @anchor presets from config before validation/preview."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        conf = Path(td) / "config-nautical.toml"
        conf.write_text('[anchor_presets]\npayday = "m:15"\n', encoding="utf-8")
        env = {"NO_COLOR": "1", "NAUTICAL_CONFIG": str(conf)}
        task = {
            "uuid": "00000000-0000-4000-8000-000000000118",
            "description": "hook test on-add anchor preset",
            "status": "pending",
            "project": "testing",
            "entry": "20260101T000000Z",
            "anchor": "@payday",
            "anchor_mode": "skip",
            "due": "20260101T090000Z",
        }
        p = _run_hook_script(hook, task, env_extra=env)
        if p.returncode != 0:
            raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:500]!r}")
        out_task = _extract_last_json(p.stdout)
        expect(out_task.get("anchor") == "@payday", f"anchor preset expression should be preserved: {out_task}")
        stderr_txt = _strip_markup(p.stderr)
        expect("Invalid anchor" not in stderr_txt, f"preset should validate cleanly: {stderr_txt[:500]!r}")
        expect("Preset" in stderr_txt and "@payday → m:15" in stderr_txt, f"preset preview should show expansion: {stderr_txt[:500]!r}")
        expect("2026-01-15" in stderr_txt, f"preset preview should use resolved anchor expression: {stderr_txt[:500]!r}")


def test_hook_on_add_anchor_unknown_preset_fails_cleanly():
    """on-add should fail clearly when an anchor preset is not configured."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        conf = Path(td) / "config-nautical.toml"
        conf.write_text("[anchor_presets]\n", encoding="utf-8")
        env = {"NO_COLOR": "1", "NAUTICAL_CONFIG": str(conf)}
        task = {
            "uuid": "00000000-0000-4000-8000-000000000119",
            "description": "hook test on-add unknown anchor preset",
            "status": "pending",
            "project": "testing",
            "entry": "20260101T000000Z",
            "anchor": "@missing",
            "anchor_mode": "skip",
            "due": "20260101T090000Z",
        }
        p = _run_hook_script(hook, task, env_extra=env)
        expect(p.returncode != 0, "on-add should fail for unknown anchor preset")
        expect((p.stdout or "").strip() == "", f"expected no stdout on unknown preset failure, got: {p.stdout!r}")
        stderr_txt = _strip_markup(p.stderr)
        expect("Invalid anchor" in stderr_txt, f"expected invalid anchor panel. stderr={stderr_txt[:500]!r}")
        expect("Unknown anchor preset '@missing'" in stderr_txt, f"expected unknown preset guidance. stderr={stderr_txt[:500]!r}")


def test_hook_on_add_anchor_composed_preset_resolves_from_config():
    """on-add should allow presets to compose with normal anchor filters."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        conf = Path(td) / "config-nautical.toml"
        conf.write_text('[anchor_presets]\nworkout = "w:mon,wed,fri"\n', encoding="utf-8")
        env = {"NO_COLOR": "1", "NAUTICAL_CONFIG": str(conf)}
        task = {
            "uuid": "00000000-0000-4000-8000-000000000120",
            "description": "hook test on-add composed anchor preset",
            "status": "pending",
            "project": "testing",
            "entry": "20260401T000000Z",
            "anchor": "@workout + y:apr",
            "anchor_mode": "skip",
            "due": "20260401T090000Z",
        }
        p = _run_hook_script(hook, task, env_extra=env)
        if p.returncode != 0:
            raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:500]!r}")
        stderr_txt = _strip_markup(p.stderr)
        expect("Invalid anchor" not in stderr_txt, f"composed preset should validate cleanly: {stderr_txt[:500]!r}")
        expect("Natural" in stderr_txt and "Apr" in stderr_txt, f"composed preset should describe resolved expression: {stderr_txt[:500]!r}")


def test_hook_on_add_anchor_recursive_preset_fails_cleanly():
    """on-add should reject recursive preset definitions with a clear message."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        conf = Path(td) / "config-nautical.toml"
        conf.write_text('[anchor_presets]\na = "@b"\nb = "@a"\n', encoding="utf-8")
        env = {"NO_COLOR": "1", "NAUTICAL_CONFIG": str(conf)}
        task = {
            "uuid": "00000000-0000-4000-8000-000000000121",
            "description": "hook test on-add recursive anchor preset",
            "status": "pending",
            "project": "testing",
            "entry": "20260101T000000Z",
            "anchor": "@a",
            "anchor_mode": "skip",
            "due": "20260101T090000Z",
        }
        p = _run_hook_script(hook, task, env_extra=env)
        expect(p.returncode != 0, "on-add should fail for recursive anchor presets")
        expect((p.stdout or "").strip() == "", f"expected no stdout on recursive preset failure, got: {p.stdout!r}")
        stderr_txt = _strip_markup(p.stderr)
        expect("Invalid Nautical configuration" in stderr_txt, f"expected invalid config panel. stderr={stderr_txt[:500]!r}")
        expect(
            "Recursive" in stderr_txt and "anchor preset reference detected" in stderr_txt,
            f"expected recursive preset guidance. stderr={stderr_txt[:500]!r}",
        )


def test_hook_on_add_omit_preset_resolves_from_config():
    """on-add should resolve @omit presets from config before omit validation/preview."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        conf = Path(td) / "config-nautical.toml"
        conf.write_text('[omit_presets]\napril = "y:apr"\n', encoding="utf-8")
        env = {"NO_COLOR": "1", "NAUTICAL_CONFIG": str(conf)}
        task = {
            "uuid": "00000000-0000-4000-8000-000000000122",
            "description": "hook test on-add omit preset",
            "status": "pending",
            "project": "testing",
            "entry": "20260301T000000Z",
            "anchor": "w:mon",
            "omit": "@april",
            "anchor_mode": "skip",
            "due": "20260302T090000Z",
        }
        p = _run_hook_script(hook, task, env_extra=env)
        if p.returncode != 0:
            raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:500]!r}")
        out_task = _extract_last_json(p.stdout)
        expect(out_task.get("omit") == "@april", f"omit preset expression should be preserved: {out_task}")
        stderr_txt = _strip_markup(p.stderr)
        expect("Invalid omit" not in stderr_txt, f"omit preset should validate cleanly: {stderr_txt[:500]!r}")
        expect("Omit preset" in stderr_txt and "@april → y:apr" in stderr_txt, f"omit preset preview should show expansion: {stderr_txt[:500]!r}")
        expect("Except" in stderr_txt and "Apr" in stderr_txt, f"omit preset preview should describe resolved omit: {stderr_txt[:500]!r}")


def test_hook_on_add_omit_unknown_preset_fails_cleanly():
    """on-add should fail clearly when an omit preset is not configured."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        conf = Path(td) / "config-nautical.toml"
        conf.write_text("[omit_presets]\n", encoding="utf-8")
        env = {"NO_COLOR": "1", "NAUTICAL_CONFIG": str(conf)}
        task = {
            "uuid": "00000000-0000-4000-8000-000000000123",
            "description": "hook test on-add unknown omit preset",
            "status": "pending",
            "project": "testing",
            "entry": "20260301T000000Z",
            "anchor": "w:mon",
            "omit": "@missing",
            "anchor_mode": "skip",
            "due": "20260302T090000Z",
        }
        p = _run_hook_script(hook, task, env_extra=env)
        expect(p.returncode != 0, "on-add should fail for unknown omit preset")
        expect((p.stdout or "").strip() == "", f"expected no stdout on unknown omit preset failure, got: {p.stdout!r}")
        stderr_txt = _strip_markup(p.stderr)
        expect("Invalid omit" in stderr_txt, f"expected invalid omit panel. stderr={stderr_txt[:500]!r}")
        expect("Unknown omit preset '@missing'" in stderr_txt, f"expected unknown omit preset guidance. stderr={stderr_txt[:500]!r}")


def test_hook_on_add_omit_recursive_preset_fails_cleanly():
    """on-add should reject recursive omit preset definitions with a clear message."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        conf = Path(td) / "config-nautical.toml"
        conf.write_text('[omit_presets]\na = "@b"\nb = "@a"\n', encoding="utf-8")
        env = {"NO_COLOR": "1", "NAUTICAL_CONFIG": str(conf)}
        task = {
            "uuid": "00000000-0000-4000-8000-000000000124",
            "description": "hook test on-add recursive omit preset",
            "status": "pending",
            "project": "testing",
            "entry": "20260301T000000Z",
            "anchor": "w:mon",
            "omit": "@a",
            "anchor_mode": "skip",
            "due": "20260302T090000Z",
        }
        p = _run_hook_script(hook, task, env_extra=env)
        expect(p.returncode != 0, "on-add should fail for recursive omit presets")
        expect((p.stdout or "").strip() == "", f"expected no stdout on recursive omit preset failure, got: {p.stdout!r}")
        stderr_txt = _strip_markup(p.stderr)
        expect("Invalid Nautical configuration" in stderr_txt, f"expected invalid config panel. stderr={stderr_txt[:500]!r}")
        expect(
            "Recursive omit" in stderr_txt and "preset reference detected" in stderr_txt,
            f"expected recursive omit preset guidance. stderr={stderr_txt[:500]!r}",
        )


def test_hook_on_add_omit_timed_preset_rejected():
    """omit presets should remain date-based and reject timed expressions."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        conf = Path(td) / "config-nautical.toml"
        conf.write_text('[omit_presets]\ntimed = "w:mon@t=09:00"\n', encoding="utf-8")
        env = {"NO_COLOR": "1", "NAUTICAL_CONFIG": str(conf)}
        task = {
            "uuid": "00000000-0000-4000-8000-000000000125",
            "description": "hook test on-add timed omit preset",
            "status": "pending",
            "project": "testing",
            "entry": "20260301T000000Z",
            "anchor": "w:mon",
            "omit": "@timed",
            "anchor_mode": "skip",
            "due": "20260302T090000Z",
        }
        p = _run_hook_script(hook, task, env_extra=env)
        expect(p.returncode != 0, "on-add should fail for timed omit preset")
        expect((p.stdout or "").strip() == "", f"expected no stdout on timed omit preset failure, got: {p.stdout!r}")
        stderr_txt = _strip_markup(p.stderr)
        expect("Invalid Nautical configuration" in stderr_txt, f"expected invalid config panel. stderr={stderr_txt[:500]!r}")
        expect(
            "omit does not" in stderr_txt and "support time modifiers" in stderr_txt,
            f"expected timed omit guidance. stderr={stderr_txt[:500]!r}",
        )


def test_hook_on_add_cp_malformed_inputs_fail_with_parser_guidance():
    """on-add should surface parser-specific guidance for malformed cp strings."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    cases = [
        ("rand(7d..3d)", ("lower", "bound", "<=", "upper")),
        ("rand(3d-7d)", ("expected", "rand(<duration>..<duration>)")),
        ("14d~abc", ("invalid", "duration", "bound")),
        ("2d~3d", ("lower", "bound", ">= 0")),
        ("3d,,7d", ("empty", "duration", "position 2")),
    ]
    for idx, (cp_value, expected_parts) in enumerate(cases, start=1):
        task = {
            "uuid": f"00000000-0000-4000-8000-00000000{130 + idx:04d}",
            "description": f"hook test malformed cp add {idx}",
            "status": "pending",
            "project": "testing",
            "entry": "20260101T000000Z",
            "cp": cp_value,
            "due": "20260101T090000Z",
        }
        p = _run_hook_script(hook, task, env_extra=env)
        expect(p.returncode != 0, f"on-add should fail for malformed cp {cp_value!r}")
        expect((p.stdout or "").strip() == "", f"expected no stdout on malformed cp add failure, got: {p.stdout!r}")
        stderr_txt = _strip_markup(p.stderr)
        expect("Invalid cp" in stderr_txt or "Invalid CP" in stderr_txt, f"expected invalid cp panel for {cp_value!r}: {stderr_txt[:500]!r}")
        for part in expected_parts:
            expect(part in stderr_txt, f"expected parser guidance fragment {part!r} for {cp_value!r}: {stderr_txt[:500]!r}")


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


def test_hook_on_add_cp_sequence_preview_accepts_string_periods():
    """on-add should accept comma-separated cp sequences now that cp is string-backed."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    task = {
        "uuid": "00000000-0000-4000-8000-000000000114",
        "description": "hook test on-add cp sequence",
        "status": "pending",
        "project": "testing",
        "entry": "20260101T000000Z",
        "cp": "3d,20d,7d",
        "due": "20260101T090000Z",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    if p.returncode != 0:
        raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:400]!r}")
    out_task = _extract_last_json(p.stdout)
    expect(out_task.get("cp") == "3d,20d,7d", f"cp sequence should be preserved as string: {out_task}")
    stderr_txt = _strip_markup(p.stderr)
    expect("Period" in stderr_txt and "3d,20d,7d" in stderr_txt, f"preview should show cp sequence: {stderr_txt[:500]!r}")
    expect("Step" in stderr_txt and "1/3 (3d)" in stderr_txt, f"preview should show sequence step and period: {stderr_txt[:500]!r}")
    for token in ("(3d)", "(20d)", "(7d)"):
        expect(token in stderr_txt, f"preview upcoming timeline should show sequence interval {token}: {stderr_txt[:500]!r}")


def test_hook_on_add_cp_random_preview_shows_selected_periods():
    """on-add random cp previews should show selected intervals, not reuse the raw rand expression."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    task = {
        "uuid": "00000000-0000-4000-8000-000000000115",
        "description": "hook test on-add cp random",
        "status": "pending",
        "project": "testing",
        "entry": "20260101T000000Z",
        "cp": "rand(15d..15d)",
        "due": "20260101T090000Z",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    if p.returncode != 0:
        raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:400]!r}")
    out_task = _extract_last_json(p.stdout)
    expect(out_task.get("cp") == "rand(15d..15d)", f"random cp should be preserved as string: {out_task}")
    stderr_txt = _strip_markup(p.stderr)
    expect("Step" in stderr_txt and "1/1" in stderr_txt, f"random cp preview should show selected step: {stderr_txt[:500]!r}")
    expect("(15d)" in stderr_txt, f"random cp preview should show whole-day picks as days: {stderr_txt[:500]!r}")
    expect("2w1d" not in stderr_txt, f"random cp preview should not show composite week/day picks: {stderr_txt[:500]!r}")
    expect("Upcoming" in stderr_txt, f"random cp preview should show upcoming dates: {stderr_txt[:500]!r}")
    expect("(rand(" not in stderr_txt, f"random cp preview should show selected durations, not raw rand tokens: {stderr_txt[:500]!r}")


def test_hook_on_add_cp_random_preview_uses_stamped_chain_id():
    """on-add should scope random previews to the chain ID stamped on the new root task."""
    hook = _find_hook_file("on-add.nautical")
    cp = "rand(11d..14d)"
    chain_id = "12345678"
    task = {
        "uuid": "12345678-0000-0000-0000-000000000115",
        "description": "chain-scoped random preview",
        "status": "pending",
        "entry": "20260101T000000Z",
        "cp": cp,
        "due": "20260101T090000Z",
    }
    proc = _run_hook_script(hook, task, env_extra={"NO_COLOR": "1"})
    expect(proc.returncode == 0, f"on-add chain-scoped random preview failed: {proc.stderr[:500]!r}")
    out_task = _assert_stdout_json_only(proc.stdout)
    expect(out_task.get("chainID") == chain_id, f"unexpected stamped chain ID: {out_task!r}")

    selected = core.cp_sequence_interval_for_link(cp, 1, chain_id)
    other_chain = core.cp_sequence_interval_for_link(cp, 1, "chain-b")
    expect(selected != other_chain, "test chains must have distinct selections at the previewed link")
    selected_days = int(selected.total_seconds() // 86400)
    stderr_txt = _strip_markup(proc.stderr)
    expect(
        f"({selected_days}d)" in stderr_txt,
        f"preview did not use the stamped chain ID selection: {stderr_txt[:500]!r}",
    )


def test_hook_on_add_cp_random_malformed_fails_with_guidance():
    """on-add should surface clear rand syntax guidance for malformed random cp ranges."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    task = {
        "uuid": "00000000-0000-4000-8000-000000000116",
        "description": "hook test on-add malformed cp random",
        "status": "pending",
        "project": "testing",
        "entry": "20260101T000000Z",
        "cp": "rand(3d-7d)",
        "due": "20260101T090000Z",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    expect(p.returncode != 0, "on-add should fail for malformed random cp")
    expect((p.stdout or "").strip() == "", f"expected no stdout on malformed random cp failure, got: {p.stdout!r}")
    stderr_txt = _strip_markup(p.stderr)
    expect("Invalid cp" in stderr_txt, f"expected invalid cp panel. stderr={stderr_txt[:500]!r}")
    expect("expected rand(<duration>..<duration>)" in stderr_txt, f"expected rand syntax guidance. stderr={stderr_txt[:500]!r}")


def test_hook_on_add_cp_jitter_preview_shows_selected_periods():
    """on-add jitter cp previews should show selected intervals, not the raw jitter expression."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    task = {
        "uuid": "00000000-0000-4000-8000-000000000117",
        "description": "hook test on-add cp jitter",
        "status": "pending",
        "project": "testing",
        "entry": "20260101T000000Z",
        "cp": "15d~0d",
        "due": "20260101T090000Z",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    if p.returncode != 0:
        raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:400]!r}")
    out_task = _extract_last_json(p.stdout)
    expect(out_task.get("cp") == "15d~0d", f"jitter cp should be preserved as string: {out_task}")
    stderr_txt = _strip_markup(p.stderr)
    expect("Step" in stderr_txt and "1/1 (15d)" in stderr_txt, f"jitter cp preview should show selected step: {stderr_txt[:500]!r}")
    expect("15d~0d" in stderr_txt, f"jitter cp preview should still show original period expression: {stderr_txt[:500]!r}")


def test_hook_on_add_anchor_scheduled_only_preserves_no_due():
    """scheduled-only anchor tasks should remain scheduled-only on add."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    task = {
        "uuid": "00000000-0000-4000-8000-000000000113",
        "description": "hook test on-add anchor scheduled-only",
        "status": "pending",
        "project": "testing",
        "entry": "20251217T000000Z",
        "anchor": "w:wed",
        "anchor_mode": "skip",
        "scheduled": "20251217T090000Z",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    if p.returncode != 0:
        raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:400]!r}")
    out_task = _extract_last_json(p.stdout)
    expect(not out_task.get("due"), f"scheduled-only anchor add should not set due: {out_task}")
    expect(out_task.get("scheduled") == task["scheduled"], f"scheduled changed unexpectedly: {out_task}")
    stderr_txt = _strip_markup(p.stderr)
    expect("First scheduled" in stderr_txt, f"preview should label scheduled anchor. stderr={stderr_txt[:500]!r}")


def test_on_add_native_until_requires_strictly_later_target():
    """Nautical additions should reject until at or before due/scheduled."""
    hook = _find_hook_file("on-add.nautical")
    cases = (
        ("due", "20260801T090000Z", "20260801T085959Z"),
        ("due", "20260801T090000Z", "20260801T090000Z"),
        ("scheduled", "20260801T090000Z", "20260801T090000Z"),
    )
    for index, (target_field, target, until) in enumerate(cases):
        task = {
            "uuid": f"00000000-0000-4000-8000-00000000012{index}",
            "description": f"invalid native until {target_field}",
            "status": "pending",
            "entry": "20260720T090000Z",
            "cp": "7d",
            target_field: target,
            "until": until,
        }
        proc = _run_hook_script(hook, task, env_extra={"NO_COLOR": "1"})
        expect(proc.returncode != 0, f"invalid {target_field}/until ordering was accepted: {task!r}")
        expect(not (proc.stdout or "").strip(), f"rejected add leaked stdout: {proc.stdout!r}")
        stderr_txt = _strip_markup(proc.stderr)
        label = "Scheduled" if target_field == "scheduled" else "Due"
        expect("Invalid expiration window" in stderr_txt, f"missing expiration guard panel: {stderr_txt!r}")
        expect(label in stderr_txt and "Expires" in stderr_txt, f"missing compared timestamps: {stderr_txt!r}")
        expect(
            f"until must be later than {target_field}" in stderr_txt,
            f"missing ordering guidance: {stderr_txt!r}",
        )

    valid = {
        "uuid": "00000000-0000-4000-8000-000000000129",
        "description": "valid native until window",
        "status": "pending",
        "entry": "20260720T090000Z",
        "cp": "7d",
        "due": "20260801T090000Z",
        "until": "20260801T090001Z",
    }
    proc = _run_hook_script(hook, valid, env_extra={"NO_COLOR": "1"})
    expect(proc.returncode == 0, f"strictly later until was rejected: {proc.stderr!r}")
    expect(_assert_stdout_json_only(proc.stdout).get("until") == valid["until"], "valid until changed")


def test_on_add_native_until_checks_generated_cp_due():
    """The expiration guard should run after CP assigns its automatic first due."""
    hook = _find_hook_file("on-add.nautical")
    task = {
        "uuid": "00000000-0000-4000-8000-000000000130",
        "description": "invalid until before generated CP due",
        "status": "pending",
        "entry": "20260801T090000Z",
        "cp": "7d",
        "until": "20260808T085959Z",
    }
    proc = _run_hook_script(hook, task, env_extra={"NO_COLOR": "1"})
    expect(proc.returncode != 0, "until before generated CP due was accepted")
    expect(not (proc.stdout or "").strip(), f"rejected generated CP due leaked stdout: {proc.stdout!r}")
    stderr_txt = _strip_markup(proc.stderr)
    expect(
        "Invalid expiration window" in stderr_txt
        and "Due" in stderr_txt
        and "Expires" in stderr_txt
        and "until must be later than due" in stderr_txt,
        stderr_txt,
    )


def test_on_add_preview_distinguishes_expiration_from_chain_end_point():
    """Add previews should distinguish native expiration from chain boundaries."""
    hook = _find_hook_file("on-add.nautical")
    base = {
        "status": "pending",
        "entry": "20260720T090000Z",
        "due": "20260803T100000Z",
        "until": "20260803T180000Z",
        # Keep the chain bound in the future so this fixture remains valid
        # as the calendar advances; chainMax still determines the last date.
        "chainUntil": "20991231T210000Z",
        "chainMax": 3,
    }
    cases = (
        dict(base, uuid="00000000-0000-4000-8000-000000000141", description="CP expiry preview", cp="7d"),
        dict(
            base,
            uuid="00000000-0000-4000-8000-000000000142",
            description="Anchor expiry preview",
            anchor="w:mon",
            anchor_mode="skip",
        ),
    )
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config-nautical.toml"
        config_path.write_text('tz = "UTC"\n', encoding="utf-8")
        for task in cases:
            proc = _run_hook_script(
                hook,
                task,
                env_extra={
                    "NO_COLOR": "1",
                    "NAUTICAL_CONFIG": str(config_path),
                    "NAUTICAL_TRUST_CONFIG_PATH": "1",
                },
            )
            expect(proc.returncode == 0, f"preview failed: {proc.stderr!r}")
            expect(_assert_stdout_json_only(proc.stdout).get("until") == task["until"], "native until changed")
            panel = _strip_markup(proc.stderr)
            for label in ("Expiration", "First expires", "Chain end point", "Last occurrence", "Future links"):
                expect(label in panel, f"{label!r} missing from preview: {panel!r}")
            expected_policy = "Same day at 18:00"
            expect(expected_policy in panel, f"calendar expiration policy missing from preview: {panel!r}")
            expect("2026-08-17" in panel, f"chainMax should determine the effective last occurrence: {panel!r}")
            expect("Final (until)" not in panel, f"ambiguous legacy label remains: {panel!r}")


def test_on_add_preview_fails_closed_when_evaluator_initialization_fails():
    """A shared evaluator failure must never fall back to legacy scheduling callbacks."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_evaluator_failure_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    now_utc = mod.core.build_local_datetime(date(2026, 4, 12), (12, 0)).astimezone(timezone.utc)
    task = {
        "uuid": "00000000-0000-4000-8000-000000000143",
        "description": "evaluator initialization failure",
        "status": "pending",
        "entry": mod.core.fmt_isoz(now_utc),
        "anchor": "w:mon",
        "anchor_mode": "skip",
        "chain": "on",
        "chainID": "00000000",
    }
    ctx = mod._module("add_composition").build_on_add_context(mod, task, now_utc, mod.core.to_local(now_utc))
    panels = []
    service_cls = importlib.import_module("nautical_core.scheduler_service").SchedulerService
    original_from_task = service_cls.__dict__["from_task"]
    def fail_from_task(cls, *args, **kwargs):
        raise RuntimeError("astronomy profile is unavailable")

    try:
        service_cls.from_task = classmethod(fail_from_task)
        mod._panel = lambda title, rows, **kwargs: panels.append((title, list(rows), kwargs))
        try:
            mod._module("add_composition").render_anchor_preview(mod, ctx, prof=mod._NoopProfiler())
        except SystemExit as exc:
            expect(exc.code == 1, f"unexpected evaluator failure exit code: {exc.code!r}")
        else:
            raise AssertionError("evaluator initialization failure was accepted")
    finally:
        service_cls.from_task = original_from_task

    expect(panels and panels[-1][0] == "❌ Invalid Chain", f"missing evaluator error panel: {panels!r}")
    rows = panels[-1][1]
    expect(any(label == "Recurrence evaluator" for label, _value in rows), f"missing evaluator error detail: {rows!r}")
    expect(any(label == "Fix" for label, _value in rows), f"missing evaluator remediation: {rows!r}")


def test_on_add_preview_reports_scheduler_exhaustion_actionably():
    """Scheduler exhaustion should become a clear panel, not a generic hook crash."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_scheduler_exhaustion_test")
    now_utc = mod.core.build_local_datetime(date(2026, 4, 12), (12, 0)).astimezone(timezone.utc)
    task = {
        "uuid": "00000000-0000-4000-8000-000000000144",
        "description": "scheduler exhaustion",
        "status": "pending",
        "entry": mod.core.fmt_isoz(now_utc),
        "anchor": "w:mon",
        "anchor_mode": "skip",
        "chain": "on",
        "chainID": "00000000",
    }
    ctx = mod._module("add_composition").build_on_add_context(mod, task, now_utc, mod.core.to_local(now_utc))
    panels = []
    preview = mod._module("add_anchor_preview")
    original_preview = preview.handle_anchor_preview_on_add
    expected = mod.core.OccurrenceSearchExhausted(
        "test preview", reference=date(2026, 4, 12), limit=1
    )

    def fail_preview(**_kwargs):
        raise expected

    try:
        preview.handle_anchor_preview_on_add = fail_preview
        mod._panel = lambda title, rows, **kwargs: panels.append((title, list(rows), kwargs))
        try:
            mod._module("add_composition").render_anchor_preview(mod, ctx, prof=mod._NoopProfiler())
        except SystemExit as exc:
            expect(exc.code == 1, f"unexpected scheduler exhaustion exit code: {exc.code!r}")
        else:
            raise AssertionError("scheduler exhaustion was accepted")
    finally:
        preview.handle_anchor_preview_on_add = original_preview

    expect(panels and panels[-1][0] == "❌ Invalid Chain", f"missing scheduler error panel: {panels!r}")
    rows = panels[-1][1]
    expect(any(label == "Scheduler" and "test preview" in value for label, value in rows), rows)
    expect(any(label == "Fix" and "less sparse" in value for label, value in rows), rows)


def test_on_add_preview_uses_evaluator_for_first_due_and_upcoming_rows():
    """Normal anchor previews must not invoke the legacy occurrence callbacks."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_evaluator_scheduler_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    now_utc = mod.core.build_local_datetime(date(2026, 4, 12), (12, 0)).astimezone(timezone.utc)
    task = {
        "uuid": "00000000-0000-4000-8000-000000000144",
        "description": "evaluator scheduler preview",
        "status": "pending",
        "entry": mod.core.fmt_isoz(now_utc),
        "anchor": "w:mon",
        "anchor_mode": "skip",
        "chain": "on",
        "chainID": "00000000",
    }
    ctx = mod._module("add_composition").build_on_add_context(mod, task, now_utc, mod.core.to_local(now_utc))
    captured = {}
    original = (mod._fmt_local_for_task, mod._panel)
    try:
        mod._fmt_local_for_task = mod.core.fmt_isoz
        mod._panel = lambda title, rows, **kwargs: captured.update({"title": title, "rows": list(rows)})
        mod._module("add_composition").render_anchor_preview(mod, ctx, prof=mod._NoopProfiler())
    finally:
        mod._fmt_local_for_task, mod._panel = original

    expect(task.get("due"), f"evaluator preview did not assign due: {captured!r}")
    expect(captured.get("title") == "⚓︎ Anchor Preview", f"evaluator preview did not render: {captured!r}")


def test_on_add_native_until_checks_generated_anchor_due():
    """The expiration guard should run after an anchor resolves its automatic first due."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_generated_anchor_until_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    now_utc = mod.core.build_local_datetime(date(2026, 4, 12), (12, 0)).astimezone(timezone.utc)
    first_due = mod.core.build_local_datetime(date(2026, 4, 13), (9, 0)).astimezone(timezone.utc)
    task = {
        "uuid": "00000000-0000-4000-8000-000000000131",
        "description": "invalid until before generated anchor due",
        "status": "pending",
        "entry": mod.core.fmt_isoz(now_utc),
        "anchor": "w:mon",
        "chain": "on",
        "chainID": "generated131",
        "link": 1,
        "until": mod.core.fmt_isoz(first_due - timedelta(seconds=1)),
    }
    ctx = mod._module("add_composition").build_on_add_context(mod, task, now_utc, mod.core.to_local(now_utc))
    panels = []
    original = mod._panel
    try:
        mod._panel = lambda title, rows, **kwargs: panels.append((title, list(rows), kwargs))
        try:
            mod._module("add_composition").render_anchor_preview(mod, ctx, prof=mod._NoopProfiler())
        except SystemExit as exc:
            expect(exc.code == 1, f"unexpected generated-anchor rejection code: {exc.code!r}")
        else:
            raise AssertionError("until before generated anchor due was accepted")
    finally:
        mod._panel = original

    expect(task.get("due"), f"anchor target was not finalized before validation: {task!r}")
    expect(panels and "Invalid expiration window" in panels[-1][0], f"missing guard panel: {panels!r}")


def test_on_add_native_until_guard_ignores_ordinary_tasks():
    """Nautical should not impose its expiration ordering on ordinary Taskwarrior tasks."""
    hook = _find_hook_file("on-add.nautical")
    task = {
        "uuid": "00000000-0000-4000-8000-000000000132",
        "description": "ordinary task with independent until",
        "status": "pending",
        "entry": "20260720T090000Z",
        "due": "20260801T090000Z",
        "until": "20260731T090000Z",
    }
    proc = _run_hook_script(hook, task, env_extra={"NO_COLOR": "1"})
    expect(proc.returncode == 0, f"ordinary task was rejected: {proc.stderr!r}")
    expect(_assert_stdout_json_only(proc.stdout) == task, "ordinary task was changed")


def test_on_add_chain_until_rejects_before_first_anchor_occurrence():
    """An auto-due anchor must not be accepted when chainUntil precedes its first match."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_chain_until_first_match_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()
    now_utc = mod.core.build_local_datetime(date(2026, 4, 12), (12, 0)).astimezone(timezone.utc)
    task = {
        "uuid": "00000000-0000-4000-8000-000000000133",
        "description": "chain endpoint before first anchor",
        "status": "pending",
        "entry": mod.core.fmt_isoz(now_utc),
        "anchor": "w:mon",
        "chain": "on",
        "chainID": "firstmatch133",
        "link": 1,
        "chainUntil": mod.core.fmt_isoz(now_utc + timedelta(hours=12)),
    }
    ctx = mod._module("add_composition").build_on_add_context(mod, task, now_utc, mod.core.to_local(now_utc))
    panels = []
    original = mod._panel
    try:
        mod._panel = lambda title, rows, **kwargs: panels.append((title, list(rows), kwargs))
        try:
            mod._module("add_composition").render_anchor_preview(mod, ctx, prof=mod._NoopProfiler())
        except SystemExit as exc:
            expect(exc.code == 1, f"unexpected chainUntil rejection code: {exc.code!r}")
        else:
            raise AssertionError("chainUntil before first anchor occurrence was accepted")
    finally:
        mod._panel = original
    expect(
        panels and any(label == "Invalid chainUntil" for label, _value in panels[-1][1]),
        f"missing chainUntil guard panel: {panels!r}",
    )


def test_on_add_native_until_rejects_strict_anchor_modes():
    """Native until should be incompatible with all and flex anchor backfill."""
    hook = _find_hook_file("on-add.nautical")
    base = {
        "uuid": "00000000-0000-4000-8000-000000000137",
        "description": "strict anchor expiration conflict",
        "status": "pending",
        "entry": "20260720T090000Z",
        "anchor": "w:mon",
        "due": "20260803T090000Z",
        "until": "20260804T090000Z",
    }
    for mode in ("all", "flex"):
        proc = _run_hook_script(hook, dict(base, anchor_mode=mode), env_extra={"NO_COLOR": "1"})
        expect(proc.returncode != 0, f"anchor_mode:{mode} accepted native until")
        expect(not (proc.stdout or "").strip(), f"rejected anchor mode leaked stdout: {proc.stdout!r}")
        stderr_txt = _strip_markup(proc.stderr)
        expect("Invalid expiration mode" in stderr_txt, f"missing mode conflict panel: {stderr_txt!r}")
        expect("Remove until or use anchor_mode:skip" in stderr_txt, f"missing mode resolution: {stderr_txt!r}")

    proc = _run_hook_script(hook, dict(base, anchor_mode="skip"), env_extra={"NO_COLOR": "1"})
    expect(proc.returncode == 0, f"anchor_mode:skip rejected valid native until: {proc.stderr!r}")
    expect(_assert_stdout_json_only(proc.stdout).get("anchor_mode") == "skip", "skip mode changed")

    validation = core._import_sibling("add_validation")
    valid, reason = validation.validate_native_until_anchor_mode(
        base["until"],
        "",
        "events.csv",
        "all",
    )
    expect(not valid and "anchor_mode:all" in str(reason), f"anchor_file all conflict was missed: {reason!r}")


def test_on_modify_native_until_rejects_invalid_window_changes():
    """Nautical modifications should reject target windows made invalid."""
    hook = _find_hook_file("on-modify.nautical")
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
            proc = _run_hook_script_raw(
                hook,
                json.dumps(base) + "\n" + json.dumps(new),
                env_extra={"NO_COLOR": "1", "TASKDATA": td},
            )
            expect(proc.returncode != 0, f"invalid modified expiration window was accepted: {new!r}")
            expect(not (proc.stdout or "").strip(), f"rejected modification leaked stdout: {proc.stdout!r}")
            stderr_txt = _strip_markup(proc.stderr)
            expect("Invalid expiration window" in stderr_txt, f"missing modification guard panel: {stderr_txt!r}")
            expect("until must be later than" in stderr_txt, f"missing modification guidance: {stderr_txt!r}")


def test_on_modify_native_until_follows_recurrence_target_move():
    """An untouched native until should follow a rescheduled recurrence target."""
    hook = _find_hook_file("on-modify.nautical")
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
            proc = _run_hook_script_raw(
                hook,
                json.dumps(old) + "\n" + json.dumps(new),
                env_extra={"NO_COLOR": "1", "TASKDATA": td},
            )
            expect(proc.returncode == 0, f"rescheduled expiration window was rejected: {proc.stderr!r}")
            result = _assert_stdout_json_only(proc.stdout)
            expect(result.get("until") == expected_until, f"until did not follow recurrence target: {result!r}")
            if idx == 0:
                panel = _strip_markup(proc.stderr)
                # The typed lifecycle path may legitimately suppress panels in
                # non-interactive hook execution; when emitted, retain the
                # semantic-content assertion.
                if panel:
                    expect("Nautical recurrence updated" in panel, f"unexpected expiration panel: {panel!r}")
                    expect("Expiration" in panel and "Carry" in panel, f"expiration carry was not explained: {panel!r}")


def test_native_until_shared_policy_covers_recurrence_kinds_and_conflicts():
    """The shared expiration policy should cover every recurrence kind with typed conflicts."""
    import nautical_core.native_until as native_until

    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_native_until_shared_policy_test")
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
    hook = _find_hook_file("on-modify.nautical")
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
        proc = _run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra={"NO_COLOR": "1", "TASKDATA": td},
        )
    expect(proc.returncode != 0, "uncarryable anchor target move retained a stale absolute until")
    expect(not (proc.stdout or "").strip(), f"rejected target move leaked stdout: {proc.stdout!r}")
    panel = _strip_markup(proc.stderr)
    expect("Invalid expiration window" in panel and "Carry" in panel, f"missing carry conflict panel: {panel!r}")


def test_on_modify_completion_reschedule_carries_native_until():
    """Completion and target rescheduling in one modify should retain expiration policy."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_completion_reschedule_until_test")
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
            _modify_effect(mod, "handle_completion", old, new, _test_operator_uow())
            expect(new.get("until") == expected_until, f"completion reschedule lost expiration carry: {new!r}")
    finally:
        mod._completion_effects.preflight_context = original_preflight


def test_on_modify_native_until_accepts_valid_window_change():
    """A modified until that remains after the target should pass through normally."""
    hook = _find_hook_file("on-modify.nautical")
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
        proc = _run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra={"NO_COLOR": "1", "TASKDATA": td},
        )
    expect(proc.returncode == 0, f"valid modified expiration window was rejected: {proc.stderr!r}")
    expect(_assert_stdout_json_only(proc.stdout).get("due") == new["due"], "valid due modification changed")


def test_on_modify_native_until_validates_recurrence_promotion():
    """Adding Nautical recurrence should validate an existing native until window."""
    hook = _find_hook_file("on-modify.nautical")
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
        proc = _run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra={"NO_COLOR": "1", "TASKDATA": td},
        )
    expect(proc.returncode != 0, "recurrence promotion accepted an invalid expiration window")
    expect(not (proc.stdout or "").strip(), f"rejected recurrence promotion leaked stdout: {proc.stdout!r}")
    expect("Invalid expiration window" in _strip_markup(proc.stderr), f"missing promotion guard: {proc.stderr!r}")


def test_on_modify_native_until_validates_simultaneous_completion():
    """Completion should not queue a child from an invalid newly modified window."""
    hook = _find_hook_file("on-modify.nautical")
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
        proc = _run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra={"NO_COLOR": "1", "TASKDATA": td},
        )
    expect(proc.returncode != 0, "simultaneous completion accepted an invalid expiration window")
    expect(not (proc.stdout or "").strip(), f"rejected completion leaked stdout: {proc.stdout!r}")
    expect("Invalid expiration window" in _strip_markup(proc.stderr), f"missing completion guard: {proc.stderr!r}")


def test_on_modify_native_until_rejects_strict_anchor_mode_changes():
    """Changing an expiring anchor task to all or flex should be rejected."""
    hook = _find_hook_file("on-modify.nautical")
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
            proc = _run_hook_script_raw(
                hook,
                json.dumps(old) + "\n" + json.dumps(new),
                env_extra={"NO_COLOR": "1", "TASKDATA": td},
            )
            expect(proc.returncode != 0, f"anchor_mode:{mode} modification accepted native until")
            expect(not (proc.stdout or "").strip(), f"rejected mode modification leaked stdout: {proc.stdout!r}")
            expect("Invalid expiration mode" in _strip_markup(proc.stderr), f"missing mode guard: {proc.stderr!r}")


def test_on_modify_native_until_rejects_legacy_all_completion():
    """Completion should not perpetuate a legacy all-plus-until configuration."""
    hook = _find_hook_file("on-modify.nautical")
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
        proc = _run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra={"NO_COLOR": "1", "TASKDATA": td},
        )
    expect(proc.returncode != 0, "legacy anchor_mode:all completion perpetuated native until")
    expect(not (proc.stdout or "").strip(), f"rejected legacy completion leaked stdout: {proc.stdout!r}")
    expect("Invalid expiration mode" in _strip_markup(proc.stderr), f"missing completion mode guard: {proc.stderr!r}")


def test_on_add_due_context_treats_due_matching_entry_as_implicit():
    """on-add should not treat due==entry as an explicit anchor due."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_due_context_entry_due_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    now_utc = mod.core.parse_dt_any("20260412T111500Z")
    task = {
        "uuid": "00000000-0000-4000-8000-000000000113a",
        "description": "hook test on-add implicit entry due context",
        "status": "pending",
        "entry": "20260412T111500Z",
        "due": "20260412T111500Z",
        "anchor": "w:mon,wed,fri",
    }
    user_provided_due, recurrence_field, due_dt, past_due_warning, due_day, due_hhmm = mod._due_context_on_add(task, now_utc)
    expect(not user_provided_due, f"due matching entry should be treated as implicit: {(user_provided_due, recurrence_field, due_dt)!r}")
    expect(recurrence_field == "due", f"unexpected recurrence field for implicit entry due: {recurrence_field!r}")
    expect(due_dt == now_utc, f"implicit entry due should fall back to now_utc context: {due_dt!r}")
    expect(past_due_warning is None, f"implicit entry due should not produce past-due warning: {past_due_warning!r}")
    expect(due_day == mod.core.to_local(now_utc).date(), f"unexpected implicit due day: {due_day!r}")
    expect(due_hhmm == (mod.core.to_local(now_utc).hour, mod.core.to_local(now_utc).minute), f"unexpected implicit due hhmm: {due_hhmm!r}")


def test_on_add_anchor_preview_auto_assigns_when_due_matches_entry():
    """on-add anchor preview should auto-assign first anchor when incoming due merely mirrors entry."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_anchor_entry_due_preview_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    task = {
        "uuid": "00000000-0000-4000-8000-000000000136",
        "description": "hook test on-add implicit entry due preview",
        "status": "pending",
        "entry": "20260412T111500Z",
        "due": "20260412T111500Z",
        "anchor": "w:mon,wed,fri",
        "anchor_mode": "skip",
        "chain": "on",
        "chainID": "entrydue136",
        "link": 1,
    }
    now_utc = mod.core.parse_dt_any(task["entry"])
    now_local = mod.core.to_local(now_utc)
    ctx = mod._module("add_composition").build_on_add_context(mod, task, now_utc, now_local)
    expect(not ctx.user_provided_due, f"build_on_add_context should treat due==entry as implicit: {ctx!r}")

    captured = {}
    orig_panel = mod._panel
    try:
        mod._panel = lambda title, rows, **_k: captured.update({"title": title, "rows": list(rows)})
        mod._module("add_composition").render_anchor_preview(mod, ctx, prof=mod._NoopProfiler())
    finally:
        mod._panel = orig_panel

    expected_due = mod._fmt_local_for_task(mod.core.build_local_datetime(date(2026, 4, 13), (9, 0)).astimezone(timezone.utc))
    expect(task.get("due") == expected_due, f"expected implicit entry due to auto-assign first anchor match: {task!r}")
    rows = captured.get("rows") or []
    labels = [label for label, _value in rows]
    expect("Next anchor" not in labels, f"auto-assigned first due should not render a separate next-anchor row: {rows!r}")
    expect("[auto-due]" in labels, f"expected auto-due row in anchor preview: {rows!r}")


def test_hook_on_add_anchor_preview_skips_omit_date():
    """on-add anchor preview should skip omitted dates when selecting the next anchor."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    task = {
        "uuid": "00000000-0000-4000-8000-000000000114",
        "description": "hook test on-add anchor omit preview",
        "status": "pending",
        "project": "testing",
        "entry": "20250108T000000Z",
        "anchor": "w:mon,wed,fri@t=09:00",
        "omit": "w:wed",
        "anchor_mode": "skip",
        "due": "20250108T090000Z",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    if p.returncode != 0:
        raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:400]!r}")
    out_task = _extract_last_json(p.stdout)
    expect(out_task.get("due") == task["due"], f"on-add changed explicit due: {out_task!r}")
    stderr_txt = _strip_markup(p.stderr)
    expect("Omit" in stderr_txt, f"expected omit row in preview. stderr={stderr_txt[:500]!r}")
    expect("Except" in stderr_txt, f"expected omit natural-language row in preview. stderr={stderr_txt[:500]!r}")
    expect("Wednesdays" in stderr_txt or "Wednesday" in stderr_txt, f"expected omit natural-language wording in preview. stderr={stderr_txt[:500]!r}")
    expect("2025-01-10" in stderr_txt, f"expected next anchor to skip Wednesday and show Friday. stderr={stderr_txt[:500]!r}")


def test_hook_on_add_anchor_preview_skips_omit_file_date():
    """on-add anchor preview should skip dates loaded from omit_file."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        omit_dir = Path(td) / "omit"
        omit_dir.mkdir()
        (omit_dir / "holidays.csv").write_text('date,description\n2025-01-10,Skip Friday\n', encoding='utf-8')
        conf = Path(td) / 'config-nautical.toml'
        conf.write_text(f'omit_file_dir = "{omit_dir}"\n', encoding='utf-8')
        env = {"NO_COLOR": "1", "NAUTICAL_CONFIG": str(conf)}
        task = {
            "uuid": "00000000-0000-4000-8000-000000000114a",
            "description": "hook test on-add anchor omit_file preview",
            "status": "pending",
            "project": "testing",
            "entry": "20250108T000000Z",
            "anchor": "w:mon,wed,fri@t=09:00",
            "omit_file": "holidays.csv",
            "anchor_mode": "skip",
            "due": "20250108T090000Z",
        }
        p = _run_hook_script(hook, task, env_extra=env)
        if p.returncode != 0:
            raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:400]!r}")
        out_task = _extract_last_json(p.stdout)
        expect(out_task.get("due") == task["due"], f"on-add changed explicit due: {out_task!r}")
        stderr_txt = _strip_markup(p.stderr)
        expect("Omit file" in stderr_txt, f"expected omit_file row in preview. stderr={stderr_txt[:500]!r}")
        expect("holidays.csv" in stderr_txt, f"expected omit file name in preview. stderr={stderr_txt[:500]!r}")
        expect("2025-01-13" in stderr_txt, f"expected next anchor to skip file-blocked Friday and show Monday. stderr={stderr_txt[:500]!r}")


def test_hook_on_add_anchor_preview_skips_omit_file_modifier_date():
    """on-add anchor preview should apply omit_file modifiers before skipping matching dates."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        omit_dir = Path(td) / "omit"
        omit_dir.mkdir()
        (omit_dir / "holidays.csv").write_text('date,description\n2026-04-25,Weekend holiday\n', encoding='utf-8')
        conf = Path(td) / 'config-nautical.toml'
        conf.write_text(f'omit_file_dir = "{omit_dir}"\n', encoding='utf-8')
        env = {"NO_COLOR": "1", "NAUTICAL_CONFIG": str(conf)}
        task = {
            "uuid": "00000000-0000-4000-8000-000000000114e",
            "description": "hook test on-add anchor omit_file modifier preview",
            "status": "pending",
            "project": "testing",
            "entry": "20260412T000000Z",
            "anchor": "y:04-25@nbd@t=09:00",
            "omit_file": "holidays.csv@nbd",
            "anchor_mode": "skip",
        }
        p = _run_hook_script(hook, task, env_extra=env)
        if p.returncode != 0:
            raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:400]!r}")
        stderr_txt = _strip_markup(p.stderr)
        expect("Mon 2027-04-26 09:00" in stderr_txt, f"expected rolled 2026 occurrence to be omitted and next year shown. stderr={stderr_txt[:500]!r}")
        expect("Mon 2026-04-27 09:00" not in stderr_txt, f"expected transformed omit_file date to be skipped. stderr={stderr_txt[:500]!r}")


def test_hook_on_add_anchor_preview_marks_omitted_future_slots():
    """on-add preview should skip omitted future anchor slots in Upcoming."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    task = {
        "uuid": "00000000-0000-4000-8000-000000000114g",
        "description": "hook test on-add omit upcoming",
        "status": "pending",
        "project": "testing",
        "entry": "20250108T000000Z",
        "anchor": "w:mon,wed,fri@t=09:00",
        "omit": "w:wed",
        "anchor_mode": "skip",
        "due": "20250108T090000Z",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    if p.returncode != 0:
        raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:400]!r}")
    stderr_txt = _strip_markup(p.stderr)
    expect("(omitted)" not in stderr_txt, f"on-add should not render omitted slots as upcoming: {stderr_txt[:700]!r}")
    expect(
        "2025-01-15" not in stderr_txt and "Wed 2025-01-15" not in stderr_txt,
        f"expected omitted Wednesday slot to be skipped in Upcoming: {stderr_txt[:700]!r}",
    )


def test_hook_on_add_anchor_preview_uses_omit_file_description_in_upcoming():
    """on-add preview should skip omit_file dates instead of rendering them as upcoming entries."""
    hook = _find_hook_file("on-add.nautical")
    with tempfile.TemporaryDirectory() as td:
        omit_dir = Path(td) / "omit"
        omit_dir.mkdir()
        (omit_dir / "holidays.csv").write_text('date,description\n2025-01-10,Company holiday blackout\n', encoding='utf-8')
        conf = Path(td) / 'config-nautical.toml'
        conf.write_text(f'omit_file_dir = "{omit_dir}"\n', encoding='utf-8')
        env = {"NO_COLOR": "1", "NAUTICAL_CONFIG": str(conf)}
        task = {
            "uuid": "00000000-0000-4000-8000-000000000114h",
            "description": "hook test on-add omit file upcoming desc",
            "status": "pending",
            "project": "testing",
            "entry": "20250108T000000Z",
            "anchor": "w:mon,wed,fri@t=09:00",
            "omit_file": "holidays.csv",
            "anchor_mode": "skip",
            "due": "20250108T090000Z",
        }
        p = _run_hook_script(hook, task, env_extra=env)
        if p.returncode != 0:
            raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:400]!r}")
        stderr_txt = _strip_markup(p.stderr)
        expect("Company holida..." not in stderr_txt, f"on-add should not render omitted descriptions: {stderr_txt[:700]!r}")
        expect(
            "Fri 2025-01-10" not in stderr_txt and "2025-01-10 09:00" not in stderr_txt,
            f"expected omitted omit_file date to be skipped in Upcoming: {stderr_txt[:700]!r}",
        )


def test_hook_on_add_anchor_preview_rolled_business_day_uses_timed_slot():
    """on-add preview should keep @t times when a yearly anchor rolls forward to the next business day."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    expr = "y:04-25@nbd@t=12:00,17:00"
    task = {
        "uuid": "00000000-0000-4000-8000-000000000114b",
        "description": "hook test on-add rolled timed business day",
        "status": "pending",
        "project": "testing",
        "entry": "20260412T111500Z",
        "anchor": expr,
        "anchor_mode": "skip",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    if p.returncode != 0:
        raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:400]!r}")
    stderr_txt = _strip_markup(p.stderr)
    dnf = core.validate_anchor_expr_strict(expr)
    today = core.to_local(core.now_utc()).date()
    first, _ = core.next_after_expr(dnf, today, default_seed=today, seed_base="preview-test")
    second, _ = core.next_after_expr(dnf, first, default_seed=today, seed_base="preview-test")
    expect(f"{first:%Y-%m-%d} 12:00" in stderr_txt, f"expected first due to use rolled timed slot. stderr={stderr_txt[:500]!r}")
    expect(f"{first:%Y-%m-%d} 17:00" in stderr_txt, f"expected later same-day second slot in preview. stderr={stderr_txt[:500]!r}")
    expect(f"{second:%Y-%m-%d} 12:00" in stderr_txt, f"expected next yearly rolled occurrence to keep timed slot. stderr={stderr_txt[:500]!r}")


def test_hook_on_add_anchor_preview_positive_day_offset_uses_timed_slot():
    """on-add preview should keep @t times when an anchor date is shifted forward by @+Nd."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    expr = "y:04-25@+10d@t=12:00"
    task = {
        "uuid": "00000000-0000-4000-8000-000000000114c",
        "description": "hook test on-add positive offset timed anchor",
        "status": "pending",
        "project": "testing",
        "entry": "20260412T111500Z",
        "anchor": expr,
        "anchor_mode": "skip",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    if p.returncode != 0:
        raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:400]!r}")
    stderr_txt = _strip_markup(p.stderr)
    dnf = core.validate_anchor_expr_strict(expr)
    today = core.to_local(core.now_utc()).date()
    first, _ = core.next_after_expr(dnf, today, default_seed=today, seed_base="preview-test")
    second, _ = core.next_after_expr(dnf, first, default_seed=today, seed_base="preview-test")
    expect(f"{first:%Y-%m-%d} 12:00" in stderr_txt, f"expected first due to use shifted timed slot. stderr={stderr_txt[:500]!r}")
    expect(f"{second:%Y-%m-%d} 12:00" in stderr_txt, f"expected next yearly shifted occurrence to keep timed slot. stderr={stderr_txt[:500]!r}")


def test_hook_on_add_anchor_preview_negative_day_offset_uses_timed_slot():
    """on-add preview should keep @t times when an anchor date is shifted earlier by @-Nd."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    expr = "y:04-25@-2d@t=12:00"
    task = {
        "uuid": "00000000-0000-4000-8000-000000000114d",
        "description": "hook test on-add negative offset timed anchor",
        "status": "pending",
        "project": "testing",
        "entry": "20260412T111500Z",
        "anchor": expr,
        "anchor_mode": "skip",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    if p.returncode != 0:
        raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:400]!r}")
    stderr_txt = _strip_markup(p.stderr)
    dnf = core.validate_anchor_expr_strict(expr)
    today = core.to_local(core.now_utc()).date()
    first, _ = core.next_after_expr(dnf, today, default_seed=today, seed_base="preview-test")
    second, _ = core.next_after_expr(dnf, first, default_seed=today, seed_base="preview-test")
    expect(f"{first:%Y-%m-%d} 12:00" in stderr_txt, f"expected first due to use shifted timed slot. stderr={stderr_txt[:500]!r}")
    expect(f"{second:%Y-%m-%d} 12:00" in stderr_txt, f"expected next yearly shifted occurrence to keep timed slot. stderr={stderr_txt[:500]!r}")


def test_hook_on_add_timed_omit_rejected():
    """on-add should reject timed omit expressions with a clear error."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    task = {
        "uuid": "00000000-0000-4000-8000-000000000115",
        "description": "hook test on-add timed omit reject",
        "status": "pending",
        "project": "testing",
        "entry": "20250108T000000Z",
        "anchor": "w:mon,wed,fri",
        "omit": "w:wed@t=09:00",
        "anchor_mode": "skip",
        "due": "20250108T090000Z",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    expect(p.returncode != 0, "on-add should fail for timed omit")
    expect((p.stdout or "").strip() == "", f"expected no stdout on timed omit failure, got: {p.stdout!r}")
    stderr_txt = _strip_markup(p.stderr)
    expect(
        "omit does not support time modifiers (@t)." in stderr_txt and "date-based only." in stderr_txt,
        f"expected timed omit validation message. stderr={stderr_txt[:500]!r}",
    )


def test_hook_on_add_invalid_omit_file_rejected():
    """on-add should reject omit_file values that are paths instead of basenames."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    task = {
        "uuid": "00000000-0000-4000-8000-000000000115a",
        "description": "hook test on-add invalid omit_file",
        "status": "pending",
        "project": "testing",
        "entry": "20250108T000000Z",
        "anchor": "w:mon,wed,fri",
        "omit_file": "../holidays.csv",
        "anchor_mode": "skip",
        "due": "20250108T090000Z",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    expect(p.returncode != 0, "on-add should fail for invalid omit_file")
    expect((p.stdout or "").strip() == "", f"expected no stdout on invalid omit_file failure, got: {p.stdout!r}")
    stderr_txt = _strip_markup(p.stderr)
    expect("omit_file must be a file name, not a path." in stderr_txt, f"expected basename validation message. stderr={stderr_txt[:500]!r}")


def test_hook_on_add_anchor_file_time_padding_hint():
    """on-add should tell the user to pad single-digit hours in anchor_file @t."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    task = {
        "uuid": "00000000-0000-4000-8000-000000000115b",
        "description": "hook test on-add anchor_file padding hint",
        "status": "pending",
        "project": "testing",
        "entry": "20250108T000000Z",
        "anchor_file": "calendar.csv@t=3:00",
        "anchor_mode": "skip",
        "due": "20250108T090000Z",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    expect(p.returncode != 0, "on-add should fail for unpadded anchor_file @t")
    expect((p.stdout or "").strip() == "", f"expected no stdout on invalid anchor_file @t failure, got: {p.stdout!r}")
    stderr_txt = _strip_markup(p.stderr)
    expect("leading zero" in stderr_txt and "03:00" in stderr_txt, f"expected padding hint in error message. stderr={stderr_txt[:500]!r}")


def test_hook_on_add_unsatisfiable_omit_fails_cleanly():
    """on-add should fail cleanly when omit removes every future anchor date."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    task = {
        "uuid": "00000000-0000-4000-8000-000000000116",
        "description": "hook test on-add unsat omit",
        "status": "pending",
        "project": "testing",
        "entry": "20250106T000000Z",
        "anchor": "w:mon",
        "omit": "w:mon",
        "anchor_mode": "skip",
        "due": "20250106T090000Z",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    expect(p.returncode != 0, "on-add should fail for unsatisfiable omit")
    expect((p.stdout or "").strip() == "", f"expected no stdout on unsatisfiable omit failure, got: {p.stdout!r}")
    stderr_txt = _strip_markup(p.stderr)
    expect("No valid anchor occurrences found after applying omit rules." in stderr_txt or "No matching anchor dates found." in stderr_txt,
           f"expected clean unsatisfiable omit failure. stderr={stderr_txt[:500]!r}")

def test_hook_on_modify_timeline_multitime_includes_all_slots():
    """on-modify timeline generator must step occurrences (date+time), not only dates."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_test")
    if not hasattr(mod, "_timeline_lines"):
        raise AssertionError("on-modify hook does not expose _timeline_lines; cannot validate timeline stepping.")
    # Avoid external calls for prev collection in unit context.
    setattr(mod, "_collect_prev_two", lambda _task: [])
    expr = "w:mon..sun@t=06:00,12:00,22:00"
    dnf = core.validate_anchor_expr_strict(expr)
    # Simulate a chain where the next due is at 22:00 on a given day.
    child_due_utc = datetime(2025, 12, 20, 22, 0, tzinfo=timezone.utc)
    task = {
        "uuid": "00000000-0000-4000-8000-000000000222",
        "description": "hook test on-modify multitime",
        "anchor": expr,
        "anchor_mode": "skip",
        "link": 1,
        # completed earlier in the day
        "end": "20251220T090000Z",
        # due is not required by _timeline_lines, but helpful for formatting.
        "due": "20251220T120000Z",
    }
    lines = _call_with_supported_kwargs(
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
    txt = _strip_markup("\n".join(lines))
    times = sorted(set(re.findall(r"\b\d{2}:\d{2}\b", txt)))
    # Expect to see at least the three slots across the timeline.
    for t in ("06:00", "12:00", "22:00"):
        if t not in times:
            raise AssertionError(f"on-modify timeline missing time {t}. found={times}. text={txt[:500]!r}")
    # Also ensure it isn't collapsing to a single daily time.
    if len(times) < 3:
        raise AssertionError(f"on-modify timeline collapsed times unexpectedly: {times}. text={txt[:500]!r}")


def test_hook_on_modify_timeline_cp_sequence_labels_future_intervals():
    """cp sequence timelines should show the interval used for future rows."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_cp_sequence_timeline_test")
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
    lines = _call_with_supported_kwargs(
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
    txt = _strip_markup("\n".join(lines))
    for token in ("(20d)", "(7d)", "(3d)"):
        expect(token in txt, f"cp sequence timeline missing {token}: {txt}")
    expect(
        evaluator_calls["count"] == 1,
        f"CP timeline rebuilt the task evaluator instead of reusing one session: {evaluator_calls}",
    )


def test_hook_on_modify_timeline_cp_random_labels_selected_intervals():
    """cp random timelines should display the selected interval for each future row."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_cp_random_timeline_test")
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
    lines = _call_with_supported_kwargs(
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
    txt = _strip_markup("\n".join(lines))
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
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_omit_timeline_test")
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
    lines = _call_with_supported_kwargs(
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
    txt = _strip_markup("\n".join(lines))
    expect("(omitted)" in txt, f"expected omitted marker in anchor timeline: {txt!r}")
    expect(
        "2025-01-08" in txt or "2025-01-08 09:00" in txt or "Wed 2025-01-08" in txt,
        f"expected omitted Wednesday slot to remain visible: {txt!r}",
    )


def test_hook_on_modify_merged_timeline_marks_projection_failures():
    """Merged anchor/anchor-file timelines should expose provider failures as warning rows."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_merged_timeline_warning_test")
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
            lines = _call_with_supported_kwargs(
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

    text = _strip_markup("\n".join(lines))
    expect("Projection unavailable" in text, f"merged projection failure was hidden: {text!r}")
    expect("merged provider contract broken" in text, f"merged warning lost failure detail: {text!r}")


def test_hook_on_modify_timeline_uses_omit_file_description_label():
    """anchor timelines should use omit_file description text for omitted markers when available."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_omit_file_desc_timeline_test")
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
            lines = _call_with_supported_kwargs(
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
    txt = _strip_markup("\n".join(lines))
    expect("(Company holida...)" in txt, f"expected truncated omit_file description marker in anchor timeline: {txt!r}")
    expect("(omitted)" not in txt, f"expected omit_file description to replace default omitted marker: {txt!r}")


def test_on_add_dnf_cache_uses_central_api_and_fingerprints_parser():
    """on-add DNF caching uses the central cache format and parser fingerprints."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_central_cache_test")
    with tempfile.TemporaryDirectory() as td:
        old_dir = getattr(mod.core, "ANCHOR_CACHE_DIR_OVERRIDE", "")
        old_cache = getattr(mod.core, "_CACHE_DIR", None)
        old_enabled = getattr(mod.core, "ENABLE_ANCHOR_CACHE", True)
        mod.core.ANCHOR_CACHE_DIR_OVERRIDE = td
        mod.core.ENABLE_ANCHOR_CACHE = True
        setattr(mod.core, "_CACHE_DIR", None)
        try:
            dnf = mod.core.validate_anchor_expr_strict("w:mon")
            expect(mod.core._dnf_cache_save("w:mon", dnf), "central DNF cache save failed")
            expect(mod.core._dnf_cache_load("w:mon") == dnf, "central DNF cache did not round-trip")
            fingerprint = mod.core._dnf_cache_fingerprint()
            expect("parser=" in fingerprint, f"parser fingerprint missing: {fingerprint}")
            expect("schema:" in fingerprint, f"cache schema fingerprint missing: {fingerprint}")
            expect("release:" in fingerprint, f"release fingerprint missing: {fingerprint}")
            cache_path = Path(mod.core._cache_path(mod.core._dnf_cache_key("w:mon")))
            expect(cache_path.suffix == ".jsonz" and cache_path.exists(), f"central cache path missing: {cache_path}")
        finally:
            mod.core.ANCHOR_CACHE_DIR_OVERRIDE = old_dir
            mod.core.ENABLE_ANCHOR_CACHE = old_enabled
            setattr(mod.core, "_CACHE_DIR", old_cache)


def test_on_add_dnf_cache_quarantines_central_corruption():
    """Central cache corruption is quarantined and treated as a miss."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_central_cache_corrupt_test")
    with tempfile.TemporaryDirectory() as td:
        old_dir = getattr(mod.core, "ANCHOR_CACHE_DIR_OVERRIDE", "")
        old_cache = getattr(mod.core, "_CACHE_DIR", None)
        mod.core.ANCHOR_CACHE_DIR_OVERRIDE = td
        mod.core._CACHE_DIR = None
        try:
            cache_path = Path(mod.core._cache_path(mod.core._dnf_cache_key("w:mon")))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(b"not a cache")
            expect(mod.core._dnf_cache_load("w:mon") is None, "corrupt central DNF cache should be a miss")
            expect(list(cache_path.parent.glob(cache_path.name + ".bad.*")), "corrupt central DNF cache was not quarantined")
        finally:
            mod.core.ANCHOR_CACHE_DIR_OVERRIDE = old_dir
            mod.core._CACHE_DIR = old_cache


def test_hooks_require_package_core_layout():
    """Hooks should resolve only the package-based nautical_core layout."""
    import tempfile

    hook_names = ["on-add.nautical", "on-modify.nautical", "on-exit.nautical"]
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        pkg = td_path / "nautical_core"
        pkg.mkdir(parents=True, exist_ok=True)
        pkg_init = pkg / "__init__.py"
        pkg_init.write_text("# package core\n", encoding="utf-8")
        legacy = td_path / "nautical_core.py"
        legacy.write_text("# legacy core\n", encoding="utf-8")
        for idx, hook_name in enumerate(hook_names):
            mod = _load_hook_module(_find_hook_file(hook_name), f"_nautical_pkg_layout_test_{idx}")
            resolved = mod._core_target_from_base(td_path)
            expect(resolved == pkg_init, f"{hook_name} should prefer package core: {resolved}")
            expect(mod._core_target_from_base(legacy) is None, f"{hook_name} should reject legacy core file")

def test_core_import_deterministic():
    """Hooks should ignore TASKDATA unless NAUTICAL_DEV=1."""
    with tempfile.TemporaryDirectory() as td:
        bad_core = Path(td) / "nautical_core/__init__.py"
        bad_core.parent.mkdir(parents=True, exist_ok=True)
        bad_core.write_text("raise RuntimeError('bad core')\n", encoding="utf-8")
        os.environ["TASKDATA"] = td
        os.environ.pop("NAUTICAL_DEV", None)
        try:
            hook = _find_hook_file("on-add.nautical")
            _ = _load_hook_module(hook, "_nautical_on_add_import_deterministic_test").core
        finally:
            os.environ.pop("TASKDATA", None)

    expect(True, "core import should ignore TASKDATA when NAUTICAL_DEV is not set")


def test_core_import_defers_optional_stacks():
    """Importing the facade should defer optional and parser API stacks."""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys, nautical_core; "
                "names=('rich','nautical_core.ui','nautical_core.astronomy',"
                "'nautical_core.natural_language','nautical_core.linting',"
                "'nautical_core.parser_api','nautical_core.parser_support_api',"
                "'nautical_core.acf_api','nautical_core.expansion_api',"
                "'nautical_core.quarter_api','nautical_core.scheduler_api',"
                "'nautical_core.cached_expansion','nautical_core.monthly_support',"
                "'nautical_core.recurrence_evaluator',"
                "'nautical_core.token_api','nautical_core.time_api',"
                "'nautical_core.business_calendar_api','nautical_core.cache_api',"
                "'nautical_core.hint_builder_api','nautical_core.natural_language_api',"
                "'nautical_core.linting_api'); "
                "loaded=sorted(name for name in names if name in sys.modules); "
                "count=sum(name.startswith('nautical_core') for name in sys.modules); "
                "assert not loaded, loaded; assert 'subprocess' not in sys.modules, 'subprocess loaded'; "
                "assert 'tempfile' not in sys.modules, 'tempfile loaded'; "
                "assert count <= 30, count; "
                "print(json.dumps({'count': count, 'loaded': loaded}))"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=12.0,
    )
    expect(probe.returncode == 0, f"optional stacks were imported eagerly: {probe.stderr or probe.stdout}")

def test_queue_claim_quarantines_poison_rows_and_queue_status_reports_them():
    """Quarantined lifecycle intents remain visible to operator diagnostics."""
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository
    from nautical_core.tools import nautical_doctor
    from nautical_core.tools import nautical_queue_status

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        repository = _LifecycleOutboxRepository(root)
        expect(repository.open().ok, "lifecycle outbox did not initialize")
        with sqlite3.connect(str(repository.path)) as conn:
            conn.execute(
                "INSERT INTO lifecycle_outbox "
                "(intent_id, plan_json, plan_fingerprint, parent_guard_json, configuration_fingerprint, "
                "schedule_fingerprint, lifecycle_stage, processing_state, attempts, failure_json, created_at, updated_at) "
                "VALUES (?, '{}', 'poison', '{}', 'cf', 'sf', 'planned', 'quarantined', 1, ?, 1.0, 1.0)",
                ("outbox-poison", json.dumps({"code": "poison_row", "message": "invalid lifecycle plan JSON"})),
            )

        summary, _budget = nautical_queue_status._status_payload(root, stale_after=300.0, limit=5)
        issues = summary.get("issues", [])
        outbox = summary.get("outbox", {})
        expect(outbox.get("states", {}).get("quarantined") == 1, f"outbox status missed quarantined row: {summary!r}")
        expect(
            any("quarantined" in issue for issue in issues),
            f"queue status missed poison issue: {issues!r}",
        )
        findings = []
        nautical_doctor._check_lifecycle_outbox(findings, root, 300.0)
        poison_finding = next((item for item in findings if item.get("id") == "outbox.poison_rows"), None)
        expect(poison_finding and poison_finding.get("severity") == "error", f"doctor missed poison row: {findings!r}")


def test_on_modify_carry_wall_clock_across_dst():
    """carry-forward should preserve local wall-clock offset across DST."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_carry_dst_test")
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

        _carry_relative_datetime(mod, parent, child, child_due_utc, "wait")
        wait_child = mod.core.parse_dt_any(child.get("wait"))
        wait_local = mod.core.to_local(wait_child)

        expect(wait_local.hour == 3 and wait_local.minute == 30, f"unexpected local wait: {wait_local}")
    finally:
        mod.core.LOCAL_TZ_NAME = previous_tz_name
        mod.core._LOCAL_TZ = previous_tz


def test_on_modify_build_child_carries_until_across_dst():
    """native until should retain its local wall-clock offset from the recurrence due."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_carry_until_dst_test")
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

        child = _build_child_draft_for_test(mod,
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

    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_native_until_carry_policy_test")
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
        return _build_child_draft_for_test(mod,
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
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_native_until_exact_dst_test")
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

        child = _build_child_draft_for_test(mod,
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
    add_hook = _find_hook_file("on-add.nautical")
    modify_hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(modify_hook, "_nautical_native_until_slot_guard_test")
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
        added = _run_hook_script(add_hook, dict(base), env_extra=env)
        expect(added.returncode != 0, "on-add accepted a same-day expiration before an anchor slot")
        expect(not added.stdout.strip(), f"rejected on-add leaked stdout: {added.stdout!r}")
        added_stderr = _strip_markup(added.stderr)
        expect("Invalid expiration window" in added_stderr, f"missing on-add expiration panel: {added_stderr!r}")
        expect("20:00" in added_stderr, f"missing conflicting anchor slot: {added_stderr!r}")

        exact = _run_hook_script(
            add_hook,
            dict(base, until="20300701T190001Z"),
            env_extra=env,
        )
        expect(exact.returncode == 0, f"+1s exact expiration should bypass calendar slot rejection: {exact.stderr!r}")

        old = dict(base, chain="on", chainID="cid_until_slots", link=1, until="20300701T210000Z")
        new = dict(old, until="20300701T190000Z")
        modified = _run_hook_script_raw(
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
        anchor_invalid = _run_hook_script_raw(
            modify_hook,
            json.dumps(anchor_old) + "\n" + json.dumps(dict(anchor_old, anchor="w:mon@t=09:00,20:00")),
            env_extra=dict(env, TASKDATA=td),
        )
        anchor_valid = _run_hook_script_raw(
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
        converted = _run_hook_script_raw(
            modify_hook,
            json.dumps(cp_old) + "\n" + json.dumps(cp_to_anchor),
            env_extra=dict(env, TASKDATA=td),
        )
    expect(modified.returncode != 0, "on-modify accepted a same-day expiration before an anchor slot")
    expect(not modified.stdout.strip(), f"rejected on-modify leaked stdout: {modified.stdout!r}")
    expect("Invalid expiration window" in _strip_markup(modified.stderr), f"missing modify expiration panel: {modified.stderr!r}")
    expect(anchor_invalid.returncode != 0, "on-modify accepted an anchor edit adding a slot after expiration")
    expect(not anchor_invalid.stdout.strip(), f"rejected anchor edit leaked stdout: {anchor_invalid.stdout!r}")
    expect("20:00" in _strip_markup(anchor_invalid.stderr), f"missing edited anchor slot: {anchor_invalid.stderr!r}")
    expect(anchor_valid.returncode == 0, f"valid anchor slot edit was rejected: {anchor_valid.stderr!r}")
    expect(
        _assert_stdout_json_only(anchor_valid.stdout).get("anchor") == "w:mon@t=09:00,18:00",
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
        from_file = _run_hook_script(
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
        modified_file = _run_hook_script_raw(
            modify_hook,
            json.dumps(file_old) + "\n" + json.dumps(file_new),
            env_extra={"NO_COLOR": "1", "NAUTICAL_CONFIG": str(config), "TASKDATA": td},
        )
    expect(from_file.returncode != 0, "anchor_file accepted a same-day expiration before a file slot")
    expect(not from_file.stdout.strip(), f"rejected anchor_file add leaked stdout: {from_file.stdout!r}")
    expect("20:00" in _strip_markup(from_file.stderr), f"missing anchor_file slot evidence: {from_file.stderr!r}")
    expect(modified_file.returncode != 0, "on-modify accepted an anchor_file edit adding a slot after expiration")
    expect(not modified_file.stdout.strip(), f"rejected anchor_file edit leaked stdout: {modified_file.stdout!r}")
    expect("20:00" in _strip_markup(modified_file.stderr), f"missing edited anchor_file slot: {modified_file.stderr!r}")

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
        _build_child_draft_for_test(mod,
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


def test_on_modify_build_child_transitions_flex_to_all():
    """A flex anchor should skip backlog once and make its child strict all mode."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_flex_child_mode_test")
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
    child = _build_child_draft_for_test(mod,
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
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_cp_due_scheduled_test")
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
        _modify_effect(mod, "handle_non_completion", old, due_only, _test_operator_uow())
        _modify_effect(mod, "handle_non_completion", old, explicit_scheduled, _test_operator_uow())
        _modify_effect(mod, "handle_non_completion", old, explicit_wait, _test_operator_uow())
        _modify_effect(mod, "handle_non_completion", old, explicit_both, _test_operator_uow())
        for invalid_old, invalid in (
            (old, malformed),
            (malformed_scheduled_old, malformed_scheduled),
            (malformed_wait_old, malformed_wait),
        ):
            try:
                _modify_effect(mod, "handle_non_completion", invalid_old, invalid, _test_operator_uow())
            except SystemExit as exc:
                expect(exc.code == 1, f"carry failure exited with unexpected status: {exc.code!r}")
            else:
                raise AssertionError(f"malformed carry was accepted: {invalid!r}")
        mod._completion_effects.preflight_context = lambda *_args, **_kwargs: None
        _modify_effect(mod, "handle_completion", old, completed, _test_operator_uow())
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
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_timing_order_panel_test")
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
            _modify_effect(mod, "handle_non_completion", base, changed, _test_operator_uow())
        _modify_effect(mod, "handle_non_completion", old, valid, _test_operator_uow())
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
    hook = _find_hook_file("on-modify.nautical")
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
        proc = _run_hook_script_raw(
            hook,
            json.dumps(old) + "\n" + json.dumps(new),
            env_extra={"TASKDATA": td},
        )

    expect(proc.returncode == 0, f"timing warning hook failed: {proc.stderr!r}")
    expect(_assert_stdout_json_only(proc.stdout) == new, f"timing warning changed task JSON: {proc.stdout!r}")
    expect("Nautical timing order" in proc.stderr, f"timing warning missing from stderr: {proc.stderr!r}")
    expect("Scheduled is after Due" in proc.stderr, f"timing problem missing from stderr: {proc.stderr!r}")


def test_on_modify_build_child_carries_configured_uda_datetime():
    """configured recurrence_update_udas fields should carry with wall-clock delta."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_carry_uda_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    try:
        from zoneinfo import ZoneInfo
    except Exception:
        return

    prev_cfg = getattr(mod, "_RECURRENCE_UPDATE_UDAS", ())
    prev_tz_name = getattr(mod.core, "LOCAL_TZ_NAME", None)
    prev_local_tz = getattr(mod.core, "_LOCAL_TZ", None)
    try:
        mod._RECURRENCE_UPDATE_UDAS = ("rappel",)
        mod.core.LOCAL_TZ_NAME = "America/New_York"
        mod.core._LOCAL_TZ = ZoneInfo("America/New_York")

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
        child = _build_child_draft_for_test(mod,
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
        mod.core._LOCAL_TZ = prev_local_tz


def test_on_modify_stable_child_uuid_is_slot_deterministic():
    """stable child UUID should be deterministic for the same parent slot and change with link."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_stable_child_uuid_test")

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
    uuid_fn = lambda value: prep.stable_child_uuid(
        value[0], value[1], task_uuid_or_empty=mod._module("modify_task_fields").task_uuid_or_empty,
        coerce_int=mod.core.coerce_int, stable_child_uuid_namespace=mod._STABLE_CHILD_UUID_NAMESPACE,
    )
    uuid_a = uuid_fn((parent, child_a))
    uuid_b = uuid_fn((parent, child_b))
    uuid_c = uuid_fn((parent, child_c))

    expect(bool(uuid_a), "stable child uuid should not be empty")
    expect(uuid_a == uuid_b, "same chain slot should yield same stable uuid")
    expect(uuid_a != uuid_c, "different link slot should yield different stable uuid")


def test_on_modify_link_limit():
    """on-modify should block spawns when link exceeds max."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_link_limit_test")
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


def test_on_modify_completion_preflight_context_happy_path():
    """completion preflight should derive link numbers, kind, and chain id for a valid chain task."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_preflight_context_test")
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
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_compute_next_limits_test")
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
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_cap_until_exact_test")
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


def test_hook_on_add_rejects_invalid_chain_max_for_cp_and_anchor():
    """on-add should reject invalid chainMax values for both recurrence branches."""
    hook = _find_hook_file("on-add.nautical")
    env = {"NO_COLOR": "1"}
    cases = [
        ({"cp": "1d", "chainMax": 0}, "cp zero"),
        ({"cp": "1d", "chainMax": -1}, "cp negative"),
        ({"cp": "1d", "chainMax": 2.5}, "cp fractional"),
        ({"anchor": "w:mon", "anchor_mode": "skip", "chainMax": 0}, "anchor zero"),
    ]
    for idx, (attrs, label) in enumerate(cases, start=1):
        task = {
            "uuid": f"00000000-0000-4000-8000-00000000{180 + idx:04d}",
            "description": f"hook test invalid chainMax add {label}",
            "status": "pending",
            "entry": "20260101T000000Z",
            "due": "20260101T090000Z",
            **attrs,
        }
        p = _run_hook_script(hook, task, env_extra=env)
        expect(p.returncode != 0, f"on-add should reject {label}")
        expect((p.stdout or "").strip() == "", f"invalid chainMax add should not emit stdout: {p.stdout!r}")
        stderr_txt = _strip_markup(p.stderr)
        expect("Invalid chainMax" in stderr_txt, f"expected chainMax panel for {label}: {stderr_txt[:500]!r}")
        expect("chainMax must be" in stderr_txt, f"expected chainMax guidance for {label}: {stderr_txt[:500]!r}")


def test_hook_on_modify_rejects_invalid_chain_max_for_cp_and_anchor():
    """on-modify should reject invalid chainMax values before completion or spawn."""
    hook = _find_hook_file("on-modify.nautical")
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
        p = _run_hook_script_raw(hook, raw, env_extra=env)
        expect(p.returncode != 0, f"on-modify should reject {label}")
        expect((p.stdout or "").strip() == "", f"invalid chainMax modify should not emit stdout: {p.stdout!r}")
        stderr_txt = _strip_markup(p.stderr)
        expect("Invalid chainMax" in stderr_txt, f"expected chainMax panel for {label}: {stderr_txt[:500]!r}")
        expect("chainMax must be" in stderr_txt, f"expected chainMax guidance for {label}: {stderr_txt[:500]!r}")


def test_on_modify_validates_chain_until_only_when_recurrence_or_caps_change():
    """Unrelated edits should pass, but changing chainUntil should trigger strict validation."""
    hook = _find_hook_file("on-modify.nautical")
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
    p = _run_hook_script_raw(hook, json.dumps(old) + "\n" + json.dumps(unrelated) + "\n", env_extra=env)
    expect(p.returncode == 0, f"unrelated modify should not revalidate an old cap: stderr={p.stderr!r}")
    expect(_extract_last_json(p.stdout) == unrelated, f"unrelated modify should pass through: {p.stdout!r}")

    invalid = dict(old)
    invalid["chainUntil"] = "not-a-date"
    p = _run_hook_script_raw(hook, json.dumps(old) + "\n" + json.dumps(invalid) + "\n", env_extra=env)
    expect(p.returncode != 0, "changing chainUntil to an invalid value should fail")
    expect((p.stdout or "").strip() == "", f"invalid chainUntil modify should not emit stdout: {p.stdout!r}")
    stderr_txt = _strip_markup(p.stderr)
    expect("Invalid chainUntil" in stderr_txt, f"expected chainUntil panel: {stderr_txt[:500]!r}")
    expect("Unrecognized datetime format" in stderr_txt, f"expected chainUntil guidance: {stderr_txt[:500]!r}")


def test_on_modify_completion_helper_returns_finalized_lifecycle_result():
    """The hook helper must expose the typed result returned by finalization."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_completion_result_boundary_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    models = core._import_sibling("modify_models")
    expected = models.CompletionLifecycleResult(state="applied", child_short="child123")
    ctx = SimpleNamespace(
        parent_short="parent01",
        base_no=1,
        next_no=2,
        kind="anchor",
        chain_id="chain01",
        chain_snapshot=SimpleNamespace(rows=[], mode="next", loaded=False),
    )
    computed = SimpleNamespace(
        child_due=core.now_utc(),
        meta={"target_field": "due"},
        dnf=[],
        until_dt=None,
        cpmax=0,
        cap_no=None,
        finals=[],
        until_cap_no=None,
    )
    fake_flow = SimpleNamespace(
        CompletionFlowServices=lambda **kwargs: kwargs,
        CompletionFinalizeServices=lambda **kwargs: kwargs,
        finalize_completion_modify=lambda **_kwargs: expected,
        handle_completion_modify=lambda *_args, **_kwargs: expected,
    )
    validation = mod._module("modify_validation_effects")
    original = {
        "validate_cp": validation.validate_cp,
        "preserve_cp": mod._transition_effects.preserve_cp_relative_offsets_on_due_change,
        "preserve_until": mod._transition_effects.preserve_native_until_on_target_change,
        "validate_until": mod._module("modify_validation_effects").validate_native_until,
        "validate_slots": mod._module("modify_validation_effects").validate_native_until_slots,
        "preflight": mod._completion_effects.preflight_context,
        "compute": mod._completion_effects.compute_next_and_limits,
        "import_module": mod.importlib.import_module,
    }
    try:
        validation.validate_cp = lambda *_a, **_k: None
        mod._transition_effects.preserve_cp_relative_offsets_on_due_change = lambda *_a, **_k: None
        mod._transition_effects.preserve_native_until_on_target_change = lambda *_a, **_k: None
        mod._module("modify_validation_effects").validate_native_until = lambda *_a, **_k: None
        mod._module("modify_validation_effects").validate_native_until_slots = lambda *_a, **_k: None
        mod._completion_effects.preflight_context = lambda *_a, **_k: ctx
        mod._completion_effects.compute_next_and_limits = lambda *_a, **_k: computed

        def fake_import(name):
            if name == "nautical_core.modify_completion_flow":
                return fake_flow
            return original["import_module"](name)

        mod.importlib.import_module = fake_import
        result = _modify_effect(mod, "handle_completion",
            {"uuid": "parent", "status": "pending"},
            {"uuid": "parent", "status": "completed"},
            _test_operator_uow(),
        )
    finally:
        validation.validate_cp = original["validate_cp"]
        mod._transition_effects.preserve_cp_relative_offsets_on_due_change = original["preserve_cp"]
        mod._transition_effects.preserve_native_until_on_target_change = original["preserve_until"]
        mod._module("modify_validation_effects").validate_native_until = original["validate_until"]
        mod._module("modify_validation_effects").validate_native_until_slots = original["validate_slots"]
        mod._completion_effects.preflight_context = original["preflight"]
        mod._completion_effects.compute_next_and_limits = original["compute"]
        mod.importlib.import_module = original["import_module"]

    expect(result is expected, f"completion helper dropped finalized result: {result!r}")


def test_on_modify_completion_chain_snapshot_modes_and_query():
    """Completion presentation modes share one authoritative chain read."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_completion_snapshot_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    from nautical_core.integration_models import Found

    saved = (mod.core.PANEL_MODE, mod._SHOW_ANALYTICS, mod._CHECK_CHAIN_INTEGRITY)
    calls = []
    repository = SimpleNamespace(
        chain_snapshot=lambda chain_id: calls.append(chain_id) or Found(tuple(), "chain snapshot")
    )

    try:
        mod._SHOW_ANALYTICS = False
        mod._CHECK_CHAIN_INTEGRITY = False
        mod.core.PANEL_MODE = "line"
        next_only = mod._completion_effects.chain_snapshot("cid", 5, 6, repository)
        expect(next_only.mode == "next" and next_only.loaded, f"unexpected line snapshot: {next_only}")

        mod.core.PANEL_MODE = "rich"
        recent = mod._completion_effects.chain_snapshot("cid", 5, 6, repository)
        expect(recent.mode == "recent" and recent.loaded, f"unexpected recent snapshot: {recent}")

        mod._CHECK_CHAIN_INTEGRITY = True
        full = mod._completion_effects.chain_snapshot("cid", 5, 6, repository)
        expect(full.mode == "full" and full.loaded, f"unexpected full snapshot: {full}")
        expect(calls == ["cid", "cid", "cid"], f"completion bypassed repository chain reads: {calls!r}")
    finally:
        mod.core.PANEL_MODE, mod._SHOW_ANALYTICS, mod._CHECK_CHAIN_INTEGRITY = saved


def test_on_modify_completion_snapshot_malformed_json_is_unavailable():
    """Malformed completion exports must not become loaded empty snapshots."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_completion_snapshot_malformed_test")
    from nautical_core.integration_models import CommandFailureKind, FailureEvidence, TaskCommand, Unavailable

    saved = (mod.core.PANEL_MODE, mod._SHOW_ANALYTICS, mod._CHECK_CHAIN_INTEGRITY)
    command = TaskCommand(("task", "export"), "completion snapshot", 3.0)
    unavailable = Unavailable(
        "chain snapshot",
        FailureEvidence(command, CommandFailureKind.INVALID_RESPONSE, 0, 1, 0.001, False, "malformed JSON"),
    )
    repository = SimpleNamespace(
        chain_snapshot=lambda *_args, **_kwargs: unavailable,
        exact_child_slot=lambda *_args, **_kwargs: unavailable,
    )
    try:
        mod.core.PANEL_MODE = "line"
        mod._SHOW_ANALYTICS = False
        mod._CHECK_CHAIN_INTEGRITY = False
        snapshot = mod._completion_effects.chain_snapshot("malformed01", 1, 2, repository)
        expect(snapshot.is_unavailable, f"malformed snapshot was accepted: {snapshot!r}")
        expect(not snapshot.loaded and snapshot.rows == [], f"malformed snapshot changed lookup state: {snapshot!r}")
        panels = []
        mod._panel = lambda title, rows, **_kwargs: panels.append((title, rows))
        mod._print_task = lambda _task: None
        allowed = mod._completion_effects.existing_next_or_fail({}, 2, snapshot, repository)
        expect(not allowed and panels and "unavailable" in panels[0][0].lower(), "unavailable snapshot did not stop spawn")
    finally:
        mod.core.PANEL_MODE, mod._SHOW_ANALYTICS, mod._CHECK_CHAIN_INTEGRITY = saved


def test_on_modify_completion_defers_chain_export_until_after_preflight():
    """completion handling should not export the chain before preflight succeeds."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_preflight_export_deferral_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    called = {"chain_export": 0}
    mod._module("modify_validation_effects").validate_cp = lambda *_a, **_k: None
    mod._completion_effects.preflight_context = lambda *_a, **_k: None
    mod._completion_effects.compute_next_and_limits = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("compute should not run after preflight failure"))
    mod._SHOW_ANALYTICS = True
    mod._SHOW_TIMELINE_GAPS = False
    mod._CHECK_CHAIN_INTEGRITY = False

    old = {"uuid": "00000000-0000-4000-8000-000000000111", "status": "pending", "cp": "P1D", "chainID": "abcd1234", "link": 1}
    new = dict(old)
    new["status"] = "completed"

    import io
    from contextlib import redirect_stdout, redirect_stderr

    stdin_raw = json.dumps(old) + "\n" + json.dumps(new)
    stdout = io.StringIO()
    stderr = io.StringIO()
    orig_stdin = sys.stdin
    try:
        sys.stdin = io.TextIOWrapper(io.BytesIO(stdin_raw.encode("utf-8")), encoding="utf-8")
        with redirect_stdout(stdout), redirect_stderr(stderr):
            _modify_effect(mod, "handle_completion", old, new, _test_operator_uow())
    finally:
        sys.stdin = orig_stdin

    expect(called["chain_export"] == 0, f"expected no chain export before preflight success, got {called}")


def test_on_modify_compute_cp_child_due_uses_scheduled_when_due_missing():
    """scheduled-only cp chains should preserve scheduled wall clock on completion."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_cp_sched_only_compute_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    child_due, meta = _compute_cp_child_due(mod,
        {
            "cp": "P1D",
            "chainID": "scheduled-only-chain",
            "scheduled": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 1), (9, 0))),
            "end": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 1), (17, 0))),
        }
    )
    child_local = mod.core.to_local(child_due)
    expect((child_local.hour, child_local.minute) == (9, 0), f"unexpected scheduled-only child time: {child_local}")
    expect(meta.get("target_field") == "scheduled", f"expected scheduled target field: {meta}")


def test_on_modify_compute_cp_sequence_selects_interval_by_link():
    """cp sequences should derive the active interval from the current link number."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_cp_sequence_compute_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    child_due, meta = _compute_cp_child_due(mod,
        {
            "cp": "3d,20d,7d",
            "chainID": "sequence-chain",
            "link": 2,
            "due": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2026, 1, 1), (9, 0))),
            "end": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2026, 1, 1), (10, 0))),
        }
    )
    child_local = mod.core.to_local(child_due)
    expect(child_local.date() == date(2026, 1, 21), f"link #2 should use 20d interval: {child_local}")
    expect((child_local.hour, child_local.minute) == (9, 0), f"whole-day sequence interval should preserve wall clock: {child_local}")
    expect(meta.get("cp_sequence_step") == 2, f"expected sequence step 2: {meta}")
    expect(meta.get("cp_sequence_len") == 3, f"expected sequence length 3: {meta}")


def test_on_modify_compute_cp_random_selects_deterministic_interval():
    """random cp ranges should resolve deterministically for the active link."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_cp_random_compute_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    chain_id = "abcd1234"
    expected_td = mod.core.cp_sequence_interval_for_link("rand(3d..7d)", 2, chain_id)
    child_due, meta = _compute_cp_child_due(mod,
        {
            "cp": "rand(3d..7d)",
            "link": 2,
            "chainID": chain_id,
            "due": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2026, 1, 1), (9, 0))),
            "end": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2026, 1, 1), (10, 0))),
        }
    )
    child_local = mod.core.to_local(child_due)
    expected_local = mod.core.to_local(mod.core.build_local_datetime(date(2026, 1, 1), (9, 0)) + expected_td)
    expect(child_local.date() == expected_local.date(), f"random cp should use deterministic selected interval: {child_local} vs {expected_local}")
    expect((child_local.hour, child_local.minute) == (9, 0), f"whole-day random interval should preserve wall clock: {child_local}")
    expect(meta.get("cp_sequence_step") == 1, f"expected random cp selected step metadata: {meta}")
    expect(meta.get("cp_sequence_len") == 1, f"expected random cp length metadata: {meta}")


def test_on_modify_cp_sequence_estimates_chainmax_final_date():
    """chainMax final-date estimation should advance through cp sequence intervals."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_cp_sequence_chainmax_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    next_due = mod.core.build_local_datetime(date(2026, 1, 4), (9, 0)).astimezone(timezone.utc)
    final_due = mod._estimate_cp_final_by_max(
        {"cp": "3d,20d,7d", "link": 1, "chainMax": 4},
        next_due,
    )
    final_local = mod.core.to_local(final_due)
    expect(final_local.date() == date(2026, 1, 31), f"expected Jan 31 final due, got {final_local}")
    expect((final_local.hour, final_local.minute) == (9, 0), f"whole-day sequence cap should preserve wall clock: {final_local}")


def test_on_modify_anchor_chainmax_forecast_is_bounded():
    """Large anchor chainMax values must not make final-date forecasting unbounded."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_anchor_chainmax_bound_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    from nautical_core.recurrence_evaluator import RecurrenceEvaluator

    original_next = RecurrenceEvaluator._default_next_occurrence_after_local_dt
    original_diag = mod._diag
    diagnostics = []
    calls = []

    def next_daily(_self, _dnf, value, **_kwargs):
        calls.append(value)
        return value + timedelta(days=1)

    RecurrenceEvaluator._default_next_occurrence_after_local_dt = next_daily
    try:
        from dataclasses import replace
        schedule = mod._module("modify_schedule_effects")
        anchor_ports = replace(schedule.anchor_completion_ports_for(mod), diagnostic=diagnostics.append)
        final_due = schedule.estimate_anchor_final_by_max(
            anchor_ports,
            {
                "uuid": "00000000-0000-4000-8000-000000000118",
                "description": "chain max bound",
                "status": "completed",
                "anchor": "w:mon",
                "link": 1,
                "chainMax": 5000,
                "chainID": "bound-test",
            },
            datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
            None,
        )
    finally:
        RecurrenceEvaluator._default_next_occurrence_after_local_dt = original_next
        mod._diag = original_diag
    expect(final_due is None, "unbounded anchor forecast returned a fabricated final date")
    expect(len(calls) == mod._MAX_ITERATIONS, f"anchor forecast exceeded its iteration budget: {len(calls)}")
    expect(diagnostics and "final date is unavailable" in diagnostics[0], f"forecast bound was not diagnosed: {diagnostics!r}")
    cp_ports = replace(schedule.cp_completion_ports_for(mod), diagnostic=diagnostics.append)
    cp_final = schedule.estimate_cp_final_by_max(
        cp_ports,
        {"cp": "1d", "link": 1, "chainMax": 5000},
        datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
    )
    expect(cp_final is None, "unbounded CP forecast returned a fabricated final date")


def test_on_modify_anchor_file_child_projection_reuses_provider():
    """Combined all-mode child projection should build one anchor-file provider."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_anchor_file_session_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()
    anchor_inclusion = mod.core._import_sibling("anchor_inclusion")
    occurrence_provider = mod.core._import_sibling("occurrence_provider")
    original_builder = anchor_inclusion._build_anchor_file_provider
    builders = []

    def build_provider(*_args, **_kwargs):
        provider = type("Provider", (), {})()
        occurrence = occurrence_provider.Occurrence(
            date(2026, 8, 4), 9, 0, source="anchor_file",
            local_datetime=mod.core.to_local(mod.core.build_local_datetime(date(2026, 8, 4), (9, 0))),
        )
        provider.next_after = lambda after_local, **_kwargs: (
            occurrence
            if occurrence.local_datetime is not None and occurrence.local_datetime > after_local
            else None
        )
        builders.append(provider)
        return provider

    anchor_inclusion._build_anchor_file_provider = build_provider
    try:
        due = mod.core.build_local_datetime(date(2026, 8, 3), (9, 0))
        child_due, _meta, _dnf = _compute_anchor_child_due(mod,
            {
                "anchor": "w:mon@t=09:00",
                "anchor_file": "calendar.csv@t=09:00",
                "anchor_mode": "all",
                "chainID": "anchor-file-session",
                "due": mod.core.fmt_isoz(due),
                "end": mod.core.fmt_isoz(due + timedelta(hours=1)),
            }
        )
    finally:
        anchor_inclusion._build_anchor_file_provider = original_builder
    expect(len(builders) == 1, f"child projection rebuilt anchor-file provider {len(builders)} times")
    expect(mod.core.to_local(child_due).strftime("%H:%M") == "09:00", f"unexpected projected child due: {child_due!r}")


def test_on_modify_pure_anchor_file_projection_reuses_provider():
    """File-only child projection should use the shared provider session."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_pure_anchor_file_session_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()
    anchor_inclusion = mod.core._import_sibling("anchor_inclusion")
    occurrence_provider = mod.core._import_sibling("occurrence_provider")
    original_builder = anchor_inclusion._build_anchor_file_provider
    builders = []

    def build_provider(*_args, **_kwargs):
        provider = type("Provider", (), {})()
        occurrence = occurrence_provider.Occurrence(
            date(2026, 8, 4), 9, 0, source="anchor_file",
            local_datetime=mod.core.to_local(mod.core.build_local_datetime(date(2026, 8, 4), (9, 0))),
        )
        provider.occurrences = lambda: [occurrence]
        provider.next_after = lambda after_local, **_kwargs: (
            occurrence
            if occurrence.local_datetime is not None and occurrence.local_datetime > after_local
            else None
        )
        builders.append(provider)
        return provider

    anchor_inclusion._build_anchor_file_provider = build_provider
    try:
        due = mod.core.build_local_datetime(date(2026, 8, 3), (9, 0))
        child_due, _meta, _dnf = _compute_anchor_child_due(mod,
            {
                "anchor_file": "calendar.csv@t=09:00",
                "anchor_mode": "skip",
                "chainID": "pure-anchor-file-session",
                "due": mod.core.fmt_isoz(due),
                "end": mod.core.fmt_isoz(due + timedelta(hours=1)),
            }
        )
    finally:
        anchor_inclusion._build_anchor_file_provider = original_builder
    expect(len(builders) == 1, f"pure anchor-file projection rebuilt provider {len(builders)} times")
    expect(mod.core.to_local(child_due).date() == date(2026, 8, 4), f"unexpected pure anchor-file child due: {child_due!r}")


def test_on_modify_compute_anchor_child_due_uses_scheduled_seed_for_all_mode():
    """scheduled-only anchor chains should compute missed occurrences from scheduled, not completion time."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_anchor_sched_only_compute_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    parent = {
            "anchor": "w:mon..sun@t=09:00",
            "anchor_mode": "all",
            "scheduled": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 6), (9, 0))),
            "end": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 8), (10, 0))),
            "chainID": "abcd1234",
        }
    child_due, meta, _dnf = _compute_anchor_child_due(mod, parent)
    expected = mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 7), (9, 0)))
    expect(mod.core.fmt_isoz(child_due) == expected, f"unexpected next scheduled anchor: {mod.core.fmt_isoz(child_due)}")
    expect(meta.get("target_field") == "scheduled", f"expected scheduled target field: {meta}")
    evaluator = _evaluator_for_fixture(parent, timezone_value=mod.core._LOCAL_TZ)
    result = evaluator.select_mode(
        "all",
        due_local=mod.core.to_local(mod.core.parse_dt_any(parent["scheduled"])),
        end_local=mod.core.to_local(mod.core.parse_dt_any(parent["end"])),
        due_explicit=False,
        fallback_hhmm=(9, 0),
    )
    expect(
        result.selected_occurrence is not None
        and result.selected_occurrence.astimezone(timezone.utc) == child_due,
        f"scheduled-only evaluator drifted from hook: {result!r} vs {child_due!r}",
    )


def test_on_modify_compute_anchor_child_due_builds_timed_slots_in_configured_timezone():
    """@t slots are local wall-clock anchors, not UTC clock values."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_anchor_timed_timezone_compute_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    child_due, _meta, _dnf = _compute_anchor_child_due(mod,
        {
            "anchor": "w:mon..sun@t=05:00,09:00,14:00,19:00",
            "anchor_mode": "skip",
            "due": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2026, 7, 4), (9, 0))),
            "end": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2026, 7, 4), (10, 0))),
            "chainID": "d07ff246",
        }
    )

    child_local = mod.core.to_local(child_due)
    expect(child_local.date() == date(2026, 7, 4), f"expected same-day next slot, got {child_local}")
    expect((child_local.hour, child_local.minute) == (14, 0), f"expected 14:00 local, got {child_local}")


def test_on_add_preview_and_completion_skip_choose_same_next_anchor():
    """Preview and completion should agree when given the same occurrence and chain seed."""
    add_hook = _find_hook_file("on-add.nautical")
    modify_hook = _find_hook_file("on-modify.nautical")
    add_mod = _load_hook_module(add_hook, "_nautical_on_add_preview_completion_agreement_test")
    modify_mod = _load_hook_module(modify_hook, "_nautical_on_modify_preview_completion_agreement_test")
    if hasattr(add_mod, "_load_core"):
        add_mod._load_core()
    if hasattr(modify_mod, "_load_core"):
        modify_mod._load_core()

    cases = [
        ("w:mon,wed,fri@t=09:00", "", date(2026, 1, 5), (9, 0)),
        ("w:mon,wed,fri + y:apr@t=09:00", "", date(2026, 4, 1), (9, 0)),
        ("m:-1bd@t=09:00", "", date(2026, 1, 30), (9, 0)),
        ("w/2:fri@t=09:00", "", date(2026, 1, 2), (9, 0)),
        ("w:mon@t=09:00,12:00,18:00", "", date(2026, 1, 5), (9, 0)),
        ("w:mon,wed,fri@t=09:00", "w:wed", date(2026, 1, 5), (9, 0)),
        ("y:rand + w:sat@t=09:00", "", date(2026, 1, 1), (9, 0)),
    ]

    chain_id = "agree1234"
    anchor_compute = add_mod._module("add_anchor_compute")

    def compute_preview_next(dnf, current, fallback_hhmm, interval_seed, seed_base, omit_dnf):
        return anchor_compute.anchor_next_occurrence_after_local_dt(
            dnf,
            current,
            fallback_hhmm,
            interval_seed,
            seed_base,
            omit_dnf=omit_dnf,
            core=add_mod.core,
            norm_t_mod=add_mod._norm_t_mod,
            resolve_time_slots=add_mod._resolve_time_slots,
        )

    for expr, omit_expr, due_day, fallback_hhmm in cases:
        due_utc = modify_mod.core.build_local_datetime(due_day, fallback_hhmm).astimezone(timezone.utc)
        due_local = modify_mod.core.to_local(due_utc)
        dnf = add_mod.core.validate_anchor_expr_strict(expr)
        omit_dnf = None
        if omit_expr:
            anchor_omit = add_mod.core._import_sibling("anchor_omit")
            omit_dnf = anchor_omit.validate_omit_expr_strict(
                omit_expr,
                validate_anchor_expr_cached=add_mod.core.validate_anchor_expr_strict,
            )

        preview_current = due_local
        completion_current = due_local
        for step in range(4):
            preview_next = compute_preview_next(
                dnf,
                preview_current,
                fallback_hhmm,
                due_day,
                chain_id,
                omit_dnf,
            )
            expect(preview_next is not None, f"{expr}: preview did not find occurrence {step + 1}")

            completion_current_utc = completion_current.astimezone(timezone.utc)
            parent = {
                "anchor": expr,
                "anchor_mode": "skip",
                "due": modify_mod.core.fmt_isoz(completion_current_utc),
                "end": modify_mod.core.fmt_isoz(completion_current_utc),
                "chainID": chain_id,
            }
            if omit_expr:
                parent["omit"] = omit_expr
            child_due, meta, _dnf = _compute_anchor_child_due(modify_mod, parent)
            completion_next = modify_mod.core.to_local(child_due)

            expect(
                preview_next == completion_next,
                f"{expr} omit={omit_expr!r} step={step + 1}: preview chose {preview_next}, completion chose {completion_next}",
            )
            expect(meta.get("basis") == "after_end", f"{expr}: expected skip-mode completion metadata: {meta}")
            preview_current = preview_next
            completion_current = completion_next


def test_on_modify_anchor_dnf_accepts_configured_preset():
    """completion-side anchor validation should resolve configured preset aliases."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_anchor_preset_dnf_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    prev_anchor_presets = getattr(mod.core, "ANCHOR_PRESETS", {})
    try:
        mod.core.ANCHOR_PRESETS = {"payday": "m:15,-1bd"}
        expr = "@payday"
        dnf = mod.core.validate_anchor_expr_strict(expr)
    finally:
        mod.core.ANCHOR_PRESETS = prev_anchor_presets

    expect(expr == "@payday", f"original preset anchor should be preserved: {expr!r}")
    expect(dnf, f"preset anchor should resolve to DNF: {dnf!r}")


def test_on_modify_omit_dnf_accepts_configured_preset():
    """completion-side omit validation should resolve configured omit preset aliases."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_omit_preset_dnf_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    prev_omit_presets = getattr(mod.core, "OMIT_PRESETS", {})
    try:
        mod.core.OMIT_PRESETS = {"april": "y:apr"}
        host = mod._module("modify_composition").hook_host(mod.__dict__, mod.__name__)
        omit_effects = mod._module("modify_anchor_effects")
        expr, omit_dnf = omit_effects.omit_dnf_from_parent(
            omit_effects.omit_ports_for(host), {"omit": "@april"}
        )
    finally:
        mod.core.OMIT_PRESETS = prev_omit_presets

    expect(expr == "@april", f"original omit preset should be preserved: {expr!r}")
    expect(omit_dnf, f"omit preset should resolve to DNF: {omit_dnf!r}")


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


def test_on_modify_reuses_task_scoped_evaluator_and_scheduler_binding():
    """One completion task should build its evaluator and scheduler binding once."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_evaluator_session_test")
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

def test_on_add_anchor_and_anchor_file_can_coexist():
    """on-add should allow anchor and anchor_file to coexist as inclusion sources."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_anchor_and_file_allowed_test")

    with tempfile.TemporaryDirectory() as td:
        anchor_dir = Path(td)
        (anchor_dir / "calendar.csv").write_text("date\n2026-04-14\n", encoding="utf-8")
        old_dir = getattr(mod.core, "ANCHOR_FILE_DIR", "")
        mod.core.ANCHOR_FILE_DIR = str(anchor_dir)
        try:
            task = {
                "description": "combined inclusions",
                "anchor": "w:mon",
                "anchor_file": "calendar.csv",
            }
            now_utc = mod.core.now_utc()
            now_local = mod.core.to_local(now_utc)
            ctx = mod._module("add_composition").build_on_add_context(mod, task, now_utc, now_local)
            expect(ctx.kind == "anchor", f"expected combined anchor kind, got {ctx.kind!r}")
        finally:
            mod.core.ANCHOR_FILE_DIR = old_dir



def test_on_add_anchor_file_root_gets_chainid_stamp():
    """on-add should stamp chainID for anchor_file roots so later completion can proceed."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_anchor_file_chainid_stamp_test")

    task = {
        "description": "anchor file chainid",
        "uuid": "12345678-1234-1234-1234-1234567890ab",
        "anchor_file": "calendar.csv",
    }
    mod._stamp_chain_id_on_add(task)
    expect(task.get("chainID") == "12345678", f"expected anchor_file root chainID stamp, got: {task!r}")


def test_on_add_chainid_stamp_failure_rejects_recurring_root():
    """A recurring root must not proceed when its mandatory chainID cannot be derived."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_chainid_failure_test")
    task = {
        "description": "chainid failure",
        "uuid": "12345678-1234-1234-1234-1234567890ab",
        "anchor": "w:mon",
    }
    captured = {}
    original_fail = mod._fail_and_exit
    original_short_uuid = mod.core.short_uuid
    try:
        def fail(title, message):
            captured.update(title=title, message=message)
            raise RuntimeError("rejected")

        mod._fail_and_exit = fail
        mod.core.short_uuid = lambda _value: (_ for _ in ()).throw(ValueError("UUID unavailable"))
        try:
            mod._stamp_chain_id_on_add(task)
        except RuntimeError as exc:
            expect(str(exc) == "rejected", f"unexpected chainID failure result: {exc}")
        else:
            raise AssertionError("chainID derivation failure was ignored")
    finally:
        mod._fail_and_exit = original_fail
        mod.core.short_uuid = original_short_uuid
    expect(captured.get("title") == "Chain identity unavailable", f"unexpected chainID failure panel: {captured}")
    expect("UUID unavailable" in str(captured.get("message")), f"chainID failure detail was lost: {captured}")
    expect(not task.get("chainID"), f"incomplete recurring root was left stamped: {task!r}")


def test_hook_on_add_anchor_file_preview_auto_assigns_first_match():
    """on-add anchor_file preview should auto-assign due from the first future file occurrence and keep task-level time."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_anchor_file_preview_test")

    with tempfile.TemporaryDirectory() as td:
        anchor_dir = Path(td)
        (anchor_dir / "calendar.csv").write_text("date,description\n2026-04-25,Party prep\n", encoding="utf-8")
        old_dir = getattr(mod.core, "ANCHOR_FILE_DIR", "")
        mod.core.ANCHOR_FILE_DIR = str(anchor_dir)
        try:
            task = {
                "uuid": "00000000-0000-4000-8000-000000000701",
                "description": "anchor file preview",
                "status": "pending",
                "chain": "on",
                "chainID": "fixture-anchor-file",
                "link": 1,
                "anchor_file": "calendar.csv@nbd@t=12:00",
                "entry": "2026-04-12T09:00:00Z",
                "due": "2026-04-12T09:00:00Z",
            }
            now_utc = mod.core.parse_dt_any("2026-04-12T09:00:00Z")
            now_local = mod.core.to_local(now_utc)
            ctx = mod._module("add_composition").build_on_add_context(mod, task, now_utc, now_local)
            expect(ctx.kind == "anchor_file", f"expected anchor_file kind, got {ctx.kind!r}")
            expect(not ctx.user_provided_due, f"expected implicit due, got {ctx!r}")

            captured = {}
            saved_panel = mod._panel
            try:
                mod._panel = lambda _title, rows, **_kwargs: captured.setdefault("rows", rows)
                mod._module("add_composition").render_anchor_preview(mod, ctx, prof=type("P", (), {"add_ms": lambda *_a, **_k: None})())
            finally:
                mod._panel = saved_panel

            due_val = task.get("due")
            expect(str(due_val).startswith("2026-04-27T12:00:00"), f"unexpected auto-assigned due for anchor_file preview: {due_val!r}")
        finally:
            mod.core.ANCHOR_FILE_DIR = old_dir

def test_hook_on_add_anchor_and_anchor_file_preview_uses_earliest_union_match():
    """combined anchor sources should preview from the earliest merged occurrence."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_anchor_and_file_preview_test")

    with tempfile.TemporaryDirectory() as td:
        anchor_dir = Path(td)
        (anchor_dir / "calendar.csv").write_text("date,description\n2026-04-14,Special date\n", encoding="utf-8")
        old_dir = getattr(mod.core, "ANCHOR_FILE_DIR", "")
        mod.core.ANCHOR_FILE_DIR = str(anchor_dir)
        try:
            task = {
                "uuid": "00000000-0000-4000-8000-000000000702",
                "description": "combined preview",
                "status": "pending",
                "chain": "on",
                "chainID": "fixture-anchor-union",
                "link": 1,
                "anchor": "w:fri@t=09:00",
                "anchor_file": "calendar.csv@t=12:00",
                "entry": "2026-04-12T09:00:00Z",
                "due": "2026-04-12T09:00:00Z",
            }
            now_utc = mod.core.parse_dt_any("2026-04-12T09:00:00Z")
            now_local = mod.core.to_local(now_utc)
            ctx = mod._module("add_composition").build_on_add_context(mod, task, now_utc, now_local)
            captured = {}
            saved_panel = mod._panel
            try:
                mod._panel = lambda _title, rows, **_kwargs: captured.setdefault("rows", rows)
                mod._module("add_composition").render_anchor_preview(mod, ctx, prof=type("P", (), {"add_ms": lambda *_a, **_k: None})())
            finally:
                mod._panel = saved_panel
            due_val = task.get("due")
            expect(str(due_val).startswith("2026-04-14T12:00:00"), f"unexpected merged due preview: {due_val!r}")
        finally:
            mod.core.ANCHOR_FILE_DIR = old_dir


def test_on_modify_compute_anchor_child_due_from_anchor_file():
    """on-modify completion should compute the next child due from anchor_file occurrences."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_anchor_file_due_test")

    with tempfile.TemporaryDirectory() as td:
        anchor_dir = Path(td)
        (anchor_dir / "calendar.csv").write_text("date,description\n2026-04-25,Party prep\n2026-05-10,Event two\n", encoding="utf-8")
        old_dir = getattr(mod.core, "ANCHOR_FILE_DIR", "")
        mod.core.ANCHOR_FILE_DIR = str(anchor_dir)
        try:
            parent = {
                "uuid": "00000000-0000-4000-8000-000000000963",
                "status": "pending",
                "description": "anchor file chain",
                "anchor_file": "calendar.csv@nbd@t=12:00",
                "anchor_mode": "skip",
                "link": 1,
                "chainID": "cid",
                "due": "2026-04-27T12:00:00Z",
                "end": "2026-04-28T12:00:00Z",
            }
            child_due, meta, dnf = _compute_anchor_child_due(mod, parent)
            expected_due = mod.core.build_local_datetime(date(2026, 5, 11), (12, 0)).astimezone(timezone.utc)
            expect(not dnf, f"anchor_file recurrence should not produce a truthy DNF payload, got {dnf!r}")
            expect(child_due == expected_due, f"unexpected anchor_file child due: {child_due!r}")
            expect(isinstance(meta, dict) and meta.get("target_field") == "due", f"unexpected anchor_file meta: {meta!r}")


            evaluator = _evaluator_for_fixture(
                parent,
                timezone=mod.core._LOCAL_TZ,
                anchor_file_dir=str(anchor_dir),
            )
            result = evaluator.select_mode(
                "skip",
                due_local=mod.core.to_local(mod.core.parse_dt_any(parent["due"])),
                end_local=mod.core.to_local(mod.core.parse_dt_any(parent["end"])),
                fallback_hhmm=(12, 0),
            )
            expect(
                result.selected_occurrence is not None
                and result.selected_occurrence.astimezone(timezone.utc) == child_due,
                f"evaluator/file mode drifted from hook mode: {result!r} vs {child_due!r}",
            )

            child = _build_child_draft_for_test(mod, parent, child_due, "due", 2, "beef", "anchor_file", 0, None)
            expect(child.get("anchor_file") == "calendar.csv@nbd@t=12:00", f"child should preserve anchor_file: {child!r}")
            expect(not child.get("anchor"), f"child should not gain anchor expr: {child!r}")

            anchor_parent = dict(parent, anchor="w:mon@t=12:00", anchor_file="null")
            anchor_child = _build_child_draft_for_test(
                mod, anchor_parent, child_due, "due", 2, "beef", "anchor", 0, None
            )
            expect(not anchor_child.get("anchor_file"), f"literal null anchor_file leaked into anchor child: {anchor_child!r}")
        finally:
            mod.core.ANCHOR_FILE_DIR = old_dir


def test_on_modify_compute_anchor_child_due_from_random_anchor_file():
    """Operational anchor-file completion should preserve chain-scoped random slots."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_random_anchor_file_due_test")

    with tempfile.TemporaryDirectory() as td:
        anchor_dir = Path(td)
        (anchor_dir / "calendar.csv").write_text("date\n2026-04-27\n", encoding="utf-8")
        old_dir = getattr(mod.core, "ANCHOR_FILE_DIR", "")
        mod.core.ANCHOR_FILE_DIR = str(anchor_dir)
        try:
            chain_id = "random-file-chain"
            target = date(2026, 4, 27)
            slots = mod.core._import_sibling("time_slots").resolve_time_slots_with_offsets(
                {"time_random": "rand(06:00..18:00/3)", "t": []},
                target,
                seed_base=chain_id,
            )
            first = mod.core.build_local_datetime(target, (slots[0][1], slots[0][2]))
            second = mod.core.build_local_datetime(target, (slots[1][1], slots[1][2]))
            parent = {
                "description": "random anchor file chain",
                "anchor_file": "calendar.csv@t=rand(06..18/3)",
                "anchor_mode": "skip",
                "link": 1,
                "chainID": chain_id,
                "due": mod.core.fmt_isoz(first.astimezone(timezone.utc)),
                "end": mod.core.fmt_isoz((first + timedelta(minutes=10)).astimezone(timezone.utc)),
            }
            child_due, meta, dnf = _compute_anchor_child_due(mod, parent)
            expect(not dnf, f"random anchor_file should not produce an anchor DNF: {dnf!r}")
            expect(child_due == second.astimezone(timezone.utc), f"random anchor_file slot was not preserved: {child_due!r} != {second!r}")
            expect(meta.get("target_field") == "due", f"unexpected random anchor_file metadata: {meta!r}")
        finally:
            mod.core.ANCHOR_FILE_DIR = old_dir


def test_on_modify_compute_anchor_child_due_from_multiple_file_times():
    """completion should retain independent times and select a later same-day occurrence from another file."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_multiple_anchor_file_due_test")

    with tempfile.TemporaryDirectory() as td:
        anchor_dir = Path(td)
        (anchor_dir / "morning.csv").write_text("date\n2026-04-25\n", encoding="utf-8")
        (anchor_dir / "afternoon.csv").write_text("date\n2026-04-25\n", encoding="utf-8")
        old_dir = getattr(mod.core, "ANCHOR_FILE_DIR", "")
        mod.core.ANCHOR_FILE_DIR = str(anchor_dir)
        try:
            due_utc = mod.core.build_local_datetime(date(2026, 4, 25), (9, 0)).astimezone(timezone.utc)
            end_utc = mod.core.build_local_datetime(date(2026, 4, 25), (10, 0)).astimezone(timezone.utc)
            parent = {
                "description": "multiple anchor file times",
                "anchor_file": "morning.csv@t=09:00 | afternoon.csv@t=15:00",
                "anchor_mode": "skip",
                "link": 1,
                "chainID": "cid",
                "due": mod.core.fmt_isoz(due_utc),
                "end": mod.core.fmt_isoz(end_utc),
            }
            child_due, meta, dnf = _compute_anchor_child_due(mod, parent)
            expected_due = mod.core.build_local_datetime(date(2026, 4, 25), (15, 0)).astimezone(timezone.utc)
            expect(not dnf, f"multiple anchor_file recurrence should not produce DNF: {dnf!r}")
            expect(child_due == expected_due, f"unexpected later same-day child due: {child_due!r}")
            expect(meta.get("target_field") == "due", f"unexpected multiple-file metadata: {meta!r}")
        finally:
            mod.core.ANCHOR_FILE_DIR = old_dir


def test_on_modify_compute_anchor_child_due_from_combined_anchor_sources():
    """completion should use the earliest next occurrence from anchor and anchor_file together."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_combined_anchor_due_test")

    with tempfile.TemporaryDirectory() as td:
        anchor_dir = Path(td)
        (anchor_dir / "calendar.csv").write_text("date\n2026-04-14\n2026-04-25\n", encoding="utf-8")
        old_dir = getattr(mod.core, "ANCHOR_FILE_DIR", "")
        mod.core.ANCHOR_FILE_DIR = str(anchor_dir)
        try:
            parent = {
                "description": "combined anchor chain",
                "anchor": "w:fri@t=09:00",
                "anchor_file": "calendar.csv@t=12:00",
                "anchor_mode": "skip",
                "link": 1,
                "chainID": "cid",
                "due": "2026-04-11T09:00:00Z",
                "end": "2026-04-12T12:00:00Z",
            }
            child_due, meta, dnf = _compute_anchor_child_due(mod, parent)
            expected_due = mod.core.build_local_datetime(date(2026, 4, 14), (12, 0)).astimezone(timezone.utc)
            expect(dnf is not None, f"combined anchor should preserve expression dnf, got {dnf!r}")
            expect(child_due == expected_due, f"unexpected combined anchor child due: {child_due!r}")
            expect(isinstance(meta, dict) and meta.get("target_field") == "due", f"unexpected combined anchor meta: {meta!r}")


            evaluator = _evaluator_for_fixture(
                parent,
                timezone=mod.core._LOCAL_TZ,
                anchor_file_dir=str(anchor_dir),
            )
            result = evaluator.select_mode(
                "skip",
                due_local=mod.core.to_local(mod.core.parse_dt_any(parent["due"])),
                end_local=mod.core.to_local(mod.core.parse_dt_any(parent["end"])),
                fallback_hhmm=(9, 0),
            )
            expect(
                result.selected_occurrence is not None
                and result.selected_occurrence.astimezone(timezone.utc) == child_due,
                f"evaluator/merged mode drifted from hook mode: {result!r} vs {child_due!r}",
            )
        finally:
            mod.core.ANCHOR_FILE_DIR = old_dir


def test_on_modify_compute_combined_overnight_sources_in_time_order():
    """Combined anchor sources should merge an overnight continuation before later file events."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_combined_overnight_due_test")

    with tempfile.TemporaryDirectory() as td:
        anchor_dir = Path(td)
        (anchor_dir / "calendar.csv").write_text("date\n2026-08-04\n", encoding="utf-8")
        old_dir = getattr(mod.core, "ANCHOR_FILE_DIR", "")
        mod.core.ANCHOR_FILE_DIR = str(anchor_dir)
        try:
            due = mod.core.build_local_datetime(date(2026, 8, 3), (22, 30))
            end = mod.core.build_local_datetime(date(2026, 8, 4), (6, 40))
            parent = {
                "description": "combined overnight sources",
                "anchor": "w:mon@t=22:30..06:30/7",
                "anchor_file": "calendar.csv@t=07:00",
                "anchor_mode": "skip",
                "link": 1,
                "chainID": "cid",
                "due": mod.core.fmt_isoz(due.astimezone(timezone.utc)),
                "end": mod.core.fmt_isoz(end.astimezone(timezone.utc)),
            }
            child_due, _meta, _dnf = _compute_anchor_child_due(mod, parent)
            expected_due = mod.core.build_local_datetime(date(2026, 8, 4), (7, 0)).astimezone(timezone.utc)
            expect(child_due == expected_due, f"combined overnight sources were not time-ordered: {child_due!r}")
        finally:
            mod.core.ANCHOR_FILE_DIR = old_dir


def test_hook_on_modify_timeline_keeps_anchor_match_after_shifted_anchor_file_child():
    """when anchor_file is shifted and anchor matches the original file date, timeline should still show the original date as the next future anchor."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_shifted_anchor_file_timeline_test")
    setattr(mod, "_collect_prev_two", lambda _task: [])
    from zoneinfo import ZoneInfo
    previous_tz_name = mod.core.LOCAL_TZ_NAME
    previous_tz = mod.core._LOCAL_TZ
    mod.core.LOCAL_TZ_NAME = "Europe/Bucharest"
    mod.core._LOCAL_TZ = ZoneInfo("Europe/Bucharest")

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
                child_due, _meta, dnf = _compute_anchor_child_due(mod, parent)
                expect(mod.core.fmt_isoz(child_due) == "2026-04-24T09:00:00Z", f"unexpected shifted child due: {mod.core.fmt_isoz(child_due)}")
                lines = _call_with_supported_kwargs(
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
        mod.core._LOCAL_TZ = previous_tz

    txt = _strip_markup("\n".join(lines))
    expect("Fri 2026-04-24 12:00" in txt, f"expected shifted anchor_file child in timeline: {txt!r}")
    expect("Sat 2026-04-25 12:00" in txt, f"expected original file date preserved via anchor match: {txt!r}")


def test_hook_on_modify_timeline_omits_shifted_anchor_file_dates_in_merged_stream():
    """merged anchor timelines should still omit shifted anchor_file dates when omit matches their shifted local date."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_shifted_anchor_file_omit_timeline_test")
    setattr(mod, "_collect_prev_two", lambda _task: [])
    from zoneinfo import ZoneInfo
    previous_tz_name = mod.core.LOCAL_TZ_NAME
    previous_tz = mod.core._LOCAL_TZ
    mod.core.LOCAL_TZ_NAME = "Europe/Bucharest"
    mod.core._LOCAL_TZ = ZoneInfo("Europe/Bucharest")

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
                lines = _call_with_supported_kwargs(
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
        mod.core._LOCAL_TZ = previous_tz

    txt = _strip_markup("\n".join(lines))
    expect("Thu 2026-04-30 12:00" in txt and "(omitted)" in txt, f"shifted omitted anchor_file date was not marked: {txt!r}")
    expect("Thu 2026-04-30 18:00" in txt and "(omitted)" in txt, f"shifted omitted anchor_file date was not marked: {txt!r}")
    expect("Mon 2026-05-04 12:00" in txt and "(omitted)" in txt, f"shifted omitted anchor_file date was not marked: {txt!r}")


def test_hook_on_modify_timeline_shows_anchor_side_omit_file_dates_in_merged_stream():
    """merged timelines should still show omitted anchor-side dates when omit_file blocks them."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_anchor_side_omit_file_timeline_test")
    if hasattr(mod, "_collect_prev_two"):
        setattr(mod, "_collect_prev_two", lambda _task: [])
    from zoneinfo import ZoneInfo
    previous_tz_name = mod.core.LOCAL_TZ_NAME
    previous_tz = mod.core._LOCAL_TZ
    mod.core.LOCAL_TZ_NAME = "Europe/Bucharest"
    mod.core._LOCAL_TZ = ZoneInfo("Europe/Bucharest")

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
                lines = _call_with_supported_kwargs(
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
        mod.core._LOCAL_TZ = previous_tz

    txt = _strip_markup("\n".join(lines))
    expect("Tue 2026-05-05" in txt, f"expected omitted anchor-side date to remain visible: {txt!r}")
    expect("(omitted)" in txt, f"expected merged timeline omitted marker for anchor-side omit_file date: {txt!r}")


def test_on_modify_compute_anchor_child_due_skips_omit_date():
    """anchor completion should skip omitted anchor dates and choose the next valid one."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_anchor_omit_skip_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    parent = {
            "anchor": "w:mon,wed,fri@t=09:00",
            "omit": "w:wed",
            "anchor_mode": "skip",
            "due": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 6), (9, 0))),
            "end": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 6), (10, 0))),
            "chainID": "omit1234",
        }
    child_due, meta, _dnf = _compute_anchor_child_due(mod, parent)
    expected = mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 10), (9, 0)))
    expect(mod.core.fmt_isoz(child_due) == expected, f"unexpected next due with omit: {mod.core.fmt_isoz(child_due)}")
    expect(meta.get("target_field") == "due", f"expected due target field: {meta}")

    evaluator = _evaluator_for_fixture(parent, timezone_value=mod.core._LOCAL_TZ)
    result = evaluator.select_mode(
        "skip",
        due_local=mod.core.to_local(mod.core.parse_dt_any(parent["due"])),
        end_local=mod.core.to_local(mod.core.parse_dt_any(parent["end"])),
        fallback_hhmm=(9, 0),
    )
    expect(
        result.selected_occurrence is not None
        and result.selected_occurrence.astimezone(timezone.utc) == child_due,
        f"omit evaluator drifted from hook: {result!r} vs {child_due!r}",
    )


def test_on_modify_compute_anchor_child_due_accepts_scheduled_after_due():
    """anchor completion should not crash when scheduled is later than due."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_anchor_sched_after_due_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    child_due, meta, _dnf = _compute_anchor_child_due(mod,
        {
            "anchor": "w:mon..sun@t=09:00",
            "anchor_mode": "all",
            "due": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 6), (9, 0))),
            "scheduled": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 8), (12, 0))),
            "end": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 8), (10, 0))),
            "chainID": "abcd1234",
        }
    )

    expected = mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 7), (9, 0)))
    expect(mod.core.fmt_isoz(child_due) == expected, f"unexpected next due with scheduled-after-due: {mod.core.fmt_isoz(child_due)}")
    expect(meta.get("target_field") == "due", f"expected due target field when due is present: {meta}")


def test_on_modify_compute_counted_random_advances_within_period():
    """Counted-random completion should emit the remaining selection in the same period."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_counted_random_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    dnf = mod.core.validate_anchor_expr_strict("m:2rand")
    seed = date(2026, 1, 1)
    first, _meta = mod.core.next_after_expr(
        dnf,
        seed,
        default_seed=seed,
        seed_base="abcd1234",
    )
    expected, _meta = mod.core.next_after_expr(
        dnf,
        first,
        default_seed=seed,
        seed_base="abcd1234",
    )
    child_due, meta, _dnf = _compute_anchor_child_due(mod,
        {
            "anchor": "m:2rand",
            "anchor_mode": "skip",
            "due": mod.core.fmt_isoz(mod.core.build_local_datetime(first, (9, 0))),
            "end": mod.core.fmt_isoz(mod.core.build_local_datetime(first, (10, 0))),
            "chainID": "abcd1234",
            "link": 1,
        }
    )
    expect(mod.core.to_local(child_due).date() == expected, f"unexpected counted-random child due: {child_due}")
    expect(expected.month == first.month, f"second counted pick should remain in the same month: {first}, {expected}")
    expect(meta.get("target_field") == "due", f"expected due target field: {meta}")


def test_on_modify_compute_anchor_child_due_unsatisfiable_omit_fails():
    """anchor completion should fail cleanly when omit removes every future anchor date."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_anchor_omit_unsat_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    try:
        _compute_anchor_child_due(mod,
            {
                "anchor": "w:mon",
                "omit": "w:mon",
                "anchor_mode": "skip",
                "due": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 6), (9, 0))),
                "end": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 6), (10, 0))),
                "chainID": "omit1234",
            }
        )
        expect(False, "expected unsatisfiable omit to fail")
    except ValueError as e:
        expect(
            "No valid anchor occurrences found after applying omit rules." in str(e),
            f"unexpected unsatisfiable omit error: {e}",
        )


def test_on_modify_completion_build_and_spawn_child_happy_path():
    """completion spawn wrapper should return child info and stamp nextLink when verified."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_completion_spawn_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    new = {
        "uuid": "00000000-0000-4000-8000-000000000111",
        "status": "completed",
        "chainID": "abcd1234",
        "link": 1,
        "cp": "P1D",
    }
    child = {"uuid": "00000000-0000-4000-8000-000000000222", "link": 2}
    from nautical_core.chain_generation import ChainGenerationService

    class StubGeneration(ChainGenerationService):
        def build_child_draft(self, parent, child_due, child_field, next_link_no, *_args, **_kwargs):
            return _task_draft({
                **child,
                "description": "typed child fixture",
                "chain": "on",
                "status": "pending",
                "chainID": parent.observation.to_mapping()["chainID"],
                "link": next_link_no,
                "cp": "P1D",
                "anchor_mode": "skip",
                child_field: child_due,
            })

    generation_effects = mod._module("modify_generation_effects")
    original_generation = generation_effects.chain_generation_service
    generation_effects.chain_generation_service = lambda _host: StubGeneration.from_core(mod.core)
    spawn_effects = mod._module("modify_spawn_effects")
    original_spawn = spawn_effects.spawn_child_atomic
    spawn_effects.spawn_child_atomic = lambda _ports, _child, _parent, **_kwargs: ("beeswax", set(), True, False, None, "si_test")
    try:
        out = mod._completion_effects.build_and_spawn_child(
            new,
            child_due=mod.core.now_utc(),
            child_field="due",
            next_no=2,
            parent_short="00000000",
            kind="cp",
            cpmax=0,
            until_dt=None,
        )
    finally:
        generation_effects.chain_generation_service = original_generation
        spawn_effects.spawn_child_atomic = original_spawn
    expect(bool(out), f"expected spawn result, got {out}")
    expect(out.child.get("uuid") == child["uuid"], f"unexpected child payload: {out}")
    expect(out.child.get("link") == 2, f"typed child lost link: {out}")
    expect(out.child_short == "beeswax", f"unexpected child short: {out}")
    expect(out.verified is True and out.deferred_spawn is False, f"unexpected verification state: {out}")
    expect(out.spawn_intent_id == "si_test", f"unexpected spawn intent id: {out}")
    expect(new.get("nextLink") == "beeswax", f"verified spawn should stamp nextLink: {new}")


def test_on_modify_completion_spawn_exception_is_retryable_with_reason():
    """A spawn command exception must remain typed and actionable for finalization."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_completion_spawn_exception_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    parent = {
        "uuid": "00000000-0000-4000-8000-000000000121",
        "status": "completed",
        "chainID": "spawn121",
        "link": 1,
        "cp": "P1D",
    }
    child = {"uuid": "00000000-0000-4000-8000-000000000122", "link": 2}
    from nautical_core.chain_generation import ChainGenerationService

    class StubGeneration(ChainGenerationService):
        def build_child_draft(self, parent, child_due, child_field, next_link_no, *_args, **_kwargs):
            return _task_draft({
                **child,
                "description": "typed child fixture",
                "chain": "on",
                "status": "pending",
                "chainID": parent.observation.to_mapping()["chainID"],
                "link": next_link_no,
                "cp": "P1D",
                "anchor_mode": "skip",
                child_field: child_due,
            })

    generation_effects = mod._module("modify_generation_effects")
    original_generation = generation_effects.chain_generation_service
    spawn_effects = mod._module("modify_spawn_effects")
    original_spawn = spawn_effects.spawn_child_atomic
    original_panel = mod._panel
    original_print = mod._print_task
    panels = []
    try:
        generation_effects.chain_generation_service = lambda _host: StubGeneration.from_core(mod.core)
        spawn_effects.spawn_child_atomic = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Taskwarrior lock busy"))
        mod._panel = lambda title, rows, *, kind=None: panels.append((title, list(rows), kind))
        mod._print_task = lambda _task: None
        result = mod._completion_effects.build_and_spawn_child(
            parent,
            child_due=mod.core.now_utc(),
            child_field="due",
            next_no=2,
            parent_short="00000000",
            kind="cp",
            cpmax=0,
            until_dt=None,
        )
    finally:
        generation_effects.chain_generation_service = original_generation
        spawn_effects.spawn_child_atomic = original_spawn
        mod._panel = original_panel
        mod._print_task = original_print

    expect(result is not None and result.outcome_state == "retryable", f"spawn exception lost typed state: {result!r}")
    expect("Taskwarrior lock busy" in result.reason, f"spawn exception lost reason: {result!r}")
    expect(not panels, f"spawn helper should not render before finalization: {panels!r}")




def test_on_modify_build_child_scheduled_only_keeps_due_unset_and_carries_wait():
    """scheduled-only child spawn should carry relative dates from scheduled."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_build_child_sched_only_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    parent = {
        "uuid": "00000000-0000-4000-8000-000000000333",
        "status": "completed",
        "link": 1,
        "scheduled": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 1), (9, 0))),
        "wait": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 1), (7, 0))),
        "until": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 2), (17, 0))),
        "cp": "1d",
        "chainID": "cid_sched",
    }
    child_due = mod.core.build_local_datetime(date(2025, 1, 2), (9, 0))
    child = _build_child_draft_for_test(mod,
        parent,
        child_due,
        "scheduled",
        2,
        "beef",
        "cp",
        0,
        None,
    )
    expect(not child.get("due"), f"scheduled-only child should not get due: {child}")
    expect(child.get("scheduled") == mod.core.fmt_isoz(child_due), f"unexpected child scheduled: {child}")
    wait_local = mod.core.to_local(mod.core.parse_dt_any(child.get("wait")))
    expect((wait_local.hour, wait_local.minute) == (7, 0), f"unexpected carried wait: {wait_local}")
    until_local = mod.core.to_local(mod.core.parse_dt_any(child.get("until")))
    expect(
        until_local.date() == date(2025, 1, 3) and (until_local.hour, until_local.minute) == (17, 0),
        f"unexpected carried until: {until_local}",
    )


def test_on_modify_render_anchor_completion_feedback_wrapper():
    """anchor completion feedback wrapper should delegate and emit a preview panel title."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_anchor_feedback_wrapper_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    mod._SHOW_TIMELINE_GAPS = False
    mod._CHAIN_COLOR_PER_CHAIN = False
    mod._append_next_wait_sched_rows = lambda *_a, **_k: None
    mod._format_root_and_age = lambda *_a, **_k: "abcd1234"
    mod._timeline_lines = lambda *_a, **_k: []

    captured = {}
    mod._panel_line = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("line mode should not be used"))
    mod._panel = lambda title, fb, **_k: captured.update({"title": title, "fb": list(fb)})

    prev_panel_mode = mod.core.PANEL_MODE
    prev_show_analytics = mod.core.SHOW_ANALYTICS
    prev_anchor_presets = getattr(mod.core, "ANCHOR_PRESETS", {})
    prev_omit_presets = getattr(mod.core, "OMIT_PRESETS", {})
    try:
        mod.core.PANEL_MODE = "panel"
        mod.core.SHOW_ANALYTICS = False
        mod.core.ANCHOR_PRESETS = {"payday": "m:15,-1bd"}
        mod.core.OMIT_PRESETS = {"wed": "w:wed"}
        mod._presentation_effects.render_anchor_completion_feedback(
            new={"anchor": "@payday", "omit": "@wed", "anchor_mode": "skip", "uuid": "00000000-0000-4000-8000-000000000111", "chainID": "abcd1234"},
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
            analytics_advice="chain looks healthy but should be hidden",
            integrity_warnings=None,
            base_no=1,
        )
    finally:
        mod.core.PANEL_MODE = prev_panel_mode
        mod.core.SHOW_ANALYTICS = prev_show_analytics
        mod.core.ANCHOR_PRESETS = prev_anchor_presets
        mod.core.OMIT_PRESETS = prev_omit_presets

    expect("title" in captured, "expected preview panel emission")
    expect("Next anchor" in captured["title"], f"unexpected panel title: {captured}")
    fb = captured.get("fb") or []
    expect(any(k == "Omit" and "@wed" in str(v) for k, v in fb), f"expected omit row in anchor feedback: {fb}")
    expect(any(k == "Preset" and "@payday → m:15,-1bd" in str(v) for k, v in fb), f"expected preset expansion row in anchor feedback: {fb}")
    expect(any(k == "Natural" and "skip @wed" in str(v) for k, v in fb), f"expected natural omit row in anchor feedback: {fb}")
    expect(any(k == "Result" and "Applied now" in str(v) for k, v in fb), f"expected applied lifecycle result in anchor feedback: {fb}")
    expect(not any(k == "Analytics" for k, _v in fb), f"analytics row should be hidden when show_analytics is false: {fb}")


def test_on_modify_reports_business_calendar_displacement():
    """Completion feedback should report the captured calendar roll in every panel mode."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_calendar_displacement_test")
    mod._SHOW_TIMELINE_GAPS = False
    mod._CHAIN_COLOR_PER_CHAIN = False
    mod._append_next_wait_sched_rows = lambda *_a, **_k: None
    mod._format_root_and_age = lambda *_a, **_k: "abcd1234"
    mod._timeline_lines = lambda *_a, **_k: []
    mod._panel_line = lambda *_a, **_k: None
    panels = []
    mod._panel = lambda title, rows, **_kwargs: panels.append((title, list(rows)))

    policy = mod.core.resolve_business_calendar_config(
        {'work': {'anchor': 'w:mon..fri', 'omit': 'y:04-24'}}
    )['work']
    dnf = mod.core.validate_anchor_expr_strict('y:04-24@nbd@t=09:00')
    previous_mode = mod.core.PANEL_MODE
    try:
        mod.core.PANEL_MODE = "minimal"
        with mod.core.use_business_calendar(policy), mod.core.capture_business_calendar_displacements():
            child_date, _meta = mod.core.next_after_expr(
                dnf,
                date(2026, 4, 20),
                date(2026, 4, 20),
            )
            child_due = mod.core.build_local_datetime(child_date, (9, 0))
            mod._presentation_effects.render_anchor_completion_feedback(
                new={
                    "anchor": "y:04-24@nbd@t=09:00",
                    "anchor_mode": "skip",
                    "bc": "work",
                    "uuid": "00000000-0000-4000-8000-000000000127",
                    "chainID": "calendar-chain",
                },
                child={"uuid": "00000000-0000-4000-8000-000000000128"},
                child_due=child_due,
                child_short="beeswax",
                next_no=2,
                parent_short="00000000",
                cap_no=None,
                finals=[],
                now_utc=mod.core.now_utc(),
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
        mod.core.PANEL_MODE = previous_mode

    calendar_panels = [rows for title, rows in panels if title == "⚓ Business calendar adjusted"]
    expect(len(calendar_panels) == 1, f"completion should emit one displacement panel: {panels!r}")
    rows = calendar_panels[0]
    expect(("Calendar", "work") in rows, f"calendar name missing: {rows!r}")
    expect(("Original", "Fri 2026-04-24") in rows, f"original occurrence missing: {rows!r}")
    expect(("Adjusted", "Mon 2026-04-27 (+3d)") in rows, f"adjusted occurrence missing: {rows!r}")


def test_on_modify_anchor_feedback_warns_when_timed_anchor_uses_utc_fallback():
    """Timed anchors should show a warning when timezone data is unavailable."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_anchor_timezone_warning_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    mod._SHOW_TIMELINE_GAPS = False
    mod._CHAIN_COLOR_PER_CHAIN = False
    mod._append_next_wait_sched_rows = lambda *_a, **_k: None
    mod._format_root_and_age = lambda *_a, **_k: "abcd1234"
    mod._timeline_lines = lambda *_a, **_k: []

    captured = {}
    mod._panel = lambda title, fb, **_k: captured.update({"title": title, "fb": list(fb)})
    prev_local_tz = getattr(mod.core, "_LOCAL_TZ", None)
    prev_panel_mode = mod.core.PANEL_MODE
    try:
        mod.core._LOCAL_TZ = None
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
        mod.core._LOCAL_TZ = prev_local_tz
        mod.core.PANEL_MODE = prev_panel_mode

    fb = captured.get("fb") or []
    expect(any("Timezone data unavailable" in str(v) for k, v in fb if k == "Integrity"), f"missing timezone fallback warning: {fb}")


def test_on_modify_render_anchor_file_completion_feedback_wrapper():
    """anchor_file completion feedback should not crash when anchor DNF is absent."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_anchor_file_feedback_wrapper_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    mod._SHOW_TIMELINE_GAPS = False
    mod._CHAIN_COLOR_PER_CHAIN = False
    mod._append_next_wait_sched_rows = lambda *_a, **_k: None
    mod._format_root_and_age = lambda *_a, **_k: "abcd1234"
    mod._timeline_lines = lambda *_a, **_k: []

    captured = {}
    mod._panel_line = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("line mode should not be used"))
    mod._panel = lambda title, fb, **_k: captured.update({"title": title, "fb": list(fb)})

    prev_panel_mode = mod.core.PANEL_MODE
    prev_anchor_dir = mod.core.ANCHOR_FILE_DIR
    anchor_dir = tempfile.TemporaryDirectory(prefix="nautical-anchor-feedback-")
    Path(anchor_dir.name, "calendar.csv").write_text("date\n2025-01-01\n", encoding="utf-8")
    try:
        mod.core.PANEL_MODE = "panel"
        mod.core.ANCHOR_FILE_DIR = anchor_dir.name
        mod._presentation_effects.render_anchor_completion_feedback(
            new={"anchor_file": "calendar.csv@t=12:00", "anchor_mode": "skip", "uuid": "00000000-0000-4000-8000-000000000333", "chainID": "abcd1234"},
            child={"uuid": "00000000-0000-4000-8000-000000000444"},
            child_due=mod.core.now_utc(),
            child_short="beeswax",
            next_no=2,
            parent_short="00000000",
            cap_no=None,
            finals=[],
            now_utc=mod.core.now_utc(),
            until_dt=None,
            until_cap_no=None,
            dnf=None,
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
        mod.core.PANEL_MODE = prev_panel_mode
        mod.core.ANCHOR_FILE_DIR = prev_anchor_dir
        anchor_dir.cleanup()

    expect("title" in captured, "expected anchor_file preview panel emission")
    expect("Next anchor" in captured["title"], f"unexpected anchor_file panel title: {captured}")
    fb = captured.get("fb") or []
    expect(any(k == "Anchor file" for k, _v in fb), f"expected anchor_file row in feedback: {fb}")



def test_on_modify_render_cp_completion_feedback_wrapper():
    """CP completion feedback wrapper should delegate and emit a preview panel title."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_cp_feedback_wrapper_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    mod._SHOW_TIMELINE_GAPS = False
    mod._CHAIN_COLOR_PER_CHAIN = False
    mod._append_next_wait_sched_rows = lambda *_a, **_k: None
    mod._format_root_and_age = lambda *_a, **_k: "abcd1234"
    mod._timeline_lines = lambda *_a, **_k: []

    captured = {}
    mod._panel_line = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("line mode should not be used"))
    mod._panel = lambda title, fb, **_k: captured.update({"title": title, "fb": list(fb)})

    prev_panel_mode = mod.core.PANEL_MODE
    try:
        mod.core.PANEL_MODE = "panel"
        mod._presentation_effects.render_cp_completion_feedback(
            new={"cp": "3d,20d,7d", "uuid": "00000000-0000-4000-8000-000000000111", "chainID": "abcd1234"},
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
            meta={"cp_sequence_step": 3, "cp_sequence_len": 3},
            deferred_spawn=False,
            spawn_intent_id=None,
            chain_by_short=None,
            analytics_advice=None,
            integrity_warnings=None,
            base_no=1,
        )
    finally:
        mod.core.PANEL_MODE = prev_panel_mode

    expect("title" in captured, "expected preview panel emission")
    expect("Next link" in captured["title"], f"unexpected panel title: {captured}")
    expect(("Step", "3/3 (7d)") in captured["fb"], f"expected sequence step period in feedback rows: {captured}")
    expect(any(k == "Result" and "Applied now" in str(v) for k, v in captured["fb"]), f"expected applied lifecycle result in CP feedback: {captured}")


def test_on_modify_completion_panel_distinguishes_expiration_and_chain_boundaries():
    """Completion panels should show the next expiration and one effective last occurrence."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_expiration_boundary_feedback_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    mod._SHOW_TIMELINE_GAPS = False
    mod._CHAIN_COLOR_PER_CHAIN = False
    mod._append_next_wait_sched_rows = lambda *_a, **_k: None
    mod._format_root_and_age = lambda *_a, **_k: "abcd1234"
    mod._timeline_lines = lambda *_a, **_k: []

    child_due = datetime(2026, 8, 10, 10, 0, tzinfo=timezone.utc)
    child_expires = child_due + timedelta(hours=8)
    chain_end = child_due + timedelta(days=35)
    last_by_max = child_due + timedelta(days=60)
    last_by_end = child_due + timedelta(days=28)
    captured = {}
    mod._panel = lambda title, rows, **_kwargs: captured.update({"title": title, "rows": list(rows)})

    previous_mode = mod.core.PANEL_MODE
    try:
        mod.core.PANEL_MODE = "panel"
        mod._presentation_effects.render_cp_completion_feedback(
            new={
                "cp": "7d",
                "chainMax": 10,
                "chainUntil": mod.core.fmt_isoz(chain_end),
                "uuid": "00000000-0000-4000-8000-000000000143",
                "chainID": "abcd1234",
            },
            child={
                "uuid": "00000000-0000-4000-8000-000000000144",
                "until": mod.core.fmt_isoz(child_expires),
            },
            child_due=child_due,
            child_short="beeswax",
            next_no=2,
            parent_short="00000000",
            cap_no=6,
            finals=[("max", last_by_max), ("until", last_by_end)],
            now_utc=datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc),
            until_dt=chain_end,
            until_cap_no=6,
            meta={},
            deferred_spawn=False,
            spawn_intent_id=None,
            chain_by_short=None,
            analytics_advice=None,
            integrity_warnings=None,
            base_no=1,
        )
    finally:
        mod.core.PANEL_MODE = previous_mode

    rows = captured.get("rows") or []
    add_validation = mod.core._import_sibling("add_validation")
    expected_policy = add_validation.describe_native_until_carry(
        child_expires,
        child_due,
        to_local=mod.core.to_local,
    )
    expect(("Expiration", expected_policy) in rows, f"expiration policy missing: {rows!r}")
    expect(any(label == "Next expires" for label, _value in rows), f"next expiration missing: {rows!r}")
    expect(("Chain cap", "#10") in rows, f"chain cap missing: {rows!r}")
    expect(
        any(label == "Chain end point" and "2026-09-14" in value for label, value in rows),
        f"chain end point missing: {rows!r}",
    )
    last_rows = [(label, value) for label, value in rows if label == "Last occurrence"]
    expect(len(last_rows) == 1, f"expected one effective last occurrence: {rows!r}")
    expect("2026-09-07" in last_rows[0][1], f"earlier boundary should determine last occurrence: {rows!r}")
    expect(not any(str(label).startswith("Final (") for label, _value in rows), f"legacy final label remains: {rows!r}")


def test_on_modify_render_cp_completion_feedback_random_selected_interval():
    """CP random completion feedback should show the selected interval, not the raw rand expression."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_cp_random_feedback_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    mod._SHOW_TIMELINE_GAPS = False
    mod._CHAIN_COLOR_PER_CHAIN = False
    mod._append_next_wait_sched_rows = lambda *_a, **_k: None
    mod._format_root_and_age = lambda *_a, **_k: "abcd1234"
    mod._timeline_lines = lambda *_a, **_k: []

    captured = {}
    mod._panel_line = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("line mode should not be used"))
    mod._panel = lambda title, fb, **_k: captured.update({"title": title, "fb": list(fb)})

    prev_panel_mode = mod.core.PANEL_MODE
    cp = "rand(11d..14d)"
    chain_id = "abcd1234"
    try:
        mod.core.PANEL_MODE = "panel"
        mod._presentation_effects.render_cp_completion_feedback(
            new={"cp": cp, "link": 2, "uuid": "00000000-0000-4000-8000-000000000111", "chainID": chain_id},
            child={"uuid": "00000000-0000-4000-8000-000000000222"},
            child_due=mod.core.now_utc(),
            child_short="beeswax",
            next_no=3,
            parent_short="00000000",
            cap_no=None,
            finals=[],
            now_utc=mod.core.now_utc(),
            until_dt=None,
            until_cap_no=None,
            meta={"cp_sequence_step": 1, "cp_sequence_len": 1},
            deferred_spawn=False,
            spawn_intent_id=None,
            chain_by_short=None,
            analytics_advice=None,
            integrity_warnings=None,
            base_no=2,
        )
    finally:
        mod.core.PANEL_MODE = prev_panel_mode

    step_rows = [v for k, v in captured.get("fb", []) if k == "Step"]
    expect(step_rows, f"expected random cp step row: {captured}")
    expect("rand(" not in str(step_rows[0]), f"expected selected interval in random cp step row: {captured}")
    selected = mod.core.cp_sequence_interval_for_link(cp, 2, chain_id)
    selected_days = int(selected.total_seconds() // 86400)
    expect(
        str(step_rows[0]) == f"1/1 ({selected_days}d)",
        f"expected chain-scoped random interval as days: {captured}",
    )


def test_on_modify_render_cp_completion_feedback_jitter_selected_interval():
    """CP jitter completion feedback should show the selected interval, not the raw jitter expression."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_cp_jitter_feedback_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    mod._SHOW_TIMELINE_GAPS = False
    mod._CHAIN_COLOR_PER_CHAIN = False
    mod._append_next_wait_sched_rows = lambda *_a, **_k: None
    mod._format_root_and_age = lambda *_a, **_k: "abcd1234"
    mod._timeline_lines = lambda *_a, **_k: []

    captured = {}
    mod._panel_line = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("line mode should not be used"))
    mod._panel = lambda title, fb, **_k: captured.update({"title": title, "fb": list(fb)})

    prev_panel_mode = mod.core.PANEL_MODE
    try:
        mod.core.PANEL_MODE = "panel"
        mod._presentation_effects.render_cp_completion_feedback(
            new={"cp": "15d~0d", "link": 2, "uuid": "00000000-0000-4000-8000-000000000111", "chainID": "abcd1234"},
            child={"uuid": "00000000-0000-4000-8000-000000000222"},
            child_due=mod.core.now_utc(),
            child_short="beeswax",
            next_no=3,
            parent_short="00000000",
            cap_no=None,
            finals=[],
            now_utc=mod.core.now_utc(),
            until_dt=None,
            until_cap_no=None,
            meta={"cp_sequence_step": 1, "cp_sequence_len": 1},
            deferred_spawn=False,
            spawn_intent_id=None,
            chain_by_short=None,
            analytics_advice=None,
            integrity_warnings=None,
            base_no=2,
        )
    finally:
        mod.core.PANEL_MODE = prev_panel_mode

    step_rows = [v for k, v in captured.get("fb", []) if k == "Step"]
    expect(step_rows, f"expected jitter cp step row: {captured}")
    expect(str(step_rows[0]) == "1/1 (15d)", f"expected selected interval in jitter cp step row: {captured}")


def test_on_modify_render_cp_completion_feedback_text_mode():
    """CP completion feedback should use concise ASCII text output in text mode."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_cp_feedback_text_mode_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    mod._SHOW_TIMELINE_GAPS = False
    mod._CHAIN_COLOR_PER_CHAIN = False
    mod._append_next_wait_sched_rows = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("wait-schedule rows should not be built in text mode"))
    mod._format_root_and_age = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("root formatting should not run in text mode"))
    mod._timeline_lines = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("timeline should not be built in text mode"))

    captured = {}
    mod._panel_line = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("panel line should not be used in text mode"))
    mod._panel = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("panel should not be used in text mode"))
    mod._text_line = lambda line, **_k: captured.update({"line": line, "kwargs": dict(_k)})

    prev_panel_mode = mod.core.PANEL_MODE
    try:
        mod.core.PANEL_MODE = "text"
        mod._presentation_effects.render_cp_completion_feedback(
            new={"cp": "P1D", "uuid": "00000000-0000-4000-8000-000000000111", "chainID": "abcd1234"},
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
            meta={},
            deferred_spawn=False,
            spawn_intent_id=None,
            chain_by_short=None,
            analytics_advice=None,
            integrity_warnings=None,
            base_no=1,
        )
    finally:
        mod.core.PANEL_MODE = prev_panel_mode

    expect("line" in captured, f"expected text-mode line emission, got {captured}")
    txt = mod.core.strip_rich_markup(str(captured["line"]))
    expect("\n" in txt, f"expected stacked text payload, got {txt!r}")
    expect("00000000 ✓" in txt, f"expected parent status line in text payload, got {txt!r}")
    expect("Next ⛓ #2 beeswax" in txt, f"expected next-link line in text payload, got {txt!r}")
    expect("Period: P1D" in txt, f"expected summary line in text payload, got {txt!r}")
    expect("Result: Applied now" in txt, f"expected lifecycle result in text payload, got {txt!r}")
    expect(captured.get("kwargs", {}).get("kind") == "preview_cp", f"unexpected text line kwargs: {captured}")
    expect(captured.get("kwargs", {}).get("markup_body") is True, f"unexpected markup handling: {captured}")


def test_on_add_preview_hard_cap():
    """on-add preview loop should respect hard cap even with large preview setting."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_preview_cap_test")

    mod.UPCOMING_PREVIEW = 1000
    mod._PREVIEW_HARD_CAP = 3

    first_date_local = date(2025, 1, 6)
    first_hhmm = (9, 0)

    def _step_once(prev_date):
        return prev_date + timedelta(days=7)

    preview = []
    cur_dt = core.to_local(core.build_local_datetime(first_date_local, first_hhmm))
    for i in range(mod._PREVIEW_HARD_CAP + 5):
        if i >= mod._PREVIEW_HARD_CAP:
            break
        nxt_date = _step_once(cur_dt.date())
        cur_dt = core.to_local(core.build_local_datetime(nxt_date, first_hhmm))
        preview.append(core.fmt_dt_local(cur_dt.astimezone(timezone.utc)))

    preview_limit = max(0, min(mod.UPCOMING_PREVIEW, 10**9, 10**9, mod._PREVIEW_HARD_CAP))
    expect(preview_limit == 3, f"unexpected preview limit: {preview_limit}")
    expect(len(preview) == 3, "preview hard cap should limit preview length")


def test_on_add_flushes_stdout():
    """on-add should flush stdout after emitting JSON."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_flush_test")

    class _FlushIO(io.StringIO):
        def __init__(self):
            super().__init__()
            self.flushed = False

        def flush(self):
            self.flushed = True
            return super().flush()

    task = {"uuid": "00000000-0000-4000-8000-000000000111", "status": "pending"}
    raw = json.dumps(task)
    stdin = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    stdout = _FlushIO()
    stderr = io.StringIO()
    orig_stdin, orig_stdout, orig_stderr = sys.stdin, sys.stdout, sys.stderr
    try:
        sys.stdin = stdin
        sys.stdout = stdout
        sys.stderr = stderr
        mod.main()
    finally:
        sys.stdin, sys.stdout, sys.stderr = orig_stdin, orig_stdout, orig_stderr

    expect(stdout.flushed, "stdout.flush should be called")


def test_on_add_profiler_lazy_init():
    """on-add should not register profiler when disabled."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_profiler_lazy_test")
    mod._PROFILE_LEVEL = 0

    called = {"ok": False}
    orig_register = mod.atexit.register
    mod.atexit.register = lambda *_a, **_k: called.update(ok=True)

    task = {"uuid": "00000000-0000-4000-8000-000000000111", "status": "pending"}
    raw = json.dumps(task)
    stdin = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()
    orig_stdin, orig_stdout, orig_stderr = sys.stdin, sys.stdout, sys.stderr
    try:
        sys.stdin = stdin
        sys.stdout = stdout
        sys.stderr = stderr
        mod.main()
    finally:
        sys.stdin, sys.stdout, sys.stderr = orig_stdin, orig_stdout, orig_stderr
        mod.atexit.register = orig_register

    expect(not called["ok"], "profiler should not register when disabled")


def test_on_add_format_anchor_rows_numbers_upcoming_from_three_with_next_anchor():
    """on-add anchor formatting should number upcoming entries from 3 when Next anchor exists."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_format_rows_next_anchor_test")
    if not hasattr(mod, "_format_anchor_rows"):
        raise AssertionError("on-add hook does not expose _format_anchor_rows")

    rows = [
        ("Pattern", "w:mon"),
        ("First due", "2025-01-01 09:00"),
        ("Next anchor", "2025-01-08 09:00"),
        ("Upcoming", "[cyan]2025-01-15 09:00[/]\n[cyan]2025-01-22 09:00[/]"),
        ("Delta", "+7d"),
        ("Chain", "enabled"),
    ]
    out = mod._format_anchor_rows(rows)
    txt = _strip_markup("\n".join(v for _, v in out if isinstance(v, str)))
    expect(" 3 ▸ 2025-01-15 09:00" in txt, f"expected #3 upcoming marker, got: {txt!r}")
    expect(" 4 ▸ 2025-01-22 09:00" in txt, f"expected #4 upcoming marker, got: {txt!r}")


def test_on_add_format_anchor_rows_numbers_upcoming_from_two_without_next_anchor():
    """on-add anchor formatting should number upcoming entries from 2 without Next anchor."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_format_rows_no_next_anchor_test")
    if not hasattr(mod, "_format_anchor_rows"):
        raise AssertionError("on-add hook does not expose _format_anchor_rows")

    rows = [
        ("Pattern", "w:mon"),
        ("First due", "2025-01-01 09:00"),
        ("Upcoming", "[cyan]2025-01-08 09:00[/]"),
        ("Delta", "+7d"),
        ("Other", "x"),
    ]
    out = mod._format_anchor_rows(rows)
    txt = _strip_markup("\n".join(v for _, v in out if isinstance(v, str)))
    expect(" 2 ▸ 2025-01-08 09:00" in txt, f"expected #2 upcoming marker, got: {txt!r}")
    expect("Δ +7d" in txt, f"expected inline delta in first-due row, got: {txt!r}")


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


def test_on_exit_emit_exit_feedback_reaches_stdout_contract():
    """on-exit failing-hook feedback should still reach stdout even after stdout redirection."""
    hook = _find_hook_file("on-exit.nautical")
    mod = _load_hook_module(hook, "_nautical_on_exit_emit_feedback_test")

    class _DevNullLike:
        def write(self, _s):
            return None
        def flush(self):
            return None

    fake_stdout = io.StringIO()
    fake_stderr = io.StringIO()
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    orig_dunder_stdout = sys.__stdout__
    try:
        sys.stdout = _DevNullLike()
        sys.stderr = fake_stderr
        sys.__stdout__ = fake_stdout
        mod._emit_exit_feedback("[nautical] test feedback")
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        sys.__stdout__ = orig_dunder_stdout

    expect("[nautical] test feedback" in fake_stdout.getvalue(), "feedback should reach stdout contract stream")
    expect("[nautical] test feedback" in fake_stderr.getvalue(), "feedback should also remain visible on stderr")


def test_on_modify_state_files_use_dedicated_dir():
    """on-modify lifecycle outbox state should live under .nautical-state."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_state_dir_test")
    mod._load_core()
    outbox = mod._module("lifecycle_outbox")
    path = outbox.lifecycle_outbox_path(Path(mod.TW_DATA_DIR))
    expect(path.parent.name == ".nautical-state", f"unexpected outbox dir: {path}")


def test_on_modify_recompleted_task_with_nextlink_skips_spawn():
    """Re-completing a reactivated task should not spawn when nextLink already exists."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_recomplete_skip_spawn_test")
    mod._SHOW_TIMELINE_GAPS = False
    mod._SHOW_ANALYTICS = False
    mod._CHECK_CHAIN_INTEGRITY = False

    called = {"spawn": False}

    def _spawn_child_atomic_stub(_child, _parent):
        called["spawn"] = True
        return ("beeswax", set(), False, True, "queued", "si_test1")

    spawn_effects = mod._module("modify_spawn_effects")
    original_spawn = spawn_effects.spawn_child_atomic
    spawn_effects.spawn_child_atomic = lambda _ports, child, parent, **_kwargs: _spawn_child_atomic_stub(child, parent)

    old = {
        "uuid": "00000000-0000-4000-8000-000000000111",
        "status": "pending",
        "description": "reactivated duplicate guard",
        "cp": "P1D",
        "chainID": "abcd1234",
        "link": 1,
        "nextLink": "beeswax",
        "due": "20250101T090000Z",
    }
    new = dict(old)
    new.update(
        {
            "status": "completed",
            "end": "20250102T090000Z",
        }
    )

    raw = json.dumps(old) + "\n" + json.dumps(new) + "\n"
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    buf_in = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    prev_stdin = sys.stdin
    try:
        sys.stdin = buf_in
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            mod.main()
    finally:
        sys.stdin = prev_stdin
        spawn_effects.spawn_child_atomic = original_spawn

    out_task = _extract_last_json(buf_out.getvalue())
    spawn_effects.spawn_child_atomic = original_spawn
    expect(not called["spawn"], "re-completion should not trigger duplicate spawn")
    expect(out_task.get("nextLink") == "beeswax", "existing nextLink should be preserved")


def test_on_modify_recompleted_task_with_existing_link_skips_spawn():
    """Re-completing should not spawn when link #N+1 already exists in chain even if nextLink is empty."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_recomplete_link_guard_test")
    mod._SHOW_TIMELINE_GAPS = False
    mod._SHOW_ANALYTICS = False
    mod._CHECK_CHAIN_INTEGRITY = False

    called = {"spawn": False}

    def _spawn_child_atomic_stub(_child, _parent):
        called["spawn"] = True
        return ("cafebabe", set(), False, True, "queued", "si_test2")

    spawn_effects = mod._module("modify_spawn_effects")
    original_spawn = spawn_effects.spawn_child_atomic
    spawn_effects.spawn_child_atomic = lambda _ports, child, parent, **_kwargs: _spawn_child_atomic_stub(child, parent)
    modify_models = mod._module("modify_models")
    mod._completion_effects.chain_snapshot = lambda chain_id, _base, _next: modify_models.CompletionChainSnapshot(
        mode="recent", rows=[], loaded=False, chain_id=str(chain_id)
    )
    def _existing_next_guard(task, *_args, **_kwargs):
        mod._print_task(task)
        return False

    mod._completion_effects.existing_next_or_fail = _existing_next_guard

    def _get_chain_export_stub(chain_id, since=None, extra=None, env=None):
        if chain_id == "abcd1234" and extra and "link:2" in extra:
            return [
                {
                    "uuid": "00000000-0000-4000-8000-000000000222",
                    "status": "pending",
                    "link": 2,
                    "chainID": "abcd1234",
                }
            ]
        return []

    mod._module("modify_composition").lifecycle_read_service_for(mod).get_chain_export = _get_chain_export_stub

    old = {
        "uuid": "00000000-0000-4000-8000-000000000111",
        "status": "pending",
        "description": "reactivated duplicate guard via link check",
        "cp": "P1D",
        "chainID": "abcd1234",
        "link": 1,
        "due": "20250101T090000Z",
    }
    new = dict(old)
    new.update(
        {
            "status": "completed",
            "end": "20250102T090000Z",
        }
    )

    raw = json.dumps(old) + "\n" + json.dumps(new) + "\n"
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    buf_in = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    prev_stdin = sys.stdin
    try:
        sys.stdin = buf_in
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            mod.main()
    finally:
        sys.stdin = prev_stdin
        spawn_effects.spawn_child_atomic = original_spawn

    _ = _extract_last_json(buf_out.getvalue())
    expect(not called["spawn"], "existing link #N+1 should prevent duplicate spawn")




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








def test_on_modify_completion_reuses_single_chain_export_when_chain_needed():
    """on-modify should reuse one full-chain export across preflight and later feedback prep when chain context is needed."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_single_chain_export_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    mod._SHOW_ANALYTICS = True
    mod._SHOW_TIMELINE_GAPS = False
    mod._CHECK_CHAIN_INTEGRITY = False
    prev_panel_mode = mod.core.PANEL_MODE
    mod.core.PANEL_MODE = "text"

    now_utc = mod.core.now_utc()
    child_due = now_utc + timedelta(days=1)
    export_calls = {"count": 0}
    parent_uuid = "00000000-0000-4000-8000-000000000111"
    child_uuid = "00000000-0000-4000-8000-000000000222"

    chain_rows = [
        {
            "uuid": parent_uuid,
            "status": "completed",
            "description": "cp spawn test",
            "cp": "P1D",
            "chainID": "abcd1234",
            "chain": "on",
            "link": 1,
            "due": "20250101T090000Z",
            "entry": "2025-01-01T09:00:00Z",
            "nextLink": "",
        }
    ]

    modify_models = mod._module("modify_models")
    mod._completion_effects.compute_next_and_limits = lambda *_a, **_k: modify_models.CompletionComputeResult(
        child_due=child_due,
        meta={},
        dnf=None,
        until_dt=None,
        cpmax=0,
        cap_no=None,
        finals=[],
        until_cap_no=None,
    )
    mod._completion_effects.build_and_spawn_child = lambda *_a, **_k: modify_models.CompletionSpawnResult(
        child={
            "uuid": child_uuid,
            "status": "pending",
            "description": "next cp",
            "chainID": "abcd1234",
            "link": 2,
            "prevLink": parent_uuid[:8],
            "due": mod.core.fmt_isoz(child_due),
        },
        child_short=child_uuid[:8],
        stripped_attrs=[],
        verified=True,
        deferred_spawn=False,
        spawn_intent_id=None,
    )
    mod._presentation_effects.render_cp_completion_feedback = lambda **_k: None
    mod._diagnostics_effects.chain_health_advice = lambda *_a, **_k: None
    mod._append_next_wait_sched_rows = lambda *_a, **_k: None
    mod._module("lifecycle_read_service").clear_cached_chain_exports()
    mod._reset_modify_runtime_state()

    old = {
        "uuid": parent_uuid,
        "status": "pending",
        "description": "cp spawn test",
        "cp": "P1D",
        "chainID": "abcd1234",
        "link": 1,
        "due": "20250101T090000Z",
    }
    new = dict(old)
    new.update({"status": "completed", "end": "20250102T090000Z"})

    from nautical_core.integration_models import CommandFailureKind, TaskCommand, TaskCommandResult

    uow = _test_operator_uow()

    class Client:
        def execute(self, args, *, purpose, timeout, **_kwargs):
            export_calls["count"] += 1
            command = TaskCommand(("task", *args), purpose, timeout)
            return TaskCommandResult(
                command,
                0,
                json.dumps(chain_rows),
                "",
                CommandFailureKind.SUCCESS,
                1,
                0.001,
            )

    uow.client = Client()

    try:
        _modify_effect(mod, "handle_completion", old, new, uow)
    finally:
        mod.core.PANEL_MODE = prev_panel_mode

    expect(export_calls["count"] == 1, f"expected one underlying chain export, got {export_calls}")


def test_on_modify_completion_snapshot_reuses_full_chain_read():
    """A completion chain snapshot should satisfy the exact child-slot read."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_completion_snapshot_reuse_test")
    mod._reset_modify_runtime_state()
    saved_analytics = mod._SHOW_ANALYTICS
    mod._SHOW_ANALYTICS = True
    from nautical_core.integration_models import CommandFailureKind, Found, TaskCommand, TaskCommandResult

    uow = _test_operator_uow()
    calls = {"count": 0}

    class Client:
        def execute(self, args, *, purpose, timeout, **_kwargs):
            calls["count"] += 1
            rows = [{
                "uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "chainID": "reuse01",
                "link": 2,
                "chain": "on",
                "status": "pending",
            }]
            command = TaskCommand(("task", *args), purpose, timeout)
            return TaskCommandResult(command, 0, json.dumps(rows), "", CommandFailureKind.SUCCESS, 1, 0.001)

    uow.client = Client()
    try:
        snapshot = mod._completion_effects.chain_snapshot("reuse01", 1, 2, uow.repository)
        expect(snapshot.loaded and snapshot.coverage == "full", f"unexpected full snapshot: {snapshot!r}")
        reused = uow.repository.exact_child_slot("reuse01", 2)
        expect(isinstance(reused, Found), f"full snapshot did not satisfy child-slot read: {reused!r}")
        expect(calls["count"] == 1, f"full snapshot was exported more than once: {calls}")
    finally:
        mod._SHOW_ANALYTICS = saved_analytics
        mod._reset_modify_runtime_state()


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


def test_on_modify_cp_completion_spawns_next_link():
    """on-modify should spawn the next CP link on completion."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_cp_spawn_test")
    mod._SHOW_TIMELINE_GAPS = False
    mod._SHOW_ANALYTICS = False
    mod._CHECK_CHAIN_INTEGRITY = False

    spawned = {}

    def _spawn_child_atomic_stub(child, parent):
        spawned["child"] = child
        return ("beeswax", set(), False, True, "queued", "si_test3")

    spawn_effects = mod._module("modify_spawn_effects")
    original_spawn = spawn_effects.spawn_child_atomic
    spawn_effects.spawn_child_atomic = lambda _ports, child, parent, **_kwargs: _spawn_child_atomic_stub(child, parent)
    modify_models = mod._module("modify_models")
    mod._completion_effects.chain_snapshot = lambda chain_id, _base, _next: modify_models.CompletionChainSnapshot(
        mode="next", rows=[], loaded=True, chain_id=str(chain_id)
    )
    mod._completion_effects.existing_next_or_fail = lambda *_a, **_k: True
    # A confirmed empty chain is distinct from an unavailable Taskwarrior
    # export; keep this spawn-path test deterministic and network-free.
    mod._module("modify_composition").lifecycle_read_service_for(mod).get_chain_export = lambda *_a, **_k: []

    old = {
        "uuid": "00000000-0000-4000-8000-000000000111",
        "status": "pending",
        "description": "cp spawn test",
        "cp": "P1D",
        "chainID": "abcd1234",
        "link": 1,
        "due": "20250101T090000Z",
    }
    new = dict(old)
    new.update(
        {
            "status": "completed",
            "end": "20250102T090000Z",
        }
    )

    raw = json.dumps(old) + "\n" + json.dumps(new) + "\n"
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    buf_in = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    prev_stdin = sys.stdin
    try:
        sys.stdin = buf_in
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            try:
                mod.main()
            except SystemExit as e:
                raise AssertionError(f"on-modify exited unexpectedly (code={e.code})")
    finally:
        sys.stdin = prev_stdin
        spawn_effects.spawn_child_atomic = original_spawn

    out_task = _extract_last_json(buf_out.getvalue())
    expect("child" in spawned, "CP completion did not trigger spawn")
    expect(out_task.get("nextLink") in (None, ""), "CP completion should not set nextLink in decision-only mode")


def test_on_modify_spawn_intent_queue_failure_is_reported():
    """_spawn_child_atomic should report queue failure instead of claiming deferred success."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_spawn_queue_failure_test")
    command_effects = mod._module("modify_command_effects")
    command_effects.reserve_child_uuid = lambda _host, _env: "00000000-0000-4000-8000-00000000abcd"
    spawn_effects = mod._module("modify_spawn_effects")
    original_enqueue = spawn_effects.enqueue_spawn_intent
    spawn_effects.enqueue_spawn_intent = lambda _host, _entry: (False, "queue lock busy")

    spawn_ports = spawn_effects.spawn_child_ports_for(mod)
    child_short, _stripped, verified, deferred, reason, intent = spawn_effects.spawn_child_atomic(spawn_ports,
        {
            "uuid": "00000000-0000-4000-8000-000000000999",
            "description": "x",
            "status": "pending",
            "chainID": "abcd1234",
            "link": 2,
            "cp": "1d",
            "anchor_mode": "skip",
            "due": "20260824T090000Z",
        },
        {
            "uuid": "00000000-0000-4000-8000-000000000111",
            "chainID": "abcd1234",
            "chain": "on",
            "link": 1,
            "status": "completed",
            "nextLink": "",
        },
    )
    spawn_effects.enqueue_spawn_intent = original_enqueue
    expect(len(child_short) == 8 and all(ch in "0123456789abcdef" for ch in child_short.lower()), f"unexpected child short: {child_short}")
    expect(not verified, "verified should be false when queue fails")
    expect(not deferred, "deferred should be false when queue fails")
    expect("queue lock busy" in (reason or ""), f"missing queue failure reason: {reason}")
    expect(bool(intent), "spawn intent id should still be generated")


def test_on_add_run_task_timeout():
    """on-add typed command execution reports timeouts."""
    hook = _find_hook_file("on-add.nautical")
    mod = _load_hook_module(hook, "_nautical_on_add_run_task_timeout_test")
    result = core.run_task_result(
        [sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.02, retries=1,
    )
    expect(not result.ok and result.kind.value == "timeout", f"on-add timeout changed: {result}")


def test_on_modify_run_task_timeout():
    """on-modify typed command execution reports timeouts."""
    hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook, "_nautical_on_modify_run_task_timeout_test")
    command = mod._module("modify_command_effects")
    result = command.run_task_result(
        command.command_ports_for(mod),
        [sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.02, retries=1,
    )
    expect(not result.ok and result.kind.value == "timeout", f"on-modify timeout changed: {result}")


def test_on_modify_missing_taskdata_uses_tw_dir():
    """on-modify uses TW_DIR when TASKDATA is missing."""
    hook = _find_hook_file("on-modify.nautical")
    orig = os.environ.get("TASKDATA")
    if "TASKDATA" in os.environ:
        del os.environ["TASKDATA"]
    try:
        mod = _load_hook_module(hook, "_nautical_on_modify_no_taskdata_test")
    finally:
        if orig is not None:
            os.environ["TASKDATA"] = orig

    expect(
        str(getattr(mod, "TW_DATA_DIR", "")) == str(getattr(mod, "TW_DIR", "")),
        "TW_DATA_DIR should fall back to TW_DIR when TASKDATA is unset",
    )


def test_hooks_no_direct_subprocess_run():
    """Hooks should not call subprocess.run outside _run_task."""
    import ast

    def _bad_calls(path: str) -> list[tuple[int, str]]:
        src = Path(path).read_text(encoding="utf-8")
        tree = ast.parse(src, filename=path)
        bad = []
        stack = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            def visit_AsyncFunctionDef(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            def visit_Call(self, node):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "run":
                    if isinstance(func.value, ast.Name) and func.value.id == "subprocess":
                        current_fn = stack[-1] if stack else ""
                        if current_fn != "_run_task":
                            bad.append((node.lineno, current_fn or "<module>"))
                self.generic_visit(node)

        Visitor().visit(tree)
        return bad

    for hook_name in ("on-add.nautical", "on-modify.nautical"):
        path = _find_hook_file(hook_name)
        bad = _bad_calls(path)
        expect(not bad, f"Direct subprocess.run found in {hook_name}: {bad}")


def test_on_add_position_selection_renders_semantic_advice():
    """The on-add preview should include one advice row without disturbing hook JSON."""
    hook = _find_hook_file("on-add.nautical")
    task = {
        "uuid": "00000000-0000-4000-8000-000000000782",
        "description": "positional semantic advice",
        "status": "pending",
        "entry": "20260715T090000Z",
        "anchor": "(y:w-1)@in-year=8th",
        "anchor_mode": "skip",
    }
    proc = _run_hook_script(hook, task, env_extra={"NO_COLOR": "1"})
    expect(proc.returncode == 0, f"on-add advice preview failed: {proc.stderr}")
    out_task = _extract_last_json(proc.stdout)
    expect(out_task.get("anchor") == task["anchor"], f"on-add changed advised anchor: {out_task}")
    stderr = _strip_markup(proc.stderr)
    expect("Advice" in stderr, f"on-add preview omitted advice row: {stderr}")
    expect("ISO-week candidates by calendar year" in stderr, f"on-add preview omitted boundary explanation: {stderr}")


def test_position_selection_on_add_and_modify_completion():
    """Add preview and modify completion should agree on monthly positional anchors."""
    expr = "(w:tue | w:thu)@in-month=last"
    add_hook = _find_hook_file("on-add.nautical")
    task = {
        "uuid": "00000000-0000-4000-8000-000000000777",
        "description": "positional anchor integration",
        "status": "pending",
        "entry": "20260701T090000Z",
        "due": "20260730T090000Z",
        "anchor": expr,
        "anchor_mode": "skip",
    }
    result = _run_hook_script(add_hook, task, env_extra={"NO_COLOR": "1"})
    expect(result.returncode == 0, f"on-add rejected positional anchor: {result.stderr}")
    out_task = _extract_last_json(result.stdout)
    expect(out_task.get("anchor") == expr, f"on-add changed positional expression: {out_task}")
    expect(out_task.get("chain") == "on", f"on-add did not enable chain: {out_task}")
    expect(
        "last matching date" in _strip_markup(result.stderr),
        f"on-add preview omitted positional natural text: {result.stderr}",
    )

    modify_hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(modify_hook, "_nautical_position_selection_modify_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()
    child_due, meta, child_dnf = _compute_anchor_child_due(mod,
        {
            "anchor": expr,
            "anchor_mode": "skip",
            "due": "20260730T090000Z",
            "end": "20260730T100000Z",
            "chainID": "abcd1234",
        }
    )
    expect(mod.core.fmt_isoz(child_due) == "2026-08-27T09:00:00Z", f"bad child due: {child_due}")
    expect(meta.get("basis") == "after_end", f"unexpected completion metadata: {meta}")
    expect(child_dnf and child_dnf[0][0].get("kind") == "select", "completion lost selection DNF")


def test_position_selection_modify_timeline_projects_future_dates():
    """Modify timelines should project future positional-selection occurrences."""
    mod = _hook
    saved_collect = getattr(mod, "_collect_prev_two", None)
    mod._collect_prev_two = lambda _task: []
    expr = "(w:tue | w:thu)@in-month=last"
    dnf = core.validate_anchor_expr_strict(expr)
    task = {
        "uuid": "00000000-0000-4000-8000-000000000778",
        "description": "positional timeline",
        "anchor": expr,
        "anchor_mode": "skip",
        "link": 1,
        "due": "20260730T090000Z",
        "end": "20260730T100000Z",
        "chainID": "abcd1234",
    }
    try:
        lines = _call_with_supported_kwargs(
            mod._timeline_lines,
            kind="anchor",
            task=task,
            child_due_utc=datetime(2026, 8, 27, 9, 0, tzinfo=timezone.utc),
            child_short="0000abcd",
            dnf=dnf,
            next_count=5,
            cap_no=None,
            cur_no=1,
        )
    finally:
        if saved_collect is not None:
            mod._collect_prev_two = saved_collect
        else:
            delattr(mod, "_collect_prev_two")
    text = _strip_markup("\n".join(lines))
    expect("2026-08-27" in text, f"timeline omitted next positional date: {text}")
    expect("2026-09-29" in text, f"timeline omitted future positional date: {text}")


def test_position_selection_post_modifiers_modify_completion():
    """Modify completion should schedule the next transformed positional occurrence and time."""
    expr = "(w:tue | w:thu)@in-month=last@+2d@t=09:00"
    add_hook = _find_hook_file("on-add.nautical")
    task = {
        "uuid": "00000000-0000-4000-8000-000000000779",
        "description": "post-selection modifier integration",
        "status": "pending",
        "entry": "20260703T090000Z",
        "due": "20260801T060000Z",
        "anchor": expr,
        "anchor_mode": "skip",
    }
    result = _run_hook_script(add_hook, task, env_extra={"NO_COLOR": "1"})
    expect(result.returncode == 0, f"on-add rejected selector modifiers: {result.stderr}")
    out_task = _extract_last_json(result.stdout)
    expect(out_task.get("anchor") == expr, f"on-add changed selector modifiers: {out_task}")
    expect("2 days later at 09:00" in _strip_markup(result.stderr), f"bad add preview: {result.stderr}")

    modify_hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(modify_hook, "_nautical_position_selection_modifier_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()
    child_due, meta, child_dnf = _compute_anchor_child_due(mod,
        {
            "anchor": expr,
            "anchor_mode": "skip",
            "due": "20260801T090000Z",
            "end": "20260801T100000Z",
            "chainID": "abcd1234",
        }
    )
    child_local = mod.core.to_local(child_due)
    expect(child_local.date() == date(2026, 8, 29), f"bad child date: {child_due}")
    expect((child_local.hour, child_local.minute) == (9, 0), f"bad child time: {child_due}")
    expect(meta.get("basis") == "after_end", f"unexpected completion metadata: {meta}")
    expect(child_dnf[0][0].get("mods", {}).get("day_offset") == 2, "completion lost selector mods")

    saved_collect = getattr(mod, "_collect_prev_two", None)
    mod._collect_prev_two = lambda _task: []
    try:
        lines = _call_with_supported_kwargs(
            mod._timeline_lines,
            kind="anchor",
            task={
                "anchor": expr,
                "anchor_mode": "skip",
                "link": 2,
                "due": "20260801T060000Z",
                "end": "20260801T070000Z",
                "chainID": "abcd1234",
            },
            child_due_utc=child_due,
            child_short="0000abcd",
            dnf=child_dnf,
            next_count=4,
            cap_no=None,
            cur_no=2,
        )
    finally:
        if saved_collect is not None:
            mod._collect_prev_two = saved_collect
        else:
            delattr(mod, "_collect_prev_two")
    timeline = _strip_markup("\n".join(lines))
    expect("2026-08-29" in timeline, f"timeline omitted transformed child: {timeline}")
    expect("2026-10-01" in timeline, f"timeline omitted next transformed date: {timeline}")


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
            "from nautical_core import position_selection, season_support\n"
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


def test_on_add_seasonal_selection_feedback():
    """The add preview should show a readable seasonal rule and its fixed boundary."""
    hook = _find_hook_file("on-add.nautical")
    task = {
        "uuid": "00000000-0000-4000-8000-000000000783",
        "description": "seasonal feedback",
        "status": "pending",
        "entry": "20260723T090000Z",
        "anchor": "(w:mon)@in-spring=first",
        "anchor_mode": "skip",
    }
    proc = _run_hook_script(hook, task, env_extra={"NO_COLOR": "1"})
    expect(proc.returncode == 0, f"on-add rejected seasonal feedback anchor: {proc.stderr}")
    out_task = _extract_last_json(proc.stdout)
    expect(out_task.get("anchor") == task["anchor"], f"on-add changed seasonal anchor: {out_task}")
    stderr = _strip_markup(proc.stderr)
    expect("the first Monday of each spring" in stderr, f"preview omitted natural season: {stderr}")
    expect(
        "Advice" in stderr
        and (
            "fixed March 1 through May 31" in stderr
            or "astronomical spring equinox through" in stderr
        )
        and "boundaries." in stderr,
        f"preview omitted season boundary: {stderr}",
    )


def test_seasonal_selection_modify_modes_times_and_timeline():
    """Completion modes should preserve seasonal slots, local times, and future projections."""
    from zoneinfo import ZoneInfo

    modify_hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(modify_hook, "_nautical_seasonal_modify_modes_test")
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
        same_day_due, _same_meta, _same_dnf = _compute_anchor_child_due(mod,
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
        all_due, all_meta, _all_dnf = _compute_anchor_child_due(mod,
            dict(common, anchor_mode="all")
        )
        skip_due, skip_meta, skip_dnf = _compute_anchor_child_due(mod,
            dict(common, anchor_mode="skip")
        )
        flex_due, flex_meta, _flex_dnf = _compute_anchor_child_due(mod,
            dict(common, anchor_mode="flex")
        )
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


        evaluator = _evaluator_for_fixture(
            common,
            timezone=mod.core._LOCAL_TZ,
        )
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
        child = _build_child_draft_for_test(mod,
            parent,
            flex_due,
            "due",
            2,
            "00000000",
            "anchor",
            0,
            None,
        )
        expect(child.get("anchor") == expression, f"child lost seasonal anchor: {child}")
        expect(child.get("anchor_mode") == "all", f"flex child did not become all mode: {child}")
        expect(child.get("chainID") == "season123", f"child lost chain identity: {child}")

        saved_collect = getattr(mod, "_collect_prev_two", None)
        mod._collect_prev_two = lambda _task: []
        try:
            lines = _call_with_supported_kwargs(
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
        timeline = _strip_markup("\n".join(lines))
        expect("2027-03-01" in timeline, f"timeline omitted seasonal child: {timeline}")
        expect("2027-05-31" in timeline, f"timeline omitted later spring slot: {timeline}")
    finally:
        mod.core._LOCAL_TZ = previous_tz
        mod.core.SEASON_HEMISPHERE = previous_hemisphere
        season_support.configure_hemisphere(previous_hemisphere)






def test_position_selection_public_period_scopes_hooks():
    """Add, completion, and timeline paths should support shifted yearly selections."""
    expr = "(w:mon)@in-year=last@+7d@t=09:00"
    add_hook = _find_hook_file("on-add.nautical")
    local_day = date(2027, 1, 4)
    local_due = core.build_local_datetime(local_day, (9, 0)).astimezone(timezone.utc)
    local_end = core.build_local_datetime(local_day, (10, 0)).astimezone(timezone.utc)
    due_token = core.fmt_isoz(local_due)
    end_token = core.fmt_isoz(local_end)
    task = {
        "uuid": "00000000-0000-4000-8000-000000000780",
        "description": "yearly positional integration",
        "status": "pending",
        "entry": "20260701T090000Z",
        "due": due_token,
        "anchor": expr,
        "anchor_mode": "skip",
    }
    result = _run_hook_script(add_hook, task, env_extra={"NO_COLOR": "1"})
    expect(result.returncode == 0, f"on-add rejected yearly selection: {result.stderr}")
    out_task = _extract_last_json(result.stdout)
    expect(out_task.get("anchor") == expr, f"on-add changed yearly selection: {out_task}")
    expect("in each year" in _strip_markup(result.stderr), f"bad add preview: {result.stderr}")

    modify_hook = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(modify_hook, "_nautical_period_selection_hook_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()
    child_due, meta, child_dnf = _compute_anchor_child_due(mod,
        {
            "anchor": expr,
            "anchor_mode": "skip",
            "due": due_token,
            "end": end_token,
            "chainID": "abcd1234",
        }
    )
    child_local = mod.core.to_local(child_due)
    expect(child_local.date() == date(2028, 1, 3), f"bad yearly child date: {child_due}")
    expect((child_local.hour, child_local.minute) == (9, 0), f"bad yearly child time: {child_due}")
    expect(meta.get("basis") == "after_end", f"unexpected completion metadata: {meta}")
    expect(child_dnf[0][0].get("scope") == "year", "completion lost yearly scope")

    read_effects = mod._module("modify_read_effects")
    saved_collect = read_effects.collect_prev_two
    read_effects.collect_prev_two = lambda _host, _task, _chain=None: []
    try:
        lines = _call_with_supported_kwargs(
            mod._timeline_lines,
            kind="anchor",
            task={
                "anchor": expr,
                "anchor_mode": "skip",
                "link": 2,
                "due": "20270104T060000Z",
                "end": "20270104T070000Z",
                "chainID": "abcd1234",
            },
            child_due_utc=child_due,
            child_short="0000abcd",
            dnf=child_dnf,
            next_count=4,
            cap_no=None,
            cur_no=2,
        )
    finally:
        read_effects.collect_prev_two = saved_collect
    timeline = _strip_markup("\n".join(lines))
    expect("2028-01-03" in timeline, f"timeline omitted yearly child: {timeline}")
    expect("2029-01-01" in timeline, f"timeline omitted next yearly selection: {timeline}")


TESTS = [
    test_year_ordinals_hooks_modes_calendar_and_timeline,
    test_on_add_position_selection_renders_semantic_advice,
    test_position_selection_on_add_and_modify_completion,
    test_position_selection_modify_timeline_projects_future_dates,
    test_position_selection_post_modifiers_modify_completion,
    test_on_add_seasonal_selection_feedback,
    test_seasonal_selection_modify_modes_times_and_timeline,
    test_position_selection_public_period_scopes_hooks,
    test_business_calendar_toml_section_resolves_lazily,
    test_hook_on_add_uses_and_normalizes_business_calendar,
    test_hook_on_add_reports_business_calendar_displacement_only_when_shifted,
    test_hook_on_add_rejects_unknown_business_calendar_cleanly,
    test_hook_on_add_rejects_invalid_timezone_for_nautical_task,
    test_discovered_malformed_config_blocks_taskdata_reload,
    test_taskdata_reload_exposes_consistent_validated_fingerprints,
    test_hook_on_modify_rejects_unknown_business_calendar_cleanly,
    test_hook_on_modify_rejects_invalid_timezone_for_nautical_task,
    test_on_modify_spawned_child_preserves_business_calendar,
    test_modifier_boundary_paths_agree_and_advance_strictly,
    *RECURRENCE_TESTS,
    *RECONCILE_TESTS,
    test_random_anchor_and_omit_presets_keep_chain_scope,
    test_chain_colour_uses_complete_root_identity,
    test_on_add_preview_uses_configured_chain_colour,
    test_cp_interval_helpers_agree_between_on_add_and_on_modify,
    test_on_modify_compute_cp_sequence_selects_interval_by_link,
    test_on_modify_compute_cp_random_selects_deterministic_interval,
    test_on_modify_cp_sequence_estimates_chainmax_final_date,
    test_on_modify_anchor_chainmax_forecast_is_bounded,
    test_on_modify_anchor_file_child_projection_reuses_provider,
    test_on_modify_pure_anchor_file_projection_reuses_provider,
    test_hook_on_add_multitime_preview_emits_all_slots,
    test_hook_on_add_time_window_preview_emits_bounded_slots,
    test_hook_on_add_overnight_window_keeps_json_and_next_day_preview,
    test_hook_on_add_random_time_window_keeps_json_and_preview,
    test_on_modify_time_window_completion_advances_within_same_day,
    test_on_modify_partitioned_window_completion_rolls_to_next_day,
    test_on_modify_overnight_window_completion_uses_next_day_slots,
    test_on_modify_random_time_window_completion_reuses_stable_slots,
    test_time_window_dst_gap_deduplicates_shifted_local_slot,
    test_partitioned_time_window_dst_gap_deduplicates_shifted_local_slot,
    test_overnight_time_window_dst_fallback_deduplicates_repeated_local_slot,
    test_chain_until_overnight_window_survives_dst_fallback,
    test_hook_on_add_live_panel_mode_preserves_captured_protocol,
    test_hook_on_add_counted_random_preview_uses_group_time,
    test_hook_on_add_accepts_group_date_modifiers,
    test_hook_on_add_anchor_preset_resolves_from_config,
    test_hook_on_add_anchor_unknown_preset_fails_cleanly,
    test_hook_on_add_anchor_composed_preset_resolves_from_config,
    test_hook_on_add_anchor_recursive_preset_fails_cleanly,
    test_hook_on_add_omit_preset_resolves_from_config,
    test_hook_on_add_omit_unknown_preset_fails_cleanly,
    test_hook_on_add_omit_recursive_preset_fails_cleanly,
    test_hook_on_add_omit_timed_preset_rejected,
    test_hook_on_modify_cp_malformed_inputs_fail_with_parser_guidance,
    test_hook_on_add_cp_sequence_preview_accepts_string_periods,
    test_hook_on_add_cp_random_preview_shows_selected_periods,
    test_hook_on_add_cp_random_preview_uses_stamped_chain_id,
    test_hook_on_add_cp_random_malformed_fails_with_guidance,
    test_hook_on_add_cp_jitter_preview_shows_selected_periods,
    test_on_add_native_until_requires_strictly_later_target,
    test_on_add_native_until_checks_generated_cp_due,
    test_on_add_preview_distinguishes_expiration_from_chain_end_point,
    test_on_add_preview_fails_closed_when_evaluator_initialization_fails,
    test_on_add_preview_reports_scheduler_exhaustion_actionably,
    test_on_add_preview_uses_evaluator_for_first_due_and_upcoming_rows,
    test_on_add_native_until_checks_generated_anchor_due,
    test_on_add_native_until_guard_ignores_ordinary_tasks,
    test_on_add_chain_until_rejects_before_first_anchor_occurrence,
    test_on_add_native_until_rejects_strict_anchor_modes,
    test_on_modify_native_until_rejects_invalid_window_changes,
    test_on_modify_native_until_follows_recurrence_target_move,
    test_native_until_shared_policy_covers_recurrence_kinds_and_conflicts,
    test_on_modify_native_until_rejects_uncarryable_anchor_target_move,
    test_on_modify_completion_reschedule_carries_native_until,
    test_on_modify_native_until_accepts_valid_window_change,
    test_on_modify_native_until_validates_recurrence_promotion,
    test_on_modify_native_until_validates_simultaneous_completion,
    test_on_modify_native_until_rejects_strict_anchor_mode_changes,
    test_on_modify_native_until_rejects_legacy_all_completion,
    test_on_add_due_context_treats_due_matching_entry_as_implicit,
    test_on_add_anchor_preview_auto_assigns_when_due_matches_entry,
    test_hook_on_add_anchor_preview_skips_omit_date,
    test_hook_on_add_anchor_preview_skips_omit_file_date,
    test_hook_on_add_anchor_preview_rolled_business_day_uses_timed_slot,
    test_hook_on_add_anchor_preview_positive_day_offset_uses_timed_slot,
    test_hook_on_add_anchor_preview_negative_day_offset_uses_timed_slot,
    test_hook_on_add_timed_omit_rejected,
    test_hook_on_add_invalid_omit_file_rejected,
    test_hook_on_add_unsatisfiable_omit_fails_cleanly,
    test_hook_on_modify_timeline_multitime_includes_all_slots,
    test_hook_on_modify_timeline_cp_sequence_labels_future_intervals,
    test_hook_on_modify_timeline_cp_random_labels_selected_intervals,
    *HOOK_TESTS,
    test_hook_protocol_loads_without_core_package,
    test_taskwarrior_mutation_service_is_guarded_idempotent_and_fail_closed,
    test_child_import_rejects_incomplete_existing_rows,
    test_lifecycle_child_prefetch_reuses_one_authoritative_snapshot,
    test_lifecycle_batch_prefetch_uses_one_union_set_read,
    test_lifecycle_batch_postverification_fails_closed_on_unavailable_snapshot,
    test_lifecycle_outbox_persists_typed_plans_and_recovers_claims,
    test_lifecycle_outbox_prunes_only_expired_acknowledged_rows,
    test_lifecycle_outbox_initialization_is_concurrent_and_rejects_unknown_schema,
    test_lifecycle_outbox_bulk_compare_and_set_operations_isolate_rows,
    test_lifecycle_outbox_claims_quarantine_exhausted_and_inconsistent_rows,
    test_full_hooks_receive_one_explicit_integration_context,
    test_light_taskdata_resolution_matches_hook_precedence,
    test_plain_hook_fast_paths_do_not_import_core_package,
    test_full_hook_modules_defer_core_import,
    test_full_hooks_reuse_wrapper_protocol_probe,
    test_hook_bootstrap_uses_symlink_path_and_core_path_rescue,
    test_hooks_survive_malformed_numeric_environment,
    test_hook_files_are_private_permissions,
    test_safe_lock_fcntl_contention,
    test_safe_lock_fallback_contention,
    test_safe_lock_fallback_stale_cleanup,
    test_safe_lock_fallback_stale_pid_cleanup,
    test_diag_log_rotation_bounds,
    test_diag_log_redacts_sensitive_fields,
    test_hook_diag_redact_msg_masks_sensitive_json_fields,
    test_core_cache_dir_and_lock_permissions,
    test_core_cache_lock_contention_matches_safe_lock,
    test_core_cache_dir_rejects_symlink_override,
    test_on_modify_invalid_json_passthrough,
    test_on_modify_read_two_invalid_trailing,
    test_on_modify_read_two_array_uuid_mismatch_fails,
    test_on_modify_read_two_array_single_missing_uuid_fails,
    test_delete_chain_summary_span_uses_stop_time_without_last_end,
    test_end_summary_history_marks_deleted_pending_tail,
    test_delete_chain_summary_uses_stopped_title,
    test_on_modify_expiration_panel_explains_carry,
    test_on_modify_expiration_delegates_to_extracted_orchestration,
    test_on_modify_expiration_internal_failure_remains_recoverable,
    test_on_modify_expiration_wrapper_preserves_json_stdout,
    test_on_modify_manual_delete_persists_chain_off,
    test_on_modify_invalid_anchor_has_no_stdout,
    test_on_modify_render_anchor_completion_feedback_wrapper,
    test_hook_on_add_cp_scheduled_only_preserves_no_due,
    test_hook_on_add_cp_malformed_inputs_fail_with_parser_guidance,
    test_hook_on_add_anchor_scheduled_only_preserves_no_due,
    test_hook_on_add_rejects_invalid_chain_max_for_cp_and_anchor,
    test_on_modify_reports_business_calendar_displacement,
    test_on_modify_anchor_feedback_warns_when_timed_anchor_uses_utc_fallback,
    test_on_modify_render_anchor_file_completion_feedback_wrapper,
    test_on_modify_render_cp_completion_feedback_wrapper,
    test_on_modify_completion_panel_distinguishes_expiration_and_chain_boundaries,
    test_on_add_rejects_oversized_stdin_early,
    test_on_modify_rejects_oversized_stdin_early,
    test_health_check_json_ok_empty_taskdata,
    test_queue_status_and_doctor_report_schema_health,
    test_queue_claim_quarantines_poison_rows_and_queue_status_reports_them,
    test_queue_status_json_ok_empty_taskdata,
    test_queue_status_explicit_prune_reports_maintenance_result,
    test_doctor_installation_json_and_verifier_contract,
    test_operator_queue_status_json_ok_empty_taskdata,
    test_queue_status_warns_on_stale_processing_and_dead_letters,
    test_doctor_reports_healthy_installation,
    test_doctor_hook_inventory_allows_third_party_and_symlink_install,
    test_doctor_hook_inventory_rejects_duplicates_without_counting_backups,
    test_doctor_hook_inventory_reports_incomplete_core_and_api_mismatch,
    test_installer_dry_run_fresh_install_and_idempotent_reinstall,
    test_installer_navigator_dependency_failure_is_actionable,
    test_installer_upgrade_rollback_restores_active_runtime,
    test_installer_migrates_legacy_core_and_rolls_back_first_switch,
    test_installer_lock_and_duplicate_hook_guards,
    test_installer_cli_and_doctor_managed_runtime_diagnostics,
    test_doctor_reports_retired_queue_state_without_migrating_it,
    test_runtime_cleanup_preserves_active_and_rollback_releases,
    test_retained_release_can_be_selected_with_dry_run_then_applied,
    test_doctor_discovers_effective_taskdata_directory,
    test_operator_doctor_loads_colocated_queue_helper,
    test_nautical_dispatches_supported_subcommands,
    test_doctor_reports_actionable_broken_installation,
    test_doctor_reports_chain_repair_plan_findings,
    test_perf_hint_benchmark_isolates_persistent_cache,
    *PERFORMANCE_TESTS,
    test_core_import_defers_panel_colour_module,
    test_core_import_defers_diagnostic_model,
    test_core_import_defers_parser_scheduler_models,
    test_deploy_sanity_enforces_removed_lifecycle_ownership,
    test_perf_hook_fast_path_ratio_enforcement,
    test_load_benchmark_installs_complete_hook_runtime,
    test_load_benchmark_queue_and_lineage_verification,
    test_deploy_sanity_rejects_missing_lazy_lifecycle_module,
    test_deploy_sanity_rejects_missing_operator_runtime_tool,
    test_deploy_sanity_rejects_unowned_taskwarrior_subprocess,
    test_hook_replay_harness_reports_ok,
    test_mixed_recurrence_loop_harness_reports_ok,
    test_soak_runner_reports_ok,
    test_ops_templates_present_and_runner_executable,
    test_tw_export_chain_extra_validation,
    test_tw_export_chain_extra_rejects_dash_prefixed_tokens,
    test_on_modify_diag_blocks_pretty_print,
    test_on_modify_lifecycle_diagnostics_are_gated_to_stderr,
    test_on_modify_run_task_diag_bucket_stats,
    test_on_exit_diag_blocks_pretty_print,
    test_on_exit_outcome_diagnostics_are_bounded,
    test_on_modify_chain_cache_thread_safety_smoke,
    test_on_modify_get_chain_export_filters_cached_chain_in_memory,
    test_on_modify_chain_cache_reads_through_typed_repository,
    test_on_modify_chain_cache_preserves_repository_unavailability,
    test_on_modify_predecessor_read_preserves_repository_unavailability,
    test_local_datetime_non_hour_dst_gap_is_shared_by_modify,
    test_modify_completion_advances_past_second_dst_fold,
    test_modify_overnight_window_advances_past_second_dst_fold,
    test_anchor_preview_explains_nonexistent_wall_time_adjustment,
    test_on_modify_collect_prev_two_prefers_live_statuses,
    test_on_add_fail_and_exit_emits_json,
    test_on_add_panic_passthrough_emits_valid_json,
    test_on_modify_panic_passthrough_uses_latest_task,
    test_on_add_ignores_unsafe_core_path_override,
    test_on_modify_ignores_unsafe_core_path_override,
    test_on_modify_promotes_chain_when_task_becomes_nautical,
    test_on_modify_promotes_chain_emits_upgrade_panel,
    test_on_modify_promotes_cp_emits_period_explanation,
    test_on_modify_disables_chain_emits_disabled_panel,
    test_on_modify_resumes_chain_emits_resumed_panel,
    test_on_modify_resume_wrapper_preserves_json_and_emits_panel,
    test_on_modify_recurrence_update_emits_ack_panel,
    test_on_modify_recurrence_update_groups_and_flattens_changes,
    test_on_modify_native_until_update_explains_carry,
    test_on_modify_limit_update_emits_effective_boundaries,
    test_on_add_lowercase_chainid_does_not_mark_nautical,
    test_on_add_read_one_fuzz_inputs,
    test_on_modify_read_two_fuzz_inputs,
    test_on_add_dnf_cache_uses_central_api_and_fingerprints_parser,
    test_on_add_dnf_cache_quarantines_central_corruption,
    test_on_exit_reads_data_arg_from_hook_argv,
    test_on_modify_no_explicit_taskdata_skips_rc_data_location,
    test_on_modify_reads_data_arg_from_hook_argv,
    test_on_add_no_explicit_taskdata_skips_rc_data_location,
    test_on_add_reads_data_arg_from_hook_argv,
    test_on_exit_data_arg_overrides_taskdata_env,
    test_on_modify_data_arg_overrides_taskdata_env,
    test_on_add_data_arg_overrides_taskdata_env,
    test_on_add_requires_integration_context_helper,
    test_on_modify_requires_integration_context_helper,
    test_on_exit_requires_integration_context_helper,
    test_on_modify_carry_wall_clock_across_dst,
    test_on_modify_build_child_carries_until_across_dst,
    test_on_modify_native_until_calendar_and_exact_carry_policy,
    test_on_modify_native_until_exact_carry_preserves_elapsed_time_across_dst,
    test_native_until_calendar_slot_guard_rejects_impossible_anchor_expirations,
    test_on_modify_build_child_transitions_flex_to_all,
    test_on_modify_cp_due_edit_preserves_relative_offsets,
    test_on_modify_explicit_timing_edits_warn_on_invalid_order,
    test_on_modify_timing_warning_wrapper_preserves_json_stdout,
    test_on_modify_link_limit,
    test_on_modify_completion_preflight_context_happy_path,
    test_on_modify_completion_compute_next_and_limits_happy_path,
    test_cap_from_until_cp_includes_exact_deadline,
    test_hook_on_modify_rejects_invalid_chain_max_for_cp_and_anchor,
    test_on_modify_validates_chain_until_only_when_recurrence_or_caps_change,
    test_on_modify_completion_chain_snapshot_modes_and_query,
    test_on_modify_completion_snapshot_malformed_json_is_unavailable,
    test_on_modify_completion_defers_chain_export_until_after_preflight,
    test_on_modify_compute_cp_child_due_uses_scheduled_when_due_missing,
    test_on_modify_compute_anchor_child_due_uses_scheduled_seed_for_all_mode,
    test_on_modify_compute_anchor_child_due_builds_timed_slots_in_configured_timezone,
    test_on_add_preview_and_completion_skip_choose_same_next_anchor,
    test_on_modify_anchor_dnf_accepts_configured_preset,
    test_on_modify_omit_dnf_accepts_configured_preset,
    test_on_add_anchor_and_anchor_file_can_coexist,
    test_on_add_anchor_file_root_gets_chainid_stamp,
    test_on_add_chainid_stamp_failure_rejects_recurring_root,
    test_hook_on_add_anchor_file_preview_auto_assigns_first_match,
    test_hook_on_add_anchor_and_anchor_file_preview_uses_earliest_union_match,
    test_on_modify_compute_anchor_child_due_from_anchor_file,
    test_on_modify_compute_anchor_child_due_from_random_anchor_file,
    test_on_modify_compute_anchor_child_due_from_multiple_file_times,
    test_on_modify_compute_anchor_child_due_from_combined_anchor_sources,
    test_on_modify_compute_combined_overnight_sources_in_time_order,
    test_hook_on_modify_timeline_marks_omitted_anchor_slots,
    test_hook_on_modify_merged_timeline_marks_projection_failures,
    test_hook_on_modify_timeline_uses_omit_file_description_label,
    test_on_modify_compute_anchor_child_due_skips_omit_date,
    test_on_modify_compute_anchor_child_due_accepts_scheduled_after_due,
    test_on_modify_compute_counted_random_advances_within_period,
    test_on_modify_compute_anchor_child_due_unsatisfiable_omit_fails,
    test_on_modify_completion_build_and_spawn_child_happy_path,
    test_on_modify_completion_spawn_exception_is_retryable_with_reason,
    test_on_modify_build_child_scheduled_only_keeps_due_unset_and_carries_wait,
    test_on_modify_render_cp_completion_feedback_random_selected_interval,
    test_on_modify_render_cp_completion_feedback_jitter_selected_interval,
    test_on_modify_render_cp_completion_feedback_text_mode,
    test_on_add_preview_hard_cap,
    test_on_add_flushes_stdout,
    test_on_add_profiler_lazy_init,
    test_on_add_format_anchor_rows_numbers_upcoming_from_three_with_next_anchor,
    test_on_add_format_anchor_rows_numbers_upcoming_from_two_without_next_anchor,
    test_on_modify_panel_fallback,
    test_on_modify_panel_forwards_live_duration,
    test_ui_live_test_term_guard_restores_environment,
    test_on_exit_emit_exit_feedback_reaches_stdout_contract,
    test_hooks_require_package_core_layout,
    test_core_import_deterministic,
    test_core_import_defers_optional_stacks,
    test_on_modify_recompleted_task_with_nextlink_skips_spawn,
    test_on_modify_recompleted_task_with_existing_link_skips_spawn,
    test_on_modify_completion_reuses_single_chain_export_when_chain_needed,
    test_on_modify_completion_snapshot_reuses_full_chain_read,
    test_on_modify_lifecycle_export_reuses_completion_chain_snapshot,
    test_on_modify_cp_completion_spawns_next_link,
    test_on_modify_spawn_intent_queue_failure_is_reported,
    test_on_add_run_task_timeout,
    test_on_modify_run_task_timeout,
    test_on_modify_state_files_use_dedicated_dir,
    test_on_modify_stable_child_uuid_is_slot_deterministic,
    test_on_modify_missing_taskdata_uses_tw_dir,
    test_hooks_no_direct_subprocess_run,
    test_core_invalid_timezone_warns_and_falls_back_to_utc,
    test_explicit_unsafe_config_blocks_scheduling_with_actionable_error,
    test_taskdata_config_reload_fails_closed_for_malformed_toml_and_timezone,
    test_core_recurrence_update_udas_config_aliases,
    test_core_live_panel_duration_config_defaults_and_clamps,
    test_core_live_panel_footer_config_defaults_and_customizes,
    test_core_uda_aliases_config_defaults_disabled_and_can_enable,
    test_on_add_expands_enabled_description_uda_aliases,
    test_hook_on_add_uda_aliases_emit_canonical_json_and_reject_conflicts,
    test_hook_on_modify_uda_aliases_route_through_thin_wrapper,
    test_hook_on_modify_uda_alias_anchor_change_emits_ack_panel,
    test_hook_on_modify_empty_uda_alias_clears_through_thin_wrapper,
    test_hook_on_add_disabled_uda_aliases_leave_description_untouched,
    test_on_modify_expands_and_clears_description_uda_aliases,
    test_astronomical_season_selection_scheduler_uses_transition_dates,
    test_on_modify_build_child_carries_configured_uda_datetime,

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
            if not args.isolated_child and "_load_core_module" in fn.__code__.co_names:
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
    *OPERATOR_TESTS,
    test_hook_on_add_anchor_file_time_padding_hint,
    test_hook_on_add_anchor_preview_marks_omitted_future_slots,
    test_hook_on_add_anchor_preview_skips_omit_file_modifier_date,
    test_hook_on_add_anchor_preview_uses_omit_file_description_in_upcoming,
    test_hook_on_modify_timeline_keeps_anchor_match_after_shifted_anchor_file_child,
    test_hook_on_modify_timeline_omits_shifted_anchor_file_dates_in_merged_stream,
    test_hook_on_modify_timeline_shows_anchor_side_omit_file_dates_in_merged_stream,
    test_navigator_surfaces_configuration_drift_warning,
    test_navigator_reloads_validated_taskdata_configuration,
    test_navigator_fallback_export_uses_empty_filter,
    test_shared_time_slot_resolver_keeps_hook_and_navigator_parity,
    test_navigator_projects_all_slots_in_a_time_window,
    test_on_modify_reuses_task_scoped_evaluator_and_scheduler_binding,
    test_random_time_window_is_stable_across_processes,
    test_navigator_reads_through_read_only_invocation_repository,
    test_navigator_uses_anchor_and_anchor_file_sources,
    test_on_modify_read_two_single_plain_delete_without_uuid_is_ignored,
    test_on_modify_read_two_uuid_mismatch_without_nautical_fields_is_ignored,
    test_config_fingerprint_invalidates_persistent_cache_keys,
    test_configuration_drift_detects_edit_and_removal,
    *INSTALLER_TESTS,
])

TESTS.append(test_on_modify_completion_helper_returns_finalized_lifecycle_result)
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
    *LIFECYCLE_TESTS,
])

if __name__ == "__main__":
    main()
