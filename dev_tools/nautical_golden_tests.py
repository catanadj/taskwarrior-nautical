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
    uow.repository.rows.pop(parent_uuid)
    deleted = service.apply(request(MutationOperation.CHAIN_DISABLE, ChainDisablePayload(parent_uuid), 0))
    expect(deleted.kind in {MutationOutcomeKind.CONFLICT, MutationOutcomeKind.RETRYABLE}, f"deleted parent was applied: {deleted}")
    expect(not uow.client.calls, "deleted parent reached the mutation command")
    uow.repository.rows[parent_uuid] = parent

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
    *TIMELINE_TESTS[:3],
    test_taskwarrior_mutation_service_is_guarded_idempotent_and_fail_closed,
    test_lifecycle_outbox_persists_typed_plans_and_recovers_claims,
    test_lifecycle_outbox_initialization_is_concurrent_and_rejects_unknown_schema,
    *STORAGE_TESTS,
    *MODIFY_TESTS[:1],
    test_on_modify_anchor_feedback_warns_when_timed_anchor_uses_utc_fallback,
    *OPERATOR_TESTS[:2],
    test_queue_claim_quarantines_poison_rows_and_queue_status_reports_them,
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
    test_on_modify_promotes_chain_when_task_becomes_nautical,
    *MODIFY_TESTS[1:6],
    *MODIFY_TESTS[6:10],
    test_on_add_requires_integration_context_helper,
    *MODIFY_TESTS[10:],
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
    test_on_modify_compute_anchor_child_due_from_anchor_file,
    test_on_modify_compute_anchor_child_due_from_random_anchor_file,
    test_on_modify_compute_anchor_child_due_from_multiple_file_times,
    test_on_modify_compute_anchor_child_due_from_combined_anchor_sources,
    test_on_modify_compute_combined_overnight_sources_in_time_order,
    *TIMELINE_TESTS[3:5],
    *TIMELINE_TESTS[5:6],
    test_on_modify_completion_build_and_spawn_child_happy_path,
    test_on_modify_completion_spawn_exception_is_retryable_with_reason,
    test_on_modify_build_child_scheduled_only_keeps_due_unset_and_carries_wait,
    test_on_modify_panel_fallback,
    test_on_modify_panel_forwards_live_duration,
    test_ui_live_test_term_guard_restores_environment,
    test_on_modify_recompleted_task_with_nextlink_skips_spawn,
    test_on_modify_recompleted_task_with_existing_link_skips_spawn,
    test_on_modify_completion_reuses_single_chain_export_when_chain_needed,
    test_on_modify_completion_snapshot_reuses_full_chain_read,
    test_on_modify_lifecycle_export_reuses_completion_chain_snapshot,
    test_on_modify_cp_completion_spawns_next_link,
    test_on_modify_spawn_intent_queue_failure_is_reported,
    test_on_modify_stable_child_uuid_is_slot_deterministic,
    *CONFIGURATION_TESTS[:3],
    *CONFIGURATION_TESTS[6:10],
    *CONFIGURATION_TESTS[12:15],
    test_on_modify_expands_and_clears_description_uda_aliases,
    *SCHEDULING_TESTS[8:9],
    test_on_modify_build_child_carries_configured_uda_datetime,

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
