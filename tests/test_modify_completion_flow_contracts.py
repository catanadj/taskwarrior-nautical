from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, get_type_hints
import unittest

import nautical_core.modify_completion_flow as flow
from nautical_core.lifecycle.read_service import LifecycleReadService
from nautical_core.task_read_repository import TaskReadRepository
from nautical_core.integration_models import (
    CommandFailureKind,
    FailureEvidence,
    TaskCommand,
    Unavailable,
)
from nautical_core.modify_completion_effects import SnapshotPorts, chain_snapshot
from nautical_core.modify_models import CompletionChainSnapshot


class ModifyCompletionFlowContracts(unittest.TestCase):
    def test_completion_clock_contract_uses_datetime(self) -> None:
        localns = {
            "LifecycleReadService": LifecycleReadService,
            "TaskReadRepository": TaskReadRepository,
        }
        service_clock = get_type_hints(flow.CompletionFlowServices, localns=localns)["now_utc"]
        finalize_clock = get_type_hints(flow.finalize_completion_modify)["now_utc"]

        self.assertEqual(service_clock, Callable[[], datetime])
        self.assertIs(finalize_clock, datetime)
        self.assertIsNot(service_clock, Any)

    def test_completion_flow_uses_runtime_and_unit_of_work_owners(self) -> None:
        from nautical_core.modify_runtime import ModifyRuntimeState
        from nautical_core.taskwarrior_uow import TaskwarriorUnitOfWork

        runtime_state = get_type_hints(
            flow.CompletionFlowServices,
            localns={
                "LifecycleReadService": LifecycleReadService,
                "TaskReadRepository": TaskReadRepository,
            },
        )["runtime_state"]
        unit_of_work = get_type_hints(flow.handle_completion_modify)["unit_of_work"]

        self.assertEqual(runtime_state, Callable[[], ModifyRuntimeState])
        self.assertIs(unit_of_work, TaskwarriorUnitOfWork)

    def test_completion_services_use_lifecycle_read_owner(self) -> None:
        localns = {
            "LifecycleReadService": LifecycleReadService,
            "TaskReadRepository": TaskReadRepository,
        }
        flow_services = get_type_hints(flow.CompletionFlowServices, localns=localns)
        finalize_services = get_type_hints(flow.CompletionFinalizeServices, localns=localns)

        self.assertIs(flow_services["lifecycle_read_service"], LifecycleReadService)
        self.assertIs(finalize_services["lifecycle_read_service"], LifecycleReadService)

    def test_preflight_callback_uses_datetime_and_task_repository(self) -> None:
        preflight = get_type_hints(
            flow.CompletionFlowServices,
            localns={
                "LifecycleReadService": LifecycleReadService,
                "TaskReadRepository": TaskReadRepository,
            },
        )["preflight_context"]

        self.assertEqual(
            preflight,
            Callable[
                [flow.TaskPayload, datetime, TaskReadRepository],
                flow.CompletionPreflightContext | None,
            ],
        )

    def test_completion_compute_callback_has_typed_preflight_context(self) -> None:
        from nautical_core.modify_models import (
            CompletionComputeCallback,
            CompletionPreflightContext,
        )

        callback_hints = get_type_hints(CompletionComputeCallback.__call__)

        self.assertEqual(callback_hints["preflight"], CompletionPreflightContext | None)
        flow_callback = get_type_hints(
            flow.CompletionFlowServices,
            localns={
                "LifecycleReadService": LifecycleReadService,
                "TaskReadRepository": TaskReadRepository,
            },
        )["compute_next_and_limits"]
        self.assertIs(flow_callback, CompletionComputeCallback)

        from nautical_core.modify_composition import ModifyRuntimeServices
        from nautical_core.modify_runtime import ModifyRuntimeState

        runtime_callback = get_type_hints(
            ModifyRuntimeServices,
            localns={
                "LifecycleReadService": LifecycleReadService,
                "ModifyRuntimeState": ModifyRuntimeState,
            },
        )["compute_next_and_limits"]
        self.assertIs(runtime_callback, CompletionComputeCallback)

    def test_completion_finalize_callback_has_explicit_arguments(self) -> None:
        from inspect import Parameter, signature
        from nautical_core.modify_models import CompletionFinalizeCallback

        callback = get_type_hints(
            flow.CompletionFlowServices,
            localns={
                "LifecycleReadService": LifecycleReadService,
                "TaskReadRepository": TaskReadRepository,
            },
        )["finalize_completion"]

        self.assertIs(callback, CompletionFinalizeCallback)
        callback_signature = signature(CompletionFinalizeCallback.__call__)
        self.assertEqual(
            tuple(callback_signature.parameters),
            (
                "self", "new", "ctx", "computed", "now_utc", "need_chain",
                "chain_snapshot_loaded", "preloaded_chain", "preloaded_chain_by_link",
                "preloaded_chain_by_short", "chain_id", "services",
            ),
        )
        self.assertTrue(all(
            callback_signature.parameters[name].kind is Parameter.KEYWORD_ONLY
            for name in (
                "new", "ctx", "computed", "now_utc", "need_chain",
                "chain_snapshot_loaded", "preloaded_chain", "preloaded_chain_by_link",
                "preloaded_chain_by_short", "chain_id", "services",
            )
        ))

    def test_unavailable_chain_export_is_not_loaded_as_empty_snapshot(self) -> None:
        command = TaskCommand(("task", "export"), "completion snapshot", 3.0)
        unavailable = Unavailable(
            "chain snapshot",
            FailureEvidence(
                command,
                CommandFailureKind.INVALID_RESPONSE,
                0,
                1,
                0.001,
                False,
                "malformed JSON",
            ),
        )
        ports = SnapshotPorts(
            repository=SimpleNamespace(
                chain_snapshot=lambda _chain_id: unavailable,
            ),
            mode=lambda: "full",
            snapshot_type=CompletionChainSnapshot,
        )

        snapshot = chain_snapshot(ports, "malformed01", 1, 2)

        self.assertFalse(snapshot.loaded)
        self.assertEqual(snapshot.rows, [])
        self.assertTrue(snapshot.is_unavailable)
        self.assertIn("malformed JSON", snapshot.error)

    def test_failed_preflight_stops_before_compute_and_chain_reads(self) -> None:
        events = []
        chain_reads = []
        repository = SimpleNamespace(
            chain_snapshot=lambda *_args: chain_reads.append("snapshot"),
            exact_child_slot=lambda *_args: chain_reads.append("child-slot"),
        )
        now = datetime(2026, 9, 28, tzinfo=timezone.utc)
        services = flow.CompletionFlowServices(
            runtime_state=lambda: SimpleNamespace(task_repository=None),
            prepare_recurrence=lambda _old, _new: ("P1D", "", ""),
            preserve_cp_relative_offsets=lambda *_args: None,
            preserve_native_until=lambda *_args: None,
            validate_native_until=lambda *_args: None,
            validate_native_until_slots=lambda *_args: None,
            now_utc=lambda: now,
            preflight_context=lambda _task, _now, _repository: events.append(
                "preflight"
            ),
            compute_next_and_limits=lambda *_args, **_kwargs: events.append(
                "compute"
            ),
            lifecycle_read_service=object(),
            diag_count=lambda *_args: None,
            diag_lifecycle_result=lambda *_args: None,
            finalize_completion=lambda **_kwargs: events.append("finalize"),
            finalize_services=None,
        )

        result = flow.handle_completion_modify(
            {"uuid": "parent", "status": "pending", "cp": "P1D"},
            {
                "uuid": "parent",
                "status": "completed",
                "cp": "P1D",
                "chainID": "abcd1234",
                "link": 1,
            },
            SimpleNamespace(repository=repository),
            services=services,
        )

        self.assertIsNone(result)
        self.assertEqual(events, ["preflight"])
        self.assertEqual(chain_reads, [])

    def test_link_limit_stops_preflight_before_snapshot_or_next_link_lookup(self) -> None:
        from nautical_core.modify_completion_preflight import (
            completion_link_numbers_or_fail,
            completion_preflight_context,
        )

        panels = []
        printed = []
        later_calls = []

        def unexpected(name):
            def fail(*_args):
                later_calls.append(name)
                raise AssertionError(f"{name} must not run after the link limit")

            return fail

        task = {
            "uuid": "00000000-0000-4000-8000-000000000111",
            "status": "completed",
            "description": "limit test",
            "anchor": "w:mon",
            "chainID": "abcd1234",
            "link": 3,
            "due": "20250101T090000Z",
        }
        services = SimpleNamespace(
            short=lambda value: str(value or "")[:8],
            completion_link_numbers_or_fail=lambda new: completion_link_numbers_or_fail(
                new,
                coerce_int=lambda value, default: int(value or default),
                max_link_number=3,
                panel=lambda *args, **kwargs: panels.append((args, kwargs)),
                print_task=printed.append,
            ),
            completion_kind_or_stop=unexpected("kind"),
            completion_chain_id_or_fail=unexpected("chain-id"),
            completion_chain_snapshot=unexpected("snapshot"),
            completion_existing_next_or_fail=unexpected("next-link"),
        )

        context = completion_preflight_context(
            task,
            datetime(2025, 1, 2, tzinfo=timezone.utc),
            services=services,
        )

        self.assertIsNone(context)
        self.assertEqual(
            panels,
            [
                (
                    (
                        "⛔ Link limit exceeded",
                        [("Reason", "Link number 4 exceeds max_link_number=3.")],
                    ),
                    {"kind": "error"},
                )
            ],
        )
        self.assertEqual(printed, [task])
        self.assertEqual(later_calls, [])


if __name__ == "__main__":
    unittest.main()
