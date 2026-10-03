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

        runtime_callback = get_type_hints(ModifyRuntimeServices)["compute_next_and_limits"]
        self.assertIs(runtime_callback, CompletionComputeCallback)

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


if __name__ == "__main__":
    unittest.main()
