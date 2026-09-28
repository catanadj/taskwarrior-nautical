from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

from nautical_core import modify_completion_flow as flow


class ModifyCompletionFlowContracts(unittest.TestCase):
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
