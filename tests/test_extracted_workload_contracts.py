from __future__ import annotations

import unittest

from dev_tools.golden_tests import installer, operator, performance, recurrence
from dev_tools.perf import (
    anchor_file_workloads,
    cache_workloads,
    calendar_workloads,
    hook_workloads,
    operator_workloads,
    outbox_workloads,
    resource_workloads,
    scheduler_workloads,
    telemetry,
)
from nautical_core import callback_ports, lifecycle_outbox_codec
from nautical_core.lifecycle_models import ExecutionStage


class ExtractedWorkloadContractTests(unittest.TestCase):
    def test_extracted_golden_modules_register_callable_cases(self) -> None:
        for module in (installer, operator, performance, recurrence):
            self.assertTrue(module.TESTS)
            self.assertTrue(all(callable(case) for case in module.TESTS))

    def test_extracted_performance_modules_expose_callable_workloads(self) -> None:
        modules = (
            anchor_file_workloads,
            cache_workloads,
            calendar_workloads,
            hook_workloads,
            operator_workloads,
            outbox_workloads,
            resource_workloads,
            scheduler_workloads,
        )
        for module in modules:
            public = [value for name, value in vars(module).items() if not name.startswith("_")]
            self.assertTrue(any(callable(value) for value in public), module.__name__)

    def test_telemetry_reports_sorted_samples_and_budget_status(self) -> None:
        measured = telemetry.measure_workflow("sample", [0.3, 0.1, 0.2], 0.15)
        self.assertEqual(measured["samples_s"], [0.1, 0.2, 0.3])
        self.assertFalse(measured["pass"])

    def test_callback_port_is_a_callable_protocol(self) -> None:
        self.assertTrue(callable(callback_ports.CallbackPort.__call__))

    def test_lifecycle_codec_round_trips_canonical_objects(self) -> None:
        encoded = lifecycle_outbox_codec.canonical_object_json(
            '{"z": 1, "a": "é"}', field="payload"
        )
        self.assertEqual(encoded, '{"a":"é","z":1}')
        self.assertTrue(
            lifecycle_outbox_codec.transition_allowed(
                ExecutionStage.PLANNED,
                ExecutionStage.PERSISTED,
            )
        )


if __name__ == "__main__":
    unittest.main()
