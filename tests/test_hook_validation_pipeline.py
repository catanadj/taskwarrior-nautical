from __future__ import annotations

import unittest

from nautical_core.hook_validation_pipeline import (
    ValidationFinding,
    ValidationInput,
    ValidationPipeline,
    ValidationStage,
    ValidationStatus,
)
from nautical_core.hook_workflow_models import WorkflowRoute
from nautical_core.task_models import TaskObservation


def _observation() -> TaskObservation:
    return TaskObservation.from_mapping(
        {
            "uuid": "11111111-1111-4111-8111-111111111111",
            "status": "pending",
            "chainID": "abcd1234",
            "link": 1,
            "chain": "on",
            "cp": "P1D",
        },
        source_query="validation-test",
    )


class ValidationPipelineTests(unittest.TestCase):
    def test_rules_run_in_declared_order_and_return_valid(self) -> None:
        seen: list[str] = []

        def syntax(value: ValidationInput):
            seen.append(value.route.value)
            return ()

        report = ValidationPipeline(((ValidationStage.SYNTAX, syntax),)).validate(
            ValidationInput(_observation(), route=WorkflowRoute.CP_ACTIVATION)
        )
        self.assertEqual(report.status, ValidationStatus.VALID)
        self.assertEqual(seen, ["cp_activation"])

    def test_retryable_finding_is_unavailable(self) -> None:
        finding = ValidationFinding(
            ValidationStage.DOMAIN,
            "config_unavailable",
            "timezone",
            "timezone configuration could not be loaded",
            retryable=True,
            correction="Fix the scheduling configuration and retry.",
        )
        report = ValidationPipeline(((ValidationStage.DOMAIN, lambda _value: (finding,)),)).validate(
            ValidationInput(_observation())
        )
        self.assertEqual(report.status, ValidationStatus.UNAVAILABLE)
        self.assertEqual(report.findings[0].correction, "Fix the scheduling configuration and retry.")

    def test_invalid_finding_is_not_unavailable(self) -> None:
        finding = ValidationFinding(
            ValidationStage.SYNTAX,
            "anchor_invalid",
            "anchor",
            "anchor expression is invalid",
            correction="Use a supported anchor expression.",
        )
        report = ValidationPipeline(((ValidationStage.SYNTAX, lambda _value: (finding,)),)).validate(
            ValidationInput(_observation())
        )
        self.assertEqual(report.status, ValidationStatus.INVALID)


if __name__ == "__main__":
    unittest.main()
