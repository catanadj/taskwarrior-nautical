from __future__ import annotations

import unittest

from nautical_core.hook_validation_pipeline import (
    ValidationFinding,
    ValidationInput,
    ValidationPipeline,
    ValidationStage,
    ValidationStatus,
    build_default_validation_pipeline,
    validate_task_transition,
    normalize_description_uda_aliases,
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
    def test_default_domain_pipeline_rejects_mixed_recurrence_sources(self) -> None:
        task = _observation().to_mapping()
        task["anchor"] = "w:mon"
        report = build_default_validation_pipeline().validate(
            ValidationInput(TaskObservation.from_mapping(task, source_query="domain-test"))
        )
        self.assertEqual(report.status, ValidationStatus.INVALID)
        self.assertEqual(report.findings[0].code, "recurrence_kind_conflict")

    def test_default_domain_pipeline_allows_missing_identity_on_activation(self) -> None:
        task = _observation().to_mapping()
        task.pop("chainID", None)
        task.pop("link", None)
        report = build_default_validation_pipeline().validate(
            ValidationInput(
                TaskObservation.from_mapping(task, source_query="activation-test"),
                route=WorkflowRoute.CP_ACTIVATION,
            )
        )
        self.assertEqual(report.status, ValidationStatus.VALID)

    def test_mapping_validation_exposes_decode_findings(self) -> None:
        _observation_value, report = __import__(
            "nautical_core.hook_validation_pipeline", fromlist=["validate_task_mapping"]
        ).validate_task_mapping(
            {"status": "pending", "chain": "on", "chainID": "bad", "link": 0},
            route=WorkflowRoute.RECURRING_EDIT,
            source_query="mapping-test",
        )
        self.assertEqual(report.status, ValidationStatus.INVALID)
        self.assertTrue(report.findings)

    def test_transition_policy_rejects_identity_edits(self) -> None:
        old = _observation()
        changed = old.to_mapping()
        changed["link"] = 2
        new = TaskObservation.from_mapping(changed, source_query="transition-test")
        report = validate_task_transition(
            old,
            new,
            route=WorkflowRoute.RECURRING_EDIT,
            source_query="transition-test",
        )
        self.assertEqual(report.status, ValidationStatus.INVALID)
        self.assertEqual(report.findings[-1].code, "chain_identity_edit")

    def test_alias_normalization_preserves_empty_clear_syntax(self) -> None:
        task = {"description": "review am:"}
        self.assertTrue(normalize_description_uda_aliases(task, enabled=True))
        self.assertEqual(task["description"], "review")
        self.assertNotIn("anchor_mode", task)

    def test_alias_normalization_is_disabled_without_mutation(self) -> None:
        task = {"description": "review am:all"}
        self.assertFalse(normalize_description_uda_aliases(task, enabled=False))
        self.assertEqual(task, {"description": "review am:all"})

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
