import unittest
from datetime import datetime, timezone

from nautical_core.chain_integrity_engine import IntegrityEngineResult
from nautical_core.chain_integrity_models import IntegrityReportStatus
from nautical_core.integrity_report import components
from nautical_core.lifecycle_models import (
    LifecycleAction,
    LifecycleEvent,
    LifecycleIdentity,
    LifecyclePlan,
    ParentGuard,
    recurrence_fingerprint,
)
from nautical_core.lifecycle_recovery_models import RecoveryPlanResult
from nautical_core.reconcile_report import describe_recovery_result
from nautical_core.task_codec import DEFAULT_TASK_CODEC


class IntegrityReportContractTests(unittest.TestCase):
    def test_recovery_evidence_includes_local_child_time_when_formatter_exists(self):
        parent = {
            "uuid": "11111111-0000-4000-8000-000000000001",
            "status": "completed",
            "description": "remote completion",
            "cp": "1d",
            "chain": "on",
            "chainID": "11111111",
            "link": 1,
            "due": "20260703T090000Z",
        }
        observation = DEFAULT_TASK_CODEC.decode_row(
            parent, source_query="local-child-time-contract"
        )
        guard = ParentGuard(
            status="completed",
            chain="on",
            chain_id="11111111",
            link=1,
            recurrence_fingerprint=recurrence_fingerprint(parent),
            modified="",
        )
        identity = LifecycleIdentity(
            chain_id="11111111",
            parent_uuid=parent["uuid"],
            source_link=1,
            target_link=2,
            event=LifecycleEvent.COMPLETE,
        )
        plan = RecoveryPlanResult(
            observation,
            LifecyclePlan(
                identity=identity,
                action=LifecycleAction.SPAWN_CHILD,
                parent_guard=guard,
                child_payload=(("due", "20260704T110000Z"),),
            ),
            reason="missing next link",
            child_due=datetime(2026, 7, 4, 11, 0, tzinfo=timezone.utc),
        )

        evidence = describe_recovery_result(
            plan, fmt_dt_local=lambda _dt: "Sat 2026-07-04 14:00 EEST"
        )

        self.assertEqual(evidence["child_local"], "Sat 2026-07-04 14:00 EEST")

    def test_healthy_internal_status_maps_to_public_ok(self):
        payload = components(IntegrityEngineResult(IntegrityReportStatus.HEALTHY))
        self.assertEqual(payload["status"], "ok")
        self.assertIsNone(payload["failure"])

    def test_doctor_and_query_payloads_share_integrity_components(self):
        from nautical_core.chain_integrity_models import IntegrityReportStatus
        from nautical_core.integrity_report import doctor_findings, public_payload

        result = IntegrityEngineResult(IntegrityReportStatus.HEALTHY, reason="")
        shared = components(result)
        payload = public_payload(
            result, query={"kind": "all"}, configuration_fingerprint="cfg"
        )
        self.assertEqual(payload["status"], shared["status"])
        self.assertEqual(payload["findings"], shared["findings"])
        self.assertEqual(payload["plans"], shared["plans"])
        self.assertEqual(doctor_findings(result)[0]["id"], "chains.integrity")


if __name__ == "__main__":
    unittest.main()
