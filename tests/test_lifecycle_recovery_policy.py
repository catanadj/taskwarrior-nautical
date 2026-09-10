import unittest
from datetime import datetime, timezone

from nautical_core.lifecycle_reconciliation import LifecycleRecoveryPolicy
from nautical_core.lifecycle_models import LifecycleAction
from nautical_core.task_models import TaskObservation


class LifecycleRecoveryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.parse = lambda value: (datetime.fromisoformat(value.replace("Z", "+00:00")), None)
        self.policy = LifecycleRecoveryPolicy(
            parse_datetime=self.parse,
            compare_datetimes=lambda left, right: (left > right) - (left < right),
            validate_child=lambda _parent, _child: "",
            virtual_uuid=lambda _plan: "dryrun-chain-2",
        )

    def test_terminal_error_accepts_future_until_after_target(self):
        child = TaskObservation.from_mapping({
            "uuid": "child", "status": "pending", "due": "2030-01-01T09:00:00+00:00",
            "until": "2030-01-01T10:00:00+00:00",
        }, source_query="test")
        self.assertEqual(
            self.policy.terminal_error(child, datetime(2029, 1, 1, tzinfo=timezone.utc)),
            "",
        )

    def test_virtual_expired_child_marks_planned_child_deleted(self):
        parent = TaskObservation.from_mapping(
            {"uuid": "parent", "chainID": "chain", "link": 1}, source_query="test"
        )
        plan = type(
            "Plan",
            (),
            {
                "action": LifecycleAction.SPAWN_CHILD,
                "identity": type("Identity", (), {"chain_id": "chain", "target_link": 2})(),
                "child_dict": lambda _self: {
                    "chainID": "chain", "link": 2, "status": "pending",
                    "until": "2020-01-01T00:00:00+00:00",
                },
            },
        )()
        virtual, error = self.policy.virtual_expired_child(
            plan, parent=parent, recovery_at=datetime(2021, 1, 1, tzinfo=timezone.utc)
        )
        self.assertEqual(error, "")
        self.assertIsNotNone(virtual)
        self.assertEqual(virtual.observation.field("status").value, "deleted")

    def test_terminal_error_converts_parser_exception_to_recovery_reason(self):
        def raising_parse(_value):
            raise ValueError("malformed datetime")

        policy = LifecycleRecoveryPolicy(
            parse_datetime=raising_parse,
            compare_datetimes=self.policy.compare_datetimes,
            validate_child=self.policy.validate_child,
            virtual_uuid=self.policy.virtual_uuid,
        )
        child = TaskObservation.from_mapping(
            {
                "uuid": "child",
                "status": "pending",
                "due": "2030-01-01T09:00:00+00:00",
                "until": "2030-01-01T10:00:00+00:00",
            },
            source_query="test",
        )
        self.assertEqual(
            policy.terminal_error(child, datetime(2029, 1, 1, tzinfo=timezone.utc)),
            "live recovery child native until could not be parsed",
        )

    def test_virtual_expired_child_converts_parser_exception_to_recovery_reason(self):
        def raising_parse(_value):
            raise ValueError("malformed datetime")

        policy = LifecycleRecoveryPolicy(
            parse_datetime=raising_parse,
            compare_datetimes=self.policy.compare_datetimes,
            validate_child=self.policy.validate_child,
            virtual_uuid=self.policy.virtual_uuid,
        )
        parent = TaskObservation.from_mapping(
            {"uuid": "parent", "chainID": "chain", "link": 1},
            source_query="test",
        )
        plan = type(
            "Plan",
            (),
            {
                "action": LifecycleAction.SPAWN_CHILD,
                "identity": type("Identity", (), {"chain_id": "chain", "target_link": 2})(),
                "child_dict": lambda _self: {
                    "chainID": "chain",
                    "link": 2,
                    "status": "pending",
                    "until": "2020-01-01T00:00:00+00:00",
                },
            },
        )()
        virtual, error = policy.virtual_expired_child(
            plan,
            parent=parent,
            recovery_at=datetime(2021, 1, 1, tzinfo=timezone.utc),
        )
        self.assertIsNone(virtual)
        self.assertEqual(error, "planned child expiration could not be parsed")


if __name__ == "__main__":
    unittest.main()
