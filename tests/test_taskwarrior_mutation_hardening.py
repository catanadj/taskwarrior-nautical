from __future__ import annotations

from types import SimpleNamespace
import unittest

from nautical_core.integration_models import (
    Absent,
    ChildImportPayload,
    CommandFailureKind,
    FailureEvidence,
    Found,
    GuardTimestamp,
    GuardTimestampField,
    MutationGuard,
    MutationOperation,
    MutationOutcomeKind,
    MutationRequest,
    TaskCommand,
    Unavailable,
)
from nautical_core.lifecycle_models import recurrence_fingerprint
from nautical_core.task_models import TaskObservation
from nautical_core.taskwarrior_mutations import TaskwarriorMutationService


class MutationHardeningTests(unittest.TestCase):
    def test_child_import_refuses_ambiguous_slot_before_dispatch(self) -> None:
        parent = {
            "uuid": "11111111-1111-4111-8111-111111111111",
            "chain": "on",
            "chainID": "chain-a",
            "link": 1,
            "status": "completed",
            "modified": "20260922T120000Z",
        }
        child = {
            "uuid": "22222222-2222-4222-8222-222222222222",
            "chain": "on",
            "chainID": "chain-a",
            "link": 2,
            "prevLink": "11111111",
            "status": "pending",
        }
        guard = MutationGuard(
            task_uuid=parent["uuid"],
            status=parent["status"],
            chain_id=parent["chainID"],
            link=parent["link"],
            recurrence_identity=recurrence_fingerprint(parent),
            timestamps=(GuardTimestamp(GuardTimestampField.MODIFIED, parent["modified"]),),
            expected_mutation_epoch=0,
            chain="on",
        )
        payload = ChildImportPayload(
            parent_uuid=parent["uuid"],
            child_uuid=child["uuid"],
            chain_id=child["chainID"],
            target_link=child["link"],
            fields=tuple(child.items()),
        )
        request = MutationRequest(MutationOperation.CHILD_IMPORT, guard, payload)
        evidence = FailureEvidence(
            TaskCommand(("task", "export"), "ambiguous slot", 1.0),
            CommandFailureKind.INVALID_RESPONSE,
            1,
            1,
            0.0,
            False,
            "duplicate slot chain-a:2",
        )

        class Repository:
            def by_uuid(self, uuid_value, *, refresh=False):
                if uuid_value == parent["uuid"]:
                    return Found(TaskObservation.from_mapping(parent, source_query="test"), "parent")
                return Absent("child", "not present")

            def exact_child_slot(self, chain_id, link, **_kwargs):
                return Unavailable("slot", evidence)

        class Client:
            def execute(self, *_args, **_kwargs):
                raise AssertionError("ambiguous slot must not dispatch import")

        service = TaskwarriorMutationService(
            SimpleNamespace(
                context=SimpleNamespace(mutation_capable=True),
                repository=Repository(),
                client=Client(),
                mutation_epoch=0,
                record_mutation=lambda **_kwargs: 0,
            )
        )

        outcome = service.import_child(request)

        self.assertIs(outcome.kind, MutationOutcomeKind.MANUAL_REVIEW)
        self.assertIn("duplicate slot", outcome.reason)
        self.assertIn("chain=chain-a", outcome.reason)
        self.assertIn("link=2", outcome.reason)
        self.assertIn("parent=11111111", outcome.reason)
        self.assertIn("expected_child=22222222", outcome.reason)

    def test_second_device_candidate_for_occupied_slot_is_not_imported(self) -> None:
        parent = {
            "uuid": "33333333-3333-4333-8333-333333333333",
            "chain": "on", "chainID": "chain-race", "link": 4,
            "status": "completed", "modified": "20260922T120000Z",
        }
        existing = {
            "uuid": "44444444-4444-4444-8444-444444444444",
            "chain": "on", "chainID": "chain-race", "link": 5,
            "prevLink": "33333333", "status": "pending",
        }
        competing = dict(existing, uuid="55555555-5555-4555-8555-555555555555")
        guard = MutationGuard(
            task_uuid=parent["uuid"], status=parent["status"], chain_id=parent["chainID"],
            link=parent["link"], recurrence_identity=recurrence_fingerprint(parent),
            timestamps=(GuardTimestamp(GuardTimestampField.MODIFIED, parent["modified"]),),
            expected_mutation_epoch=0, chain="on",
        )
        payload = ChildImportPayload(
            parent_uuid=parent["uuid"], child_uuid=competing["uuid"],
            chain_id=competing["chainID"], target_link=competing["link"],
            fields=tuple(competing.items()),
        )
        request = MutationRequest(MutationOperation.CHILD_IMPORT, guard, payload)

        class Repository:
            def by_uuid(self, uuid_value, *, refresh=False):
                if uuid_value == parent["uuid"]:
                    return Found(TaskObservation.from_mapping(parent, source_query="test"), "parent")
                if uuid_value == existing["uuid"]:
                    return Found(TaskObservation.from_mapping(existing, source_query="test"), "child")
                return Absent("child", "not present")

            def exact_child_slot(self, chain_id, link, **_kwargs):
                return Found(TaskObservation.from_mapping(existing, source_query="test"), "slot")

        class Client:
            def execute(self, *_args, **_kwargs):
                raise AssertionError("competing device must not import a second child")

        service = TaskwarriorMutationService(
            SimpleNamespace(
                context=SimpleNamespace(mutation_capable=True), repository=Repository(), client=Client(),
                mutation_epoch=0, record_mutation=lambda **_kwargs: 0,
            )
        )

        outcome = service.import_child(request)

        self.assertIs(outcome.kind, MutationOutcomeKind.MANUAL_REVIEW)
        self.assertIn("occupied by 44444444", outcome.reason)
        self.assertIn("chain=chain-race", outcome.reason)
        self.assertIn("link=5", outcome.reason)
        self.assertIn("parent=33333333", outcome.reason)
        self.assertIn("expected_child=55555555", outcome.reason)


if __name__ == "__main__":
    unittest.main()
