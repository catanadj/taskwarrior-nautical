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
    ParentLinkPayload,
    TaskCommand,
    Unavailable,
)
from nautical_core.lifecycle_models import recurrence_fingerprint
from nautical_core.task_models import TaskObservation
from nautical_core.taskwarrior_mutations import TaskwarriorMutationService


class MutationHardeningTests(unittest.TestCase):
    def test_batch_postverification_fails_closed_for_untrusted_snapshots(self) -> None:
        from nautical_core.task_set_reads import SetReadResult, SetReadStatus

        parent_uuid = "00000000-0000-4000-8000-000000000930"
        child_uuid = "00000000-0000-4000-8000-000000000931"
        parent = {
            "uuid": parent_uuid, "status": "completed", "chain": "on",
            "chainID": "batch-verify", "link": 1,
            "modified": "20260813T120000Z", "cp": "1d",
            "nextLink": child_uuid[:8],
        }
        child = {
            "uuid": child_uuid, "chainID": "batch-verify", "link": 2,
            "prevLink": parent_uuid[:8], "status": "pending", "chain": "on",
            "cp": "1d",
        }
        guard = MutationGuard(
            task_uuid=parent_uuid, status=parent["status"], chain_id=parent["chainID"],
            link=parent["link"], recurrence_identity=recurrence_fingerprint(parent),
            timestamps=(GuardTimestamp(GuardTimestampField.MODIFIED, parent["modified"]),),
            expected_mutation_epoch=0, chain="on",
        )
        child_payload = ChildImportPayload(
            parent_uuid=parent_uuid, child_uuid=child_uuid,
            chain_id=child["chainID"], target_link=2, fields=tuple(child.items()),
        )
        child_request = MutationRequest(MutationOperation.CHILD_IMPORT, guard, child_payload)
        parent_request = MutationRequest(
            MutationOperation.PARENT_LINK,
            guard,
            ParentLinkPayload(parent_uuid, child_uuid[:8]),
        )

        class Repository:
            def __init__(self, mode):
                self.mode = mode

            def read_uuid_set(self, request):
                if self.mode == "unavailable":
                    evidence = FailureEvidence(
                        TaskCommand(("task", "export"), "verify", 1.0),
                        CommandFailureKind.BUSY, 1, 1, 0.01, True, "lock active",
                    )
                    return SetReadResult(
                        SetReadStatus.UNAVAILABLE, request.uuids, failures=(evidence,),
                    )
                if self.mode == "malformed":
                    return SetReadResult(
                        SetReadStatus.MALFORMED, request.uuids, evidence=("malformed",),
                    )
                found = {child_uuid: dict(child), parent_uuid: dict(parent)}
                if self.mode == "stale":
                    found[child_uuid]["link"] = 99
                    found[parent_uuid]["nextLink"] = "stale00"
                return SetReadResult(
                    SetReadStatus.DUPLICATE if self.mode == "duplicate" else SetReadStatus.COMPLETE,
                    request.uuids,
                    found=found,
                    complete_for_requested_identities=self.mode != "duplicate",
                    evidence=("duplicate identity",) if self.mode == "duplicate" else (),
                )

        for mode, expected in (
            ("unavailable", MutationOutcomeKind.RETRYABLE),
            ("malformed", MutationOutcomeKind.MANUAL_REVIEW),
            ("stale", MutationOutcomeKind.MANUAL_REVIEW),
            ("duplicate", MutationOutcomeKind.MANUAL_REVIEW),
        ):
            with self.subTest(snapshot=mode):
                service = TaskwarriorMutationService(SimpleNamespace(
                    repository=Repository(mode), mutation_epoch=0,
                ))
                child_outcome = service.verify_lifecycle_children((child_request,))[child_uuid]
                parent_outcome = service.verify_lifecycle_parents((parent_request,))[parent_uuid]
                self.assertIs(child_outcome.kind, expected)
                self.assertIs(parent_outcome.kind, expected)

        from nautical_core.taskwarrior_mutations import _child_import_matches

        null_payload = ChildImportPayload(
            parent_uuid=parent_uuid, child_uuid=child_uuid,
            chain_id=child["chainID"], target_link=2,
            fields=tuple(dict(child, anchor_file="null").items()),
        )
        self.assertTrue(_child_import_matches(child, null_payload, parent_uuid))

    def test_lifecycle_child_prefetch_reuses_authoritative_uuid_set_read(self) -> None:
        from nautical_core.task_set_reads import SetReadResult, SetReadStatus

        parent_uuid = "00000000-0000-4000-8000-000000000928"
        child_uuid = "00000000-0000-4000-8000-000000000929"
        parent = {
            "uuid": parent_uuid, "status": "completed", "chain": "on",
            "chainID": "prefetch-chain", "link": 1,
            "modified": "20260813T120000Z",
        }
        payload = ChildImportPayload(
            parent_uuid=parent_uuid,
            child_uuid=child_uuid,
            chain_id="prefetch-chain",
            target_link=2,
            fields=(("uuid", child_uuid), ("chainID", "prefetch-chain"),
                    ("link", 2), ("prevLink", parent_uuid[:8])),
        )

        class Repository:
            def __init__(self) -> None:
                self.requests = []
                self.uuid_reads = []
                self.broad_reads = 0

            def read_uuid_set(self, request):
                self.requests.append(request)
                return SetReadResult(
                    SetReadStatus.COMPLETE,
                    request.uuids,
                    found={parent_uuid: parent},
                    absent=tuple(identity for identity in request.uuids if identity != parent_uuid),
                    complete_for_requested_identities=True,
                )

            def by_uuid(self, uuid_value, *, refresh=False):
                del refresh
                self.uuid_reads.append(uuid_value)
                return Found(parent, f"uuid:{uuid_value}")

            def broad_snapshot(self, **_kwargs):
                self.broad_reads += 1
                return Unavailable("unexpected broad read")

        repository = Repository()
        service = TaskwarriorMutationService(SimpleNamespace(
            repository=repository, mutation_epoch=0,
        ))
        service.preflight_lifecycle_batch(
            (payload,), parent_expectations=((parent_uuid, child_uuid[:8]),),
        )

        self.assertEqual(len(repository.requests), 1)
        self.assertEqual(set(repository.requests[0].uuids), {parent_uuid, child_uuid})
        self.assertEqual(repository.uuid_reads, [])
        self.assertEqual(repository.broad_reads, 0)
        self.assertIn(child_uuid.lower(), service._prefetched_children)
        self.assertEqual(service._prefetched_parents.get(parent_uuid), parent)

    def test_lifecycle_batch_prefetch_uses_one_union_uuid_set_read(self) -> None:
        from nautical_core.task_set_reads import SetReadResult, SetReadStatus

        parent_uuids = tuple(f"00000000-0000-4000-8000-00000000093{i}" for i in range(3))
        child_uuids = tuple(f"00000000-0000-4000-8000-00000000094{i}" for i in range(3))
        parents = {
            uuid: {
                "uuid": uuid, "status": "completed", "chain": "on",
                "chainID": "prefetch-batch", "link": index + 1,
                "modified": "20260813T120000Z",
            }
            for index, uuid in enumerate(parent_uuids)
        }
        payloads = tuple(
            ChildImportPayload(
                parent_uuid=parent_uuid,
                child_uuid=child_uuid,
                chain_id="prefetch-batch",
                target_link=index + 2,
                fields=(("uuid", child_uuid), ("chainID", "prefetch-batch"),
                        ("link", index + 2), ("prevLink", parent_uuid[:8])),
            )
            for index, (parent_uuid, child_uuid) in enumerate(zip(parent_uuids, child_uuids))
        )

        class Repository:
            def __init__(self) -> None:
                self.requests = []

            def read_uuid_set(self, request):
                self.requests.append(request)
                return SetReadResult(
                    SetReadStatus.COMPLETE,
                    request.uuids,
                    found=parents,
                    absent=tuple(identity for identity in request.uuids if identity not in parents),
                    complete_for_requested_identities=True,
                )

        repository = Repository()
        service = TaskwarriorMutationService(SimpleNamespace(
            repository=repository, mutation_epoch=0,
        ))
        service.preflight_lifecycle_batch(
            payloads,
            parent_expectations=tuple(
                (uuid, f"{index + 2:08x}") for index, uuid in enumerate(parent_uuids)
            ),
        )

        self.assertEqual(len(repository.requests), 1)
        self.assertEqual(set(repository.requests[0].uuids), set(parent_uuids + child_uuids))

    def test_existing_child_is_acknowledged_only_when_complete_and_matching(self) -> None:
        parent = {
            "uuid": "00000000-0000-4000-8000-000000000926",
            "status": "completed", "chain": "on", "chainID": "child-check",
            "link": 4, "modified": "20260813T110000Z",
        }
        child = {
            "uuid": "00000000-0000-4000-8000-000000000927",
            "chainID": "child-check", "link": 5, "prevLink": parent["uuid"][:8],
            "status": "pending", "chain": "on", "cp": "1d", "description": "child",
            "due": "20260814T090000Z",
        }
        payload = ChildImportPayload(
            parent_uuid=parent["uuid"], child_uuid=child["uuid"],
            chain_id=child["chainID"], target_link=5, fields=tuple(child.items()),
        )
        guard = MutationGuard(
            task_uuid=parent["uuid"], status=parent["status"], chain_id=parent["chainID"],
            link=parent["link"], recurrence_identity=recurrence_fingerprint(parent),
            timestamps=(GuardTimestamp(GuardTimestampField.MODIFIED, parent["modified"]),),
            expected_mutation_epoch=0, chain="on",
        )

        class Repository:
            def __init__(self, existing):
                self.rows = {parent["uuid"]: dict(parent), child["uuid"]: existing}

            def by_uuid(self, uuid_value, *, refresh=False):
                del refresh
                row = self.rows.get(uuid_value)
                if row is None:
                    return Absent(f"uuid:{uuid_value}", "not present")
                return Found(TaskObservation.from_mapping(row, source_query="uuid"), "uuid")

            def exact_child_slot(self, chain_id, link, **_kwargs):
                row = next((item for item in self.rows.values()
                            if item.get("chainID") == chain_id and int(item.get("link", 0)) == link), None)
                if row is None:
                    return Absent("slot", "not present")
                return Found(TaskObservation.from_mapping(row, source_query="slot"), "slot")

        class Client:
            def execute(self, *_args, **_kwargs):
                raise AssertionError("an existing child must not trigger a mutation")

        def apply(existing, payload_values=None):
            request_payload = payload
            if payload_values is not None:
                request_payload = ChildImportPayload(
                    parent_uuid=parent["uuid"], child_uuid=child["uuid"],
                    chain_id=str(payload_values["chainID"]), target_link=int(payload_values["link"]),
                    fields=tuple(payload_values.items()),
                )
            repository = Repository(existing)
            service = TaskwarriorMutationService(SimpleNamespace(
                context=SimpleNamespace(mutation_capable=True), repository=repository,
                client=Client(), mutation_epoch=0, record_mutation=lambda **_kwargs: 1,
            ))
            return service.apply(MutationRequest(MutationOperation.CHILD_IMPORT, guard, request_payload))

        invalid_rows = (
            ("missing prevLink", {key: value for key, value in child.items() if key != "prevLink"}),
            ("wrong status", dict(child, status="completed")),
            ("disabled chain", dict(child, chain="off")),
            ("changed recurrence metadata", dict(child, cp="2d")),
            ("sync replacement", dict(child, chainID="replacement-chain", link=99)),
            ("future deleted child", dict(child, status="deleted", until="29990101T000000Z")),
        )
        for label, existing in invalid_rows:
            with self.subTest(existing=label):
                outcome = apply(existing)
                self.assertIs(outcome.kind, MutationOutcomeKind.CONFLICT)
                self.assertFalse(outcome.postconditions)

        expired_values = dict(child, status="deleted", until="20000101T000000Z")
        expired = apply(expired_values, payload_values=expired_values)
        self.assertIs(expired.kind, MutationOutcomeKind.ALREADY_APPLIED, expired)
        future_values = dict(child, status="deleted", until="29990101T000000Z")
        future = apply(future_values, payload_values=future_values)
        self.assertIs(future.kind, MutationOutcomeKind.CONFLICT)
        valid = apply(dict(child))
        self.assertIs(valid.kind, MutationOutcomeKind.ALREADY_APPLIED)

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
