from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import json
import unittest

from nautical_core.modify_models import TaskView
from nautical_core.task_changes import ChangeAction, PatchOperation, TaskChangeError, TaskPatch
from nautical_core.task_codec import DEFAULT_TASK_CODEC, TASK_OBSERVATION_SCHEMA, TaskCodec, TaskCodecError
from nautical_core.integration_models import (
    Absent,
    CommandFailureKind,
    FailureEvidence,
    Found,
    IntegrationContractError,
    TaskCommand,
    TaskCommandResult,
    Unavailable,
)
from nautical_core.task_models import (
    FieldPresence,
    InvalidTask,
    NauticalTask,
    TaskDraft,
    TaskObservation,
    TaskOperation,
    TaskStatus,
    TaskTimestamp,
    ValidatedTask,
    validate_task,
)


class TaskDomainModelTests(unittest.TestCase):
    def test_mutation_guard_and_outcome_reject_incomplete_evidence(self) -> None:
        from nautical_core.integration_models import (
            CommandFailureKind,
            FailureEvidence,
            GuardTimestamp,
            GuardTimestampField,
            IntegrationContractError,
            MutationGuard,
            MutationOperation,
            MutationOutcome,
            MutationOutcomeKind,
            MutationPostcondition,
            TaskCommand,
            Unavailable,
        )

        command = TaskCommand(("task", "uuid", "modify", "chain:off"), "disable chain", 12.0)
        busy = FailureEvidence(
            command, CommandFailureKind.BUSY, 1, 1, 0.2, True, "lock active"
        )
        timestamps = (
            GuardTimestamp(GuardTimestampField.MODIFIED, "20260813T070000Z"),
            GuardTimestamp(GuardTimestampField.DUE, "20260814T070000Z"),
        )
        guard = MutationGuard(
            "00000000-0000-4000-8000-000000000921",
            "pending",
            "chain-1",
            7,
            "rf1-example",
            timestamps,
            2,
        )
        success = MutationOutcome(
            MutationOperation.CHAIN_DISABLE,
            MutationOutcomeKind.APPLIED,
            guard,
            (MutationPostcondition.CHAIN_DISABLED,),
        )
        retry = MutationOutcome(
            MutationOperation.CHAIN_DISABLE,
            MutationOutcomeKind.RETRYABLE,
            guard,
            reason="Taskwarrior is busy",
            failure=busy,
        )
        self.assertEqual(success.postconditions, (MutationPostcondition.CHAIN_DISABLED,))
        self.assertIs(retry.failure, busy)

        invalid_cases = (
            lambda: MutationGuard("", "pending", "chain-1", 7, "rf1", timestamps, 0),
            lambda: MutationGuard("uuid", "pending", "", 7, "rf1", timestamps, 0),
            lambda: MutationGuard("uuid", "pending", "chain-1", -1, "rf1", timestamps, 0),
            lambda: MutationGuard("uuid", "pending", "chain-1", 7, "", timestamps, 0),
            lambda: MutationGuard(
                "uuid", "pending", "chain-1", 7, "rf1",
                (GuardTimestamp(GuardTimestampField.DUE, "20260814T070000Z"),), 0,
            ),
            lambda: MutationGuard("uuid", "pending", "chain-1", 7, "rf1", timestamps * 2, 0),
            lambda: MutationOutcome(
                MutationOperation.CHAIN_DISABLE, MutationOutcomeKind.APPLIED, guard, ()
            ),
            lambda: MutationOutcome(
                MutationOperation.CHAIN_DISABLE, MutationOutcomeKind.REJECTED, guard,
                (MutationPostcondition.CHAIN_DISABLED,), "guard mismatch",
            ),
            lambda: MutationOutcome(
                MutationOperation.CHAIN_DISABLE, MutationOutcomeKind.RETRYABLE, guard,
                reason="busy without evidence",
            ),
            lambda: MutationOutcome(
                MutationOperation.CHAIN_DISABLE, MutationOutcomeKind.APPLIED,
                Unavailable("guard lookup", busy),
                (MutationPostcondition.CHAIN_DISABLED,),
            ),
        )
        for make_invalid in invalid_cases:
            with self.subTest(make_invalid=make_invalid), self.assertRaises(IntegrationContractError):
                make_invalid()

    def test_every_mutation_operation_has_its_matching_success_postcondition(self) -> None:
        from nautical_core.integration_models import (
            GuardTimestamp,
            GuardTimestampField,
            MutationGuard,
            MutationOperation,
            MutationOutcome,
            MutationOutcomeKind,
            MutationPostcondition,
        )

        guard = MutationGuard(
            "parent-uuid",
            "completed",
            "chain-3",
            2,
            "rf1-states",
            (GuardTimestamp(GuardTimestampField.MODIFIED, "20260813T090000Z"),),
            0,
        )
        expected = {
            MutationOperation.CHILD_IMPORT: MutationPostcondition.CHILD_IMPORTED,
            MutationOperation.CHILD_COMPENSATION: MutationPostcondition.CHILD_COMPENSATED,
            MutationOperation.PARENT_LINK: MutationPostcondition.PARENT_LINKED,
            MutationOperation.PARENT_LINK_CLEAR: MutationPostcondition.PARENT_LINK_CLEARED,
            MutationOperation.CHAIN_DISABLE: MutationPostcondition.CHAIN_DISABLED,
            MutationOperation.NATIVE_UNTIL_REPAIR: MutationPostcondition.NATIVE_UNTIL_REPAIRED,
            MutationOperation.METADATA_REPAIR: MutationPostcondition.METADATA_REPAIRED,
        }
        for operation, postcondition in expected.items():
            with self.subTest(operation=operation):
                outcome = MutationOutcome(
                    operation,
                    MutationOutcomeKind.APPLIED,
                    guard,
                    (postcondition,),
                )
                self.assertIs(outcome.operation, operation)
    def test_named_mutation_requests_require_typed_targets_and_preserve_payloads(self) -> None:
        from nautical_core.integration_models import (
            ChainDisablePayload,
            ChildImportPayload,
            GuardTimestamp,
            GuardTimestampField,
            IntegrationContractError,
            MetadataRepairPayload,
            MutationGuard,
            MutationOperation,
            MutationRequest,
            NativeUntilRepairPayload,
            ParentLinkPayload,
        )
        from nautical_core.task_changes import TaskPatch
        from nautical_core.task_models import TaskUUID

        parent_uuid = "00000000-0000-4000-8000-000000000922"
        child_uuid = "00000000-0000-4000-8000-000000000923"
        guard = MutationGuard(
            parent_uuid,
            "completed",
            "chain-requests",
            7,
            "rf1-requests",
            (
                GuardTimestamp(GuardTimestampField.MODIFIED, "20260813T070000Z"),
                GuardTimestamp(GuardTimestampField.UNTIL, "20260813T200000Z"),
            ),
            0,
        )
        child = ChildImportPayload(
            parent_uuid,
            child_uuid,
            "chain-requests",
            8,
            (
                ("uuid", child_uuid),
                ("chainID", "chain-requests"),
                ("link", 8),
                ("prevLink", parent_uuid[:8]),
                ("description", "typed child"),
            ),
        )
        imported = MutationRequest(MutationOperation.CHILD_IMPORT, guard, child)
        self.assertEqual(imported.payload.child_uuid, child_uuid)
        self.assertEqual(imported.payload.to_dict()["description"], "typed child")

        parent_link = MutationRequest.parent_link(
            guard,
            TaskPatch.parent_link(TaskUUID(parent_uuid), TaskUUID(child_uuid)),
        )
        self.assertEqual(parent_link.payload, ParentLinkPayload(parent_uuid, child_uuid[:8]))
        disabled = MutationRequest.chain_disable(
            guard, TaskPatch.chain_disable(TaskUUID(parent_uuid))
        )
        self.assertEqual(disabled.payload, ChainDisablePayload(parent_uuid))
        repaired_until = MutationRequest.native_until_repair(
            guard,
            TaskPatch.native_until_repair(
                TaskUUID(parent_uuid), TaskTimestamp(datetime(2026, 8, 14, 20, tzinfo=timezone.utc))
            ),
        )
        self.assertEqual(repaired_until.payload, NativeUntilRepairPayload(
            parent_uuid, "20260813T200000Z", "2026-08-14T20:00:00Z"
        ))
        metadata = MutationRequest.metadata_repair(
            guard,
            TaskPatch.metadata_repair(TaskUUID(parent_uuid), nextLink=child_uuid[:8]),
        )
        self.assertEqual(metadata.payload.to_dict(), {"nextLink": child_uuid[:8]})

        with self.assertRaises(IntegrationContractError):
            MutationRequest(
                MutationOperation.CHILD_IMPORT,
                guard,
                {"uuid": child_uuid},  # type: ignore[arg-type]
            )
        with self.assertRaises(IntegrationContractError):
            MutationRequest.parent_link(
                guard,
                TaskPatch.parent_link(TaskUUID(child_uuid), TaskUUID(parent_uuid)),
            )

    def test_outbox_identity_is_stable_and_finalization_requires_all_verified_mutations(self) -> None:
        from nautical_core.integration_models import (
            GuardTimestamp,
            GuardTimestampField,
            IntegrationContractError,
            MutationGuard,
            MutationOperation,
            MutationOutcome,
            MutationOutcomeKind,
            MutationPostcondition,
            OutboxIntent,
            OutboxOutcome,
            OutboxOutcomeKind,
            OutboxStage,
        )
        from nautical_core.lifecycle_models import LifecycleEvent, LifecycleIdentity

        guard = MutationGuard(
            "parent-uuid",
            "completed",
            "chain-2",
            9,
            "rf1-outbox",
            (GuardTimestamp(GuardTimestampField.MODIFIED, "20260813T080000Z"),),
            0,
        )
        identity = LifecycleIdentity(
            "chain-2", "parent-uuid", 9, 10, LifecycleEvent.COMPLETE
        )
        operations = (MutationOperation.CHILD_IMPORT, MutationOperation.PARENT_LINK)
        postconditions = (
            MutationPostcondition.CHILD_IMPORTED,
            MutationPostcondition.PARENT_LINKED,
        )
        intent = OutboxIntent(identity, guard, operations, postconditions)
        retried = OutboxIntent(
            identity, guard, operations, postconditions, max_attempts=8
        )
        self.assertEqual(intent.intent_id, retried.intent_id)

        imported = MutationOutcome(
            MutationOperation.CHILD_IMPORT,
            MutationOutcomeKind.APPLIED,
            guard,
            (MutationPostcondition.CHILD_IMPORTED,),
        )
        linked = MutationOutcome(
            MutationOperation.PARENT_LINK,
            MutationOutcomeKind.ALREADY_APPLIED,
            guard,
            (MutationPostcondition.PARENT_LINKED,),
        )
        finalized = OutboxOutcome(
            intent, OutboxStage.FINALIZED, OutboxOutcomeKind.FINALIZED,
            (imported, linked),
        )
        self.assertEqual(finalized.intent.intent_id, intent.intent_id)
        with self.assertRaises(IntegrationContractError):
            OutboxOutcome(
                intent, OutboxStage.FINALIZED, OutboxOutcomeKind.FINALIZED,
                (imported,),
            )

    def test_integration_reads_preserve_absent_unavailable_and_validation(self) -> None:
        command = TaskCommand(("task", "rc.hooks=off", "export"), "chain snapshot", 12.0)
        result = TaskCommandResult(
            command, 0, '[{"description":"Répéter 🌊"}]', "", CommandFailureKind.SUCCESS, 1, 0.25
        )
        self.assertTrue(result.ok)

        evidence = FailureEvidence(
            command, CommandFailureKind.BUSY, 1, 2, 0.4, True, "Taskwarrior lock active"
        )
        found = Found({"uuid": "task-uuid"}, "uuid lookup")
        absent = Absent("uuid lookup", "authoritative export contained no match")
        unavailable = Unavailable("uuid lookup", evidence)
        self.assertEqual(found.value["uuid"], "task-uuid")
        self.assertNotEqual(absent, unavailable)
        self.assertTrue(unavailable.retryable)

        invalid_cases = (
            lambda: TaskCommand((), "query", 1.0),
            lambda: TaskCommand(("task",), "", 1.0),
            lambda: TaskCommand(("task",), "query", 0.0),
            lambda: TaskCommandResult(command, 1, "", "", CommandFailureKind.SUCCESS, 1, 0.1),
            lambda: FailureEvidence(command, CommandFailureKind.ABSENT, 1, 1, 0.1, False),
            lambda: FailureEvidence(command, CommandFailureKind.REJECTED, 1, 1, 0.1, True),
            lambda: Found(None, "uuid lookup"),
            lambda: Absent("uuid lookup", ""),
            lambda: Unavailable("uuid lookup", object()),
        )
        for make_invalid in invalid_cases:
            with self.subTest(make_invalid=make_invalid), self.assertRaises(IntegrationContractError):
                make_invalid()
        with self.assertRaises(FrozenInstanceError):
            command.purpose = "changed"  # type: ignore[misc]

    def test_observation_preserves_source_evidence_and_is_immutable(self) -> None:
        row = {
            "uuid": "00000000-0000-4000-8000-000000000001", "status": "pending", "link": 1,
            "due": "20260821T090000Z", "description": "héllo", "tags": ["a", "b"],
            "id": 7, "urgency": 4.2,
        }
        observation = TaskObservation.from_mapping(row, source_query="uuid:00000000", snapshot_id="snap-1")
        row["tags"].append("mutated")
        self.assertIs(observation.field("uuid").presence, FieldPresence.VALUE)
        self.assertIs(observation.field("status").value, TaskStatus.PENDING)
        self.assertIs(observation.field("until").presence, FieldPresence.ABSENT)
        self.assertIs(
            TaskObservation.from_mapping({"until": None}, source_query="uuid:00000000").field("until").presence,
            FieldPresence.NULL,
        )
        self.assertIs(observation.field("missing").presence, FieldPresence.ABSENT)
        self.assertEqual(observation.to_mapping()["tags"], ["a", "b"])
        self.assertNotIn("id", observation.semantic_fingerprint)
        self.assertNotIn("urgency", observation.semantic_fingerprint)

        equivalent = TaskObservation.from_mapping(
            {
                **observation.to_mapping(), "link": 1.0,
                "due": "2026-08-21T09:00:00+00:00", "id": 99, "urgency": 100.0,
            },
            source_query="uuid:00000000", snapshot_id="snap-1",
        )
        self.assertEqual(observation, equivalent)
        self.assertEqual(equivalent.field("link").value.value, 1)
        malformed = TaskObservation.from_mapping(
            {"uuid": "bad", "link": 1.5, "status": "future"}, source_query="broad:all"
        )
        self.assertEqual({issue.code for issue in malformed.issues}, {"invalid_value", "unknown_status"})

    def test_validated_task_projection_retains_observation_and_operation_rules(self) -> None:
        row = {
            "uuid": "00000000-0000-4000-8000-000000000002", "status": "pending",
            "chain": "on", "chainID": "chain-2", "link": 2.0, "anchor": "w:mon",
            "anchor_mode": "skip", "due": "20260824T090000Z", "until": None,
            "description": "typed task",
        }
        observation = TaskObservation.from_mapping(row, source_query="chain:chain-2", snapshot_id="snap-2")
        result = validate_task(observation, TaskOperation.SCHEDULE)
        self.assertIsInstance(result, ValidatedTask)
        self.assertIs(NauticalTask.from_observation(observation), result.task)
        self.assertIs(result.task.status, TaskStatus.PENDING)
        self.assertEqual(result.task.recurrence.kind.value, "anchor")
        self.assertEqual(result.task.temporal.presence["until"].value, "null")

        malformed = TaskObservation.from_mapping(
            {"uuid": "bad", "status": "pending", "chain": "on", "anchor": "w:mon"},
            source_query="uuid:bad",
        )
        rejected = validate_task(malformed, TaskOperation.QUERY)
        self.assertIsInstance(rejected, InvalidTask)
        self.assertIs(rejected.observation, malformed)
        self.assertTrue(rejected.issues)

        missing_reference = TaskObservation.from_mapping(
            {**row, "due": None, "scheduled": None}, source_query="chain:chain-2", snapshot_id="snap-3"
        )
        self.assertIsInstance(validate_task(missing_reference, TaskOperation.COMPLETION), InvalidTask)

    def test_task_view_preserves_typed_temporal_presence(self) -> None:
        base = {
            "uuid": "00000000-0000-4000-8000-000000000003", "status": "pending",
            "chainID": "view-chain", "link": 1, "anchor": "w:mon", "due": "20260824T090000Z",
        }
        view = TaskView.from_mapping(base)
        self.assertIsNotNone(view.timestamp("due"))
        self.assertIsNone(view.timestamp("scheduled"))
        self.assertEqual(TaskView.from_mapping({**base, "due": None}).temporal.presence["due"].value, "null")
        self.assertIsNone(TaskView.from_mapping({**base, "due": "not-a-date"}).timestamp("due"))

    def test_codec_keeps_serialization_contracts_separate_and_strict(self) -> None:
        codec = TaskCodec()
        row = {
            "uuid": "00000000-0000-4000-8000-000000000003", "status": "pending",
            "chainID": "codec-chain", "link": 3.0, "anchor": "w:mon",
            "description": "Répéter 🌊", "tags": ["one", "two"],
        }
        observation = codec.decode_export(
            json.dumps([row], ensure_ascii=False), source_query="chain:codec-chain", snapshot_id="codec-snap"
        )[0]
        self.assertEqual(json.loads(codec.encode_task_import(observation))["description"], "Répéter 🌊")
        hook_json = codec.encode_hook_stdout(observation.to_mapping())
        self.assertIn("Répéter 🌊", hook_json)
        self.assertEqual(json.loads(hook_json)["uuid"], row["uuid"])
        self.assertEqual(json.loads(codec.encode_query_json({"schema": "nautical.query.test", "value": "🌊"}))["schema"], "nautical.query.test")
        diagnostic = json.loads(codec.encode_diagnostic(observation))
        self.assertEqual((diagnostic["schema"], diagnostic["version"]), (TASK_OBSERVATION_SCHEMA, 1))

        for invalid in ("", "{}", "[1]", '[{"link": NaN}]'):
            with self.subTest(invalid=invalid), self.assertRaises(TaskCodecError):
                codec.decode_export(invalid, source_query="invalid")
        with self.assertRaises(TaskCodecError):
            codec.encode_hook_stdout({"bad": object()})

    def test_codec_decodes_leading_hook_rows_once_and_keeps_bad_row_evidence(self) -> None:
        first = {"uuid": "00000000-0000-4000-8000-000000000101", "status": "pending"}
        second = {"uuid": "00000000-0000-4000-8000-000000000102", "status": "completed"}
        concatenated = json.dumps(first) + json.dumps(second)
        rows, index, error = DEFAULT_TASK_CODEC.decode_leading_rows(concatenated, source_query="hook framing")
        self.assertFalse(error)
        self.assertEqual((len(rows), index), (2, len(concatenated)))
        rows, index, error = DEFAULT_TASK_CODEC.decode_leading_rows(
            json.dumps([first, second]), source_query="hook array"
        )
        self.assertFalse(error)
        self.assertEqual(len(rows), 2)
        self.assertGreater(index, 0)
        rows, _index, error = DEFAULT_TASK_CODEC.decode_leading_rows(
            json.dumps([first, {"status": "pending", "link": "bad"}]), source_query="malformed hook array"
        )
        self.assertFalse(error)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[1].issues)

    def test_draft_and_patch_make_mutation_semantics_explicit(self) -> None:
        row = {
            "uuid": "00000000-0000-4000-8000-000000000004", "status": "pending", "chain": "on",
            "chainID": "draft-chain", "link": 4, "anchor": "w:mon", "anchor_mode": "skip",
            "due": "20260824T090000Z", "description": "draft source",
        }
        validated = validate_task(TaskObservation.from_mapping(row, source_query="chain:draft-chain"), TaskOperation.SCHEDULE)
        self.assertIsInstance(validated, ValidatedTask)
        target_time = TaskTimestamp(datetime(2026, 8, 31, 9, tzinfo=timezone.utc))
        draft = TaskDraft(
            identity=validated.task.identity, description="draft child", recurrence=validated.task.recurrence,
            target=target_time, fields={"project": "Routines", "tags": ["🌊"]},
        )
        self.assertEqual(draft.to_mapping()["status"], "pending")
        self.assertEqual(draft.to_mapping()["tags"], ["🌊"])
        reversed_fields = TaskDraft(
            validated.task.identity, "draft child", validated.task.recurrence, target_time,
            {"tags": ["🌊"], "project": "Routines"},
        )
        self.assertEqual(draft.fingerprint, reversed_fields.fingerprint)

        target = validated.task.identity.task_uuid
        patch = TaskPatch.set(target, PatchOperation.ORDINARY_CARRY, scheduled="2026-08-31T08:30:00Z")
        clear = TaskPatch.clear(target, PatchOperation.ORDINARY_CARRY, "wait")
        self.assertEqual(patch.set_values()["scheduled"], "2026-08-31T08:30:00Z")
        self.assertEqual(clear.clear_fields(), ("wait",))
        self.assertIs(TaskPatch.parent_link(target, target).changes[0].action, ChangeAction.SET)
        self.assertEqual(
            TaskPatch.set(target, PatchOperation.METADATA_REPAIR, a=1, b=2).fingerprint,
            TaskPatch.set(target, PatchOperation.METADATA_REPAIR, b=2, a=1).fingerprint,
        )
        invalid_patches = (
            lambda: TaskPatch.set(target, PatchOperation.METADATA_REPAIR, modified="now"),
            lambda: TaskPatch.set(target, PatchOperation.METADATA_REPAIR, chainID="other"),
            lambda: TaskPatch(target, PatchOperation.ORDINARY_CARRY, ()),
        )
        for make_invalid in invalid_patches:
            with self.subTest(make_invalid=make_invalid), self.assertRaises(TaskChangeError):
                make_invalid()
        for field in ("uuid", "chainID", "nextLink", "description", "anchor"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                TaskDraft(
                    validated.task.identity, "invalid draft", validated.task.recurrence,
                    target_time, fields={field: "unexpected"},
                )
