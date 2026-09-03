import json
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from nautical_core.backup_service import StorageIO, create_manifest, publish_manifest
from nautical_core.restore_service import restore_backup, validate_backup
from nautical_core.lifecycle_models import ExecutionStage, LifecycleAction, LifecycleEvent, LifecycleIdentity, LifecyclePlan, ParentGuard
from nautical_core.lifecycle_outbox import LifecycleOutboxRepository
from dev_tools.nautical_golden_tests import _task_draft


class RestoreServiceTests(unittest.TestCase):
    def _backup(self, root: Path) -> Path:
        backup = root / "backup"
        backup.mkdir()
        (backup / "taskwarrior-export.json").write_text('[{"uuid":"u1","description":"café"}]\n', encoding="utf-8")
        state = backup / "source-state"
        state.mkdir()
        database = state / "outbox.db"
        connection = sqlite3.connect(database)
        connection.execute("CREATE TABLE marker (value TEXT)")
        connection.execute("INSERT INTO marker VALUES ('ok')")
        connection.commit()
        connection.close()
        (backup / "lifecycle-outbox.db").write_bytes(database.read_bytes())
        publish_manifest(backup / "manifest.json", create_manifest(backup, files=("taskwarrior-export.json", "lifecycle-outbox.db")))
        return backup

    def test_validation_is_read_only_and_checks_export_and_outbox(self):
        with tempfile.TemporaryDirectory() as td:
            source = self._backup(Path(td))
            result = validate_backup(source)
            self.assertEqual(result.status, "validated")
            self.assertEqual(result.tasks, 1)
            self.assertFalse((source / ".nautical-state").exists())

    def test_restore_source_permission_failure_is_structured(self):
        import nautical_core.restore_service as service
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._backup(root)
            target = root / "restored"
            with patch.object(service.Path, "read_text", side_effect=PermissionError("read-only source")):
                result = restore_backup(source, target, apply=True)
            self.assertEqual(result.status, "rejected")
            self.assertTrue(any("manifest is unreadable" in error for error in result.errors))
            self.assertFalse(target.exists())

    def test_inspection_does_not_create_target_without_apply(self):
        with tempfile.TemporaryDirectory() as td:
            source = self._backup(Path(td))
            target = Path(td) / "restored"
            result = restore_backup(source, target)
            self.assertEqual(result.status, "validated")
            self.assertFalse(target.exists())

    def test_apply_restores_only_to_new_or_empty_target(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._backup(root)
            target = root / "restored"
            result = restore_backup(source, target, apply=True)
            self.assertEqual(result.status, "restored")
            self.assertEqual(json.loads((target / "taskwarrior-export.json").read_text(encoding="utf-8"))[0]["description"], "café")
            self.assertTrue((target / ".nautical-state" / ".nautical_lifecycle_outbox.db").is_file())

            empty = root / "empty"
            empty.mkdir()
            result = restore_backup(source, empty, apply=True)
            self.assertEqual(result.status, "restored")
            self.assertTrue((empty / "taskwarrior-export.json").is_file())

    def test_apply_refuses_nonempty_target(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._backup(root)
            target = root / "restored"
            target.mkdir()
            (target / "keep").write_text("x", encoding="utf-8")
            result = restore_backup(source, target, apply=True)
            self.assertEqual(result.status, "rejected")
            self.assertTrue((target / "keep").exists())

    def test_restore_destination_permission_failure_is_structured(self):
        import nautical_core.restore_service as service
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._backup(root)
            target = root / "restored"
            with patch.object(service.Path, "mkdir", side_effect=PermissionError("read-only destination")):
                result = restore_backup(source, target, apply=True)
            self.assertEqual(result.status, "rejected")
            self.assertTrue(any("publish" in error for error in result.errors))
            self.assertFalse(target.exists())

    def test_apply_restores_explicit_resources(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._backup(root)
            resources = source / "resources"
            resources.mkdir()
            (resources / "calendar.json").write_text('{"name":"café"}\n', encoding="utf-8")
            publish_manifest(
                source / "manifest.json",
                create_manifest(
                    source,
                    files=("taskwarrior-export.json", "lifecycle-outbox.db", "resources/calendar.json"),
                ),
            )
            target = root / "restored"
            result = restore_backup(source, target, apply=True)
            self.assertEqual(result.status, "restored")
            self.assertIn("café", (target / "resources" / "calendar.json").read_text(encoding="utf-8"))

    def test_apply_restores_runtime_and_hooks_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._backup(root)
            (source / "runtime" / "releases" / "r-test").mkdir(parents=True)
            (source / "runtime" / "releases" / "r-test" / "manifest.json").write_text("{}\n", encoding="utf-8")
            (source / "hooks").mkdir()
            (source / "hooks" / "on-add.nautical").write_text("hook\n", encoding="utf-8")
            publish_manifest(
                source / "manifest.json",
                create_manifest(source, files=(
                    "taskwarrior-export.json", "lifecycle-outbox.db",
                    "runtime/releases/r-test/manifest.json", "hooks/on-add.nautical",
                )),
            )
            target = root / "restored"
            result = restore_backup(source, target, apply=True)
            self.assertEqual(result.status, "restored")
            self.assertTrue((target / ".nautical-runtime" / "releases" / "r-test" / "manifest.json").is_file())
            self.assertEqual((target / ".nautical-runtime" / "current").resolve().name, "r-test")
            self.assertEqual((target / "hooks" / "on-add.nautical").read_text(encoding="utf-8"), "hook\n")

    def test_restore_preserves_partial_intent_for_stage_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            live = root / "live"
            repository = LifecycleOutboxRepository(live)
            plan = LifecyclePlan.from_draft(
                identity=LifecycleIdentity("restore-chain", "00000000-0000-4000-8000-000000000801", 1, 2, LifecycleEvent.COMPLETE),
                action=LifecycleAction.SPAWN_CHILD,
                parent_guard=ParentGuard("completed", "on", "restore-chain", 1, "rf-restore", "20260101T000000Z"),
                draft=_task_draft({
                    "uuid": "00000000-0000-4000-8000-000000000802", "description": "restored child",
                    "status": "pending", "chain": "on", "chainID": "restore-chain", "link": 2,
                    "prevLink": "00000000", "cp": "1d", "due": "20260102T000000Z",
                }),
                parent_patch={"nextLink": "00000000"},
                expected_postconditions=("child_present", "parent_linked", "verified"),
            )
            repository.enqueue(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            intent_id = plan.identity.idempotency_key
            repository.claim_intent(owner="before-backup", lease_seconds=30, intent_id=intent_id)
            repository.advance_stage(intent_id=intent_id, owner="before-backup", stage=ExecutionStage.CHILD_PRESENT)
            backup = root / "backup"
            backup.mkdir()
            (backup / "taskwarrior-export.json").write_text("[]\n", encoding="utf-8")
            (backup / "lifecycle-outbox.db").write_bytes(repository.path.read_bytes())
            publish_manifest(backup / "manifest.json", create_manifest(backup, files=("taskwarrior-export.json", "lifecycle-outbox.db")))
            target = root / "restored"
            result = restore_backup(backup, target, apply=True)
            self.assertEqual(result.status, "restored")
            restored = LifecycleOutboxRepository(target)
            status = restored.status(limit=5)[1]
            self.assertEqual(status["records"][0]["stage"], "child_present")
            resumed = restored.advance_stage(intent_id=intent_id, owner="before-backup", stage=ExecutionStage.PARENT_LINKED)
            self.assertEqual(resumed.kind.value, "applied")

    def test_corrupt_export_is_rejected_before_apply(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._backup(root)
            (source / "taskwarrior-export.json").write_text("not-json", encoding="utf-8")
            target = root / "restored"
            result = restore_backup(source, target, apply=True)
            self.assertEqual(result.status, "rejected")
            self.assertFalse(target.exists())

    def test_corrupt_outbox_is_rejected_before_target_publication(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._backup(root)
            (source / "lifecycle-outbox.db").write_bytes(b"corrupt sqlite")
            publish_manifest(
                source / "manifest.json",
                create_manifest(source, files=("taskwarrior-export.json", "lifecycle-outbox.db")),
            )
            target = root / "restored"
            result = restore_backup(source, target, apply=True)
            self.assertEqual(result.status, "rejected")
            self.assertTrue(any("quick_check failed" in error for error in result.errors))
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob(".restored.restore-*")), [])

    def test_missing_outbox_is_unavailable_before_target_publication(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._backup(root)
            (source / "lifecycle-outbox.db").unlink()
            publish_manifest(
                source / "manifest.json",
                create_manifest(source, files=("taskwarrior-export.json",)),
            )
            target = root / "restored"
            result = restore_backup(source, target, apply=True)
            self.assertEqual(result.status, "rejected")
            self.assertTrue(any("outbox" in error and "missing" in error for error in result.errors))
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob(".restored.restore-*")), [])

    def test_outbox_quick_check_failure_is_rejected_before_target_publication(self):
        import nautical_core.restore_service as service

        class FailingConnection:
            def execute(self, query: str) -> object:
                raise sqlite3.DatabaseError("simulated quick_check failure")

            def close(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._backup(root)
            target = root / "restored"
            with patch.object(service.sqlite3, "connect", return_value=FailingConnection()):
                result = restore_backup(source, target, apply=True)
            self.assertEqual(result.status, "rejected")
            self.assertTrue(any("quick_check failed" in error for error in result.errors))
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob(".restored.restore-*")), [])

    def test_restore_publication_interruption_preserves_empty_target(self):
        import nautical_core.restore_service as service
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._backup(root)
            target = root / "restored"
            target.mkdir()
            with patch.object(service.os, "replace", side_effect=OSError("simulated interruption")):
                result = restore_backup(source, target, apply=True)
            self.assertEqual(result.status, "rejected")
            self.assertTrue(target.is_dir())
            self.assertEqual(list(target.iterdir()), [])
            self.assertEqual(list(root.glob(".restored.restore-*")), [])

    def test_restore_accepts_injected_publication_operation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._backup(root)
            target = root / "restored"
            calls = []

            def replace(source_path, destination):
                calls.append("replace")
                service.os.replace(source_path, destination)

            import nautical_core.restore_service as service
            result = restore_backup(source, target, apply=True, storage=StorageIO(replace=replace))
            self.assertEqual(result.status, "restored")
            self.assertEqual(calls, ["replace"])

    def test_restore_resource_copy_failure_cleans_staging_and_target(self):
        import nautical_core.restore_service as service
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._backup(root)
            resources = source / "resources"
            resources.mkdir()
            (resources / "calendar.json").write_text('{"name":"café"}\n', encoding="utf-8")
            from nautical_core.backup_service import create_manifest, publish_manifest
            publish_manifest(
                source / "manifest.json",
                create_manifest(
                    source,
                    files=("taskwarrior-export.json", "lifecycle-outbox.db", "resources/calendar.json"),
                ),
            )
            target = root / "restored"
            with patch.object(service.shutil, "copytree", side_effect=service.shutil.Error("simulated resource failure")):
                result = restore_backup(source, target, apply=True)
            self.assertEqual(result.status, "rejected")
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob(".restored.restore-*")), [])


if __name__ == "__main__":
    unittest.main()
