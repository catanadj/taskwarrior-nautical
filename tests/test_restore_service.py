import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from nautical_core.backup_service import create_manifest, publish_manifest
from nautical_core.restore_service import restore_backup, validate_backup


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

    def test_corrupt_export_is_rejected_before_apply(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._backup(root)
            (source / "taskwarrior-export.json").write_text("not-json", encoding="utf-8")
            target = root / "restored"
            result = restore_backup(source, target, apply=True)
            self.assertEqual(result.status, "rejected")
            self.assertFalse(target.exists())

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


if __name__ == "__main__":
    unittest.main()
