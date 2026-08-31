import json
import os
from pathlib import Path
import stat
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from nautical_core.backup_service import (
    BackupManifestError,
    BackupExportError,
    backup_outbox_database,
    capture_taskwarrior_export,
    create_manifest,
    inventory,
    manifest_json,
    publish_manifest,
    validate_manifest,
    verify_manifest,
    prune_backup_generations,
    build_backup_metadata,
)


class BackupServiceTests(unittest.TestCase):
    def test_backup_metadata_is_explicit_and_deterministic(self):
        values = dict(
            active_release="r-test",
            runtime_digest="a" * 64,
            taskwarrior_version="3.5.0",
            python_version="3.11.2",
            timezone="Europe/Bucharest",
            timezone_data_identity="tzdata-2026a",
        )
        expected = {
            "metadata_schema": 1,
            **values,
        }
        self.assertEqual(build_backup_metadata(**values), expected)
        self.assertEqual(build_backup_metadata(**values), build_backup_metadata(**values))

    def test_backup_metadata_omits_unsupplied_values(self):
        self.assertEqual(build_backup_metadata(timezone=" UTC ", python_version=""), {"metadata_schema": 1, "timezone": "UTC"})
    def _generation(self, root: Path, name: str, content: str = "ok") -> Path:
        generation = root / name
        generation.mkdir()
        (generation / "payload").write_text(content, encoding="utf-8")
        publish_manifest(generation / "manifest.json", create_manifest(generation, files=("payload",)))
        return generation

    def test_prune_keeps_newest_verified_generations(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            older = self._generation(root, "old")
            middle = self._generation(root, "middle")
            newest = self._generation(root, "newest")
            for index, path in enumerate((older, middle, newest), start=1):
                os.utime(path, (index, index))
            result = prune_backup_generations(root)
            self.assertEqual(result.kept, ("newest", "middle"))
            self.assertEqual(result.removed, ("old",))
            self.assertFalse(older.exists())

    def test_prune_leaves_invalid_and_unverified_generations(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            valid = self._generation(root, "valid")
            invalid = root / "invalid"
            invalid.mkdir()
            (invalid / "manifest.json").write_text("not-json", encoding="utf-8")
            unverified = self._generation(root, "unverified")
            (unverified / "payload").write_text("changed", encoding="utf-8")
            result = prune_backup_generations(root, keep=1)
            self.assertEqual(result.kept, ("valid",))
            self.assertEqual(result.removed, ())
            self.assertTrue(invalid.exists())
            self.assertTrue(unverified.exists())

    def test_prune_honors_keep_override(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("one", "two", "three"):
                self._generation(root, name)
            result = prune_backup_generations(root, keep=1)
            self.assertEqual(len(result.kept), 1)
            self.assertEqual(len(result.removed), 2)

    def test_prune_refuses_zero_or_boolean_retention(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(BackupManifestError):
                prune_backup_generations(root, keep=0)
            with self.assertRaises(BackupManifestError):
                prune_backup_generations(root, keep=True)
    def test_outbox_online_backup_copies_wal_and_checks_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            taskdata = root / "taskdata"
            state = taskdata / ".nautical-state"
            state.mkdir(parents=True)
            source = state / ".nautical_lifecycle_outbox.db"
            connection = sqlite3.connect(source)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE values_table (value TEXT NOT NULL)")
            connection.execute("INSERT INTO values_table VALUES ('pending')")
            connection.commit()
            # Leave the connection open so the row remains a WAL-backed source.
            destination = root / "backup" / "outbox.db"
            result = backup_outbox_database(taskdata, destination)
            connection.close()
            self.assertEqual(result.status, "captured")
            self.assertEqual(result.quick_check, "ok")
            copied = sqlite3.connect(destination)
            self.assertEqual(copied.execute("SELECT value FROM values_table").fetchone()[0], "pending")
            copied.close()

    def test_outbox_backup_refuses_live_or_existing_destinations(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            taskdata = root / "taskdata"
            state = taskdata / ".nautical-state"
            state.mkdir(parents=True)
            source = state / ".nautical_lifecycle_outbox.db"
            connection = sqlite3.connect(source)
            connection.execute("CREATE TABLE values_table (value TEXT)")
            connection.commit()
            with self.assertRaises(BackupExportError):
                backup_outbox_database(taskdata, taskdata / "outbox.db")
            target = root / "existing.db"
            target.write_bytes(b"existing")
            with self.assertRaises(BackupExportError):
                backup_outbox_database(taskdata, target)
            connection.close()

    def _fake_task(self, root: Path, output: str) -> Path:
        script = root / "task-fake.py"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "assert sys.argv[1:] == ['rc.hooks=off', 'rc.verbose=nothing', 'export']\n"
            f"print({output!r})\n",
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        return script

    def test_capture_export_uses_hooks_off_and_publishes_atomically(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            taskdata = root / "taskdata"
            taskdata.mkdir()
            destination = root / "backup" / "tasks.json"
            task = self._fake_task(root, '[{"uuid":"abc","description":"café"}]')
            captured = capture_taskwarrior_export(taskdata, destination, task_bin=str(task))
            self.assertEqual(captured.status, "captured")
            self.assertEqual(captured.tasks, 1)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8"))[0]["description"], "café")
            self.assertEqual(list(destination.parent.glob(".*")), [])

    def test_capture_export_refuses_live_or_existing_destinations(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            taskdata = root / "taskdata"
            taskdata.mkdir()
            task = self._fake_task(root, "[]")
            with self.assertRaises(BackupExportError):
                capture_taskwarrior_export(taskdata, taskdata / "export.json", task_bin=str(task))
            destination = root / "export.json"
            destination.write_text("old", encoding="utf-8")
            with self.assertRaises(BackupExportError):
                capture_taskwarrior_export(taskdata, destination, task_bin=str(task))

    def test_capture_export_rejects_invalid_json_without_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            taskdata = root / "taskdata"
            taskdata.mkdir()
            destination = root / "export.json"
            task = self._fake_task(root, "not-json")
            with self.assertRaises(BackupExportError):
                capture_taskwarrior_export(taskdata, destination, task_bin=str(task))
            self.assertFalse(destination.exists())

    def test_manifest_inventory_is_stable_and_unicode_safe(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "nested").mkdir()
            (root / "nested" / "task-data.json").write_text('{"description":"café"}\n', encoding="utf-8")
            manifest = create_manifest(root, metadata={"timezone": "Europe/Bucharest", "label": "café"})
            self.assertEqual([item["path"] for item in manifest["files"]], ["nested/task-data.json"])
            self.assertEqual(verify_manifest(root, manifest).status, "verified")
            self.assertIn("café", manifest_json(manifest))

    def test_publish_is_atomic_and_verifies(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "one").write_text("one", encoding="utf-8")
            manifest = create_manifest(root)
            target = root / "backup" / "manifest.json"
            publish_manifest(target, manifest)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["version"], 1)
            self.assertEqual(verify_manifest(root, manifest).status, "verified")

    def test_publish_interruption_removes_temporary_and_preserves_previous(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "payload").write_text("new", encoding="utf-8")
            manifest = create_manifest(root, files=("payload",))
            target = root / "manifest.json"
            target.write_text('{"status":"previous"}\n', encoding="utf-8")
            with patch("nautical_core.backup_service.os.replace", side_effect=OSError("simulated interruption")), self.assertRaises(BackupManifestError):
                publish_manifest(target, manifest)
            self.assertEqual(target.read_text(encoding="utf-8"), '{"status":"previous"}\n')
            self.assertEqual(list(root.glob(".manifest.json.*")), [])

    def test_outbox_failure_removes_partial_target(self):
        import nautical_core.backup_service as service
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            taskdata = root / "taskdata"
            source_dir = taskdata / ".nautical-state"
            source_dir.mkdir(parents=True)
            source = source_dir / ".nautical_lifecycle_outbox.db"
            source.write_bytes(b"source")
            target = root / "outbox.db"

            class FakeConnection:
                def __init__(self, is_target: bool) -> None:
                    self.is_target = is_target
                def backup(self, other: object) -> None:
                    target.write_bytes(b"partial")
                def commit(self) -> None:
                    return None
                def execute(self, query: str) -> object:
                    raise sqlite3.DatabaseError("simulated quick_check failure")
                def close(self) -> None:
                    return None

            calls = iter((FakeConnection(False), FakeConnection(True)))
            with patch.object(service.sqlite3, "connect", side_effect=lambda *args, **kwargs: next(calls)):
                with self.assertRaises(BackupExportError):
                    backup_outbox_database(taskdata, target)
            self.assertFalse(target.exists())

    def test_outbox_connection_failure_rejects_without_artifacts(self):
        import nautical_core.backup_service as service
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            taskdata = root / "taskdata"
            source_dir = taskdata / ".nautical-state"
            source_dir.mkdir(parents=True)
            (source_dir / ".nautical_lifecycle_outbox.db").write_bytes(b"source")
            target = root / "outbox.db"
            with patch.object(service.sqlite3, "connect", side_effect=sqlite3.OperationalError("simulated open failure")):
                with self.assertRaises(BackupExportError):
                    backup_outbox_database(taskdata, target)
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob(".outbox.db.*")), [])


    def test_changed_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "taskdata"
            path.write_text("before", encoding="utf-8")
            manifest = create_manifest(root)
            path.write_text("after", encoding="utf-8")
            result = verify_manifest(root, manifest)
            self.assertEqual(result.status, "rejected")
            self.assertIn("checksum mismatch", result.errors[0])

    def test_manifest_rejects_traversal_duplicate_and_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "file").write_text("x", encoding="utf-8")
            with self.assertRaises(BackupManifestError):
                inventory(root, ["../file"])
            with self.assertRaises(BackupManifestError):
                inventory(root, ["file", "file"])
            outside = root.parent / "outside"
            outside.write_text("x", encoding="utf-8")
            try:
                (root / "escape").symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            with self.assertRaises(BackupManifestError):
                inventory(root)

    def test_validate_rejects_unsafe_manifest_without_touching_disk(self):
        with self.assertRaises(BackupManifestError):
            validate_manifest(
                {
                    "schema": "nautical.backup",
                    "version": 1,
                    "metadata": {},
                    "files": [{"path": "/etc/passwd", "bytes": 1, "sha256": "0" * 64}],
                }
            )

    def test_validate_rejects_coerced_artifact_types(self):
        base = {"schema": "nautical.backup", "version": 1, "metadata": {}}
        for artifact in (
            {"path": "file", "bytes": True, "sha256": "0" * 64},
            {"path": "file", "bytes": 1.5, "sha256": "0" * 64},
            {"path": "file", "bytes": 1, "sha256": 0},
        ):
            with self.subTest(artifact=artifact), self.assertRaises(BackupManifestError):
                validate_manifest({**base, "files": [artifact]})

    def test_validate_rejects_oversized_manifest(self):
        with self.assertRaises(BackupManifestError):
            validate_manifest(
                {
                    "schema": "nautical.backup",
                    "version": 1,
                    "metadata": {"padding": "x" * (16 * 1024 * 1024)},
                    "files": [],
                }
            )


if __name__ == "__main__":
    unittest.main()
