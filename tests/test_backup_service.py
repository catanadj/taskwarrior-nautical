import json
from pathlib import Path
import tempfile
import unittest

from nautical_core.backup_service import (
    BackupManifestError,
    create_manifest,
    inventory,
    manifest_json,
    publish_manifest,
    validate_manifest,
    verify_manifest,
)


class BackupServiceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
