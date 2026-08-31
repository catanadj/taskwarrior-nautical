import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from nautical_core.backup_service import create_manifest, publish_manifest


class RestoreCliTests(unittest.TestCase):
    def _backup(self, root: Path) -> Path:
        backup = root / "backup"
        backup.mkdir()
        (backup / "taskwarrior-export.json").write_text("[]\n", encoding="utf-8")
        database = backup / "lifecycle-outbox.db"
        connection = sqlite3.connect(database)
        connection.execute("CREATE TABLE marker (value TEXT)")
        connection.commit()
        connection.close()
        publish_manifest(backup / "manifest.json", create_manifest(backup, files=("taskwarrior-export.json", "lifecycle-outbox.db")))
        return backup

    def test_default_is_inspection_only_and_emits_json(self):
        script = Path(__file__).parents[1] / "nautical_core" / "tools" / "nautical_restore.py"
        with tempfile.TemporaryDirectory(prefix="nautical-restore-cli-") as td:
            root = Path(td)
            source = self._backup(root)
            target = root / "target"
            result = subprocess.run([sys.executable, str(script), "--source", str(source), "--target", str(target)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "validated")
            self.assertFalse(target.exists())

    def test_apply_creates_target_only_with_explicit_flag(self):
        script = Path(__file__).parents[1] / "nautical_core" / "tools" / "nautical_restore.py"
        with tempfile.TemporaryDirectory(prefix="nautical-restore-cli-") as td:
            root = Path(td)
            source = self._backup(root)
            target = root / "target"
            result = subprocess.run([sys.executable, str(script), "--source", str(source), "--target", str(target), "--apply", "--json"], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "restored")
            self.assertTrue((target / "manifest.json").is_file())

    def test_missing_source_is_actionable_json_error(self):
        script = Path(__file__).parents[1] / "nautical_core" / "tools" / "nautical_restore.py"
        result = subprocess.run([sys.executable, str(script), "--source", "/tmp/no-such-nautical-backup"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "rejected")
        self.assertIn("backup source", json.loads(result.stdout)["errors"][0])


if __name__ == "__main__":
    unittest.main()
