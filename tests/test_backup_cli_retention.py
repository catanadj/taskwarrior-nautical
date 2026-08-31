import json
import sqlite3
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class BackupCliRetentionTests(unittest.TestCase):
    def test_prune_is_explicit_and_returns_retention_details(self):
        script = Path(__file__).parents[1] / "nautical_core" / "tools" / "nautical_backup.py"
        with tempfile.TemporaryDirectory(prefix="nautical-backup-retention-") as td:
            root = Path(td)
            taskdata = root / "taskdata"
            (taskdata / ".nautical-state").mkdir(parents=True)
            connection = sqlite3.connect(taskdata / ".nautical-state" / ".nautical_lifecycle_outbox.db")
            connection.execute("CREATE TABLE marker (value TEXT)")
            connection.commit()
            connection.close()
            fake_task = root / "task"
            fake_task.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "assert sys.argv[1:] == ['rc.hooks=off', 'rc.verbose=nothing', 'export']\n"
                "print('[]')\n", encoding="utf-8"
            )
            fake_task.chmod(fake_task.stat().st_mode | stat.S_IXUSR)
            for name in ("one", "two", "three"):
                result = subprocess.run(
                    [sys.executable, str(script), "--taskdata", str(taskdata), "--destination", str(root / name), "--task-bin", str(fake_task), "--json"],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run(
                [sys.executable, str(script), "--taskdata", str(taskdata), "--destination", str(root / "four"), "--task-bin", str(fake_task), "--prune", "--keep", "2", "--json"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["retention"]["status"], "pruned")
            self.assertEqual(len(payload["retention"]["kept"]), 2)
            self.assertEqual(len(payload["retention"]["removed"]), 2)
            self.assertFalse((root / "one").exists())

    def test_backup_without_prune_reports_no_deletion(self):
        script = Path(__file__).parents[1] / "nautical_core" / "tools" / "nautical_backup.py"
        with tempfile.TemporaryDirectory(prefix="nautical-backup-retention-") as td:
            root = Path(td)
            taskdata = root / "taskdata"
            (taskdata / ".nautical-state").mkdir(parents=True)
            connection = sqlite3.connect(taskdata / ".nautical-state" / ".nautical_lifecycle_outbox.db")
            connection.execute("CREATE TABLE marker (value TEXT)")
            connection.commit()
            connection.close()
            fake_task = root / "task"
            fake_task.write_text("#!/usr/bin/env python3\nprint('[]')\n", encoding="utf-8")
            fake_task.chmod(fake_task.stat().st_mode | stat.S_IXUSR)
            target = root / "generation"
            result = subprocess.run([sys.executable, str(script), "--taskdata", str(taskdata), "--destination", str(target), "--task-bin", str(fake_task), "--json"], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)["retention"]["status"], "not_requested")


if __name__ == "__main__":
    unittest.main()
