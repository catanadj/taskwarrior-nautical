import json
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nautical_core import backup_service
from nautical_core.tools import nautical_backup


class BackupCliTests(unittest.TestCase):
    def test_publication_interruption_removes_staging_and_preserves_previous(self):
        with tempfile.TemporaryDirectory(prefix="nautical-backup-cli-") as td:
            root = Path(td)
            taskdata = root / "taskdata"
            (taskdata / ".nautical-state").mkdir(parents=True)
            connection = sqlite3.connect(taskdata / ".nautical-state" / ".nautical_lifecycle_outbox.db")
            connection.execute("CREATE TABLE marker (value TEXT)")
            connection.commit()
            connection.close()
            task = root / "task"
            task.write_text("#!/usr/bin/env python3\nprint('[]')\n", encoding="utf-8")
            task.chmod(task.stat().st_mode | stat.S_IXUSR)
            destination = root / "backup"
            previous = root / "previous"
            previous.mkdir()
            with patch.object(nautical_backup.os, "replace", side_effect=OSError("simulated interruption")):
                with self.assertRaises(backup_service.BackupExportError):
                    nautical_backup.create_backup(taskdata, destination, task_bin=str(task), timeout=5.0)
            self.assertFalse(destination.exists())
            self.assertTrue(previous.is_dir())
            self.assertEqual(list(root.glob(".backup.*")), [])

    def test_named_options_capture_export_and_outbox(self):
        script = Path(__file__).parents[1] / "nautical_core" / "tools" / "nautical_backup.py"
        with tempfile.TemporaryDirectory(prefix="nautical-backup-cli-") as td:
            root = Path(td)
            taskdata = root / "taskdata"
            (taskdata / ".nautical-state").mkdir(parents=True)
            connection = sqlite3.connect(taskdata / ".nautical-state" / ".nautical_lifecycle_outbox.db")
            connection.execute("CREATE TABLE marker (value TEXT)")
            connection.execute("INSERT INTO marker VALUES ('ok')")
            connection.commit()
            connection.close()
            fake_task = root / "task"
            fake_task.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "assert sys.argv[1:] == ['rc.hooks=off', 'rc.verbose=nothing', 'export']\n"
                "print('[{\"uuid\":\"abc\",\"description\":\"café\"}]')\n",
                encoding="utf-8",
            )
            fake_task.chmod(fake_task.stat().st_mode | stat.S_IXUSR)
            destination = root / "backup"
            result = subprocess.run(
                [sys.executable, str(script), "--taskdata", str(taskdata), "--destination", str(destination), "--task-bin", str(fake_task), "--json"],
                capture_output=True, text=True, check=False, env={**os.environ, "PYTHONPATH": str(root)},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "created")
            self.assertEqual(json.loads((destination / "taskwarrior-export.json").read_text(encoding="utf-8"))[0]["description"], "café")
            self.assertTrue((destination / "manifest.json").is_file())

    def test_named_provenance_is_recorded_without_implicit_values(self):
        script = Path(__file__).parents[1] / "nautical_core" / "tools" / "nautical_backup.py"
        with tempfile.TemporaryDirectory(prefix="nautical-backup-cli-") as td:
            root = Path(td)
            taskdata = root / "taskdata"
            (taskdata / ".nautical-state").mkdir(parents=True)
            connection = sqlite3.connect(taskdata / ".nautical-state" / ".nautical_lifecycle_outbox.db")
            connection.execute("CREATE TABLE marker (value TEXT)")
            connection.commit()
            connection.close()
            task = root / "task"
            task.write_text("#!/usr/bin/env python3\nprint('[]')\n", encoding="utf-8")
            task.chmod(task.stat().st_mode | stat.S_IXUSR)
            destination = root / "backup"
            result = subprocess.run([
                sys.executable, str(script), "--taskdata", str(taskdata), "--destination", str(destination), "--task-bin", str(task),
                "--active-release", "r-test", "--runtime-digest", "a" * 64, "--timezone", "Europe/Bucharest", "--json",
            ], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metadata = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))["metadata"]
            self.assertEqual(metadata["active_release"], "r-test")
            self.assertEqual(metadata["runtime_digest"], "a" * 64)
            self.assertEqual(metadata["timezone"], "Europe/Bucharest")
            self.assertNotIn("python_version", metadata)

    def test_include_copies_unicode_resource_and_checksums_it(self):
        script = Path(__file__).parents[1] / "nautical_core" / "tools" / "nautical_backup.py"
        with tempfile.TemporaryDirectory(prefix="nautical-backup-cli-") as td:
            root = Path(td)
            taskdata = root / "taskdata"
            (taskdata / ".nautical-state").mkdir(parents=True)
            connection = sqlite3.connect(taskdata / ".nautical-state" / ".nautical_lifecycle_outbox.db")
            connection.execute("CREATE TABLE marker (value TEXT)")
            connection.commit()
            connection.close()
            resource = root / "calendar.json"
            resource.write_text('{"description":"café"}\n', encoding="utf-8")
            task = root / "task"
            task.write_text("#!/usr/bin/env python3\nprint('[]')\n", encoding="utf-8")
            task.chmod(task.stat().st_mode | stat.S_IXUSR)
            destination = root / "backup"
            result = subprocess.run([sys.executable, str(script), "--taskdata", str(taskdata), "--destination", str(destination), "--task-bin", str(task), "--include", "calendar=" + str(resource), "--json"], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((destination / "resources" / "calendar").read_text(encoding="utf-8"), '{"description":"café"}\n')

    def test_include_refuses_duplicate_and_taskdata_resource(self):
        script = Path(__file__).parents[1] / "nautical_core" / "tools" / "nautical_backup.py"
        with tempfile.TemporaryDirectory(prefix="nautical-backup-cli-") as td:
            root = Path(td)
            taskdata = root / "taskdata"
            (taskdata / ".nautical-state").mkdir(parents=True)
            connection = sqlite3.connect(taskdata / ".nautical-state" / ".nautical_lifecycle_outbox.db")
            connection.execute("CREATE TABLE marker (value TEXT)")
            connection.commit()
            connection.close()
            task = root / "task"
            task.write_text("#!/usr/bin/env python3\nprint('[]')\n", encoding="utf-8")
            task.chmod(task.stat().st_mode | stat.S_IXUSR)
            destination = root / "backup"
            result = subprocess.run([sys.executable, str(script), "--taskdata", str(taskdata), "--destination", str(destination), "--task-bin", str(task), "--include", "x=" + str(taskdata / ".nautical-state" / ".nautical_lifecycle_outbox.db"), "--json"], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn("outside Taskdata", json.loads(result.stdout)["error"])

    def test_missing_destination_is_structured_error(self):
        script = Path(__file__).parents[1] / "nautical_core" / "tools" / "nautical_backup.py"
        result = subprocess.run([sys.executable, str(script), "--taskdata", "/tmp/no-taskdata"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["error"], "--destination is required")
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
