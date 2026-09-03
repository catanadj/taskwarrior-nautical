import json
import os
import tarfile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class OfflineKitTests(unittest.TestCase):
    def test_build_and_verify_is_self_contained(self):
        script = Path(__file__).parents[1] / "dev_tools" / "nautical_offline_kit.py"
        with tempfile.TemporaryDirectory(prefix="nautical-kit-test-") as td:
            kit = Path(td) / "kit"
            built = subprocess.run([sys.executable, str(script), "build", str(kit)], capture_output=True, text=True, check=False)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            self.assertEqual(json.loads(built.stdout)["status"], "created")
            verified = subprocess.run([sys.executable, str(script), "verify", str(kit)], capture_output=True, text=True, check=False)
            self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
            self.assertEqual(json.loads(verified.stdout)["status"], "verified")
            manifest = json.loads((kit / "kit-manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["inventory"]["platform"])
            self.assertTrue(manifest["inventory"]["architecture"])
            paths = [item["path"] for item in manifest["files"]]
            self.assertFalse(any(".nautical-cache" in path or "__pycache__" in path or path.endswith(".pyc") for path in paths))

    def test_verify_rejects_changed_runtime(self):
        script = Path(__file__).parents[1] / "dev_tools" / "nautical_offline_kit.py"
        with tempfile.TemporaryDirectory(prefix="nautical-kit-test-") as td:
            kit = Path(td) / "kit"
            subprocess.run([sys.executable, str(script), "build", str(kit)], check=True, capture_output=True, text=True)
            target = kit / "nautical"
            target.write_bytes(target.read_bytes() + b"\n")
            verified = subprocess.run([sys.executable, str(script), "verify", str(kit)], capture_output=True, text=True, check=False)
            self.assertEqual(verified.returncode, 2)
            self.assertIn("checksum mismatch", verified.stdout)

    def test_build_archive_and_verify_archive(self):
        script = Path(__file__).parents[1] / "dev_tools" / "nautical_offline_kit.py"
        with tempfile.TemporaryDirectory(prefix="nautical-kit-test-") as td:
            kit = Path(td) / "kit"
            archive = Path(td) / "kit.tar.gz"
            built = subprocess.run(
                [sys.executable, str(script), "build", str(kit), "--archive", str(archive)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            self.assertTrue(archive.is_file())
            verified = subprocess.run(
                [sys.executable, str(script), "verify", str(archive)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
            self.assertEqual(json.loads(verified.stdout)["status"], "verified")

    def test_verified_local_kit_installs_without_network(self):
        script = Path(__file__).parents[1] / "dev_tools" / "nautical_offline_kit.py"
        install = Path(__file__).parents[1] / "nautical_core" / "tools" / "nautical_install.py"
        with tempfile.TemporaryDirectory(prefix="nautical-kit-test-") as td:
            root = Path(td)
            kit = root / "kit"
            target = root / "taskdata"
            launcher = root / "launcher"
            env = {**os.environ, "TASKRC": str(root / "taskrc")}
            built = subprocess.run(
                [sys.executable, str(script), "build", str(kit)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            verified = subprocess.run(
                [sys.executable, str(script), "verify", str(kit)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
            planned = subprocess.run(
                [sys.executable, str(install), "--source", str(kit), "--taskdata", str(target),
                 "--launcher-path", str(launcher), "--dry-run", "--json"],
                capture_output=True, text=True, check=False, env=env,
            )
            self.assertEqual(planned.returncode, 0, planned.stdout + planned.stderr)
            self.assertFalse(target.exists())
            applied = subprocess.run(
                [sys.executable, str(install), "--source", str(kit), "--taskdata", str(target),
                 "--launcher-path", str(launcher), "--json"],
                capture_output=True, text=True, check=False, env=env,
            )
            self.assertEqual(applied.returncode, 0, applied.stdout + applied.stderr)
            self.assertTrue((target / ".nautical-runtime" / "current").is_symlink())
            self.assertTrue(launcher.is_file())

    def test_verify_rejects_unsafe_archive_member(self):
        script = Path(__file__).parents[1] / "dev_tools" / "nautical_offline_kit.py"
        with tempfile.TemporaryDirectory(prefix="nautical-kit-test-") as td:
            archive = Path(td) / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                info = tarfile.TarInfo("../escape")
                info.size = 1
                handle.addfile(info, __import__("io").BytesIO(b"x"))
            verified = subprocess.run(
                [sys.executable, str(script), "verify", str(archive)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(verified.returncode, 2)
            self.assertIn("unsafe archive", verified.stdout)


if __name__ == "__main__":
    unittest.main()
