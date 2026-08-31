import json
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


if __name__ == "__main__":
    unittest.main()
