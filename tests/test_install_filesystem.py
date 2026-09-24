import os
import stat
import tempfile
import unittest
from pathlib import Path

from nautical_core.install_filesystem import (
    InstallError,
    InstallLock,
    atomic_copy,
    atomic_symlink,
    atomic_write_text,
    pointer_snapshot,
    restore_file,
    restore_pointer,
    snapshot_file,
)


class InstallFilesystemTests(unittest.TestCase):
    def test_install_lock_rejects_overlapping_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "runtime" / "install.lock"
            with InstallLock(lock_path):
                with self.assertRaisesRegex(InstallError, "already running"):
                    with InstallLock(lock_path):
                        pass

    def test_atomic_helpers_replace_targets_and_leave_no_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "nested" / "payload.txt"
            atomic_write_text("first", target)
            self.assertEqual(target.read_text(encoding="utf-8"), "first")

            source = root / "source.txt"
            source.write_text("second", encoding="utf-8")
            atomic_copy(source, target)
            self.assertEqual(target.read_text(encoding="utf-8"), "second")

            pointer = root / "current"
            atomic_symlink(str(target), pointer)
            self.assertTrue(pointer.is_symlink())
            self.assertEqual(pointer.resolve(), target.resolve())
            self.assertEqual(list(target.parent.glob(".*.tmp-*")), [])

    def test_snapshot_file_creates_private_backup_and_restores_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "managed.conf"
            source.write_text("secret", encoding="utf-8")
            source.chmod(0o644)
            backup_dir = root / "backups"

            snapshot = snapshot_file(source, backup_dir)

            self.assertEqual(snapshot["kind"], "file")
            self.assertEqual(stat.S_IMODE(backup_dir.stat().st_mode), 0o700)
            backup = Path(str(snapshot["backup"]))
            self.assertEqual(stat.S_IMODE(backup.stat().st_mode), 0o600)
            source.write_text("changed", encoding="utf-8")
            restore_file(source, snapshot)
            self.assertEqual(source.read_text(encoding="utf-8"), "secret")

    def test_snapshot_and_restore_cover_missing_and_symlink_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "managed"
            backup_dir = root / "backups"

            missing = snapshot_file(target, backup_dir)
            self.assertEqual(missing, {"kind": "missing"})
            target.write_text("value", encoding="utf-8")
            restore_file(target, missing)
            self.assertFalse(target.exists())

            target.write_text("value", encoding="utf-8")
            pointer = root / "pointer"
            pointer.symlink_to(target)
            symlink = pointer_snapshot(pointer)
            self.assertEqual(symlink["kind"], "symlink")
            pointer.unlink()
            restore_pointer(pointer, symlink)
            self.assertTrue(pointer.is_symlink())
            self.assertEqual(pointer.resolve(), target.resolve())

    def test_snapshots_reject_unmanaged_path_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            managed_dir = root / "managed-dir"
            managed_dir.mkdir()
            with self.assertRaisesRegex(InstallError, "not a file or symlink"):
                snapshot_file(managed_dir, root / "backups")
            with self.assertRaisesRegex(InstallError, "not a symlink"):
                pointer_snapshot(managed_dir)


if __name__ == "__main__":
    unittest.main()
