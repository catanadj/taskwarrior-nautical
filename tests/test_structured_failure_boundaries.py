from __future__ import annotations

import sqlite3
import stat
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from nautical_core.lifecycle_outbox import (
    LifecycleOutboxError,
    LifecycleOutboxRepository,
    OUTBOX_MAINTENANCE_FILESYSTEM_FAILURE,
    OutboxResultKind,
)
from nautical_core.panel_diagnostics import file_source_warnings


class StructuredFailureBoundaryTests(unittest.TestCase):
    def test_panel_config_warning_reports_missing_explicit_config(self) -> None:
        from nautical_core.panel_diagnostics import config_warnings

        with TemporaryDirectory() as directory:
            missing_config = Path(directory) / "missing-nautical.toml"
            with patch.dict(os.environ, {"NAUTICAL_CONFIG": str(missing_config)}):
                warnings = config_warnings()

        self.assertTrue(
            any("NAUTICAL_CONFIG points to a missing file" in item for item in warnings),
            warnings,
        )

    def test_panel_file_warnings_report_empty_sources_and_unmatched_patterns(self) -> None:
        from nautical_core import anchor_files, omit_files

        with TemporaryDirectory() as directory:
            Path(directory, "weekend.csv").write_text(
                "date,description\n2026-07-04,weekend\n", encoding="utf-8"
            )
            file_modules = {"anchor_files": anchor_files, "omit_files": omit_files}
            core = SimpleNamespace(
                ANCHOR_FILE_DIR=directory,
                OMIT_FILE_DIR=directory,
                _import_sibling=lambda name: file_modules[name],
            )
            warnings = file_source_warnings(
                core,
                {"anchor_file": "weekend.csv@bd", "omit_file": "weekend.csv@bd"},
            )
            pattern_warnings = file_source_warnings(
                core,
                {
                    "anchor_file": "missing-*.csv | weekend.csv",
                    "omit_file": "missing-?.txt",
                },
            )

        self.assertIn("anchor_file 'weekend.csv' has no usable dates.", warnings)
        self.assertIn("omit_file 'weekend.csv' has no usable dates.", warnings)
        self.assertIn("anchor_file pattern 'missing-*.csv' matched no files.", pattern_warnings)
        self.assertIn("omit_file pattern 'missing-?.txt' matched no files.", pattern_warnings)

    def test_outbox_rejects_symlinked_state_directory_without_chmodding_target(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            taskdata = root / "taskdata"
            taskdata.mkdir()
            target = root / "external-state"
            target.mkdir(mode=0o755)
            target.chmod(0o755)
            (taskdata / ".nautical-state").symlink_to(target, target_is_directory=True)

            with self.assertRaises(LifecycleOutboxError):
                LifecycleOutboxRepository(taskdata)._connect()

            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o755)
            self.assertEqual(list(target.iterdir()), [])

    def test_outbox_rejects_symlinked_database_without_chmodding_target(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            taskdata = root / "taskdata"
            state = taskdata / ".nautical-state"
            state.mkdir(parents=True, mode=0o700)
            target = root / "external.db"
            with sqlite3.connect(target) as connection:
                connection.execute("CREATE TABLE sentinel (value TEXT)")
            target.chmod(0o644)
            (state / ".nautical_lifecycle_outbox.db").symlink_to(target)

            with self.assertRaises(LifecycleOutboxError):
                LifecycleOutboxRepository(taskdata)._connect()

            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    def test_outbox_rejects_symlinked_sqlite_sidecar(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            taskdata = root / "taskdata"
            state = taskdata / ".nautical-state"
            state.mkdir(parents=True, mode=0o700)
            target = root / "external-wal"
            target.write_text("sentinel", encoding="utf-8")
            target.chmod(0o644)
            (state / ".nautical_lifecycle_outbox.db-wal").symlink_to(target)

            with self.assertRaises(LifecycleOutboxError):
                LifecycleOutboxRepository(taskdata)._connect()

            self.assertEqual(target.read_text(encoding="utf-8"), "sentinel")
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    def test_outbox_closes_connection_when_setup_fails(self) -> None:
        with TemporaryDirectory() as td:
            repository = LifecycleOutboxRepository(Path(td))
            opened: list[sqlite3.Connection] = []
            real_connect = sqlite3.connect

            class FailingConnection(sqlite3.Connection):
                def execute(self, sql, parameters=(), /):
                    if sql == "PRAGMA synchronous=FULL":
                        raise LifecycleOutboxError("denied")
                    return super().execute(sql, parameters)

            def tracked_connect(*args, **kwargs):
                connection = real_connect(*args, factory=FailingConnection, **kwargs)
                opened.append(connection)
                return connection

            with patch("nautical_core.lifecycle_outbox.sqlite3.connect", side_effect=tracked_connect):
                with self.assertRaises(LifecycleOutboxError):
                    repository._connect()

            self.assertEqual(len(opened), 1)
            with self.assertRaises(sqlite3.ProgrammingError):
                opened[0].execute("SELECT 1")

    def test_prune_classifies_filesystem_security_failure_and_closes_connection(self) -> None:
        with TemporaryDirectory() as td:
            repository = LifecycleOutboxRepository(Path(td))
            connection = Mock()
            with patch.object(repository, "_connect", return_value=connection), \
                    patch.object(repository, "_initialize"), \
                    patch.object(repository, "_secure_state_files", side_effect=PermissionError("denied")):
                result = repository.prune_acknowledged()

            self.assertEqual(result.kind, OutboxResultKind.REJECTED)
            self.assertTrue(result.reason.startswith(f"{OUTBOX_MAINTENANCE_FILESYSTEM_FAILURE}:"))
            connection.close.assert_called_once_with()

    def test_housekeeping_classifies_filesystem_security_failure_and_closes_connection(self) -> None:
        with TemporaryDirectory() as td:
            repository = LifecycleOutboxRepository(Path(td))
            repository.path.parent.mkdir(parents=True)
            repository.path.touch()
            connection = Mock()
            with patch.object(repository, "_connect", return_value=connection), \
                    patch.object(repository, "_initialize"), \
                    patch.object(repository, "_secure_state_files", side_effect=PermissionError("denied")):
                result = repository.opportunistic_housekeeping()

            self.assertEqual(result.kind, OutboxResultKind.REJECTED)
            self.assertTrue(result.reason.startswith(f"{OUTBOX_MAINTENANCE_FILESYSTEM_FAILURE}:"))
            connection.close.assert_called_once_with()

    def test_panel_source_missing_file_is_optional(self) -> None:
        loader = SimpleNamespace(
            load_anchor_file_dates=Mock(side_effect=FileNotFoundError("gone")),
            unmatched_anchor_file_patterns=Mock(return_value=()),
        )
        core = SimpleNamespace(_import_sibling=Mock(return_value=loader), ANCHOR_FILE_DIR="")

        self.assertEqual(file_source_warnings(core, {"anchor_file": "dates.txt"}), [])

    def test_panel_source_loader_permission_error_surfaces(self) -> None:
        loader = SimpleNamespace(
            load_anchor_file_dates=Mock(side_effect=PermissionError("denied")),
            unmatched_anchor_file_patterns=Mock(return_value=()),
        )
        core = SimpleNamespace(_import_sibling=Mock(return_value=loader), ANCHOR_FILE_DIR="")

        with self.assertRaises(PermissionError):
            file_source_warnings(core, {"anchor_file": "dates.txt"})

    def test_panel_source_loader_parse_error_surfaces(self) -> None:
        loader = SimpleNamespace(
            load_omit_file_dates=Mock(side_effect=ValueError("invalid date")),
            unmatched_omit_file_patterns=Mock(return_value=()),
        )
        core = SimpleNamespace(_import_sibling=Mock(return_value=loader), OMIT_FILE_DIR="")

        with self.assertRaises(ValueError):
            file_source_warnings(core, {"omit_file": "omit.txt"})


if __name__ == "__main__":
    unittest.main()
