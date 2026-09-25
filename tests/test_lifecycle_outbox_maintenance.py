from __future__ import annotations

from contextlib import contextmanager
import sqlite3
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from nautical_core.lifecycle_outbox_maintenance import (
    housekeeping_rows,
    prune_acknowledged_rows,
)


class LifecycleOutboxMaintenanceTests(unittest.TestCase):
    def _connection(self) -> tuple[TemporaryDirectory[str], sqlite3.Connection]:
        temporary = TemporaryDirectory()
        connection = sqlite3.connect(Path(temporary.name) / "outbox.db", isolation_level=None)
        connection.execute(
            "CREATE TABLE lifecycle_outbox (intent_id TEXT PRIMARY KEY, processing_state TEXT NOT NULL, "
            "acknowledged_at REAL NOT NULL DEFAULT 0)"
        )
        connection.execute(
            "CREATE TABLE lifecycle_maintenance (key TEXT PRIMARY KEY, value REAL NOT NULL)"
        )
        return temporary, connection

    @staticmethod
    @contextmanager
    def _transaction(connection: sqlite3.Connection):
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()

    @staticmethod
    def _insert(connection: sqlite3.Connection, intent_id: str, state: str, acknowledged_at: float) -> None:
        connection.execute(
            "INSERT INTO lifecycle_outbox(intent_id, processing_state, acknowledged_at) VALUES (?, ?, ?)",
            (intent_id, state, acknowledged_at),
        )

    def test_prune_is_inclusive_deterministic_bounded_and_acknowledged_only(self) -> None:
        temporary, connection = self._connection()
        self.addCleanup(connection.close)
        self.addCleanup(temporary.cleanup)
        self._insert(connection, "b", "acknowledged", 990.0)
        self._insert(connection, "c", "acknowledged", 990.0)
        self._insert(connection, "a", "acknowledged", 989.0)
        self._insert(connection, "newer", "acknowledged", 991.0)
        self._insert(connection, "review", "manual_review", 1.0)

        removed = prune_acknowledged_rows(
            connection, cutoff=990.0, limit=1, transaction=self._transaction
        )

        self.assertEqual(removed, 1)
        removed_at_cutoff = prune_acknowledged_rows(
            connection, cutoff=990.0, limit=1, transaction=self._transaction
        )
        self.assertEqual(removed_at_cutoff, 1)
        self.assertEqual(
            [row[0] for row in connection.execute("SELECT intent_id FROM lifecycle_outbox ORDER BY intent_id")],
            ["c", "newer", "review"],
        )

    def test_housekeeping_no_work_and_cooldown_are_explicit_skips(self) -> None:
        temporary, connection = self._connection()
        self.addCleanup(connection.close)
        self.addCleanup(temporary.cleanup)

        no_work = housekeeping_rows(
            connection, now=1000.0, cutoff=990.0, interval_seconds=100.0,
            size_threshold_bytes=1024, limit=10, checkpoint=True,
            database_size=lambda: 1, transaction=self._transaction,
        )
        self.assertTrue(no_work.skipped)
        self.assertEqual(no_work.reason, "no_work")
        self._insert(connection, "ack", "acknowledged", 1.0)
        with connection:
            connection.execute(
                "INSERT INTO lifecycle_maintenance(key, value) VALUES('housekeeping_last_attempt', 999.0)"
            )

        cooldown = housekeeping_rows(
            connection, now=1000.0, cutoff=990.0, interval_seconds=100.0,
            size_threshold_bytes=0, limit=10, checkpoint=True,
            database_size=lambda: 1, transaction=self._transaction,
        )
        self.assertTrue(cooldown.skipped)
        self.assertEqual(cooldown.reason, "cooldown")
        self.assertEqual(cooldown.removed, 0)

    def test_housekeeping_size_trigger_is_bounded_and_checkpoints_after_removal(self) -> None:
        temporary, connection = self._connection()
        self.addCleanup(connection.close)
        self.addCleanup(temporary.cleanup)
        self._insert(connection, "b", "acknowledged", 1.0)
        self._insert(connection, "a", "acknowledged", 1.0)
        self._insert(connection, "review", "manual_review", 1.0)

        result = housekeeping_rows(
            connection, now=1000.0, cutoff=990.0, interval_seconds=100.0,
            size_threshold_bytes=1, limit=1, checkpoint=True,
            database_size=lambda: 1, transaction=self._transaction,
        )

        self.assertEqual(result.removed, 1)
        self.assertFalse(result.skipped)
        self.assertEqual(result.checkpoint, "completed")
        self.assertEqual(
            [row[0] for row in connection.execute("SELECT intent_id FROM lifecycle_outbox ORDER BY intent_id")],
            ["b", "review"],
        )


if __name__ == "__main__":
    unittest.main()
