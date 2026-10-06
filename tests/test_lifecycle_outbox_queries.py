from __future__ import annotations

import sqlite3
import unittest
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from pathlib import Path

from nautical_core.lifecycle.outbox_queries import (
    OutboxStatusFailureSummary,
    StatusRowPoison,
    snapshot_rows,
    status_summary,
)


@dataclass(frozen=True)
class _Value:
    value: str


@dataclass(frozen=True)
class _Guard:
    def to_dict(self) -> dict[str, str]:
        return {"status": "completed"}


@dataclass(frozen=True)
class _Failure:
    code: str
    message: str


@dataclass(frozen=True)
class _Identity:
    event: _Value
    chain_id: str
    parent_uuid: str
    source_link: int
    target_link: int


@dataclass(frozen=True)
class _Plan:
    action: _Value
    identity: _Identity
    parent_guard: _Guard

    def child_dict(self) -> dict[str, str]:
        return {"uuid": "child"}


@dataclass(frozen=True)
class _Record:
    intent_id: str
    state: object
    stage: object
    attempts: int
    lease_expires_at: float
    failure: object
    plan: _Plan


class LifecycleOutboxQueryTests(unittest.TestCase):
    def _database(self) -> tuple[TemporaryDirectory[str], sqlite3.Connection]:
        temporary = TemporaryDirectory()
        connection = sqlite3.connect(Path(temporary.name) / "outbox.db")
        connection.row_factory = sqlite3.Row
        connection.execute(
            "CREATE TABLE lifecycle_outbox ("
            "intent_id TEXT, work_kind TEXT, plan_json TEXT, plan_fingerprint TEXT, "
            "parent_guard_json TEXT, configuration_fingerprint TEXT, schedule_fingerprint TEXT, "
            "lifecycle_stage TEXT, processing_state TEXT, lease_owner TEXT, lease_expires_at REAL, "
            "attempts INTEGER, failure_json TEXT, created_at REAL, updated_at REAL, acknowledged_at REAL)"
        )
        return temporary, connection

    @staticmethod
    def _record(row: sqlite3.Row) -> _Record:
        return _Record(
            intent_id=str(row["intent_id"]), state=_Value(str(row["processing_state"])),
            stage=_Value(str(row["lifecycle_stage"])), attempts=int(row["attempts"]),
            lease_expires_at=float(row["lease_expires_at"]),
            failure=_Failure("temporary", "retry later") if row["intent_id"] == "b" else None,
            plan=_Plan(
                action=_Value("spawn_child"),
                identity=_Identity(_Value("complete"), "chain", "parent", 1, 2),
                parent_guard=_Guard(),
            ),
        )

    @staticmethod
    def _insert(connection: sqlite3.Connection, intent_id: str, state: str, updated_at: float,
                acknowledged_at: float = 0.0, lease_expires_at: float = 0.0) -> None:
        connection.execute(
            "INSERT INTO lifecycle_outbox VALUES (?, 'lifecycle', '{}', '', '{}', 'cfg', 'sch', "
            "'planned', ?, '', ?, 2, '', 1, ?, ?)",
            (intent_id, state, lease_expires_at, updated_at, acknowledged_at),
        )

    def test_status_summary_projects_metrics_order_and_poison_without_owning_connection(self) -> None:
        temporary, connection = self._database()
        self.addCleanup(connection.close)
        self.addCleanup(temporary.cleanup)
        self._insert(connection, "b", "ready", 20.0)
        self._insert(connection, "a", "claimed", 10.0, lease_expires_at=70.0)
        self._insert(connection, "c", "acknowledged", 5.0, acknowledged_at=5.0)
        self._insert(connection, "poison", "ready", 1.0)

        def decode(row: sqlite3.Row) -> _Record:
            if row["intent_id"] == "poison":
                raise StatusRowPoison("broken payload")
            return self._record(row)

        summary = status_summary(
            connection, now=100.0, limit=3, stale_after=30.0,
            retention_seconds=60.0, intent_id=None, decode_row=decode,
        )

        self.assertEqual(summary.states, {"ready": 2, "claimed": 1, "acknowledged": 1})
        self.assertEqual(summary.stale_claims, 1)
        self.assertEqual(summary.max_attempts, 2)
        self.assertEqual(summary.acknowledged, 1)
        self.assertEqual(summary.eligible, 1)
        self.assertEqual(summary.oldest_age_s, 95)
        self.assertEqual([row.intent_id for row in summary.records], ["a", "poison", "b"])
        self.assertEqual(summary.records[1].reason, "broken payload")
        self.assertEqual(summary.records[0].lease_age_s, 30)
        self.assertEqual(summary.records[0].plan.action, "spawn_child")
        self.assertEqual(summary.records[0].plan.child_uuid, "child")
        self.assertEqual(summary.records[2].failure, OutboxStatusFailureSummary("temporary", "retry later"))
        self.assertEqual(connection.execute("SELECT 1").fetchone()[0], 1)

    def test_status_summary_filters_one_intent(self) -> None:
        temporary, connection = self._database()
        self.addCleanup(connection.close)
        self.addCleanup(temporary.cleanup)
        self._insert(connection, "a", "ready", 1.0)
        self._insert(connection, "b", "ready", 2.0)

        summary = status_summary(
            connection, now=100.0, limit=10, stale_after=30.0,
            retention_seconds=60.0, intent_id=" b ", decode_row=self._record,
        )

        self.assertEqual([row.intent_id for row in summary.records], ["b"])

    def test_snapshot_rows_are_complete_sorted_and_do_not_close_connection(self) -> None:
        temporary, connection = self._database()
        self.addCleanup(connection.close)
        self.addCleanup(temporary.cleanup)
        self._insert(connection, "z", "ready", 1.0)
        self._insert(connection, "a", "ready", 1.0)
        connection.execute("UPDATE lifecycle_outbox SET work_kind='integrity' WHERE intent_id='z'")

        rows = snapshot_rows(
            connection,
            decode_lifecycle_row=lambda row: ("lifecycle", str(row["intent_id"])),
            decode_integrity_row=lambda row: ("integrity", str(row["intent_id"])),
        )

        self.assertEqual(rows, (("lifecycle", "a"), ("integrity", "z")))
        self.assertEqual(connection.execute("SELECT 1").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
