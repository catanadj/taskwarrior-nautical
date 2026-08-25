from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

from nautical_core.hook_workflow_context import (
    InvocationCaches,
    SnapshotLease,
    WorkflowInvocationContext,
)
from nautical_core.integration_context import (
    IntegrationAccess,
    IntegrationContext,
    SilentDiagnostics,
    ValidatedNauticalConfiguration,
)


class _FixedClock:
    def __init__(self) -> None:
        self.calls = 0

    def now_utc(self) -> datetime:
        self.calls += 1
        return datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)


class _Calendar:
    name = "weekday"

    def is_business_day(self, value) -> bool:
        return value.weekday() < 5


class WorkflowContextTests(unittest.TestCase):
    def _integration(self, clock: _FixedClock, taskdata: Path) -> IntegrationContext:
        config = ValidatedNauticalConfiguration(
            source="test-config",
            fingerprint="config-1",
            scheduler_fingerprint="schedule-1",
            timezone_name="Europe/Bucharest",
            values=(("timezone", "Europe/Bucharest"),),
        )
        return IntegrationContext(
            taskdata=taskdata,
            taskdata_source="test",
            command_prefix=("task",),
            configuration=config,
            local_timezone=ZoneInfo("Europe/Bucharest"),
            diagnostics=SilentDiagnostics(),
            clock=clock,
            invocation_id="invocation-1",
            command_budget=8,
            access=IntegrationAccess.READ_ONLY,
        )

    def test_capture_samples_clock_once_and_derives_local_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clock = _FixedClock()
            context = WorkflowInvocationContext.capture(
                self._integration(clock, Path(directory)),
                configuration_lease=SnapshotLease("config-1"),
                task_lease=SnapshotLease("taskdata-1"),
                business_calendar=_Calendar(),
            )
            self.assertEqual(clock.calls, 1)
            self.assertEqual(context.now_utc.hour, 12)
            self.assertEqual(context.now_local.hour, 15)
            self.assertEqual(context.configuration_lease.source_identity, "config-1")
            self.assertEqual(context.business_calendar_name, "weekday")

    def test_cache_is_bounded_and_cleared_on_close(self) -> None:
        caches = InvocationCaches(max_entries=2)
        cache = caches.store("tasks")
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("c"), 3)
        caches.clear()
        self.assertEqual(caches.names(), ())

    def test_context_cleanup_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = WorkflowInvocationContext.capture(
                self._integration(_FixedClock(), Path(directory)),
                configuration_lease=SnapshotLease("config-1"),
                task_lease=SnapshotLease("taskdata-1"),
            )
            context.caches.store("evidence").put("row", object())
        context.close()
        self.assertTrue(context.closed)
        with self.assertRaises(RuntimeError):
            context.caches.store("after-close")
            self.assertEqual(context.caches.names(), ())


if __name__ == "__main__":
    unittest.main()
