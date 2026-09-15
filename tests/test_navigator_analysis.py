import unittest
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

from nautical_navigator import (
    NavigatorAnalysisView,
    NavigatorCalendarView,
    NavigatorProjectionView,
    NavigatorSnapshot,
    analyze_navigator_snapshot,
)
import nautical_navigator as navigator


class NavigatorAnalysisTests(unittest.TestCase):
    def test_display_timezone_matches_the_configured_core_timezone(self) -> None:
        module_name = "_nautical_navigator_timezone_contract_test"
        loader = importlib.machinery.SourceFileLoader(
            module_name, str(Path(navigator.__file__).resolve())
        )
        spec = importlib.util.spec_from_loader(module_name, loader)
        self.assertIsNotNone(spec)
        isolated = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = isolated
        try:
            loader.exec_module(isolated)
            configured = str(getattr(isolated.core, "LOCAL_TZ_NAME", "")).strip()
            self.assertTrue(configured)
            self.assertEqual(getattr(isolated.LOCAL_ZONE, "key", ""), configured)
        finally:
            sys.modules.pop(module_name, None)

    def test_snapshot_analysis_is_presentation_free(self) -> None:
        snapshot = NavigatorSnapshot(
            rows=(), snapshot_id="snapshot-1", coverage="complete",
            configuration_fingerprint="config-1",
        )
        view = analyze_navigator_snapshot(
            snapshot,
            calendar=NavigatorCalendarView(),
            projection=NavigatorProjectionView(("projection unavailable",)),
        )
        self.assertIsInstance(view, NavigatorAnalysisView)
        self.assertEqual(view.chain_size, 0)
        self.assertEqual(view.projection.warnings, ("projection unavailable",))

    def test_authoritative_empty_snapshot_produces_no_chained_tasks(self) -> None:
        snapshot = NavigatorSnapshot(
            rows=(), snapshot_id="empty", coverage="complete",
            configuration_fingerprint="config-1",
        )

        with patch.object(navigator, "_run_chain_snapshot", return_value=snapshot):
            tasks = navigator.TaskAnalyzer().get_all_chained_tasks()

        self.assertEqual(tasks, [])

    def test_configuration_drift_failures_are_not_silently_ignored(self) -> None:
        failure = RuntimeError("configuration probe failed")
        with patch.object(navigator.core, "configuration_drift", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "configuration probe failed"):
                navigator._show_config_drift_warning()


if __name__ == "__main__":
    unittest.main()
