"""Integrity checks for the developer golden-test registry.

These checks deliberately inspect the runner's registry without moving golden
assertions into another harness.  The explicit retired set documents the
small number of characterization helpers whose coverage now lives in direct
contract tests.
"""

import importlib
import unittest


GOLDEN_MODULE = "dev_tools.nautical_golden_tests"
RETIRED_CHARACTERIZATION_TESTS = frozenset(
    {
        "test_reconcile_candidate_discovery_is_narrow_and_deterministic",
        "test_reconcile_delayed_expiration_dry_run_converges_to_live_slot",
        "test_reconcile_empty_snapshot_is_authoritative",
        "test_reconcile_lifecycle_outcomes_preserve_retry_and_manual_review",
        "test_reconcile_planning_configuration_drift_is_partial",
        "test_reconcile_reuses_verified_live_recovery_child",
        "test_reconcile_snapshot_reuses_initial_chain_export",
        "test_prev_weekday_natural_text",
        "test_natural_interval_or_branches_keep_cadence_with_subject",
        "test_natural_compresses_repeated_within_variants",
        "test_natural_compresses_repeated_fall_on_variants",
        "test_time_window_natural_language_uses_bounded_interval",
    }
)


class GoldenRegistryIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = importlib.import_module(GOLDEN_MODULE)

    def test_registered_functions_are_unique_and_callable(self):
        registered = [*self.golden.TESTS, *self.golden.DEEP_TESTS]
        names = [fn.__name__ for fn in registered]
        self.assertEqual(len(names), len(set(names)), "golden registry contains duplicates")
        self.assertTrue(all(callable(fn) for fn in registered))
        self.assertTrue(all(name.startswith("test_") for name in names))

    def test_live_top_level_tests_are_registered_or_explicitly_retired(self):
        registered = {
            fn.__name__ for fn in (*self.golden.TESTS, *self.golden.DEEP_TESTS)
        }
        top_level = {
            name
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        }
        self.assertEqual(
            top_level - registered,
            set(RETIRED_CHARACTERIZATION_TESTS),
            "unregistered golden tests must be migrated or explicitly retired",
        )
        self.assertFalse(RETIRED_CHARACTERIZATION_TESTS & registered)

    def test_registry_inventory_counts_match_documented_snapshot(self):
        registered = [*self.golden.TESTS, *self.golden.DEEP_TESTS]
        top_level = [
            value
            for name, value in vars(self.golden).items()
            if name.startswith("test_") and callable(value)
        ]
        self.assertEqual(len(top_level), 995)
        self.assertEqual(len(registered), 983)
        self.assertEqual(len(RETIRED_CHARACTERIZATION_TESTS), 12)


if __name__ == "__main__":
    unittest.main()
