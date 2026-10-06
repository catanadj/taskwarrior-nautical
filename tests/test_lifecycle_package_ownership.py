"""Contracts for canonical lifecycle package ownership."""

from __future__ import annotations

import importlib
import importlib.util
import unittest


class LifecyclePackageOwnershipTests(unittest.TestCase):
    def test_lifecycle_owners_live_under_package_without_old_module_paths(self) -> None:
        package_spec = importlib.util.find_spec("nautical_core.lifecycle")
        self.assertIsNotNone(package_spec, "lifecycle domain must have a package owner")

        package = importlib.import_module("nautical_core.lifecycle")
        models = importlib.import_module("nautical_core.lifecycle.models")
        outbox = importlib.import_module("nautical_core.lifecycle.outbox")
        read_service = importlib.import_module("nautical_core.lifecycle.read_service")
        self.assertIsNotNone(package)
        self.assertTrue(hasattr(models, "LifecyclePlan"))
        self.assertTrue(hasattr(outbox, "LifecycleOutboxRepository"))
        self.assertTrue(hasattr(read_service, "LifecycleReadService"))

        old_modules = (
            "lifecycle_application",
            "lifecycle_execution_policy",
            "lifecycle_models",
            "lifecycle_operator_owner",
            "lifecycle_outbox",
            "lifecycle_outbox_claims",
            "lifecycle_outbox_codec",
            "lifecycle_outbox_maintenance",
            "lifecycle_outbox_operations",
            "lifecycle_outbox_queries",
            "lifecycle_outbox_schema",
            "lifecycle_planner",
            "lifecycle_read_service",
            "lifecycle_reconciliation",
            "lifecycle_recovery_models",
            "lifecycle_state",
        )
        for module_name in old_modules:
            with self.subTest(module=module_name):
                self.assertIsNone(
                    importlib.util.find_spec(f"nautical_core.{module_name}"),
                    f"old internal module path remains importable: {module_name}",
                )


if __name__ == "__main__":
    unittest.main()
