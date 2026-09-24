import unittest

import nautical_core.lifecycle_outbox as lifecycle_outbox
import nautical_core.lifecycle_outbox_operations as operations


class OutboxCompatibilityRemovalTests(unittest.TestCase):
    def test_canonical_repository_contract_is_exported(self) -> None:
        # The repository implementation remains private; this is the one
        # supported module-level contract for internal callers.
        self.assertTrue(hasattr(lifecycle_outbox, "LifecycleOutboxRepository"))
        for name in ("RepositoryOutboxOperations", "RepositoryLifecycleExecution", "repository_for_taskdata"):
            self.assertFalse(hasattr(operations, name))


if __name__ == "__main__":
    unittest.main()
