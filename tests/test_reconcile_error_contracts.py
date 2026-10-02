"""Error contracts for the reconcile operator boundary."""

from types import SimpleNamespace
import unittest

import nautical_core.tools.nautical_reconcile as reconcile
from nautical_core.task_models import TaskObservation


class ReconcileErrorContracts(unittest.TestCase):
    def test_native_until_match_propagates_internal_parser_fault(self) -> None:
        class BrokenParser:
            def parse(self, _value: object) -> tuple[None, str]:
                raise RuntimeError("unexpected parser failure")

        task = TaskObservation.from_mapping(
            {"until": "2026-01-01T00:00:00Z"}, source_query="error-contract"
        )
        token = reconcile._RECONCILE_RUNTIME.set(
            SimpleNamespace(datetime_parser=BrokenParser())
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "unexpected parser failure"):
                reconcile._native_until_matches(task, "2026-01-02T00:00:00Z", None)
        finally:
            reconcile._RECONCILE_RUNTIME.reset(token)


if __name__ == "__main__":
    unittest.main()
