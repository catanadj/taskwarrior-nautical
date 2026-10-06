"""Contracts for on-exit composition services."""

from __future__ import annotations

import unittest

from nautical_core.exit_composition import ExitServices
from nautical_core.hook_results import ExitHookResponse
from nautical_core.on_exit_models import ExitDrainStats


class ExitCompositionContractTests(unittest.TestCase):
    def test_result_uses_the_supported_exit_hook_response(self) -> None:
        services = ExitServices(
            redirect_stdout=lambda: None,
            drain_outbox=lambda _runtime: ExitDrainStats(),
            strict_feedback=lambda _stats: None,
        )

        result = services.result(
            exit_code=0,
            feedback_message=None,
            stats=ExitDrainStats(),
        )

        self.assertEqual(result, ExitHookResponse(exit_code=0, stats=ExitDrainStats()))


if __name__ == "__main__":
    unittest.main()
