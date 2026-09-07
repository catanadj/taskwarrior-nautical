from __future__ import annotations

import unittest

from nautical_core.modify_completion_compute import completion_compute_child_due, completion_compute_next_and_limits
from nautical_core.modify_models import CompletionComputeServices, CompletionLifecycleResult
from nautical_core.scheduler_models import OccurrenceSearchExhausted


class CompletionComputeTerminalEvidenceTests(unittest.TestCase):
    def test_exhaustion_is_not_collapsed_after_terminal_presentation(self) -> None:
        observed: list[OccurrenceSearchExhausted] = []
        error = OccurrenceSearchExhausted(
            "monthly recurrence", reference="2026-01-01", limit=32,
            kind=OccurrenceSearchExhausted.SEARCH_LIMIT,
        )

        def compute(_task: dict[str, object]):
            raise error

        with self.assertRaisesRegex(OccurrenceSearchExhausted, "monthly recurrence"):
            completion_compute_child_due(
                {"chain": "on", "chainID": "abcd1234", "link": 3},
                "anchor",
                compute_anchor_child_due=compute,
                compute_cp_child_due=compute,
                panel=lambda *_args, **_kwargs: None,
                print_task=lambda *_args: None,
                on_terminal=observed.append,
            )
        self.assertIs(observed[0], error)

    def test_mutation_decision_retains_search_limit_kind(self) -> None:
        error = OccurrenceSearchExhausted(
            "monthly recurrence", reference="2026-01-01", limit=32,
            kind=OccurrenceSearchExhausted.SEARCH_LIMIT,
        )
        services = CompletionComputeServices(
            completion_compute_child_due=lambda *_args: (_ for _ in ()).throw(error),
            completion_until_or_fail=lambda *_args: None,
            completion_until_guard_or_stop=lambda *_args: True,
            completion_require_child_due_or_fail=lambda *_args: True,
            completion_warn_unreasonable_duration=lambda *_args: None,
            completion_caps=lambda *_args: (0, None, None, [], None),
            completion_cap_guard_or_stop=lambda *_args: True,
        )
        result = completion_compute_next_and_limits(
            {"chain": "on", "chainID": "abcd1234", "link": 3},
            "anchor",
            4,
            None,
            services=services,
        )
        self.assertIsInstance(result, CompletionLifecycleResult)
        assert isinstance(result, CompletionLifecycleResult)
        self.assertEqual(result.state, "retryable")
        self.assertEqual(result.diagnostic.failure_kind, "search_limit")


if __name__ == "__main__":
    unittest.main()
