from __future__ import annotations

import inspect
import unittest

import nautical_core.add_anchor_compute as add_anchor_compute


class AddAnchorComputeContractTests(unittest.TestCase):
    def test_required_evaluator_guards_do_not_recheck_none(self) -> None:
        for name in ("anchor_until_summary", "anchor_build_preview"):
            with self.subTest(name=name):
                source = inspect.getsource(getattr(add_anchor_compute, name))
                self.assertNotIn("if evaluator is not None", source)
