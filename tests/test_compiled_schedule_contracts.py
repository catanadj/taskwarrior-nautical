"""Direct contracts for canonical schedule compilation and reuse."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import nautical_core as core
from nautical_core.compiled_schedule import CompiledSchedule, CompiledScheduleCache
from nautical_core.recurrence_evaluator import RecurrenceEvaluator
from nautical_core.task_codec import DEFAULT_TASK_CODEC


def _compiled(row: dict) -> CompiledSchedule:
    observation = DEFAULT_TASK_CODEC.decode_row(
        row, source_query="test:compiled-schedule"
    )
    return CompiledSchedule.from_observation(observation)


class CompiledScheduleContractTests(unittest.TestCase):
    def test_equivalent_fields_share_canonical_fingerprint_and_cache_entry(self) -> None:
        first = _compiled(
            {
                "uuid": "00000000-0000-4000-8000-000000000507",
                "status": "pending",
                "chainID": "compiled-chain",
                "link": 1,
                "anchor": " w:mon ",
                "anchor_mode": "SKIP",
                "chainMax": "4",
            }
        )
        equivalent = _compiled(
            {
                "uuid": "00000000-0000-4000-8000-000000000508",
                "status": "pending",
                "chainID": "compiled-chain",
                "link": 1,
                "anchor": "w:mon",
                "anchor_mode": "skip",
                "chainMax": 4,
            }
        )

        self.assertEqual(first.fingerprint, equivalent.fingerprint)
        cache = CompiledScheduleCache(max_entries=2)
        cached = cache.get_or_compile(first.spec)
        self.assertIs(cached, cache.get_or_compile(equivalent.spec))
        self.assertEqual(first.to_dict()["compiler_schema"], 1)
        self.assertTrue(first.cache_key.startswith("compiled-schedule:1:cs1-"))
        self.assertEqual(json.loads(first.to_diagnostic_json()), first.to_dict())

    def test_compiled_evaluator_uses_normalized_dnfs_without_reparsing(self) -> None:
        observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000503",
                "status": "pending",
                "chainID": "compiled-observation-chain",
                "link": 1,
                "anchor": "w:mon + y:jul@t=09:00",
                "omit": "y:07-04",
                "chainMax": 4,
            },
            source_query="test:compiled-schedule",
        )
        compiled = CompiledSchedule.from_observation(observation)
        evaluator = RecurrenceEvaluator.from_compiled(compiled)

        self.assertEqual(evaluator.spec.context.chain_id, "compiled-observation-chain")
        self.assertEqual(evaluator.spec, compiled.spec)
        normalized = compiled.to_dict()["schedule"]["normalized"]
        self.assertEqual(normalized["provider"]["kind"], "anchor")
        self.assertEqual(normalized["identity"], "compiled-observation-chain")
        self.assertTrue(normalized["anchor_dnf"])
        self.assertTrue(normalized["omit_dnf"])
        self.assertTrue(normalized["time_projection"])

        with patch.object(
            core,
            "parse_anchor_expr_to_dnf_cached",
            side_effect=AssertionError("compiled schedule reparsed its anchor"),
        ):
            self.assertTrue(evaluator.anchor_dnf)

    def test_invalid_recurrence_shapes_are_rejected_during_compilation(self) -> None:
        for recurrence, message in (
            ({}, "recurrence requires"),
            ({"cp": "1d", "anchor": "w:mon"}, "both cp and anchor"),
            ({"anchor": "w:mon", "chainMax": 0}, "chainMax"),
            ({"anchor": "w:mon", "anchor_mode": "unknown"}, "anchor_mode"),
        ):
            with self.subTest(recurrence=recurrence), self.assertRaisesRegex(
                ValueError, message
            ):
                _compiled(
                    {
                        "uuid": "00000000-0000-4000-8000-000000000511",
                        "status": "pending",
                        "chainID": "compiled-chain",
                        "link": 1,
                        **recurrence,
                    }
                )


if __name__ == "__main__":
    unittest.main()
