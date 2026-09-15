from __future__ import annotations

import unittest

from nautical_core import modify_lifecycle


class ModifyLifecycleContractTests(unittest.TestCase):
    def test_new_nautical_recurrence_is_promoted_and_transitions_are_explicit(self) -> None:
        self.assertFalse(modify_lifecycle.task_has_nautical_fields({"chainid": "legacy-1234"}))
        self.assertFalse(modify_lifecycle.task_has_nautical_fields({"anchor_mode": "skip"}))
        self.assertTrue(modify_lifecycle.task_has_nautical_chain_fields({"chainID": "abcd1234"}))
        self.assertTrue(modify_lifecycle.task_has_nautical_fields({"chainID": "abcd1234"}))

        old = {"uuid": "00000000-0000-4000-8000-000000000447", "status": "pending"}
        new = {**old, "anchor_file": "2026.csv", "chain": "off"}
        route = modify_lifecycle.classify_modify_route(
            old,
            new,
            is_non_completion_modify=lambda before, after: (
                before.get("status") == after.get("status") or after.get("status") != "completed"
            ),
        )
        self.assertFalse(route.is_deleted)
        self.assertTrue(route.has_nautical_fields)
        self.assertTrue(route.is_non_completion)
        source = modify_lifecycle.promote_newly_nautical_task(
            old, new, short_uuid=lambda uuid: str(uuid).split("-")[0] if uuid else ""
        )
        self.assertEqual(source, "anchor_file")
        self.assertEqual(new["chain"], "on")
        self.assertTrue(new["chainID"])

        disabled_old = {
            "uuid": "00000000-0000-4000-8000-000000000448", "status": "pending",
            "anchor": "w:mon", "chain": "on", "chainID": "00000000",
        }
        disabled_new = dict(disabled_old, chain="off")
        disabled = modify_lifecycle.apply_nautical_transition(
            disabled_old, disabled_new,
            short_uuid=lambda uuid: str(uuid).split("-")[0] if uuid else "",
        )
        self.assertEqual(disabled.state, "disabled")
        self.assertIn("chain:off", disabled.reason)
        self.assertEqual(disabled_new["chain"], "off")

        resumed_new = dict(disabled_new, chain="on")
        resumed = modify_lifecycle.apply_nautical_transition(
            disabled_new, resumed_new,
            short_uuid=lambda uuid: str(uuid).split("-")[0] if uuid else "",
        )
        self.assertEqual((resumed.state, resumed.source), ("resumed", "anchor"))
        self.assertEqual(
            modify_lifecycle.recurrence_setting_changes(
                {"anchor": "w:mon", "omit": "", "chain": "on"},
                {"anchor": "w:tue", "omit": "y:apr", "chain": "on"},
            ),
            [("anchor", "w:mon", "w:tue"), ("omit", "", "y:apr")],
        )


if __name__ == "__main__":
    unittest.main()
