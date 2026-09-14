"""Direct contracts for stable recurrence identity selection."""

import unittest
import uuid

import nautical_core as core
from nautical_core.add_anchor_preview import _preview_seed_base
from nautical_core.modify_schedule_effects import recurrence_seed_base
from nautical_core.modify_spawn_prep import SpawnIdentityError, stable_child_uuid
from nautical_core.modify_timeline import _timeline_seed_base


class RecurrenceIdentityContracts(unittest.TestCase):
    def test_modify_schedule_prefers_chain_id_and_has_preview_fallback(self):
        self.assertEqual(
            recurrence_seed_base({"chainID": "chain-a", "uuid": "uuid-a"}),
            "chain-a",
        )
        self.assertEqual(recurrence_seed_base({}), "preview")

    def test_timeline_prefers_chain_id_and_uses_uuid_fallback(self):
        self.assertEqual(
            _timeline_seed_base({"chainID": "chain-a", "uuid": "uuid-a"}),
            "chain-a",
        )
        self.assertEqual(_timeline_seed_base({}), "preview")

    def test_add_preview_prefers_chain_id_and_uses_root_uuid_fallback(self):
        self.assertEqual(
            _preview_seed_base({"chainID": "chain-a", "uuid": "uuid-a"}, "root"),
            "chain-a",
        )
        self.assertEqual(_preview_seed_base({}, "root"), "root")

    def test_legacy_lowercase_chainid_cannot_supply_stable_child_identity(self):
        def stable(parent, child):
            return stable_child_uuid(
                parent,
                child,
                task_uuid_or_empty=lambda task: str(task.get("uuid") or "").strip(),
                coerce_int=core.coerce_int,
                stable_child_uuid_namespace=uuid.NAMESPACE_URL,
            )

        with self.assertRaises(SpawnIdentityError):
            stable(
                {"chainid": "legacy-1", "uuid": "parent-uuid"},
                {"chainid": "legacy-1", "link": 2},
            )
        with self.assertRaises(SpawnIdentityError):
            stable({"uuid": "parent-uuid"}, {"link": 2})


if __name__ == "__main__":
    unittest.main()
