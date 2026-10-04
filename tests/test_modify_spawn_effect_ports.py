"""Host contracts for modify spawn effect factories."""

from __future__ import annotations

import unittest
from typing import Any, get_type_hints


class ModifySpawnEffectPortTests(unittest.TestCase):
    def test_spawn_intent_factory_has_typed_host_contract(self) -> None:
        from nautical_core.modify_spawn_effects import spawn_intent_ports_for

        self.assertIsNot(get_type_hints(spawn_intent_ports_for)["host"], Any)

    def test_child_uuid_factory_has_typed_host_contract(self) -> None:
        from nautical_core.modify_spawn_effects import child_uuid_ports_for

        self.assertIsNot(get_type_hints(child_uuid_ports_for)["host"], Any)
