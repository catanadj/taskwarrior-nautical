"""Direct contracts for CP interval agreement across hook owners."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from nautical_core.add_preview_composition import cp_sequence_period_for_link
import nautical_core.cp_parser as cp_parser
from nautical_core.modify_schedule_effects import SequencePorts, sequence_period_for_link


class CpIntervalCrossPathContractTests(unittest.TestCase):
    def test_on_add_and_modify_select_the_same_interval_for_each_link(self) -> None:
        core = SimpleNamespace(
            cp_sequence_interval_for_token=cp_parser.cp_sequence_interval_for_token
        )
        add_host = SimpleNamespace(core=core)
        ports = SequencePorts(cp_parser.cp_sequence_interval_for_token)
        chain_id = "abcd1234"

        for cp, link_no in (
            ("3d,20d,7d", 1),
            ("3d,20d,7d", 3),
            ("3d,20d,7d", 4),
            ("3d,rand(10d..20d),7d", 2),
            ("3d,14d~2d,7d", 2),
            ("rand(12h..36h)", 5),
        ):
            with self.subTest(cp=cp, link=link_no):
                tokens = cp_parser.parse_cp_sequence_tokens(cp)
                add_interval = cp_sequence_period_for_link(
                    add_host, tokens, cp, link_no, chain_id
                )
                modify_interval = sequence_period_for_link(
                    ports, tokens, cp, link_no, chain_id
                )
                core_interval = cp_parser.cp_sequence_interval_for_link(
                    cp, link_no, chain_id
                )
                self.assertEqual(add_interval, modify_interval)
                self.assertEqual(modify_interval, core_interval)


if __name__ == "__main__":
    unittest.main()
