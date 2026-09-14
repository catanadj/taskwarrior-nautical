"""Direct contracts for CP duration parsing and interval selection."""

import unittest
from datetime import date, timedelta, timezone

import nautical_core as core
from nautical_core import cp_parser


class CpSequenceContractTests(unittest.TestCase):
    def test_duration_and_sequence_parser_expands_and_rejects_invalid_values(self) -> None:
        self.assertEqual(
            cp_parser.parse_cp_duration("P1DT2H30M"),
            timedelta(days=1, hours=2, minutes=30),
        )
        self.assertEqual(cp_parser.parse_cp_duration("P1D"), timedelta(days=1))
        self.assertEqual(cp_parser.parse_cp_duration("3d"), timedelta(days=3))
        self.assertEqual(cp_parser.parse_cp_duration("6h30m"), timedelta(hours=6, minutes=30))
        self.assertEqual(cp_parser.parse_cp_duration("24h+1s"), timedelta(hours=24, seconds=1))
        self.assertEqual(cp_parser.parse_cp_duration("3d-1s"), timedelta(days=3, seconds=-1))
        self.assertEqual(
            cp_parser.parse_cp_sequence("3d,P20D,16h"),
            [timedelta(days=3), timedelta(days=20), timedelta(hours=16)],
        )
        self.assertEqual(
            cp_parser.parse_cp_sequence("7d*3,14d"),
            [timedelta(days=7)] * 3 + [timedelta(days=14)],
        )
        self.assertEqual(
            cp_parser.cp_sequence_interval_for_link("7d*3,14d", 4), timedelta(days=14)
        )
        self.assertEqual(
            cp_parser.cp_sequence_interval_for_link("7d*3,14d", 5), timedelta(days=7)
        )
        self.assertEqual(
            cp_parser.cp_sequence_interval_for_link("rand(3d..7d)*2", 2, "cp-repeat"),
            cp_parser.cp_sequence_interval_for_link(
                "rand(3d..7d),rand(3d..7d)", 2, "cp-repeat"
            ),
        )
        self.assertIsNone(cp_parser.parse_cp_sequence("3d,,7d"))
        self.assertEqual(cp_parser.cp_sequence_parse_error("3d,,7d"), "empty duration at position 2")
        self.assertEqual(
            cp_parser.cp_sequence_parse_error("3d,abc"), "invalid duration 'abc' at position 2"
        )
        self.assertIn("positive repeat count", cp_parser.cp_sequence_parse_error("7d*0") or "")
        self.assertIn(
            "lower bound must be <= upper bound",
            cp_parser.cp_sequence_parse_error("rand(7d..3d)") or "",
        )
        self.assertIn(
            "lower bound must be >= 0",
            cp_parser.cp_sequence_parse_error("2d~3d") or "",
        )
        self.assertIn(
            "expected rand(<duration>..<duration>)",
            cp_parser.cp_sequence_parse_error("rand(3d-7d)") or "",
        )
        self.assertIn(
            "expected rand(<duration>..<duration>)",
            cp_parser.cp_sequence_parse_error("3d,rand(3d..7d") or "",
        )
        self.assertIn(
            "invalid duration bound",
            cp_parser.cp_sequence_parse_error("14d~abc") or "",
        )

    def test_link_selection_cycles_and_clamps_boundary_links(self) -> None:
        cp = "3d,20d,7d"
        expected = {
            0: timedelta(days=3),
            1: timedelta(days=3),
            2: timedelta(days=20),
            3: timedelta(days=7),
            4: timedelta(days=3),
            5: timedelta(days=20),
            9999: timedelta(days=7),
            10000: timedelta(days=3),
        }

        for link_number, interval in expected.items():
            with self.subTest(link_number=link_number):
                self.assertEqual(cp_parser.cp_sequence_interval_for_link(cp, link_number), interval)

    def test_random_and_jitter_selection_is_deterministic_bounded_and_chain_scoped(self) -> None:
        cases = (
            ("rand(3d..7d)", 1, timedelta(days=3), timedelta(days=7), 86400),
            ("rand(12h..36h)", 1, timedelta(hours=12), timedelta(hours=36), 3600),
            ("rand(30m..90m)", 1, timedelta(minutes=30), timedelta(minutes=90), 60),
            ("14d~2d", 1, timedelta(days=12), timedelta(days=16), 86400),
            ("3d,14d~2d,7d", 2, timedelta(days=12), timedelta(days=16), 86400),
        )
        for cp, link_number, lower, upper, granularity in cases:
            with self.subTest(cp=cp):
                first = cp_parser.cp_sequence_interval_for_link(cp, link_number)
                second = cp_parser.cp_sequence_interval_for_link(cp, link_number)
                self.assertEqual(first, second)
                self.assertGreaterEqual(first, lower)
                self.assertLessEqual(first, upper)
                self.assertEqual(int(first.total_seconds()) % granularity, 0)

        sequence = "3d,rand(10d..20d),7d"
        self.assertEqual(
            cp_parser.cp_sequence_interval_for_link(sequence, 1), timedelta(days=3)
        )
        selected = cp_parser.cp_sequence_interval_for_link(sequence, 2)
        self.assertGreaterEqual(selected, timedelta(days=10))
        self.assertLessEqual(selected, timedelta(days=20))
        self.assertEqual(
            cp_parser.cp_sequence_interval_for_link(sequence, 3), timedelta(days=7)
        )

        jittered_sequence = cp_parser.cp_sequence_interval_for_link("3d,14d~2d,7d", 2)
        self.assertGreaterEqual(jittered_sequence, timedelta(days=12))
        self.assertLessEqual(jittered_sequence, timedelta(days=16))

        cp = "rand(11d..14d)"

        def sequence(chain_id: str) -> list[int]:
            return [
                int(cp_parser.cp_sequence_interval_for_link(cp, link_number, chain_id).total_seconds() // 86400)
                for link_number in range(1, 17)
            ]

        chain_a = sequence("chain-a")
        chain_b = sequence("chain-b")
        self.assertNotEqual(chain_a, chain_b)
        self.assertEqual(chain_a, sequence(" CHAIN-A "))
        self.assertTrue(all(11 <= value <= 14 for value in chain_a + chain_b))

    def test_whole_day_step_preserves_local_wall_clock_across_dst(self) -> None:
        if core._LOCAL_TZ is None:
            self.skipTest("timezone data is unavailable; core is in UTC-only mode")

        start_local = core.build_local_datetime(date(2026, 3, 28), (10, 0))
        start_utc = start_local.astimezone(timezone.utc).replace(microsecond=0)
        naive_next_utc = (start_utc + timedelta(days=1)).replace(microsecond=0)

        local_start = core.to_local(start_utc)
        preserved_local = core.build_local_datetime(
            (local_start + timedelta(days=1)).date(), (local_start.hour, local_start.minute)
        )
        preserved_next_utc = preserved_local.astimezone(timezone.utc).replace(microsecond=0)
        local_preserved = core.to_local(preserved_next_utc)
        self.assertEqual((local_preserved.hour, local_preserved.minute), (local_start.hour, local_start.minute))

        local_naive = core.to_local(naive_next_utc)
        if local_start.utcoffset() != local_naive.utcoffset():
            self.assertNotEqual((local_naive.hour, local_naive.minute), (local_start.hour, local_start.minute))


if __name__ == "__main__":
    unittest.main()
