import unittest

from nautical_core.operator_presentation import ordered_records


class OperatorPresentationTests(unittest.TestCase):
    def test_ordered_records_is_deterministic_for_shuffled_canonical_rows(self) -> None:
        rows = [
            {"domain": "chains", "severity": "warning", "actionability": "repairable", "chain_id": "b", "link": 2, "code": "z"},
            {"domain": "config", "severity": "info", "actionability": "informational", "code": "a"},
            {"domain": "chains", "severity": "error", "actionability": "blocking", "chain_id": "a", "link": 1, "code": "x"},
        ]
        expected = tuple(ordered_records(rows))
        self.assertEqual(expected, tuple(ordered_records(list(reversed(rows)))))
        self.assertEqual(expected[0]["domain"], "chains")
        self.assertEqual(expected[0]["severity"], "error")


if __name__ == "__main__":
    unittest.main()
