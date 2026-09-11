from __future__ import annotations

import unittest

from nautical_core.modify_format_effects import HumanDeltaPort, human_delta


class HumanDeltaTests(unittest.TestCase):
    def test_uses_three_argument_formatter_contract(self) -> None:
        class Core:
            def humanize_delta(self, start, end, *, use_months_days):
                return (start, end, use_months_days)

        result = human_delta(HumanDeltaPort(Core().humanize_delta), 1, 2, False)
        self.assertEqual(result, (1, 2, False))

    def test_formatter_type_errors_are_not_hidden(self) -> None:
        class Core:
            def humanize_delta(self, *_args, **_kwargs):
                raise TypeError("internal failure")

        with self.assertRaisesRegex(TypeError, "internal failure"):
            human_delta(HumanDeltaPort(Core().humanize_delta), 1, 2)


if __name__ == "__main__":
    unittest.main()
