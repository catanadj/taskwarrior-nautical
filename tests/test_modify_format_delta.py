from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from typing import Callable, get_type_hints

from nautical_core.modify_format_effects import HumanDeltaPort, human_delta


class HumanDeltaTests(unittest.TestCase):
    def test_human_delta_port_has_datetime_text_contract(self) -> None:
        self.assertEqual(
            get_type_hints(HumanDeltaPort)["humanize"],
            Callable[[datetime, datetime, bool], str],
        )
        self.assertEqual(
            get_type_hints(human_delta),
            {
                "port": HumanDeltaPort,
                "start": datetime,
                "end": datetime,
                "prefer_months": bool,
                "return": str,
            },
        )

    def test_uses_three_argument_formatter_contract(self) -> None:
        class Core:
            def humanize_delta(self, start, end, use_months_days):
                return f"{start:%Y-%m-%d}|{end:%Y-%m-%d}|{use_months_days}"

        start = datetime(2026, 1, 1)
        end = start + timedelta(days=1)
        result = human_delta(HumanDeltaPort(Core().humanize_delta), start, end, False)
        self.assertEqual(result, "2026-01-01|2026-01-02|False")

    def test_formatter_type_errors_are_not_hidden(self) -> None:
        class Core:
            def humanize_delta(self, *_args, **_kwargs):
                raise TypeError("internal failure")

        with self.assertRaisesRegex(TypeError, "internal failure"):
            human_delta(HumanDeltaPort(Core().humanize_delta), 1, 2)


if __name__ == "__main__":
    unittest.main()
