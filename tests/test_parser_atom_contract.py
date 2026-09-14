from __future__ import annotations

from functools import partial
import re
import unittest

from nautical_core.parsing import parser_atoms


class ParserAtomContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.weekdays = {
            "mon": 0,
            "tue": 1,
            "wed": 2,
            "thu": 3,
            "fri": 4,
            "sat": 5,
            "sun": 6,
        }
        self.parse_atom_mods = partial(
            parser_atoms.parse_atom_mods,
            split_csv_tokens=lambda value: [part.strip() for part in value.split(",")],
            parse_hhmm=lambda value: parser_atoms.parse_hhmm(
                value, hhmm_re=re.compile(r"^(\d{1,2}):(\d{2})$")
            ),
            next_prev_wd_re=re.compile(r"^(next|prev)-(mon|tue|wed|thu|fri|sat|sun)$"),
            weekdays=self.weekdays,
            day_offset_re=re.compile(r"^([+-]\d+)d$"),
            parse_error_cls=ValueError,
        )

    def _build_atom_dnf(self, head: str, tail: str):
        return parser_atoms.build_anchor_atom_dnf(
            head,
            tail,
            parse_atom_head=lambda value: parser_atoms.parse_atom_head(
                value, re_mod=re, parse_error_cls=ValueError
            ),
            parse_group_with_inline_mods=lambda *_args: None,
            normalize_monthly_ordinal_spec=lambda value: parser_atoms.normalize_monthly_ordinal_spec(
                value, re_mod=re
            ),
            split_csv_lower=lambda value: [part.strip().lower() for part in value.split(",")],
            parse_atom_mods=self.parse_atom_mods,
            parse_error_cls=ValueError,
        )

    def test_atom_head_intervals_are_normalized_without_clamping(self) -> None:
        self.assertEqual(
            parser_atoms.parse_atom_head(" W/1000 ", re_mod=re, parse_error_cls=ValueError),
            ("w", 1000),
        )
        self.assertEqual(
            parser_atoms.parse_atom_head("moon", re_mod=re, parse_error_cls=ValueError),
            ("moon", 1),
        )
        with self.assertRaisesRegex(ValueError, "Invalid anchor head"):
            parser_atoms.parse_atom_head("q/2", re_mod=re, parse_error_cls=ValueError)

    def test_atom_modifiers_preserve_times_and_signed_offsets(self) -> None:
        mods = self.parse_atom_mods("t=09:00,12:00@+1d@-2bd")
        self.assertEqual(mods["t"], [(9, 0), (12, 0)])
        self.assertEqual(mods["day_offset"], 1)
        self.assertEqual(mods["business_day_offset"], -2)
        self.assertEqual(
            self.parse_atom_mods("t=09:00@next-mon")["wd"],
            self.weekdays["mon"],
        )
        with self.assertRaisesRegex(ValueError, "Duplicate '@t='"):
            self.parse_atom_mods("t=09:00@t=12:00")

    def test_monthly_ordinal_builds_canonical_atom_dnf(self) -> None:
        self.assertEqual(
            self._build_atom_dnf("m", "1st-mon@t=09:00"),
            [[
                {
                    "typ": "m",
                    "spec": "1mon",
                    "ival": 1,
                    "mods": {
                        "t": (9, 0),
                        "roll": None,
                        "wd": None,
                        "bd": False,
                        "day_offset": 0,
                        "business_day_offset": 0,
                    },
                }
            ]],
        )

    def test_atom_parser_stops_before_next_expression_term(self) -> None:
        expression = "w:mon@t=09:00 + m:1"
        node, next_index = parser_atoms.parse_anchor_atom_at(
            expression,
            0,
            len(expression),
            skip_ws_pos=lambda text, index, limit: len(text[:limit]) - len(text[:limit].lstrip()),
            raise_if_comma_joined_anchors=lambda _tail: None,
            build_anchor_atom_dnf=self._build_atom_dnf,
            parse_error_cls=ValueError,
        )
        self.assertEqual(node[0][0]["spec"], "mon")
        self.assertEqual(node[0][0]["mods"]["t"], (9, 0))
        self.assertEqual(expression[next_index:].lstrip(), "+ m:1")


if __name__ == "__main__":
    unittest.main()
