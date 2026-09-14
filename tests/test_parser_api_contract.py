import functools
import re
import unittest
from types import SimpleNamespace

import nautical_core as core
from nautical_core import parser_api
from nautical_core.parsing import parser_frontend
from nautical_core.parsing import parser_support_api


class ParseError(ValueError):
    pass


def split_csv(value):
    return [part.strip() for part in value.split(",")]


class ParserFrontendContractTests(unittest.TestCase):
    def test_split_top_level_respects_parentheses_and_drops_empty_tail(self):
        self.assertEqual(
            parser_frontend.split_top_level("a+(b+c)+", "+"),
            ["a", "(b+c)"],
        )
        self.assertEqual(
            parser_frontend.split_top_level("a|(b|c)|", "|"),
            ["a", "(b|c)"],
        )

    def test_normalize_anchor_input_unquotes_aliases_and_expands_weekly_times(self):
        normalized = parser_frontend.normalize_anchor_expr_input(
            '  "m:1,2 + y:06-rand + w:mon@t=09:00,fri@t=15:00"  ',
            unwrap_quotes=lambda value: (
                value.strip()[1:-1] if value.strip()[:1] in "'\"" and value.strip()[-1:] == value.strip()[:1]
                else value
            ),
            rewrite_weekly_multi_time_atoms=lambda value: parser_frontend.rewrite_weekly_multi_time_atoms(
                value, split_csv_tokens=split_csv, re_mod=re
            ),
            re_mod=re,
            parse_error_cls=ParseError,
        )
        self.assertEqual(
            normalized,
            "(m:1,2) + y:rand-06 + (w:mon@t=09:00 | w:fri@t=15:00)",
        )

    def test_normalize_quoted_month_random_alias(self):
        self.assertEqual(
            parser_frontend.normalize_anchor_expr_input(
                '"07-rand"',
                unwrap_quotes=lambda value: value.strip()[1:-1],
                rewrite_weekly_multi_time_atoms=lambda value: value,
                re_mod=re,
                parse_error_cls=ParseError,
            ),
            "rand-07",
        )

    def test_normalize_anchor_input_rejects_oversized_expression(self):
        with self.assertRaisesRegex(ParseError, "max 1024"):
            parser_frontend.normalize_anchor_expr_input(
                "x" * 1025,
                unwrap_quotes=lambda value: value,
                rewrite_weekly_multi_time_atoms=lambda value: value,
                re_mod=re,
                parse_error_cls=ParseError,
            )

    def test_weekly_multi_time_rewrite_leaves_unrelated_expression_unchanged(self):
        self.assertEqual(
            parser_frontend.rewrite_weekly_multi_time_atoms(
                "m:1 + y:01-01", split_csv_tokens=split_csv, re_mod=re
            ),
            "m:1 + y:01-01",
        )
        self.assertEqual(
            parser_frontend.rewrite_weekly_multi_time_atoms(
                "w:mon@t=09:00,fri@t=15:00",
                split_csv_tokens=split_csv,
                re_mod=re,
            ),
            "(w:mon@t=09:00 | w:fri@t=15:00)",
        )

    def test_validation_reports_malformed_yearly_tokens_and_comma_joins(self):
        def fatal(tail):
            return parser_frontend.fatal_bad_colon_in_year_tail(
                tail, split_csv_tokens=split_csv, re_mod=re, yearfmt=lambda: "MD"
            )

        for value, message in (
            ("y:01:02", "uses ':' between numbers"),
            ("y:05:15", "uses ':' between numbers"),
            ("y:q1:q2", "must use '..'"),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ParseError, message):
                    parser_frontend.raise_on_bad_colon_year_tokens(
                        value,
                        re_mod=re,
                        fatal_bad_colon_in_year_tail=fatal,
                        parse_error_cls=ParseError,
                    )
        self.assertIn(
            "uses ':' between numbers",
            parser_frontend.fatal_bad_colon_in_year_tail(
                "05:15",
                split_csv_tokens=split_csv,
                re_mod=re,
                yearfmt=lambda: "MD",
            ),
        )
        with self.assertRaisesRegex(ParseError, "must be joined"):
            parser_frontend.raise_if_comma_joined_anchors(
                "m:31,w:sun", re_mod=re, parse_error_cls=ParseError
            )
        with self.assertRaisesRegex(ParseError, "It looks like you used a comma"):
            parser_frontend.raise_if_comma_joined_anchors(
                "31@t=14:00, w:sun", re_mod=re, parse_error_cls=ParseError
            )


class ParserSupportContractTests(unittest.TestCase):
    def _binding(self, *, context=False):
        namespace = {
            "_ttl_lru_cache": functools.lru_cache,
            "_parser_frontend": parser_frontend,
            "_split_csv_tokens": split_csv,
            "re": re,
            "ParseError": ParseError,
            "_yearfmt": lambda: "MD",
        }
        if context:
            return parser_support_api.for_core(
                module=SimpleNamespace(**namespace),
                context=SimpleNamespace(namespace=namespace),
            )
        return parser_support_api.for_core(module=SimpleNamespace(**namespace))

    def test_for_core_delegates_frontend_validation_and_rewrite(self):
        binding = self._binding()
        self.assertEqual(
            binding._rewrite_weekly_multi_time_atoms("w:mon@t=09:00,fri@t=15:00"),
            "(w:mon@t=09:00 | w:fri@t=15:00)",
        )
        with self.assertRaisesRegex(ParseError, "uses ':' between numbers"):
            binding._raise_on_bad_colon_year_tokens("y:01:02")
        with self.assertRaisesRegex(ParseError, "must be joined"):
            binding._raise_if_comma_joined_anchors("m:31,w:sun")

    def test_for_core_context_namespace_is_authoritative(self):
        binding = self._binding(context=True)
        self.assertEqual(binding._skip_ws_pos("  x", 0, 3), 2)


class ParserPresetContractTests(unittest.TestCase):
    def _binding(self, *, anchors=None, omits=None):
        namespace = {
            **vars(core),
            "ParseError": core.ParseError,
            "ANCHOR_PRESETS": anchors or {},
            "OMIT_PRESETS": omits or {},
        }
        return parser_api.for_core(module=core, namespace=namespace)

    def test_unknown_presets_list_available_aliases_and_config_table(self):
        binding = self._binding(
            anchors={"payday": "m:15", "workout": "w:mon,wed,fri"},
            omits={"april": "y:apr", "weekends": "w:sat,sun"},
        )
        with self.assertRaisesRegex(
            core.ParseError,
            r"Unknown anchor preset '@missing'.*Available anchor presets: @payday, @workout.*\[anchor_presets\]",
        ):
            binding.resolve_anchor_presets("@missing")
        with self.assertRaisesRegex(
            core.ParseError,
            r"Unknown omit preset '@missing'.*Available omit presets: @april, @weekends.*\[omit_presets\]",
        ):
            binding.resolve_omit_presets("@missing")

    def test_recursive_preset_diagnostic_preserves_reference_chain(self):
        binding = self._binding(anchors={"a": "@b", "b": "@c", "c": "@a"})
        with self.assertRaisesRegex(
            core.ParseError,
            r"Recursive anchor preset reference detected: @a -> @b -> @c -> @a",
        ):
            binding.resolve_anchor_presets("@a")

    def test_nested_preset_display_shows_resolved_leaf(self):
        binding = self._binding(
            anchors={"payday": "m:15,-1bd", "salary": "@payday"},
            omits={"april": "y:apr", "spring": "@april"},
        )
        self.assertEqual(
            binding.anchor_preset_display("@salary"),
            ("Preset", "@salary → m:15,-1bd"),
        )
        self.assertEqual(
            binding.omit_preset_display("@spring"),
            ("Omit preset", "@spring → y:apr"),
        )


if __name__ == "__main__":
    unittest.main()
