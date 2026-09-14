from __future__ import annotations

from datetime import date
import random
import unittest

import nautical_core as core


def _random_anchor_expr(rng: random.Random) -> str:
    weekly = ("mon", "fri", "mon..fri", "mon,tue", "rand", "rand,mon", "bad")
    monthly = ("1", "-1", "2nd-mon", "last-fri", "rand", "1..15", "35", "bad")
    yearly = ("01-01", "12-31", "01-01..01-31", "q1", "rand-07", "13-01", "bad")
    modifiers = ("", "@t=09:00", "@bd", "@wd", "@nbd", "@pbd", "@nw", "@t=99:99", "@@@")
    separators = (" + ", " | ", "  ", "")

    def atom() -> str:
        kind = rng.choice(("w", "m", "y", "x"))
        if kind == "w":
            spec = rng.choice(weekly)
        elif kind == "m":
            spec = rng.choice(monthly)
        elif kind == "y":
            spec = rng.choice(yearly)
        else:
            spec = rng.choice(("bad", "noop", ""))
        return f"{kind}{rng.choice(('', '/2', '/3', '/0'))}:{spec}{rng.choice(modifiers)}"

    mode = rng.randint(0, 4)
    if mode == 0:
        return atom()
    if mode == 1:
        return atom() + rng.choice(separators) + atom()
    if mode == 2:
        return f"({atom()}{rng.choice(separators)}{atom()})"
    if mode == 3:
        return "".join(
            rng.choice("()|+@:,- abcXYZ0123") for _ in range(rng.randint(1, 40))
        )
    return f" {atom()} "


class ParserFuzzContractTests(unittest.TestCase):
    def test_mixed_valid_and_invalid_parser_inputs_never_raise_unexpected_errors(self) -> None:
        samples = (
            "", " ", "w:mon", "m:15", "y:06-01", "w:mon..fri@t=09:00",
            "m:rand", "y:rand-12", "w:rand,mon", "w:mon..fri + m:bad",
            "w:mon | m:1st-mon", "w:mon..fri@t=99:99", "x:bad",
        )
        for expression in samples:
            with self.subTest(expression=expression):
                try:
                    dnf = core.parse_anchor_expr_to_dnf_cached(expression)
                except core.ParseError:
                    continue
                self.assertIsInstance(dnf, list)

    def test_seeded_parse_validate_fuzz_accepts_only_expected_parse_errors(self) -> None:
        rng = random.Random(20260309)
        for index in range(250):
            expression = _random_anchor_expr(rng)
            with self.subTest(index=index, expression=expression):
                try:
                    dnf = core.validate_anchor_expr_strict(expression)
                except core.ParseError:
                    continue
                self.assertIsInstance(dnf, list)
                normalized = core.validate_anchor_expr_strict(dnf)
                self.assertIsInstance(normalized, list)

    def test_parsed_dnf_round_trip_preserves_scheduled_occurrences(self) -> None:
        expressions = (
            "w:mon", "w:mon,tue,wed", "w/2:fri", "m:15", "m:last-fri",
            "m:rand", "y:01-01..01-31", "w:mon + m:1", "m:rand + y:01-01..01-31",
        )
        start = date(2025, 1, 1)
        for expression in expressions:
            with self.subTest(expression=expression):
                from_string = core.validate_anchor_expr_strict(expression)
                from_dnf = core.validate_anchor_expr_strict(from_string)
                left = right = start
                for _ in range(5):
                    next_left, _meta = core.next_after_expr(
                        from_string, left, seed_base="roundtrip"
                    )
                    next_right, _meta = core.next_after_expr(
                        from_dnf, right, seed_base="roundtrip"
                    )
                    self.assertEqual(next_left, next_right)
                    left, right = next_left, next_right

    def test_anchor_normalization_is_idempotent_and_cache_isolation_holds(self) -> None:
        expressions = (
            "w:mon,wed,fri + y:apr", "(w:mon | w:fri) + y:apr", "y:jan",
            "m:1bd + y:q4", "m:-1bd + y:q4", "w/2:fri@t=09:00",
            "y:rand + w:sat", "w:mon@t=09:00,fri@t=15:00", "y:07-rand",
            "m:15@+2d | m:last-fri@nbd", "@workout + y:apr",
        )
        previous_presets = core.ANCHOR_PRESETS
        core.ANCHOR_PRESETS = {**previous_presets, "workout": "w:mon,wed,fri"}
        try:
            start = date(2026, 1, 1)
            for expression in expressions:
                with self.subTest(expression=expression):
                    parsed = core.parse_anchor_expr_to_dnf_cached(expression)
                    parsed_again = core.parse_anchor_expr_to_dnf_cached(expression)
                    validated = core.validate_anchor_expr_strict(expression)
                    revalidated = core.validate_anchor_expr_strict(parsed)
                    revalidated_again = core.validate_anchor_expr_strict(revalidated)
                    self.assertEqual(parsed, parsed_again)
                    self.assertIsNot(parsed, parsed_again)
                    self.assertEqual(validated, revalidated)
                    self.assertEqual(revalidated, revalidated_again)

                    cache_probe = core.parse_anchor_expr_to_dnf_cached(expression)
                    original_spec = cache_probe[0][0]["spec"]
                    cache_probe[0][0]["spec"] = "__mutated__"
                    after_mutation = core.parse_anchor_expr_to_dnf_cached(expression)
                    self.assertEqual(parsed_again[0][0]["spec"], original_spec)
                    self.assertEqual(after_mutation[0][0]["spec"], original_spec)
                    self.assertEqual(
                        core.describe_anchor_expr(expression),
                        core._describe_anchor_expr_from_dnf(revalidated),
                    )

                    references = [start, start, start]
                    streams = [[], [], []]
                    dnfs = (validated, revalidated, revalidated_again)
                    for _ in range(20):
                        for index, dnf in enumerate(dnfs):
                            occurrence, _meta = core.next_after_expr(
                                dnf,
                                references[index],
                                default_seed=start,
                                seed_base="normalization-invariance-v1",
                            )
                            self.assertIsNotNone(occurrence)
                            self.assertGreater(occurrence, references[index])
                            streams[index].append(occurrence)
                            references[index] = occurrence
                    self.assertEqual(streams[0], streams[1])
                    self.assertEqual(streams[1], streams[2])
            self.assertEqual(
                core.describe_anchor_expr("w:mon,wed,fri + y:apr"),
                "Mondays, Wednesdays, or Fridays in Apr each year",
            )
        finally:
            core.ANCHOR_PRESETS = previous_presets

    def test_expression_characterization_matrix(self) -> None:
        cases = (
            (
                "w:mon,wed,fri + y:apr",
                [[("w", "mon", 1, None, False, 0), ("y", "04-01..04-31", 1, None, False, 0)],
                 [("w", "wed", 1, None, False, 0), ("y", "04-01..04-31", 1, None, False, 0)],
                 [("w", "fri", 1, None, False, 0), ("y", "04-01..04-31", 1, None, False, 0)]],
                "Mondays, Wednesdays, or Fridays in Apr each year",
                ("2026-04-01", "2026-04-03", "2026-04-06", "2026-04-08", "2026-04-10", "2026-04-13"),
                None,
            ),
            (
                "w:mon | m:15",
                [[("w", "mon", 1, None, False, 0)], [("m", "15", 1, None, False, 0)]],
                "either Mondays or the 15th day of each month",
                ("2026-01-05", "2026-01-12", "2026-01-15", "2026-01-19", "2026-01-26", "2026-02-02"),
                None,
            ),
            ("m:last-fri", [[("m", "last-fri", 1, None, False, 0)]], "the last Friday of each month", ("2026-01-30", "2026-02-27", "2026-03-27", "2026-04-24", "2026-05-29", "2026-06-26"), None),
            ("m:-1bd", [[("m", "-1bd", 1, None, False, 0)]], "the last business day of each month", ("2026-01-30", "2026-02-27", "2026-03-31", "2026-04-30", "2026-05-29", "2026-06-30"), None),
            ("y:02-29", [[("y", "02-29", 1, None, False, 0)]], "Feb 29 each leap year", ("2028-02-29", "2032-02-29", "2036-02-29"), None),
            ("w/2:fri", [[("w", "fri", 2, None, False, 0)]], "every 2 weeks: Fridays", ("2026-01-02", "2026-01-16", "2026-01-30", "2026-02-13", "2026-02-27", "2026-03-13"), None),
            ("w:mon..fri@t=09:00", [[("w", "mon..fri", 1, (9, 0), False, 0)]], "Mondays through Fridays at 09:00", ("2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"), None),
            ("y:rand + w:sat", [[("y", "rand", 1, None, False, 0), ("w", "sat", 1, None, False, 0)]], "one random Saturday each year", None, lambda day: day.weekday() == 5),
            ("m:rand + y:apr", [[("m", "rand", 1, None, False, 0), ("y", "04-01..04-31", 1, None, False, 0)]], "one random day each month and within Apr each year", None, lambda day: day.month == 4),
        )

        def signature(dnf):
            return [
                [
                    (
                        atom.get("typ"),
                        atom.get("spec"),
                        atom.get("ival"),
                        (atom.get("mods") or {}).get("t"),
                        bool((atom.get("mods") or {}).get("bd")),
                        (atom.get("mods") or {}).get("day_offset"),
                    )
                    for atom in term
                ]
                for term in dnf
            ]

        start = date(2026, 1, 1)
        for expression, expected_terms, natural, expected_dates, is_valid in cases:
            with self.subTest(expression=expression):
                parsed = core.parse_anchor_expr_to_dnf_cached(expression)
                validated = core.validate_anchor_expr_strict(expression)
                revalidated = core.validate_anchor_expr_strict(parsed)
                self.assertEqual(signature(parsed), expected_terms)
                self.assertEqual(signature(validated), expected_terms)
                self.assertEqual(signature(revalidated), expected_terms)
                self.assertEqual(core.describe_anchor_expr(expression), natural)
                cursor = start
                scheduled = []
                count = len(expected_dates) if expected_dates is not None else 3
                for _ in range(count):
                    cursor, _meta = core.next_after_expr(
                        validated,
                        cursor,
                        default_seed=start,
                        seed_base="anchor-characterization-v1",
                    )
                    self.assertIsNotNone(cursor)
                    scheduled.append(cursor.isoformat())
                if expected_dates is not None:
                    self.assertEqual(tuple(scheduled), expected_dates)
                else:
                    self.assertTrue(all(is_valid(date.fromisoformat(value)) for value in scheduled))

    def test_deeply_nested_parser_input_fails_with_parse_error(self) -> None:
        expression = "(" * 64 + "w:mon" + ")" * 64
        with self.assertRaisesRegex(core.ParseError, "(?i)nesting too deep"):
            core.parse_anchor_expr_to_dnf_cached(expression)

    def test_legacy_tuple_error_payload_is_rejected_as_parse_error(self) -> None:
        with self.assertRaises(core.ParseError) as ctx:
            core.validate_anchor_expr_strict(("legacy parser error", None))
        self.assertEqual(str(ctx.exception), "legacy parser error")

    def test_seeded_random_anchor_repeats_the_same_occurrence(self) -> None:
        dnf = core.parse_anchor_expr_to_dnf_cached("m:rand")
        after = date(2025, 1, 1)
        first, _first_meta = core.next_after_expr(dnf, after, seed_base="test-seed")
        replay, _replay_meta = core.next_after_expr(dnf, after, seed_base="test-seed")
        self.assertEqual(first, replay)

    def test_next_after_preserves_branch_metadata_contracts(self) -> None:
        dnf = core.parse_anchor_expr_to_dnf_cached("w:mon")
        occurrence, metadata = core.next_after_expr(dnf, date(2024, 12, 11))
        self.assertEqual(occurrence, date(2024, 12, 16))
        self.assertEqual(metadata["basis"], "simple_weekly")

        dnf = core.parse_anchor_expr_to_dnf_cached("w:mon + m:1")
        occurrence, metadata = core.next_after_expr(dnf, date(2024, 12, 1))
        self.assertEqual(occurrence, date(2025, 9, 1))
        self.assertEqual(metadata["basis"], "term")

        for expression, expected_month, expected_pattern in (
            ("m:rand", None, r"\d{6}"),
            ("y:rand-07", 7, r"\d{4}-\d{2}"),
            ("m:rand + y:01-01..01-31", 1, None),
        ):
            with self.subTest(expression=expression):
                dnf = core.parse_anchor_expr_to_dnf_cached(expression)
                occurrence, metadata = core.next_after_expr(
                    dnf, date(2025, 1, 1), seed_base="branch-seed"
                )
                self.assertEqual(metadata["basis"], "rand" if expected_pattern else "rand+yearly")
                if expected_pattern:
                    self.assertRegex(metadata["rand_period"], expected_pattern)
                if expected_month is not None:
                    self.assertEqual(occurrence.month, expected_month)
                self.assertGreater(occurrence, date(2025, 1, 1))


if __name__ == "__main__":
    unittest.main()
