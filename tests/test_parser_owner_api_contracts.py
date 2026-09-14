"""Behavioral contracts for core-bound parser owner APIs."""

from __future__ import annotations

import unittest
from datetime import date

import nautical_core as core
from nautical_core import acf_api, description_aliases, expansion_api, quarter_api, satisfiability


class ParserOwnerApiContractTests(unittest.TestCase):
    def test_incompatible_moon_phases_are_rejected_by_both_public_parsers(self) -> None:
        for parser in (core.validate_anchor_expr_strict, core.parse_anchor_expr_to_dnf):
            with self.subTest(parser=parser.__name__):
                with self.assertRaisesRegex(Exception, "incompatible moon phases"):
                    parser("moon:full + moon:new")

    def test_acf_spec_normalization_bounds_input_and_rejects_unknown_types(self) -> None:
        self.assertEqual(core._normalize_spec_for_acf_cached("w", "mon", "MD"), "mon")
        self.assertIsNone(core._normalize_spec_for_acf_cached("w", "x" * 300, "MD"))
        self.assertIsNone(core._normalize_spec_for_acf_cached("q", "mon", "MD"))

    @classmethod
    def setUpClass(cls) -> None:
        cls.acf = acf_api.for_core(module=core)
        cls.expansion = expansion_api.for_core(module=core)
        cls.quarter = quarter_api.for_core(module=core)

    def test_acf_round_trip_rejects_checksum_corruption(self) -> None:
        packed = self.acf._build_acf_impl("w:mon")
        self.assertTrue(self.acf.is_valid_acf(packed))
        self.assertEqual(self.acf.acf_to_original_format(packed), "w:mon")
        self.assertFalse(self.acf.is_valid_acf(packed[:-1] + ("0" if packed[-1] != "0" else "1")))

    def test_expansion_produces_hand_checked_weekdays_and_month_days(self) -> None:
        self.assertEqual(self.expansion._weekly_spec_to_wset("mon..wed,fri"), {0, 1, 2, 4})
        self.assertEqual(self.expansion._doms_for_weekly_spec("mon", 2026, 1), {5, 12, 19, 26})
        self.assertEqual(self.expansion._doms_for_monthly_token("1..3", 2026, 1), {1, 2, 3})

    def test_quarter_api_recognizes_and_formats_adjacent_quarters(self) -> None:
        self.assertTrue(self.quarter._has_quarter_tokens("q1,q2"))
        self.assertFalse(self.quarter._has_quarter_tokens("jan,feb"))
        self.assertEqual(self.quarter._format_quarter_set([1, 2]), "Q1–Q2")

    def test_description_aliases_extract_short_udas(self) -> None:
        description, fields = description_aliases.parse_description_aliases(
            "test task a:(w:mon | w:fri) am:all"
        )

        self.assertEqual(description, "test task")
        self.assertEqual(fields, {"anchor": "(w:mon | w:fri)", "anchor_mode": "all"})

    def test_description_aliases_preserve_prose_and_reject_duplicates(self) -> None:
        prose, prose_fields = description_aliases.parse_description_aliases(
            "read a: book today"
        )
        self.assertEqual(prose, "read a: book today")
        self.assertFalse(prose_fields)

        description, fields = description_aliases.parse_description_aliases(
            "water a:water am:all"
        )
        self.assertEqual(description, "water a:water")
        self.assertEqual(fields, {"anchor_mode": "all"})

        with self.assertRaisesRegex(ValueError, "more than once"):
            description_aliases.parse_description_aliases(
                "test task a:w:mon a:w:fri"
            )

        cleared, clear_fields = description_aliases.parse_description_aliases("test am:")
        self.assertEqual(cleared, "test")
        self.assertEqual(clear_fields, {"anchor_mode": ""})
        with self.assertRaisesRegex(ValueError, "leave it empty"):
            description_aliases.parse_description_aliases("test am:-")

        note, note_fields = description_aliases.parse_description_aliases(
            "note a:book today"
        )
        self.assertEqual(note, "note a:book today")
        self.assertFalse(note_fields)

    def test_quarter_selector_modes_accept_supported_tokens_and_reject_ambiguity(self) -> None:
        selector = self.quarter._quarter_month_selector_mode
        self.assertEqual(selector([{"spec": "1bd"}]), "quarter_start")
        self.assertEqual(selector([{"spec": "1st-mon"}]), "quarter_start")
        self.assertEqual(selector([{"spec": "-1bd"}]), "quarter_end")
        self.assertEqual(selector([{"spec": "last-fri"}]), "quarter_end")

        cases = (
            ([{"spec": "rand"}], "cannot be combined with m:rand"),
            ([{"spec": "1,15"}], "require a single monthly selector token"),
            ([{"spec": "15"}], "ambiguous"),
            ([{"spec": "1bd"}, {"spec": "-1bd"}], "cannot be combined with multiple monthly atoms"),
        )
        for atoms, expected in cases:
            with self.subTest(atoms=atoms), self.assertRaisesRegex(core.ParseError, expected):
                selector(atoms)

    def test_quarter_rewrite_modes_preserve_specs_and_context_metadata(self) -> None:
        mode = self.quarter._term_quarter_rewrite_mode
        self.assertEqual(
            mode([{"typ": "y", "spec": "q4"}], [{"typ": "m", "spec": "-1bd"}]),
            "quarter_end",
        )
        self.assertEqual(
            mode([{"typ": "y", "spec": "q4s"}], [{"typ": "m", "spec": "-1bd"}]),
            "first_month",
        )

        qmap = {}
        self.assertEqual(
            self.quarter._rewrite_quarter_spec_mode("q4", "quarter_end", meta_out=qmap),
            "12-01..12-31",
        )
        self.assertEqual(qmap, {"12-01..12-31": "Q4 end month"})
        qmap = {}
        self.assertEqual(
            self.quarter._rewrite_quarter_spec_mode(
                "q1..q2", "first_month", meta_out=qmap
            ),
            "01-01..01-31,04-01..04-30",
        )
        self.assertEqual(
            qmap,
            {
                "01-01..01-31": "Q1 first month",
                "04-01..04-30": "Q2 first month",
            },
        )
        self.assertEqual(
            self.quarter._rewrite_quarter_spec_mode("q1s..q2s", "first_month"),
            "01-01..01-31,04-01..04-30",
        )

        dnf = [[{"typ": "m", "spec": "-1bd"}, {"typ": "y", "spec": "q4"}]]
        rewritten = self.quarter._rewrite_quarters_in_context(dnf)
        self.assertEqual(rewritten[0][1]["spec"], "12-01..12-31")
        self.assertEqual(
            rewritten[0][1]["_qmap"], {"12-01..12-31": "Q4 end month"}
        )

    def test_time_window_metadata_round_trips_and_resolves_partitioned_slots(self) -> None:
        from datetime import date
        from nautical_core.time_slots import resolve_time_slots_with_offsets

        dnf = core.parse_anchor_expr_to_dnf("w:mon@t=04:30..19:30/3")
        mods = dnf[0][0]["mods"]
        self.assertEqual(mods["time_window"], "04:30..19:30/3")
        self.assertEqual(mods["t"], [(4, 30), (12, 0), (19, 30)])

        overnight = core.parse_anchor_expr_to_dnf("w:mon@t=22:30..06:30/7")
        overnight_mods = overnight[0][0]["mods"]
        self.assertEqual(overnight_mods["time_window"], "22:30..06:30/7")
        self.assertEqual(
            overnight_mods["time_window_offsets"],
            [(0, 22, 30), (0, 23, 50), (1, 1, 10), (1, 2, 30), (1, 3, 50), (1, 5, 10), (1, 6, 30)],
        )
        self.assertEqual(
            resolve_time_slots_with_offsets(
                {"t": [(22, 30)], "time_window": "22:30..06:30/7"},
                date(2026, 8, 4),
            ),
            [(0, 22, 30), (0, 23, 50), (1, 1, 10), (1, 2, 30), (1, 3, 50), (1, 5, 10), (1, 6, 30)],
        )

        expression = "w:mon@t=06..18/3h@+1d@+15m"
        canonical = core.acf_to_original_format(core.build_acf(expression))
        self.assertIn("@t=06:00..18:00/3h", canonical)
        self.assertIn("@+1d", canonical)
        self.assertIn("@+15m", canonical)

    def test_random_time_metadata_resolves_from_consistent_chain_identity(self) -> None:
        from datetime import date
        from nautical_core.recurrence_context import RecurrenceContext
        from nautical_core.time_slots import resolve_time_slots_with_offsets

        dnf = core.parse_anchor_expr_to_dnf("w:mon@t=rand(06..18/3)")
        mods = dnf[0][0]["mods"]
        self.assertEqual(mods["time_random"], "rand(06:00..18:00/3)")
        self.assertIn(
            "@t=rand(06:00..18:00/3)",
            core.acf_to_original_format(core.build_acf("w:mon@t=rand(06..18/3)")),
        )
        resolved = resolve_time_slots_with_offsets(mods, date(2026, 8, 3), seed_base="chain-a")
        self.assertEqual(
            resolved,
            resolve_time_slots_with_offsets(
                mods, date(2026, 8, 3), context=RecurrenceContext(chain_id="chain-a")
            ),
        )
        self.assertEqual(len(resolved), 3)
        self.assertEqual(resolved, sorted(resolved))
        with self.assertRaisesRegex(ValueError, "Conflicting recurrence identities"):
            resolve_time_slots_with_offsets(
                mods,
                date(2026, 8, 3),
                seed_base="chain-a",
                context=RecurrenceContext(chain_id="chain-b"),
            )
        with self.assertRaisesRegex(ValueError, "chain ID"):
            RecurrenceContext(chain_id="")

    def test_time_list_and_composable_schedule_parser_metadata_round_trip(self) -> None:
        from nautical_core import anchor_files

        parsed = core.parse_anchor_expr_to_dnf("w:mon@t=9,12:30,18")
        self.assertEqual(parsed[0][0]["mods"]["t"], [(9, 0), (12, 30), (18, 0)])
        _name, file_modifiers = anchor_files.parse_anchor_file_spec(
            "events.csv@t=9,12:30,18"
        )
        self.assertEqual(file_modifiers["t"], [(9, 0), (12, 30), (18, 0)])

        expression = "w:mon@t=06..18/3h,22"
        schedule = core.parse_anchor_expr_to_dnf(expression)[0][0]["mods"]
        self.assertEqual(schedule["time_schedule"], "06:00..18:00/3h,22:00")
        self.assertEqual(schedule["t"], [(6, 0), (9, 0), (12, 0), (15, 0), (18, 0), (22, 0)])
        self.assertIn("@t=06:00..18:00/3h,22:00", core.acf_to_original_format(core.build_acf(expression)))
        self.assertIn("every 3h within 06:00–18:00 plus 22:00", core.describe_anchor_expr(expression))

        grouped = core.parse_anchor_expr_to_dnf(
            "(w:mon | w:fri)@t=06..12/2h,16..20/2h,22@+15m"
        )
        self.assertEqual(len(grouped), 2)
        for term in grouped:
            self.assertEqual(
                term[0]["mods"]["time_schedule"],
                "06:00..12:00/2h,16:00..20:00/2h,22:00",
            )
            self.assertEqual(term[0]["mods"]["time_offset_minutes"], 15)
        core.validate_anchor_expr_strict(
            "(w:mon | w:fri)@t=06..12/2h,16..20/2h,22@+15m"
        )
        self.assertIn(
            "@+15m",
            core.acf_to_original_format(
                core.build_acf("(w:mon | w:fri)@t=06..12/2h,16..20/2h,22@+15m")
            ),
        )
        with self.assertRaisesRegex(core.ParseError, "timed term"):
            core.parse_anchor_expr_to_dnf(
                "(w:mon@t=06..12/2h | w:fri)@t=16..20/2h"
            )

    def test_composable_schedule_parser_rejects_invalid_members(self) -> None:
        for expression, expected in (
            ("w:mon@t=06..18/3h,sunset", "numeric"),
            ("w:mon@t=06..18/3h,", "empty"),
            ("w:mon@t=06..18/3h,,22", "empty"),
        ):
            with self.subTest(expression=expression), self.assertRaisesRegex(
                core.ParseError, expected
            ):
                core.parse_anchor_expr_to_dnf(expression)

    def test_random_time_window_composition_requires_separate_anchor_branches(self) -> None:
        with self.assertRaisesRegex(core.ParseError, "separate anchor branches"):
            core.parse_anchor_expr_to_dnf("w:mon@t=rand(06..18),22")

    def test_time_window_natural_language_describes_bounds_not_slot_lists(self) -> None:
        descriptions = (
            ("w:mon..fri@t=06..17/3h", "every 3h within 06:00–17:00"),
            ("w:mon..fri@t=06..18/3", "3 evenly spaced times (every 6h) within 06:00–18:00"),
            ("w:mon..fri@t=06:00..18:01/4", "4 evenly spaced times (every ~4h)"),
            ("w:mon@t=22:30..06:30/7", "7 evenly spaced times (every 1h20m) within 22:30–06:30 next day"),
            ("w:mon@t=rand(06..18/3)", "3 deterministic random times, one per bucket, within 06:00–18:00"),
        )
        for expression, expected in descriptions:
            with self.subTest(expression=expression):
                self.assertIn(expected, core.describe_anchor_expr(expression))
        self.assertNotIn("06:00, 09:00", core.describe_anchor_expr(descriptions[0][0]))

    def test_cached_time_metadata_rejects_inconsistent_shapes(self) -> None:
        valid = [[{"typ": "w", "spec": "mon", "ival": 1, "mods": {
            "time_window": "06:00..17:00/3h",
            "t": [[6, 0], [9, 0], [12, 0], [15, 0]],
        }}]]
        self.assertEqual(
            core._normalize_dnf_cached(valid)[0][0]["mods"]["t"],
            [(6, 0), (9, 0), (12, 0), (15, 0)],
        )
        for value in (
            [[{"typ": "w", "spec": "mon", "ival": 1, "mods": {
                "time_window": "06:00..17:00/3h", "t": [[6, 0], [10, 0]],
            }}]],
            [[{"typ": "w", "spec": "mon", "ival": 1, "mods": {
                "time_random": "rand(06..18/3)", "t": [],
            }}]],
            [[{"typ": "w", "spec": "mon", "ival": 1, "mods": {
                "time_schedule": "06:00..12:00/2h,18:00",
                "t": [[6, 0], [8, 0], [10, 0], [18, 0]],
            }}]],
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                core._normalize_dnf_cached(value)

        with self.assertRaisesRegex(core.ParseError, "cannot be combined"):
            core.validate_anchor_expr_strict([[{"typ": "w", "spec": "mon", "ival": 1, "mods": {
                "time_random": "rand(06:00..18:00/3)",
                "time_window": "06:00..18:00/3h",
                "t": [],
            }}]])

    def test_cached_random_metadata_requires_canonical_spec_and_valid_offsets(self) -> None:
        valid = [[{"typ": "w", "spec": "mon", "ival": 1, "mods": {
            "time_random": "rand(06:00..18:00/3)", "t": [],
        }}]]
        core._normalize_dnf_cached(valid)
        invalid = [[{"typ": "w", "spec": "mon", "ival": 1, "mods": {
            "time_random": "rand(06..18/3)", "t": [],
        }}]]
        with self.assertRaisesRegex(ValueError, "random"):
            core._normalize_dnf_cached(invalid)

    def test_parser_rejects_term_explosion_before_cartesian_expansion(self) -> None:
        group = "(w:mon|w:tue|w:wed|w:thu|w:fri|w:sat)"
        expression = "+".join([group] * 6)

        with self.assertRaisesRegex(core.ParseError, "too complex"):
            core.parse_anchor_expr_to_dnf(expression)

    def test_parser_satisfiability_matches_scheduler_acceptance(self) -> None:
        from datetime import date

        start = date(2026, 1, 1)
        accepted = (
            ("w:mon + y:01-01", date(2029, 1, 1)),
            ("y:rand + w:sat", None),
            ("m:rand + y:02-29", date(2028, 2, 29)),
            ("m:5th-mon + y:02-29", date(2044, 2, 29)),
            ("m:last-mon + y:02-29", date(2044, 2, 29)),
        )
        for expression, expected in accepted:
            with self.subTest(expression=expression):
                dnf = core.parse_anchor_expr_to_dnf(expression)
                validated = core.validate_anchor_expr_strict(dnf)
                occurrence, _meta = core.next_after_expr(
                    validated,
                    start,
                    default_seed=start,
                    seed_base="satisfiability-agreement-v1",
                )
                self.assertIsNotNone(occurrence)
                self.assertGreater(occurrence, start)
                if expected is not None:
                    self.assertEqual(occurrence, expected)

        impossible = (
            "w:mon + w:sun",
            "y:01-01 + y:12-25",
            "m:31 + y:apr",
            "m:30 + y:feb",
        )
        for expression in impossible:
            with self.subTest(expression=expression), self.assertRaises(
                core.AndTermUnsatisfiable
            ):
                core.parse_anchor_expr_to_dnf(expression)

    def test_satisfiability_owner_rejects_disjoint_weekly_and_yearly_sets(self) -> None:
        with self.assertRaisesRegex(core.AndTermUnsatisfiable, "never coincide"):
            satisfiability.quick_weekly_and_check(
                [{"typ": "w", "spec": "sat"}, {"typ": "w", "spec": "mon"}],
                weekday_set_from_weekly_atom=lambda atom: {
                    "sat": {5}, "mon": {0}
                }[atom["spec"]],
                and_term_unsatisfiable_cls=core.AndTermUnsatisfiable,
            )

        yearly_sets = {"01-01": {(1, 1)}, "12-25": {(12, 25)}}
        with self.assertRaisesRegex(core.AndTermUnsatisfiable, "never overlap within a year"):
            satisfiability.quick_yearly_and_check(
                [{"typ": "y", "spec": "01-01"}, {"typ": "y", "spec": "12-25"}],
                md_pairs_from_yearly_spec=lambda spec: yearly_sets[spec],
                and_term_unsatisfiable_cls=core.AndTermUnsatisfiable,
            )

        self.assertTrue(
            satisfiability.term_has_any_match_within(
                [{"typ": "w", "spec": "mon"}, {"typ": "m", "spec": "1"}],
                date(2026, 1, 1),
                date(2026, 1, 1),
                atom_matches_on=lambda atom, day, _seed: day.weekday() == 0
                if atom["typ"] == "w"
                else day.day == 1,
                years=2,
            )
        )

    def test_quarter_parser_selectors_schedule_in_expected_months(self) -> None:
        from datetime import date

        start = date(2026, 1, 1)
        expected_months = (
            ("m:1bd + y:q4", 10),
            ("m:-1bd + y:q4", 12),
        )
        for expression, month in expected_months:
            with self.subTest(expression=expression):
                dnf = core.parse_anchor_expr_to_dnf(expression)
                occurrence, _meta = core.next_after_expr(
                    dnf,
                    start,
                    default_seed=start,
                    seed_base="quarter-month-selector-contract",
                )
                self.assertIsNotNone(occurrence)
                self.assertEqual(occurrence.month, month)

        with self.assertRaisesRegex(core.ParseError, "(?i)ambiguous.*quarter|quarter.*ambiguous"):
            core.parse_anchor_expr_to_dnf("m:2nd-mon + y:q1..q2")

    def test_astronomical_time_tokens_are_shared_by_parser_and_runtime(self) -> None:
        from nautical_core import astronomy

        for event in sorted(astronomy.EVENT_NAMES):
            with self.subTest(event=event):
                dnf = core.parse_anchor_expr_to_dnf(f"w:mon@t={event}")
                self.assertEqual(dnf[0][0]["mods"]["t"], event)
                self.assertTrue(astronomy.is_event_name(event))

        with self.assertRaises(core.ParseError):
            core.parse_anchor_expr_to_dnf("w:mon@t=not-an-event")

    def test_symbolic_and_numeric_time_modifiers_keep_typed_parser_values(self) -> None:
        for event in ("sunrise", "sunset", "dawn", "dusk", "moonrise", "moonset"):
            with self.subTest(event=event):
                dnf = core.validate_anchor_expr_strict(f"w:mon@t={event}")
                self.assertEqual(dnf[0][0]["mods"]["t"], event)
        numeric = core.validate_anchor_expr_strict("w:mon@t=09:00")
        self.assertEqual(numeric[0][0]["mods"]["t"], (9, 0))
        offset = core.validate_anchor_expr_strict("w:mon@t=dawn@-45m")
        self.assertEqual(offset[0][0]["mods"]["time_offset_minutes"], -45)

    def test_parser_entry_point_rejects_oversized_expression(self) -> None:
        expression = "w:mon" + ("+w:mon" * 300)
        with self.assertRaises(core.ParseError):
            core.parse_anchor_expr_to_dnf(expression)

    def test_interval_heads_parse_for_weekly_monthly_and_yearly_anchors(self) -> None:
        for expression in ("w/2:sun", "m/3:1st-mon", "y/4:06-01"):
            with self.subTest(expression=expression):
                self.assertTrue(core.validate_anchor_expr_strict(expression))

    def test_yearly_token_parser_handles_quarters_aliases_and_invalid_dates(self) -> None:
        expected = (
            ("q1", ("quarter", "q1")),
            ("q2s", ("quarter", "q2s")),
            ("01-06", ("day", (1, 6))),
            ("06-jan", ("day", (6, 1))),
        )
        for token, parsed in expected:
            with self.subTest(token=token):
                self.assertEqual(core._parse_y_token(token), parsed)
        self.assertIsNone(core._parse_y_token("13-01"))
        self.assertIsNone(core._parse_y_token("31-04"))

    def test_parser_rejects_commas_between_atoms_after_modifiers(self) -> None:
        with self.assertRaises(core.ParseError) as ctx:
            core.validate_anchor_expr_strict("m:31@t=14:00,w:sun@t=22:00")
        message = str(ctx.exception).lower()
        self.assertTrue("join" in message or "use '+' (and) or '|'" in message)

    def test_linter_reports_stable_validation_guidance(self) -> None:
        cases = (
            ('"w:mon-fri"', "Weekly ranges must use '..' (e.g., 'w:mon..fri')."),
            ("w:mon:fri", "Weekly ranges must use '..' (e.g., 'w:mon..fri')."),
            ("y:01-01:12-31", "Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."),
            ("y:05:15", "Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."),
            ("w:mno", "Unknown weekday 'mno'. Did you mean 'mon'?"),
            ("m:6th-mon", "Invalid ordinal '6th'. Only 1st..5th are supported."),
            (
                "w:sat + w:mon",
                "These anchors joined with '+' don't share any possible date. If you meant 'either/or', use '|'.",
            ),
            (
                "y:q4..q2",
                "Invalid quarter range 'qX..qY': end quarter precedes start quarter. Split across the year boundary, e.g., 'q4, q1'.",
            ),
        )
        for expression, expected in cases:
            with self.subTest(expression=expression):
                fatal, warnings = core.lint_anchor_expr(expression)
                self.assertEqual(fatal, expected)
                self.assertEqual(warnings, [])

        fatal, warnings = core.lint_anchor_expr("w:mon" + ("x" * 1100))
        self.assertEqual(fatal, "Anchor expression too long (max 1024 characters).")
        self.assertEqual(warnings, [])

        for expression in (
            "w:mon,wed,fri + y:apr",
            "m:1,15,-1bd + w:mon..fri",
            "y:apr,jul,oct + w:sat",
        ):
            with self.subTest(expression=expression):
                fatal, warnings = core.lint_anchor_expr(expression)
                self.assertIsNone(fatal)
                self.assertEqual(warnings, [])

    def test_month_name_and_unsatisfiable_hints_use_yearly_alias(self) -> None:
        for expression, expected in (
            ("m:jan", "Month names belong to yearly anchors."),
            ("w:wed + m:apr", "Example: w:wed | y:apr"),
        ):
            with self.subTest(expression=expression), self.assertRaises(core.ParseError) as ctx:
                core.validate_anchor_expr_strict(expression)
            self.assertIn(expected, str(ctx.exception))
        with self.assertRaises(core.ParseError) as ctx:
            core.validate_anchor_expr_strict("m:jan")
        self.assertIn("Use 'y:jan'", str(ctx.exception))

    def test_grouped_weekday_lists_keep_filters_on_every_branch(self) -> None:
        from datetime import date

        dnf = core.validate_anchor_expr_strict("w:mon,wed,fri + y:apr")
        self.assertEqual(len(dnf), 3)
        self.assertEqual(
            core.next_after_expr(
                dnf,
                date(2026, 4, 12),
                default_seed=date(2026, 4, 11),
                seed_base="anchor-test",
            )[0],
            date(2026, 4, 13),
        )
        self.assertEqual(
            core.next_after_expr(
                dnf,
                date(2026, 4, 30),
                default_seed=date(2026, 4, 11),
                seed_base="anchor-test",
            )[0],
            date(2027, 4, 2),
        )

    def test_weekday_time_lists_parse_to_single_time_branches(self) -> None:
        expression = "w:mon@t=09:00,fri@t=15:00"
        fatal, warnings = core.lint_anchor_expr(expression)
        self.assertIsNone(fatal)
        self.assertEqual(warnings, [])
        dnf = core.validate_anchor_expr_strict(expression)
        self.assertEqual(
            [(term[0]["spec"], term[0]["mods"]["t"]) for term in dnf],
            [("mon", (9, 0)), ("fri", (15, 0))],
        )

    def test_trailing_weekday_time_applies_to_every_list_item(self) -> None:
        from datetime import date

        dnf = core.parse_anchor_expr_to_dnf("w:mon,wed,fri@t=05:00")
        self.assertEqual([term[0]["spec"] for term in dnf], ["mon", "wed", "fri"])
        self.assertTrue(all(term[0]["mods"]["t"] == (5, 0) for term in dnf))
        seed = date(2026, 7, 7)
        cursor = seed
        slots = []
        for _ in range(4):
            occurrence, _meta = core.next_after_expr(
                dnf, cursor, default_seed=seed, seed_base="weekly-list-trailing-time-test"
            )
            self.assertIsNotNone(occurrence)
            slots.append((occurrence, core.pick_hhmm_from_dnf_for_date(
                dnf, occurrence, seed, seed_base="weekly-list-trailing-time-test"
            )))
            cursor = occurrence
        self.assertEqual(
            slots,
            [
                (date(2026, 7, 8), (5, 0)),
                (date(2026, 7, 10), (5, 0)),
                (date(2026, 7, 13), (5, 0)),
                (date(2026, 7, 15), (5, 0)),
            ],
        )

    def test_group_time_modifier_distributes_and_survives_acf_round_trip(self) -> None:
        expression = "(w:mon | m:last-fri)@t=09:00"
        dnf = core.parse_anchor_expr_to_dnf(expression)
        self.assertEqual(len(dnf), 2)
        self.assertTrue(
            all(atom["mods"]["t"] == (9, 0) for term in dnf for atom in term)
        )
        self.assertIn("09:00", core.describe_anchor_expr(expression))
        canonical = core.acf_to_original_format(core.build_acf(expression))
        reparsed = core.parse_anchor_expr_to_dnf(canonical)
        self.assertTrue(
            all(atom["mods"]["t"] == (9, 0) for term in reparsed for atom in term)
        )

    def test_group_time_modifier_preserves_multiple_clock_times(self) -> None:
        dnf = core.parse_anchor_expr_to_dnf("(w:mon | w:fri)@t=09:00,17:30")
        self.assertTrue(
            all(atom["mods"]["t"] == [(9, 0), (17, 30)] for term in dnf for atom in term)
        )

    def test_group_astronomical_time_and_offset_apply_to_every_atom(self) -> None:
        dnf = core.parse_anchor_expr_to_dnf("(moon:full + y:jul)@t=sunset@+45m")
        self.assertEqual(len(dnf), 1)
        self.assertEqual(len(dnf[0]), 2)
        self.assertTrue(
            all(
                atom["mods"]["t"] == "sunset"
                and atom["mods"]["time_offset_minutes"] == 45
                for atom in dnf[0]
            )
        )

    def test_grouped_date_modifiers_schedule_and_round_trip(self) -> None:
        from datetime import date

        expression = "(y:12-24 | (y:12-30 | y:12-31))@pbd@-1bd@t=09:00"
        dnf = core.validate_anchor_expr_strict(expression)
        self.assertEqual(len(dnf), 3)
        for term in dnf:
            mods = term[0]["mods"]
            self.assertEqual(mods["roll"], "pbd")
            self.assertEqual(mods["business_day_offset"], -1)
            self.assertEqual(mods["t"], (9, 0))
        self.assertEqual(
            core.next_after_expr(dnf, date(2026, 12, 1))[0], date(2026, 12, 23)
        )
        description = core.describe_anchor_expr(expression)
        self.assertIn("previous business day", description)
        self.assertIn("1 business day earlier", description)
        canonical = core.acf_to_original_format(core.build_acf(expression))
        self.assertEqual(core.parse_anchor_expr_to_dnf(canonical), dnf)

        all_mods = core.parse_anchor_expr_to_dnf(
            "(y:04-24 | y:04-30)@nbd@bd@+1d@-2bd@t=09:00"
        )
        self.assertTrue(all(term[0]["mods"]["bd"] for term in all_mods))
        cancelled = core.parse_anchor_expr_to_dnf("(w:mon | w:fri)@+1d@-1d")
        self.assertTrue(all(term[0]["mods"]["day_offset"] == 0 for term in cancelled))

    def test_grouped_modifiers_reject_semantically_ambiguous_inputs(self) -> None:
        cases = (
            ("(w:mon@t=10:00 | w:fri)@t=09:00", "already has"),
            ("(m:1@pbd | m:15)@nbd", "already inside"),
            ("(m:1 + w:sun)@nbd", "OR-only"),
            ("(w:mon)@", "empty"),
        )
        for expression, message in cases:
            with self.subTest(expression=expression), self.assertRaises(core.ParseError) as ctx:
                core.parse_anchor_expr_to_dnf(expression)
            self.assertIn(message, str(ctx.exception))

    def test_expansion_helpers_cover_business_and_yearly_ranges(self) -> None:
        self.assertEqual(
            core._weekly_spec_to_wset("rand", mods={"bd": True}),
            {0, 1, 2, 3, 4},
        )
        self.assertEqual(
            core._y_ranges_from_spec("rand-07,01-10..01-12"),
            [(7, 1, 7, 31), (1, 10, 1, 12)],
        )
        self.assertEqual(
            core._doms_allowed_by_year(2026, 7, ["rand-07"]),
            set(range(1, 32)),
        )

    def test_monthly_expansion_helpers_resolve_valid_occurrence_months(self) -> None:
        from datetime import date

        self.assertEqual(core._doms_for_monthly_token("last-fri", 2026, 1), {30})
        self.assertFalse(core._month_has_hit("5th-mon", 2026, 2))
        self.assertTrue(core._month_has_hit("5th-mon", 2026, 3))
        self.assertEqual(core._next_valid_month_on_or_after("5th-mon", 2026, 2), (2026, 3))
        self.assertEqual(
            core._first_hit_after_probe_in_month("5th-mon", 2026, 3, date(2026, 3, 1)),
            date(2026, 3, 30),
        )

    def test_leap_day_anchor_schedules_only_leap_days(self) -> None:
        from datetime import date

        dnf = core.validate_anchor_expr_strict("y:02-29")
        seed = date(2025, 1, 1)
        occurrences = []
        cursor = seed
        for _ in range(8):
            occurrence, _meta = core.next_after_expr(
                dnf,
                cursor,
                default_seed=seed,
                seed_base="leap-day-contract",
            )
            self.assertIsNotNone(occurrence)
            occurrences.append(occurrence)
            cursor = occurrence
        self.assertTrue(occurrences)
        self.assertTrue(all((day.month, day.day) == (2, 29) for day in occurrences))
        self.assertEqual(occurrences[0], date(2028, 2, 29))


if __name__ == "__main__":
    unittest.main()
