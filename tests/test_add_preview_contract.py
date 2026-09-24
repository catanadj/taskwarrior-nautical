import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from nautical_core import add_anchor_compute, add_anchor_preview, add_preview_composition, anchor_omit
from nautical_core.occurrence_provider import Occurrence, OccurrenceBatch
from nautical_core.scheduler_models import OccurrenceSearchExhausted


UTC = timezone.utc


class _Host:
    _MAX_PREVIEW_ITERATIONS = 20
    _MAX_ITERATIONS = 20

    def __init__(self):
        self.core = SimpleNamespace(
            to_local=lambda value: value,
            build_local_datetime=lambda day, hhmm: datetime(
                day.year, day.month, day.day, hhmm[0], hhmm[1], tzinfo=UTC
            ),
            fmt_dt_local=lambda value: value.strftime("%Y-%m-%d %H:%M"),
            cp_sequence_interval_for_token=lambda token, **kwargs: token.get("td", timedelta()),
            _import_sibling=lambda name: SimpleNamespace(
                compare_datetimes=lambda left, right: (left > right) - (left < right)
            ) if name == "timeutil" else None,
        )

    def _human_delta(self, *_args):
        return "tomorrow"

    def _fmt_cp_interval_token(self, td):
        return str(td)


class AddPreviewCompositionTests(unittest.TestCase):
    def test_omit_natural_text_propagates_unexpected_conversion_failures(self):
        class FailingParser:
            @staticmethod
            def resolve_omit_presets(_value):
                raise RuntimeError("omit configuration failed")

        core = SimpleNamespace(
            _parser_api=FailingParser(),
            _import_sibling=lambda _name: SimpleNamespace(normalize_omit_expr=lambda value: value),
            describe_anchor_expr=lambda value: value,
        )
        with self.assertRaisesRegex(RuntimeError, "omit configuration failed"):
            add_anchor_preview._anchor_omit_natural_text({"omit": "w:sun"}, core=core)

    def test_omit_natural_text_treats_expected_description_errors_as_empty(self):
        core = SimpleNamespace(
            _parser_api=SimpleNamespace(resolve_omit_presets=lambda value: value),
            _import_sibling=lambda _name: SimpleNamespace(normalize_omit_expr=lambda value: value),
            describe_anchor_expr=lambda _value: (_ for _ in ()).throw(ValueError("invalid omit")),
        )
        self.assertEqual(
            add_anchor_preview._anchor_omit_natural_text({"omit": "w:sun"}, core=core),
            "w:sun",
        )

    def test_compact_anchor_preview_requests_only_its_first_occurrence(self):
        self.assertEqual(add_anchor_preview._initial_occurrence_limit(200, True), 1)
        self.assertEqual(add_anchor_preview._initial_occurrence_limit(3, False), 19)

    def test_timezone_fallback_warning_requires_a_timed_or_recurrence_source(self):
        core = SimpleNamespace(_LOCAL_TZ=None)
        self.assertTrue(
            add_anchor_preview._timezone_fallback_warning_needed(
                core, "w:mon@t=09:00", ""
            )
        )
        self.assertTrue(
            add_anchor_preview._timezone_fallback_warning_needed(
                core, "", "calendar.csv@t=09:00"
            )
        )
        self.assertTrue(
            add_anchor_preview._timezone_fallback_warning_needed(core, "w:mon", "")
        )
        self.assertFalse(add_anchor_preview._timezone_fallback_warning_needed(core, "", ""))
        core._LOCAL_TZ = object()
        self.assertFalse(
            add_anchor_preview._timezone_fallback_warning_needed(
                core, "w:mon@t=09:00", ""
            )
        )

    def test_daily_period_preserves_local_clock_and_sequence_preview(self):
        host = _Host()
        start = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
        add_day = add_preview_composition.cp_add_period_builder(host, timedelta(days=1))
        self.assertEqual(add_day(start), datetime(2026, 1, 2, 9, 30, tzinfo=UTC))
        self.assertEqual(add_preview_composition.cp_preview_lines(host, start, None, 2, add_day), [
            "[bright_cyan]2026-01-02 09:30[/bright_cyan]",
            "[cyan]2026-01-03 09:30[/cyan]",
        ])

    def test_until_summary_honors_date_limit(self):
        host = _Host()
        start = datetime(2026, 1, 1, 9, tzinfo=UTC)
        add_day = lambda value: value + timedelta(days=1)
        count, last = add_preview_composition.cp_until_summary(
            host, start, datetime(2026, 1, 3, 9, tzinfo=UTC), add_day
        )
        self.assertEqual(count, 2)
        self.assertEqual(last, datetime(2026, 1, 3, 9, tzinfo=UTC))

    def test_sequence_period_uses_zero_duration_for_unresolved_token(self):
        # The parser normally rejects this input.  At this lower boundary an
        # unresolved provider token has the stable no-op duration contract.
        self.assertEqual(
            add_preview_composition.cp_sequence_period_for_link(_Host(), [{}], "bad", 1),
            timedelta(),
        )

    def test_limit_rows_reports_minimum_cap_and_until(self):
        host = _Host()
        rows = []
        due = datetime(2026, 1, 1, 9, tzinfo=UTC)
        add_preview_composition.cp_limit_rows(
            host, rows, cpmax=4, due_dt=due,
            until_dt=datetime(2026, 1, 3, 9, tzinfo=UTC), exact_until_count=2,
            final_until_dt=datetime(2026, 1, 3, 9, tzinfo=UTC), add_period=lambda value: value,
            now_utc=due,
        )
        self.assertEqual(rows[0], ("Chain cap", "[bold yellow]#4[/]"))
        self.assertEqual(rows[-1], ("Future links", "[white]2[/]"))


class AddAnchorComputeTests(unittest.TestCase):
    def test_anchor_step_preserves_scheduler_exhaustion_identity(self) -> None:
        expected = OccurrenceSearchExhausted(
            "test adapter", reference=date(2026, 1, 1), limit=1
        )

        class FakeOmit:
            @staticmethod
            def next_after_expr_with_omit(*_args, **_kwargs):
                raise expected

        class FakeCore:
            MAX_ANCHOR_ITER = 1

            @staticmethod
            def _import_sibling(_name):
                return FakeOmit

        with self.assertRaises(OccurrenceSearchExhausted) as raised:
            add_anchor_compute.anchor_step_once_with_omit(
                [],
                date(2026, 1, 1),
                date(2026, 1, 1),
                "test",
                omit_dnf=None,
                core=FakeCore(),
            )

        self.assertIs(raised.exception, expected)

    def test_anchor_step_propagates_unexpected_failures(self) -> None:
        class FailingOmit:
            @staticmethod
            def next_after_expr_with_omit(*_args, **_kwargs):
                raise RuntimeError("scheduler configuration failed")

        class FakeCore:
            MAX_ANCHOR_ITER = 1

            @staticmethod
            def _import_sibling(_name):
                return FailingOmit

        with self.assertRaisesRegex(RuntimeError, "scheduler configuration failed"):
            add_anchor_compute.anchor_step_once_with_omit(
                [], date(2026, 1, 1), date(2026, 1, 1), "test",
                omit_dnf=None, core=FakeCore(),
            )

    def test_anchor_term_propagates_unexpected_failures(self) -> None:
        class FailingEngine:
            @staticmethod
            def factor_matches_on(*_args, **_kwargs):
                raise RuntimeError("factor evaluation failed")

        with self.assertRaisesRegex(RuntimeError, "factor evaluation failed"):
            add_anchor_compute.anchor_term_fires_on_date(
                [{}], date(2026, 1, 1), date(2026, 1, 1), "test",
                core=SimpleNamespace(_scheduler_api=FailingEngine()),
            )

    def test_anchor_expression_propagates_unexpected_failures(self) -> None:
        class FailingEngine:
            @staticmethod
            def dnf_has_counted_random(*_args, **_kwargs):
                raise RuntimeError("parser state failed")

        with self.assertRaisesRegex(RuntimeError, "parser state failed"):
            add_anchor_compute.anchor_expr_fires_on_date_with_omit(
                [], date(2026, 1, 1), date(2026, 1, 1), "test",
                omit_dnf=None,
                core=SimpleNamespace(
                    _scheduler_api=FailingEngine(),
                    _import_sibling=lambda _name: SimpleNamespace(
                        omit_expr_fires_on_date=lambda *_args, **_kwargs: False
                    ),
                ),
            )

    def test_anchor_counted_random_fallback_propagates_unexpected_failures(self) -> None:
        class FailingEngine:
            @staticmethod
            def factor_matches_on(*_args, **_kwargs):
                return False

            @staticmethod
            def dnf_has_counted_random(*_args, **_kwargs):
                return True

            @staticmethod
            def next_after_expr(*_args, **_kwargs):
                raise RuntimeError("counted random evaluation failed")

        with self.assertRaisesRegex(RuntimeError, "counted random evaluation failed"):
            add_anchor_compute.anchor_times_for_date(
                [[{}]], date(2026, 1, 1), date(2026, 1, 1), "test",
                core=SimpleNamespace(
                    _scheduler_api=FailingEngine(),
                    _import_sibling=lambda _name: SimpleNamespace(
                        omit_expr_fires_on_date=lambda *_args, **_kwargs: False
                    ),
                ),
                norm_t_mod=lambda _mods: [],
            )

    def test_omit_calendar_loader_propagates_unexpected_failures(self) -> None:
        class FailingCalendar:
            @staticmethod
            def active_business_calendar():
                raise RuntimeError("calendar configuration failed")

        core = SimpleNamespace(
            _scheduler_api=SimpleNamespace(),
            _import_sibling=lambda _name: FailingCalendar,
        )
        with self.assertRaisesRegex(RuntimeError, "calendar configuration failed"):
            anchor_omit._scheduler_business_calendar(core)

    def test_anchor_build_preview_formats_events_and_respects_until(self):
        first = datetime(2026, 1, 1, 9, tzinfo=UTC)
        events = OccurrenceBatch([
            Occurrence(date(2026, 1, 2), 9, 0, local_datetime=datetime(2026, 1, 2, 9, tzinfo=UTC)),
            Occurrence(date(2026, 1, 4), 9, 0, local_datetime=datetime(2026, 1, 4, 9, tzinfo=UTC)),
        ])
        evaluator = SimpleNamespace(collect_after=lambda *_args, **_kwargs: events)
        core = SimpleNamespace(fmt_dt_local=lambda value: value.strftime("%Y-%m-%d"))
        result = add_anchor_compute.anchor_build_preview(
            [], first, 4, datetime(2026, 1, 3, 9, tzinfo=UTC), (9, 0), date(2026, 1, 1), "chain",
            core=core, evaluator=evaluator,
        )
        self.assertEqual(result, ["[bright_cyan]2026-01-02[/bright_cyan]"])
        self.assertIsNone(result.terminal)

    def test_anchor_until_summary_reports_count_and_last_event(self):
        start = datetime(2026, 1, 1, 9, tzinfo=UTC)
        last = datetime(2026, 1, 3, 9, tzinfo=UTC)
        events = OccurrenceBatch([
            Occurrence(date(2026, 1, 1), 9, 0, local_datetime=start),
            Occurrence(date(2026, 1, 3), 9, 0, local_datetime=last),
        ])
        evaluator = SimpleNamespace(
            events_between=lambda *_args, **_kwargs: events,
        )
        core = SimpleNamespace(build_local_datetime=lambda day, hhmm: datetime(*day.timetuple()[:3], *hhmm, tzinfo=UTC))
        count, actual_last = add_anchor_compute.anchor_until_summary(
            [], datetime(2026, 1, 4, tzinfo=UTC), date(2026, 1, 1), (9, 0), date(2026, 1, 1),
            "chain", core=core, to_local_cached=lambda value: value, max_iterations=10,
            evaluator=evaluator,
        )
        self.assertEqual((count, actual_last), (1, last))

    def test_anchor_until_summary_requires_evaluator_dependency(self):
        with self.assertRaisesRegex(TypeError, "requires the evaluator"):
            add_anchor_compute.anchor_until_summary(
                [], datetime(2026, 1, 2, tzinfo=UTC), date(2026, 1, 1), (9, 0), date(2026, 1, 1),
                "chain", core=SimpleNamespace(), to_local_cached=lambda value: value, max_iterations=2,
            )

    def test_anchor_build_preview_propagates_typed_provider_exhaustion(self):
        failure = OccurrenceSearchExhausted(
            "anchor preview", reference=date(9999, 12, 31), limit=3,
            kind=OccurrenceSearchExhausted.DATE_LIMIT,
        )
        evaluator = SimpleNamespace(collect_after=lambda *_args, **_kwargs: (_ for _ in ()).throw(failure))
        core = SimpleNamespace(fmt_dt_local=lambda value: value.isoformat())
        with self.assertRaises(OccurrenceSearchExhausted) as ctx:
            add_anchor_compute.anchor_build_preview(
                [], datetime(2026, 1, 1, 9, tzinfo=UTC), 3, None, (9, 0),
                date(2026, 1, 1), "chain", core=core, evaluator=evaluator,
            )
        self.assertIs(ctx.exception, failure)

    def test_until_summary_preserves_typed_provider_exhaustion(self):
        failure = OccurrenceSearchExhausted(
            "anchor summary", reference=date(9999, 12, 31), limit=4,
            kind=OccurrenceSearchExhausted.DATE_LIMIT,
        )
        evaluator = SimpleNamespace(events_between=lambda *_args, **_kwargs: (_ for _ in ()).throw(failure))
        core = SimpleNamespace(
            build_local_datetime=lambda day, hhmm: datetime(
                day.year, day.month, day.day, hhmm[0], hhmm[1], tzinfo=UTC
            )
        )
        with self.assertRaises(OccurrenceSearchExhausted) as ctx:
            add_anchor_compute.anchor_until_summary(
                [], datetime(2026, 1, 2, tzinfo=UTC), date(2026, 1, 1), (9, 0),
                date(2026, 1, 1), "chain", core=core,
                to_local_cached=lambda value: value, max_iterations=4, evaluator=evaluator,
            )
        self.assertIs(ctx.exception, failure)

    def test_anchor_preview_filters_omitted_events_but_preserves_them_in_provider_stream(self):
        occurrences = [
            Occurrence(
                date(2026, 1, day),
                9,
                0,
                local_datetime=datetime(2026, 1, day, 9, tzinfo=UTC),
                omitted=day in {1, 2, 3},
            )
            for day in range(1, 6)
        ]
        batch = OccurrenceBatch(occurrences)
        scheduler_result = SimpleNamespace(occurrences=batch, terminal=None)
        scheduler_service = SimpleNamespace(
            session=SimpleNamespace(evaluator=SimpleNamespace(context=SimpleNamespace(timezone=UTC))),
            collect=lambda *_args, **_kwargs: scheduler_result,
        )
        kwargs = dict(
            after_local_dt=datetime(2026, 1, 1, tzinfo=UTC),
            inclusive=False, limit_included=2, fallback_hhmm=(9, 0),
            default_seed_date=date(2026, 1, 1),
            scheduler_service=scheduler_service,
        )
        stream = add_anchor_preview._collect_events_with_provider(**kwargs, return_occurrences=True)
        self.assertEqual(stream, occurrences)
        included_kwargs = {
            "after_local_dt": kwargs["after_local_dt"],
            "inclusive": kwargs["inclusive"],
            "limit": 2,
            "fallback_hhmm": kwargs["fallback_hhmm"],
            "default_seed_date": kwargs["default_seed_date"],
            "scheduler_service": scheduler_service,
        }
        included_only = add_anchor_preview._collect_included_with_provider(**included_kwargs)
        self.assertEqual(
            included_only,
            [occurrences[3].local_datetime, occurrences[4].local_datetime],
        )


class AddAnchorPreviewTests(unittest.TestCase):
    def test_seed_context_uses_entry_date_for_implicit_due(self):
        task = {"entry": "20260809T090000Z", "chainID": "stable-id"}
        now = datetime(2026, 9, 21, 10, tzinfo=UTC)
        result = add_anchor_preview.anchor_preview_seed_context(
            task, date(2026, 9, 21), now, False, root_uuid_from=lambda _task: "root"
        )
        self.assertEqual(result[:2], (date(2026, 8, 9), date(2026, 8, 9)))

    def test_seed_context_prefers_chain_identity_and_due_day(self):
        task = {"chainID": "  stable-id "}
        now = datetime(2026, 2, 5, 10, tzinfo=UTC)
        result = add_anchor_preview.anchor_preview_seed_context(
            task, date(2026, 2, 1), now, True, root_uuid_from=lambda _task: "root"
        )
        self.assertEqual(result, (date(2026, 2, 1), date(2026, 2, 1), "stable-id"))

    def test_prepare_dnf_records_pattern_and_mode(self):
        rows = []
        prof = SimpleNamespace(add_ms=lambda *_args: None)
        core = SimpleNamespace(describe_anchor_dnf=lambda *_args: "Every Monday")
        result = add_anchor_preview.anchor_preview_prepare_dnf(
            {}, "w:mon", datetime.now(UTC), rows, prof,
            core=core, validate_anchor_syntax_strict=lambda _value: ([[{"typ": "w"}]], None),
            validate_anchor_mode=lambda _value: ("skip", None), error_and_exit=lambda value: (_ for _ in ()).throw(AssertionError(value)),
        )
        self.assertEqual(result[1], "skip")
        self.assertEqual(rows[0][0], "Pattern")
        self.assertEqual(rows[1], ("Natural", "[white]Every Monday[/]"))

    def test_prepare_dnf_uses_injected_error_for_malformed_input(self):
        with self.assertRaisesRegex(RuntimeError, "bad anchor"):
            add_anchor_preview.anchor_preview_prepare_dnf(
                {}, "bad", datetime.now(UTC), [], SimpleNamespace(add_ms=lambda *_args: None),
                core=SimpleNamespace(), validate_anchor_syntax_strict=lambda _value: (None, "bad anchor"),
                validate_anchor_mode=lambda _value: ("skip", None),
                error_and_exit=lambda value: (_ for _ in ()).throw(RuntimeError(value[0][1])),
            )


if __name__ == "__main__":
    unittest.main()
