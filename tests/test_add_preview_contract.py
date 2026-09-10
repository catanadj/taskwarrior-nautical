import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from nautical_core import add_anchor_compute, add_anchor_preview, add_preview_composition
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

    def test_sequence_period_rejects_empty_tokens(self):
        with self.assertRaises(ZeroDivisionError):
            add_preview_composition.cp_sequence_period_for_link(_Host(), [], "", 1)

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


class AddAnchorPreviewTests(unittest.TestCase):
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
