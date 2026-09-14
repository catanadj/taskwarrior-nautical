"""Direct contracts for completion analytics and chain integrity warnings."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest

from nautical_core.modify_analytics import chain_health_advice, chain_integrity_warnings
from nautical_core.modify_value_effects import format_delta


def parse_utc(value: object) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(str(value), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


class ModifyAnalyticsContractTests(unittest.TestCase):
    def test_chain_integrity_warnings_report_gaps_and_missing_identity(self) -> None:
        chain = [
            {
                "uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "link": 1,
                "nextLink": "bbbbbbbb",
                "chainID": "cid",
            },
            {
                "uuid": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "link": 3,
                "prevLink": "aaaaaaaa",
                "chainID": "",
            },
        ]
        warnings = chain_integrity_warnings(
            chain,
            expected_chain_id="cid",
            coerce_int=lambda value, default: int(value) if value is not None else default,
            short=lambda value: str(value or "")[:8],
        )
        self.assertTrue(any("missing link(s): 2" in item for item in warnings))
        self.assertTrue(any("missing chainID" in item for item in warnings))

    def test_coach_advice_reports_healthy_on_time_streak(self) -> None:
        chain = [
            {"uuid": "a", "link": 1, "status": "completed", "due": "20250101T090000Z", "end": "20250101T090500Z"},
            {"uuid": "b", "link": 2, "status": "completed", "due": "20250104T090000Z", "end": "20250104T091000Z"},
            {"uuid": "c", "link": 3, "status": "completed", "due": "20250107T090000Z", "end": "20250107T090800Z"},
            {"uuid": "d", "link": 4, "status": "pending", "due": "20250110T090000Z"},
        ]
        core = SimpleNamespace(
            cp_sequence_interval_for_link=lambda *_args: timedelta(days=3)
        )
        advice = chain_health_advice(
            chain, "cp", {"cp": "3d", "chainID": "cid", "link": 4},
            core=core, parse_datetime=parse_utc, format_delta=format_delta,
            coerce_int=lambda value, default: int(value) if value is not None else default,
            tol_secs=3600, style="coach",
        )
        self.assertEqual(
            advice,
            "Chain looks healthy with a 3-link on-time streak; keep the current cadence.",
        )

    def test_coach_advice_identifies_low_on_time_rate(self) -> None:
        chain = [
            {"uuid": "a", "link": 1, "status": "completed", "due": "20250101T090000Z", "end": "20250102T120000Z"},
            {"uuid": "b", "link": 2, "status": "completed", "due": "20250102T090000Z", "end": "20250103T140000Z"},
            {"uuid": "c", "link": 3, "status": "completed", "due": "20250103T090000Z", "end": "20250103T093000Z"},
            {"uuid": "d", "link": 4, "status": "pending", "due": "20250105T090000Z"},
        ]
        core = SimpleNamespace(
            cp_sequence_interval_for_link=lambda *_args: timedelta(days=1)
        )
        advice = chain_health_advice(
            chain, "cp", {"cp": "1d", "chainID": "cid", "link": 4},
            core=core, parse_datetime=parse_utc, format_delta=format_delta,
            coerce_int=lambda value, default: int(value) if value is not None else default,
            tol_secs=3600, style="coach",
        )
        self.assertEqual(
            advice,
            "Chain needs attention (on-time rate is low); try smaller scopes or later due times.",
        )

    def test_clinical_advice_normalizes_style_and_reports_drift(self) -> None:
        chain = [
            {"uuid": "a", "link": 1, "status": "completed", "due": "20250101T090000Z", "end": "20250101T100000Z"},
            {"uuid": "b", "link": 2, "status": "completed", "due": "20250102T090000Z", "end": "20250102T090500Z"},
            {"uuid": "c", "link": 3, "status": "completed", "due": "20250103T090000Z", "end": "20250103T090500Z"},
            {"uuid": "d", "link": 4, "status": "completed", "due": "20250105T090000Z", "end": "20250105T090500Z"},
        ]
        advice = chain_health_advice(
            chain, "anchor", {}, core=object(), parse_datetime=parse_utc,
            format_delta=format_delta,
            coerce_int=lambda value, default: int(value) if value is not None else default,
            tol_secs=3600, style=" Clinical ",
        )
        self.assertEqual(advice, "OT 100% | Drift +1d 00h:00m | Streak 4 | Vol 0d 00h:23m")


if __name__ == "__main__":
    unittest.main()
