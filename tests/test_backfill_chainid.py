"""Deterministic tests for the chainID backfill planning helpers."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from dev_tools import nautical_backfill_chainid as backfill


def _uuid(prefix: str, suffix: str) -> str:
    """Build a stable UUID-shaped value whose short prefix is predictable."""
    return f"{prefix}-{suffix}-0000-0000-000000000000"


class BackfillChainIdTests(unittest.TestCase):
    def test_index_tasks_skips_missing_uuid_and_indexes_full_and_short_forms(self) -> None:
        first = _uuid("aaaaaaaa", "1111")
        second = _uuid("bbbbbbbb", "2222")
        by_full, by_short = backfill.index_tasks(
            [{"description": "missing"}, {"uuid": first}, {"uuid": second.upper()}]
        )

        self.assertEqual(set(by_full), {first, second})
        self.assertEqual(by_short["aaaaaaaa"], [first])
        self.assertEqual(by_short["bbbbbbbb"], [second])

    def test_resolve_uuid_uses_reciprocal_adjacency_for_ambiguous_short_uuid(self) -> None:
        current = _uuid("cccccccc", "3333")
        linked = _uuid("aaaaaaaa", "1111")
        unrelated = _uuid("aaaaaaaa", "2222")
        rows = [
            {"uuid": current, "prevLink": "aaaaaaaa"},
            {"uuid": linked, "nextLink": "cccccccc"},
            {"uuid": unrelated, "nextLink": "different"},
        ]
        by_full, by_short = backfill.index_tasks(rows)

        self.assertEqual(
            backfill.resolve_uuid(
                "aaaaaaaa", by_full, by_short,
                context_full=current, direction="prev",
            ),
            linked,
        )

    def test_resolve_uuid_leaves_unresolved_ambiguity_safe(self) -> None:
        current = _uuid("cccccccc", "3333")
        first = _uuid("aaaaaaaa", "1111")
        second = _uuid("aaaaaaaa", "2222")
        by_full, by_short = backfill.index_tasks(
            [{"uuid": current}, {"uuid": first}, {"uuid": second}]
        )

        self.assertIsNone(backfill.resolve_uuid("aaaaaaaa", by_full, by_short))
        self.assertIsNone(backfill.resolve_uuid("missing", by_full, by_short))

    def test_walk_chain_requires_reciprocal_links(self) -> None:
        first = _uuid("aaaaaaaa", "1111")
        middle = _uuid("bbbbbbbb", "2222")
        last = _uuid("cccccccc", "3333")
        bridge = _uuid("dddddddd", "4444")
        rows = [
            {"uuid": first, "nextLink": "bbbbbbbb"},
            {"uuid": middle, "prevLink": "aaaaaaaa", "nextLink": "cccccccc"},
            {"uuid": last, "prevLink": "bbbbbbbb"},
            # This task points at the chain but does not point back.
            {"uuid": bridge, "prevLink": "bbbbbbbb"},
        ]
        by_full, by_short = backfill.index_tasks(rows)

        walked = backfill.walk_chain(first, by_full, by_short)
        self.assertEqual({task["uuid"] for task in walked}, {first, middle, last})

    def test_select_root_prefers_empty_prev_then_entry(self) -> None:
        first = _uuid("aaaaaaaa", "1111")
        second = _uuid("bbbbbbbb", "2222")
        rows = [
            {"uuid": first, "entry": "20260102T000000Z", "prevLink": "missing"},
            {"uuid": second, "entry": "20260101T000000Z", "prevLink": ""},
        ]

        self.assertEqual(backfill.select_root_from_chain(rows), second)

    def test_main_dry_run_reports_only_filtered_candidate_chain_updates(self) -> None:
        first = _uuid("aaaaaaaa", "1111")
        second = _uuid("bbbbbbbb", "2222")
        rows = [
            {
                "uuid": first,
                "anchor": "RRULE:FREQ=DAILY",
                "project": "home",
                "chainID": "",
                "nextLink": "bbbbbbbb",
            },
            {
                "uuid": second,
                "cp": "20260101T000000Z",
                "project": "home",
                "chainID": "",
                "prevLink": "aaaaaaaa",
            },
            {
                "uuid": _uuid("eeeeeeee", "5555"),
                "anchor": "ignored",
                "project": "work",
            },
        ]
        output = io.StringIO()
        with patch.object(backfill, "export_all", return_value=rows), patch.object(
            backfill.sys, "argv", ["nautical_backfill_chainid.py", "--dry-run", "--only-project", "home"]
        ), redirect_stdout(output):
            backfill.main()

        text = output.getvalue()
        self.assertIn(f"DRY: task {first} modify chainID:aaaaaaaa", text)
        self.assertIn(f"DRY: task {second} modify chainID:aaaaaaaa", text)
        self.assertNotIn("eeeeeeee", text)


if __name__ == "__main__":
    unittest.main()
