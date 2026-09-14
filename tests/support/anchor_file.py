"""Temporary recurrence-file fixture shared by tests and golden acceptance cases."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator
import importlib


@contextmanager
def canonical_anchor_file_fixture(
    *,
    count: int,
    omitted,
) -> Iterator[tuple[object, Path, date]]:
    """Yield a real anchor-file source and deterministic omission boundary."""
    first_date = date(2026, 1, 1)
    with TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "calendar.csv").write_text(
            "date\n" + "\n".join(
                (first_date + timedelta(days=index)).isoformat() for index in range(count)
            ) + "\n",
            encoding="utf-8",
        )
        core = type(
            "AnchorFileFixtureCore",
            (),
            {
                "ANCHOR_FILE_DIR": str(root),
                "build_local_datetime": staticmethod(lambda day, hhmm: datetime(day.year, day.month, day.day, *hhmm)),
                "to_local": staticmethod(lambda value: value),
                "_import_sibling": staticmethod(lambda name: importlib.import_module(f"nautical_core.{name}")),
            },
        )()
        inclusion = importlib.import_module("nautical_core.anchor_inclusion")
        original = inclusion._anchor_file_occurrence_is_omitted
        inclusion._anchor_file_occurrence_is_omitted = lambda item, **_kwargs: bool(omitted(item, first_date))
        try:
            yield core, root, first_date
        finally:
            inclusion._anchor_file_occurrence_is_omitted = original
