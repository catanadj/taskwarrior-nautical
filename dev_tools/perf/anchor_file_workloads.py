"""Anchor-file provider benchmark workloads."""

from __future__ import annotations

import importlib
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path


def provider(rounds: int) -> float:
    anchor_files = importlib.import_module("nautical_core.anchor_files")
    with tempfile.TemporaryDirectory(prefix="nautical-perf-anchor-file-") as td:
        path = Path(td) / "calendar.csv"
        rows = ["date,description"]
        for index in range(365):
            item_date = date(2026, 1, 1) + timedelta(days=index)
            rows.append(f"{item_date.isoformat()},{index}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        occurrence_provider = anchor_files.AnchorFileOccurrenceProvider(
            "calendar.csv@t=09:00", td, (9, 0)
        )
        build = lambda day, hhmm: datetime(day.year, day.month, day.day, *hhmm)
        identity = lambda value: value
        started = time.perf_counter()
        for index in range(max(1, rounds)):
            after = datetime(2026, 1, 1, 8, 0) + timedelta(days=index % 364)
            if occurrence_provider.next_after(after, build_local_datetime=build, to_local=identity) is None:
                raise RuntimeError("anchor-file provider benchmark unexpectedly exhausted")
        return time.perf_counter() - started


def batch_provider(rounds: int) -> float:
    anchor_files = importlib.import_module("nautical_core.anchor_files")
    occurrence_provider = importlib.import_module("nautical_core.occurrence_provider")
    with tempfile.TemporaryDirectory(prefix="nautical-perf-anchor-batch-") as td:
        path = Path(td) / "calendar.csv"
        rows = ["date,description"]
        for index in range(365):
            item_date = date(2026, 1, 1) + timedelta(days=index)
            rows.append(f"{item_date.isoformat()},{index}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        occurrence_provider_instance = anchor_files.AnchorFileOccurrenceProvider(
            "calendar.csv@t=09:00", td, (9, 0)
        )
        build = lambda day, hhmm: datetime(day.year, day.month, day.day, *hhmm)
        identity = lambda value: value
        started = time.perf_counter()
        for index in range(max(1, rounds)):
            after = datetime(2026, 1, 1, 8, 0) + timedelta(days=index % 350)
            batch = occurrence_provider.collect_after(
                occurrence_provider_instance,
                after,
                limit=5,
                build_local_datetime=build,
                to_local=identity,
                require_contract=True,
            )
            if not batch:
                raise RuntimeError("anchor-file batch benchmark unexpectedly returned no occurrences")
        return time.perf_counter() - started


def large_provider(
    rounds: int,
    *,
    row_count: int = 5000,
    mode: str = "hot",
    business_day_only: bool = False,
) -> float:
    """Exercise large file-backed calendars and cached lookup cursors."""
    anchor_files = importlib.import_module("nautical_core.anchor_files")
    business_calendar = importlib.import_module("nautical_core.business_calendar")
    row_count = max(1000, int(row_count))
    with tempfile.TemporaryDirectory(prefix="nautical-perf-anchor-large-") as td:
        path = Path(td) / "calendar.csv"
        rows = ["date,description"]
        first_day = date(2020, 1, 1)
        for index in range(row_count):
            item_date = first_day + timedelta(days=index)
            rows.append(f"{item_date.isoformat()},event-{index}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        suffix = "@bd" if business_day_only else ""
        provider_name = f"calendar.csv{suffix}"
        calendar = business_calendar.DEFAULT_BUSINESS_CALENDAR if business_day_only else None
        build = lambda day, hhmm: datetime(day.year, day.month, day.day, *hhmm)
        identity = lambda value: value
        query_indexes = list(range(0, row_count - 2, max(1, row_count // 37)))
        if mode == "nonmonotonic":
            query_indexes = query_indexes[::2] + list(reversed(query_indexes[1::2]))
        started = time.perf_counter()
        provider_instance = None
        if mode != "cold":
            provider_instance = anchor_files.AnchorFileOccurrenceProvider(
                provider_name, td, (9, 0), business_calendar=calendar
            )
            if provider_instance.next_after(
                datetime.combine(first_day - timedelta(days=1), datetime.min.time()),
                build_local_datetime=build,
                to_local=identity,
            ) is None:
                raise RuntimeError("large anchor-file provider failed to load its first occurrence")
        for _ in range(max(1, int(rounds))):
            for query_index in query_indexes:
                current = provider_instance
                if mode == "cold":
                    current = anchor_files.AnchorFileOccurrenceProvider(
                        provider_name, td, (9, 0), business_calendar=calendar
                    )
                after = datetime.combine(first_day + timedelta(days=query_index), datetime.min.time())
                occurrence = current.next_after(
                    after, build_local_datetime=build, to_local=identity
                )
                if occurrence is None:
                    raise RuntimeError(
                        f"large anchor-file provider exhausted unexpectedly ({mode}, query={query_index})"
                    )
                if business_day_only and occurrence.day.weekday() >= 5:
                    raise RuntimeError("business-day anchor-file benchmark returned a weekend")
        return time.perf_counter() - started
