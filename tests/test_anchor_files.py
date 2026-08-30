import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from nautical_core import anchor_files


class AnchorFileOccurrenceCacheTests(unittest.TestCase):
    def _provider(self, directory: str) -> anchor_files.AnchorFileOccurrenceProvider:
        return anchor_files.AnchorFileOccurrenceProvider("calendar.csv", directory, (9, 0))

    @staticmethod
    def _build(day, hhmm):
        return datetime(day.year, day.month, day.day, *hhmm)

    def test_static_records_are_reused_across_cold_providers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "calendar.csv").write_text("date,description\n2026-08-24,review\n", encoding="utf-8")
            original = anchor_files._load_anchor_source_data
            with patch.object(anchor_files, "_load_anchor_source_data", wraps=original) as loader:
                first = self._provider(directory).occurrences()
                second = self._provider(directory).occurrences()
            self.assertEqual(first, second)
            self.assertEqual(loader.call_count, 1)

    def test_file_metadata_change_invalidates_static_record_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "calendar.csv")
            path.write_text("date,description\n2026-08-24,old\n", encoding="utf-8")
            self.assertEqual(self._provider(directory).occurrences()[0].description, "old")
            path.write_text("date,description\n2026-08-25,new\n", encoding="utf-8")
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            values = self._provider(directory).occurrences()
            self.assertEqual([(item.day, item.description) for item in values], [(values[0].day, "new")])


if __name__ == "__main__":
    unittest.main()
