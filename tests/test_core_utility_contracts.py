from __future__ import annotations

from contextlib import redirect_stderr
from datetime import date
import io
import os
import unittest
from unittest.mock import patch

from nautical_core.common import coerce_int, sanitize_text, short_uuid
import nautical_core.scheduler_api as scheduler_api
from nautical_core.schedule_utils import weeks_between


class CoreUtilityContractTests(unittest.TestCase):
    def test_week_count_uses_iso_week_boundaries(self) -> None:
        self.assertEqual(weeks_between(date(2024, 12, 31), date(2025, 1, 1)), 0)
        self.assertEqual(weeks_between(date(2024, 12, 29), date(2024, 12, 30)), 1)

    def test_scheduler_api_does_not_own_a_week_count_forwarder(self) -> None:
        self.assertFalse(hasattr(scheduler_api, "_weeks_between"))

    def test_scheduler_api_does_not_own_a_roll_acceptance_forwarder(self) -> None:
        self.assertFalse(hasattr(scheduler_api, "_accept_roll_candidate"))

    def test_short_uuid_handles_invalid_and_short_values(self) -> None:
        self.assertEqual(short_uuid(None), "")
        self.assertEqual(short_uuid(1234), "")
        self.assertEqual(short_uuid("abcd"), "abcd")

    def test_integer_coercion_rejects_values_above_supported_bounds(self) -> None:
        too_large = 2**63
        self.assertEqual(coerce_int(too_large, default=7), 7)
        self.assertEqual(coerce_int(float(too_large), default=7), 7)

    def test_integer_coercion_does_not_hide_unexpected_string_conversion_failure(self) -> None:
        class BrokenString:
            def __str__(self) -> str:
                raise RuntimeError("conversion implementation failed")

        with self.assertRaisesRegex(RuntimeError, "conversion implementation failed"):
            coerce_int(BrokenString(), default=7)

    def test_sanitization_survives_optional_diagnostic_write_failure(self) -> None:
        class BrokenStderr(io.StringIO):
            def write(self, _text: str) -> int:
                raise OSError("stderr is unavailable")

        with patch.dict(os.environ, {"NAUTICAL_DIAG": "1"}), redirect_stderr(BrokenStderr()):
            self.assertEqual(sanitize_text("abcdef", max_len=3), "abc")

    def test_sanitization_does_not_hide_unexpected_diagnostic_failure(self) -> None:
        class BrokenStderr(io.StringIO):
            def write(self, _text: str) -> int:
                raise RuntimeError("diagnostic implementation failed")

        with patch.dict(os.environ, {"NAUTICAL_DIAG": "1"}), redirect_stderr(BrokenStderr()):
            with self.assertRaisesRegex(RuntimeError, "diagnostic implementation failed"):
                sanitize_text("abcdef", max_len=3)


if __name__ == "__main__":
    unittest.main()
