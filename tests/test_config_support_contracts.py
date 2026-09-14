"""Direct contracts for configuration file selection and validation."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import tomllib
except ImportError:  # pragma: no cover - supported Python 3.10 fallback
    import tomli as tomllib

from nautical_core import config_support


class ConfigSupportContractTests(unittest.TestCase):
    def _read_result(self, path: str, *, error_sink=None):
        return config_support.read_toml_result(
            path,
            tomllib_mod=tomllib,
            warn_missing_toml_parser=lambda _path: None,
            warn_toml_parse_error=lambda _path, _err: None,
            error_sink=error_sink,
        )

    def test_malformed_discovered_file_returns_no_values_and_retains_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nautical.toml"
            path.write_text("tz = [\n", encoding="utf-8")
            errors: list[str] = []

            data = config_support.read_toml(
                str(path),
                tomllib_mod=tomllib,
                warn_missing_toml_parser=lambda _path: None,
                warn_toml_parse_error=lambda _path, _err: None,
                error_sink=errors.append,
            )

            self.assertEqual(data, {})
            self.assertTrue(errors)
            self.assertIn(str(path), errors[0])

    def test_world_writable_file_is_rejected_with_path_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nautical.toml"
            path.write_text('tz = "Pacific/Auckland"\n', encoding="utf-8")
            try:
                path.chmod(0o666)
            except OSError as exc:
                self.skipTest(f"file permissions cannot be changed here: {exc}")

            errors: list[str] = []
            data = config_support.read_toml(
                str(path),
                tomllib_mod=tomllib,
                warn_missing_toml_parser=lambda _path: None,
                warn_toml_parse_error=lambda _path, _err: None,
                error_sink=errors.append,
            )

            self.assertEqual(data, {})
            self.assertTrue(errors)
            self.assertIn(str(path), errors[0])
            self.assertIn("world-writable", errors[0])

    def test_empty_candidate_is_authoritative_and_invalid_candidate_blocks_fallback(self) -> None:
        defaults = {
            "wrand_salt": "default",
            "tz": "UTC",
            "holiday_region": "",
            "anchor_file_dir": "",
            "omit_file_dir": "",
            "anchor_presets": {},
            "omit_presets": {},
            "business_calendar": {},
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            os.environ.pop("NAUTICAL_CONFIG", None)
            high = Path(directory) / "high.toml"
            low = Path(directory) / "low.toml"
            low.write_text('tz = "Pacific/Auckland"\n', encoding="utf-8")

            high.write_text("", encoding="utf-8")
            selected_empty = config_support.load_config(
                defaults=defaults,
                config_paths=lambda: [str(high), str(low)],
                read_toml_result=self._read_result,
                normalize_keys=config_support.normalize_keys,
            )
            self.assertEqual(selected_empty["tz"], "UTC")

            errors: list[str] = []
            high.write_text("tz = [\n", encoding="utf-8")

            def read_invalid(path: str):
                return self._read_result(path, error_sink=errors.append)

            blocked = config_support.load_config(
                defaults=defaults,
                config_paths=lambda: [str(high), str(low)],
                read_toml_result=read_invalid,
                normalize_keys=config_support.normalize_keys,
            )
            self.assertEqual(blocked["tz"], "UTC")
            self.assertTrue(errors)
            self.assertIn(str(high), errors[0])

            with patch.object(
                config_support.os.path,
                "exists",
                side_effect=OSError("stat unavailable"),
            ):
                inspection_errors: list[str] = []
                inspected = self._read_result(
                    str(high), error_sink=inspection_errors.append
                )

            self.assertTrue(inspected.is_invalid)
            self.assertTrue(inspection_errors)
            self.assertIn("inspection failed", inspection_errors[0])


if __name__ == "__main__":
    unittest.main()
