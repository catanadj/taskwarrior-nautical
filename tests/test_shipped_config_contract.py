"""Direct contracts for the shipped Taskwarrior/Nautical configuration."""

from pathlib import Path
import unittest

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from nautical_core import config_schema


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config-nautical.toml"


class ShippedConfigContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_hook_and_panel_settings_remain_top_level_after_preset_tables(self) -> None:
        config = self.config
        self.assertEqual(config.get("panel_mode"), "rich")
        self.assertEqual(config.get("live_panel_duration_ms"), 160)
        self.assertIs(config.get("show_analytics"), False)
        omit_presets = config.get("omit_presets")
        if not isinstance(omit_presets, dict):
            omit_presets = {}
        self.assertNotIn("panel_mode", omit_presets)
        self.assertNotIn("show_analytics", omit_presets)

    def test_every_shipped_setting_matches_the_runtime_schema(self) -> None:
        issues = config_schema.validate_config(self.config)
        self.assertEqual(issues, [])
        self.assertNotIn("verify_import", self.config)
        self.assertNotIn("holiday_region", self.config)

    def test_season_mode_default_and_supported_values_are_explicit(self) -> None:
        self.assertEqual(config_schema.spec_default("season_mode"), "fixed")
        self.assertEqual(config_schema.validate_config({"season_mode": "fixed"}), [])
        self.assertEqual(
            config_schema.validate_config({"season_mode": "astronomical"}), []
        )
        issues = config_schema.validate_config({"season_mode": "lunar"})
        self.assertTrue(any(issue.get("kind") == "choice" for issue in issues))

    def test_schema_reports_deprecated_invalid_unknown_and_effective_values(self) -> None:
        issues = config_schema.validate_config(
            {
                "verify_import": False,
                "show_analytics": "yes",
                "outbox_drain_max_items": 0,
                "panel_mode": "sparkle",
                "setting_typo": True,
            }
        )

        self.assertTrue(
            {"deprecated", "type", "range", "choice", "unknown"}
            <= {issue["kind"] for issue in issues}
        )
        drain = next(issue for issue in issues if issue["key"] == "outbox_drain_max_items")
        panel = next(issue for issue in issues if issue["key"] == "panel_mode")
        self.assertEqual(drain.get("effective"), 1)
        self.assertEqual(panel.get("effective"), "rich")


if __name__ == "__main__":
    unittest.main()
