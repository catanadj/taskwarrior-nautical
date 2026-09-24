"""Direct contracts for effective runtime configuration snapshots."""

import unittest
from unittest.mock import patch

import nautical_core as core
from nautical_core import core_config


class RuntimeConfigContracts(unittest.TestCase):
    def test_domain_validation_fails_closed_for_astronomy_presets_and_calendars(self) -> None:
        with patch.object(
            core._astronomy,
            "validate_configuration",
            side_effect=ValueError("latitude must be within range"),
        ):
            with self.assertRaisesRegex(RuntimeError, "latitude"):
                core.validate_scheduling_configuration()

        with (
            patch.object(core, "ANCHOR_PRESETS", {"broken": "w:not-a-day"}),
            patch.object(core._astronomy, "validate_configuration", return_value=None),
            patch.object(core, "resolve_anchor_presets", return_value="w:not-a-day"),
            patch.object(core, "configured_business_calendars", return_value={}),
        ):
            with self.assertRaisesRegex(RuntimeError, "weekly"):
                core.validate_scheduling_configuration()

        with (
            patch.object(core, "ANCHOR_PRESETS", {}),
            patch.object(core, "OMIT_PRESETS", {}),
            patch.object(core._astronomy, "validate_configuration", return_value=None),
            patch.object(
                core,
                "configured_business_calendars",
                side_effect=ValueError("business_calendar.work.anchor is invalid"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "business_calendar.work.anchor"):
                core.validate_scheduling_configuration()

    def test_effective_snapshot_is_provenanced_and_isolated(self) -> None:
        snapshot = core.effective_config_snapshot()
        values = snapshot.get("values")
        self.assertIsInstance(values, dict)
        self.assertTrue(str(snapshot.get("source") or ""))

        original_timezone = values.get("tz")
        values["tz"] = "mutated-in-test"

        self.assertEqual(core._core_config.LOCAL_TZ_NAME, original_timezone)

    def test_hot_config_fingerprint_does_not_stat_filesystem(self) -> None:
        first = core_config.effective_config_fingerprint()
        original_stat = core_config.os.stat
        try:
            core_config.os.stat = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("hot fingerprint touched the filesystem")
            )
            self.assertEqual(core_config.effective_config_fingerprint(), first)
        finally:
            core_config.os.stat = original_stat

    def test_loaded_config_accessors_are_canonical_and_isolated(self) -> None:
        snapshot = core_config.loaded_config_snapshot()
        self.assertIsInstance(snapshot, dict)
        self.assertEqual(core_config.loaded_config_value("tz"), snapshot["tz"])
        snapshot["tz"] = "mutated-in-test"
        self.assertNotEqual(core_config.loaded_config_value("tz"), "mutated-in-test")


if __name__ == "__main__":
    unittest.main()
