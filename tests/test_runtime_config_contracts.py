"""Direct contracts for effective runtime configuration snapshots."""

import unittest
from types import SimpleNamespace

import nautical_core as core
import nautical_core.core_config as core_config
from nautical_core.configuration_facade import validate_scheduling


def _validate_scheduling_owner(
    *, anchor_presets=None, omit_presets=None, validate_anchor=None,
    configured_calendars=None, astronomy_validation=None,
):
    def import_sibling(name):
        if name == "astronomy":
            return SimpleNamespace(validate_configuration=lambda _config: astronomy_validation())
        if name == "anchor_omit":
            return SimpleNamespace(validate_omit_expr_strict=lambda *_args, **_kwargs: None)
        raise AssertionError(f"unexpected sibling: {name}")

    validate_scheduling(
        core_config=core_config,
        astronomy_config={},
        anchor_presets=anchor_presets or {},
        omit_presets=omit_presets or {},
        resolve_anchor_presets=lambda name: str((anchor_presets or {}).get(name[1:], name)),
        validate_anchor_expr=validate_anchor or (lambda _expression: None),
        resolve_omit_presets=lambda name: str((omit_presets or {}).get(name[1:], name)),
        configured_business_calendars=configured_calendars or (lambda: {}),
        import_sibling=import_sibling,
    )


class RuntimeConfigContracts(unittest.TestCase):
    def test_domain_validation_fails_closed_for_astronomy_presets_and_calendars(self) -> None:
        def invalid_astronomy():
            raise ValueError("latitude must be within range")

        with self.assertRaisesRegex(RuntimeError, "latitude"):
            _validate_scheduling_owner(astronomy_validation=invalid_astronomy)

        def invalid_anchor(_expression):
            raise ValueError("weekly selector is invalid")

        with self.subTest(case="invalid anchor preset"):
            with self.assertRaisesRegex(RuntimeError, "weekly"):
                _validate_scheduling_owner(
                    anchor_presets={"broken": "w:not-a-day"},
                    validate_anchor=invalid_anchor,
                    astronomy_validation=lambda: None,
                )

        def invalid_calendars():
            raise ValueError("business_calendar.work.anchor is invalid")

        with self.subTest(case="invalid business calendar"):
            with self.assertRaisesRegex(RuntimeError, "business_calendar.work.anchor"):
                _validate_scheduling_owner(
                    configured_calendars=invalid_calendars,
                    astronomy_validation=lambda: None,
                )

    def test_effective_snapshot_is_provenanced_and_isolated(self) -> None:
        snapshot = core.effective_config_snapshot()
        values = snapshot.get("values")
        self.assertIsInstance(values, dict)
        self.assertTrue(str(snapshot.get("source") or ""))

        original_timezone = values.get("tz")
        values["tz"] = "mutated-in-test"

        self.assertEqual(core_config.LOCAL_TZ_NAME, original_timezone)

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
