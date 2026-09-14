"""Direct contracts for effective runtime configuration snapshots."""

import unittest

import nautical_core as core
from nautical_core import core_config


class RuntimeConfigContracts(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
