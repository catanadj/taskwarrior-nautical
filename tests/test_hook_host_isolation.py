from __future__ import annotations

import unittest

from nautical_core.modify_composition import hook_host


class HookHostIsolationTests(unittest.TestCase):
    def test_hosts_keep_composition_namespaces_separate(self) -> None:
        first_values = {"value": "first"}
        second_values = {"value": "second"}

        first = hook_host(first_values, "first-hook")
        second = hook_host(second_values, "second-hook")

        self.assertEqual(first.value, "first")
        self.assertEqual(second.value, "second")
        self.assertEqual(first.__name__, "first-hook")
        self.assertEqual(second.__name__, "second-hook")

        first_values["value"] = "updated"
        self.assertEqual(first.value, "updated")
        self.assertEqual(second.value, "second")

    def test_missing_composition_attribute_does_not_fall_through(self) -> None:
        host = hook_host({}, "isolated-hook")
        with self.assertRaises(AttributeError):
            _ = host.not_in_composition


if __name__ == "__main__":
    unittest.main()
