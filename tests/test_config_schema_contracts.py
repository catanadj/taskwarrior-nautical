from __future__ import annotations

import builtins
import unittest
from unittest.mock import patch

import nautical_core.config_schema as config_schema


class ConfigSchemaContracts(unittest.TestCase):
    def test_integer_normalization_does_not_hide_unexpected_conversion_failures(self) -> None:
        builtin_int = builtins.int

        def parse_int(value):
            if value == "23":
                raise RuntimeError("configuration integer conversion failed")
            return builtin_int(value)

        with patch.object(config_schema, "int", side_effect=parse_int, create=True):
            with self.assertRaisesRegex(RuntimeError, "configuration integer conversion failed"):
                config_schema._effective_int({"default": 7}, "23")


if __name__ == "__main__":
    unittest.main()
