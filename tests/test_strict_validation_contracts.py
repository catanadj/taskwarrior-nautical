from __future__ import annotations

import builtins
import unittest
from unittest.mock import patch

import nautical_core.strict_validation as strict_validation


class StrictValidationContracts(unittest.TestCase):
    def test_yearly_month_conversion_does_not_hide_unexpected_integer_failures(self) -> None:
        builtin_int = builtins.int

        def parse_int(value):
            if value == "broken":
                raise RuntimeError("integer parser implementation failed")
            return builtin_int(value)

        with patch.object(strict_validation, "int", side_effect=parse_int, create=True):
            with self.assertRaisesRegex(RuntimeError, "integer parser implementation failed"):
                strict_validation.validate_anchor_atom_strict(
                    {"typ": "y", "spec": "rand-broken", "ival": 1},
                    validate_weekly_spec=lambda _spec: None,
                    validate_monthly_spec=lambda _spec: None,
                    active_mod_keys=lambda _mods: set(),
                    validate_yearly_token_format=lambda _spec: None,
                    parse_error_cls=ValueError,
                )


if __name__ == "__main__":
    unittest.main()
