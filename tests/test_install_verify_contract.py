from __future__ import annotations

from collections.abc import Mapping
from typing import get_type_hints
import unittest

from nautical_core.tools import nautical_install_verify


class InstallationVerificationInputContractTests(unittest.TestCase):
    def test_finding_reader_ignores_malformed_non_list_findings(self) -> None:
        self.assertEqual(nautical_install_verify._findings({"operator_findings": 7}), [])

    def test_finding_reader_uses_object_mapping_boundary(self) -> None:
        hints = get_type_hints(nautical_install_verify._findings)

        self.assertEqual(hints["payload"], Mapping[str, object])
        self.assertEqual(hints["return"], list[Mapping[str, object]])

    def test_finding_filter_uses_object_mapping_boundary(self) -> None:
        hints = get_type_hints(nautical_install_verify._items)

        self.assertEqual(hints["payload"], Mapping[str, object])
        self.assertEqual(hints["return"], list[Mapping[str, object]])


if __name__ == "__main__":
    unittest.main()
