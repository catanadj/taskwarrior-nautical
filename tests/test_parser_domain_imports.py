import importlib
import unittest


class ParserDomainImportTests(unittest.TestCase):
    def test_canonical_parser_modules_are_importable(self):
        package = importlib.import_module("nautical_core.parsing")
        for name in package.__all__:
            self.assertIsNotNone(importlib.import_module(f"nautical_core.parsing.{name}"))

    def test_legacy_parser_modules_remain_compatible(self):
        canonical = importlib.import_module("nautical_core.parsing.parser_models")
        legacy = importlib.import_module("nautical_core.parser_models")
        self.assertIs(legacy.AnchorDNF, canonical.AnchorDNF)


if __name__ == "__main__":
    unittest.main()
