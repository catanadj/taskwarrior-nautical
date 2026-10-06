import importlib
import unittest


class ParserDomainImportTests(unittest.TestCase):
    def test_canonical_parser_modules_are_importable(self):
        package = importlib.import_module("nautical_core.parsing")
        for name in package.__all__:
            self.assertIsNotNone(importlib.import_module(f"nautical_core.parsing.{name}"))

    def test_root_parser_forwarding_modules_are_removed(self):
        for name in (
            "nautical_core.parser_models",
            "nautical_core.parser_frontend",
            "nautical_core.parser_support_api",
        ):
            with self.subTest(name=name), self.assertRaises(ModuleNotFoundError):
                importlib.import_module(name)


if __name__ == "__main__":
    unittest.main()
