from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

from nautical_core.add_validation import validate_datetime_field
from nautical_core.chain_generation import ChainGenerationService
from nautical_core.task_datetime import ConfiguredTaskDatetimeParser
from nautical_core.tools.nautical_reconcile import _parse_datetime


UTC = timezone.utc
VALID = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)


class TaskDatetimeContractTests(unittest.TestCase):
    def test_all_consumers_share_empty_valid_malformed_and_wrong_type_results(self):
        def parse(value):
            if value == "valid":
                return VALID
            if value == "malformed":
                return None
            return (_ for _ in ()).throw(TypeError("text required"))

        parser = ConfiguredTaskDatetimeParser(parse)
        expected = {
            "": (None, None),
            "valid": (VALID, None),
            "malformed": (None, "Unrecognized datetime format 'malformed'"),
            17: (None, "Datetime value must be text"),
        }
        for value, result in expected.items():
            self.assertEqual(parser.parse(value), result)
            self.assertEqual(parser.parse(value), result)
            self.assertEqual(
                validate_datetime_field(value, "due", parser=parser),
                (result[0], f"due: {result[1]}" if result[1] else None),
            )
        core = SimpleNamespace(parse_dt_any=parse)
        generation = ChainGenerationService.from_core(core)
        for value, result in expected.items():
            self.assertEqual(generation.parse_datetime(value), result)
        hook = SimpleNamespace(core=core)
        for value, result in expected.items():
            self.assertEqual(_parse_datetime(hook, value), result)

    def test_parser_exceptions_are_normalized_and_diagnosed(self):
        diagnostics = []

        def raising(_value):
            raise RuntimeError("parser exploded")

        parser = ConfiguredTaskDatetimeParser(raising, diagnostic=diagnostics.append)
        expected = (None, "Datetime parsing failed")
        self.assertEqual(parser.parse("value"), expected)
        self.assertEqual(parser.parse("value"), expected)
        self.assertEqual(validate_datetime_field("value", "until", parser=parser), (None, "until: Datetime parsing failed"))
        self.assertEqual(len(diagnostics), 3)
        self.assertTrue(all("parser exploded" in item for item in diagnostics))

    def test_parser_rejects_non_datetime_dependency_results(self):
        diagnostics = []
        parser = ConfiguredTaskDatetimeParser(lambda _value: "not-a-datetime", diagnostic=diagnostics.append)
        self.assertEqual(parser.parse("value"), (None, "Datetime parser returned an invalid value"))
        self.assertEqual(len(diagnostics), 1)


if __name__ == "__main__":
    unittest.main()
