import json
import unittest

from nautical_core.task_codec import DEFAULT_TASK_CODEC


class TaskCodecContainerRoundTripTests(unittest.TestCase):
    def test_arbitrary_json_containers_keep_their_shape(self):
        row = {
            "description": "café",
            "tags": [],
            "annotations": [],
            "depends": [],
            "custom_array": [["key", "value"]],
            "custom_object": {"key": "value"},
            "nested": [{"items": []}, []],
        }

        observed = DEFAULT_TASK_CODEC.decode_row(row, source_query="test:roundtrip")

        self.assertEqual(observed.to_mapping(), row)
        self.assertEqual(json.loads(DEFAULT_TASK_CODEC.encode_task_import(observed)), row)

    def test_array_and_object_have_distinct_fingerprints(self):
        array = DEFAULT_TASK_CODEC.decode_row({"custom": [["key", "value"]]}, source_query="test:array")
        object_value = DEFAULT_TASK_CODEC.decode_row({"custom": {"key": "value"}}, source_query="test:object")

        self.assertNotEqual(array.semantic_fingerprint, object_value.semantic_fingerprint)

    def test_returned_containers_are_copies(self):
        row = {"custom": {"nested": []}}
        observed = DEFAULT_TASK_CODEC.decode_row(row, source_query="test:immutability")

        row["custom"]["nested"].append("source mutation")
        returned = observed.to_mapping()
        returned["custom"]["nested"].append("returned mutation")

        self.assertEqual(observed.to_mapping(), {"custom": {"nested": []}})


if __name__ == "__main__":
    unittest.main()
