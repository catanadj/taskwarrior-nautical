import unittest

from nautical_core.exit_presentation import ExitDrainProgress


class ExitDrainProgressTests(unittest.TestCase):
    def test_internal_drain_stages_use_concise_user_facing_labels(self) -> None:
        expected = {
            "starting intent": "Preparing",
            "child mutation": "Created",
            "child verified": "Confirmed",
            "child mutation and verification": "Created",
            "parent mutation": "Linked",
            "parent verified": "Confirmed",
            "parent mutation and verification": "Linked",
            "intent verified": "Verified",
            "intent acknowledged": "Recorded",
            "intent finished": "Completed",
        }
        self.assertTrue(all(" " not in label for label in expected.values()))
        for internal, friendly in expected.items():
            with self.subTest(internal=internal):
                self.assertEqual(
                    ExitDrainProgress._description(internal),
                    friendly,
                )

    def test_unknown_detail_does_not_expose_internal_vocabulary(self) -> None:
        self.assertEqual(
            ExitDrainProgress._description("future_internal_phase"),
            "Processing",
        )

    def test_empty_detail_uses_compact_title(self) -> None:
        self.assertEqual(ExitDrainProgress._description(), "Processing")


if __name__ == "__main__":
    unittest.main()
