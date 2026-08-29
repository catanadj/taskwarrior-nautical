import unittest

from nautical_core.exit_presentation import ExitDrainProgress


class ExitDrainProgressTests(unittest.TestCase):
    def test_internal_drain_stages_have_distinct_user_facing_labels(self) -> None:
        expected = {
            "starting intent": "Preparing update",
            "child mutation": "Next task created",
            "child verified": "Next task confirmed",
            "child mutation and verification": "Next task created and confirmed",
            "parent mutation": "Task sequence linked",
            "parent verified": "Task link confirmed",
            "parent mutation and verification": "Task sequence linked and confirmed",
            "intent verified": "Update verified",
            "intent acknowledged": "Completion recorded",
            "intent finished": "Update complete",
        }
        self.assertEqual(len(set(expected.values())), len(expected))
        for internal, friendly in expected.items():
            with self.subTest(internal=internal):
                self.assertEqual(
                    ExitDrainProgress._description(internal),
                    f"⚓ Updating recurring tasks · {friendly}",
                )

    def test_unknown_detail_does_not_expose_internal_vocabulary(self) -> None:
        self.assertEqual(
            ExitDrainProgress._description("future_internal_phase"),
            "⚓ Updating recurring tasks · Processing update",
        )

    def test_empty_detail_uses_compact_title(self) -> None:
        self.assertEqual(ExitDrainProgress._description(), "⚓ Updating recurring tasks")


if __name__ == "__main__":
    unittest.main()
