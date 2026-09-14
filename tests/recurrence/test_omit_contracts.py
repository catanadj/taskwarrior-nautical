"""Direct contracts for date-based recurrence omission."""

from datetime import date
import unittest

import nautical_core as core
from nautical_core import anchor_omit


class OmitContracts(unittest.TestCase):
    def test_omit_expressions_are_date_only(self) -> None:
        with self.assertRaises(ValueError) as raised:
            anchor_omit.validate_omit_expr_strict(
                "w:mon@t=09:00",
                validate_anchor_expr_cached=core.validate_anchor_expr_strict,
            )
        self.assertIn(
            "omit does not support time modifiers (@t). Omit rules are date-based only.",
            str(raised.exception),
        )

    def test_omit_scheduler_skips_matching_expression_dates(self) -> None:
        anchor_dnf = core.validate_anchor_expr_strict("w:mon,wed,fri")
        omit_dnf = anchor_omit.validate_omit_expr_strict(
            "w:wed",
            validate_anchor_expr_cached=core.validate_anchor_expr_strict,
        )
        next_date, _metadata = anchor_omit.next_after_expr_with_omit(
            anchor_dnf,
            date(2025, 1, 6),
            default_seed=date(2025, 1, 6),
            seed_base="omit-test",
            omit_dnf=omit_dnf,
            core=core,
        )
        self.assertEqual(next_date, date(2025, 1, 10))

    def test_omit_scheduler_skips_dates_from_file_state(self) -> None:
        anchor_dnf = core.validate_anchor_expr_strict("w:mon,wed,fri")
        omit_state = anchor_omit.combine_omit_state(omit_dates={date(2025, 1, 10)})
        next_date, _metadata = anchor_omit.next_after_expr_with_omit(
            anchor_dnf,
            date(2025, 1, 8),
            default_seed=date(2025, 1, 6),
            seed_base="omit-file-test",
            omit_dnf=omit_state,
            core=core,
        )
        self.assertEqual(next_date, date(2025, 1, 13))

    def test_grouped_list_plus_expression_filters_every_list_member(self) -> None:
        omit_dnf = anchor_omit.validate_omit_expr_strict(
            "w:mon,wed,fri + y:apr",
            validate_anchor_expr_cached=core.validate_anchor_expr_strict,
        )
        self.assertTrue(
            anchor_omit.omit_expr_fires_on_date(
                omit_dnf, date(2026, 4, 13), date(2026, 4, 11), "omit-test", core=core
            )
        )
        self.assertTrue(
            anchor_omit.omit_expr_fires_on_date(
                omit_dnf, date(2026, 4, 15), date(2026, 4, 11), "omit-test", core=core
            )
        )
        self.assertFalse(
            anchor_omit.omit_expr_fires_on_date(
                omit_dnf, date(2026, 5, 4), date(2026, 4, 11), "omit-test", core=core
            )
        )
        self.assertFalse(
            anchor_omit.omit_expr_fires_on_date(
                omit_dnf, date(2026, 5, 6), date(2026, 4, 11), "omit-test", core=core
            )
        )

    def test_business_day_roll_omits_the_rolled_date(self) -> None:
        omit_dnf = anchor_omit.validate_omit_expr_strict(
            "y:04-25@nbd",
            validate_anchor_expr_cached=core.validate_anchor_expr_strict,
        )
        self.assertTrue(
            anchor_omit.omit_expr_fires_on_date(
                omit_dnf,
                date(2026, 4, 27),
                date(2026, 4, 12),
                "omit-roll-test",
                core=core,
            )
        )

    def test_positive_calendar_and_business_offsets_match_shifted_dates(self) -> None:
        calendar_offset = anchor_omit.validate_omit_expr_strict(
            "y:04-25@+2d",
            validate_anchor_expr_cached=core.validate_anchor_expr_strict,
        )
        self.assertTrue(
            anchor_omit.omit_expr_fires_on_date(
                calendar_offset,
                date(2026, 4, 27),
                date(2026, 4, 12),
                "omit-offset-test",
                core=core,
            )
        )

        business_offset = anchor_omit.validate_omit_expr_strict(
            "y:04-24@+1bd",
            validate_anchor_expr_cached=core.validate_anchor_expr_strict,
        )
        self.assertTrue(
            anchor_omit.omit_expr_fires_on_date(
                business_offset,
                date(2026, 4, 27),
                date(2026, 4, 12),
                "omit-business-offset-test",
                core=core,
            )
        )


if __name__ == "__main__":
    unittest.main()
