from __future__ import annotations

import unittest

import nautical_core.query_models as query_models


class QueryModelsContractTests(unittest.TestCase):
    def test_query_status_validators_share_one_allowed_values_set(self) -> None:
        self.assertEqual(
            query_models.QUERY_STATUSES,
            frozenset(("found", "empty", "exhausted", "absent", "unavailable", "invalid")),
        )
