"""Direct contracts for recurrence hint precomputation and cache validation."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import nautical_core as core
from nautical_core import natural_language_api
from nautical_core.precompute import build_and_cache_hints


class PrecomputeContractTests(unittest.TestCase):
    def _builder(self, result=None):
        builder = Mock()
        builder.build.return_value = result or {"next_dates": ["2026-09-14T00:00"]}
        return builder

    def _build(self, *, cached=None, validate=None, save=None, builder=None):
        validate = validate or (lambda expression: [[{"typ": "w", "spec": expression[-3:]}]])
        return build_and_cache_hints(
            "w:mon",
            anchor_mode="skip",
            default_due_dt=None,
            cache_key_for_task=lambda *_args: "key",
            cache_load=lambda _key: cached,
            validate_anchor_expr_strict=validate,
            describe_anchor_expr_from_dnf=lambda _dnf, **_kwargs: "Mondays",
            cache_save=save or (lambda *_args: None),
            anchor_year_fmt="%Y",
            wrand_salt="seed",
            local_tz_name="UTC",
            business_calendar_fingerprint="calendar",
            hint_builder=builder,
        )

    def test_matching_cache_entry_returns_without_constructing_builder(self) -> None:
        dnf = [[{"typ": "w", "spec": "mon"}]]
        cached = {"dnf": dnf, "natural": "Mondays", "next_dates": ["cached"]}
        validate = Mock(return_value=dnf)
        builder = self._builder()

        result = self._build(cached=cached, validate=validate, builder=builder)

        self.assertEqual(result, cached)
        builder.build.assert_not_called()
        validate.assert_called_once_with("w:mon")

    def test_cached_hint_payload_is_isolated_from_caller_mutation(self) -> None:
        first = core.build_and_cache_hints("w:mon@t=09:00", "skip")
        second = core.build_and_cache_hints("w:mon@t=09:00", "skip")

        self.assertIsNot(first, second)
        first_dnf = first.get("dnf") or []
        second_dnf = second.get("dnf") or []
        if first_dnf and second_dnf:
            first_dnf[0][0]["spec"] = "fri"
            self.assertEqual(second_dnf[0][0]["spec"], "mon")

    def test_production_hint_path_uses_scheduler_service_not_legacy_callbacks(self) -> None:
        with (
            patch.object(
                core,
                "next_after_expr",
                side_effect=AssertionError("hint build used a legacy scheduler callback"),
            ),
            patch.object(
                core,
                "_next_for_or",
                side_effect=AssertionError("hint build used a legacy scheduler callback"),
            ),
            patch.object(core, "cache_load", return_value=None),
            patch.object(core, "cache_save", return_value=None),
        ):
            payload = core.build_and_cache_hints("w:thu@t=08:45", "skip")

        self.assertTrue(payload["next_dates"])
        self.assertGreater(payload["per_year"]["est"], 0)

    def test_stale_shape_valid_entry_is_rebuilt_and_replaced(self) -> None:
        cached = {
            "dnf": [[{"typ": "w", "spec": "tue"}]],
            "natural": "Tuesdays",
        }
        saved = []
        builder = self._builder({"next_dates": ["2026-09-14T00:00"]})

        result = self._build(cached=cached, save=lambda key, payload: saved.append((key, payload)), builder=builder)

        self.assertEqual(result["dnf"], [[{"typ": "w", "spec": "mon"}]])
        self.assertEqual(result["natural"], "Mondays")
        self.assertEqual(saved, [("key", result)])
        builder.build.assert_called_once()

    def test_cache_miss_validates_once_and_uses_typed_builder_factory(self) -> None:
        validated = [[{"typ": "w", "spec": "mon"}]]
        validate = Mock(return_value=validated)
        builder = self._builder()
        factory = Mock(return_value=builder)
        saved = []

        result = build_and_cache_hints(
            "w:mon",
            anchor_mode="skip",
            default_due_dt=None,
            cache_key_for_task=lambda *_args: "key",
            cache_load=lambda _key: None,
            validate_anchor_expr_strict=validate,
            describe_anchor_expr_from_dnf=lambda _dnf, **_kwargs: "Mondays",
            cache_save=lambda key, payload: saved.append((key, payload)),
            anchor_year_fmt="%Y",
            wrand_salt="seed",
            local_tz_name="UTC",
            hint_builder_factory=factory,
        )

        validate.assert_called_once_with("w:mon")
        factory.assert_called_once_with()
        builder.build.assert_called_once()
        self.assertEqual(result["dnf"], validated)
        self.assertEqual(saved, [("key", result)])

    def test_hint_payload_matches_parser_normalization_and_natural_text(self) -> None:
        natural = natural_language_api.for_core(module=core)
        expressions = (
            "w:mon",
            "m:15",
            "y:12-25",
            "w:mon@t=09:00",
            "m:rand",
            "w/2:mon",
        )
        for expression in expressions:
            with self.subTest(expression=expression):
                expected_dnf = core.validate_anchor_expr_strict(expression)
                expected_natural = core.describe_anchor_expr(expression)
                result = build_and_cache_hints(
                    expression,
                    anchor_mode="skip",
                    default_due_dt=None,
                    cache_key_for_task=lambda *_args: expression,
                    cache_load=lambda _key: None,
                    validate_anchor_expr_strict=core.validate_anchor_expr_strict,
                    describe_anchor_expr_from_dnf=natural._describe_anchor_expr_from_dnf,
                    cache_save=lambda *_args: None,
                    anchor_year_fmt="%Y",
                    wrand_salt="seed",
                    local_tz_name="UTC",
                    hint_builder=self._builder({"next_dates": []}),
                )

                self.assertEqual(result["dnf"], expected_dnf)
                self.assertEqual(result["natural"], expected_natural)


if __name__ == "__main__":
    unittest.main()
