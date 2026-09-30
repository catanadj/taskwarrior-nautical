from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nautical_core.add_validation as add_validation
import nautical_core.modify_expiration as modify_expiration
from nautical_core.lifecycle.models import LifecycleAction


class ModifyExpirationContractTests(unittest.TestCase):
    def _deleted_services(self, expiration: object, *, warnings: list[tuple[dict, str]]) -> modify_expiration.DeletedModifyServices:
        return modify_expiration.DeletedModifyServices(
            expiration=expiration,
            terminal_chain_off=lambda *_args: self.fail("expiration must not stop the chain"),
            now_utc=lambda: datetime(2026, 7, 27, tzinfo=timezone.utc),
            end_chain_summary=lambda *_args, **_kwargs: self.fail("expiration must not render a stop summary"),
            format_root_and_age=lambda *_args: "root",
            short=lambda value: str(value),
            panel=lambda *_args, **_kwargs: None,
            diag=lambda _message: None,
            recovery_warning=lambda task, reason: warnings.append((task, reason)),
        )

    def test_expiration_disposition_delegates_to_recovery_owner(self) -> None:
        old = {"status": "pending", "chainID": "expiration-chain"}
        new = dict(old, status="deleted", until="20260726T235900Z", end="20260727T000000Z")
        expiration_services = object()
        warnings: list[tuple[dict, str]] = []
        called: list[tuple[dict, object]] = []
        evidence = SimpleNamespace(disposition=SimpleNamespace(value="expiration"), reason="expired")
        services = self._deleted_services(expiration_services, warnings=warnings)

        with (
            patch.object(modify_expiration, "classify_deleted_task", return_value=evidence),
            patch.object(
                modify_expiration,
                "handle_expired_deleted_modify",
                side_effect=lambda task, *, services: called.append((task, services)) or True,
            ),
        ):
            modify_expiration.handle_deleted_modify(old, new, services=services)

        self.assertEqual(called, [(new, expiration_services)])
        self.assertEqual(warnings, [])

    def test_expiration_panel_explains_native_until_carry(self) -> None:
        child_due = datetime(2026, 7, 27, 9, tzinfo=timezone.utc)
        child_until = datetime(2026, 8, 2, 23, 59, tzinfo=timezone.utc)
        rows: list[tuple[str, str]] = []
        core = SimpleNamespace(
            coerce_int=lambda value, default: int(value or default),
            fmt_dt_local=lambda value: value.isoformat(),
            _import_sibling=lambda _name: add_validation,
            to_local=lambda value: value,
        )

        def safe_parse_datetime(value: object) -> tuple[datetime | None, str | None]:
            if not isinstance(value, str):
                return None, "not text"
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")), None
            except ValueError as exc:
                return None, str(exc)

        plan = SimpleNamespace(
            plan=SimpleNamespace(
                action=LifecycleAction.SPAWN_CHILD,
                child_dict=lambda: {"until": child_until.isoformat()},
            ),
            child_due=child_due,
            next_link=2,
            reason="expired link missing next link",
        )
        services = SimpleNamespace(
            core=core,
            safe_parse_datetime=safe_parse_datetime,
            panel=lambda _title, panel_rows, **_kwargs: rows.extend(panel_rows),
        )

        modify_expiration._render_recovery_panel(
            {"description": "Take the trash out", "link": 1},
            plan,
            services=services,
            result="Next occurrence created",
            child_short="22222222",
        )

        expected_carry = add_validation.describe_native_until_carry(
            child_until, child_due, to_local=core.to_local,
        )
        self.assertIn(("Expiration", expected_carry), rows)
        self.assertTrue(any(label == "Next expires" for label, _value in rows))

    def test_expiration_recovery_failure_warns_without_stopping_chain(self) -> None:
        old = {
            "status": "pending",
            "chainID": "expiration-chain",
            "chain": "on",
        }
        new = dict(old, status="deleted", until="20260726T235900Z", end="20260727T000000Z")
        warnings: list[tuple[dict, str]] = []
        evidence = SimpleNamespace(disposition=SimpleNamespace(value="expiration"), reason="expired")
        services = self._deleted_services(object(), warnings=warnings)

        with (
            patch.object(modify_expiration, "classify_deleted_task", return_value=evidence),
            patch.object(
                modify_expiration,
                "handle_expired_deleted_modify",
                side_effect=RuntimeError("missing recovery module"),
            ),
        ):
            modify_expiration.handle_deleted_modify(old, new, services=services)

        self.assertEqual(new["chain"], "on")
        self.assertEqual(len(warnings), 1)
        self.assertIs(warnings[0][0], new)
        self.assertIn("chain remains active", warnings[0][1])


if __name__ == "__main__":
    unittest.main()
