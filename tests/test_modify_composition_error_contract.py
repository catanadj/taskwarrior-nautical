from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import nautical_core.modify_composition as modify_composition
from nautical_core.business_calendar_config import BusinessCalendarConfigError


class ModifyCompositionErrorContractTests(unittest.TestCase):
    def test_business_calendar_startup_does_not_hide_internal_failures(self) -> None:
        valid = object()
        validation = SimpleNamespace(
            WorkflowRoute=SimpleNamespace(RECURRING_EDIT=object()),
            ValidationStatus=SimpleNamespace(VALID=valid),
            validate_task_mapping=lambda *_args, **_kwargs: (
                None,
                SimpleNamespace(status=valid),
            ),
        )
        state = SimpleNamespace(diag_stats={})
        fail_and_exit = Mock()
        host = SimpleNamespace(
            _reset_modify_runtime_state=lambda: None,
            _modify_runtime_state=lambda: state,
            _ptime=SimpleNamespace(perf_counter=lambda: 1.0),
            _read_two=lambda: ({}, {"chain": "on"}),
            _PARSED_OLD_OBSERVATION=None,
            _PARSED_NEW_OBSERVATION=None,
            _load_core=lambda: None,
            _apply_description_uda_aliases=lambda _old, _new: None,
            _fail_and_exit=fail_and_exit,
            core=SimpleNamespace(
                _import_sibling=lambda _name: validation,
                scheduling_configuration_error=lambda: "",
                BusinessCalendarConfigError=BusinessCalendarConfigError,
                use_task_business_calendar=lambda _task: (_ for _ in ()).throw(
                    RuntimeError("business calendar owner invariant failed")
                ),
            ),
        )
        capabilities = SimpleNamespace(
            hook_results=SimpleNamespace(emit_passthrough_json=Mock()),
            modify_lifecycle=SimpleNamespace(
                task_has_nautical_fields=lambda _task: True,
            ),
            hook_context=object(),
            hook_engine=object(),
        )

        with patch.object(modify_composition, "capabilities_for", return_value=capabilities):
            with self.assertRaisesRegex(RuntimeError, "owner invariant"):
                modify_composition.run_on_modify(host)

        fail_and_exit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
