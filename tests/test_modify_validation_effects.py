from types import SimpleNamespace
import unittest

from nautical_core.modify_validation_effects import validate_anchor


class ModifyValidationEffectsTests(unittest.TestCase):
    def test_anchor_validation_does_not_persist_unused_hints(self) -> None:
        calls = []

        class Core:
            ENABLE_ANCHOR_CACHE = True

            @staticmethod
            def lint_anchor_expr(_expr):
                return (), ()

            @staticmethod
            def validate_anchor_expr_strict(expr):
                calls.append(("validate", expr))
                return object()

            @staticmethod
            def build_and_cache_hints(*_args, **_kwargs):
                calls.append(("hints",))
                raise AssertionError("hint persistence must not run during validation")

        host = SimpleNamespace(
            core=Core(),
            _module=lambda _name: SimpleNamespace(panel=lambda *_args, **_kwargs: None),
            _fail_and_exit=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected failure")),
        )
        validate_anchor(host, {}, {}, "w:mon")
        self.assertEqual(calls, [("validate", "w:mon")])


if __name__ == "__main__":
    unittest.main()
