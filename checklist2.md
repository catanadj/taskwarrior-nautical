  2. [x] Raise type-check strictness gradually (start with nautical_core.py, reduce disable_error_code list in mypy.ini
     step by step).
     - Implemented: removed `call-arg` and `index` from `disable_error_code` in `mypy.ini`; mypy remains green.
  3. [x] Add targeted perf micro-benchs for hot paths (next_after_expr, hint generation, cache load/save) and gate
     regressions in CI.
     - Implemented: added `cache_save` and `cache_load_hot` checks to `tools/nautical_perf_budget.py` and
       `tools/perf_budget.json`; existing perf CI gate enforces budgets.
  4. [x] Split the most complex anchor logic into smaller pure helpers (parser/validator/rewriter boundaries) with direct
     unit coverage.
     - Implemented: extracted yearly token validation into pure helpers in `nautical_core.py`
       (`_validate_yearly_spec_token`, `_yearly_check_day_month`, etc.) and added direct helper tests.
  5. [x] Add deployment sanity checks (single-command verify that required runtime files are present and hooks still
     output strict JSON).
     - Implemented: `tools/nautical_deploy_sanity.py` plus CI execution in `.github/workflows/type-check.yml`.
