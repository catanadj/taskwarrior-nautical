## 13. Performance And Call-Budget Pass

- [x] Benchmark cold imports and ordinary thin hooks before and after each
  ownership extraction.
  (`test_performance_benchmark_cold_imports`)
- [x] Benchmark fresh/idempotent CP and anchor completion, empty/populated
  outbox, partial recovery, reconcile, doctor, and navigator.
  (`test_performance_benchmark_workflow_paths`)
- [x] Assert Taskwarrior call counts by purpose so a faster but incorrect path
  cannot pass.
  (`test_taskwarrior_call_count_validation`)
- [x] Reuse one authoritative export across compatible reads and use narrow
  fallbacks only for deliberately absent scope.
  (`test_taskwarrior_call_budget_reuse`)
- [x] Avoid SQLite initialization, schema adoption, or WAL negotiation on
  proven empty thin-hook paths.
  (`test_taskwarrior_sqlite_minimized`)
- [x] Preserve bounded work and responsive diagnostics on slow Termux devices.
  (`test_taskwarrior_diagnostics_bounded`)
- [ ] Run enforced profiles on desktop and both Termux devices after the final
  deletion pass.

All tests use local in-memory storage and are deterministic.