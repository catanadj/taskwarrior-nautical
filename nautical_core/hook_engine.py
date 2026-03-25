from __future__ import annotations


def handle_on_add(
    task: dict,
    *,
    prof,
    runtime,
    core_ref,
    task_has_nautical_fields,
    load_core,
    diag,
    fail_and_exit,
    emit_task_json,
    build_on_add_context,
    stamp_chain_id_on_add,
    handle_anchor_preview_on_add,
    handle_cp_preview_on_add,
) -> None:
    if not task_has_nautical_fields(task):
        emit_task_json(task, sanitize=False)
        return

    try:
        load_core()
    except Exception as exc:
        diag(f'core load failed: {exc}')
        fail_and_exit('Hook misconfigured', 'Failed to initialize nautical core')
    try:
        if getattr(prof, 'enabled', False) and runtime.import_ms is not None:
            prof.import_ms = runtime.import_ms
    except Exception:
        pass

    with prof.section('clock:now'):
        core = core_ref()
        now_utc = core.now_utc()
        now_local = core.to_local(now_utc)

    ctx = build_on_add_context(task, now_utc, now_local, prof=prof)
    if not ctx.kind:
        emit_task_json(task, sanitize=True, prof=prof)
        return

    stamp_chain_id_on_add(task)
    if ctx.kind == 'anchor':
        handle_anchor_preview_on_add(ctx, prof=prof)
        return

    handle_cp_preview_on_add(ctx, prof=prof)



def handle_on_exit(
    *,
    runtime,
    redirect_stdout_to_devnull,
    drain_queue,
    emit_stats_diag,
    strict_exit_result,
    emit_exit_feedback,
) -> int:
    redirect_stdout_to_devnull()
    stats = drain_queue()
    emit_stats_diag(stats)
    strict_msg = strict_exit_result(stats)
    if strict_msg:
        emit_exit_feedback(strict_msg)
        return 1
    return 0
