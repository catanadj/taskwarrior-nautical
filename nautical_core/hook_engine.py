from __future__ import annotations


def handle_on_add(
    request,
    *,
    json_result_cls,
    core_ref,
    task_has_nautical_fields,
    load_core,
    diag,
    fail_and_exit,
    build_on_add_context,
    stamp_chain_id_on_add,
    handle_anchor_preview_on_add,
    handle_cp_preview_on_add,
) -> None:
    task = request.task
    prof = request.prof
    runtime = request.runtime
    if not task_has_nautical_fields(task):
        return json_result_cls(task=task, sanitize=False, prof=prof)

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
        return json_result_cls(task=task, sanitize=True, prof=prof)

    stamp_chain_id_on_add(task)
    if ctx.kind == 'anchor':
        handle_anchor_preview_on_add(ctx, prof=prof)
        return None

    handle_cp_preview_on_add(ctx, prof=prof)
    return None



def handle_on_exit(
    request,
    *,
    exit_result_cls,
    redirect_stdout_to_devnull,
    drain_queue,
    strict_exit_result,
):
    _ = request.runtime
    redirect_stdout_to_devnull()
    stats = drain_queue()
    strict_msg = strict_exit_result(stats)
    if strict_msg:
        return exit_result_cls(exit_code=1, feedback_message=strict_msg, stats=stats)
    return exit_result_cls(exit_code=0, stats=stats)



def handle_on_modify(
    request,
    *,
    json_result_cls,
    task_has_nautical_fields,
    load_core,
    diag,
    fail_and_exit,
    is_non_completion_modify,
    handle_non_completion_modify,
    handle_completion_modify,
):
    old, new = request.old, request.new

    if (new.get('status') or '').lower() == 'deleted':
        return json_result_cls(task=new, sanitize=False)

    if not task_has_nautical_fields(old, new):
        return json_result_cls(task=new, sanitize=False)

    try:
        load_core()
    except Exception as exc:
        diag(f'core load failed: {exc}')
        fail_and_exit('Hook misconfigured', 'Failed to initialize nautical core')

    if is_non_completion_modify(old, new):
        handle_non_completion_modify(old, new)
        return None

    handle_completion_modify(old, new)
    return None
