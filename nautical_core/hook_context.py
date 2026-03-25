from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable


@dataclass(slots=True)
class HookRuntimeContext:
    hook_name: str
    taskdata_dir: str
    use_rc_data_location: bool
    tw_dir: str
    hook_dir: str
    profile_level: int = 0
    import_ms: float | None = None


@dataclass(slots=True)
class OnAddContext:
    task: dict[str, Any]
    now_utc: datetime
    now_local: datetime
    cp_str: str
    anchor_str: str
    kind: str | None
    chain_state: str
    until_dt: datetime | None
    user_provided_due: bool
    due_dt: datetime
    past_due_warning: str | None
    due_day: Any
    due_hhmm: tuple[int, int]


def build_hook_runtime_context(
    *,
    hook_name: str,
    taskdata_dir: str,
    use_rc_data_location: bool,
    tw_dir: str,
    hook_dir: str,
    profile_level: int = 0,
    import_ms: float | None = None,
) -> HookRuntimeContext:
    return HookRuntimeContext(
        hook_name=hook_name,
        taskdata_dir=taskdata_dir,
        use_rc_data_location=bool(use_rc_data_location),
        tw_dir=tw_dir,
        hook_dir=hook_dir,
        profile_level=int(profile_level or 0),
        import_ms=float(import_ms) if import_ms is not None else None,
    )


def build_on_add_context(
    task: dict[str, Any],
    now_utc: datetime,
    now_local: datetime,
    *,
    validate_kind_not_conflicting: Callable[[str, str], tuple[bool, str]],
    kind_and_defaults_on_add: Callable[[dict[str, Any], str, str], tuple[str | None, str]],
    validate_chain_limits_on_add: Callable[[dict[str, Any], datetime], datetime | None],
    due_context_on_add: Callable[
        [dict[str, Any], datetime],
        tuple[bool, datetime, str | None, Any, tuple[int, int]],
    ],
) -> OnAddContext:
    cp_str = (task.get('cp') or '').strip()
    anchor_str = (task.get('anchor') or '').strip()
    is_valid, err = validate_kind_not_conflicting(cp_str, anchor_str)
    if not is_valid:
        raise ValueError(err)

    kind, chain_state = kind_and_defaults_on_add(task, cp_str, anchor_str)
    if not kind:
        until_dt = None
        user_provided_due = bool(task.get('due'))
        due_dt = now_utc
        past_due_warning = None
        due_local = now_local if isinstance(now_local, datetime) else now_utc
        due_day = due_local.date()
        due_hhmm = (due_local.hour, due_local.minute)
    else:
        until_dt = validate_chain_limits_on_add(task, now_utc)
        (
            user_provided_due,
            due_dt,
            past_due_warning,
            due_day,
            due_hhmm,
        ) = due_context_on_add(task, now_utc)
    return OnAddContext(
        task=task,
        now_utc=now_utc,
        now_local=now_local,
        cp_str=cp_str,
        anchor_str=anchor_str,
        kind=kind,
        chain_state=chain_state,
        until_dt=until_dt,
        user_provided_due=user_provided_due,
        due_dt=due_dt,
        past_due_warning=past_due_warning,
        due_day=due_day,
        due_hhmm=due_hhmm,
    )
