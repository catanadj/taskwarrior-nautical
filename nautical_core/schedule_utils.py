from __future__ import annotations

from datetime import date, timedelta


def roll_apply(dt: date, mods: dict, *, parse_error_cls) -> date:
    roll = mods.get("roll")

    if roll in ("pbd", "nbd", "nw"):
        if dt.weekday() > 4:
            if roll == "pbd":
                for _ in range(8):
                    if dt.weekday() <= 4:
                        break
                    dt -= timedelta(days=1)
                else:
                    raise parse_error_cls("roll_apply: failed to reach business day (pbd)")
            elif roll == "nbd":
                for _ in range(8):
                    if dt.weekday() <= 4:
                        break
                    dt += timedelta(days=1)
                else:
                    raise parse_error_cls("roll_apply: failed to reach business day (nbd)")
            else:
                prev_dt = dt
                next_dt = dt
                for _ in range(8):
                    if prev_dt.weekday() <= 4:
                        break
                    prev_dt -= timedelta(days=1)
                else:
                    raise parse_error_cls("roll_apply: failed to reach business day (nw prev)")
                for _ in range(8):
                    if next_dt.weekday() <= 4:
                        break
                    next_dt += timedelta(days=1)
                else:
                    raise parse_error_cls("roll_apply: failed to reach business day (nw next)")
                dt = prev_dt if (dt - prev_dt) <= (next_dt - dt) else next_dt

    elif roll in ("next-wd", "prev-wd"):
        tgt = mods.get("wd")
        if tgt is not None:
            if roll == "next-wd":
                dt += timedelta(days=1)
                for _ in range(8):
                    if dt.weekday() == tgt:
                        break
                    dt += timedelta(days=1)
                else:
                    raise parse_error_cls("roll_apply: failed to reach target weekday (next-wd)")
            else:
                dt -= timedelta(days=1)
                for _ in range(8):
                    if dt.weekday() == tgt:
                        break
                    dt -= timedelta(days=1)
                else:
                    raise parse_error_cls("roll_apply: failed to reach target weekday (prev-wd)")

    return dt


def weeks_between(d1: date, d2: date) -> int:
    iso1 = d1.isocalendar()
    iso2 = d2.isocalendar()
    mon1 = date.fromisocalendar(iso1.year, iso1.week, 1)
    mon2 = date.fromisocalendar(iso2.year, iso2.week, 1)
    return (mon2 - mon1).days // 7


def apply_day_offset(d: date, mods: dict) -> date:
    off = int(mods.get("day_offset", 0) or 0)
    return d + timedelta(days=off) if off else d


def expr_has_m_or_y(dnf) -> bool:
    for term in dnf:
        for atom in term:
            if atom["typ"] in ("m", "y"):
                return True
    return False


def pick_hhmm_from_dnf_for_date(
    dnf,
    target: date,
    default_seed: date,
    *,
    atom_matches_on,
):
    for term in dnf:
        if all(atom_matches_on(atom, target, default_seed) for atom in term):
            for atom in term:
                tval = atom["mods"].get("t")
                if not tval:
                    continue
                if isinstance(tval, list):
                    return tval[0] if tval else None
                return tval
    return None
