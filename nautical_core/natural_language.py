from __future__ import annotations

import os
import re
from calendar import month_name


_WDNAME = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
_MONTH_ABBR = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
_MONTH_FULL = list(month_name)
_WD_INDEX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
_WD_FULL = {
    "mon": "Monday",
    "tue": "Tuesday",
    "wed": "Wednesday",
    "thu": "Thursday",
    "fri": "Friday",
    "sat": "Saturday",
    "sun": "Sunday",
}


def ordinal(n: int) -> str:
    n = int(n)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def term_collect_mods(term: list) -> dict:
    merged = {}
    for atom in term:
        mods = atom.get("mods") or {}
        for key, value in mods.items():
            merged[key] = value
    return merged


def fmt_hhmm_for_term(term: list, default_due_dt):
    _ = default_due_dt
    tmod = term_collect_mods(term).get("t")
    if isinstance(tmod, tuple):
        return f"{tmod[0]:02d}:{tmod[1]:02d}"
    if isinstance(tmod, list):
        parts = []
        for value in tmod:
            if isinstance(value, tuple) and len(value) == 2:
                parts.append(f"{value[0]:02d}:{value[1]:02d}")
        return ", ".join(parts) if parts else None
    if isinstance(tmod, str) and tmod:
        return tmod
    return None


def fmt_weekdays_list(spec: str, *, expand_weekly_aliases, split_csv_lower, wday_idx_any) -> str:
    spec = expand_weekly_aliases(spec)
    tokens = split_csv_lower(spec)
    if not tokens:
        return ""

    plural = {
        0: "Mondays",
        1: "Tuesdays",
        2: "Wednesdays",
        3: "Thursdays",
        4: "Fridays",
        5: "Saturdays",
        6: "Sundays",
    }

    names: list[str] = []
    for token in tokens:
        if token == "rand":
            names.append("one random day each week")
            continue

        if ".." in token:
            left, right = token.split("..", 1)
            i_left, i_right = wday_idx_any(left), wday_idx_any(right)
            if i_left is None or i_right is None:
                continue
            if i_left == i_right:
                names.append(plural[i_left])
            else:
                names.append(f"{plural[i_left]} through {plural[i_right]}")
            continue

        idx = wday_idx_any(token)
        if idx is not None:
            names.append(plural[idx])

    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " or " + names[-1]


def fmt_monthly_atom(
    spec: str,
    *,
    monthly_alias: dict,
    safe_match,
    nth_wd_re,
    bd_re,
) -> str:
    text = (spec or "").lower().strip()
    if text in monthly_alias:
        text = monthly_alias[text]
    if text == "rand":
        return "one random day each month"

    match = safe_match(nth_wd_re, text)
    if match:
        idx, wd = match.group(1), match.group(2)
        name = _WDNAME[_WD_INDEX[wd]]
        if idx == "last":
            return f"the last {name} of each month"
        nth = int(re.sub(r"(st|nd|rd|th)$", "", idx))
        if nth < 0:
            return f"the {ordinal(abs(nth))} last {name} of each month"
        return f"the {ordinal(nth)} {name} of each month"

    match = bd_re.match(text)
    if match:
        nth = int(match.group(1))
        if nth > 0:
            return f"the {ordinal(nth)} business day of each month"
        if nth == -1:
            return "the last business day of each month"
        return f"the {ordinal(abs(nth))} last business day of each month"

    if ".." in text:
        left, right = text.split("..", 1)
        try:
            i_left = int(left)
            i_right = int(right)
            if i_left > 0 and i_right > 0:
                return f"days {i_left}–{i_right} of each month"

            def _dword(value: int) -> str:
                if value == -1:
                    return "last day"
                return ordinal(value) if value > 0 else f"{ordinal(abs(value))} last day"

            return f"days {_dword(i_left)}–{_dword(i_right)} of each month"
        except Exception:
            pass

    try:
        nth = int(text)
        if nth == -1:
            return "the last day of each month"
        if nth < 0:
            return f"the {ordinal(abs(nth))} last day of each month"
        return f"the {ordinal(nth)} day of each month"
    except Exception:
        return f"[unknown monthly token '{spec}']"


def fmt_yearly_atom(
    tok: str,
    *,
    rand_mm_re,
    md_range_re,
    yearfmt,
) -> str:
    text = (tok or "").strip().lower()

    if text == "rand":
        return "one random day each year"

    match_rand_month = rand_mm_re.fullmatch(text)
    if match_rand_month:
        month = int(match_rand_month.group(1))
        if 1 <= month <= 12:
            return f"one random day in {_MONTH_ABBR[month - 1]} each year"

    match = md_range_re.fullmatch(text)
    if not match:
        return tok

    def _pair(a: int, b: int) -> tuple[int, int]:
        return (b, a) if yearfmt() == "MD" else (a, b)

    a, b = int(match.group(1)), int(match.group(2))
    if match.group(3):
        c, d = int(match.group(3)), int(match.group(4))
        d1, m1 = _pair(a, b)
        d2, m2 = _pair(c, d)
        if m1 == m2 and d1 == 1 and 28 <= d2 <= 31:
            return f"{_MONTH_ABBR[m1 - 1]} each year"
        if m1 == m2:
            if yearfmt() == "DM":
                return f"{d1}–{d2} {_MONTH_ABBR[m1 - 1]} each year"
            return f"{_MONTH_ABBR[m1 - 1]} {d1}–{d2} each year"
        if d1 == 1 and 28 <= d2 <= 31:
            return f"{_MONTH_ABBR[m1 - 1]}–{_MONTH_ABBR[m2 - 1]} each year"
        if yearfmt() == "DM":
            left = f"{d1} {_MONTH_ABBR[m1 - 1]}"
            right = f"{d2} {_MONTH_ABBR[m2 - 1]}"
        else:
            left = f"{_MONTH_ABBR[m1 - 1]} {d1}"
            right = f"{_MONTH_ABBR[m2 - 1]} {d2}"
        return f"{left}–{right} each year"

    d1, m1 = _pair(a, b)
    if m1 == 2 and d1 == 29:
        return "Feb 29 each leap year"
    if yearfmt() == "DM":
        return f"{d1} {_MONTH_ABBR[m1 - 1]} each year"
    return f"{_MONTH_ABBR[m1 - 1]} {d1} each year"


def describe_monthly_tokens(spec: str, *, split_csv_lower):
    return split_csv_lower(spec)


def describe_is_pure_nth_weekday_spec(spec: str, *, split_csv_lower, safe_match, nth_wd_re):
    toks = describe_monthly_tokens(spec, split_csv_lower=split_csv_lower)
    if not toks:
        return False, []
    out = []
    for tok in toks:
        match = safe_match(nth_wd_re, tok)
        if not match:
            return False, []
        n_raw, wd = match.group(1), match.group(2)
        if n_raw == "last":
            nth = -1
        else:
            nth = int(re.sub(r"(st|nd|rd|th)$", "", n_raw))
        out.append((nth, wd))
    return True, out


def describe_is_pure_dom_spec(spec: str, *, split_csv_lower):
    toks = describe_monthly_tokens(spec, split_csv_lower=split_csv_lower)
    if not toks:
        return False, []
    out = []
    for tok in toks:
        if not tok.isdigit():
            return False, []
        day = int(tok)
        if day < 1 or day > 31:
            return False, []
        out.append(day)
    return True, out


def describe_single_full_month_from_yearly_spec(spec: str, *, year_range_colon_re):
    match = year_range_colon_re.match(str(spec or "").strip())
    if not match:
        return None
    d1, m1, d2, m2 = map(int, match.groups())
    if m1 != m2 or d1 != 1:
        return None
    if d2 < 28 or d2 > 31:
        return None
    return m1


def describe_term_roll_shift(term) -> str | None:
    saw = set()
    for atom in term:
        roll = (atom.get("mods") or {}).get("roll")
        if roll in ("nw", "pbd", "nbd"):
            saw.add(roll)
    if "nw" in saw:
        return "nw"
    if "pbd" in saw:
        return "pbd"
    if "nbd" in saw:
        return "nbd"
    return None


def describe_term_bd_filter(term) -> bool:
    return any((atom.get("mods") or {}).get("bd") for atom in term)


def describe_roll_suffix(roll: str) -> str:
    if roll == "pbd":
        return " if business day; otherwise the previous business day"
    if roll == "nbd":
        return " if business day; otherwise the next business day"
    if roll == "nw":
        return " if business day; otherwise the nearest business day (Fri if Saturday, Mon if Sunday)"
    return ""


def describe_inject_schedule_suffixes(txt: str, term) -> str:
    roll = describe_term_roll_shift(term)
    if roll:
        suffix = describe_roll_suffix(roll)
    elif describe_term_bd_filter(term):
        suffix = " only if a business day (skipped if weekend)"
    else:
        suffix = ""

    if not suffix:
        return txt

    targets = [
        "the last day of each month",
        "the first day of each month",
        "the last day of the month",
        "the first day of the month",
        "the last day of each quarter",
        "the first day of each quarter",
    ]
    for target in targets:
        if target in txt:
            return txt.replace(target, target + suffix)

    if " at " in txt:
        head, _sep, tail = txt.partition(" at ")
        return f"{head}{suffix} at {tail}"
    return txt + suffix


def describe_anchor_term_collect(
    term,
    *,
    fmt_weekdays_list,
    split_csv_tokens,
    fmt_monthly_atom,
    fmt_yearly_atom,
):
    m_parts = []
    y_parts = []
    w_phrase = None
    bd_filter = False
    wk_ival = mo_ival = yr_ival = 1
    monthly_specs = []
    yearly_specs = []

    for atom in term:
        typ = (atom.get("typ") or atom.get("type") or "").lower()
        spec = str(atom.get("spec") or atom.get("value") or "").strip().lower()
        ival = int(atom.get("ival") or atom.get("intv") or 1)

        if typ == "w":
            wk_ival = max(wk_ival, ival)
            w_phrase = fmt_weekdays_list(spec)
            if wk_ival > 1 and spec == "rand":
                w_phrase = f"one random day every {wk_ival} weeks"
        elif typ == "m":
            mo_ival = max(mo_ival, ival)
            monthly_specs.append(spec)
            for tok in split_csv_tokens(spec):
                m_parts.append(fmt_monthly_atom(tok))
        elif typ == "y":
            yr_ival = max(yr_ival, ival)
            yearly_specs.append(spec)
            qmap = atom.get("_qmap") or {}
            for tok in split_csv_tokens(spec):
                phr = fmt_yearly_atom(tok)
                if phr and qmap and tok in qmap and not phr.startswith("one random day"):
                    phr = f"{phr} ({qmap[tok]})"
                y_parts.append(phr)

        mods = atom.get("mods") or {}
        bd_filter = bd_filter or bool(mods.get("bd") or (mods.get("wd") is True))

    return w_phrase, m_parts, y_parts, bd_filter, wk_ival, mo_ival, yr_ival, monthly_specs, yearly_specs


def describe_anchor_term_fused_month_year(
    term,
    default_due_dt,
    monthly_specs,
    yearly_specs,
    yr_ival: int,
    bd_filter: bool,
    m_parts: list[str],
    *,
    describe_is_pure_nth_weekday_spec,
    describe_single_full_month_from_yearly_spec,
    fmt_hhmm_for_term,
):
    if len(monthly_specs) != 1 or len(yearly_specs) != 1:
        return None
    mspec = monthly_specs[0]
    yspec = yearly_specs[0]
    is_nth, pairs = describe_is_pure_nth_weekday_spec(mspec)
    fuse_month = describe_single_full_month_from_yearly_spec(yspec)
    if not (is_nth and fuse_month and len(pairs) == 1):
        return None
    nth, wd = pairs[0]
    if nth < 0:
        nth_txt = "last" if nth == -1 else f"{ordinal(abs(nth))} last"
    else:
        nth_txt = ordinal(nth)
    main = f"the {nth_txt} {_WD_FULL[wd]} of {_MONTH_FULL[fuse_month]}"
    hhmm = fmt_hhmm_for_term(term, default_due_dt)
    if yr_ival > 1:
        main = f"{main} every {yr_ival} years"
    if hhmm:
        main = f"{main} at {hhmm}"
    if bd_filter and any("random day each month" in part for part in m_parts):
        main = f"{main} on a business day"
    return describe_inject_schedule_suffixes(main, term)


def describe_anchor_term_interval_prefix(
    wk_ival,
    mo_ival,
    yr_ival,
    monthly_specs,
    *,
    describe_is_pure_nth_weekday_spec,
    describe_is_pure_dom_spec,
):
    interval_prefix = None
    suppress_tail = False

    if wk_ival > 1:
        interval_prefix = f"every {wk_ival} weeks: "
    elif mo_ival > 1:
        monthly_prefix = f"every {mo_ival} months"
        clarifier = ""
        if len(monthly_specs) == 1:
            mspec = monthly_specs[0]
            is_nth, pairs = describe_is_pure_nth_weekday_spec(mspec)
            if is_nth:
                if len(pairs) == 1:
                    nth, wd = pairs[0]
                    if nth < 0:
                        nth_txt = "last" if nth == -1 else f"{ordinal(abs(nth))} last"
                    else:
                        nth_txt = ordinal(nth)
                    clarifier = f" among months that have the {nth_txt} {_WD_FULL[wd]}"
                else:
                    clarifier = " among months that satisfy the specified nth-weekdays"
            else:
                is_dom, doms = describe_is_pure_dom_spec(mspec)
                if is_dom and any(day >= 29 for day in doms):
                    clarifier = (
                        f" among months that have day {doms[0]}"
                        if len(doms) == 1
                        else " among months that have those days"
                    )

        if clarifier:
            interval_prefix = monthly_prefix + clarifier
            suppress_tail = True
        else:
            interval_prefix = monthly_prefix + ": "
    elif yr_ival > 1:
        interval_prefix = f"every {yr_ival} years: "

    return interval_prefix, suppress_tail


def describe_anchor_term_parts(w_phrase, m_parts, y_parts, bd_filter: bool) -> list[str]:
    parts = []
    if w_phrase:
        parts.append(w_phrase)

    if m_parts:
        monthly = ", ".join(m_parts)
        if not w_phrase:
            parts.append(monthly)
        else:
            parts.append(f"that fall on {monthly}")

    if y_parts:
        yearly = " or ".join(y_parts) if len(y_parts) > 1 else y_parts[0]
        if yearly.startswith("one random day"):
            parts.append(yearly)
        elif w_phrase or m_parts:
            parts.append(f"and within {yearly}")
        else:
            parts.append(yearly)

    if bd_filter and any("random day each month" in part for part in m_parts):
        parts.append("on a business day")
    return parts


def term_prevnext_wd(term, *, wdname: dict) -> tuple[str, str] | None:
    for atom in term:
        mods = atom.get("mods") or {}
        roll = mods.get("roll")
        if roll in ("next-wd", "prev-wd"):
            wd = mods.get("wd")
            if wd is not None:
                return ("next" if roll == "next-wd" else "prev", wdname.get(wd, ""))
    return None


def inject_prevnext_phrase(txt: str, term, *, wdname: dict) -> str:
    tup = term_prevnext_wd(term, wdname=wdname)
    if not tup:
        return txt

    dir_word, dayname = tup
    rel = "before" if dir_word == "prev" else "after"
    adj = "previous" if dir_word == "prev" else "next"

    targets = [
        "the last day of each month",
        "the first day of each month",
        "the last day of the month",
        "the first day of the month",
        "the last day of each quarter",
        "the first day of each quarter",
    ]

    for target in targets:
        if target in txt:
            return txt.replace(target, f"the {adj} {dayname} {rel} {target}")

    phrase = f", then the {adj} {dayname}"
    if " at " in txt:
        head, sep, tail = txt.partition(" at ")
        return f"{head}{phrase} at {tail}"
    return txt + phrase


def describe_anchor_term(
    term: list,
    default_due_dt=None,
    *,
    fmt_weekdays_list,
    split_csv_tokens,
    fmt_monthly_atom,
    fmt_yearly_atom,
    describe_is_pure_nth_weekday_spec,
    describe_single_full_month_from_yearly_spec,
    fmt_hhmm_for_term,
    describe_is_pure_dom_spec,
):
    (
        w_phrase,
        m_parts,
        y_parts,
        bd_filter,
        wk_ival,
        mo_ival,
        yr_ival,
        monthly_specs,
        yearly_specs,
    ) = describe_anchor_term_collect(
        term,
        fmt_weekdays_list=fmt_weekdays_list,
        split_csv_tokens=split_csv_tokens,
        fmt_monthly_atom=fmt_monthly_atom,
        fmt_yearly_atom=fmt_yearly_atom,
    )

    fused = describe_anchor_term_fused_month_year(
        term,
        default_due_dt,
        monthly_specs,
        yearly_specs,
        yr_ival,
        bd_filter,
        m_parts,
        describe_is_pure_nth_weekday_spec=describe_is_pure_nth_weekday_spec,
        describe_single_full_month_from_yearly_spec=describe_single_full_month_from_yearly_spec,
        fmt_hhmm_for_term=fmt_hhmm_for_term,
    )
    if fused is not None:
        return fused

    interval_prefix, suppress_tail = describe_anchor_term_interval_prefix(
        wk_ival,
        mo_ival,
        yr_ival,
        monthly_specs,
        describe_is_pure_nth_weekday_spec=describe_is_pure_nth_weekday_spec,
        describe_is_pure_dom_spec=describe_is_pure_dom_spec,
    )
    parts = describe_anchor_term_parts(w_phrase, m_parts, y_parts, bd_filter)
    hhmm = fmt_hhmm_for_term(term, default_due_dt)

    if suppress_tail:
        txt = interval_prefix
        if hhmm:
            txt = f"{txt} at {hhmm}"
        return describe_inject_schedule_suffixes(txt or "any day", term)

    if hhmm:
        parts.append(f"at {hhmm}")
    txt = " ".join(part for part in parts if part)
    if interval_prefix:
        txt = interval_prefix + txt

    txt = inject_prevnext_phrase(txt, term, wdname=_WDNAME)
    txt = describe_inject_schedule_suffixes(txt or "any day", term)
    return txt or "any day"


def describe_anchor_expr_from_dnf(dnf: list, default_due_dt=None, *, describe_anchor_term) -> str:
    nat_terms = []
    for term in dnf or []:
        try:
            text = describe_anchor_term(term, default_due_dt=default_due_dt)
        except Exception:
            text = ""
        if text:
            nat_terms.append(text)

    if not nat_terms:
        return ""

    seen = set()
    ordered = []
    for text in nat_terms:
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered[0] if len(ordered) == 1 else compress_natural_or_terms(ordered)


def describe_anchor_expr(anchor_expr: str, default_due_dt=None, *, parse_anchor_expr_to_dnf_cached, describe_anchor_expr_from_dnf) -> str:
    if not anchor_expr or not str(anchor_expr).strip():
        return ""
    try:
        dnf = parse_anchor_expr_to_dnf_cached(anchor_expr)
    except Exception:
        return ""
    return describe_anchor_expr_from_dnf(dnf, default_due_dt=default_due_dt)


def join_natural_or_terms(terms: list[str]) -> str:
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"either {terms[0]} or {terms[1]}"
    return "either " + ", ".join(terms[:-1]) + ", or " + terms[-1]


def join_plain_or_terms(terms: list[str]) -> str:
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"{terms[0]} or {terms[1]}"
    return ", ".join(terms[:-1]) + ", or " + terms[-1]


def longest_common_suffix(parts: list[str]) -> str:
    if not parts:
        return ""
    rev = [part[::-1] for part in parts if isinstance(part, str)]
    if not rev:
        return ""
    prefix = os.path.commonprefix(rev)
    return prefix[::-1]


def compress_or_terms_by_shared_rest(terms: list[str], delim: str) -> str | None:
    if not terms or len(terms) < 2:
        return None
    if not isinstance(delim, str) or not delim:
        return None

    split: list[tuple[str, str]] = []
    for term in terms:
        if not isinstance(term, str):
            return None
        idx = term.find(delim)
        if idx <= 0:
            return None
        prefix = term[:idx].strip()
        rest = term[idx + len(delim):].strip()
        if not prefix or not rest:
            return None
        split.append((prefix, rest))

    rests = {rest for _, rest in split}
    if len(rests) != 1:
        return None
    prefixes = [prefix for prefix, _ in split]
    if len(set(prefixes)) <= 1:
        return None
    rest = split[0][1]
    prefix_text = join_plain_or_terms(prefixes)
    if delim == " and within ":
        prep = "on" if re.search(r"\b\d{1,2}\b", rest) else "in"
        return f"{prefix_text} {prep} {rest}"
    return f"{prefix_text}{delim}{rest}"


def compress_or_terms_by_clause(terms: list[str], delim: str) -> str | None:
    if not terms or len(terms) < 2:
        return None
    if not isinstance(delim, str) or not delim:
        return None

    split: list[tuple[str, str]] = []
    for term in terms:
        if not isinstance(term, str):
            return None
        idx = term.find(delim)
        if idx <= 0:
            return None
        prefix = term[:idx]
        rest = term[idx + len(delim):]
        if not rest:
            return None
        split.append((prefix, rest))

    prefixes = {prefix for prefix, _ in split}
    if len(prefixes) != 1:
        return None
    prefix = split[0][0]
    rests = [rest for _, rest in split]
    if len(set(rests)) <= 1:
        return None

    common_tail = longest_common_suffix(rests)
    if delim == " and within ":
        if common_tail and re.match(r"^\s+\d{1,2}\b", common_tail) and "each year" in common_tail:
            alt_tail = " each year"
            if all(rest.endswith(alt_tail) for rest in rests):
                common_tail = alt_tail

    variants: list[str] = []
    for rest in rests:
        variant = rest[:-len(common_tail)] if common_tail else rest
        variant = variant.strip(" ,")
        if not variant:
            return None
        variants.append(variant)

    return f"{prefix}{delim}{join_natural_or_terms(variants)}{common_tail}"


def compress_natural_or_terms(terms: list[str]) -> str:
    return (
        compress_or_terms_by_shared_rest(terms, " and within ")
        or compress_or_terms_by_shared_rest(terms, " that fall on ")
        or compress_or_terms_by_clause(terms, " and within ")
        or compress_or_terms_by_clause(terms, " that fall on ")
        or join_natural_or_terms(terms)
    )


def normalize_range_token(tok: str, *, safe_match, int_range_re) -> str | None:
    text = (tok or "").strip().lower()
    match = safe_match(int_range_re, text)
    if not match:
        return None
    left, right = [int(x) for x in text.split("..")]
    return f"{left}–{right}"


def rand_bucket_time_from_mods(mods: dict) -> str | None:
    tmod = mods.get("t")
    if isinstance(tmod, tuple):
        return f"{tmod[0]:02d}:{tmod[1]:02d}"
    if isinstance(tmod, str) and tmod:
        return tmod
    return None


def rand_bucket_merge_mods(mods: dict, time_str: str | None, bd_flag: bool) -> tuple[str | None, bool]:
    if time_str is None:
        time_str = rand_bucket_time_from_mods(mods)
    bd_flag = bd_flag or bool(mods.get("bd") or (mods.get("wd") is True))
    return time_str, bd_flag


def rand_bucket_signature(term: list[dict], *, normalize_range_token) -> tuple | None:
    has_rand = False
    range_norm = None
    ival_seen = None
    time_str = None
    bd_flag = False

    for atom in term:
        typ = (atom.get("typ") or atom.get("type") or "").lower()
        if typ in ("w", "y") or typ != "m":
            return None
        spec = str(atom.get("spec") or atom.get("value") or "").lower()
        ival = int(atom.get("ival") or atom.get("intv") or 1)
        ival_seen = ival if ival_seen is None else ival_seen
        mods = atom.get("mods") or {}
        time_str, bd_flag = rand_bucket_merge_mods(mods, time_str, bd_flag)
        if spec == "rand":
            has_rand = True
            continue
        norm = normalize_range_token(spec)
        if not norm:
            return None
        if range_norm and norm != range_norm:
            return None
        range_norm = norm

    if not (has_rand and range_norm):
        return None
    return (ival_seen or 1, time_str, bd_flag, range_norm)


def try_bucket_rand_monthly(dnf: list[list[dict]], task: dict, *, rand_bucket_signature) -> str | None:
    _ = task
    if not dnf or any(len(term) == 0 for term in dnf):
        return None

    sig = None
    ranges = []
    for term in dnf:
        result = rand_bucket_signature(term)
        if not result:
            return None
        cur_sig = (result[0], result[1], result[2])
        if sig is None:
            sig = cur_sig
        elif cur_sig != sig:
            return None
        ranges.append(result[3])

    def _start_val(range_text):
        left = range_text.split("–", 1)[0]
        try:
            return int(left)
        except Exception:
            return 0

    ranges = sorted(ranges, key=_start_val)

    if sig is None:
        return None
    ival, time_str, bd_flag = sig
    parts = []
    lead = "one random "
    if bd_flag:
        lead += "business "
    lead += "day"
    if ival and int(ival) > 1:
        lead = f"every {ival} months: " + lead
    parts.append(lead + " in each monthly bucket")
    parts.append(f"({', '.join([f'days {rng}' for rng in ranges])})")
    if time_str:
        parts.append(f"at {time_str}")
    return " ".join(parts)


def describe_anchor_dnf(
    dnf: list,
    task: dict,
    *,
    try_bucket_rand_monthly,
    parse_dt_any,
    describe_anchor_term,
):
    def _mode_tail(mode: str) -> str:
        if mode == "all":
            return "backfill all missed anchors"
        if mode == "flex":
            return "skip past anchors; respect future anchors"
        if mode == "skip":
            return "skip missed anchors"
        return ""

    bucket = try_bucket_rand_monthly(dnf, task)
    if bucket:
        mode = (task.get("anchor_mode") or "skip").lower()
        tail = _mode_tail(mode)
        return f"{bucket}; {tail}" if tail else bucket

    due_dt = parse_dt_any(task.get("due")) if task else None
    terms = [describe_anchor_term(term, due_dt) for term in (dnf or [])]
    if not terms:
        return ""

    seen = set()
    uniq_terms = []
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            uniq_terms.append(term)
    if not uniq_terms:
        return ""

    sentence = compress_natural_or_terms(uniq_terms)
    mode = (task.get("anchor_mode") or "skip").lower()
    tail = _mode_tail(mode)
    return f"{sentence}; {tail}" if tail else sentence
