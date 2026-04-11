from __future__ import annotations


def iter_y_segments(s: str, *, re_mod):
    """
    Yield the raw yearly-spec segments that follow 'y:' up to the next
    term delimiter (+, |, ) or end. We don't fully parse here; it's
    just for linting.
    """
    for match in re_mod.finditer(r"y\s*:\s*([^\+\|\)]*)", s):
        yield (match.group(1) or "").strip()


def lint_expand_year_month_aliases(s: str, *, month_from_alias, year_full_month_range_token, re_mod) -> str:
    # Allow bare month aliases: replace 'y:jun' with a canonical monthly window for linting.
    def _lint_month_alias_sub(match):
        mm = month_from_alias(match.group(1))
        if not mm:
            return match.group(0)
        return f"y:{year_full_month_range_token(mm)}"

    # Allow bare month aliases ONLY when they are not part of a numeric day-month like 'y:01-13'.
    #  - 'y:jan' or 'y:03' -> expand to full month window
    #  - do NOT touch 'y:01-13' / 'y:jun-01' etc.
    s = re_mod.sub(r"\by:([a-z]{3})(?=\b(?!-)|[,+|()])", _lint_month_alias_sub, s)
    s = re_mod.sub(r"\by:(\d{2})(?=(?:\b(?!-)|[,+|()]))", _lint_month_alias_sub, s)
    return s


def lint_check_weekly_delimiter_contract(s: str, *, re_mod) -> str | None:
    if re_mod.search(
        r"\bw(?:/\d+)?\s*:\s*(?:mon|tue|wed|thu|fri|sat|sun)\s*-\s*(?:mon|tue|wed|thu|fri|sat|sun)\b",
        s,
    ):
        return "Weekly ranges must use '..' (e.g., 'w:mon..fri')."
    if re_mod.search(
        r"\bw(?:/\d+)?\s*:\s*(?:mon|tue|wed|thu|fri|sat|sun)(?:\s*:\s*(?:mon|tue|wed|thu|fri|sat|sun))+\b",
        s,
    ):
        return "Weekly ranges must use '..' (e.g., 'w:mon..fri')."
    return None


def lint_check_yearly_segments(s: str, *, yearfmt, iter_y_segments, split_csv_tokens, re_mod) -> str | None:
    fmt = yearfmt()
    for seg in iter_y_segments(s):
        for tok in split_csv_tokens(seg):
            if re_mod.fullmatch(r"\d{2}:\d{2}", tok):
                return "Yearly day/month must use '-', not ':'. Try '05-15' (not '05:15')."
            if ":" in tok:
                return "Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."
            if re_mod.fullmatch(r"\d{2}-\d{2}", tok):
                a, b = tok.split("-")
                x, y = int(a), int(b)
                if fmt == "MD":
                    if x > 12 and 1 <= y <= 12:
                        return f"'{tok}' looks like DD-MM but config expects MM-DD. Try '{y:02d}-{x:02d}'."
                else:
                    if y > 12 and 1 <= x <= 12:
                        return f"'{tok}' looks like MM-DD but config expects DD-MM. Try '{y:02d}-{x:02d}'."
                continue
            if re_mod.fullmatch(r"\d{2}-\d{2}\.\.\d{2}-\d{2}", tok):
                left, right = tok.split("..", 1)
                a, b = left.split("-", 1)
                c, d = right.split("-", 1)
                x, y, u, v = int(a), int(b), int(c), int(d)
                if fmt == "MD":
                    if x > 12 and 1 <= y <= 12:
                        return (
                            f"'{tok}' starts like DD-MM but config expects MM-DD. "
                            f"Try '{y:02d}-{x:02d}..{v:02d}-{u:02d}'."
                        )
                else:
                    if y > 12 and 1 <= x <= 12:
                        return (
                            f"'{tok}' starts like MM-DD but config expects DD-MM. "
                            f"Try '{y:02d}-{x:02d}..{v:02d}-{u:02d}'."
                        )
                continue
    return None


def lint_check_global_md_dm_confusion(s: str, *, yearfmt, re_mod) -> str | None:
    for match in re_mod.finditer(r"\b(\d{2})-(\d{2})(?=([^\d:]|$))", s):
        a, b = int(match.group(1)), int(match.group(2))
        fmt = yearfmt()
        if fmt == "MD":
            if a > 12 and 1 <= b <= 12:
                return f"'{match.group(0)}' looks like DD-MM but config expects MM-DD. Try '{b:02d}-{a:02d}'."
        else:
            if b > 12 and 1 <= a <= 12:
                return f"'{match.group(0)}' looks like MM-DD but config expects DD-MM. Try '{b:02d}-{a:02d}'."
    return None


def lint_check_invalid_weekday_names(s: str, *, wd_abbr, re_mod, difflib_mod) -> str | None:
    wd_set = set(wd_abbr)
    for wd in re_mod.findall(r"\b[a-z]{3,}\b", s):
        if wd in wd_set or wd in ("rand", "rand*"):
            continue
        if re_mod.search(
            rf"(?:^|[\s\+\|,:@-])(w:|@prev-|@next-|last-|1st|2nd|3rd|4th|5th-){wd}\b",
            s,
        ):
            sug = difflib_mod.get_close_matches(wd, list(wd_set), n=1, cutoff=0.6)
            if sug:
                return f"Unknown weekday '{wd}'. Did you mean '{sug[0]}'?"
    return None


def lint_check_nth_weekday_suffixes(s: str, *, re_mod) -> str | None:
    ord_ok = {"1": "1st", "2": "2nd", "3": "3rd", "4": "4th", "5": "5th"}
    for match in re_mod.finditer(r"\b(\d+)(st|nd|rd|th)-([a-z]+)\b", s):
        n, suff, wd = match.group(1), match.group(2), match.group(3)
        if n not in ord_ok:
            return f"Invalid ordinal '{n}{suff}'. Only 1st..5th are supported."
        expect = ord_ok[n]
        if f"{n}{suff}" != expect:
            return f"Did you mean '{expect}-{wd}' instead of '{n}{suff}-{wd}'?"
    return None


def lint_check_unsat_pure_weekly_and(s: str, *, wd_abbr, split_csv_tokens, re_mod) -> str | None:
    wd_set = set(wd_abbr)
    and_terms = [term.strip() for term in re_mod.split(r"\|", s)]
    for term in and_terms:
        atoms = [atom.strip() for atom in re_mod.split(r"\+", term)]
        wsets, only_weekly = [], True
        for atom in atoms:
            match = re_mod.match(r"^w(?:(/\d+)?):([a-z0-9\-\:\,]+)$", atom)
            if not match:
                only_weekly = False
                break
            spec = match.group(2)
            ws = set()
            simple = True
            for tok in split_csv_tokens(spec):
                if "-" in tok or ":" in tok:
                    simple = False
                    break
                if tok in wd_set:
                    ws.add(tok)
            if not simple:
                only_weekly = False
                break
            if ws:
                wsets.append(ws)
        if only_weekly and wsets and not set.intersection(*wsets):
            return (
                "These anchors joined with '+' don't share any possible date. "
                "If you meant 'either/or', join them with ',' or '|'."
            )
    return None


def lint_check_backward_quarter_ranges(s: str, *, re_mod) -> str | None:
    match = re_mod.search(r"\bq([1-4])\s*\.\.\s*q([1-4])\b", s)
    if match and int(match.group(2)) < int(match.group(1)):
        return (
            "Invalid quarter range 'qX..qY': end quarter precedes start quarter. "
            "Split across the year boundary, e.g., 'q4, q1'."
        )
    return None


def lint_collect_warnings(s: str, *, re_mod) -> list[str]:
    warnings: list[str] = []
    if re_mod.search(r"y:[^|+)]*@t=\d{2}:\d{2},", s):
        warnings.append("Multiple @t times inside a single 'y:' atom; ensure each spec has its own @t or use '|'.")
    if re_mod.search(r"(?<!\()\b(?:w|m|y)(?:/\d+)?\s*:\s*[^|()]*,[^|()]*\s*\+", s):
        warnings.append(
            "A comma list joined with '+' only applies the '+' filter to the last list item. "
            "If you want it to apply to every item, group it with parentheses, e.g. '(w:mon,wed,fri) + y:apr'."
        )
    return warnings


def lint_anchor_expr(
    expr: str,
    *,
    unwrap_quotes,
    lint_expand_year_month_aliases,
    lint_check_weekly_delimiter_contract,
    lint_check_yearly_segments,
    lint_check_global_md_dm_confusion,
    lint_check_invalid_weekday_names,
    lint_check_nth_weekday_suffixes,
    lint_check_unsat_pure_weekly_and,
    lint_check_backward_quarter_ranges,
    lint_collect_warnings,
    re_mod,
) -> tuple[str | None, list[str]]:
    s = unwrap_quotes(expr or "").strip().lower()
    if len(s) > 1024:
        return ("Anchor expression too long (max 1024 characters).", [])
    s = re_mod.sub(r"\b(\d{2})-rand\b", r"rand-\1", s)
    s = lint_expand_year_month_aliases(s)
    if not s:
        return None, []

    fatal = lint_check_weekly_delimiter_contract(s)
    if fatal:
        return fatal, []

    fatal = lint_check_yearly_segments(s)
    if fatal:
        return fatal, []

    fatal = lint_check_global_md_dm_confusion(s)
    if fatal:
        return fatal, []

    fatal = lint_check_invalid_weekday_names(s)
    if fatal:
        return fatal, []

    fatal = lint_check_nth_weekday_suffixes(s)
    if fatal:
        return fatal, []

    fatal = lint_check_unsat_pure_weekly_and(s)
    if fatal:
        return fatal, []

    fatal = lint_check_backward_quarter_ranges(s)
    if fatal:
        return fatal, []

    return None, lint_collect_warnings(s)
