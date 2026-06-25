#!/usr/bin/env python3
import argparse, json, os, subprocess, sys
from collections import defaultdict

# ---------- Task helpers ------------------------------------------------------
def run_task(args, expect_json=False, timeout=60):
    """
    Run Taskwarrior with hooks off + no prompts.
    Returns (rc, stdout, stderr) or parsed JSON when expect_json=True.
    """
    base = [
        "task",
        "rc.hooks=off",
        "rc.confirmation=off",
        "rc.dependency.confirmation=off",
        "rc.recurrence.confirmation=off",
        "rc.pager=off",
        "rc.color=off",
        "rc.bulk=0",
        "rc.verbose=nothing",
    ]
    cmd = base + list(args)
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    out = p.stdout or ""
    err = p.stderr or ""
    if expect_json:
        try:
            return p.returncode, (json.loads(out) if out.strip().startswith("[") else []), err
        except Exception:
            return p.returncode, [], err
    return p.returncode, out, err


def modify_bulk_uuids(uuid_list, chain_id_value, chunk=120, timeout=120):
    """
    Modify many tasks (same chainID) in chunks. Checks rc and falls back to singles.
    Returns number of tasks successfully updated.
    """
    total = len(uuid_list)
    done  = 0
    for i in range(0, total, chunk):
        group = uuid_list[i:i+chunk]
        filt = "(" + " or ".join(f"uuid:{u}" for u in group) + ")"
        rc, out, err = run_task([filt, "modify", f"chainID:{chain_id_value}"], timeout=timeout)
        if rc == 0:
            done += len(group)
            print(f"Applied {done}/{total} for chainID {chain_id_value}", flush=True)
            continue

        # Fallback: singles, and count successes
        print(f"Chunk failed (rc={rc}). Falling back to single modifies for {len(group)} tasks...", flush=True)
        for u in group:
            rc1, out1, err1 = run_task([f"uuid:{u}", "modify", f"chainID:{chain_id_value}"], timeout=30)
            if rc1 == 0:
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"Applied {done}/{total} for chainID {chain_id_value}", flush=True)
            else:
                # Show a couple of errors to help the user (usually UDA not defined)
                if "not recognized" in (err1 or "").lower() or "not recognized" in (out1 or "").lower():
                    print(f"ERR modifying {u}: chainID UDA not recognized; define it in ~/.taskrc", flush=True)
                elif "No matches" in (out1 or ""):
                    print(f"ERR modifying {u}: no matches; check uuid exists", flush=True)

    return done


def _uda_exists(name="chainID"):
    rc, out, err = run_task(["show", f"uda.{name}.type"])
    # 'task show uda.chainID.type' prints the kv when present, empty otherwise
    return (rc == 0) and (name in (out or ""))


def export_all():
    rc, data, err = run_task(["rc.json.array=1", "export"], expect_json=True, timeout=120)
    return data if rc == 0 else []

def export_candidates():
    # much faster on big DBs
    rc, data, err = run_task(
        ["rc.json.array=1", "( anchor.any: or cp.any: )", "export"],
        expect_json=True,
        timeout=120,
    )
    return data if rc == 0 else []

def _export_uuid_full(u: str, env=None) -> dict | None:
    rc, out, err = run_task(["rc.json.array=1", f"export uuid:{u}"], expect_json=True)
    return out[0] if (rc == 0 and out) else None


def modify_uuid(full_uuid, kv):
    """
    Modify a single task by UUID with given key=value (dict).
    Hooks disabled and confirmation off.
    """
    parts = []
    for k, v in kv.items():
        parts.append(f"{k}:{v}")
    return run_task(["rc.confirmation=off", f"uuid:{full_uuid}", "modify"] + parts)

# ---------- Chain helpers -----------------------------------------------------
def short_uuid(u):
    u = (u or "").strip().lower()
    return u.split("-")[0] if u else ""

def index_tasks(rows):
    by_full, by_short = {}, defaultdict(list)
    for t in rows:
        u = (t.get("uuid") or "").strip().lower()
        if not u:
            continue
        by_full[u] = t
        by_short[short_uuid(u)].append(u)
    return by_full, by_short

def resolve_uuid(token, by_full, by_short, *, context_full=None, direction=None):
    """
    Resolve a full UUID from a short token. If ambiguous, use link adjacency
    relative to 'context_full' and 'direction' ('prev' or 'next') to choose.
    Returns full uuid or None.
    """
    if not token:
        return None
    tok = token.strip().lower()

    # Full UUID path
    if len(tok) == 36 and tok in by_full:
        return tok

    # Short/prefix path
    if len(tok) <= 8:
        cands = by_short.get(tok, [])
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]

        # Ambiguous: try to pick the neighbor actually linked to 'context_full'
        if context_full and context_full in by_full:
            cur = by_full[context_full]
            cur_short = short_uuid(context_full)

            def is_linked(full_u):
                t = by_full.get(full_u)
                if not t:
                    return False
                if direction == "prev":
                    # prev of current should have nextLink == current short
                    return (t.get("nextLink") or "").strip() == cur_short
                if direction == "next":
                    # next of current should have prevLink == current short
                    return (t.get("prevLink") or "").strip() == cur_short
                # if unknown, accept any adjacency
                return ((t.get("nextLink") or "").strip() == cur_short or
                        (t.get("prevLink") or "").strip() == cur_short)

            linked = [u for u in cands if is_linked(u)]
            if len(linked) == 1:
                return linked[0]

        # Still ambiguous
        return None

    return None

def root_uuid_from(u, by_full, by_short, max_walk=5000):
    seen = set(); cur = u; steps = 0
    while cur and cur not in seen and steps < max_walk:
        seen.add(cur); steps += 1
        t = by_full.get(cur)
        if not t:
            break
        prev_tok = (t.get("prevLink") or "").strip()
        prev = resolve_uuid(prev_tok, by_full, by_short,
                            context_full=cur, direction="prev")
        if not prev:
            break
        cur = prev
    return cur or u

def walk_chain(seed_full_uuid, by_full, by_short, max_walk=5000):
    """
    BFS over prev/next links but ONLY traverse when the neighbor has a reciprocal link:
      - prev hop allowed iff prev.nextLink == current.short
      - next hop allowed iff next.prevLink == current.short
    This prevents accidental bridges across unrelated chains.
    """
    def _neighbors(u_full):
        cur = by_full.get(u_full)
        if not cur:
            return []
        cur_short = short_uuid(u_full)
        nbrs = []

        # prev side: token is a short; find all candidates with that short
        p_tok = (cur.get("prevLink") or "").strip().lower()
        if p_tok:
            for cand_full in by_short.get(p_tok, []):
                t = by_full.get(cand_full)
                if not t:
                    continue
                # reciprocal check: their nextLink must point back to us
                if (t.get("nextLink") or "").strip().lower() == cur_short:
                    nbrs.append(cand_full)

        # next side
        n_tok = (cur.get("nextLink") or "").strip().lower()
        if n_tok:
            for cand_full in by_short.get(n_tok, []):
                t = by_full.get(cand_full)
                if not t:
                    continue
                # reciprocal check: their prevLink must point back to us
                if (t.get("prevLink") or "").strip().lower() == cur_short:
                    nbrs.append(cand_full)

        return nbrs

    seen, q, out = set(), [seed_full_uuid], []
    steps = 0
    while q and steps < max_walk:
        u = q.pop(0); steps += 1
        if u in seen:
            continue
        seen.add(u)
        t = by_full.get(u)
        if t:
            out.append(t)
            for v in _neighbors(u):
                if v not in seen:
                    q.append(v)
    return out


def _chain_map(chain):
    """full_uuid -> task for quick lookups."""
    cmap = {}
    for t in chain:
        u = (t.get("uuid") or "").strip().lower()
        if u:
            cmap[u] = t
    return cmap

def _resolve_prev_within_chain(task, cmap):
    """
    Resolve task.prevLink to a full UUID **within the same chain** (cmap),
    handling short-UUID ambiguity by adjacency.
    """
    cur_u = (task.get("uuid") or "").strip().lower()
    cur_short = short_uuid(cur_u)
    prev_tok = (task.get("prevLink") or "").strip().lower()
    if not prev_tok:
        return None

    # candidates: any task in chain whose short == prev_tok
    cands = [u for u in cmap.keys() if short_uuid(u) == prev_tok]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]

    # ambiguous: pick the one whose nextLink points back to us (adjacency)
    for u in cands:
        if (cmap[u].get("nextLink") or "").strip().lower() == cur_short:
            return u
    return None  # still ambiguous -> treat as no-prev inside chain

def select_root_from_chain(chain):
    """
    Pick the root **inside this chain**:
      1) Node whose prev can't be resolved to a member of the same chain
         (i.e., no valid in-chain predecessor).
      2) If multiple, prefer the one with empty prevLink field.
      3) Tie-break by earliest 'entry' (lexicographically ok for Taskwarrior timestamps),
         then by earliest 'due', then by UUID.
    Returns the full UUID string.
    """
    if not chain:
        return None

    cmap = _chain_map(chain)
    # Compute which nodes have an in-chain predecessor
    has_in_prev = set()
    for u, t in cmap.items():
        prev_u = _resolve_prev_within_chain(t, cmap)
        if prev_u:
            has_in_prev.add(u)

    # Roots are nodes with no in-chain predecessor
    roots = [u for u in cmap.keys() if u not in has_in_prev]
    if not roots:
        # weird cycle: fall back to earliest entry/due
        roots = list(cmap.keys())

    def _key(u):
        t = cmap[u]
        prev_empty = not (t.get("prevLink") or "").strip()
        entry = (t.get("entry") or "99991231T235959Z")
        due   = (t.get("due")   or "99991231T235959Z")
        return (not prev_empty, entry, due, u)  # prefer prev_empty==True, then earliest

    roots.sort(key=_key)
    return roots[0]


def _tasks_by_chain_id(tasks):
    groups = {}
    for x in tasks:
        cid = (x.get("chainID") or "").strip() or "(empty)"
        groups.setdefault(cid, []).append(x)
    return groups

def _print_conflict(root, chain, groups, max_per_group=10):
    root_short = short_uuid(root)
    cids = ", ".join(groups.keys())
    print(f"\nCONFLICT: root {root} ({root_short}) has multiple chainIDs: {cids}")
    for cid, items in groups.items():
        print(f"  chainID={cid}: {len(items)} task(s)")
        for x in items[:max_per_group]:
            su = short_uuid(x.get("uuid"))
            desc = (x.get("description") or "").strip()
            prev_s = (x.get("prevLink") or "").strip()
            next_s = (x.get("nextLink") or "").strip()
            print(f"    • {su}  prev:{prev_s or '-'}  next:{next_s or '-'}  {desc}")
        if len(items) > max_per_group:
            print(f"    … and {len(items)-max_per_group} more")



# ---------- Main logic --------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Backfill short chainID for anchor/cp chains.")
    ap.add_argument("--dry-run", action="store_true", help="Show planned changes without modifying tasks.")
    ap.add_argument("--fix-inconsistent", action="store_true",
                    help="Unify conflicting chainIDs within a chain to the root's short UUID.")
    ap.add_argument("--only-project", metavar="PROJ", help="Limit to a single project (client-side filter).")
    ap.add_argument("--only-tag", metavar="TAG", help="Limit to a single tag (client-side filter).")
    ap.add_argument("--include-completed", action="store_true", help="Include completed tasks (default: yes; we filter client-side anyway).")
    args = ap.parse_args()

    # 0) Fetch everything, filter locally (simplest & safest)
    rows = export_all()
    if not rows:
        print("No tasks exported. Nothing to do.")
        return

    # Optional local filters
    def keep(t):
        if args.only_project and (t.get("project") or "") != args.only_project:
            return False
        if args.only_tag:
            tags = t.get("tags") or []
            if args.only_tag not in tags:
                return False
        return True

    # 1) Index
    by_full, by_short = index_tasks(rows)

    # 2) Candidates: tasks that clearly use Nautical (anchor or cp)
    candidates = [t for t in rows
                  if keep(t) and ((t.get("anchor") or "").strip() or (t.get("cp") or "").strip())]

    # 3) Walk by chains, compute the chainID, and prepare modifications
    touched = 0
    chains_fixed = 0
    already_ok = 0
    collisions = 0

    # New: de-dupe planned mods and processed roots
    mods_map = {}            # uuid -> {"chainID": cid}
    processed_roots = set()  # full root uuids we've already handled

    for t in candidates:
        u = (t.get("uuid") or "").strip().lower()
        if not u:
            continue

        # Build the chain first from the current task,
        # then select the root **from that chain**.
        chain = walk_chain(u, by_full, by_short)

        root = select_root_from_chain(chain) or u
        if root in processed_roots:
            continue
        processed_roots.add(root)

        cid = short_uuid(root)


        # Determine current chainIDs in this chain
        existing = set()
        for x in chain:
            v = (x.get("chainID") or "").strip()
            if v:
                existing.add(v)

        if not existing:
            # assign cid to all tasks in the chain (only missing/incorrect ones get added)
            for x in chain:
                if (x.get("chainID") or "").strip() != cid:
                    mods_map[x["uuid"]] = {"chainID": cid}
            chains_fixed += 1
        else:
            if existing == {cid}:
                already_ok += len(chain)
            else:
                collisions += 1
                if args.fix_inconsistent:
                    for x in chain:
                        if (x.get("chainID") or "").strip() != cid:
                            mods_map[x["uuid"]] = {"chainID": cid}
                    chains_fixed += 1
                # else: leave inconsistent chain as-is, just count it

    touched = len(mods_map)


    # 4) Apply or show
#    print(f"Planned updates:  {touched} task(s)")
    if collisions and not args.fix_inconsistent:
        print(f"Note: {collisions} chain(s) have conflicting chainIDs. Re-run with --fix-inconsistent to unify.")

    if args.dry_run:
        # show at most 50 examples
        shown = 0
        for uuid, kv in mods_map.items():
            if shown < 50:
                print(f"DRY: task {uuid} modify " + " ".join([f"{k}:{v}" for k, v in kv.items()]))
                shown += 1
            else:
                break
        if touched > shown:
            print(f"...and {touched - shown} more.")
        return

    # Execute (bulk by chainID)
    from collections import defaultdict
    groups = defaultdict(list)  # chainID -> [uuids]
    for uuid, kv in mods_map.items():
        groups[kv["chainID"]].append(uuid)

    applied = 0
    for cid, uuids in groups.items():
        print(f"Updating {len(uuids)} task(s) to chainID {cid}...", flush=True)
        applied += modify_bulk_uuids(uuids, cid, chunk=120)

    print(f"Applied: {applied} update(s). Rechecking...", flush=True)

    # Recheck to verify conflicts are gone
    rows2 = export_all()  # scan everything to detect any remaining inconsistencies
    by_full2, by_short2 = index_tasks(rows2)

    remaining_conflicts = 0
    # Report exact chains & tasks still conflicted
    for t in rows2:
        u = (t.get("uuid") or "").strip().lower()
        if not u:
            continue
        root = root_uuid_from(u, by_full2, by_short2)
        if root in processed_roots:
            continue
        processed_roots.add(root)
        chain = walk_chain(root, by_full2, by_short2)
        groups = _tasks_by_chain_id(chain)
        # Ignore the easy case: exactly one chainID and it's the root's short
        if len(groups) == 1:
            continue
        # Otherwise, show details
        _print_conflict(root, chain, groups)
        remaining_conflicts += 1

    print(f"\nRemaining conflicting chains: {remaining_conflicts}")
    if remaining_conflicts:
        print("Tip: rerun with --fix-inconsistent to unify every task in a conflicted chain to the root’s short UUID.")
    processed_roots = set()



    if not _uda_exists("chainID"):
        print("WARNING: UDA 'chainID' is not defined in ~/.taskrc; modifies may fail.")
        print("Add:\n  uda.chainID.type=string\n  uda.chainID.label=ChainID\n")


if __name__ == "__main__":
    # Safety hint: user should configure UDA first, but not required for storage.
    # Recommended in ~/.taskrc:
    #   uda.chainID.type=string
    #   uda.chainID.label=ChainID
    main()

