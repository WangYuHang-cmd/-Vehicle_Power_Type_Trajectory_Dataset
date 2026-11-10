#!/usr/bin/env python3
"""Owner's download list for commaCarSegments (CANDIDATES.md section 4.1, step 2).

database.json (platform key -> ["<dongle>/<route>/<seg>/s", ...]) -> fold legacy keys with opendbc MIGRATION -> drop MOCK and
keys that neither opendbc nor the label tables know (forks) -> remove the devices shared with the lab library
(shared_dongles.txt) -> label every platform (ccs_platforms.Labeller: model, footprint, powertrain, stratum) -> select:
  * 'seen' models (training / validation models): at most --cap segments per platform from at least --min-devices devices,
    taken as pieces of up to --chunk consecutive segments of one route (so a drive yields several windows), round-robin
    over the devices (devices with consecutive runs first), multi-segment pieces before singletons;
  * 'held-out' (the 41 test models) and 'absent' (not in the lab inventory) platforms: every segment (or --cap-heldout /
    --cap-absent with the same algorithm when > 0).
Writes <out> (one <dongle>/<route>/<seg> per line, sorted; for the owner's machine only - never copy it into the repo),
<out dir>/select_summary.json (per-platform counts, no identifiers) and the markdown report CCS_SELECTION.md (counts,
strata, the exact download command with a placeholder path). Prints the totals and the estimated download size
(--mb-per-seg: 1.58 MB = 298.6 GB / 188,883 segments of the index; the five verification samples averaged 1.97 MB).

Runs under $CPY (pandas only; opendbc is parsed from source, see ccs_platforms.find_opendbc_root):
  source env.sh && $CPY experiments/transfer/ccs_select.py --index $EXT/ccs/database.json --out $EXT/ccs/select.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.transfer import ccs_platforms as P  # noqa: E402

SCRATCH = P.DATA_ROOT / "scratch" / "ccs_verify" / "dl"
HF_BASE = "https://huggingface.co/datasets/commaai/commaCarSegments/resolve/main"


def first_existing(*paths: Path) -> Path | None:
    return next((p for p in paths if p and Path(p).is_file()), None)


def parse_entry(s: str) -> tuple[str, str, int]:
    parts = s.strip().split("/")
    if len(parts) < 3:
        raise ValueError(f"unexpected index entry {s!r}")
    return parts[0], parts[1], int(parts[2])


def consecutive_runs(segs) -> list[list[int]]:
    runs: list[list[int]] = []
    for s in sorted(set(int(x) for x in segs)):
        if runs and s == runs[-1][-1] + 1:
            runs[-1].append(s)
        else:
            runs.append([s])
    return runs


def select_platform(entries: list[tuple[str, str, int]], cap: int, min_devices: int, chunk: int) -> list[tuple[str, str, int]]:
    """Capped selection of one platform's entries (see the module docstring); cap <= 0 or few entries -> all."""
    if cap <= 0 or len(entries) <= cap:
        return sorted(set(entries))
    by_dev: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for d, r, s in entries:
        by_dev[d][r].append(s)
    queues: dict[str, deque] = {}
    multi_segs: dict[str, int] = {}
    for d, routes in by_dev.items():
        pieces = []
        for r, segs in routes.items():
            for run in consecutive_runs(segs):
                for i in range(0, len(run), chunk):
                    piece = run[i:i + chunk]
                    pieces.append((len(piece) >= 2, len(piece), r, piece))
        pieces.sort(key=lambda p: (not p[0], -p[1], p[2]))          # multi-segment pieces first, longest first, then route
        queues[d] = deque(pieces)
        multi_segs[d] = sum(p[1] for p in pieces if p[0])
    n_dev = len(queues)
    per_dev_cap = max(chunk, math.ceil(cap / max(1, min(n_dev, min_devices))))
    order = sorted(queues, key=lambda d: (-multi_segs[d], d))
    taken: dict[str, int] = defaultdict(int)
    selected: list[tuple[str, str, int]] = []
    progress = True
    while len(selected) < cap and progress:
        progress = False
        for d in order:
            if len(selected) >= cap:
                break
            q = queues[d]
            if taken[d] >= per_dev_cap:
                q.clear()
            if not q:
                continue
            _, _, r, piece = q.popleft()
            room = min(cap - len(selected), per_dev_cap - taken[d])
            piece = piece[:room]
            selected.extend((d, r, s) for s in piece)
            taken[d] += len(piece)
            progress = True
    return sorted(set(selected))


def run_share(sel: list[tuple[str, str, int]]) -> float:
    """Share of the selected segments that have a selected neighbour (seg +- 1) in the same route."""
    if not sel:
        return float("nan")
    have = set(sel)
    n = sum(1 for d, r, s in sel if (d, r, s - 1) in have or (d, r, s + 1) in have)
    return n / len(sel)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=None, help="database.json (default: $EXT/ccs/database.json, else the verification copy)")
    ap.add_argument("--shared", type=Path, default=None, help="shared_dongles.txt (default: $EXT/ccs/shared_dongles.txt, else the verification copy)")
    ap.add_argument("--out", type=Path, default=P.CCS_DIR / "select.txt", help="selection list (owner's machine only)")
    ap.add_argument("--md", type=Path, default=HERE / "CCS_SELECTION.md", help="markdown report (no identifiers)")
    ap.add_argument("--cap", type=int, default=400, help="segments per 'seen' platform")
    ap.add_argument("--min-devices", type=int, default=20, help="devices a capped platform should draw from (when it has that many)")
    ap.add_argument("--chunk", type=int, default=5, help="max consecutive segments of one route taken per device and round")
    ap.add_argument("--cap-heldout", type=int, default=0, help="cap for held-out models (0 = every segment)")
    ap.add_argument("--cap-absent", type=int, default=0, help="cap for models absent from the lab inventory (0 = every segment)")
    ap.add_argument("--budget-cap-heldout", type=int, default=1000,
                    help="also write <out dir>/select_budget.txt with held-out platforms capped at this many segments (0 = off)")
    ap.add_argument("--mb-per-seg", type=float, default=1.58, help="index-wide mean segment size in MB (298.6 GB / 188,883)")
    ap.add_argument("--mb-per-seg-sample", type=float, default=1.97, help="mean size of the five verification samples (upper estimate)")
    ap.add_argument("--opendbc-root", type=Path, default=None, help="opendbc source tree for MIGRATION / platform names")
    a = ap.parse_args(argv)

    index = a.index or first_existing(P.CCS_DIR / "database.json", SCRATCH / "database.json")
    shared = a.shared or first_existing(P.CCS_DIR / "shared_dongles.txt", SCRATCH / "shared_dongles.txt")
    if index is None or not Path(index).is_file():
        print("index database.json not found", file=sys.stderr)
        return 2
    if shared is None or not Path(shared).is_file():
        print("shared_dongles.txt not found", file=sys.stderr)
        return 2
    root = P.find_opendbc_root(a.opendbc_root)
    migration, platforms = P.load_opendbc_names(root)
    lab = P.Labeller()
    db = json.loads(Path(index).read_text())
    shared_devs = {ln.strip() for ln in Path(shared).read_text().splitlines() if ln.strip()}

    # ---- fold, drop, de-share
    folded: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    raw_keys: dict[str, set[str]] = defaultdict(set)
    dropped: list[dict] = []
    n_index = 0
    for key, entries in db.items():
        n_index += len(entries)
        f = P.fold(key, migration)
        if key in P.DROP_KEYS or f in P.DROP_KEYS:
            dropped.append(dict(key=key, folded=f, reason="mock", n=len(entries)))
            continue
        lbl = lab.label(f, key)
        if f not in platforms and lbl["label_source"] == "none":
            dropped.append(dict(key=key, folded=f, reason="fork / unknown to opendbc and to the label tables", n=len(entries)))
            continue
        raw_keys[f].add(key)
        folded[f].extend(parse_entry(e) for e in entries)
    n_folded = sum(len(v) for v in folded.values())

    rows: list[dict] = []
    selection: dict[str, list[tuple[str, str, int]]] = {}
    for f in sorted(folded):
        ents = folded[f]
        devs_idx = {d for d, _, _ in ents}
        kept = [e for e in ents if e[0] not in shared_devs]
        devs_kept = {d for d, _, _ in kept}
        lbl = lab.label(f, sorted(raw_keys[f])[0] if raw_keys[f] else None)
        stratum = lbl["stratum"]
        cap = a.cap if stratum == "seen" else (a.cap_heldout if stratum == "held-out" else a.cap_absent)
        sel = select_platform(kept, cap, a.min_devices, a.chunk)
        selection[f] = sel
        rows.append(dict(platform=f, index_keys=len(raw_keys[f]), model=lbl["model"], brand=lbl["brand"], stratum=stratum,
                         footprint_m2=(None if not math.isfinite(lbl["footprint_m2"]) else round(lbl["footprint_m2"], 3)),
                         size_class=lbl["size_class"], powertrain=lbl["powertrain"], powertrain_raw=lbl["powertrain_raw"],
                         borderline=lbl["borderline"], label_source=lbl["label_source"], in_opendbc=f in platforms,
                         n_index=len(ents), dev_index=len(devs_idx), n_shared_removed=len(ents) - len(kept),
                         n_after=len(kept), dev_after=len(devs_kept), n_selected=len(sel),
                         dev_selected=len({d for d, _, _ in sel}), routes_selected=len({(d, r) for d, r, _ in sel}),
                         run_share=(round(run_share(sel), 3) if sel else None), note=lbl["note"]))

    # ---- write the list (owner's machine only)
    lines = [f"{d}/{r}/{s}" for f in sorted(selection) for d, r, s in selection[f]]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + ("\n" if lines else ""))
    n_sel = len(lines)
    n_after = sum(r["n_after"] for r in rows)
    gb = n_sel * a.mb_per_seg / 1000.0
    gb_hi = n_sel * a.mb_per_seg_sample / 1000.0

    budget = None
    if a.budget_cap_heldout > 0:
        b_sel: dict[str, list] = {}
        for r in rows:
            f = r["platform"]
            if r["stratum"] == "held-out":
                kept = [e for e in folded[f] if e[0] not in shared_devs]
                b_sel[f] = select_platform(kept, a.budget_cap_heldout, a.min_devices, a.chunk)
            else:
                b_sel[f] = selection[f]
        b_lines = [f"{d}/{r}/{s}" for f in sorted(b_sel) for d, r, s in b_sel[f]]
        b_out = a.out.parent / "select_budget.txt"
        b_out.write_text("\n".join(b_lines) + ("\n" if b_lines else ""))
        budget = dict(file=b_out.name, cap_heldout=a.budget_cap_heldout, selected=len(b_lines),
                      devices=len({ln.split("/")[0] for ln in b_lines}), routes=len({"/".join(ln.split("/")[:2]) for ln in b_lines}),
                      est_gb=round(len(b_lines) * a.mb_per_seg / 1000.0, 1),
                      heldout_segments=sum(len(b_sel[r["platform"]]) for r in rows if r["stratum"] == "held-out"),
                      heldout_devices=sum(len({d for d, _, _ in b_sel[r["platform"]]}) for r in rows if r["stratum"] == "held-out"))
        for r in rows:
            r["n_budget"] = len(b_sel[r["platform"]])

    strata = {}
    for s in P.STRATA:
        rs = [r for r in rows if r["stratum"] == s and r["n_selected"] > 0]
        strata[s] = dict(platforms=len(rs), models=len({r["model"] for r in rs}), segments=sum(r["n_selected"] for r in rs),
                         devices=sum(r["dev_selected"] for r in rs), routes=sum(r["routes_selected"] for r in rs),
                         gb=round(sum(r["n_selected"] for r in rs) * a.mb_per_seg / 1000.0, 1))
    summary = dict(written=str(date.today()), index_segments=n_index, index_keys=len(db), folded_platforms=len(folded),
                   folded_segments=n_folded, dropped=dropped, shared_devices=len(shared_devs),
                   shared_segments_removed=sum(r["n_shared_removed"] for r in rows), after_shared_removal=n_after,
                   selected=n_sel, selected_devices=len({ln.split("/")[0] for ln in lines}),
                   selected_routes=len({"/".join(ln.split("/")[:2]) for ln in lines}),
                   est_gb=round(gb, 1), est_gb_upper=round(gb_hi, 1), mb_per_seg=a.mb_per_seg, mb_per_seg_sample=a.mb_per_seg_sample,
                   cap=a.cap, min_devices=a.min_devices, chunk=a.chunk, cap_heldout=a.cap_heldout, cap_absent=a.cap_absent,
                   opendbc=P.opendbc_version(root), migration_entries=len(migration), opendbc_platforms=len(platforms),
                   strata=strata, budget=budget, platforms=rows)
    (a.out.parent / "select_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    write_md(a.md, summary)
    print(f"index: {n_index} segments in {len(db)} keys -> {len(folded)} platforms after MIGRATION ({n_folded} segments); "
          f"dropped {sum(d['n'] for d in dropped)} segments in {len(dropped)} keys (mock / fork)")
    print(f"shared devices removed: {len(shared_devs)} devices, {summary['shared_segments_removed']} segments -> {n_after} segments remain")
    for s in P.STRATA:
        st = strata[s]
        print(f"  {s:9s}: {st['platforms']:3d} platforms, {st['models']:3d} models, {st['segments']:6d} segments, {st['devices']:5d} devices, {st['routes']:5d} routes, ~{st['gb']} GB")
    print(f"TOTAL selected: {n_sel} segments from {summary['selected_devices']} devices / {summary['selected_routes']} routes; "
          f"estimated {gb:.1f} GB at {a.mb_per_seg} MB/segment (upper estimate {gb_hi:.1f} GB at {a.mb_per_seg_sample} MB)")
    if budget:
        print(f"budget alternative (held-out cap {budget['cap_heldout']}): {budget['selected']} segments from {budget['devices']} devices, "
              f"~{budget['est_gb']} GB -> {a.out.parent / budget['file']}")
    print(f"list: {a.out}  summary: {a.out.parent / 'select_summary.json'}  report: {a.md}")
    return 0


def write_md(path: Path, S: dict) -> None:
    rows = sorted(S["platforms"], key=lambda r: (P.STRATA.index(r["stratum"]), -(r["n_selected"]), r["platform"]))

    def f(x, nd=3):
        return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    L = [f"# commaCarSegments: owner download selection ({S['written']})", "",
         f"Built by `experiments/transfer/ccs_select.py` from the dataset index (`database.json`, {S['index_segments']:,} segments in "
         f"{S['index_keys']} keys). Legacy keys folded with opendbc MIGRATION ({S['opendbc']}, {S['migration_entries']} entries, "
         f"{S['opendbc_platforms']} platforms) -> {S['folded_platforms']} platforms. Dropped: MOCK and fork keys "
         f"({sum(d['n'] for d in S['dropped'])} segments). The {S['shared_devices']} devices shared with the lab's raw library "
         f"were removed ({S['shared_segments_removed']:,} segments), leaving {S['after_shared_removal']:,} segments.", "",
         "Selection rule: 'seen' models (models with training / validation drives in the lab's `split_platform` scheme) keep at most "
         f"{S['cap']} segments per platform from at least {S['min_devices']} devices, taken round-robin as pieces of up to {S['chunk']} "
         "consecutive segments of one route (multi-segment pieces before singletons, devices with consecutive runs first); "
         f"held-out models (the 41 test models) and models absent from the lab inventory keep every segment"
         + (f" (held-out cap {S['cap_heldout']})" if S["cap_heldout"] else "") + (f" (absent cap {S['cap_absent']})" if S["cap_absent"] else "") + ".",
         "", f"**Total: {S['selected']:,} segments from {S['selected_devices']:,} devices and {S['selected_routes']:,} routes; "
         f"about {S['est_gb']} GB at {S['mb_per_seg']} MB per segment (index-wide mean), upper estimate {S['est_gb_upper']} GB at "
         f"{S['mb_per_seg_sample']} MB (mean of the five verification samples).**", "",
         "The list itself (`select.txt`, one `<dongle>/<route>/<seg>` per line) stays on the data disk next to the index; it carries "
         "device ids and is never copied into the repository or into any markdown.", "",
         "## Strata", "", "| stratum | platforms | models | segments | devices | routes | ~GB |", "|---|---|---|---|---|---|---|"]
    for s in P.STRATA:
        st = S["strata"][s]
        L.append(f"| {s} | {st['platforms']} | {st['models']} | {st['segments']:,} | {st['devices']:,} | {st['routes']:,} | {st['gb']} |")
    L += ["", "## Per platform", "",
          "Columns: index = segments / devices in the index; shared = segments removed with the lab-shared devices; selected = "
          "segments / devices / routes in the list; runs = share of the selected segments that have a selected neighbour segment "
          "in the same route (several windows per drive). Footprint and class from the lab table (`lab`) or the public spec table "
          "`specs_external.csv` (`public`); `bl` = borderline (|footprint - cut| < 0.05 m^2).", "",
          "| platform | model | stratum | footprint | class | bl | powertrain | label | index seg / dev | shared | selected seg / dev / routes | runs |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['platform']} | {r['model'] or '-'} | {r['stratum']} | {f(r['footprint_m2'])} | {r['size_class'] or '-'} | "
                 f"{'yes' if r['borderline'] else ''} | {r['powertrain']}{'' if r['powertrain_raw'] in ('ICE','HEV','PHEV','BEV','FCEV') else ' (' + r['powertrain_raw'] + ')'} | "
                 f"{r['label_source']} | {r['n_index']:,} / {r['dev_index']} | {r['n_shared_removed']:,} | "
                 f"{r['n_selected']:,} / {r['dev_selected']} / {r['routes_selected']} | {f(r['run_share'], 2)} |")
    if S.get("budget"):
        b = S["budget"]
        L += ["", "## Budget alternative", "",
              f"`{b['file']}` (same folder as `select.txt`) keeps every 'seen' and 'absent' selection but caps the held-out platforms at "
              f"{b['cap_heldout']} segments each with the same >= {S['min_devices']}-device round-robin rule: {b['selected']:,} segments from "
              f"{b['devices']:,} devices and {b['routes']:,} routes, about {b['est_gb']} GB (held-out stratum {b['heldout_segments']:,} segments "
              f"from {b['heldout_devices']:,} devices). Use it when the full list ({S['est_gb']} GB) is too much for one sitting; the two lists "
              "are nested (the budget list is a subset), so a budget download can be topped up later with `select.txt` and the same command.",
              "", "Per-platform budget counts (held-out platforms whose selection differs):", "",
              "| platform | model | selected | budget |", "|---|---|---|---|"]
        for r in rows:
            if r["stratum"] == "held-out" and r.get("n_budget") is not None and r["n_budget"] != r["n_selected"]:
                L.append(f"| {r['platform']} | {r['model'] or '-'} | {r['n_selected']:,} | {r['n_budget']:,} |")
    L += ["", "## Dropped index keys", "", "| key | folded | reason | segments |", "|---|---|---|---|"]
    for d in S["dropped"]:
        L.append(f"| {d['key']} | {d['folded']} | {d['reason']} | {d['n']} |")
    zero = [r for r in rows if r["n_selected"] == 0]
    if zero:
        L += ["", "Platforms with no segment left after the shared-device removal: " + ", ".join(r["platform"] for r in zero) + "."]
    L += ["", "## Download (owner's machine)", "",
          "Resumable, 8 parallel streams, 5 retries per file; `$EXT` is the external-data folder on the large data disk "
          "(`select.txt` was written there by `ccs_select.py`). `download_owner_sets.sh --only ccs` runs the same loop and keeps a "
          "failure list in `$EXT/logs/ccs_failed.txt`.", "", "```bash",
          "export EXT=<folder for external data on the large data disk>",
          f"curl -sfL --create-dirs -o $EXT/ccs/database.json {HF_BASE}/database.json   # index, 9.4 MB (already present when ccs_select.py ran)",
          "# $EXT/ccs/select.txt: built by  $CPY experiments/transfer/ccs_select.py --index $EXT/ccs/database.json --out $EXT/ccs/select.txt",
          "xargs -P 8 -I{} sh -c 'curl -sfL -C - --retry 5 --retry-delay 5 --create-dirs -o \"$EXT/ccs/segments/{}/rlog.zst\" "
          f"\"{HF_BASE}/segments/{{}}/rlog.zst\"' < $EXT/ccs/select.txt",
          "# equivalent: hf download commaai/commaCarSegments --repo-type dataset --local-dir $EXT/ccs --include 'segments/<dongle>/<route>/<seg>/rlog.zst' ... (one pattern per line of select.txt)",
          "# afterwards: find $EXT/ccs/segments -name rlog.zst -size +0 | wc -l   # should equal the line count of select.txt",
          "```", "",
          f"Budget: about {S['mb_per_seg']} MB per segment -> {S['est_gb']} GB for the whole list (the full dataset, 299 GB, is not needed). "
          "Decoding: `experiments/transfer/ccs_decode.py` (see `CCS_PIPELINE.md`).", ""]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
