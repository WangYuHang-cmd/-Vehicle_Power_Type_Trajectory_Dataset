#!/usr/bin/env python3
"""Consistency check of the Honda gearbox option of ccs_decode.py (`--honda-gearbox-from-can`, the `ccs_honda` re-decode).

Part 1 (default): every route of `--list` (Honda platforms that decode under both settings: Odyssey, Pilot, Accord, drawn by
`ccs_honda_list.py --limit 20`) goes through `ccs_decode.process_route` twice, with the option off and on, in one process;
the windows are matched on (drive, w_idx) and every one of the 20 training channels is compared on the 10 Hz grid:
max and 99th-percentile absolute difference (continuous channels, NaN in both = equal, NaN in one = counted as a
difference), share of differing samples (binary channels), lead_valid agreement, `can_valid` of both decodes per segment,
and the `meta.note` fields (identical up to the extra `gearbox=` entry). Part 2 (`--legacy-dumps DIR`): the per-sample
CarState dumps of the diagnosis (100 Hz rows `t, vEgo, aEgo, steeringAngleDeg, steeringRateDeg, steeringTorque, yawRate,
gasPressed, brakePressed, steeringPressed, standstill, cruiseState.enabled, canValid, cruiseState.available, vEgoRaw`;
`dump_031_<i>.npy` = opendbc 0.3.1 with the option, `dump_legacy_<i>.npy` = the 2024-08 openpilot checkout's car stack on the
same segment) are held onto the same 10 Hz grid and compared the same way, whole minute and after the first 0.5 s (the
parser start-up transient). Prints the tables (no identifiers; the platform of a dump pair comes from the first column of
`--legacy-list`) and writes `--out` (JSON). Runs under the decode venv like ccs_decode.py:

  nice -n 10 $D/ext/venv_opendbc/bin/python3 experiments/transfer/ccs_honda_check.py --segments-root $EXT/ccs/segments \\
      --list $EXT/ccs/ccs_honda_check.txt --legacy-dumps $D/tmp/ccs_honda --legacy-list $EXT/ccs/ccs_honda_probe.txt \\
      --out $D/tmp/ccs_honda/ccs_honda_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.transfer import ccs_decode as X  # noqa: E402
from experiments.transfer import common as T  # noqa: E402

DUMP_COLS = ("t", "v_mps", "a_mps2", "steer_angle_deg", "steer_rate_degps", "steer_torque", "yaw_rate_radps", "gas_pressed", "brake_pressed",
             "steer_pressed", "standstill", "cruise_enabled", "can_valid", "cruise_available", "v_raw_mps")
DUMP_BINARY = {"gas_pressed", "brake_pressed", "steer_pressed", "standstill", "cruise_enabled", "can_valid", "cruise_available"}


def _fmt(x) -> str:
    return "-" if x is None or not np.isfinite(x) else (f"{x:.3g}" if abs(x) < 1e-3 or abs(x) >= 1e4 else f"{x:.4f}")


def compare_arrays(a: np.ndarray, b: np.ndarray, binary: bool) -> dict:
    """Difference statistics of two equally long channel arrays (NaN-aware)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = np.isnan(a), np.isnan(b)
    both = ~na & ~nb
    nan_mismatch = int((na != nb).sum())
    if binary:
        diff = int(((a[both] > 0.5) != (b[both] > 0.5)).sum()) + nan_mismatch
        return dict(n=int(a.size), differing=diff, share_differing=(diff / a.size if a.size else 0.0), nan_mismatch=nan_mismatch)
    d = np.abs(a[both] - b[both])
    return dict(n=int(a.size), compared=int(both.sum()), nan_mismatch=nan_mismatch,
                max_abs=(float(d.max()) if d.size else 0.0), p99_abs=(float(np.percentile(d, 99)) if d.size else 0.0),
                differing=int((d > 0).sum()) + nan_mismatch)


# ----------------------------------------------------------------------------- part 1: option off vs on
def decode_both(root: Path, list_file: Path, cereal_dir: str) -> dict:
    X.init_worker(cereal_dir, "derived", True, 1.0, honda_gearbox=False)
    X._W["dataset"] = "ccs_honda_check"
    tasks = X.find_segments(root, list_file)
    res = {}
    for flag in (False, True):
        X._W["honda_gearbox"] = flag
        wins, notes, qc = {}, {}, []
        for task in tasks:
            out = X.process_route(task)
            qc += out["qc"]
            if out["X"] is None:
                continue
            for i, row in out["meta"].reset_index(drop=True).iterrows():
                k = (str(row["drive_id"]), int(row["w_idx"]))
                wins[k] = out["X"][i]
                notes[k] = str(row["note"])
        res[flag] = dict(wins=wins, notes=notes, qc=qc)
    n_seg = sum(len(t[2]) for t in tasks)
    keys = sorted(set(res[False]["wins"]) & set(res[True]["wins"]))
    only = dict(off=len(set(res[False]["wins"]) - set(keys)), on=len(set(res[True]["wins"]) - set(keys)))
    chans = {}
    for c in T.CHANNELS:
        j = T.CH[c]
        a = np.concatenate([res[False]["wins"][k][:, j] for k in keys]) if keys else np.zeros(0)
        b = np.concatenate([res[True]["wins"][k][:, j] for k in keys]) if keys else np.zeros(0)
        chans[c] = compare_arrays(a, b, c in T.BINARY_CHANNELS)
    note_diff = 0
    for k in keys:
        n0 = [f for f in res[False]["notes"][k].split(";") if f]
        n1 = [f for f in res[True]["notes"][k].split(";") if f and not f.startswith("gearbox=")]
        note_diff += int(n0 != n1)
    qc0 = {(q["drive_id"], q["seg"]): q for q in res[False]["qc"]}
    qc1 = {(q["drive_id"], q["seg"]): q for q in res[True]["qc"]}
    cv = [(qc0[k]["can_valid_frac"], qc1[k]["can_valid_frac"]) for k in sorted(set(qc0) & set(qc1))]
    gear = {}
    for q in res[True]["qc"]:
        gear[str(q.get("gearbox"))] = gear.get(str(q.get("gearbox")), 0) + 1
    plats = {}
    for q in res[True]["qc"]:
        plats[q["platform"]] = plats.get(q["platform"], 0) + 1
    return dict(segments=n_seg, routes=len(tasks), decoded_off=len(res[False]["qc"]), decoded_on=len(res[True]["qc"]), windows_matched=len(keys),
                windows_only=only, platforms=plats, gearbox=gear, channels=chans, notes_differing=note_diff,
                can_valid=dict(n=len(cv), min_off=(min(c[0] for c in cv) if cv else None), min_on=(min(c[1] for c in cv) if cv else None),
                               max_abs_diff=(max(abs(c[0] - c[1]) for c in cv) if cv else None)))


# ----------------------------------------------------------------------------- part 2: option (0.3.1) vs the legacy car stack (dumps)
def grid_hold(A: np.ndarray, grid: np.ndarray) -> np.ndarray:
    t = A[:, 0]
    j = np.searchsorted(t, grid + 1e-6, side="right") - 1
    return A[np.clip(j, 0, len(t) - 1)]


def compare_dumps(dump_dir: Path, platforms: list[str] | None, skip_s: float) -> dict:
    pairs = []
    i = 0
    while (dump_dir / f"dump_031_{i}.npy").exists() and (dump_dir / f"dump_legacy_{i}.npy").exists():
        pairs.append(i); i += 1
    per_seg = []
    pooled = {c: ([], []) for c in DUMP_COLS[1:]}
    pooled_tail = {c: ([], []) for c in DUMP_COLS[1:]}
    for i in pairs:
        A = np.load(dump_dir / f"dump_031_{i}.npy"); B = np.load(dump_dir / f"dump_legacy_{i}.npy")
        t0 = min(A[0, 0], B[0, 0])
        grid = t0 + np.arange(T.WINDOW_SAMPLES) / T.HZ
        Ga, Gb = grid_hold(A, grid), grid_hold(B, grid)
        tail = grid - t0 >= skip_s
        row = dict(i=i, platform=(platforms[i] if platforms and i < len(platforms) else "?"), rows_031=int(len(A)), rows_legacy=int(len(B)),
                   can_valid_031=float(A[:, 12].mean()), can_valid_legacy=float(B[:, 12].mean()), channels={})
        for j, c in enumerate(DUMP_COLS):
            if j == 0:
                continue
            binary = c in DUMP_BINARY
            row["channels"][c] = dict(whole=compare_arrays(Ga[:, j], Gb[:, j], binary), tail=compare_arrays(Ga[tail, j], Gb[tail, j], binary))
            pooled[c][0].append(Ga[:, j]); pooled[c][1].append(Gb[:, j])
            pooled_tail[c][0].append(Ga[tail, j]); pooled_tail[c][1].append(Gb[tail, j])
        per_seg.append(row)
    chans = {}
    for c in DUMP_COLS[1:]:
        binary = c in DUMP_BINARY
        chans[c] = dict(whole=compare_arrays(np.concatenate(pooled[c][0]), np.concatenate(pooled[c][1]), binary) if pairs else None,
                        tail=compare_arrays(np.concatenate(pooled_tail[c][0]), np.concatenate(pooled_tail[c][1]), binary) if pairs else None)
    return dict(pairs=len(pairs), skip_s=skip_s, channels=chans, per_segment=per_seg)


# ----------------------------------------------------------------------------- report
def table_part1(r: dict) -> str:
    rows = []
    for c, s in r["channels"].items():
        if "share_differing" in s:
            rows.append([c, "binary", str(s["n"]), "-", "-", f"{100 * s['share_differing']:.2f} %", str(s["nan_mismatch"])])
        else:
            rows.append([c, "continuous", str(s["n"]), _fmt(s["max_abs"]), _fmt(s["p99_abs"]), f"{s['differing']}", str(s["nan_mismatch"])])
    return T.md_table(["channel", "kind", "samples", "max abs diff", "p99 abs diff", "differing samples", "NaN mismatches"], rows)


def table_part2(r: dict) -> str:
    rows = []
    for c, s in r["channels"].items():
        w, t = s["whole"], s["tail"]
        if w is None:
            continue
        if "share_differing" in w:
            rows.append([c, "binary", f"{100 * w['share_differing']:.3f} %", f"{100 * t['share_differing']:.3f} %", "-", "-"])
        else:
            rows.append([c, "continuous", f"{100 * w['differing'] / w['n']:.2f} %", f"{100 * t['differing'] / t['n']:.2f} %",
                         f"{_fmt(w['max_abs'])} / {_fmt(w['p99_abs'])}", f"{_fmt(t['max_abs'])} / {_fmt(t['p99_abs'])}"])
    return T.md_table(["CarState field", "kind", "differing, whole minute", f"differing, after {r['skip_s']} s", "max / p99 abs diff, whole minute",
                       f"max / p99 abs diff, after {r['skip_s']} s"], rows)


def table_part2_segments(r: dict) -> str:
    rows = []
    for s in r["per_segment"]:
        v = s["channels"]["v_mps"]["tail"]; a = s["channels"]["a_mps2"]["tail"]; sa = s["channels"]["steer_angle_deg"]["whole"]
        tq = s["channels"]["steer_torque"]["whole"]
        flags = max(s["channels"][c]["whole"]["share_differing"] for c in ("gas_pressed", "brake_pressed", "steer_pressed", "standstill", "cruise_enabled"))
        rows.append([str(s["i"]), s["platform"], f"{s['can_valid_031']:.3f} / {s['can_valid_legacy']:.3f}", _fmt(v["max_abs"]), _fmt(a["max_abs"]),
                     _fmt(sa["max_abs"]), _fmt(tq["max_abs"]), f"{100 * flags:.2f} %"])
    return T.md_table(["pair", "platform", "can_valid 0.3.1 / legacy", f"max |dv| after {r['skip_s']} s", f"max |da| after {r['skip_s']} s", "max |d steer|",
                       "max |d torque|", "flags differing (max over 5)"], rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments-root", type=Path, default=None, help="folder holding <dongle>/<route>/<seg>/rlog.zst")
    ap.add_argument("--list", type=Path, default=None, help="segment list of the off/on comparison (ccs_honda_list.py --limit 20)")
    ap.add_argument("--cereal-dir", type=Path, default=X.CEREAL_DIR)
    ap.add_argument("--legacy-dumps", type=Path, default=None, help="folder with dump_031_<i>.npy / dump_legacy_<i>.npy pairs")
    ap.add_argument("--legacy-list", type=Path, default=None, help="tab-separated file whose first column names the platform of pair i (ids are not printed)")
    ap.add_argument("--skip-s", type=float, default=0.5, help="start-up transient excluded in the 'tail' statistics of part 2")
    ap.add_argument("--out", type=Path, required=True, help="JSON result")
    a = ap.parse_args(argv)
    R: dict = {}
    L: list[str] = []
    if a.segments_root and a.list:
        R["off_vs_on"] = r = decode_both(a.segments_root, a.list, str(a.cereal_dir))
        L += [f"Part 1: option off vs on, {r['segments']} segments / {r['routes']} routes, decoded {r['decoded_off']} / {r['decoded_on']}, "
              f"{r['windows_matched']} windows matched (only off {r['windows_only']['off']}, only on {r['windows_only']['on']}); platforms {r['platforms']}; "
              f"gearbox rule {r['gearbox']}; can_valid min off {r['can_valid']['min_off']} / on {r['can_valid']['min_on']}, max |diff| {r['can_valid']['max_abs_diff']}; "
              f"notes differing (ignoring gearbox=): {r['notes_differing']}", "", table_part1(r), ""]
    if a.legacy_dumps:
        plats = None
        if a.legacy_list and a.legacy_list.exists():
            plats = [ln.split("\t")[0] for ln in a.legacy_list.read_text().splitlines() if ln.strip()]
        R["fix_vs_legacy"] = r2 = compare_dumps(a.legacy_dumps, plats, a.skip_s)
        L += [f"Part 2: opendbc 0.3.1 with the option vs the legacy car stack, {r2['pairs']} segment pairs on the 10 Hz grid", "", table_part2(r2), "",
              table_part2_segments(r2), ""]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=1, default=str))
    print("\n".join(L))
    print(f"[ccs_honda_check] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
