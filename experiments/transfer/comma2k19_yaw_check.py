#!/usr/bin/env python3
"""Reviewer check of the understeer finding of RESULTS.md 2.14 (CCS_PIPELINE.md known gap 9): the decoder's steering-derived yaw
(kinematic bicycle model, `yaw_rate_radps` of <external>/comma2k19/work/windows.npz) against the measured `carState.yawRate` of
the same 2018 comma2k19 segments, read with the lab LogReader, by speed bin. Runs under the lab openpilot venv ($PY).

  source env.sh && nice -n 10 $PY experiments/transfer/comma2k19_yaw_check.py [--per-car 5] [--all] [--out <work>/yaw_check.json]

Per segment: the decoder's 10 Hz grid is rebuilt as in ccs_decode.py (first `can` event time + k / 10 s), the logged carState
(100 Hz) is matched to it by nearest time (sample 0 excluded: the fresh parser's all-zero CarState), and three things are compared
by speed bin: (1) the decoder's `yaw_rate_radps` channel against the logged `yawRate` (ratio = least-squares slope through the
origin, median of the sample ratios over |yawRate| > 0.02 rad/s, correlation, same-sign share); (2) the bicycle model recomputed
from the LOGGED steering angle and speed with the decoder's steerRatio / wheelbase against the logged `yawRate` on every 100 Hz
sample of the segment (a physics check independent of the decoder); (3) a fit of ratio = 1 + k v^2 over the bins. Output: a JSON
with counts and ratios only (pseudonymous drive ids and segment numbers; no dongle id, route stamp or path).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BINS = ((1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40))
YAW_MIN = 0.02      # rad/s: samples below this are left out of the median ratio and the sign check


def pseudonym(kind: str, *parts: str) -> str:   # = experiments.transfer.ccs_platforms.pseudonym
    return f"ccs_{kind}_" + hashlib.sha1(("ccs|" + "|".join(parts)).encode()).hexdigest()[:10]


def read_log(path: str):
    """(t0 of the first can event [s], carState rows (t, vEgo, yawRate, steeringAngleDeg))."""
    from tools.lib.logreader import LogReader
    t0 = None
    rows = []
    for ev in LogReader(path):
        w = ev.which()
        if w == "can":
            if t0 is None:
                t0 = ev.logMonoTime * 1e-9
        elif w == "carState":
            cs = ev.carState
            ang = cs.steeringAngleDeg if hasattr(cs, "steeringAngleDeg") else cs.steeringAngle
            rows.append((ev.logMonoTime * 1e-9, float(cs.vEgo), float(cs.yawRate), float(ang)))
    return t0, np.asarray(rows, dtype=np.float64)


def ratio_stats(d: np.ndarray, m: np.ndarray) -> dict:
    """d = derived yaw, m = measured yaw on the same samples."""
    ok = np.isfinite(d) & np.isfinite(m)
    d, m = d[ok], m[ok]
    out = dict(n=int(len(d)))
    if len(d) < 20 or float((m * m).sum()) <= 0:
        return out
    big = np.abs(m) > YAW_MIN
    out["slope"] = float((d * m).sum() / (m * m).sum())
    out["corr"] = float(np.corrcoef(d, m)[0, 1]) if d.std() > 0 and m.std() > 0 else float("nan")
    if big.sum() >= 20:
        out["median_ratio"] = float(np.median(d[big] / m[big]))
        out["same_sign"] = float((np.sign(d[big]) == np.sign(m[big])).mean())
        out["n_big"] = int(big.sum())
    return out


def by_bins(v: np.ndarray, d: np.ndarray, m: np.ndarray) -> dict:
    out = {}
    for lo, hi in BINS:
        sel = (v >= lo) & (v < hi)
        out[f"{lo}-{hi}"] = ratio_stats(d[sel], m[sel])
    out["all"] = ratio_stats(d[v >= 1.0], m[v >= 1.0])
    return out


def fit_k(bins: dict) -> float | None:
    """ratio = 1 + k v^2 through the bin slopes (bin centre speed; bins with >= 50 samples)."""
    xs, ys = [], []
    for name, s in bins.items():
        if name == "all" or s.get("n", 0) < 50 or "slope" not in s:
            continue
        lo, hi = (float(x) for x in name.split("-"))
        xs.append(((lo + hi) / 2) ** 2); ys.append(s["slope"] - 1.0)
    if len(xs) < 2:
        return None
    xs, ys = np.asarray(xs), np.asarray(ys)
    return float((xs * ys).sum() / (xs * xs).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if "VTT_EXTERNAL" not in os.environ and "VTT_DATA_ROOT" not in os.environ:
        raise SystemExit("source env.sh first (VTT_DATA_ROOT / VTT_EXTERNAL set the data root)")
    ext = Path(os.environ.get("VTT_EXTERNAL", str(Path(os.environ.get("VTT_DATA_ROOT", "")) / "external")))
    ap.add_argument("--root", default=str(ext / "comma2k19" / "segments"))
    ap.add_argument("--work", default=str(ext / "comma2k19" / "work"))
    ap.add_argument("--openpilot", default=str(Path(__file__).resolve().parents[3] / "openpilot"), help="checkout that holds tools/lib/logreader.py")
    ap.add_argument("--per-car", type=int, default=5, help="segments per car, spread evenly over the car's windows")
    ap.add_argument("--all", action="store_true", help="every window instead of --per-car")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sys.path.insert(0, a.openpilot)
    work = Path(a.work)
    z = np.load(work / "windows.npz", allow_pickle=True)
    X = z["X"]; ch = {str(c): i for i, c in enumerate(z["channels"])}
    m = pd.read_parquet(work / "meta.parquet").reset_index(drop=True)
    qc = pd.read_csv(work / "segments_qc.csv")
    geom = qc.groupby("platform")[["steer_ratio", "wheelbase_m"]].first()
    # segment folders -> meta rows (pseudonym of dongle | route folder, segment number)
    row_of = {}
    for i, r in m.iterrows():
        row_of[(str(r["drive_id"]), int(r["w_idx"]))] = i
    segs = []   # (platform, meta row, path)
    for chunk in sorted(p for p in Path(a.root).iterdir() if p.is_dir()):
        for route_dir in sorted(p for p in chunk.iterdir() if p.is_dir()):
            dongle = route_dir.name.split("|", 1)[0]
            did = pseudonym("d", dongle, route_dir.name)
            for seg_dir in sorted((p for p in route_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
                f = seg_dir / "raw_log.bz2"
                i = row_of.get((did, int(seg_dir.name)))
                if f.exists() and i is not None:
                    segs.append((str(m.loc[i, "platform"]), i, str(f)))
    print(f"[yaw_check] {len(segs)} windows matched to a segment folder (meta rows {len(m)})")
    assert len(segs) == len(m), "every window must map to one segment folder"
    chosen = []
    for plat in sorted({s[0] for s in segs}):
        mine = [s for s in segs if s[0] == plat]
        if a.all:
            chosen += mine
        else:
            idx = np.unique(np.round(np.linspace(0, len(mine) - 1, a.per_car)).astype(int))
            chosen += [mine[i] for i in idx]
    out = dict(segments_per_car=("all" if a.all else a.per_car), yaw_min_radps=YAW_MIN, bins=[f"{lo}-{hi}" for lo, hi in BINS], cars={})
    pooled = {k: [] for k in ("v", "d", "m", "vl", "dl", "ml")}
    for plat in sorted({s[0] for s in chosen}):
        sr, wb = float(geom.loc[plat, "steer_ratio"]), float(geom.loc[plat, "wheelbase_m"])
        V, D, M, VL, DL, ML = [], [], [], [], [], []
        used = []
        dt_max = 0.0
        for _, i, path in [s for s in chosen if s[0] == plat]:
            t0, cs = read_log(path)
            if t0 is None or len(cs) < 100:
                print(f"[yaw_check] skip a segment of {plat}: no can / carState"); continue
            grid = t0 + np.arange(X.shape[1]) / 10.0
            j = np.clip(np.searchsorted(cs[:, 0], grid), 1, len(cs) - 1)
            j = np.where(np.abs(cs[j, 0] - grid) < np.abs(cs[j - 1, 0] - grid), j, j - 1)
            dt_max = max(dt_max, float(np.abs(cs[j, 0] - grid)[1:].max()))
            v_dec = X[i, :, ch["v_mps"]].astype(float); yaw_dec = X[i, :, ch["yaw_rate_radps"]].astype(float)
            k = np.arange(1, X.shape[1])          # sample 0 excluded
            V.append(v_dec[k]); D.append(yaw_dec[k]); M.append(cs[j[k], 2])
            # physics check on every logged sample: bicycle model from the LOGGED angle and speed
            vl, yl, al = cs[:, 1], cs[:, 2], cs[:, 3]
            VL.append(vl); ML.append(yl); DL.append(vl * np.tan(np.deg2rad(al / sr)) / wb)
            ang_dec = X[i, :, ch["steer_angle_deg"]].astype(float)
            dv = np.abs(v_dec[k] - cs[j[k], 1]); mv = cs[j[k], 1] > 1.0
            da = np.abs(ang_dec[k] - cs[j[k], 3])
            used.append(dict(drive=str(m.loc[i, "drive_id"]), seg=int(m.loc[i, "w_idx"]), v_max=round(float(np.nanmax(v_dec)), 1), das=round(float(m.loc[i, "das_frac"]), 2),
                             v_absdiff_max=round(float(dv.max()), 3), v_absdiff_p99=round(float(np.percentile(dv, 99)), 3), v_ratio_median=(round(float(np.median(v_dec[k][mv] / cs[j[k], 1][mv])), 4) if mv.sum() > 20 else None),
                             angle_absdiff_max=round(float(da.max()), 2), angle_absdiff_p99=round(float(np.percentile(da, 99)), 2), angle_offset_median=round(float(np.median(ang_dec[k] - cs[j[k], 3])), 2),
                             angle_corr=(round(float(np.corrcoef(ang_dec[k], cs[j[k], 3])[0, 1]), 4) if ang_dec[k].std() > 0 else None)))
        v, d, mm = (np.concatenate(x) for x in (V, D, M)); vl, dl, ml = (np.concatenate(x) for x in (VL, DL, ML))
        for key, arr in zip(("v", "d", "m", "vl", "dl", "ml"), (v, d, mm, vl, dl, ml)):
            pooled[key].append(arr)
        dec_bins = by_bins(v, d, mm); log_bins = by_bins(vl, dl, ml)
        sign_angle = float((np.sign(cs[:, 3][np.abs(cs[:, 2]) > YAW_MIN]) == np.sign(cs[:, 2][np.abs(cs[:, 2]) > YAW_MIN])).mean()) if (np.abs(cs[:, 2]) > YAW_MIN).sum() > 20 else None
        out["cars"][plat] = dict(model=str(m.loc[[s[1] for s in chosen if s[0] == plat][0], "model"]), steer_ratio=sr, wheelbase_m=wb, n_segments=len(used), segments=used,
                                 match_dt_max_s=round(dt_max, 4), yaw_meas_nonzero=float((mm != 0).mean()),
                                 decoder_vs_logged=dec_bins, k_fit_decoder=fit_k(dec_bins), bicycle_from_log_vs_logged=log_bins, k_fit_log=fit_k(log_bins),
                                 sign_angle_vs_yaw_last_segment=sign_angle)
        print(f"  {plat} ({len(used)} segments; steerRatio {sr}, wheelbase {wb} m; nearest-match dt max {dt_max * 1e3:.0f} ms):")
        for name in list(dec_bins):
            s = dec_bins[name]; s2 = log_bins[name]
            print(f"    {name:6s} m/s: decoder/logged n={s['n']:6d} slope {s.get('slope', float('nan')):.2f} median {s.get('median_ratio', float('nan')):.2f} corr {s.get('corr', float('nan')):.3f} same sign {s.get('same_sign', float('nan')):.3f}"
                  f" | bicycle(log)/logged n={s2['n']:7d} slope {s2.get('slope', float('nan')):.2f} median {s2.get('median_ratio', float('nan')):.2f} corr {s2.get('corr', float('nan')):.3f}")
        print(f"    k fit (1 + k v^2): decoder channel {out['cars'][plat]['k_fit_decoder']}, bicycle from the log {out['cars'][plat]['k_fit_log']}")
    v, d, mm, vl, dl, ml = (np.concatenate(pooled[k]) for k in ("v", "d", "m", "vl", "dl", "ml"))
    out["both_cars"] = dict(decoder_vs_logged=by_bins(v, d, mm), bicycle_from_log_vs_logged=by_bins(vl, dl, ml))
    out["both_cars"]["k_fit_decoder"] = fit_k(out["both_cars"]["decoder_vs_logged"]); out["both_cars"]["k_fit_log"] = fit_k(out["both_cars"]["bicycle_from_log_vs_logged"])
    print(f"  both cars: k fit decoder {out['both_cars']['k_fit_decoder']}, bicycle from the log {out['both_cars']['k_fit_log']}; overall corr decoder/logged {out['both_cars']['decoder_vs_logged']['all'].get('corr', float('nan')):.4f}")
    dest = Path(a.out) if a.out else work / ("yaw_check_all.json" if a.all else "yaw_check.json")
    dest.write_text(json.dumps(out, indent=1))
    print(f"[yaw_check] wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
