#!/usr/bin/env python3
"""VED (Vehicle Energy Dataset, U. Michigan, Ann Arbor 2017-18) -> transfer windows of the experiments/transfer contract.

Source (external/ved/raw): 54 weekly CSVs of OBD-II records (trip-relative `Timestamp(ms)` at irregular ~1 Hz steps, 0.1-s
bursts included; `Vehicle Speed[km/h]`, 1 km/h resolution on ICE / HEV cars, sub-km/h on the plug-ins) and two static
xlsx tables (powertrain ICE / HEV / PHEV / EV per VehId, 500-lb weight bins; make / model withheld except that the three
EVs are 2013 Nissan Leafs). Only the speed column is read: every engine / fuel / battery column is direct powertrain
leakage and never enters the windows.

Per trip ((VehId, Trip) = one contiguous recording = drive_id "<VehId>-<Trip>"):
  native (t, v_kmh) --linear; a node inside a source gap longer than --bridge-s (3 s) is missing--> 1 Hz nodes at integer trip seconds
  --round to --speed-res-kph (1 km/h)--> segments (a missing node or a glitch step ends one) --> linear between
  ADJACENT nodes only --> 10 Hz grid --> rows
  without a value dropped (= time gaps, so common.cut_windows restarts there) --> a_mps2 = common.smooth_derivative(v,
  win_s=--a-smooth-s (1.0 s)) per gap-free run, standstill = v < --standstill-v (0.1 m/s), das_enabled = 0 (no cruise
  information) --> common.cut_windows(align='run'); every other channel NaN (missing).
Glitch rule: a node-to-node step with |dv| > --a-max (8 m/s^2, the training clip bound) x dt is physically impossible. In
VED it is the OBD speed frozen at 0 km/h while the car is moving, then jumping to 40-50 km/h within a second (every case
inspected). The run is split at such a step (no interpolation across it) and the zero-speed nodes within
--glitch-zero-margin-s (10 s) before an upward step / after a downward step are discarded as stale.
Why the 1 Hz / 1 km/h homogenisation: resolution and sampling density are class-correlated in VED (all three EVs and
21 / 24 PHEVs report sub-km/h speeds; the EVs sample ~40 % of their steps at 0.1 s against ~23 % for ICE cars), so the
texture of a directly interpolated trace would identify the plug-ins. --native-timing and --native-resolution switch the
homogenisation off for sensitivity runs. The smoothing window is one source interval (a 1-s boxcar nulls the 1-Hz
staircase of the interpolant) and was checked against the training tensors: with 1.0 s the derived acceleration has the
autocorrelation 0.995 / 0.891 / 0.703 / 0.556 at lags 1 / 5 / 10 / 20 samples against 0.961 / 0.879 / 0.767 / 0.554 for the
training aEgo (0.5 s: 0.991 / 0.826 / 0.601 / 0.515; 1.5 s: 0.997 / 0.939 / 0.806 / 0.605).
Cap: at most --cap (200) windows per vehicle, evenly spaced over the vehicle's chronologically ordered windows.
Labels: powertrain from the static table (EV -> BEV); size_class NaN (no footprint) except the three Leafs, which take the
Nissan Leaf ZE0 row of specs_external.csv (4.149 m^2, SMALL; --no-leaf-footprint disables); model = "VED <powertrain>
<weight bin> lb" (the only public grouping, so the scorer's per-model table is the weight-bin table) or "Nissan Leaf (2013)".

Outputs (<external>/ved/work/): windows.npz + meta.parquet + windows_summary.json (common.save_windows);
ved_all_windows.parquet (every eligible window before the cap), ved_trips.parquet (per-trip statistics),
ved_vehicles.parquet (per-vehicle table), ved_build_report.json, ved_tables.md (tables for DS_VED.md),
ved_validation.json (validate step). Chunks of the build live in work/tmp_chunks/ and are deleted unless --keep-chunks.

Usage:  source env.sh && nice -n 10 $CPY experiments/transfer/ds_ved.py build [--workers 4] [--limit-files N]
        $CPY experiments/transfer/ds_ved.py validate
        $CPY experiments/transfer/ds_ved.py build --native-timing --native-resolution --dataset ved_native   # sensitivity set
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.transfer import common as cm  # noqa: E402

DATASET = "ved"
RAW_DEFAULT = cm.EXTERNAL_ROOT / "ved" / "raw"
SPECS_CSV = Path(__file__).resolve().parent / "specs_external.csv"
COLS = ["DayNum", "VehId", "Trip", "Timestamp(ms)", "Vehicle Speed[km/h]"]
PRESENT = ("v_mps", "a_mps2", "standstill", "das_enabled")
CHANNEL_MAP = {c: c for c in PRESENT}
DENSE_DT_S = 0.25            # a source step <= 0.25 s counts as "dense" (the 0.1-s bursts) in the sampling statistics
A_MAX_DEFAULT = 8.0          # m/s^2: the training clip bound of a_mps2; a larger node-to-node |dv| / dt is a glitch
ZERO_MARGIN_DEFAULT = 10.0   # s: zero-speed nodes this close to an impossible step are stale readings
KMH = 3.6
LEAF_FALLBACK = dict(model="Nissan Leaf (2013)", footprint_m2=4.149, wheelbase_mm=2700.0, track_front_mm=1539.0, track_rear_mm=1534.0,
                     source="DOE AVT fact sheet 2013 Nissan Leaf (fallback constants; specs_external.csv row not found)")
HEX16 = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{16}(?![0-9a-fA-F])")


# ----------------------------------------------------------------------------- static table and labels
def load_static(raw: Path) -> pd.DataFrame:
    """One row per VehId: powertrain (contract vocabulary), pt_raw, weight_lb (NaN = NO DATA), vehicle_class, engine, transmission."""
    p_ice = raw / "VED" / "Data" / "VED_Static_Data_ICE&HEV.xlsx"
    p_ev = raw / "VED" / "Data" / "VED_Static_Data_PHEV&EV.xlsx"
    a = pd.read_excel(p_ice).rename(columns={"Vehicle Type": "pt_raw"})
    b = pd.read_excel(p_ev).rename(columns={"EngineType": "pt_raw"})
    st = pd.concat([a, b], ignore_index=True)
    st.columns = [str(c).strip() for c in st.columns]
    st["VehId"] = pd.to_numeric(st["VehId"], errors="raise").astype(int)
    if st["VehId"].duplicated().any():
        raise ValueError("VehId duplicated across the static tables")
    st["pt_raw"] = st["pt_raw"].astype(str).str.strip()
    st["powertrain"] = [cm.normalise_powertrain(p) for p in st["pt_raw"]]
    st["weight_lb"] = pd.to_numeric(st["Generalized_Weight"], errors="coerce")

    def clean(s):
        s = "" if s is None or (isinstance(s, float) and not np.isfinite(s)) else str(s).strip()
        return None if s.upper() in ("", "NO DATA", "NAN") else s

    st["vehicle_class"] = [clean(x) for x in st["Vehicle Class"]]
    st["engine"] = [clean(x) for x in st["Engine Configuration & Displacement"]]
    st["transmission"] = [clean(x) for x in st["Transmission"]]
    return st.set_index("VehId")[["powertrain", "pt_raw", "weight_lb", "vehicle_class", "engine", "transmission"]]


def leaf_spec(specs_csv: Path = SPECS_CSV) -> dict:
    """The Nissan Leaf ZE0 (2013) row of specs_external.csv (datasets contains VED), else LEAF_FALLBACK."""
    if specs_csv.exists():
        sp = pd.read_csv(specs_csv)
        m = sp["model"].astype(str).str.strip().eq("Nissan Leaf") & sp["datasets"].astype(str).str.contains("VED", case=False, na=False)
        if m.any():
            r = sp[m].iloc[0]
            fp = float(r["footprint_m2"])
            return dict(model="Nissan Leaf (2013)", footprint_m2=fp, wheelbase_mm=float(r["wheelbase_mm"]), track_front_mm=float(r["track_front_mm"]),
                        track_rear_mm=float(r["track_rear_mm"]), source=f"specs_external.csv: {r['generation_or_years']}; {r['source_url']}")
    return dict(LEAF_FALLBACK)


def vehicle_labels(static: pd.DataFrame, use_leaf: bool, leaf: dict) -> dict[int, dict]:
    """VehId -> dict(model, powertrain, footprint_m2, size_class, borderline, weight_lb, vehicle_class, engine)."""
    out: dict[int, dict] = {}
    for veh, r in static.iterrows():
        pt = r["powertrain"]
        w = r["weight_lb"]
        wtxt = f"{int(w)} lb" if np.isfinite(w) else "weight n/a"
        lab = dict(model=f"VED {pt} {wtxt}", powertrain=pt, footprint_m2=float("nan"), weight_lb=(float(w) if np.isfinite(w) else float("nan")),
                   vehicle_class=r["vehicle_class"], engine=r["engine"])
        if pt == "BEV" and use_leaf:                      # README: all three EVs are 2013 Nissan Leafs (24 kWh)
            lab.update(model=leaf["model"], footprint_m2=float(leaf["footprint_m2"]))
        out[int(veh)] = lab
    return out


def make_note(lab: dict, cfg: dict) -> str:
    timing = "native stamps" if cfg["native_timing"] else "1-s nodes"
    res = "native resolution" if cfg["speed_res_kph"] <= 0 else f"speed rounded to {cfg['speed_res_kph']:g} km/h"
    parts = [f"upsampled from 1 Hz (OBD speed, irregular stamps; {timing}, linear interpolation, bridge<={cfg['bridge_s']:g}s; {res})",
             f"a_mps2 smoothed derivative ({cfg['a_smooth_s']:g}s)", f"steps >{cfg['a_max']:g} m/s^2 split the run (frozen-zero OBD glitch)",
             f"standstill=v<{cfg['standstill_v']:g} m/s", "das_enabled assumed 0",
             f"weight_bin_lb={int(lab['weight_lb']) if np.isfinite(lab['weight_lb']) else 'n/a'}"]
    if lab.get("vehicle_class"):
        parts.append(f"class={lab['vehicle_class']}")
    if lab.get("engine"):
        parts.append(f"engine={lab['engine']}")
    if lab["powertrain"] == "BEV":
        parts.append("make/model from README (2013 Nissan Leaf)" if np.isfinite(lab["footprint_m2"]) else "2013 Nissan Leaf per README (footprint not applied)")
    return "; ".join(parts)


# ----------------------------------------------------------------------------- one trip -> 10 Hz frame
def node_values(t: np.ndarray, v: np.ndarray, nodes: np.ndarray, bridge_s: float) -> np.ndarray:
    """Linear interpolation of the native samples at the node times; a node is NaN when it falls inside a source gap longer
    than bridge_s (the WHOLE gap decides, so a long gap is never partly bridged; a node on a sample is always valid)."""
    j = np.clip(np.searchsorted(t, nodes + 1e-9, side="right") - 1, 0, len(t) - 2)   # native interval [t[j], t[j+1]] around the node
    span = t[j + 1] - t[j]
    on_sample = (np.abs(nodes - t[j]) < 1e-6) | (np.abs(nodes - t[j + 1]) < 1e-6)
    out = np.interp(nodes, t, v)
    out[~(on_sample | (span <= bridge_s + 1e-9))] = np.nan
    return out


def trip_series(t_s: np.ndarray, v_kmh: np.ndarray, cfg: dict):
    """(df10hz, stats) of one trip or (None, stats). df columns t_s, v_mps, a_mps2, standstill, das_enabled; rows exist only
    where the speed is defined (grid gaps mark the source gaps > bridge_s)."""
    t = np.asarray(t_s, float)
    v = np.asarray(v_kmh, float)
    stats = dict(n_native=int(len(t)), n_speed_nan=int((~np.isfinite(v)).sum()), nonint_share=float("nan"), dense_share=float("nan"),
                 dt_mean=float("nan"), n_gap_native=0, n_grid=0, dur_s=float("nan"), n_glitch_steps=0, n_glitch_nodes=0)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    if len(t) >= 2:
        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]
        keep = np.r_[True, np.diff(t) > 0]
        t, v = t[keep], v[keep]
    if len(t) < 2:
        return None, stats
    d = np.diff(t)
    stats.update(nonint_share=float(((v % 1.0) != 0).mean()), dense_share=float((d <= DENSE_DT_S).mean()), dt_mean=float(d.mean()),
                 n_gap_native=int((d > cfg["bridge_s"]).sum()), dur_s=float(t[-1] - t[0]))
    res = float(cfg["speed_res_kph"])
    if cfg["native_timing"]:
        if res > 0:
            v = np.round(v / res) * res
        tn, vn, far_s, min_dt = t, v, cfg["bridge_s"], 0.5           # dense 0.1-s steps: judge the glitch over >= 0.5 s
    else:
        nodes = np.arange(np.ceil(t[0] - 1e-9), np.floor(t[-1] + 1e-9) + 1.0)        # integer trip seconds
        if len(nodes) < 2:
            return None, stats
        vn = node_values(t, v, nodes, cfg["bridge_s"])
        if res > 0:
            vn = np.round(vn / res) * res
        okn = np.isfinite(vn)
        if okn.sum() < 2:
            return None, stats
        tn, vn, far_s, min_dt = nodes[okn], vn[okn], 1.0 + 1e-6, 0.0                  # adjacent nodes only
    # segments: a step wider than far_s (non-adjacent nodes = an unbridged source gap) or a physically impossible step
    # (glitch rule) ends a segment, and interpolation never crosses a segment boundary; stale zero readings next to a
    # glitch step are dropped
    dvn = np.diff(vn) / KMH
    dtn = np.diff(tn)
    bad = np.abs(dvn) > float(cfg["a_max"]) * np.maximum(dtn, min_dt)
    far = dtn > far_s
    seg = np.cumsum(np.r_[0, bad | far])
    invalid = np.zeros(len(tn), bool)
    margin = float(cfg["zero_margin_s"])
    for i in np.flatnonzero(bad):
        if dvn[i] > 0:                                   # frozen at zero before an upward jump
            j = i
            while j >= 0 and vn[j] <= 0 and tn[i] - tn[j] < margin:
                invalid[j] = True
                j -= 1
        else:                                            # sudden drop to a frozen zero
            j = i + 1
            while j < len(tn) and vn[j] <= 0 and tn[j] - tn[i + 1] < margin:
                invalid[j] = True
                j += 1
    stats["n_glitch_steps"] = int(bad.sum())
    stats["n_glitch_nodes"] = int(invalid.sum())
    gs, vs = [], []
    for k in range(int(seg[-1]) + 1):
        m = (seg == k) & ~invalid
        if m.sum() < 2:
            continue
        g_k, v_k = cm.resample_to_10hz(tn[m], vn[m], "linear")          # every step inside a segment is adjacent / bridged
        f_k = np.isfinite(v_k)
        gs.append(g_k[f_k])
        vs.append(v_k[f_k])
    if not gs:
        return None, stats
    g = np.concatenate(gs)
    vg = np.concatenate(vs) / KMH
    if len(g) == 0:
        return None, stats
    brk = np.flatnonzero(np.diff(g) > cm.GAP_S + 1e-6) + 1
    a = np.full(len(g), np.nan)
    for s, e in zip(np.r_[0, brk], np.r_[brk, len(g)]):
        if e - s >= 3:
            a[s:e] = cm.smooth_derivative(vg[s:e], win_s=cfg["a_smooth_s"])
    ok = np.isfinite(a)
    g, vg, a = g[ok], vg[ok], a[ok]
    stats["n_grid"] = int(len(g))
    if len(g) == 0:
        return None, stats
    df = pd.DataFrame(dict(t_s=g, v_mps=vg.astype(np.float32), a_mps2=a.astype(np.float32),
                           standstill=(vg < cfg["standstill_v"]).astype(np.float32), das_enabled=np.zeros(len(g), np.float32)))
    return df, stats


# ----------------------------------------------------------------------------- one weekly file -> chunk on disk
def process_file(path: str, labels: dict[int, dict], cfg: dict, chunk_dir: str) -> dict:
    p = Path(path)
    cd = Path(chunk_dir)
    t0 = time.time()
    d = pd.read_csv(p, usecols=COLS)
    d = d.sort_values(["VehId", "Trip", "Timestamp(ms)"], kind="stable")
    Xs, metas, trips = [], [], []
    n_unlabelled = 0
    for (veh, trip), tr in d.groupby(["VehId", "Trip"], sort=False):
        veh, trip = int(veh), int(trip)
        lab = labels.get(veh)
        if lab is None:
            n_unlabelled += 1
            lab = dict(model="VED unknown", powertrain="unknown", footprint_m2=float("nan"), weight_lb=float("nan"), vehicle_class=None, engine=None)
        df, st = trip_series(tr["Timestamp(ms)"].to_numpy(float) / 1000.0, tr["Vehicle Speed[km/h]"].to_numpy(float), cfg)
        row = dict(VehId=veh, Trip=trip, drive_id=f"{veh}-{trip}", DayNum=float(tr["DayNum"].iloc[0]), file=p.name, powertrain=lab["powertrain"],
                   model=lab["model"], **st, n_rows=0, n_gap_rows=0, n_runs=0, n_candidates=0, n_dropped_gap=0, n_dropped_stationary=0, n_windows=0)
        if df is not None:
            info = dict(dataset=cfg["dataset"], vehicle_id=str(veh), model=lab["model"], platform=None, drive_id=f"{veh}-{trip}",
                        footprint_m2=lab["footprint_m2"], powertrain=lab["powertrain"], note=make_note(lab, cfg))
            r = cm.cut_windows(df, CHANNEL_MAP, t_col="t_s", info=info, align="run")
            row.update(r.stats)
            if r.X.shape[0]:
                Xs.append(r.X)
                metas.append(r.meta.assign(DayNum=row["DayNum"], VehId=veh, Trip=trip, file=p.name))
        trips.append(row)
    stem = p.stem
    n_w = 0
    if Xs:
        X = np.concatenate(Xs)
        meta = pd.concat(metas, ignore_index=True)
        meta["chunk"] = stem
        meta["row"] = np.arange(len(meta), dtype=np.int64)
        np.save(cd / f"{stem}.npy", X)
        meta.to_parquet(cd / f"{stem}.windows.parquet", index=False)
        n_w = int(len(meta))
    pd.DataFrame(trips).to_parquet(cd / f"{stem}.trips.parquet", index=False)
    return dict(file=p.name, n_rows=int(len(d)), n_trips=int(len(trips)), n_windows=n_w, n_unlabelled_trips=n_unlabelled, seconds=round(time.time() - t0, 1))


# ----------------------------------------------------------------------------- assembly, cap, save
def select_capped(all_meta: pd.DataFrame, cap: int) -> pd.DataFrame:
    """At most `cap` windows per vehicle, evenly spaced over the vehicle's windows ordered by (DayNum, drive, t0)."""
    picks = []
    for _, g in all_meta.groupby("VehId", sort=True):
        g = g.sort_values(["DayNum", "Trip", "t0_s"], kind="stable")
        n = len(g)
        if cap > 0 and n > cap:
            idx = np.unique(np.round(np.linspace(0, n - 1, cap)).astype(int))
            g = g.iloc[idx]
        picks.append(g)
    sel = pd.concat(picks, ignore_index=True) if picks else all_meta.iloc[0:0]
    return sel.sort_values(["VehId", "DayNum", "Trip", "w_idx"], kind="stable").reset_index(drop=True)


def gather_X(sel: pd.DataFrame, chunk_dir: Path) -> np.ndarray:
    X = np.empty((len(sel), cm.WINDOW_SAMPLES, cm.N_CHANNELS), dtype=np.float32)
    pos = np.arange(len(sel))
    for chunk, g in sel.groupby("chunk", sort=False):
        arr = np.load(chunk_dir / f"{chunk}.npy", mmap_mode="r")
        rows = g["row"].to_numpy(np.int64)
        X[pos[g.index.to_numpy()]] = np.asarray(arr[rows], dtype=np.float32)
    return X


def _hours(n: int) -> float:
    return round(n * cm.WINDOW_SAMPLES * cm.DT_S / 3600.0, 2)


def build_report(static: pd.DataFrame, labels: dict, trips: pd.DataFrame, all_meta: pd.DataFrame, meta: pd.DataFrame, cfg: dict,
                 file_stats: list[dict], leaf: dict, work: Path) -> dict:
    pt_order = ["ICE", "HEV", "PHEV", "BEV", "unknown"]
    st = static.copy()
    veh_all = all_meta.groupby("VehId").size().rename("win_all")
    veh_sel = meta.assign(VehId=meta["vehicle_id"].astype(int)).groupby("VehId").agg(win=("w_idx", "size"), drives=("drive_id", "nunique"))
    tv = trips.groupby("VehId").agg(trips=("Trip", "size"), trip_hours=("dur_s", lambda s: float(np.nansum(s)) / 3600.0),
                                    rows=("n_native", "sum"), dense=("dense_share", "mean"), nonint=("nonint_share", "mean"), dt_mean=("dt_mean", "mean"),
                                    n_candidates=("n_candidates", "sum"), n_dropped_gap=("n_dropped_gap", "sum"), n_dropped_stationary=("n_dropped_stationary", "sum"))
    veh = st.join(tv, how="left").join(veh_all, how="left").join(veh_sel, how="left")
    for c in ("trips", "rows", "win_all", "win", "drives", "n_candidates", "n_dropped_gap", "n_dropped_stationary"):
        veh[c] = veh[c].fillna(0).astype(int)
    veh["trip_hours"] = veh["trip_hours"].fillna(0.0)
    veh["model"] = [labels[int(i)]["model"] for i in veh.index]
    veh["footprint_m2"] = [labels[int(i)]["footprint_m2"] for i in veh.index]
    veh["size_class"] = [cm.classify_footprint(labels[int(i)]["footprint_m2"]) for i in veh.index]
    veh = veh.reset_index().rename(columns={"index": "VehId"})
    veh.to_parquet(work / "ved_vehicles.parquet", index=False)

    def per_group(key):
        rows = []
        for k, g in veh.groupby(key, sort=False, dropna=False):
            rows.append(dict(**{key: (k if not (isinstance(k, float) and np.isnan(k)) else "n/a")}, vehicles=int(len(g)), vehicles_with_windows=int((g["win"] > 0).sum()),
                             trips=int(g["trips"].sum()), trip_hours=round(float(g["trip_hours"].sum()), 1), windows_all=int(g["win_all"].sum()),
                             windows_kept=int(g["win"].sum()), hours_kept=_hours(int(g["win"].sum())), drives_kept=int(g["drives"].sum()),
                             median_windows_per_vehicle=float(g["win"].median()), vehicles_at_cap=int((g["win"] >= cfg["cap"]).sum()) if cfg["cap"] > 0 else 0,
                             dense_share_mean=round(float(np.nanmean(g["dense"])), 3) if g["dense"].notna().any() else None,
                             nonint_share_mean=round(float(np.nanmean(g["nonint"])), 3) if g["nonint"].notna().any() else None,
                             dt_mean=round(float(np.nanmean(g["dt_mean"])), 3) if g["dt_mean"].notna().any() else None))
        return rows

    pt_rows = sorted(per_group("powertrain"), key=lambda r: pt_order.index(r["powertrain"]) if r["powertrain"] in pt_order else 99)
    veh["weight_bin"] = veh["weight_lb"].map(lambda w: f"{int(w)}" if np.isfinite(w) else "n/a")
    wb_rows = sorted(per_group("weight_bin"), key=lambda r: (r["weight_bin"] == "n/a", float(r["weight_bin"]) if r["weight_bin"] != "n/a" else 0))
    model_rows = sorted(per_group("model"), key=lambda r: (pt_order.index(r["model"].split()[1]) if r["model"].startswith("VED ") and r["model"].split()[1] in pt_order else 98, r["model"]))
    cut = dict(trips=int(len(trips)), trips_with_series=int((trips["n_grid"] > 0).sum()), trips_with_windows=int((trips["n_windows"] > 0).sum()),
               candidates=int(trips["n_candidates"].sum()), dropped_gap=int(trips["n_dropped_gap"].sum()), dropped_stationary=int(trips["n_dropped_stationary"].sum()),
               windows_all=int(len(all_meta)), windows_kept=int(len(meta)), native_rows=int(trips["n_native"].sum()), speed_nan_rows=int(trips["n_speed_nan"].sum()),
               native_gaps_over_bridge=int(trips["n_gap_native"].sum()), glitch_steps=int(trips["n_glitch_steps"].sum()),
               glitch_nodes_dropped=int(trips["n_glitch_nodes"].sum()), trips_with_glitch=int((trips["n_glitch_steps"] > 0).sum()),
               trip_hours=round(float(np.nansum(trips["dur_s"])) / 3600.0, 1),
               grid_hours=round(float(trips["n_grid"].sum()) / 36000.0, 1))
    rep = dict(dataset=cfg["dataset"], config=cfg, leaf_spec=leaf, files=file_stats, cut=cut, per_powertrain=pt_rows, per_weight_bin=wb_rows, per_model=model_rows,
               n_vehicles_static=int(len(static)), n_vehicles_with_windows=int((veh["win"] > 0).sum()), n_models=int(meta["model"].nunique()),
               n_drives=int(meta["drive_id"].nunique()), hours_kept=_hours(int(len(meta))))
    (work / "ved_build_report.json").write_text(json.dumps(rep, indent=1, default=str))
    # markdown tables for DS_VED.md
    L = [f"# ved build tables ({time.strftime('%Y-%m-%d')})", "", "## Per powertrain", "",
         cm.md_table(["powertrain", "vehicles", "with windows", "trips", "trip h", "eligible windows", "kept windows", "kept h", "kept drives", "median win/veh", "at cap", "dense-step share", "non-integer speed share", "mean dt s"],
                     [[r["powertrain"], r["vehicles"], r["vehicles_with_windows"], r["trips"], r["trip_hours"], r["windows_all"], r["windows_kept"], r["hours_kept"], r["drives_kept"],
                       r["median_windows_per_vehicle"], r["vehicles_at_cap"], r["dense_share_mean"], r["nonint_share_mean"], r["dt_mean"]] for r in pt_rows]),
         "", "## Per weight bin (lb)", "",
         cm.md_table(["weight bin", "vehicles", "with windows", "trips", "trip h", "eligible windows", "kept windows", "kept h"],
                     [[r["weight_bin"], r["vehicles"], r["vehicles_with_windows"], r["trips"], r["trip_hours"], r["windows_all"], r["windows_kept"], r["hours_kept"]] for r in wb_rows]),
         "", "## Per model (powertrain x weight bin)", "",
         cm.md_table(["model", "vehicles", "with windows", "kept windows", "kept drives", "kept h"],
                     [[r["model"], r["vehicles"], r["vehicles_with_windows"], r["windows_kept"], r["drives_kept"], r["hours_kept"]] for r in model_rows]),
         "", "## Cutting", "", cm.md_table(["item", "value"], [[k, v] for k, v in cut.items()]), ""]
    (work / "ved_tables.md").write_text("\n".join(L))
    return rep


def cmd_build(a: argparse.Namespace) -> None:
    raw = Path(a.raw)
    work = cm.work_dir(a.dataset)
    work.mkdir(parents=True, exist_ok=True)
    chunk_dir = work / "tmp_chunks"
    if chunk_dir.exists() and not a.resume:
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(dataset=a.dataset, bridge_s=float(a.bridge_s), a_smooth_s=float(a.a_smooth_s), standstill_v=float(a.standstill_v),
               a_max=float(a.a_max), zero_margin_s=float(a.glitch_zero_margin_s),
               speed_res_kph=0.0 if a.native_resolution else float(a.speed_res_kph), native_timing=bool(a.native_timing), cap=int(a.cap),
               use_leaf_footprint=not a.no_leaf_footprint, align="run", gap_s=cm.GAP_S, min_moving_frac=cm.MIN_MOVING_FRAC)
    static = load_static(raw)
    leaf = leaf_spec()
    labels = vehicle_labels(static, cfg["use_leaf_footprint"], leaf)
    files = sorted((raw / "extracted").glob("VED_DynamicData_Part*/VED_*_week.csv"))
    if not files:
        raise SystemExit(f"no weekly CSVs under {raw / 'extracted'}")
    if a.limit_files:
        files = files[: a.limit_files]
    print(f"[ved] {len(files)} weekly files, {len(static)} vehicles in the static tables "
          f"({static['powertrain'].value_counts().to_dict()}), config {cfg}")
    todo = [f for f in files if not (a.resume and (chunk_dir / f"{f.stem}.trips.parquet").exists())]
    file_stats: list[dict] = []
    t0 = time.time()
    if a.workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(process_file, str(f), labels, cfg, str(chunk_dir)): f for f in todo}
            for fut in as_completed(futs):
                r = fut.result()
                file_stats.append(r)
                print(f"[ved] {r['file']}: rows {r['n_rows']:,} trips {r['n_trips']:,} windows {r['n_windows']:,} ({r['seconds']}s)")
    else:
        for f in todo:
            r = process_file(str(f), labels, cfg, str(chunk_dir))
            file_stats.append(r)
            print(f"[ved] {r['file']}: rows {r['n_rows']:,} trips {r['n_trips']:,} windows {r['n_windows']:,} ({r['seconds']}s)")
    file_stats = sorted(file_stats, key=lambda r: r["file"])
    print(f"[ved] pass 1 done in {time.time() - t0:.0f}s")
    # assembly
    trips = pd.concat([pd.read_parquet(p) for p in sorted(chunk_dir.glob("VED_*_week.trips.parquet"))], ignore_index=True)
    wparts = sorted(chunk_dir.glob("VED_*_week.windows.parquet"))
    all_meta = pd.concat([pd.read_parquet(p) for p in wparts], ignore_index=True) if wparts else None
    if all_meta is None or not len(all_meta):
        raise SystemExit("no eligible windows")
    trips.to_parquet(work / "ved_trips.parquet", index=False)
    all_meta.to_parquet(work / "ved_all_windows.parquet", index=False)
    sel = select_capped(all_meta, cfg["cap"])
    print(f"[ved] eligible windows {len(all_meta):,} from {all_meta['VehId'].nunique()} vehicles -> kept {len(sel):,} (cap {cfg['cap']} / vehicle)")
    X = gather_X(sel, chunk_dir)
    meta = sel[[c for c, _ in cm.META_COLUMNS]].reset_index(drop=True)
    out = cm.save_windows(a.dataset, X, meta)
    rep = build_report(static, labels, trips, all_meta, meta, cfg, file_stats, leaf, work)
    print(f"[ved] per powertrain: " + ", ".join(f"{r['powertrain']} {r['windows_kept']} win / {r['vehicles_with_windows']} veh" for r in rep["per_powertrain"]))
    print(f"[ved] warnings: {out['warnings']}")
    if not a.keep_chunks:
        shutil.rmtree(chunk_dir, ignore_errors=True)
    print(f"[ved] wrote {out['npz']}, {out['parquet']}, {work / 'ved_build_report.json'}, {work / 'ved_tables.md'} in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- validation
def cmd_validate(a: argparse.Namespace) -> None:
    X, meta = cm.load_windows(a.dataset)
    work = cm.work_dir(a.dataset)
    rep_path = work / "ved_build_report.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    cfg = rep.get("config", {})
    cap = int(cfg.get("cap", a.cap))
    problems: list[str] = []
    notes: list[str] = []

    def check(cond: bool, msg: str):
        (notes if cond else problems).append(("OK   " if cond else "FAIL ") + msg)

    n = X.shape[0]
    check(X.shape[1:] == (cm.WINDOW_SAMPLES, cm.N_CHANNELS) and X.dtype == np.float32, f"shape {X.shape} float32")
    check(len(meta) == n, f"meta rows {len(meta)} = windows {n}")
    present = ",".join(PRESENT)
    check(bool((meta["channels_present"] == present).all()), f"channels_present == '{present}' on every window")
    for c in PRESENT:
        col = X[:, :, cm.CH[c]]
        check(bool(np.isfinite(col).all()), f"{c}: no NaN ({int((~np.isfinite(col)).sum())} NaN samples)")
    absent = [c for c in cm.CHANNELS if c not in PRESENT]
    for c in absent:
        col = X[:, :, cm.CH[c]]
        check(bool(np.isnan(col).all()), f"{c}: NaN everywhere (missing channel)")
    v = X[:, :, cm.CH["v_mps"]]
    acc = X[:, :, cm.CH["a_mps2"]]
    ss = X[:, :, cm.CH["standstill"]]
    das = X[:, :, cm.CH["das_enabled"]]
    check(bool(np.isin(ss, (0.0, 1.0)).all() and np.isin(das, (0.0, 1.0)).all()), "flags standstill / das_enabled are 0 / 1")
    check(bool((das == 0).all()) and bool((meta["das_frac"].fillna(0) == 0).all()), "das_enabled = 0 everywhere, das_frac = 0")
    sv = float(cfg.get("standstill_v", 0.1))
    check(bool(((v < sv).astype(np.float32) == ss).all()), f"standstill == (v < {sv} m/s)")
    vmin, vmax = float(v.min()), float(v.max())
    check(vmin >= 0.0 and vmax <= 60.0, f"speed range [{vmin:.2f}, {vmax:.2f}] m/s inside [0, 60] (max {vmax * KMH:.0f} km/h)")
    q = np.quantile(acc, [0.001, 0.01, 0.5, 0.99, 0.999])
    n_a5 = int((np.abs(acc) > 5).sum())
    n_a8 = int((np.abs(acc) > 8).sum())
    check(n_a8 == 0, f"|a| <= 8 m/s^2 everywhere (|a| > 5: {n_a5} samples = {n_a5 / acc.size * 100:.4f} %, |a| > 8: {n_a8}); a quantiles 0.1/1/50/99/99.9 % = {np.round(q, 3).tolist()}")
    mf = (v >= cm.MOVING_V_MPS).mean(1)
    check(bool((mf >= cm.MIN_MOVING_FRAC).all()), f"moving rule: min moving_frac {mf.min():.3f} >= {cm.MIN_MOVING_FRAC}; mean moving share {mf.mean():.3f}, standstill share {ss.mean():.3f}")
    check(bool(np.allclose(meta["moving_frac"].to_numpy(np.float32), mf.astype(np.float32), atol=1e-3)), "meta.moving_frac matches X")
    # jumps: consecutive-sample speed steps (gap-free by construction; the glitch rule bounds node steps by a_max x dt)
    a_max = float(cfg.get("a_max", A_MAX_DEFAULT))
    dv = np.abs(np.diff(v, axis=1))
    check(float(dv.max()) <= a_max * cm.DT_S + 2e-3, f"max |dv| between consecutive 10 Hz samples {dv.max():.3f} m/s <= a_max x 0.1 s = {a_max * cm.DT_S:.2f}")
    # meta / labels
    per_veh = meta.groupby("vehicle_id").size()
    check(bool((per_veh <= cap).all()) if cap > 0 else True, f"<= {cap} windows per vehicle (max {int(per_veh.max())}, vehicles at cap {int((per_veh >= cap).sum())}, vehicles {len(per_veh)})")
    for c in ("model", "powertrain", "size_class"):
        k = meta.groupby("vehicle_id")[c].nunique(dropna=False)
        check(bool((k <= 1).all()), f"one {c} per vehicle")
    check(bool(meta["drive_id"].str.match(r"^\d+-\d+$").all()), "drive_id = '<VehId>-<Trip>'")
    check(bool(meta["vehicle_id"].str.match(r"^\d+$").all()), "vehicle_id = VehId")
    check(bool((meta["drive_id"].str.split("-").str[0] == meta["vehicle_id"]).all()), "drive_id prefix equals vehicle_id")
    check(not meta.duplicated(["drive_id", "w_idx"]).any(), "(drive_id, w_idx) unique")
    check(bool(meta["note"].str.contains("upsampled from 1 Hz").all()), "every note carries 'upsampled from 1 Hz'")
    check(bool(meta["powertrain"].isin(["ICE", "HEV", "PHEV", "BEV"]).all()), f"powertrain vocabulary {sorted(meta['powertrain'].unique())}")
    leaf = meta[meta["powertrain"] == "BEV"]
    if cfg.get("use_leaf_footprint", True):
        check(bool((leaf["size_class"] == "SMALL").all() and np.allclose(leaf["footprint_m2"], float(rep.get("leaf_spec", {}).get("footprint_m2", 4.149)), atol=1e-3) and (~leaf["borderline"]).all()),
              f"Leaf windows: size_class SMALL, footprint {leaf['footprint_m2'].iloc[0] if len(leaf) else 'n/a'}, not borderline ({len(leaf)} windows, {leaf['vehicle_id'].nunique()} vehicles)")
        check(bool(meta.loc[meta['powertrain'] != 'BEV', 'size_class'].isna().all()), "size_class NaN for every non-Leaf vehicle")
    else:
        check(bool(meta["size_class"].isna().all()), "size_class NaN everywhere (--no-leaf-footprint)")
    blob = " ".join(meta[c].astype(str).str.cat(sep=" ") for c in ("vehicle_id", "model", "drive_id", "note"))
    check(not HEX16.search(blob), "no 16-hex identifier-like token in meta strings")
    # tables
    tab = meta.groupby("powertrain").agg(vehicles=("vehicle_id", "nunique"), drives=("drive_id", "nunique"), windows=("w_idx", "size")).reindex(["ICE", "HEV", "PHEV", "BEV"]).dropna(how="all")
    tab["hours"] = (tab["windows"] * cm.WINDOW_SAMPLES * cm.DT_S / 3600.0).round(1)
    tab["v_mean_mps"] = [float(v[(meta["powertrain"] == p).to_numpy()].mean()) for p in tab.index]
    tab["moving_share"] = [float(mf[(meta["powertrain"] == p).to_numpy()].mean()) for p in tab.index]
    tab["a_std_moving"] = [float(acc[(meta["powertrain"] == p).to_numpy()][v[(meta["powertrain"] == p).to_numpy()] >= 1].std()) for p in tab.index]
    model_tab = meta.groupby("model").agg(vehicles=("vehicle_id", "nunique"), drives=("drive_id", "nunique"), windows=("w_idx", "size"), powertrain=("powertrain", "first"),
                                           size_class=("size_class", "first")).sort_values(["powertrain", "model"])
    # acceleration statistics against the training reference
    A = acc
    def ac(l):
        x = A[:, :-l]; y = A[:, l:]; xm = x - x.mean(); ym = y - y.mean()
        return float((xm * ym).mean() / np.sqrt((xm ** 2).mean() * (ym ** 2).mean()))
    a_stats = dict(std_moving=float(acc[v >= 1].std()), q01=float(np.quantile(acc[v >= 1], 0.01)), q99=float(np.quantile(acc[v >= 1], 0.99)),
                   autocorr_lag_1_5_10_20=[round(ac(l), 3) for l in (1, 5, 10, 20)], jerk_std=float((np.diff(A, axis=1) * cm.HZ).std()),
                   training_reference=dict(std_moving=0.615, q01=-1.98, q99=1.80, autocorr_lag_1_5_10_20=[0.961, 0.879, 0.767, 0.554], jerk_std=1.66))
    v_stats = dict(min=vmin, max=vmax, mean=float(v.mean()), q50=float(np.median(v)), q99=float(np.quantile(v, 0.99)), moving_share=float(mf.mean()), standstill_share=float(ss.mean()),
                   training_reference=dict(q50=17.6, q99=36.3, moving_share=0.915, standstill_share=0.074))
    out = dict(dataset=a.dataset, n_windows=int(n), n_vehicles=int(meta["vehicle_id"].nunique()), n_drives=int(meta["drive_id"].nunique()), n_models=int(meta["model"].nunique()),
               hours=_hours(int(n)), per_powertrain=tab.reset_index().to_dict(orient="records"), per_model=model_tab.reset_index().to_dict(orient="records"),
               speed=v_stats, acceleration=a_stats, checks_ok=notes, checks_failed=problems)
    (work / "ved_validation.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"[ved validate] windows {n:,} vehicles {out['n_vehicles']} drives {out['n_drives']:,} models {out['n_models']} hours {out['hours']}")
    print(tab.round(3).to_string())
    print(model_tab.to_string())
    print("speed:", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v_stats.items() if k != "training_reference"})
    print("acceleration:", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in a_stats.items() if k != "training_reference"})
    for line in notes + problems:
        print(line)
    print(f"[ved validate] {len(problems)} problems -> {work / 'ved_validation.json'}")
    if problems:
        sys.exit(1)


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="read the raw files, cut and save the windows")
    b.add_argument("--raw", default=str(RAW_DEFAULT), help="external/ved/raw (contains VED/ and extracted/)")
    b.add_argument("--dataset", default=DATASET, help="work folder name under external/ (default ved)")
    b.add_argument("--bridge-s", type=float, default=3.0, help="largest source gap bridged by linear interpolation (s); longer gaps break the run")
    b.add_argument("--a-smooth-s", type=float, default=1.0, help="moving-average window of the speed derivative (s); 1.0 = one source interval")
    b.add_argument("--a-max", type=float, default=A_MAX_DEFAULT, help="node-to-node |dv|/dt above this (m/s^2) is a glitch: the run is split there")
    b.add_argument("--glitch-zero-margin-s", type=float, default=ZERO_MARGIN_DEFAULT, help="zero-speed nodes within this many seconds of a glitch step are dropped")
    b.add_argument("--standstill-v", type=float, default=0.1, help="standstill = v < this (m/s); the training flag is 1 at 99 %% of v = 0 and 1.5 %% of 0.1-0.3 m/s")
    b.add_argument("--speed-res-kph", type=float, default=1.0, help="round the speed to this resolution before upsampling (km/h); 0 = native")
    b.add_argument("--native-resolution", action="store_true", help="keep the native speed resolution (same as --speed-res-kph 0)")
    b.add_argument("--native-timing", action="store_true", help="interpolate the native stamps directly to 10 Hz instead of through 1-s nodes")
    b.add_argument("--cap", type=int, default=200, help="max windows per vehicle (0 = no cap), evenly spaced in time")
    b.add_argument("--no-leaf-footprint", action="store_true", help="do not give the three Leafs the ZE0 footprint / SMALL label")
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--limit-files", type=int, default=0, help="smoke test: only the first N weekly files")
    b.add_argument("--resume", action="store_true", help="reuse chunks already in work/tmp_chunks")
    b.add_argument("--keep-chunks", action="store_true")
    b.set_defaults(func=cmd_build)
    vld = sub.add_parser("validate", help="check the saved windows against the contract and the physical ranges")
    vld.add_argument("--dataset", default=DATASET)
    vld.add_argument("--cap", type=int, default=200)
    vld.set_defaults(func=cmd_validate)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
