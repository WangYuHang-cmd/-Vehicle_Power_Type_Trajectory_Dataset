#!/usr/bin/env python3
"""OpenACC (EC JRC, CC BY 4.0) adapter for the zero-shot transfer test (experiments/transfer, CANDIDATES.md 2.4).

Reads the raw campaign CSVs under <external>/openacc/raw/<campaign>/ (platoon files: 5 metadata rows, header, 10 Hz
rows; column suffix n = n-th vehicle of the `Vehicle_order` metadata row) and the one-vehicle campaign
(`D<driver>_part<n>.csv`, columns Time / Speed / Engine Speed / Gear), builds the longitudinal-only channel set of the
training contract and writes two window sets with common.save_windows:

  openacc        the six platoon campaigns (AstaZero, Casale, Cherasco, JRC low speed, Vicolungo, ZalaZone): one window
                 set per (vehicle, run file); v_mps = GNSS Doppler speed, a_mps2 = smoothed derivative of v, standstill
                 = v < STANDSTILL_V_MPS, das_enabled = per-sample Driver column (ACC / Human) where present else the
                 campaign flag / protocol default, lead group from the inter-vehicle spacing IVS(n-1) and the speed of
                 the vehicle ahead (lead_v_rel = v_lead - v_ego), leaders lead_valid = 0; lateral and pedal channels NaN
                 (score with --mask-channels V2 or the longitudinal-only variant).
  openacc_500x   the one-vehicle / 20-driver Fiat 500X campaign (438 trips, ~336 h, manual driving): speed-only set
                 (v, derived a, standstill; das_enabled = 0, lead_valid = 0); engine speed and gear are never read
                 (score with --mask-channels V3 or the speed-only variant).

Labels: Vehicle_order header name -> canonical public model name -> specs_external.csv (footprint, size class,
powertrain, borderline) cross-checked against the campaign vehicle-specification CSV (propulsion type, power, model
year). Excluded by default: the Peugeot 3008 / 5008 (identity unresolved), the unnamed BMW / Smart / Skoda leaders, the
hybrids and the fuel-cell car (Ioniq hybrid, Niro PHEV, Outlander PHEV, i3s REx, RAV4 hybrid, Nexo; --keep-hybrids keeps
them with their HEV / PHEV / FCEV labels, i.e. outside the BEV-vs-ICE task but usable for the size task). The two
48 V / 12 V mild hybrids (Mercedes GLE 450, Audi A4 Avant) are kept for the size task and labelled HEV as in the JRC
spec file, so the scorer keeps them out of the BEV / ICE score (--mhev-as-ice applies the lab's mild-hybrid = ICE
convention instead; --drop-mhev removes them).

Speed QC (both sets, clean_speed): speed outside [-1, 70] m/s -> missing; a frozen sensor (>= 30 identical consecutive values
= 3.0 s while v > 0.5 m/s, e.g. 1.8 h of constant 25.8 m/s in one 500X trip) -> missing; an impossible step (|dv| / dt >
15 m/s^2 = 1.5 m/s per 0.1 s) splits the series (no window straddles it) and removes the stale hold of >= 2 identical values
before the jump. Spacing QC: IVS <= 0.5 m (incl. the documented negative spacings) or > 150 m (radar-range emulation;
training leads never exceed 165 m) -> lead_valid 0 on those samples. Missing speed samples are dropped; the lab gap rule
(> 0.3 s between consecutive samples) and the moving rule (>= 20 % of samples with v >= 1 m/s) then decide the windows.

Run:  source env.sh && $CPY experiments/transfer/ds_openacc.py [--raw DIR] [--root DIR] [--campaigns A,B] [--skip-one-vehicle]
      [--only-one-vehicle] [--keep-hybrids] [--drop-mhev] [--mhev-as-ice] [--no-merge-fleet] [--standstill-v 0.2]
      [--lead-max-m 150] [--align run|drive] [--dry-run] [--limit-files N]
Outputs: <external>/openacc/work/{windows.npz, meta.parquet, windows_summary.json, qc_vehicle_files.csv, qc_summary.json,
qc_tables.md}; <external>/openacc_500x/work/{windows.npz, meta.parquet, windows_summary.json, qc_vehicle_files.csv,
qc_summary.json, qc_tables.md}. The raw folder is never written.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.transfer import common as T  # noqa: E402
from vtt import config as C  # noqa: E402

DATASET = "openacc"
DATASET_500X = "openacc_500x"
RAW_DEFAULT = T.EXTERNAL_ROOT / "openacc" / "raw"
SPECS_CSV = Path(__file__).with_name("specs_external.csv")
ONE_VEHICLE_DIR = "One_vehicle_multiple_drivers_on_road_campaign"
PLATOON_CAMPAIGNS = ("AstaZero", "Casale", "Cherasco", "JRC low speed", "Vicolungo", "ZalaZone")

STANDSTILL_V_MPS = 0.2      # training: standstill = CAN flag; 99.9 % of standstill samples have v < 0.10 m/s and 94 % of the
                            # training samples with v < 0.2 m/s are standstill -> threshold for GNSS Doppler speed (noise ~0.01-0.2)
LEAD_MAX_M = 150.0          # training leads come from radar: 99.9 % of valid lead_d_m < 135 m, max 165 m (clip 200)
LEAD_MIN_M = float(C.LEAD_VALID_DREL_MIN_M)   # 0.5 m, the lab's lead_valid distance condition
V_MIN_PLAUSIBLE, V_MAX_PLAUSIBLE = -1.0, 70.0  # GNSS speed outside -> treated as missing (counted)
HOLD_MAX_SAMPLES = 30       # >= 30 identical consecutive speed values (3.0 s) while moving = frozen / held sensor sample -> NaN
                            # (AstaZero / Vicolungo speeds are quantised to 0.01 m/s and show legitimate 1-2 s plateaus on ACC)
HOLD_V_MPS = 0.5            # plateaus at or below this speed are real standstills, never "frozen"
STEP_MAX_ACC = 15.0         # |dv| / dt above this (m/s^2, = 1.5 m/s per 0.1 s) is impossible for a car -> series split (stale hold removed)
STEP_MAX_MPS = STEP_MAX_ACC / T.HZ
EQ_TOL_MPS = 1e-5           # two speed samples closer than this are "identical" (the spline of a stale GNSS speed repeats to 1e-9;
                            # float32 windows resolve ~4e-6 at 30 m/s; no real speed change over 0.1 s is that small)

# campaign acquisition notes carried into meta.note
CAMPAIGN_NOTE = {
    "AstaZero": "AstaZero test track 2019, OxTS RT-Range dGNSS 100 Hz downsampled to 10 Hz",
    "Casale": "Ispra-Casale public roads 2020, u-blox 9 GNSS cubic-splined to 10 Hz",
    "Cherasco": "Ispra-Cherasco freeway 2018, u-blox 8 GNSS 3-5 Hz cubic-splined to 10 Hz",
    "JRC low speed": "JRC Ispra site 2020, low speed, u-blox 9 GNSS cubic-splined to 10 Hz (gaps <= 1 s interpolated by JRC)",
    "Vicolungo": "Ispra-Vicolungo motorway 2019, u-blox 8 GNSS 3-5 Hz cubic-splined to 10 Hz",
    "ZalaZone": "ZalaZONE proving ground 2019, 30-60 km/h, VBOX + u-blox fused, 2nd-degree B-spline to 10 Hz",
    "OneVehicle": "one Fiat 500X, 20 volunteer drivers 2016-17, CAN/OBD speed interpolated to 10 Hz, manual driving",
}


@dataclass(frozen=True)
class Veh:
    model: str | None          # canonical public model name (specs_external.csv `model`), None = excluded
    spec_name: str | None      # row name in the campaign's vehicle-specification CSV (None when absent)
    exclude: str | None = None  # None (included), 'unnamed', 'ambiguous', 'hybrid', 'mhev'
    fleet_id: str | None = None  # physical-car id shared across campaigns (same JRC fleet car), None = per campaign


# Vehicle_order header token -> vehicle; one dict per campaign (names exactly as they appear in the CSV headers)
VEHICLES: dict[str, dict[str, Veh]] = {
    "AstaZero": {
        "Audi(A8)": Veh("Audi A8", "Audi (A8)"),
        "Tesla(Model3)": Veh("Tesla Model 3", "Tesla (Model 3)"),
        "BMW(X5)": Veh("BMW X5", "BMW (X5)"),
        "Audi(A6)": Veh("Audi A6", "Audi (A6)"),
        "Mercedes(AClass)": Veh("Mercedes-Benz A-Class", "Mercedes (A Class)"),
    },
    "Casale": {
        "Rexton": Veh("SsangYong Rexton", "SsangYong REXTON"),
        "Hyundai": Veh("Hyundai Nexo", "Hyundai Nexo", exclude="hybrid", fleet_id="JRC fleet/Hyundai Nexo"),
    },
    "Cherasco": {
        "Fiat(500X)": Veh("Fiat 500X", "Fiat (500X)", fleet_id="JRC fleet/Fiat 500X"),
        "Hyundai(Ioniqhybrid)": Veh("Hyundai Ioniq Hybrid", "Hyundai (Ioniq hybrid)", exclude="hybrid"),
        "Volvo(XC40)": Veh("Volvo XC40", "Volvo (XC40)"),
        "VW(Polo)": Veh("Volkswagen Polo", "VW (Polo)"),
    },
    "JRC low speed": {
        "BMW": Veh(None, None, exclude="unnamed"),
        "Hyundai": Veh("Hyundai Nexo", "Hyundai Nexo", exclude="hybrid", fleet_id="JRC fleet/Hyundai Nexo"),
        "Ford": Veh("Ford S-Max", "Ford S-Max", fleet_id="JRC fleet/Ford S-Max"),
    },
    "Vicolungo": {
        "Mitsubishi(SpaceStar)": Veh("Mitsubishi Space Star", "Mitsubishi (SpaceStar)"),
        "Ford(S-Max)": Veh("Ford S-Max", "Ford S-Max", fleet_id="JRC fleet/Ford S-Max"),
        "Peugeot(3008GTLine)": Veh(None, "Peugeot (5008 GT Line)", exclude="ambiguous"),
        "KIA(Niro)": Veh("Kia Niro PHEV", "KIA (Niro)", exclude="hybrid"),
        "Mini(Cooper)": Veh("MINI Cooper", "Mini (Cooper)"),
        "VW(GolfE)": Veh("Volkswagen e-Golf", "VW (Golf E)"),
        "Mitsubishi(OutlanderPHEV)": Veh("Mitsubishi Outlander PHEV", "Mitsubishi (Outlander PHEV)", exclude="hybrid"),
    },
    "ZalaZone": {
        "SMART_TARGET": Veh(None, None, exclude="unnamed"),
        "SKODA_TARGET": Veh(None, None, exclude="unnamed"),
        "BMW_I3": Veh("BMW i3s REx", "BMW (I3 s)", exclude="hybrid"),
        "MERCEDES_GLE450": Veh("Mercedes-Benz GLE 450 4MATIC", "Mercedes-Benz (GLE 450 4Matic)", exclude="mhev"),
        "JAGUAR_I_PACE": Veh("Jaguar I-Pace", "Jaguar (I-Pace)"),
        "TESLA_MODELX": Veh("Tesla Model X", "Tesla (model X)"),
        "TESLA_MODEL3": Veh("Tesla Model 3", "Tesla (model 3)"),
        "TESLA_MODELS": Veh("Tesla Model S", "Tesla (model S)"),
        "AUDI_E_TRON": Veh("Audi e-tron", "Audi (E-tron)"),
        "TOYOTA_RAV4": Veh("Toyota RAV4 Hybrid", "Toyota (Rav 4)", exclude="hybrid"),
        "MAZDA_3": Veh("Mazda 3", "Mazda (3)"),
        "AUDI_A4": Veh("Audi A4 Avant", "Audi (A4 Avant)", exclude="mhev"),
    },
    "OneVehicle": {
        "Fiat500X": Veh("Fiat 500X", "Fiat500X", fleet_id="JRC fleet/Fiat 500X"),
    },
}
# the Tesla Model 3 of AstaZero (150 kW, spec file) and of ZalaZone (250 kW) are different cars -> no fleet_id


# ----------------------------------------------------------------------------- specs
def load_specs(path: Path = SPECS_CSV) -> pd.DataFrame:
    """specs_external.csv rows that serve OpenACC, indexed by canonical model name (one row per model)."""
    df = pd.read_csv(path)
    m = df["datasets"].fillna("").str.contains("OpenACC", case=False)
    s = df[m].copy()
    dup = s["model"][s["model"].duplicated()].tolist()
    if dup:
        raise RuntimeError(f"specs_external.csv has several OpenACC rows for {dup}")
    return s.set_index("model")


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_campaign_spec(camp_dir: Path) -> pd.DataFrame | None:
    """The campaign's vehicle-specification CSV (Vehicles, Max power, Drive-Fuel, ..., Propulsion type, Model year)."""
    cands = [p for p in camp_dir.glob("*.csv") if "pecification" in p.name]
    if not cands:
        return None
    df = pd.read_csv(cands[0])
    df.columns = [c.strip() for c in df.columns]
    df["Vehicles"] = df["Vehicles"].astype(str).str.strip()
    df["_key"] = df["Vehicles"].map(_norm_name)
    return df


def spec_row(camp_spec: pd.DataFrame | None, spec_name: str | None) -> dict:
    if camp_spec is None or spec_name is None:
        return {}
    hit = camp_spec[camp_spec["_key"] == _norm_name(spec_name)]
    if len(hit) != 1:
        return {}
    r = hit.iloc[0]
    return dict(propulsion=str(r.get("Propulsion type", "")).strip(), power_kw=str(r.get("Max power (kW)", "")).strip(),
                fuel=str(r.get("Drive-Fuel", "")).strip(), model_year=str(r.get("Model year", "")).strip())


# ----------------------------------------------------------------------------- raw readers
def read_platoon_file(path: Path) -> tuple[dict, pd.DataFrame]:
    """(metadata, frame) of one platoon CSV: metadata = Date, Vehicle_order (list), Number_of_vehicles, ACC ('0'/'1'/'2'/''),
    Distance_setting; frame columns stripped (Time, Speed<n>, ..., IVS<n>, Driver<n>)."""
    meta: dict = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for _ in range(5):
            line = fh.readline().rstrip("\r\n")
            parts = [p.strip() for p in line.split(",")]
            key, vals = parts[0], [p for p in parts[1:] if p != ""]
            meta[key] = vals
    md = dict(date=meta.get("Date", []), vehicle_order=meta.get("Vehicle_order", []),
              n_vehicles=int(meta.get("Number_of_vehicles", ["0"])[0]) if meta.get("Number_of_vehicles") else 0,
              acc=(meta.get("ACC") or [""])[0], distance=" ".join(meta.get("Distance_setting") or []))
    df = pd.read_csv(path, skiprows=5, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return md, df


def finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(start, stop)) of the True runs of a boolean mask."""
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def clean_speed(v: np.ndarray, t: np.ndarray | None = None, *, hold_max: int = HOLD_MAX_SAMPLES, step_max_acc: float = STEP_MAX_ACC,
                hold_v: float = HOLD_V_MPS, gap_s: float = T.GAP_S) -> tuple[np.ndarray, np.ndarray, dict]:
    """Speed QC of one series on the 10 Hz grid -> (v_clean, segment id per sample (-1 = missing), counters).
    Both tests run on the sequence of finite samples (a missing sample in between does not reset them):
    1. frozen sensor: >= hold_max identical (within EQ_TOL_MPS) consecutive finite values at v > hold_v (a held / stale
       sample, e.g. 1.8 h of constant 25.8 m/s in one 500X trip, or a stale GNSS speed carried through the JRC spline for
       3-4 s; 3 s at the 0.01 m/s quantisation of the AstaZero speeds is far beyond the 1-2 s plateaus a car on ACC
       produces) -> NaN;
    2. impossible step: |dv| / dt > step_max_acc (15 m/s^2, beyond any car) between finite samples at most gap_s apart ->
       the series is split there (no window straddles the jump) and a hold of >= 2 identical values ending at the jump
       (the stale samples before it) -> NaN.
    Segments: finite samples at most gap_s apart belong to one segment (the lab's hole tolerance), a larger hole or a
    step starts a new one; the derivative and the windows are computed per segment."""
    v = np.array(v, dtype=np.float64, copy=True)
    n = len(v)
    t = np.arange(n, dtype=np.float64) / T.HZ if t is None else np.asarray(t, dtype=np.float64)
    qc = dict(n_v_frozen=0, n_step_breaks=0, n_hold_before_step=0)
    seg = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return v, seg, qc
    fin = np.flatnonzero(np.isfinite(v))
    if len(fin) == 0:
        return v, seg, qc
    vf = v[fin]
    eq = np.zeros(len(fin), dtype=bool)
    eq[1:] = np.abs(vf[1:] - vf[:-1]) <= EQ_TOL_MPS
    for s, e in finite_runs(eq):                      # finite samples fin[s-1 : e] share one value
        length = e - s + 1
        if length >= hold_max and np.max(np.abs(vf[s - 1:e])) > hold_v:
            v[fin[s - 1:e]] = np.nan
            qc["n_v_frozen"] += int(length)
    fin = np.flatnonzero(np.isfinite(v))
    if len(fin) == 0:
        return v, seg, qc
    vf, tf = v[fin], t[fin]
    dt = np.diff(tf)
    dv = np.diff(vf)
    bad = np.flatnonzero((dt <= gap_s + 1e-6) & (np.abs(dv) > step_max_acc * np.maximum(dt, 1e-3)))
    break_at = np.zeros(n, dtype=bool)                # grid index that starts a new segment
    for k in bad:
        break_at[fin[k + 1]] = True
        qc["n_step_breaks"] += 1
        j = k
        while j - 1 >= 0 and abs(vf[j - 1] - vf[k]) <= EQ_TOL_MPS:
            j -= 1
        if k - j + 1 >= 2:                            # a hold of >= 2 identical values ends at the jump: stale samples
            v[fin[j:k + 1]] = np.nan
            qc["n_hold_before_step"] += int(k - j + 1)
    fin = np.flatnonzero(np.isfinite(v))
    if len(fin) == 0:
        return v, seg, qc
    new = np.ones(len(fin), dtype=bool)
    new[1:] = (np.diff(t[fin]) > gap_s + 1e-6) | break_at[fin[1:]]
    seg[fin] = np.cumsum(new) - 1
    return v, seg, qc


def derived_accel(v: np.ndarray, seg: np.ndarray | None = None, hz: int = T.HZ) -> np.ndarray:
    """common.smooth_derivative per segment (holes of missing samples inside a segment are bridged linearly for the
    differentiation only; the result is kept at the finite samples; edge-padded inside the segment, NaN elsewhere);
    segments with fewer than 3 samples stay NaN."""
    a = np.full(len(v), np.nan)
    if seg is None:
        seg = np.where(np.isfinite(v), 0, -1)
    for k in np.unique(seg[seg >= 0]):
        idx = np.flatnonzero(seg == k)
        if len(idx) < 3:
            continue
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        grid = np.arange(lo, hi)
        vv = np.interp(grid, idx, v[idx])             # bridge the holes (<= gap_s) for the derivative only
        a[idx] = T.smooth_derivative(vv, hz=hz, win_s=0.5)[idx - lo]
    return a


def cut_frame_segments(frame: pd.DataFrame, channel_map: dict, info: dict, align: str) -> T.CutResult:
    """common.cut_windows per segment of `frame['seg']` (windows never straddle a segment break); w_idx is offset so that
    (drive_id, w_idx) stays unique; the stats are summed."""
    keys = ("n_rows", "n_gap_rows", "n_runs", "n_candidates", "n_dropped_gap", "n_dropped_stationary", "n_windows")
    stats = {k: 0 for k in keys}
    Xs, Ms = [], []
    off = 0
    for _, g in frame.groupby("seg", sort=True):
        if len(g) < T.WINDOW_SAMPLES:
            stats["n_rows"] += int(len(g))
            continue
        r = T.cut_windows(g.reset_index(drop=True), channel_map, t_col="t_s", info=info, align=align)
        for k in keys:
            stats[k] += int(r.stats[k])
        if r.X.shape[0]:
            m = r.meta.copy()
            m["w_idx"] = (m["w_idx"].to_numpy() + off).astype(np.int32)
            off += int(r.X.shape[0])
            Xs.append(r.X); Ms.append(m)
    if not Xs:
        return T.CutResult(np.zeros((0, T.WINDOW_SAMPLES, T.N_CHANNELS), np.float32), pd.DataFrame(columns=[c for c, _ in T.META_COLUMNS]), stats)
    return T.CutResult(np.concatenate(Xs), pd.concat(Ms, ignore_index=True), stats)


def das_series(df: pd.DataFrame, n: int, acc_flag: str, role: str) -> tuple[np.ndarray, str]:
    """das_enabled for vehicle n: per-sample Driver<n> column (ACC -> 1, Human -> 0) where present; else the campaign ACC
    flag (1 -> 1, 0 -> 0); else (flag 2 = mixed or blank) the protocol default: followers 1 (ACC on apart from manual
    braking instances), leaders 0 (human). Returns (array, source tag)."""
    col = f"Driver{n}"
    if col in df.columns:
        s = df[col].astype(str).str.strip().str.lower()
        out = np.where(s == "acc", 1.0, np.where(s == "human", 0.0, np.nan))
        if np.isnan(out).any():
            fb = 1.0 if (role == "follower" and acc_flag != "0") else 0.0
            out = np.where(np.isnan(out), fb, out)
        return out, "Driver column (per sample)"
    if role == "leader":
        # the campaign flag describes the followers' ACC use (Vicolungo ACC=1 files carry Driver1 = Human); a leader
        # without a Driver column (Casale Rexton, JRC low speed BMW / Nexo) is taken as human-driven
        return np.full(len(df), 0.0), "leader without Driver column: assumed human"
    if acc_flag in ("0", "1"):
        return np.full(len(df), float(acc_flag)), f"campaign ACC flag = {acc_flag}"
    return np.full(len(df), 1.0), f"protocol default (ACC flag {acc_flag or 'blank'}, follower on ACC)"


# ----------------------------------------------------------------------------- per-vehicle series -> windows
def vehicle_series(df: pd.DataFrame, n: int, md: dict, role: str, lead_max_m: float, standstill_v: float) -> tuple[pd.DataFrame, dict]:
    """10 Hz frame (rows with a finite speed) of vehicle n with the longitudinal channels, plus QC counters."""
    t = pd.to_numeric(df["Time"], errors="coerce").to_numpy(np.float64)
    v = pd.to_numeric(df[f"Speed{n}"], errors="coerce").to_numpy(np.float64)
    qc = dict(n_rows=int(len(df)), n_v_nan=int(np.isnan(v).sum()))
    bad = np.isfinite(v) & ((v < V_MIN_PLAUSIBLE) | (v > V_MAX_PLAUSIBLE))
    qc["n_v_implausible"] = int(bad.sum())
    v = np.where(bad, np.nan, v)
    qc["n_v_negative_small"] = int((np.isfinite(v) & (v < 0)).sum())
    v, seg, cq = clean_speed(v, t)
    qc.update(cq)
    a = derived_accel(v, seg)
    ss = np.where(np.isfinite(v), (v < standstill_v).astype(float), np.nan)
    das, das_src = das_series(df, n, md["acc"], role)
    qc["das_source"] = das_src
    # lead: vehicle n-1 and the spacing IVS(n-1)
    p = n - 1
    lead_valid = np.zeros(len(df), dtype=bool)
    d = np.full(len(df), np.nan); vr = np.full(len(df), np.nan)
    qc.update(lead_source="none (leader)" if role == "leader" else "none (vehicle ahead not in file)",
              n_lead_rows=0, n_ivs_negative=0, n_ivs_below_min=0, n_ivs_above_max=0, n_ivs_nan_while_moving=0)
    if role == "follower" and f"Speed{p}" in df.columns and f"IVS{p}" in df.columns:
        vp = pd.to_numeric(df[f"Speed{p}"], errors="coerce").to_numpy(np.float64)
        ivs = pd.to_numeric(df[f"IVS{p}"], errors="coerce").to_numpy(np.float64)
        vp = np.where(np.isfinite(vp) & ((vp < V_MIN_PLAUSIBLE) | (vp > V_MAX_PLAUSIBLE)), np.nan, vp)
        fin = np.isfinite(ivs) & np.isfinite(vp) & np.isfinite(v)
        qc["n_ivs_negative"] = int((fin & (ivs < 0)).sum())
        qc["n_ivs_below_min"] = int((fin & (ivs >= 0) & (ivs <= LEAD_MIN_M)).sum())
        qc["n_ivs_above_max"] = int((fin & (ivs > lead_max_m)).sum())
        qc["n_ivs_nan_while_moving"] = int((np.isfinite(v) & ~np.isfinite(ivs)).sum())
        lead_valid = fin & (ivs > LEAD_MIN_M) & (ivs <= lead_max_m)
        d = np.where(lead_valid, ivs, np.nan)
        vr = np.where(lead_valid, vp - v, np.nan)
        qc["lead_source"] = f"IVS{p} + Speed{p}"
        qc["n_lead_rows"] = int(lead_valid.sum())
    lead = T.lead_channels(v, lead_valid, d, vr, t=t)
    frame = pd.DataFrame(dict(t_s=t, seg=seg, v_mps=v, a_mps2=a, standstill=ss, das_enabled=das, **lead))
    keep = np.isfinite(frame["v_mps"].to_numpy()) & np.isfinite(frame["a_mps2"].to_numpy()) & np.isfinite(t) & (seg >= 0)
    qc["n_rows_dropped_short_run"] = int((np.isfinite(v) & ~np.isfinite(a)).sum())
    frame = frame[keep].reset_index(drop=True)
    qc["n_rows_kept"] = int(len(frame))
    if len(frame):
        vv = frame["v_mps"].to_numpy()
        qc.update(v_min=float(vv.min()), v_max=float(vv.max()), moving_frac=float((vv >= T.MOVING_V_MPS).mean()),
                  das_frac=float(np.nanmean(frame["das_enabled"].to_numpy())),
                  lead_frac=float(frame["lead_valid"].to_numpy().mean()))
    else:
        qc.update(v_min=np.nan, v_max=np.nan, moving_frac=np.nan, das_frac=np.nan, lead_frac=np.nan)
    return frame, qc


CHANNEL_MAP_LON = {c: c for c in ("v_mps", "a_mps2", "standstill", "das_enabled") + T.LEAD_CHANNELS}
CHANNEL_MAP_SPEED = {c: c for c in ("v_mps", "a_mps2", "standstill", "das_enabled", "lead_valid")}


def label_info(model: str, specs: pd.DataFrame, veh: Veh, mhev_as_ice: bool) -> dict:
    r = specs.loc[model]
    pt = str(r["powertrain"])
    if veh.exclude == "mhev" and not mhev_as_ice:
        pt = "HEV"
    platform = r["vtt_platform"] if (isinstance(r.get("spec_source"), str) and r["spec_source"] == "VTT table"
                                    and isinstance(r.get("vtt_platform"), str) and r["vtt_platform"]) else None
    return dict(footprint_m2=float(r["footprint_m2"]), size_class=str(r["size_class"]), powertrain=pt,
                borderline=bool(str(r["borderline"]).strip().lower() == "true"), platform=platform,
                wheelbase_mm=r["wheelbase_mm"], generation=str(r["generation_or_years"]))


def build_platoon(raw: Path, campaigns, specs: pd.DataFrame, *, keep_hybrids: bool, drop_mhev: bool, mhev_as_ice: bool,
                  merge_fleet: bool, standstill_v: float, lead_max_m: float, align: str, limit_files: int | None, log=print):
    results: list[T.CutResult] = []
    qc_rows: list[dict] = []
    skipped: dict[str, int] = {}
    label_rows: dict[tuple, dict] = {}
    for camp in campaigns:
        cdir = raw / camp
        if not cdir.is_dir():
            log(f"[{camp}] missing folder {cdir}, skipped")
            continue
        cspec = load_campaign_spec(cdir)
        files = sorted(p for p in cdir.glob("*.csv") if "pecification" not in p.name and not p.name.startswith("Long setting"))
        if limit_files:
            files = files[:limit_files]
        t0 = time.time()
        n_win_c = 0
        for path in files:
            md, df = read_platoon_file(path)
            order = md["vehicle_order"]
            if not order:
                log(f"[{camp}] {path.name}: no Vehicle_order metadata, skipped")
                continue
            n_present = [n for n in range(1, len(order) + 1) if f"Speed{n}" in df.columns]
            for n in range(1, len(order) + 1):
                name = order[n - 1]
                veh = VEHICLES[camp].get(name)
                if veh is None:
                    raise KeyError(f"[{camp}] {path.name}: unknown vehicle name {name!r} in Vehicle_order; add it to VEHICLES")
                key = (camp, name)
                if key not in label_rows:
                    sp = spec_row(cspec, veh.spec_name)
                    label_rows[key] = dict(campaign=camp, header_name=name, model=veh.model, spec_file_name=veh.spec_name or "-",
                                           spec_propulsion=sp.get("propulsion", "-"), spec_power_kw=sp.get("power_kw", "-"),
                                           spec_model_year=sp.get("model_year", "-"), exclude=veh.exclude or "", n_files=0, n_windows=0)
                label_rows[key]["n_files"] += 1
                if f"Speed{n}" not in df.columns:
                    skipped[f"{camp}: vehicle column absent"] = skipped.get(f"{camp}: vehicle column absent", 0) + 1
                    continue
                excl = veh.exclude
                if excl == "hybrid" and not keep_hybrids:
                    skipped[f"excluded hybrid/FCEV: {veh.model}"] = skipped.get(f"excluded hybrid/FCEV: {veh.model}", 0) + 1
                    continue
                if excl == "mhev" and drop_mhev:
                    skipped[f"excluded MHEV: {veh.model}"] = skipped.get(f"excluded MHEV: {veh.model}", 0) + 1
                    continue
                if excl in ("unnamed", "ambiguous") or veh.model is None:
                    skipped[f"excluded {excl}: {name}"] = skipped.get(f"excluded {excl}: {name}", 0) + 1
                    continue
                role = "leader" if n == 1 else "follower"
                frame, qc = vehicle_series(df, n, md, role, lead_max_m, standstill_v)
                lab = label_info(veh.model, specs, veh, mhev_as_ice)
                vehicle_id = veh.fleet_id if (merge_fleet and veh.fleet_id) else f"{camp}/{veh.model}"
                sp = spec_row(cspec, veh.spec_name)
                note = (f"{camp}: {CAMPAIGN_NOTE[camp]}; platoon position {n}/{len(order)} ({role}); "
                        f"das_enabled from {qc['das_source']}; lead from {qc['lead_source']}; a_mps2 = smoothed derivative of GNSS speed; "
                        f"standstill = v < {standstill_v:g} m/s; JRC spec: {sp.get('propulsion', '?')} {sp.get('power_kw', '?')} kW MY{sp.get('model_year', '?')}")
                if veh.exclude == "mhev":
                    note += "; mild hybrid (MHEV): labelled " + ("ICE (lab convention)" if mhev_as_ice else "HEV (JRC spec; outside the BEV/ICE score)")
                if veh.exclude == "hybrid":
                    note += "; hybrid/FCEV kept with --keep-hybrids (outside the BEV/ICE score)"
                if merge_fleet and veh.fleet_id:
                    note += "; vehicle_id merged across campaigns (same JRC fleet car per spec files)"
                info = dict(dataset=DATASET, vehicle_id=vehicle_id, model=veh.model, platform=lab["platform"],
                            drive_id=f"{camp}/{path.stem}/{name}", footprint_m2=lab["footprint_m2"], size_class=lab["size_class"],
                            powertrain=lab["powertrain"], borderline=lab["borderline"], note=note)
                res = cut_frame_segments(frame, CHANNEL_MAP_LON, info, align)
                results.append(res)
                n_win_c += res.stats["n_windows"]
                label_rows[key]["n_windows"] += res.stats["n_windows"]
                qc_rows.append(dict(dataset=DATASET, campaign=camp, file=path.name, position=n, n_in_order=len(order), n_in_file=len(n_present),
                                    role=role, header_name=name, model=veh.model, vehicle_id=vehicle_id, acc_flag=md["acc"], distance_setting=md["distance"],
                                    **qc, **{k: res.stats[k] for k in ("n_runs", "n_candidates", "n_dropped_gap", "n_dropped_stationary", "n_windows")}))
        log(f"[{camp}] {len(files)} files -> {n_win_c} windows ({time.time() - t0:.1f} s)")
    X, meta = T.concat_results(results)
    return X, meta, pd.DataFrame(qc_rows), skipped, pd.DataFrame(list(label_rows.values()))


def build_one_vehicle(raw: Path, specs: pd.DataFrame, *, merge_fleet: bool, standstill_v: float, align: str,
                      limit_files: int | None, log=print):
    cdir = raw / ONE_VEHICLE_DIR
    veh = VEHICLES["OneVehicle"]["Fiat500X"]
    cspec = load_campaign_spec(cdir)
    sp = spec_row(cspec, veh.spec_name)
    lab = label_info(veh.model, specs, veh, mhev_as_ice=False)
    files = sorted(cdir.glob("D*_part*.csv"), key=lambda p: (int(re.match(r"D(\d+)_part(\d+)", p.stem).group(1)),
                                                                int(re.match(r"D(\d+)_part(\d+)", p.stem).group(2))))
    if limit_files:
        files = files[:limit_files]
    vehicle_id = veh.fleet_id if merge_fleet else "OneVehicle/Fiat 500X"
    results: list[T.CutResult] = []
    qc_rows: list[dict] = []
    t0 = time.time()
    for i, path in enumerate(files):
        m = re.match(r"(D\d+)_part(\d+)", path.stem)
        driver = m.group(1)
        df = pd.read_csv(path, usecols=["Time", "Speed"])   # Engine Speed and Gear are never read (they reveal the engine)
        t = pd.to_numeric(df["Time"], errors="coerce").to_numpy(np.float64)
        v = pd.to_numeric(df["Speed"], errors="coerce").to_numpy(np.float64)
        qc = dict(n_rows=int(len(df)), n_v_nan=int(np.isnan(v).sum()))
        bad = np.isfinite(v) & ((v < V_MIN_PLAUSIBLE) | (v > V_MAX_PLAUSIBLE))
        qc["n_v_implausible"] = int(bad.sum())
        v = np.where(bad, np.nan, v)
        v, seg, cq = clean_speed(v, t)
        qc.update(cq)
        a = derived_accel(v, seg)
        ss = np.where(np.isfinite(v), (v < standstill_v).astype(float), np.nan)
        frame = pd.DataFrame(dict(t_s=t, seg=seg, v_mps=v, a_mps2=a, standstill=ss, das_enabled=np.zeros(len(v)), lead_valid=np.zeros(len(v))))
        keep = np.isfinite(v) & np.isfinite(a) & np.isfinite(t) & (seg >= 0)
        frame = frame[keep].reset_index(drop=True)
        qc.update(n_rows_kept=int(len(frame)), v_min=float(frame["v_mps"].min()) if len(frame) else np.nan,
                  v_max=float(frame["v_mps"].max()) if len(frame) else np.nan,
                  moving_frac=float((frame["v_mps"] >= T.MOVING_V_MPS).mean()) if len(frame) else np.nan)
        note = (f"OneVehicle: {CAMPAIGN_NOTE['OneVehicle']}; driver {driver}; speed-only set (engine speed and gear dropped); "
                f"das_enabled = 0 (manual driving, no ACC); lead_valid = 0 (no lead sensor); a_mps2 = smoothed derivative of CAN speed; "
                f"standstill = v < {standstill_v:g} m/s; JRC spec: {sp.get('propulsion', '?')} {sp.get('power_kw', '?')} kW MY{sp.get('model_year', '?')}")
        info = dict(dataset=DATASET_500X, vehicle_id=vehicle_id, model=veh.model, platform=lab["platform"], drive_id=f"OneVehicle/{path.stem}",
                    footprint_m2=lab["footprint_m2"], size_class=lab["size_class"], powertrain=lab["powertrain"], borderline=lab["borderline"], note=note)
        res = cut_frame_segments(frame, CHANNEL_MAP_SPEED, info, align)
        results.append(res)
        qc_rows.append(dict(dataset=DATASET_500X, campaign="OneVehicle", file=path.name, driver=driver, role="single", model=veh.model,
                            vehicle_id=vehicle_id, **qc, **{k: res.stats[k] for k in ("n_runs", "n_candidates", "n_dropped_gap", "n_dropped_stationary", "n_windows")}))
        if (i + 1) % 50 == 0:
            log(f"[OneVehicle] {i + 1}/{len(files)} files, {sum(r.stats['n_windows'] for r in results)} windows ({time.time() - t0:.0f} s)")
    X, meta = T.concat_results(results)
    log(f"[OneVehicle] {len(files)} files -> {len(meta)} windows ({time.time() - t0:.0f} s)")
    return X, meta, pd.DataFrame(qc_rows)


# ----------------------------------------------------------------------------- validation
RANGES = {"v_mps": (-0.5, 60.0), "a_mps2": (-10.0, 10.0), "lead_d_m": (0.0, 300.0), "lead_v_rel_mps": (-40.0, 40.0),
          "lead_speed_mps": (-1.0, 70.0), "lead_a_slope_mps2": (-4.0, 4.0), "thw_s": (0.0, np.inf), "ttc_s": (0.0, np.inf)}
ALWAYS_FINITE = ("v_mps", "a_mps2", "standstill", "das_enabled", "lead_valid")


def validate_set(X: np.ndarray, meta: pd.DataFrame, lead_group_expected: bool) -> tuple[dict, list[str]]:
    """Adapter-level checks beyond common.validate_windows: no NaN inside declared channels, lead channels finite exactly
    while lead_valid = 1 (slope / thw / ttc may carry definitional NaN), flags binary, physical ranges, moving rule,
    label consistency per vehicle. Returns (stats, problems)."""
    probs: list[str] = []
    st: dict = {}
    n = X.shape[0]
    if n == 0:
        return dict(n_windows=0), ["no windows"]
    present = np.isfinite(X).any(axis=(0, 1))
    st["channels_present"] = [c for c, p in zip(T.CHANNELS, present) if p]
    for c in ALWAYS_FINITE:
        k = int(np.isnan(X[:, :, T.CH[c]]).sum())
        if k:
            probs.append(f"{c}: {k} NaN samples inside windows")
    for c in T.BINARY_CHANNELS:
        col = X[:, :, T.CH[c]]
        fin = col[np.isfinite(col)]
        if fin.size and not np.isin(fin, (0.0, 1.0)).all():
            probs.append(f"{c}: non-binary values")
    lv = X[:, :, T.CH["lead_valid"]] == 1
    st["lead_valid_frac"] = float(lv.mean())
    if lead_group_expected:
        for c in ("lead_d_m", "lead_v_rel_mps", "lead_speed_mps"):
            col = X[:, :, T.CH[c]]
            k = int(np.isnan(col[lv]).sum())
            if k:
                probs.append(f"{c}: {k} NaN samples while lead_valid = 1")
            k2 = int(np.isfinite(col[~lv]).sum())
            if k2:
                probs.append(f"{c}: {k2} finite samples while lead_valid = 0")
        sl = X[:, :, T.CH["lead_a_slope_mps2"]]
        st["slope_finite_frac_given_lead"] = float(np.isfinite(sl[lv]).mean()) if lv.any() else float("nan")
        th = X[:, :, T.CH["thw_s"]]
        st["thw_finite_frac_given_lead"] = float(np.isfinite(th[lv]).mean()) if lv.any() else float("nan")
        v = X[:, :, T.CH["v_mps"]]; vr = X[:, :, T.CH["lead_v_rel_mps"]]; ls = X[:, :, T.CH["lead_speed_mps"]]
        m = lv & np.isfinite(ls)
        if m.any():
            err = float(np.nanmax(np.abs(ls[m] - (v[m] + vr[m]))))
            if err > 1e-3:
                probs.append(f"lead_speed_mps != v + v_rel (max |err| {err:.4f})")
    else:
        for c in T.LEAD_CHANNELS[1:]:
            if present[T.CH[c]]:
                probs.append(f"{c}: finite values in a speed-only set")
        if lv.any():
            probs.append("lead_valid = 1 in a speed-only set")
    ranges = {}
    for c, (lo, hi) in RANGES.items():
        col = X[:, :, T.CH[c]]
        fin = col[np.isfinite(col)]
        if fin.size == 0:
            continue
        ranges[c] = dict(min=float(fin.min()), max=float(fin.max()), p001=float(np.quantile(fin, 0.001)), p999=float(np.quantile(fin, 0.999)))
        k = int(((fin < lo) | (fin > hi)).sum())
        if k:
            probs.append(f"{c}: {k} samples outside [{lo}, {hi}] (min {fin.min():.2f}, max {fin.max():.2f})")
    st["ranges"] = ranges
    v = X[:, :, T.CH["v_mps"]]
    dv = np.abs(np.diff(v, axis=1))
    st["n_steps_above_1p5"] = int((dv > STEP_MAX_MPS).sum()); st["n_steps_above_1p0"] = int((dv > 1.0).sum())
    if st["n_steps_above_1p5"]:
        probs.append(f"{st['n_steps_above_1p5']} speed steps > {STEP_MAX_MPS} m/s per sample remain inside windows")
    eq = (dv <= EQ_TOL_MPS) & (v[:, 1:] > HOLD_V_MPS)
    # frozen plateaus of >= HOLD_MAX_SAMPLES equal values inside windows (should be none after clean_speed)
    k = np.zeros(eq.shape[0], dtype=np.int64)
    run = np.zeros(eq.shape[0], dtype=np.int64)
    for j in range(eq.shape[1]):
        run = np.where(eq[:, j], run + 1, 0)
        k = np.maximum(k, run)
    st["n_windows_with_frozen_plateau"] = int((k + 1 >= HOLD_MAX_SAMPLES).sum())
    if st["n_windows_with_frozen_plateau"]:
        probs.append(f"{st['n_windows_with_frozen_plateau']} windows hold a frozen speed plateau >= {HOLD_MAX_SAMPLES} samples")
    mf = (v >= T.MOVING_V_MPS).mean(axis=1)
    st["moving_frac_mean"] = float(mf.mean()); st["moving_frac_min"] = float(mf.min())
    st["standstill_frac"] = float(np.nanmean(X[:, :, T.CH["standstill"]]))
    st["das_frac"] = float(np.nanmean(X[:, :, T.CH["das_enabled"]]))
    st["v_mean_mps"] = float(np.nanmean(v)); st["v_max_mps"] = float(np.nanmax(v))
    if (mf < T.MIN_MOVING_FRAC).any():
        probs.append(f"{int((mf < T.MIN_MOVING_FRAC).sum())} windows below the moving rule")
    for c in ("model", "size_class", "powertrain", "footprint_m2"):
        k = meta.groupby("vehicle_id")[c].nunique(dropna=False)
        if (k > 1).any():
            probs.append(f"{int((k > 1).sum())} vehicles carry more than one {c}")
    dup = meta.duplicated(subset=["drive_id", "w_idx"]).sum()
    if dup:
        probs.append(f"{int(dup)} duplicated (drive_id, w_idx) pairs")
    st["n_windows"] = int(n); st["n_vehicles"] = int(meta["vehicle_id"].nunique()); st["n_drives"] = int(meta["drive_id"].nunique())
    st["n_models"] = int(meta["model"].nunique()); st["hours"] = round(n * T.WINDOW_SAMPLES * T.DT_S / 3600.0, 2)
    return st, probs


def per_model_table(meta: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    v = X[:, :, T.CH["v_mps"]]
    lv = X[:, :, T.CH["lead_valid"]]
    df = meta.assign(_v_mean=np.nanmean(v, axis=1), _v_max=np.nanmax(v, axis=1), _lead=np.nanmean(lv, axis=1))
    g = df.groupby("model", sort=True)
    out = pd.DataFrame({
        "windows": g.size(), "vehicles": g["vehicle_id"].nunique(), "drives": g["drive_id"].nunique(),
        "hours": (g.size() * T.WINDOW_SAMPLES * T.DT_S / 3600.0).round(2),
        "footprint_m2": g["footprint_m2"].first().round(3), "size_class": g["size_class"].first(), "powertrain": g["powertrain"].first(),
        "borderline": g["borderline"].first(), "das_frac": g["das_frac"].mean().round(3), "lead_frac": g["_lead"].mean().round(3),
        "v_mean_mps": g["_v_mean"].mean().round(1), "v_max_mps": g["_v_max"].max().round(1), "moving_frac": g["moving_frac"].mean().round(3),
    }).reset_index()
    return out


def per_campaign_table(meta: pd.DataFrame) -> pd.DataFrame:
    camp = meta["drive_id"].str.split("/").str[0]
    g = meta.assign(_camp=camp).groupby("_camp", sort=True)
    return pd.DataFrame({"windows": g.size(), "vehicles": g["vehicle_id"].nunique(), "models": g["model"].nunique(), "drives": g["drive_id"].nunique(),
                         "hours": (g.size() * T.WINDOW_SAMPLES * T.DT_S / 3600.0).round(2), "das_frac": g["das_frac"].mean().round(3)}).reset_index().rename(columns={"_camp": "campaign"})


def class_tables(meta: pd.DataFrame) -> dict:
    out = {}
    for col in ("size_class", "powertrain"):
        g = meta.groupby(col, dropna=False)
        out[col] = pd.DataFrame({"windows": g.size(), "vehicles": g["vehicle_id"].nunique(), "models": g["model"].nunique()}).reset_index()
    return out


def md_tables(name: str, meta: pd.DataFrame, X: np.ndarray, qc: pd.DataFrame, st: dict, probs: list[str], skipped: dict | None,
              labels: pd.DataFrame | None, args_note: str) -> str:
    L = [f"## {name}: generated tables ({pd.Timestamp.now():%Y-%m-%d %H:%M})", "", f"Build options: {args_note}", "",
         f"windows {st['n_windows']}, vehicles {st['n_vehicles']}, drives {st['n_drives']}, models {st['n_models']}, hours {st['hours']}; "
         f"channels present: {', '.join(st['channels_present'])}", ""]
    pm = per_model_table(meta, X)
    L += ["### Per model", "", T.md_table(list(pm.columns), pm.values.tolist()), ""]
    pc = per_campaign_table(meta)
    L += ["### Per campaign", "", T.md_table(list(pc.columns), pc.values.tolist()), ""]
    for col, tab in class_tables(meta).items():
        L += [f"### Per {col}", "", T.md_table(list(tab.columns), tab.values.tolist()), ""]
    if labels is not None and len(labels):
        L += ["### Vehicle header names, JRC spec file and decision", "", T.md_table(list(labels.columns), labels.values.tolist()), ""]
    if skipped:
        L += ["### Skipped vehicle-file series", "", T.md_table(["reason", "vehicle-files"], [[k, v] for k, v in sorted(skipped.items())]), ""]
    # QC aggregates
    if len(qc):
        agg = dict(vehicle_files=int(len(qc)), rows=int(qc["n_rows"].sum()), v_nan=int(qc["n_v_nan"].sum()), v_implausible=int(qc["n_v_implausible"].sum()),
                   rows_kept=int(qc["n_rows_kept"].sum()), candidates=int(qc["n_candidates"].sum()), dropped_gap=int(qc["n_dropped_gap"].sum()),
                   dropped_stationary=int(qc["n_dropped_stationary"].sum()), windows=int(qc["n_windows"].sum()))
        for k in ("n_v_frozen", "n_step_breaks", "n_hold_before_step", "n_ivs_negative", "n_ivs_below_min", "n_ivs_above_max", "n_ivs_nan_while_moving",
                  "n_lead_rows", "n_v_negative_small", "n_rows_dropped_short_run"):
            if k in qc:
                agg[k[2:] if k.startswith("n_") else k] = int(qc[k].sum())
        L += ["### QC aggregates (vehicle-file series)", "", T.md_table(["quantity", "value"], [[k, v] for k, v in agg.items()]), ""]
        if "das_source" in qc:
            ds = qc.groupby("das_source").agg(vehicle_files=("file", "size"), windows=("n_windows", "sum")).reset_index()
            L += ["### das_enabled source", "", T.md_table(list(ds.columns), ds.values.tolist()), ""]
        neg = qc[qc.get("n_ivs_negative", pd.Series(0, index=qc.index)) > 0] if "n_ivs_negative" in qc else qc.iloc[0:0]
        if len(neg):
            L += ["### Files with negative spacings (set lead_valid = 0 on those samples)", "",
                  T.md_table(["campaign", "file", "vehicle", "n_ivs_negative", "n_lead_rows"], neg[["campaign", "file", "model", "n_ivs_negative", "n_lead_rows"]].values.tolist()), ""]
    rg = st.get("ranges", {})
    if rg:
        L += ["### Physical ranges of the finite samples", "",
              T.md_table(["channel", "min", "p0.1", "p99.9", "max"], [[c, r["min"], r["p001"], r["p999"], r["max"]] for c, r in rg.items()]), ""]
    L += ["### Validation", ""]
    L += [f"* moving_frac mean {st['moving_frac_mean']:.3f} (min {st['moving_frac_min']:.3f}); standstill share {st['standstill_frac']:.3f}; "
          f"das share {st['das_frac']:.3f}; lead_valid share {st['lead_valid_frac']:.3f}; v mean {st['v_mean_mps']:.1f} m/s, max {st['v_max_mps']:.1f} m/s"]
    if "slope_finite_frac_given_lead" in st:
        L += [f"* lead_a_slope finite in {st['slope_finite_frac_given_lead']:.3f} of the lead samples, thw finite in {st['thw_finite_frac_given_lead']:.3f} "
              f"(training: 0.861 / 0.907)"]
    L += ["* problems: " + ("none" if not probs else "; ".join(probs)), ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_DEFAULT, help="OpenACC raw folder (campaign sub-folders)")
    ap.add_argument("--root", type=Path, default=None, help="external root for the outputs (default common.EXTERNAL_ROOT)")
    ap.add_argument("--specs", type=Path, default=SPECS_CSV)
    ap.add_argument("--campaigns", default=",".join(PLATOON_CAMPAIGNS), help="comma list of platoon campaign folders")
    ap.add_argument("--skip-one-vehicle", action="store_true", help="do not build the openacc_500x speed-only set")
    ap.add_argument("--only-one-vehicle", action="store_true", help="build only the openacc_500x set")
    ap.add_argument("--keep-hybrids", action="store_true", help="keep HEV / PHEV / FCEV vehicles (labels outside the BEV/ICE task)")
    ap.add_argument("--drop-mhev", action="store_true", help="drop the two mild hybrids (GLE 450, A4 Avant) instead of labelling them HEV")
    ap.add_argument("--mhev-as-ice", action="store_true", help="label the two mild hybrids ICE (lab convention) instead of HEV")
    ap.add_argument("--no-merge-fleet", action="store_true", help="one vehicle_id per campaign even for the shared JRC fleet cars (S-Max, 500X, Nexo)")
    ap.add_argument("--standstill-v", type=float, default=STANDSTILL_V_MPS, help="standstill = v < this (m/s)")
    ap.add_argument("--lead-max-m", type=float, default=LEAD_MAX_M, help="spacing above this -> lead_valid 0 (radar-range emulation)")
    ap.add_argument("--align", choices=("run", "drive"), default="run")
    ap.add_argument("--dry-run", action="store_true", help="build and validate, do not write")
    ap.add_argument("--limit-files", type=int, default=None, help="debug: first N files per campaign")
    a = ap.parse_args(argv)
    if not a.raw.is_dir():
        ap.error(f"raw folder {a.raw} not found")
    specs = load_specs(a.specs)
    args_note = (f"standstill_v={a.standstill_v:g} m/s, lead_max={a.lead_max_m:g} m, align={a.align}, keep_hybrids={a.keep_hybrids}, "
                 f"drop_mhev={a.drop_mhev}, mhev_as_ice={a.mhev_as_ice}, merge_fleet={not a.no_merge_fleet}")
    report = {}

    def finish(dataset: str, X, meta, qc, st, probs, skipped, labels, lead_expected: bool):
        wd = T.work_dir(dataset, a.root)
        md = md_tables(dataset, meta, X, qc, st, probs, skipped, labels, args_note)
        print(f"[{dataset}] windows={st['n_windows']} vehicles={st['n_vehicles']} drives={st['n_drives']} models={st['n_models']} hours={st['hours']}")
        for p in probs:
            print(f"[{dataset}] PROBLEM: {p}")
        if a.dry_run:
            print(f"[{dataset}] dry run: nothing written")
            return
        wd.mkdir(parents=True, exist_ok=True)
        out = T.save_windows(dataset, X, meta, root=a.root)
        qc.to_csv(wd / "qc_vehicle_files.csv", index=False)
        (wd / "qc_tables.md").write_text(md)
        (wd / "qc_summary.json").write_text(json.dumps(dict(dataset=dataset, options=args_note, stats=st, problems=probs, skipped=skipped or {},
                                                            save_warnings=out["warnings"]), indent=1, default=str))
        report[dataset] = dict(n=out["n"], problems=probs, warnings=out["warnings"])

    if not a.only_one_vehicle:
        camps = [c.strip() for c in a.campaigns.split(",") if c.strip()]
        X, meta, qc, skipped, labels = build_platoon(a.raw, camps, specs, keep_hybrids=a.keep_hybrids, drop_mhev=a.drop_mhev, mhev_as_ice=a.mhev_as_ice,
                                                     merge_fleet=not a.no_merge_fleet, standstill_v=a.standstill_v, lead_max_m=a.lead_max_m,
                                                     align=a.align, limit_files=a.limit_files)
        st, probs = validate_set(X, meta, lead_group_expected=True)
        finish(DATASET, X, meta, qc, st, probs, skipped, labels, True)
        del X
    if not a.skip_one_vehicle:
        X, meta, qc = build_one_vehicle(a.raw, specs, merge_fleet=not a.no_merge_fleet, standstill_v=a.standstill_v, align=a.align, limit_files=a.limit_files)
        st, probs = validate_set(X, meta, lead_group_expected=False)
        finish(DATASET_500X, X, meta, qc, st, probs, None, None, False)
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
