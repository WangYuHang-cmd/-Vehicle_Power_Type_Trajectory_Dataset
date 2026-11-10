#!/usr/bin/env python3
"""Adapter: microSIM-ACC / EV (FAU + UC Berkeley) and the UGA Mobility Lab EV platoon -> VTT transfer windows.

Two GNSS-only car-following sets, both delivered as separate window sets under the common.py contract
(longitudinal-only channels; lateral channels, pedals and driver torque are NaN = missing, mask preset V2):

  microsim_ev  <external>/microsim_ev/raw/unzipped/<car>/.../<gap>/<desired>/<free-flow mph>/<dip mph>/<trial>.csv
               50 Hz grid (interpolated 25 Hz RaceBox GNSS) of follower / leader speed (km/h) and spacing (m); the
               follower is an ACC-engaged EV (Hyundai Ioniq 5 2022, Polestar 2 2023, Tesla Model 3 2022), the human-driven
               lead a Toyota Camry 2021 (Ioniq trials) or a Toyota Corolla 2022 sedan (Polestar / Tesla trials).
  uga_ev       <external>/uga_ev/raw/rawData/{LEADCAR,FLWCAR1,FLWCAR2,FLWCAR3}/*<run>.csv
               10 Hz RaceBox GNSS (UTC time, lat / lon / alt, speed km/h, device-frame g-force) of a 4-car platoon:
               Lexus UX 200 2021 (ICE lead) -> Ford Mustang Mach-E 2024 -> Kia Niro EV 2023 -> Kia Niro EV 2023,
               12 scripted runs (README: 1-8 ACC followers, 9-12 manual; OPD / TPD pedal mode; longest / shortest headway).

Channel map (status in brackets): v_mps = GNSS speed / 3.6 [measured]; a_mps2 = common.smooth_derivative(v) at 10 Hz
[derived]; standstill = v < 0.1 m/s [derived; the rule agrees with carState.standstill on 99.8 % of the training samples];
das_enabled = 1 for ACC-engaged followers by protocol, 0 for manual runs and for the scripted lead cars [protocol];
lead_valid = 1 while a lead gap is available (microSIM: every sample; UGA: the car ahead in the platoon is within range and
the pair's GNSS clocks are consistent), 0 for lead cars; lead_d_m = bumper-to-bumper gap = front-to-front distance minus
the lead car's length [derived]; lead_v_rel_mps = lead speed - ego speed [measured]; lead_speed / thw / ttc / slope by
common.lead_channels [lab rules]. Everything else NaN.

Usage:  source env.sh && $CPY experiments/transfer/ds_microsim_uga.py [--datasets microsim_ev,uga_ev] [--no-length-correction]
                                                                      [--dry-run] [--limit N]
Writes <external>/<dataset>/work/windows.npz + meta.parquet (common.save_windows) and adapter_report.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.transfer import common as T  # noqa: E402
from vtt import config as C  # noqa: E402

KMH = 1.0 / 3.6
STANDSTILL_V_MPS = 0.1          # training: standstill <-> v < 0.1 m/s agree on 99.8 % of samples
LEAD_D_MIN_M = float(C.LEAD_VALID_DREL_MIN_M)   # 0.5 m, the lab's lead_valid distance floor
LEAD_D_MAX_M = 165.0            # largest valid lead distance seen in the training tensors (radar range)
LEAD_LATERAL_MAX_M = 5.0        # UGA: the car ahead counts as the lead only within this lateral offset (lane + GNSS noise)
PASS_GAP_MIN_S = 0.3            # UGA: a front-to-front time gap below this at any road point = inconsistent GNSS clocks
PASS_GAP_P10_MIN_S = 0.5        # UGA: ... or a 10th percentile below this
SPECS_CSV = HERE / "specs_external.csv"

# public overall lengths (m) of the lead cars, used to turn front-to-front GNSS distances into bumper-to-bumper gaps
VEHICLE_LENGTH_M = {
    "Toyota Camry": 4.885,            # XV70 2021 (LE / XLE 4,880-4,905 mm)
    "Toyota Corolla": 4.630,          # E210 sedan 2022 (the 1.8 l, 139 hp, 2,910 lb car of the dissertation is the sedan)
    "Lexus UX 200": 4.495,            # ZA10
    "Ford Mustang Mach-E": 4.713,     # Mk1 (GT 4,743)
    "Kia Niro EV": 4.420,             # SG2 2023
}

# microSIM: folder name -> follower model, its lead car and model year (dissertation ch. 3)
MICROSIM_CARS = {
    "Hyundai Ioniq 5": dict(model="Hyundai Ioniq 5", year=2022, lead="Toyota Camry", lead_year=2021),
    "Polestar 2": dict(model="Polestar 2", year=2023, lead="Toyota Corolla", lead_year=2022),
    "Tesla Model 3": dict(model="Tesla Model 3", year=2022, lead="Toyota Corolla", lead_year=2022),
}

# UGA: README experiment table (run -> pedal mode, control mode, ACC headway, lead speed-adaptation rate)
UGA_EXPERIMENTS = {
    1: ("OPD", "ACC", "Longest", "Low"), 2: ("OPD", "ACC", "Longest", "High"), 3: ("TPD", "ACC", "Longest", "Low"),
    4: ("TPD", "ACC", "Longest", "High"), 5: ("OPD", "ACC", "Shortest", "Low"), 6: ("OPD", "ACC", "Shortest", "High"),
    7: ("TPD", "ACC", "Shortest", "Low"), 8: ("TPD", "ACC", "Shortest", "High"), 9: ("OPD", "Manual", "N/A", "Low"),
    10: ("OPD", "Manual", "N/A", "High"), 11: ("TPD", "Manual", "N/A", "Low"), 12: ("TPD", "Manual", "N/A", "High"),
}
# UGA: folder -> (model, model year, platoon position, predecessor folder); README order lead -> 2nd -> 3rd -> 4th
UGA_ROLES = {
    "LEADCAR": ("Lexus UX 200", 2021, "lead", None),
    "FLWCAR1": ("Ford Mustang Mach-E", 2024, "2nd", "LEADCAR"),
    "FLWCAR2": ("Kia Niro EV", 2023, "3rd", "FLWCAR1"),
    "FLWCAR3": ("Kia Niro EV", 2023, "4th", "FLWCAR2"),
}
UGA_ORDER = ("LEADCAR", "FLWCAR1", "FLWCAR2", "FLWCAR3")


# ----------------------------------------------------------------------------- labels
def load_specs(path: Path = SPECS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["borderline"] = df["borderline"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    return df


def label_for(specs: pd.DataFrame, model: str, dataset_tag: str) -> dict:
    """Row of specs_external.csv for `model` whose `datasets` column names this dataset (one model may have several
    generations) -> footprint / class / powertrain / borderline / platform."""
    rows = specs[(specs["model"].str.strip() == model) & specs["datasets"].fillna("").str.contains(dataset_tag, case=False, regex=False)]
    if len(rows) != 1:
        raise KeyError(f"specs_external.csv: {len(rows)} rows for model {model!r} with dataset tag {dataset_tag!r}")
    r = rows.iloc[0]
    plat = r.get("vtt_platform")
    plat = None if (plat is None or (isinstance(plat, float) and np.isnan(plat)) or str(plat).strip() == "") else str(plat).strip()
    return dict(footprint_m2=float(r["footprint_m2"]), size_class=str(r["size_class"]).upper(), powertrain=T.normalise_powertrain(r["powertrain"]),
                borderline=bool(r["borderline"]), platform=plat, generation=str(r["generation_or_years"]))


# ----------------------------------------------------------------------------- shared helpers
def to_grid(t: np.ndarray, series: dict[str, np.ndarray], kind: str = "linear", max_age_s: float | None = None) -> tuple[np.ndarray, dict]:
    """Every series of one recording onto the common 10 Hz grid of its time span (common.resample_to_10hz)."""
    g = T.make_grid(float(t[0]), float(t[-1]))
    out = {k: T.resample_to_10hz(t, v, kind, t_grid=g, max_age_s=max_age_s)[1] for k, v in series.items()}
    return g, out


def standstill_from_speed(v: np.ndarray) -> np.ndarray:
    s = (np.asarray(v, float) < STANDSTILL_V_MPS).astype(np.float64)
    s[~np.isfinite(v)] = np.nan
    return s


def local_xy(lat, lon, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular local metres around (lat0, lon0): fine for a few km."""
    R = 6371008.8
    return np.radians(np.asarray(lon, float) - lon0) * R * np.cos(np.radians(lat0)), np.radians(np.asarray(lat, float) - lat0) * R


def heading_unit(x: np.ndarray, y: np.ndarray, min_step_m: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Unit heading from position increments on the 10 Hz grid; undefined while (nearly) stopped -> last known heading."""
    hx, hy = np.gradient(x), np.gradient(y)
    nrm = np.hypot(hx, hy)
    ok = np.isfinite(nrm) & (nrm >= min_step_m)
    hx = np.where(ok, hx / np.where(ok, nrm, 1.0), np.nan)
    hy = np.where(ok, hy / np.where(ok, nrm, 1.0), np.nan)
    hx = pd.Series(hx).ffill().bfill().to_numpy()
    hy = pd.Series(hy).ffill().bfill().to_numpy()
    return hx, hy


def arc_length_on_path(px: np.ndarray, py: np.ndarray, qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    """Arc-length coordinate (m) of points q along the polyline p (nearest vertex); p is the lead car's path."""
    s_path = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    try:
        from scipy.spatial import cKDTree
        j = cKDTree(np.c_[px, py]).query(np.c_[qx, qy])[1]
    except Exception:                                     # brute force in chunks
        j = np.empty(len(qx), dtype=np.int64)
        for a in range(0, len(qx), 500):
            b = min(a + 500, len(qx))
            d2 = (qx[a:b, None] - px[None, :]) ** 2 + (qy[a:b, None] - py[None, :]) ** 2
            j[a:b] = np.argmin(d2, axis=1)
    return s_path[j]


def pass_gaps(t_a, s_a, t_b, s_b, step_m: float = 20.0, margin_m: float = 50.0) -> np.ndarray | None:
    """Front-to-front time gaps (s) between car a (ahead) and car b at fixed road points: t_b(s) - t_a(s). A platoon
    cannot show gaps near zero (the car ahead is 4.4-4.9 m long), so small / negative values reveal GNSS clock offsets
    between the two loggers."""
    lo = max(np.nanmin(s_a), np.nanmin(s_b)) + margin_m
    hi = min(np.nanmax(s_a), np.nanmax(s_b)) - margin_m
    grid = np.arange(lo, hi, step_m)
    if len(grid) < 5:
        return None

    def t_at(t, s):
        idx = np.searchsorted(np.maximum.accumulate(np.nan_to_num(s, nan=-1.0)), grid, side="left")
        return t[np.clip(idx, 0, len(t) - 1)]

    return t_at(np.asarray(t_b), np.asarray(s_b)) - t_at(np.asarray(t_a), np.asarray(s_a))


def lon_channel_map(g, v, a, standstill, das, lead: dict | None) -> dict:
    cm = {"v_mps": v, "a_mps2": a, "standstill": standstill, "das_enabled": np.full(len(g), float(das))}
    if lead is None:
        cm["lead_valid"] = np.zeros(len(g))
    else:
        cm.update(lead)
    return cm


# ----------------------------------------------------------------------------- microSIM
_MS_PATH = re.compile(r"^(?P<gap>Short|Medium|Long|XLong)/(?P<des>\d+)_desired/(?P<ff>\d+)/(?P<dip>\d+)/(?P<trial>\d+)\.csv$")


def microsim_files(raw: Path) -> list[tuple[str, dict, Path]]:
    out = []
    for car, info in MICROSIM_CARS.items():
        for f in sorted((raw / car).rglob("*.csv")):
            rel = f.relative_to(raw / car).as_posix()
            parts = rel.split("/")
            tail = "/".join(parts[-5:])                      # Polestar / Tesla have one extra folder level
            m = _MS_PATH.match(tail)
            if not m:
                print(f"[microsim] skip (unexpected path) {rel}")
                continue
            out.append((car, dict(info, **m.groupdict(), rel=rel), f))
    return out


def microsim_drive(car: str, info: dict, f: Path, label: dict, correct_length: bool) -> tuple[T.CutResult, dict]:
    df = pd.read_csv(f)
    need = ["Time", "Speed Follower", "Speed Leader", "Spacing"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{f}: missing columns {miss}")
    t = pd.to_numeric(df["Time"], errors="coerce").to_numpy(float)
    ok = np.isfinite(t)
    t = t[ok]
    if (np.diff(t) <= 0).any():
        raise ValueError(f"{f}: Time not strictly increasing")
    src = {k: pd.to_numeric(df[k], errors="coerce").to_numpy(float)[ok] for k in need[1:]}
    g, s = to_grid(t, src, kind="linear", max_age_s=0.3)
    v = s["Speed Follower"] * KMH
    v_lead = s["Speed Leader"] * KMH
    dist = s["Spacing"]
    L = VEHICLE_LENGTH_M[info["lead"]] if correct_length else 0.0
    gap = dist - L
    valid = np.isfinite(gap) & np.isfinite(v_lead) & np.isfinite(v) & (gap > LEAD_D_MIN_M) & (gap <= LEAD_D_MAX_M)
    lead = T.lead_channels(v, valid, gap, v_lead - v, t=g)
    a = T.smooth_derivative(v)
    frame = pd.DataFrame({"t_s": g})
    cm = lon_channel_map(g, v, a, standstill_from_speed(v), 1.0, lead)
    note = (f"microSIM trial {info['rel']}: ACC gap setting {info['gap']}, set speed +{info['des']} mph, free-flow {info['ff']} mph, "
            f"dip to {info['dip']} mph; lead {info['lead']} {info['lead_year']} (human); v = RaceBox GNSS speed (25 Hz -> 50 Hz by the authors), "
            f"a = 0.5-s smoothed derivative; standstill = v < 0.1 m/s; das_enabled = 1 by protocol; "
            f"lead_d = Spacing{' - lead length %.2f m' % L if correct_length else ' (front-to-front, uncorrected)'}; lateral channels, pedals, torque missing")
    res = T.cut_windows(frame, cm, t_col="t_s", align="run",
                        info=dict(dataset="microsim_ev", vehicle_id=f"microSIM {info['model']} {info['year']}", model=info["model"],
                                  platform=label["platform"], drive_id=f"{car}/{info['rel'][:-4]}", footprint_m2=label["footprint_m2"],
                                  size_class=label["size_class"], powertrain=label["powertrain"], borderline=label["borderline"], note=note))
    qc = dict(file=info["rel"], car=car, n_raw_rows=int(len(t)), dur_s=float(t[-1] - t[0]), n10=int(len(g)), v_max=float(np.nanmax(v)),
              gap_min=float(np.nanmin(gap)), gap_max=float(np.nanmax(gap)), lead_valid_frac=float(np.mean(valid)), **res.stats)
    return res, qc


def build_microsim(raw: Path, specs: pd.DataFrame, correct_length: bool, limit: int | None) -> tuple[np.ndarray, pd.DataFrame, dict]:
    files = microsim_files(raw)
    if limit:
        files = files[:limit]
    labels = {car: label_for(specs, info["model"], "microSIM") for car, info in MICROSIM_CARS.items()}
    results, qcs = [], []
    for car, info, f in files:
        res, qc = microsim_drive(car, info, f, labels[car], correct_length)
        results.append(res)
        qcs.append(qc)
    X, meta = T.concat_results(results)
    Q = pd.DataFrame(qcs)
    per_car = Q.groupby("car").agg(trials=("file", "size"), trials_ge_60s=("dur_s", lambda d: int((d >= 60).sum())), windows=("n_windows", "sum"),
                                  dur_med_s=("dur_s", "median"), dur_max_s=("dur_s", "max"), v_max=("v_max", "max"),
                                  gap_min=("gap_min", "min"), gap_max=("gap_max", "max"), lead_valid_frac=("lead_valid_frac", "mean"),
                                  dropped_gap=("n_dropped_gap", "sum"), dropped_stationary=("n_dropped_stationary", "sum"))
    report = dict(n_files=len(files), per_car=per_car.reset_index().to_dict(orient="records"), labels=labels,
                  length_correction=correct_length, lead_lengths_m={k: VEHICLE_LENGTH_M[v["lead"]] for k, v in MICROSIM_CARS.items()})
    return X, meta, report


# ----------------------------------------------------------------------------- UGA
def uga_file(raw: Path, folder: str, run: int) -> Path:
    name = f"leadingcar{run}.csv" if folder == "LEADCAR" else f"followingcar_{folder[-1]}_{run}.csv"
    return raw / folder / name


def uga_load(f: Path, day0: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.Timestamp]:
    df = pd.read_csv(f)
    need = ["Time", "Latitude", "Longitude", "Speed"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{f}: missing columns {miss}")
    ts = pd.to_datetime(df["Time"], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{f}: unparsable timestamps")
    if day0 is None:
        day0 = ts.iloc[0].normalize()
    t = (ts - day0).dt.total_seconds().to_numpy(float)
    if (np.diff(t) <= 0).any():
        raise ValueError(f"{f}: Time not strictly increasing")
    out = pd.DataFrame({"t": t, "lat": pd.to_numeric(df["Latitude"], errors="coerce"), "lon": pd.to_numeric(df["Longitude"], errors="coerce"),
                        "v": pd.to_numeric(df["Speed"], errors="coerce") * KMH})
    return out, day0


def build_uga(raw: Path, specs: pd.DataFrame, correct_length: bool, limit: int | None) -> tuple[np.ndarray, pd.DataFrame, dict]:
    labels = {m: label_for(specs, m, "UGA") for m in {r[0] for r in UGA_ROLES.values()}}
    runs = sorted({int(re.search(r"(\d+)\.csv$", f.name).group(1)) for f in (raw / "LEADCAR").glob("leadingcar*.csv")})
    if limit:
        runs = runs[:limit]
    results, qcs, gates = [], [], []
    for run in runs:
        exp = UGA_EXPERIMENTS.get(run)
        raw_d, day0 = {}, None
        for folder in UGA_ORDER:
            raw_d[folder], day0 = uga_load(uga_file(raw, folder, run), day0)
        lat0, lon0 = float(raw_d["LEADCAR"]["lat"].iloc[0]), float(raw_d["LEADCAR"]["lon"].iloc[0])
        # every car on its own 10 Hz grid, positions in local metres
        G = {}
        for folder in UGA_ORDER:
            d = raw_d[folder]
            g, s = to_grid(d["t"].to_numpy(), {"lat": d["lat"].to_numpy(), "lon": d["lon"].to_numpy(), "v": d["v"].to_numpy()}, "linear", max_age_s=0.3)
            x, y = local_xy(s["lat"], s["lon"], lat0, lon0)
            G[folder] = dict(g=g, x=x, y=y, v=s["v"], t_raw=d["t"].to_numpy())
        # arc-length coordinate of every car along the lead's path (for the clock-consistency gate)
        px, py = G["LEADCAR"]["x"], G["LEADCAR"]["y"]
        for folder in UGA_ORDER:
            G[folder]["s"] = arc_length_on_path(px, py, G[folder]["x"], G[folder]["y"])
        for folder in UGA_ORDER:
            model, year, pos, pred = UGA_ROLES[folder]
            lab = labels[model]
            g, x, y, v = G[folder]["g"], G[folder]["x"], G[folder]["y"], G[folder]["v"]
            a = T.smooth_derivative(v)
            das = 1.0 if (pred is not None and exp is not None and exp[1] == "ACC") else 0.0
            lead, gate = None, None
            if pred is not None:
                P = G[pred]
                # predecessor position / speed on this car's grid (no bridging beyond 0.3 s)
                _, ps = T.resample_to_10hz(P["g"], P["x"], "linear", t_grid=g, max_age_s=0.3)
                _, py_ = T.resample_to_10hz(P["g"], P["y"], "linear", t_grid=g, max_age_s=0.3)
                _, pv = T.resample_to_10hz(P["g"], P["v"], "linear", t_grid=g, max_age_s=0.3)
                hx, hy = heading_unit(x, y)
                dx, dy = ps - x, py_ - y
                along = dx * hx + dy * hy                                     # front-to-front distance along the heading
                lateral = np.abs(-dx * hy + dy * hx)
                L = VEHICLE_LENGTH_M[UGA_ROLES[pred][0]] if correct_length else 0.0
                gap = along - L
                valid = np.isfinite(gap) & np.isfinite(pv) & np.isfinite(v) & (gap > LEAD_D_MIN_M) & (gap <= LEAD_D_MAX_M) & (lateral < LEAD_LATERAL_MAX_M)
                pg = pass_gaps(P["g"], P["s"], g, G[folder]["s"])
                gate = dict(run=run, car=folder, pred=pred, pass_gap_min_s=float(np.min(pg)) if pg is not None else np.nan,
                            pass_gap_p10_s=float(np.percentile(pg, 10)) if pg is not None else np.nan,
                            pass_gap_med_s=float(np.median(pg)) if pg is not None else np.nan, n_points=int(len(pg)) if pg is not None else 0)
                clocks_ok = pg is not None and gate["pass_gap_min_s"] >= PASS_GAP_MIN_S and gate["pass_gap_p10_s"] >= PASS_GAP_P10_MIN_S
                gate["lead_kept"] = bool(clocks_ok)
                gate["lead_valid_frac"] = float(np.mean(valid)) if clocks_ok else 0.0
                gate["gap_med_m"] = float(np.nanmedian(np.where(valid, gap, np.nan))) if valid.any() else np.nan
                gates.append(gate)
                if clocks_ok:
                    lead = T.lead_channels(v, valid, gap, pv - v, t=g)
                lead_note = (f"lead = {UGA_ROLES[pred][0]} ahead, from paired GNSS positions projected on the heading"
                             f"{' minus its length %.2f m' % L if correct_length else ' (front-to-front, uncorrected)'}"
                             if clocks_ok else
                             f"lead group dropped (lead_valid = 0): GNSS clocks of this car and the car ahead inconsistent "
                             f"(front-to-front time gap min {gate['pass_gap_min_s']:.1f} s / p10 {gate['pass_gap_p10_s']:.1f} s)")
            else:
                lead_note = "platoon lead car: no vehicle ahead (lead_valid = 0); scripted speed profile, assumed human-driven (das_enabled = 0)"
            exp_s = f"experiment {run}: pedal {exp[0]}, control {exp[1]}, ACC headway {exp[2]}, lead adaptation rate {exp[3]}" if exp else f"run {run}: not in the README table"
            note = (f"UGA EV platoon {exp_s}; position {pos}; v = RaceBox 10 Hz GNSS speed, a = 0.5-s smoothed derivative; standstill = v < 0.1 m/s; "
                    f"das_enabled = {int(das)} ({'ACC by protocol' if das else 'manual / lead'}); {lead_note}; lateral channels, pedals, torque missing")
            frame = pd.DataFrame({"t_s": g})
            cm = lon_channel_map(g, v, a, standstill_from_speed(v), das, lead)
            vid = f"UGA {model} {year} (platoon {pos})"
            res = T.cut_windows(frame, cm, t_col="t_s", align="run",
                                info=dict(dataset="uga_ev", vehicle_id=vid, model=model, platform=lab["platform"], drive_id=f"UGA run{run:02d} {folder}",
                                          footprint_m2=lab["footprint_m2"], size_class=lab["size_class"], powertrain=lab["powertrain"],
                                          borderline=lab["borderline"], note=note))
            results.append(res)
            qcs.append(dict(run=run, car=folder, model=model, n_raw_rows=int(len(G[folder]["t_raw"])), dur_s=float(g[-1] - g[0]), v_max=float(np.nanmax(v)),
                            das=das, lead_kept=(gate or {}).get("lead_kept"), **res.stats))
    X, meta = T.concat_results(results)
    Q = pd.DataFrame(qcs)
    per_car = Q.groupby(["car", "model"]).agg(files=("run", "size"), windows=("n_windows", "sum"), hours=("dur_s", lambda d: float(d.sum() / 3600)),
                                              v_max=("v_max", "max"), dropped_gap=("n_dropped_gap", "sum"), dropped_stationary=("n_dropped_stationary", "sum"))
    report = dict(runs=runs, per_car=per_car.reset_index().to_dict(orient="records"), clock_gates=gates, labels=labels,
                  length_correction=correct_length, pass_gap_thresholds_s=dict(min=PASS_GAP_MIN_S, p10=PASS_GAP_P10_MIN_S))
    return X, meta, report


# ----------------------------------------------------------------------------- validation
def validate_report(X: np.ndarray, meta: pd.DataFrame, name: str) -> dict:
    """Counts, ranges and consistency checks of a finished window set; raises on a hard violation."""
    C = T.CH
    n = X.shape[0]
    out = dict(dataset=name, n_windows=int(n), n_vehicles=int(meta["vehicle_id"].nunique()), n_drives=int(meta["drive_id"].nunique()),
               hours=round(n * 60 / 3600, 2))
    if n == 0:
        return out
    g = meta.groupby("model", sort=True)
    out["per_model"] = [dict(model=m, windows=int(len(d)), vehicles=int(d["vehicle_id"].nunique()), drives=int(d["drive_id"].nunique()),
                             size_class=d["size_class"].iloc[0], powertrain=d["powertrain"].iloc[0], footprint_m2=float(d["footprint_m2"].iloc[0]),
                             borderline=bool(d["borderline"].iloc[0]), das_frac=float(d["das_frac"].mean()), moving_frac=float(d["moving_frac"].mean()))
                        for m, d in g]
    out["per_vehicle"] = meta.groupby("vehicle_id").size().to_dict()
    out["per_size_class"] = meta["size_class"].fillna("unknown").value_counts().to_dict()
    out["per_powertrain"] = meta["powertrain"].value_counts().to_dict()
    v = X[:, :, C["v_mps"]]
    a = X[:, :, C["a_mps2"]]
    out["v_mps"] = dict(min=float(np.nanmin(v)), max=float(np.nanmax(v)), mean=float(np.nanmean(v)), nan_frac=float(np.isnan(v).mean()))
    out["a_mps2"] = dict(min=float(np.nanmin(a)), max=float(np.nanmax(a)), std=float(np.nanstd(a)), nan_frac=float(np.isnan(a).mean()),
                         std_diff=float(np.nanstd(np.diff(a, axis=1))))
    out["moving_frac"] = dict(mean=float(meta["moving_frac"].mean()), min=float(meta["moving_frac"].min()))
    out["standstill_frac"] = float(np.nanmean(X[:, :, C["standstill"]]))
    out["das_frac"] = float(np.nanmean(X[:, :, C["das_enabled"]]))
    lv = X[:, :, C["lead_valid"]]
    out["lead_valid_frac"] = float(np.nanmean(lv))
    out["windows_with_lead"] = int((np.nanmean(lv, axis=1) > 0).sum())
    out["nan_frac_per_channel"] = {c: float(np.isnan(X[:, :, i]).mean()) for c, i in C.items()}
    probs = []
    # hard checks
    for c in ("v_mps", "a_mps2", "standstill", "das_enabled", "lead_valid"):
        if np.isnan(X[:, :, C[c]]).any():
            probs.append(f"{c} has NaN samples")
    for c in T.BINARY_CHANNELS:
        col = X[:, :, C[c]]
        fin = col[np.isfinite(col)]
        if fin.size and not np.isin(fin, (0.0, 1.0)).all():
            probs.append(f"{c} not 0/1")
    for c in ("lead_d_m", "lead_v_rel_mps", "lead_speed_mps"):
        col = X[:, :, C[c]]
        if np.isnan(col[lv == 1]).any():
            probs.append(f"{c} NaN while lead_valid = 1")
        if np.isfinite(col[lv == 0]).any():
            probs.append(f"{c} finite while lead_valid = 0")
    for c in T.LATERAL_CHANNELS + ("gas_pressed", "brake_pressed"):
        if np.isfinite(X[:, :, C[c]]).any():
            probs.append(f"{c} should be all NaN (missing) in a GNSS-only set")
    d = X[:, :, C["lead_d_m"]]
    vr = X[:, :, C["lead_v_rel_mps"]]
    rng = {"v_mps": (v, -0.01, 45.0), "a_mps2": (a, -8.0, 8.0), "lead_d_m": (d, 0.5, LEAD_D_MAX_M + 1e-6), "lead_v_rel_mps": (vr, -25.0, 25.0),
           "thw_s": (X[:, :, C["thw_s"]], 0.0, np.inf), "ttc_s": (X[:, :, C["ttc_s"]], 0.0, np.inf), "lead_a_slope_mps2": (X[:, :, C["lead_a_slope_mps2"]], -4.0, 4.0)}
    out["ranges"] = {}
    for c, (col, lo, hi) in rng.items():
        fin = col[np.isfinite(col)]
        if fin.size:
            out["ranges"][c] = dict(min=float(fin.min()), p01=float(np.percentile(fin, 1)), p99=float(np.percentile(fin, 99)), max=float(fin.max()))
            if fin.min() < lo or fin.max() > hi:
                probs.append(f"{c} outside [{lo}, {hi}]: min {fin.min():.3f} max {fin.max():.3f}")
    if (meta["moving_frac"] < T.MIN_MOVING_FRAC).any():
        probs.append("windows below the moving rule")
    out["problems"] = probs
    if probs:
        raise RuntimeError(f"{name}: validation problems: {probs}")
    return out


def round_floats(obj, nd: int = 4):
    """Round every float in a nested report (keeps the JSON free of long digit runs that look like hex ids to scanners)."""
    if isinstance(obj, float):
        return round(obj, nd) if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, nd) for v in obj]
    if isinstance(obj, np.floating):
        return round(float(obj), nd) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def print_report(rep: dict) -> None:
    print(f"\n== {rep['dataset']}: {rep['n_windows']} windows, {rep['n_vehicles']} vehicles, {rep['n_drives']} drives, {rep['hours']} h")
    if rep.get("per_model"):
        print(T.md_table(["model", "windows", "vehicles", "drives", "size", "powertrain", "footprint", "borderline", "das_frac", "moving_frac"],
                         [[m["model"], m["windows"], m["vehicles"], m["drives"], m["size_class"], m["powertrain"], m["footprint_m2"], m["borderline"],
                           round(m["das_frac"], 3), round(m["moving_frac"], 3)] for m in rep["per_model"]]))
        print("per vehicle:", rep["per_vehicle"])
        print("speed m/s:", {k: round(x, 3) for k, x in rep["v_mps"].items()}, "| accel:", {k: round(x, 3) for k, x in rep["a_mps2"].items()})
        print("moving_frac:", rep["moving_frac"], "standstill_frac %.4f das_frac %.3f lead_valid_frac %.3f windows_with_lead %d" %
              (rep["standstill_frac"], rep["das_frac"], rep["lead_valid_frac"], rep["windows_with_lead"]))
        print("NaN frac per channel:", {k: round(x, 3) for k, x in rep["nan_frac_per_channel"].items()})
        print("ranges:", {k: {kk: round(vv, 2) for kk, vv in d.items()} for k, d in rep["ranges"].items()})


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="microsim_ev,uga_ev", help="comma list among microsim_ev, uga_ev")
    ap.add_argument("--root", type=Path, default=T.EXTERNAL_ROOT, help="external root holding <dataset>/raw and <dataset>/work")
    ap.add_argument("--microsim-raw", type=Path, default=None, help="default <root>/microsim_ev/raw/unzipped")
    ap.add_argument("--uga-raw", type=Path, default=None, help="default <root>/uga_ev/raw/rawData")
    ap.add_argument("--no-length-correction", action="store_true", help="keep front-to-front GNSS distances as lead_d_m (default subtracts the lead car's length)")
    ap.add_argument("--dry-run", action="store_true", help="build and validate, write nothing")
    ap.add_argument("--limit", type=int, default=None, help="first N trials (microSIM) / runs (UGA) for a smoke test")
    args = ap.parse_args(argv)
    specs = load_specs()
    correct = not args.no_length_correction
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        t0 = time.time()
        if ds == "microsim_ev":
            raw = args.microsim_raw or (args.root / "microsim_ev" / "raw" / "unzipped")
            X, meta, rep = build_microsim(raw, specs, correct, args.limit)
        elif ds == "uga_ev":
            raw = args.uga_raw or (args.root / "uga_ev" / "raw" / "rawData")
            X, meta, rep = build_uga(raw, specs, correct, args.limit)
        else:
            raise SystemExit(f"unknown dataset {ds}")
        X, meta, warns = T.validate_windows(X, meta, strict=True)
        val = validate_report(X, meta, ds)
        print_report(val)
        for w in warns:
            print(f"[{ds}] validate_windows warning: {w}")
        if ds == "uga_ev":
            print(T.md_table(["run", "car", "ahead", "pass-gap min s", "p10 s", "median s", "lead kept", "lead_valid frac", "gap median m"],
                             [[g["run"], g["car"], g["pred"], g["pass_gap_min_s"], g["pass_gap_p10_s"], g["pass_gap_med_s"], g["lead_kept"],
                               round(g["lead_valid_frac"], 3), g["gap_med_m"]] for g in rep["clock_gates"]]))
        print(f"[{ds}] built in {time.time() - t0:.1f} s")
        if args.dry_run:
            continue
        info = T.save_windows(ds, X, meta, root=args.root)
        wd = T.work_dir(ds, args.root)
        (wd / "adapter_report.json").write_text(json.dumps(round_floats(dict(build=rep, validation=val, save_warnings=info["warnings"],
                                                                              adapter="ds_microsim_uga.py", written=time.strftime("%Y-%m-%d %H:%M:%S"))),
                                                           indent=1, default=str))
        print(f"[{ds}] -> {info['npz']} ({info['n']} windows); report {wd / 'adapter_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
