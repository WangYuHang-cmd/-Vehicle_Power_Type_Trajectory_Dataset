#!/usr/bin/env python3
"""Adapter "tesla_yuan": Yuan et al. 2024 Tesla Model 3 real-world trips -> VTT transfer windows (speed-only, mask V3).

Source: figshare article 23904102 v2, file 50103315 `DATA S1 update20241031.zip` (CC BY 4.0), extracted under
<external>/tesla_yuan/raw/extracted/DATA S1/. One car (2019 Tesla Model 3 Standard Range Plus RWD, BEV), 106 real-world
trips `data/real_world_data/001.xlsx .. 106.xlsx` (Changchun, April-May 2023), sheet `Continuous20Hz`: `PhaseTime` (s,
0.05-s steps), `DAActualSpeed` (km/h, the vehicle speed from the CAN bus), GPS `lat` / `long` (about 1 Hz), `elevation`,
`gear`, battery current / voltage and energy counters (never read here: they reveal the powertrain). The 46 chassis-
dynamometer cycles under `data/lab_data_m1/` are not road data and are ignored.

What this adapter does (every step is documented in DS_TESLA_YUAN.md):
  1. reads each trip workbook (openpyxl), keeps PhaseTime / speed / GPS / elevation / gear, checks the 20 Hz cadence and
     caches the frame as <work>/cache_20hz/<trip>.parquet (+ a stats json);
  2. verifies the time base and the speed unit against the GPS (1-Hz fix cadence in PhaseTime, GPS path length vs the
     integral of the speed) -- the wall-clock `Time` column is a logger artefact and is not used;
  3. holds the 20 Hz speed onto the 10 Hz grid (common.resample_to_10hz 'decimate': samples, not bin means), derives
     a_mps2 with common.smooth_derivative per contiguous run, sets standstill = v < 0.1 m/s and das_enabled = 0 (Autopilot
     use unknown -> assumed off), leaves the other 16 channels NaN;
  4. cuts 60-s windows with common.cut_windows (lab rule) per trip, saves them with common.save_windows, then validates
     the saved file (counts, ranges, NaN pattern, flags, labels) -> <work>/validation.json and <work>/trips.csv.

Run (any cwd; `-I` keeps the interpreter from importing anything next to the data):
  source env.sh && nice -n 10 $CPY -I experiments/transfer/ds_tesla_yuan.py [--limit 3] [--workers 4] [--align run|drive]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common as T  # noqa: E402

DATASET = "tesla_yuan"
MODEL = "Tesla Model 3"                      # public model name, as in specs_external.csv and the lab table
PLATFORM = "TESLA_MODEL_3"                   # lab platform key (2017-2023 pre-facelift; the lab's three Model 3 spellings share these specs)
VEHICLE_ID = "yuan2024_model3"               # the single physical car of the study (2019 SR+ RWD)
POWERTRAIN = "BEV"
GENERATION = "2019 Model 3 Standard Range Plus RWD (pre-facelift)"
FALLBACK_SPEC = dict(wheelbase_mm=2875.0, track_front_mm=1590.0, track_rear_mm=1590.0, footprint_m2=4.571)
SHEET = "Continuous20Hz"
REQUIRED_COLS = ("PhaseTime", "DAActualSpeed", "lat", "long")
KEEP_COLS = {"PhaseTime": "t_s", "DAActualSpeed": "v_kmh", "lat": "lat", "long": "lon", "elevation": "elev_m", "gear": "gear"}
SRC_DT_S = 0.05                              # nominal 20 Hz
TRIP_RE = re.compile(r"^(\d{3})\.xlsx$")
KMH_TO_MPS = 1.0 / 3.6
STANDSTILL_MPS = 0.1                         # standstill = speed below 0.1 m/s (training tensors: P(standstill | v < 0.1) = 0.98, P(v < 0.1 | standstill) = 0.999)
ACCEL_WIN_S = 0.5                            # moving-average window of common.smooth_derivative (the contract default)
PRESENT = ("v_mps", "a_mps2", "standstill", "das_enabled")
NOTE = ("speed-only source: 20 Hz CAN speed (km/h) held to 10 Hz; a_mps2 = smoothed derivative of v; standstill = v < 0.1 m/s; "
        "das_enabled unknown -> 0; steering / yaw / pedals / lead absent (NaN)")


# ----------------------------------------------------------------------------- spec lookup
def load_spec(specs_csv: Path) -> dict:
    """Footprint of the Model 3 from specs_external.csv (row `model == MODEL`), else the lab values hard-coded above."""
    spec = dict(FALLBACK_SPEC, source="fallback constants (lab table TESLA_MODEL_3: 2875 x 1590 mm)")
    if specs_csv.exists():
        df = pd.read_csv(specs_csv)
        row = df[df["model"].astype(str).str.strip() == MODEL]
        if len(row):
            r = row.iloc[0]
            spec = dict(wheelbase_mm=float(r["wheelbase_mm"]), track_front_mm=float(r["track_front_mm"]), track_rear_mm=float(r["track_rear_mm"]),
                        footprint_m2=float(r["footprint_m2"]), source=f"{specs_csv.name}: {r['source_url']}")
            if str(r["powertrain"]).strip().upper() != POWERTRAIN:
                raise RuntimeError(f"specs_external.csv powertrain {r['powertrain']!r} != {POWERTRAIN}")
    fp_calc = spec["wheelbase_mm"] / 1000.0 * (spec["track_front_mm"] + spec["track_rear_mm"]) / 2000.0
    if abs(fp_calc - spec["footprint_m2"]) > 0.002:
        raise RuntimeError(f"footprint {spec['footprint_m2']} inconsistent with wheelbase x mean track = {fp_calc:.3f}")
    spec["size_class"] = T.classify_footprint(spec["footprint_m2"])
    spec["borderline"] = bool(T.is_borderline(spec["footprint_m2"]))
    return spec


# ----------------------------------------------------------------------------- raw reading (one worker per trip)
def find_trips(raw: Path) -> list[Path]:
    trips = sorted(p for p in raw.iterdir() if TRIP_RE.match(p.name))
    if not trips:
        raise FileNotFoundError(f"no NNN.xlsx trip files in {raw}")
    return trips


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def gps_stats(t: np.ndarray, v_mps: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> dict:
    """Time-base and unit evidence from the GPS: cadence of fix changes in PhaseTime while moving (a 1-Hz receiver must
    give 1.000 s if PhaseTime is real time), GPS path length vs the integral of the CAN speed (ratio ~1 iff km/h), and the
    regression of GPS ground speed on the CAN speed over the 1-s fix intervals."""
    out = dict(gps_n_fix=0, gps_cadence_s=np.nan, gps_path_km=np.nan, int_v_km=np.nan, gps_over_int=np.nan, gps_slope=np.nan, gps_icpt=np.nan, gps_n_reg=0)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if ok.sum() < 3:
        return out
    ch = np.flatnonzero((np.diff(lat) != 0) | (np.diff(lon) != 0)) + 1          # first row of each new fix
    out["gps_n_fix"] = int(len(ch))
    out["int_v_km"] = float(np.trapezoid(v_mps, t) / 1000.0)
    if len(ch) < 3:
        return out
    seg_d = _haversine_m(lat[ch[:-1]], lon[ch[:-1]], lat[ch[1:]], lon[ch[1:]])
    seg_t = np.diff(t[ch])
    mv = (v_mps[ch[1:]] > 1.4) & (v_mps[ch[:-1]] > 1.4)                            # both ends above 5 km/h
    out["gps_path_km"] = float(seg_d.sum() / 1000.0)
    out["gps_over_int"] = float(out["gps_path_km"] / out["int_v_km"]) if out["int_v_km"] > 0 else np.nan
    if mv.sum() >= 5:
        out["gps_cadence_s"] = float(np.median(seg_t[mv]))
        sel = mv & (seg_t > 0.5) & (seg_t < 1.5)
        if sel.sum() >= 10:
            cs = np.concatenate([[0.0], np.cumsum(v_mps)])
            a, b = ch[:-1][sel], ch[1:][sel]
            can_v = (cs[b] - cs[a]) / (b - a)
            gps_v = seg_d[sel] / seg_t[sel]
            slope, icpt = np.linalg.lstsq(np.vstack([can_v, np.ones_like(can_v)]).T, gps_v, rcond=None)[0]
            out.update(gps_slope=float(slope), gps_icpt=float(icpt), gps_n_reg=int(sel.sum()))
    return out


def read_trip(path: Path, cache_dir: Path, use_cache: bool = True) -> dict:
    """Read one trip workbook -> cache parquet (t_s, v_kmh, lat, lon, elev_m, gear) + stats json; returns the stats."""
    trip = TRIP_RE.match(path.name).group(1)
    pq, js = cache_dir / f"{trip}.parquet", cache_dir / f"{trip}.json"
    if use_cache and pq.exists() and js.exists():
        st = json.loads(js.read_text())
        if st.get("src_size") == path.stat().st_size:
            st["from_cache"] = True
            return st
    t0 = time.time()
    df = pd.read_excel(path, sheet_name=SHEET, engine="openpyxl")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: columns {missing} missing; has {list(df.columns)}")
    keep = {src: dst for src, dst in KEEP_COLS.items() if src in df.columns}
    out = df[list(keep)].rename(columns=keep).copy()
    for c in ("t_s", "v_kmh", "lat", "lon"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float64)
    st: dict = dict(trip=trip, file=path.name, src_size=path.stat().st_size, n_rows_20hz=int(len(out)), columns=list(df.columns),
                    n_nan_t=int(out["t_s"].isna().sum()), n_nan_v=int(out["v_kmh"].isna().sum()), read_s=round(time.time() - t0, 1))
    # wall clock column: documented only (increments ~0.01 s per 4 rows with one jump of ~1,650 s; useless as a time base)
    if "Time" in df.columns:
        tw = pd.to_datetime(df["Time"].astype(str), format="ISO8601", errors="coerce")
        st["wall_span_s"] = float((tw.max() - tw.min()).total_seconds()) if tw.notna().any() else np.nan
    out = out.dropna(subset=["t_s", "v_kmh"]).sort_values("t_s", kind="stable").reset_index(drop=True)
    t = out["t_s"].to_numpy(); v = out["v_kmh"].to_numpy() * KMH_TO_MPS
    dt = np.diff(t)
    st.update(duration_s=float(t[-1] - t[0]) if len(t) > 1 else 0.0, dt_median=float(np.median(dt)) if len(dt) else np.nan,
              n_dt_irregular=int((np.abs(dt - SRC_DT_S) > 0.005).sum()), n_src_gaps=int((dt > T.GAP_S).sum()), n_dt_nonpos=int((dt <= 0).sum()),
              t_start=float(t[0]), v_max_kmh=float(out["v_kmh"].max()), v_min_kmh=float(out["v_kmh"].min()), v_mean_kmh=float(out["v_kmh"].mean()),
              frac_v0=float(np.mean(v == 0)), frac_moving_20hz=float(np.mean(v >= T.MOVING_V_MPS)),
              gear_values=",".join(str(int(g)) for g in sorted(pd.unique(out["gear"].dropna()))) if "gear" in out else "")
    st.update(gps_stats(t, v, out["lat"].to_numpy(), out["lon"].to_numpy()))
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pq, index=False)
    js.write_text(json.dumps(_round(st, 6), indent=1, default=float))
    st["from_cache"] = False
    return st


# ----------------------------------------------------------------------------- 10 Hz series
def build_10hz(t20: np.ndarray, v20_mps: np.ndarray, standstill_mps: float = STANDSTILL_MPS, accel_win_s: float = ACCEL_WIN_S) -> pd.DataFrame:
    """20 Hz speed -> 10 Hz frame (t_s, v_mps, a_mps2, standstill, das_enabled). Speed by causal hold on the 10 Hz grid
    (max age 1.5 x source dt), rows without a fresh sample dropped (so a source gap becomes a time gap for cut_windows);
    a_mps2 = common.smooth_derivative (central difference + 0.5-s moving average) inside each contiguous run."""
    g, v10 = T.resample_to_10hz(t20, v20_mps, "decimate")
    ok = np.isfinite(v10)
    g, v10 = g[ok], v10[ok]
    a10 = np.full(len(g), np.nan)
    if len(g):
        breaks = np.flatnonzero(np.diff(g) > 1.5 * T.DT_S) + 1
        for s, e in zip(np.r_[0, breaks], np.r_[breaks, len(g)]):
            if e - s >= 3:
                a10[s:e] = T.smooth_derivative(v10[s:e], hz=T.HZ, win_s=accel_win_s)
    return pd.DataFrame(dict(t_s=g, v_mps=v10.astype(np.float32), a_mps2=a10.astype(np.float32),
                             standstill=(v10 < standstill_mps).astype(np.float32), das_enabled=np.zeros(len(g), np.float32)))


def cut_trip(trip: str, df10: pd.DataFrame, spec: dict, align: str) -> T.CutResult:
    info = dict(dataset=DATASET, vehicle_id=VEHICLE_ID, model=MODEL, platform=PLATFORM, drive_id=f"trip_{trip}",
                footprint_m2=spec["footprint_m2"], size_class=spec["size_class"], powertrain=POWERTRAIN, borderline=spec["borderline"], note=NOTE)
    cmap = {"v_mps": "v_mps", "a_mps2": "a_mps2", "standstill": "standstill", "das_enabled": "das_enabled"}
    return T.cut_windows(df10, cmap, t_col="t_s", info=info, align=align)


# ----------------------------------------------------------------------------- validation of the saved set
def validate_saved(spec: dict, trips: pd.DataFrame) -> dict:
    X, meta = T.load_windows(DATASET)
    n = X.shape[0]
    rep: dict = dict(n_windows=n, n_vehicles=int(meta["vehicle_id"].nunique()), n_models=int(meta["model"].nunique()), n_drives=int(meta["drive_id"].nunique()),
                     hours=round(n * T.WINDOW_SAMPLES * T.DT_S / 3600.0, 2), problems=[])
    prob = rep["problems"]
    rep["per_vehicle"] = meta.groupby("vehicle_id").size().to_dict()
    rep["per_model"] = meta.groupby("model").size().to_dict()
    rep["per_size_class"] = meta["size_class"].fillna("unknown").value_counts().to_dict()
    rep["per_powertrain"] = meta["powertrain"].value_counts().to_dict()
    rep["per_borderline"] = meta["borderline"].value_counts().to_dict()
    rep["footprint_values"] = sorted(float(x) for x in np.unique(meta["footprint_m2"].to_numpy(np.float32)))
    if set(meta["size_class"]) != {spec["size_class"]} or set(meta["powertrain"]) != {POWERTRAIN} or meta["borderline"].any() != spec["borderline"]:
        prob.append("label columns differ from the spec")
    if abs(rep["footprint_values"][0] - spec["footprint_m2"]) > 1e-3 or len(rep["footprint_values"]) != 1:
        prob.append("footprint_m2 differs from the spec")
    # channel presence pattern
    present_idx = [T.CH[c] for c in PRESENT]
    absent_idx = [i for i in range(T.N_CHANNELS) if i not in present_idx]
    nan_present = {c: int(np.isnan(X[:, :, T.CH[c]]).sum()) for c in PRESENT}
    finite_absent = {T.CHANNELS[i]: int(np.isfinite(X[:, :, i]).sum()) for i in absent_idx}
    rep["nan_in_present_channels"] = nan_present
    rep["finite_in_absent_channels"] = {k: v for k, v in finite_absent.items() if v}
    if any(nan_present.values()):
        prob.append(f"NaN inside declared-present channels: {nan_present}")
    if any(finite_absent.values()):
        prob.append(f"values inside channels declared absent: {rep['finite_in_absent_channels']}")
    cp = set(meta["channels_present"].unique())
    rep["channels_present"] = sorted(cp)
    if cp != {",".join(PRESENT)}:
        prob.append(f"channels_present {cp} != {','.join(PRESENT)}")
    # physical ranges and flags
    v = X[:, :, T.CH["v_mps"]]; a = X[:, :, T.CH["a_mps2"]]; ss = X[:, :, T.CH["standstill"]]; das = X[:, :, T.CH["das_enabled"]]
    rep["v_mps"] = dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean()), std=float(v.std()), p50=float(np.median(v)),
                        frac_moving=float((v >= T.MOVING_V_MPS).mean()), frac_zero=float((v == 0).mean()), max_kmh=float(v.max() * 3.6))
    rep["a_mps2"] = dict(min=float(a.min()), max=float(a.max()), std=float(a.std()), p01=float(np.quantile(a, 0.01)), p99=float(np.quantile(a, 0.99)),
                         p001=float(np.quantile(a, 0.001)), p999=float(np.quantile(a, 0.999)), frac_abs_gt_4=float((np.abs(a) > 4).mean()))
    rep["standstill"] = dict(mean=float(ss.mean()), values=sorted(float(x) for x in np.unique(ss)), agrees_with_v=bool(np.array_equal(ss, (v < STANDSTILL_MPS).astype(np.float32))))
    rep["das_enabled"] = dict(mean=float(das.mean()), values=sorted(float(x) for x in np.unique(das)))
    if v.min() < 0 or v.max() > 60:
        prob.append(f"v_mps outside [0, 60]: {rep['v_mps']}")
    if np.abs(a).max() > 8:
        prob.append(f"|a_mps2| above 8 m/s^2 (training clip): max {np.abs(a).max():.2f}")
    for c, arr in (("standstill", ss), ("das_enabled", das)):
        if not np.isin(np.unique(arr), (0.0, 1.0)).all():
            prob.append(f"{c} not 0/1")
    if not rep["standstill"]["agrees_with_v"]:
        prob.append("standstill differs from v < 0.1 m/s")
    # window rule and meta consistency
    mf = (v >= T.MOVING_V_MPS).mean(1)
    rep["moving_frac"] = dict(min=float(mf.min()), median=float(np.median(mf)), mean=float(mf.mean()), max=float(mf.max()))
    if (mf < T.MIN_MOVING_FRAC).any():
        prob.append("windows below the moving rule")
    if np.abs(meta["moving_frac"].to_numpy(np.float32) - mf).max() > 1e-5:
        prob.append("meta.moving_frac disagrees with X")
    if not (meta["das_frac"].to_numpy(np.float32) == 0).all():
        prob.append("das_frac != 0")
    # windows per trip; trips with none
    per_drive = meta.groupby("drive_id").size()
    rep["windows_per_trip"] = dict(min=int(per_drive.min()), median=float(per_drive.median()), max=int(per_drive.max()), n_trips_with_windows=int(len(per_drive)))
    rep["trips_without_windows"] = sorted(set(trips["trip"].astype(str)) - {d.split("_")[1] for d in per_drive.index})
    # ids
    bad = [s for s in pd.concat([meta["vehicle_id"], meta["drive_id"]]).unique() if not re.match(r"^[A-Za-z0-9][A-Za-z0-9 ._+()/-]*$", s)]
    if bad:
        prob.append(f"ids with unusual characters: {bad[:5]}")
    hex16 = re.compile(r"\b[0-9a-f]{16}\b")
    if any(hex16.search(str(s)) for s in pd.concat([meta["vehicle_id"], meta["drive_id"], meta["note"]]).unique()):
        prob.append("16-hex token inside meta")
    # window time coverage vs trip duration
    tot_trip_s = float(trips["duration_s"].sum())
    rep["coverage"] = dict(trip_hours=round(tot_trip_s / 3600.0, 2), window_hours=rep["hours"], share=round(n * 60.0 / tot_trip_s, 3) if tot_trip_s else np.nan)
    # order / round trip
    w = meta.groupby("drive_id")["w_idx"].apply(lambda s: bool((np.diff(s.to_numpy()) == 1).all()))
    if not w.all():
        prob.append("w_idx not consecutive inside a drive")
    return rep


def _round(o, nd: int = 4):
    """Round every float in a nested structure (keeps the JSON free of 16-digit fragments that look like hex ids)."""
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    if isinstance(o, (float, np.floating)):
        return round(float(o), nd)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


# ----------------------------------------------------------------------------- markdown helpers
def mapping_rows() -> list[list]:
    """channel -> source / status table for DS_TESLA_YUAN.md."""
    rows = [
        ["v_mps", "`DAActualSpeed` (km/h, 20 Hz, CAN vehicle speed) / 3.6, held onto the 10 Hz grid (`resample_to_10hz` 'decimate': the sample at or before each grid time, max age 0.075 s)", "measured"],
        ["a_mps2", "`common.smooth_derivative(v_mps)` on the 10 Hz series: central difference + 0.5-s moving average, per contiguous run", "derived"],
        ["standstill", "`v_mps < 0.1` (the speed is exactly 0 at rest; 0.0216 m/s resolution)", "derived"],
        ["das_enabled", "constant 0: Autopilot / TACC use is not recorded in the data set or the paper -> assumed human-driven", "assumed (documented)"],
    ]
    for c in T.CHANNELS:
        if c not in PRESENT:
            rows.append([c, "not in the data set (no steering, yaw / IMU, pedal, torque or lead measurements)", "missing (NaN)"])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=T.EXTERNAL_ROOT / DATASET / "raw" / "extracted" / "DATA S1" / "data" / "real_world_data")
    ap.add_argument("--work", type=Path, default=T.work_dir(DATASET))
    ap.add_argument("--specs", type=Path, default=HERE / "specs_external.csv")
    ap.add_argument("--align", default="run", choices=["run", "drive"], help="cut_windows alignment (no gaps here, so both are identical)")
    ap.add_argument("--standstill-mps", type=float, default=STANDSTILL_MPS)
    ap.add_argument("--accel-win", type=float, default=ACCEL_WIN_S, help="smoothing window (s) of the derived a_mps2; 0.5 = common.smooth_derivative default")
    ap.add_argument("--workers", type=int, default=4, help="parallel workbook readers")
    ap.add_argument("--limit", type=int, default=0, help="only the first N trips (smoke test)")
    ap.add_argument("--no-cache", action="store_true", help="re-read the workbooks even when a cache exists")
    ap.add_argument("--dry-run", action="store_true", help="build and validate in memory, do not save")
    args = ap.parse_args()

    spec = load_spec(args.specs)
    print(f"[spec] {MODEL}: footprint {spec['footprint_m2']:.3f} m^2 -> {spec['size_class']} (borderline {spec['borderline']}), {POWERTRAIN}; {spec['source']}")
    trips = find_trips(args.raw)
    dyno_dir = args.raw.parent / "lab_data_m1"
    n_dyno = len(list(dyno_dir.rglob("*.xlsx"))) if dyno_dir.exists() else 0
    if args.limit:
        trips = trips[: args.limit]
    print(f"[raw] {len(trips)} real-world trip workbooks in {args.raw} ({n_dyno} dynamometer workbooks ignored)")
    cache = args.work / "cache_20hz"
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        stats = list(ex.map(read_trip, trips, [cache] * len(trips), [not args.no_cache] * len(trips)))
    print(f"[raw] read {len(stats)} trips in {time.time() - t0:.0f} s ({sum(s['from_cache'] for s in stats)} from cache)")

    results, wstats = [], []
    for st in stats:
        df20 = pd.read_parquet(cache / f"{st['trip']}.parquet")
        df10 = build_10hz(df20["t_s"].to_numpy(), df20["v_kmh"].to_numpy() * KMH_TO_MPS, args.standstill_mps, args.accel_win)
        r = cut_trip(st["trip"], df10, spec, args.align)
        results.append(r)
        n_grid = len(T.make_grid(float(df20["t_s"].min()), float(df20["t_s"].max())))
        wstats.append(dict(trip=st["trip"], n_rows_10hz=len(df10), v10_nan_dropped=n_grid - len(df10), **r.stats))
    X, meta = T.concat_results(results)
    print(f"[cut] {X.shape[0]} windows from {len(results)} trips; candidates {sum(w['n_candidates'] for w in wstats)}, "
          f"dropped gap {sum(w['n_dropped_gap'] for w in wstats)}, dropped stationary {sum(w['n_dropped_stationary'] for w in wstats)}")
    tr = pd.DataFrame(stats).drop(columns=["columns", "from_cache"]).merge(pd.DataFrame(wstats), on="trip")
    if args.dry_run:
        _, m2, warns = T.validate_windows(X, meta)
        print(f"[dry-run] validate_windows ok: {len(m2)} rows; warnings {warns}")
        print(tr.to_string())
        return 0
    args.work.mkdir(parents=True, exist_ok=True)
    tr.round(6).to_csv(args.work / "trips.csv", index=False)
    res = T.save_windows(DATASET, X, meta, root=args.work.parent.parent)
    rep = validate_saved(spec, tr)
    rep.update(spec=spec, n_trips_read=len(stats), n_dyno_ignored=n_dyno, align=args.align, standstill_mps=args.standstill_mps, accel_win_s=args.accel_win,
               raw_checks=dict(n_dt_irregular=int(tr["n_dt_irregular"].sum()), n_src_gaps=int(tr["n_src_gaps"].sum()), n_dt_nonpos=int(tr["n_dt_nonpos"].sum()),
                               n_nan_v=int(tr["n_nan_v"].sum()), n_nan_t=int(tr["n_nan_t"].sum()), dt_median=sorted(set(np.round(tr["dt_median"], 4))),
                               gps_cadence_s=dict(min=float(tr["gps_cadence_s"].min()), median=float(tr["gps_cadence_s"].median()), max=float(tr["gps_cadence_s"].max())),
                               gps_over_int=dict(min=float(tr["gps_over_int"].min()), median=float(tr["gps_over_int"].median()), max=float(tr["gps_over_int"].max())),
                               gps_slope=dict(min=float(tr["gps_slope"].min()), median=float(tr["gps_slope"].median()), max=float(tr["gps_slope"].max())),
                               wall_over_phase=dict(min=float((tr["wall_span_s"] / tr["duration_s"]).min()), max=float((tr["wall_span_s"] / tr["duration_s"]).max())),
                               duration_s=dict(min=float(tr["duration_s"].min()), median=float(tr["duration_s"].median()), max=float(tr["duration_s"].max()), total_h=float(tr["duration_s"].sum() / 3600)),
                               v_max_kmh=float(tr["v_max_kmh"].max()), v_min_kmh=float(tr["v_min_kmh"].min()), gear_values=sorted(set(tr["gear_values"])),
                               columns=stats[0]["columns"]),
               save_warnings=res["warnings"])
    (args.work / "validation.json").write_text(json.dumps(_round(rep), indent=1, default=float))
    print(f"[validate] problems: {rep['problems'] or 'none'}")
    print(f"[validate] windows {rep['n_windows']} ({rep['hours']} h) from {rep['windows_per_trip']['n_trips_with_windows']} trips; per trip min/median/max "
          f"{rep['windows_per_trip']['min']}/{rep['windows_per_trip']['median']}/{rep['windows_per_trip']['max']}; trips without windows {rep['trips_without_windows']}")
    r4 = _round(rep)
    print(f"[validate] v_mps {r4['v_mps']}")
    print(f"[validate] a_mps2 {r4['a_mps2']}")
    print(f"[validate] standstill {r4['standstill']}  das {r4['das_enabled']}  moving_frac {r4['moving_frac']}  coverage {r4['coverage']}")
    print(f"[validate] raw checks {r4['raw_checks']}")
    print("\n" + T.md_table(["channel", "source", "status"], mapping_rows()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
