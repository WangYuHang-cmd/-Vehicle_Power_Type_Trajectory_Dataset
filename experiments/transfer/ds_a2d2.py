#!/usr/bin/env python3
"""A2D2 (Audi Autonomous Driving Dataset) sensor-fusion bus signals -> VTT transfer windows (experiments/transfer contract).

Input: the three bus-signal JSONs <external>/a2d2/raw/bus/<seq>_bus_signals.json (one per sensor-fusion sequence), each a
dict {signal: {"unit": str | None, "values": [[timestamp_us, value], ...]}}; every signal runs on its own clock (Unix
microseconds), all start within 20 ms of each other and none has a gap (50 Hz speed, 100 Hz steering / pedals, 200 Hz IMU).
Single recording car: one Audi Q7 e-tron quattro (4M, 3.0 TDI diesel plug-in hybrid), driven manually.

Mapping to the 20 training channels (status measured / derived / missing; details and checks in DS_A2D2.md):
  v_mps             vehicle_speed (km/h, 50 Hz) / 3.6, zero-order hold onto the 10 Hz grid (lab sampling rule)    measured
  a_mps2            common.smooth_derivative of the 10 Hz speed; the lab aEgo is a speed-filter output and the IMU
                    acceleration_x is gravity-contaminated (pitch) and rougher -> default 'speed', --accel imu       derived
  steer_angle_deg   steering_angle_calculated x (+1 if steering_angle_calculated_sign == 0 else -1), 100 Hz, hold;
                    sign bit 1 = right turn, so the result is left-positive (checked against the IMU yaw rate)      measured
  steer_rate_degps  common.smooth_derivative of the 10 Hz signed angle (no CAN rate signal)                        derived
  steer_torque_norm no driver torque                                                                                missing
  yaw_rate_radps    angular_velocity_omega_z (deg/s, 200 Hz) x pi/180, hold; left-positive (GPS heading check)      measured
  curv_1pm, lat_acc_mps2  common.curvature_and_lat_acc(v, yaw) = yaw / v (v >= 1) and v x yaw (lab definitions);
                    --lat-acc imu puts acceleration_y there instead (roll-contaminated, kept for comparison)        derived
  gas_pressed       accelerator_pedal (%, 100 Hz) > --gas-pct (default 0: opendbc Volkswagen gasPressed = gas > 0)  derived
  brake_pressed     brake_pressure (bar, 100 Hz) > --brake-bar (default 0.5: two 0.3-bar steps above the zero level,
                    whose readings wander between -1.2 and 0 bar)                                                   derived
  steer_pressed     unknown (no torque, no hands-on flag) -> NaN by default (no training augmentation covers this: see
                    DS_A2D2.md); --steer-pressed zero | one | active (|steer_rate| > 5 deg/s or |angle| > 15 deg) are
                    diagnostic substitutes for sensitivity runs, never the default                                    missing
  standstill        v < 0.5 m/s                                                                                       derived
  das_enabled       0 everywhere (manual driving, no automation on the car)                                          derived
  lead group        masked exactly as the training augmentation drops it: lead_valid = 0, the six lead values NaN    missing
Labels: wheelbase 2994 mm x mean track 1685 mm = 5.045 m^2 -> LARGE, not borderline; powertrain PHEV (outside the
BEV-vs-ICE task: the scorer reports predicted shares only).

Usage:  source env.sh && $CPY -I experiments/transfer/ds_a2d2.py [--raw DIR] [--align drive|run] [--accel speed|imu]
        [--lat-acc yaw|imu] [--steer-pressed nan|zero|one|active] [--gas-pct 0] [--brake-bar 0.5] [--dataset NAME] [--dry-run]
Writes <external>/a2d2/work/: windows.npz + meta.parquet + windows_summary.json (common.save_windows),
series_<drive>.parquet (10 Hz frames incl. diagnostic columns) and validation.json (every check printed at the end).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.transfer import common as T  # noqa: E402

DATASET = "a2d2"
RAW_DEFAULT = T.EXTERNAL_ROOT / DATASET / "raw" / "bus"
SEQ_SUFFIX = "_bus_signals.json"

# ----------------------------------------------------------------------------- labels (public specs, SPECS_EXTERNAL.md row "Audi Q7 e-tron quattro")
VEHICLE = dict(vehicle_id="a2d2-audi-q7-etron", model="Audi Q7 e-tron", platform="Audi Q7 e-tron quattro 4M 2016-2018",
               footprint_m2=5.045, size_class="LARGE", powertrain="PHEV", borderline=False)
SPEC = dict(wheelbase_mm=2994.0, track_front_mm=1679.0, track_rear_mm=1691.0, curb_mass_kg=2445.0)
LOCATIONS = {"Gaimersheim": (48.81, 11.37), "Ingolstadt": (48.765, 11.425), "Munich": (48.137, 11.575)}   # README-SensorFusion

# ----------------------------------------------------------------------------- thresholds and constants
GAS_PCT = 0.0            # gas_pressed = accelerator_pedal > 0 %
BRAKE_BAR = 0.5          # brake_pressed = brake_pressure > 0.5 bar
STANDSTILL_MPS = 0.5
HOLD_AGE_FACTOR = 3.0    # zero-order hold tolerates 3 nominal periods (gap-free streams never reach it)
DEG2RAD = np.pi / 180.0
KMH_TO_MPS = 1.0 / 3.6
GPS_FAST_MPS = 6.0       # heading check uses fixes while the car is faster than this
ACTIVE_RATE_DEGPS = 5.0  # --steer-pressed active: driver 'steering' while |steer_rate| > 5 deg/s or |steer_angle| > 15 deg
ACTIVE_ANGLE_DEG = 15.0

SIGNALS = {  # json key -> short name
    "vehicle_speed": "v_kmh", "steering_angle_calculated": "steer_mag_deg", "steering_angle_calculated_sign": "steer_sign",
    "angular_velocity_omega_z": "gyro_z_degps", "acceleration_x": "imu_ax", "acceleration_y": "imu_ay",
    "accelerator_pedal": "gas_pct", "brake_pressure": "brake_bar", "latitude_degree": "lat_deg", "longitude_degree": "lon_deg",
}
UNITS = {  # substring of the json unit string (A2D2 spells "Unit_KiloMeterPerHour", "Unit_DegreOfArcPerSecon", ...)
    "vehicle_speed": "KiloMeterPerHour", "steering_angle_calculated": "DegreOfArc", "angular_velocity_omega_z": "DegreOfArcPerSecon",
    "acceleration_x": "MeterPerSeconSquar", "acceleration_y": "MeterPerSeconSquar", "accelerator_pedal": "PerCent",
    "brake_pressure": "Bar", "latitude_degree": "DegreOfArc", "longitude_degree": "DegreOfArc",
}
GRID_SIGNALS = ("v_kmh", "steer_mag_deg", "steer_sign", "gyro_z_degps", "imu_ax", "imu_ay", "gas_pct", "brake_bar")
MODEL_COLUMNS = ("v_mps", "a_mps2", "steer_angle_deg", "steer_rate_degps", "yaw_rate_radps", "curv_1pm", "lat_acc_mps2",
                 "gas_pressed", "brake_pressed", "standstill", "das_enabled")
CHANNEL_MAP = {c: (c if c in MODEL_COLUMNS else None) for c in T.CHANNELS}     # None = NaN (torque, steer_pressed, lead group)
STEER_PRESSED_MODES = ("nan", "zero", "one", "active")
PHYSICAL_RANGE = {"v_mps": (0.0, 60.0), "a_mps2": (-8.0, 8.0), "steer_angle_deg": (-720.0, 720.0), "steer_rate_degps": (-720.0, 720.0),
                  "yaw_rate_radps": (-1.5, 1.5), "curv_1pm": (-0.3, 0.3), "lat_acc_mps2": (-8.0, 8.0)}


# ----------------------------------------------------------------------------- raw -> arrays
def load_sequence(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{short name: (t_us int64 sorted, values float64)} for the signals in SIGNALS; checks presence and units."""
    with open(path) as f:
        d = json.load(f)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, name in SIGNALS.items():
        if key not in d or "values" not in d[key]:
            raise KeyError(f"{path.name}: signal {key!r} missing")
        unit = d[key].get("unit")
        exp = UNITS.get(key)
        if exp and exp not in str(unit):
            raise ValueError(f"{path.name}: {key} has unit {unit!r}, expected *{exp}*")
        arr = np.asarray(d[key]["values"], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
            raise ValueError(f"{path.name}: {key} is not a [timestamp, value] list")
        t_us = np.rint(arr[:, 0]).astype(np.int64)                 # Unix microseconds are exact in float64 (< 2^53)
        v = arr[:, 1]
        if (np.diff(t_us) < 0).any():
            order = np.argsort(t_us, kind="stable")
            t_us, v = t_us[order], v[order]
        out[name] = (t_us, v)
    del d
    return out


def signal_rate_hz(t_s: np.ndarray) -> float:
    d = np.diff(t_s)
    d = d[d > 0]
    return float(1.0 / np.median(d)) if len(d) else float("nan")


# ----------------------------------------------------------------------------- arrays -> 10 Hz frame
def build_series(sig: dict, *, accel: str = "speed", lat_acc: str = "yaw", gas_pct: float = GAS_PCT, brake_bar: float = BRAKE_BAR,
                 steer_pressed: str = "nan") -> tuple[pd.DataFrame, dict, dict]:
    """10 Hz frame (t_s relative to the first speed sample) with the model columns plus diagnostic columns, the GPS fixes
    (own rate) and a dict of source-stream facts (rates, max dt, clock offsets)."""
    t_ref = int(sig["v_kmh"][0][0])
    rel = {k: (sig[k][0] - t_ref) / 1e6 for k in sig}                    # seconds relative to the first speed sample
    facts = {k: dict(n=int(len(rel[k])), rate_hz=round(signal_rate_hz(rel[k]), 2), max_dt_s=round(float(np.max(np.diff(rel[k]))), 4),
                     first_s=round(float(rel[k][0]), 4), last_s=round(float(rel[k][-1]), 4)) for k in sig}
    t_start = max(rel[k][0] for k in GRID_SIGNALS)
    t_end = min(rel[k][-1] for k in GRID_SIGNALS)
    g = T.make_grid(t_start, t_end)

    def hold(name: str, values=None) -> np.ndarray:
        t = rel[name]
        v = sig[name][1] if values is None else np.asarray(values, dtype=np.float64)
        age = HOLD_AGE_FACTOR / signal_rate_hz(t)
        return T.resample_to_10hz(t, v, "hold", t_grid=g, max_age_s=age)[1]

    v = hold("v_kmh") * KMH_TO_MPS
    # steering: magnitude and sign bit are two 100 Hz streams; combine at the source rate, then hold
    t_mag, mag = rel["steer_mag_deg"], sig["steer_mag_deg"][1]
    t_sgn, sgn = rel["steer_sign"], sig["steer_sign"][1]
    if len(t_sgn) != len(t_mag) or not np.allclose(t_sgn, t_mag, atol=1e-6):
        j = np.clip(np.searchsorted(t_sgn, t_mag + 1e-6, side="right") - 1, 0, len(t_sgn) - 1)
        sgn = sgn[j]
    signed = np.where(sgn > 0.5, -mag, mag)                              # sign bit 1 = right (negative), 0 = left (positive)
    steer = hold("steer_mag_deg", signed)
    yaw = hold("gyro_z_degps") * DEG2RAD
    imu_ax, imu_ay = hold("imu_ax"), hold("imu_ay")
    gas10, brake10 = hold("gas_pct"), hold("brake_bar")
    gas_flag = hold("gas_pct", (sig["gas_pct"][1] > gas_pct).astype(np.float64))
    brake_flag = hold("brake_bar", (sig["brake_bar"][1] > brake_bar).astype(np.float64))
    a_speed = T.smooth_derivative(v)
    steer_rate = T.smooth_derivative(steer)
    curv, lat_yaw = T.curvature_and_lat_acc(v, yaw)
    if steer_pressed not in STEER_PRESSED_MODES:
        raise ValueError(f"steer_pressed mode {steer_pressed!r} not in {STEER_PRESSED_MODES}")
    sp = np.full(len(g), np.nan)
    if steer_pressed == "zero":
        sp[:] = 0.0
    elif steer_pressed == "one":
        sp[:] = 1.0
    elif steer_pressed == "active":
        sp = ((np.abs(steer_rate) > ACTIVE_RATE_DEGPS) | (np.abs(steer) > ACTIVE_ANGLE_DEG)).astype(np.float64)
    df = pd.DataFrame(dict(
        t_s=g, v_mps=v, a_mps2=(a_speed if accel == "speed" else imu_ax), steer_angle_deg=steer, steer_rate_degps=steer_rate,
        yaw_rate_radps=yaw, curv_1pm=curv, lat_acc_mps2=(lat_yaw if lat_acc == "yaw" else imu_ay), gas_pressed=gas_flag,
        brake_pressed=brake_flag, standstill=(v < STANDSTILL_MPS).astype(np.float64), das_enabled=np.zeros(len(g)), steer_pressed=sp,
        # diagnostics, not model inputs
        imu_ax_mps2=imu_ax, imu_ay_mps2=imu_ay, a_speed_mps2=a_speed, lat_acc_yaw_mps2=lat_yaw, gas_pct=gas10, brake_bar=brake10,
    ))
    gps = dict(t=rel["lat_deg"], lat=sig["lat_deg"][1], lon=sig["lon_deg"][1])
    facts["_grid"] = dict(t_start_s=float(g[0]), t_end_s=float(g[-1]), n=int(len(g)))
    return df, gps, facts


# ----------------------------------------------------------------------------- per-sequence checks
def _corr(x, y) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 3 else float("nan")


def _fit(x, y) -> tuple[float, float]:
    """(slope, intercept) of y ~ x over finite pairs."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    s, i = np.polyfit(x[ok], y[ok], 1)
    return float(s), float(i)


def _roughness(x: np.ndarray, sel: np.ndarray, win: int = 15) -> float:
    """std of x minus its centred `win`-sample moving average (high-frequency content) over `sel`."""
    k = np.ones(win) / win
    pad = win // 2
    lp = np.convolve(np.pad(x, (pad, win - 1 - pad), mode="edge"), k, mode="valid")
    return float(np.nanstd((x - lp)[sel]))


def _bearing(la1, lo1, la2, lo2):
    dlon = lo2 - lo1
    x = np.sin(dlon) * np.cos(la2)
    y = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlon)
    return np.arctan2(x, y)


def gps_heading_check(gps: dict, sig: dict, t_ref_us: int, df: pd.DataFrame, k_base: int = 6, m_span: int = 18) -> dict:
    """Heading change from GPS (clockwise-positive compass bearing of ~1-s baselines, differenced ~3 s apart) regressed on
    the integrated raw gyro over the same intervals while v > GPS_FAST_MPS. A left-positive gyro gives slope ~ -1."""
    lat, lon, t = np.deg2rad(gps["lat"]), np.deg2rad(gps["lon"]), gps["t"]
    if len(t) <= k_base + m_span + 2:
        return dict(n=0, slope=float("nan"), corr=float("nan"), verdict="insufficient GPS")
    b = _bearing(lat[:-k_base], lon[:-k_base], lat[k_base:], lon[k_base:])
    tb = 0.5 * (t[:-k_base] + t[k_base:])
    db = b[m_span:] - b[:-m_span]
    db = (db + np.pi) % (2 * np.pi) - np.pi
    ta, tz = tb[:-m_span], tb[m_span:]
    t_w, wz = (sig["gyro_z_degps"][0] - t_ref_us) / 1e6, sig["gyro_z_degps"][1]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (wz[1:] + wz[:-1]) * np.diff(t_w))]) * DEG2RAD
    dyaw = np.interp(tz, t_w, cum) - np.interp(ta, t_w, cum)
    v_mid = np.interp(0.5 * (ta + tz), df["t_s"].to_numpy(), df["v_mps"].to_numpy())
    ok = (v_mid > GPS_FAST_MPS) & np.isfinite(db) & np.isfinite(dyaw)
    if ok.sum() < 30:
        return dict(n=int(ok.sum()), slope=float("nan"), corr=float("nan"), verdict="insufficient fast GPS")
    slope, _ = _fit(dyaw[ok], db[ok])
    big = ok & (np.abs(dyaw) > 0.2)
    ratio = float(np.median(db[big] / dyaw[big])) if big.sum() >= 5 else float("nan")
    verdict = "gyro left-positive (OK)" if slope < -0.5 else ("gyro appears clockwise-positive: SIGN ERROR" if slope > 0.5 else "ambiguous")
    return dict(n=int(ok.sum()), slope=round(slope, 3), median_ratio=round(ratio, 3), corr=round(_corr(dyaw[ok], db[ok]), 3), verdict=verdict)


def nearest_location(gps: dict) -> str:
    la, lo = float(np.median(gps["lat"])), float(np.median(gps["lon"]))
    name, dist = min(((n, np.hypot(la - c[0], lo - c[1])) for n, c in LOCATIONS.items()), key=lambda x: x[1])
    return name if dist < 0.15 else "unknown"


def sequence_diagnostics(df: pd.DataFrame, gps: dict, sig: dict, facts: dict) -> dict:
    v = df["v_mps"].to_numpy()
    steer = df["steer_angle_deg"].to_numpy()
    yaw = df["yaw_rate_radps"].to_numpy()
    moving = v >= T.MOVING_V_MPS
    fast = v >= 3.0
    d: dict = dict(n_rows=int(len(df)), duration_s=round(float(df["t_s"].iloc[-1] - df["t_s"].iloc[0]) + T.DT_S, 1),
                   location=nearest_location(gps), streams=facts)
    d["speed"] = dict(max_mps=round(float(v.max()), 2), p50_mps=round(float(np.percentile(v, 50)), 2), p99_mps=round(float(np.percentile(v, 99)), 2),
                      moving_share=round(float(moving.mean()), 3), standstill_share=round(float((v < STANDSTILL_MPS).mean()), 3))
    # lateral consistency: steering vs IMU yaw, implied steering ratio (small-angle bicycle model), IMU lateral accel vs v x yaw
    sel = fast & (np.abs(yaw) > 0.05)
    ratio = float(np.median(v[sel] * np.deg2rad(np.abs(steer[sel])) / (SPEC["wheelbase_mm"] / 1000.0 * np.abs(yaw[sel])))) if sel.sum() > 20 else float("nan")
    ay = df["imu_ay_mps2"].to_numpy()
    lat_yaw = df["lat_acc_yaw_mps2"].to_numpy()
    s_ay, i_ay = _fit(lat_yaw[fast], ay[fast])
    d["lateral"] = dict(corr_steer_vs_yaw_moving=round(_corr(steer[fast], yaw[fast]), 3), implied_steer_ratio=round(ratio, 2),
                        imu_ay_vs_v_yaw=dict(corr=round(_corr(lat_yaw[fast], ay[fast]), 3), slope=round(s_ay, 3), offset_mps2=round(i_ay, 3),
                                             resid_std_mps2=round(float(np.nanstd((ay - lat_yaw)[fast])), 3)),
                        gps_heading=gps_heading_check(gps, sig, int(sig["v_kmh"][0][0]), df))
    # longitudinal: IMU a_x vs differentiated speed
    ax = df["imu_ax_mps2"].to_numpy()
    a_sp = df["a_speed_mps2"].to_numpy()
    s_ax, i_ax = _fit(a_sp[moving], ax[moving])
    d["longitudinal"] = dict(imu_ax_vs_dv_dt=dict(corr=round(_corr(a_sp[moving], ax[moving]), 3), slope=round(s_ax, 3), offset_mps2=round(i_ax, 3),
                                                  resid_std_mps2=round(float(np.nanstd((ax - a_sp)[moving])), 3)),
                             roughness_moving=dict(imu_ax=round(_roughness(ax, moving), 3), dv_dt_smooth=round(_roughness(a_sp, moving), 3)),
                             std_moving=dict(imu_ax=round(float(np.nanstd(ax[moving])), 3), dv_dt_smooth=round(float(np.nanstd(a_sp[moving])), 3)))
    # sensor zero levels during long stops (>= 5 s, inner part)
    z = (v < 0.05).astype(int)
    e = np.diff(np.concatenate([[0], z, [0]]))
    inner = np.zeros(len(v), bool)
    for a0, b0 in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if b0 - a0 >= 50:
            inner[a0 + 10:b0 - 10] = True
    if inner.sum() >= 20:
        d["standstill_zero_levels"] = dict(n=int(inner.sum()), gyro_z_mean_degps=round(float(np.mean(yaw[inner]) / DEG2RAD), 3),
                                           gyro_z_std_degps=round(float(np.std(yaw[inner]) / DEG2RAD), 3), imu_ax_mean=round(float(np.mean(ax[inner])), 3),
                                           imu_ax_std=round(float(np.std(ax[inner])), 3), imu_ay_mean=round(float(np.mean(ay[inner])), 3))
    else:
        d["standstill_zero_levels"] = dict(n=int(inner.sum()))
    # pedal thresholds (10 Hz samples)
    gp, bb = df["gas_pct"].to_numpy(), df["brake_bar"].to_numpy()
    d["pedal_thresholds"] = dict(gas_pressed_share={f">{t:g}%": round(float((gp > t).mean()), 3) for t in (0.0, 2.0, 5.0, 10.0)},
                                 brake_pressed_share={f">{t:g}bar": round(float((bb > t).mean()), 3) for t in (0.0, 0.5, 1.0, 2.0, 3.0)},
                                 gas_and_brake_share=round(float(((gp > GAS_PCT) & (bb > BRAKE_BAR)).mean()), 4),
                                 gas_pct_max=round(float(gp.max()), 1), brake_bar_max=round(float(bb.max()), 1),
                                 brake_bar_min=round(float(bb.min()), 1))
    # completeness of the model columns on the grid
    nan = {c: int(np.isnan(df[c].to_numpy()).sum()) for c in MODEL_COLUMNS}
    curv_nan_ok = bool(np.array_equal(np.isnan(df["curv_1pm"].to_numpy()), ~moving))
    d["grid_nan"] = dict(counts={k: n for k, n in nan.items() if n}, curv_nan_only_where_v_below_1=curv_nan_ok)
    return d


# ----------------------------------------------------------------------------- post-cut checks
def check_windows(X: np.ndarray, meta: pd.DataFrame) -> tuple[dict, list[str]]:
    """Counts, ranges and completeness of the saved windows; returns (report, problems)."""
    problems: list[str] = []
    rep: dict = dict(n_windows=int(X.shape[0]), shape=list(X.shape), dtype=str(X.dtype), hours=round(X.shape[0] / 60.0, 3))
    if X.shape[1:] != (T.WINDOW_SAMPLES, T.N_CHANNELS):
        problems.append(f"shape {X.shape}")
    rep["per_drive"] = meta.groupby("drive_id", sort=True).agg(windows=("w_idx", "size"), moving_frac_mean=("moving_frac", "mean"),
                                                                moving_frac_min=("moving_frac", "min")).round(3).reset_index().to_dict("records")
    rep["per_vehicle"] = meta.groupby("vehicle_id")["w_idx"].size().to_dict()
    rep["per_model"] = meta.groupby("model")["w_idx"].size().to_dict()
    rep["per_size_class"] = meta["size_class"].value_counts().to_dict()
    rep["per_powertrain"] = meta["powertrain"].value_counts().to_dict()
    rep["borderline_windows"] = int(meta["borderline"].astype(bool).sum())
    present = sorted({c for s in meta["channels_present"] for c in s.split(",") if c}, key=T.CH.get)
    rep["channels_present"] = present
    v = X[:, :, T.CH["v_mps"]]
    moving = v >= T.MOVING_V_MPS
    rep["moving_frac"] = dict(min=round(float(moving.mean(1).min()), 3), mean=round(float(moving.mean(1).mean()), 3))
    if (moving.mean(1) < T.MIN_MOVING_FRAC).any():
        problems.append("window below the moving rule")
    # completeness: present channels have no NaN (curv_1pm: NaN exactly where v < 1), missing channels are all NaN
    nan_present = {}
    for c in present:
        col = X[:, :, T.CH[c]]
        if c == "curv_1pm":
            if not np.array_equal(np.isnan(col), ~moving):
                problems.append("curv_1pm NaN pattern differs from v < 1 m/s")
            nan_present[c] = f"{int(np.isnan(col).sum())} (= samples with v < 1 m/s)"
        elif np.isnan(col).any():
            problems.append(f"{c}: {int(np.isnan(col).sum())} NaN inside a present channel")
            nan_present[c] = int(np.isnan(col).sum())
    rep["nan_in_present_channels"] = nan_present
    missing = [c for c in T.CHANNELS if c not in present]
    rep["channels_missing"] = missing
    for c in missing:
        col = X[:, :, T.CH[c]]
        if c == "lead_valid":
            if not np.all(col == 0.0):
                problems.append("lead_valid must be 0 everywhere (masked lead group)")
        elif np.isfinite(col).any():
            problems.append(f"{c} should be NaN everywhere")
    # flags
    for c in T.BINARY_CHANNELS:
        col = X[:, :, T.CH[c]]
        fin = col[np.isfinite(col)]
        if fin.size and not np.isin(fin, (0.0, 1.0)).all():
            problems.append(f"{c} not binary")
    das = X[:, :, T.CH["das_enabled"]]
    if not np.all(das == 0.0):
        problems.append("das_enabled must be 0 everywhere")
    rep["flag_shares"] = {c: round(float(np.nanmean(X[:, :, T.CH[c]])), 3) for c in T.BINARY_CHANNELS if np.isfinite(X[:, :, T.CH[c]]).any()}
    # physical ranges and distribution summary of the continuous channels
    rng = {}
    for c, (lo, hi) in PHYSICAL_RANGE.items():
        col = X[:, :, T.CH[c]]
        fin = col[np.isfinite(col)]
        out = int(((fin < lo) | (fin > hi)).sum())
        rng[c] = dict(min=round(float(fin.min()), 4), p01=round(float(np.percentile(fin, 1)), 4), p50=round(float(np.percentile(fin, 50)), 4),
                      p99=round(float(np.percentile(fin, 99)), 4), max=round(float(fin.max()), 4), mean=round(float(fin.mean()), 4),
                      std=round(float(fin.std()), 4), outside_physical_range=out)
        if out and c not in ("curv_1pm",):
            problems.append(f"{c}: {out} samples outside {lo}..{hi}")
    rep["ranges"] = rng
    if (X[:, :, T.CH["v_mps"]] < 0).any():
        problems.append("negative speed")
    return rep, problems


# ----------------------------------------------------------------------------- main
def make_note(a, location: str) -> str:
    acc = "a from speed (smooth_derivative)" if a.accel == "speed" else "a = IMU acceleration_x"
    lat = "lat_acc = v*yaw" if a.lat_acc == "yaw" else "lat_acc = IMU acceleration_y"
    sp = "steer_pressed unknown" if a.steer_pressed == "nan" else f"steer_pressed := {a.steer_pressed} (diagnostic substitute)"
    return (f"manual driving (das_enabled=0); lead+torque masked; {sp}; {acc}; steer_rate derived; yaw = IMU gyro z; "
            f"{lat}; gas>{a.gas_pct:g}%; brake>{a.brake_bar:g} bar; {location}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_DEFAULT, help="folder with <seq>_bus_signals.json")
    ap.add_argument("--root", type=Path, default=None, help="external root for save_windows (default common.EXTERNAL_ROOT)")
    ap.add_argument("--dataset", default=DATASET, help="window set name (default a2d2; use another name for variant builds)")
    ap.add_argument("--align", default="drive", choices=["drive", "run"], help="window blocks: fixed 600-row blocks from the first row (lab) or restart after gaps")
    ap.add_argument("--accel", default="speed", choices=["speed", "imu"], help="a_mps2 source (default: differentiated speed)")
    ap.add_argument("--lat-acc", default="yaw", choices=["yaw", "imu"], help="lat_acc_mps2 source (default: v x yaw, the lab definition)")
    ap.add_argument("--steer-pressed", default="nan", choices=list(STEER_PRESSED_MODES),
                    help="steer_pressed channel: nan (default, unknown) or a diagnostic substitute (zero, one, active = |rate| > 5 deg/s or |angle| > 15 deg)")
    ap.add_argument("--gas-pct", type=float, default=GAS_PCT, help="gas_pressed = accelerator_pedal > this percent")
    ap.add_argument("--brake-bar", type=float, default=BRAKE_BAR, help="brake_pressed = brake_pressure > this many bar")
    ap.add_argument("--dry-run", action="store_true", help="build and validate, write nothing")
    a = ap.parse_args(argv)
    files = sorted(Path(a.raw).glob(f"*{SEQ_SUFFIX}"))
    if not files:
        print(f"no *{SEQ_SUFFIX} under {a.raw}")
        return 2
    wd = T.work_dir(a.dataset, a.root)
    if not a.dry_run:
        wd.mkdir(parents=True, exist_ok=True)
    results, diags = [], {}
    t0 = time.time()
    for f in files:
        seq = f.name[: -len(SEQ_SUFFIX)]
        drive_id = f"{seq[:8]}_{seq[8:]}" if (seq.isdigit() and len(seq) == 14) else seq       # the A2D2 scene folder name
        sig = load_sequence(f)
        df, gps, facts = build_series(sig, accel=a.accel, lat_acc=a.lat_acc, gas_pct=a.gas_pct, brake_bar=a.brake_bar, steer_pressed=a.steer_pressed)
        diag = sequence_diagnostics(df, gps, sig, facts)
        info = dict(VEHICLE, dataset=a.dataset, drive_id=drive_id, note=make_note(a, diag["location"]))
        cmap = dict(CHANNEL_MAP, steer_pressed=("steer_pressed" if a.steer_pressed != "nan" else None))
        res = T.cut_windows(df, cmap, t_col="t_s", info=info, align=a.align)
        diag["cut"] = res.stats
        diags[drive_id] = diag
        results.append(res)
        if not a.dry_run:
            df.to_parquet(wd / f"series_{drive_id}.parquet", index=False)
        print(f"[{drive_id}] {diag['location']:12s} rows={len(df)} dur={diag['duration_s']}s v_max={diag['speed']['max_mps']} m/s "
              f"moving={diag['speed']['moving_share']:.3f} windows={res.stats['n_windows']}/{res.stats['n_candidates']} "
              f"(gap-drop {res.stats['n_dropped_gap']}, stationary-drop {res.stats['n_dropped_stationary']}) "
              f"GPS-heading slope={diag['lateral']['gps_heading'].get('slope')} corr(steer,yaw)={diag['lateral']['corr_steer_vs_yaw_moving']} "
              f"ratio={diag['lateral']['implied_steer_ratio']}", flush=True)
    X, meta = T.concat_results(results)
    X = T.apply_channel_mask(X, T.CHANNEL_GROUPS["lead"])          # lead_valid = 0, six lead values NaN (training 'no lead sensor')
    rep, problems = check_windows(X, meta)
    report = dict(dataset=a.dataset, built=time.strftime("%Y-%m-%d %H:%M:%S"), options=dict(align=a.align, accel=a.accel, lat_acc=a.lat_acc,
                  steer_pressed=a.steer_pressed, gas_pct=a.gas_pct, brake_bar=a.brake_bar, standstill_mps=STANDSTILL_MPS), vehicle=VEHICLE, spec=SPEC,
                  sequences=diags, windows=rep, problems=problems, seconds=round(time.time() - t0, 1))
    print("\n== windows ==")
    print(T.md_table(["drive", "windows", "moving_frac mean", "moving_frac min"],
                     [[r["drive_id"], r["windows"], r["moving_frac_mean"], r["moving_frac_min"]] for r in rep["per_drive"]]))
    print(f"total {rep['n_windows']} windows ({rep['hours']} h), vehicles {rep['per_vehicle']}, models {rep['per_model']}, "
          f"size {rep['per_size_class']}, powertrain {rep['per_powertrain']}, borderline {rep['borderline_windows']}")
    print(f"present: {','.join(rep['channels_present'])}\nmissing: {','.join(rep['channels_missing'])}")
    print("flag shares:", rep["flag_shares"])
    print(T.md_table(["channel", "min", "p01", "p50", "p99", "max", "mean", "std", "outside range"],
                     [[c, r["min"], r["p01"], r["p50"], r["p99"], r["max"], r["mean"], r["std"], r["outside_physical_range"]] for c, r in rep["ranges"].items()]))
    print("problems:", problems if problems else "none")
    if problems:
        print("refusing to save: fix the problems above")
        return 1
    if a.dry_run:
        print("dry run: nothing written")
        return 0
    saved = T.save_windows(a.dataset, X, meta, root=a.root)
    report["saved"] = {k: str(v) for k, v in saved.items() if k != "warnings"}
    report["save_warnings"] = saved["warnings"]
    (wd / "validation.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"validation -> {wd / 'validation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
