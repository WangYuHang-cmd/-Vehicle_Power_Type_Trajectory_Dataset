"""Synthetic drives in the public schema, for tests, the GitHub `sample/` folder and starter-kit demos.

Nothing here derives from real data: a small stochastic car-following simulation (10 Hz) produces two behaviour
profiles ('bev': stronger acceleration, strong lift-off deceleration, quick catch-up; 'ice': the opposite) so that
the starter-kit features have something to pick up. Tables are written with `vtt.schema.write_public_parquet`, so
they pass the public schema check. Units follow `vtt.schema.DRIVE_COLUMNS` (m/s, m/s^2, deg, rad/s, s).

Used by `scripts/s11_github_export.py` when `vtt.synth` (the pipeline's own synthetic writer) is not importable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vtt import config as C
from vtt import schema as S

DT_S = C.NOMINAL_DT_S
PROFILES = {
    # a_max: max accel [m/s^2]; coast_a: lift-off decel [m/s^2]; resp_delay_s: reaction to the lead [s];
    # k_steer_degpm: steering-wheel angle per unit curvature [deg*m]; lead_source 1 radar / 2 camera model
    "bev": dict(a_max=3.0, brake_a=-3.0, coast_a=-0.9, resp_delay_s=0.6, k_steer_degpm=900.0, v_cruise=22.0, lead_source=2),
    "ice": dict(a_max=2.0, brake_a=-2.5, coast_a=-0.3, resp_delay_s=1.2, k_steer_degpm=1150.0, v_cruise=20.0, lead_source=1),
}
# public labels of the two synthetic platforms (clearly fictitious names)
PLATFORM_META = {
    "bev": dict(brand="SynthBrandA", model="Model-E", platform="SYNTH_E_GEN1", body="suv", size_code="E",
                wheelbase_mm=2900.0, track_front_mm=1640.0, track_rear_mm=1650.0, curb_mass_kg=2150.0,
                powertrain="BEV", size_class="LARGE", radar_unavailable=True),
    "ice": dict(brand="SynthBrandB", model="Model-G", platform="SYNTH_G_GEN1", body="sedan", size_code="D",
                wheelbase_mm=2700.0, track_front_mm=1560.0, track_rear_mm=1570.0, curb_mass_kg=1450.0,
                powertrain="ICE", size_class="MID", radar_unavailable=False),
}


PHASE_CYCLE = ("cruise", "coast", "cruise", "stop")


def _phases(rng: np.random.Generator, n_rows: int, v_cruise: float) -> np.ndarray:
    """Target speed per sample [m/s]: the cycle cruise / coast (lower target, gas off) / cruise / stop repeated with
    random phase lengths, so that every drive of >= 60 s contains an acceleration and a lift-off phase."""
    tgt = np.zeros(n_rows)
    i, k, v = 0, 0, 0.0
    while i < n_rows:
        kind = PHASE_CYCLE[k % len(PHASE_CYCLE)]
        n = int(rng.integers(100, 250))
        if kind == "stop":
            v = 0.0
        elif kind == "cruise":
            v = float(np.clip(v_cruise + rng.normal(0, 3), 8, 35))
        else:
            v = max(v - rng.uniform(3, 6), 2.0)
        tgt[i:i + n] = v
        i += n; k += 1
    return tgt


def _lead_profile(rng: np.random.Generator, n_rows: int, tgt: np.ndarray):
    """Lead vehicle present over random stretches of the moving phases; its speed is a staircase of +3 / -3 m/s
    ramps (3 s each, slope 1 m/s^2) separated by 10-16 s of constant speed, so that lead-acceleration onsets with
    a quiet period before them exist. Returns (present, lead_speed)."""
    present = np.zeros(n_rows, bool)
    v_lead = np.full(n_rows, np.nan)
    i = 0
    while i < n_rows:
        n = int(rng.integers(250, 500))
        if rng.random() < 0.8:
            sl = slice(i, min(i + n, n_rows))
            m = sl.stop - sl.start
            present[sl] = True                      # the lead stays tracked through stops (queueing)
            base = np.maximum(tgt[sl] - rng.uniform(0, 3), 0.0)
            stair = np.zeros(m)
            k, sign = int(rng.integers(30, 80)), 1.0
            while k < m:
                stair[k:] += sign * 3.0 * np.clip((np.arange(k, m) - k) * DT_S / 3.0, 0, 1)
                k += int(rng.integers(100, 160)); sign = -sign
            v_lead[sl] = np.maximum(base + stair, 0.0)
        i += n
    return present, v_lead


def _simulate(rng: np.random.Generator, n_rows: int, prof: dict) -> dict[str, np.ndarray]:
    """Ego motion with a delayed first-order speed controller and lead following; returns raw arrays."""
    tgt = _phases(rng, n_rows, prof["v_cruise"])
    present, v_lead = _lead_profile(rng, n_rows, tgt)
    delay = int(round(prof["resp_delay_s"] / DT_S))
    v = np.zeros(n_rows); a = np.zeros(n_rows); d = np.full(n_rows, np.nan)
    gas = np.zeros(n_rows, bool); brake = np.zeros(n_rows, bool); coast = np.zeros(n_rows, bool)
    dist = 30.0
    for k in range(1, n_rows):
        kd = max(k - delay, 0)
        goal = tgt[kd]
        if present[kd] and np.isfinite(v_lead[kd]):
            goal = min(goal, v_lead[kd] + 0.5 * (dist - 2.0 * max(v[k - 1], 1.0) - 5.0) / 5.0)
        err = goal - v[k - 1]
        if err > 0.3:
            a_cmd = min(prof["a_max"], 0.8 * err); gas[k] = True
        elif err < -2.0:
            a_cmd = max(prof["brake_a"], 0.8 * err); brake[k] = True
        elif err < -0.3:
            a_cmd = prof["coast_a"] * min(1.0, v[k - 1] / 10.0); coast[k] = True
        else:                                   # holding speed needs throttle (no free-wheeling cruise)
            a_cmd = 0.3 * err; gas[k] = v[k - 1] > 0.5
        a[k] = a_cmd + rng.normal(0, 0.05)
        v[k] = max(v[k - 1] + a[k] * DT_S, 0.0)
        if v[k] == 0.0:
            a[k] = 0.0; gas[k] = False
        if present[k] and np.isfinite(v_lead[k]):
            dist = float(np.clip(dist + (v_lead[k] - v[k]) * DT_S, 3.0, 120.0))
            d[k] = dist
        else:
            dist = 30.0
    return dict(v=v, a=a, present=present & np.isfinite(d), v_lead=v_lead, d=d, gas=gas, brake=brake, coast=coast)


def _causal_slope(t: np.ndarray, y: np.ndarray, valid: np.ndarray, win_s: float = 1.0, min_n: int = 4) -> np.ndarray:
    """Causal least-squares slope of y over (t - win_s, t] restricted to valid samples (NaN with < min_n)."""
    out = np.full(len(t), np.nan)
    for i in range(len(t)):
        if not valid[i]:
            continue
        j0 = np.searchsorted(t, t[i] - win_s, side="right")
        sel = np.flatnonzero(valid[j0:i + 1]) + j0
        if len(sel) >= min_n:
            tt = t[sel] - t[sel].mean()
            den = float((tt * tt).sum())
            out[i] = float((tt * (y[sel] - y[sel].mean())).sum() / den) if den > 1e-9 else np.nan
    return out


def synth_drive(seed: int, n_rows: int = 1200, profile: str = "bev", das_span: tuple[int, int] | None = None) -> pd.DataFrame:
    """One synthetic drive table (all `DRIVE_COLUMNS`, n_rows samples at 10 Hz).

    das_span: (row0, row1) over which the driving automation system is engaged (longitudinal + lateral active);
    default = rows 58-83 % of the drive, so that the first 60 s window of a 120 s drive is human-only."""
    rng = np.random.default_rng(seed)
    prof = PROFILES[profile]
    sim = _simulate(rng, n_rows, prof)
    t = np.arange(n_rows) * DT_S
    v, a = sim["v"], sim["a"]
    das0, das1 = das_span if das_span is not None else (int(0.58 * n_rows), int(0.83 * n_rows))
    das = np.zeros(n_rows, bool); das[das0:das1] = True
    moving = v >= C.MOVING_V_MPS
    curv = 0.012 * np.sin(2 * np.pi * t / 40.0 + rng.uniform(0, 6.28)) * moving + rng.normal(0, 2e-4, n_rows)
    yaw = v * curv
    angle = prof["k_steer_degpm"] * curv + rng.normal(0, 0.2, n_rows)
    rate = np.gradient(angle, DT_S)
    torque_raw = 120.0 * curv / 0.012 * moving + rng.normal(0, 4.0, n_rows)
    torque_raw[das] *= 0.1                                   # the system steers, the driver mostly rests
    pressed_mask = moving & ~das
    q = np.quantile(np.abs(torque_raw[pressed_mask]), C.TORQUE_NORM_Q) if pressed_mask.sum() >= C.TORQUE_NORM_MIN_SAMPLES else np.nan
    scale = float(q) if np.isfinite(q) and q > 0 else 1.0
    steer_pressed = (np.abs(torque_raw) > 0.5 * scale) & ~das
    lead_valid = sim["present"]
    d = np.where(lead_valid, sim["d"], np.nan)
    v_lead = np.where(lead_valid, sim["v_lead"], np.nan)
    v_rel = v_lead - v
    with np.errstate(divide="ignore", invalid="ignore"):
        thw = np.where(lead_valid & (v > C.THW_V_MIN_MPS), d / v, np.nan)
        closing = -v_rel
        ttc = np.where(lead_valid & (closing > C.TTC_CLOSING_MIN_MPS), d / closing, np.nan)
        curv_pub = np.where(moving, yaw / np.maximum(v, 1e-6), np.nan)
    gas_pressed = sim["gas"] & ~das
    brake_pressed = sim["brake"] & ~das
    df = pd.DataFrame({
        "t_s": t, "seg": (np.arange(n_rows) // 600).astype(np.int16), "gap": np.zeros(n_rows, bool),
        "v_mps": v, "v_raw_mps": v + rng.normal(0, 0.02, n_rows), "a_mps2": a, "standstill": v < 0.05,
        "steer_angle_deg": angle, "steer_rate_degps": rate, "steer_torque_raw": torque_raw,
        "steer_torque_norm": torque_raw / scale, "steer_pressed": steer_pressed,
        "yaw_rate_radps": yaw + rng.normal(0, 1e-3, n_rows), "yaw_valid": np.ones(n_rows, bool),
        "pose_source": np.ones(n_rows, np.int8), "curv_1pm": curv_pub, "lat_acc_mps2": v * yaw,
        "gas": np.where(gas_pressed, np.clip(a / prof["a_max"], 0, 1), 0.0), "gas_pressed": gas_pressed,
        "brake": np.where(brake_pressed, np.clip(-a / 4.0, 0, 1), 0.0), "brake_pressed": brake_pressed,
        "blinker_left": np.zeros(n_rows, bool), "blinker_right": np.zeros(n_rows, bool),
        "das_enabled": das, "das_any": das, "op_lat_active": das, "op_long_active": das,
        "cruise_available": moving, "v_set_mps": np.where(das, prof["v_cruise"], np.nan),
        "lead_valid": lead_valid, "lead_d_m": d, "lead_y_m": np.where(lead_valid, rng.normal(0, 0.3, n_rows), np.nan),
        "lead_v_rel_mps": v_rel, "lead_speed_mps": v_lead,
        "lead_a_slope_mps2": _causal_slope(t, np.nan_to_num(v_lead), lead_valid),
        "lead_source": np.where(lead_valid, prof["lead_source"], 0).astype(np.int8),
        "lead_prob": np.where(lead_valid, 0.9, np.nan), "thw_s": thw, "ttc_s": ttc,
        "lead2_valid": np.zeros(n_rows, bool), "lead2_d_m": np.nan, "lead2_v_rel_mps": np.nan,
        "qc_lead_age_s": np.where(lead_valid, rng.uniform(0, 0.25, n_rows), np.nan),
        "qc_yaw_age_s": rng.uniform(0, 0.2, n_rows),
    })
    for c in df.columns:                                     # 4 decimals: compresses well, far below sensor resolution
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(4)
    df.attrs["steer_torque_scale"] = scale
    return df


def _window_rows(df: pd.DataFrame, drive: dict, n_win: int = C.WINDOW_SAMPLES) -> list[dict]:
    """Minimal `WINDOWS_INDEX_COLUMNS` rows (eligibility + a few stats) for consecutive non-overlapping windows."""
    rows = []
    prev_das_end = -10 ** 9
    for w in range(len(df) // n_win):
        r0, r1 = w * n_win, (w + 1) * n_win
        s = df.iloc[r0:r1]
        das_any = s["das_any"].to_numpy()
        moving = (s["v_mps"] >= C.MOVING_V_MPS).to_numpy()
        lead = s["lead_valid"].to_numpy()
        thw = s["thw_s"].to_numpy()
        human_only = not das_any.any()
        strict = human_only and (r0 - prev_das_end) * DT_S >= C.HUMAN_GUARD_S
        if das_any.any():
            prev_das_end = r0 + int(np.flatnonzero(das_any).max()) + 1
        kind = "human" if human_only else ("das" if das_any.all() else "mixed")
        rows.append(dict(
            window_id=f"{drive['drive_id']}_w{w}", drive_id=drive["drive_id"], vehicle_id=drive["vehicle_id"],
            group_id=drive["group_id"], w_idx=w, row0=r0, row1=r1, t0_s=r0 * DT_S, span_s=(n_win - 1) * DT_S,
            eligible=bool(moving.mean() >= C.WINDOW_MIN_MOVING_FRAC), inelig_reason="" if moving.mean() >= C.WINDOW_MIN_MOVING_FRAC else "moving_frac",
            kind=kind, das_frac=float(s["das_enabled"].mean()), das_any_frac=float(das_any.mean()),
            op_long_frac=float(s["op_long_active"].mean()), op_lat_frac=float(s["op_lat_active"].mean()),
            moving_frac=float(moving.mean()), standstill_frac=float(s["standstill"].mean()), gap_count=0,
            v_mean=float(s["v_mps"].mean()), v_max=float(s["v_mps"].max()), v_std=float(s["v_mps"].std()),
            a_min=float(s["a_mps2"].min()), a_max=float(s["a_mps2"].max()), a_std=float(s["a_mps2"].std()),
            n_accel_ep=0, n_decel_ep=0, lead_frac=float(lead.mean()),
            cf_frac=float((thw[lead] <= C.THW_CF_S).mean()) if lead.any() else np.nan,
            thw_med=float(np.nanmedian(thw)) if np.isfinite(thw).any() else np.nan,
            lead_speed_mean=float(np.nanmean(s["lead_speed_mps"])) if lead.any() else np.nan, n_lead_accel_events=0,
            yaw_valid_frac=1.0, yaw_abs_mean=float(np.abs(s["yaw_rate_radps"]).mean()),
            curv_abs_mean=float(np.nanmean(np.abs(s["curv_1pm"]))) if np.isfinite(s["curv_1pm"]).any() else np.nan,
            steer_pressed_frac=float(s["steer_pressed"].mean()), gas_pressed_frac=float(s["gas_pressed"].mean()),
            brake_pressed_frac=float(s["brake_pressed"].mean()), human_only=human_only, human_only_strict=strict,
            size_class=drive["size_class"], powertrain=drive["powertrain"], powertrain_raw=drive["powertrain_raw"],
            platform=drive["platform"], brand=drive["brand"], model=drive["model"], body=drive["body"],
            size_code=drive["size_code"], footprint_m2=drive["footprint_m2"], split_device=drive["split_device"],
            split_platform=drive["split_platform"], task_size_eligible=True, task_bev_ice_eligible=True, low_support_cell=False))
    return rows


def write_sample(out_dir: Path, n_vehicles: int = 2, n_drives: int = 2, rows_per_drive: int = 1200, seed: int = 0) -> dict:
    """Write a self-contained miniature release (data/, index/, splits/, README.md) and return counts.

    Vehicle i uses profile 'bev' for even i and 'ice' for odd i; odd vehicles are the test split of both protocols
    so that the starter kit can be exercised end to end on the sample."""
    out = Path(out_dir)
    drives, windows = [], []
    for vi in range(n_vehicles):
        profile = "bev" if vi % 2 == 0 else "ice"
        meta = PLATFORM_META[profile]
        vehicle_id = C.VEHICLE_FMT.format(vi + 1)
        group_id = C.GROUP_FMT.format(vi + 1)
        split = "train" if vi % 2 == 0 else "test"
        for di in range(n_drives):
            drive_id = C.DRIVE_FMT.format(vi * n_drives + di + 1)
            df = synth_drive(seed * 1000 + vi * 100 + di, rows_per_drive, profile)
            rel = f"data/{vehicle_id}/{drive_id}.parquet"
            S.write_public_parquet(df, out / rel, S.DRIVE_COLUMNS, "drive")
            fp = meta["wheelbase_mm"] / 1000.0 * (meta["track_front_mm"] + meta["track_rear_mm"]) / 2000.0
            row = dict(vehicle_id=vehicle_id, drive_id=drive_id, group_id=group_id, file=rel, n_rows=len(df),
                       duration_s=len(df) * DT_S, n_segments=int(df["seg"].max()) + 1, n_seg_rlog_dec=0, n_gaps=0, max_gap_s=0.0,
                       brand=meta["brand"], model=meta["model"], platform=meta["platform"], body=meta["body"], size_code=meta["size_code"],
                       wheelbase_mm=meta["wheelbase_mm"], track_front_mm=meta["track_front_mm"], track_rear_mm=meta["track_rear_mm"],
                       footprint_m2=fp, curb_mass_kg=meta["curb_mass_kg"], size_class=meta["size_class"], powertrain=meta["powertrain"],
                       powertrain_raw=meta["powertrain"], label_source="synthetic", label_note=None, region="other_unknown",
                       software_era="other", device_type="synthetic", pose_source="livePose", radar_unavailable=meta["radar_unavailable"],
                       op_long=True, gas_alive=True, brake_alive=True, torque_alive=True, has_pose=True,
                       moving_s=float((df["v_mps"] >= C.MOVING_V_MPS).sum() * DT_S), das_s=float(df["das_enabled"].sum() * DT_S),
                       das_any_s=float(df["das_any"].sum() * DT_S), human_s=float((~df["das_any"]).sum() * DT_S),
                       lead_valid_frac=float(df["lead_valid"].mean()),
                       cf_frac=float((df.loc[df["lead_valid"], "thw_s"] <= C.THW_CF_S).mean()) if df["lead_valid"].any() else np.nan,
                       yaw_valid_frac=1.0, steer_torque_scale=df.attrs["steer_torque_scale"], split_device=split, split_platform=split,
                       task_size_eligible=True, task_bev_ice_eligible=True, low_support_cell=False)
            drives.append(row)
            windows.extend(_window_rows(df, row))
    drives_df = pd.DataFrame(drives)
    windows_df = pd.DataFrame(windows)
    S.write_public_parquet(drives_df, out / "index" / "drives.parquet", S.DRIVES_INDEX_COLUMNS, "drives")
    S.write_public_parquet(windows_df, out / "index" / "windows.parquet", S.WINDOWS_INDEX_COLUMNS, "windows")
    (out / "splits").mkdir(parents=True, exist_ok=True)
    for split_col, suffix in (("split_platform", ""), ("split_device", "_device")):    # S8 naming per split scheme
        test_w = windows_df[(windows_df[split_col] == "test") & windows_df.eligible]
        for task in ("size", "bev_ice"):
            test_w[["window_id"]].to_parquet(out / "splits" / f"eval_balanced_{task}_test{suffix}.parquet", index=False)
    (out / "splits" / "splits.json").write_text(json.dumps({"version": "synthetic", "seed": seed, "cells": [], "groups": {
        g: {"split_device": s, "split_platform": s} for g, s in zip(drives_df.group_id, drives_df.split_platform)}}, indent=1))
    (out / "README.md").write_text(
        "# Synthetic sample\n\nThese tables are generated by `vtt/synth_sample.py` from a toy car-following simulation. "
        "They contain no real measurements and no real vehicle; they exist so that the starter kit and the tests run "
        "without downloading the dataset. The layout mirrors the release: `data/<vehicle_id>/<drive_id>.parquet`, "
        "`index/drives.parquet`, `index/windows.parquet`, `splits/`.\n")
    return {"n_vehicles": n_vehicles, "n_drives": len(drives_df), "n_windows": len(windows_df),
            "bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file())}
