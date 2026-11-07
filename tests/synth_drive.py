"""Synthetic public drive tables for the S6 / S7 tests (no real identifiers anywhere: pseudonyms are the public
vehicle_/drive_/group_ formats, fake route uids are 'f' * 10 + '%02x').

make_drive builds a frame with every schema.DRIVE_COLUMNS column from a speed profile and optional flags;
write_drive / drives_index_row / write_drives_index put a tiny build tree under a temp data root.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vtt import config as C
from vtt import schema as S

DT = C.NOMINAL_DT_S


def fake_route_uid(k: int) -> str:
    return "f" * 10 + f"{k:02x}"


def ramp_speed(n: int, segments: list[tuple[float, float]]) -> np.ndarray:
    """Piecewise-linear speed: segments = [(duration_s, target_speed), ...] starting from 0 m/s, held at the last
    target afterwards; total length n samples."""
    v = np.zeros(n)
    t = np.arange(n) * DT
    cur_t, cur_v = 0.0, 0.0
    for dur, target in segments:
        m = (t >= cur_t) & (t < cur_t + dur)
        v[m] = cur_v + (target - cur_v) * (t[m] - cur_t) / dur
        cur_t, cur_v = cur_t + dur, target
    v[t >= cur_t] = cur_v
    return v


def make_drive(n: int, v: np.ndarray | None = None, das_any: np.ndarray | None = None, das_enabled: np.ndarray | None = None,
               gap_rows=(), lead_valid: np.ndarray | None = None, lead_a_slope: np.ndarray | None = None,
               lead_d: np.ndarray | None = None, yaw: np.ndarray | None = None, lead_source: int = 1,
               has_pose: bool = True, gas_alive: bool = True, brake_alive: bool = True) -> pd.DataFrame:
    idx = np.arange(n)
    v = np.full(n, 15.0) if v is None else np.asarray(v, float)
    a = np.gradient(v, DT) if n > 1 else np.zeros(n)
    das_any = np.zeros(n, bool) if das_any is None else np.asarray(das_any, bool)
    das_enabled = das_any.copy() if das_enabled is None else np.asarray(das_enabled, bool)
    gap = np.zeros(n, bool)
    gap[list(gap_rows)] = True
    lead_valid = np.zeros(n, bool) if lead_valid is None else np.asarray(lead_valid, bool)
    lead_d = np.where(lead_valid, 30.0, np.nan) if lead_d is None else np.asarray(lead_d, float)
    lead_a_slope = np.where(lead_valid, 0.0, np.nan) if lead_a_slope is None else np.asarray(lead_a_slope, float)
    yaw = np.full(n, 0.01) if yaw is None else np.asarray(yaw, float)
    if not has_pose:
        yaw = np.full(n, np.nan)
    moving = v >= C.MOVING_V_MPS
    with np.errstate(divide="ignore", invalid="ignore"):
        thw = np.where(lead_valid & (v > C.THW_V_MIN_MPS), lead_d / v, np.nan)
        curv = np.where(moving & np.isfinite(yaw), yaw / v, np.nan)
    df = pd.DataFrame({
        "t_s": idx * DT, "seg": idx // 600, "gap": gap, "v_mps": v, "v_raw_mps": v, "a_mps2": a, "standstill": v < 0.05,
        "steer_angle_deg": np.zeros(n), "steer_rate_degps": np.zeros(n), "steer_torque_raw": np.zeros(n),
        "steer_torque_norm": np.zeros(n), "steer_pressed": np.zeros(n, bool),
        "yaw_rate_radps": yaw, "yaw_valid": np.isfinite(yaw), "pose_source": np.full(n, 1 if has_pose else 0),
        "curv_1pm": curv, "lat_acc_mps2": np.where(np.isfinite(yaw), v * yaw, np.nan),
        "gas": np.where(gas_alive, np.clip(a, 0, None) / 3.0, np.nan), "gas_pressed": a > 0.05,
        "brake": np.where(brake_alive, np.clip(-a, 0, None) / 3.0, np.nan), "brake_pressed": a < -0.05,
        "blinker_left": np.zeros(n, bool), "blinker_right": np.zeros(n, bool),
        "das_enabled": das_enabled, "das_any": das_any, "op_lat_active": das_any, "op_long_active": das_any,
        "cruise_available": np.ones(n, bool), "v_set_mps": np.full(n, np.nan),
        "lead_valid": lead_valid, "lead_d_m": lead_d, "lead_y_m": np.where(lead_valid, 0.0, np.nan),
        "lead_v_rel_mps": np.where(lead_valid, 0.0, np.nan), "lead_speed_mps": np.where(lead_valid, v, np.nan),
        "lead_a_slope_mps2": lead_a_slope, "lead_source": np.where(lead_valid, lead_source, 0),
        "lead_prob": np.where(lead_valid, 0.9, np.nan), "thw_s": thw, "ttc_s": np.full(n, np.nan),
        "lead2_valid": np.zeros(n, bool), "lead2_d_m": np.full(n, np.nan), "lead2_v_rel_mps": np.full(n, np.nan),
        "qc_lead_age_s": np.where(lead_valid, 0.1, np.nan), "qc_yaw_age_s": np.where(np.isfinite(yaw), 0.1, np.nan),
    })
    return df[S.names(S.DRIVE_COLUMNS)]


def write_drive(df: pd.DataFrame, data_root: Path, vehicle_id: str, drive_id: str) -> Path:
    path = data_root / "build" / "data" / vehicle_id / f"{drive_id}.parquet"
    return S.write_public_parquet(df, path, S.DRIVE_COLUMNS, "drive")


def drives_index_row(df: pd.DataFrame, vehicle_id: str, drive_id: str, group_id: str, platform: str = "PLATFORM_A",
                     powertrain: str | None = "ICE", powertrain_raw: str = "ICE", size_class: str | None = "MID",
                     brand: str = "BrandA", model: str = "ModelA", region: str = "north_america", radar_unavailable: bool = False,
                     n_segments: int | None = None, label_note: str | None = None, **extra) -> dict:
    n = len(df)
    v = df["v_mps"].to_numpy(float)
    gap = df["gap"].to_numpy(bool)
    lead = df["lead_valid"].to_numpy(bool)
    thw = df["thw_s"].to_numpy(float)
    row = dict(
        vehicle_id=vehicle_id, drive_id=drive_id, group_id=group_id, file=f"data/{vehicle_id}/{drive_id}.parquet",
        n_rows=n, duration_s=n * DT, n_segments=int(df["seg"].max()) + 1 if n_segments is None else n_segments,
        n_seg_rlog_dec=0, n_gaps=int(gap.sum()), max_gap_s=0.0 if not gap.any() else 1.0,
        brand=brand, model=model, platform=platform, body="sedan", size_code="D", wheelbase_mm=2700.0, track_front_mm=1550.0,
        track_rear_mm=1550.0, footprint_m2=4.185, curb_mass_kg=1500.0, size_class=size_class, powertrain=powertrain,
        powertrain_raw=powertrain_raw, label_source="fingerprint", label_note=label_note, region=region, software_era="0.9",
        device_type="tici", pose_source="livePose", radar_unavailable=radar_unavailable, op_long=False,
        gas_alive=bool(np.isfinite(df["gas"]).any()), brake_alive=bool(np.isfinite(df["brake"]).any()), torque_alive=True,
        has_pose=bool(df["yaw_valid"].any()), moving_s=float((v >= C.MOVING_V_MPS).sum() * DT),
        das_s=float(df["das_enabled"].sum() * DT), das_any_s=float(df["das_any"].sum() * DT),
        human_s=float((~df["das_any"].to_numpy(bool)).sum() * DT), lead_valid_frac=float(lead.mean()) if n else 0.0,
        cf_frac=float(np.mean(thw[lead] <= C.THW_CF_S)) if lead.any() else float("nan"),
        yaw_valid_frac=float(df["yaw_valid"].mean()) if n else 0.0, steer_torque_scale=1.0,
        split_device=None, split_platform=None, task_size_eligible=size_class is not None,
        task_bev_ice_eligible=powertrain in C.MAIN_POWERTRAIN_TASK, low_support_cell=None,
    )
    row.update(extra)
    return row


def write_drives_index(rows: list[dict], data_root: Path) -> Path:
    path = data_root / "build" / "index" / "drives.parquet"
    return S.write_public_parquet(pd.DataFrame(rows), path, S.DRIVES_INDEX_COLUMNS, "drives")
