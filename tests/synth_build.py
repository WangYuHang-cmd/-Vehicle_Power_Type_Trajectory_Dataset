"""Synthetic build/ tree for the S9 tests: fictitious platforms, deterministic signals, nothing from real data.

make_build(root) writes data/<vehicle>/<drive>.parquet (vtt.schema.DRIVE_COLUMNS), build/index/drives.parquet,
build/index/windows.parquet and build/splits/* exactly as the upstream stages would, so that vtt.package and the gate
can be exercised end to end. Units follow vtt.schema (m/s, m/s^2, deg, rad/s, s). Identifiers are pseudonyms only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vtt import config as C
from vtt import schema as S

DT = C.NOMINAL_DT_S
PLATFORMS = [
    dict(platform="SYNTH_A_GEN1", brand="SynthBrandA", model="Model-A", body="suv", size_code="E", wheelbase_mm=2950.0,
         track_front_mm=1650.0, track_rear_mm=1660.0, curb_mass_kg=2200.0, size_class="LARGE", powertrain="BEV",
         powertrain_raw="BEV", split_platform="train", radar_unavailable=True),
    dict(platform="SYNTH_B_GEN2", brand="SynthBrandB", model="Model-B", body="sedan", size_code="D", wheelbase_mm=2700.0,
         track_front_mm=1560.0, track_rear_mm=1570.0, curb_mass_kg=1450.0, size_class="MID", powertrain="ICE",
         powertrain_raw="ICE", split_platform="test", radar_unavailable=False),
    dict(platform="SYNTH_C_GEN1", brand="SynthBrandC", model="Model-C", body="hatchback", size_code="B", wheelbase_mm=2500.0,
         track_front_mm=1480.0, track_rear_mm=1490.0, curb_mass_kg=1250.0, size_class="SMALL", powertrain="HEV",
         powertrain_raw="HEV", split_platform="validation", radar_unavailable=True),
    dict(platform="SYNTH_D_GEN3", brand="SynthBrandB", model="Model-D", body="suv", size_code="C", wheelbase_mm=None,
         track_front_mm=None, track_rear_mm=None, curb_mass_kg=None, size_class=None, powertrain=None,
         powertrain_raw="MIXED", split_platform="train", radar_unavailable=False),
]
DEVICE_SPLITS = ("train", "validation", "test", "train")


def synth_drive(seed: int, n_rows: int, das_span: tuple[int, int] | None) -> pd.DataFrame:
    """One drive table with smooth deterministic signals; the vehicle moves throughout (every full window eligible)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows) * DT
    v = 12.0 + 6.0 * np.sin(2 * np.pi * t / 120.0) + rng.normal(0, 0.05, n_rows)
    a = np.gradient(v, DT)
    das = np.zeros(n_rows, bool)
    if das_span is not None:
        das[das_span[0]:das_span[1]] = True
    curv = 0.01 * np.sin(2 * np.pi * t / 45.0)
    yaw = v * curv
    angle = 900.0 * curv + rng.normal(0, 0.1, n_rows)
    torque = 80.0 * np.sin(2 * np.pi * t / 45.0) * (~das)
    scale = float(np.quantile(np.abs(torque[torque != 0]), C.TORQUE_NORM_Q)) if (torque != 0).sum() > 50 else 1.0
    lead = np.zeros(n_rows, bool)
    lead[200:900] = True
    d = np.where(lead, 30.0 + 5.0 * np.sin(2 * np.pi * t / 60.0), np.nan)
    v_rel = np.where(lead, 0.5 * np.cos(2 * np.pi * t / 60.0), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        thw = np.where(lead & (v > C.THW_V_MIN_MPS), d / v, np.nan)
        closing = -v_rel
        ttc = np.where(lead & (closing > C.TTC_CLOSING_MIN_MPS), d / closing, np.nan)
    gas_pressed = (a > 0.2) & ~das
    brake_pressed = (a < -0.2) & ~das
    return pd.DataFrame({
        "t_s": t, "seg": (np.arange(n_rows) // 600).astype(np.int16), "gap": np.zeros(n_rows, bool),
        "v_mps": v, "v_raw_mps": v + 0.01, "a_mps2": a, "standstill": np.zeros(n_rows, bool),
        "steer_angle_deg": angle, "steer_rate_degps": np.gradient(angle, DT), "steer_torque_raw": torque,
        "steer_torque_norm": torque / scale, "steer_pressed": np.abs(torque) > 20.0,
        "yaw_rate_radps": yaw, "yaw_valid": np.ones(n_rows, bool), "pose_source": np.ones(n_rows, np.int8),
        "curv_1pm": curv, "lat_acc_mps2": v * yaw, "gas": np.where(gas_pressed, 0.3, 0.0), "gas_pressed": gas_pressed,
        "brake": np.where(brake_pressed, 0.2, 0.0), "brake_pressed": brake_pressed,
        "blinker_left": np.zeros(n_rows, bool), "blinker_right": np.zeros(n_rows, bool),
        "das_enabled": das, "das_any": das, "op_lat_active": das, "op_long_active": das, "cruise_available": np.ones(n_rows, bool),
        "v_set_mps": np.where(das, 20.0, np.nan), "lead_valid": lead, "lead_d_m": d, "lead_y_m": np.where(lead, 0.1, np.nan),
        "lead_v_rel_mps": v_rel, "lead_speed_mps": v + v_rel, "lead_a_slope_mps2": np.where(lead, 0.05, np.nan),
        "lead_source": np.where(lead, 1, 0).astype(np.int8), "lead_prob": np.where(lead, 0.9, np.nan), "thw_s": thw, "ttc_s": ttc,
        "lead2_valid": np.zeros(n_rows, bool), "lead2_d_m": np.nan, "lead2_v_rel_mps": np.nan,
        "qc_lead_age_s": np.where(lead, 0.1, np.nan), "qc_yaw_age_s": np.full(n_rows, 0.1),
    }), scale


def _windows(df: pd.DataFrame, drive: dict) -> list[dict]:
    rows = []
    n = C.WINDOW_SAMPLES
    n_win = int(np.ceil(len(df) / n))
    for w in range(n_win):
        r0, r1 = w * n, min((w + 1) * n, len(df))
        s = df.iloc[r0:r1]
        full = (r1 - r0) == n
        das_any = s["das_any"].to_numpy()
        human_only = not das_any.any()
        before = df["das_any"].iloc[max(0, r0 - int(C.HUMAN_GUARD_S / DT)):r0].to_numpy()
        moving = (s["v_mps"] >= C.MOVING_V_MPS).to_numpy()
        lead = s["lead_valid"].to_numpy()
        rows.append(dict(
            window_id=f"{drive['drive_id']}_w{w}", drive_id=drive["drive_id"], vehicle_id=drive["vehicle_id"], group_id=drive["group_id"],
            w_idx=w, row0=r0, row1=r1, t0_s=r0 * DT, span_s=(r1 - r0 - 1) * DT, eligible=full, inelig_reason=None if full else "short",
            kind="human" if human_only else ("das" if das_any.all() else "mixed"),
            das_frac=float(s["das_enabled"].mean()), das_any_frac=float(das_any.mean()), op_long_frac=float(s["op_long_active"].mean()),
            op_lat_frac=float(s["op_lat_active"].mean()), moving_frac=float(moving.mean()), standstill_frac=0.0, gap_count=0,
            v_mean=float(s["v_mps"].mean()), v_max=float(s["v_mps"].max()), v_std=float(s["v_mps"].std()),
            a_min=float(s["a_mps2"].min()), a_max=float(s["a_mps2"].max()), a_std=float(s["a_mps2"].std()), n_accel_ep=1, n_decel_ep=1,
            lead_frac=float(lead.mean()), cf_frac=float((s["thw_s"][lead] <= C.THW_CF_S).mean()) if lead.any() else np.nan,
            thw_med=float(np.nanmedian(s["thw_s"])) if lead.any() else np.nan,
            lead_speed_mean=float(np.nanmean(s["lead_speed_mps"])) if lead.any() else np.nan, n_lead_accel_events=0,
            yaw_valid_frac=1.0, yaw_abs_mean=float(np.abs(s["yaw_rate_radps"]).mean()), curv_abs_mean=float(np.abs(s["curv_1pm"]).mean()),
            steer_pressed_frac=float(s["steer_pressed"].mean()), gas_pressed_frac=float(s["gas_pressed"].mean()),
            brake_pressed_frac=float(s["brake_pressed"].mean()), human_only=human_only, human_only_strict=human_only and not before.any(),
            size_class=drive["size_class"], powertrain=drive["powertrain"], powertrain_raw=drive["powertrain_raw"], platform=drive["platform"],
            brand=drive["brand"], model=drive["model"], body=drive["body"], size_code=drive["size_code"], footprint_m2=drive["footprint_m2"],
            split_device=drive["split_device"], split_platform=drive["split_platform"], task_size_eligible=drive["task_size_eligible"],
            task_bev_ice_eligible=drive["task_bev_ice_eligible"], low_support_cell=False))
    return rows


def make_build(root: Path, drives_per_vehicle: int = 2, n_rows: int = 1300, seed: int = 0) -> dict:
    """Write a complete synthetic build tree under `root` (one vehicle per fictitious platform)."""
    root = Path(root)
    drives, windows = [], []
    k = 0
    for vi, meta in enumerate(PLATFORMS):
        vehicle_id = C.VEHICLE_FMT.format(vi + 1)
        group_id = C.GROUP_FMT.format(vi + 1)
        for di in range(drives_per_vehicle):
            k += 1
            drive_id = C.DRIVE_FMT.format(k)
            das_span = (650, 900) if di % 2 == 0 else None
            df, scale = synth_drive(seed * 1000 + k, n_rows, das_span)
            rel = f"data/{vehicle_id}/{drive_id}.parquet"
            S.write_public_parquet(df, root / rel, S.DRIVE_COLUMNS, "drive")
            fp = (meta["wheelbase_mm"] / 1000.0 * (meta["track_front_mm"] + meta["track_rear_mm"]) / 2000.0) if meta["wheelbase_mm"] else None
            row = dict(vehicle_id=vehicle_id, drive_id=drive_id, group_id=group_id, file=rel, n_rows=len(df), duration_s=len(df) * DT,
                       n_segments=int(df["seg"].max()) + 1, n_seg_rlog_dec=0, n_gaps=0, max_gap_s=0.0, brand=meta["brand"], model=meta["model"],
                       platform=meta["platform"], body=meta["body"], size_code=meta["size_code"], wheelbase_mm=meta["wheelbase_mm"],
                       track_front_mm=meta["track_front_mm"], track_rear_mm=meta["track_rear_mm"], footprint_m2=fp, curb_mass_kg=meta["curb_mass_kg"],
                       size_class=meta["size_class"], powertrain=meta["powertrain"], powertrain_raw=meta["powertrain_raw"], label_source="synthetic",
                       label_note=None, region="other_unknown", software_era="other", device_type="synthetic", pose_source="livePose",
                       radar_unavailable=meta["radar_unavailable"], op_long=True, gas_alive=True, brake_alive=vi % 2 == 0, torque_alive=True,
                       has_pose=True, moving_s=float((df["v_mps"] >= C.MOVING_V_MPS).sum() * DT), das_s=float(df["das_enabled"].sum() * DT),
                       das_any_s=float(df["das_any"].sum() * DT), human_s=float((~df["das_any"]).sum() * DT),
                       lead_valid_frac=float(df["lead_valid"].mean()), cf_frac=0.9, yaw_valid_frac=1.0, steer_torque_scale=scale,
                       split_device=DEVICE_SPLITS[vi], split_platform=meta["split_platform"],
                       task_size_eligible=meta["size_class"] is not None, task_bev_ice_eligible=meta["powertrain"] in C.MAIN_POWERTRAIN_TASK,
                       low_support_cell=False)
            drives.append(row)
            windows.extend(_windows(df, row))
    drives_df = pd.DataFrame(drives)
    windows_df = pd.DataFrame(windows)
    S.write_public_parquet(drives_df, root / "index" / "drives.parquet", S.DRIVES_INDEX_COLUMNS, "drives")
    S.write_public_parquet(windows_df, root / "index" / "windows.parquet", S.WINDOWS_INDEX_COLUMNS, "windows")
    (root / "splits").mkdir(parents=True, exist_ok=True)
    el = windows_df[windows_df.eligible]
    for task, col in (("size", "task_size_eligible"), ("bev_ice", "task_bev_ice_eligible")):
        for split in ("validation", "test"):
            sel = el[el[col] & (el.split_platform == split)][["window_id"]]
            sel.to_parquet(root / "splits" / f"eval_balanced_{task}_{split}.parquet", index=False)   # pandas metadata on purpose
        tr = el[el[col] & (el.split_platform == "train")][["window_id"]].copy()
        tr["weight"] = 1.0
        tr.to_parquet(root / "splits" / f"train_uniform_{task}.parquet", index=False)
    groups = {g: {"split_device": d, "split_platform": p} for g, d, p in
              drives_df[["group_id", "split_device", "split_platform"]].drop_duplicates().itertuples(index=False)}
    (root / "splits" / "splits.json").write_text(json.dumps({
        "version": C.CONFIG_VERSION, "seed": C.SPLIT_SEED, "config": {"fracs": C.SPLIT_FRACS}, "cells": [], "groups": groups,
        "heldout_platforms": [m["platform"] for m in PLATFORMS if m["split_platform"] != "train"],
        "class_weights": {"size": {}, "bev_ice": {}}, "sha256_size_classes": None}, indent=1, sort_keys=True))
    return {"n_drives": len(drives_df), "n_windows": len(windows_df), "n_eligible": int(windows_df.eligible.sum()),
            "n_shard_expected": int((el.task_size_eligible | el.task_bev_ice_eligible).sum())}
