#!/usr/bin/env python3
"""MoreTorque/rlogs adapter: community openpilot rlog.bz2 -> the lab's 10 Hz drive timeline -> 20 training channels -> windows.

Runs under the openpilot venv ($PY after `source env.sh`; it has numpy / pandas / pyarrow and the cereal LogReader):

  source env.sh && cd $R
  nice -n 10 $PY -I experiments/transfer/ds_moretorque.py --smoke 20 --dataset moretorque_smoke          # 20-segment test
  nohup nice -n 10 $PY -I experiments/transfer/ds_moretorque.py --workers 6 > $D/external/moretorque/work/logs/run.log 2>&1 &
  $PY -I experiments/transfer/ds_moretorque.py --stage assemble [--steer-rate can] [--align run] [--dataset NAME]   # re-cut only

Stage `decode` (worker pool, at most 6 processes, nice inherited): one task = one DRIVE = a run of consecutive segment numbers
of one route (`<FOLDER>/<device>/<route>--<seg>--rlog.bz2`). Every rlog is read with the cereal LogReader (bz2 / zst), no
CAN decoding: the logs already carry carState at 100 Hz. The lab timeline is reproduced the way vtt/drives.py builds it from
qlogs: row k = the k-th carState sample of the qlog, i.e. every 10th carState message of the continuous 100 Hz stream
(openpilot's qlog decimation; the phase runs across segment boundaries), real sample times kept for the gap rule (dt > 0.3 s);
the other topics are brought to their qlog rates the same way (carControl every 10th = 10 Hz, liveLocationKalman every 5th
= 4 Hz, modelV2 leads every 5th = the 4 Hz of radarState in a qlog; `--native-rates` keeps 100 / 20 Hz) and put onto the
carState sample times by the lab's causal zero-order hold with the lab max ages (0.3 s controls, 0.625 s pose, 0.75 s lead).
Channels: v = carState.vEgo, a = aEgo, steer_angle = steeringAngleDeg, steer_rate = steeringRateDeg (CAN sample), torque raw =
steeringTorque, flags = gasPressed / brakePressed / steeringPressed / standstill, das_enabled = held carControl.enabled (these
logs have no controlsState / selfdriveState; carControl.enabled is the same controller flag) OR carState.cruiseState.enabled,
yaw_rate = -liveLocationKalman.angularVelocityCalibrated.z (left-positive; the lab sign POSE_YAW_SIGN = -1 applied to the
calibrated frame because angularVelocityDevice is empty in these logs), lead = modelV2.leadsV3[0] (the t = 0 vision lead:
valid when prob >= 0.5 and d > 0.5 m, d = x - 1.52 m exactly as radard turns a vision lead into radarState.leadOne.dRel,
v_rel = lead v - ego v). The per-drive 10 Hz table goes to <external>/<dataset>/work/drives10hz/<drive_id>.parquet.

Stage `assemble` (main process): per vehicle it decides which channels are alive (a pedal flag that never fires, a steering
rate that is always 0 while the wheel moves, a torque that never changes sign), then per drive: steer_torque_norm by the lab
rule (q95 of |torque| over moving & steering samples, >= 50 of them), curvature = yaw / v (v >= 1 m/s and yaw valid),
lat_acc = v x yaw, the seven lead channels through common.lead_channels, the steering rate from the CAN sample or, where the
CAN rate is dead, common.smooth_derivative of the angle (`--steer-rate auto|can|derived`), and common.cut_windows with the
lab's fixed 600-row blocks (`--align drive`, default) -> common.save_windows. Labels: folder -> platform; the Sonata from the
lab platform table (seen model), the three community-fork platforms (Mazda 3 BP, CX-50, Omoda E5) from specs_external.csv.
Pseudonyms: vehicle_id = veh_<sha1('vtt-transfer' + device)[:8]>, drive_id = drv_<sha1('vtt-transfer' + device/route/seg0)[:10]>;
no device, route or path identifier is written anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.transfer import common as T  # noqa: E402
from experiments.transfer import ccs_platforms as P  # noqa: E402
from vtt import config as C  # noqa: E402

DATASET = "moretorque"
RAW_ROOT = Path(os.environ.get("VTT_MORETORQUE_RAW", str(T.EXTERNAL_ROOT / "moretorque")))
SALT = "vtt-transfer"
EXT_SPECS = HERE / "specs_external.csv"
SEG_RE = re.compile(r"^(?P<route>.+)--(?P<seg>\d+)--rlog\.(?:bz2|zst)$")
DEVICE_RE = re.compile(r"^[0-9a-f]{16}$")

# folder name -> label resolution. 'lab' = vtt/specs_data/platform_specs.csv row (through ccs_platforms.Labeller),
# 'external' = specs_external.csv row picked by model + generation text. `fingerprint` is the carParams.carFingerprint the
# folder is expected to carry (community forks use their own spellings); `powertrain_folder` is the folder's own claim.
FOLDERS: dict[str, dict] = {
    "HYUNDAI SONATA 2020": dict(platform="HYUNDAI_SONATA", fingerprint="HYUNDAI SONATA 2020", model="Hyundai Sonata",
                                source="lab", ext_model="Hyundai Sonata", ext_gen="DN8", powertrain_folder="ICE", community=False),
    "MAZDA 3 2019": dict(platform="MAZDA_3_2019", fingerprint="MAZDA 3 2019", model="Mazda 3",
                         source="external", ext_model="Mazda 3", ext_gen="BP 2019", powertrain_folder="ICE", community=True),
    "MAZDA CX-50": dict(platform="MAZDA_CX50", fingerprint="MAZDA CX-50", model="Mazda CX-50",
                        source="external", ext_model="Mazda CX-50", ext_gen="2023", powertrain_folder="ICE", community=True),
    "OMODA_E5": dict(platform="OMODA_E5", fingerprint="OMODA_E5", model="Omoda E5",
                     source="external", ext_model="Omoda E5", ext_gen="2024", powertrain_folder="BEV", community=True),
}

# qlog decimation of the openpilot services (cereal/services.py: carState 100 Hz / 10, carControl 100 Hz / 10,
# liveLocationKalman 20 Hz / 5, radarState 20 Hz / 5 -> the modelV2 lead is decimated like radarState)
DECIM = dict(carState=10, carControl=10, liveLocationKalman=5, modelV2=5)
CTRL_MAX_AGE_S = float(C.CTRL_MAX_AGE_S)     # 0.3
POSE_MAX_AGE_S = float(C.POSE_MAX_AGE_S)     # 0.625
LEAD_MAX_AGE_S = float(C.LEAD_MAX_AGE_S)     # 0.75
POSE_YAW_SIGN = float(C.POSE_YAW_SIGN)       # -1: left-positive = -angularVelocity z
LEAD_PROB_MIN = 0.5                          # radard: a vision lead exists when prob > 0.5
RADAR_TO_CAMERA_M = 1.52                     # radard.RADAR_TO_CAMERA: vision x is measured from the camera, radarState from the bumper
LEAD_DREL_MIN_M = float(C.LEAD_VALID_DREL_MIN_M)
MIN_CAN_VALID_FRAC = 0.5
DEAD_FLAG_MIN_FIRES = int(C.DEAD_CHANNEL_MIN_PRESSED)   # 10: a flag firing fewer times over a whole vehicle is dead ...
DEAD_FLAG_MIN_ROWS = 6000                               # ... provided the vehicle has >= 10 min of 10 Hz samples (else: keep the flag)
RATE_ALIVE_MIN_FRAC = 0.005                  # CAN steering rate alive when non-zero on >= 0.5 % of the moving samples
SIGNAL_COLS = ("v_mps", "a_mps2", "steer_angle_deg", "steer_rate_can_degps", "steer_torque_raw", "gas_pressed", "brake_pressed",
               "steer_pressed", "standstill", "cruise_enabled", "cruise_available", "op_enabled", "lat_active", "long_active",
               "das_enabled", "can_valid", "yaw_rate_radps", "yaw_valid", "lead_prob", "lead_x_m", "lead_v_abs_mps",
               "lead_valid", "lead_d_m", "lead_v_rel_mps", "gas_mag", "brake_mag")

_W: dict = {}


# ----------------------------------------------------------------------------- pseudonyms, paths
def vehicle_pseudonym(device: str) -> str:
    return "veh_" + hashlib.sha1((SALT + device).encode()).hexdigest()[:8]


def drive_pseudonym(device: str, route: str, seg0: int) -> str:
    return "drv_" + hashlib.sha1((SALT + f"{device}/{route}/{seg0}").encode()).hexdigest()[:10]


def work_dir(dataset: str) -> Path:
    return T.work_dir(dataset)


# ----------------------------------------------------------------------------- segment discovery
def find_drives(root: Path, folders: list[str] | None = None) -> list[dict]:
    """[{folder, device, route, segs: [(seg, path)], seg0, seg1, vehicle_id, drive_id}], one per run of consecutive segments."""
    out = []
    for folder in sorted(FOLDERS):
        if folders and folder not in folders:
            continue
        fdir = root / folder
        if not fdir.is_dir():
            continue
        for dev_dir in sorted(p for p in fdir.iterdir() if p.is_dir()):
            device = dev_dir.name
            if not DEVICE_RE.match(device):
                continue
            by_route: dict[str, list[tuple[int, str]]] = defaultdict(list)
            for f in dev_dir.iterdir():
                m = SEG_RE.match(f.name)
                if m and f.is_file():
                    by_route[m.group("route")].append((int(m.group("seg")), str(f)))
            for route, segs in sorted(by_route.items()):
                segs = sorted(set(segs))
                run: list[tuple[int, str]] = []
                for s in segs:
                    if run and s[0] != run[-1][0] + 1:
                        out.append(_drive(folder, device, route, run))
                        run = []
                    run.append(s)
                if run:
                    out.append(_drive(folder, device, route, run))
    return out


def _drive(folder: str, device: str, route: str, run: list[tuple[int, str]]) -> dict:
    return dict(folder=folder, device=device, route=route, segs=list(run), seg0=run[0][0], seg1=run[-1][0],
                vehicle_id=vehicle_pseudonym(device), drive_id=drive_pseudonym(device, route, run[0][0]))


def public_drive_row(d: dict) -> dict:
    """The drive's index row without any identifier (folder, pseudonyms, segment numbers and count only)."""
    return dict(drive_id=d["drive_id"], vehicle_id=d["vehicle_id"], folder=d["folder"], platform=FOLDERS[d["folder"]]["platform"],
                model=FOLDERS[d["folder"]]["model"], seg0=int(d["seg0"]), seg1=int(d["seg1"]), n_segments=len(d["segs"]))


# ----------------------------------------------------------------------------- worker: one segment
def init_worker(native_rates: bool) -> None:
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from openpilot.tools.lib.logreader import LogReader
    _W["LogReader"] = LogReader
    _W["decim"] = {k: 1 for k in DECIM} if native_rates else dict(DECIM)


def _which(m):
    try:
        return m.which()
    except Exception:            # noqa: BLE001  union member unknown to this cereal
        return None


def read_segment(path: str) -> dict:
    """Full-rate arrays of the topics the adapter uses + carParams facts; raises on an unreadable file."""
    msgs = list(_W["LogReader"](path, sort_by_time=True))
    cs, cc, llk, mv, cp = [], [], [], [], None
    n_unknown = 0
    for m in msgs:
        w = _which(m)
        if w == "carState":
            s = m.carState
            cs.append((m.logMonoTime, s.vEgo, s.aEgo, s.steeringAngleDeg, s.steeringRateDeg, s.steeringTorque, s.gasPressed, s.brakePressed,
                       s.steeringPressed, s.standstill, s.cruiseState.enabled, s.cruiseState.available, s.canValid, s.gas, s.brake))
        elif w == "carControl":
            c = m.carControl
            cc.append((m.logMonoTime, c.enabled, c.latActive, c.longActive))
        elif w == "liveLocationKalman":
            k = m.liveLocationKalman
            av = k.angularVelocityCalibrated
            wz = av.value[2] if len(av.value) == 3 else float("nan")
            llk.append((m.logMonoTime, wz, av.valid, k.inputsOK))
        elif w == "modelV2":
            x = m.modelV2
            if len(x.leadsV3):
                l0 = x.leadsV3[0]
                mv.append((m.logMonoTime, l0.prob, l0.x[0] if len(l0.x) else float("nan"), l0.v[0] if len(l0.v) else float("nan")))
            else:
                mv.append((m.logMonoTime, 0.0, float("nan"), float("nan")))
        elif w == "carParams" and cp is None:
            c = m.carParams
            cp = dict(fingerprint=str(c.carFingerprint), car_name=str(c.carName), wheelbase_m=float(c.wheelbase), steer_ratio=float(c.steerRatio),
                      mass_kg=float(c.mass), radar_unavailable=bool(c.radarUnavailable), op_long=bool(c.openpilotLongitudinalControl))
        elif w is None:
            n_unknown += 1
    return dict(carState=np.asarray(cs, dtype=np.float64).reshape(-1, 15), carControl=np.asarray(cc, dtype=np.float64).reshape(-1, 4),
                liveLocationKalman=np.asarray(llk, dtype=np.float64).reshape(-1, 4), modelV2=np.asarray(mv, dtype=np.float64).reshape(-1, 4),
                carParams=cp, n_msgs=len(msgs), n_unknown=n_unknown)


def _decimate(A: np.ndarray, step: int, seen: dict, key: str) -> np.ndarray:
    """Every step-th row of the continuous stream: the phase continues from the rows already seen in this drive."""
    n = len(A)
    start = (-seen.get(key, 0)) % step
    seen[key] = seen.get(key, 0) + n
    return A[start::step] if n else A


def _hold(src_t, values, t_grid, max_age_s: float, as_bool: bool = False) -> np.ndarray:
    if len(src_t) == 0:
        return np.zeros(len(t_grid)) if as_bool else np.full(len(t_grid), np.nan)
    vals = np.asarray(values).astype(bool) if as_bool else np.asarray(values, dtype=np.float64)
    _, out = T.resample_to_10hz(np.asarray(src_t, dtype=np.float64), vals, "hold", t_grid=t_grid, max_age_s=max_age_s)
    return out


# ----------------------------------------------------------------------------- worker: one drive -> 10 Hz table
def process_drive(task: dict) -> dict:
    """task = find_drives() entry + out_dir, expected fingerprint. Writes <out_dir>/<drive_id>.parquet; returns QC (no identifiers)."""
    t_start = time.time()
    exp_fp = FOLDERS[task["folder"]]["fingerprint"]
    res = dict(drive_id=task["drive_id"], vehicle_id=task["vehicle_id"], folder=task["folder"], segments=[], n_rows=0, ok=False, error="")
    seen: dict = {}
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    seg_of_rows: list[np.ndarray] = []
    cp_first = None
    try:
        for seg, path in task["segs"]:
            q = dict(seg=int(seg), status="ok", reason="", n_msgs=0, n_carState=0, span_s=np.nan, decode_s=np.nan, fingerprint=None)
            t0 = time.time()
            try:
                r = read_segment(path)
            except Exception as ex:          # noqa: BLE001  a corrupt file must not kill the drive
                q.update(status="failed", reason=f"{type(ex).__name__}: {str(ex)[:100]}", decode_s=round(time.time() - t0, 2))
                res["segments"].append(q)
                continue
            A = r["carState"]
            q.update(n_msgs=r["n_msgs"], n_carState=int(len(A)), n_unknown=int(r["n_unknown"]), decode_s=round(time.time() - t0, 2))
            if r["carParams"] is not None:
                q["fingerprint"] = r["carParams"]["fingerprint"]
                if cp_first is None:
                    cp_first = r["carParams"]
            if len(A) < 10:
                q.update(status="skipped", reason=f"only {len(A)} carState messages")
                res["segments"].append(q)
                continue
            if q["fingerprint"] is not None and q["fingerprint"] != exp_fp:
                q.update(status="skipped", reason=f"fingerprint {q['fingerprint']!r} is not the folder's {exp_fp!r}")
                res["segments"].append(q)
                continue
            can_valid_frac = float(A[:, 12].mean())
            q["can_valid_frac"] = round(can_valid_frac, 3)
            if can_valid_frac < MIN_CAN_VALID_FRAC:
                q.update(status="skipped", reason=f"canValid only {can_valid_frac:.2f}")
                res["segments"].append(q)
                continue
            q["span_s"] = round(float((A[-1, 0] - A[0, 0]) * 1e-9), 2)
            dec = _W["decim"]
            cs10 = _decimate(A, dec["carState"], seen, "carState")
            parts["carState"].append(cs10)
            seg_of_rows.append(np.full(len(cs10), int(seg), dtype=np.int16))
            parts["carControl"].append(_decimate(r["carControl"], dec["carControl"], seen, "carControl"))
            parts["liveLocationKalman"].append(_decimate(r["liveLocationKalman"], dec["liveLocationKalman"], seen, "liveLocationKalman"))
            parts["modelV2"].append(_decimate(r["modelV2"], dec["modelV2"], seen, "modelV2"))
            # per-segment QC on the decimated carState samples
            v = cs10[:, 1]
            q.update(v_min=round(float(v.min()), 2), v_max=round(float(v.max()), 2), v_mean=round(float(v.mean()), 2),
                     moving_frac=round(float((v >= T.MOVING_V_MPS).mean()), 3), a_min=round(float(cs10[:, 2].min()), 2), a_max=round(float(cs10[:, 2].max()), 2),
                     steer_min=round(float(cs10[:, 3].min()), 1), steer_max=round(float(cs10[:, 3].max()), 1),
                     rate_nonzero=round(float((cs10[:, 4] != 0).mean()), 3), torque_nonzero=round(float((cs10[:, 5] != 0).mean()), 3),
                     torque_min=round(float(cs10[:, 5].min()), 1), torque_max=round(float(cs10[:, 5].max()), 1),
                     gas_frac=round(float(cs10[:, 6].mean()), 3), brake_frac=round(float(cs10[:, 7].mean()), 3), steer_pressed_frac=round(float(cs10[:, 8].mean()), 3),
                     standstill_frac=round(float(cs10[:, 9].mean()), 3), cruise_frac=round(float(cs10[:, 10].mean()), 3),
                     op_enabled_frac=round(float(r["carControl"][:, 1].mean()), 3) if len(r["carControl"]) else np.nan,
                     lead_prob_frac=round(float((r["modelV2"][:, 1] >= LEAD_PROB_MIN).mean()), 3) if len(r["modelV2"]) else np.nan,
                     n_llk=int(len(r["liveLocationKalman"])), n_modelV2=int(len(r["modelV2"])), n_carControl=int(len(r["carControl"])))
            res["segments"].append(q)
        if not parts["carState"]:
            res["error"] = "no decodable segment"
            return res
        CS = np.concatenate(parts["carState"])
        order = np.argsort(CS[:, 0], kind="stable")
        CS = CS[order]
        seg_col = np.concatenate(seg_of_rows)[order]
        t0_ns = CS[0, 0]
        t = (CS[:, 0] - t0_ns) * 1e-9
        n = len(t)
        v = CS[:, 1]
        df = dict(t_s=t, seg=seg_col, v_mps=v, a_mps2=CS[:, 2], steer_angle_deg=CS[:, 3], steer_rate_can_degps=CS[:, 4], steer_torque_raw=CS[:, 5],
                  gas_pressed=CS[:, 6], brake_pressed=CS[:, 7], steer_pressed=CS[:, 8], standstill=CS[:, 9], cruise_enabled=CS[:, 10],
                  cruise_available=CS[:, 11], can_valid=CS[:, 12], gas_mag=CS[:, 13], brake_mag=CS[:, 14])
        # controller flag (carControl.enabled = controlsState.enabled) held with the lab control max age
        CC = np.concatenate(parts["carControl"]) if parts["carControl"] else np.zeros((0, 4))
        if len(CC):
            CC = CC[np.argsort(CC[:, 0], kind="stable")]
            tc = (CC[:, 0] - t0_ns) * 1e-9
            df["op_enabled"] = _hold(tc, CC[:, 1] > 0.5, t, CTRL_MAX_AGE_S, as_bool=True)
            df["lat_active"] = _hold(tc, CC[:, 2] > 0.5, t, CTRL_MAX_AGE_S, as_bool=True)
            df["long_active"] = _hold(tc, CC[:, 3] > 0.5, t, CTRL_MAX_AGE_S, as_bool=True)
        else:
            df["op_enabled"] = df["lat_active"] = df["long_active"] = np.zeros(n)
        df["das_enabled"] = ((df["op_enabled"] > 0.5) | (df["cruise_enabled"] > 0.5)).astype(np.float64)
        # yaw from the localizer, left-positive, held with the lab pose max age
        LK = np.concatenate(parts["liveLocationKalman"]) if parts["liveLocationKalman"] else np.zeros((0, 4))
        if len(LK):
            LK = LK[np.argsort(LK[:, 0], kind="stable")]
            tk = (LK[:, 0] - t0_ns) * 1e-9
            yaw = _hold(tk, POSE_YAW_SIGN * LK[:, 1], t, POSE_MAX_AGE_S)
            valid = _hold(tk, (LK[:, 2] > 0.5) & (LK[:, 3] > 0.5) & np.isfinite(LK[:, 1]), t, POSE_MAX_AGE_S, as_bool=True)
            df["yaw_rate_radps"] = yaw
            df["yaw_valid"] = ((valid > 0.5) & np.isfinite(yaw)).astype(np.float64)
        else:
            df["yaw_rate_radps"] = np.full(n, np.nan)
            df["yaw_valid"] = np.zeros(n)
        # vision lead (modelV2.leadsV3[0]) held with the lab lead max age, converted like radard's vision lead
        MV = np.concatenate(parts["modelV2"]) if parts["modelV2"] else np.zeros((0, 4))
        if len(MV):
            MV = MV[np.argsort(MV[:, 0], kind="stable")]
            tm = (MV[:, 0] - t0_ns) * 1e-9
            prob = _hold(tm, MV[:, 1], t, LEAD_MAX_AGE_S)
            x0 = _hold(tm, MV[:, 2], t, LEAD_MAX_AGE_S)
            v0 = _hold(tm, MV[:, 3], t, LEAD_MAX_AGE_S)
            d = x0 - RADAR_TO_CAMERA_M
            valid = np.isfinite(prob) & (prob >= LEAD_PROB_MIN) & np.isfinite(d) & (d > LEAD_DREL_MIN_M) & np.isfinite(v0)
            df["lead_prob"] = prob
            df["lead_x_m"] = x0
            df["lead_v_abs_mps"] = v0
            df["lead_valid"] = valid.astype(np.float64)
            df["lead_d_m"] = np.where(valid, d, np.nan)
            df["lead_v_rel_mps"] = np.where(valid, v0 - v, np.nan)
        else:
            for k in ("lead_prob", "lead_x_m", "lead_v_abs_mps", "lead_d_m", "lead_v_rel_mps"):
                df[k] = np.full(n, np.nan)
            df["lead_valid"] = np.zeros(n)
        frame = pd.DataFrame(df)
        for c in SIGNAL_COLS:
            frame[c] = frame[c].astype(np.float32)
        frame["seg"] = frame["seg"].astype(np.int16)
        out_dir = Path(task["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f".{task['drive_id']}.tmp.parquet"
        frame.to_parquet(tmp, index=False)
        tmp.replace(out_dir / f"{task['drive_id']}.parquet")
        dt = np.diff(t)
        res.update(ok=True, n_rows=int(n), n_gaps=int((dt > T.GAP_S).sum()), max_gap_s=round(float(dt.max()), 3) if len(dt) else 0.0,
                   dt_med_s=round(float(np.median(dt)), 4) if len(dt) else np.nan, span_s=round(float(t[-1]), 1),
                   n_llk=int(len(LK)), n_modelV2=int(len(MV)), n_carControl=int(len(CC)), car_params=cp_first,
                   yaw_valid_frac=round(float(frame["yaw_valid"].mean()), 3), lead_valid_frac=round(float(frame["lead_valid"].mean()), 3),
                   das_frac=round(float(frame["das_enabled"].mean()), 3), decode_s=round(time.time() - t_start, 1))
    except Exception:            # noqa: BLE001
        res["error"] = "drive exception: " + traceback.format_exc().splitlines()[-1][:200]
    return res


# ----------------------------------------------------------------------------- labels
def load_labels() -> dict[str, dict]:
    """folder -> dict(platform, model, footprint_m2, size_class, powertrain, borderline, stratum, label_source, spec_note, brand_seen)."""
    lab = P.Labeller()
    ext = pd.read_csv(EXT_SPECS, dtype=str, keep_default_na=False)
    out = {}
    for folder, f in FOLDERS.items():
        row = dict(platform=f["platform"], model=f["model"], fingerprint=f["fingerprint"], community=f["community"],
                   powertrain_folder=f["powertrain_folder"])
        if f["source"] == "lab":
            lb = lab.label(f["platform"])
            row.update(footprint_m2=float(lb["footprint_m2"]), powertrain=lb["powertrain"], label_source="lab platform table",
                       spec_note="", confidence="lab")
        else:
            cand = ext[(ext["model"] == f["ext_model"]) & ext["generation_or_years"].str.contains(f["ext_gen"], regex=False)]
            if len(cand) != 1:
                raise RuntimeError(f"{folder}: {len(cand)} specs_external rows for model {f['ext_model']!r} / generation {f['ext_gen']!r}")
            r = cand.iloc[0]
            fp = P.footprint_from_mm(r["wheelbase_mm"], r["track_front_mm"], r["track_rear_mm"])
            if not math.isfinite(fp):
                fp = float(r["footprint_m2"] or "nan")
            row.update(footprint_m2=fp, powertrain=T.normalise_powertrain(r["powertrain"]), label_source="specs_external.csv (public spec)",
                       spec_note=f"{r['generation_or_years']}; {r['note']}", confidence=r["confidence"], wheelbase_mm=r["wheelbase_mm"],
                       track_front_mm=r["track_front_mm"], track_rear_mm=r["track_rear_mm"], curb_mass_kg=r["curb_mass_kg"])
        if row["powertrain"] != f["powertrain_folder"]:
            raise RuntimeError(f"{folder}: spec powertrain {row['powertrain']} differs from the folder's {f['powertrain_folder']}")
        row["size_class"] = T.classify_footprint(row["footprint_m2"])
        row["borderline"] = bool(T.is_borderline(row["footprint_m2"]))
        row["stratum"] = lab.stratum(f["model"])
        brand = f["model"].split()[0]
        row["brand_seen_models"] = sorted(m for m in lab.seen if m.split()[0] == brand)
        row["brand_heldout_models"] = sorted(m for m in lab.heldout if m.split()[0] == brand)
        out[folder] = row
    return out


# ----------------------------------------------------------------------------- assembly
def vehicle_decisions(tables: list[pd.DataFrame]) -> dict:
    """Which channel groups are usable for one vehicle, from all its 10 Hz tables (lab dead-channel logic extended to flags)."""
    v = np.concatenate([t["v_mps"].to_numpy(np.float64) for t in tables])
    moving = v >= T.MOVING_V_MPS
    cat = lambda c: np.concatenate([t[c].to_numpy(np.float64) for t in tables])  # noqa: E731
    gas, brake, sp = cat("gas_pressed"), cat("brake_pressed"), cat("steer_pressed")
    rate, tq, sa, yaw, yv = cat("steer_rate_can_degps"), cat("steer_torque_raw"), cat("steer_angle_deg"), cat("yaw_rate_radps"), cat("yaw_valid")
    n_mov = int(moving.sum())
    d = dict(n_rows=int(len(v)), n_moving=n_mov, gas_fires=int(gas.sum()), brake_fires=int(brake.sum()), steer_fires=int(sp.sum()))
    enough = d["n_rows"] >= DEAD_FLAG_MIN_ROWS                       # a flag is only declared dead with >= 10 min of evidence
    d["gas_alive"] = d["gas_fires"] >= DEAD_FLAG_MIN_FIRES or not enough
    d["brake_alive"] = d["brake_fires"] >= DEAD_FLAG_MIN_FIRES or not enough
    d["rate_nonzero_moving_frac"] = float((rate[moving] != 0).mean()) if n_mov else float("nan")
    d["rate_alive"] = bool(n_mov and d["rate_nonzero_moving_frac"] >= RATE_ALIVE_MIN_FRAC)
    steer_on = sp > 0.5
    d["torque_nonzero_frac"] = float((tq != 0).mean()) if len(tq) else float("nan")
    d["torque_min"], d["torque_max"] = (float(np.nanmin(tq)), float(np.nanmax(tq))) if len(tq) else (float("nan"), float("nan"))
    d["torque_alive"] = bool(len(tq) and not (steer_on.sum() >= DEAD_FLAG_MIN_PRESSED_ALIAS() and float(np.nanmax(np.abs(tq))) == 0.0))
    both_dirs = bool((sa[steer_on] > 5).any() and (sa[steer_on] < -5).any()) if steer_on.any() else False
    d["torque_signed"] = bool(len(tq) and (np.nanmin(tq[steer_on]) < 0 < np.nanmax(tq[steer_on]))) if steer_on.any() else False
    d["torque_unsigned"] = bool(d["torque_alive"] and both_dirs and not d["torque_signed"])
    ok = moving & (v >= 3.0) & (yv > 0.5) & np.isfinite(yaw)
    d["yaw_steer_corr"] = float(np.corrcoef(sa[ok], yaw[ok])[0, 1]) if ok.sum() >= 100 and np.std(sa[ok]) > 0 and np.std(yaw[ok]) > 0 else float("nan")
    d["yaw_valid_frac"] = float(np.mean(yv > 0.5)) if len(yv) else float("nan")
    # automation semantics of the fork: controller flag against the stock cruise flag (pcmCruise cars: they should agree)
    op, cr, la = cat("op_enabled") > 0.5, cat("cruise_enabled") > 0.5, cat("lat_active") > 0.5
    d["op_enabled_frac"] = float(op.mean()); d["cruise_enabled_frac"] = float(cr.mean()); d["lat_active_frac"] = float(la.mean())
    d["op_cruise_disagree_frac"] = float((op != cr).mean())
    d["lat_active_without_op_frac"] = float((la & ~op).mean())
    d["lead_valid_frac"] = float(np.mean(cat("lead_valid") > 0.5))
    d["standstill_frac"] = float(np.mean(cat("standstill") > 0.5))
    d["v_max"] = float(v.max()) if len(v) else float("nan")
    return d


def DEAD_FLAG_MIN_PRESSED_ALIAS() -> int:
    return DEAD_FLAG_MIN_FIRES


def derived_rate(angle: np.ndarray, t: np.ndarray) -> np.ndarray:
    """common.smooth_derivative of the 10 Hz angle, computed inside every gap-free run (no spike across a gap)."""
    out = np.full(len(angle), np.nan)
    gap = np.flatnonzero(np.diff(t) > T.GAP_S) + 1
    for a, b in zip(np.r_[0, gap], np.r_[gap, len(angle)]):
        if b - a >= 3:
            out[a:b] = T.smooth_derivative(angle[a:b])
    return out


def build_drive_windows(tab: pd.DataFrame, dec: dict, lab: dict, drow: dict, args, dataset: str) -> tuple[T.CutResult, dict]:
    t = tab["t_s"].to_numpy(np.float64)
    v = tab["v_mps"].to_numpy(np.float64)
    sp = tab["steer_pressed"].to_numpy(np.float64) > 0.5
    tq = tab["steer_torque_raw"].to_numpy(np.float64)
    # torque: lab rule per drive, absent when the vehicle's torque is dead or unsigned
    torque_status = "drive-q95"
    scale, tq_norm = float("nan"), np.full(len(t), np.nan)
    if args.torque == "auto" and not dec["torque_alive"]:
        torque_status = "absent-dead"
    elif args.torque == "auto" and dec["torque_unsigned"]:
        torque_status = "absent-unsigned"
    else:
        scale, tq_norm = T.torque_norm(tq, v, sp)
        if not math.isfinite(scale):
            torque_status = "absent-few-samples"
    # steering rate
    rate_can = tab["steer_rate_can_degps"].to_numpy(np.float64)
    sa = tab["steer_angle_deg"].to_numpy(np.float64)
    if args.steer_rate == "can" or (args.steer_rate == "auto" and dec["rate_alive"]):
        rate, rate_status = rate_can, "can"
    else:
        rate, rate_status = derived_rate(sa, t), "derived"
    # yaw -> curvature / lateral acceleration (lab definitions)
    yaw = tab["yaw_rate_radps"].to_numpy(np.float64)
    yv = tab["yaw_valid"].to_numpy(np.float64) > 0.5
    curv = np.full(len(t), np.nan)
    m = yv & np.isfinite(yaw) & (v >= T.MOVING_V_MPS)
    np.divide(yaw, v, out=curv, where=m)
    lat = v * yaw
    lead = T.lead_channels(v, tab["lead_valid"].to_numpy(np.float64) > 0.5, tab["lead_d_m"].to_numpy(np.float64), tab["lead_v_rel_mps"].to_numpy(np.float64), t=t)
    gas = tab["gas_pressed"].to_numpy(np.float64)
    brake = tab["brake_pressed"].to_numpy(np.float64)
    pedals_status = "present"
    op = tab["op_enabled"].to_numpy(np.float64) > 0.5
    lat_act = tab["lat_active"].to_numpy(np.float64) > 0.5
    das = tab["das_enabled"].to_numpy(np.float64)
    if args.das == "any":
        das = ((das > 0.5) | lat_act).astype(np.float64)
    lat_only = (lat_act & ~op).astype(np.float64)             # lateral automation while the controller is not "enabled" (fork MADS)
    cmap = dict(v_mps=v, a_mps2=tab["a_mps2"].to_numpy(np.float64), steer_angle_deg=sa, steer_rate_degps=rate,
                steer_torque_norm=(tq_norm if torque_status == "drive-q95" else None), yaw_rate_radps=yaw, curv_1pm=curv, lat_acc_mps2=lat,
                gas_pressed=gas, brake_pressed=brake, steer_pressed=sp.astype(np.float64), standstill=tab["standstill"].to_numpy(np.float64),
                das_enabled=das, **lead)
    if args.pedals == "auto":
        if not dec["gas_alive"] and not dec["brake_alive"]:
            cmap["gas_pressed"] = cmap["brake_pressed"] = None
            pedals_status = "absent-dead"
        elif not dec["gas_alive"]:
            cmap["gas_pressed"] = None
            pedals_status = "gas-absent-dead"
        elif not dec["brake_alive"]:
            cmap["brake_pressed"] = None
            pedals_status = "brake-absent-dead"
    note = ";".join([f"stratum={lab['stratum']}", f"label={'lab' if lab['label_source'].startswith('lab') else 'public'}", "lead=vision-modelV2",
                     "yaw=llk-calibrated", f"rate={rate_status}", f"torque={torque_status}", f"pedals={pedals_status}",
                     "das=carControl|cruise" + ("|latActive" if args.das == "any" else "")]
                    + (["community-fork platform"] if lab["community"] else []))
    info = dict(dataset=dataset, vehicle_id=drow["vehicle_id"], model=lab["model"], platform=lab["platform"], drive_id=drow["drive_id"],
                footprint_m2=lab["footprint_m2"], size_class=lab["size_class"], powertrain=lab["powertrain"], borderline=lab["borderline"], note=note)
    df = pd.DataFrame(dict(t_s=t))
    res = T.cut_windows(df, cmap, t_col="t_s", info=info, align=args.align)
    if res.X.shape[0]:
        # per-window share of lateral-only automation (latActive without enabled), appended to the note as lat_only=<share>
        r0 = np.searchsorted(t, res.meta["t0_s"].to_numpy(np.float64) - 1e-6)
        shares = [float(lat_only[a:a + T.WINDOW_SAMPLES].mean()) for a in r0]
        res.meta["note"] = [f"{n};lat_only={x:.2f}" for n, x in zip(res.meta["note"], shares)]
    qc = dict(drive_id=drow["drive_id"], vehicle_id=drow["vehicle_id"], platform=lab["platform"], model=lab["model"], n_rows=int(len(t)),
              span_s=round(float(t[-1] - t[0]), 1) if len(t) else 0.0, n_segments=int(drow["n_segments"]), torque_status=torque_status,
              torque_scale=(round(scale, 2) if math.isfinite(scale) else None), rate_status=rate_status, pedals_status=pedals_status,
              moving_frac=round(float((v >= T.MOVING_V_MPS).mean()), 3), das_frac=round(float(np.mean(tab["das_enabled"])), 3),
              lead_valid_frac=round(float(np.mean(lead["lead_valid"])), 3), yaw_valid_frac=round(float(yv.mean()), 3),
              gas_frac=round(float(gas.mean()), 3), brake_frac=round(float(brake.mean()), 3), steer_pressed_frac=round(float(sp.mean()), 3),
              lat_only_frac=round(float(lat_only.mean()), 3), op_enabled_frac=round(float(op.mean()), 3),
              v_max=round(float(v.max()), 2), **{f"n_{k}": int(res.stats[f"n_{k}"]) for k in ("gap_rows", "runs", "candidates", "dropped_gap", "dropped_stationary", "windows")})
    return res, qc


def assemble(args, dataset: str) -> dict:
    wd = work_dir(dataset)
    idx_path = wd / "drives_index.parquet"
    if not idx_path.exists():
        raise FileNotFoundError(f"{idx_path} missing: run the decode stage first")
    drives = pd.read_parquet(idx_path)
    drives = drives[drives["ok"]].copy()
    labels = load_labels()
    results, dqc, vqc = [], [], []
    for vid, g in drives.groupby("vehicle_id", sort=True):
        folder = g["folder"].iloc[0]
        lab = labels[folder]
        tables = {r["drive_id"]: pd.read_parquet(wd / "drives10hz" / f"{r['drive_id']}.parquet") for _, r in g.iterrows()}
        dec = vehicle_decisions(list(tables.values()))
        n_win = 0
        for _, r in g.iterrows():
            res, qc = build_drive_windows(tables[r["drive_id"]], dec, lab, r, args, dataset)
            results.append(res)
            dqc.append(qc)
            n_win += res.X.shape[0]
        vqc.append(dict(vehicle_id=vid, platform=lab["platform"], model=lab["model"], size_class=lab["size_class"], powertrain=lab["powertrain"],
                        stratum=lab["stratum"], n_drives=int(len(g)), n_segments=int(g["n_segments_ok"].sum()), n_windows=int(n_win),
                        hours_10hz=round(dec["n_rows"] / 36000.0, 2), moving_hours=round(dec["n_moving"] / 36000.0, 2),
                        **{k: (round(dec[k], 3) if isinstance(dec[k], float) else dec[k]) for k in
                           ("gas_alive", "brake_alive", "gas_fires", "brake_fires", "rate_alive", "rate_nonzero_moving_frac", "torque_alive", "torque_signed",
                            "torque_unsigned", "torque_min", "torque_max", "yaw_steer_corr", "yaw_valid_frac", "op_enabled_frac", "cruise_enabled_frac",
                            "lat_active_frac", "op_cruise_disagree_frac", "lat_active_without_op_frac", "lead_valid_frac", "standstill_frac", "v_max")}))
        print(f"  vehicle {vid} ({lab['model']}): {len(g)} drives -> {n_win} windows; rate_alive={dec['rate_alive']} gas/brake alive={dec['gas_alive']}/{dec['brake_alive']} "
              f"torque signed={dec['torque_signed']} yaw~steer corr={dec['yaw_steer_corr']:.2f}", flush=True)
    X, meta = T.concat_results(results)
    saved = T.save_windows(dataset, X, meta) if len(meta) else None
    pd.DataFrame(dqc).to_csv(wd / "drives_qc.csv", index=False)
    pd.DataFrame(vqc).to_csv(wd / "vehicles_qc.csv", index=False)
    (wd / "labels.json").write_text(json.dumps(labels, indent=1, default=str))
    return dict(X=X, meta=meta, saved=saved, dqc=pd.DataFrame(dqc), vqc=pd.DataFrame(vqc), labels=labels)


def _hex16(text: str) -> bool:
    """True when the text holds a device-id-like token: 16 hex characters with at least one letter, not inside a number
    (a 16-digit float fragment such as 0.37400001287460327 is not a token; cf. the lab's qc scan)."""
    return any(re.search(r"[a-f]", m) for m in re.findall(r"(?<![0-9a-f.])[0-9a-f]{16}(?![0-9a-f])", text))


# ----------------------------------------------------------------------------- validation report
def validation_report(args, dataset: str, asm: dict, decode_summary: dict | None) -> dict:
    wd = work_dir(dataset)
    X, meta, dqc, vqc, labels = asm["X"], asm["meta"], asm["dqc"], asm["vqc"], asm["labels"]
    rep = dict(dataset=dataset, n_windows=int(len(meta)), n_vehicles=int(meta["vehicle_id"].nunique()) if len(meta) else 0,
               n_drives_with_windows=int(meta["drive_id"].nunique()) if len(meta) else 0, hours=round(len(meta) / 60.0, 2),
               align=args.align, steer_rate=args.steer_rate, torque=args.torque, pedals=args.pedals, das=args.das, decode=decode_summary)
    per_plat = []
    for folder, lab in labels.items():
        g = meta[meta["platform"] == lab["platform"]] if len(meta) else meta
        d = dqc[dqc["platform"] == lab["platform"]]
        vq = vqc[vqc["platform"] == lab["platform"]]
        row = dict(folder=folder, platform=lab["platform"], model=lab["model"], footprint_m2=round(lab["footprint_m2"], 3), size_class=lab["size_class"],
                   powertrain=lab["powertrain"], borderline=lab["borderline"], stratum=lab["stratum"], label_source=lab["label_source"],
                   n_vehicles=int(vq["vehicle_id"].nunique()), n_drives=int(len(d)), n_segments=int(vq["n_segments"].sum()) if len(vq) else 0,
                   n_windows=int(len(g)), hours=round(len(g) / 60.0, 2), das_frac=round(float(g["das_frac"].mean()), 3) if len(g) else None,
                   moving_frac=round(float(g["moving_frac"].mean()), 3) if len(g) else None)
        if len(g):
            Xi = X[g.index.to_numpy()]
            ch = lambda c: Xi[:, :, T.CH[c]]  # noqa: E731
            lv = ch("lead_valid")
            row.update(v_max=round(float(np.nanmax(ch("v_mps"))), 1), v_mean=round(float(np.nanmean(ch("v_mps"))), 1),
                       a_min=round(float(np.nanmin(ch("a_mps2"))), 2), a_max=round(float(np.nanmax(ch("a_mps2"))), 2),
                       steer_abs_p99=round(float(np.nanquantile(np.abs(ch("steer_angle_deg")), 0.99)), 1),
                       rate_abs_p99=round(float(np.nanquantile(np.abs(ch("steer_rate_degps")), 0.99)), 1) if np.isfinite(ch("steer_rate_degps")).any() else None,
                       torque_norm_abs_p95=round(float(np.nanquantile(np.abs(ch("steer_torque_norm")), 0.95)), 2) if np.isfinite(ch("steer_torque_norm")).any() else None,
                       yaw_abs_p99=round(float(np.nanquantile(np.abs(ch("yaw_rate_radps")), 0.99)), 3),
                       gas_frac=round(float(np.nanmean(ch("gas_pressed"))), 3) if np.isfinite(ch("gas_pressed")).any() else None,
                       brake_frac=round(float(np.nanmean(ch("brake_pressed"))), 3) if np.isfinite(ch("brake_pressed")).any() else None,
                       steer_pressed_frac=round(float(np.nanmean(ch("steer_pressed"))), 3), standstill_frac=round(float(np.nanmean(ch("standstill"))), 3),
                       lead_valid_sample_frac=round(float(np.nanmean(lv)), 3), windows_with_lead_frac=round(float((np.nansum(lv, axis=1) > 0).mean()), 3),
                       thw_med=round(float(np.nanmedian(ch("thw_s"))), 2) if np.isfinite(ch("thw_s")).any() else None,
                       yaw_steer_corr_vehicles=[round(float(x), 3) for x in vq["yaw_steer_corr"].tolist()],
                       channels_present=sorted({c for s in g["channels_present"] for c in s.split(",") if c}, key=T.CH.get))
            row["channels_absent"] = [c for c in T.CHANNELS if c not in row["channels_present"]]
        if len(d):
            row["lat_only_sample_frac"] = round(float((d["lat_only_frac"] * d["n_rows"]).sum() / max(d["n_rows"].sum(), 1)), 3)
        if len(g):
            lo = g["note"].str.extract(r"lat_only=([0-9.]+)")[0].astype(float)
            row["windows_lat_only_gt_half_frac"] = round(float((lo > 0.5).mean()), 3)
            row["windows_lat_only_any_frac"] = round(float((lo > 0.0).mean()), 3)
        per_plat.append(row)
    rep["per_platform"] = per_plat
    rep["per_vehicle"] = vqc.to_dict(orient="records")
    rep["windows_dropped"] = dict(gap=int(dqc["n_dropped_gap"].sum()), stationary=int(dqc["n_dropped_stationary"].sum()), candidates=int(dqc["n_candidates"].sum()))
    rep["size_class"] = meta["size_class"].fillna("unknown").value_counts().to_dict() if len(meta) else {}
    rep["powertrain"] = meta["powertrain"].value_counts().to_dict() if len(meta) else {}
    checks = []
    if len(meta):
        for c in T.BINARY_CHANNELS:
            col = X[:, :, T.CH[c]]
            fin = col[np.isfinite(col)]
            checks.append(dict(check=f"{c} is 0/1", ok=bool(np.isin(fin, (0.0, 1.0)).all())))
        lv = X[:, :, T.CH["lead_valid"]]
        for c in T.LEAD_CHANNELS[1:]:
            col = X[:, :, T.CH[c]]
            checks.append(dict(check=f"{c} NaN where lead_valid = 0", ok=bool(np.isnan(col[lv < 0.5]).all())))
        checks.append(dict(check="lead_valid never NaN", ok=bool(np.isfinite(lv).all())))
        checks.append(dict(check="lead_speed = v + v_rel", ok=bool(np.nanmax(np.abs(X[:, :, T.CH["lead_speed_mps"]] - X[:, :, T.CH["v_mps"]] - X[:, :, T.CH["lead_v_rel_mps"]])) < 1e-3)))
        v = X[:, :, T.CH["v_mps"]]
        cv = X[:, :, T.CH["curv_1pm"]]
        checks.append(dict(check="curv NaN below 1 m/s", ok=bool(np.isnan(cv[v < T.MOVING_V_MPS]).all())))
        checks.append(dict(check="moving rule >= 0.2", ok=bool((meta["moving_frac"] >= T.MIN_MOVING_FRAC).all())))
        checks.append(dict(check="yaw left-positive (corr with steering > 0 for every vehicle)", ok=bool((vqc["yaw_steer_corr"].dropna() > 0).all())))
        checks.append(dict(check="no 16-hex device token in meta strings", ok=not any(_hex16(str(x)) for c in ("vehicle_id", "drive_id", "note", "model", "platform") for x in meta[c].unique())))
    rep["checks"] = checks
    (wd / "validation.json").write_text(json.dumps(rep, indent=1, default=str))
    # markdown tables for the report
    lines = ["## Counts per platform", "",
             T.md_table(["folder", "platform", "model", "footprint", "class", "powertrain", "stratum", "vehicles", "drives", "segments", "windows", "hours", "DAS share", "lateral-only share", "lead sample share", "windows with lead"],
                        [[r["folder"], r["platform"], r["model"], r["footprint_m2"], r["size_class"], r["powertrain"], r["stratum"], r["n_vehicles"], r["n_drives"], r["n_segments"],
                          r["n_windows"], r["hours"], r.get("das_frac"), r.get("lat_only_sample_frac"), r.get("lead_valid_sample_frac"), r.get("windows_with_lead_frac")] for r in per_plat]), "",
             "## Per vehicle", "",
             T.md_table(["vehicle", "model", "drives", "segments", "windows", "10 Hz hours", "gas alive", "brake alive", "rate alive", "torque signed", "yaw~steer corr", "yaw valid"],
                        [[r["vehicle_id"], r["model"], r["n_drives"], r["n_segments"], r["n_windows"], r["hours_10hz"], r["gas_alive"], r["brake_alive"], r["rate_alive"], r["torque_signed"],
                          r["yaw_steer_corr"], r["yaw_valid_frac"]] for r in rep["per_vehicle"]]), "",
             "## Signal ranges in the windows", "",
             T.md_table(["platform", "v max", "v mean", "a min", "a max", "|steer| p99", "|rate| p99", "|torque_norm| p95", "|yaw| p99", "gas", "brake", "steer pressed", "standstill", "THW median"],
                        [[r["platform"], r.get("v_max"), r.get("v_mean"), r.get("a_min"), r.get("a_max"), r.get("steer_abs_p99"), r.get("rate_abs_p99"), r.get("torque_norm_abs_p95"),
                          r.get("yaw_abs_p99"), r.get("gas_frac"), r.get("brake_frac"), r.get("steer_pressed_frac"), r.get("standstill_frac"), r.get("thw_med")] for r in per_plat]), "",
             "## Checks", "", T.md_table(["check", "ok"], [[c["check"], c["ok"]] for c in checks])]
    (wd / "report_tables.md").write_text("\n".join(lines) + "\n")
    return rep


# ----------------------------------------------------------------------------- comparison with the training distribution
def compare_training(dataset: str) -> Path | None:
    """Per-channel summary of the saved windows (per platform) next to the training reference statistics
    (<work>/training_reference_stats.json: nan share, mean, std, p01, p99 on a 4,000-window sample of the training split,
    computed once with $CPY from work/tensors/X_f16.npy). Writes <work>/compare_training.md."""
    wd = work_dir(dataset)
    ref_path = wd / "training_reference_stats.json"
    if not ref_path.exists():
        print(f"[ds_moretorque] no {ref_path.name}: skipping the training comparison")
        return None
    allref = json.loads(ref_path.read_text())
    ref = allref["train"]
    X, meta = T.load_windows(dataset)
    groups = [("training sample", None)] + [(p, p) for p in sorted(meta["platform"].unique())]
    # the lab's own windows of the same brand / model (all splits) as a second reference where available
    extra = [("lab Hyundai Sonata windows", allref.get("sonata_all")), ("lab Mazda windows (CX-5 / CX-9 / 6)", allref.get("mazda_all"))]
    lines = ["## Channel statistics: MoreTorque windows per platform against the training sample", "",
             "nan = share of NaN samples (a dropped channel is NaN everywhere; lead values are NaN without a lead); flags are shares.", ""]
    for c in T.CHANNELS:
        rows = []
        r = ref[c]
        rows.append(["training sample", r["nan_frac"], r["mean"], r["std"], r["p01"], r["p99"]])
        for name, er in extra:
            if er and c in er:
                rows.append([name, er[c]["nan_frac"], er[c]["mean"], er[c]["std"], er[c]["p01"], er[c]["p99"]])
        for name, plat in groups[1:]:
            col = X[meta["platform"].to_numpy() == plat][:, :, T.CH[c]]
            fin = np.isfinite(col)
            if fin.any():
                rows.append([name, round(float(1 - fin.mean()), 3), round(float(np.nanmean(col)), 4), round(float(np.nanstd(col)), 4),
                             round(float(np.nanquantile(col, 0.01)), 3), round(float(np.nanquantile(col, 0.99)), 3)])
            else:
                rows.append([name, 1.0, None, None, None, None])
        lines += [f"### {c}", "", T.md_table(["set", "nan", "mean", "std", "p01", "p99"], rows), ""]
    lv = X[:, :, T.CH["lead_valid"]]
    lines += ["### windows with at least one valid lead sample", "",
              T.md_table(["set", "share"], [["training sample", ref["_windows_with_lead_frac"]]] +
                         [[name, er["_windows_with_lead_frac"]] for name, er in extra if er] +
                         [[p, round(float((np.nansum(lv[meta["platform"].to_numpy() == p], axis=1) > 0).mean()), 3)] for p in sorted(meta["platform"].unique())]), ""]
    out = wd / "compare_training.md"
    out.write_text("\n".join(lines))
    print(f"[ds_moretorque] wrote {out}")
    return out


# ----------------------------------------------------------------------------- decode stage driver
def pick_smoke(drives: list[dict], n_segments: int) -> list[dict]:
    """~n_segments / 4 consecutive segments from one drive per folder (the longest drive of the first vehicle with >= 5 segments)."""
    per = max(1, n_segments // len(FOLDERS))
    out = []
    for folder in sorted(FOLDERS):
        cands = [d for d in drives if d["folder"] == folder and len(d["segs"]) >= per]
        if not cands:
            cands = [d for d in drives if d["folder"] == folder]
        if not cands:
            continue
        d = dict(sorted(cands, key=lambda x: (-len(x["segs"]), x["drive_id"]))[0])
        d["segs"] = d["segs"][:per]
        d["seg1"] = d["segs"][-1][0]
        out.append(d)
    return out


def decode_stage(args, dataset: str) -> dict:
    wd = work_dir(dataset)
    (wd / "drives10hz").mkdir(parents=True, exist_ok=True)
    drives = find_drives(RAW_ROOT, args.folders)
    n_seg_all = sum(len(d["segs"]) for d in drives)
    print(f"[ds_moretorque] {n_seg_all} segments in {len(drives)} drives, {len({d['device'] for d in drives})} devices, "
          f"{len({(d['device'], d['route']) for d in drives})} routes under {RAW_ROOT.name}/", flush=True)
    if args.smoke:
        drives = pick_smoke(drives, args.smoke)
    if args.limit_drives:
        drives = drives[:args.limit_drives]
    if args.resume:
        before = len(drives)
        drives = [d for d in drives if not (wd / "drives10hz" / f"{d['drive_id']}.parquet").exists()]
        print(f"[ds_moretorque] resume: {before - len(drives)} drives already decoded, {len(drives)} to go", flush=True)
    tasks = [dict(d, out_dir=str(wd / "drives10hz")) for d in drives]
    n_seg = sum(len(t["segs"]) for t in tasks)
    print(f"[ds_moretorque] decoding {n_seg} segments in {len(tasks)} drives with {args.workers} workers (qlog decimation {'off' if args.native_rates else 'on'})", flush=True)
    t0 = time.time()
    results: list[dict] = []
    done_seg = 0

    def _progress(r):
        nonlocal done_seg
        done_seg += len(r["segments"])
        if len(results) % 25 == 0 or len(results) == len(tasks):
            el = time.time() - t0
            print(f"  drives {len(results)}/{len(tasks)}  segments {done_seg}/{n_seg}  {el:.0f}s  ({done_seg / max(el, 1e-9):.2f} seg/s)", flush=True)

    if args.workers <= 1:
        init_worker(args.native_rates)
        for t in tasks:
            results.append(process_drive(t))
            _progress(results[-1])
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=init_worker, initargs=(args.native_rates,)) as ex:
            tasks_sorted = sorted(tasks, key=lambda t: -len(t["segs"]))        # long drives first for load balance
            for r in ex.map(process_drive, tasks_sorted, chunksize=1):
                results.append(r)
                _progress(r)
    seconds = time.time() - t0
    # index + QC (pseudonymous)
    by_id = {t["drive_id"]: t for t in tasks}
    idx_rows, seg_rows, cps = [], [], []
    for r in results:
        t = by_id[r["drive_id"]]
        row = public_drive_row(t)
        n_ok = sum(1 for s in r["segments"] if s["status"] == "ok")
        row.update(ok=bool(r["ok"]), error=r.get("error", ""), n_segments_ok=n_ok, n_segments_failed=sum(1 for s in r["segments"] if s["status"] == "failed"),
                   n_segments_skipped=sum(1 for s in r["segments"] if s["status"] == "skipped"), n_rows=int(r.get("n_rows", 0)), n_gaps=r.get("n_gaps"),
                   max_gap_s=r.get("max_gap_s"), dt_med_s=r.get("dt_med_s"), span_s=r.get("span_s"), yaw_valid_frac=r.get("yaw_valid_frac"),
                   lead_valid_frac=r.get("lead_valid_frac"), das_frac=r.get("das_frac"), decode_s=r.get("decode_s"))
        cp = r.get("car_params") or {}
        row.update(fingerprint=cp.get("fingerprint"), cp_wheelbase_m=cp.get("wheelbase_m"), cp_steer_ratio=cp.get("steer_ratio"), cp_mass_kg=cp.get("mass_kg"),
                   cp_radar_unavailable=cp.get("radar_unavailable"), cp_op_long=cp.get("op_long"))
        idx_rows.append(row)
        for s in r["segments"]:
            seg_rows.append(dict(drive_id=r["drive_id"], vehicle_id=r["vehicle_id"], folder=r["folder"], platform=FOLDERS[r["folder"]]["platform"], **s))
    # append to an existing index on --resume
    idx = pd.DataFrame(idx_rows)
    seg = pd.DataFrame(seg_rows)
    if args.resume and (wd / "drives_index.parquet").exists():
        old = pd.read_parquet(wd / "drives_index.parquet")
        idx = pd.concat([old[~old["drive_id"].isin(idx["drive_id"])], idx], ignore_index=True) if len(idx) else old
        if (wd / "segments_qc.csv").exists():
            olds = pd.read_csv(wd / "segments_qc.csv")
            seg = pd.concat([olds[~olds["drive_id"].isin(seg["drive_id"])], seg], ignore_index=True) if len(seg) else olds
    if len(idx):
        idx.to_parquet(wd / "drives_index.parquet", index=False)
    if len(seg):
        seg.to_csv(wd / "segments_qc.csv", index=False)
    fails = seg[seg["status"] != "ok"] if len(seg) else seg
    summary = dict(dataset=dataset, raw_folders=sorted({t["folder"] for t in tasks}), segments_listed_total=n_seg_all, drives_listed_total=len(find_drives(RAW_ROOT, args.folders)) if not args.smoke else None,
                   segments_in_run=n_seg, drives_in_run=len(tasks), drives_ok=int(idx["ok"].sum()) if len(idx) else 0,
                   segments_ok=int((seg["status"] == "ok").sum()) if len(seg) else 0, segments_failed=int((seg["status"] == "failed").sum()) if len(seg) else 0,
                   segments_skipped=int((seg["status"] == "skipped").sum()) if len(seg) else 0,
                   skip_reasons=fails["reason"].str.split(":").str[0].value_counts().to_dict() if len(fails) else {},
                   fingerprints=seg.groupby("folder")["fingerprint"].agg(lambda s: s.dropna().value_counts().to_dict()).to_dict() if len(seg) else {},
                   seconds=round(seconds, 1), seg_per_s=round(n_seg / max(seconds, 1e-9), 2), workers=args.workers, qlog_decimation=not args.native_rates,
                   decimation=DECIM, max_ages_s=dict(ctrl=CTRL_MAX_AGE_S, pose=POSE_MAX_AGE_S, lead=LEAD_MAX_AGE_S), yaw_sign=POSE_YAW_SIGN, radar_to_camera_m=RADAR_TO_CAMERA_M,
                   per_folder=seg[seg["status"] == "ok"].groupby("folder").agg(segments=("seg", "size"), drives=("drive_id", "nunique"), vehicles=("vehicle_id", "nunique"),
                                                                             v_max=("v_max", "max"), v_mean=("v_mean", "mean"), moving_frac=("moving_frac", "mean"), gas=("gas_frac", "mean"),
                                                                             brake=("brake_frac", "mean"), steer_pressed=("steer_pressed_frac", "mean"), standstill=("standstill_frac", "mean"),
                                                                             cruise=("cruise_frac", "mean"), op_enabled=("op_enabled_frac", "mean"), lead_prob=("lead_prob_frac", "mean"),
                                                                             rate_nonzero=("rate_nonzero", "mean"), torque_min=("torque_min", "min"), torque_max=("torque_max", "max"),
                                                                             decode_s=("decode_s", "mean")).round(3).reset_index().to_dict(orient="records") if len(seg) else [])
    (wd / "decode_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"[ds_moretorque] decoded {summary['segments_ok']} segments in {summary['drives_ok']} drives; failed {summary['segments_failed']}, skipped {summary['segments_skipped']}; "
          f"{summary['seconds']} s ({summary['seg_per_s']} seg/s)", flush=True)
    for reason, n in summary["skip_reasons"].items():
        print(f"    {n} x {reason}")
    return summary


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("all", "decode", "assemble", "compare"), default="all")
    ap.add_argument("--dataset", default=DATASET, help="window set name (default moretorque; e.g. moretorque_smoke for the test)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", type=int, default=0, help="decode only this many segments (consecutive ones of one drive per folder)")
    ap.add_argument("--limit-drives", type=int, default=0)
    ap.add_argument("--folders", nargs="*", default=None, help="restrict to these folder names")
    ap.add_argument("--resume", action="store_true", help="skip drives whose 10 Hz table exists")
    ap.add_argument("--native-rates", action="store_true", help="hold the 100 / 20 Hz topics directly (no qlog-like decimation)")
    ap.add_argument("--align", choices=("drive", "run"), default="drive", help="window blocks: fixed 600-row blocks from the drive start (lab) or restarted after gaps")
    ap.add_argument("--steer-rate", choices=("auto", "can", "derived"), default="auto", help="auto = CAN rate where the vehicle's rate is alive, else smooth_derivative(angle)")
    ap.add_argument("--torque", choices=("auto", "keep"), default="auto", help="auto = torque group absent when dead or unsigned over the vehicle")
    ap.add_argument("--pedals", choices=("auto", "keep"), default="auto", help="auto = pedal flags absent (NaN) when a vehicle's flag never fires")
    ap.add_argument("--das", choices=("lab", "any"), default="lab",
                    help="das_enabled = controller enabled OR stock cruise enabled (lab definition, default) | additionally carControl.latActive (the forks steer with lateral-only automation, MADS)")
    a = ap.parse_args(argv)
    if a.workers > 6:
        raise SystemExit("at most 6 worker processes")
    t0 = time.time()
    dec = None
    if a.stage in ("all", "decode"):
        dec = decode_stage(a, a.dataset)
    rep = None
    if a.stage in ("all", "assemble"):
        if dec is None and (work_dir(a.dataset) / "decode_summary.json").exists():
            dec = json.loads((work_dir(a.dataset) / "decode_summary.json").read_text())
        print("[ds_moretorque] assembling windows", flush=True)
        asm = assemble(a, a.dataset)
        rep = validation_report(a, a.dataset, asm, dec)
        print(f"[ds_moretorque] {rep['n_windows']} windows, {rep['n_vehicles']} vehicles, {rep['n_drives_with_windows']} drives with windows, {rep['hours']} h; "
              f"size {rep['size_class']} powertrain {rep['powertrain']}")
        for r in rep["per_platform"]:
            print(f"    {r['platform']:<16s} {r['model']:<15s} {str(r['size_class']):<5s}/{r['powertrain']:<3s} {r['stratum']:<8s} vehicles {r['n_vehicles']:>2d} drives {r['n_drives']:>4d} "
                  f"segments {r['n_segments']:>5d} windows {r['n_windows']:>5d} ({r['hours']} h) DAS {r.get('das_frac')} lead samples {r.get('lead_valid_sample_frac')} "
                  f"windows-with-lead {r.get('windows_with_lead_frac')} yaw~steer {r.get('yaw_steer_corr_vehicles')} absent {r.get('channels_absent')}")
        bad = [c for c in rep["checks"] if not c["ok"]]
        print(f"    checks: {len(rep['checks']) - len(bad)} ok, {len(bad)} failed" + (f" -> {[c['check'] for c in bad]}" if bad else ""))
    if a.stage in ("all", "assemble", "compare"):
        compare_training(a.dataset)
    print(f"[ds_moretorque] total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
