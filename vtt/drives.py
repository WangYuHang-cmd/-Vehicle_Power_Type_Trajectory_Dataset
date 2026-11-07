"""Stage S5: one public drive table per extracted route (vtt.schema.DRIVE_COLUMNS), its drives-index row and a private QC row.

Timeline (plan section 3 'public timeline'): row k of a drive = the k-th carState sample in logMonoTime order (never by
segment number); ``t_s = k / 10`` is nominal, the real sample-time statistics go only to the private QC frame; ``gap`` marks
a row whose real dt to the previous sample exceeds config.CS_GAP_S; ``seg`` is the ordinal of the source segment by first
appearance. Every other topic is put onto the carState sample times by a causal zero-order hold with a maximum age
(config.CTRL_MAX_AGE_S / LEAD_MAX_AGE_S / POSE_MAX_AGE_S; stale -> NaN / False), so no column carries future information.
Units are in the column names (m/s, m/s^2, deg, rad/s, 1/m, s); flags are bool.

Dead channels: a pedal magnitude that stays at 0 while its *_pressed flag fires >= config.DEAD_CHANNEL_MIN_PRESSED times
(or is NaN throughout) is dead for the whole drive -> the column is NaN and gas_alive / brake_alive is False; the driver
torque channel likewise (torque_alive). steer_torque_norm = raw / quantile(|raw|, TORQUE_NORM_Q) over moving & steering
samples (NaN with fewer than TORQUE_NORM_MIN_SAMPLES of them).

Lead columns are derived from the native radarState samples (tocf.lead: validity, episodes, causal 1-s speed slope,
source) and then held; lead_speed_mps = v_mps + lead_v_rel_mps (never the radar's own vLead).
The builder never reads the raw library and never sees an identifier other than the internal route_uid.
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vtt import config as C
from vtt import schema as S
from vtt import signals as SG

V_SET_MAX_MPS = 45.0           # = tocf.config.V_SET_MAX_MS: a set speed above this (incl. the 255 km/h unset sentinel) is 'unset'
KPH_TO_MPS = 1.0 / 3.6
LEAD_SOURCE_CODES = {"none": 0, "radar": 1, "vision": 2}      # schema.DRIVE_COLUMNS lead_source
POSE_SOURCE_NAMES = {v: k for k, v in SG.POSE_SOURCE_CODES.items()}
CTRL_TOPICS = ("carControl", "controlsState", "selfdriveState")
_VER_RE = re.compile(r"(\d+)\.(\d+)")

# private QC frame (private/qc/drives_private_qc.parquet): real clock statistics and raw provenance, never released
PRIVATE_QC_COLUMNS = (
    "route_uid", "vehicle_id", "drive_id", "n_rows", "n_carState", "n_carControl", "n_controlsState", "n_selfdriveState",
    "n_radarState", "n_pose", "dt_med_s", "dt_p01_s", "dt_p99_s", "dt_max_s", "real_span_s", "nominal_span_s", "gap_total_s",
    "drift_s", "n_gaps", "max_gap_s", "version", "software_era", "device_type", "pose_source_meta", "pose_source",
    "radar_unavailable", "op_long", "pcm_cruise", "n_seg_meta", "n_seg_table", "n_seg_rlog_dec", "n_seg_none",
    "unknown_union_n", "gas_alive", "brake_alive", "torque_alive", "gas_pressed_n", "brake_pressed_n", "steer_pressed_n",
    "steer_torque_scale", "cruise_speed_max_mps", "v_set_valid_frac", "cruise_enabled_frac", "ctrl_enabled_frac",
    "sds_enabled_frac", "cc_enabled_frac", "lead_valid_frac", "lead_radar_frac", "lead_vision_frac", "yaw_valid_frac",
    "extractor_version", "config_version", "built_at",
)


# ----------------------------------------------------------------------------- roots
class Roots:
    """Every S5 path under a data root, with the same relative layout as vtt.paths.P (so --data-root can point at a temp root)."""

    def __init__(self, data_root: Path | str = C.DATA_ROOT):
        self.data = Path(data_root)
        self.work = self.data / "work"
        self.signals = self.work / "signals"
        self.logs = self.work / "logs"
        self.failures = self.logs / "s05_failures.csv"
        self.build = self.data / "build"
        self.data_dir = self.build / "data"
        self.index = self.build / "index"
        self.drives = self.index / "drives.parquet"
        self.drive_names = self.build / "drive_names.parquet"
        self.routes_labels = self.build / "routes_labels.parquet"
        self.routes_all = self.build / "routes_all.csv"
        self.private = self.data / "private"
        self.private_qc = self.private / "qc" / "drives_private_qc.parquet"

    def drive_file(self, vehicle_id: str, drive_id: str) -> Path:
        return self.data_dir / vehicle_id / f"{drive_id}.parquet"


def drive_rel_file(vehicle_id: str, drive_id: str) -> str:
    """The drives-index 'file' column: path of the drive table relative to the build / release root."""
    return f"data/{vehicle_id}/{drive_id}.parquet"


# ----------------------------------------------------------------------------- small helpers
def software_era(version) -> str:
    """Coarse software generation of a version string (config.SOFTWARE_ERAS): '0.8' / '0.9' / '0.10' / '0.11' for 0.x
    releases, '2025' / '2026' for year-based ones, 'other' for anything else. Defers to vtt.manifest.software_era when
    that module exists (S1 owns the definition)."""
    try:
        from vtt.manifest import software_era as _se           # lazy: written by another stage
        return _se(version)
    except (ImportError, AttributeError):
        pass
    m = _VER_RE.search(str(version or ""))
    if not m:
        return "other"
    major, minor = int(m.group(1)), int(m.group(2))
    era = f"0.{minor}" if major == 0 else str(major)
    return era if era in C.SOFTWARE_ERAS else "other"


def _opt(x):
    """Label / meta value -> None when missing (None, NaN, pandas NA)."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return x


def _opt_bool(x):
    x = _opt(x)
    return None if x is None else bool(x)


def _frac(mask) -> float:
    m = np.asarray(mask, dtype=bool)
    return float(m.mean()) if len(m) else float("nan")


def dead_channel(mag, pressed, min_pressed: int = C.DEAD_CHANNEL_MIN_PRESSED) -> bool:
    """A magnitude channel is dead when it is NaN throughout, or when its finite values never leave 0 while the
    matching pressed flag is True on at least min_pressed samples (the flag proves the pedal was used)."""
    mag = np.asarray(mag, dtype="float64"); pressed = np.asarray(pressed, dtype=bool)
    fin = np.isfinite(mag)
    if not fin.any():
        return True
    if int(pressed.sum()) < min_pressed:
        return False
    return float(np.abs(mag[fin]).max()) == 0.0


def torque_norm(torque, v, steer_pressed) -> tuple[float, np.ndarray]:
    """(scale, torque / scale): scale = quantile(|torque|, TORQUE_NORM_Q) over samples moving (v >= MOVING_V_MPS) with the
    driver steering (steer_pressed); (NaN, all-NaN) with fewer than TORQUE_NORM_MIN_SAMPLES such samples or a zero scale."""
    torque = np.asarray(torque, dtype="float64")
    sel = np.isfinite(torque) & (np.asarray(v, dtype="float64") >= C.MOVING_V_MPS) & np.asarray(steer_pressed, dtype=bool)
    nan = np.full(len(torque), np.nan)
    if int(sel.sum()) < C.TORQUE_NORM_MIN_SAMPLES:
        return float("nan"), nan
    scale = float(np.quantile(np.abs(torque[sel]), C.TORQUE_NORM_Q))
    if not scale > 0:
        return float("nan"), nan
    return scale, torque / scale


def valid_set_speed_kph(kph) -> np.ndarray:
    """km/h on the wire -> m/s where finite and in (0, V_SET_MAX_MPS] after conversion, else NaN (0 = nothing set / dead
    field, 255 km/h = openpilot's unset sentinel; same rule as tocf.pose.valid_set_speed_ms)."""
    v = np.asarray(kph, dtype="float64") * KPH_TO_MPS
    return np.where(np.isfinite(v) & (v > 0.0) & (v <= V_SET_MAX_MPS), v, np.nan)


def valid_set_speed_mps(mps) -> np.ndarray:
    """carState.cruiseState.speed is already m/s: keep finite values in (0, V_SET_MAX_MPS]."""
    v = np.asarray(mps, dtype="float64")
    return np.where(np.isfinite(v) & (v > 0.0) & (v <= V_SET_MAX_MPS), v, np.nan)


def _first_valid(*cands) -> np.ndarray:
    out = np.asarray(cands[0], dtype="float64").copy()
    for c in cands[1:]:
        miss = np.isnan(out)
        out[miss] = np.asarray(c, dtype="float64")[miss]
    return out


# ----------------------------------------------------------------------------- per-topic columns
def control_columns(sig: dict, t: np.ndarray) -> dict:
    """carControl / controlsState / selfdriveState held onto the carState times (max age CTRL_MAX_AGE_S).
    Returns cc_enabled, lat_active, long_active, ctrl_enabled, sds_enabled (bool) and ctrl_vcruise, ctrl_vcruise_cluster,
    sds_vcruise, sds_vcruise_cluster (km/h, NaN when stale), plus n_<topic> counts."""
    n = len(t)
    out = {k: np.zeros(n, dtype=bool) for k in ("cc_enabled", "lat_active", "long_active", "ctrl_enabled", "sds_enabled")}
    out.update({k: np.full(n, np.nan) for k in ("ctrl_vcruise", "ctrl_vcruise_cluster", "sds_vcruise", "sds_vcruise_cluster")})
    for topic in CTRL_TOPICS:
        df = sig.get(topic)
        out[f"n_{topic}"] = 0 if df is None else int(len(df))
        if df is None or len(df) == 0:
            continue
        ts = SG.rel_seconds(df["logMonoTime"].to_numpy(), sig["_t0_ns"])
        j, _, fresh = SG.hold(ts, t, C.CTRL_MAX_AGE_S)
        if topic == "carControl":
            out["cc_enabled"] = SG.hold_bool(df["enabled"].to_numpy(), j, fresh)
            out["lat_active"] = SG.hold_bool(df["latActive"].to_numpy(), j, fresh)
            out["long_active"] = SG.hold_bool(df["longActive"].to_numpy(), j, fresh)
        else:
            key = "ctrl" if topic == "controlsState" else "sds"
            out[f"{key}_enabled"] = SG.hold_bool(df["enabled"].to_numpy(), j, fresh)
            out[f"{key}_vcruise"] = SG.hold_float(df["vCruise"].to_numpy(), j, fresh)
            out[f"{key}_vcruise_cluster"] = SG.hold_float(df["vCruiseCluster"].to_numpy(), j, fresh)
    return out


def set_speed(cs: pd.DataFrame, ctrl: dict) -> np.ndarray:
    """v_set_mps per sample: the first valid of carState.vCruise, carState.vCruiseCluster, held controlsState.vCruise /
    vCruiseCluster, held selfdriveState.vCruise / vCruiseCluster (all km/h on the wire, openpilot's own set speed) and
    finally carState.cruiseState.speed (m/s, the stock cruise set speed); NaN when nothing is set."""
    return _first_valid(valid_set_speed_kph(cs["vCruise"].to_numpy()), valid_set_speed_kph(cs["vCruiseCluster"].to_numpy()),
                        valid_set_speed_kph(ctrl["ctrl_vcruise"]), valid_set_speed_kph(ctrl["ctrl_vcruise_cluster"]),
                        valid_set_speed_kph(ctrl["sds_vcruise"]), valid_set_speed_kph(ctrl["sds_vcruise_cluster"]),
                        valid_set_speed_mps(cs["cruiseState_speed"].to_numpy()))


def pose_columns(pose: pd.DataFrame | None, t0_ns: int, t: np.ndarray, v: np.ndarray) -> dict:
    """yaw_rate_radps (left positive = POSE_YAW_SIGN x angularVelocityDevice_z), yaw_valid (fresh <= POSE_MAX_AGE_S and
    angularVelocityDevice_valid and inputsOK and finite), pose_source (int8 codes), curv_1pm (yaw / v for v >= MOVING_V_MPS
    and yaw_valid), lat_acc_mps2 (v x yaw), qc_yaw_age_s; plus n_pose and pose_source (route-level string)."""
    n = len(t)
    out = dict(yaw_rate_radps=np.full(n, np.nan), yaw_valid=np.zeros(n, dtype=bool), pose_source=np.zeros(n, dtype=np.int64),
               curv_1pm=np.full(n, np.nan), lat_acc_mps2=np.full(n, np.nan), qc_yaw_age_s=np.full(n, np.nan),
               n_pose=0, pose_source_name="none")
    if pose is None or len(pose) == 0:
        return out
    out["n_pose"] = int(len(pose))
    tp = SG.rel_seconds(pose["logMonoTime"].to_numpy(), t0_ns)
    yaw_src = C.POSE_YAW_SIGN * pose["angularVelocityDevice_z"].to_numpy(dtype="float64")
    valid_src = SG.as_bool(pose["angularVelocityDevice_valid"].to_numpy()) & SG.as_bool(pose["inputsOK"].to_numpy()) & np.isfinite(yaw_src)
    codes = pose["pose_source"].map(lambda s: SG.POSE_SOURCE_CODES.get(s, 0) if s is not None else 0).to_numpy(dtype=np.int64)
    j, age, fresh = SG.hold(tp, t, C.POSE_MAX_AGE_S)
    yaw = SG.hold_float(yaw_src, j, fresh)
    yaw_valid = fresh & valid_src[j] & np.isfinite(yaw)
    out["yaw_rate_radps"] = yaw
    out["yaw_valid"] = yaw_valid
    out["pose_source"] = np.where(fresh, codes[j], 0)
    mask = yaw_valid & (v >= C.MOVING_V_MPS)
    curv = np.full(n, np.nan)
    np.divide(yaw, v, out=curv, where=mask)
    out["curv_1pm"] = curv
    out["lat_acc_mps2"] = v * yaw
    out["qc_yaw_age_s"] = age
    present = set(np.unique(out["pose_source"][fresh]).tolist()) - {0}
    out["pose_source_name"] = "none" if not present else ("mixed" if len(present) > 1 else POSE_SOURCE_NAMES[present.pop()])
    return out


def lead_columns(radar: pd.DataFrame | None, t0_ns: int, t: np.ndarray, v: np.ndarray, radar_unavailable: bool) -> dict:
    """Lead columns from the native radarState samples, then held (max age LEAD_MAX_AGE_S): lead_valid (tocf.lead.lead_valid
    and fresh), lead_d_m, lead_y_m, lead_v_rel_mps, lead_speed_mps (= v + vRel), lead_a_slope_mps2 (tocf.lead.al_slope within
    tocf.lead.lead_episodes on the native samples; the radar's vLead, else ego speed + vRel when the field is NaN),
    lead_source (tocf.lead.lead_source with the route's radarUnavailable), lead_prob, thw_s, ttc_s, lead2_*, qc_lead_age_s;
    plus n_radarState, lead_radar_frac, lead_vision_frac."""
    n = len(t)
    out = dict(lead_valid=np.zeros(n, dtype=bool), lead_d_m=np.full(n, np.nan), lead_y_m=np.full(n, np.nan),
               lead_v_rel_mps=np.full(n, np.nan), lead_speed_mps=np.full(n, np.nan), lead_a_slope_mps2=np.full(n, np.nan),
               lead_source=np.zeros(n, dtype=np.int64), lead_prob=np.full(n, np.nan), thw_s=np.full(n, np.nan),
               ttc_s=np.full(n, np.nan), lead2_valid=np.zeros(n, dtype=bool), lead2_d_m=np.full(n, np.nan),
               lead2_v_rel_mps=np.full(n, np.nan), qc_lead_age_s=np.full(n, np.nan), n_radarState=0,
               lead_radar_frac=float("nan"), lead_vision_frac=float("nan"))
    if radar is None or len(radar) == 0:
        return out
    from vtt.tocf_bridge import tocf
    L = tocf("lead")
    out["n_radarState"] = int(len(radar))
    tr = SG.rel_seconds(radar["logMonoTime"].to_numpy(), t0_ns)
    f = lambda c: radar[c].to_numpy(dtype="float64")  # noqa: E731
    d, y, vrel, vlead, rflag, tid, prob = (f(c) for c in ("leadOne_dRel", "leadOne_yRel", "leadOne_vRel", "leadOne_vLead",
                                                            "leadOne_radar", "leadOne_radarTrackId", "leadOne_modelProb"))
    valid_r = L.lead_valid(f("leadOne_status"), d)
    valid2_r = L.lead_valid(f("leadTwo_status"), f("leadTwo_dRel"))
    ep, _ = L.lead_episodes(tr, valid_r, d, y, rflag, tid, valid2_r, f("leadTwo_radarTrackId"), L.radar_gap_thr(tr))
    jv, _, fv = SG.hold(t, tr, np.inf)                      # ego speed at the radar sample (causal, for the vLead fallback)
    v_at_r = SG.hold_float(v, jv, fv)
    vlead_src = np.where(np.isfinite(vlead), vlead, v_at_r + vrel)
    slope_r = L.al_slope(tr, vlead_src, ep)
    src_r = pd.Series(L.lead_source(valid_r, rflag, bool(radar_unavailable))).map(LEAD_SOURCE_CODES).to_numpy(dtype=np.int64)

    j, age, fresh = SG.hold(tr, t, C.LEAD_MAX_AGE_S)
    valid = fresh & valid_r[j]
    held = lambda a: np.where(valid, a[j], np.nan)  # noqa: E731
    out["lead_valid"] = valid
    out["lead_d_m"] = held(d)
    out["lead_y_m"] = held(y)
    out["lead_v_rel_mps"] = held(vrel)
    out["lead_speed_mps"] = v + out["lead_v_rel_mps"]
    out["lead_a_slope_mps2"] = held(slope_r)
    out["lead_source"] = np.where(fresh, src_r[j], 0)
    out["lead_prob"] = np.where(fresh, prob[j], np.nan)
    out["qc_lead_age_s"] = age
    thw = np.full(n, np.nan)
    m = valid & (v > C.THW_V_MIN_MPS) & (out["lead_d_m"] > C.LEAD_VALID_DREL_MIN_M)
    np.divide(out["lead_d_m"], v, out=thw, where=m)
    out["thw_s"] = thw
    closing = -out["lead_v_rel_mps"]
    ttc = np.full(n, np.nan)
    m = valid & (closing > C.TTC_CLOSING_MIN_MPS)
    np.divide(out["lead_d_m"], closing, out=ttc, where=m)
    out["ttc_s"] = ttc
    valid2 = fresh & valid2_r[j]
    out["lead2_valid"] = valid2
    out["lead2_d_m"] = np.where(valid2, f("leadTwo_dRel")[j], np.nan)
    out["lead2_v_rel_mps"] = np.where(valid2, f("leadTwo_vRel")[j], np.nan)
    nv = int(valid.sum())
    out["lead_radar_frac"] = float((out["lead_source"][valid] == 1).sum() / nv) if nv else float("nan")
    out["lead_vision_frac"] = float((out["lead_source"][valid] == 2).sum() / nv) if nv else float("nan")
    return out


# ----------------------------------------------------------------------------- the drive
def build_drive(sig: dict, meta: dict) -> tuple[pd.DataFrame, dict]:
    """(drive frame with the DRIVE_COLUMNS names, info dict) for one route from its normalised signal frames
    (vtt.signals.read_signals) and route_meta.json. Raises ValueError when there is no carState sample."""
    cs = sig.get("carState")
    if cs is None or len(cs) == 0:
        raise ValueError("no carState samples")
    n = len(cs)
    t0_ns = int(cs["logMonoTime"].iloc[0])
    sig = dict(sig); sig["_t0_ns"] = t0_ns
    t = SG.rel_seconds(cs["logMonoTime"].to_numpy(), t0_ns)
    col = lambda c: cs[c].to_numpy(dtype="float64")  # noqa: E731
    flag = lambda c: SG.as_bool(cs[c].to_numpy())  # noqa: E731
    v = col("vEgo")
    df: dict[str, np.ndarray] = {}
    df["t_s"] = np.arange(n, dtype="float64") / C.NOMINAL_HZ
    df["seg"] = SG.seg_ordinal(cs["seg"].to_numpy())
    dt = np.diff(t, prepend=np.nan)
    df["gap"] = np.nan_to_num(dt, nan=0.0) > C.CS_GAP_S
    df["v_mps"] = v
    df["v_raw_mps"] = col("vEgoRaw")
    df["a_mps2"] = col("aEgo")
    df["standstill"] = flag("standstill")
    df["steer_angle_deg"] = col("steeringAngleDeg")
    df["steer_rate_degps"] = col("steeringRateDeg")
    steer_pressed = flag("steeringPressed")
    torque = col("steeringTorque")
    torque_alive = not dead_channel(torque, steer_pressed)
    scale, norm = torque_norm(torque, v, steer_pressed) if torque_alive else (float("nan"), np.full(n, np.nan))
    df["steer_torque_raw"] = torque if torque_alive else np.full(n, np.nan)
    df["steer_torque_norm"] = norm
    df["steer_pressed"] = steer_pressed

    pose = pose_columns(sig.get("pose"), t0_ns, t, v)
    for k in ("yaw_rate_radps", "yaw_valid", "pose_source", "curv_1pm", "lat_acc_mps2"):
        df[k] = pose[k]

    gas, gas_pressed = col("gas"), flag("gasPressed")
    brake, brake_pressed = col("brake"), flag("brakePressed")
    gas_alive = not dead_channel(gas, gas_pressed)
    brake_alive = not dead_channel(brake, brake_pressed)
    df["gas"] = gas if gas_alive else np.full(n, np.nan)
    df["gas_pressed"] = gas_pressed
    df["brake"] = brake if brake_alive else np.full(n, np.nan)
    df["brake_pressed"] = brake_pressed
    df["blinker_left"] = flag("leftBlinker")
    df["blinker_right"] = flag("rightBlinker")

    ctrl = control_columns(sig, t)
    cruise_enabled = flag("cruiseState_enabled")
    df["das_enabled"] = ctrl["ctrl_enabled"] | cruise_enabled
    df["das_any"] = df["das_enabled"] | ctrl["sds_enabled"] | ctrl["cc_enabled"] | ctrl["lat_active"] | ctrl["long_active"]
    df["op_lat_active"] = ctrl["lat_active"]
    df["op_long_active"] = ctrl["long_active"]
    df["cruise_available"] = flag("cruiseState_available")
    df["v_set_mps"] = set_speed(cs, ctrl)

    lead = lead_columns(sig.get("radarState"), t0_ns, t, v, bool(meta.get("radarUnavailable") or False))
    for k in ("lead_valid", "lead_d_m", "lead_y_m", "lead_v_rel_mps", "lead_speed_mps", "lead_a_slope_mps2", "lead_source",
              "lead_prob", "thw_s", "ttc_s", "lead2_valid", "lead2_d_m", "lead2_v_rel_mps", "qc_lead_age_s"):
        df[k] = lead[k]
    df["qc_yaw_age_s"] = pose["qc_yaw_age_s"]
    frame = pd.DataFrame(df, columns=S.names(S.DRIVE_COLUMNS))

    segs = meta.get("segments") or []
    stats = SG.dt_stats(t)
    nv = int(frame["lead_valid"].sum())
    info = dict(
        n_rows=n, n_carState=n, n_segments=int(len(np.unique(frame["seg"][frame["seg"] >= 0]))),
        n_seg_meta=_opt(meta.get("n_seg")), n_seg_rlog_dec=sum(1 for s in segs if s.get("src_kind") == "rlog_dec"),
        n_seg_none=sum(1 for s in segs if s.get("src_kind") == "none"),
        unknown_union_n=int(sum(int(s.get("unknown_union_n") or 0) for s in segs)),
        gas_alive=gas_alive, brake_alive=brake_alive, torque_alive=torque_alive, steer_torque_scale=scale,
        has_pose=bool(pose["n_pose"] > 0 and bool(np.isfinite(frame["yaw_rate_radps"]).any())),
        pose_source=pose["pose_source_name"], n_pose=pose["n_pose"], n_radarState=lead["n_radarState"],
        moving_s=float((v >= C.MOVING_V_MPS).sum() / C.NOMINAL_HZ), das_s=float(frame["das_enabled"].sum() / C.NOMINAL_HZ),
        das_any_s=float(frame["das_any"].sum() / C.NOMINAL_HZ), human_s=float((~frame["das_any"]).sum() / C.NOMINAL_HZ),
        lead_valid_frac=_frac(frame["lead_valid"]),
        cf_frac=float((frame["thw_s"].to_numpy() <= C.THW_CF_S).sum() / nv) if nv else float("nan"),
        yaw_valid_frac=_frac(frame["yaw_valid"]), lead_radar_frac=lead["lead_radar_frac"], lead_vision_frac=lead["lead_vision_frac"],
        gas_pressed_n=int(gas_pressed.sum()), brake_pressed_n=int(brake_pressed.sum()), steer_pressed_n=int(steer_pressed.sum()),
        cruise_speed_max_mps=float(np.nanmax(col("cruiseState_speed"))) if np.isfinite(col("cruiseState_speed")).any() else float("nan"),
        v_set_valid_frac=_frac(np.isfinite(frame["v_set_mps"])), cruise_enabled_frac=_frac(cruise_enabled),
        ctrl_enabled_frac=_frac(ctrl["ctrl_enabled"]), sds_enabled_frac=_frac(ctrl["sds_enabled"]), cc_enabled_frac=_frac(ctrl["cc_enabled"]),
        n_carControl=ctrl["n_carControl"], n_controlsState=ctrl["n_controlsState"], n_selfdriveState=ctrl["n_selfdriveState"],
    )
    info.update(stats)
    return frame, info


# ----------------------------------------------------------------------------- index / QC rows
LABEL_PASSTHROUGH = ("brand", "model", "platform", "body", "size_code", "wheelbase_mm", "track_front_mm", "track_rear_mm",
                     "footprint_m2", "curb_mass_kg", "size_class", "powertrain", "powertrain_raw", "label_source", "label_note",
                     "region")


def index_row(info: dict, names: dict, labels: dict | None, meta: dict) -> dict:
    """One vtt.schema.DRIVES_INDEX_COLUMNS row. Labels come from build/routes_labels.parquet (None when the route has
    none); software_era from the labels frame, else from route_meta.version; device / radar / op_long from route_meta;
    pose_source from the drive's own held samples. Split columns stay None (S8)."""
    lab = labels or {}
    size_class = _opt(lab.get("size_class")); powertrain = _opt(lab.get("powertrain"))
    row = dict(vehicle_id=names["vehicle_id"], drive_id=names["drive_id"], group_id=_opt(names.get("group_id")),
               file=drive_rel_file(names["vehicle_id"], names["drive_id"]), n_rows=int(info["n_rows"]),
               duration_s=float(info["n_rows"]) / C.NOMINAL_HZ, n_segments=int(info["n_segments"]),
               n_seg_rlog_dec=int(info["n_seg_rlog_dec"]), n_gaps=int(info["n_gaps"]), max_gap_s=float(info["max_gap_s"]))
    for k in LABEL_PASSTHROUGH:
        row[k] = _opt(lab.get(k))
    row["software_era"] = _opt(lab.get("software_era")) or software_era(meta.get("version"))
    row["device_type"] = _opt(meta.get("device_type"))
    row["pose_source"] = info["pose_source"]
    row["radar_unavailable"] = _opt_bool(meta.get("radarUnavailable"))
    row["op_long"] = _opt_bool(meta.get("opLong"))
    for k in ("gas_alive", "brake_alive", "torque_alive", "has_pose"):
        row[k] = bool(info[k])
    for k in ("moving_s", "das_s", "das_any_s", "human_s", "lead_valid_frac", "cf_frac", "yaw_valid_frac", "steer_torque_scale"):
        row[k] = float(info[k])
    row.update(split_device=None, split_platform=None,
               task_size_eligible=size_class in C.SIZE_CLASSES,
               task_bev_ice_eligible=powertrain in C.MAIN_POWERTRAIN_TASK, low_support_cell=None)
    return row


def qc_row(info: dict, names: dict, meta: dict, route_uid: str, era: str) -> dict:
    """One PRIVATE_QC_COLUMNS row (real clock statistics + raw provenance; private root only)."""
    row = {k: None for k in PRIVATE_QC_COLUMNS}
    row.update(route_uid=route_uid, vehicle_id=names["vehicle_id"], drive_id=names["drive_id"])
    for k in PRIVATE_QC_COLUMNS:
        if k in info:
            row[k] = info[k]
    row.update(version=_opt(meta.get("version")), software_era=era, device_type=_opt(meta.get("device_type")),
               pose_source_meta=_opt(meta.get("pose_source")), pose_source=info["pose_source"],
               radar_unavailable=_opt_bool(meta.get("radarUnavailable")), op_long=_opt_bool(meta.get("opLong")),
               pcm_cruise=_opt_bool(meta.get("pcmCruise")), n_seg_table=int(info["n_segments"]),
               extractor_version=_opt(meta.get("extractor_version")), config_version=_opt(meta.get("config_version")),
               built_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    return row


# ----------------------------------------------------------------------------- worker
def build_route(route_uid: str, names: dict, labels: dict | None, data_root: Path | str) -> dict:
    """Build and write one drive: data/<vehicle_id>/<drive_id>.parquet under the build root. Returns a dict with
    status ('ok' | 'no_signals' | 'empty' | 'schema' | 'error'), index (row), qc (row), n_rows, err (masked text)."""
    roots = Roots(data_root)
    res = dict(route_uid=route_uid, drive_id=names.get("drive_id"), status="error", index=None, qc=None, n_rows=0, err="")
    sig_dir = roots.signals / route_uid
    if not (sig_dir / "carState.parquet").exists():
        res["status"] = "no_signals"
        return res
    try:
        sig = SG.read_signals(sig_dir)
        meta = SG.read_meta(sig_dir)
        try:
            frame, info = build_drive(sig, meta)
        except ValueError as e:
            res.update(status="empty", err=str(e))
            return res
        path = roots.drive_file(names["vehicle_id"], names["drive_id"])
        S.write_public_parquet(frame, path, S.DRIVE_COLUMNS, "drive")
        problems = S.check_public_parquet(path, S.DRIVE_COLUMNS, "drive")
        row = index_row(info, names, labels, meta)
        res.update(index=row, qc=qc_row(info, names, meta, route_uid, row["software_era"]), n_rows=len(frame),
                   status="ok" if not problems else "schema", err="; ".join(problems))
    except Exception as e:      # noqa: BLE001  one bad route never stops the pool; the text is masked before it is stored
        from vtt.private import mask
        res.update(status="error", err=mask(f"{type(e).__name__}: {e}")[:300])
    return res


# ----------------------------------------------------------------------------- relabel (S5 --relabel)
def relabel_index(index: pd.DataFrame, names: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Refresh the label-derived columns of an existing drives index from build/routes_labels.parquet without reading
    any drive table: LABEL_PASSTHROUGH, software_era (when the labels carry one) and the task_*_eligible flags.
    The split columns and low_support_cell are reset to None (S8 has to run again). `names` maps route_uid -> drive_id."""
    idx = index.copy()
    r2d = names[["route_uid", "drive_id"]].dropna()
    d2r = dict(zip(r2d["drive_id"], r2d["route_uid"]))
    lab = labels.drop_duplicates("route_uid").set_index("route_uid")
    lab_rows = {uid: row for uid, row in zip(lab.index, lab.to_dict("records"))}
    counts = dict(drives=int(len(idx)), relabelled=0, no_label=0)
    cols = list(LABEL_PASSTHROUGH)
    new: dict[str, list] = {c: [] for c in cols + ["software_era", "task_size_eligible", "task_bev_ice_eligible"]}
    eras = idx["software_era"].tolist() if "software_era" in idx.columns else [None] * len(idx)
    for did, era_old in zip(idx["drive_id"], eras):
        row = lab_rows.get(d2r.get(did))
        if row is None:
            counts["no_label"] += 1
            for c in cols:
                new[c].append(None)
            new["software_era"].append(_opt(era_old))
            new["task_size_eligible"].append(False)
            new["task_bev_ice_eligible"].append(False)
            continue
        counts["relabelled"] += 1
        for c in cols:
            new[c].append(_opt(row.get(c)))
        new["software_era"].append(_opt(row.get("software_era")) or _opt(era_old))
        sc, pt = _opt(row.get("size_class")), _opt(row.get("powertrain"))
        new["task_size_eligible"].append(sc in C.SIZE_CLASSES)
        new["task_bev_ice_eligible"].append(pt in C.MAIN_POWERTRAIN_TASK)
    for c, v in new.items():
        idx[c] = v
    for c in ("split_device", "split_platform", "low_support_cell"):
        idx[c] = None
    return idx, counts
