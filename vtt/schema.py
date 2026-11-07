"""Public table contracts of the release (column order, dtype, unit, description) and the parquet writer.

Column names are snake_case with unit suffixes. 'origin' names the upstream message field the value derives from.
Anything not listed here never reaches a public parquet (gate G3 checks exact column sets).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "1.0"

# (name, dtype, unit, description, origin)
DRIVE_COLUMNS: list[tuple[str, str, str, str, str]] = [
    ("t_s", "float64", "s", "nominal time from the first sample of the drive, = row_index / 10", "row index"),
    ("seg", "int16", "", "ordinal of the source 1-min segment within the drive (0-based)", "segment number"),
    ("gap", "bool", "", "True when more than 0.3 s elapsed since the previous sample (missing data before this row)", "sample times"),
    ("v_mps", "float32", "m/s", "ego speed", "carState.vEgo"),
    ("v_raw_mps", "float32", "m/s", "unfiltered ego speed", "carState.vEgoRaw"),
    ("a_mps2", "float32", "m/s^2", "ego longitudinal acceleration (vehicle estimate)", "carState.aEgo"),
    ("standstill", "bool", "", "vehicle reports standstill", "carState.standstill"),
    ("steer_angle_deg", "float32", "deg", "steering wheel angle, left positive", "carState.steeringAngleDeg"),
    ("steer_rate_degps", "float32", "deg/s", "steering wheel angle rate", "carState.steeringRateDeg"),
    ("steer_torque_raw", "float32", "raw", "driver steering torque in the vehicle's own CAN units (not comparable across makes)", "carState.steeringTorque"),
    ("steer_torque_norm", "float32", "", "steer_torque_raw divided by the drive's 95th percentile of |torque| while moving and steering", "derived"),
    ("steer_pressed", "bool", "", "driver torque above the vehicle's threshold", "carState.steeringPressed"),
    ("yaw_rate_radps", "float32", "rad/s", "body yaw rate, left positive (device IMU / localizer)", "livePose or liveLocationKalman angularVelocityDevice.z"),
    ("yaw_valid", "bool", "", "yaw sample fresh (<= 0.625 s) and flagged valid by the localizer", "derived"),
    ("pose_source", "int8", "", "0 none, 1 livePose (5 Hz), 2 liveLocationKalman (4 Hz)", "derived"),
    ("curv_1pm", "float32", "1/m", "path curvature = yaw_rate / v (v >= 1 m/s and yaw_valid, else NaN)", "derived"),
    ("lat_acc_mps2", "float32", "m/s^2", "lateral acceleration = v * yaw_rate", "derived"),
    ("gas", "float32", "0-1", "accelerator pedal position (NaN on drives where the channel is dead)", "carState.gas"),
    ("gas_pressed", "bool", "", "accelerator pedal pressed", "carState.gasPressed"),
    ("brake", "float32", "0-1", "brake pedal position / pressure (NaN on drives where the channel is dead)", "carState.brake"),
    ("brake_pressed", "bool", "", "brake pedal pressed", "carState.brakePressed"),
    ("blinker_left", "bool", "", "left turn signal", "carState.leftBlinker"),
    ("blinker_right", "bool", "", "right turn signal", "carState.rightBlinker"),
    ("das_enabled", "bool", "", "driving automation engaged: controller enabled OR stock cruise enabled", "controlsState.enabled | carState.cruiseState.enabled"),
    ("das_any", "bool", "", "das_enabled OR any lateral/longitudinal automation activity flag", "das_enabled | selfdriveState.enabled | carControl.enabled | latActive | longActive"),
    ("op_lat_active", "bool", "", "automation steering active", "carControl.latActive"),
    ("op_long_active", "bool", "", "automation longitudinal control active", "carControl.longActive"),
    ("cruise_available", "bool", "", "cruise control available", "carState.cruiseState.available"),
    ("v_set_mps", "float32", "m/s", "cruise set speed (NaN when unset)", "carState.cruiseState.speed / vCruise"),
    ("lead_valid", "bool", "", "lead vehicle tracked (status and distance > 0.5 m, sample fresh <= 0.75 s)", "radarState.leadOne"),
    ("lead_d_m", "float32", "m", "longitudinal distance to lead", "radarState.leadOne.dRel"),
    ("lead_y_m", "float32", "m", "lateral offset of lead", "radarState.leadOne.yRel"),
    ("lead_v_rel_mps", "float32", "m/s", "lead speed minus ego speed", "radarState.leadOne.vRel"),
    ("lead_speed_mps", "float32", "m/s", "lead speed = v_mps + lead_v_rel_mps", "derived"),
    ("lead_a_slope_mps2", "float32", "m/s^2", "lead acceleration: causal 1 s least-squares slope of lead speed within the lead episode", "derived"),
    ("lead_source", "int8", "", "0 none, 1 radar track, 2 camera model track", "radarState.leadOne.radar / carParams.radarUnavailable"),
    ("lead_prob", "float32", "", "model probability of the lead (NaN on older software)", "radarState.leadOne.modelProb"),
    ("thw_s", "float32", "s", "time headway = lead_d_m / v_mps (v > 0.5 m/s)", "derived"),
    ("ttc_s", "float32", "s", "time to collision = lead_d_m / closing speed (closing > 0.1 m/s)", "derived"),
    ("lead2_valid", "bool", "", "second lead tracked", "radarState.leadTwo"),
    ("lead2_d_m", "float32", "m", "distance to second lead", "radarState.leadTwo.dRel"),
    ("lead2_v_rel_mps", "float32", "m/s", "second lead relative speed", "radarState.leadTwo.vRel"),
    ("qc_lead_age_s", "float32", "s", "age of the held lead sample (provenance, not a model input)", "derived"),
    ("qc_yaw_age_s", "float32", "s", "age of the held yaw sample (provenance, not a model input)", "derived"),
]

DRIVES_INDEX_COLUMNS: list[tuple[str, str, str]] = [   # (name, dtype, description)
    ("vehicle_id", "str", "pseudonymous vehicle (device x platform), random order"),
    ("drive_id", "str", "pseudonymous drive (one ignition cycle / openpilot route), random order"),
    ("group_id", "str", "identity cluster used as the unit of all splits"),
    ("file", "str", "relative path of the drive table"),
    ("n_rows", "int32", "rows of the drive table (10 Hz samples)"),
    ("duration_s", "float32", "nominal duration = n_rows / 10"),
    ("n_segments", "int16", "source segments"),
    ("n_seg_rlog_dec", "int16", "segments decimated from the high-rate log (no low-rate log present)"),
    ("n_gaps", "int16", "rows flagged gap"),
    ("max_gap_s", "float32", "largest gap"),
    ("brand", "str", "manufacturer"),
    ("model", "str", "base model"),
    ("platform", "str", "canonical vehicle platform (generation) name"),
    ("body", "str", "body style"),
    ("size_code", "str", "legacy US segment code"),
    ("wheelbase_mm", "float32", "wheelbase"),
    ("track_front_mm", "float32", "front track width"),
    ("track_rear_mm", "float32", "rear track width"),
    ("footprint_m2", "float32", "wheelbase x mean track"),
    ("curb_mass_kg", "float32", "curb mass (representative)"),
    ("size_class", "str", "SMALL / MID / LARGE by footprint thresholds (null when unknown)"),
    ("powertrain", "str", "ICE / HEV / PHEV / BEV (null when the platform is shared by several powertrains)"),
    ("powertrain_raw", "str", "7-value platform code incl. MIXED / UNK / FCEV"),
    ("label_source", "str", "how the platform was identified"),
    ("label_note", "str", "known label-noise notes"),
    ("region", "str", "north_america / europe / asia / other_unknown"),
    ("software_era", "str", "coarse software generation"),
    ("device_type", "str", "logging device hardware class"),
    ("pose_source", "str", "livePose / liveLocationKalman / mixed / none"),
    ("radar_unavailable", "bool", "lead comes from the camera model (True) or radar (False)"),
    ("op_long", "bool", "automation stack controlled longitudinal motion when engaged"),
    ("gas_alive", "bool", "gas magnitude channel populated"),
    ("brake_alive", "bool", "brake magnitude channel populated"),
    ("torque_alive", "bool", "steering torque channel populated"),
    ("has_pose", "bool", "yaw-rate stream present"),
    ("moving_s", "float32", "seconds with v >= 1 m/s"),
    ("das_s", "float32", "seconds with das_enabled"),
    ("das_any_s", "float32", "seconds with das_any"),
    ("human_s", "float32", "seconds without das_any"),
    ("lead_valid_frac", "float32", "share of samples with a valid lead"),
    ("cf_frac", "float32", "share of valid-lead samples with THW <= 6 s"),
    ("yaw_valid_frac", "float32", "share of samples with a valid yaw sample"),
    ("steer_torque_scale", "float32", "divisor used for steer_torque_norm"),
    ("split_device", "str", "train / validation / test (device-disjoint)"),
    ("split_platform", "str", "train / validation / test / excluded (headline: test platforms never appear in train or validation; "
                              "excluded = the drive's group also drove a held-out platform, the drive belongs to no split)"),
    ("task_size_eligible", "bool", "size_class known"),
    ("task_bev_ice_eligible", "bool", "powertrain in {BEV, ICE}"),
    ("low_support_cell", "bool", "cell has too few groups / platforms / hours for headline metrics"),
]

WINDOWS_INDEX_COLUMNS: list[tuple[str, str, str]] = [
    ("window_id", "str", "<drive_id>_w<index>"),
    ("drive_id", "str", ""), ("vehicle_id", "str", ""), ("group_id", "str", ""),
    ("w_idx", "int32", "window index within the drive"),
    ("row0", "int32", "first row (inclusive)"), ("row1", "int32", "last row (exclusive)"),
    ("t0_s", "float32", "nominal start time"), ("span_s", "float32", "nominal span of the window's samples = (rows - 1) x 0.1 s (59.9 s for a full window); the 60-s guarantee is gap_count == 0"),
    ("eligible", "bool", "600 samples, no gap row inside the window, moving fraction >= 0.2"),
    ("inelig_reason", "str", ""),
    ("kind", "str", "human / das / mixed by das_any"),
    ("das_frac", "float32", ""), ("das_any_frac", "float32", ""), ("op_long_frac", "float32", ""), ("op_lat_frac", "float32", ""),
    ("moving_frac", "float32", ""), ("standstill_frac", "float32", ""), ("gap_count", "int16", ""),
    ("v_mean", "float32", ""), ("v_max", "float32", ""), ("v_std", "float32", ""),
    ("a_min", "float32", ""), ("a_max", "float32", ""), ("a_std", "float32", ""),
    ("n_accel_ep", "int16", "acceleration episodes: monotone runs of the smoothed speed (0.3 m/s tolerance) lasting >= 3 s with dv >= 2 m/s and mean acceleration >= 0.1 m/s^2; a run includes the steady cruise around a manoeuvre, which dilutes the mean of a short acceleration inside a long cruise"),
    ("n_decel_ep", "int16", "deceleration episodes, same rule with the sign reversed"),
    ("lead_frac", "float32", ""), ("cf_frac", "float32", "THW <= 6 s share of valid-lead samples"),
    ("thw_med", "float32", ""), ("lead_speed_mean", "float32", ""), ("n_lead_accel_events", "int16", ""),
    ("yaw_valid_frac", "float32", ""), ("yaw_abs_mean", "float32", ""), ("curv_abs_mean", "float32", ""),
    ("steer_pressed_frac", "float32", ""), ("gas_pressed_frac", "float32", ""), ("brake_pressed_frac", "float32", ""),
    ("human_only", "bool", "no das_any sample in the window"),
    ("human_only_strict", "bool", "human_only and no das_any in the preceding 10 s"),
    ("size_class", "str", ""), ("powertrain", "str", ""), ("powertrain_raw", "str", ""), ("platform", "str", ""),
    ("brand", "str", ""), ("model", "str", ""), ("body", "str", ""), ("size_code", "str", ""), ("footprint_m2", "float32", ""),
    ("split_device", "str", ""), ("split_platform", "str", ""),
    ("task_size_eligible", "bool", ""), ("task_bev_ice_eligible", "bool", ""), ("low_support_cell", "bool", ""),
]

QC_COLUMNS = tuple(c for c, *_ in DRIVE_COLUMNS if c.startswith("qc_"))
PARQUET_METADATA_KEYS = (b"vtt_table", b"vtt_schema_version")

_PA = {"float64": pa.float64(), "float32": pa.float32(), "int8": pa.int8(), "int16": pa.int16(), "int32": pa.int32(),
       "int64": pa.int64(), "bool": pa.bool_(), "str": pa.string()}


def names(spec) -> list[str]:
    return [t[0] for t in spec]


def dtypes(spec) -> dict[str, str]:
    return {t[0]: t[1] for t in spec}


def arrow_schema(spec, table_name: str) -> pa.Schema:
    fields = [pa.field(t[0], _PA[t[1]], nullable=True) for t in spec]
    return pa.schema(fields, metadata={b"vtt_table": table_name.encode(), b"vtt_schema_version": SCHEMA_VERSION.encode()})


def coerce(df: pd.DataFrame, spec) -> pd.DataFrame:
    """Order and cast a frame to a spec; missing columns are added as null; extra columns raise."""
    extra = [c for c in df.columns if c not in dtypes(spec)]
    if extra:
        raise ValueError(f"columns not in schema: {extra}")
    out = pd.DataFrame(index=df.index)
    for name, dt in dtypes(spec).items():
        if name not in df.columns:
            col = pd.Series([None] * len(df), index=df.index)
        else:
            col = df[name]
        if dt == "bool":
            out[name] = np.asarray([bool(x) if (x is not None and x == x) else False for x in pd.Series(col).to_numpy(dtype=object)], dtype=bool)
        elif dt.startswith("int"):
            out[name] = pd.to_numeric(col, errors="coerce").fillna(-1).astype(dt).to_numpy()
        elif dt.startswith("float"):
            out[name] = pd.to_numeric(col, errors="coerce").astype(dt).to_numpy()
        else:
            out[name] = col.astype("object").where(pd.notna(col), None).to_numpy()
    return out


def write_public_parquet(df: pd.DataFrame, path: Path, spec, table_name: str, row_group_size: int = 6000) -> Path:
    """Write with the exact schema and only the two allowed metadata keys (no pandas metadata, no index)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = coerce(df, spec)
    table = pa.Table.from_pandas(out, schema=arrow_schema(spec, table_name), preserve_index=False, safe=False)
    table = table.replace_schema_metadata({b"vtt_table": table_name.encode(), b"vtt_schema_version": SCHEMA_VERSION.encode()})
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp, compression="zstd", row_group_size=row_group_size)
    tmp.replace(path)
    return path


def check_public_parquet(path: Path, spec, table_name: str) -> list[str]:
    """Return a list of schema problems (empty = OK)."""
    problems: list[str] = []
    sch = pq.read_schema(path)
    if sch.names != names(spec):
        problems.append(f"columns differ: {sch.names[:5]}... vs spec")
    meta = sch.metadata or {}
    if set(meta.keys()) != set(PARQUET_METADATA_KEYS):
        problems.append(f"metadata keys {sorted(meta.keys())}")
    if meta.get(b"vtt_table", b"").decode() != table_name:
        problems.append("vtt_table mismatch")
    return problems
