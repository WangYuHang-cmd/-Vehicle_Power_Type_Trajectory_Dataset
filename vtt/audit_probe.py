"""S2b: label-audit probe of a raw route (private/route_audit.parquet).

Reads the first segments' logs of a route with tocf.capnp_io.iter_events (no openpilot LogReader) and summarises what the
car itself reports about its powertrain and drivetrain: carParams.carFw ECU names (a 'hybrid' or 'transmission' ECU),
transmissionType, mass / wheelbase / steerRatio, and carState engineRpm / regenBraking / charging / fuelGauge / gearShifter /
yawRate / steeringTorqueEps statistics. Fields a writer did not have (wire layout, capnp_io.present_fields) are NaN / null,
never a capnp default. The table is private: it carries route_uid only, but the raw route directory is resolved from the
private uid_map. Units: mass kg, wheelbase m, speeds m/s, engineRpm rpm, fuelGauge 0..1, fractions 0..1.

One writer per route is assumed (a route is one openpilot session, one software version): the carState wire layout is taken
from the first carState message of the route and applied to every sample, and carParams is read once, from the first segment
that holds one (every segment's log head repeats it). A field present in a later segment only would stay NaN.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from vtt.private import write_parquet_atomic
from vtt.tocf_bridge import tocf

MOVING_V_MPS = 5.0                 # engine_off_moving_frac: samples faster than this
CARPARAMS_FIELDS = ("carFw", "transmissionType", "mass", "wheelbase", "steerRatio")
CARSTATE_FIELDS = ("vEgo", "engineRpm", "regenBraking", "charging", "fuelGauge", "gearShifter", "yawRate", "steeringTorqueEps")
AUDIT_SCHEMA = pa.schema([
    ("route_uid", pa.string()), ("n_segments_read", pa.int64()), ("ecus", pa.string()), ("has_hybrid_ecu", pa.bool_()),
    ("has_transmission_ecu", pa.bool_()), ("transmission", pa.string()), ("mass_kg", pa.float64()), ("wheelbase_m", pa.float64()),
    ("steer_ratio", pa.float64()), ("engine_rpm_pos_frac", pa.float64()), ("engine_off_moving_frac", pa.float64()),
    ("regen_any", pa.bool_()), ("charging_any", pa.bool_()), ("fuel_gauge_med", pa.float64()), ("gear_values", pa.string()),
    ("yaw_rate_nonzero_frac", pa.float64()), ("eps_torque_nonzero_frac", pa.float64()), ("n_samples", pa.int64()), ("err", pa.string()),
])
AUDIT_COLUMNS = AUDIT_SCHEMA.names


def empty_row(route_uid: str = "", err: str = "") -> dict:
    """A row with every measurement null (route unreadable or nothing logged)."""
    row = {c: None for c in AUDIT_COLUMNS}
    row.update(route_uid=route_uid, n_segments_read=0, n_samples=0, err=err)
    for c in ("mass_kg", "wheelbase_m", "steer_ratio", "engine_rpm_pos_frac", "engine_off_moving_frac", "fuel_gauge_med",
              "yaw_rate_nonzero_frac", "eps_torque_nonzero_frac"):
        row[c] = np.nan
    return row


# ----------------------------------------------------------------------------- carParams
def carparams_fields(cp) -> dict:
    """ECU names (JSON list, sorted unique), hybrid / transmission ECU flags, transmissionType name and the CarSpecs numbers of
    one carParams reader; a field outside the writer's layout stays null / NaN."""
    K = tocf("capnp_io")
    present = K.present_fields(cp)
    out = {"ecus": None, "has_hybrid_ecu": None, "has_transmission_ecu": None, "transmission": None,
           "mass_kg": np.nan, "wheelbase_m": np.nan, "steer_ratio": np.nan}
    if "carFw" in present:
        ecus = sorted({str(K.to_py(f.ecu)) for f in cp.carFw})
        out.update(ecus=json.dumps(ecus), has_hybrid_ecu="hybrid" in ecus, has_transmission_ecu="transmission" in ecus)
    if "transmissionType" in present:
        out["transmission"] = str(K.to_py(cp.transmissionType))
    for name, col in (("mass", "mass_kg"), ("wheelbase", "wheelbase_m"), ("steerRatio", "steer_ratio")):
        if name in present:
            out[col] = float(getattr(cp, name))
    return out


# ----------------------------------------------------------------------------- carState
def carstate_present(cs) -> set[str]:
    """CARSTATE_FIELDS inside this writer's carState layout (computed once per route: one writer per route)."""
    present = tocf("capnp_io").present_fields(cs)
    return {f for f in CARSTATE_FIELDS if f in present}


def carstate_sample(cs, present: set[str]) -> dict:
    """Plain-python values of the present CARSTATE_FIELDS of one message."""
    K = tocf("capnp_io")
    out = {}
    for f in present:
        v = getattr(cs, f)
        out[f] = str(K.to_py(v)) if f == "gearShifter" else v
    return out


def carstate_stats(samples: list[dict], present: set[str]) -> dict:
    """Statistics over the collected samples; a field absent from the writer -> NaN / null.
    engine_off_moving_frac = share of samples with vEgo > MOVING_V_MPS whose engineRpm == 0 (NaN without moving samples)."""
    out = {"engine_rpm_pos_frac": np.nan, "engine_off_moving_frac": np.nan, "regen_any": None, "charging_any": None,
           "fuel_gauge_med": np.nan, "gear_values": None, "yaw_rate_nonzero_frac": np.nan, "eps_torque_nonzero_frac": np.nan,
           "n_samples": len(samples)}
    if not samples:
        return out
    col = lambda f: np.asarray([s[f] for s in samples])   # noqa: E731
    if "engineRpm" in present:
        rpm = col("engineRpm").astype(float)
        out["engine_rpm_pos_frac"] = float(np.mean(rpm > 0))
        if "vEgo" in present:
            moving = col("vEgo").astype(float) > MOVING_V_MPS
            out["engine_off_moving_frac"] = float(np.mean(rpm[moving] == 0)) if moving.any() else np.nan
    if "regenBraking" in present:
        out["regen_any"] = bool(col("regenBraking").astype(bool).any())
    if "charging" in present:
        out["charging_any"] = bool(col("charging").astype(bool).any())
    if "fuelGauge" in present:
        out["fuel_gauge_med"] = float(np.median(col("fuelGauge").astype(float)))
    if "gearShifter" in present:
        out["gear_values"] = json.dumps(sorted({str(s["gearShifter"]) for s in samples}))
    if "yawRate" in present:
        out["yaw_rate_nonzero_frac"] = float(np.mean(col("yawRate").astype(float) != 0))
    if "steeringTorqueEps" in present:
        out["eps_torque_nonzero_frac"] = float(np.mean(col("steeringTorqueEps").astype(float) != 0))
    return out


# ----------------------------------------------------------------------------- one route
def probe_route(route_dir: Path, max_segs: int = 3) -> dict:
    """AUDIT_COLUMNS row (without route_uid) of a raw route: the first max_segs segments holding a log are read (qlog preferred,
    rlog when a segment has no qlog). A cut or unreadable segment contributes what was read; err names the last exception type
    or 'missing_dir' / 'no_logs'."""
    rawlib, K = tocf("rawlib"), tocf("capnp_io")
    route_dir = Path(route_dir)
    row = empty_row()
    if not route_dir.is_dir():
        row["err"] = "missing_dir"
        return row
    segs = rawlib.segment_table(route_dir)
    logs = [f["qlog"] or f["rlog"] for _, f in segs.items() if f["qlog"] or f["rlog"]][:max_segs]
    if not logs:
        row["err"] = "no_logs"
        return row
    cp_done, present, samples, errs = False, None, [], []
    for log in logs:
        try:
            for w, ev in K.iter_events(log):
                if w == "carState":
                    cs = ev.carState
                    if present is None:
                        present = carstate_present(cs)
                    samples.append(carstate_sample(cs, present))
                elif w == "carParams" and not cp_done:
                    row.update(carparams_fields(ev.carParams))
                    cp_done = True
        except Exception as e:                  # noqa: BLE001  keep what was read, remember the failure type
            errs.append(type(e).__name__)
        row["n_segments_read"] += 1
    row.update(carstate_stats(samples, present or set()))
    row["err"] = ",".join(errs)
    return row


# ----------------------------------------------------------------------------- table I/O
def audit_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    for c in ("n_segments_read", "n_samples"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    df["err"] = df["err"].fillna("").astype(str)
    return df


def read_audit(path: Path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else audit_frame([])


def merge_audit(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """New rows replace old rows of the same route_uid; sorted by route_uid."""
    if len(old):
        old = old[~old["route_uid"].isin(set(new["route_uid"]))]
    out = pd.concat([old, new], ignore_index=True) if len(old) else new.copy()
    return out.sort_values("route_uid", kind="stable").reset_index(drop=True)


def write_audit(df: pd.DataFrame, path: Path) -> Path:
    """Atomic private write with the explicit AUDIT_SCHEMA (nullable bools / strings survive an all-null column)."""
    out = audit_frame(df.to_dict("records")) if list(df.columns) != AUDIT_COLUMNS else df
    return write_parquet_atomic(out, Path(path), private=True, schema=AUDIT_SCHEMA)
