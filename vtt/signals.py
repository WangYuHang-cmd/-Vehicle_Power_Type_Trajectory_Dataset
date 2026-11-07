"""Read and normalise the extractor's per-route signal tables for the drive builder (stage S5).

Input: ``work/signals/<route_uid>/{carState,carControl,controlsState,selfdriveState,radarState,pose}.parquet`` as written
by the ADAS-TO-CF extractor (``tocf.schema.SIGNAL_COLUMNS``): ``logMonoTime`` int64 ns on the device boot clock, ``seg``
int64 source segment number, then the topic's flat fields. Dtypes drift across routes (an all-missing column is float64,
an older schema lacks a field entirely, bools may come back as float), so every frame is normalised here:

* every topic column of the contract exists (missing -> NaN),
* numeric and boolean fields are float64 (booleans 0.0 / 1.0, NaN = unknown), string fields are object,
* rows are sorted by ``logMonoTime`` (stable), never by ``seg``; rows without a time are dropped.

Also here: the causal zero-order hold used to put every other topic onto the carState sample times, the sample-time
statistics (real dt median / percentiles / drift, private QC only) and the segment ordinal.
Units: times are seconds unless a name ends in ``_ns``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vtt import config as C

TOPICS = ("carState", "carControl", "controlsState", "selfdriveState", "radarState", "pose")
STRING_FIELDS = frozenset({"state", "personality", "pose_source"})
# mirror of tocf.schema.SIGNAL_COLUMNS for the six topics S5 reads (tests assert it matches the extractor's list)
_SIGNAL_COLUMNS_MIRROR = {
    "carState": ["vEgo", "aEgo", "vEgoRaw", "standstill", "steeringAngleDeg", "steeringRateDeg", "steeringTorque",
                 "steeringPressed", "gas", "gasPressed", "brake", "brakePressed", "leftBlinker", "rightBlinker",
                 "cruiseState_enabled", "cruiseState_available", "cruiseState_speed", "vCruise", "vCruiseCluster"],
    "carControl": ["enabled", "latActive", "longActive"],
    "controlsState": ["enabled", "active", "state", "personality", "experimentalMode", "vCruise", "vCruiseCluster"],
    "selfdriveState": ["enabled", "active", "state", "personality", "experimentalMode", "vCruise", "vCruiseCluster"],
    "pose": ["angularVelocityDevice_x", "angularVelocityDevice_y", "angularVelocityDevice_z", "angularVelocityDevice_valid",
             "inputsOK", "posenetOK", "pose_source"],
    "radarState": ["leadOne_status", "leadOne_dRel", "leadOne_yRel", "leadOne_vRel", "leadOne_aRel", "leadOne_vLead",
                   "leadOne_radar", "leadOne_radarTrackId", "leadOne_modelProb",
                   "leadTwo_status", "leadTwo_dRel", "leadTwo_yRel", "leadTwo_vRel", "leadTwo_radar", "leadTwo_radarTrackId"],
}
POSE_SOURCE_CODES = {"livePose": 1, "liveLocationKalman": 2}      # schema.DRIVE_COLUMNS pose_source: 0 none


def signal_columns(topic: str) -> list[str]:
    """Flat field names of a topic: the extractor's own list when tocf is importable, else the mirror above."""
    try:
        from vtt.tocf_bridge import tocf
        return list(tocf("schema").SIGNAL_COLUMNS[topic])
    except Exception:            # noqa: BLE001  tocf absent (e.g. a clone without the extractor): the mirror is the contract
        return list(_SIGNAL_COLUMNS_MIRROR[topic])


# ----------------------------------------------------------------------------- reading
def normalise(df: pd.DataFrame | None, topic: str) -> pd.DataFrame:
    """Return a frame with logMonoTime (int64 ns), seg (int64) and every contract column, typed as described in the module
    docstring and sorted by logMonoTime. An empty / None input gives an empty frame with the right columns."""
    cols = signal_columns(topic)
    if df is None:
        df = pd.DataFrame(columns=["logMonoTime", "seg"] + cols)
    out = pd.DataFrame(index=range(len(df)))
    t = pd.to_numeric(df["logMonoTime"], errors="coerce") if "logMonoTime" in df else pd.Series(np.nan, index=range(len(df)))
    keep = np.isfinite(t.to_numpy(dtype="float64"))
    out["logMonoTime"] = t.to_numpy(dtype="float64")
    out["seg"] = pd.to_numeric(df["seg"], errors="coerce").to_numpy(dtype="float64") if "seg" in df else np.zeros(len(df))
    for c in cols:
        if c not in df:
            out[c] = np.full(len(df), None, dtype=object) if c in STRING_FIELDS else np.full(len(df), np.nan)
        elif c in STRING_FIELDS:
            v = df[c]
            out[c] = v.astype(object).where(v.notna(), None).to_numpy()
        else:
            out[c] = _to_float(df[c])
    out = out[keep]
    out = out.sort_values("logMonoTime", kind="stable").reset_index(drop=True)
    out["logMonoTime"] = out["logMonoTime"].to_numpy().astype(np.int64)
    out["seg"] = np.nan_to_num(out["seg"].to_numpy(dtype="float64"), nan=-1.0).astype(np.int64)
    return out


def _to_float(v: pd.Series) -> np.ndarray:
    """Numeric / boolean / object column -> float64 with NaN for missing (True/False strings and bools -> 1.0 / 0.0)."""
    if v.dtype == bool:
        return v.to_numpy().astype(np.float64)
    if v.dtype.kind in "fiu":
        return v.to_numpy(dtype="float64")
    if str(v.dtype) == "boolean":
        return v.astype("float64").to_numpy(dtype="float64", na_value=np.nan)
    num = pd.to_numeric(v, errors="coerce")
    if num.notna().sum() == v.notna().sum():
        return num.to_numpy(dtype="float64")
    s = v.astype(object).map(lambda x: 1.0 if x is True or str(x).lower() == "true" else
                             0.0 if x is False or str(x).lower() == "false" else np.nan)
    return np.where(num.notna(), num.to_numpy(dtype="float64"), s.to_numpy(dtype="float64"))


def read_topic(sig_dir: Path, topic: str) -> pd.DataFrame | None:
    """Normalised frame of one topic, or None when the parquet is absent (tolerated: the drive gets NaN / False columns)."""
    p = Path(sig_dir) / f"{topic}.parquet"
    if not p.exists():
        return None
    return normalise(pd.read_parquet(p), topic)


def read_signals(sig_dir: Path) -> dict[str, pd.DataFrame | None]:
    """{topic: normalised frame or None} for every topic in TOPICS."""
    return {t: read_topic(sig_dir, t) for t in TOPICS}


def read_meta(sig_dir: Path) -> dict:
    """route_meta.json as a dict ({} when absent or unreadable)."""
    p = Path(sig_dir) / "route_meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:            # noqa: BLE001
        return {}


# ----------------------------------------------------------------------------- arrays
def as_bool(x) -> np.ndarray:
    """float (0/1/NaN) or bool array -> bool, NaN / None -> False."""
    a = np.asarray(x, dtype="float64") if not isinstance(x, np.ndarray) or x.dtype != bool else x
    return a if a.dtype == bool else (np.nan_to_num(a, nan=0.0) > 0.5)


def rel_seconds(ns, t0_ns: int) -> np.ndarray:
    """Boot-clock nanoseconds -> float seconds relative to t0_ns (the first carState sample of the drive)."""
    return (np.asarray(ns, dtype=np.int64) - np.int64(t0_ns)) / 1e9


def seg_ordinal(seg) -> np.ndarray:
    """Ordinal (0-based) of each row's source segment by first appearance in time order; -1 stays -1."""
    seg = np.asarray(seg)
    out = np.full(len(seg), -1, dtype=np.int64)
    order: dict = {}
    for i, s in enumerate(seg.tolist()):
        if s < 0:
            continue
        if s not in order:
            order[s] = len(order)
        out[i] = order[s]
    return out


# ----------------------------------------------------------------------------- causal zero-order hold
def zoh_index(t_src, t_q) -> np.ndarray:
    """Index of the last source sample with t_src <= t_q (-1 before the first). t_src must be sorted."""
    return np.searchsorted(np.asarray(t_src, dtype="float64"), np.asarray(t_q, dtype="float64"), side="right") - 1


def hold(t_src, t_q, max_age_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal sample-and-hold bookkeeping onto the query times: (index clipped to [0, n-1], age in s, fresh).
    fresh = a source sample exists at or before the query and is at most max_age_s old. Age is NaN where not fresh.
    The same semantics as tocf.engage.zoh_last (a NaN source value is held as NaN: an invalid observation is still the
    latest observation)."""
    t_src = np.asarray(t_src, dtype="float64"); t_q = np.asarray(t_q, dtype="float64")
    n = len(t_q)
    if len(t_src) == 0:
        return np.zeros(n, dtype=np.int64), np.full(n, np.nan), np.zeros(n, dtype=bool)
    idx = zoh_index(t_src, t_q)
    j = np.clip(idx, 0, len(t_src) - 1)
    age = t_q - t_src[j]
    fresh = (idx >= 0) & (age <= max_age_s + 1e-9)
    age = np.where(fresh, age, np.nan)
    return j, age, fresh


def hold_float(values, j, fresh) -> np.ndarray:
    """Held float values: values[j] where fresh, else NaN."""
    v = np.asarray(values, dtype="float64")
    if len(v) == 0:
        return np.full(len(j), np.nan)
    return np.where(fresh, v[j], np.nan)


def hold_bool(values, j, fresh) -> np.ndarray:
    """Held boolean values: values[j] (NaN -> False) where fresh, else False."""
    v = as_bool(values)
    if len(v) == 0:
        return np.zeros(len(j), dtype=bool)
    return np.where(fresh, v[j], False)


# ----------------------------------------------------------------------------- sample-time statistics (private QC)
def dt_stats(t_rel, gap_s: float = C.CS_GAP_S, nominal_dt_s: float = C.NOMINAL_DT_S) -> dict:
    """Real inter-sample statistics of the carState timeline (seconds). Keys: dt_med_s, dt_p01_s, dt_p99_s, dt_max_s,
    real_span_s, nominal_span_s (= (n-1) * nominal dt), gap_total_s (time missing inside gap rows: sum of dt - nominal
    over rows flagged gap), drift_s (= real_span_s - gap_total_s - nominal_span_s: how far the nominal 10 Hz clock runs
    from the real clock once gaps are excluded), n_gaps, max_gap_s."""
    t = np.asarray(t_rel, dtype="float64")
    n = len(t)
    if n < 2:
        return dict(dt_med_s=np.nan, dt_p01_s=np.nan, dt_p99_s=np.nan, dt_max_s=np.nan, real_span_s=0.0,
                    nominal_span_s=0.0, gap_total_s=0.0, drift_s=np.nan, n_gaps=0, max_gap_s=0.0)
    dt = np.diff(t)
    gap = dt > gap_s
    gap_total = float(np.sum(dt[gap] - nominal_dt_s)) if gap.any() else 0.0
    real_span = float(t[-1] - t[0]); nominal_span = (n - 1) * nominal_dt_s
    return dict(dt_med_s=float(np.median(dt)), dt_p01_s=float(np.quantile(dt, 0.01)), dt_p99_s=float(np.quantile(dt, 0.99)),
                dt_max_s=float(dt.max()), real_span_s=real_span, nominal_span_s=float(nominal_span), gap_total_s=gap_total,
                drift_s=float(real_span - gap_total - nominal_span), n_gaps=int(gap.sum()),
                max_gap_s=float(dt[gap].max()) if gap.any() else 0.0)
