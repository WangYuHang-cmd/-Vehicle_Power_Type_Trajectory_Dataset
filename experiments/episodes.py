"""Event episodes of a drive: detection, anchor-aligned crops, speed-indexed grids and the per-episode index row.

Reads one public drive table (build/data/<vehicle_id>/<drive_id>.parquet, via vtt.windows.read_drive) and returns, per
drive, the arrays that experiments/episodes_build.py caches under work/episodes/ (contract (1) of the method-research
plan, Part II stage 3):
  * E (n, L_STORE = 180, 22) float16: 18-s crops of the 20 official input channels + a_sg (Savitzky-Golay acceleration,
    5 taps / order 2 / first derivative of v_mps) + valid; the anchor sits at sample 40 (STOP episodes: sample 140).
    Samples outside the drive, or separated from the anchor by a gap row, are NaN with valid = 0;
  * G (n_g, 72, 6) float16: ACC / DEC episodes re-indexed on the speed grid 0 .. 36 m/s (exclusive) step 0.5 with the
    channels [a_sg, gas_pressed, brake_pressed, das_enabled, valid, t_from_onset_s]; bins between the episode's minimum
    and maximum smoothed speed are interpolated, the others are NaN with valid = 0;
  * the index rows (INDEX_COLUMNS): ids, type_id, rows, kinematics, automation flags, human_strict, gap_in_crop,
    w_idx = anchor_row // 600 (window_id is joined at assembly) and grid_idx (-1 when the type has no grid).
No label or split column is produced: consumers join them from build/index/drives.parquet on drive_id.

Episode types (TYPES, type_id = position): ACC / DEC_LIFT / DEC_BRK from vtt.features_core.speed_episodes with the trim
rule (PARAMS["accel"]), the deceleration split by the brake_pressed fraction of the run; STOP = standstill run >= 1 s
preceded within 15 s by v > 2 m/s (anchor = standstill onset); LAUNCH = standstill run end followed by v > 2 m/s within
15 s (anchor = first moving sample); the look-back / look-ahead stops at the neighbouring qualifying standstill run
(stop_bound_at_runs), so stop-and-go traffic gives one STOP per approach and one LAUNCH per departure and no STOP /
LAUNCH interval contains another qualifying run; LEAD_ACC / LEAD_DEC = lead-acceleration onsets of the relaxed detector (quiet 2 s
with |slope| < 0.3, then slope >= slope_on held 1 s, up to 0.5 s of transition; re-implemented here so the experiments
never import the starter kit) in car-following range at the anchor (THW <= 6 s, or, while the ego stands (v <= 0.5 m/s,
THW undefined), lead distance <= lead_standstill_d_m and the ego starts moving within horizon_s: the lead pulling away
from a queue is the most frequent lead acceleration, and thw_anchor stays NaN there so consumers can still apply the
literal THW rule), >= 8 s apart; TURN (switchable) = runs
of |curv_1pm| >= 0.01 lasting >= 1 s at 2 <= v <= 12 m/s with a valid yaw sample. Detection runs inside gap-free
segments of the drive (a gap row marks missing data before it), so no episode spans a gap; crops may, hence gap_in_crop.
The anchor of an ACC / DEC episode is refined from the trimmed run start to the first sample of the run whose a_sg
reaches min_mean_a in the episode direction (the trim rule leads a ramp by up to half its 2-s slope window).
das_any is used only for the human_strict flag and the das_any_frac column, never as a stored channel.
Units: m/s, m/s^2, s; indices are 0-based rows of the drive table; [i0, i1) is half-open. Prints nothing.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vtt import config as C
from vtt import features_core as F
from vtt import windows as W

__all__ = ["TYPES", "TYPE_ID", "EP_CHANNELS", "GRID_CHANNELS", "L_STORE", "PRE_SAMPLES", "PRE_DEFAULT", "SPEED_GRID",
           "PARAMS", "INDEX_COLUMNS", "CONTEXT_COLUMNS", "Episode", "params_hash", "load_drive_arrays", "ensure_derived",
           "lead_onsets", "detect_episodes", "episode_flags", "episode_context", "context_matrix", "crop_episode",
           "speed_grid", "episode_row", "coerce_index", "extract_drive", "type_counts"]

TYPES: tuple[str, ...] = ("ACC", "DEC_LIFT", "DEC_BRK", "STOP", "LAUNCH", "LEAD_ACC", "LEAD_DEC", "TURN")
TYPE_ID: dict[str, int] = {t: i for i, t in enumerate(TYPES)}
GRID_TYPES: frozenset[str] = frozenset({"ACC", "DEC_LIFT", "DEC_BRK"})
EP_CHANNELS: tuple[str, ...] = tuple(C.OFFICIAL_INPUTS) + ("a_sg", "valid")
GRID_CHANNELS: tuple[str, ...] = ("a_sg", "gas_pressed", "brake_pressed", "das_enabled", "valid", "t_from_onset_s")
L_STORE = 180
PRE_DEFAULT = 40
PRE_SAMPLES: dict[str, int] = {"STOP": 140}
SPEED_GRID: np.ndarray = np.arange(0.0, 36.0, 0.5)              # 72 bins, 0 .. 35.5 m/s
F16_MAX = 65000.0                                                # float16 finite range guard (thw / ttc tails)
# columns read from the drive table: the stored channels plus the detection / flag inputs
READ_COLUMNS: tuple[str, ...] = tuple(C.OFFICIAL_INPUTS) + ("das_any", "gap", "yaw_valid")

PARAMS: dict = dict(
    dt=C.NOMINAL_DT_S,
    accel=dict(C.ACCEL_EP, trim_win_s=2.0),
    lead=dict(C.LEAD_ACCEL_EVENT, trans_s=0.5),
    lead_decel_slope_on=0.6,
    lead_min_spacing_s=8.0,
    thw_cf_s=C.THW_CF_S,
    lead_standstill_d_m=30.0,      # ego at <= THW_V_MIN_MPS (THW undefined): keep the onset when the lead is within this distance; 0 = off
    stop_min_standstill_s=1.0,
    stop_lookback_s=15.0,
    stop_v_min=2.0,
    stop_bound_at_runs=True,       # clip the STOP look-back / LAUNCH look-ahead at the neighbouring qualifying standstill run
    turn_enabled=True,
    turn_curv_min=0.01,
    turn_min_s=1.0,
    turn_v_range=(2.0, 12.0),
    lift_brake_frac_max=0.1,
    human_guard_s=C.HUMAN_GUARD_S,
)

# (name, dtype) of episodes.parquet; episode_id and grid_idx are local to the drive in extract_drive and made global at
# assembly; window_id is None until the assembly joins the windows index on (drive_id, w_idx)
INDEX_COLUMNS: list[tuple[str, str]] = [
    ("episode_id", "int64"), ("drive_id", "str"), ("vehicle_id", "str"), ("group_id", "str"), ("type_id", "int8"),
    ("anchor_row", "int32"), ("i0", "int32"), ("i1", "int32"), ("crop_row0", "int32"),
    ("v_anchor", "float32"), ("dv", "float32"), ("dur_s", "float32"), ("mean_a", "float32"), ("peak_a", "float32"),
    ("resp_delay_s", "float32"), ("das_frac", "float32"), ("das_any_frac", "float32"), ("human_strict", "bool"),
    ("lead_valid_frac", "float32"), ("thw_anchor", "float32"), ("gap_in_crop", "bool"), ("w_idx", "int32"),
    ("window_id", "str"), ("grid_idx", "int32"),
]
# the 9-float per-episode context vector of contract (1), in order (computed from the index row by the consumer)
CONTEXT_COLUMNS: tuple[str, ...] = ("v_anchor", "dv", "dur_s", "mean_a", "peak_a", "das_frac", "lead_valid_frac",
                                    "thw_anchor", "gap_in_crop")
_CONTEXT_SCALE = np.array([1 / 30.0, 1 / 10.0, 1 / 20.0, 1.0, 1.0, 1.0, 1.0, 1 / 10.0, 1.0])


@dataclass
class Episode:
    """One detected episode. Kinematics are of the ego over [i0, i1): dv = v_sg[i1 - 1] - v_sg[i0], dur_s = (i1 - i0) dt,
    mean_a = dv / dur_s, peak_a = the a_sg extreme in the episode direction (largest |a_sg| for TURN).
    resp_delay_s (LEAD_* only) = ego response delay after the onset (vtt.features_core.ego_response_delay on +-a_sg)."""
    type_id: int
    anchor_row: int
    i0: int
    i1: int
    v_anchor: float
    dv: float
    dur_s: float
    mean_a: float
    peak_a: float
    resp_delay_s: float = float("nan")
    thw_anchor: float = float("nan")

    @property
    def type(self) -> str:
        return TYPES[self.type_id]

    @property
    def pre(self) -> int:
        return PRE_SAMPLES.get(self.type, PRE_DEFAULT)

    @property
    def crop_row0(self) -> int:
        return self.anchor_row - self.pre


def params_hash(p: dict = PARAMS) -> str:
    """Short, stable digest of the detector parameters (8 hex characters; stored in the parts and params.json)."""
    import hashlib
    return hashlib.sha256(json.dumps(p, sort_keys=True, default=list).encode()).hexdigest()[:8]


# ----------------------------------------------------------------------------- reading and derived arrays
def load_drive_arrays(path: Path | str) -> dict[str, np.ndarray]:
    """The channels needed here as numpy arrays (vtt.windows.read_drive: bool -> bool, float -> float64, '__n__')."""
    return W.read_drive(Path(path), READ_COLUMNS)


def segments(d: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    """Gap-free [s0, s1) row ranges of the drive (a gap row starts a new segment; row 0 never does)."""
    n = int(d["__n__"][0])
    if n == 0:
        return []
    cuts = np.flatnonzero(d["gap"][1:]) + 1
    bounds = [0, *cuts.tolist(), n]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1) if bounds[k + 1] > bounds[k]]


def ensure_derived(d: dict[str, np.ndarray], p: dict = PARAMS) -> dict[str, np.ndarray]:
    """Add v_sg (smoothed speed) and a_sg (smoothed acceleration, m/s^2) computed inside each gap-free segment."""
    if "a_sg" in d and "v_sg" in d:
        return d
    n = int(d["__n__"][0])
    v_sg = np.full(n, np.nan)
    a_sg = np.full(n, np.nan)
    win, order = p["accel"]["sg_win"], p["accel"]["sg_order"]
    for s0, s1 in segments(d):
        if s1 - s0 < 2:                      # a one-row segment (gap row right after a gap) has no derivative
            v_sg[s0:s1] = d["v_mps"][s0:s1]
            a_sg[s0:s1] = np.nan
            continue
        v = d["v_mps"][s0:s1]
        if not np.isfinite(v).any():
            continue
        v_sg[s0:s1] = F.savgol_smooth(v, win, order)
        a_sg[s0:s1] = F.savgol_smooth(v, win, order, deriv=1, dt=p["dt"])
    d["v_sg"], d["a_sg"] = v_sg, a_sg
    return d


# ----------------------------------------------------------------------------- detectors
def _held(mask: np.ndarray, n: int) -> np.ndarray:
    """True at i when mask is True on every sample of [i, i + n)."""
    mask = np.asarray(mask, bool)
    if n <= 1:
        return mask.copy()
    c = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    out = np.zeros(len(mask), bool)
    k = len(mask) - n + 1
    if k > 0:
        out[:k] = (c[n:n + k] - c[:k]) == n
    return out


def lead_onsets(slope, lead_valid, dt: float, slope_on: float, hold_s: float, quiet_s: float, quiet_abs: float,
                trans_s: float = 0.5, **_ignored) -> np.ndarray:
    """Onset rows of lead accelerations on a slope trace (pass -slope for decelerations): slope >= slope_on on every
    sample of [i, i + hold_s) (i = the first sample of the hold), preceded by quiet_s of |slope| < quiet_abs ending at
    most trans_s before i, every sample with a valid lead. trans_s = 0 is the strict rule of vtt.features_core.
    Extra keyword arguments (the response constants) are ignored so PARAMS["lead"] can be passed whole."""
    s = np.asarray(slope, float)
    valid = np.asarray(lead_valid, bool)
    n_hold = max(1, int(round(hold_s / dt)))
    n_quiet = max(1, int(round(quiet_s / dt)))
    n_trans = max(0, int(round(trans_s / dt)))
    n = len(s)
    if n < n_quiet + n_hold:
        return np.zeros(0, np.int64)
    hi = _held(valid & (np.nan_to_num(s, nan=-np.inf) >= slope_on), n_hold)
    quiet = _held(valid & (np.nan_to_num(np.abs(s), nan=np.inf) < quiet_abs), n_quiet)
    idx = np.arange(n_quiet, n)
    first = hi[idx] & ~hi[idx - 1]                         # first sample of each qualifying hold
    ok = np.zeros(len(idx), bool)
    for g in range(n_trans + 1):                           # quiet window ending g samples before the hold
        j = idx - n_quiet - g
        m = j >= 0
        ok[m] |= quiet[j[m]]
    return idx[first & ok].astype(np.int64)


def _onset_in_run(a_sg: np.ndarray, i0: int, i1: int, sign: int, min_a: float) -> int:
    """First row of [i0, i1) where sign * a_sg >= min_a (the trimmed run start when none)."""
    seg = sign * np.nan_to_num(a_sg[i0:i1], nan=-np.inf)
    hit = np.flatnonzero(seg >= min_a)
    return int(i0 + hit[0]) if len(hit) else int(i0)


def _peak(a_sg: np.ndarray, i0: int, i1: int, sign: int) -> float:
    seg = a_sg[i0:i1]
    seg = seg[np.isfinite(seg)]
    if not len(seg):
        return float("nan")
    if sign > 0:
        return float(seg.max())
    if sign < 0:
        return float(seg.min())
    return float(seg[np.argmax(np.abs(seg))])


def _kinematics(d: dict[str, np.ndarray], i0: int, i1: int, dt: float) -> tuple[float, float, float]:
    """(dv, dur_s, mean_a) of the smoothed speed over [i0, i1)."""
    v_sg = d["v_sg"]
    dur = (i1 - i0) * dt
    dv = float(v_sg[i1 - 1] - v_sg[i0]) if i1 > i0 else float("nan")
    return dv, float(dur), (dv / dur if dur > 0 else float("nan"))


def _make(d: dict[str, np.ndarray], type_name: str, anchor: int, i0: int, i1: int, sign: int, p: dict,
          resp: float = float("nan")) -> Episode:
    dv, dur, mean_a = _kinematics(d, i0, i1, p["dt"])
    return Episode(type_id=TYPE_ID[type_name], anchor_row=int(anchor), i0=int(i0), i1=int(i1),
                   v_anchor=float(d["v_mps"][anchor]), dv=dv, dur_s=dur, mean_a=mean_a, peak_a=_peak(d["a_sg"], i0, i1, sign),
                   resp_delay_s=float(resp), thw_anchor=float(d["thw_s"][anchor]))


def _speed_episodes_seg(d: dict[str, np.ndarray], s0: int, s1: int, p: dict) -> list[Episode]:
    out: list[Episode] = []
    v = d["v_mps"][s0:s1]
    brake = d["brake_pressed"]
    min_a = p["accel"]["min_mean_a"]
    for sign in (1, -1):
        for i0, i1, _dv, _ma in F.speed_episodes(v, p["dt"], sign, **p["accel"]):
            i0, i1 = s0 + int(i0), s0 + int(i1)
            anchor = _onset_in_run(d["a_sg"], i0, i1, sign, min_a)
            if sign > 0:
                name = "ACC"
            else:
                name = "DEC_LIFT" if float(brake[i0:i1].mean()) < p["lift_brake_frac_max"] else "DEC_BRK"
            out.append(_make(d, name, anchor, i0, i1, sign, p))
    return out


def _stop_launch_seg(d: dict[str, np.ndarray], s0: int, s1: int, p: dict) -> list[Episode]:
    """STOP / LAUNCH episodes of the standstill runs >= stop_min_standstill_s inside [s0, s1). With stop_bound_at_runs the
    STOP look-back and the LAUNCH look-ahead (stop_lookback_s) are clipped at the neighbouring qualifying runs: one approach
    in stop-and-go traffic yields one STOP (at its first qualifying stop) and one LAUNCH (at its last), no STOP / LAUNCH
    interval contains another qualifying run, and a creep that exceeds stop_v_min between two stops separates them.
    Standstill blips shorter than the minimum inside an approach or a departure are hesitations, not stops."""
    out: list[Episode] = []
    dt = p["dt"]
    v = np.nan_to_num(d["v_mps"][s0:s1], nan=-np.inf)
    ss = d["standstill"][s0:s1]
    min_n = max(1, int(round(p["stop_min_standstill_s"] / dt)))
    look = int(round(p["stop_lookback_s"] / dt))
    v_min = p["stop_v_min"]
    bound = bool(p.get("stop_bound_at_runs", True))
    runs = [(r0, r1) for r0, r1 in F.true_runs(ss) if r1 - r0 >= min_n]
    for k, (r0, r1) in enumerate(runs):
        prev_r1 = runs[k - 1][1] if bound and k > 0 else 0
        next_r0 = runs[k + 1][0] if bound and k + 1 < len(runs) else len(v)
        # STOP: the final approach from v_min down to the first standstill sample
        lo = max(0, r0 - look, prev_r1)
        before = v[lo:r0]
        if len(before) and (before > v_min).any():
            j = int(np.flatnonzero(before > v_min)[-1]) + lo                       # last sample above v_min
            out.append(_make(d, "STOP", s0 + r0, s0 + j, s0 + r0 + 1, -1, p))
        # LAUNCH: from the first moving sample up to the first sample above v_min
        hi = min(len(v), r1 + look, next_r0)
        after = v[r1:hi]
        if len(after) and (after > v_min).any():
            m = int(np.flatnonzero(after > v_min)[0]) + r1
            out.append(_make(d, "LAUNCH", s0 + r1, s0 + r1, s0 + m + 1, 1, p))
    return out


def _lead_seg(d: dict[str, np.ndarray], s0: int, s1: int, p: dict) -> list[Episode]:
    out: list[Episode] = []
    dt = p["dt"]
    lp = p["lead"]
    slope = d["lead_a_slope_mps2"][s0:s1]
    valid = d["lead_valid"][s0:s1]
    thw = d["thw_s"][s0:s1]
    v = d["v_mps"][s0:s1]
    d_lead = d["lead_d_m"][s0:s1]
    a_sg = d["a_sg"][s0:s1]
    hor_n = int(round(lp["horizon_s"] / dt))
    spacing_n = int(round(p["lead_min_spacing_s"] / dt))
    d_still = p.get("lead_standstill_d_m", 0.0)
    with np.errstate(invalid="ignore"):
        in_cf = np.nan_to_num(thw, nan=np.inf) <= p["thw_cf_s"]
        if d_still > 0:                     # THW is undefined while the ego (nearly) stands: use the distance instead
            in_cf |= ~np.isfinite(thw) & (np.nan_to_num(v, nan=np.inf) <= C.THW_V_MIN_MPS) & (np.nan_to_num(d_lead, nan=np.inf) <= d_still)
    for name, sign, slope_on in (("LEAD_ACC", 1, lp["slope_on"]), ("LEAD_DEC", -1, p["lead_decel_slope_on"])):
        last = -10 ** 9
        for i in lead_onsets(sign * slope, valid, dt, slope_on=slope_on, hold_s=lp["hold_s"], quiet_s=lp["quiet_s"],
                             quiet_abs=lp["quiet_abs"], trans_s=lp["trans_s"]):
            i = int(i)
            if i - last < spacing_n or not in_cf[i]:
                continue
            if not np.isfinite(thw[i]) and not (np.nan_to_num(v[i:i + hor_n], nan=-np.inf) > C.THW_V_MIN_MPS).any():
                continue                    # standing ego that never moves within the horizon: parked car or another lane
            last = i
            i1 = min(len(slope), i + hor_n)
            resp = F.ego_response_delay(sign * a_sg, i, dt, resp_a=lp["resp_a"], resp_hold_s=lp["resp_hold_s"],
                                        horizon_s=lp["horizon_s"])
            out.append(_make(d, name, s0 + i, s0 + i, s0 + i1, sign, p, resp=resp))
    return out


def _turn_seg(d: dict[str, np.ndarray], s0: int, s1: int, p: dict) -> list[Episode]:
    out: list[Episode] = []
    v = d["v_mps"][s0:s1]
    curv = d["curv_1pm"][s0:s1]
    lo, hi = p["turn_v_range"]
    with np.errstate(invalid="ignore"):
        m = d["yaw_valid"][s0:s1] & np.isfinite(curv) & (np.abs(curv) >= p["turn_curv_min"]) & (v >= lo) & (v <= hi)
    min_n = max(1, int(round(p["turn_min_s"] / p["dt"])))
    for r0, r1 in F.true_runs(m):
        if r1 - r0 >= min_n:
            out.append(_make(d, "TURN", s0 + r0, s0 + r0, s0 + r1, 0, p))
    return out


def detect_episodes(d: dict[str, np.ndarray], p: dict = PARAMS) -> list[Episode]:
    """All episodes of a drive (arrays from load_drive_arrays), detected inside gap-free segments, ordered by
    (anchor_row, type_id)."""
    ensure_derived(d, p)
    out: list[Episode] = []
    for s0, s1 in segments(d):
        if s1 - s0 < 2:
            continue
        out += _speed_episodes_seg(d, s0, s1, p)
        out += _stop_launch_seg(d, s0, s1, p)
        out += _lead_seg(d, s0, s1, p)
        if p.get("turn_enabled", True):
            out += _turn_seg(d, s0, s1, p)
    out.sort(key=lambda e: (e.anchor_row, e.type_id))
    return out


# ----------------------------------------------------------------------------- flags, context, crop, grid
def _crop_bounds(d: dict[str, np.ndarray], ep: Episode) -> tuple[int, int, int, int, bool]:
    """(r0, lo, hi, n, gap_in_crop): r0 = crop_row0; [lo, hi) = rows of the crop that are inside the drive and in the
    same gap-free segment as the anchor; gap_in_crop = a gap row lies strictly inside the in-drive part of the crop."""
    n = int(d["__n__"][0])
    r0 = ep.crop_row0
    lo, hi = max(r0, 0), min(r0 + L_STORE, n)
    gaps = np.flatnonzero(d["gap"][lo + 1:hi]) + lo + 1 if hi > lo + 1 else np.zeros(0, np.int64)
    gap_in_crop = bool(len(gaps))
    for r in gaps.tolist():
        if r <= ep.anchor_row:
            lo = r                                        # data before a gap row at / before the anchor is discontinuous
        else:
            hi = min(hi, r)                               # data from the first gap row after the anchor onwards
            break
    return r0, lo, hi, n, gap_in_crop


def episode_flags(d: dict[str, np.ndarray], ep: Episode, p: dict = PARAMS) -> dict:
    """das_frac (das_enabled over [i0, i1)), das_any_frac (das_any over [i0, i1)), human_strict = no das_any sample from
    min(anchor - human_guard_s, crop start) to the crop end (the whole crop plus the 10-s guard before the anchor; the
    episode interval may extend past the crop, so human_strict does not imply das_any_frac == 0), lead_valid_frac over
    [i0, i1), gap_in_crop."""
    r0, lo, hi, n, gap_in_crop = _crop_bounds(d, ep)
    sl = slice(ep.i0, ep.i1)
    guard_n = int(round(p["human_guard_s"] / p["dt"]))
    g0, g1 = max(0, min(ep.anchor_row - guard_n, r0)), min(n, r0 + L_STORE)   # the crop plus a guard before the anchor
    return dict(das_frac=float(d["das_enabled"][sl].mean()) if ep.i1 > ep.i0 else float("nan"),
                das_any_frac=float(d["das_any"][sl].mean()) if ep.i1 > ep.i0 else float("nan"),
                human_strict=bool(not d["das_any"][g0:g1].any()),
                lead_valid_frac=float(d["lead_valid"][sl].mean()) if ep.i1 > ep.i0 else float("nan"),
                gap_in_crop=gap_in_crop)


def _context_from(values: dict) -> np.ndarray:
    x = np.array([float(values[c]) for c in CONTEXT_COLUMNS], dtype=np.float64)
    x[7] = 0.0 if not np.isfinite(x[7]) else x[7]          # thw_anchor: 0 when no lead
    x = x * _CONTEXT_SCALE
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def episode_context(d: dict[str, np.ndarray], ep: Episode, p: dict = PARAMS) -> np.ndarray:
    """The 9-float context vector of contract (1): v_anchor/30, dv/10, dur_s/20, mean_a, peak_a, das_frac,
    lead_valid_frac, thw_anchor/10 (0 when NaN), gap_in_crop. Finite by construction (NaN -> 0)."""
    vals = dict(asdict(ep), **episode_flags(d, ep, p))
    return _context_from(vals)


def context_matrix(index: pd.DataFrame) -> np.ndarray:
    """(n, 9) float32 context vectors from index rows (the consumer-side form of episode_context)."""
    x = index[list(CONTEXT_COLUMNS)].to_numpy(dtype=np.float64, na_value=np.nan)
    x[:, 7] = np.where(np.isfinite(x[:, 7]), x[:, 7], 0.0)
    x = x * _CONTEXT_SCALE
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def crop_episode(d: dict[str, np.ndarray], ep: Episode) -> np.ndarray:
    """(L_STORE, 22) float16: EP_CHANNELS over rows [crop_row0, crop_row0 + L_STORE); NaN with valid = 0 outside the drive
    or beyond a gap row (relative to the anchor), bool channels as 0 / 1; values clipped to the float16 range."""
    ensure_derived(d)
    r0, lo, hi, n, _ = _crop_bounds(d, ep)
    out = np.full((L_STORE, len(EP_CHANNELS)), np.nan, dtype=np.float32)
    out[:, -1] = 0.0
    if hi > lo:
        sl = slice(lo - r0, hi - r0)
        for k, ch in enumerate(C.OFFICIAL_INPUTS):
            out[sl, k] = d[ch][lo:hi]
        out[sl, len(C.OFFICIAL_INPUTS)] = d["a_sg"][lo:hi]
        out[sl, -1] = 1.0
    return np.clip(out, -F16_MAX, F16_MAX).astype(np.float16)


def speed_grid(d: dict[str, np.ndarray], ep: Episode) -> np.ndarray | None:
    """(72, 6) float16 speed-indexed representation of an ACC / DEC episode (None for the other types): GRID_CHANNELS
    interpolated onto SPEED_GRID between the episode's minimum and maximum smoothed speed (valid = 1 there), NaN with
    valid = 0 elsewhere. Speed is made monotone (running max for ACC, running min for DEC) before interpolation."""
    if ep.type not in GRID_TYPES:
        return None
    ensure_derived(d)
    out = np.full((len(SPEED_GRID), len(GRID_CHANNELS)), np.nan, dtype=np.float32)
    out[:, 4] = 0.0
    sl = slice(ep.i0, ep.i1)
    x = d["v_sg"][sl]
    if len(x) < 2 or not np.isfinite(x).all():
        return out.astype(np.float16)
    acc = ep.type == "ACC"
    x = np.maximum.accumulate(x) if acc else np.minimum.accumulate(x)
    ys = [d["a_sg"][sl], d["gas_pressed"][sl].astype(float), d["brake_pressed"][sl].astype(float),
          d["das_enabled"][sl].astype(float), (np.arange(ep.i0, ep.i1) - ep.anchor_row) * PARAMS["dt"]]
    if not acc:
        x = x[::-1]
        ys = [y[::-1] for y in ys]
    cov = (SPEED_GRID >= x[0]) & (SPEED_GRID <= x[-1])
    if cov.any():
        g = SPEED_GRID[cov]
        for k, y in zip((0, 1, 2, 3, 5), ys):
            out[cov, k] = np.interp(g, x, y)
        out[cov, 4] = 1.0
    return np.clip(out, -F16_MAX, F16_MAX).astype(np.float16)


def episode_row(d: dict[str, np.ndarray], ep: Episode, p: dict = PARAMS) -> dict:
    """The index row of one episode (ids, episode_id, window_id and grid_idx are filled by extract_drive / assembly)."""
    row = asdict(ep)
    row.update(episode_flags(d, ep, p))
    row.update(crop_row0=ep.crop_row0, w_idx=ep.anchor_row // C.WINDOW_SAMPLES)
    return row


def coerce_index(df: pd.DataFrame) -> pd.DataFrame:
    """Order and cast an index frame to INDEX_COLUMNS (missing columns become null / NaN / -1 / False)."""
    out = pd.DataFrame(index=df.index)
    for name, dt in INDEX_COLUMNS:
        col = df[name] if name in df.columns else pd.Series([None] * len(df), index=df.index, dtype=object)
        if dt == "bool":
            out[name] = col.fillna(False).astype(bool).to_numpy()
        elif dt.startswith("int"):
            out[name] = pd.to_numeric(col, errors="coerce").fillna(-1).astype(dt).to_numpy()
        elif dt.startswith("float"):
            out[name] = pd.to_numeric(col, errors="coerce").astype(dt).to_numpy()
        else:
            out[name] = col.astype(object).where(pd.notna(col), None).to_numpy()
    return out


def extract_drive(path: Path | str, drive_id: str, vehicle_id: str, group_id: str | None,
                  p: dict = PARAMS) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(E, G, index) of one drive: E (n, 180, 22) float16, G (n_g, 72, 6) float16, index with INDEX_COLUMNS where
    episode_id and grid_idx are local to the drive and window_id is None."""
    d = load_drive_arrays(path)
    eps = detect_episodes(d, p)
    crops, grids, grid_idx, rows = [], [], [], []
    for ep in eps:
        crops.append(crop_episode(d, ep))
        g = speed_grid(d, ep)
        if g is None:
            grid_idx.append(-1)
        else:
            grid_idx.append(len(grids))
            grids.append(g)
        rows.append(episode_row(d, ep, p))
    E = np.stack(crops) if crops else np.zeros((0, L_STORE, len(EP_CHANNELS)), np.float16)
    G = np.stack(grids) if grids else np.zeros((0, len(SPEED_GRID), len(GRID_CHANNELS)), np.float16)
    idx = pd.DataFrame(rows) if rows else pd.DataFrame()
    idx["episode_id"] = np.arange(len(eps), dtype=np.int64)
    idx["drive_id"], idx["vehicle_id"], idx["group_id"] = drive_id, vehicle_id, group_id
    idx["window_id"] = None
    idx["grid_idx"] = np.asarray(grid_idx, dtype=np.int32)
    return E, G, coerce_index(idx)


def type_counts(index: pd.DataFrame) -> dict[str, int]:
    """{type name: n episodes} over an index frame (every type listed, zeros included)."""
    counts = np.bincount(index["type_id"].to_numpy(np.int64), minlength=len(TYPES)) if len(index) else np.zeros(len(TYPES), int)
    return {t: int(counts[i]) for i, t in enumerate(TYPES)}
