#!/usr/bin/env python3
"""Window contract and shared helpers of the external zero-shot transfer test (experiments/transfer).

Every dataset adapter produces exactly two things and nothing else:
  X     float32 (n, 600, 20): 60-s windows at 10 Hz in the RAW PHYSICAL UNITS of the 20 training channels
        (CHANNELS = work/tensors/channels.json = vtt.config.OFFICIAL_INPUTS, same order), NaN wherever the dataset cannot
        provide a channel or a sample of it;
  meta  one row per window with META_COLUMNS (dataset, vehicle_id, model, platform, drive_id, w_idx, t0_s, das_frac,
        moving_frac, footprint_m2, size_class, powertrain, borderline, channels_present, note).
save_windows / load_windows read and write <data-root>/external/<dataset>/work/windows.npz + meta.parquet. The scorer
(score.py) applies the training pre-processing (per-channel clip -> standardise with the training statistics -> NaN -> 0),
so adapters never clip, standardise or fill.

Value conventions of the training tensors (measured on work/tensors/X_f16.npy; adapters must reproduce them):
* the six flags gas_pressed, brake_pressed, steer_pressed, standstill, das_enabled, lead_valid are 0 / 1 floats, never NaN;
* the six lead channels after lead_valid (lead_d_m ... ttc_s) are NaN whenever lead_valid = 0: that is the normal "no lead"
  state (60 % of training samples), not a missing channel; lead_speed_mps = v + lead_v_rel_mps; thw_s = d / v while
  v > 0.5 m/s; ttc_s = d / closing speed while closing > 0.1 m/s; lead_a_slope_mps2 = causal 1-s least-squares slope of the
  lead speed inside a lead episode (>= 3 samples, clipped +-4); training clips thw / ttc to 30 / 60 s, so large values are fine;
* curv_1pm = yaw_rate / v only while v >= 1 m/s (NaN otherwise), lat_acc_mps2 = v x yaw_rate; yaw is left-positive and comes
  from the device pose (IMU), not from the steering wheel; steering angle left-positive in degrees;
* steer_torque_norm = driver torque / the 95th percentile of |torque| over moving samples with steer_pressed (per recording,
  NaN with fewer than 50 such samples) - no platform constant is needed;
* a_mps2 is the vehicle's own aEgo sample (openpilot's speed Kalman filter), v_mps the filtered CAN speed; the 10 Hz rows are
  SAMPLES of the 100 Hz CAN stream (openpilot's qlog decimation: every 10th message), not bin averages;
* a channel the dataset lacks is NaN for every sample (-> 0 after standardisation, the training "missing" value); the training
  augmentation dropped whole groups that way (CHANNEL_GROUPS lead / torque / pedals, with lead_valid := 0), so those three
  groups may be absent without leaving the training distribution; any other absent channel is out of distribution and needs
  a reduced-input model variant (score.py --variant) or an explicit "reduced input" label on the result.
Window rule (docs/PROTOCOLS.md, vtt/windows.py): 600 consecutive samples, no gap (> 0.3 s between consecutive samples)
inside the window, >= 20 % of the samples moving (v >= 1 m/s).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vtt import config as C  # noqa: E402

# ----------------------------------------------------------------------------- roots
DATA_ROOT = Path(C.DATA_ROOT)
EXTERNAL_ROOT = Path(os.environ.get("VTT_EXTERNAL", str(DATA_ROOT / "external")))
TENSOR_DIR = DATA_ROOT / "work" / "tensors"
CKPT_DIR = DATA_ROOT / "experiments" / "ckpt"
RESULTS_DIR = DATA_ROOT / "experiments" / "results"

# ----------------------------------------------------------------------------- channels and constants
_OFFICIAL = list(C.OFFICIAL_INPUTS)


def _load_channels() -> list[str]:
    """work/tensors/channels.json when the data root is mounted (must equal vtt.config.OFFICIAL_INPUTS), else the config list."""
    p = TENSOR_DIR / "channels.json"
    if p.exists():
        ch = json.loads(p.read_text())
        if list(ch) != _OFFICIAL:
            raise RuntimeError(f"{p} differs from vtt.config.OFFICIAL_INPUTS")
        return list(ch)
    return _OFFICIAL


CHANNELS: list[str] = _load_channels()
CH = {c: i for i, c in enumerate(CHANNELS)}              # name -> column
N_CHANNELS = len(CHANNELS)                               # 20
WINDOW_SAMPLES = int(C.WINDOW_SAMPLES)                   # 600
HZ = int(C.NOMINAL_HZ)                                   # 10
DT_S = 1.0 / HZ
GAP_S = float(C.CS_GAP_S)                                # 0.3 s between consecutive samples = gap
MOVING_V_MPS = float(C.MOVING_V_MPS)                     # 1 m/s
MIN_MOVING_FRAC = float(C.WINDOW_MIN_MOVING_FRAC)        # 0.2
THRESHOLDS: tuple[float, float] = tuple(float(x) for x in (C.FOOTPRINT_THRESHOLDS_M2 or (4.43, 4.88)))
if THRESHOLDS != (4.43, 4.88):
    raise RuntimeError(f"footprint thresholds {THRESHOLDS} differ from the locked 4.43 / 4.88")
BORDERLINE_M2 = 0.05
SIZE_CLASSES = tuple(C.SIZE_CLASSES)                     # SMALL MID LARGE
BEV_ICE = tuple(C.MAIN_POWERTRAIN_TASK)                  # BEV ICE
POWERTRAINS = ("BEV", "ICE", "HEV", "PHEV", "FCEV", "unknown")
TASK_LABEL = {"size": ("size_class", SIZE_CLASSES), "bev_ice": ("powertrain", BEV_ICE)}
BINARY_CHANNELS = ("gas_pressed", "brake_pressed", "steer_pressed", "standstill", "das_enabled", "lead_valid")
LEAD_CHANNELS = ("lead_valid", "lead_d_m", "lead_v_rel_mps", "lead_speed_mps", "lead_a_slope_mps2", "thw_s", "ttc_s")
LATERAL_CHANNELS = ("steer_angle_deg", "steer_rate_degps", "steer_torque_norm", "yaw_rate_radps", "curv_1pm", "lat_acc_mps2", "steer_pressed")
# the training augmentation groups (experiments.data.CHANNEL_GROUPS) + the masks of the planned model variants (CANDIDATES.md 2)
CHANNEL_GROUPS: dict[str, tuple[str, ...]] = {
    "lead": LEAD_CHANNELS,
    "torque": ("steer_torque_norm",),
    "pedals": ("gas_pressed", "brake_pressed"),
    "lateral": LATERAL_CHANNELS,
    "yaw": ("yaw_rate_radps", "curv_1pm", "lat_acc_mps2"),
    "steer": ("steer_angle_deg", "steer_rate_degps", "steer_pressed", "steer_torque_norm"),
}
MASK_PRESETS: dict[str, tuple[str, ...]] = {
    "V1": tuple(dict.fromkeys(LEAD_CHANNELS + ("steer_torque_norm", "gas_pressed", "brake_pressed"))),            # commaSteeringControl
    "V2": tuple(dict.fromkeys(LATERAL_CHANNELS + ("gas_pressed", "brake_pressed"))),                              # longitudinal only
    "V3": tuple(c for c in CHANNELS if c not in ("v_mps", "a_mps2", "standstill")),                              # speed only
}
META_COLUMNS: list[tuple[str, str]] = [            # (name, pandas dtype)
    ("dataset", "object"), ("vehicle_id", "object"), ("model", "object"), ("platform", "object"), ("drive_id", "object"),
    ("w_idx", "int32"), ("t0_s", "float64"), ("das_frac", "float32"), ("moving_frac", "float32"), ("footprint_m2", "float32"),
    ("size_class", "object"), ("powertrain", "object"), ("borderline", "bool"), ("channels_present", "object"), ("note", "object"),
]
REQUIRED_META = ("dataset", "vehicle_id", "model", "drive_id")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+()/-]*$")


# ----------------------------------------------------------------------------- labels
def classify_footprint(fp):
    """Footprint (m^2) -> SMALL / MID / LARGE (None / NaN -> None). Works on scalars and arrays (object array of str / None)."""
    lo, hi = THRESHOLDS
    if np.ndim(fp) == 0:
        f = float(fp) if fp is not None else float("nan")
        if not np.isfinite(f):
            return None
        return "SMALL" if f < lo else ("MID" if f < hi else "LARGE")
    f = np.asarray(fp, dtype=float)
    out = np.full(f.shape, None, dtype=object)
    ok = np.isfinite(f)
    out[ok & (f < lo)] = "SMALL"
    out[ok & (f >= lo) & (f < hi)] = "MID"
    out[ok & (f >= hi)] = "LARGE"
    return out


def is_borderline(fp, tol: float = BORDERLINE_M2):
    """|footprint - a cut| < tol (NaN -> False); scalar or array."""
    f = np.asarray(fp, dtype=float)
    ok = np.isfinite(f)
    b = ok & ((np.abs(f - THRESHOLDS[0]) < tol) | (np.abs(f - THRESHOLDS[1]) < tol))
    return bool(b) if np.ndim(fp) == 0 else b


def normalise_powertrain(p) -> str:
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "unknown"
    s = str(p).strip()
    up = s.upper()
    aliases = {"EV": "BEV", "ELECTRIC": "BEV", "BATTERY ELECTRIC": "BEV", "PETROL": "ICE", "GASOLINE": "ICE", "DIESEL": "ICE",
               "GAS": "ICE", "HYBRID": "HEV", "MILD HYBRID": "HEV", "MHEV": "HEV", "PLUG-IN HYBRID": "PHEV", "FUEL CELL": "FCEV",
               "UNKNOWN": "unknown", "UNK": "unknown", "NONE": "unknown", "NAN": "unknown", "": "unknown"}
    up = aliases.get(up, up)
    if up not in POWERTRAINS:
        raise ValueError(f"powertrain {p!r} not in {POWERTRAINS}")
    return up


def task_labels(meta: pd.DataFrame, task: str) -> np.ndarray:
    """(n,) int64 class index of the task for every window, -1 when the label is unknown or outside the task classes."""
    col, classes = TASK_LABEL[task]
    pos = {c: i for i, c in enumerate(classes)}
    return np.array([pos.get(x, -1) for x in meta[col].to_numpy(object)], dtype=np.int64)


# ----------------------------------------------------------------------------- channel masks
def resolve_mask(spec) -> list[str]:
    """Comma string or iterable of tokens -> sorted channel names. A token is a channel name, a CHANNEL_GROUPS name (lead,
    torque, pedals, lateral, yaw, steer) or a MASK_PRESETS name (V1, V2, V3). Empty -> []."""
    if spec is None:
        return []
    toks = [t.strip() for t in spec.split(",")] if isinstance(spec, str) else [str(t).strip() for t in spec]
    out: set[str] = set()
    for t in toks:
        if not t:
            continue
        if t in CH:
            out.add(t)
        elif t in CHANNEL_GROUPS:
            out.update(CHANNEL_GROUPS[t])
        elif t.upper() in MASK_PRESETS:
            out.update(MASK_PRESETS[t.upper()])
        else:
            raise ValueError(f"unknown channel / group / preset {t!r}; channels {CHANNELS}, groups {list(CHANNEL_GROUPS)}, presets {list(MASK_PRESETS)}")
    return sorted(out, key=CH.get)


def apply_channel_mask(X: np.ndarray, names: Iterable[str]) -> np.ndarray:
    """Copy of X with the named channels set to the training "missing" state: NaN (-> 0 after standardisation), except
    lead_valid, which becomes 0 (= no lead) exactly as the training augmentation does when it drops the lead group."""
    names = list(names)
    X = np.array(X, dtype=np.float32, copy=True)
    for c in names:
        X[:, :, CH[c]] = 0.0 if c == "lead_valid" else np.nan
    return X


def mask_tag(names: Iterable[str]) -> str:
    """Short file-name token of a mask: 'full' or 'mask-<preset>' when the set equals a preset, else 'mask-<n>ch-<hash>'."""
    names = sorted(set(names), key=CH.get)
    if not names:
        return "full"
    for k, v in MASK_PRESETS.items():
        if set(v) == set(names):
            return f"mask-{k}"
    for k, v in CHANNEL_GROUPS.items():
        if set(v) == set(names):
            return f"mask-{k}"
    import hashlib
    h = hashlib.sha1(",".join(names).encode()).hexdigest()[:6]
    return f"mask-{len(names)}ch-{h}"


# ----------------------------------------------------------------------------- validation, save, load
def _as_float32_3d(X) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[1] != WINDOW_SAMPLES or X.shape[2] != N_CHANNELS:
        raise ValueError(f"X must be (n, {WINDOW_SAMPLES}, {N_CHANNELS}), got {X.shape}")
    return np.ascontiguousarray(X, dtype=np.float32)


def validate_windows(X, meta: pd.DataFrame, strict: bool = True) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Check the contract and complete the meta frame (size_class from footprint, borderline, powertrain vocabulary,
    das_frac / moving_frac / channels_present from X when absent). Returns (X float32, meta in META_COLUMNS order,
    warnings). strict=True raises on a violation that would corrupt the score (shape, lengths, ids, non-binary flags,
    windows below the moving rule); warnings cover the rest."""
    X = _as_float32_3d(X)
    n = X.shape[0]
    warns: list[str] = []
    if len(meta) != n:
        raise ValueError(f"meta has {len(meta)} rows for {n} windows")
    m = meta.reset_index(drop=True).copy()
    missing = [c for c in REQUIRED_META if c not in m.columns]
    if missing:
        raise ValueError(f"meta lacks required columns {missing}")
    for c in REQUIRED_META:
        bad = m[c].isna() | (m[c].astype(str).str.strip() == "")
        if bad.any():
            raise ValueError(f"meta.{c} has {int(bad.sum())} empty values")
        m[c] = m[c].astype(str)
    for c in ("vehicle_id", "drive_id"):
        bad = [s for s in m[c].unique() if not _ID_RE.match(s)]
        if bad:
            warns.append(f"{c}: {len(bad)} ids with unusual characters, e.g. {bad[:3]}")
    if "platform" not in m.columns:
        m["platform"] = None
    if "note" not in m.columns:
        m["note"] = ""
    m["note"] = m["note"].fillna("").astype(str)
    if "w_idx" not in m.columns:
        m["w_idx"] = m.groupby("drive_id", sort=False).cumcount().astype(np.int32)
        warns.append("w_idx missing: numbered the windows of every drive in row order")
    if "t0_s" not in m.columns:
        m["t0_s"] = np.nan
    v = X[:, :, CH["v_mps"]]
    moving = np.nan_to_num(v, nan=-np.inf) >= MOVING_V_MPS
    mf = moving.mean(1).astype(np.float32)
    if "moving_frac" not in m.columns or m["moving_frac"].isna().all():
        m["moving_frac"] = mf
    else:
        d = np.nanmax(np.abs(m["moving_frac"].to_numpy(np.float32) - mf)) if n else 0.0
        if d > 0.01:
            warns.append(f"moving_frac disagrees with X by up to {d:.3f}; X wins")
        m["moving_frac"] = mf
    low = int((mf < MIN_MOVING_FRAC).sum())
    if low:
        msg = f"{low} windows below the moving rule (moving_frac < {MIN_MOVING_FRAC})"
        if strict:
            raise ValueError(msg + "; drop them (cut_windows does) before saving")
        warns.append(msg)
    das = X[:, :, CH["das_enabled"]]
    das_frac = np.where(np.isfinite(das).any(1), np.nanmean(np.where(np.isfinite(das), das, np.nan), axis=1), np.nan).astype(np.float32)
    if "das_frac" not in m.columns or m["das_frac"].isna().all():
        m["das_frac"] = das_frac
    for c in BINARY_CHANNELS:
        col = X[:, :, CH[c]]
        fin = col[np.isfinite(col)]
        if fin.size and not np.isin(fin, (0.0, 1.0)).all():
            raise ValueError(f"channel {c} must be 0 / 1 / NaN, found other values (e.g. {np.unique(fin)[:5]})")
    lv = X[:, :, CH["lead_valid"]]
    if np.isnan(lv).any():
        warns.append("lead_valid has NaN samples: the training value for 'no lead' is 0, NaN means 'unknown' (standardised 0)")
    if not np.isfinite(v).any():
        raise ValueError("v_mps is NaN everywhere")
    # labels
    if "footprint_m2" not in m.columns:
        m["footprint_m2"] = np.nan
    m["footprint_m2"] = pd.to_numeric(m["footprint_m2"], errors="coerce").astype(np.float32)
    derived = classify_footprint(m["footprint_m2"].to_numpy(np.float64))
    if "size_class" not in m.columns:
        m["size_class"] = derived
    else:
        sc = m["size_class"].astype(object).where(m["size_class"].notna(), None)
        sc = sc.map(lambda s: None if s is None else str(s).upper())
        bad = sorted({s for s in sc.dropna().unique() if s not in SIZE_CLASSES})
        if bad:
            raise ValueError(f"size_class values {bad} not in {SIZE_CLASSES}")
        fill = sc.isna() & pd.notna(derived)
        sc[fill] = derived[fill.to_numpy()]
        dis = int((pd.notna(derived) & sc.notna() & (sc.to_numpy(object) != derived)).sum())
        if dis:
            warns.append(f"{dis} windows whose size_class disagrees with the footprint thresholds (given class kept)")
        m["size_class"] = sc
    if "powertrain" not in m.columns:
        m["powertrain"] = "unknown"
    m["powertrain"] = [normalise_powertrain(p) for p in m["powertrain"].to_numpy(object)]
    m["borderline"] = is_borderline(m["footprint_m2"].to_numpy(np.float64)) if "borderline" not in m.columns \
        else m["borderline"].fillna(False).astype(bool) | is_borderline(m["footprint_m2"].to_numpy(np.float64))
    if "channels_present" not in m.columns or m["channels_present"].isna().all():
        present = np.isfinite(X).any(1)                                             # (n, C)
        m["channels_present"] = [",".join(c for c, p in zip(CHANNELS, row) if p) for row in present]
        warns.append("channels_present derived from X (a lead channel counts as absent in windows without any lead)")
    # mixed labels inside a vehicle are a labelling error
    for c in ("model", "size_class", "powertrain"):
        k = m.groupby("vehicle_id")[c].nunique(dropna=False)
        if (k > 1).any():
            warns.append(f"{int((k > 1).sum())} vehicles carry more than one {c}")
    out = pd.DataFrame({name: m[name] for name, _ in META_COLUMNS})
    for name, dt in META_COLUMNS:
        if dt == "object":
            out[name] = out[name].astype(object).where(out[name].notna(), None)
        else:
            out[name] = out[name].astype(dt)
    return X, out, warns


def work_dir(dataset: str, root: Path | str | None = None) -> Path:
    return Path(root or EXTERNAL_ROOT) / dataset / "work"


def save_windows(dataset: str, X, meta: pd.DataFrame, root: Path | str | None = None, strict: bool = True) -> dict:
    """Validate and write <root>/<dataset>/work/windows.npz (X, channels) + meta.parquet + windows_summary.json.
    Returns dict(npz, parquet, summary, n, warnings)."""
    X, meta, warns = validate_windows(X, meta, strict=strict)
    wd = work_dir(dataset, root)
    wd.mkdir(parents=True, exist_ok=True)
    npz, pq, summ = wd / "windows.npz", wd / "meta.parquet", wd / "windows_summary.json"
    tmp = wd / ".windows.tmp.npz"
    np.savez_compressed(tmp, X=X, channels=np.array(CHANNELS, dtype=object))
    tmp.replace(npz)
    meta.to_parquet(pq, index=False)
    s = summarize_windows(meta)
    summ.write_text(json.dumps(dict(dataset=dataset, n_windows=int(len(meta)), n_vehicles=int(meta["vehicle_id"].nunique()),
                                    n_drives=int(meta["drive_id"].nunique()), n_models=int(meta["model"].nunique()),
                                    hours=round(len(meta) * WINDOW_SAMPLES * DT_S / 3600.0, 2),
                                    size_class=meta["size_class"].fillna("unknown").value_counts().to_dict(),
                                    powertrain=meta["powertrain"].value_counts().to_dict(),
                                    channels_present=sorted({c for s_ in meta["channels_present"].dropna() for c in s_.split(",") if c}, key=CH.get),
                                    per_model=s.to_dict(orient="records"), warnings=warns), indent=1, default=str))
    for w in warns:
        print(f"[save_windows:{dataset}] warning: {w}")
    print(f"[save_windows:{dataset}] n={len(meta)} vehicles={meta['vehicle_id'].nunique()} drives={meta['drive_id'].nunique()} "
          f"models={meta['model'].nunique()} -> {npz}")
    return dict(npz=npz, parquet=pq, summary=summ, n=int(len(meta)), warnings=warns)


def load_windows(dataset: str, root: Path | str | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """(X float32 (n, 600, 20), meta) of a saved window set; checks the channel order."""
    wd = work_dir(dataset, root)
    z = np.load(wd / "windows.npz", allow_pickle=True)
    X = _as_float32_3d(z["X"])
    if "channels" in z.files and [str(c) for c in z["channels"]] != CHANNELS:
        raise ValueError("windows.npz channel order differs from the training channels")
    meta = pd.read_parquet(wd / "meta.parquet")
    if len(meta) != X.shape[0]:
        raise ValueError(f"{len(meta)} meta rows for {X.shape[0]} windows")
    return X, meta


def summarize_windows(meta: pd.DataFrame) -> pd.DataFrame:
    """Per car model: windows, vehicles, drives, hours, footprint, size_class, powertrain, borderline, DAS share."""
    g = meta.groupby("model", sort=True, dropna=False)
    out = pd.DataFrame({
        "n_windows": g.size(), "n_vehicles": g["vehicle_id"].nunique(), "n_drives": g["drive_id"].nunique(),
        "hours": (g.size() * WINDOW_SAMPLES * DT_S / 3600.0).round(2),
        "footprint_m2": g["footprint_m2"].first().round(3),
        "size_class": g["size_class"].first(), "powertrain": g["powertrain"].first(),
        "borderline": g["borderline"].first(), "das_frac": g["das_frac"].mean().round(3),
    }).reset_index()
    return out


# ----------------------------------------------------------------------------- resampling utilities (source -> 10 Hz grid)
def make_grid(t0: float, t1: float, hz: int = HZ) -> np.ndarray:
    """Grid times k / hz inside [t0, t1] (first grid point = ceil(t0 x hz) / hz)."""
    k0 = int(np.ceil(round(t0 * hz, 6)))
    k1 = int(np.floor(round(t1 * hz, 6)))
    return np.arange(k0, k1 + 1, dtype=np.float64) / hz


def resample_to_10hz(t, values, kind: str = "linear", *, t_grid=None, hz: int = HZ, max_age_s: float | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Bring one source series (times `t` in seconds, any rate, need not be regular) onto a 10 Hz grid -> (t_grid, out).

    kind: 'linear'   interpolate a continuous signal (speed, angle, distance); NaN outside the source span and, with
                     max_age_s, wherever the previous source sample is older than max_age_s (no bridging of long gaps);
          'hold'     causal zero-order hold = the sample at or before the grid time (how the lab put every non-carState
                     topic on the timeline; lab max ages: 0.3 s controls, 0.75 s radar lead, 0.625 s pose); stale -> NaN,
                     or 0 for boolean input (lab hold_bool);
          'decimate' alias of 'hold' with max_age_s defaulting to 1.5 / source rate: the right way to bring a regular
                     100 Hz CAN stream to 10 Hz (the training rows are samples, not bin means);
          'nearest'  nearest source sample in time (either side) within max_age_s;
          'any'      boolean flags from a denser source: 1 when any source sample in the bin (g - 1/hz, g] is true, 0
                     otherwise (NaN when the bin holds no sample; e.g. a 100 Hz brake flag that must not miss a tap);
          'mean'     mean of the source samples in the bin (g - 1/hz, g] (NaN when empty); not the lab convention for CAN
                     signals, use it for noisy GNSS speed only.
    NaN source values are ignored by 'linear' / 'mean' / 'any' and held as NaN by 'hold' / 'nearest' (an invalid sample is
    still the latest observation, as in the lab). The default grid runs from ceil(t[0] hz)/hz to t[-1] in steps of 1/hz."""
    t = np.asarray(t, dtype=np.float64)
    is_bool = np.asarray(values).dtype == bool
    v = np.asarray(values, dtype=np.float64)
    if t.ndim != 1 or v.shape != t.shape:
        raise ValueError("t and values must be 1-D of the same length")
    if len(t) and (np.diff(t) < 0).any():
        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]
    if t_grid is None:
        if len(t) == 0:
            return np.zeros(0), np.zeros(0)
        t_grid = make_grid(t[0], t[-1], hz)
    g = np.asarray(t_grid, dtype=np.float64)
    out = np.full(len(g), np.nan)
    if len(t) == 0:
        return g, out
    eps = 1e-6                                                 # grid tolerance: 0.01 k and k / 10 differ in the last bits
    if kind == "decimate":
        kind = "hold"
        if max_age_s is None:
            d = np.diff(t)
            d = d[d > 0]
            max_age_s = 1.5 * float(np.median(d)) if len(d) else 1.5 / hz
    if kind == "linear":
        ok = np.isfinite(v)
        if ok.sum() >= 1:
            ts, vs = t[ok], v[ok]
            inside = (g >= ts[0]) & (g <= ts[-1])
            out[inside] = np.interp(g[inside], ts, vs)
            if max_age_s is not None:
                j = np.searchsorted(ts, g + eps, side="right") - 1
                age = g - ts[np.clip(j, 0, len(ts) - 1)]
                out[(j < 0) | (age > max_age_s + 1e-9)] = np.nan
        return g, out
    if kind in ("hold", "nearest"):
        j = np.searchsorted(t, g + eps, side="right") - 1
        jc = np.clip(j, 0, len(t) - 1)
        if kind == "nearest":
            jn = np.clip(j + 1, 0, len(t) - 1)
            use_next = (np.abs(t[jn] - g) < np.abs(g - t[jc])) | (j < 0)
            jc = np.where(use_next, jn, jc)
            age = np.abs(t[jc] - g)
            fresh = np.ones(len(g), bool) if max_age_s is None else age <= max_age_s + 1e-9
        else:
            age = g - t[jc]
            fresh = j >= 0
            if max_age_s is not None:
                fresh &= age <= max_age_s + 1e-9
        out = np.where(fresh, v[jc], np.nan)
        if is_bool:
            out = np.where(fresh, np.nan_to_num(out, nan=0.0), 0.0)
        return g, out
    if kind in ("any", "mean"):
        ok = np.isfinite(v)
        ts, vs = t[ok], v[ok]
        if len(ts) == 0:
            return g, out
        b = np.searchsorted(ts, g - 1.0 / hz + eps, side="right")   # first sample > g - dt
        e = np.searchsorted(ts, g + eps, side="right")               # first sample > g
        cs = np.concatenate([[0.0], np.cumsum(vs)])
        cnt = e - b
        has = cnt > 0
        s = cs[e] - cs[b]
        if kind == "any":
            out[has] = (s[has] > 0).astype(float)
        else:
            out[has] = s[has] / cnt[has]
        return g, out
    raise ValueError(f"unknown kind {kind!r}; choose linear / hold / decimate / nearest / any / mean")


# ----------------------------------------------------------------------------- derived channels (the lab's rules)
def curvature_and_lat_acc(v, yaw_rate) -> tuple[np.ndarray, np.ndarray]:
    """(curv_1pm, lat_acc_mps2) from speed and a left-positive yaw rate: curv = yaw / v only while v >= 1 m/s (else NaN),
    lat_acc = v x yaw (NaN where either input is NaN)."""
    v = np.asarray(v, float); yaw = np.asarray(yaw_rate, float)
    curv = np.full(len(v), np.nan)
    ok = np.isfinite(v) & np.isfinite(yaw) & (v >= MOVING_V_MPS)
    np.divide(yaw, v, out=curv, where=ok)
    return curv, v * yaw


def yaw_from_steering(v, steer_angle_deg, steer_ratio: float, wheelbase_m: float) -> np.ndarray:
    """Kinematic bicycle model yaw rate (rad/s) = v tan(delta) / L with delta = steering wheel angle / steer ratio. A
    steering-derived substitute for the IMU yaw of training (status 'derived': say so in meta.note)."""
    v = np.asarray(v, float); a = np.deg2rad(np.asarray(steer_angle_deg, float) / float(steer_ratio))
    return v * np.tan(a) / float(wheelbase_m)


def torque_norm(torque, v, steer_pressed) -> tuple[float, np.ndarray]:
    """The lab rule (vtt.drives.torque_norm): scale = 95th percentile of |torque| over samples moving (v >= 1) with
    steer_pressed; (scale, torque / scale); (NaN, all NaN) with fewer than 50 such samples or a zero scale."""
    tq = np.asarray(torque, float)
    sel = np.isfinite(tq) & (np.asarray(v, float) >= MOVING_V_MPS) & np.asarray(steer_pressed).astype(bool)
    if int(sel.sum()) < int(C.TORQUE_NORM_MIN_SAMPLES):
        return float("nan"), np.full(len(tq), np.nan)
    scale = float(np.quantile(np.abs(tq[sel]), float(C.TORQUE_NORM_Q)))
    if not scale > 0:
        return float("nan"), np.full(len(tq), np.nan)
    return scale, tq / scale


def smooth_derivative(x, hz: int = HZ, win_s: float = 0.5) -> np.ndarray:
    """d x / d t at the grid rate (central differences), then a centred moving average of win_s seconds; NaN propagate
    through the window. For a_mps2 from a speed trace (status 'derived': noisier than the CAN aEgo) or
    steer_rate_degps from the angle."""
    x = np.asarray(x, float)
    if len(x) < 3:
        return np.full(len(x), np.nan)
    d = np.gradient(x) * hz
    w = max(1, int(round(win_s * hz)))
    if w <= 1:
        return d
    k = np.ones(w) / w
    pad = w // 2
    dp = np.pad(d, (pad, w - 1 - pad), mode="edge")
    return np.convolve(dp, k, mode="valid")


def lead_episodes(t, lead_valid, gap_s: float = 0.6) -> np.ndarray:
    """Episode index per sample (-1 while no valid lead): a run of consecutive valid samples breaks at an invalid sample or a
    time gap > max(2 x median dt, gap_s) (tocf.lead rule without the radar-track-id tests, which external data lack)."""
    t = np.asarray(t, float); valid = np.asarray(lead_valid).astype(bool)
    d = np.diff(t); d = d[d > 0]
    thr = max(2.0 * float(np.median(d)), gap_s) if len(d) else gap_s
    ep = np.full(len(t), -1, dtype=np.int64)
    k = -1
    prev = False
    for i in range(len(t)):
        if valid[i]:
            if not prev or (t[i] - t[i - 1]) > thr:
                k += 1
            ep[i] = k
        prev = valid[i]
    return ep


def causal_slope(t, x, ep, win_s: float = 1.0, min_n: int = 3, clip: float = 4.0) -> np.ndarray:
    """Causal least-squares slope of x over (t_i - win_s, t_i] inside the sample's episode (>= min_n samples), clipped to
    +-clip (tocf.lead.al_slope)."""
    t = np.asarray(t, float); x = np.asarray(x, float); ep = np.asarray(ep)
    out = np.full(len(t), np.nan)
    for e in np.unique(ep[ep >= 0]):
        ii = np.flatnonzero(ep == e)
        te, xe = t[ii], x[ii]
        ok = np.isfinite(xe)
        te, xe, ii = te[ok], xe[ok], ii[ok]
        m = len(te)
        if m < min_n:
            continue
        c1 = np.concatenate([[0.0], np.cumsum(te)]); c2 = np.concatenate([[0.0], np.cumsum(te * te)])
        cx = np.concatenate([[0.0], np.cumsum(xe)]); ctx = np.concatenate([[0.0], np.cumsum(te * xe)])
        j0 = np.searchsorted(te, te - win_s, side="right")
        k = np.arange(m)
        n_ = k - j0 + 1
        st = c1[k + 1] - c1[j0]; st2 = c2[k + 1] - c2[j0]; sx = cx[k + 1] - cx[j0]; stx = ctx[k + 1] - ctx[j0]
        den = n_ * st2 - st * st
        with np.errstate(divide="ignore", invalid="ignore"):
            sl = np.where((n_ >= min_n) & (den > 1e-9), (n_ * stx - st * sx) / den, np.nan)
        out[ii] = np.clip(sl, -clip, clip)
    return out


def lead_channels(v, lead_valid, lead_d_m, lead_v_rel_mps, t=None) -> dict[str, np.ndarray]:
    """The seven lead channels from speed, a validity flag, the gap and the relative speed (lead minus ego, m/s), with the
    lab's definitions: lead_speed = v + v_rel; thw = d / v while valid, v > 0.5 m/s and d > 0.5 m; ttc = d / (-v_rel) while
    closing faster than 0.1 m/s; lead_a_slope = causal 1-s slope of lead_speed inside lead episodes (>= 3 samples, +-4);
    every lead value is NaN where lead_valid is 0. `t` defaults to the 10 Hz grid."""
    v = np.asarray(v, float); d = np.asarray(lead_d_m, float); vr = np.asarray(lead_v_rel_mps, float)
    valid = np.asarray(lead_valid).astype(bool) & np.isfinite(d) & np.isfinite(vr)
    n = len(v)
    t = np.arange(n) * DT_S if t is None else np.asarray(t, float)
    d = np.where(valid, d, np.nan); vr = np.where(valid, vr, np.nan)
    speed = v + vr
    thw = np.full(n, np.nan)
    m = valid & (v > float(C.THW_V_MIN_MPS)) & (d > float(C.LEAD_VALID_DREL_MIN_M))
    np.divide(d, v, out=thw, where=m)
    closing = -vr
    ttc = np.full(n, np.nan)
    m = valid & (closing > float(C.TTC_CLOSING_MIN_MPS))
    np.divide(d, closing, out=ttc, where=m)
    slope = causal_slope(t, speed, lead_episodes(t, valid))
    return dict(lead_valid=valid.astype(np.float32), lead_d_m=d, lead_v_rel_mps=vr, lead_speed_mps=speed,
                lead_a_slope_mps2=slope, thw_s=thw, ttc_s=ttc)


# ----------------------------------------------------------------------------- window cutting
class CutResult(NamedTuple):
    X: np.ndarray            # (n, 600, 20) float32, raw units, NaN = missing
    meta: pd.DataFrame       # META_COLUMNS
    stats: dict              # n_rows, n_gap_rows, n_runs, n_candidates, n_dropped_gap, n_dropped_stationary, n_windows


def cut_windows(df: pd.DataFrame, channel_map: dict, *, t_col: str | None = "t_s", info: dict | None = None,
                gap_s: float = GAP_S, hz: int = HZ, min_moving_frac: float = MIN_MOVING_FRAC, align: str = "run") -> CutResult:
    """Cut non-overlapping 600-sample windows from a 10 Hz frame of ONE contiguous recording (one drive).

    df: one row per 10 Hz sample in time order; t_col: its time column in seconds (None = assume a perfect 10 Hz grid).
    channel_map: {training channel: column name | array-like of len(df) | None}; channels not mapped (or None) are NaN;
    v_mps is required. Booleans become 0 / 1; flag channels must be 0 / 1 / NaN.
    A gap is a sample whose dt to the previous one exceeds gap_s (lab rule, 0.3 s); the eligible window has no gap after
    its first sample and >= min_moving_frac of its samples with v >= 1 m/s.
    align='run' (default) restarts the 600-sample blocks at the first sample after every gap, 'drive' uses blocks
    0-599, 600-1199, ... from the first row as the lab did (a block that straddles a gap is dropped). The trailing partial
    block is always dropped.
    info: constants for the meta columns (dataset, vehicle_id, model, platform, drive_id, footprint_m2, size_class,
    powertrain, note); size_class and borderline follow from footprint_m2 when not given. w_idx counts the kept windows of
    this call, t0_s is the time of the first sample (t_col units) and das_frac / moving_frac come from the window."""
    n = len(df)
    if align not in ("run", "drive"):
        raise ValueError("align must be 'run' or 'drive'")
    if "v_mps" not in channel_map or channel_map["v_mps"] is None:
        raise ValueError("channel_map must provide v_mps")
    A = np.full((n, N_CHANNELS), np.nan, dtype=np.float32)
    present: list[str] = []
    for c, src in channel_map.items():
        if c not in CH:
            raise ValueError(f"unknown channel {c!r}")
        if src is None:
            continue
        col = df[src].to_numpy() if isinstance(src, str) else np.asarray(src)
        if len(col) != n:
            raise ValueError(f"channel {c}: {len(col)} values for {n} rows")
        if col.dtype == bool or str(col.dtype) == "boolean":
            col = pd.Series(col).fillna(False).astype(float).to_numpy()
        col = pd.to_numeric(pd.Series(col), errors="coerce").to_numpy(dtype=np.float64)
        if c in BINARY_CHANNELS:
            fin = col[np.isfinite(col)]
            if fin.size and not np.isin(fin, (0.0, 1.0)).all():
                raise ValueError(f"channel {c} must be 0 / 1 / NaN")
        A[:, CH[c]] = col.astype(np.float32)
        present.append(c)
    present = sorted(set(present), key=CH.get)
    if t_col is None:
        t = np.arange(n, dtype=np.float64) / hz
    else:
        t = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(t).all():
            raise ValueError(f"{t_col} has NaN")
        if (np.diff(t) < 0).any():
            raise ValueError(f"{t_col} is not non-decreasing; sort the frame first")
    dt = np.diff(t, prepend=np.nan)
    gap = np.nan_to_num(dt, nan=0.0) > gap_s
    v = A[:, CH["v_mps"]]
    moving = np.nan_to_num(v, nan=-np.inf) >= MOVING_V_MPS
    das = A[:, CH["das_enabled"]]
    starts: list[int] = []
    if align == "drive":
        starts = list(range(0, n - WINDOW_SAMPLES + 1, WINDOW_SAMPLES))
        n_runs = int(gap.sum()) + 1
    else:
        bounds = [0] + np.flatnonzero(gap).tolist() + [n]
        n_runs = len(bounds) - 1
        for a, b in zip(bounds[:-1], bounds[1:]):
            starts += list(range(a, b - WINDOW_SAMPLES + 1, WINDOW_SAMPLES))
    n_gap_drop = n_stat_drop = 0
    rows, Xs = [], []
    inf = dict(info or {})
    for r0 in starts:
        r1 = r0 + WINDOW_SAMPLES
        if gap[r0 + 1:r1].any():                       # a gap flag on the first row marks missing data BEFORE the window
            n_gap_drop += 1
            continue
        mf = float(moving[r0:r1].mean())
        if not mf >= min_moving_frac:
            n_stat_drop += 1
            continue
        dw = das[r0:r1]
        rows.append(dict(w_idx=len(rows), t0_s=float(t[r0]), moving_frac=mf,
                         das_frac=float(np.nanmean(dw)) if np.isfinite(dw).any() else float("nan")))
        Xs.append(A[r0:r1])
    X = np.stack(Xs).astype(np.float32) if Xs else np.zeros((0, WINDOW_SAMPLES, N_CHANNELS), dtype=np.float32)
    meta = pd.DataFrame(rows, columns=["w_idx", "t0_s", "moving_frac", "das_frac"])
    fp = inf.get("footprint_m2", np.nan)
    fp = float(fp) if fp is not None and np.isfinite(float(fp)) else float("nan")
    size_class = inf.get("size_class") or classify_footprint(fp)
    meta = meta.assign(dataset=str(inf.get("dataset", "")), vehicle_id=str(inf.get("vehicle_id", "")), model=str(inf.get("model", "")),
                       platform=inf.get("platform"), drive_id=str(inf.get("drive_id", "")), footprint_m2=np.float32(fp),
                       size_class=size_class, powertrain=normalise_powertrain(inf.get("powertrain")),
                       borderline=bool(inf.get("borderline", False)) or is_borderline(fp), channels_present=",".join(present),
                       note=str(inf.get("note", "") or ""))
    meta = meta[[c for c, _ in META_COLUMNS]]
    stats = dict(n_rows=int(n), n_gap_rows=int(gap.sum()), n_runs=int(n_runs), n_candidates=int(len(starts)),
                 n_dropped_gap=int(n_gap_drop), n_dropped_stationary=int(n_stat_drop), n_windows=int(len(rows)))
    return CutResult(X, meta, stats)


def concat_results(results: Iterable[CutResult]) -> tuple[np.ndarray, pd.DataFrame]:
    """Stack the windows of several cut_windows calls (one per drive) -> (X, meta)."""
    results = [r for r in results if r.X.shape[0]]
    if not results:
        return np.zeros((0, WINDOW_SAMPLES, N_CHANNELS), dtype=np.float32), pd.DataFrame(columns=[c for c, _ in META_COLUMNS])
    return np.concatenate([r.X for r in results]), pd.concat([r.meta for r in results], ignore_index=True)


# ----------------------------------------------------------------------------- markdown
def md_table(headers: list[str], rows: list[list]) -> str:
    def fmt(x):
        if x is None:
            return "-"
        if isinstance(x, float):
            return "-" if not np.isfinite(x) else (f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}")
        return str(x)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


__all__ = [
    "CHANNELS", "CH", "N_CHANNELS", "WINDOW_SAMPLES", "HZ", "DT_S", "GAP_S", "MOVING_V_MPS", "MIN_MOVING_FRAC", "THRESHOLDS",
    "BORDERLINE_M2", "SIZE_CLASSES", "BEV_ICE", "POWERTRAINS", "TASK_LABEL", "BINARY_CHANNELS", "LEAD_CHANNELS", "LATERAL_CHANNELS",
    "CHANNEL_GROUPS", "MASK_PRESETS", "META_COLUMNS", "EXTERNAL_ROOT", "CKPT_DIR", "RESULTS_DIR", "TENSOR_DIR", "DATA_ROOT",
    "classify_footprint", "is_borderline", "normalise_powertrain", "task_labels", "resolve_mask", "apply_channel_mask", "mask_tag",
    "validate_windows", "save_windows", "load_windows", "summarize_windows", "work_dir", "make_grid", "resample_to_10hz",
    "curvature_and_lat_acc", "yaw_from_steering", "torque_norm", "smooth_derivative", "lead_episodes", "causal_slope",
    "lead_channels", "CutResult", "cut_windows", "concat_results", "md_table",
]
