"""Episode cache, drive bags and batches for the episode-centric multiple-instance (MIL) model.

Reads <data-root>/work/episodes/{E_f16.npy, G_f16.npy, episodes.parquet, channels.json} (the episode cache of
experiments/episodes_build.py: 18-s anchor-aligned crops, speed-indexed grids of the ACC / DEC episodes and the episode
table), joins labels, splits and auxiliary targets from build/index/drives.parquet, window flags from
build/index/windows.parquet and, optionally, the 60-s window embeddings work/tensors/wemb_<tag>.npy of
experiments/window_embed.py (tensor-cache order, aligned by window_id through work/tensors/windows.parquet). Provides
the drive-level eligibility sets lifted from the window protocols, the inner validation sets (split_platform: the
car-model-disjoint folds of experiments.internal_folds when present, else a seeded carve with the same rule;
split_device: the official device-disjoint validation, the protocol of its test, or a seeded carve of identity
groups), the auxiliary regression targets with the plausibility masks of experiments.data (PLAUSIBLE), normalisation
statistics, stratified bag sampling, collation and model-capped drive batches. make_synthetic() builds the same structures from
random arrays for the tests and the --synthetic-smoke mode; write_synthetic() writes them in the cache layout.
Nothing here reads the raw library; no identifier is printed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from vtt import config as C
from experiments.data import CLIP, PLAUSIBLE, TASKS, balanced_list_name

# ----------------------------------------------------------------------------- cache layout (contract with episodes_build)
EPISODE_TYPES = ("ACC", "DEC_LIFT", "DEC_BRK", "STOP", "LAUNCH", "LEAD_ACC", "LEAD_DEC", "TURN")
WINDOW_TYPE_ID = len(EPISODE_TYPES)              # the 60-s window embedding is the 9th instance type
N_TYPES = len(EPISODE_TYPES) + 1
CROP_LEN, CROP_OUT, CROP_OFFSET, JITTER_MAX = 180, 160, 10, 20     # stored crop, model input, eval offset, train jitter range
ONSET_SAMPLE, STOP_ONSET_SAMPLE = 40, 140
GRID_LEN = 72
CROP_CHANNELS = tuple(C.OFFICIAL_INPUTS) + ("a_sg", "valid")
GRID_CHANNELS = ("a_sg", "gas_pressed", "brake_pressed", "das_enabled", "valid", "t_from_onset_s")
CTX_NAMES = ("v_anchor_30", "dv_10", "dur_20", "mean_a", "peak_a", "das_frac", "lead_valid_frac", "thw_10", "gap_in_crop")
CTX_DIM = len(CTX_NAMES)
PRIORITY_TYPES = (5, 6, 7)                       # LEAD_ACC, LEAD_DEC, TURN are sampled first
PRIORITY_CAP = 16
LEAD_TYPES = (5, 6)
LATERAL_CHANNELS = ("steer_angle_deg", "steer_rate_degps", "steer_torque_norm", "yaw_rate_radps", "curv_1pm", "lat_acc_mps2")
BINARY_CHANNELS = ("gas_pressed", "brake_pressed", "steer_pressed", "standstill", "das_enabled", "lead_valid", "valid")
CLIP_EXTRA = {"a_sg": (-8.0, 8.0), "t_from_onset_s": (-30.0, 60.0)}
EPISODE_COLS = ("episode_id", "drive_id", "vehicle_id", "group_id", "type_id", "anchor_row", "i0", "i1", "crop_row0", "v_anchor",
                "dv", "dur_s", "mean_a", "peak_a", "resp_delay_s", "das_frac", "das_any_frac", "human_strict", "lead_valid_frac",
                "thw_anchor", "gap_in_crop", "w_idx", "window_id", "grid_idx")
DRIVE_COLS = ("drive_id", "vehicle_id", "group_id", "model", "size_class", "powertrain", "powertrain_raw", "footprint_m2",
              "curb_mass_kg", "wheelbase_mm", "split_platform", "split_device", "task_size_eligible", "task_bev_ice_eligible")
WIN_COLS = ("window_id", "drive_id", "w_idx", "eligible", "human_only_strict", "size_class", "powertrain", "split_platform",
            "split_device", "task_size_eligible", "task_bev_ice_eligible")
SIZE_CLASSES, BEV_CLASSES, PT4_CLASSES = tuple(C.SIZE_CLASSES), tuple(C.MAIN_POWERTRAIN_TASK), tuple(C.POWERTRAIN4)
FALLBACK_HOLDOUT_FRAC = 0.15
INNER_VAL_MODES = ("auto", "models", "groups", "official")   # auto: models for split_platform, official for split_device
Bag = tuple[np.ndarray, np.ndarray]              # (episode row indices, tensor indices of the window instances)
_EMPTY = np.zeros(0, dtype=np.int64)


# ----------------------------------------------------------------------------- pure helpers
def ctx_matrix(ep: pd.DataFrame) -> np.ndarray:
    """The 9-float context of every episode (contract: v/30, dv/10, dur/20, mean_a, peak_a, das_frac, lead_valid_frac,
    thw/10 (0 when NaN), gap_in_crop); non-finite values become 0."""
    f = lambda c: ep[c].to_numpy(dtype=np.float64, na_value=np.nan)  # noqa: E731
    thw = f("thw_anchor")
    thw = np.where(np.isfinite(thw), thw, 0.0)
    cols = [f("v_anchor") / 30.0, f("dv") / 10.0, f("dur_s") / 20.0, f("mean_a"), f("peak_a"), f("das_frac"), f("lead_valid_frac"),
            thw / 10.0, ep["gap_in_crop"].fillna(False).to_numpy(bool).astype(np.float64)]
    ctx = np.stack(cols, axis=1).astype(np.float32)
    return np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)


def drive_targets(drives: pd.DataFrame) -> pd.DataFrame:
    """Per-drive labels / auxiliary targets / eligibility (index = drive_id): y_size, y_bev, y_pt4 (int64, -1 = none),
    log_fp + fp_mask, log_mass + mass_mask, elig_size, elig_bev, plus model, vehicle_id, group_id and the split columns.
    The regression masks follow experiments.data.aux_targets (PLAUSIBLE): the curb mass must lie in 800-3500 kg and the
    footprint target needs a 2200-4000 mm wheelbase (a missing wheelbase column masks every footprint target)."""
    d = drives.drop_duplicates("drive_id").set_index("drive_id")
    size = d["size_class"].to_numpy(object)
    pt = d["powertrain"].to_numpy(object)
    out = pd.DataFrame(index=d.index)
    out["y_size"] = np.array([SIZE_CLASSES.index(s) if s in SIZE_CLASSES else -1 for s in size], dtype=np.int64)
    out["y_bev"] = np.array([BEV_CLASSES.index(p) if p in BEV_CLASSES else -1 for p in pt], dtype=np.int64)
    out["y_pt4"] = np.array([PT4_CLASSES.index(p) if p in PT4_CLASSES else -1 for p in pt], dtype=np.int64)
    fp = d["footprint_m2"].to_numpy(dtype=np.float64, na_value=np.nan)
    mass = d["curb_mass_kg"].to_numpy(dtype=np.float64, na_value=np.nan)
    wb = d["wheelbase_mm"].to_numpy(dtype=np.float64, na_value=np.nan) if "wheelbase_mm" in d else np.full(len(d), np.nan)
    (wb_lo, wb_hi), (m_lo, m_hi) = PLAUSIBLE["wheelbase_mm"], PLAUSIBLE["curb_mass_kg"]
    with np.errstate(invalid="ignore"):
        out["fp_mask"] = np.isfinite(fp) & (fp > 0) & np.isfinite(wb) & (wb >= wb_lo) & (wb <= wb_hi)
        out["mass_mask"] = np.isfinite(mass) & (mass >= m_lo) & (mass <= m_hi)
    out["log_fp"] = np.where(out["fp_mask"], np.log(np.where(out["fp_mask"], fp, 1.0)), 0.0).astype(np.float32)
    out["log_mass"] = np.where(out["mass_mask"], np.log(np.where(out["mass_mask"], mass, 1.0)), 0.0).astype(np.float32)
    out["elig_size"] = d["task_size_eligible"].fillna(False).to_numpy(bool) & (out["y_size"].to_numpy() >= 0)
    out["elig_bev"] = d["task_bev_ice_eligible"].fillna(False).to_numpy(bool) & (out["y_bev"].to_numpy() >= 0)
    for c in ("model", "vehicle_id", "group_id", "split_platform", "split_device"):
        out[c] = d[c].to_numpy(object)
    return out


def carve_models(drives: pd.DataFrame, seed: int, frac: float = FALLBACK_HOLDOUT_FRAC) -> tuple[set[str], set[str], list[str]]:
    """Fallback model-disjoint carve (same rule as contract (3) without the stratification): hold out `frac` of the
    car models of the pool with a seeded RNG; val = drives of a held-out model, train = pool minus val minus every
    drive whose group also drove a held-out model. `drives` has drive_id, model, group_id."""
    models = sorted({str(m) for m in drives["model"].dropna().unique()})
    if len(models) < 2:
        raise ValueError("the fallback carve needs at least two car models in the pool")
    rng = np.random.default_rng(seed)
    n_hold = min(len(models) - 1, max(1, int(round(frac * len(models)))))
    held = sorted(rng.choice(np.array(models, dtype=object), size=n_hold, replace=False).tolist())
    is_held = drives["model"].isin(held).to_numpy()
    val = set(drives["drive_id"][is_held].astype(str))
    tainted_groups = set(drives["group_id"][is_held].astype(str))
    keep = ~is_held & ~drives["group_id"].astype(str).isin(tainted_groups).to_numpy()
    train = set(drives["drive_id"][keep].astype(str))
    return train, val, held


def carve_groups(drives: pd.DataFrame, seed: int, frac: float = FALLBACK_HOLDOUT_FRAC) -> tuple[set[str], set[str], list[str]]:
    """Device-disjoint carve: hold out `frac` of the identity groups of the pool with a seeded RNG; val = their drives,
    train = the rest (car models may be shared, as in the split_device test). `drives` has drive_id and group_id."""
    groups = sorted({str(g) for g in drives["group_id"].dropna().unique()})
    if len(groups) < 2:
        raise ValueError("the group carve needs at least two identity groups in the pool")
    rng = np.random.default_rng(seed)
    n_hold = min(len(groups) - 1, max(1, int(round(frac * len(groups)))))
    held = sorted(rng.choice(np.array(groups, dtype=object), size=n_hold, replace=False).tolist())
    is_held = drives["group_id"].astype(str).isin(held).to_numpy()
    return set(drives["drive_id"][~is_held].astype(str)), set(drives["drive_id"][is_held].astype(str)), held


def resolve_inner_val(split_col: str, mode: str = "auto") -> str:
    """The inner-validation rule of a run: models (car-model-disjoint folds / carve), groups (device-disjoint carve) or
    official (the published validation split). auto = models for split_platform, official for split_device, whose test
    shares car models and holds out devices, so its validation must do the same."""
    if mode not in INNER_VAL_MODES:
        raise ValueError(f"unknown inner validation mode {mode!r}; choose from {INNER_VAL_MODES}")
    if mode != "auto":
        return mode
    return "models" if split_col == "split_platform" else "official"


def drive_sampling_weights(tgt: pd.DataFrame, drive_ids: Sequence[str], task: str = "size", power: float = C.CLASS_WEIGHT_POWER,
                           cap: float = C.CLASS_WEIGHT_CAP, model_power: float = 0.5) -> np.ndarray:
    """Per-drive sampling weight = class weight (N / (K n_k))^power (the dataset's train_uniform rule, capped) times a
    within-class car-model balance (n_class / (n_models_in_class * n_model))^model_power, capped at `cap`; drives without
    a label of the task get the class weight 1. Normalised to mean 1 over the given drives."""
    sub = tgt.loc[list(drive_ids)]
    y = sub["y_size" if task == "size" else "y_bev"].to_numpy()
    k = len(SIZE_CLASSES if task == "size" else BEV_CLASSES)
    w = np.ones(len(sub), dtype=np.float64)
    lab = y >= 0
    if lab.any():
        counts = np.bincount(y[lab], minlength=k).astype(np.float64)
        w_class = np.minimum((lab.sum() / (k * np.maximum(counts, 1))) ** power, cap)
        w[lab] = w_class[y[lab]]
    models = sub["model"].astype(str).to_numpy()
    for cls in np.unique(y):
        sel = np.flatnonzero(y == cls)
        ms, inv, cnt = np.unique(models[sel], return_inverse=True, return_counts=True)
        w_model = np.minimum((len(sel) / (len(ms) * cnt)) ** model_power, cap)
        w[sel] *= w_model[inv]
    return (w / w.mean()).astype(np.float64)


def load_wemb(wemb: Mapping[str, Path | str] | None) -> tuple[np.ndarray | None, list[str]]:
    """Concatenate the window embeddings of the given checkpoint tags (sorted by tag) along the feature axis."""
    if not wemb:
        return None, []
    tags, parts = sorted(wemb), []
    for tag in tags:
        p = Path(wemb[tag])
        arr = np.load(p, mmap_mode="r")
        side = p.with_suffix(".json")
        if side.exists():
            meta = json.loads(side.read_text())
            if int(meta.get("dim", arr.shape[1])) != arr.shape[1]:
                raise ValueError(f"embedding width {arr.shape[1]} differs from the sidecar of tag #{tags.index(tag)}")
        parts.append(np.ascontiguousarray(arr, dtype=np.float16))
    if len({p.shape[0] for p in parts}) != 1:
        raise ValueError("window embeddings of different row counts cannot be concatenated")
    return (parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)), tags


def pq_columns(path: Path) -> list[str]:
    """Column names of a parquet file without reading its rows."""
    import pyarrow.parquet as pq
    return list(pq.read_schema(path).names)


def _clip_bounds(channels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    table = {**CLIP, **CLIP_EXTRA}
    lo = np.array([table.get(c, (-np.inf, np.inf))[0] for c in channels], dtype=np.float32)
    hi = np.array([table.get(c, (-np.inf, np.inf))[1] for c in channels], dtype=np.float32)
    return lo, hi


def _stats(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, channels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean / std of (n, L, C) after clipping, NaN ignored; binary flags keep mean 0 / std 1."""
    if len(x) == 0:
        return np.zeros(len(channels), np.float32), np.ones(len(channels), np.float32)
    x = np.clip(np.asarray(x, dtype=np.float32), lo, hi)
    with np.errstate(all="ignore"):
        mean = np.nanmean(x, axis=(0, 1))
        std = np.nanstd(x, axis=(0, 1))
    mean[~np.isfinite(mean)] = 0.0
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    binary = np.array([c in BINARY_CHANNELS for c in channels])
    mean[binary], std[binary] = 0.0, 1.0
    return mean.astype(np.float32), std.astype(np.float32)


# ----------------------------------------------------------------------------- the data object
class EpisodeData:
    """Episode cache + drive table + window flags (+ window embeddings), with the sampling / collation utilities."""

    def __init__(self, root: Path | str = C.DATA_ROOT, wemb: Mapping[str, Path | str] | None = None, in_ram: bool = True):
        root = Path(root)
        edir = root / "work" / "episodes"
        E = np.load(edir / "E_f16.npy", mmap_mode="r")
        G = np.load(edir / "G_f16.npy", mmap_mode="r")
        if in_ram:
            E, G = np.array(E, copy=True), np.array(G, copy=True)        # real copies (ascontiguousarray would keep the memmap)
        ep = pd.read_parquet(edir / "episodes.parquet")
        channels = json.loads((edir / "channels.json").read_text())
        dpath = root / "build" / "index" / "drives.parquet"
        present = set(pq_columns(dpath))
        drives = pd.read_parquet(dpath, columns=[c for c in DRIVE_COLS if c in present])
        win = pd.read_parquet(root / "build" / "index" / "windows.parquet", columns=list(WIN_COLS))
        tmeta = pd.read_parquet(root / "work" / "tensors" / "windows.parquet", columns=["window_id", "tensor_idx"])
        wemb_arr, tags = load_wemb(wemb)
        self._setup(E, G, ep, channels, drives, win, tmeta, wemb_arr, tags, root=root)

    @classmethod
    def from_arrays(cls, E: np.ndarray, G: np.ndarray, ep: pd.DataFrame, channels: dict, drives: pd.DataFrame, win: pd.DataFrame,
                    tmeta: pd.DataFrame, wemb: np.ndarray | None = None, wemb_tags: Sequence[str] = (), root: Path | None = None,
                    balanced: dict | None = None) -> "EpisodeData":
        obj = cls.__new__(cls)
        obj._setup(E, G, ep, channels, drives, win, tmeta, wemb, list(wemb_tags), root=root, balanced=balanced)
        return obj

    def _setup(self, E, G, ep, channels, drives, win, tmeta, wemb, wemb_tags, root=None, balanced=None) -> None:
        if tuple(channels["crop"]) != CROP_CHANNELS or tuple(channels["grid"]) != GRID_CHANNELS:
            raise ValueError("episode cache channel order differs from the contract")
        if tuple(channels.get("types", EPISODE_TYPES)) != EPISODE_TYPES:
            raise ValueError("episode type order differs from the contract")
        if E.ndim != 3 or E.shape[1:] != (CROP_LEN, len(CROP_CHANNELS)):
            raise ValueError(f"E has shape {tuple(E.shape)}, expected (N, {CROP_LEN}, {len(CROP_CHANNELS)})")
        if G.ndim != 3 or G.shape[1:] != (GRID_LEN, len(GRID_CHANNELS)):
            raise ValueError(f"G has shape {tuple(G.shape)}, expected (N_g, {GRID_LEN}, {len(GRID_CHANNELS)})")
        if len(ep) != E.shape[0]:
            raise ValueError(f"{len(ep)} episode rows for {E.shape[0]} crops")
        missing = [c for c in EPISODE_COLS if c not in ep.columns]
        if missing:
            raise ValueError(f"episodes table lacks {len(missing)} contract columns: {missing}")
        self.root = Path(root) if root is not None else None          # None: in-memory (synthetic) data, no fold file lookup
        self.E, self.G = E, G
        self.channels = dict(crop=list(CROP_CHANNELS), grid=list(GRID_CHANNELS), types=list(EPISODE_TYPES))
        self.ep = ep.reset_index(drop=True)
        self.type_id = self.ep["type_id"].to_numpy().astype(np.int64)
        if self.type_id.min(initial=0) < 0 or self.type_id.max(initial=0) >= len(EPISODE_TYPES):
            raise ValueError("episode type_id outside 0..7")
        self.anchor_row = self.ep["anchor_row"].to_numpy().astype(np.int64)
        self.human_strict = self.ep["human_strict"].fillna(False).to_numpy(bool)
        self.grid_idx = self.ep["grid_idx"].fillna(-1).to_numpy().astype(np.int64)
        if (self.grid_idx >= G.shape[0]).any():
            raise ValueError("grid_idx beyond the grid array")
        self.ep_drive = self.ep["drive_id"].astype(str).to_numpy(object)
        self.ctx = ctx_matrix(self.ep)
        # drives: labels, targets, eligibility, splits
        self.tgt = drive_targets(drives)
        self.drives_df = drives.drop_duplicates("drive_id").reset_index(drop=True)
        self._drive_pos = {d: i for i, d in enumerate(self.tgt.index.astype(str))}
        unknown = sum(1 for d in np.unique(self.ep_drive) if d not in self._drive_pos)
        if unknown:
            raise ValueError(f"{unknown} episode drives are missing from the drives index")
        # windows: flags + tensor position (-1 when the window is not in the tensor cache)
        w = win.drop_duplicates("window_id").merge(tmeta[["window_id", "tensor_idx"]].drop_duplicates("window_id"), on="window_id", how="left")
        w["tensor_idx"] = w["tensor_idx"].fillna(-1).astype(np.int64)
        self.win = w.reset_index(drop=True)
        self.win_drive = self.win["drive_id"].astype(str).to_numpy(object)
        self.win_tidx = self.win["tensor_idx"].to_numpy()
        self.win_human = self.win["human_only_strict"].fillna(False).to_numpy(bool)
        # embeddings
        self.wemb, self.wemb_tags = wemb, list(wemb_tags)
        self.wemb_dim = 0 if wemb is None else int(wemb.shape[1])
        if wemb is not None and self.win_tidx.max(initial=-1) >= wemb.shape[0]:
            raise ValueError("tensor_idx beyond the window embedding array")
        # index maps
        self._ep_by_drive = {str(k): np.asarray(v, dtype=np.int64) for k, v in self.ep.groupby("drive_id", sort=False).indices.items()}
        wid = self.ep["window_id"]
        has_w = wid.notna().to_numpy()
        self._ep_by_window = {str(k): np.asarray(v, dtype=np.int64) for k, v in self.ep[has_w].groupby("window_id", sort=False).indices.items()}
        for k, v in self._ep_by_window.items():            # groupby on the filtered frame returns positions in that frame
            self._ep_by_window[k] = np.flatnonzero(has_w)[v]
        has_t = self.win_tidx >= 0
        self._win_by_drive = {str(k): np.flatnonzero(has_t)[np.asarray(v, dtype=np.int64)]
                              for k, v in self.win[has_t].groupby("drive_id", sort=False).indices.items()}
        self.crop_lo, self.crop_hi = _clip_bounds(CROP_CHANNELS)
        self.grid_lo, self.grid_hi = _clip_bounds(GRID_CHANNELS)
        self.lateral_idx = np.array([CROP_CHANNELS.index(c) for c in LATERAL_CHANNELS], dtype=np.int64)
        self.binary_idx = np.array([i for i, c in enumerate(CROP_CHANNELS) if c in BINARY_CHANNELS], dtype=np.int64)
        self.splits_dir = (self.root or Path(C.DATA_ROOT)) / "build" / "splits"
        self._balanced = dict(balanced) if balanced else None

    # ------------------------------------------------------------------ counts
    @property
    def n_episodes(self) -> int:
        return int(self.E.shape[0])

    def n_episodes_of(self, drive_id: str, human_only: bool = False) -> int:
        idx = self._ep_by_drive.get(drive_id, _EMPTY)
        return int(self.human_strict[idx].sum()) if human_only else int(len(idx))

    def summary(self) -> dict:
        return dict(n_episodes=self.n_episodes, n_grids=int(self.G.shape[0]), n_drives_with_episodes=len(self._ep_by_drive),
                    n_windows=int(len(self.win)), n_windows_in_cache=int((self.win_tidx >= 0).sum()), wemb_dim=self.wemb_dim,
                    type_counts={t: int(c) for t, c in zip(EPISODE_TYPES, np.bincount(self.type_id, minlength=len(EPISODE_TYPES)))})

    # ------------------------------------------------------------------ windows -> drives
    def balanced_ids(self, task: str, split: str = "test", split_col: str = "split_platform") -> set[str]:
        if self._balanced is not None:
            return set(self._balanced.get((task, split, split_col), set()))
        return set(pd.read_parquet(self.splits_dir / balanced_list_name(task, split, split_col))["window_id"].astype(str))

    def window_mask(self, task: str, split_col: str, split: str, protocol: str = "test_full") -> np.ndarray:
        """Row mask of self.win with the window protocol definitions of experiments.data.WindowData.mask."""
        t = TASKS[task]
        w = self.win
        m = (w[split_col].to_numpy(object) == split)
        m &= w["eligible"].fillna(False).to_numpy(bool)
        m &= w[t["eligible"]].fillna(False).to_numpy(bool)
        m &= w[t["label"]].isin(t["classes"]).to_numpy()
        if protocol == "test_balanced":
            m &= w["window_id"].isin(self.balanced_ids(task, split, split_col)).to_numpy()
        elif protocol == "test_human_only":
            m &= self.win_human
        elif protocol != "test_full":
            raise ValueError(protocol)
        return m

    def window_rows(self, task: str, split_col: str, split: str, protocol: str = "test_full") -> np.ndarray:
        return np.flatnonzero(self.window_mask(task, split_col, split, protocol))

    def drive_sets(self, task: str, split_col: str, split: str, protocol: str = "test_full") -> set[str]:
        """Drives with >= 1 window of the protocol (test_full -> drive_full; test_human_only -> drive_human_only)."""
        return set(self.win_drive[self.window_mask(task, split_col, split, protocol)])

    def drive_pool(self, split_col: str, split: str) -> set[str]:
        """Union over the two tasks of the drives with >= 1 eligible window in the split."""
        return self.drive_sets("size", split_col, split) | self.drive_sets("bev_ice", split_col, split)

    def task_drives(self, task: str, split_col: str, split: str) -> np.ndarray:
        """Drives of the split that are eligible for the task and labelled: the unit universe of contract (4) (the ones
        without an eligible window are counted as n_units_without_windows), as experiments.train.unit_protocols does."""
        t = TASKS[task]
        d = self.drives_df
        m = (d[split_col].to_numpy(object) == split) & d[t["eligible"]].fillna(False).to_numpy(bool) & d[t["label"]].isin(t["classes"]).to_numpy()
        return d.loc[m, "drive_id"].astype(str).to_numpy(object)

    def window_labels(self, task: str, rows: np.ndarray) -> np.ndarray:
        t = TASKS[task]
        cls = {c: i for i, c in enumerate(t["classes"])}
        return np.array([cls[x] for x in self.win[t["label"]].to_numpy(object)[rows]], dtype=np.int64)

    # ------------------------------------------------------------------ folds
    def fold_sets(self, split_col: str, fold: int = 0, seed: int = 0, task: str = "size", mode: str = "auto") -> tuple[set[str], set[str], dict]:
        """Inner-validation (train, validation) drive sets inside the train + validation pool of `split_col`, plus info
        (source, mode, fold, seed, held-out counts).

        mode (resolve_inner_val): models = car-model-disjoint: experiments.internal_folds (contract (3), split_platform
        only) when the module and its JSON exist, else the seeded carve `carve_models`; groups = device-disjoint carve
        of identity groups (`carve_groups`); official = the published validation split. auto = models for
        split_platform and official for split_device, whose test shares car models and holds out devices. Both sets are
        intersected with the drives that have >= 1 eligible window of either task."""
        resolved = resolve_inner_val(split_col, mode)
        pool = self.drive_pool(split_col, "train") | self.drive_pool(split_col, "validation")
        pool_df = self.drives_df[self.drives_df["drive_id"].astype(str).isin(pool)]
        info: dict = dict(fold=int(fold), seed=int(seed), split_col=split_col, mode=resolved)
        if resolved == "official":
            info.update(source="official_validation")
            return self.drive_pool(split_col, "train"), self.drive_pool(split_col, "validation"), info
        if resolved == "groups":
            tr, va, held = carve_groups(pool_df, seed=seed * 1000 + fold)
            info.update(source="group_carve", n_heldout_groups=len(held))
            return tr & pool, va & pool, info
        if split_col == "split_platform" and self.root is not None:
            try:
                from experiments import internal_folds as IF          # contract (3): experiments/internal_folds.py (package C1)
                folds = IF.load_folds(task, data_root=self.root)
                tr, va = IF.fold_drive_sets(self.drives_df, folds, fold)
                tr, va = set(map(str, tr)) & pool, set(map(str, va)) & pool
                if tr and va:
                    info.update(source="internal_folds", n_heldout_models=len(folds["folds"][fold]["heldout_models"]))
                    return tr, va, info
                info["internal_folds_note"] = "fold gave an empty train or validation set; fell back to the carve"
            except (ImportError, FileNotFoundError, KeyError, IndexError, ValueError, TypeError) as e:
                info["internal_folds_note"] = f"{type(e).__name__}; fell back to the carve"
        tr, va, held = carve_models(pool_df, seed=seed * 1000 + fold)
        info.update(source="fallback_carve", n_heldout_models=len(held))
        return tr & pool, va & pool, info

    # ------------------------------------------------------------------ statistics (train drives only)
    def episode_indices(self, drive_ids: Sequence[str]) -> np.ndarray:
        parts = [self._ep_by_drive[d] for d in drive_ids if d in self._ep_by_drive]
        return np.concatenate(parts) if parts else _EMPTY

    def norm_stats(self, drive_ids: Sequence[str], max_episodes: int = 20000, seed: int = 0) -> dict:
        """Per crop channel and per grid channel mean / std after clipping (NaN ignored, flags untouched)."""
        rng = np.random.default_rng(seed)
        idx = self.episode_indices(drive_ids)
        sel = np.sort(rng.choice(idx, size=min(max_episodes, len(idx)), replace=False)) if len(idx) else idx
        cm, cs = _stats(self.E[sel] if len(sel) else np.zeros((0, CROP_LEN, len(CROP_CHANNELS)), np.float32), self.crop_lo, self.crop_hi, CROP_CHANNELS)
        g = self.grid_idx[sel]
        g = np.sort(g[g >= 0])
        gm, gs = _stats(self.G[g] if len(g) else np.zeros((0, GRID_LEN, len(GRID_CHANNELS)), np.float32), self.grid_lo, self.grid_hi, GRID_CHANNELS)
        return dict(crop_mean=cm, crop_std=cs, grid_mean=gm, grid_std=gs, n_episodes=int(len(sel)), n_grids=int(len(g)))

    def target_stats(self, drive_ids: Sequence[str]) -> dict:
        """Mean / std of log footprint and log curb mass over the given drives (masked), for standardised regression."""
        sub = self.tgt.loc[list(drive_ids)]
        out = {}
        for name, mask in (("log_fp", "fp_mask"), ("log_mass", "mass_mask")):
            v = sub[name].to_numpy(np.float64)[sub[mask].to_numpy(bool)]
            mu = float(v.mean()) if len(v) else 0.0
            sd = float(v.std()) if len(v) > 1 and v.std() > 1e-6 else 1.0
            out[f"{name}_mu"], out[f"{name}_sd"] = mu, sd
        return out

    def sampling_weights(self, drive_ids: Sequence[str], task: str = "size", **kw) -> np.ndarray:
        return drive_sampling_weights(self.tgt, drive_ids, task=task, **kw)

    def model_of(self, drive_ids: Sequence[str]) -> np.ndarray:
        return self.tgt.loc[list(drive_ids), "model"].astype(str).to_numpy(object)

    def vehicle_of(self, drive_ids: Sequence[str]) -> np.ndarray:
        return self.tgt.loc[list(drive_ids), "vehicle_id"].astype(str).to_numpy(object)

    # ------------------------------------------------------------------ bags
    def sample_bag(self, drive_id: str, K: int = 64, K_w: int = 8, rng: np.random.Generator | None = None, human_only: bool = False) -> Bag:
        """Stratified episode bag of one drive: LEAD_ACC, LEAD_DEC and TURN first (at most PRIORITY_CAP each), then the
        remaining episodes fill up to K at random; plus up to K_w of the drive's cached windows (embedding instances).
        human_only keeps human_strict episodes and human_only_strict windows only."""
        rng = rng or np.random.default_rng(0)
        cand = self._ep_by_drive.get(drive_id, _EMPTY)
        if human_only:
            cand = cand[self.human_strict[cand]]
        chosen: list[np.ndarray] = []
        n = 0
        types = self.type_id[cand]
        for t in PRIORITY_TYPES:
            c = cand[types == t]
            take = min(len(c), PRIORITY_CAP, K - n)
            if take > 0:
                chosen.append(rng.choice(c, size=take, replace=False))
                n += take
        taken = np.concatenate(chosen) if chosen else _EMPTY
        rest = np.setdiff1d(cand, taken, assume_unique=True)
        fill = min(len(rest), K - n)
        if fill > 0:
            chosen.append(rng.choice(rest, size=fill, replace=False))
        ep_idx = np.sort(np.concatenate(chosen)) if chosen else _EMPTY
        t_idx = _EMPTY
        if self.wemb is not None and K_w > 0:
            wr = self._win_by_drive.get(drive_id, _EMPTY)
            if human_only:
                wr = wr[self.win_human[wr]]
            if len(wr):
                wr = np.sort(rng.choice(wr, size=min(K_w, len(wr)), replace=False))
                t_idx = self.win_tidx[wr].astype(np.int64)
        return ep_idx.astype(np.int64), t_idx

    def bag_for_window(self, win_row: int, human_only: bool = False, use_windows: bool = True) -> Bag:
        """Instances of one window: the episodes whose anchor falls inside it + the window's own embedding."""
        wid = self.win["window_id"].iat[int(win_row)]
        ep_idx = self._ep_by_window.get(str(wid), _EMPTY)
        if human_only:
            ep_idx = ep_idx[self.human_strict[ep_idx]]
        t = int(self.win_tidx[win_row])
        t_idx = np.array([t], dtype=np.int64) if (use_windows and self.wemb is not None and t >= 0) else _EMPTY
        return ep_idx.astype(np.int64), t_idx

    def collate_bags(self, bags: Sequence[Bag], drive_ids: Sequence[str], train: bool = False, rng: np.random.Generator | None = None) -> dict:
        """Flat instance arrays + padded bag mask + per-bag targets (numpy; see to_torch).

        x (M, 160, 22) f16 crops at offset 10 (train: a random anchor jitter 0..20 per instance), grid (M_g, 72, 6),
        grid_map (M,) row into grid or -1, ctx (M, 9), type (M,), bag_idx (M,), pos (M,), wemb (M_w, dim), wemb_bag_idx,
        wemb_pos, mask (B, Kmax), y_size, y_bev, y_pt4, log_fp, fp_mask, log_mass, mass_mask, elig_size, elig_bev."""
        B = len(bags)
        if B != len(drive_ids):
            raise ValueError("one drive id per bag")
        n_ep = np.array([len(b[0]) for b in bags], dtype=np.int64)
        n_w = np.array([len(b[1]) for b in bags], dtype=np.int64)
        all_ep = np.concatenate([b[0] for b in bags]) if B else _EMPTY
        all_t = np.concatenate([b[1] for b in bags]) if B else _EMPTY
        M, M_w = len(all_ep), len(all_t)
        if train and rng is not None:
            offs = rng.integers(0, JITTER_MAX + 1, size=M)
        else:
            offs = np.full(M, CROP_OFFSET, dtype=np.int64)
        if M:
            order = np.argsort(all_ep, kind="stable")        # sorted reads from the (possibly memory-mapped) crop array
            crops = np.asarray(self.E[all_ep[order]])
            inv = np.empty_like(order)
            inv[order] = np.arange(M)
            crops = crops[inv]
            x = crops[np.arange(M)[:, None], offs[:, None] + np.arange(CROP_OUT)[None, :]]
        else:
            x = np.zeros((0, CROP_OUT, len(CROP_CHANNELS)), dtype=np.float16)
        gi = self.grid_idx[all_ep] if M else _EMPTY
        has = gi >= 0
        grid = np.asarray(self.G[gi[has]]) if has.any() else np.zeros((0, GRID_LEN, len(GRID_CHANNELS)), dtype=np.float16)
        grid_map = np.full(M, -1, dtype=np.int64)
        grid_map[has] = np.arange(int(has.sum()))
        bag_idx = np.repeat(np.arange(B), n_ep)
        pos = np.concatenate([np.arange(n) for n in n_ep]) if M else _EMPTY
        wemb_bag_idx = np.repeat(np.arange(B), n_w)
        wemb_pos = np.concatenate([n_ep[b] + np.arange(n_w[b]) for b in range(B)]) if M_w else _EMPTY
        wemb = (np.asarray(self.wemb[all_t]) if (M_w and self.wemb is not None)
                else np.zeros((0, self.wemb_dim), dtype=np.float16))
        kmax = int(max(1, (n_ep + n_w).max() if B else 1))
        mask = np.zeros((B, kmax), dtype=bool)
        for b in range(B):
            mask[b, : n_ep[b] + n_w[b]] = True
        t = self.tgt.loc[list(drive_ids)]
        return dict(x=np.ascontiguousarray(x, dtype=np.float16), grid=np.ascontiguousarray(grid, dtype=np.float16), grid_map=grid_map,
                    ctx=self.ctx[all_ep] if M else np.zeros((0, CTX_DIM), np.float32), type=self.type_id[all_ep] if M else _EMPTY,
                    bag_idx=bag_idx, pos=pos.astype(np.int64), wemb=wemb, wemb_bag_idx=wemb_bag_idx, wemb_pos=wemb_pos.astype(np.int64),
                    mask=mask, y_size=t["y_size"].to_numpy(np.int64), y_bev=t["y_bev"].to_numpy(np.int64), y_pt4=t["y_pt4"].to_numpy(np.int64),
                    log_fp=t["log_fp"].to_numpy(np.float32), fp_mask=t["fp_mask"].to_numpy(bool), log_mass=t["log_mass"].to_numpy(np.float32),
                    mass_mask=t["mass_mask"].to_numpy(bool), elig_size=t["elig_size"].to_numpy(bool), elig_bev=t["elig_bev"].to_numpy(bool),
                    n_inst=(n_ep + n_w).astype(np.int64), drive_id=np.asarray(list(drive_ids), dtype=object))


def to_torch(batch: dict, device=None) -> dict:
    """numpy collate -> torch tensors (float16 crops kept; bool masks kept; drive_id stays a numpy array)."""
    import torch
    out = {}
    for k, v in batch.items():
        if k == "drive_id":
            out[k] = v
            continue
        t = torch.from_numpy(np.ascontiguousarray(v))
        out[k] = t.to(device, non_blocking=True) if device is not None else t
    return out


def iterate_drive_batches(drive_ids: Sequence[str], weights: np.ndarray | None, model_of, B: int, rng: np.random.Generator,
                          max_model_share: float = 0.10, n_draws: int | None = None) -> Iterator[np.ndarray]:
    """Weighted draws with replacement, packed into batches of B drives in which no car model exceeds
    max(1, floor(max_model_share * B)) drives (the cap is raised when the pool has too few models to fill a batch).
    model_of: mapping drive -> model, or an array aligned with drive_ids. Yields ceil(n_draws / B) batches (n_draws
    defaults to len(drive_ids))."""
    ids = np.asarray(list(drive_ids), dtype=object)
    if len(ids) == 0:
        return
    if isinstance(model_of, (np.ndarray, list, tuple)) and len(model_of) == len(ids) and not isinstance(model_of, Mapping):
        models = np.asarray(list(model_of), dtype=object)
    else:
        models = np.asarray([model_of[d] for d in ids], dtype=object)
    _, code = np.unique(models.astype(str), return_inverse=True)
    n_models = int(code.max()) + 1
    cap = max(1, int(math.floor(max_model_share * B)))
    if n_models * cap < B:
        cap = int(math.ceil(B / n_models))
    p = None if weights is None else np.asarray(weights, dtype=np.float64) / np.sum(weights)
    n = n_draws or len(ids)
    for _ in range(int(math.ceil(n / B))):
        counts = np.zeros(n_models, dtype=np.int64)
        chosen: list[int] = []
        attempts = 0
        while len(chosen) < B and attempts < 50:
            cands = rng.choice(len(ids), size=4 * B, replace=True, p=p)
            for i in cands:
                if counts[code[i]] < cap:
                    counts[code[i]] += 1
                    chosen.append(int(i))
                    if len(chosen) == B:
                        break
            attempts += 1
        yield ids[np.array(chosen, dtype=np.int64)]


# ----------------------------------------------------------------------------- synthetic structures (tests / smoke)
_SYNTH_MODELS = [
    # footprint m2, curb mass kg, wheelbase mm, powertrain, split_platform
    (4.10, 1300.0, 2550.0, "ICE", "train"), (4.25, 1700.0, 2600.0, "BEV", "train"), (4.60, 1500.0, 2700.0, "ICE", "train"),
    (4.70, 2000.0, 2750.0, "BEV", "train"), (5.00, 2300.0, 2900.0, "ICE", "train"), (5.20, 2600.0, 3000.0, "BEV", "train"),
    (4.35, 1600.0, 2650.0, "HEV", "train"), (np.nan, np.nan, np.nan, None, "train"),
    (4.50, 1550.0, 2700.0, "BEV", "validation"), (5.05, 2400.0, 2950.0, "ICE", "validation"),
    (4.20, 1400.0, 2600.0, "BEV", "test"), (4.65, 1800.0, 2750.0, "ICE", "test"), (5.10, 2500.0, 2950.0, "BEV", "test"),
]


def _size_class(fp: float) -> str | None:
    if not np.isfinite(fp):
        return None
    t1, t2 = C.FOOTPRINT_THRESHOLDS_M2
    return "SMALL" if fp < t1 else ("MID" if fp < t2 else "LARGE")


def make_synthetic(n_drives: int = 60, rng: np.random.Generator | None = None, wemb_dim: int = 16, episodes_per_drive: tuple[int, int] = (6, 24),
                   windows_per_drive: tuple[int, int] = (1, 4)) -> EpisodeData:
    """Random episode cache + drive / window tables with the contract layout (13 fictitious car models, model-disjoint
    split_platform, per-drive split_device, one model without footprint and one without a powertrain label). Signals
    are smooth random curves whose acceleration amplitude grows with the size class so that learning is possible."""
    rng = rng or np.random.default_rng(0)
    nm = len(_SYNTH_MODELS)
    model_of_drive = np.array([i % nm if i < 2 * nm else int(rng.integers(nm)) for i in range(n_drives)], dtype=np.int64)
    n_veh = np.array([int(rng.integers(1, 4)) for _ in range(nm)])
    veh_offset = np.concatenate([[0], np.cumsum(n_veh)[:-1]])
    veh_of_drive = np.array([veh_offset[m] + int(rng.integers(n_veh[m])) for m in model_of_drive], dtype=np.int64)
    dev_split = rng.choice(np.array(["train", "validation", "test"], dtype=object), size=n_drives, p=[0.7, 0.15, 0.15])
    drows, wrows, erows = [], [], []
    E_parts, G_parts, wemb_rows = [], [], []
    grid_count = 0
    t_idx = 0
    t_crop = np.arange(CROP_LEN, dtype=np.float32)
    for i in range(n_drives):
        m = model_of_drive[i]
        fp, mass, wb, pt, split_p = _SYNTH_MODELS[m]
        size = _size_class(fp)
        did, vid, gid = C.DRIVE_FMT.format(i), C.VEHICLE_FMT.format(veh_of_drive[i]), C.GROUP_FMT.format(veh_of_drive[i])
        drows.append(dict(drive_id=did, vehicle_id=vid, group_id=gid, model=f"Synth-{m:02d}", size_class=size, powertrain=pt,
                          powertrain_raw=pt or "MIXED", footprint_m2=fp, curb_mass_kg=mass, wheelbase_mm=wb, split_platform=split_p,
                          split_device=dev_split[i],
                          task_size_eligible=size is not None, task_bev_ice_eligible=pt in BEV_CLASSES))
        y_size = SIZE_CLASSES.index(size) if size else 1
        n_w = int(rng.integers(windows_per_drive[0], windows_per_drive[1] + 1))
        wins = {}
        for w in range(n_w):
            elig = bool(rng.random() > 0.15)
            wid = f"{did}_w{w}"
            wrows.append(dict(window_id=wid, drive_id=did, vehicle_id=vid, group_id=gid, w_idx=w, eligible=elig,
                              human_only_strict=bool(rng.random() < 0.5), size_class=size, powertrain=pt, split_platform=split_p,
                              split_device=dev_split[i], task_size_eligible=size is not None, task_bev_ice_eligible=pt in BEV_CLASSES,
                              tensor_idx=t_idx if elig else -1))
            if elig:
                wins[w] = wid
                wemb_rows.append((rng.normal(0, 1, wemb_dim) + 0.5 * y_size).astype(np.float16))
                t_idx += 1
        n_ep = int(rng.integers(episodes_per_drive[0], episodes_per_drive[1] + 1))
        types = rng.choice(len(EPISODE_TYPES), size=n_ep, p=[0.22, 0.13, 0.13, 0.08, 0.08, 0.12, 0.12, 0.12])
        for e in range(n_ep):
            ty = int(types[e])
            anchor = int(rng.integers(0, n_w * 600))
            w_idx = anchor // 600
            v0 = float(rng.uniform(0.0, 30.0))
            dv = float(rng.uniform(1.0, 8.0)) * (1 if ty in (0, 4, 5) else -1)
            dur = float(rng.uniform(3.0, 14.0))
            amp = (1.0 + 0.25 * y_size) * abs(dv) / dur
            das_any = float(rng.choice([0.0, rng.uniform(0.1, 1.0)]))
            das = das_any * float(rng.uniform(0.3, 1.0))
            lead_valid = float(rng.uniform(0.0, 1.0)) if ty in LEAD_TYPES or rng.random() < 0.5 else 0.0
            gap = bool(rng.random() < 0.1)
            has_grid = ty in (0, 1, 2)
            erows.append(dict(episode_id=len(erows), drive_id=did, vehicle_id=vid, group_id=gid, type_id=ty, anchor_row=anchor,
                              i0=max(0, anchor - ONSET_SAMPLE), i1=anchor + int(dur * 10), crop_row0=anchor - (STOP_ONSET_SAMPLE if ty == 3 else ONSET_SAMPLE),
                              v_anchor=v0, dv=dv, dur_s=dur, mean_a=dv / dur, peak_a=amp * np.sign(dv), resp_delay_s=float(rng.uniform(0.3, 3.0)) if ty in LEAD_TYPES else np.nan,
                              das_frac=das, das_any_frac=das_any, human_strict=das_any == 0.0, lead_valid_frac=lead_valid,
                              thw_anchor=float(rng.uniform(0.5, 6.0)) if rng.random() < 0.7 else np.nan, gap_in_crop=gap, w_idx=w_idx,
                              window_id=wins.get(w_idx), grid_idx=grid_count if has_grid else -1))
            onset = STOP_ONSET_SAMPLE if ty == 3 else ONSET_SAMPLE
            s = 1.0 / (1.0 + np.exp(-(t_crop - onset) / (dur * 10 / 6)))
            v = v0 + dv * s
            a = np.gradient(v) * 10.0 * (amp / max(1e-3, abs(np.gradient(v)).max() * 10.0))
            crop = np.zeros((CROP_LEN, len(CROP_CHANNELS)), dtype=np.float32)
            ch = {c: k for k, c in enumerate(CROP_CHANNELS)}
            crop[:, ch["v_mps"]] = v
            crop[:, ch["a_mps2"]] = a + rng.normal(0, 0.05, CROP_LEN)
            crop[:, ch["a_sg"]] = a
            ang = 5.0 * np.sin(t_crop / 20.0 + rng.uniform(0, 6.28)) * (3.0 if ty == 7 else 1.0)
            crop[:, ch["steer_angle_deg"]] = ang
            crop[:, ch["steer_rate_degps"]] = np.gradient(ang) * 10.0
            crop[:, ch["steer_torque_norm"]] = ang / 10.0
            crop[:, ch["yaw_rate_radps"]] = ang / 100.0
            crop[:, ch["curv_1pm"]] = ang / 1000.0
            crop[:, ch["lat_acc_mps2"]] = ang / 50.0
            crop[:, ch["gas_pressed"]] = (a > 0.2).astype(np.float32)
            crop[:, ch["brake_pressed"]] = (a < -0.2).astype(np.float32)
            crop[:, ch["steer_pressed"]] = (np.abs(ang) > 8).astype(np.float32)
            crop[:, ch["standstill"]] = (v < 0.5).astype(np.float32)
            crop[:, ch["das_enabled"]] = (t_crop < das * CROP_LEN).astype(np.float32)
            if lead_valid > 0:
                crop[:, ch["lead_valid"]] = 1.0
                crop[:, ch["lead_d_m"]] = 30.0 + rng.normal(0, 2, CROP_LEN)
                crop[:, ch["lead_v_rel_mps"]] = rng.normal(0, 1, CROP_LEN)
                crop[:, ch["lead_speed_mps"]] = v + crop[:, ch["lead_v_rel_mps"]]
                crop[:, ch["lead_a_slope_mps2"]] = rng.normal(0, 0.3, CROP_LEN)
                crop[:, ch["thw_s"]] = crop[:, ch["lead_d_m"]] / np.maximum(v, 0.5)
                crop[:, ch["ttc_s"]] = np.where(crop[:, ch["lead_v_rel_mps"]] < -0.1, crop[:, ch["lead_d_m"]] / np.maximum(-crop[:, ch["lead_v_rel_mps"]], 0.1), np.nan)
            else:
                for c in ("lead_d_m", "lead_v_rel_mps", "lead_speed_mps", "lead_a_slope_mps2", "thw_s", "ttc_s"):
                    crop[:, ch[c]] = np.nan
            crop[:, ch["valid"]] = 1.0
            if gap:
                cut = int(rng.integers(100, CROP_LEN))
                crop[cut:, :] = np.nan
                crop[cut:, ch["valid"]] = 0.0
            E_parts.append(crop.astype(np.float16))
            if has_grid:
                g = np.zeros((GRID_LEN, len(GRID_CHANNELS)), dtype=np.float32)
                speeds = np.arange(GRID_LEN) * 0.5
                lo, hi = sorted((v0, v0 + dv))
                inside = (speeds >= lo) & (speeds <= hi)
                g[:, 0] = np.where(inside, amp * np.sign(dv), np.nan)
                g[:, 1] = np.where(inside, float(dv > 0), 0.0)
                g[:, 2] = np.where(inside, float(dv < 0), 0.0)
                g[:, 3] = np.where(inside, float(das > 0.5), 0.0)
                g[:, 4] = inside.astype(np.float32)
                g[:, 5] = np.where(inside, (speeds - v0) / max(1e-3, dv / dur), np.nan)
                G_parts.append(g.astype(np.float16))
                grid_count += 1
    E = np.stack(E_parts) if E_parts else np.zeros((0, CROP_LEN, len(CROP_CHANNELS)), np.float16)
    G = np.stack(G_parts) if G_parts else np.zeros((0, GRID_LEN, len(GRID_CHANNELS)), np.float16)
    ep = pd.DataFrame(erows, columns=list(EPISODE_COLS))
    ep["type_id"] = ep["type_id"].astype(np.int8)
    for c in ("anchor_row", "i0", "i1", "crop_row0", "w_idx", "grid_idx"):
        ep[c] = ep[c].astype(np.int32)
    ep["episode_id"] = ep["episode_id"].astype(np.int64)
    drives = pd.DataFrame(drows)
    win = pd.DataFrame(wrows)
    tmeta = win[win["tensor_idx"] >= 0][["window_id", "drive_id", "vehicle_id", "group_id", "w_idx", "tensor_idx"]].reset_index(drop=True)
    win = win.drop(columns=["tensor_idx"])
    wemb = np.stack(wemb_rows).astype(np.float16) if (wemb_dim > 0 and wemb_rows) else None
    balanced = {}
    for task in ("size", "bev_ice"):
        for split in ("test", "validation"):
            for split_col in ("split_platform", "split_device"):
                t = TASKS[task]
                sel = win[(win[split_col] == split) & win["eligible"] & win[t["eligible"]] & win[t["label"]].isin(t["classes"])]
                ids = sel["window_id"].to_numpy(object)
                balanced[(task, split, split_col)] = set(rng.choice(ids, size=len(ids) // 2, replace=False).tolist()) if len(ids) > 1 else set(ids)
    channels = dict(crop=list(CROP_CHANNELS), grid=list(GRID_CHANNELS), types=list(EPISODE_TYPES))
    return EpisodeData.from_arrays(E, G, ep, channels, drives, win, tmeta, wemb, ["synthetic"] if wemb is not None else [], balanced=balanced)


def write_synthetic(data: EpisodeData, root: Path | str) -> dict[str, Path]:
    """Write a synthetic EpisodeData in the cache layout (contracts (1) and (2) + the index tables + balanced lists)
    so that EpisodeData(root, wemb=...) exercises the disk loader. Returns the wemb mapping to pass to the loader."""
    root = Path(root)
    edir, idir, tdir, sdir = root / "work" / "episodes", root / "build" / "index", root / "work" / "tensors", root / "build" / "splits"
    for p in (edir, idir, tdir, sdir):
        p.mkdir(parents=True, exist_ok=True)
    np.save(edir / "E_f16.npy", np.asarray(data.E, dtype=np.float16))
    np.save(edir / "G_f16.npy", np.asarray(data.G, dtype=np.float16))
    data.ep.to_parquet(edir / "episodes.parquet", index=False)
    (edir / "channels.json").write_text(json.dumps(data.channels))
    (edir / "params.json").write_text(json.dumps({"synthetic": True}))
    data.drives_df.to_parquet(idir / "drives.parquet", index=False)
    win = data.win.drop(columns=["tensor_idx"])
    win.to_parquet(idir / "windows.parquet", index=False)
    tm = data.win[data.win["tensor_idx"] >= 0][["window_id", "drive_id", "w_idx", "tensor_idx"]].sort_values("tensor_idx")
    tm.to_parquet(tdir / "windows.parquet", index=False)
    wemb = {}
    if data.wemb is not None:
        tag = "synthetic"
        np.save(tdir / f"wemb_{tag}.npy", np.asarray(data.wemb, dtype=np.float16))
        (tdir / f"wemb_{tag}.json").write_text(json.dumps({"ckpt": tag, "dim": data.wemb_dim, "hook": "input of model.head.norm"}))
        wemb[tag] = tdir / f"wemb_{tag}.npy"
    if data._balanced:
        for (task, split, split_col), ids in data._balanced.items():
            pd.DataFrame({"window_id": sorted(ids)}).to_parquet(sdir / balanced_list_name(task, split, split_col), index=False)
    return wemb
