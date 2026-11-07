"""Window tensors, labels, splits and protocol masks for the experiments.

The tensor cache (work/tensors/X_f16.npy, (N, 600, 20) float16, NaN kept) is joined to build/index/windows.parquet by
window_id; protocol masks follow docs/PROTOCOLS.md (test_full / test_balanced / test_human_only, split_platform
headline, split_device secondary). Pre-processing = per-channel clipping of heavy tails, standardisation with train
statistics, NaN -> 0 (lead channels carry their own validity flag `lead_valid`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vtt import config as C

TASKS = {
    "size": dict(label="size_class", eligible="task_size_eligible", classes=tuple(C.SIZE_CLASSES)),
    "bev_ice": dict(label="powertrain", eligible="task_bev_ice_eligible", classes=tuple(C.MAIN_POWERTRAIN_TASK)),
}
PROTOCOLS = ("test_full", "test_balanced", "test_human_only")
SPLIT_COLS = ("split_platform", "split_device")
# heavy-tailed channels are clipped before standardisation (units of the channel)
CLIP = {
    "ttc_s": (0.0, 60.0), "thw_s": (0.0, 30.0), "lead_d_m": (0.0, 200.0), "lead_v_rel_mps": (-30.0, 30.0),
    "lead_speed_mps": (-5.0, 60.0), "lead_a_slope_mps2": (-6.0, 6.0), "steer_angle_deg": (-720.0, 720.0),
    "steer_rate_degps": (-720.0, 720.0), "a_mps2": (-8.0, 8.0), "lat_acc_mps2": (-8.0, 8.0), "curv_1pm": (-0.3, 0.3),
    "yaw_rate_radps": (-1.5, 1.5), "steer_torque_norm": (-5.0, 5.0), "v_mps": (-1.0, 60.0),
}


def balanced_list_name(task: str, split: str, split_col: str) -> str:
    return f"eval_balanced_{task}_{split}{'' if split_col == 'split_platform' else '_device'}.parquet"


class WindowData:
    """The tensor cache plus the windows index, aligned by tensor_idx."""

    def __init__(self, data_root: Path | str = C.DATA_ROOT, tensor_dir: Path | str | None = None, in_ram: bool = True):
        root = Path(data_root)
        tdir = Path(tensor_dir) if tensor_dir else root / "work" / "tensors"
        X = np.load(tdir / "X_f16.npy", mmap_mode="r")
        self.X = np.ascontiguousarray(X) if in_ram else X
        self.channels: list[str] = json.loads((tdir / "channels.json").read_text())
        meta = pd.read_parquet(tdir / "windows.parquet")
        win = pd.read_parquet(root / "build" / "index" / "windows.parquet")
        cols = [c for c in win.columns if c not in ("drive_id", "vehicle_id", "group_id", "w_idx")]
        self.win = meta.merge(win[cols], on="window_id", how="left", validate="one_to_one")
        self.win = self.win.sort_values("tensor_idx").reset_index(drop=True)
        if not (self.win["tensor_idx"].to_numpy() == np.arange(len(self.win))).all():
            raise ValueError("tensor_idx is not 0..N-1 after the join")
        if len(self.win) != self.X.shape[0]:
            raise ValueError(f"{len(self.win)} index rows for {self.X.shape[0]} tensors")
        self.splits_dir = root / "build" / "splits"
        lo = np.array([CLIP.get(c, (-np.inf, np.inf))[0] for c in self.channels], dtype=np.float32)
        hi = np.array([CLIP.get(c, (-np.inf, np.inf))[1] for c in self.channels], dtype=np.float32)
        self.clip_lo, self.clip_hi = lo, hi

    # ------------------------------------------------------------------ selections
    def balanced_ids(self, task: str, split: str = "test", split_col: str = "split_platform") -> set[str]:
        return set(pd.read_parquet(self.splits_dir / balanced_list_name(task, split, split_col))["window_id"].astype(str))

    def mask(self, task: str, split_col: str, split: str, protocol: str = "test_full") -> np.ndarray:
        t = TASKS[task]
        w = self.win
        m = (w[split_col].to_numpy(object) == split)
        m &= w["eligible"].fillna(False).to_numpy(bool)
        m &= w[t["eligible"]].fillna(False).to_numpy(bool)
        m &= w[t["label"]].isin(t["classes"]).to_numpy()
        if protocol == "test_balanced":
            m &= w["window_id"].isin(self.balanced_ids(task, split, split_col)).to_numpy()
        elif protocol == "test_human_only":
            m &= w["human_only_strict"].fillna(False).to_numpy(bool)
        elif protocol != "test_full":
            raise ValueError(protocol)
        return m

    def indices(self, task: str, split_col: str, split: str, protocol: str = "test_full") -> np.ndarray:
        return np.flatnonzero(self.mask(task, split_col, split, protocol))

    def labels(self, task: str, idx: np.ndarray) -> np.ndarray:
        t = TASKS[task]
        cls = {c: i for i, c in enumerate(t["classes"])}
        lab = self.win[t["label"]].to_numpy(object)[idx]
        return np.array([cls[x] for x in lab], dtype=np.int64)

    def class_sampling_weights(self, y: np.ndarray, n_classes: int, power: float = C.CLASS_WEIGHT_POWER,
                               cap: float = C.CLASS_WEIGHT_CAP) -> np.ndarray:
        """Per-window sampling weight (N / (K n_k))^power, capped: the train_uniform rule of the dataset."""
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        w_class = np.minimum((len(y) / (n_classes * np.maximum(counts, 1))) ** power, cap)
        return w_class[y]

    # ------------------------------------------------------------------ tensors
    def norm_stats(self, idx: np.ndarray, max_windows: int = 20000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Per-channel mean / std over a random subset of the given windows, after clipping; NaN ignored."""
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(idx, size=min(max_windows, len(idx)), replace=False))
        x = np.asarray(self.X[sel], dtype=np.float32)
        x = np.clip(x, self.clip_lo, self.clip_hi)
        mean = np.nanmean(x, axis=(0, 1))
        std = np.nanstd(x, axis=(0, 1))
        mean[~np.isfinite(mean)] = 0.0
        std[~np.isfinite(std) | (std < 1e-6)] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    def batch(self, idx: np.ndarray) -> torch.Tensor:
        """Raw float16 windows (B, 600, C) as a CPU tensor (NaN kept)."""
        order = np.argsort(idx, kind="stable")
        x = self.X[idx[order]]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return torch.from_numpy(np.ascontiguousarray(x[inv]))


class Preprocess(torch.nn.Module):
    """clip -> standardise -> NaN to 0, on the device."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean))
        self.register_buffer("std", torch.as_tensor(std))
        self.register_buffer("lo", torch.as_tensor(np.nan_to_num(lo, neginf=-1e30, posinf=1e30)))
        self.register_buffer("hi", torch.as_tensor(np.nan_to_num(hi, neginf=-1e30, posinf=1e30)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = torch.maximum(torch.minimum(x, self.hi), self.lo)
        x = (x - self.mean) / self.std
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def iterate_batches(idx: np.ndarray, batch_size: int, rng: np.random.Generator | None = None,
                    weights: np.ndarray | None = None, n_draws: int | None = None):
    """Yield index arrays. With `rng`: weighted sampling with replacement (n_draws, default len(idx)); else sequential."""
    if rng is None:
        for i in range(0, len(idx), batch_size):
            yield idx[i:i + batch_size]
        return
    n = n_draws or len(idx)
    p = None if weights is None else weights / weights.sum()
    draw = rng.choice(idx, size=n, replace=True, p=p)
    for i in range(0, n, batch_size):
        yield draw[i:i + batch_size]
