"""Window tensors, labels, splits, protocol masks, auxiliary targets, sampling weights and augmentation.

The tensor cache (work/tensors/X_f16.npy, (N, 600, 20) float16, NaN kept) is joined to build/index/windows.parquet by
window_id; protocol masks follow docs/PROTOCOLS.md (test_full / test_balanced / test_human_only, split_platform
headline, split_device secondary). The drives index (build/index/drives.parquet) is kept as `WindowData.drives` and
its curb mass / wheelbase are merged onto the windows for the auxiliary regression targets; `extra_mask` selects the
windows with a known label outside the task classes (HEV / PHEV for bev_ice) that may train without a class label.
Pre-processing =
per-channel clipping of heavy tails, standardisation with train statistics, NaN -> 0 (lead channels carry their own
validity flag `lead_valid`). `Augment` works in the standardised domain, after `Preprocess`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
# auxiliary targets: plausible spec ranges (outside -> masked), powertrain codes, per-window hours
PLAUSIBLE = {"curb_mass_kg": (800.0, 3500.0), "wheelbase_mm": (2200.0, 4000.0)}
PT4_CODES = {p: i for i, p in enumerate(C.POWERTRAIN4)}            # ICE 0, HEV 1, PHEV 2, BEV 3; -1 unknown
WINDOW_H = C.WINDOW_SAMPLES / C.NOMINAL_HZ / 3600.0                # hours per window (60 s)
DRIVE_EXTRA_COLUMNS = ("curb_mass_kg", "wheelbase_mm", "body")      # merged from the drives index when missing
# augmentation: the lateral channels whose sign flips together under a mirrored road, the binary flags that get no
# noise, and the channel groups that can be dropped as a whole
FLIP_CHANNELS = ("steer_angle_deg", "steer_rate_degps", "steer_torque_norm", "yaw_rate_radps", "curv_1pm", "lat_acc_mps2")
BINARY_CHANNELS = ("gas_pressed", "brake_pressed", "steer_pressed", "standstill", "das_enabled", "lead_valid")
CHANNEL_GROUPS = {"lead": ("lead_valid", "lead_d_m", "lead_v_rel_mps", "lead_speed_mps", "lead_a_slope_mps2", "thw_s", "ttc_s"),
                  "torque": ("steer_torque_norm",), "pedals": ("gas_pressed", "brake_pressed")}


def balanced_list_name(task: str, split: str, split_col: str) -> str:
    return f"eval_balanced_{task}_{split}{'' if split_col == 'split_platform' else '_device'}.parquet"


class WindowData:
    """The tensor cache plus the windows index, aligned by tensor_idx; the drives index as `drives`."""

    def __init__(self, data_root: Path | str = C.DATA_ROOT, tensor_dir: Path | str | None = None, in_ram: bool = True):
        root = Path(data_root)
        tdir = Path(tensor_dir) if tensor_dir else root / "work" / "tensors"
        X = np.load(tdir / "X_f16.npy", mmap_mode="r")
        self.X = np.array(X, dtype=np.float16, copy=True) if in_ram else X     # a real copy: ascontiguousarray returns the memmap itself
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
        self.drives = pd.read_parquet(root / "build" / "index" / "drives.parquet")
        extra = [c for c in DRIVE_EXTRA_COLUMNS if c not in self.win.columns and c in self.drives.columns]
        if extra:
            self.win = self.win.merge(self.drives[["drive_id", *extra]], on="drive_id", how="left", validate="many_to_one")
        self.model_codes: dict[str, int] = {m: i for i, m in enumerate(sorted(self.win["model"].dropna().astype(str).unique()))}
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

    def extra_mask(self, task: str, split_col: str, split: str) -> np.ndarray:
        """Eligible windows of the split whose task label is known but outside the task classes: the HEV / PHEV windows
        of the bev_ice task (every known size label is a size class, so the size task has none). With
        run(include_hybrids=True) they join the training set without a class label (y = -1, ignored by the cls / CORAL
        losses) so the pt4 / mass / fp heads train on them (plan stage S4)."""
        t = TASKS[task]
        w = self.win
        lab = w[t["label"]]
        m = (w[split_col].to_numpy(object) == split)
        m &= w["eligible"].fillna(False).to_numpy(bool)
        m &= lab.notna().to_numpy() & ~lab.isin(t["classes"]).to_numpy()
        return m

    def extra_indices(self, task: str, split_col: str, split: str) -> np.ndarray:
        return np.flatnonzero(self.extra_mask(task, split_col, split))

    def labels(self, task: str, idx: np.ndarray) -> np.ndarray:
        t = TASKS[task]
        cls = {c: i for i, c in enumerate(t["classes"])}
        lab = self.win[t["label"]].to_numpy(object)[idx]
        return np.array([cls[x] for x in lab], dtype=np.int64)

    def class_sampling_weights(self, y: np.ndarray, n_classes: int, power: float = C.CLASS_WEIGHT_POWER,
                               cap: float = C.CLASS_WEIGHT_CAP) -> np.ndarray:
        """Per-window sampling weight (N / (K n_k))^power, capped: the train_uniform rule of the dataset. Windows without
        a class label (y = -1, the hybrids of `extra_mask`) get the mean weight of the labelled windows, so they are
        drawn in proportion to their number."""
        y = np.asarray(y, dtype=np.int64)
        lab = y >= 0
        counts = np.bincount(y[lab], minlength=n_classes).astype(np.float64)
        w_class = np.minimum((int(lab.sum()) / (n_classes * np.maximum(counts, 1))) ** power, cap)
        w = np.empty(len(y), dtype=np.float64)
        w[lab] = w_class[y[lab]]
        w[~lab] = float(w[lab].mean()) if lab.any() else 1.0
        return w

    def sampling_weights(self, idx: np.ndarray, y: np.ndarray, n_classes: int, class_power: float = C.CLASS_WEIGHT_POWER,
                         unit: str | None = "model", unit_power: float = 0.5, cap: float = C.CLASS_WEIGHT_CAP,
                         das_beta: float = 0.0) -> np.ndarray:
        """Class-balanced sampling weights, optionally re-balanced over a unit (car model) inside every class.

        unit=None reproduces `class_sampling_weights`. With a unit column: inside class c the share of unit u becomes
        proportional to hours_u^unit_power (hours = windows of u in c x 60 s) instead of hours_u; the per-window ratio
        to the plain class weight is capped at `cap` (strictly) and the class totals are kept up to the mass the cap
        removes, so the class balance is unchanged.
        das_beta > 0 multiplies by (1 - beta) + beta (1 - das_any_frac): windows with automation activity count less."""
        w = self.class_sampling_weights(y, n_classes, class_power, cap).astype(np.float64)
        if unit is not None:
            units = self.win[unit].to_numpy(object)[idx]
            units = np.array([("NA" if pd.isna(u) else str(u)) for u in units], dtype=object)
            for c in range(n_classes):
                sel = np.flatnonzero(y == c)
                if not len(sel):
                    continue
                _, inv, cnt = np.unique(units[sel], return_inverse=True, return_counts=True)
                hours = cnt * WINDOW_H
                target = hours ** unit_power
                target = target / target.sum()
                actual = cnt / cnt.sum()
                base_c = w[sel]
                w_c = base_c * np.minimum(target / actual, cap)[inv]
                for _ in range(3):                                   # keep the class total, then enforce the cap strictly
                    w_c = w_c * (base_c.sum() / w_c.sum())
                    w_c = np.minimum(w_c, cap * base_c)
                w[sel] = w_c
        if das_beta:
            das = np.nan_to_num(self.win["das_any_frac"].to_numpy(np.float64)[idx], nan=0.0)
            w = w * ((1.0 - das_beta) + das_beta * (1.0 - np.clip(das, 0.0, 1.0)))
        return w

    # ------------------------------------------------------------------ auxiliary targets
    def aux_targets(self, idx: np.ndarray, stats: dict | None = None) -> dict:
        """Auxiliary targets aligned with `idx`: standardised log footprint (`log_fp`, mask `fp_mask`), standardised log
        curb mass (`log_mass`, mask `mass_mask`: 800-3500 kg; the footprint mask also needs a 2200-4000 mm wheelbase),
        four-class powertrain code `pt4` (-1 unknown), car-model code `model_id` (-1 unknown; codes are fixed over the
        whole windows table), `das_any_frac` (NaN -> 0) and the standardisation `stats` (computed on `idx` when None,
        so pass the training statistics for other splits)."""
        w = self.win
        fp = w["footprint_m2"].to_numpy(np.float64)[idx]
        wb = w["wheelbase_mm"].to_numpy(np.float64)[idx] if "wheelbase_mm" in w else np.full(len(idx), np.nan)
        mass = w["curb_mass_kg"].to_numpy(np.float64)[idx] if "curb_mass_kg" in w else np.full(len(idx), np.nan)
        wlo, whi = PLAUSIBLE["wheelbase_mm"]
        mlo, mhi = PLAUSIBLE["curb_mass_kg"]
        with np.errstate(invalid="ignore"):
            fp_mask = np.isfinite(fp) & (fp > 0) & np.isfinite(wb) & (wb >= wlo) & (wb <= whi)
            mass_mask = np.isfinite(mass) & (mass >= mlo) & (mass <= mhi)
        log_fp = np.log(np.where(fp_mask, fp, 1.0))
        log_mass = np.log(np.where(mass_mask, mass, 1.0))
        if stats is None:
            stats = {}
            for key, val, m in (("fp", log_fp, fp_mask), ("mass", log_mass, mass_mask)):
                mu = float(val[m].mean()) if m.any() else 0.0
                sd = float(val[m].std()) if m.sum() > 1 else 1.0
                stats[f"{key}_mu"], stats[f"{key}_sd"] = mu, (sd if sd > 1e-6 else 1.0)
        pt = w["powertrain"].to_numpy(object)[idx]
        models = w["model"].to_numpy(object)[idx]
        return dict(
            log_fp=np.where(fp_mask, (log_fp - stats["fp_mu"]) / stats["fp_sd"], 0.0).astype(np.float32),
            fp_mask=fp_mask,
            log_mass=np.where(mass_mask, (log_mass - stats["mass_mu"]) / stats["mass_sd"], 0.0).astype(np.float32),
            mass_mask=mass_mask,
            pt4=np.array([PT4_CODES.get(p, -1) for p in pt], dtype=np.int64),
            model_id=np.array([-1 if pd.isna(m) else self.model_codes.get(str(m), -1) for m in models], dtype=np.int64),
            das_any_frac=np.nan_to_num(w["das_any_frac"].to_numpy(np.float32)[idx], nan=0.0),
            stats=dict(stats),
        )

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


class Augment(nn.Module):
    """Physically safe training-time augmentation of standardised windows (B, L, C); identity in eval mode.

    flip: with p_flip the six lateral channels (FLIP_CHANNELS) change sign together (a mirrored road; the channels are
    close to zero-mean, so the standardised value is negated and missing values stay 0);
    mask: with p_mask one span of mask_s seconds is set to 0 (= missing) on every channel and `lead_valid` to its
    standardised 0 there (no lead known);
    chdrop: with p_chdrop one channel group (lead / steer torque / pedals, CHANNEL_GROUPS) is set to missing for the
    whole window (lead_valid -> 0 when the lead group is dropped);
    noise: Gaussian noise of standard deviation `noise` on the continuous channels (none on the binary flags);
    cutmix (p_cutmix, needs y and model_id): a span of the window is replaced by the same span of another window of the
    same class and a different car model from the batch (labels unchanged; windows without a class label, y = -1, are
    neither cut nor used as a partner).
    No speed or amplitude scaling and no time reversal (both change the vehicle signature)."""

    def __init__(self, channels: list[str], mean: np.ndarray | None = None, std: np.ndarray | None = None, p_flip: float = 0.5,
                 p_mask: float = 0.5, mask_s: tuple[float, float] = (3.0, 15.0), p_chdrop: float = 0.1, noise: float = 0.02,
                 p_cutmix: float = 0.0, hz: int = C.NOMINAL_HZ):
        super().__init__()
        self.channels = list(channels)
        self.p_flip, self.p_mask, self.p_chdrop, self.noise, self.p_cutmix = float(p_flip), float(p_mask), float(p_chdrop), float(noise), float(p_cutmix)
        self.mask_len = (max(1, int(round(mask_s[0] * hz))), max(1, int(round(mask_s[1] * hz))))
        self.flip_idx = [self.channels.index(c) for c in FLIP_CHANNELS if c in self.channels]
        groups = [[self.channels.index(c) for c in grp if c in self.channels] for grp in CHANNEL_GROUPS.values()]
        self.groups = [g for g in groups if g]
        self.lead_valid = self.channels.index("lead_valid") if "lead_valid" in self.channels else None
        C_ = len(self.channels)
        flip = torch.zeros(C_, dtype=torch.bool)
        flip[self.flip_idx] = True
        cont = torch.tensor([c not in BINARY_CHANNELS for c in self.channels], dtype=torch.float32)
        gtab = torch.zeros(max(1, len(self.groups)), C_, dtype=torch.bool)
        for gi, g in enumerate(self.groups):
            gtab[gi, g] = True
        lv0 = 0.0
        if self.lead_valid is not None and mean is not None and std is not None:
            lv0 = float((0.0 - float(mean[self.lead_valid])) / float(std[self.lead_valid]))
        self.register_buffer("flip_mask", flip)
        self.register_buffer("cont_mask", cont)
        self.register_buffer("group_table", gtab)
        self.register_buffer("lead_valid_zero", torch.tensor(lv0, dtype=torch.float32))

    def _spans(self, B: int, L: int, active: torch.Tensor) -> torch.Tensor:
        """(B, L) bool: one random span of mask_len samples per active row."""
        lo, hi = self.mask_len
        hi = min(hi, L)
        lo = min(lo, hi)
        length = torch.randint(lo, hi + 1, (B,), device=active.device)
        start = (torch.rand(B, device=active.device) * (L - length + 1).float()).long()
        t = torch.arange(L, device=active.device)[None, :]
        return (t >= start[:, None]) & (t < (start + length)[:, None]) & active[:, None]

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, model_id: torch.Tensor | None = None) -> torch.Tensor:
        if not self.training:
            return x
        B, L, C_ = x.shape
        dev = x.device
        if self.p_cutmix > 0 and y is not None:
            x0 = x
            x = x.clone()
            active = torch.rand(B, device=dev) < self.p_cutmix
            span = self._spans(B, L, active)
            same = (y[:, None] == y[None, :]) & (y[:, None] >= 0)
            if model_id is not None:
                same &= model_id[:, None] != model_id[None, :]
            same.fill_diagonal_(False)
            for i in torch.nonzero(active & same.any(1)).flatten().tolist():
                j = int(torch.multinomial(same[i].float(), 1))
                x[i] = torch.where(span[i][:, None], x0[j], x0[i])
        if self.p_flip > 0 and self.flip_idx:
            active = torch.rand(B, device=dev) < self.p_flip
            m = active[:, None, None] & self.flip_mask[None, None, :]
            x = torch.where(m, -x, x)
        if self.p_mask > 0:
            active = torch.rand(B, device=dev) < self.p_mask
            span = self._spans(B, L, active)
            x = torch.where(span[:, :, None], torch.zeros((), device=dev, dtype=x.dtype), x)
            if self.lead_valid is not None:
                lv = x[:, :, self.lead_valid]
                x = x.clone()
                x[:, :, self.lead_valid] = torch.where(span, self.lead_valid_zero.to(x.dtype), lv)
        if self.p_chdrop > 0 and self.groups:
            active = torch.rand(B, device=dev) < self.p_chdrop
            g = torch.randint(len(self.groups), (B,), device=dev)
            chmask = self.group_table[g] & active[:, None]                       # (B, C)
            x = torch.where(chmask[:, None, :], torch.zeros((), device=dev, dtype=x.dtype), x)
            if self.lead_valid is not None:
                lvdrop = chmask[:, self.lead_valid]
                x = x.clone()
                x[:, :, self.lead_valid] = torch.where(lvdrop[:, None], self.lead_valid_zero.to(x.dtype), x[:, :, self.lead_valid])
        if self.noise > 0:
            x = x + torch.randn_like(x) * self.noise * self.cont_mask
        return x


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
