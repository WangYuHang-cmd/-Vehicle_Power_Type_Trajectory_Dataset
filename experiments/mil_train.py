#!/usr/bin/env python3
"""Train and evaluate the episode-centric multiple-instance model (DriveMIL) on drive bags, both tasks jointly.

Data: the episode cache (experiments/mil_data.EpisodeData), drive labels from the index tables and, optionally, the 60-s
window embeddings of saved window checkpoints (work/tensors/wemb_<tag>.npy). Embeddings are discovered with a provenance
rule: same split scheme as the run (sidecar split_col, else the tag prefix), no smoke `_limit` files and, when the inner
validation is car-model- or device-disjoint (val_mode internal with the model folds or a group carve), only embeddings
of fold-k checkpoints (tags fold{k}__..., contract (5) recipe checkpoints embedded with window_embed.py --ckpt), because
the published checkpoints were trained on every train drive including the held-out models: their features would leak
those models into the selection metric. Without fold-k embeddings the window instances are switched off (logged);
--wemb-allow-full admits the full-split embeddings and the JSON then says cfg.wemb_inner_val_clean = false (also in
fold_info). Training: batches of drives (no car model above a share of the batch), stratified bags of K episodes + K_w
window instances per drive, loss = CE(size) + CE(bev) (label smoothing, masked by task eligibility) + 0.3 instance CEs +
0.5 Huber(log footprint) + 0.25 Huber(log mass) + 0.25 CE(powertrain-4) + 0.25 BCE(ordinal); AdamW, warm-up + cosine,
gradient clipping, bf16 autocast for the encoders; augmentation = anchor jitter, lateral sign flip, instance / lead
dropout, Gaussian noise. Early stopping on the mean over the two tasks of the inner-validation drive macro-F1, of the
direct bag prediction (select_on inner_bag_f1, cheap, default) or of the window-mean drive_full protocol
(inner_drive_f1); a separate best checkpoint per task; fixed_epochs trains that many epochs and keeps the last state.
Evaluation: window-level protocols (instances whose anchor falls inside the window + the window's embedding). The
contract keys drive_level / unit_level / vehicle_level hold the window-mean protocols of docs/PROTOCOLS.md (mean of the
per-window probability vectors per drive or vehicle -> argmax, ties -> earlier class, drives without eligible windows
counted as n_units_without_windows), computed with experiments.train.unit_level so they compare one-to-one with the
window models; the model's own drive prediction (direct bag over a deterministic K_eval subsample of all episodes of the
drive) is listed separately under drive_level_bag / unit_level_bag / vehicle_level_bag and
inner_val_metrics.drive_full_bag. Outputs: final runs (fold None, test evaluated) write
<data-root>/experiments/results/<tag>.json with tag <split_col>__<task>__<model>__s<seed> (model "episode_mil" or
"episode_mil_meanlogit" plus a --variant suffix) + <tag>_probs.npz (window_id, probs, y) + <tag>_drive_probs.npz
(drive_id, vehicle_id, probs, y, n_episodes, n_human_episodes); fold / selection runs (--fold or --no-test) write
<data-root>/experiments/recipe/results/fold{k}__{task}__{model}__{recipe_id}__s{seed}.json + <tag>_oof.npz (the
inner-validation windows) and never a JSON without protocols into experiments/results. Checkpoints go to
experiments/ckpt_mil (final) or experiments/recipe/ckpt_mil (fold), train.py's dict format plus kind = "episode_mil", so
regen_probs / window_embed never meet them. --synthetic-smoke trains 2 epochs on random data on the CPU and writes to
<data-root>/tmp/mil_smoke. Prints counts and metrics only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402
from experiments.data import PROTOCOLS, TASKS, Preprocess  # noqa: E402
from experiments.train import EXP_ROOT, RECIPE_ROOT, lr_lambda, macro_f1, metrics, unit_level  # noqa: E402
from experiments import mil_data as MD  # noqa: E402
from experiments.mil_data import INNER_VAL_MODES, EpisodeData, iterate_drive_batches, make_synthetic, resolve_inner_val, to_torch  # noqa: E402
from experiments.mil_models import DriveMIL, POOLS, n_params  # noqa: E402

MODEL_NAMES = {"gated_attn": "episode_mil", "mean_logit": "episode_mil_meanlogit"}
TASK_HEAD = {"size": "size", "bev_ice": "bev"}
LOSS_W = dict(inst=0.3, fp=0.5, mass=0.25, pt4=0.25, ordinal=0.25)
AUG = dict(p_flip=0.5, inst_drop=0.2, lead_drop=0.1, noise=0.02)
DEFAULTS = dict(epochs=40, patience=8, batch_drives=24, K=64, K_w=8, K_eval=128, K_w_eval=16, lr=1e-3, wd=0.05, d=128, dropout=0.1,
                label_smoothing=0.05, warmup_frac=0.03, eval_bags=48, max_model_share=0.10)
SMOKE = dict(n_drives=60, epochs=2, d=32, K=16, K_w=2, K_eval=24, K_w_eval=4, batch_drives=12, eval_bags=64)
TYPE_NAMES = tuple(MD.EPISODE_TYPES) + ("WINDOW",)
SELECT_ON = ("inner_bag_f1", "inner_drive_f1")      # early-stopping metric: direct bag prediction / window-mean drive_full
CKPT_DIRNAME = "ckpt_mil"                            # MIL checkpoints never share a directory with the window models
CKPT_KIND = "episode_mil"
DEFAULT_RECIPE_ID = "mil"
DECISION = "softmax"


# ----------------------------------------------------------------------------- loss
def masked_mean(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.float()
    return (v.float() * m).sum() / m.sum().clamp_min(1.0)


def mil_loss(out: dict, batch: dict, tstats: dict, aux: bool = True, inst_w: float = LOSS_W["inst"], label_smoothing: float = 0.05,
             inst_keep: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """Bag + instance + auxiliary losses (float32). Every term is a masked mean: ineligible bags (and their instances)
    contribute nothing, so their outputs receive zero gradient. Returns (total, parts as detached floats)."""
    es, eb = batch["elig_size"].bool(), batch["elig_bev"].bool()
    ys, yb = batch["y_size"].clamp_min(0), batch["y_bev"].clamp_min(0)
    bi = out["inst_bag_idx"]
    keep = inst_keep.bool() if inst_keep is not None else torch.ones_like(bi, dtype=torch.bool)
    ls = label_smoothing
    parts = {
        "ce_size": masked_mean(F.cross_entropy(out["size"].float(), ys, reduction="none", label_smoothing=ls), es),
        "ce_bev": masked_mean(F.cross_entropy(out["bev"].float(), yb, reduction="none", label_smoothing=ls), eb),
        "ce_inst_size": masked_mean(F.cross_entropy(out["inst_size"].float(), ys[bi], reduction="none", label_smoothing=ls), es[bi] & keep),
        "ce_inst_bev": masked_mean(F.cross_entropy(out["inst_bev"].float(), yb[bi], reduction="none", label_smoothing=ls), eb[bi] & keep),
    }
    total = parts["ce_size"] + parts["ce_bev"] + inst_w * (parts["ce_inst_size"] + parts["ce_inst_bev"])
    if aux:
        fp_t = (batch["log_fp"].float() - tstats["log_fp_mu"]) / tstats["log_fp_sd"]
        mass_t = (batch["log_mass"].float() - tstats["log_mass_mu"]) / tstats["log_mass_sd"]
        yp = batch["y_pt4"]
        ord_t = torch.stack([(batch["y_size"] > 0), (batch["y_size"] > 1)], dim=-1).float()
        parts["huber_fp"] = masked_mean(F.huber_loss(out["footprint"].float(), fp_t, reduction="none"), batch["fp_mask"].bool())
        parts["huber_mass"] = masked_mean(F.huber_loss(out["logmass"].float(), mass_t, reduction="none"), batch["mass_mask"].bool())
        parts["ce_pt4"] = masked_mean(F.cross_entropy(out["pt4"].float(), yp.clamp_min(0), reduction="none"), yp >= 0)
        parts["bce_ordinal"] = masked_mean(F.binary_cross_entropy_with_logits(out["ordinal"].float(), ord_t, reduction="none").mean(-1), es)
        total = total + LOSS_W["fp"] * parts["huber_fp"] + LOSS_W["mass"] * parts["huber_mass"] + LOSS_W["pt4"] * parts["ce_pt4"] \
            + LOSS_W["ordinal"] * parts["bce_ordinal"]
    parts = {k: float(v.detach()) for k, v in parts.items()}
    parts["total"] = float(total.detach())
    return total, parts


# ----------------------------------------------------------------------------- pre-processing + augmentation
class BatchPrep:
    """clip -> standardise -> NaN to 0 for crops and grids; training augmentation (lateral sign flip before the
    standardisation, Gaussian noise on the continuous channels after it, instance and lead dropout on the bag mask)."""

    def __init__(self, data: EpisodeData, stats: dict, device: torch.device, aug: dict = AUG, seed: int = 0):
        self.crop = Preprocess(stats["crop_mean"], stats["crop_std"], data.crop_lo, data.crop_hi).to(device)
        self.grid = Preprocess(stats["grid_mean"], stats["grid_std"], data.grid_lo, data.grid_hi).to(device)
        self.lateral = torch.as_tensor(data.lateral_idx, device=device)
        self.binary = torch.as_tensor(data.binary_idx, device=device)
        self.lead_types = torch.as_tensor(np.array(MD.LEAD_TYPES), device=device)
        self.aug = dict(aug)
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def _rand(self, n: int) -> torch.Tensor:
        return torch.rand(n, generator=self.gen, device=self.device)

    def __call__(self, tb: dict, train: bool = False) -> dict:
        x = tb["x"].to(torch.float32, copy=True)
        M = x.shape[0]
        if train and M and self.aug["p_flip"] > 0:
            sign = torch.where(self._rand(M) < self.aug["p_flip"], -1.0, 1.0).to(x.dtype)
            x[:, :, self.lateral] = x[:, :, self.lateral] * sign[:, None, None]
        x = self.crop(x)
        if train and M and self.aug["noise"] > 0:
            n = torch.randn(x.shape, generator=self.gen, device=self.device) * self.aug["noise"]
            n[:, :, self.binary] = 0.0
            x = x + n
        tb["x"] = x
        tb["grid"] = self.grid(tb["grid"].float()) if tb["grid"].shape[0] else tb["grid"].float()
        if train and (self.aug["inst_drop"] > 0 or self.aug["lead_drop"] > 0):
            mask0 = tb["mask"]
            mask = mask0.clone()
            if M:
                drop = self._rand(M) < self.aug["inst_drop"]
                is_lead = torch.isin(tb["type"], self.lead_types)
                drop |= is_lead & (self._rand(M) < self.aug["lead_drop"])
                mask[tb["bag_idx"][drop], tb["pos"][drop]] = False
            Mw = tb["wemb_bag_idx"].shape[0]
            if Mw:
                dw = self._rand(Mw) < self.aug["inst_drop"]
                mask[tb["wemb_bag_idx"][dw], tb["wemb_pos"][dw]] = False
            emptied = ~mask.any(1) & mask0.any(1)                 # never drop a whole bag: restore its instances
            mask[emptied] = mask0[emptied]
            tb["mask"] = mask
        return tb


# ----------------------------------------------------------------------------- prediction
@torch.no_grad()
def predict_bags(model: DriveMIL, data: EpisodeData, bags: list, drive_ids: list, prep: BatchPrep, device: torch.device, amp: bool,
                 eval_bags: int = 128) -> dict:
    """Probabilities per bag for both heads + instance-level predictions (bags with zero instances get uniform probs)."""
    model.eval()
    n = len(bags)
    k_size, k_bev = model.n_out["size"], model.n_out["bev"]
    p_size = np.full((n, k_size), 1.0 / k_size, dtype=np.float32)
    p_bev = np.full((n, k_bev), 1.0 / k_bev, dtype=np.float32)
    n_inst = np.zeros(n, dtype=np.int64)
    inst_type, inst_size, inst_bev, inst_bag = [], [], [], []
    for i in range(0, n, eval_bags):
        sub = bags[i:i + eval_bags]
        tb = prep(to_torch(data.collate_bags(sub, drive_ids[i:i + eval_bags], train=False), device), train=False)
        out = model(tb, amp=amp)
        has = tb["mask"].any(1).cpu().numpy()
        ps = torch.softmax(out["size"].float(), -1).cpu().numpy()
        pb = torch.softmax(out["bev"].float(), -1).cpu().numpy()
        p_size[i:i + len(sub)][has] = ps[has]
        p_bev[i:i + len(sub)][has] = pb[has]
        n_inst[i:i + len(sub)] = tb["mask"].sum(1).cpu().numpy()
        inst_type.append(out["inst_type"].cpu().numpy())
        inst_size.append(out["inst_size"].float().argmax(-1).cpu().numpy())
        inst_bev.append(out["inst_bev"].float().argmax(-1).cpu().numpy())
        inst_bag.append(out["inst_bag_idx"].cpu().numpy() + i)
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, dtype=np.int64)   # noqa: E731
    return dict(size=p_size, bev_ice=p_bev, n_inst=n_inst, inst_type=cat(inst_type), inst_size=cat(inst_size), inst_bev=cat(inst_bev),
                inst_bag=cat(inst_bag))


def eval_drives(model: DriveMIL, data: EpisodeData, drive_ids: list, task: str, prep: BatchPrep, device: torch.device, amp: bool, cfg: dict,
                seed: int, use_windows: bool, human_only: bool = False) -> dict:
    """Direct bag prediction per drive (deterministic bag subsample of K_eval episodes + K_w_eval windows)."""
    rng = np.random.default_rng(seed + 7919)
    bags = [data.sample_bag(d, cfg["K_eval"], cfg["K_w_eval"] if use_windows else 0, rng, human_only=human_only) for d in drive_ids]
    n_inst = np.array([len(a) + len(b) for a, b in bags], dtype=np.int64)
    keep = np.flatnonzero(n_inst > 0)
    ids = [drive_ids[i] for i in keep]
    bags = [bags[i] for i in keep]
    pred = predict_bags(model, data, bags, ids, prep, device, amp, cfg["eval_bags"]) if ids else None
    tg = data.tgt.loc[ids]
    y = tg["y_size" if task == "size" else "y_bev"].to_numpy(np.int64)
    probs = pred[task] if pred is not None else np.zeros((0, len(TASKS[task]["classes"])), np.float32)
    return dict(drive_ids=np.asarray(ids, dtype=object), probs=probs, y=y, n_inst=n_inst[keep], vehicles=tg["vehicle_id"].to_numpy(object),
                models=tg["model"].astype(str).to_numpy(object), n_without_instances=int(len(drive_ids) - len(ids)),
                n_episodes=np.array([len(b[0]) for b in bags], dtype=np.int64),
                n_human_episodes=np.array([int(data.human_strict[b[0]].sum()) for b in bags], dtype=np.int64), pred=pred)


def eval_windows(model: DriveMIL, data: EpisodeData, rows: np.ndarray, task: str, prep: BatchPrep, device: torch.device, amp: bool, cfg: dict,
                 use_windows: bool) -> dict:
    """Window-level prediction: the episodes anchored inside the window + the window's own embedding."""
    bags = [data.bag_for_window(int(r), use_windows=use_windows) for r in rows]
    ids = list(data.win_drive[rows])
    pred = predict_bags(model, data, bags, ids, prep, device, amp, cfg["eval_bags"]) if len(rows) else None
    probs = pred[task] if pred is not None else np.zeros((0, len(TASKS[task]["classes"])), np.float32)
    n_inst = pred["n_inst"] if pred is not None else np.zeros(0, np.int64)
    return dict(rows=rows, probs=probs, y=data.window_labels(task, rows), drive_ids=np.asarray(ids, dtype=object), n_inst=n_inst,
                n_without_instances=int((n_inst == 0).sum()))


def per_type_instance_f1(pred: dict | None, y_bag: np.ndarray, task: str, k: int) -> tuple[dict, dict]:
    """Instance-type counts and the instance-level macro-F1 per type (instance head vs the bag label)."""
    if pred is None or len(pred["inst_type"]) == 0:
        return {t: 0 for t in TYPE_NAMES}, {t: None for t in TYPE_NAMES}
    it = pred["inst_type"]
    ip = pred["inst_size"] if task == "size" else pred["inst_bev"]
    yb = y_bag[pred["inst_bag"]]
    counts, f1s = {}, {}
    for t, name in enumerate(TYPE_NAMES):
        sel = (it == t) & (yb >= 0)
        counts[name] = int((it == t).sum())
        f1s[name] = macro_f1(yb[sel], ip[sel], k) if sel.any() else None
    return counts, f1s


# ----------------------------------------------------------------------------- unit-level protocols (contract (4)) and the bag variants
def _empty_metrics(classes: tuple) -> dict:
    return dict(n=0, macro_f1=float("nan"), balanced_acc=float("nan"), accuracy=float("nan"), per_class_recall={c: None for c in classes},
                support={c: 0 for c in classes}, confusion=[[0] * len(classes) for _ in classes], classes=list(classes))


def unit_metrics(y: np.ndarray, probs: np.ndarray, classes: tuple, models, **counts) -> dict:
    """metrics() of the argmax (ties -> earlier class) plus n_units, n_models and the given integer counts."""
    m = metrics(y, probs.argmax(1), classes) if len(y) else _empty_metrics(classes)
    m.update(n_units=int(len(y)), n_models=int(pd.Series(np.asarray(models, dtype=object)).nunique()), **{k: int(v) for k, v in counts.items()})
    return m


def window_frame(drive_ids, vehicles, models) -> pd.DataFrame:
    """One row per window (drive_id, vehicle_id, model): the table experiments.train.unit_level reads."""
    return pd.DataFrame(dict(drive_id=np.asarray(drive_ids, dtype=object), vehicle_id=np.asarray(vehicles, dtype=object),
                             model=np.asarray(models, dtype=object)))


def unit_level_from_windows(frame: pd.DataFrame, probs: np.ndarray, y: np.ndarray, classes: tuple, unit_col: str, all_units=None) -> dict:
    """Contract (4) through experiments.train.unit_level: mean of the per-window probability vectors per unit -> argmax
    (ties -> earlier class in task order) with n_units, n_windows_used, n_models and, given the unit universe `all_units`,
    n_units_without_windows (units of the universe without a window here)."""
    if len(y) == 0:
        m = unit_metrics(y, probs, classes, [], n_windows_used=0)
        if all_units is not None:
            m["n_units_without_windows"] = int(len(set(map(str, all_units))))
        return m
    return unit_level(frame, np.arange(len(frame)), probs, y, classes, unit_col, all_units=all_units)


def drive_level_from_windows(drive_ids, probs: np.ndarray, y: np.ndarray, classes: tuple, models, vehicles=None, all_drives=None) -> dict:
    """drive_full / drive_human_only over the given windows (+ n_drives = n_units for the readers of the old drive_level key)."""
    frame = window_frame(drive_ids, drive_ids if vehicles is None else vehicles, models)
    m = unit_level_from_windows(frame, probs, y, classes, "drive_id", all_drives)
    m["n_drives"] = m["n_units"]
    return m


def vehicle_level_from_windows(vehicles, drive_ids, probs: np.ndarray, y: np.ndarray, classes: tuple, models, min_drives: int = 1) -> dict:
    """vehicle_full (min_drives 1) / vehicle_full_multi (2): mean over all test_full windows of a vehicle, its drives pooled;
    the multi variant keeps the vehicles with >= min_drives distinct drives among these windows."""
    frame = window_frame(drive_ids, vehicles, models)
    sel = np.ones(len(frame), dtype=bool)
    if min_drives > 1 and len(frame):
        n_dr = frame.groupby("vehicle_id")["drive_id"].nunique()
        sel = frame["vehicle_id"].map(n_dr).to_numpy() >= min_drives
    m = unit_level_from_windows(frame[sel].reset_index(drop=True), np.asarray(probs)[sel], np.asarray(y)[sel], classes, "vehicle_id")
    m["n_vehicles"] = m["n_units"]
    return m


def bag_drive_level(ed: dict, classes: tuple) -> dict:
    """The model's own drive prediction (direct bag of eval_drives): unit dict with n_instances_used and
    n_units_without_instances (drives of the set without a single instance)."""
    m = unit_metrics(ed["y"], ed["probs"], classes, ed["models"], n_instances_used=int(ed["n_inst"].sum()),
                     n_units_without_instances=int(ed["n_without_instances"]))
    m["n_drives"] = m["n_units"]
    return m


def vehicle_level_bag(vehicles: np.ndarray, probs: np.ndarray, y: np.ndarray, models: np.ndarray, classes: tuple, min_drives: int = 1) -> dict:
    """One prediction per vehicle = argmax of the mean drive log-probability over its (test) drives' bag predictions."""
    df = pd.DataFrame(np.log(np.clip(probs, 1e-8, None)), columns=[f"p{i}" for i in range(len(classes))])
    df["vehicle_id"], df["y"], df["model"] = vehicles, y, models
    g = df.groupby("vehicle_id")
    cnt = g.size()
    keep = cnt.index[cnt >= min_drives]
    lp = g[[f"p{i}" for i in range(len(classes))]].mean().loc[keep].to_numpy()
    yt = g["y"].first().loc[keep].to_numpy()
    mdl = g["model"].first().loc[keep].to_numpy()
    out = unit_metrics(yt, lp, classes, mdl, n_drives_used=int(cnt.loc[keep].sum()))
    out["n_vehicles"] = int(len(keep))
    return out


def _py(o):
    """numpy scalars / arrays -> JSON-serialisable Python objects (recursive)."""
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


# ----------------------------------------------------------------------------- window-embedding provenance (contract (2))
def wemb_sidecar(path: Path) -> dict:
    side = Path(path).with_suffix(".json")
    return json.loads(side.read_text()) if side.exists() else {}


def wemb_tag_allowed(tag: str, meta: dict, split_col: str, fold: int | None = None, allow_full: bool = False) -> tuple[bool, str]:
    """Provenance rule of one window-embedding tag -> (allowed, reason). Rejected: smoke files (`_limit` in the stem or
    sidecar limit > 0, "smoke"); embeddings of the other split scheme (sidecar split_col, else the tag prefix; fold tags
    are split_platform by contract (3), "other_split"); with `fold` k (model- or device-disjoint inner validation) every
    tag that is not a fold-k checkpoint (prefix fold{k}__, "not_fold_checkpoint") unless allow_full."""
    if "_limit" in tag or int(meta.get("limit", 0) or 0) > 0:
        return False, "smoke"
    is_fold_tag = tag.startswith("fold")
    sc = meta.get("split_col") or ("split_platform" if is_fold_tag else tag.split("__")[0])
    if sc != split_col:
        return False, "other_split"
    if fold is not None and not allow_full and not tag.startswith(f"fold{int(fold)}__"):
        return False, "not_fold_checkpoint"
    return True, "ok"


def discover_wemb(root: Path, tags: str | None, split_col: str, fold: int | None = None, allow_full: bool = False, log=print) -> tuple[dict, dict]:
    """wemb_<tag>.npy files under work/tensors that pass wemb_tag_allowed: every file, or the comma-separated `tags`
    (which must all pass, else ValueError). Returns (mapping tag -> path, info with n_candidates, n_kept, dropped per
    reason); prints counts only."""
    tdir = Path(root) / "work" / "tensors"
    if tags:
        cand = {t: tdir / f"wemb_{t}.npy" for t in tags.split(",") if t}
    else:
        cand = {p.stem[len("wemb_"):]: p for p in sorted(tdir.glob("wemb_*.npy"))}
    kept, dropped = {}, {}
    for t, p in cand.items():
        ok, why = wemb_tag_allowed(t, wemb_sidecar(p), split_col, fold, allow_full)
        if ok:
            kept[t] = p
        else:
            dropped[why] = dropped.get(why, 0) + 1
    if tags and dropped:
        raise ValueError(f"{sum(dropped.values())} of the given window-embedding tags violate the provenance rule: {dropped}")
    info = dict(n_candidates=len(cand), n_kept=len(kept), dropped=dropped, fold_rule=fold is not None, allow_full=bool(allow_full))
    log(f"[wemb] candidates={len(cand)} kept={len(kept)} dropped={dropped} fold_rule={fold is not None}")
    return kept, info


# ----------------------------------------------------------------------------- output locations (contract (5))
def run_locations(split_col: str, model_name: str, seed: int, fold: int | None, eval_test: bool, recipe_id: str = "", synthetic: bool = False,
                  out_dir: Path | None = None, ckpt_dir: Path | None = None) -> dict:
    """Tags and directories of a run: final runs (fold None and the test evaluated) -> experiments/results with the
    train.py tag <split_col>__<task>__<model>__s<seed>; fold / selection runs -> experiments/recipe/results with
    fold{k}__{task}__{model}__{recipe_id}__s{seed}; synthetic runs without out_dir -> <data-root>/tmp/mil_smoke.
    Checkpoints: the sibling ckpt_mil directory, or out_dir/ckpt_mil when out_dir is given, or ckpt_dir. A run without
    the test evaluation (no protocols) must not go into experiments/results, where run_all / make_tables read."""
    is_fold_run = fold is not None or not eval_test
    fold_eff = int(fold) if fold is not None else 0
    recipe_id = recipe_id or DEFAULT_RECIPE_ID
    if is_fold_run:
        tags = {task: f"fold{fold_eff}__{task}__{model_name}__{recipe_id}__s{seed}" for task in TASKS}
        root = RECIPE_ROOT
    else:
        tags = {task: f"{split_col}__{task}__{model_name}__s{seed}" for task in TASKS}
        root = EXP_ROOT
    if out_dir is None:
        out = Path(C.DATA_ROOT) / "tmp" / "mil_smoke" if synthetic else root / "results"
        ck = out / CKPT_DIRNAME if synthetic else root / CKPT_DIRNAME
    else:
        out = Path(out_dir)
        ck = out / CKPT_DIRNAME
    if ckpt_dir is not None:
        ck = Path(ckpt_dir)
    if not eval_test and out.resolve() == (EXP_ROOT / "results").resolve():
        raise ValueError("a run without the test evaluation writes no protocols and must not go into experiments/results "
                         "(fold / selection runs belong in experiments/recipe/results)")
    return dict(tags=tags, out_dir=out, ckpt_dir=ck, is_fold_run=is_fold_run, fold=fold_eff, recipe_id=recipe_id)


# ----------------------------------------------------------------------------- one run (both tasks)
def run(split_col: str = "split_platform", seed: int = 0, data: EpisodeData | None = None, pool: str = "gated_attn", val_mode: str = "internal",
        fold: int | None = None, epochs: int | None = None, patience: int | None = None, batch_drives: int | None = None, K: int | None = None,
        K_w: int | None = None, lr: float | None = None, wd: float | None = None, human_only_train: bool = False, use_grid: bool = True,
        use_windows: bool = True, aux: bool = True, max_model_share: float | None = None, eval_test: bool = True, synthetic: bool = False,
        out_dir: Path | None = None, log=print, variant: str = "", wemb_tags: str | None = None, limit_train: int = 0, n_synthetic: int = 60,
        device: str | None = None, select_on: str = "inner_bag_f1", inner_val: str = "auto", recipe_id: str = "", fixed_epochs: int | None = None,
        wemb_allow_full: bool = False, ckpt_dir: Path | None = None, **hp) -> dict:
    """Train DriveMIL once (both tasks) and write the two results JSONs; returns {task: result}.

    val_mode internal: inner validation from EpisodeData.fold_sets (inner_val auto = car-model-disjoint folds for
    split_platform, the official device-disjoint validation for split_device; models / groups / official force one);
    official: the official validation split. select_on: inner_bag_f1 (direct bag drive macro-F1) or inner_drive_f1
    (window-mean drive_full); fixed_epochs: no early stopping, last state kept; recipe_id names fold runs; wemb_tags:
    comma-separated embedding tags (all must pass the provenance rule); wemb_allow_full admits full-split embeddings
    under a model- or device-disjoint inner validation (reported as wemb_inner_val_clean = false)."""
    if pool not in POOLS:
        raise ValueError(f"unknown pool {pool!r}")
    if val_mode not in ("internal", "official"):
        raise ValueError(f"unknown val_mode {val_mode!r}")
    if select_on not in SELECT_ON:
        raise ValueError(f"unknown select_on {select_on!r}; choose from {SELECT_ON}")
    inner_mode = "official" if val_mode == "official" else resolve_inner_val(split_col, inner_val)
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in dict(epochs=epochs, patience=patience, batch_drives=batch_drives, K=K, K_w=K_w, lr=lr, wd=wd,
                                       max_model_share=max_model_share).items() if v is not None})
    cfg.update({k: v for k, v in hp.items() if v is not None})
    if fixed_epochs:
        cfg["epochs"] = int(fixed_epochs)
    model_name = MODEL_NAMES[pool] + (f"_{variant}" if variant else "")
    loc = run_locations(split_col, model_name, seed, fold, eval_test, recipe_id, synthetic, out_dir, ckpt_dir)
    out_dir, ckpt_dir, tag_of, fold_eff, recipe_id = loc["out_dir"], loc["ckpt_dir"], loc["tags"], loc["fold"], loc["recipe_id"]
    cfg.update(pool=pool, val_mode=val_mode, inner_val=inner_mode, fold=fold, human_only_train=bool(human_only_train), use_grid=bool(use_grid),
               use_windows=bool(use_windows), aux=bool(aux), synthetic=bool(synthetic), variant=variant, select_on=select_on,
               recipe_id=recipe_id, fixed_epochs=fixed_epochs, wemb_allow_full=bool(wemb_allow_full))
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    # ---- data (+ window embeddings under the provenance rule)
    fold_rule = inner_mode in ("models", "groups")             # held-out models / devices: only fold-k embeddings are clean
    wemb_info: dict = dict(fold_rule=fold_rule)
    if data is None:
        if synthetic:
            data = make_synthetic(n_synthetic, np.random.default_rng(seed))
        else:
            mapping = None
            if use_windows:
                mapping, wemb_info = discover_wemb(C.DATA_ROOT, wemb_tags, split_col, fold_eff if fold_rule else None, wemb_allow_full, log)
            data = EpisodeData(C.DATA_ROOT, wemb=mapping or None)
    use_windows = bool(use_windows and data.wemb_dim > 0)
    wemb_clean = (not use_windows) or (not fold_rule) or all(t.startswith(f"fold{fold_eff}__") for t in data.wemb_tags)
    cfg.update(use_windows=use_windows, wemb_dim=int(data.wemb_dim), wemb_tags=list(data.wemb_tags), wemb_inner_val_clean=bool(wemb_clean))
    if fold_rule and not use_windows and wemb_info.get("dropped", {}).get("not_fold_checkpoint"):
        log("[wemb] no fold-matched window embeddings: window instances are off for this run (--wemb-allow-full admits the full-split "
            "embeddings; the inner validation is then contaminated and the JSON says so)")
    dev = torch.device(device) if device else torch.device("cpu" if synthetic or not torch.cuda.is_available() else "cuda")
    amp = dev.type == "cuda"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- drive sets
    train_ids, val_ids, fold_info = data.fold_sets(split_col, fold_eff, seed=seed, mode=inner_mode)
    fold_info.update(val_mode=val_mode, wemb_inner_val_clean=bool(wemb_clean), wemb=wemb_info)
    pool_splits = ("train", "validation")
    sets, val_rows, val_universe = {}, {}, {}
    for task in TASKS:
        pool_t = set().union(*[data.drive_sets(task, split_col, s) for s in pool_splits])
        sets[task] = dict(train=sorted(train_ids & pool_t), val=sorted(val_ids & pool_t), test=sorted(data.drive_sets(task, split_col, "test")),
                          test_human=sorted(data.drive_sets(task, split_col, "test", "test_human_only")))
        vm = np.zeros(len(data.win), dtype=bool)
        for s in pool_splits:
            vm |= data.window_mask(task, split_col, s)
        vm &= np.isin(data.win_drive, np.array(sets[task]["val"], dtype=object))
        val_rows[task] = np.flatnonzero(vm)
        universe = set().union(*[set(data.task_drives(task, split_col, s)) for s in pool_splits])
        val_universe[task] = sorted(universe & val_ids)
    train_list = sorted(d for d in train_ids if data.n_episodes_of(d, human_only=human_only_train) > 0
                        or (use_windows and not human_only_train and d in data._win_by_drive))
    n_dropped = len(train_ids) - len(train_list)
    if limit_train and len(train_list) > limit_train:
        train_list = sorted(rng.choice(np.array(train_list, dtype=object), size=limit_train, replace=False).tolist())
    if len(train_list) < 2:
        raise ValueError("fewer than two training drives with instances")
    weights = 0.5 * (data.sampling_weights(train_list, "size") + data.sampling_weights(train_list, "bev_ice"))
    model_of = data.model_of(train_list)
    stats = data.norm_stats(train_list, seed=seed)
    tstats = data.target_stats(train_list)
    prep = BatchPrep(data, stats, dev, seed=seed)
    model = DriveMIL(d=cfg["d"], pool=pool, use_grid=use_grid, use_windows=use_windows, wemb_dim=data.wemb_dim, dropout=cfg["dropout"]).to(dev)
    log(f"[{model_name}] train_drives={len(train_list)} (dropped_without_instances={n_dropped}) val_drives={len(val_ids)} "
        f"test_drives={ {t: len(sets[t]['test']) for t in TASKS} } params={n_params(model)} device={dev.type} amp={amp} "
        f"inner_val={fold_info.get('source')} select_on={select_on} wemb_dim={data.wemb_dim} wemb_clean={wemb_clean} "
        f"episodes={data.n_episodes} fold_run={loc['is_fold_run']}")

    def selection_metric(task: str) -> tuple[float, int]:
        """The inner-validation drive macro-F1 of the task under select_on (and the number of units it rests on)."""
        k = len(TASKS[task]["classes"])
        if select_on == "inner_drive_f1":
            ew_v = eval_windows(model, data, val_rows[task], task, prep, dev, amp, cfg, use_windows)
            if len(ew_v["y"]) == 0:
                return float("nan"), 0
            dl = drive_level_from_windows(ew_v["drive_ids"], ew_v["probs"], ew_v["y"], TASKS[task]["classes"], data.model_of(ew_v["drive_ids"]))
            return float(dl["macro_f1"]), int(dl["n_units"])
        ev = eval_drives(model, data, sets[task]["val"], task, prep, dev, amp, cfg, seed, use_windows)
        if len(ev["y"]) == 0:
            return float("nan"), 0
        return float(macro_f1(ev["y"], ev["probs"].argmax(1), k)), int(len(ev["y"]))

    # ---- optimisation
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(train_list) / cfg["batch_drives"])
    total = steps_per_epoch * cfg["epochs"]
    warm = max(1, int(cfg["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total, warm))
    best = {task: dict(f1=-1.0, epoch=-1, state=None) for task in TASKS}
    best_mean, bad, history = -1.0, 0, []
    last_parts: dict = {}
    for epoch in range(cfg["epochs"]):
        model.train()
        sums: dict[str, float] = {}
        n_steps = 0
        for drives in iterate_drive_batches(train_list, weights, model_of, cfg["batch_drives"], rng, cfg["max_model_share"]):
            bags = [data.sample_bag(d, cfg["K"], cfg["K_w"] if use_windows else 0, rng, human_only=human_only_train) for d in drives]
            tb = prep(to_torch(data.collate_bags(bags, list(drives), train=True, rng=rng), dev), train=True)
            out = model(tb, amp=amp)
            keep = tb["mask"][out["inst_bag_idx"], out["inst_pos"]]
            loss, parts = mil_loss(out, tb, tstats, aux=aux, label_smoothing=cfg["label_smoothing"], inst_keep=keep)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n_steps += 1
        last_parts = {k: v / max(1, n_steps) for k, v in sums.items()}
        # ---- inner validation: drive-level macro-F1 per task (select_on); fixed_epochs keeps the last state
        f1s = {}
        for task in TASKS:
            f1s[task], n_units = selection_metric(task)
            improved = n_units > 0 and f1s[task] > best[task]["f1"] + 1e-4
            if fixed_epochs or improved:
                best[task] = dict(f1=f1s[task], epoch=epoch, state=copy.deepcopy({kk: v.detach().cpu() for kk, v in model.state_dict().items()}))
        score = float(np.nanmean([v for v in f1s.values()])) if any(np.isfinite(v) for v in f1s.values()) else float("nan")
        history.append(dict(epoch=epoch, loss=last_parts.get("total", float("nan")), val_drive_f1_size=f1s["size"], val_drive_f1_bev=f1s["bev_ice"],
                            val_macro_f1=score, minutes=(time.time() - t0) / 60, parts=last_parts))
        log(f"[{model_name}] epoch {epoch + 1}/{cfg['epochs']} loss={last_parts.get('total', float('nan')):.4f} "
            f"val_drive_f1 size={f1s['size']:.4f} bev={f1s['bev_ice']:.4f} elapsed={(time.time() - t0) / 60:.1f} min")
        if np.isfinite(score) and score > best_mean + 1e-4:
            best_mean, bad = score, 0
        else:
            bad += 1
            if bad >= cfg["patience"] and not fixed_epochs:
                log(f"[{model_name}] early stop after {epoch + 1} epochs")
                break
    train_minutes = (time.time() - t0) / 60

    # ---- evaluation per task with that task's best checkpoint
    results = {}
    for task in TASKS:
        classes = TASKS[task]["classes"]
        k = len(classes)
        tag = tag_of[task]
        if best[task]["state"] is not None:
            model.load_state_dict(best[task]["state"])
        state = {kk: v.detach().cpu() for kk, v in model.state_dict().items()}
        torch.save(dict(kind=CKPT_KIND, state=state, mean=stats["crop_mean"], std=stats["crop_std"], grid_mean=stats["grid_mean"],
                        grid_std=stats["grid_std"], target_stats=tstats, cfg=cfg, task=task, split_col=split_col, model=model_name, pool=pool,
                        channels=data.channels, classes=classes, fold=fold, recipe_id=recipe_id, val_mode=val_mode), ckpt_dir / f"{tag}.pt")
        # inner validation with the best state: window metrics, window-mean drive_full (contract (4)) and the direct bag
        inner: dict = {}
        if sets[task]["val"]:
            ew_v = eval_windows(model, data, val_rows[task], task, prep, dev, amp, cfg, use_windows)
            inner["window"] = metrics(ew_v["y"], ew_v["probs"].argmax(1), classes) if len(ew_v["y"]) else None
            inner["drive_full"] = drive_level_from_windows(ew_v["drive_ids"], ew_v["probs"], ew_v["y"], classes, data.model_of(ew_v["drive_ids"]),
                                                           data.vehicle_of(ew_v["drive_ids"]), all_drives=val_universe[task])
            eb = eval_drives(model, data, sets[task]["val"], task, prep, dev, amp, cfg, seed, use_windows)
            inner["drive_full_bag"] = bag_drive_level(eb, classes)
            if loc["is_fold_run"]:
                np.savez_compressed(out_dir / f"{tag}_oof.npz", window_id=data.win["window_id"].to_numpy(object)[val_rows[task]].astype(str),
                                    probs=ew_v["probs"], y=ew_v["y"])
        val_metrics = inner.get("drive_full" if select_on == "inner_drive_f1" else "drive_full_bag")
        result = dict(tag=tag, task=task, split_col=split_col, model=model_name, arch=model_name, seed=seed, n_params=n_params(model), cfg=cfg,
                      n_train=len(sets[task]["train"]), n_val=len(sets[task]["val"]), n_test=len(sets[task]["test"]), n_train_drives_used=len(train_list),
                      epochs_run=len(history), best_epoch=best[task]["epoch"] + 1, val_macro_f1=best[task]["f1"], val_metrics=val_metrics,
                      inner_val_metrics=inner, fold=fold, recipe_id=recipe_id, decision=DECISION, select_on=select_on, fixed_epochs=fixed_epochs,
                      val_mode=val_mode, fold_info=fold_info, pool=pool, train_minutes=train_minutes, eval_test=bool(eval_test),
                      loss_parts=last_parts, history=history, protocols={}, drive_level={}, unit_level={}, vehicle_level={}, drive_level_bag={},
                      unit_level_bag={}, vehicle_level_bag={}, instance_type_counts={}, per_type_instance_f1={})
        if eval_test:
            # window-level protocols: predictions for every test_full window, the other protocols are subsets
            rows_full = data.window_rows(task, split_col, "test", "test_full")
            ew = eval_windows(model, data, rows_full, task, prep, dev, amp, cfg, use_windows)
            models_w, veh_w = data.model_of(ew["drive_ids"]), data.vehicle_of(ew["drive_ids"])
            test_drives = data.task_drives(task, split_col, "test")            # the unit universe of contract (4)
            pos = {int(r): j for j, r in enumerate(rows_full)}
            res_prot, res_drive = {}, {}
            for prot in PROTOCOLS:
                rows = data.window_rows(task, split_col, "test", prot)
                sel = np.array([pos[int(r)] for r in rows], dtype=np.int64)
                yp, pp = ew["y"][sel], ew["probs"][sel]
                res_prot[prot] = metrics(yp, pp.argmax(1), classes)
                res_prot[prot]["n_windows_without_instances"] = int((ew["n_inst"][sel] == 0).sum())
                if prot != "test_balanced":
                    res_drive[prot] = drive_level_from_windows(ew["drive_ids"][sel], pp, yp, classes, models_w[sel], veh_w[sel], all_drives=test_drives)
            vfw = vehicle_level_from_windows(veh_w, ew["drive_ids"], ew["probs"], ew["y"], classes, models_w, 1)
            vmw = vehicle_level_from_windows(veh_w, ew["drive_ids"], ew["probs"], ew["y"], classes, models_w, 2)
            # the model's own drive prediction: direct bags (all instances / human-only instances), listed separately
            ed = eval_drives(model, data, sets[task]["test"], task, prep, dev, amp, cfg, seed, use_windows)
            eh = eval_drives(model, data, sets[task]["test_human"], task, prep, dev, amp, cfg, seed, use_windows, human_only=True)
            dm, hm = bag_drive_level(ed, classes), bag_drive_level(eh, classes)
            vf = vehicle_level_bag(ed["vehicles"], ed["probs"], ed["y"], ed["models"], classes, 1)
            vmulti = vehicle_level_bag(ed["vehicles"], ed["probs"], ed["y"], ed["models"], classes, 2)
            counts, f1_type = per_type_instance_f1(ed["pred"], ed["y"], task, k)
            result.update(protocols=res_prot, drive_level=res_drive,
                          unit_level=dict(drive_full=res_drive["test_full"], drive_human_only=res_drive["test_human_only"], vehicle_full=vfw,
                                          vehicle_full_multi=vmw),
                          vehicle_level=dict(vehicle_full=vfw, vehicle_full_multi=vmw),
                          drive_level_bag={"test_full": dm, "test_human_only": hm},
                          unit_level_bag=dict(drive_full=dm, drive_human_only=hm, vehicle_full=vf, vehicle_full_multi=vmulti),
                          vehicle_level_bag=dict(vehicle_full=vf, vehicle_full_multi=vmulti), instance_type_counts=counts, per_type_instance_f1=f1_type)
            np.savez_compressed(out_dir / f"{tag}_probs.npz", window_id=data.win["window_id"].to_numpy(object)[rows_full].astype(str),
                                probs=ew["probs"], y=ew["y"])
            np.savez_compressed(out_dir / f"{tag}_drive_probs.npz", drive_id=ed["drive_ids"].astype(str), vehicle_id=ed["vehicles"].astype(str),
                                probs=ed["probs"], y=ed["y"], n_episodes=ed["n_episodes"], n_human_episodes=ed["n_human_episodes"])
            log(f"[{tag}] DONE best_val_f1={best[task]['f1']:.4f} window_full_f1={res_prot['test_full']['macro_f1']:.4f} "
                f"drive_full_f1={res_drive['test_full']['macro_f1']:.4f} drive_bag_f1={dm['macro_f1']:.4f} "
                f"vehicle_f1={vfw['macro_f1']:.4f} minutes={(time.time() - t0) / 60:.1f}")
        else:
            dv = inner.get("drive_full")
            log(f"[{tag}] DONE (no test) best_val_f1={best[task]['f1']:.4f} inner_drive_full_f1={(dv or {}).get('macro_f1', float('nan')):.4f} "
                f"minutes={(time.time() - t0) / 60:.1f}")
        (out_dir / f"{tag}.json").write_text(json.dumps(_py(result), indent=1))
        results[task] = result
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-col", choices=["split_platform", "split_device"], default="split_platform")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", choices=list(POOLS), default="gated_attn")
    ap.add_argument("--val-mode", choices=["internal", "official"], default="internal")
    ap.add_argument("--inner-val", choices=list(INNER_VAL_MODES), default="auto",
                    help="internal validation: auto = model-disjoint folds (split_platform) / official device-disjoint validation (split_device)")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--select-on", choices=list(SELECT_ON), default="inner_bag_f1")
    ap.add_argument("--fixed-epochs", type=int, default=None, help="train exactly this many epochs, no early stopping (final runs)")
    ap.add_argument("--recipe-id", default="", help=f"name of the recipe in fold-run tags (default {DEFAULT_RECIPE_ID!r})")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch-drives", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--K-w", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--human-only-train", action="store_true")
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--no-windows", action="store_true")
    ap.add_argument("--no-aux", action="store_true")
    ap.add_argument("--max-model-share", type=float, default=None)
    ap.add_argument("--no-test", action="store_true", help="skip the test evaluation (fold / selection runs -> experiments/recipe/results)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--ckpt-dir", type=Path, default=None)
    ap.add_argument("--variant", default="", help="suffix of the model name for ablations (keeps result tags apart)")
    ap.add_argument("--wemb-tags", default=None, help="comma-separated checkpoint tags of work/tensors/wemb_<tag>.npy (default: every file that "
                                                      "passes the provenance rule)")
    ap.add_argument("--wemb-allow-full", action="store_true", help="admit full-split embeddings under a model-disjoint inner validation (leaky selection)")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--synthetic-smoke", action="store_true", help="2 epochs on make_synthetic(60) on the CPU; prints loss parts and metric shapes")
    a = ap.parse_args(argv)
    kw = dict(split_col=a.split_col, seed=a.seed, pool=a.pool, val_mode=a.val_mode, inner_val=a.inner_val, fold=a.fold, select_on=a.select_on,
              fixed_epochs=a.fixed_epochs, recipe_id=a.recipe_id, epochs=a.epochs, patience=a.patience, batch_drives=a.batch_drives, K=a.K, K_w=a.K_w,
              lr=a.lr, wd=a.wd, human_only_train=a.human_only_train, use_grid=not a.no_grid, use_windows=not a.no_windows, aux=not a.no_aux,
              max_model_share=a.max_model_share, eval_test=not a.no_test, out_dir=a.out_dir, ckpt_dir=a.ckpt_dir, variant=a.variant,
              wemb_tags=a.wemb_tags, wemb_allow_full=a.wemb_allow_full, limit_train=a.limit_train, d=a.d)
    if a.synthetic_smoke:
        t0 = time.time()
        smoke = dict(SMOKE)
        kw.update(synthetic=True, n_synthetic=smoke.pop("n_drives"), epochs=a.epochs or smoke.pop("epochs"), d=a.d or smoke.pop("d"),
                  K=a.K or smoke.pop("K"), K_w=a.K_w or smoke.pop("K_w"), batch_drives=a.batch_drives or smoke.pop("batch_drives"))
        kw.update({k: v for k, v in smoke.items() if k in ("K_eval", "K_w_eval", "eval_bags")})
        res = run(**kw)
        for task, r in res.items():
            print(f"[smoke] {task}: loss_parts=" + json.dumps({k: round(v, 4) for k, v in r["loss_parts"].items()}))
            shapes = {"protocols": {p: len(m["confusion"]) for p, m in r["protocols"].items()},
                      "drive_level": {p: m["n_units"] for p, m in r["drive_level"].items()},
                      "unit_level": {p: m["n_units"] for p, m in r["unit_level"].items()},
                      "unit_level_bag": {p: m["n_units"] for p, m in r["unit_level_bag"].items()},
                      "inner_val": {p: (m["n"] if m else None) for p, m in r["inner_val_metrics"].items()},
                      "instance_type_counts": r["instance_type_counts"]}
            print(f"[smoke] {task}: metric shapes {json.dumps(shapes)}")
        print(f"[smoke] wrote {len(res)} results to the out dir in {time.time() - t0:.1f} s")
        return 0
    run(**kw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
