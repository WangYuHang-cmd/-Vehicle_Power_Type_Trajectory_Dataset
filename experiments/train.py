#!/usr/bin/env python3
"""Train and evaluate one (task, split scheme, model, seed) configuration on the 60-s windows.

Training: class-balanced sampling with the dataset's train_uniform rule ((N / K n_k)^0.5, capped), optionally
re-balanced over car models inside each class, AdamW, warm-up + cosine schedule, label smoothing, early stopping on a
selection metric (validation window macro-F1 by default; the drive-level macro-F1 of an internal car-model-disjoint
fold with fold / select_on). Recipe options: auxiliary heads (CORAL ordinal, standardised log-footprint and log-mass
regressions, four-class powertrain, conditional domain head through gradient reversal), physically safe augmentation,
down-weighting of windows with automation activity, group-DRO-lite over car models, fixed epochs, decision rules
(softmax / CORAL / footprint bucket / average). Evaluation: window-level macro-F1 / balanced accuracy / accuracy on
test_full, test_balanced and test_human_only, drive-level metrics (mean class probability over the windows of a drive)
and the unit-level protocols drive_full / drive_human_only / vehicle_full / vehicle_full_multi. Results go to
<data-root>/experiments/results/<tag>.json with the window probabilities in results/<tag>_probs.npz; fold runs go to
<data-root>/experiments/recipe/results/<tag>.json + <tag>_oof.npz (predictions on the held-out car models); a non-fold
run that skips the test split goes to <data-root>/experiments/results_notest (run_paths), because experiments/results is
rendered into RESULTS.md and needs the test protocols. include_hybrids lets the HEV / PHEV windows of the bev_ice task
train without a class label (auxiliary heads only). Prints counts and metrics only.
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
from experiments.data import PROTOCOLS, TASKS, Augment, Preprocess, WindowData, iterate_batches  # noqa: E402
from experiments.models import MODELS, build_model, coral_predict, coral_targets, n_params  # noqa: E402
from experiments import internal_folds as IF  # noqa: E402

EXP_ROOT = Path(C.DATA_ROOT) / "experiments"
RECIPE_ROOT = EXP_ROOT / "recipe"
NOTEST_ROOT = EXP_ROOT / "results_notest"      # non-fold runs that skip the test split (RESULTS.md renders experiments/results)
DEFAULTS = dict(epochs=25, batch=256, lr=1e-3, wd=0.05, d=128, n_layers=4, dropout=0.1, patience=6, label_smoothing=0.05,
                warmup_frac=0.03, eval_batch=1024)
LR_BY_MODEL = {"transformer": 5e-4}
NO_AMP = ("gru",)                    # cuDNN GRU stays in float32
AUX_PARTS = ("cls", "coral", "fp", "mass", "pt4")
LOSS_WEIGHTS = dict(cls=1.0, coral=0.5, fp=1.0, mass=0.5, pt4=0.3, dom=1.0)
SAMPLING = ("class", "class_model")
AUGMENTS = ("flip", "mask", "chdrop", "noise", "cutmix")
DECISIONS = ("softmax", "coral", "fp", "avg")
SELECT_ON = ("val_window_f1", "inner_drive_f1", "inner_window_f1")
AGGREGATIONS = ("mean", "logodds", "median", "vote")
UNIT_PROTOCOLS = ("drive_full", "drive_human_only", "vehicle_full", "vehicle_full_multi")


# ----------------------------------------------------------------------------- metrics (same definitions as starter_kit.eval)
def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Unweighted mean F1 over the classes with support."""
    cm = confusion(y_true, y_pred, k)
    f1s = []
    for c in range(k):
        tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp
        if tp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    cm = confusion(y_true, y_pred, k)
    rec = [cm[c, c] / cm[c].sum() for c in range(k) if cm[c].sum() > 0]
    return float(np.mean(rec)) if rec else float("nan")


def metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple) -> dict:
    k = len(classes)
    y_true = np.asarray(y_true, dtype=np.int64); y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion(y_true, y_pred, k)
    return dict(n=int(len(y_true)), macro_f1=macro_f1(y_true, y_pred, k), balanced_acc=balanced_accuracy(y_true, y_pred, k),
                accuracy=float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
                per_class_recall={c: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None) for i, c in enumerate(classes)},
                support={c: int(cm[i].sum()) for i, c in enumerate(classes)}, confusion=cm.tolist(), classes=list(classes))


# ----------------------------------------------------------------------------- aggregation over units (drives, vehicles)
def aggregate_probs(probs: np.ndarray, units: np.ndarray, how: str = "mean", weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One probability vector per unit from the windows of that unit -> (sorted unit ids, (U, K) array).

    mean: (weighted) mean probability; logodds: softmax of the (weighted) mean log probability (geometric mean);
    median: per-class median (weights ignored); vote: (weighted) share of the per-window argmax. The unweighted mean
    is the dataset's reference drive-level rule."""
    if how not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {how!r}; choose from {AGGREGATIONS}")
    probs = np.asarray(probs)
    units = np.asarray(units, dtype=object)
    k = probs.shape[1]
    cols = [f"p{i}" for i in range(k)]
    if how == "mean" and weights is None:
        df = pd.DataFrame(probs, columns=cols)
        df["unit"] = units
        g = df.groupby("unit")[cols].mean()
        return g.index.to_numpy(object), g.to_numpy()
    uid, inv = np.unique(units.astype(str), return_inverse=True)
    n_units = len(uid)
    w = np.ones(len(units), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if how in ("mean", "logodds"):
        vals = probs.astype(np.float64) if how == "mean" else np.log(np.clip(probs.astype(np.float64), 1e-6, None))
        num = np.zeros((n_units, k)); den = np.zeros(n_units)
        np.add.at(num, inv, vals * w[:, None])
        np.add.at(den, inv, w)
        out = num / np.maximum(den, 1e-12)[:, None]
        if how == "logodds":
            out = np.exp(out - out.max(1, keepdims=True))
            out /= out.sum(1, keepdims=True)
        return uid.astype(object), out
    if how == "median":
        out = np.zeros((n_units, k))
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(n_units + 1))
        for u in range(n_units):
            out[u] = np.median(probs[order[bounds[u]:bounds[u + 1]]], axis=0)
        return uid.astype(object), out
    out = np.zeros((n_units, k))
    np.add.at(out, (inv, probs.argmax(1)), w)
    return uid.astype(object), out / np.maximum(out.sum(1, keepdims=True), 1e-12)


def _unit_labels(uid: np.ndarray, units: np.ndarray, y: np.ndarray) -> np.ndarray:
    first = pd.Series(np.asarray(y, dtype=np.int64)).groupby(np.asarray(units, dtype=object)).first()
    return first.reindex(uid).to_numpy(dtype=np.int64)


def drive_level(win: pd.DataFrame, idx: np.ndarray, probs: np.ndarray, y: np.ndarray, classes: tuple, how: str = "mean",
                weights: np.ndarray | None = None) -> dict:
    """Aggregated probability over the windows of each drive -> one prediction per drive (ties -> earlier class)."""
    units = win["drive_id"].to_numpy(object)[idx]
    uid, p = aggregate_probs(probs, units, how, weights)
    m = metrics(_unit_labels(uid, units, y), p.argmax(1), classes)
    m["n_drives"] = int(len(uid))
    return m


def unit_level(win: pd.DataFrame, idx: np.ndarray, probs: np.ndarray, y: np.ndarray, classes: tuple, unit_col: str = "drive_id",
               all_units: np.ndarray | None = None, how: str = "mean", weights: np.ndarray | None = None) -> dict:
    """Unit-level protocol (docs/PROTOCOLS.md): mean probability over the selected windows of each unit (drive_id or
    vehicle_id) -> argmax (ties -> earlier class in task order). Returns the metrics dict plus n_units, n_windows_used,
    n_models and, when `all_units` lists every unit that should have a prediction, n_units_without_windows."""
    units = win[unit_col].to_numpy(object)[idx]
    uid, p = aggregate_probs(probs, units, how, weights)
    m = metrics(_unit_labels(uid, units, y), p.argmax(1), classes)
    m["n_units"] = int(len(uid))
    m["n_windows_used"] = int(len(idx))
    m["n_models"] = int(pd.Series(win["model"].to_numpy(object)[idx]).nunique())
    if all_units is not None:
        m["n_units_without_windows"] = int(len(set(map(str, all_units)) - set(map(str, uid))))
    return m


def unit_protocols(data: WindowData, task: str, split_col: str, idx_full: np.ndarray, p_full: np.ndarray, y_full: np.ndarray,
                   idx_human: np.ndarray, p_human: np.ndarray, y_human: np.ndarray, classes: tuple) -> dict:
    """The four unit-level protocols of the test split: drive_full, drive_human_only (test_human_only windows), vehicle_full
    (all test drives of a vehicle pooled) and vehicle_full_multi (vehicles with >= 2 drives among the test_full windows)."""
    t = TASKS[task]
    d = data.drives
    test_drives = d.loc[(d[split_col] == "test") & d[t["eligible"]].fillna(False).astype(bool) & d[t["label"]].isin(t["classes"]), "drive_id"].astype(str).to_numpy()
    out = {"drive_full": unit_level(data.win, idx_full, p_full, y_full, classes, "drive_id", all_units=test_drives),
           "drive_human_only": unit_level(data.win, idx_human, p_human, y_human, classes, "drive_id", all_units=test_drives),
           "vehicle_full": unit_level(data.win, idx_full, p_full, y_full, classes, "vehicle_id")}
    veh = data.win["vehicle_id"].to_numpy(object)[idx_full]
    drv = data.win["drive_id"].to_numpy(object)[idx_full]
    n_dr = pd.Series(drv).groupby(veh).nunique()
    multi = set(n_dr[n_dr >= 2].index)
    sel = np.array([v in multi for v in veh], dtype=bool)
    out["vehicle_full_multi"] = unit_level(data.win, idx_full[sel], p_full[sel], y_full[sel], classes, "vehicle_id")
    return out


# ----------------------------------------------------------------------------- schedule
def lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


# ----------------------------------------------------------------------------- losses
def multitask_loss(out: dict, targets: dict, weights: dict | None = None, label_smoothing: float = 0.0,
                   sample_weights: torch.Tensor | None = None, return_per_sample: bool = False):
    """Weighted sum of the head losses -> (loss, parts[, per-sample total]).

    cls: cross-entropy with label smoothing; coral: binary cross-entropy of the K-1 ordinal logits (both ignore windows
    without a class label, y = -1, and average over the labelled ones); fp / mass: SmoothL1 on the standardised log
    targets, masked (fp_mask / mass_mask) and averaged over the valid samples; pt4 / dom: cross-entropy ignoring -1 targets. Default weights LOSS_WEIGHTS (cls 1, coral 0.5, fp 1, mass 0.5, pt4 0.3, dom 1);
    `sample_weights` (B,) re-weight the samples (group DRO)."""
    w = dict(LOSS_WEIGHTS)
    w.update({k: float(v) for k, v in (weights or {}).items() if v is not None})
    y = targets["y"]
    n = y.shape[0]
    lab = (y >= 0).float()                                          # y = -1: window without a class label (hybrids)
    scale = n / lab.sum().clamp_min(1.0)
    ce = F.cross_entropy(out["cls"].float(), y.clamp_min(0), label_smoothing=label_smoothing, reduction="none")
    per = {"cls": ce * lab * scale}
    if "coral" in out:
        t = coral_targets(y.clamp_min(0), out["coral"].shape[1] + 1)
        per["coral"] = F.binary_cross_entropy_with_logits(out["coral"].float(), t, reduction="none").mean(1) * lab * scale
    for key, tkey, mkey in (("fp", "log_fp", "fp_mask"), ("mass", "log_mass", "mass_mask")):
        if key in out:
            m = targets[mkey].float()
            l1 = F.smooth_l1_loss(out[key].float().reshape(n), targets[tkey].float(), reduction="none") * m
            per[key] = l1 * (n / m.sum().clamp_min(1.0))
    for key, tkey in (("pt4", "pt4"), ("dom", "model_id")):
        if key in out:
            tgt = targets[tkey]
            valid = (tgt >= 0).float()
            ce = F.cross_entropy(out[key].float(), tgt.clamp_min(0), reduction="none") * valid
            per[key] = ce * (n / valid.sum().clamp_min(1.0))

    def reduce(v: torch.Tensor) -> torch.Tensor:
        if sample_weights is None:
            return v.mean()
        return (v * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)

    total_vec = sum(w.get(k, 1.0) * v for k, v in per.items())
    loss = reduce(total_vec)
    parts = {k: float(reduce(v).detach()) for k, v in per.items()}
    parts["total"] = float(loss.detach())
    return (loss, parts, total_vec) if return_per_sample else (loss, parts)


class GroupDRO:
    """Group-DRO-lite over car models: an exponential moving average of the loss per model; a batch's samples are
    weighted q_g / n_g with q = softmax(eta x running loss) over the models present, so the worst models count more."""

    def __init__(self, n_groups: int, eta: float = 1.0, momentum: float = 0.9):
        self.ema = torch.zeros(max(1, n_groups))
        self.seen = torch.zeros(max(1, n_groups), dtype=torch.bool)
        self.eta, self.momentum = eta, momentum

    def weights(self, per_sample: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        g = group.detach().cpu()
        v = per_sample.detach().float().cpu()
        valid = g >= 0
        sw = torch.ones(len(g))
        if valid.any():
            gv = g[valid]
            sums = torch.zeros_like(self.ema).index_add_(0, gv, v[valid])
            cnt = torch.zeros_like(self.ema).index_add_(0, gv, torch.ones(int(valid.sum())))
            present = cnt > 0
            mean = sums[present] / cnt[present]
            self.ema[present] = torch.where(self.seen[present], self.momentum * self.ema[present] + (1 - self.momentum) * mean, mean)
            self.seen[present] = True
            q = torch.softmax(self.eta * self.ema[present], dim=0)
            per_group = torch.zeros_like(self.ema)
            per_group[present] = q / cnt[present]
            sw[valid] = per_group[gv]
            sw[~valid] = float(sw[valid].mean()) if (~valid).any() else 0.0
        return (sw / sw.mean().clamp_min(1e-12)).to(per_sample.device)


# ----------------------------------------------------------------------------- prediction and decision rules
def model_n_classes(model) -> int:
    return model.head.fc2.out_features


@torch.no_grad()
def predict_probs(model, prep, data: WindowData, idx: np.ndarray, device, batch: int, amp: bool) -> np.ndarray:
    model.eval()
    out = np.zeros((len(idx), model_n_classes(model)), dtype=np.float32)
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        x = prep(data.batch(b).to(device, non_blocking=True))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        out[i:i + len(b)] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out


@torch.no_grad()
def predict(model, prep, data: WindowData, idx: np.ndarray, device, batch: int, amp: bool, fp_stats: tuple | None = None,
            mass_stats: tuple | None = None, with_emb: bool = False) -> dict:
    """Every head on the given windows: probs (softmax), coral (CORAL class probabilities), fp_mu / mass_mu (regression
    means in log units, de-standardised with (mu, sd) when given), emb (float16 embeddings, with_emb)."""
    model.eval()
    n, k = len(idx), model_n_classes(model)
    res: dict[str, np.ndarray] = {"probs": np.zeros((n, k), dtype=np.float32)}
    for i in range(0, n, batch):
        b = idx[i:i + batch]
        sl = slice(i, i + len(b))
        x = prep(data.batch(b).to(device, non_blocking=True))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            out = model.forward_all(x) if hasattr(model, "forward_all") else {"cls": model(x)}
        res["probs"][sl] = torch.softmax(out["cls"].float(), dim=-1).cpu().numpy()
        if "coral" in out:
            res.setdefault("coral", np.zeros((n, k), dtype=np.float32))[sl] = coral_predict(out["coral"]).cpu().numpy()
        for key, stats in (("fp", fp_stats), ("mass", mass_stats)):
            if key in out:
                mu, sd = stats or (0.0, 1.0)
                res.setdefault(f"{key}_mu", np.zeros(n, dtype=np.float32))[sl] = (out[key].float().reshape(len(b)) * sd + mu).cpu().numpy()
        if with_emb and "emb" in out:
            e = out["emb"].float().cpu().numpy().astype(np.float16)
            res.setdefault("emb", np.zeros((n, e.shape[1]), dtype=np.float16))[sl] = e
    return res


def class_probs_from_regression(mu: np.ndarray, sigma: float, thresholds: tuple = tuple(C.FOOTPRINT_THRESHOLDS_M2 or (4.43, 4.88))) -> np.ndarray:
    """Gaussian bucket probabilities: mu (n,) predicted log footprint (log m^2), sigma its standard deviation (log units);
    P(class j) = mass of N(mu, sigma^2) between the log thresholds -> (n, len(thresholds) + 1)."""
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    lt = np.log(np.asarray(thresholds, dtype=np.float64))
    sigma = max(float(sigma), 1e-6)
    z = torch.as_tensor((lt[None, :] - mu[:, None]) / sigma)
    cdf = torch.special.ndtr(z).numpy()                                       # P(log fp < log t_j)
    p = np.concatenate([cdf[:, :1], np.diff(cdf, axis=1), 1.0 - cdf[:, -1:]], axis=1)
    bad = ~np.isfinite(mu)
    p[bad] = 1.0 / p.shape[1]
    return np.clip(p, 0.0, 1.0)


def decide(pred: dict, decision: str = "softmax", sigma: float | None = None, thresholds: tuple | None = None) -> np.ndarray:
    """Class probabilities used for the metrics: softmax head, CORAL head, footprint-regression buckets (`fp`, needs
    sigma) or the mean of softmax and regression buckets (`avg`)."""
    if decision not in DECISIONS:
        raise ValueError(f"unknown decision {decision!r}; choose from {DECISIONS}")
    if decision == "softmax":
        return pred["probs"]
    if decision == "coral":
        if "coral" not in pred:
            raise ValueError("decision 'coral' needs the coral head (aux includes 'coral')")
        return pred["coral"]
    if "fp_mu" not in pred:
        raise ValueError(f"decision {decision!r} needs the footprint head (aux includes 'fp')")
    if sigma is not None and not np.isfinite(sigma):
        raise ValueError(f"decision {decision!r} needs a finite footprint sigma, got {sigma!r} (no validation window with a valid "
                         "footprint target, or the regression head diverged): the bucket probabilities would be NaN")
    thr = tuple(thresholds or C.FOOTPRINT_THRESHOLDS_M2)
    if pred["probs"].shape[1] != len(thr) + 1:
        raise ValueError("footprint-bucket decisions apply to the size task only")
    reg = class_probs_from_regression(pred["fp_mu"], sigma if sigma is not None else 0.1, thr).astype(np.float32)
    return reg if decision == "fp" else 0.5 * (pred["probs"] + reg)


def _parts(spec: str, allowed: tuple, sep: str, what: str) -> list[str]:
    parts = [p for p in (spec or "").split(sep) if p]
    bad = sorted(set(parts) - set(allowed))
    if bad:
        raise ValueError(f"unknown {what} {bad}; choose from {allowed}")
    return parts


def _fp_sigma(pred: dict, log_fp_nat: np.ndarray, mask: np.ndarray) -> float:
    """Root-mean-square residual of the footprint regression on the masked windows (log units)."""
    if "fp_mu" not in pred or not mask.any():
        return float("nan")
    r = pred["fp_mu"][mask].astype(np.float64) - log_fp_nat[mask]
    return float(np.sqrt(np.mean(r * r)))


def _select_metric(select_on: str, data: WindowData, va: np.ndarray, probs: np.ndarray, y: np.ndarray, classes: tuple) -> float:
    if select_on in ("val_window_f1", "inner_window_f1"):
        return macro_f1(y, probs.argmax(1), len(classes))
    return drive_level(data.win, va, probs, y, classes)["macro_f1"]


def run_paths(task: str, split_col: str, model_name: str, seed: int, recipe_id: str = "", fold: int | None = None,
              eval_test: bool = True, out_dir: Path | None = None, ckpt_dir: Path | None = None) -> tuple[str, str, Path, Path]:
    """(tag, model key, results folder, checkpoint folder) of a run. Non-fold runs keep the published layout: tag
    split_col__task__model[__recipe_id]__sSEED, results in experiments/results (rendered into RESULTS.md by run_all) and
    checkpoints in experiments/ckpt; one that skips the test split has no protocols for that table, so it goes to
    experiments/results_notest (an explicit experiments/results is refused). Fold runs: tag
    foldK__task__model__recipe_id__sSEED under experiments/recipe/{results,ckpt}."""
    model_key = model_name if not recipe_id else f"{model_name}__{recipe_id}"
    if fold is None:
        tag = f"{split_col}__{task}__{model_key}__s{seed}"
        out = Path(out_dir or (EXP_ROOT / "results" if eval_test else NOTEST_ROOT))
        if not eval_test and out.resolve() == (EXP_ROOT / "results").resolve():
            raise ValueError("a non-fold run without the test split cannot be written to experiments/results (RESULTS.md needs the "
                             "test protocols); pass another out_dir or let it go to experiments/results_notest")
        return tag, model_key, out, Path(ckpt_dir or EXP_ROOT / "ckpt")
    tag = f"fold{fold}__{task}__{model_name}__{recipe_id}__s{seed}"
    return tag, model_key, Path(out_dir or RECIPE_ROOT / "results"), Path(ckpt_dir or RECIPE_ROOT / "ckpt")


# ----------------------------------------------------------------------------- one run
def run(task: str, split_col: str, model_name: str, seed: int = 0, data: WindowData | None = None, log=print,
        out_dir: Path | None = None, *, aux: str = "cls", loss_weights: dict | None = None, sampling: str = "class",
        unit_power: float = 0.5, das_beta: float = 0.0, augment: str = "", fold: int | None = None, folds: dict | None = None,
        select_on: str = "val_window_f1", fixed_epochs: int | None = None, decision: str = "softmax", adv_lambda: float = 0.0,
        dro: bool = False, eval_test: bool = True, predict_train: bool = False, save_emb: bool = False, limit_train: int = 0,
        recipe_id: str = "", limit_val: int = 0, ckpt_dir: Path | None = None, include_hybrids: bool = False, **hp) -> dict:
    """Train one configuration and write its results JSON (+ npz predictions); returns the result dict.

    aux: heads joined by "+" out of cls / coral / fp / mass / pt4; sampling: class | class_model (model-balanced inside
    each class, hours^unit_power); das_beta: automation down-weighting; augment: parts joined by "," out of flip / mask /
    chdrop / noise / cutmix; fold: index of an internal car-model-disjoint fold (training / validation from
    experiments.internal_folds, split_col must be the folds' split column); select_on: early-stopping metric
    (val_window_f1 | inner_window_f1 | inner_drive_f1 = drive_full on the validation drives); fixed_epochs: train that
    many epochs without early stopping; decision: softmax | coral | fp | avg; adv_lambda > 0 adds the conditional
    domain head; dro: group-DRO-lite over car models; eval_test=False skips the test split (fold runs); predict_train
    saves the training-window probabilities; save_emb stores embeddings in the npz; limit_train / limit_val subsample
    the training / validation windows (debugging only); recipe_id names the recipe in the tag; ckpt_dir overrides the
    checkpoint folder (experiments/ckpt, or experiments/recipe/ckpt for fold runs); include_hybrids adds the training
    windows whose task label is known but outside the task classes (bev_ice: HEV / PHEV, WindowData.extra_mask) without
    a class label (y = -1: ignored by the cls / CORAL losses, drawn in proportion to their number, never cut-mixed) so
    the pt4 / mass / fp heads train on them (plan stage S4). A non-fold run with eval_test=False is written to
    experiments/results_notest (run_paths): experiments/results is rendered into RESULTS.md and needs the test protocols."""
    cfg = dict(DEFAULTS)
    cfg["lr"] = LR_BY_MODEL.get(model_name, cfg["lr"])
    cfg.update({k: v for k, v in hp.items() if v is not None})
    if fixed_epochs:
        cfg["epochs"] = int(fixed_epochs)
    aux_parts = _parts(aux, AUX_PARTS, "+", "aux parts") or ["cls"]
    aug_parts = _parts(augment, AUGMENTS, ",", "augmentations")
    if sampling not in SAMPLING:
        raise ValueError(f"unknown sampling {sampling!r}; choose from {SAMPLING}")
    if decision not in DECISIONS:
        raise ValueError(f"unknown decision {decision!r}; choose from {DECISIONS}")
    if select_on not in SELECT_ON:
        raise ValueError(f"unknown select_on {select_on!r}; choose from {SELECT_ON}")
    if decision == "coral" and "coral" not in aux_parts:
        raise ValueError("decision 'coral' needs aux to include 'coral'")
    if decision in ("fp", "avg") and "fp" not in aux_parts:
        raise ValueError(f"decision {decision!r} needs aux to include 'fp'")
    classes = TASKS[task]["classes"]
    k = len(classes)
    if decision in ("fp", "avg") and k != len(C.FOOTPRINT_THRESHOLDS_M2) + 1:
        raise ValueError("footprint-bucket decisions apply to the size task only")
    tag, model_key, out_dir, ckpt_dir = run_paths(task, split_col, model_name, seed, recipe_id, fold, eval_test, out_dir, ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data = data or WindowData()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # ---- index sets
    heldout: list[str] = []
    if fold is None:
        tr = data.indices(task, split_col, "train")
        va = data.indices(task, split_col, "validation")
    else:
        folds = folds or IF.load_folds(task)
        if folds.get("split_col", "split_platform") != split_col:
            raise ValueError(f"fold runs use split_col {folds.get('split_col')!r}")
        pool = data.mask(task, split_col, "train") | data.mask(task, split_col, "validation")
        tr_m, va_m = IF.fold_window_masks(data.win, folds, fold, drives_df=data.drives)
        tr = np.flatnonzero(pool & tr_m)
        va = np.flatnonzero(pool & va_m)
        heldout = list(folds["folds"][fold]["heldout_models"])
    te = data.indices(task, split_col, "test") if eval_test else np.zeros(0, dtype=np.int64)
    y_all = np.full(len(data.win), -1, dtype=np.int64)
    for arr in (tr, va, te):
        if len(arr):
            y_all[arr] = data.labels(task, arr)
    if include_hybrids:                                             # known label outside the task classes: trains with y = -1
        if fold is None:
            extra = data.extra_indices(task, split_col, "train")
        else:
            extra = np.flatnonzero((data.extra_mask(task, split_col, "train") | data.extra_mask(task, split_col, "validation")) & tr_m)
        tr = np.sort(np.concatenate([tr, extra]))
    if limit_train and len(tr) > limit_train:
        tr = np.sort(rng.choice(tr, size=int(limit_train), replace=False))
    if limit_val and len(va) > limit_val:
        va = np.sort(rng.choice(va, size=int(limit_val), replace=False))
    n_extra = int((y_all[tr] < 0).sum())
    if len(tr) - n_extra == 0 or len(va) == 0:
        raise ValueError(f"empty labelled training ({len(tr) - n_extra}) or validation ({len(va)}) set")
    w_tr = data.sampling_weights(tr, y_all[tr], k, unit=("model" if sampling == "class_model" else None), unit_power=unit_power,
                                 das_beta=das_beta)
    aux_tr = data.aux_targets(tr)
    aux_all = data.aux_targets(np.arange(len(data.win)), stats=aux_tr["stats"])
    fp_stats = (aux_tr["stats"]["fp_mu"], aux_tr["stats"]["fp_sd"])
    mass_stats = (aux_tr["stats"]["mass_mu"], aux_tr["stats"]["mass_sd"])
    if decision in ("fp", "avg") and not aux_all["fp_mask"][va].any():
        raise ValueError(f"decision {decision!r}: no validation window has a valid footprint target (footprint + wheelbase), so the "
                         "regression sigma cannot be estimated")
    with np.errstate(invalid="ignore", divide="ignore"):
        log_fp_nat = np.log(data.win["footprint_m2"].to_numpy(np.float64))
    n_models = len(data.model_codes)

    # ---- model, heads, augmentation
    heads: dict[str, int] = {}
    if "coral" in aux_parts:
        heads["coral"] = k
    if "fp" in aux_parts:
        heads["fp"] = 1
    if "mass" in aux_parts:
        heads["mass"] = 1
    if "pt4" in aux_parts:
        heads["pt4"] = len(C.POWERTRAIN4)
    if adv_lambda > 0:
        heads["dom"] = n_models
    mean, std = data.norm_stats(tr, seed=seed)
    prep = Preprocess(mean, std, data.clip_lo, data.clip_hi).to(device)
    model = build_model(model_name, len(data.channels), k, d=cfg["d"], n_layers=cfg["n_layers"], dropout=cfg["dropout"],
                        heads=heads or None).to(device)
    use_all = bool(heads)
    aug = None
    if aug_parts:
        aug = Augment(data.channels, mean, std, p_flip=0.5 if "flip" in aug_parts else 0.0, p_mask=0.5 if "mask" in aug_parts else 0.0,
                      p_chdrop=0.1 if "chdrop" in aug_parts else 0.0, noise=0.02 if "noise" in aug_parts else 0.0,
                      p_cutmix=0.25 if "cutmix" in aug_parts else 0.0).to(device)
        aug.train()
    dro_state = GroupDRO(n_models, eta=float(cfg.get("dro_eta", 1.0))) if dro else None
    amp = model_name not in NO_AMP and device.type == "cuda"
    log(f"[{tag}] train={len(tr)} val={len(va)} test={len(te)} params={n_params(model)} amp={amp} "
        f"class_counts={np.bincount(y_all[tr][y_all[tr] >= 0], minlength=k).tolist()} extra={n_extra} aux={'+'.join(aux_parts)} sampling={sampling} "
        f"augment={augment or '-'} decision={decision} select_on={select_on}"
        + (f" fold={fold} heldout_models={len(heldout)}" if fold is not None else ""))

    # ---- optimisation
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(tr) / cfg["batch"])
    total = steps_per_epoch * cfg["epochs"]
    warm = max(1, int(cfg["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total, warm))
    best = dict(metric=-1.0, epoch=-1, state=None, f1=float("nan"), sigma=float("nan"), parts={})
    history = []
    t0 = time.time()
    bad = 0

    def to_device(arr: np.ndarray, b: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr[b])).to(device)

    for epoch in range(cfg["epochs"]):
        model.train()
        tot_loss = n_seen = 0
        part_sum: dict[str, float] = {}
        for b in iterate_batches(tr, cfg["batch"], rng=rng, weights=w_tr):
            x = prep(data.batch(b).to(device, non_blocking=True))
            targets = {"y": to_device(y_all, b)}
            if use_all:
                targets.update(log_fp=to_device(aux_all["log_fp"], b), fp_mask=to_device(aux_all["fp_mask"], b),
                               log_mass=to_device(aux_all["log_mass"], b), mass_mask=to_device(aux_all["mass_mask"], b),
                               pt4=to_device(aux_all["pt4"], b), model_id=to_device(aux_all["model_id"], b))
            if aug is not None:
                x = aug(x, y=targets["y"], model_id=to_device(aux_all["model_id"], b) if "cutmix" in aug_parts else None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                out = model.forward_all(x, grl_lambda=adv_lambda) if use_all else {"cls": model(x)}
            if dro_state is not None:
                _, _, vec = multitask_loss(out, targets, loss_weights, cfg["label_smoothing"], return_per_sample=True)
                sw = dro_state.weights(vec, to_device(aux_all["model_id"], b))
                loss, parts = multitask_loss(out, targets, loss_weights, cfg["label_smoothing"], sample_weights=sw)
            else:
                loss, parts = multitask_loss(out, targets, loss_weights, cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(b)
            n_seen += len(b)
            for kk, v in parts.items():
                part_sum[kk] = part_sum.get(kk, 0.0) + v * len(b)
        parts_epoch = {kk: v / max(1, n_seen) for kk, v in part_sum.items()}
        pred_va = predict(model, prep, data, va, device, cfg["eval_batch"], amp, fp_stats, mass_stats)
        sigma = _fp_sigma(pred_va, log_fp_nat[va], aux_all["fp_mask"][va])
        probs_va = decide(pred_va, decision, sigma)
        f1 = macro_f1(y_all[va], probs_va.argmax(1), k)
        drive_f1 = drive_level(data.win, va, probs_va, y_all[va], classes)["macro_f1"]
        sel = _select_metric(select_on, data, va, probs_va, y_all[va], classes)
        history.append(dict(epoch=epoch, loss=tot_loss / max(1, n_seen), parts=parts_epoch, val_macro_f1=f1, val_drive_f1=drive_f1,
                            select_metric=sel, fp_sigma=sigma, minutes=(time.time() - t0) / 60))
        log(f"[{tag}] epoch {epoch + 1}/{cfg['epochs']} loss={tot_loss / max(1, n_seen):.4f} val_f1={f1:.4f} val_drive_f1={drive_f1:.4f} "
            f"select={sel:.4f} elapsed={(time.time() - t0) / 60:.1f} min")
        improved = sel > best["metric"] + 1e-4
        if fixed_epochs or improved:
            best = dict(metric=sel, epoch=epoch, f1=f1, sigma=sigma, parts=parts_epoch,
                        state=copy.deepcopy({kk: v.detach().cpu() for kk, v in model.state_dict().items()}))
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log(f"[{tag}] early stop after {epoch + 1} epochs (best epoch {best['epoch'] + 1})")
                break
    model.load_state_dict(best["state"])
    torch.save(dict(state=best["state"], mean=mean, std=std, cfg=cfg, task=task, split_col=split_col, model=model_name,
                    channels=data.channels, classes=classes, heads=heads, aux="+".join(aux_parts), recipe_id=recipe_id, fold=fold,
                    fp_stats=fp_stats, mass_stats=mass_stats, fp_sigma=best["sigma"], decision=decision), ckpt_dir / f"{tag}.pt")

    # ---- validation (official split or the fold's held-out models)
    pred_va = predict(model, prep, data, va, device, cfg["eval_batch"], amp, fp_stats, mass_stats, with_emb=save_emb)
    sigma = _fp_sigma(pred_va, log_fp_nat[va], aux_all["fp_mask"][va])
    probs_va = decide(pred_va, decision, sigma)
    val_metrics = metrics(y_all[va], probs_va.argmax(1), classes)
    inner = {"window": val_metrics, "drive_full": drive_level(data.win, va, probs_va, y_all[va], classes)}
    ids = data.win["window_id"].to_numpy(object)

    def npz_payload(pred: dict, idx: np.ndarray, probs: np.ndarray) -> dict:
        payload = dict(window_id=ids[idx].astype(str), probs=probs, y=y_all[idx])
        if decision != "softmax":
            payload["softmax"] = pred["probs"]
        for key in ("coral", "fp_mu", "mass_mu", "emb"):
            if key in pred:
                payload[key] = pred[key]
        return payload

    # ---- evaluation on every protocol (all protocols are subsets of test_full)
    res_prot, res_drive, res_unit = {}, {}, {}
    if eval_test:
        pred_te = predict(model, prep, data, te, device, cfg["eval_batch"], amp, fp_stats, mass_stats, with_emb=save_emb)
        p_te = decide(pred_te, decision, sigma)
        pos = {int(i): j for j, i in enumerate(te)}
        prot_idx, prot_p = {}, {}
        for prot in PROTOCOLS:
            idx = data.indices(task, split_col, "test", prot)
            sel = np.array([pos[int(i)] for i in idx], dtype=np.int64)
            yp, pp = y_all[idx], p_te[sel]
            res_prot[prot] = metrics(yp, pp.argmax(1), classes)
            if prot != "test_balanced":
                res_drive[prot] = drive_level(data.win, idx, pp, yp, classes)
            prot_idx[prot], prot_p[prot] = idx, pp
        res_unit = unit_protocols(data, task, split_col, prot_idx["test_full"], prot_p["test_full"], y_all[prot_idx["test_full"]],
                                  prot_idx["test_human_only"], prot_p["test_human_only"], y_all[prot_idx["test_human_only"]], classes)
        np.savez_compressed(out_dir / f"{tag}_probs.npz", **npz_payload(pred_te, te, p_te))
    if fold is not None:
        np.savez_compressed(out_dir / f"{tag}_oof.npz", **npz_payload(pred_va, va, probs_va))
    if predict_train:
        pred_tr = predict(model, prep, data, tr, device, cfg["eval_batch"], amp, fp_stats, mass_stats, with_emb=save_emb)
        np.savez_compressed(out_dir / f"{tag}_trainprobs.npz", **npz_payload(pred_tr, tr, decide(pred_tr, decision, sigma)))

    result = dict(tag=tag, task=task, split_col=split_col, model=model_key, arch=model_name, seed=seed, n_params=n_params(model), cfg=cfg,
                  n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)), epochs_run=len(history), best_epoch=best["epoch"] + 1,
                  val_macro_f1=val_metrics["macro_f1"], val_metrics=val_metrics, train_minutes=(time.time() - t0) / 60,
                  protocols=res_prot, drive_level=res_drive, history=history,
                  unit_level=res_unit, inner_val_metrics=inner, fold=fold, recipe_id=recipe_id, decision=decision,
                  loss_parts=best["parts"], aux="+".join(aux_parts), sampling=sampling, unit_power=unit_power, das_beta=das_beta,
                  augment=augment, select_on=select_on, fixed_epochs=fixed_epochs, adv_lambda=adv_lambda, dro=bool(dro),
                  eval_test=bool(eval_test), limit_train=int(limit_train), fp_sigma=sigma, heldout_models=heldout,
                  best_select_metric=best["metric"], include_hybrids=bool(include_hybrids), n_train_extra=n_extra)
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=1))
    msg = f"[{tag}] DONE best_epoch={best['epoch'] + 1} val_f1={val_metrics['macro_f1']:.4f} val_drive_f1={inner['drive_full']['macro_f1']:.4f}"
    if eval_test:
        msg += (f" test_full_f1={res_prot['test_full']['macro_f1']:.4f} balanced_f1={res_prot['test_balanced']['macro_f1']:.4f} "
                f"human_only_f1={res_prot['test_human_only']['macro_f1']:.4f} drive_f1={res_drive['test_full']['macro_f1']:.4f} "
                f"vehicle_f1={res_unit['vehicle_full']['macro_f1']:.4f}")
    log(msg + f" minutes={(time.time() - t0) / 60:.1f}")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--split-col", choices=["split_platform", "split_device"], default="split_platform")
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--ckpt-dir", type=Path, default=None, help="checkpoint folder (default experiments/ckpt, experiments/recipe/ckpt for fold runs)")
    ap.add_argument("--aux", default="cls", help="heads joined by '+': cls, coral, fp, mass, pt4")
    ap.add_argument("--sampling", choices=list(SAMPLING), default="class")
    ap.add_argument("--unit-power", type=float, default=0.5)
    ap.add_argument("--das-beta", type=float, default=0.0)
    ap.add_argument("--augment", default="", help="parts joined by ',': flip, mask, chdrop, noise, cutmix")
    ap.add_argument("--fold", type=int, default=None, help="internal car-model-disjoint fold index")
    ap.add_argument("--select-on", choices=list(SELECT_ON), default="val_window_f1")
    ap.add_argument("--fixed-epochs", type=int, default=None)
    ap.add_argument("--decision", choices=list(DECISIONS), default="softmax")
    ap.add_argument("--adv-lambda", type=float, default=0.0)
    ap.add_argument("--dro", action="store_true")
    ap.add_argument("--no-test", action="store_true", help="skip the test split (fold runs; a non-fold run then goes to experiments/results_notest)")
    ap.add_argument("--include-hybrids", action="store_true", help="bev_ice: HEV / PHEV training windows join without a class label (pt4 / mass / fp heads only)")
    ap.add_argument("--predict-train", action="store_true")
    ap.add_argument("--save-emb", action="store_true")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--recipe-id", default="")
    a = ap.parse_args(argv)
    run(a.task, a.split_col, a.model, a.seed, epochs=a.epochs, batch=a.batch, lr=a.lr, d=a.d, n_layers=a.n_layers,
        patience=a.patience, out_dir=a.out_dir, aux=a.aux, sampling=a.sampling, unit_power=a.unit_power, das_beta=a.das_beta,
        augment=a.augment, fold=a.fold, select_on=a.select_on, fixed_epochs=a.fixed_epochs, decision=a.decision,
        adv_lambda=a.adv_lambda, dro=a.dro, eval_test=not a.no_test, predict_train=a.predict_train, save_emb=a.save_emb,
        limit_train=a.limit_train, limit_val=a.limit_val, recipe_id=a.recipe_id, include_hybrids=a.include_hybrids, ckpt_dir=a.ckpt_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
