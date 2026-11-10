#!/usr/bin/env python3
"""Zero-shot scorer: run the final VTT window classifiers on an external window set (experiments/transfer/common.py
contract) with the exact training pre-processing, save per-window probabilities and report window / drive / vehicle /
model-level metrics with a cluster bootstrap over vehicles.

  score.py --dataset ultraav                      # both tasks, the final checkpoints (ResNet B5 bev_ice, GRU S9 size, seeds 0-2)
  score.py --dataset ultraav --task bev_ice --mask-channels V2        # full model with the longitudinal-only mask applied
  score.py --dataset ultraav --variant lon       # reduced-input variants: <ckpt-dir>/transfer__*lon*.pt (grouped by set)
  score.py --dataset ultraav --ckpts 'split_platform__size__resnet__S9__s*'   # any checkpoint glob / tag / path
  score.py --self-test [--n 300]                 # reproduce the saved test probabilities of the final checkpoints

Pre-processing = experiments.data.Preprocess with the checkpoint's training mean / std and the training clip bounds
(experiments.regen_probs.load_checkpoint, which also rebuilds the auxiliary heads and the channel mask of a reduced-input
checkpoint, cfg["channels"]). Mixed precision follows training evaluation (bfloat16 autocast except for the GRU; --no-amp
turns it off). The class probabilities follow the checkpoint's decision rule (softmax for every final model).
--mask-channels sets channels / groups / presets (common.resolve_mask) to the training "missing" state BEFORE the
pre-processing (NaN -> 0 after standardisation; lead_valid -> 0).

Outputs in <external-root>/<dataset>/work/: probs__<checkpoint>__<mask>.npz per checkpoint (probs aligned with
meta.parquet rows, y = task label or -1), probs__<set>__ens__<mask>.npz (mean over the seeds of a set),
score__<set>__<mask>.md + .json. Aggregation: mean of the window probability vectors per unit -> arg-max (ties ->
earlier class); drive = drive_id, vehicle = vehicle_id (all its drives pooled), model = all windows of a car model.
Metrics: accuracy with the chance levels 1 / K and the majority share, per-class recall, macro-F1 when >= 2 classes are
present, 95 % intervals from a cluster bootstrap over vehicles (1,000 resamples). Windows whose label is outside the task
classes (HEV / PHEV / unknown) are scored but reported separately as predicted class shares.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.transfer import common as T  # noqa: E402
from experiments.models import coral_predict  # noqa: E402
from experiments.regen_probs import load_checkpoint  # noqa: E402
from experiments.train import NO_AMP, class_probs_from_regression  # noqa: E402

FINAL_SETS = {"bev_ice": "split_platform__bev_ice__resnet__B5", "size": "split_platform__size__gru__S9"}
SEEDS = (0, 1, 2)
LEVELS = ("window", "drive", "vehicle", "model")
UNIT_COL = {"drive": "drive_id", "vehicle": "vehicle_id", "model": "model"}
_SEED_RE = re.compile(r"__s\d+$")


# ----------------------------------------------------------------------------- checkpoints
def set_name(tag: str) -> str:
    return _SEED_RE.sub("", tag)


def find_checkpoints(ckpts: str | None = None, variant: str | None = None, tasks=("size", "bev_ice"),
                     ckpt_dir: Path = T.CKPT_DIR) -> list[Path]:
    """Checkpoint files: explicit --ckpts (comma list of paths, tags or globs over tags), or --variant NAME (files
    transfer__*NAME*.pt), or the final sets of the requested tasks."""
    ckpt_dir = Path(ckpt_dir)
    out: list[Path] = []
    if ckpts:
        for tok in [t.strip() for t in ckpts.split(",") if t.strip()]:
            p = Path(tok)
            if p.suffix == ".pt" and p.exists():
                out.append(p)
                continue
            hits = sorted(q for q in ckpt_dir.glob("*.pt") if fnmatch.fnmatch(q.stem, tok) or q.stem == tok)
            if not hits:
                raise FileNotFoundError(f"no checkpoint matches {tok!r} in {ckpt_dir}")
            out += hits
    elif variant:
        pat = variant if any(ch in variant for ch in "*?[") else f"*{variant}*"
        out = sorted(q for q in ckpt_dir.glob("transfer__*.pt") if fnmatch.fnmatch(q.stem, f"transfer__{pat}") or fnmatch.fnmatch(q.stem, pat))
        if not out:
            raise FileNotFoundError(f"no transfer__*.pt checkpoint matches {variant!r} in {ckpt_dir}")
    else:
        for task in tasks:
            for s in SEEDS:
                p = ckpt_dir / f"{FINAL_SETS[task]}__s{s}.pt"
                if p.exists():
                    out.append(p)
        if not out:
            raise FileNotFoundError(f"final checkpoints not found in {ckpt_dir}")
    seen, uniq = set(), []
    for p in out:
        if p.resolve() not in seen:
            seen.add(p.resolve()); uniq.append(p)
    return uniq


def ckpt_summary(ck: dict, path: Path) -> dict:
    cfg = ck.get("cfg", {})
    return dict(tag=path.stem, task=ck["task"], model=ck["model"], classes=list(ck["classes"]), recipe=ck.get("recipe_id", ""),
                decision=ck.get("decision", "softmax"), heads=ck.get("heads"), split_col=ck.get("split_col"),
                input_channels=(cfg.get("channels") if cfg.get("channels") else "all"), epochs=cfg.get("epochs"))


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def predict_array(model, prep, ck: dict, X: np.ndarray, device: torch.device, batch: int = 1024, amp: bool = False) -> np.ndarray:
    """Class probabilities (n, K) of raw windows X (float32, NaN kept) under the checkpoint's decision rule
    (experiments.train.predict / decide without the WindowData dependency)."""
    model.eval()
    n, k = X.shape[0], len(ck["classes"])
    decision = ck.get("decision", "softmax")
    probs = np.zeros((n, k), dtype=np.float32)
    coral = np.zeros((n, k), dtype=np.float32) if decision == "coral" else None
    fp_mu = np.zeros(n, dtype=np.float32) if decision in ("fp", "avg") else None
    for i in range(0, n, batch):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(device, non_blocking=True)
        x = prep(xb)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bool(amp) and device.type == "cuda"):
            if decision == "softmax":
                out = {"cls": model(x)}
            else:
                out = model.forward_all(x)
        probs[i:i + len(xb)] = torch.softmax(out["cls"].float(), dim=-1).cpu().numpy()
        if coral is not None:
            coral[i:i + len(xb)] = coral_predict(out["coral"]).cpu().numpy()
        if fp_mu is not None:
            mu, sd = ck.get("fp_stats") or (0.0, 1.0)
            fp_mu[i:i + len(xb)] = (out["fp"].float().reshape(-1) * sd + mu).cpu().numpy()
    if decision == "softmax":
        return probs
    if decision == "coral":
        return coral
    reg = class_probs_from_regression(fp_mu, ck.get("fp_sigma") or 0.1, T.THRESHOLDS).astype(np.float32)
    return reg if decision == "fp" else (0.5 * (probs + reg)).astype(np.float32)


def pick_device(name: str = "auto") -> torch.device:
    if name == "cpu" or (name == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but CUDA is not available")
    return torch.device("cuda")


def amp_for(ck: dict, device: torch.device, no_amp: bool) -> bool:
    return (not no_amp) and ck["model"] not in NO_AMP and device.type == "cuda"


def run_checkpoint(path: Path, X: np.ndarray, device: torch.device, batch: int = 1024, no_amp: bool = False) -> tuple[np.ndarray, dict]:
    """(probs (n, K), checkpoint summary incl. amp, device and seconds) for one checkpoint on raw windows X. A CUDA
    out-of-memory error (the GPU is shared with other jobs) falls back to the CPU in float32 (info["fallback_cpu"])."""
    fallback = False
    for attempt in range(2):
        try:
            model, prep, ck = load_checkpoint(Path(path), device)
            amp = amp_for(ck, device, no_amp)
            t0 = time.time()
            probs = predict_array(model, prep, ck, X, device, batch, amp)
            break
        except torch.OutOfMemoryError:
            if device.type != "cuda" or attempt:
                raise
            print(f"[{Path(path).stem}] CUDA out of memory (GPU shared): falling back to the CPU", flush=True)
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            fallback = True
    info = ckpt_summary(ck, Path(path))
    info.update(amp=amp, device=device.type, fallback_cpu=fallback, seconds=round(time.time() - t0, 1), n=int(X.shape[0]))
    del model, prep
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs, info


# ----------------------------------------------------------------------------- metrics
def confusion(y: np.ndarray, p: np.ndarray, k: int, w: np.ndarray | None = None) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.float64)
    np.add.at(cm, (y, p), 1.0 if w is None else w)
    return cm


def cm_metrics(cm: np.ndarray) -> dict:
    """accuracy, macro-F1 over the classes with support (NaN with < 2 such classes), balanced accuracy, recalls, chance."""
    cm = np.asarray(cm, dtype=np.float64)
    k = cm.shape[0]
    n = cm.sum()
    tp = np.diag(cm); support = cm.sum(1); pred = cm.sum(0)
    fn = support - tp; fp = pred - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(support > 0, tp / support, np.nan)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    has = support > 0
    return dict(n=float(n), accuracy=float(tp.sum() / n) if n else float("nan"),
                macro_f1=float(f1[has].mean()) if has.sum() >= 2 else float("nan"),
                balanced_acc=float(np.nanmean(rec)) if has.any() else float("nan"),
                per_class_recall=[None if not np.isfinite(r) else float(r) for r in rec],
                support=[int(round(s)) for s in support], predicted=[int(round(s)) for s in pred],
                n_classes_present=int(has.sum()), chance_uniform=1.0 / k,
                chance_majority=float(support.max() / n) if n else float("nan"), confusion=cm.astype(int).tolist())


def _rep_metrics(cm_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(accuracy, macro-F1) of B confusion matrices (B, K, K)."""
    tp = np.einsum("bii->bi", cm_b)
    support = cm_b.sum(2); pred = cm_b.sum(1)
    n = support.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = tp.sum(1) / n
        den = 2 * tp + (pred - tp) + (support - tp)
        f1 = np.where(den > 0, 2 * tp / den, 0.0)
        has = support > 0
        f1m = np.where(has.sum(1) >= 2, (f1 * has).sum(1) / np.maximum(has.sum(1), 1), np.nan)
    return acc, f1m


def _ci(vals: np.ndarray) -> list | None:
    v = vals[np.isfinite(vals)]
    if len(v) < 20:
        return None
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def aggregate(probs: np.ndarray, units: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(unit ids sorted, mean probability (U, K), window count per unit, inverse index per window)."""
    uid, inv, cnt = np.unique(units.astype(str), return_inverse=True, return_counts=True)
    P = np.zeros((len(uid), probs.shape[1]), dtype=np.float64)
    np.add.at(P, inv, probs.astype(np.float64))
    return uid, P / cnt[:, None], cnt, inv


def level_eval(meta: pd.DataFrame, probs: np.ndarray, y: np.ndarray, k: int, level: str, B: int = 1000, seed: int = 0) -> dict:
    """Metrics of one level on the labelled windows (y >= 0) plus the vehicle-cluster bootstrap interval."""
    lab = y >= 0
    if not lab.any():
        return dict(n_units=0, n_windows=0)
    m = meta.loc[lab].reset_index(drop=True)
    p = probs[lab]; yy = y[lab]
    veh_ids, veh_idx = np.unique(m["vehicle_id"].astype(str).to_numpy(), return_inverse=True)
    V = len(veh_ids)
    rng = np.random.default_rng(seed)
    mult = np.zeros((B, V), dtype=np.float64)                     # vehicle multiplicities of the resamples
    for b in range(B):
        mult[b] = np.bincount(rng.integers(0, V, V), minlength=V)
    if level == "window":
        pred = p.argmax(1)
        cm = confusion(yy, pred, k)
        res = cm_metrics(cm)
        res.update(n_units=int(len(yy)), n_windows=int(len(yy)))
        cm_v = np.zeros((V, k, k)); np.add.at(cm_v, (veh_idx, yy, pred), 1.0)
        cm_b = (mult @ cm_v.reshape(V, k * k)).reshape(B, k, k)
    else:
        col = UNIT_COL[level]
        units = m[col].astype(str).to_numpy()
        uid, P, cnt, inv = aggregate(p, units)
        pred_u = P.argmax(1)
        # unit truth = the label shared by its windows (majority if a unit were mixed)
        y_u = np.zeros(len(uid), dtype=np.int64)
        for u in range(len(uid)):
            y_u[u] = np.bincount(yy[inv == u], minlength=k).argmax()
        cm = confusion(y_u, pred_u, k)
        res = cm_metrics(cm)
        res.update(n_units=int(len(uid)), n_windows=int(len(yy)), windows_per_unit_median=float(np.median(cnt)))
        if level in ("drive", "vehicle"):                          # units nested in vehicles: resample = reweight
            veh_u = np.zeros(len(uid), dtype=np.int64)
            veh_u[inv] = veh_idx
            cm_v = np.zeros((V, k, k)); np.add.at(cm_v, (veh_u, y_u, pred_u), 1.0)
            cm_b = (mult @ cm_v.reshape(V, k * k)).reshape(B, k, k)
        else:                                                      # model pools vehicles: recompute the pooled mean
            M = len(uid)
            S = np.zeros((M, V, k)); N = np.zeros((M, V))
            np.add.at(S, (inv, veh_idx), p.astype(np.float64)); np.add.at(N, (inv, veh_idx), 1.0)
            Pb = (mult @ S.transpose(1, 0, 2).reshape(V, M * k)).reshape(B, M, k)
            Nb = mult @ N.T                                        # (B, M)
            pred_b = Pb.argmax(2)
            cm_b = np.zeros((B, k, k))
            bb, mm = np.nonzero(Nb > 0)
            np.add.at(cm_b, (bb, y_u[mm], pred_b[bb, mm]), 1.0)
    acc_b, f1_b = _rep_metrics(cm_b)
    res["ci95_accuracy"] = _ci(acc_b)
    res["ci95_macro_f1"] = _ci(f1_b)
    res["n_vehicles"] = int(V)
    res["bootstrap"] = int(B)
    return res


def per_model_rows(meta: pd.DataFrame, probs: np.ndarray, y: np.ndarray, classes: tuple) -> list[dict]:
    """One row per car model with a task label: counts, mean P(true), window / drive accuracies, vehicles correct,
    model-level prediction."""
    k = len(classes)
    rows = []
    lab = y >= 0
    for model, g in meta.loc[lab].groupby("model", sort=True):
        idx = g.index.to_numpy()
        p = probs[idx]; yy = y[idx]
        yt = int(np.bincount(yy, minlength=k).argmax())
        pm = p.mean(0)
        # drives
        duid, dP, _, _ = aggregate(p, g["drive_id"].astype(str).to_numpy())
        vuid, vP, _, _ = aggregate(p, g["vehicle_id"].astype(str).to_numpy())
        rows.append(dict(model=str(model), label=classes[yt], footprint_m2=float(g["footprint_m2"].iloc[0]) if pd.notna(g["footprint_m2"].iloc[0]) else None,
                         borderline=bool(g["borderline"].iloc[0]), powertrain=str(g["powertrain"].iloc[0]),
                         n_vehicles=int(len(vuid)), n_drives=int(len(duid)), n_windows=int(len(idx)),
                         das_frac=float(np.nanmean(g["das_frac"].to_numpy(float))) if np.isfinite(g["das_frac"].to_numpy(float)).any() else None,
                         mean_p_true=float(pm[yt]), window_acc=float((p.argmax(1) == yt).mean()),
                         drive_acc=float((dP.argmax(1) == yt).mean()), vehicles_correct=int((vP.argmax(1) == yt).sum()),
                         model_pred=classes[int(pm.argmax())], model_probs=[round(float(x), 3) for x in pm]))
    return rows


def unlabelled_rows(meta: pd.DataFrame, probs: np.ndarray, y: np.ndarray, task: str, classes: tuple) -> list[dict]:
    """Predicted class shares of the windows without a task label (HEV / PHEV / unknown ...), per model."""
    col = T.TASK_LABEL[task][0]
    rows = []
    unl = y < 0
    if not unl.any():
        return rows
    for (model, lab), g in meta.loc[unl].groupby(["model", col], sort=True, dropna=False):
        idx = g.index.to_numpy()
        p = probs[idx]
        share = np.bincount(p.argmax(1), minlength=len(classes)) / len(idx)
        rows.append(dict(model=str(model), label=("unknown" if pd.isna(lab) else str(lab)), n_vehicles=int(g["vehicle_id"].nunique()),
                         n_windows=int(len(idx)), pred_share=[round(float(s), 3) for s in share], mean_probs=[round(float(x), 3) for x in p.mean(0)]))
    return rows


# ----------------------------------------------------------------------------- reporting
def _pct(x) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:.1f}"


def _ci_str(ci) -> str:
    return "" if not ci else f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def render_markdown(R: dict) -> str:
    classes = R["classes"]
    L = []
    L.append(f"# {R['dataset']}: zero-shot score of `{R['set']}` ({R['task']}, input {R['mask_tag']})")
    L.append("")
    L.append(f"Checkpoints: {', '.join(c['tag'] for c in R['checkpoints'])} (arch {R['checkpoints'][0]['model']}, decision "
             f"{R['checkpoints'][0]['decision']}, amp {R['checkpoints'][0]['amp']}, {R['checkpoints'][0]['device']}, input channels {R['checkpoints'][0]['input_channels']}). "
             f"Masked channels: {', '.join(R['masked_channels']) or 'none'}.")
    L.append(f"Windows: {R['n_windows']} ({R['n_labelled']} with a {R['task']} label, {R['n_windows'] - R['n_labelled']} without); "
             f"vehicles {R['n_vehicles']}, drives {R['n_drives']}, car models {R['n_models']}; classes {classes}; "
             f"label counts {R['label_counts']}. Chance accuracy: uniform {_pct(1 / len(classes))} %, majority {_pct(R['levels'].get('window', {}).get('chance_majority'))} % (windows).")
    L.append("")
    L.append("## Ensemble (mean probability over the seeds), by aggregation level")
    L.append("")
    rows = []
    for lv in LEVELS:
        r = R["levels"].get(lv, {})
        if not r or not r.get("n_units"):
            continue
        rec = " / ".join(f"{c} {_pct(x)}" for c, x in zip(classes, r["per_class_recall"]))
        rows.append([lv, r["n_units"], _pct(r["accuracy"]) + _ci_str(r.get("ci95_accuracy")), _pct(r["macro_f1"]) + _ci_str(r.get("ci95_macro_f1")),
                     _pct(r["chance_majority"]), rec, "/".join(str(s) for s in r["support"])])
    L.append(T.md_table(["level", "units", "accuracy % [95% CI]", "macro-F1 % [95% CI]", "majority %", "recall per class %", "support"], rows))
    if R.get("levels_no_borderline"):
        L.append("")
        L.append(f"Same, excluding the borderline models (|footprint - cut| < {T.BORDERLINE_M2} m^2: {', '.join(R['borderline_models'])}):")
        L.append("")
        rows = []
        for lv in LEVELS:
            r = R["levels_no_borderline"].get(lv, {})
            if not r or not r.get("n_units"):
                continue
            rows.append([lv, r["n_units"], _pct(r["accuracy"]) + _ci_str(r.get("ci95_accuracy")), _pct(r["macro_f1"]) + _ci_str(r.get("ci95_macro_f1")),
                         "/".join(str(s) for s in r["support"])])
        L.append(T.md_table(["level", "units", "accuracy % [95% CI]", "macro-F1 % [95% CI]", "support"], rows))
    L.append("")
    L.append("## Single seeds")
    L.append("")
    rows = []
    for c in R["checkpoints"]:
        lv = c["levels"]
        rows.append([c["tag"], _pct(lv["window"].get("accuracy")), _pct(lv["window"].get("macro_f1")), _pct(lv["drive"].get("accuracy")),
                     _pct(lv["drive"].get("macro_f1")), _pct(lv["vehicle"].get("accuracy")), _pct(lv["vehicle"].get("macro_f1")),
                     _pct(lv["model"].get("accuracy")), f"{c['seconds']:.1f}"])
    L.append(T.md_table(["checkpoint", "window acc %", "window F1 %", "drive acc %", "drive F1 %", "vehicle acc %", "vehicle F1 %", "model acc %", "seconds"], rows))
    L.append("")
    L.append("## Per car model (ensemble)")
    L.append("")
    rows = []
    for r in R["per_model"]:
        rows.append([r["model"], r["label"], r["footprint_m2"], "yes" if r["borderline"] else "", r["n_vehicles"], r["n_drives"], r["n_windows"],
                     _pct(r["das_frac"]) if r["das_frac"] is not None else "-", f"{r['mean_p_true']:.3f}", _pct(r["window_acc"]), _pct(r["drive_acc"]),
                     f"{r['vehicles_correct']}/{r['n_vehicles']}", f"{r['model_pred']} ({', '.join(str(x) for x in r['model_probs'])})"])
    L.append(T.md_table(["model", "label", "footprint", "border", "veh", "drives", "win", "DAS %", "mean P(true)", "win acc %", "drive acc %", "veh correct", "model-level pred (P)"], rows))
    if R["unlabelled"]:
        L.append("")
        L.append(f"## Windows without a {R['task']} label (scored, not in the metrics): predicted class shares")
        L.append("")
        rows = [[r["model"], r["label"], r["n_vehicles"], r["n_windows"], ", ".join(f"{c} {_pct(s)}" for c, s in zip(classes, r["pred_share"]))] for r in R["unlabelled"]]
        L.append(T.md_table(["model", "label", "veh", "win", "predicted share %"], rows))
    L.append("")
    L.append(f"Generated {R['generated']} by experiments/transfer/score.py; bootstrap {R['bootstrap']} vehicle resamples, seed {R['seed']}.")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- scoring one set
def score_set(dataset: str, X: np.ndarray, meta: pd.DataFrame, paths: list[Path], task: str, masked: list[str], device: torch.device,
              out_dir: Path, batch: int = 1024, no_amp: bool = False, B: int = 1000, seed: int = 0, exclude_borderline: bool = False,
              log=print) -> dict:
    """Score one set of checkpoints (same task; usually the seeds of one recipe) on a window set; writes the probability
    files and the report; returns the result dict."""
    classes = None
    mt = T.mask_tag(masked)
    y = T.task_labels(meta, task)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Xm = T.apply_channel_mask(X, masked) if masked else X
    members, infos = [], []
    for p in paths:
        probs, info = run_checkpoint(p, Xm, device, batch, no_amp)
        if info["task"] != task:
            raise ValueError(f"{p.stem} is a {info['task']} checkpoint, set task is {task}")
        classes = tuple(info["classes"])
        np.savez_compressed(out_dir / f"probs__{p.stem}__{mt}.npz", probs=probs, y=y, classes=np.array(classes, dtype=object),
                            vehicle_id=meta["vehicle_id"].astype(str).to_numpy(), drive_id=meta["drive_id"].astype(str).to_numpy(),
                            w_idx=meta["w_idx"].to_numpy(np.int32), masked=np.array(masked, dtype=object), checkpoint=p.stem)
        info["levels"] = {lv: _quick_levels(meta, probs, y, len(classes), lv) for lv in LEVELS}
        log(f"[{dataset}] {p.stem} amp={info['amp']} n={info['n']} window_acc={_pct(info['levels']['window'].get('accuracy'))} "
            f"drive_acc={_pct(info['levels']['drive'].get('accuracy'))} vehicle_acc={_pct(info['levels']['vehicle'].get('accuracy'))} {info['seconds']} s")
        members.append(probs); infos.append(info)
    ens = np.mean(np.stack(members), axis=0).astype(np.float32)
    sname = set_name(paths[0].stem) if len({set_name(p.stem) for p in paths}) == 1 else "+".join(sorted({set_name(p.stem) for p in paths}))
    np.savez_compressed(out_dir / f"probs__{sname}__ens__{mt}.npz", probs=ens, y=y, classes=np.array(classes, dtype=object),
                        vehicle_id=meta["vehicle_id"].astype(str).to_numpy(), drive_id=meta["drive_id"].astype(str).to_numpy(),
                        w_idx=meta["w_idx"].to_numpy(np.int32), masked=np.array(masked, dtype=object),
                        members=np.array([p.stem for p in paths], dtype=object))
    levels = {lv: level_eval(meta, ens, y, len(classes), lv, B=B, seed=seed) for lv in LEVELS}
    R = dict(dataset=dataset, task=task, set=sname, classes=list(classes), mask_tag=mt, masked_channels=masked, checkpoints=infos,
             n_windows=int(len(meta)), n_labelled=int((y >= 0).sum()), n_vehicles=int(meta["vehicle_id"].nunique()),
             n_drives=int(meta["drive_id"].nunique()), n_models=int(meta["model"].nunique()),
             label_counts={classes[i]: int((y == i).sum()) for i in range(len(classes))}, levels=levels,
             per_model=per_model_rows(meta, ens, y, classes), unlabelled=unlabelled_rows(meta, ens, y, task, classes),
             bootstrap=B, seed=seed, generated=time.strftime("%Y-%m-%d %H:%M"), files=dict(ens=str(out_dir / f"probs__{sname}__ens__{mt}.npz")))
    bl = meta["borderline"].fillna(False).astype(bool).to_numpy() & (y >= 0)
    if task == "size" and (exclude_borderline or bl.any()) and bl.any() and (~bl & (y >= 0)).any():
        y_nb = np.where(bl, -1, y)
        R["levels_no_borderline"] = {lv: level_eval(meta, ens, y_nb, len(classes), lv, B=B, seed=seed) for lv in LEVELS}
        R["borderline_models"] = sorted(meta.loc[bl, "model"].astype(str).unique().tolist())
    md = render_markdown(R)
    (out_dir / f"score__{sname}__{mt}.md").write_text(md)
    (out_dir / f"score__{sname}__{mt}.json").write_text(json.dumps(R, indent=1, default=_json_default))
    R["markdown"] = md
    return R


def _quick_levels(meta, probs, y, k, level) -> dict:
    """Point metrics of a level without the bootstrap (single seeds)."""
    lab = y >= 0
    if not lab.any():
        return {}
    m = meta.loc[lab].reset_index(drop=True); p = probs[lab]; yy = y[lab]
    if level == "window":
        return cm_metrics(confusion(yy, p.argmax(1), k)) | {"n_units": int(len(yy))}
    uid, P, cnt, inv = aggregate(p, m[UNIT_COL[level]].astype(str).to_numpy())
    y_u = np.array([np.bincount(yy[inv == u], minlength=k).argmax() for u in range(len(uid))], dtype=np.int64)
    return cm_metrics(confusion(y_u, P.argmax(1), k)) | {"n_units": int(len(uid))}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def group_sets(paths: list[Path], tasks: set[str] | None) -> dict[tuple[str, str], list[Path]]:
    """{(task, set name): [checkpoint paths]} from the checkpoint headers (reads task / classes only)."""
    sets: dict[tuple[str, str], list[Path]] = {}
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        task = ck["task"]
        if tasks and task not in tasks:
            continue
        sets.setdefault((task, set_name(p.stem)), []).append(p)
    return sets


# ----------------------------------------------------------------------------- self-test against the official pipeline
def self_test(n: int = 300, seed: int = 0, batch: int = 1024, B: int = 200, ckpts: str | None = None, log=print,
              device: torch.device | None = None) -> dict:
    """Score N random windows of OUR test split (test_full of each final checkpoint's task, from work/tensors/X_f16.npy via
    experiments.data.WindowData) through the scorer path (save_windows -> load_windows -> mask -> Preprocess -> model) and
    compare with the saved test probabilities of the same checkpoint (results/<tag>_probs.npz); reports the max abs
    difference with the training AMP rule and in float32. Also exercises cut_windows / resample_to_10hz on synthetic data."""
    from experiments.data import WindowData
    rep: dict = {"unit_checks": _unit_checks(), "checkpoints": []}
    log("unit checks: " + ", ".join(f"{k}={v}" for k, v in rep["unit_checks"].items()))
    device = device or pick_device()
    data = WindowData(in_ram=False)
    pos = {w: i for i, w in enumerate(data.win["window_id"].to_numpy(object))}
    paths = find_checkpoints(ckpts) if ckpts else find_checkpoints()
    by_task: dict[str, list[Path]] = {}
    for p in paths:
        by_task.setdefault(torch.load(p, map_location="cpu", weights_only=False)["task"], []).append(p)
    for task, tpaths in by_task.items():
        ref = np.load(T.RESULTS_DIR / f"{tpaths[0].stem}_probs.npz", allow_pickle=True)
        wid_all = ref["window_id"].astype(str)
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(wid_all), size=min(n, len(wid_all)), replace=False))
        wids = wid_all[sel]
        tidx = np.array([pos[w] for w in wids], dtype=np.int64)
        order = np.argsort(tidx)
        X = np.asarray(data.X[tidx[order]], dtype=np.float32)
        X = X[np.argsort(order)]                                                     # back to the wids order
        w = data.win.iloc[tidx]
        meta = pd.DataFrame(dict(dataset=f"_selftest_{task}", vehicle_id=w["vehicle_id"].astype(str).to_numpy(), model=w["model"].astype(str).to_numpy(),
                                 platform=w["platform"].to_numpy(object), drive_id=w["drive_id"].astype(str).to_numpy(),
                                 w_idx=w["w_idx"].to_numpy(np.int32), t0_s=w["t0_s"].to_numpy(float), das_frac=w["das_frac"].to_numpy(float),
                                 moving_frac=w["moving_frac"].to_numpy(float), footprint_m2=w["footprint_m2"].to_numpy(float),
                                 size_class=w["size_class"].to_numpy(object), powertrain=w["powertrain"].to_numpy(object),
                                 note="internal test split window (self-test)"))
        dname = f"_selftest_{task}"
        T.save_windows(dname, X, meta)
        X2, meta2 = T.load_windows(dname)
        assert np.array_equal(np.nan_to_num(X2, nan=-9e9), np.nan_to_num(X, nan=-9e9)), "save/load round trip changed X"
        for p in tpaths:
            saved = np.load(T.RESULTS_DIR / f"{p.stem}_probs.npz", allow_pickle=True)
            spos = {w_: i for i, w_ in enumerate(saved["window_id"].astype(str))}
            ref_p = saved["probs"][[spos[w_] for w_ in wids]]
            out = dict(tag=p.stem, task=task, n=int(len(wids)))
            for label, no_amp in (("amp_rule", False), ("fp32", True)):
                probs, info = run_checkpoint(p, X2, device, batch, no_amp)
                d = np.abs(probs.astype(np.float64) - ref_p.astype(np.float64))
                out[label] = dict(amp=info["amp"], device=info["device"], max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
                                  argmax_agree=float((probs.argmax(1) == ref_p.argmax(1)).mean()), pass_1e3=bool(d.max() <= 1e-3))
            log(f"[self-test] {p.stem}: amp-rule(amp={out['amp_rule']['amp']}, {out['amp_rule']['device']}) max|dp|={out['amp_rule']['max_abs_diff']:.2e} "
                f"fp32 max|dp|={out['fp32']['max_abs_diff']:.2e} argmax agree {out['amp_rule']['argmax_agree']:.4f}")
            rep["checkpoints"].append(out)
        # the full scoring path on this set (bootstrap small: it is a smoke test)
        R = score_set(dname, X2, meta2, tpaths, task, [], device, T.work_dir(dname), batch=batch, no_amp=False, B=B, seed=seed, log=log)
        rep[f"score_{task}"] = {lv: {k_: R["levels"][lv].get(k_) for k_ in ("n_units", "accuracy", "macro_f1", "ci95_accuracy")} for lv in LEVELS}
        log(R["markdown"])
        # masked run (lead group) as a smoke test of --mask-channels
        Rm = score_set(dname, X2, meta2, tpaths[:1], task, T.resolve_mask("lead"), device, T.work_dir(dname), batch=batch, B=50, seed=seed, log=log)
        rep[f"score_{task}_mask_lead_window_acc"] = Rm["levels"]["window"].get("accuracy")
    rep["all_pass_1e3"] = all(c["amp_rule"]["pass_1e3"] for c in rep["checkpoints"])
    log(f"self-test: all checkpoints within 1e-3 under the training AMP rule: {rep['all_pass_1e3']}")
    (T.work_dir("_selftest_size").parent.parent / "_selftest_report.json").write_text(json.dumps(rep, indent=1, default=_json_default))
    return rep


def _unit_checks() -> dict:
    """Synthetic checks of cut_windows / resample_to_10hz / apply_channel_mask; returns {check: 'ok'} or raises."""
    out = {}
    n = 1900
    t = np.arange(n) / 10.0
    t[1300:] += 2.0                                              # one 2-s gap before row 1300
    v = np.full(n, 15.0); v[600:1200] = 0.0                      # second block stationary
    df = pd.DataFrame(dict(t_s=t, v=v, a=np.zeros(n), gas=(np.arange(n) % 7 == 0), das=np.ones(n, dtype=bool)))
    cmap = {"v_mps": "v", "a_mps2": "a", "gas_pressed": "gas", "das_enabled": "das"}
    r = T.cut_windows(df, cmap, info=dict(dataset="synthetic", vehicle_id="car1", model="Test Car", drive_id="d1", footprint_m2=4.5, powertrain="ice"))
    # run alignment: run 1 rows 0-1299 -> blocks 0, 600 (stationary -> dropped); run 2 rows 1300-1899 -> one block
    assert r.stats["n_candidates"] == 3 and r.stats["n_dropped_stationary"] == 1 and r.X.shape == (2, 600, 20), r.stats
    assert r.meta["size_class"].iloc[0] == "MID" and r.meta["powertrain"].iloc[0] == "ICE" and not bool(r.meta["borderline"].iloc[0])
    assert np.isnan(r.X[0, :, T.CH["steer_angle_deg"]]).all() and r.X[0, :, T.CH["gas_pressed"]].max() == 1.0
    assert r.meta["t0_s"].iloc[1] == t[1300] and r.meta["channels_present"].iloc[0] == "v_mps,a_mps2,gas_pressed,das_enabled"
    r2 = T.cut_windows(df, cmap, align="drive", info=dict(dataset="synthetic", vehicle_id="car1", model="Test Car", drive_id="d1"))
    assert r2.stats["n_candidates"] == 3 and r2.stats["n_dropped_gap"] == 1 and r2.X.shape[0] == 1, r2.stats   # block 1200-1799 straddles the gap
    out["cut_windows"] = "ok"
    ts = np.arange(0, 10, 0.01); xs = ts * 2.0                   # 100 Hz ramp
    g, lin = T.resample_to_10hz(ts, xs, "linear")
    assert len(g) == 100 and np.allclose(lin, g * 2.0)
    g2, dec = T.resample_to_10hz(ts, xs, "decimate", t_grid=g)
    assert np.allclose(dec, g * 2.0)
    flags = np.zeros(len(ts), dtype=bool); flags[355] = True     # one 10-ms brake tap at t = 3.55
    _, anyf = T.resample_to_10hz(ts, flags, "any", t_grid=g)
    assert anyf[36] == 1.0 and anyf.sum() == 1.0
    _, held = T.resample_to_10hz(np.array([0.0, 1.0, 5.0]), np.array([1.0, 2.0, 3.0]), "hold", t_grid=g, max_age_s=0.75)
    assert held[0] == 1.0 and np.isnan(held[20]) and held[52] == 3.0
    out["resample_to_10hz"] = "ok"
    Xm = T.apply_channel_mask(np.ones((2, 600, 20), np.float32), T.resolve_mask("lead"))
    assert np.isnan(Xm[:, :, T.CH["thw_s"]]).all() and (Xm[:, :, T.CH["lead_valid"]] == 0).all() and (Xm[:, :, T.CH["v_mps"]] == 1).all()
    out["apply_channel_mask"] = "ok"
    lc = T.lead_channels(np.full(50, 20.0), np.ones(50), np.full(50, 30.0), np.linspace(-2, 0, 50))
    assert np.isclose(lc["thw_s"][0], 1.5) and np.isclose(lc["ttc_s"][0], 15.0) and np.isnan(lc["ttc_s"][-1]) and np.isfinite(lc["lead_a_slope_mps2"][10])
    out["lead_channels"] = "ok"
    return out


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="window set name under the external root (<root>/<dataset>/work/windows.npz)")
    ap.add_argument("--root", type=Path, default=None, help=f"external root (default {T.EXTERNAL_ROOT})")
    ap.add_argument("--task", default="both", choices=["size", "bev_ice", "both"])
    ap.add_argument("--ckpts", default=None, help="comma list of checkpoint paths / tags / globs over tags (default: the final sets)")
    ap.add_argument("--variant", default=None, help="reduced-input variants: transfer__*<variant>*.pt in the checkpoint folder")
    ap.add_argument("--ckpt-dir", type=Path, default=T.CKPT_DIR)
    ap.add_argument("--mask-channels", default=None, help="comma list of channels / groups (lead, torque, pedals, lateral, yaw, steer) / presets (V1, V2, V3) set to missing before pre-processing")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--no-amp", action="store_true", help="float32 inference for every architecture (training evaluation used bfloat16 except for the GRU)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="inference device (auto = cuda when available; CUDA out-of-memory falls back to the CPU)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="vehicle-cluster bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-borderline", action="store_true", help="also report the size metrics without the borderline models (done automatically when any exist)")
    ap.add_argument("--self-test", action="store_true", help="reproduce the saved test probabilities of the final checkpoints on N random internal test windows")
    ap.add_argument("--n", type=int, default=300, help="self-test: windows per task")
    a = ap.parse_args(argv)
    device = pick_device(a.device)
    if a.self_test:
        rep = self_test(n=a.n, seed=a.seed, batch=a.batch, ckpts=a.ckpts, device=device)
        return 0 if rep["all_pass_1e3"] else 2
    if not a.dataset:
        ap.error("--dataset or --self-test is required")
    tasks = {"size", "bev_ice"} if a.task == "both" else {a.task}
    X, meta = T.load_windows(a.dataset, a.root)
    X, meta, warns = T.validate_windows(X, meta, strict=False)
    for w in warns:
        print(f"[{a.dataset}] warning: {w}")
    masked = T.resolve_mask(a.mask_channels)
    paths = find_checkpoints(a.ckpts, a.variant, tasks=sorted(tasks), ckpt_dir=a.ckpt_dir)
    sets = group_sets(paths, tasks)
    if not sets:
        raise SystemExit("no checkpoint of the requested task(s)")
    print(f"[{a.dataset}] windows={len(meta)} vehicles={meta['vehicle_id'].nunique()} drives={meta['drive_id'].nunique()} models={meta['model'].nunique()} "
          f"device={device.type} mask={T.mask_tag(masked)} sets={[k[1] for k in sets]}")
    out_dir = T.work_dir(a.dataset, a.root)
    for (task, sname), spaths in sets.items():
        R = score_set(a.dataset, X, meta, spaths, task, masked, device, out_dir, batch=a.batch, no_amp=a.no_amp, B=a.bootstrap, seed=a.seed,
                      exclude_borderline=a.exclude_borderline)
        print()
        print(R["markdown"])
        print(f"[{a.dataset}] wrote {out_dir / f'score__{sname}__{T.mask_tag(masked)}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
