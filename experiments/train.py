#!/usr/bin/env python3
"""Train and evaluate one (task, split scheme, model, seed) configuration on the 60-s windows.

Training: class-balanced sampling with the dataset's train_uniform rule ((N / K n_k)^0.5, capped), AdamW, warm-up +
cosine schedule, label smoothing, early stopping on the validation macro-F1 of the same split scheme. Evaluation:
window-level macro-F1 / balanced accuracy / accuracy on test_full, test_balanced and test_human_only, plus drive-level
metrics (mean class probability over the windows of a drive). Results go to <data-root>/experiments/results/<tag>.json
with the window probabilities in results/<tag>_probs.npz; prints counts and metrics only.
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
from experiments.data import PROTOCOLS, TASKS, Preprocess, WindowData, iterate_batches  # noqa: E402
from experiments.models import MODELS, build_model, n_params  # noqa: E402

EXP_ROOT = Path(C.DATA_ROOT) / "experiments"
DEFAULTS = dict(epochs=25, batch=256, lr=1e-3, wd=0.05, d=128, n_layers=4, dropout=0.1, patience=6, label_smoothing=0.05,
                warmup_frac=0.03, eval_batch=1024)
LR_BY_MODEL = {"transformer": 5e-4}
NO_AMP = ("gru",)                    # cuDNN GRU stays in float32


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
    cm = confusion(y_true, y_pred, k)
    return dict(n=int(len(y_true)), macro_f1=macro_f1(y_true, y_pred, k), balanced_acc=balanced_accuracy(y_true, y_pred, k),
                accuracy=float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
                per_class_recall={c: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None) for i, c in enumerate(classes)},
                support={c: int(cm[i].sum()) for i, c in enumerate(classes)}, confusion=cm.tolist(), classes=list(classes))


# ----------------------------------------------------------------------------- schedule
def lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


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


def model_n_classes(model) -> int:
    return model.head.fc2.out_features


def drive_level(win: pd.DataFrame, idx: np.ndarray, probs: np.ndarray, y: np.ndarray, classes: tuple) -> dict:
    """Mean probability over the windows of each drive -> one prediction per drive."""
    df = pd.DataFrame(probs, columns=[f"p{i}" for i in range(len(classes))])
    df["drive_id"] = win["drive_id"].to_numpy(object)[idx]
    df["y"] = y
    g = df.groupby("drive_id")
    p = g[[f"p{i}" for i in range(len(classes))]].mean().to_numpy()
    yt = g["y"].first().to_numpy()
    m = metrics(yt, p.argmax(1), classes)
    m["n_drives"] = int(len(yt))
    return m


# ----------------------------------------------------------------------------- one run
def run(task: str, split_col: str, model_name: str, seed: int = 0, data: WindowData | None = None, log=print,
        out_dir: Path | None = None, **hp) -> dict:
    cfg = dict(DEFAULTS)
    cfg["lr"] = LR_BY_MODEL.get(model_name, cfg["lr"])
    cfg.update({k: v for k, v in hp.items() if v is not None})
    tag = f"{split_col}__{task}__{model_name}__s{seed}"
    out_dir = Path(out_dir or EXP_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "ckpt").mkdir(parents=True, exist_ok=True)
    data = data or WindowData()
    classes = TASKS[task]["classes"]
    k = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    tr = data.indices(task, split_col, "train")
    va = data.indices(task, split_col, "validation")
    te = data.indices(task, split_col, "test")
    y_all = np.full(len(data.win), -1, dtype=np.int64)
    for arr in (tr, va, te):
        y_all[arr] = data.labels(task, arr)
    w_tr = data.class_sampling_weights(y_all[tr], k)
    mean, std = data.norm_stats(tr, seed=seed)
    prep = Preprocess(mean, std, data.clip_lo, data.clip_hi).to(device)
    model = build_model(model_name, len(data.channels), k, d=cfg["d"], n_layers=cfg["n_layers"], dropout=cfg["dropout"]).to(device)
    amp = model_name not in NO_AMP and device.type == "cuda"
    log(f"[{tag}] train={len(tr)} val={len(va)} test={len(te)} params={n_params(model)} amp={amp} "
        f"class_counts={np.bincount(y_all[tr], minlength=k).tolist()}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(tr) / cfg["batch"])
    total = steps_per_epoch * cfg["epochs"]
    warm = max(1, int(cfg["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total, warm))
    best = dict(f1=-1.0, epoch=-1, state=None)
    history = []
    t0 = time.time()
    bad = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        tot_loss = n_seen = 0
        for b in iterate_batches(tr, cfg["batch"], rng=rng, weights=w_tr):
            x = prep(data.batch(b).to(device, non_blocking=True))
            y = torch.from_numpy(y_all[b]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += float(loss) * len(b)
            n_seen += len(b)
        pv = predict_probs(model, prep, data, va, device, cfg["eval_batch"], amp)
        f1 = macro_f1(y_all[va], pv.argmax(1), k)
        history.append(dict(epoch=epoch, loss=tot_loss / max(1, n_seen), val_macro_f1=f1, minutes=(time.time() - t0) / 60))
        log(f"[{tag}] epoch {epoch + 1}/{cfg['epochs']} loss={tot_loss / max(1, n_seen):.4f} val_f1={f1:.4f} "
            f"elapsed={(time.time() - t0) / 60:.1f} min")
        if f1 > best["f1"] + 1e-4:
            best = dict(f1=f1, epoch=epoch, state=copy.deepcopy({kk: v.detach().cpu() for kk, v in model.state_dict().items()}))
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log(f"[{tag}] early stop after {epoch + 1} epochs (best epoch {best['epoch'] + 1})")
                break
    model.load_state_dict(best["state"])
    torch.save(dict(state=best["state"], mean=mean, std=std, cfg=cfg, task=task, split_col=split_col, model=model_name,
                    channels=data.channels, classes=classes), EXP_ROOT / "ckpt" / f"{tag}.pt")

    # ---- evaluation on every protocol (all protocols are subsets of test_full)
    p_te = predict_probs(model, prep, data, te, device, cfg["eval_batch"], amp)
    pos = {int(i): j for j, i in enumerate(te)}
    res_prot, res_drive = {}, {}
    for prot in PROTOCOLS:
        idx = data.indices(task, split_col, "test", prot)
        sel = np.array([pos[int(i)] for i in idx], dtype=np.int64)
        yp, pp = y_all[idx], p_te[sel]
        res_prot[prot] = metrics(yp, pp.argmax(1), classes)
        if prot != "test_balanced":
            res_drive[prot] = drive_level(data.win, idx, pp, yp, classes)
    pv = predict_probs(model, prep, data, va, device, cfg["eval_batch"], amp)
    result = dict(tag=tag, task=task, split_col=split_col, model=model_name, seed=seed, n_params=n_params(model), cfg=cfg,
                  n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)), epochs_run=len(history), best_epoch=best["epoch"] + 1,
                  val_macro_f1=best["f1"], val_metrics=metrics(y_all[va], pv.argmax(1), classes), train_minutes=(time.time() - t0) / 60,
                  protocols=res_prot, drive_level=res_drive, history=history)
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(out_dir / f"{tag}_probs.npz", window_id=data.win["window_id"].to_numpy(object)[te].astype(str),
                        probs=p_te, y=y_all[te])
    log(f"[{tag}] DONE best_val_f1={best['f1']:.4f} test_full_f1={res_prot['test_full']['macro_f1']:.4f} "
        f"balanced_f1={res_prot['test_balanced']['macro_f1']:.4f} human_only_f1={res_prot['test_human_only']['macro_f1']:.4f} "
        f"drive_f1={res_drive['test_full']['macro_f1']:.4f} minutes={(time.time() - t0) / 60:.1f}")
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
    a = ap.parse_args(argv)
    run(a.task, a.split_col, a.model, a.seed, epochs=a.epochs, batch=a.batch, lr=a.lr, d=a.d, n_layers=a.n_layers,
        patience=a.patience, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
