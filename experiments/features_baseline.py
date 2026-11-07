#!/usr/bin/env python3
"""Hand-crafted window statistics + linear / gradient-boosting classifiers (the non-deep baselines).

Features: for each of the 20 official channels the NaN-aware mean, std, min, max, 10/50/90 % quantiles, mean absolute
first difference and the finite fraction (180 features). Classifiers (scikit-learn): logistic regression on standardised
features (NaN -> 0) and a histogram gradient-boosting classifier (handles NaN natively), both class-balanced. Results are
written in the same JSON format as experiments/train.py so run_all.py renders them in the same table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.data import PROTOCOLS, TASKS, WindowData  # noqa: E402
from experiments.train import EXP_ROOT, drive_level, metrics  # noqa: E402

STATS = ("mean", "std", "min", "max", "q10", "q50", "q90", "absdiff", "finite")


def window_features(x: np.ndarray) -> np.ndarray:
    """(n, L, C) float32 with NaN -> (n, C * len(STATS)) float32."""
    n, L, Cc = x.shape
    fin = np.isfinite(x)
    nfin = fin.sum(1)                                             # (n, C)
    mean = np.nanmean(np.where(fin, x, np.nan), axis=1)
    std = np.nanstd(np.where(fin, x, np.nan), axis=1)
    xs = np.sort(x, axis=1)                                       # NaN last
    idx = lambda q: np.clip(np.floor(q * np.maximum(nfin - 1, 0)).astype(np.int64), 0, L - 1)   # noqa: E731
    take = lambda i: np.take_along_axis(xs, i[:, None, :], axis=1)[:, 0, :]                    # noqa: E731
    q10, q50, q90 = take(idx(0.10)), take(idx(0.50)), take(idx(0.90))
    mn = xs[:, 0, :]
    mx = take(np.clip(nfin - 1, 0, L - 1))
    d = np.abs(np.diff(x, axis=1))
    absdiff = np.nanmean(d, axis=1)
    out = np.stack([mean, std, mn, mx, q10, q50, q90, absdiff, nfin / L], axis=-1)     # (n, C, S)
    out = np.where(nfin[..., None] > 0, out, np.nan)
    out[..., -1] = nfin / L
    return out.reshape(n, Cc * len(STATS)).astype(np.float32)


def all_features(data: WindowData, cache: Path, chunk: int = 8000) -> np.ndarray:
    if cache.exists():
        f = np.load(cache)
        if f.shape[0] == len(data.win):
            return f
    n = len(data.win)
    out = np.zeros((n, len(data.channels) * len(STATS)), dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, chunk):
        x = np.asarray(data.X[i:i + chunk], dtype=np.float32)
        x = np.clip(x, data.clip_lo, data.clip_hi)
        out[i:i + len(x)] = window_features(x)
        if (i // chunk) % 5 == 0:
            print(f"features {i + len(x)}/{n} elapsed_s={time.time() - t0:.0f}", flush=True)
    np.save(cache, out)
    return out


def fit_predict(name: str, Xtr, ytr, Xev, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    kind = name.rsplit("_", 1)[-1]                      # feat_logreg / feat_hgb / feat_llm_hgb / feat_llmstats_hgb ...
    if kind == "logreg":
        sc = StandardScaler().fit(np.nan_to_num(Xtr, nan=0.0))
        a = np.nan_to_num(sc.transform(np.nan_to_num(Xtr, nan=0.0)), nan=0.0)
        b = np.nan_to_num(sc.transform(np.nan_to_num(Xev, nan=0.0)), nan=0.0)
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed).fit(a, ytr)
        return clf.predict_proba(b)
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                             class_weight="balanced", early_stopping=True, validation_fraction=0.1,
                                             random_state=seed).fit(Xtr, ytr)
        return clf.predict_proba(Xev)
    raise ValueError(name)


def run(name: str, task: str, split_col: str, data: WindowData, feats: np.ndarray, seed: int, out_dir: Path) -> dict:
    tag = f"{split_col}__{task}__{name}__s{seed}"
    classes = TASKS[task]["classes"]
    tr = data.indices(task, split_col, "train"); va = data.indices(task, split_col, "validation"); te = data.indices(task, split_col, "test")
    y_all = np.full(len(data.win), -1, dtype=np.int64)
    for arr in (tr, va, te):
        y_all[arr] = data.labels(task, arr)
    t0 = time.time()
    ev = np.concatenate([va, te])
    p = fit_predict(name, feats[tr], y_all[tr], feats[ev], seed)
    pv, pt = p[: len(va)], p[len(va):]
    pos = {int(i): j for j, i in enumerate(te)}
    res_prot, res_drive = {}, {}
    for prot in PROTOCOLS:
        idx = data.indices(task, split_col, "test", prot)
        sel = np.array([pos[int(i)] for i in idx], dtype=np.int64)
        res_prot[prot] = metrics(y_all[idx], pt[sel].argmax(1), classes)
        if prot != "test_balanced":
            res_drive[prot] = drive_level(data.win, idx, pt[sel], y_all[idx], classes)
    vm = metrics(y_all[va], pv.argmax(1), classes)
    result = dict(tag=tag, task=task, split_col=split_col, model=name, seed=seed, n_params=0, cfg={"features": len(STATS) * len(data.channels)},
                  n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)), epochs_run=0, best_epoch=0, val_macro_f1=vm["macro_f1"],
                  val_metrics=vm, train_minutes=(time.time() - t0) / 60, protocols=res_prot, drive_level=res_drive, history=[])
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(out_dir / f"{tag}_probs.npz", window_id=data.win["window_id"].to_numpy(object)[te].astype(str), probs=pt, y=y_all[te])
    print(f"[{tag}] val_f1={vm['macro_f1']:.4f} test_full_f1={res_prot['test_full']['macro_f1']:.4f} "
          f"balanced_f1={res_prot['test_balanced']['macro_f1']:.4f} human_only_f1={res_prot['test_human_only']['macro_f1']:.4f} "
          f"drive_f1={res_drive['test_full']['macro_f1']:.4f} minutes={(time.time() - t0) / 60:.1f}", flush=True)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="feat_logreg,feat_hgb")
    ap.add_argument("--tasks", default="size,bev_ice")
    ap.add_argument("--split-cols", default="split_platform,split_device")
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args(argv)
    data = WindowData(in_ram=False)              # chunked reads from the memory map are enough here
    cache = Path(data.splits_dir).parents[1] / "work" / "tensors" / "features_f32.npy"
    feats = all_features(data, cache)
    out_dir = EXP_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_col in a.split_cols.split(","):
        for task in a.tasks.split(","):
            for name in a.models.split(","):
                for seed in [int(s) for s in a.seeds.split(",")]:
                    if (out_dir / f"{split_col}__{task}__{name}__s{seed}.json").exists():
                        continue
                    run(name, task, split_col, data, feats, seed, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
