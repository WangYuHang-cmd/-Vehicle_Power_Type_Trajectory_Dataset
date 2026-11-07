"""Shared pieces of the audits: macro-F1, group-disjoint folds, two tiny classifiers under cross-validation
(per-key majority vote, nearest centroid), chance levels and categorical keys. numpy / pandas only."""
from __future__ import annotations

import numpy as np
import pandas as pd

NA = "NA"


def to_str(values) -> np.ndarray:
    """Object array of strings; null -> 'NA'; booleans / numbers printed plainly (vectorised: 200k values in ms)."""
    s = pd.Series(values.to_numpy(object) if isinstance(values, (pd.Series, pd.Index)) else list(values), dtype=object)
    out = s.astype(str).to_numpy(object)
    out[s.isna().to_numpy()] = NA
    return out


def _codes(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Integer codes of two string arrays over their joint label set (sorted): (codes_true, codes_pred, n_labels)."""
    labs, inv = np.unique(np.concatenate([y_true, y_pred]).astype(str), return_inverse=True)
    return inv[:len(y_true)], inv[len(y_true):], len(labs)


def _f1_from_codes(t: np.ndarray, p: np.ndarray, k: int) -> float:
    """Macro-F1 from integer codes: bincount of true / predicted / correct per class, O(n + k)."""
    n_true = np.bincount(t, minlength=k); n_pred = np.bincount(p, minlength=k)
    tp = np.bincount(t[t == p], minlength=k)
    denom = n_true + n_pred                      # = 2 tp + fp + fn
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(k, dtype=float), where=denom > 0)
    sup = n_true > 0
    return float(f1[sup].mean()) if sup.any() else np.nan


def macro_f1(y_true, y_pred) -> float:
    """Unweighted mean F1 over the classes with support in y_true (NaN when empty). Labels are compared as strings
    (null -> 'NA'); the cost is O(n log n) whatever the number of classes."""
    yt = to_str(y_true); yp = to_str(y_pred)
    if len(yt) == 0:
        return np.nan
    if len(yp) != len(yt):
        raise ValueError(f"{len(yp)} predictions for {len(yt)} labels")
    return _f1_from_codes(*_codes(yt, yp))


def group_folds(groups, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per row such that a group never spans folds; groups go to the currently smallest fold (by rows),
    largest groups first, ties broken by a seeded shuffle. Fewer groups than folds -> one fold per group."""
    g = to_str(groups)
    uniq, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.permutation(len(uniq)), -counts))
    k = min(n_folds, len(uniq))
    load = np.zeros(k, dtype=np.int64)
    fold_of_group = np.zeros(len(uniq), dtype=np.int64)
    for gi in order:
        f = int(np.argmin(load))
        fold_of_group[gi] = f
        load[f] += counts[gi]
    return fold_of_group[inv]


def _majority(values) -> str:
    vals, cnt = np.unique(np.asarray(list(values), dtype=object).astype(str), return_counts=True)
    return str(vals[np.argmax(cnt)])              # ties -> alphabetically first (np.unique sorts)


def majority_vote_cv(keys, y, groups, n_folds: int = 5, seed: int = 0) -> dict:
    """Per-key majority vote: the train folds map every key to its most frequent label (unseen key -> the
    train majority); predictions pooled over folds. Returns macro_f1, accuracy, n, n_groups, n_keys, n_folds."""
    keys = to_str(keys); y = to_str(y)
    folds = group_folds(groups, n_folds, seed)
    pred = np.empty(len(y), dtype=object)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        table = pd.Series(y[tr]).groupby(keys[tr]).agg(_majority).to_dict()
        fallback = _majority(y[tr])
        pred[te] = [table.get(k, fallback) for k in keys[te]]
    return dict(macro_f1=macro_f1(y, pred), accuracy=float(np.mean(pred == y)), n=int(len(y)),
                n_groups=int(len(np.unique(to_str(groups)))), n_keys=int(len(np.unique(keys))), n_folds=int(len(np.unique(folds))))


def nearest_centroid_cv(X, y, groups, n_folds: int = 5, seed: int = 0) -> dict:
    """Nearest class centroid in standardised feature space (train-fold medians impute NaN, train mean / std scale);
    predictions pooled over folds."""
    X = np.asarray(X, float); y = to_str(y)
    folds = group_folds(groups, n_folds, seed)
    pred = np.empty(len(y), dtype=object)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        med = np.nanmedian(X[tr], axis=0); med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(X[tr]), X[tr], med); Xte = np.where(np.isfinite(X[te]), X[te], med)
        mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd == 0] = 1.0
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        labs = sorted(set(y[tr].tolist()))
        cent = np.stack([Xtr[y[tr] == lab].mean(axis=0) for lab in labs])
        d = ((Xte[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        pred[te] = np.array(labs, dtype=object)[np.argmin(d, axis=1)]
    return dict(macro_f1=macro_f1(y, pred), accuracy=float(np.mean(pred == y)), n=int(len(y)),
                n_groups=int(len(np.unique(to_str(groups)))), n_folds=int(len(np.unique(folds))))


def chance_levels(y, n_perm: int = 20, seed: int = 0) -> dict:
    """Macro-F1 of the always-majority predictor and the mean over label permutations (computed on integer codes:
    permuting the codes with the same generator state permutes the labels identically). Depends on the label vector
    only, so the audits compute it once per target and share it across their method rows."""
    y = to_str(y)
    if len(y) == 0:
        return dict(chance_majority_f1=np.nan, chance_permuted_f1=np.nan)
    labs, t = np.unique(y.astype(str), return_inverse=True)
    k = len(labs)
    maj = int(np.argmax(np.bincount(t, minlength=k)))        # ties -> first code = alphabetically first (as _majority)
    rng = np.random.default_rng(seed)
    perm = [_f1_from_codes(t, rng.permutation(t), k) for _ in range(n_perm)]
    return dict(chance_majority_f1=_f1_from_codes(t, np.full(len(t), maj, dtype=t.dtype), k), chance_permuted_f1=float(np.nanmean(perm)))


def joint_key(frame: pd.DataFrame, cols) -> np.ndarray:
    """One categorical key per row: the column values joined with '|' (null -> 'NA')."""
    parts = [to_str(frame[c]) for c in cols]
    return np.array(["|".join(p) for p in zip(*parts)], dtype=object) if parts else np.array([NA] * len(frame), dtype=object)


def quantile_bins(x_train: np.ndarray, x: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Bin index per value using train-fold quantile edges (NaN -> bin -1), as strings."""
    xt = x_train[np.isfinite(x_train)]
    if len(xt) == 0:
        return np.array([NA] * len(x), dtype=object)
    edges = np.unique(np.quantile(xt, np.linspace(0, 1, n_bins + 1)[1:-1]))
    b = np.searchsorted(edges, x, side="right")
    return np.array([NA if not np.isfinite(v) else str(int(k)) for v, k in zip(x, b)], dtype=object)
