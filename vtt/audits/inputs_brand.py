"""Official-input shortcut audit: how predictable is the BRAND (and the platform) from window summary statistics
of the official inputs, compared with the task labels from the same statistics.

Two small numpy classifiers under group-disjoint cross-validation: nearest centroid on all standardised statistics
and a per-column quantile-binned majority vote (which single statistic carries the most brand information).
Unit of analysis = one eligible window (row of the windows index, subsampled to max_rows with a fixed seed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vtt.audits import common as CM

STAT_COLS = ("v_mean", "v_max", "v_std", "a_min", "a_max", "a_std", "n_accel_ep", "n_decel_ep", "lead_frac", "cf_frac",
             "thw_med", "lead_speed_mean", "yaw_abs_mean", "curv_abs_mean", "steer_pressed_frac", "gas_pressed_frac",
             "brake_pressed_frac")
TARGETS = ("brand", "platform", "powertrain", "size_class")
COLUMNS = ["target", "method", "n", "n_groups", "n_classes", "macro_f1", "accuracy", "chance_majority_f1",
           "chance_permuted_f1", "lift_vs_permuted"]


def select_windows(windows: pd.DataFrame, eligible_only: bool = True, human_only: bool = False, max_rows: int = 200000,
                   seed: int = 0) -> pd.DataFrame:
    """Eligible (and optionally human-only) windows, subsampled to max_rows."""
    df = windows
    if eligible_only and "eligible" in df:
        df = df[df["eligible"].fillna(False).astype(bool)]
    if human_only and "human_only" in df:
        df = df[df["human_only"].fillna(False).astype(bool)]
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    return df.reset_index(drop=True)


def _result(target: str, method: str, res: dict, y, seed: int, ch: dict | None = None) -> dict:
    """One table row; `ch` = chance_levels(y) shared by every method row of the target (computed when not given)."""
    ch = CM.chance_levels(y, seed=seed) if ch is None else ch
    return dict(target=target, method=method, n=res["n"], n_groups=res["n_groups"], n_classes=int(len(set(CM.to_str(y)))),
                macro_f1=res["macro_f1"], accuracy=res["accuracy"], chance_majority_f1=ch["chance_majority_f1"],
                chance_permuted_f1=ch["chance_permuted_f1"], lift_vs_permuted=res["macro_f1"] - ch["chance_permuted_f1"])


def binned_majority_cv(x: np.ndarray, y, groups, n_bins: int = 8, n_folds: int = 5, seed: int = 0) -> dict:
    """Majority vote on the quantile bin of ONE statistic (bins from the train folds)."""
    x = np.asarray(x, float); y = CM.to_str(y)
    folds = CM.group_folds(groups, n_folds, seed)
    pred = np.empty(len(y), dtype=object)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        ktr = CM.quantile_bins(x[tr], x[tr], n_bins); kte = CM.quantile_bins(x[tr], x[te], n_bins)
        table = pd.Series(y[tr]).groupby(ktr).agg(CM._majority).to_dict()
        fallback = CM._majority(y[tr])
        pred[te] = [table.get(k, fallback) for k in kte]
    return dict(macro_f1=CM.macro_f1(y, pred), accuracy=float(np.mean(pred == y)), n=int(len(y)),
                n_groups=int(len(np.unique(CM.to_str(groups)))))


def audit(windows: pd.DataFrame, stat_cols=STAT_COLS, targets=TARGETS, group_col: str = "group_id", n_folds: int = 5,
          seed: int = 0, n_bins: int = 8, max_rows: int = 200000, min_rows: int = 20) -> pd.DataFrame:
    """Rows: per target, `nearest_centroid:all` and `binned_majority:<stat>` for every statistic present."""
    df = select_windows(windows, max_rows=max_rows, seed=seed)
    cols = [c for c in stat_cols if c in df.columns]
    rows = []
    for target in targets:
        if target not in df.columns or not cols:
            continue
        sub = df[df[target].notna()]
        if len(sub) < min_rows or sub[target].nunique() < 2:
            continue
        y, groups = sub[target].to_numpy(object), sub[group_col].to_numpy(object)
        X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        ch = CM.chance_levels(y, seed=seed)                      # the label vector is the same for every method
        rows.append(_result(target, "nearest_centroid:all", CM.nearest_centroid_cv(X, y, groups, n_folds, seed), y, seed, ch))
        for j, c in enumerate(cols):
            rows.append(_result(target, f"binned_majority:{c}", binned_majority_cv(X[:, j], y, groups, n_bins, n_folds, seed), y, seed, ch))
    return pd.DataFrame(rows, columns=COLUMNS)


def summary_lines(table: pd.DataFrame) -> list[str]:
    """Markdown bullet per target: nearest-centroid macro-F1 against chance and the most informative statistic."""
    out = []
    for target, g in table.groupby("target", sort=False):
        nc = g[g.method == "nearest_centroid:all"]
        best = g[g.method.str.startswith("binned_majority:")].sort_values("macro_f1", ascending=False)
        if nc.empty:
            continue
        r = nc.iloc[0]
        line = (f"* `{target}` from {len(g) - 1} window statistics (nearest centroid): macro-F1 {r.macro_f1:.2f} "
                f"(chance {r.chance_permuted_f1:.2f}, majority {r.chance_majority_f1:.2f}; {r.n_classes} classes, n = {r.n} windows, "
                f"{r.n_groups} groups)")
        if not best.empty:
            b = best.iloc[0]
            line += f"; most informative single statistic `{b.method.split(':', 1)[1]}` (macro-F1 {b.macro_f1:.2f})"
        out.append(line + ".")
    return out
