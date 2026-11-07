"""Nuisance audit: how predictable are the task labels and the splits from drive metadata that is NOT an official
input (logging device class, software era, region, pose stream, lead source, channel liveness ...).

Per (target, predictor): Cramer's V (association strength, 0..1) and the macro-F1 of a per-value majority-vote
predictor under group-disjoint cross-validation, next to two chance levels (always-majority; label permutation).
Unit of analysis = one drive (row of index/drives.parquet); no hours weighting. A high lift of `ALL` (the joint key
of every nuisance column) over chance means metadata alone separates the classes / the splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vtt.audits import common as CM

NUISANCE_COLS = ("device_type", "software_era", "region", "pose_source", "radar_unavailable", "op_long",
                 "gas_alive", "brake_alive", "torque_alive", "has_pose", "rlog_dec_any")
TARGETS = ("powertrain", "size_class", "split_platform", "split_device")
ALL_KEY = "ALL"
COLUMNS = ["target", "predictor", "n", "n_groups", "n_levels", "cramers_v", "macro_f1", "accuracy",
           "chance_majority_f1", "chance_permuted_f1", "lift_vs_permuted"]


def prepare(drives: pd.DataFrame) -> pd.DataFrame:
    """Copy with the derived nuisance column `rlog_dec_any` (any segment decimated from the high-rate log)."""
    df = drives.copy()
    if "n_seg_rlog_dec" in df:
        df["rlog_dec_any"] = pd.to_numeric(df["n_seg_rlog_dec"], errors="coerce").fillna(0) > 0
    return df


def cramers_v(x, y) -> float:
    """Cramer's V of two categorical arrays (NaN when either has < 2 levels)."""
    tab = pd.crosstab(pd.Series(CM.to_str(x)), pd.Series(CM.to_str(y))).to_numpy(float)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    n = tab.sum()
    exp = np.outer(tab.sum(1), tab.sum(0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(exp > 0, (tab - exp) ** 2 / exp, 0.0))
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def _row(target: str, predictor: str, keys, y, groups, n_folds: int, seed: int, ch: dict | None = None) -> dict:
    """One table row; `ch` = chance_levels(y) shared by every row of the target (computed here when not given)."""
    res = CM.majority_vote_cv(keys, y, groups, n_folds, seed)
    ch = CM.chance_levels(y, seed=seed) if ch is None else ch
    return dict(target=target, predictor=predictor, n=res["n"], n_groups=res["n_groups"], n_levels=res["n_keys"],
                cramers_v=cramers_v(keys, y), macro_f1=res["macro_f1"], accuracy=res["accuracy"],
                chance_majority_f1=ch["chance_majority_f1"], chance_permuted_f1=ch["chance_permuted_f1"],
                lift_vs_permuted=res["macro_f1"] - ch["chance_permuted_f1"])


def audit(drives: pd.DataFrame, predictors=NUISANCE_COLS, targets=TARGETS, group_col: str = "group_id",
          n_folds: int = 5, seed: int = 0, min_rows: int = 20) -> pd.DataFrame:
    """One row per (target, predictor) plus the joint key `ALL`; targets with < 2 classes or < min_rows labelled
    drives are skipped (e.g. the split columns before S8 has run)."""
    df = prepare(drives)
    preds = [c for c in predictors if c in df.columns]
    rows = []
    for target in targets:
        if target not in df.columns:
            continue
        sub = df[df[target].notna()]
        if len(sub) < min_rows or sub[target].nunique() < 2 or not preds:
            continue
        y, groups = sub[target].to_numpy(object), sub[group_col].to_numpy(object)
        ch = CM.chance_levels(y, seed=seed)                      # the label vector is the same for every predictor
        for p in preds:
            rows.append(_row(target, p, sub[p].to_numpy(object), y, groups, n_folds, seed, ch))
        rows.append(_row(target, ALL_KEY, CM.joint_key(sub, preds), y, groups, n_folds, seed, ch))
    return pd.DataFrame(rows, columns=COLUMNS)


def summary_lines(table: pd.DataFrame) -> list[str]:
    """Markdown bullet per target: the joint-key macro-F1 against chance and the strongest single predictor."""
    out = []
    for target, g in table.groupby("target", sort=False):
        allrow = g[g.predictor == ALL_KEY]
        single = g[g.predictor != ALL_KEY].sort_values("macro_f1", ascending=False)
        if allrow.empty or single.empty:
            continue
        a, s = allrow.iloc[0], single.iloc[0]
        out.append(f"* `{target}` from all nuisance columns: macro-F1 {a.macro_f1:.2f} (chance {a.chance_permuted_f1:.2f}, "
                   f"majority {a.chance_majority_f1:.2f}; n = {a.n} drives, {a.n_groups} groups); strongest single column "
                   f"`{s.predictor}` (macro-F1 {s.macro_f1:.2f}, Cramer's V {s.cramers_v:.2f}).")
    return out
