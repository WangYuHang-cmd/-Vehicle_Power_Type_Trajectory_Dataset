#!/usr/bin/env python3
"""Aggregation study: from per-window class probabilities to one prediction per drive or per vehicle.

Reads out-of-fold window probabilities of the recipe runs (<data-root>/experiments/recipe/results/<tag>_oof.npz with
their <tag>.json, written by experiments.train fold runs) or, when none exist yet, the saved test-set probabilities of
the published models (<data-root>/experiments/results/<tag>_probs.npz) as a post-hoc demonstration, plus the windows
index (build/index/windows.parquet: unit keys and per-window statistics). Compares aggregation rules (probability
mean, log-odds mean, per-class median, majority vote, footprint mean), informativeness weights
((1 - das_any_frac + 0.1)^a (1 + n_accel_ep + n_decel_ep)^b (steer_pressed_frac + 0.05)^c for a, b, c in {0, 0.5, 1}),
a temperature, a correctness gate and stacking of several runs. Vehicle levels: every rule on the vehicle's pooled
windows (`mean / uniform` = the protocol `vehicle_full`, the reference) and, as a candidate, in two stages (suffix
`>drives`: log-odds mean of the drive decisions weighted by n_windows^0.5). Only runs that are peers are compared: on
the out-of-fold path the pooled fold runs of a configuration (probes and single-fold sweep runs are left out unless
asked for), and stacking members must share their windows. Writes experiments/AGGREGATION.md and CSV tables under
<data-root>/experiments/aggregation/. numpy / pandas / scikit-learn only (no torch); prints counts and metrics only.

Everything fitted here (temperature, gate, stacker) is fitted out of sample: by fold when the fold runs of one
configuration are pooled, otherwise leave-one-car-model-out inside the evaluated windows.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402
from starter_kit.eval import TASKS  # noqa: E402   (numpy / pandas only; experiments.data would import torch)

REPO = Path(__file__).resolve().parents[1]
EXP_ROOT = Path(C.DATA_ROOT) / "experiments"
EPS = 1e-6
METHODS = ("mean", "logodds", "median", "vote", "fp_mean")
BASE_HOW = {"mean_T": "mean", "gate_mean": "mean", "gate_logodds": "logodds"}
DRIVES_SUFFIX = ">drives"                      # two-stage vehicle rule: <drive rule> then log-odds mean over the drives
LEVELS = ("drive", "vehicle", "vehicle_multi")
PROTOCOL_NAMES = ("full", "human_only")
UNIFORM = "uniform"
VEHICLE_POWER = 0.5
MIN_COVER = 0.5                                # stacking: least share of a member's windows the common window set must cover
PROBE_RE = re.compile(r"^L\d+(_|$)")            # leave-one-car-model-out probe recipe ids (experiments/run_recipe.py)
GATE_FEATURES = ("das_any_frac", "n_accel_ep", "n_decel_ep", "n_lead_accel_events", "steer_pressed_frac", "gas_pressed_frac",
                 "brake_pressed_frac", "moving_frac", "v_mean", "v_std", "a_std", "lead_frac", "cf_frac", "max_prob", "entropy")
INDEX_COLUMNS = ("window_id", "drive_id", "vehicle_id", "model", "human_only_strict", "das_any_frac", "n_accel_ep", "n_decel_ep",
                 "n_lead_accel_events", "steer_pressed_frac", "gas_pressed_frac", "brake_pressed_frac", "moving_frac", "v_mean",
                 "v_std", "a_std", "lead_frac", "cf_frac")
DEEP_MODELS = ("gru", "resnet", "transformer", "mamba2")
RENDER_MARK = "<!-- rendered by experiments/aggregate.py: everything below this line is overwritten -->"

HEADER = """# Aggregation study: from window probabilities to drive- and vehicle-level decisions

How the per-window class probabilities of a trained window model are turned into one decision per drive (`drive_full`,
`drive_human_only`) and per vehicle (`vehicle_full`, `vehicle_full_multi`), see docs/PROTOCOLS.md for the unit
protocols. Everything is computed from saved window probabilities by `experiments/aggregate.py`; no model is trained here.

Rules compared (per unit; ties go to the earlier class in task order):

- `mean`: weighted mean of the probability vectors (the published `drive_level` rule with uniform weights);
- `logodds`: weighted mean of log probabilities, mapped back through a softmax (geometric mean, insensitive to a
  temperature);
- `median`: per-class (weighted) median, renormalised; `vote`: weighted share of per-window argmax votes;
- `mean_T`: mean of temperature-scaled probabilities, the temperature fitted by negative log-likelihood out of sample;
- `gate_mean` / `gate_logodds`: mean / log-odds mean with weights = out-of-sample probability that the window is
  classified correctly (logistic regression on window statistics: automation share, episode and lead-event counts,
  pedal / steering shares, speed and acceleration statistics, confidence and entropy of the window);
- `fp_mean` (size task, runs that carry a footprint estimate): mean footprint over the windows, cut at the class
  thresholds.

Weights: `das{a}_ep{b}_steer{c}` = (1 - das_any_frac + 0.1)^a (1 + n_accel_ep + n_decel_ep)^b (steer_pressed_frac + 0.05)^c
with a, b, c in {0, 0.5, 1}; `uniform` is (0, 0, 0).

Vehicle levels (`vehicle`, `vehicle_multi`): every rule is applied in two ways. Applied to the pooled windows of the
vehicle (method name unchanged), `mean / uniform` is exactly the protocol `vehicle_full` / `vehicle_full_multi` of
docs/PROTOCOLS.md (all the vehicle's evaluated windows with the same weight) and is the reference the deltas are
measured against. Applied in two stages (method name with the suffix `>drives`), the rule's drive decisions are
combined by a log-odds mean weighted by n_windows^0.5 (the candidate rule of the plan: a vehicle with many short drives
is not dominated by its single longest one); it is a candidate, not the protocol. `vehicle_multi` keeps the vehicles
with at least two drives among the evaluated windows. The vehicle levels under protocol `human_only` are exploratory
(no published protocol pools human-only windows per vehicle). Stacking: multinomial logistic regression on the centred
log probabilities of several runs, fitted out of sample.

Which runs are compared: on the out-of-fold path only the pooled fold runs of a configuration (all folds of one task x
model x recipe x seed, kind `oof_pooled`) are tabulated and stacked; single-fold sweep runs, the leave-one-car-model-out
probes (recipe ids `L<k>_...`, whose windows are one car model and disjoint from the fold windows) and other kinds are
left out unless asked for (`--kinds`, `--recipes`, `--include-probes`). Stacking members must share their windows: a
member whose windows are covered by less than half by the common window set is dropped (`--min-cover`).

Out-of-sample means: by fold when the fold runs of one configuration are pooled, otherwise leave-one-car-model-out
inside the evaluated windows (the temperature, gate or stacker applied to a car model never saw that model). Numbers in
a section marked post-hoc come from the published models' test-set predictions: they show that the code works and the
size of the effects, but choosing a rule from them would be optimistic; the rule used for the final evaluation is
chosen on the internal folds only (experiments/CAMPAIGN.md).
"""


# ----------------------------------------------------------------------------- metrics (the definitions of experiments.train, torch-free)
def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
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


def metrics(y_true, y_pred, classes) -> dict:
    """The dict of experiments.train.metrics (n, macro_f1, balanced_acc, accuracy, per_class_recall, support, confusion,
    classes), computed here so that nothing in this module imports torch."""
    classes = tuple(classes)
    k = len(classes)
    y_true = np.asarray(y_true, dtype=np.int64); y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion(y_true, y_pred, k)
    return dict(n=int(len(y_true)), macro_f1=macro_f1(y_true, y_pred, k), balanced_acc=balanced_accuracy(y_true, y_pred, k),
                accuracy=float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
                per_class_recall={c: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None) for i, c in enumerate(classes)},
                support={c: int(cm[i].sum()) for i, c in enumerate(classes)}, confusion=cm.tolist(), classes=list(classes))


# ----------------------------------------------------------------------------- small helpers
def softmax_rows(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    L = L - L.max(axis=1, keepdims=True)
    e = np.exp(L)
    return e / e.sum(axis=1, keepdims=True)


def _group_codes(keys) -> tuple[np.ndarray, np.ndarray]:
    """Integer code per row and the sorted unique keys (the order pandas groupby uses)."""
    codes, units = pd.factorize(pd.Series(np.asarray(keys, dtype=object)), sort=True)
    if (codes < 0).any():
        raise ValueError("unit keys must not be missing")
    return codes.astype(np.int64), np.asarray(units, dtype=object)


def _group_sum(codes: np.ndarray, n_units: int, values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return np.bincount(codes, weights=values, minlength=n_units)
    return np.stack([np.bincount(codes, weights=values[:, c], minlength=n_units) for c in range(values.shape[1])], axis=1)


def _weighted_median(values: np.ndarray, w: np.ndarray) -> float:
    """Weighted median; when the cumulative weight hits exactly half at a value, the midpoint to the next value is
    returned (uniform weights then reproduce numpy's median)."""
    order = np.argsort(values, kind="stable")
    v, cw = values[order], np.cumsum(w[order])
    half = 0.5 * cw[-1]
    i = min(int(np.searchsorted(cw, half)), len(v) - 1)
    if i + 1 < len(v) and abs(cw[i] - half) <= 1e-12 * max(cw[-1], 1.0):
        return float(0.5 * (v[i] + v[i + 1]))
    return float(v[i])


# ----------------------------------------------------------------------------- aggregation rules
def aggregate(P, keys, how: str = "mean", w=None, thresholds=None) -> tuple[np.ndarray, np.ndarray]:
    """One probability vector per unit.

    P: (n, k) window probabilities (for `fp_mean`: (n,) or (n, 1) footprint estimates in m^2); keys: unit id per window;
    how: mean | logodds | median | vote | fp_mean; w: non-negative weight per window (None = uniform). Returns the sorted
    unique units and an (n_units, k) array whose argmax is the unit decision (`fp_mean`: one-hot of the class the mean
    footprint falls into, k = len(thresholds) + 1). A unit whose windows all have weight 0 falls back to uniform weights.
    """
    if how not in METHODS:
        raise ValueError(f"unknown aggregation {how!r}; choose from {METHODS}")
    P = np.asarray(P, dtype=np.float64)
    codes, units = _group_codes(keys)
    n_units = len(units)
    if len(codes) != len(P):
        raise ValueError(f"{len(codes)} keys for {len(P)} probability rows")
    w = np.ones(len(P)) if w is None else np.asarray(w, dtype=np.float64)
    if w.shape != (len(P),) or not np.all(np.isfinite(w)) or (w < 0).any():
        raise ValueError("weights must be finite, non-negative and one per window")
    den = _group_sum(codes, n_units, w)
    empty = den <= 0
    if empty.any():
        w = np.where(empty[codes], 1.0, w)
        den = _group_sum(codes, n_units, w)
    if how == "fp_mean":
        fp = P.reshape(len(P), -1)
        if fp.shape[1] != 1:
            raise ValueError("fp_mean expects one footprint estimate per window")
        thr = tuple(thresholds if thresholds is not None else C.FOOTPRINT_THRESHOLDS_M2)
        mean_fp = _group_sum(codes, n_units, w * fp[:, 0]) / den
        agg = np.zeros((n_units, len(thr) + 1))
        agg[np.arange(n_units), np.digitize(mean_fp, thr)] = 1.0
        return units, agg
    if P.ndim != 2:
        raise ValueError("P must be (n, k)")
    k = P.shape[1]
    if how == "mean":
        agg = _group_sum(codes, n_units, P * w[:, None]) / den[:, None]
    elif how == "logodds":
        L = np.log(np.clip(P, EPS, 1.0))
        agg = softmax_rows(_group_sum(codes, n_units, L * w[:, None]) / den[:, None])
    elif how == "vote":
        agg = np.zeros((n_units, k))
        np.add.at(agg, (codes, P.argmax(axis=1)), w)
        agg /= den[:, None]
    else:                                                   # median
        agg = np.zeros((n_units, k))
        order = np.argsort(codes, kind="stable")
        bounds = np.searchsorted(codes[order], np.arange(n_units + 1))
        for u in range(n_units):
            sel = order[bounds[u]:bounds[u + 1]]
            for c in range(k):
                agg[u, c] = _weighted_median(P[sel, c], w[sel])
        s = agg.sum(axis=1, keepdims=True)
        agg = np.where(s > 0, agg / np.where(s > 0, s, 1.0), 1.0 / k)
    return units, agg


def unit_labels(keys, y) -> tuple[np.ndarray, np.ndarray]:
    """(sorted units, label per unit) = the first label of each unit; raises when a unit carries several labels."""
    codes, units = _group_codes(keys)
    g = pd.DataFrame({"c": codes, "y": np.asarray(y)}).groupby("c")["y"]
    if (g.nunique() > 1).any():
        raise ValueError("a unit carries more than one label")
    return units, g.first().to_numpy()


def unit_metrics(units: np.ndarray, y_units: np.ndarray, agg: np.ndarray, classes) -> dict:
    """metrics() of the unit decisions (argmax of `agg`) plus n_units."""
    if len(units) != len(agg) or len(y_units) != len(agg):
        raise ValueError("units, labels and aggregated probabilities must have one row per unit")
    m = metrics(y_units, agg.argmax(axis=1), classes)
    m["n_units"] = int(len(units))
    return m


def weight_grid(meta: pd.DataFrame, exponents=(0.0, 0.5, 1.0)) -> dict[str, np.ndarray]:
    """Named informativeness weights (1 - das_any_frac + 0.1)^a (1 + n_accel_ep + n_decel_ep)^b (steer_pressed_frac + 0.05)^c
    for every (a, b, c) in exponents^3; the all-zero exponent is called `uniform`. Missing statistics count as 0."""
    col = lambda c: pd.to_numeric(meta[c], errors="coerce").fillna(0.0).to_numpy(np.float64) if c in meta else np.zeros(len(meta))  # noqa: E731
    das = 1.0 - col("das_any_frac") + 0.1
    ep = 1.0 + col("n_accel_ep") + col("n_decel_ep")
    st = col("steer_pressed_frac") + 0.05
    out = {}
    for a in exponents:
        for b in exponents:
            for c in exponents:
                name = UNIFORM if a == b == c == 0 else f"das{a:g}_ep{b:g}_steer{c:g}"
                out[name] = das ** a * ep ** b * st ** c
    return out


def vehicle_level(drive_units: np.ndarray, drive_agg: np.ndarray, drive_vehicle: dict, drive_n_windows: dict,
                  multi: bool = False, power: float = VEHICLE_POWER) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-stage vehicle rule (the plan's candidate, not the protocol): log-odds mean of the drives' aggregated
    probabilities weighted by n_windows^power. Returns (vehicles, agg, mask of the drives kept); `multi` keeps only
    vehicles with >= 2 drives."""
    veh = np.array([drive_vehicle[d] for d in drive_units], dtype=object)
    nw = np.array([drive_n_windows[d] for d in drive_units], dtype=np.float64)
    keep = np.ones(len(veh), dtype=bool)
    if multi:
        counts = pd.Series(veh).value_counts()
        keep = np.array([counts[v] >= 2 for v in veh], dtype=bool)
    if not keep.any():
        return np.array([], dtype=object), np.zeros((0, drive_agg.shape[1])), keep
    units, agg = aggregate(drive_agg[keep], veh[keep], "logodds", nw[keep] ** power)
    return units, agg, keep


# ----------------------------------------------------------------------------- temperature
def fit_temperature(P, y, log_bounds=(math.log(0.05), math.log(20.0)), iters: int = 50) -> float:
    """Temperature T minimising the negative log-likelihood of softmax(log P / T) (golden-section search on log T)."""
    L = np.log(np.clip(np.asarray(P, dtype=np.float64), EPS, 1.0))
    y = np.asarray(y, dtype=np.int64)
    rows = np.arange(len(y))

    def nll(log_t: float) -> float:
        Z = L / math.exp(log_t)
        m = Z.max(axis=1)
        return float(np.mean(np.log(np.exp(Z - m[:, None]).sum(axis=1)) + m - Z[rows, y]))

    a, b = log_bounds
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = nll(d)
    return math.exp(0.5 * (a + b))


def apply_temperature(P, T: float) -> np.ndarray:
    return softmax_rows(np.log(np.clip(np.asarray(P, dtype=np.float64), EPS, 1.0)) / T)


# ----------------------------------------------------------------------------- correctness gate
def window_stats(P, meta: pd.DataFrame) -> pd.DataFrame:
    """Gate features per window: the index statistics present in `meta` (row-aligned with P) plus max_prob and entropy."""
    P = np.asarray(P, dtype=np.float64)
    if len(meta) != len(P):
        raise ValueError(f"{len(meta)} statistics rows for {len(P)} probability rows")
    df = pd.DataFrame(index=np.arange(len(P)))
    for c in GATE_FEATURES:
        if c in meta.columns:
            df[c] = pd.to_numeric(meta[c].to_numpy(), errors="coerce").astype(np.float64)
    Pc = np.clip(P, EPS, 1.0)
    df["max_prob"] = P.max(axis=1)
    df["entropy"] = -(Pc * np.log(Pc)).sum(axis=1)
    return df


def fit_gate(oof_df: pd.DataFrame, features=None, C_reg: float = 1.0, seed: int = 0) -> dict:
    """Logistic regression predicting window correctness (`correct` column) from the window statistics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    feats = [f for f in (features or GATE_FEATURES) if f in oof_df.columns]
    if not feats:
        raise ValueError("no gate feature column present")
    yc = oof_df["correct"].to_numpy(bool)
    if yc.all() or (~yc).all() or len(yc) < 2:
        return dict(model=None, scaler=None, features=feats, constant=float(yc.mean()) if len(yc) else 0.5)
    X = oof_df[feats].to_numpy(np.float64)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C_reg, max_iter=1000, random_state=seed).fit(np.nan_to_num(sc.transform(X)), yc.astype(int))
    return dict(model=clf, scaler=sc, features=feats, constant=None)


def apply_gate(gate: dict, df: pd.DataFrame) -> np.ndarray:
    """Probability that each window is classified correctly (used as an aggregation weight)."""
    if gate["model"] is None:
        return np.full(len(df), gate["constant"], dtype=np.float64)
    X = df[gate["features"]].to_numpy(np.float64)
    return gate["model"].predict_proba(np.nan_to_num(gate["scaler"].transform(X)))[:, 1]


# ----------------------------------------------------------------------------- stacking
def member_features(P) -> np.ndarray:
    """Centred log probabilities (k columns per member)."""
    L = np.log(np.clip(np.asarray(P, dtype=np.float64), EPS, 1.0))
    return L - L.mean(axis=1, keepdims=True)


def align_runs(oof_runs: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Common window ids (sorted) of several (window_id, probs) members and every member's probabilities in that order."""
    common = None
    for wid, _ in oof_runs.values():
        s = set(np.asarray(wid).astype(str).tolist())
        common = s if common is None else common & s
    ids = np.array(sorted(common or ()), dtype=object)
    out = {}
    for name, (wid, P) in oof_runs.items():
        pos = pd.Series(np.arange(len(wid)), index=np.asarray(wid).astype(str))
        out[name] = np.asarray(P, dtype=np.float64)[pos.reindex(ids).to_numpy(np.int64)]
    return ids, out


def select_members(window_ids: dict, min_cover: float = MIN_COVER) -> tuple[list[str], list[str], np.ndarray]:
    """Members whose window sets overlap enough to be stacked. The common window set must cover at least `min_cover` of
    every kept member's windows; until it does, the member whose removal enlarges the common set the most (ties: the
    member the common set covers least, then fewer windows, then name order) is dropped and the common set recomputed.
    window_ids: name -> window ids. Returns (kept names in input order, dropped names, sorted common window ids); a probe
    run (one car model, disjoint from the fold windows) or a single-fold run among pooled runs is dropped this way."""
    sets = {n: set(np.asarray(w).astype(str).tolist()) for n, w in window_ids.items()}
    kept, dropped = list(window_ids), []
    common: set = set()
    while kept:
        common = set.intersection(*(sets[n] for n in kept))
        cover = {n: (len(common) / len(sets[n]) if sets[n] else 0.0) for n in kept}
        if all(c >= min_cover for c in cover.values()):
            break
        gain = {n: (len(set.intersection(*(sets[m] for m in kept if m != n))) if len(kept) > 1 else 0) for n in kept}
        worst = max(sorted(kept), key=lambda n: (gain[n], -cover[n], -len(sets[n])))
        kept.remove(worst)
        dropped.append(worst)
    return kept, dropped, np.array(sorted(common) if kept else [], dtype=object)


def _stack_matrix(ids: np.ndarray, members: dict, names: list[str], stats: pd.DataFrame | None, stat_cols: list[str]) -> np.ndarray:
    X = np.hstack([member_features(members[n]) for n in names])
    if stat_cols:
        X = np.hstack([X, stats.reindex(ids)[stat_cols].to_numpy(np.float64)])
    return X


def stack(oof_runs: dict, y, stats: pd.DataFrame | None = None, n_classes: int | None = None, C_reg: float = 1.0,
          seed: int = 0, max_iter: int = 1000) -> dict:
    """Multinomial logistic regression on the members' centred log probabilities (+ optional window statistics indexed by
    window_id). oof_runs: name -> (window_id, probs); y: labels indexed by window_id (Series or dict)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ids, members = align_runs(oof_runs)
    names = list(oof_runs)
    k = n_classes or int(next(iter(members.values())).shape[1])
    stat_cols = list(stats.columns) if stats is not None else []
    X = _stack_matrix(ids, members, names, stats, stat_cols)
    yv = pd.Series(y).reindex(ids).to_numpy()
    if pd.isna(yv).any():
        raise ValueError("labels missing for some common windows")
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C_reg, max_iter=max_iter, random_state=seed).fit(np.nan_to_num(sc.transform(X)), yv.astype(int))
    return dict(model=clf, scaler=sc, members=names, stat_cols=stat_cols, n_classes=k, n_fit=int(len(ids)))


def predict_stack(stacker: dict, runs: dict, stats: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(window_id, (n, n_classes) probabilities) of the stacker on the common windows of `runs` (classes absent at fit
    time get probability 0)."""
    ids, members = align_runs({n: runs[n] for n in stacker["members"]})
    X = _stack_matrix(ids, members, stacker["members"], stats, stacker["stat_cols"])
    p = stacker["model"].predict_proba(np.nan_to_num(stacker["scaler"].transform(X)))
    out = np.zeros((len(ids), stacker["n_classes"]))
    out[:, stacker["model"].classes_.astype(int)] = p
    return ids, out


# ----------------------------------------------------------------------------- cross-fitting
def crossfit(groups, fit, predict) -> np.ndarray:
    """Leave-one-group-out: for every group, `fit(train_mask)` on the other groups and `predict(model, test_mask)` on it."""
    groups = np.asarray(groups, dtype=object)
    uniq = pd.unique(groups)
    if len(uniq) < 2:
        raise ValueError("cross-fitting needs at least two groups")
    out = None
    for g in uniq:
        te = groups == g
        pred = np.asarray(predict(fit(~te), te), dtype=np.float64)
        if out is None:
            out = np.zeros((len(groups),) + pred.shape[1:])
        out[te] = pred
    return out


def crossfit_temperature(P, y, groups) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    return crossfit(groups, lambda tr: fit_temperature(P[tr], y[tr]), lambda T, te: apply_temperature(P[te], T))


def crossfit_gate(P, y, meta: pd.DataFrame, groups) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    st = window_stats(P, meta)
    correct = P.argmax(axis=1) == np.asarray(y, dtype=np.int64)
    return crossfit(groups, lambda tr: fit_gate(st[tr].assign(correct=correct[tr])), lambda g, te: apply_gate(g, st[te]))


def crossfit_stack(members: dict, y: pd.Series, groups: pd.Series, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Stacked probabilities on the members' common windows, fitted leave-one-group-out (groups indexed by window_id)."""
    ids, aligned = align_runs(members)
    g = groups.reindex(ids).to_numpy(object)

    def fit(tr):
        return stack({n: (ids[tr], aligned[n][tr]) for n in members}, y, n_classes=n_classes)

    def predict(stk, te):
        return predict_stack(stk, {n: (ids[te], aligned[n][te]) for n in members})[1]

    return ids, crossfit(g, fit, predict)


# ----------------------------------------------------------------------------- runs and index
def load_windows_index(data_root: Path | str | None = None) -> pd.DataFrame:
    """The windows index columns needed here, indexed by window_id."""
    root = Path(data_root or C.DATA_ROOT)
    win = pd.read_parquet(root / "build" / "index" / "windows.parquet", columns=list(INDEX_COLUMNS))
    return win.rename(columns={"model": "car_model"}).set_index("window_id")


def load_runs(oof_dir: Path, results_dir: Path | None = None) -> list[dict]:
    """Fold runs (<tag>_oof.npz + <tag>.json, kind `oof`) or, when there are none, the published test predictions
    (<tag>_probs.npz, kind `test_posthoc`). Each run: tag, task, split_col, model (the architecture: `arch` of the JSON,
    else `model`), seed, fold, recipe_id, kind, window_id, probs, y and, when the npz carries the footprint head
    (`fp_mu` = predicted natural-log footprint in m^2, de-standardised by experiments.train.predict, contract 5), fp_m2 =
    exp(fp_mu) with non-finite values as NaN."""
    def read(path: Path, suffix: str, kind: str) -> dict | None:
        tag = path.name[: -len(suffix)]
        meta = path.with_name(f"{tag}.json")
        if not meta.exists():
            return None
        r = json.loads(meta.read_text())
        z = np.load(path, allow_pickle=True)
        run = dict(tag=tag, task=r["task"], split_col=r["split_col"], model=str(r.get("arch") or r["model"]), seed=r.get("seed", 0), fold=r.get("fold"),
                   recipe_id=str(r.get("recipe_id") or ""), kind=kind, window_id=z["window_id"].astype(str), probs=np.asarray(z["probs"], np.float64),
                   y=np.asarray(z["y"], np.int64))
        if "fp_mu" in z.files:
            with np.errstate(over="ignore", invalid="ignore"):
                fp = np.exp(np.asarray(z["fp_mu"], np.float64).reshape(-1))
            run["fp_m2"] = np.where(np.isfinite(fp), fp, np.nan)
        return run

    runs = [r for r in (read(p, "_oof.npz", "oof") for p in sorted(Path(oof_dir).glob("*_oof.npz"))) if r]
    if runs or results_dir is None:
        return runs
    return [r for r in (read(p, "_probs.npz", "test_posthoc") for p in sorted(Path(results_dir).glob("*_probs.npz"))) if r]


def pool_folds(runs: list[dict]) -> list[dict]:
    """Concatenate the fold runs of one (task, split_col, model, recipe_id, seed) into a pooled run whose `groups` are the
    fold ids (cross-fitting by fold); runs without several folds are returned unchanged."""
    by_cfg: dict[tuple, list[dict]] = {}
    out = []
    for r in runs:
        if r["kind"] == "oof" and r.get("fold") is not None:
            by_cfg.setdefault((r["task"], r["split_col"], r["model"], r["recipe_id"], r["seed"]), []).append(r)
        else:
            out.append(r)
    for key, rs in by_cfg.items():
        if len(rs) < 2:
            out.extend(rs)
            continue
        rs = sorted(rs, key=lambda r: r["fold"])
        task, split_col, model, rid, seed = key
        pooled = dict(tag=f"pooled__{task}__{model}__{rid}__s{seed}", task=task, split_col=split_col, model=model, seed=seed, fold=None,
                      recipe_id=rid, kind="oof_pooled", window_id=np.concatenate([r["window_id"] for r in rs]),
                      probs=np.concatenate([r["probs"] for r in rs]), y=np.concatenate([r["y"] for r in rs]),
                      groups=np.concatenate([np.full(len(r["y"]), f"fold{r['fold']}", dtype=object) for r in rs]), n_folds=len(rs))
        if all("fp_m2" in r for r in rs):
            pooled["fp_m2"] = np.concatenate([r["fp_m2"] for r in rs])
        out.append(pooled)
    return out


def is_probe(recipe_id) -> bool:
    """Leave-one-car-model-out probe run (recipe id `L<k>` or `L<k>_<base>`, experiments/run_recipe.py)."""
    return bool(PROBE_RE.match(str(recipe_id or "")))


def select_runs(runs: list[dict], kinds=None, recipes=None, models=None, include_probes: bool = False,
                fold: int | None = None) -> tuple[list[dict], dict[str, int]]:
    """The runs compared as peers. kinds: run kinds kept (default: `oof_pooled` when any exists, else the `oof` runs of
    the most frequent fold, else every kind present); recipes: recipe ids kept (default all); models: architecture names
    kept (default all); fold: keep one fold of single-fold runs; the probes (is_probe) are dropped unless include_probes.
    Returns (kept runs, number of dropped runs by reason)."""
    present = {r["kind"] for r in runs}
    if kinds is None:
        if "oof_pooled" in present:
            kinds = ("oof_pooled",)
        elif "oof" in present:
            kinds = ("oof",)
            if fold is None:
                folds = [r.get("fold") for r in runs if r["kind"] == "oof" and (include_probes or not is_probe(r.get("recipe_id")))]
                fold = Counter(folds).most_common(1)[0][0] if folds else None
        else:
            kinds = tuple(sorted(present))
    kinds, recipes, models = set(kinds), (None if recipes is None else set(recipes)), (None if models is None else set(models))
    kept, dropped = [], Counter()
    for r in runs:
        if r["kind"] not in kinds:
            dropped["kind"] += 1
        elif not include_probes and is_probe(r.get("recipe_id")):
            dropped["probe"] += 1
        elif recipes is not None and str(r.get("recipe_id") or "") not in recipes:
            dropped["recipe"] += 1
        elif models is not None and r["model"] not in models:
            dropped["model"] += 1
        elif fold is not None and r.get("fold") is not None and r.get("fold") != fold:
            dropped["fold"] += 1
        else:
            kept.append(r)
    return kept, dict(dropped)


def join_run(run: dict, windows_index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    """(probs, y, per-window metadata, n_missing) of a run aligned with the windows index (unknown ids are dropped)."""
    idx = windows_index if windows_index.index.name == "window_id" else windows_index.set_index("window_id")
    if "car_model" not in idx.columns and "model" in idx.columns:
        idx = idx.rename(columns={"model": "car_model"})
    meta = idx.reindex(np.asarray(run["window_id"]).astype(str))
    ok = meta["drive_id"].notna().to_numpy()
    meta = meta[ok].reset_index()
    meta["groups"] = (np.asarray(run["groups"], dtype=object)[ok] if run.get("groups") is not None else meta["car_model"].to_numpy(object))
    if "fp_m2" in run:
        meta["fp_m2"] = np.asarray(run["fp_m2"], np.float64)[ok]
    return np.asarray(run["probs"], np.float64)[ok], np.asarray(run["y"], np.int64)[ok], meta, int((~ok).sum())


# ----------------------------------------------------------------------------- evaluation
def _candidates(P: np.ndarray, y: np.ndarray, meta: pd.DataFrame, exponents, log=print) -> list[tuple[str, str, np.ndarray, np.ndarray | None]]:
    """(method, weights name, probabilities, weights) for every rule evaluated on one run."""
    cands: list[tuple[str, str, np.ndarray, np.ndarray | None]] = []
    for name, w in weight_grid(meta, exponents).items():
        cands.append(("mean", name, P, w))
        cands.append(("logodds", name, P, w))
    cands += [("median", UNIFORM, P, None), ("vote", UNIFORM, P, None)]
    groups = meta["groups"].to_numpy(object)
    try:
        cands.append(("mean_T", UNIFORM, crossfit_temperature(P, y, groups), None))
        wg = crossfit_gate(P, y, meta, groups)
        cands += [("gate_mean", "gate", P, wg), ("gate_logodds", "gate", P, wg)]
    except ValueError as e:                                  # a single group: nothing to cross-fit on
        log(f"cross-fitting skipped ({e})")
    if "fp_m2" in meta.columns:
        fp = meta["fp_m2"].to_numpy(np.float64)
        ok = np.isfinite(fp)
        if ok.any():
            if not ok.all():
                log(f"fp_mean: {int((~ok).sum())} windows without a finite footprint estimate get weight 0")
            cands.append(("fp_mean", UNIFORM, np.where(ok, fp, 0.0), None if ok.all() else ok.astype(np.float64)))
    return cands


def evaluate_aggregations(runs: list[dict], windows_index: pd.DataFrame, task: str, split_col: str, level="drive",
                          protocol: str = "full", exponents=(0.0, 0.5, 1.0), log=print) -> pd.DataFrame:
    """One row per run x level x rule x weights with macro_f1, balanced_acc, accuracy, n_units, n_windows, n_models.

    level: drive (`drive_full`: one decision per drive), vehicle (`vehicle_full`) or vehicle_multi (`vehicle_full_multi`:
    vehicles with >= 2 drives among the evaluated windows), one name or a sequence of names (the cross-fitted rules are
    then computed once per run). At the vehicle levels every rule appears twice: applied to the vehicle's pooled windows
    (method name unchanged; `mean / uniform` is then exactly the protocol rule of docs/PROTOCOLS.md) and in two stages
    (method name + `>drives`: the rule's drive decisions combined by vehicle_level(), the plan's candidate; not for
    `fp_mean`). protocol: full (every window of the run) or human_only (human_only_strict windows).
    """
    levels = (level,) if isinstance(level, str) else tuple(level)
    for lv in levels:
        if lv not in LEVELS:
            raise ValueError(f"unknown level {lv!r}; choose from {LEVELS}")
    if protocol not in PROTOCOL_NAMES:
        raise ValueError(f"unknown protocol {protocol!r}; choose from {PROTOCOL_NAMES}")
    classes = TASKS[task]["classes"]
    rows = []
    for run in runs:
        if run["task"] != task or run["split_col"] != split_col:
            continue
        P, y, meta, n_missing = join_run(run, windows_index)
        if n_missing:
            log(f"[{run['tag']}] {n_missing} window ids not in the index were dropped")
        if protocol == "human_only":
            keep = meta["human_only_strict"].fillna(False).to_numpy(bool)
            P, y, meta = P[keep], y[keep], meta[keep].reset_index(drop=True)
        if len(P) == 0:
            continue
        drive_keys = meta["drive_id"].to_numpy(object)
        vehicle_keys = meta["vehicle_id"].to_numpy(object)
        drive_units, y_drive = unit_labels(drive_keys, y)
        drive_vehicle = dict(zip(meta["drive_id"], meta["vehicle_id"]))
        drive_n = meta.groupby("drive_id").size().to_dict()
        n_drives_veh = meta.groupby("vehicle_id")["drive_id"].nunique()
        veh_sel = {"vehicle": np.ones(len(meta), dtype=bool), "vehicle_multi": (meta["vehicle_id"].map(n_drives_veh).to_numpy() >= 2)}
        veh_truth = {lv: unit_labels(vehicle_keys[s], y[s]) for lv, s in veh_sel.items() if s.any()}
        base = dict(tag=run["tag"], model=run["model"], seed=run["seed"], kind=run["kind"], fold=run.get("fold"), recipe_id=run.get("recipe_id", ""),
                    task=task, split_col=split_col, protocol=protocol, n_windows=int(len(P)), n_models=int(meta["car_model"].nunique()))

        def add(lv: str, method: str, wname: str, m: dict) -> None:
            rows.append(dict(base, level=lv, method=method, weights=wname, macro_f1=m["macro_f1"], balanced_acc=m["balanced_acc"],
                             accuracy=m["accuracy"], n_units=m["n_units"]))

        for method, wname, Pm, wm in _candidates(P, y, meta, exponents, log=log):
            how = BASE_HOW.get(method, method)
            if how == "fp_mean" and task != "size":
                continue
            d_units, d_agg = aggregate(Pm, drive_keys, how, wm)
            if how == "fp_mean" and d_agg.shape[1] != len(classes):
                continue
            for lv in levels:
                if lv == "drive":
                    add(lv, method, wname, unit_metrics(d_units, y_drive, d_agg, classes))
                    continue
                if lv not in veh_truth:
                    continue
                sel = veh_sel[lv]
                units_v, y_v = veh_truth[lv]
                v_units, v_agg = aggregate(Pm[sel], vehicle_keys[sel], how, None if wm is None else wm[sel])
                if not np.array_equal(v_units, units_v):
                    raise RuntimeError("vehicle units out of order")
                add(lv, method, wname, unit_metrics(v_units, y_v, v_agg, classes))
                if how == "fp_mean":
                    continue
                units2, agg2, _ = vehicle_level(d_units, d_agg, drive_vehicle, drive_n, multi=(lv == "vehicle_multi"))
                if not np.array_equal(units2, units_v):
                    raise RuntimeError("vehicle units out of order")
                add(lv, method + DRIVES_SUFFIX, wname, unit_metrics(units2, y_v, agg2, classes))
    cols = ["tag", "model", "seed", "kind", "fold", "recipe_id", "task", "split_col", "level", "protocol", "method", "weights", "macro_f1",
            "balanced_acc", "accuracy", "n_units", "n_windows", "n_models"]
    return pd.DataFrame(rows, columns=cols)


def summarise(df: pd.DataFrame, reference: tuple[str, str] = ("mean", UNIFORM)) -> pd.DataFrame:
    """Per (method, weights): mean / std macro-F1 over runs, paired difference to the reference rule and how many runs improved."""
    if df.empty:
        return pd.DataFrame(columns=["method", "weights", "n_runs", "macro_f1_mean", "macro_f1_std", "delta_mean", "improved", "n_units"])
    ref = df[(df["method"] == reference[0]) & (df["weights"] == reference[1])].set_index("tag")["macro_f1"]
    rows = []
    for (m, w), sub in df.groupby(["method", "weights"], sort=False):
        d = sub["macro_f1"].to_numpy() - ref.reindex(sub["tag"]).to_numpy()
        d = d[np.isfinite(d)]
        rows.append(dict(method=m, weights=w, n_runs=int(len(sub)), macro_f1_mean=float(sub["macro_f1"].mean()), macro_f1_std=float(sub["macro_f1"].std(ddof=0)),
                         delta_mean=float(d.mean()) if len(d) else float("nan"), improved=int((d > 1e-12).sum()), n_units=float(sub["n_units"].mean())))
    return pd.DataFrame(rows).sort_values(["macro_f1_mean", "method"], ascending=[False, True]).reset_index(drop=True)


def stack_table(runs: list[dict], windows_index: pd.DataFrame, task: str, split_col: str, log=print, min_cover: float = MIN_COVER) -> pd.DataFrame:
    """Window- and drive-level macro-F1 of every member, of the plain probability mean over members and of the
    cross-fitted stacker, on the members' common windows. Members whose windows the common set covers by less than
    `min_cover` are dropped first (select_members); fewer than two members left -> empty table."""
    classes = TASKS[task]["classes"]
    cols = ["member", "window_macro_f1", "drive_macro_f1_mean", "drive_macro_f1_logodds", "n_windows"]
    cand = {r["tag"]: r for r in runs if r["task"] == task and r["split_col"] == split_col}
    if len(cand) < 2:
        return pd.DataFrame(columns=cols)
    kept, dropped, _ = select_members({t: r["window_id"] for t, r in cand.items()}, min_cover)
    if dropped:
        log(f"stacking: {len(dropped)} members dropped (windows not shared with the others): {', '.join(dropped)}")
    if len(kept) < 2:
        log("stacking skipped (fewer than two members share their windows)")
        return pd.DataFrame(columns=cols)
    members = {t: (cand[t]["window_id"], cand[t]["probs"]) for t in kept}
    first = cand[kept[0]]
    ids, aligned = align_runs(members)
    meta = windows_index.reindex(ids)
    ok = meta["drive_id"].notna().to_numpy()
    ids, meta = ids[ok], meta[ok]
    aligned = {n: a[ok] for n, a in aligned.items()}
    first_ids = np.asarray(first["window_id"]).astype(str)
    y = pd.Series(np.asarray(first["y"], np.int64), index=first_ids).reindex(ids).to_numpy(np.int64)
    groups = pd.Series(first.get("groups") if first.get("groups") is not None else windows_index["car_model"].reindex(first_ids).to_numpy(object), index=first_ids)
    drive_keys = meta["drive_id"].to_numpy(object)
    _, yd = unit_labels(drive_keys, y)

    def row(name: str, P: np.ndarray) -> dict:
        wm = metrics(y, P.argmax(1), classes)
        out = dict(member=name, window_macro_f1=wm["macro_f1"], n_windows=int(len(ids)))
        for how in ("mean", "logodds"):
            units, agg = aggregate(P, drive_keys, how)
            out[f"drive_macro_f1_{how}"] = unit_metrics(units, yd, agg, classes)["macro_f1"]
        return out

    rows = [row(n, aligned[n]) for n in members]
    rows.append(row("ensemble_mean", np.mean([aligned[n] for n in members], axis=0)))
    try:
        sids, sp = crossfit_stack({n: (ids, aligned[n]) for n in members}, pd.Series(y, index=ids), groups, len(classes))
        if not np.array_equal(sids, ids):
            raise RuntimeError("stack ids out of order")
        rows.append(row("stack_crossfit", sp))
    except ValueError as e:
        log(f"stacking skipped ({e})")
    return pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------------------------------- report
def _fmt(x) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.1f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [f"| {' | '.join(headers)} |", f"|{'---|' * len(headers)}"] + [f"| {' | '.join(r)} |" for r in rows] + [""]


def _ranked(s: pd.DataFrame, top: int) -> list[list[str]]:
    keep = pd.concat([s.head(top), s[s["weights"].isin([UNIFORM, "gate"]) & ~s.index.isin(s.head(top).index)]])
    return [[r.method, r.weights, _fmt(r.macro_f1_mean), _fmt(r.macro_f1_std), f"{100 * r.delta_mean:+.1f}", f"{r.improved}/{r.n_runs}", str(r.n_runs)]
            for r in keep.itertuples()]


def render_task(task: str, split_col: str, kind: str, tables: dict[tuple[str, str], pd.DataFrame], stacks: pd.DataFrame | None, top: int = 10) -> list[str]:
    """Markdown section of one task: reference vs best rule per level / protocol (with the plan's two-stage vehicle rule
    as a named candidate), the ranked rules at drive and vehicle level and the stacking table."""
    tname = {"size": "Size: SMALL / MID / LARGE", "bev_ice": "Powertrain: BEV vs ICE"}.get(task, task)
    posthoc = kind == "test_posthoc"
    suffix = " - POST-HOC on the test predictions of the published models" if posthoc else " - internal folds (out-of-fold)"
    L = [f"## {tname} ({split_col}){suffix}", ""]
    if posthoc:
        L += ["**Post-hoc.** These numbers come from the test-set predictions of the already published models (no fold runs exist "
              "yet). They demonstrate the code and the size of the effects; a rule must not be chosen from them.", ""]
    any_df = next((d for d in tables.values() if not d.empty), None)
    if any_df is None:
        return L + ["No runs found.", ""]
    n_runs = any_df["tag"].nunique()
    kinds = ", ".join(sorted(any_df["kind"].astype(str).unique()))
    recipes = ", ".join(sorted(r for r in any_df["recipe_id"].astype(str).unique() if r)) or "-"
    L += [f"Runs: {n_runs} ({', '.join(sorted(any_df['model'].unique()))}); kinds: {kinds}; recipes: {recipes}; macro-F1 in percent, mean over runs; "
          "`delta` = mean paired difference to `mean / uniform` (the protocol rule: mean over the drive's windows at the drive level, "
          "mean over the vehicle's pooled windows at the vehicle levels), `improved` = runs with a positive difference; "
          f"`mean{DRIVES_SUFFIX} / uniform` is the plan's two-stage vehicle candidate (log-odds mean of the drive decisions, n_windows^0.5 weights).", ""]
    rows = []
    for (level, protocol), df in tables.items():
        if df.empty:
            continue
        s = summarise(df)
        ref = s[(s["method"] == "mean") & (s["weights"] == UNIFORM)]
        two = s[(s["method"] == "mean" + DRIVES_SUFFIX) & (s["weights"] == UNIFORM)]
        best = s.iloc[0]
        rows.append([level, protocol, _fmt(float(ref["macro_f1_mean"].iloc[0])) if len(ref) else "-",
                     _fmt(float(two["macro_f1_mean"].iloc[0])) if len(two) else "-", f"{best['method']} / {best['weights']}",
                     _fmt(float(best["macro_f1_mean"])), f"{100 * best['delta_mean']:+.1f}", f"{int(best['improved'])}/{int(best['n_runs'])}",
                     f"{best['n_units']:.0f}"])
    L += ["### Reference rule vs best rule per level and protocol", ""]
    L += _table(["level", "protocol", "mean / uniform (protocol)", f"mean{DRIVES_SUFFIX} / uniform", "best rule", "macro-F1", "delta", "improved", "units"], rows)
    for level, title in (("drive", "Drive level"), ("vehicle", "Vehicle level")):
        df = tables.get((level, "full"))
        if df is not None and not df.empty:
            L += [f"### {title}, protocol full: ranked rules (top {top} plus every unweighted rule)", ""]
            L += _table(["method", "weights", "macro-F1", "std over runs", "delta", "improved", "runs"], _ranked(summarise(df), top))
    if stacks is not None and not stacks.empty:
        L += ["### Stacking over runs (common windows; stacker fitted out of sample)", ""]
        L += _table(["member", "window macro-F1", "drive macro-F1 (mean)", "drive macro-F1 (log-odds)", "windows"],
                    [[r.member, _fmt(r.window_macro_f1), _fmt(r.drive_macro_f1_mean), _fmt(r.drive_macro_f1_logodds), str(int(r.n_windows))] for r in stacks.itertuples()])
    return L


def write_report(path: Path, sections: list[str]) -> None:
    """Keep the hand-written header above RENDER_MARK (or HEADER when the file does not exist) and replace the rest."""
    header = HEADER
    if path.exists():
        text = path.read_text()
        if RENDER_MARK in text:
            header = text.split(RENDER_MARK)[0]
    path.write_text(header.rstrip("\n") + "\n\n" + RENDER_MARK + "\n\n" + "\n".join(sections))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof-dir", type=Path, default=EXP_ROOT / "recipe" / "results")
    ap.add_argument("--results-dir", type=Path, default=EXP_ROOT / "results", help="post-hoc fallback when --oof-dir has no fold runs")
    ap.add_argument("--tasks", default="size,bev_ice")
    ap.add_argument("--split-col", default="split_platform")
    ap.add_argument("--models", default=",".join(DEEP_MODELS), help="architecture names kept for the aggregation tables")
    ap.add_argument("--stack-extra", default="feat_llmstats_hgb", help="additional model names used as stacking members only (post-hoc path)")
    ap.add_argument("--kinds", default=None, help="run kinds kept, comma list of oof_pooled / oof / test_posthoc (default: pooled fold runs when any "
                                                  "exist, else the single-fold runs of the most frequent fold, else the post-hoc test predictions)")
    ap.add_argument("--recipes", default=None, help="recipe ids kept (comma list; default: every non-probe recipe)")
    ap.add_argument("--include-probes", action="store_true", help="keep the leave-one-car-model-out probe runs (recipe ids L<k>_...)")
    ap.add_argument("--min-cover", type=float, default=MIN_COVER, help="stacking: drop a member when the common window set covers less than this share of its windows")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--protocols", default=",".join(PROTOCOL_NAMES))
    ap.add_argument("--no-stack", action="store_true")
    ap.add_argument("--limit-runs", type=int, default=0, help="keep only the first N runs per task (smoke)")
    ap.add_argument("--out-dir", type=Path, default=EXP_ROOT / "aggregation")
    ap.add_argument("--md", type=Path, default=REPO / "experiments" / "AGGREGATION.md")
    a = ap.parse_args(argv)
    runs = pool_folds(load_runs(a.oof_dir, a.results_dir))
    if not runs:
        print("no predictions found")
        return 2
    posthoc = all(r["kind"] == "test_posthoc" for r in runs)
    kind = "test_posthoc" if posthoc else "oof"
    models = set(a.models.split(","))
    extra = set(a.stack_extra.split(",")) if (a.stack_extra and posthoc) else set()
    runs, dropped = select_runs(runs, kinds=a.kinds.split(",") if a.kinds else None, recipes=a.recipes.split(",") if a.recipes else None,
                                models=models | extra, include_probes=a.include_probes)
    print(f"runs={len(runs)} kinds={sorted({r['kind'] for r in runs})} recipes={sorted({r['recipe_id'] for r in runs if r['recipe_id']})} dropped={dropped}", flush=True)
    if not runs:
        print("no runs left after the selection")
        return 2
    win = load_windows_index()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"kind={kind} windows_index={len(win)}", flush=True)
    sections: list[str] = []
    for task in a.tasks.split(","):
        task_runs = [r for r in runs if r["task"] == task and r["split_col"] == a.split_col]
        agg_runs = [r for r in task_runs if r["model"] in models]
        if a.limit_runs:
            agg_runs, task_runs = agg_runs[: a.limit_runs], task_runs[: a.limit_runs]
        tables: dict[tuple[str, str], pd.DataFrame] = {}
        for protocol in a.protocols.split(","):
            all_levels = evaluate_aggregations(agg_runs, win, task, a.split_col, a.levels.split(","), protocol, log=lambda s: print(s, flush=True))
            for level in a.levels.split(","):
                df = all_levels[all_levels["level"] == level].reset_index(drop=True)
                tables[(level, protocol)] = df
                df.to_csv(a.out_dir / f"agg_{task}_{a.split_col}_{level}_{protocol}.csv", index=False)
                s = summarise(df)
                s.to_csv(a.out_dir / f"summary_{task}_{a.split_col}_{level}_{protocol}.csv", index=False)
                if not s.empty:
                    ref = s[(s["method"] == "mean") & (s["weights"] == UNIFORM)]["macro_f1_mean"]
                    print(f"{task} {level} {protocol}: runs={df['tag'].nunique()} rows={len(df)} ref_mean_uniform={_fmt(float(ref.iloc[0])) if len(ref) else '-'} "
                          f"best={s.iloc[0]['method']}/{s.iloc[0]['weights']} {_fmt(float(s.iloc[0]['macro_f1_mean']))} (delta {100 * s.iloc[0]['delta_mean']:+.1f})", flush=True)
        stacks = None
        if not a.no_stack and len(task_runs) >= 2:
            stacks = stack_table(task_runs, win, task, a.split_col, log=lambda s: print(s, flush=True), min_cover=a.min_cover)
            stacks.to_csv(a.out_dir / f"stack_{task}_{a.split_col}.csv", index=False)
            if not stacks.empty:
                print(f"{task} stacking: members={len(stacks) - 2} " + "; ".join(f"{r.member}: window {_fmt(r.window_macro_f1)} drive {_fmt(r.drive_macro_f1_mean)}"
                                                                                  for r in stacks.itertuples() if r.member in ("ensemble_mean", "stack_crossfit")), flush=True)
        sections += render_task(task, a.split_col, kind, tables, stacks)
    write_report(a.md, sections)
    print(f"wrote {a.md.relative_to(REPO) if a.md.is_relative_to(REPO) else a.md.name} and CSVs ({len(list(a.out_dir.glob('*.csv')))} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
