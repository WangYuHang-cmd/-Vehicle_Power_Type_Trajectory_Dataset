"""Official evaluation protocols and metrics of the dataset (numpy / pandas only).

Protocols (docs/PROTOCOLS.md) select rows of the windows index:
  test_full        every eligible window of the task in the chosen split (default: the test split)
  test_balanced    the published window list `splits/eval_balanced_<task>_<split>.parquet` (class-balanced, with
                   per-group and per-platform caps; the `split_device` lists end in `_device`), intersected with
                   the test_full conditions of the chosen split column as a sanity filter
  test_human_only  test_full restricted to `human_only_strict` windows (no driving automation activity in the
                   window or in the 10 s before it)
Split columns: `split_platform` (headline: no vehicle platform shared between train and test) and `split_device`.
Tasks: size (size_class SMALL / MID / LARGE), bev_ice (powertrain BEV vs ICE, the main powertrain task),
powertrain4 (ICE / HEV / PHEV / BEV, secondary).
Metrics: macro-F1 over the classes present in the selection, computed on all selected windows and on the windows
outside low-support cells ("supported"); per-cell accuracy (cell = size_class x powertrain); confusion matrix CSV.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from vtt import config as C
    SIZE_CLASSES, POWERTRAIN4, MAIN_POWERTRAIN_TASK = C.SIZE_CLASSES, C.POWERTRAIN4, C.MAIN_POWERTRAIN_TASK
except ImportError:                                      # pragma: no cover - mirrors vtt.config
    SIZE_CLASSES, POWERTRAIN4, MAIN_POWERTRAIN_TASK = ("SMALL", "MID", "LARGE"), ("ICE", "HEV", "PHEV", "BEV"), ("BEV", "ICE")

TASKS = {
    "size": dict(label="size_class", eligible="task_size_eligible", classes=tuple(SIZE_CLASSES)),
    "bev_ice": dict(label="powertrain", eligible="task_bev_ice_eligible", classes=tuple(MAIN_POWERTRAIN_TASK)),
    "powertrain4": dict(label="powertrain", eligible=None, classes=tuple(POWERTRAIN4)),
}
PROTOCOLS = ("test_full", "test_balanced", "test_human_only")
SPLIT_COLS = ("split_platform", "split_device")
HEADLINE_SPLIT_COL = "split_platform"


def task_labels(windows: pd.DataFrame, task: str) -> np.ndarray:
    """Label per window row (object array; None where the window has no class of the task)."""
    t = TASKS[task]
    lab = windows[t["label"]].to_numpy(object)
    return np.array([x if x in t["classes"] else None for x in lab], dtype=object)


def protocol_mask(windows: pd.DataFrame, task: str, protocol: str = "test_full", split_col: str = HEADLINE_SPLIT_COL,
                  split: str = "test", balanced_ids=None, human_col: str = "human_only_strict") -> np.ndarray:
    """Boolean row mask of the windows index for (task, protocol, split column, split).

    test_full: `eligible & <task eligibility> & (split_col == split)` with a class label of the task;
    test_balanced: test_full & `window_id.isin(balanced_ids)` (the published list of the matching split scheme is a
    subset of test_full, so the intersection is a sanity filter: ids outside test_full, e.g. the list of the other
    scheme, are dropped with a warning);
    test_human_only: test_full & human_only_strict."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; choose from {PROTOCOLS}")
    if split_col not in SPLIT_COLS:
        raise ValueError(f"unknown split column {split_col!r}; choose from {SPLIT_COLS}")
    t = TASKS[task]
    m = (windows[split_col].to_numpy(object) == split)
    if "eligible" in windows:
        m &= windows["eligible"].fillna(False).to_numpy(bool)
    if t["eligible"] is not None and t["eligible"] in windows:
        m &= windows[t["eligible"]].fillna(False).to_numpy(bool)
    m &= np.array([x is not None for x in task_labels(windows, task)])
    if protocol == "test_balanced":
        if balanced_ids is None:
            raise ValueError("test_balanced needs the published window-id list (balanced_ids)")
        listed = windows["window_id"].isin(set(balanced_ids)).to_numpy()
        n_outside = int((listed & ~m).sum())
        if n_outside:
            warnings.warn(f"{n_outside} window(s) of the balanced list are outside test_full for {split_col} == {split!r} / "
                          f"task {task} and are dropped; is the list from the other split scheme?")
        m &= listed
    elif protocol == "test_human_only":
        m &= windows[human_col].fillna(False).to_numpy(bool)
    return m


def confusion_matrix(y_true, y_pred, labels) -> np.ndarray:
    """Rows = true class, columns = predicted class (predictions outside `labels` are ignored)."""
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def macro_f1(y_true, y_pred, labels=None) -> float:
    """Unweighted mean F1 over the classes with support (> 0 true samples); NaN with no supported class."""
    y_true = np.asarray(list(y_true), dtype=object); y_pred = np.asarray(list(y_pred), dtype=object)
    labels = list(labels) if labels is not None else sorted({x for x in y_true if x is not None})
    f1s = []
    for lab in labels:
        tp = int(np.sum((y_true == lab) & (y_pred == lab))); fn = int(np.sum((y_true == lab) & (y_pred != lab)))
        fp = int(np.sum((y_true != lab) & (y_pred == lab)))
        if tp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s)) if f1s else np.nan


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(list(y_true), dtype=object); y_pred = np.asarray(list(y_pred), dtype=object)
    return float(np.mean(y_true == y_pred)) if len(y_true) else np.nan


def per_cell_accuracy(windows_sub: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    """Accuracy inside every (size_class, powertrain) cell of the selection, with the low-support flag."""
    df = pd.DataFrame({"size_class": windows_sub["size_class"].to_numpy(object), "powertrain": windows_sub["powertrain"].to_numpy(object),
                       "low_support_cell": windows_sub["low_support_cell"].fillna(False).to_numpy(bool) if "low_support_cell" in windows_sub else False,
                       "hit": np.asarray(list(y_true), dtype=object) == np.asarray(list(y_pred), dtype=object)})
    g = df.groupby(["size_class", "powertrain"], dropna=False)
    out = g.agg(n=("hit", "size"), accuracy=("hit", "mean"), low_support_cell=("low_support_cell", "max")).reset_index()
    return out.sort_values(["size_class", "powertrain"]).reset_index(drop=True)


def chance_levels(y_true, labels, n_perm: int = 20, seed: int = 0) -> dict:
    """Macro-F1 of the always-majority predictor and the mean over label permutations (same class frequencies)."""
    y = np.asarray(list(y_true), dtype=object)
    if len(y) == 0:
        return dict(majority_f1=np.nan, permuted_f1=np.nan)
    vals, cnt = np.unique(y.astype(str), return_counts=True)
    maj = vals[np.argmax(cnt)]
    rng = np.random.default_rng(seed)
    perm = [macro_f1(y, rng.permutation(y), labels) for _ in range(n_perm)]
    return dict(majority_f1=macro_f1(y, np.full(len(y), maj, dtype=object), labels), permuted_f1=float(np.nanmean(perm)))


def write_confusion_csv(cm: np.ndarray, labels, path: Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels]).to_csv(path)
    return path


def align_predictions(windows_sub: pd.DataFrame, y_pred) -> np.ndarray:
    """Predictions as an object array aligned with `windows_sub`: a Series indexed by window_id, a dict, or an
    array already aligned with the selection."""
    if isinstance(y_pred, pd.Series):
        return y_pred.reindex(windows_sub["window_id"]).to_numpy(object)
    if isinstance(y_pred, dict):
        return np.array([y_pred.get(w) for w in windows_sub["window_id"]], dtype=object)
    arr = np.asarray(list(y_pred), dtype=object)
    if len(arr) != len(windows_sub):
        raise ValueError(f"{len(arr)} predictions for {len(windows_sub)} selected windows")
    return arr


def evaluate(windows: pd.DataFrame, y_pred, task: str, protocol: str = "test_full", split_col: str = HEADLINE_SPLIT_COL,
             split: str = "test", balanced_ids=None, out_dir: Path | None = None, name: str | None = None) -> dict:
    """Protocol selection + metrics for one prediction set. Returns a dict with the macro-F1 (all / supported
    windows), accuracy, chance levels, counts and the per-cell table; writes the confusion CSV when out_dir is given."""
    mask = protocol_mask(windows, task, protocol, split_col, split, balanced_ids)
    sub = windows[mask]
    y_true = task_labels(sub, task)
    pred = align_predictions(sub, y_pred)
    labels = list(TASKS[task]["classes"])
    supported = ~sub["low_support_cell"].fillna(False).to_numpy(bool) if "low_support_cell" in sub else np.ones(len(sub), bool)
    res = dict(task=task, protocol=protocol, split_col=split_col, split=split, n_windows=int(len(sub)),
               n_groups=int(sub["group_id"].nunique()) if "group_id" in sub else -1,
               n_platforms=int(sub["platform"].nunique()) if "platform" in sub else -1,
               macro_f1=macro_f1(y_true, pred, labels), macro_f1_supported=macro_f1(y_true[supported], pred[supported], labels),
               accuracy=accuracy(y_true, pred), **chance_levels(y_true, labels),
               per_cell=per_cell_accuracy(sub, y_true, pred), confusion=confusion_matrix(y_true, pred, labels), labels=labels)
    if out_dir is not None:
        tag = name or f"{task}_{protocol}_{split_col}_{split}"
        res["confusion_csv"] = str(write_confusion_csv(res["confusion"], labels, Path(out_dir) / f"{tag}_confusion.csv"))
        res["per_cell"].to_csv(Path(out_dir) / f"{tag}_per_cell.csv", index=False)
    return res


# ----------------------------------------------------------------------------- drive- and vehicle-level protocols
# unit protocol -> the window protocol that selects the evidence, the unit column and the minimum number of drives per unit
UNIT_PROTOCOLS = {
    "drive_full": dict(windows="test_full", unit_col="drive_id", min_drives=1),
    "drive_human_only": dict(windows="test_human_only", unit_col="drive_id", min_drives=1),
    "vehicle_full": dict(windows="test_full", unit_col="vehicle_id", min_drives=1),
    "vehicle_full_multi": dict(windows="test_full", unit_col="vehicle_id", min_drives=2),
}
# window -> unit aggregation rules; "mean" is the official rule, the others exist for the aggregation studies
AGGREGATIONS = ("mean", "logodds", "median", "confidence", "trimmed", "vote")
LOGODDS_CLIP = 1e-4
TRIM_FRAC = 0.2


def one_hot(labels, classes) -> np.ndarray:
    """(n, k) float64 one-hot matrix of class labels; a label outside `classes` (or None) raises."""
    classes = list(classes)
    pos = {c: i for i, c in enumerate(classes)}
    lab = np.asarray(list(labels), dtype=object)
    bad = [x for x in lab if x not in pos]
    if bad:
        raise ValueError(f"{len(bad)} label(s) outside the classes {classes} (or missing)")
    out = np.zeros((len(lab), len(classes)), dtype=np.float64)
    out[np.arange(len(lab)), [pos[x] for x in lab]] = 1.0
    return out


def _prob_columns(probs: pd.DataFrame, classes: list) -> list:
    cols = list(probs.columns)
    if all(c in cols for c in classes):
        return classes
    if all(f"p_{c}" in cols for c in classes):
        return [f"p_{c}" for c in classes]
    if all(i in cols for i in range(len(classes))):
        return list(range(len(classes)))
    raise ValueError(f"probability table needs the columns {classes} (or p_<class>, or 0..{len(classes) - 1})")


def probs_matrix(windows_sub: pd.DataFrame, probs, classes) -> np.ndarray:
    """(n, k) float64 class probabilities aligned with `windows_sub`.

    Accepted forms: a DataFrame indexed by window_id with one column per class (named by class, `p_<class>` or
    0..k-1); a label Series / dict indexed by window_id (-> one-hot); an (n, k) array aligned with the selection; a
    label sequence aligned with the selection (-> one-hot). A selected window without an entry raises."""
    classes = list(classes)
    ids = windows_sub["window_id"].astype(str).to_numpy()
    if isinstance(probs, dict):
        probs = pd.Series(probs, dtype=object)
    if isinstance(probs, pd.DataFrame):
        df = probs[_prob_columns(probs, classes)].copy()
        df.index = df.index.astype(str)
        if not df.index.is_unique:
            raise ValueError("duplicate window_id in the probability table")
        m = df.reindex(ids).to_numpy(dtype=np.float64)
        missing = int(np.isnan(m).any(axis=1).sum())
        if missing:
            raise ValueError(f"{missing} selected window(s) have no probabilities")
        return m
    if isinstance(probs, pd.Series):
        s = probs.copy()
        s.index = s.index.astype(str)
        if not s.index.is_unique:
            raise ValueError("duplicate window_id in the label series")
        return one_hot(s.reindex(ids).to_numpy(object), classes)
    arr = np.asarray(probs)
    if arr.ndim == 2:
        if arr.shape != (len(ids), len(classes)):
            raise ValueError(f"probabilities of shape {arr.shape} for {len(ids)} selected windows and {len(classes)} classes")
        return arr.astype(np.float64)
    if arr.ndim == 1:
        if len(arr) != len(ids):
            raise ValueError(f"{len(arr)} labels for {len(ids)} selected windows")
        return one_hot(arr.astype(object), classes)
    raise ValueError("probs must be a DataFrame, a Series, a dict, an (n, k) array or a label sequence")


def _label_column(windows_sub: pd.DataFrame, classes: list) -> str:
    for t in TASKS.values():
        if tuple(t["classes"]) == tuple(classes):
            return t["label"]
    for col in ("size_class", "powertrain"):
        if col in windows_sub and windows_sub[col].dropna().isin(classes).all():
            return col
    raise ValueError(f"no label column of the windows index holds the classes {classes}")


def _sum_by(codes: np.ndarray, M: np.ndarray, n_units: int) -> np.ndarray:
    return np.stack([np.bincount(codes, weights=M[:, j], minlength=n_units) for j in range(M.shape[1])], axis=1)


def aggregate_windows(windows_sub: pd.DataFrame, probs, unit_col: str, classes, how: str = "mean",
                      label_col: str | None = None) -> pd.DataFrame:
    """One row per unit (`unit_col` value) of the selected windows: unit, truth, pred, n_windows, n_drives, p_<class>...

    how = "mean" (official): mean of the per-window probability vectors; "logodds": mean per-class log-odds with the
    probabilities clipped to [1e-4, 1 - 1e-4], mapped back through the logistic function and renormalised; "median":
    per-class median, renormalised; "confidence": mean weighted by (max p - 1/k), plain mean when every weight is 0;
    "trimmed": per-class mean after dropping the lowest and highest 20 % of the windows (plain mean when fewer than
    5 windows); "vote": share of windows whose arg-max is the class. pred = arg-max of the aggregated vector, ties go
    to the earlier class in the task's class order. truth = the label of the unit's first selected window."""
    classes = list(classes)
    k = len(classes)
    if how not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {how!r}; choose from {AGGREGATIONS}")
    P = probs_matrix(windows_sub, probs, classes)
    label_col = label_col or _label_column(windows_sub, classes)
    codes, uniq = pd.factorize(windows_sub[unit_col].astype(str).to_numpy(), sort=True)
    n_units = len(uniq)
    counts = np.bincount(codes, minlength=n_units).astype(np.float64)
    mean = _sum_by(codes, P, n_units) / counts[:, None] if n_units else np.zeros((0, k))
    if how == "mean":
        agg = mean
    elif how == "logodds":
        Q = np.clip(P, LOGODDS_CLIP, 1.0 - LOGODDS_CLIP)
        L = _sum_by(codes, np.log(Q) - np.log1p(-Q), n_units) / counts[:, None]
        agg = 1.0 / (1.0 + np.exp(-L))
        agg = agg / agg.sum(axis=1, keepdims=True)
    elif how == "confidence":
        w = P.max(axis=1) - 1.0 / k
        ws = np.bincount(codes, weights=w, minlength=n_units)
        num = _sum_by(codes, P * w[:, None], n_units)
        pos = ws > 0
        agg = mean.copy()
        agg[pos] = num[pos] / ws[pos, None]
    elif how == "vote":
        agg = np.zeros((n_units, k))
        np.add.at(agg, (codes, P.argmax(axis=1)), 1.0)
        agg = agg / counts[:, None]
    else:                                                   # median / trimmed: per-unit blocks
        order = np.argsort(codes, kind="stable")
        bounds = np.flatnonzero(np.diff(codes[order])) + 1
        agg = np.zeros((n_units, k))
        for u, rows in enumerate(np.split(order, bounds) if n_units else []):
            block = P[rows]
            if how == "median":
                v = np.median(block, axis=0)
            else:
                t = int(np.floor(TRIM_FRAC * len(rows)))
                v = np.sort(block, axis=0)[t:len(rows) - t].mean(axis=0) if len(rows) - 2 * t > 0 else block.mean(axis=0)
            s = v.sum()
            agg[u] = v / s if s > 0 else block.mean(axis=0)
    pred = agg.argmax(axis=1) if n_units else np.zeros(0, dtype=int)
    first = np.unique(codes, return_index=True)[1] if n_units else np.zeros(0, dtype=int)
    truth = windows_sub[label_col].to_numpy(object)[first]
    out = pd.DataFrame({"unit": uniq, "truth": truth, "pred": np.array([classes[i] for i in pred], dtype=object),
                        "n_windows": counts.astype(np.int64)})
    if "drive_id" in windows_sub and unit_col != "drive_id":
        nd = pd.Series(windows_sub["drive_id"].astype(str).to_numpy()).groupby(codes).nunique()
        out["n_drives"] = nd.reindex(range(n_units)).fillna(0).astype(np.int64).to_numpy()
    else:
        out["n_drives"] = 1
    for j, c in enumerate(classes):
        out[f"p_{c}"] = agg[:, j]
    return out


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """Unweighted mean F1 over the classes with support, from a (k, k) confusion matrix (rows = true)."""
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp
        if tp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def unit_metrics(y_true, y_pred, classes) -> dict:
    """Metric dict in the shape of experiments.train.metrics: n, macro_f1, balanced_acc, accuracy, per_class_recall,
    support, confusion (rows = true), classes; labels are class names."""
    classes = list(classes)
    cm = confusion_matrix(y_true, y_pred, classes)
    n = int(cm.sum())
    rec = [cm[c, c] / cm[c].sum() for c in range(len(classes)) if cm[c].sum() > 0]
    return dict(n=n, macro_f1=macro_f1_from_confusion(cm), balanced_acc=float(np.mean(rec)) if rec else float("nan"),
                accuracy=float(np.trace(cm) / n) if n else float("nan"),
                per_class_recall={c: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None) for i, c in enumerate(classes)},
                support={c: int(cm[i].sum()) for i, c in enumerate(classes)}, confusion=cm.tolist(), classes=classes)


def cluster_bootstrap_ci(y_true, y_pred, clusters, classes, n_boot: int = 1000, seed: int = 0) -> list[float] | None:
    """95 % percentile interval of the macro-F1 under a cluster bootstrap: the clusters (car models) are resampled
    with replacement, every unit of a drawn cluster enters once per draw. None with fewer than two clusters."""
    classes = list(classes)
    k = len(classes)
    pos = {c: i for i, c in enumerate(classes)}
    codes, uniq = pd.factorize(np.asarray([str(c) for c in clusters], dtype=object))
    M = len(uniq)
    if M < 2:
        return None
    ti = np.array([pos[x] for x in y_true]); pi = np.array([pos[x] for x in y_pred])
    cms = np.zeros((M, k, k))
    np.add.at(cms, (codes, ti, pi), 1.0)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, M, size=M), minlength=M).astype(np.float64)
        vals[b] = macro_f1_from_confusion(np.tensordot(cnt, cms, axes=(0, 0)))
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _unit_attributes(sub: pd.DataFrame, unit_col: str) -> pd.DataFrame:
    """Per-unit car model (`model`), group_id and low-support flag from the selected windows."""
    key = sub[unit_col].astype(str)
    out = pd.DataFrame(index=pd.Index(sorted(key.unique()), name="unit"))
    for col in ("model", "group_id"):
        if col in sub:
            out[col] = sub.groupby(key)[col].first().reindex(out.index)
    if "low_support_cell" in sub:
        out["low_support_cell"] = sub.groupby(key)["low_support_cell"].apply(lambda s: bool(s.fillna(False).astype(bool).any())).reindex(out.index)
    return out


def _unit_population(table: pd.DataFrame, task: str, split_col: str, split: str, unit_col: str, min_drives: int) -> set:
    """Units (drives or vehicles) of the split with a class label of the task, from the drives index or, lacking it,
    from the windows index (drives without any listed window are then invisible)."""
    t = TASKS[task]
    m = (table[split_col].to_numpy(object) == split) & table[t["label"]].isin(t["classes"]).to_numpy()
    if t["eligible"] is not None and t["eligible"] in table:
        m &= table[t["eligible"]].fillna(False).to_numpy(bool)
    rows = table[m]
    if min_drives > 1 and "drive_id" in rows:
        nd = rows.groupby(rows[unit_col].astype(str))["drive_id"].nunique()
        return set(nd.index[nd >= min_drives])
    return set(rows[unit_col].astype(str))


def evaluate_units(windows: pd.DataFrame, probs, task: str, protocol: str = "drive_full", split_col: str = HEADLINE_SPLIT_COL,
                   split: str = "test", out_dir: Path | None = None, name: str | None = None, n_boot: int = 1000, seed: int = 0,
                   drives: pd.DataFrame | None = None, how: str = "mean") -> dict:
    """Drive- or vehicle-level evaluation (docs/PROTOCOLS.md, "Drive- and vehicle-level protocols").

    The unit protocol selects windows with its window protocol (`test_full` or `test_human_only`), aggregates the
    per-window probability vectors per unit (`how`, official rule "mean"), predicts the arg-max (ties -> earlier
    class) and scores the units: the metric dict of `unit_metrics` plus n_units, n_units_without_windows (units of
    the split with a task label but no selected window; exact when the drives index is passed as `drives`),
    n_windows_used, n_models, macro_f1_supported / n_units_supported (units outside low-support cells) and
    ci95_macro_f1 (cluster bootstrap over car models, fallback identity groups; n_boot resamples, `seed`).
    `probs` takes the forms accepted by `probs_matrix`. Writes <tag>_units.csv and <tag>_confusion.csv to out_dir."""
    if protocol not in UNIT_PROTOCOLS:
        raise ValueError(f"unknown unit protocol {protocol!r}; choose from {tuple(UNIT_PROTOCOLS)}")
    spec = UNIT_PROTOCOLS[protocol]
    t = TASKS[task]
    classes = list(t["classes"])
    sub = windows[protocol_mask(windows, task, spec["windows"], split_col, split)]
    units = aggregate_windows(sub, probs, spec["unit_col"], classes, how=how, label_col=t["label"])
    units = units.merge(_unit_attributes(sub, spec["unit_col"]), left_on="unit", right_index=True, how="left")
    if spec["min_drives"] > 1:
        units = units[units["n_drives"] >= spec["min_drives"]].reset_index(drop=True)
    population = _unit_population(drives if drives is not None else windows, task, split_col, split, spec["unit_col"], spec["min_drives"])
    res = dict(task=task, protocol=protocol, window_protocol=spec["windows"], unit_col=spec["unit_col"], split_col=split_col,
               split=split, how=how)
    res.update(unit_metrics(units["truth"], units["pred"], classes))
    res["n_units"] = int(len(units))
    res["n_units_without_windows"] = int(len(population - set(units["unit"].astype(str))))
    res["n_windows_used"] = int(units["n_windows"].sum())
    res["n_drives_used"] = int(units["n_drives"].sum())
    cluster_col = "model" if "model" in units and units["model"].notna().any() else ("group_id" if "group_id" in units else None)
    res["n_models"] = int(units["model"].nunique()) if "model" in units else -1
    res["cluster_col"] = cluster_col
    res["ci95_macro_f1"] = (cluster_bootstrap_ci(units["truth"], units["pred"], units[cluster_col].astype(str), classes, n_boot, seed)
                            if cluster_col is not None and len(units) else None)
    supported = ~units["low_support_cell"].fillna(False).astype(bool).to_numpy() if "low_support_cell" in units else np.ones(len(units), bool)
    res["macro_f1_supported"] = macro_f1(units["truth"].to_numpy(object)[supported], units["pred"].to_numpy(object)[supported], classes)
    res["n_units_supported"] = int(supported.sum())
    if out_dir is not None:
        tag = name or f"{task}_{protocol}_{split_col}_{split}"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        units.to_csv(out_dir / f"{tag}_units.csv", index=False)
        res["units_csv"] = str(out_dir / f"{tag}_units.csv")
        res["confusion_csv"] = str(write_confusion_csv(np.array(res["confusion"]), classes, out_dir / f"{tag}_confusion.csv"))
    return res
