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
