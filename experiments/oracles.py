#!/usr/bin/env python3
"""Zero-training oracle studies on the saved test predictions: post-hoc ceilings for the already-published models.

Reads <data-root>/experiments/results/<tag>_probs.npz with <tag>.json (and <tag>_valprobs.npz where regen_probs.py has
run), build/index/windows.parquet and build/index/drives.parquet; writes <data-root>/experiments/oracles/*.csv and
experiments/ORACLES.md (counts, rates and car-model names only). CPU only: numpy / pandas, scikit-learn for the optional
stacking. Studies, per task and split scheme on the test split: (a) window-to-drive aggregation rules x window masks,
(b) ensembles at window and drive level, (c) vehicle pooling, (d) evidence sufficiency (drive accuracy by amount of
evidence, window subsampling), (e) upper bounds (per-car-model majority oracle, footprint-noise oracle, curb-mass
oracle, separability of the held-out car models). Nothing here trains a network; the training split is touched only by
the curb-mass cuts (fitted on training drives) and the stacking model (fitted on validation probabilities).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402
from starter_kit.eval import AGGREGATIONS, TASKS, aggregate_windows, macro_f1_from_confusion, protocol_mask, unit_metrics  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXP_ROOT = Path(C.DATA_ROOT) / "experiments"
DEEP = ("resnet", "gru", "transformer", "mamba2", "mamba1")
FEATURE_MODELS = ("feat_logreg", "feat_hgb", "feat_llm_logreg", "feat_llm_hgb", "feat_llmstats_hgb")
ARCH_ORDER = FEATURE_MODELS + DEEP
MASKS = {"all": "every test_full window", "moving": "moving_frac >= 0.5", "episodes": "n_accel_ep + n_decel_ep >= 1",
         "human": "human_only_strict"}
SPLIT_COLS = ("split_platform", "split_device")
TASK_LIST = ("size", "bev_ice")
EVIDENCE_BINS = ((1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, None))
COUNT_BINS = ((0, 0),) + EVIDENCE_BINS
EVIDENCE_QUANTITIES = ("n_windows", "n_accel_ep", "n_decel_ep", "n_lead_accel_events")
SUBSAMPLE_M = (1, 2, 4, 8, 16, 32)
SUBSAMPLE_MIN_WINDOWS = 32
SUBSAMPLE_REPEATS = 50
NOISE_LEVELS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
NOISE_REPEATS = 200
NEAR_GAP_M2 = 0.15
LOG_CLIP = 1e-6
WIN_COLS = ["window_id", "drive_id", "vehicle_id", "group_id", "model", "size_class", "powertrain", "footprint_m2", "eligible",
            "task_size_eligible", "task_bev_ice_eligible", "split_platform", "split_device", "moving_frac", "n_accel_ep",
            "n_decel_ep", "n_lead_accel_events", "human_only_strict", "low_support_cell"]
DRIVE_COLS = ["drive_id", "vehicle_id", "group_id", "model", "size_class", "powertrain", "footprint_m2", "curb_mass_kg",
              "task_size_eligible", "task_bev_ice_eligible", "split_platform", "split_device"]


# ----------------------------------------------------------------------------- small helpers
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion k / n (NaN, NaN when n == 0)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def bin_label(b: tuple) -> str:
    lo, hi = b
    if hi is None:
        return f">{lo - 1}"
    return str(lo) if lo == hi else f"{lo}-{hi}"


def in_bin(v: np.ndarray, b: tuple) -> np.ndarray:
    lo, hi = b
    return (v >= lo) & ((v <= hi) if hi is not None else True)


def size_class_of(fp: np.ndarray, thresholds=C.FOOTPRINT_THRESHOLDS_M2) -> np.ndarray:
    """0 SMALL below t1, 1 MID up to t2 (inclusive), 2 LARGE above (docs/SIZE_CLASSES.md)."""
    t1, t2 = thresholds
    fp = np.asarray(fp, dtype=np.float64)
    return (fp >= t1).astype(np.int64) + (fp > t2).astype(np.int64)


def window_mask(sub: pd.DataFrame, name: str) -> np.ndarray:
    if name == "all":
        return np.ones(len(sub), dtype=bool)
    if name == "moving":
        return sub["moving_frac"].fillna(0).to_numpy(dtype=float) >= 0.5
    if name == "episodes":
        return (sub["n_accel_ep"].fillna(0).to_numpy(dtype=float) + sub["n_decel_ep"].fillna(0).to_numpy(dtype=float)) >= 1
    if name == "human":
        return sub["human_only_strict"].fillna(False).to_numpy(dtype=bool)
    raise ValueError(f"unknown mask {name!r}; choose from {tuple(MASKS)}")


def class_index(labels, classes) -> np.ndarray:
    pos = {c: i for i, c in enumerate(classes)}
    return np.array([pos[x] for x in labels], dtype=np.int64)


def window_f1(P: np.ndarray, y: np.ndarray, k: int) -> float:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, P.argmax(1)), 1)
    return macro_f1_from_confusion(cm)


def units_of(sub: pd.DataFrame, P: np.ndarray, classes, label_col: str, how: str = "mean", mask: np.ndarray | None = None,
             unit_col: str = "drive_id") -> pd.DataFrame:
    if mask is not None:
        sub, P = sub[mask], P[mask]
    return aggregate_windows(sub, P, unit_col, classes, how=how, label_col=label_col)


def f1_of(units: pd.DataFrame, classes) -> float:
    return unit_metrics(units["truth"], units["pred"], classes)["macro_f1"] if len(units) else float("nan")


def arch_key(a: str) -> int:
    return ARCH_ORDER.index(a) if a in ARCH_ORDER else 99


# ----------------------------------------------------------------------------- loading
def load_runs(results_dir: Path, split_col: str, task: str) -> list[dict]:
    """Saved test predictions of (split scheme, task): tag, arch, seed, window_id, probs, y (+ val when regenerated)."""
    out = []
    for p in sorted(Path(results_dir).glob("*_probs.npz")):
        tag = p.name[: -len("_probs.npz")]
        meta = Path(results_dir) / f"{tag}.json"
        if not meta.exists():
            continue
        r = json.loads(meta.read_text())
        if r.get("split_col") != split_col or r.get("task") != task:
            continue
        z = np.load(p, allow_pickle=True)
        run = dict(tag=tag, arch=r["model"], seed=int(r["seed"]), window_id=z["window_id"].astype(str),
                   probs=z["probs"].astype(np.float64), y=z["y"].astype(np.int64))
        vp = Path(results_dir) / f"{tag}_valprobs.npz"
        if vp.exists():
            zv = np.load(vp, allow_pickle=True)
            run["val"] = dict(window_id=zv["window_id"].astype(str), probs=zv["probs"].astype(np.float64), y=zv["y"].astype(np.int64))
        out.append(run)
    return out


def align_runs(runs: list[dict], sub: pd.DataFrame) -> list[dict]:
    """Reorder every run's test probabilities to the rows of `sub` (key P); runs missing a selected window are dropped."""
    ids = pd.Index(sub["window_id"].astype(str))
    kept = []
    for r in runs:
        pos = pd.Index(r["window_id"]).get_indexer(ids)
        if (pos < 0).any():
            continue
        r = dict(r)
        r["P"] = r["probs"][pos]
        kept.append(r)
    return kept


# ----------------------------------------------------------------------------- (a) aggregation rules x masks
def study_aggregation(sub: pd.DataFrame, runs: list[dict], classes, label_col: str) -> pd.DataFrame:
    rows = []
    masks = {m: window_mask(sub, m) for m in MASKS}
    for r in runs:
        for m, mk in masks.items():
            for how in AGGREGATIONS:
                u = units_of(sub, r["P"], classes, label_col, how=how, mask=mk)
                met = unit_metrics(u["truth"], u["pred"], classes)
                rows.append(dict(arch=r["arch"], seed=r["seed"], mask=m, how=how, macro_f1=met["macro_f1"],
                                 balanced_acc=met["balanced_acc"], accuracy=met["accuracy"], n_drives=int(len(u)), n_windows=int(mk.sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- (b) ensembles
def ensemble_sets(runs: list[dict]) -> list[dict]:
    """Named ensembles: parts = probability arrays averaged with equal weight, runs = the raw members (vote / stacking)."""
    sets = []
    by_arch: dict[str, list[dict]] = {}
    for r in runs:
        by_arch.setdefault(r["arch"], []).append(r)
    for a in sorted(by_arch, key=arch_key):
        rs = by_arch[a]
        if len(rs) >= 2:
            sets.append(dict(name=f"seeds: {a}", parts=[r["P"] for r in rs], runs=rs))
    deep0 = [r for r in runs if r["arch"] in DEEP and r["seed"] == 0]
    deep = [r for r in runs if r["arch"] in DEEP]
    if len(deep0) >= 2:
        sets.append(dict(name="architectures: deep, seed 0", parts=[r["P"] for r in deep0], runs=deep0))
    if len(deep) >= 2:
        sets.append(dict(name="architectures: deep, all seeds", parts=[r["P"] for r in deep], runs=deep))
        deep_mean = np.mean([r["P"] for r in deep], axis=0)
        for f in ("feat_hgb", "feat_llmstats_hgb"):
            fr = by_arch.get(f, [])
            if fr:
                sets.append(dict(name=f"deep (all seeds) + {f}", parts=[deep_mean] + [r["P"] for r in fr], runs=deep + fr))
    if len(runs) >= 2:
        sets.append(dict(name="all saved runs", parts=[r["P"] for r in runs], runs=list(runs)))
    return sets


def vote_over_members(member_units: list[pd.DataFrame], classes) -> pd.DataFrame:
    """Majority vote of the members' per-unit predictions (ties -> earlier class); truth from the first member."""
    base = member_units[0][["unit", "truth"]].copy()
    counts = np.zeros((len(base), len(classes)))
    pos = {c: i for i, c in enumerate(classes)}
    for u in member_units:
        pred = u.set_index("unit")["pred"].reindex(base["unit"]).to_numpy(object)
        counts[np.arange(len(base)), [pos[p] for p in pred]] += 1
    base["pred"] = np.array([classes[i] for i in counts.argmax(1)], dtype=object)
    return base


def stacked_probs(members: list[dict], classes) -> np.ndarray | None:
    """Logistic-regression stacker on the members' validation log-probabilities, applied to the test probabilities.
    None when a member has no regenerated validation probabilities or scikit-learn is missing."""
    if any("val" not in m for m in members):
        return None
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:                                      # pragma: no cover
        return None
    ref = members[0]["val"]
    ids = pd.Index(ref["window_id"])
    Xv, Xt = [], []
    for m in members:
        pos = pd.Index(m["val"]["window_id"]).get_indexer(ids)
        if (pos < 0).any():
            return None
        Xv.append(np.log(np.clip(m["val"]["probs"][pos], LOG_CLIP, 1.0)))
        Xt.append(np.log(np.clip(m["P"], LOG_CLIP, 1.0)))
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(np.hstack(Xv), ref["y"])
    pr = clf.predict_proba(np.hstack(Xt))
    out = np.zeros((len(pr), len(classes)))
    out[:, clf.classes_] = pr
    return out


def study_ensembles(sub: pd.DataFrame, runs: list[dict], classes, label_col: str, y: np.ndarray) -> pd.DataFrame:
    k = len(classes)
    rows = []
    single = {r["tag"]: units_of(sub, r["P"], classes, label_col) for r in runs}
    for r in runs:
        rows.append(dict(ensemble=f"single: {r['arch']} s{r['seed']}", kind="single", n_members=1, window_macro_f1=window_f1(r["P"], y, k),
                         drive_macro_f1=f1_of(single[r["tag"]], classes), drive_vote_macro_f1=np.nan, stacked_window_macro_f1=np.nan,
                         stacked_drive_macro_f1=np.nan, member_drive_f1_mean=np.nan, member_drive_f1_max=np.nan, n_drives=int(len(single[r["tag"]]))))
    for s in ensemble_sets(runs):
        P = np.mean(s["parts"], axis=0)
        u = units_of(sub, P, classes, label_col)
        member_f1 = [f1_of(single[r["tag"]], classes) for r in s["runs"]]
        vote = vote_over_members([single[r["tag"]] for r in s["runs"]], classes)
        st = stacked_probs(s["runs"], classes)
        rows.append(dict(ensemble=s["name"], kind="ensemble", n_members=len(s["runs"]), window_macro_f1=window_f1(P, y, k),
                         drive_macro_f1=f1_of(u, classes), drive_vote_macro_f1=f1_of(vote, classes),
                         stacked_window_macro_f1=window_f1(st, y, k) if st is not None else np.nan,
                         stacked_drive_macro_f1=f1_of(units_of(sub, st, classes, label_col), classes) if st is not None else np.nan,
                         member_drive_f1_mean=float(np.mean(member_f1)), member_drive_f1_max=float(np.max(member_f1)), n_drives=int(len(u))))
    return pd.DataFrame(rows)


def reference_probs(runs: list[dict]) -> tuple[np.ndarray, str]:
    """Reference predictions for the evidence and ceiling studies: the mean over every deep run (all architectures and
    seeds), falling back to every saved run."""
    deep = [r for r in runs if r["arch"] in DEEP]
    use = deep or runs
    return np.mean([r["P"] for r in use], axis=0), f"mean probability of {len(use)} {'deep ' if deep else ''}runs"


# ----------------------------------------------------------------------------- (c) vehicle pooling
def study_vehicle_counts(sub: pd.DataFrame, label_col: str) -> pd.DataFrame:
    d = sub.groupby(sub["drive_id"].astype(str)).agg(vehicle_id=("vehicle_id", "first"), truth=(label_col, "first"),
                                                        size_class=("size_class", "first"), powertrain=("powertrain", "first"))
    d[["size_class", "powertrain"]] = d[["size_class", "powertrain"]].fillna("unknown")
    rows = []
    for level, keys in (("class", ["truth"]), ("cell", ["size_class", "powertrain"])):
        for key, g in d.groupby(keys, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            per_vehicle = g.groupby("vehicle_id").size()
            rows.append(dict(level=level, key=" | ".join(str(x) for x in key), n_drives=int(len(g)), n_vehicles=int(len(per_vehicle)),
                             single_drive_vehicles=int((per_vehicle == 1).sum()), share_single_drive_vehicles=float((per_vehicle == 1).mean())))
    per_vehicle = d.groupby("vehicle_id").size()
    rows.append(dict(level="all", key="all", n_drives=int(len(d)), n_vehicles=int(len(per_vehicle)),
                     single_drive_vehicles=int((per_vehicle == 1).sum()), share_single_drive_vehicles=float((per_vehicle == 1).mean())))
    return pd.DataFrame(rows)


def study_vehicle_pooling(sub: pd.DataFrame, runs: list[dict], classes, label_col: str) -> pd.DataFrame:
    rows = []
    for r in runs:
        drv = units_of(sub, r["P"], classes, label_col)
        veh = units_of(sub, r["P"], classes, label_col, unit_col="vehicle_id")
        multi = veh[veh["n_drives"] >= 2]
        rows.append(dict(arch=r["arch"], seed=r["seed"], drive_macro_f1=f1_of(drv, classes), n_drives=int(len(drv)),
                         vehicle_macro_f1=f1_of(veh, classes), n_vehicles=int(len(veh)),
                         vehicle_multi_macro_f1=f1_of(multi, classes), n_vehicles_multi=int(len(multi))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- (d) evidence sufficiency
def drive_evidence(sub: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """Per drive (rows of `units`): n_windows and the window-index episode / lead-event counts summed over its windows."""
    g = sub.groupby(sub["drive_id"].astype(str))
    out = units[["unit", "truth", "pred", "n_windows"]].copy()
    for col in ("n_accel_ep", "n_decel_ep", "n_lead_accel_events"):
        out[col] = g[col].sum().reindex(out["unit"]).fillna(0).to_numpy(dtype=np.int64)
    out["correct"] = (out["truth"] == out["pred"]).to_numpy(bool)
    return out


def study_evidence(ev: pd.DataFrame, classes) -> pd.DataFrame:
    rows = []
    for q in EVIDENCE_QUANTITIES:
        v = ev[q].to_numpy(dtype=np.int64)
        for b in (EVIDENCE_BINS if q == "n_windows" else COUNT_BINS):
            m = in_bin(v, b)
            n = int(m.sum())
            k = int(ev["correct"].to_numpy()[m].sum())
            lo, hi = wilson_ci(k, n)
            rows.append(dict(quantity=q, bin=bin_label(b), n_drives=n, accuracy=(k / n if n else np.nan), ci_lo=lo, ci_hi=hi,
                             macro_f1=f1_of(ev[m], classes) if n else np.nan))
    return pd.DataFrame(rows)


def study_subsampling(sub: pd.DataFrame, P: np.ndarray, classes, label_col: str, seed: int = 0) -> pd.DataFrame:
    """Drives with >= 32 selected windows: mean probability over m randomly drawn windows -> accuracy and macro-F1,
    averaged over 50 repeats (plus the all-windows reference for the same drives)."""
    codes, uniq = pd.factorize(sub["drive_id"].astype(str).to_numpy(), sort=True)
    counts = np.bincount(codes, minlength=len(uniq))
    keep = np.flatnonzero(counts >= SUBSAMPLE_MIN_WINDOWS)
    rows = []
    if not len(keep):
        return pd.DataFrame(rows)
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    blocks = np.split(order, bounds)
    truth = class_index(sub[label_col].to_numpy(object)[[blocks[u][0] for u in keep]], classes)
    k = len(classes)
    rng = np.random.default_rng(seed)

    def score(pred: np.ndarray) -> tuple[float, float]:
        cm = np.zeros((k, k), dtype=np.int64)
        np.add.at(cm, (truth, pred), 1)
        return float((pred == truth).mean()), macro_f1_from_confusion(cm)

    full = score(np.array([P[blocks[u]].mean(0).argmax() for u in keep]))
    for m in SUBSAMPLE_M:
        accs, f1s = [], []
        for _ in range(SUBSAMPLE_REPEATS):
            pred = np.array([P[rng.choice(blocks[u], size=m, replace=False)].mean(0).argmax() for u in keep])
            a, f = score(pred)
            accs.append(a); f1s.append(f)
        rows.append(dict(m_windows=m, n_drives=int(len(keep)), accuracy=float(np.mean(accs)), accuracy_std=float(np.std(accs)),
                         macro_f1=float(np.mean(f1s)), macro_f1_std=float(np.std(f1s))))
    rows.append(dict(m_windows=0, n_drives=int(len(keep)), accuracy=full[0], accuracy_std=0.0, macro_f1=full[1], macro_f1_std=0.0))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- (e) upper bounds
def majority_prediction(units: pd.DataFrame, car_model: pd.Series, classes) -> np.ndarray:
    """Every unit gets the majority prediction of its car model (ties -> earlier class)."""
    pos = {c: i for i, c in enumerate(classes)}
    cm = car_model.reindex(units["unit"].astype(str)).to_numpy(object)
    codes, uniq = pd.factorize(np.array([str(x) for x in cm], dtype=object))
    counts = np.zeros((len(uniq), len(classes)))
    np.add.at(counts, (codes, [pos[p] for p in units["pred"]]), 1.0)
    maj = counts.argmax(1)
    return np.array([classes[maj[c]] for c in codes], dtype=object)


def footprint_noise_oracle(fp: np.ndarray, model_codes: np.ndarray, truth: np.ndarray, thresholds=C.FOOTPRINT_THRESHOLDS_M2,
                           sigmas=NOISE_LEVELS, taus=NOISE_LEVELS, n_rep: int = NOISE_REPEATS, seed: int = 0) -> pd.DataFrame:
    """Drive macro-F1 when the size class is cut from footprint + N(0, sigma) per drive + N(0, tau) per car model."""
    rng = np.random.default_rng(seed)
    k = 3
    n_models = int(model_codes.max()) + 1 if len(model_codes) else 0
    rows = []
    for sigma in sigmas:
        for tau in taus:
            f1s, accs = [], []
            for _ in range(n_rep if (sigma or tau) else 1):
                x = fp.copy()
                if sigma:
                    x = x + rng.normal(0.0, sigma, size=len(fp))
                if tau:
                    x = x + rng.normal(0.0, tau, size=n_models)[model_codes]
                pred = size_class_of(x, thresholds)
                cm = np.zeros((k, k), dtype=np.int64)
                np.add.at(cm, (truth, pred), 1)
                f1s.append(macro_f1_from_confusion(cm)); accs.append(float((pred == truth).mean()))
            rows.append(dict(sigma=sigma, tau=tau, macro_f1_mean=float(np.mean(f1s)), macro_f1_std=float(np.std(f1s)),
                             accuracy_mean=float(np.mean(accs)), n_drives=int(len(fp)), n_models=n_models))
    return pd.DataFrame(rows)


def apply_mass_cuts(mass: np.ndarray, c1: float, c2: float) -> np.ndarray:
    mass = np.asarray(mass, dtype=np.float64)
    return (mass > c1).astype(np.int64) + (mass > c2).astype(np.int64)


def fit_mass_cuts(mass: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """Two curb-mass cuts (midpoints between observed masses) maximising the three-class macro-F1 over the given
    drives; returns (c1, c2, macro_f1). NaN triple with fewer than three distinct masses."""
    mass = np.asarray(mass, dtype=np.float64)
    u = np.unique(mass[np.isfinite(mass)])
    if len(u) < 3:
        return (float("nan"), float("nan"), float("nan"))
    cands = (u[:-1] + u[1:]) / 2.0
    above = mass[None, :] > cands[:, None]                              # (n_cands, n_drives)
    best = (float("nan"), float("nan"), -1.0)
    k = 3
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            pred = above[i].astype(np.int64) + above[j].astype(np.int64)
            cm = np.zeros((k, k), dtype=np.int64)
            np.add.at(cm, (truth, pred), 1)
            f1 = macro_f1_from_confusion(cm)
            if f1 > best[2]:
                best = (float(cands[i]), float(cands[j]), f1)
    return best


def separability_table(heldout: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    """For every held-out car model (rows: car_model, size_class, footprint, n_drives, accuracy): the nearest
    differently-labelled training car model by footprint and the gap."""
    rows = []
    for r in heldout.itertuples(index=False):
        other = training[training["size_class"] != r.size_class]
        if len(other) and np.isfinite(r.footprint):
            gap = (other["footprint"] - r.footprint).abs()
            j = gap.idxmin()
            rows.append(dict(car_model=r.car_model, size_class=r.size_class, footprint=r.footprint, n_drives=r.n_drives, accuracy=r.accuracy,
                             majority_correct=r.majority_correct, nearest_other_class_model=other.loc[j, "car_model"],
                             nearest_other_class=other.loc[j, "size_class"], footprint_gap=float(gap.loc[j])))
        else:
            rows.append(dict(car_model=r.car_model, size_class=r.size_class, footprint=r.footprint, n_drives=r.n_drives, accuracy=r.accuracy,
                             majority_correct=r.majority_correct, nearest_other_class_model=None, nearest_other_class=None, footprint_gap=np.nan))
    return pd.DataFrame(rows).sort_values(["size_class", "footprint_gap"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- rendering
def pct(x) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * float(x):.1f}"


def md_table(header: list[str], rows: list[list]) -> list[str]:
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""])


def seed_mean(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    g = df.groupby(by, dropna=False)
    out = g[cols].mean().reset_index()
    out["n_seeds"] = g.size().to_numpy()
    return out


def render_block(split_col: str, task: str, classes, n_sub: int, runs: list[dict], ref_name: str, agg: pd.DataFrame, ens: pd.DataFrame,
                 counts: pd.DataFrame, pool: pd.DataFrame, evidence: pd.DataFrame, subsample: pd.DataFrame, majority: pd.DataFrame,
                 fp_noise: pd.DataFrame | None, mass: dict | None, sep: pd.DataFrame | None) -> list[str]:
    L = [f"## {task} / {split_col}", "",
         f"{len(runs)} saved runs ({len({r['arch'] for r in runs})} model families), {n_sub} test windows, "
         f"{int(agg[(agg['mask'] == 'all') & (agg['how'] == 'mean')]['n_drives'].max())} test drives with at least one window. "
         f"Reference predictions for (d) and (e): {ref_name}.", ""]
    # (a)
    L += ["### (a) Aggregation rule x window mask (drive macro-F1, mean over seeds)", ""]
    a_all = seed_mean(agg[agg["mask"] == "all"], ["arch", "how"], ["macro_f1"])
    archs = sorted(a_all["arch"].unique(), key=arch_key)
    rows = []
    for a in archs:
        s = a_all[a_all["arch"] == a].set_index("how")["macro_f1"]
        rows.append([a] + [pct(s.get(h)) for h in AGGREGATIONS] + [int(a_all[a_all["arch"] == a]["n_seeds"].iloc[0])])
    L += md_table(["model", *AGGREGATIONS, "seeds"], rows)
    a_mask = seed_mean(agg[agg["how"] == "mean"], ["arch", "mask"], ["macro_f1", "n_drives"])
    rows = []
    for a in archs:
        s = a_mask[a_mask["arch"] == a].set_index("mask")
        rows.append([a] + [f"{pct(s.loc[m, 'macro_f1'])} ({int(round(s.loc[m, 'n_drives']))})" if m in s.index else "-" for m in MASKS])
    L += ["Mean-probability rule under the window masks (drive macro-F1 (drives kept)): " +
          "; ".join(f"`{m}` = {d}" for m, d in MASKS.items()) + ".", ""]
    L += md_table(["model", *MASKS], rows)
    # (b)
    L += ["### (b) Ensembles (macro-F1)", ""]
    e = ens[ens["kind"] == "ensemble"]
    rows = [[r.ensemble, r.n_members, pct(r.window_macro_f1), pct(r.drive_macro_f1), pct(r.drive_vote_macro_f1), pct(r.stacked_drive_macro_f1),
             pct(r.member_drive_f1_mean), pct(r.member_drive_f1_max)] for r in e.itertuples(index=False)]
    L += md_table(["ensemble", "members", "window", "drive (mean prob)", "drive (vote over members)", "drive (stacked)", "members: mean drive", "members: best drive"], rows)
    if e["stacked_drive_macro_f1"].isna().all():
        L += ["Stacking skipped: no regenerated validation probabilities for a complete member set (run `experiments/regen_probs.py`).", ""]
    # (c)
    L += ["### (c) Vehicle pooling", ""]
    rows = [[r.level, r.key, r.n_drives, r.n_vehicles, r.single_drive_vehicles, pct(r.share_single_drive_vehicles)] for r in counts.itertuples(index=False)]
    L += md_table(["level", "class / cell", "drives", "vehicles", "single-drive vehicles", "share single-drive (%)"], rows)
    p = seed_mean(pool, ["arch"], ["drive_macro_f1", "vehicle_macro_f1", "vehicle_multi_macro_f1", "n_drives", "n_vehicles", "n_vehicles_multi"])
    rows = [[r.arch, pct(r.drive_macro_f1), f"{pct(r.vehicle_macro_f1)} ({int(round(r.n_vehicles))})",
             f"{pct(r.vehicle_multi_macro_f1)} ({int(round(r.n_vehicles_multi))})", int(r.n_seeds)] for r in p.sort_values("arch", key=lambda s: s.map(arch_key)).itertuples(index=False)]
    L += md_table(["model", "drive_full", "vehicle_full (vehicles)", "vehicle_full_multi (vehicles)", "seeds"], rows)
    # (d)
    L += ["### (d) Evidence sufficiency (reference predictions)", "",
          "Drive accuracy with a Wilson 95 % interval by the amount of evidence in the drive (`n_windows` = selected windows; the "
          "other quantities are window-index counts summed over the drive's windows).", ""]
    for q in EVIDENCE_QUANTITIES:
        s = evidence[evidence["quantity"] == q]
        rows = [[r.bin, r.n_drives, pct(r.accuracy), f"{pct(r.ci_lo)}-{pct(r.ci_hi)}", pct(r.macro_f1)] for r in s.itertuples(index=False) if r.n_drives]
        L += [f"`{q}`", ""] + md_table(["bin", "drives", "accuracy", "95 % CI", "macro-F1"], rows)
    if len(subsample):
        rows = [[("all" if r.m_windows == 0 else r.m_windows), r.n_drives, f"{pct(r.accuracy)} ± {pct(r.accuracy_std)}", f"{pct(r.macro_f1)} ± {pct(r.macro_f1_std)}"]
                for r in subsample.itertuples(index=False)]
        L += [f"Subsampling m windows per drive (drives with >= {SUBSAMPLE_MIN_WINDOWS} windows, {SUBSAMPLE_REPEATS} repeats, mean ± std):", ""]
        L += md_table(["m", "drives", "accuracy", "macro-F1"], rows)
    # (e)
    L += ["### (e) Upper bounds", ""]
    m = seed_mean(majority[majority["arch"] != "reference"], ["arch"], ["drive_macro_f1", "majority_macro_f1"])
    rows = [[r.arch, pct(r.drive_macro_f1), pct(r.majority_macro_f1), int(r.n_seeds)] for r in m.sort_values("arch", key=lambda s: s.map(arch_key)).itertuples(index=False)]
    ref = majority[majority["arch"] == "reference"]
    if len(ref):
        rows.append(["reference", pct(ref["drive_macro_f1"].iloc[0]), pct(ref["majority_macro_f1"].iloc[0]), "-"])
    L += ["Per-car-model majority oracle: every drive of a car model receives the model's majority drive prediction "
          "(the ceiling of a predictor that is perfectly consistent within a car model).", ""]
    L += md_table(["model", "drive_full", "majority oracle", "seeds"], rows)
    if fp_noise is not None and len(fp_noise):
        L += ["Footprint-noise oracle (size task): the class cut from the true footprint plus per-drive noise sigma and per-car-model bias tau "
              f"(m^2, {NOISE_REPEATS} repeats), drive macro-F1:", ""]
        piv = fp_noise.pivot(index="sigma", columns="tau", values="macro_f1_mean")
        rows = [[f"sigma {s:.2f}"] + [pct(piv.loc[s, t]) for t in piv.columns] for s in piv.index]
        L += md_table(["", *[f"tau {t:.2f}" for t in piv.columns]], rows)
    if mass is not None:
        L += [f"Curb-mass oracle (size task): two mass cuts fitted on {mass['n_train_drives']} training drives "
              f"({mass['n_train_models']} car models; training macro-F1 {pct(mass['train_macro_f1'])}), applied to "
              f"{mass['n_test_drives']} test drives with a known mass ({mass['n_test_without_mass']} without): drive macro-F1 "
              f"{pct(mass['test_macro_f1'])}, accuracy {pct(mass['test_accuracy'])}.", ""]
    if sep is not None and len(sep):
        near = int((sep["footprint_gap"] < NEAR_GAP_M2).sum())
        L += [f"Separability of the {len(sep)} held-out car models (size task): nearest differently-labelled training car model by footprint; "
              f"{near} of {len(sep)} lie within {NEAR_GAP_M2} m^2 of one. Accuracy = share of the model's drives the reference predicts correctly.", ""]
        rows = [[r.car_model, r.size_class, f"{r.footprint:.2f}", r.n_drives, pct(r.accuracy), "yes" if r.majority_correct else "no",
                 r.nearest_other_class_model, r.nearest_other_class, f"{r.footprint_gap:.2f}"] for r in sep.itertuples(index=False)]
        L += md_table(["held-out car model", "class", "footprint", "drives", "accuracy", "majority correct", "nearest other-class training model", "its class", "gap m^2"], rows)
    return L


def render(blocks: list[str], n_runs: int) -> str:
    head = ["# Oracle studies on the saved test predictions (post-hoc ceilings)", "",
            "Every number on this page is computed after the fact from the saved test-split predictions of the already-published models "
            f"({n_runs} runs; `experiments/RESULTS.md`) and from the public index tables; nothing was trained for it. These are **ceilings "
            "and diagnostics, not results**: because they look at the test split, none of them may select a model, a rule or a "
            "hyper-parameter. Any idea they suggest has to win on the internal model-disjoint folds first (`experiments/CAMPAIGN.md`). "
            "Macro-F1 in percent; `drive_full` etc. follow docs/PROTOCOLS.md (drive = mean class probability over the drive's test "
            "windows, arg-max). Car-model names are public labels of the dataset; no drive or vehicle identifier appears here.", "",
            "Aggregation rules: `mean` = mean probability (official); `logodds` = mean per-class log-odds (probabilities clipped at 1e-4); "
            "`median` = per-class median; `confidence` = mean weighted by (max p - 1/k); `trimmed` = 20 % trimmed mean per class; "
            "`vote` = majority of the per-window arg-max.", ""]
    return "\n".join(head + blocks)


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=EXP_ROOT / "results")
    ap.add_argument("--out-dir", type=Path, default=EXP_ROOT / "oracles", help="CSV output folder")
    ap.add_argument("--md", type=Path, default=REPO / "experiments" / "ORACLES.md")
    ap.add_argument("--split-cols", default=",".join(SPLIT_COLS))
    ap.add_argument("--tasks", default=",".join(TASK_LIST))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    root = Path(C.DATA_ROOT)
    win = pd.read_parquet(root / "build" / "index" / "windows.parquet", columns=WIN_COLS).rename(columns={"model": "car_model"})
    drives = pd.read_parquet(root / "build" / "index" / "drives.parquet", columns=DRIVE_COLS).rename(columns={"model": "car_model"})
    a.out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[pd.DataFrame]] = {}
    blocks: list[str] = []
    n_runs_total = 0

    def add(name: str, df: pd.DataFrame, split_col: str, task: str) -> None:
        if df is None or not len(df):
            return
        tables.setdefault(name, []).append(df.assign(split_col=split_col, task=task))

    for split_col in a.split_cols.split(","):
        for task in a.tasks.split(","):
            classes = list(TASKS[task]["classes"])
            label_col = TASKS[task]["label"]
            sub = win[protocol_mask(win, task, "test_full", split_col)].reset_index(drop=True)
            runs = align_runs(load_runs(a.results_dir, split_col, task), sub)
            if not runs:
                print(f"{task}/{split_col}: no saved runs cover the test windows", flush=True)
                continue
            n_runs_total += len(runs)
            y = class_index(sub[label_col].to_numpy(object), classes)
            agg = study_aggregation(sub, runs, classes, label_col)
            ens = study_ensembles(sub, runs, classes, label_col, y)
            counts = study_vehicle_counts(sub, label_col)
            pool = study_vehicle_pooling(sub, runs, classes, label_col)
            P_ref, ref_name = reference_probs(runs)
            ref_units = units_of(sub, P_ref, classes, label_col)
            ev = drive_evidence(sub, ref_units)
            evidence = study_evidence(ev, classes)
            subsample = study_subsampling(sub, P_ref, classes, label_col, seed=a.seed)
            car_of_drive = sub.groupby(sub["drive_id"].astype(str))["car_model"].first()
            maj_rows = []
            for r in runs:
                u = units_of(sub, r["P"], classes, label_col)
                maj = majority_prediction(u, car_of_drive, classes)
                maj_rows.append(dict(arch=r["arch"], seed=r["seed"], drive_macro_f1=f1_of(u, classes),
                                     majority_macro_f1=unit_metrics(u["truth"], maj, classes)["macro_f1"], n_drives=int(len(u)),
                                     n_models=int(car_of_drive.reindex(u["unit"]).nunique())))
            maj_ref = majority_prediction(ref_units, car_of_drive, classes)
            maj_rows.append(dict(arch="reference", seed=-1, drive_macro_f1=f1_of(ref_units, classes),
                                 majority_macro_f1=unit_metrics(ref_units["truth"], maj_ref, classes)["macro_f1"], n_drives=int(len(ref_units)),
                                 n_models=int(car_of_drive.reindex(ref_units["unit"]).nunique())))
            majority = pd.DataFrame(maj_rows)
            fp_noise = mass = sep = None
            if task == "size" and C.FOOTPRINT_THRESHOLDS_M2:
                dsub = sub.groupby(sub["drive_id"].astype(str)).agg(car_model=("car_model", "first"), footprint=("footprint_m2", "first"),
                                                                     truth=(label_col, "first")).reindex(ref_units["unit"])
                ok = dsub["footprint"].notna().to_numpy()
                codes, _ = pd.factorize(dsub["car_model"].astype(str).to_numpy()[ok])
                fp_noise = footprint_noise_oracle(dsub["footprint"].to_numpy(dtype=np.float64)[ok], codes,
                                                  class_index(dsub["truth"].to_numpy(object)[ok], classes), seed=a.seed)
                add("footprint_noise", fp_noise, split_col, task)
                # curb-mass oracle: cuts fitted on training drives, applied to the test drives of the drive_full population
                tr = drives[(drives[split_col] == "train") & drives["task_size_eligible"].fillna(False) & drives["size_class"].isin(classes)
                            & drives["curb_mass_kg"].notna()]
                c1, c2, f1_tr = fit_mass_cuts(tr["curb_mass_kg"].to_numpy(), class_index(tr["size_class"].to_numpy(object), classes))
                te = drives.set_index("drive_id").reindex(ref_units["unit"])
                has = te["curb_mass_kg"].notna().to_numpy()
                pred = apply_mass_cuts(te["curb_mass_kg"].to_numpy(dtype=np.float64)[has], c1, c2)
                truth = class_index(ref_units["truth"].to_numpy(object)[has], classes)
                cm = np.zeros((3, 3), dtype=np.int64)
                np.add.at(cm, (truth, pred), 1)
                mass = dict(cut1_kg=c1, cut2_kg=c2, train_macro_f1=f1_tr, n_train_drives=int(len(tr)), n_train_models=int(tr["car_model"].nunique()),
                            test_macro_f1=macro_f1_from_confusion(cm), test_accuracy=float((pred == truth).mean()) if has.any() else np.nan,
                            n_test_drives=int(has.sum()), n_test_without_mass=int((~has).sum()))
                add("mass_oracle", pd.DataFrame([mass]), split_col, task)
                if split_col == "split_platform":
                    per_model = ev.assign(car_model=car_of_drive.reindex(ev["unit"]).to_numpy(object), truth=ev["truth"]) \
                        .groupby("car_model").agg(size_class=("truth", "first"), n_drives=("unit", "size"), accuracy=("correct", "mean"))
                    per_model["footprint"] = dsub.groupby("car_model")["footprint"].median().reindex(per_model.index)
                    maj_correct = pd.Series(maj_ref == ref_units["truth"].to_numpy(object), index=ref_units["unit"]).groupby(
                        car_of_drive.reindex(ref_units["unit"]).to_numpy(object)).all()
                    per_model["majority_correct"] = maj_correct.reindex(per_model.index).fillna(False).astype(bool)
                    trn = drives[drives[split_col].isin(["train", "validation"]) & drives["task_size_eligible"].fillna(False)
                                 & drives["size_class"].isin(classes) & drives["footprint_m2"].notna()]
                    training = trn.groupby("car_model").agg(size_class=("size_class", "first"), footprint=("footprint_m2", "median")).reset_index()
                    sep = separability_table(per_model.reset_index(), training)
                    add("separability", sep, split_col, task)
            for name, df in (("aggregation", agg), ("ensembles", ens), ("vehicle_counts", counts), ("vehicle_pooling", pool),
                             ("evidence", evidence), ("subsample", subsample), ("majority_oracle", majority)):
                add(name, df, split_col, task)
            blocks += render_block(split_col, task, classes, len(sub), runs, ref_name, agg, ens, counts, pool, evidence, subsample, majority,
                                   fp_noise, mass, sep)
            print(f"{task}/{split_col}: runs={len(runs)} windows={len(sub)} drives={len(ref_units)} elapsed_s={time.time() - t0:.0f}", flush=True)
    if not blocks:
        print("nothing to report")
        return 2
    for name, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_csv(a.out_dir / f"{name}.csv", index=False)
    a.md.write_text(render(blocks, n_runs_total))
    print(f"wrote {a.md.relative_to(REPO) if a.md.is_relative_to(REPO) else a.md.name} and {len(tables)} CSV tables ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
