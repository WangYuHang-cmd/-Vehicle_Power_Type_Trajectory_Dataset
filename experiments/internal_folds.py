#!/usr/bin/env python3
"""Internal car-model-disjoint folds for model selection (the published validation split shares most car models with
the training split, so early stopping on it rewards memorising car models).

Pool = drives of `split_platform` in {train, validation}. Candidate models (per task) have >= 2 identity groups and
>= 100 eligible pool windows outside fleet groups (the windows a held-out model is validated on) and account for
<= 35 % of the pool windows of their size x powertrain cell (so a cell's dominant model is never held out). Three folds
of held-out models, balanced over the six size x {BEV, ICE} cells by validated hours (hybrids / unknown powertrains
count with the combustion bucket), each fold holding >= 2 models whose footprint lies within 0.15 m^2 of a size
threshold. Fold k: validation = pool drives of the held-out models; training = the rest of the pool minus every drive
whose identity group also drove a held-out model, so an identity group never sits on both sides of a fold (as in the
published splits, whose test groups never appear in train or validation).

Fleet rule: an identity group whose drives span >= `fleet_min_models` car models is a shared device pool (the lab
fleet: one group with 19 car models, 33 vehicles and about 480 h, all in the training split), not an owner whose
habits could stand in for a car model. The literal group rule would remove it from the training set of every fold
(13-17 % of the pool windows never trained on in any fold, all from that one group). Such a group is pinned to the
training side instead: its drives of held-out models leave the fold entirely (neither trained on nor validated), its
other drives stay in training. `fleet_min_models=0` restores the literal rule.

Deterministic given the seed (ties by vtt.splits.tie_hash). Reads build/index/{drives,windows}.parquet, writes
<data-root>/experiments/internal_folds_<task>.json and experiments/INTERNAL_FOLDS.md (model names, hours and counts only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402
from vtt.splits import tie_hash  # noqa: E402
from experiments.data import TASKS, WINDOW_H  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
POOL_SPLITS = ("train", "validation")
SPLIT_COL = "split_platform"
PARAMS = dict(n_folds=3, min_groups=2, min_windows=100, max_cell_share=0.35, boundary_tol_m2=0.15, min_boundary_models=2,
              fleet_min_models=5, thresholds_m2=list(C.FOOTPRINT_THRESHOLDS_M2 or ()), pool_splits=list(POOL_SPLITS), split_col=SPLIT_COL,
              hours="eligible pool windows x 60 s", val_hours="eligible pool windows outside fleet groups x 60 s",
              cell_for_cap="dominant size_class x powertrain (4 levels, NA) of the model's eligible windows",
              cell_for_balance="size_class x {BEV, ICE}; HEV / PHEV / unknown count as ICE; balanced by val_hours",
              fleet_rule="an identity group spanning >= fleet_min_models car models stays on the training side: its drives of held-out "
                         "models are dropped from the fold (neither trained on nor validated), its other drives train; min_windows counts "
                         "the windows outside fleet groups; 0 disables the rule")


def folds_path(task: str, data_root: Path | None = None) -> Path:
    return Path(data_root or C.DATA_ROOT) / "experiments" / f"internal_folds_{task}.json"


def _clean(x) -> str:
    return "NA" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)


def fleet_groups(drives_df: pd.DataFrame, min_models: int | None) -> set[str]:
    """Identity groups whose drives span >= min_models distinct car models (a shared device pool); empty when min_models
    is 0 / None. Works on any table with group_id and model columns (drives index or windows table)."""
    if not min_models:
        return set()
    n_models = drives_df.groupby(drives_df["group_id"].astype(str))["model"].nunique()
    return set(n_models.index[n_models.to_numpy() >= int(min_models)])


def pool_windows(task: str, win: pd.DataFrame) -> pd.DataFrame:
    """Eligible windows of the task inside the pool (train + validation of split_platform)."""
    t = TASKS[task]
    m = win[SPLIT_COL].isin(POOL_SPLITS).to_numpy()
    m &= win["eligible"].fillna(False).to_numpy(bool)
    m &= win[t["eligible"]].fillna(False).to_numpy(bool)
    m &= win[t["label"]].isin(t["classes"]).to_numpy()
    m &= win["model"].notna().to_numpy()
    return win.loc[m]


def model_table(task: str, win: pd.DataFrame, params: dict = PARAMS, drives: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per car model of the pool: windows (all / outside fleet groups), hours, groups, drives, dominant cell, cell
    share, footprint, flags. Fleet groups are detected on `drives` (the drives index) when given, else on the windows."""
    w = pool_windows(task, win)
    fleet = fleet_groups(drives if drives is not None else win, params.get("fleet_min_models"))
    cell4 = w["size_class"].map(_clean) + "|" + w["powertrain"].map(_clean)
    w = w.assign(cell4=cell4.to_numpy(), in_fleet=w["group_id"].astype(str).isin(fleet).to_numpy())
    cell_total = w.groupby("cell4").size()
    rows = []
    for model, sub in w.groupby("model"):
        dom = sub["cell4"].value_counts().index[0]
        n_in_cell = int((sub["cell4"] == dom).sum())
        size, pt = dom.split("|")
        fp = float(sub["footprint_m2"].median()) if sub["footprint_m2"].notna().any() else float("nan")
        thr = params["thresholds_m2"]
        boundary = bool(np.isfinite(fp) and thr and min(abs(fp - t) for t in thr) <= params["boundary_tol_m2"])
        n_val = int((~sub["in_fleet"]).sum())
        rows.append(dict(model=str(model), n_windows=int(len(sub)), n_windows_val=n_val, hours=float(len(sub) * WINDOW_H),
                         val_hours=float(n_val * WINDOW_H), n_groups=int(sub["group_id"].nunique()), n_drives=int(sub["drive_id"].nunique()),
                         cell=dom, cell_share=n_in_cell / float(cell_total[dom]), cell6=f"{size}|{'BEV' if pt == 'BEV' else 'ICE'}",
                         footprint_m2=fp, boundary=boundary))
    df = pd.DataFrame(rows)
    df["candidate"] = (df["n_groups"] >= params["min_groups"]) & (df["n_windows_val"] >= params["min_windows"]) & (df["cell_share"] <= params["max_cell_share"])
    return df.sort_values(["hours", "model"], ascending=[False, True]).reset_index(drop=True)


def assign_folds(table: pd.DataFrame, seed: int = 0, params: dict = PARAMS) -> list[list[str]]:
    """Deterministic assignment of the candidate models to n_folds folds: boundary models first, by validated hours
    descending, each to the fold with the fewest boundary models (so every fold gets its minimum; ties: least validated
    hours of the model's cell, then least total hours, then tie_hash), then every six-cell in turn, each model to the fold
    with the least validated hours of that cell (ties: least total validated hours, then tie_hash)."""
    n_folds = params["n_folds"]
    cand = table[table["candidate"]].copy()
    cand["h"] = [tie_hash(seed, m) for m in cand["model"]]
    cand = cand.sort_values(["val_hours", "h"], ascending=[False, True])
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    hours_tot = np.zeros(n_folds)
    hours_cell: dict[str, np.ndarray] = {}
    n_boundary = np.zeros(n_folds, dtype=int)

    def put(k: int, row) -> None:
        folds[k].append(row.model)
        hours_tot[k] += row.val_hours
        hours_cell.setdefault(row.cell6, np.zeros(n_folds))[k] += row.val_hours
        n_boundary[k] += int(row.boundary)

    def pick(keys: list[np.ndarray], row) -> int:
        order = sorted(range(n_folds), key=lambda k: (*[float(a[k]) for a in keys], hours_tot[k], tie_hash(seed, k, row.model)))
        return order[0]

    for row in cand[cand["boundary"]].itertuples():
        put(pick([n_boundary.astype(float), hours_cell.setdefault(row.cell6, np.zeros(n_folds))], row), row)
    for cell in sorted(cand["cell6"].unique()):
        for row in cand[(cand["cell6"] == cell) & ~cand["boundary"]].itertuples():
            put(pick([hours_cell.setdefault(cell, np.zeros(n_folds))], row), row)
    short = [k for k in range(n_folds) if n_boundary[k] < params["min_boundary_models"]]
    if short:
        raise ValueError(f"{len(short)} fold(s) hold fewer than {params['min_boundary_models']} near-threshold models")
    return [sorted(f) for f in folds]


def fold_drive_sets(drives_df: pd.DataFrame, folds: dict, k: int) -> tuple[set[str], set[str]]:
    """(train_drive_ids, val_drive_ids) of fold k. Pool = drives with split_platform in {train, validation}; validation
    = pool drives of a held-out model; training = pool minus validation minus every drive whose identity group also has
    a drive (anywhere in the index) of a held-out model. Fleet rule (folds["params"]["fleet_min_models"], default
    PARAMS): a group spanning that many car models stays on the training side: its drives of held-out models belong to
    neither set, its other drives train."""
    held = set(folds["folds"][k]["heldout_models"])
    split_col = folds.get("split_col", SPLIT_COL)
    fleet = fleet_groups(drives_df, (folds.get("params") or {}).get("fleet_min_models", PARAMS["fleet_min_models"]))
    pool = drives_df[drives_df[split_col].isin(POOL_SPLITS)]
    gid = pool["group_id"].astype(str)
    in_fleet = gid.isin(fleet).to_numpy()
    is_held = pool["model"].isin(held).to_numpy()
    val = set(pool.loc[is_held & ~in_fleet, "drive_id"].astype(str))
    tainted = set(drives_df.loc[drives_df["model"].isin(held), "group_id"].astype(str)) - fleet
    keep = ~is_held & ~gid.isin(tainted).to_numpy()
    train = set(pool.loc[keep, "drive_id"].astype(str))
    return train, val


def fold_window_masks(win_df: pd.DataFrame, folds: dict, k: int, drives_df: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(train_mask, val_mask) over the rows of a windows table, by drive membership (fold_drive_sets). Without a drives
    table the drive memberships (and the fleet groups) are derived from the windows table itself (drive_id, group_id,
    model, split column)."""
    if drives_df is None:
        cols = ["drive_id", "group_id", "model", folds.get("split_col", SPLIT_COL)]
        drives_df = win_df[cols].drop_duplicates("drive_id")
    train, val = fold_drive_sets(drives_df, folds, k)
    d = win_df["drive_id"].astype(str)
    return d.isin(train).to_numpy(), d.isin(val).to_numpy()


def build_folds(task: str, drives: pd.DataFrame, win: pd.DataFrame, seed: int = 0, params: dict = PARAMS) -> tuple[dict, pd.DataFrame]:
    """The fold file of one task (dict) and its model table. Besides the contract keys every fold records its window
    counts (val / train / dropped, of which fleet) and the params carry the pool summary: fleet groups, pool windows never
    trained on in any fold (also under the literal group rule, for the record) and pool windows in no fold at all."""
    table = model_table(task, win, params, drives)
    assigned = assign_folds(table, seed, params)
    pw = pool_windows(task, win)
    fleet = fleet_groups(drives, params.get("fleet_min_models"))
    pool_drives = drives[drives[SPLIT_COL].isin(POOL_SPLITS)]
    pool_gid = pool_drives["group_id"].astype(str)
    pw_drive = pw["drive_id"].astype(str)
    pw_fleet = pw["group_id"].astype(str).isin(fleet).to_numpy()
    n_trained = np.zeros(len(pw), dtype=int)
    n_trained_literal = np.zeros(len(pw), dtype=int)
    n_used = np.zeros(len(pw), dtype=int)
    out_folds = []
    for k, models in enumerate(assigned):
        stub = {"folds": [{"heldout_models": models}] * (k + 1), "split_col": SPLIT_COL, "params": {"fleet_min_models": params.get("fleet_min_models")}}
        tr, va = fold_drive_sets(drives, stub, k)
        tr_lit, _ = fold_drive_sets(drives, dict(stub, params={"fleet_min_models": 0}), k)
        in_tr, in_va = pw_drive.isin(tr).to_numpy(), pw_drive.isin(va).to_numpy()
        n_trained += in_tr
        n_trained_literal += pw_drive.isin(tr_lit).to_numpy()
        n_used += in_tr | in_va
        dropped = ~in_tr & ~in_va
        sub = table[table["model"].isin(models)]
        out_folds.append(dict(fold=k, heldout_models=models, n_heldout_drives=len(va), n_train_drives=len(tr),
                              n_dropped_drives=int(len(pool_drives) - len(va) - len(tr)),
                              n_val_windows=int(in_va.sum()), n_train_windows=int(in_tr.sum()), n_dropped_windows=int(dropped.sum()),
                              n_fleet_dropped_windows=int((dropped & pw_fleet).sum()),
                              heldout_hours=float(sub["hours"].sum()), heldout_val_hours=float(sub["val_hours"].sum()),
                              n_boundary_models=int(sub["boundary"].sum()),
                              hours_by_cell={c: float(h) for c, h in sub.groupby("cell6")["hours"].sum().items()},
                              val_hours_by_cell={c: float(h) for c, h in sub.groupby("cell6")["val_hours"].sum().items()}))
    never_lit = n_trained_literal == 0
    summary = dict(n_fleet_groups=len(fleet), n_fleet_pool_groups=len(fleet & set(pool_gid)), n_fleet_pool_drives=int(pool_gid.isin(fleet).sum()),
                   n_fleet_pool_windows=int(pw_fleet.sum()), n_pool_windows_never_trained=int((n_trained == 0).sum()),
                   never_trained_share=float((n_trained == 0).mean()) if len(pw) else 0.0,
                   n_pool_windows_never_trained_literal_rule=int(never_lit.sum()),
                   never_trained_share_literal_rule=float(never_lit.mean()) if len(pw) else 0.0,
                   n_never_trained_literal_rule_in_fleet=int((never_lit & pw_fleet).sum()),
                   n_pool_windows_in_no_fold=int((n_used == 0).sum()))
    res = dict(task=task, split_col=SPLIT_COL, seed=seed, n_folds=params["n_folds"], folds=out_folds,
               params=dict(params, n_pool_models=int(len(table)), n_candidates=int(table["candidate"].sum()),
                           n_pool_windows=int(len(pw)), pool_hours=float(len(pw) * WINDOW_H), **summary))
    return res, table


def load_folds(task: str, data_root: Path | None = None) -> dict:
    return json.loads(folds_path(task, data_root).read_text())


def _cell_label(cell: str) -> str:
    return cell.replace("|", " / ")            # the bar would end a Markdown table cell


def render_md(results: dict[str, tuple[dict, pd.DataFrame]]) -> str:
    L = ["# Internal car-model-disjoint folds", "",
         "Selection folds for the method study (hyper-parameters, early stopping, aggregation and decision rules are chosen "
         "on these, never on the test split). Pool = drives of `split_platform` in {train, validation}. A fold holds out "
         "whole car models; its training set is the rest of the pool minus every drive whose identity group also drove a "
         "held-out model, so an identity group never sits on both sides of a fold (the published splits keep test groups "
         "out of train and validation in the same way). Candidates: >= 2 identity groups, >= 100 eligible windows outside "
         "fleet groups, <= 35 % of their size x powertrain cell's pool windows (so a cell's dominant model, e.g. the pickup / "
         "SUV pair that carries most LARGE hours or the compact that carries most SMALL hours, is never held out). Folds "
         "are balanced over the six size x {BEV, ICE} cells by validated hours (hybrids and unknown powertrains count with "
         "the combustion bucket) and each holds >= 2 models whose footprint lies within 0.15 m^2 of a size threshold "
         "(`near` column). Hours = eligible pool windows x 60 s; validated hours leave out the fleet groups' windows.", "",
         "**Fleet rule.** One identity group of the pool is a shared device pool (the lab fleet: 19 car models, 33 vehicles, "
         "about 480 h, every drive in the training split), not an owner whose habits could stand in for a car model. Under "
         "the literal group rule every fold holds out some of its car models, so the whole group leaves the training set of "
         "every fold and 13-17 % of the pool windows are never trained on (100 % of them from this one group), while the "
         "fold validation sets still contain its drives: a population shift inside every fold and a fold training population "
         "unlike the final full-train run. Groups spanning >= 5 car models are therefore pinned to the training side: their "
         "drives of held-out models are dropped from the fold (neither trained on nor validated; they train in the other "
         "folds), their other drives stay in training. Validation keeps only drives of the other identity groups, which is "
         "what the test split holds as well. `internal_folds.py --fleet-min-models 0` rebuilds the literal rule.", ""]
    for task, (res, table) in results.items():
        p = res["params"]
        share = 100.0 * p["n_fleet_pool_windows"] / max(1, p["n_pool_windows"])
        L += [f"## Task `{task}`", "",
              f"Pool: {p['n_pool_windows']:,} eligible windows ({p['pool_hours']:.0f} h) from {p['n_pool_models']} car models; "
              f"{p['n_candidates']} candidate models; seed {res['seed']}; {res['n_folds']} folds.", "",
              f"Fleet groups in the pool: {p['n_fleet_pool_groups']} (>= {p['fleet_min_models']} car models; {p['n_fleet_pool_drives']} drives, "
              f"{p['n_fleet_pool_windows']:,} windows = {share:.1f} % of the pool). Pool windows never trained on in any fold: "
              f"{p['n_pool_windows_never_trained']:,} ({100 * p['never_trained_share']:.1f} %), against "
              f"{p['n_pool_windows_never_trained_literal_rule']:,} ({100 * p['never_trained_share_literal_rule']:.1f} %, "
              f"{p['n_never_trained_literal_rule_in_fleet']:,} of them in fleet groups) under the literal group rule; "
              f"pool windows in no fold at all (neither trained on nor validated anywhere): {p['n_pool_windows_in_no_fold']:,}.", ""]
        L += ["| fold | held-out models | hours (validated) | drives (val / train / dropped) | windows (val / train / dropped, of which fleet) | near-threshold models | validated hours by cell |",
              "|---|---|---|---|---|---|---|"]
        for f in res["folds"]:
            cells = ", ".join(f"{_cell_label(c)} {h:.0f}" for c, h in sorted(f["val_hours_by_cell"].items()))
            L.append(f"| {f['fold']} | {len(f['heldout_models'])} | {f['heldout_hours']:.0f} ({f['heldout_val_hours']:.0f}) | "
                     f"{f['n_heldout_drives']} / {f['n_train_drives']} / {f['n_dropped_drives']} | "
                     f"{f['n_val_windows']:,} / {f['n_train_windows']:,} / {f['n_dropped_windows']:,}, {f['n_fleet_dropped_windows']:,} | "
                     f"{f['n_boundary_models']} | {cells} |")
        L.append("")
        for f in res["folds"]:
            L += [f"### Fold {f['fold']}", "", "| car model | cell | hours | validated hours | groups | drives | footprint m^2 | near |", "|---|---|---|---|---|---|---|---|"]
            sub = table[table["model"].isin(f["heldout_models"])].sort_values(["cell6", "val_hours"], ascending=[True, False])
            for r in sub.itertuples():
                fp = f"{r.footprint_m2:.2f}" if np.isfinite(r.footprint_m2) else "-"
                L.append(f"| {r.model} | {_cell_label(r.cell6)} | {r.hours:.1f} | {r.val_hours:.1f} | {r.n_groups} | {r.n_drives} | {fp} | {'yes' if r.boundary else ''} |")
            L.append("")
        never = table[~table["candidate"]].sort_values("hours", ascending=False)
        L += ["### Never held out (not a candidate)", "", "| car model | cell | hours | groups | windows (outside fleet groups) | cell share |", "|---|---|---|---|---|---|"]
        for r in never.head(25).itertuples():
            L.append(f"| {r.model} | {_cell_label(r.cell)} | {r.hours:.1f} | {r.n_groups} | {r.n_windows:,} ({r.n_windows_val:,}) | {100 * r.cell_share:.0f} % |")
        if len(never) > 25:
            L.append(f"| ... {len(never) - 25} more models with fewer hours | | | | | |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="size,bev_ice")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fleet-min-models", type=int, default=PARAMS["fleet_min_models"],
                    help="identity groups spanning this many car models stay on the training side (0 = literal group rule)")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=REPO / "experiments" / "INTERNAL_FOLDS.md")
    a = ap.parse_args(argv)
    root = Path(a.data_root or C.DATA_ROOT)
    params = dict(PARAMS, fleet_min_models=int(a.fleet_min_models))
    drives = pd.read_parquet(root / "build" / "index" / "drives.parquet")
    win = pd.read_parquet(root / "build" / "index" / "windows.parquet")
    results = {}
    for task in a.tasks.split(","):
        res, table = build_folds(task, drives, win, seed=a.seed, params=params)
        path = folds_path(task, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=1))
        results[task] = (res, table)
        p = res["params"]
        print(f"[{task}] candidates={p['n_candidates']}/{p['n_pool_models']} models; fleet_pool_groups={p['n_fleet_pool_groups']} "
              f"never_trained={p['n_pool_windows_never_trained']} ({100 * p['never_trained_share']:.1f} %, literal rule "
              f"{100 * p['never_trained_share_literal_rule']:.1f} %); folds: " +
              "; ".join(f"k{f['fold']}: {len(f['heldout_models'])} models {f['heldout_val_hours']:.0f} h val_windows={f['n_val_windows']} "
                        f"train_windows={f['n_train_windows']} dropped={f['n_dropped_windows']} near={f['n_boundary_models']}" for f in res["folds"]), flush=True)
    a.out_md.write_text(render_md(results))
    print(f"wrote {a.out_md.relative_to(REPO)} and {len(results)} fold file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
