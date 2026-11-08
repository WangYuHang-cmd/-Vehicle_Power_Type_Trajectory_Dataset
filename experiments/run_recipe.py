#!/usr/bin/env python3
"""Resumable runner of the recipe grid on the internal model-disjoint folds, and the renderer of experiments/RECIPE.md.

Stages (plan part II, stage 2): `sweep` runs the recipe ladder of one task on one fold (S0-S9 for size, B0-B5 for
BEV vs ICE); `probes` runs leave-one-car-model-out probes (L1-L4: Honda Odyssey, Hyundai Palisade, Hyundai Sonata,
Tesla Model 3) as fold definitions passed through the `folds` argument of experiments.train.run; `folds` runs the
best recipe so far (or --recipes) on the three folds with the GRU and the ResNet. Every configuration is a dict of
run() keyword arguments; a finished configuration (its results JSON exists) is skipped. Results and out-of-fold
probabilities are written by run() to <data-root>/experiments/recipe/results/<tag>.json and <tag>_oof.npz with
tag = fold<k>__<task>__<model>__<recipe_id>__s<seed>; the test split is never evaluated here (eval_test=False).
--dry-run prints the configurations without importing torch; --render rewrites RECIPE.md from the saved JSON files.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXP_ROOT = Path(C.DATA_ROOT) / "experiments"
RECIPE_DIR = EXP_ROOT / "recipe"
RESULTS_DIR = RECIPE_DIR / "results"
SPLIT_COL = "split_platform"
N_FOLDS = 3
FOLD_MODELS = ("gru", "resnet")
SELECT_ON = "inner_drive_f1"
AUG = "flip,mask,chdrop,noise"
RENDER_MARK = "<!-- rendered by experiments/run_recipe.py: everything below this line is overwritten -->"

# contract (5): allowed values of the run() options
ALLOWED = dict(aux={"cls", "coral", "fp", "mass", "pt4"}, sampling={"class", "class_model"}, augment={"flip", "mask", "chdrop", "noise", "cutmix"},
               decision={"softmax", "coral", "fp", "avg"}, select_on={"val_window_f1", "inner_drive_f1", "inner_window_f1"})

# recipe ladder: (id, change relative to the previous row, run() keyword arguments). S2-S6 are cumulative, S7-S9 are S5 + one change.
_S1 = dict(sampling="class_model")
_S2 = dict(_S1, aux="cls+fp")
_S3 = dict(_S1, aux="cls+fp+coral")
_S4 = dict(_S1, aux="cls+fp+coral+mass+pt4")
_S5 = dict(_S4, augment=AUG)
_B1 = dict(sampling="class_model")
_B2 = dict(_B1, aux="cls+fp")
_B3 = dict(_B1, aux="cls+fp+mass+pt4", include_hybrids=True)      # the four-class head trains on HEV / PHEV windows too
_B4 = dict(_B3, augment=AUG)
RECIPES: dict[str, list[tuple[str, str, dict]]] = {
    "size": [
        ("S0", "reference recipe (class-balanced sampling, cross-entropy only)", {}),
        ("S1", "sampling by car model inside each class (sampling=class_model)", _S1),
        ("S2", "S1 + footprint regression head (aux fp)", _S2),
        ("S3", "S2 + CORAL ordinal head", _S3),
        ("S4", "S3 + curb-mass and four-class powertrain heads (mass, pt4)", _S4),
        ("S5", "S4 + physics-safe augmentation (flip, mask, chdrop, noise)", _S5),
        ("S6", "S5 + same-class cutmix", dict(_S5, augment=AUG + ",cutmix")),
        ("S7", "S5 + DAS down-weighting (das_beta 0.5)", dict(_S5, das_beta=0.5)),
        ("S8", "S5 + conditional adversarial car-model head (adv_lambda 0.1)", dict(_S5, adv_lambda=0.1)),
        ("S9", "S5 + group-DRO-lite", dict(_S5, dro=True)),
    ],
    "bev_ice": [
        ("B0", "reference recipe (class-balanced sampling, cross-entropy only)", {}),
        ("B1", "sampling by car model inside each class (sampling=class_model)", _B1),
        ("B2", "B1 + footprint regression head (aux fp)", _B2),
        ("B3", "B2 + curb-mass and four-class powertrain heads (mass, pt4)", _B3),
        ("B4", "B3 + physics-safe augmentation (flip, mask, chdrop, noise)", _B4),
        ("B5", "B4 + DAS down-weighting (das_beta 0.5)", dict(_B4, das_beta=0.5)),
    ],
}
PROBES = (("L1", "Honda Odyssey"), ("L2", "Hyundai Palisade"), ("L3", "Hyundai Sonata"), ("L4", "Tesla Model 3"))

HEADER = """# Recipe grid on the internal model-disjoint folds

Window-model recipes of plan part II stage 2, trained and selected on the internal car-model-disjoint folds
(experiments/INTERNAL_FOLDS.md); the published validation split shares most car models with the training split, so
nothing here looks at it or at the test split. Every row is one run of `experiments.train.run` with the keyword
arguments printed by `experiments/run_recipe.py --dry-run`; `inner drive_full F1` is the macro-F1 of the held-out
fold's drives (mean class probability over the drive's windows), `inner window F1` the window-level macro-F1 of the
same fold, both at the selected epoch. The ladder is cumulative from S1 to S6; S7-S9 each add one change to S5. B0-B5
are the BEV vs ICE counterparts (no ordinal head, no cutmix: the task is binary and the LARGE-class composition problem
does not apply). L1-L4 are leave-one-car-model-out probes implemented as single-model folds (the probe car model is
held out of train and validation, every drive of an identity group that drove it is removed from the training set).
Stage `folds` runs the best recipe on all three folds with the GRU and the ResNet to produce the out-of-fold
probabilities used by experiments/aggregate.py.
"""


# ----------------------------------------------------------------------------- configurations
def check_kwargs(kw: dict) -> None:
    """Raise when a run() option is outside the values of contract (5)."""
    for key, sep in (("aux", "+"), ("augment", ",")):
        if key in kw and kw[key]:
            bad = set(kw[key].split(sep)) - ALLOWED[key]
            if bad:
                raise ValueError(f"{key}: unknown values {sorted(bad)}")
    for key in ("sampling", "decision", "select_on"):
        if key in kw and kw[key] not in ALLOWED[key]:
            raise ValueError(f"{key}: unknown value {kw[key]!r}")


def recipe_table(task: str) -> dict[str, tuple[str, dict]]:
    return {rid: (changes, dict(kw)) for rid, changes, kw in RECIPES[task]}


def probe_folds(task: str, drives=None) -> dict:
    """Fold definitions of contract (3) with one held-out car model per fold (drive counts filled from the drives index
    when it is given, -1 otherwise). The counts follow experiments.internal_folds.fold_drive_sets: pool = train +
    validation drives, held out = pool drives of the probe model, training = the rest minus every identity group that
    has a drive of the probe model anywhere in the index (test split included)."""
    folds = []
    for k, (rid, model) in enumerate(PROBES):
        n_held = n_train = -1
        if drives is not None:
            pool = drives[drives[SPLIT_COL].isin(["train", "validation"])]
            held = pool["model"] == model
            tainted = pool["group_id"].isin(drives.loc[drives["model"] == model, "group_id"])
            n_held, n_train = int(held.sum()), int((~held & ~tainted).sum())
        folds.append(dict(fold=k, probe=rid, heldout_models=[model], n_heldout_drives=n_held, n_train_drives=n_train))
    return dict(task=task, split_col=SPLIT_COL, seed=0, n_folds=len(PROBES), folds=folds, params=dict(kind="leave_one_model_out_probe"))


def make_tag(fold: int, task: str, model: str, recipe_id: str, seed: int) -> str:
    return f"fold{fold}__{task}__{model}__{recipe_id}__s{seed}"


def configs(stage: str, task: str, fold: int | None = None, models=("gru",), recipes=None, seed: int = 0, probe_base: str | None = None,
            select_on: str = SELECT_ON, epochs: int | None = None, limit_train: int = 0, drives=None, results=None, limit_val: int = 0) -> list[dict]:
    """The run configurations of one stage: dicts with id, changes, task, model, fold, seed, folds (probe definitions or
    None), tag and kwargs (keyword arguments of experiments.train.run beyond the positional ones)."""
    table = recipe_table(task)
    common = dict(select_on=select_on)
    if epochs:
        common["epochs"] = int(epochs)
    if limit_train:
        common["limit_train"] = int(limit_train)
    if limit_val:
        common["limit_val"] = int(limit_val)
    out = []
    if stage == "sweep":
        k = 0 if fold is None else fold
        ids = recipes or list(table)
        for model in models:
            for rid in ids:
                changes, kw = table[rid]
                out.append(dict(id=rid, changes=changes, task=task, model=model, fold=k, seed=seed, folds=None, kwargs=dict(common, **kw)))
    elif stage == "folds":
        ids = recipes or [best_recipe(results or [], task) or list(table)[0]]
        ks = [fold] if fold is not None else list(range(N_FOLDS))
        for model in models:
            for rid in ids:
                changes, kw = table[rid]
                for k in ks:
                    out.append(dict(id=rid, changes=changes, task=task, model=model, fold=k, seed=seed, folds=None, kwargs=dict(common, **kw)))
    elif stage == "probes":
        base = probe_base or best_recipe(results or [], task) or list(table)[0]
        changes, kw = table[base]
        pf = probe_folds(task, drives)
        for model in models:
            for k, (rid, car) in enumerate(PROBES):
                if fold is not None and k != fold:
                    continue
                out.append(dict(id=f"{rid}_{base}", changes=f"{base} with {car} held out", task=task, model=model, fold=k, seed=seed, folds=pf,
                                kwargs=dict(common, **kw)))
    else:
        raise ValueError(f"unknown stage {stage!r}")
    for c in out:
        check_kwargs(c["kwargs"])
        c["tag"] = make_tag(c["fold"], c["task"], c["model"], c["id"], c["seed"])
    return out


def describe(c: dict) -> str:
    kw = json.dumps(c["kwargs"], sort_keys=True)
    return f"{c['tag']}  [{c['id']}] {c['changes']}  kwargs={kw}" + (f"  heldout={c['folds']['folds'][c['fold']]['heldout_models']}" if c["folds"] else "")


# ----------------------------------------------------------------------------- execution
def report_path(results_dir: Path, md: Path | None = None) -> Path:
    """Where the table is rendered: `md` when given; the repository's experiments/RECIPE.md for the default results
    folder; otherwise <results-dir>/RECIPE.md, so a smoke run into another folder never overwrites the repository table."""
    if md is not None:
        return Path(md)
    return REPO / "experiments" / "RECIPE.md" if Path(results_dir).resolve() == RESULTS_DIR.resolve() else Path(results_dir) / "RECIPE.md"


def execute(cfgs: list[dict], results_dir: Path = RESULTS_DIR, in_ram: bool = True, log=print, render_md: Path | None = None,
            ckpt_dir: Path | None = None) -> int:
    """Run every configuration whose results JSON does not exist yet and re-render the table (report_path(results_dir,
    render_md)) after each run; returns the number of runs started."""
    from experiments.data import WindowData          # lazy: torch
    from experiments.train import run                 # lazy: torch; written concurrently (contract 5)
    params = inspect.signature(run).parameters
    needed = {"fold", "folds", "select_on", "eval_test", "recipe_id"}
    if not needed <= set(params) and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        raise SystemExit(f"experiments.train.run lacks the fold options {sorted(needed - set(params))} (contract 5)")
    results_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cfgs if not (results_dir / f"{c['tag']}.json").exists()]
    log(f"configs={len(cfgs)} done={len(cfgs) - len(todo)} todo={len(todo)}")
    if not todo:
        return 0
    data = WindowData(in_ram=in_ram)
    log(f"windows={len(data.win)} channels={len(data.channels)}")
    for i, c in enumerate(todo, 1):
        log(f"[{i}/{len(todo)}] start {c['tag']} {time.strftime('%H:%M')}")
        run(c["task"], SPLIT_COL, c["model"], c["seed"], data=data, log=log, out_dir=results_dir, fold=c["fold"], folds=c["folds"],
            eval_test=False, recipe_id=c["id"], **({"ckpt_dir": ckpt_dir} if ckpt_dir else {}), **c["kwargs"])
        report_path(results_dir, render_md).write_text(render(load_results(results_dir)))
    return len(todo)


# ----------------------------------------------------------------------------- report
def load_results(results_dir: Path = RESULTS_DIR) -> list[dict]:
    out = []
    for p in sorted(Path(results_dir).glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except ValueError:
            continue
        if isinstance(r, dict) and "task" in r:
            out.append(r)
    return out


def _inner(r: dict, level: str, key: str = "macro_f1"):
    return ((r.get("inner_val_metrics") or {}).get(level) or {}).get(key)


def _arch(r: dict) -> str:
    """Architecture name: experiments.train writes it as `arch` and keeps `model` = <arch>__<recipe_id> for fold runs."""
    return str(r.get("arch") or r.get("model"))


def best_recipe(results: list[dict], task: str, model: str | None = None) -> str | None:
    """Recipe id with the highest mean inner drive_full macro-F1 (window macro-F1 as fallback) among the ladder rows of the task."""
    ids = set(recipe_table(task))
    best, best_v = None, -1.0
    by: dict[str, list[float]] = {}
    for r in results:
        if r.get("task") != task or r.get("recipe_id") not in ids or (model and _arch(r) != model):
            continue
        v = _inner(r, "drive_full")
        v = _inner(r, "window") if v is None else v
        if v is not None:
            by.setdefault(r["recipe_id"], []).append(float(v))
    for rid, vals in by.items():
        if np.mean(vals) > best_v:
            best, best_v = rid, float(np.mean(vals))
    return best


def _agg(vals: list) -> str:
    v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    if not v:
        return "-"
    return f"{100 * np.mean(v):.1f}" if len(v) == 1 else f"{100 * np.mean(v):.1f} ± {100 * np.std(v):.1f}"


def render(results: list[dict]) -> str:
    """RECIPE.md: per task, per (model, fold) a table over the recipe ids present, then the probes."""
    L = [HEADER.rstrip("\n"), "", RENDER_MARK, ""]
    if not results:
        return "\n".join(L + ["No fold runs yet: run `experiments/run_recipe.py --stage sweep --task size`.", ""])
    for task, tname in (("size", "Size: SMALL / MID / LARGE"), ("bev_ice", "Powertrain: BEV vs ICE")):
        sub = [r for r in results if r.get("task") == task]
        if not sub:
            continue
        table = recipe_table(task)
        rec_cls = "LARGE" if task == "size" else "BEV"
        L += [f"## {tname}", ""]
        best = best_recipe(sub, task)
        if best:
            L += [f"Best recipe so far by inner drive_full macro-F1: `{best}` (used by `--stage folds` unless --recipes is given).", ""]
        ladder = [r for r in sub if r.get("recipe_id") in table]
        keys = sorted({(_arch(r), r.get("fold")) for r in ladder}, key=lambda t: (str(t[0]), -1 if t[1] is None else t[1]))
        for model, fold in keys:
            rs = [r for r in ladder if _arch(r) == model and r.get("fold") == fold]
            L += [f"### {model}, fold {fold}", "",
                  f"| id | changes | inner drive_full F1 | inner window F1 | {rec_cls} recall (drive) | decision | best epoch | minutes | seeds |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for rid in table:
                rr = [r for r in rs if r.get("recipe_id") == rid]
                if not rr:
                    continue
                rec = [(((r.get("inner_val_metrics") or {}).get("drive_full") or {}).get("per_class_recall") or {}).get(rec_cls) for r in rr]
                dec = sorted({str(r.get("decision", "softmax")) for r in rr})
                L.append(f"| {rid} | {table[rid][0]} | {_agg([_inner(r, 'drive_full') for r in rr])} | {_agg([_inner(r, 'window') for r in rr])} | "
                         f"{_agg(rec)} | {'/'.join(dec)} | {_agg_int([r.get('best_epoch') for r in rr])} | "
                         f"{_agg_int([r.get('train_minutes') for r in rr])} | {len(rr)} |")
            L.append("")
        probes = [r for r in sub if str(r.get("recipe_id", "")).startswith("L")]
        if probes:
            L += ["### Leave-one-car-model-out probes", "", "| id | held-out car model | model | inner drive_full F1 | inner window F1 | best epoch | minutes |", "|---|---|---|---|---|---|---|"]
            names = dict(PROBES)
            for r in sorted(probes, key=lambda r: (str(r.get("recipe_id")), _arch(r))):
                rid = str(r.get("recipe_id"))
                L.append(f"| {rid} | {names.get(rid.split('_')[0], '-')} | {_arch(r)} | {_agg([_inner(r, 'drive_full')])} | {_agg([_inner(r, 'window')])} | "
                         f"{_agg_int([r.get('best_epoch')])} | {_agg_int([r.get('train_minutes')])} |")
            L.append("")
    return "\n".join(L)


def _agg_int(vals: list) -> str:
    v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    if not v:
        return "-"
    return f"{np.mean(v):.0f}" if len(v) == 1 else f"{np.mean(v):.0f} ± {np.std(v):.0f}"


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["sweep", "folds", "probes"], default="sweep")
    ap.add_argument("--task", choices=list(RECIPES), default="size")
    ap.add_argument("--fold", type=int, default=None, help="sweep: the fold (default 0); folds: one fold instead of all; probes: one probe index")
    ap.add_argument("--models", default=None, help="comma list; default gru for sweep / probes, gru,resnet for folds")
    ap.add_argument("--recipes", default=None, help="comma list of recipe ids (sweep / folds)")
    ap.add_argument("--probe-base", default=None, help="probes: the recipe id to probe with (default: best so far, else the reference)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select-on", choices=sorted(ALLOWED["select_on"]), default=SELECT_ON)
    ap.add_argument("--epochs", type=int, default=None, help="override the epoch budget (smoke tests)")
    ap.add_argument("--limit-train", type=int, default=0, help="train on at most N windows (smoke tests)")
    ap.add_argument("--limit-val", type=int, default=0, help="validate on at most N windows (smoke tests)")
    ap.add_argument("--md", type=Path, default=None, help="where the table is rendered (default: experiments/RECIPE.md for the default "
                                                           "results folder, else <results-dir>/RECIPE.md)")
    ap.add_argument("--mmap", action="store_true", help="read the tensor cache from disk instead of RAM")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--ckpt-dir", type=Path, default=None, help="checkpoint folder (default: the recipe checkpoint folder of experiments.train)")
    ap.add_argument("--dry-run", action="store_true", help="print the configurations and exit")
    ap.add_argument("--render", action="store_true", help="only rewrite experiments/RECIPE.md from the saved results")
    a = ap.parse_args(argv)
    md = report_path(a.results_dir, a.md)
    if a.render:
        md.write_text(render(load_results(a.results_dir)))
        print(f"wrote {md.name} ({len(load_results(a.results_dir))} runs)")
        return 0
    models = tuple(a.models.split(",")) if a.models else (FOLD_MODELS if a.stage == "folds" else ("gru",))
    recipes = a.recipes.split(",") if a.recipes else None
    results = load_results(a.results_dir) if a.results_dir.exists() else []
    drives = None
    drives_path = Path(C.DATA_ROOT) / "build" / "index" / "drives.parquet"
    if a.stage == "probes" and drives_path.exists():
        import pandas as pd
        drives = pd.read_parquet(drives_path, columns=["drive_id", "group_id", "model", SPLIT_COL])
    cfgs = configs(a.stage, a.task, a.fold, models, recipes, a.seed, a.probe_base, a.select_on, a.epochs, a.limit_train, drives, results, a.limit_val)
    if a.dry_run:
        for c in cfgs:
            print(describe(c))
        print(f"{len(cfgs)} configurations ({a.stage}, {a.task}); done={sum((a.results_dir / (c['tag'] + '.json')).exists() for c in cfgs)}")
        return 0
    n = execute(cfgs, a.results_dir, in_ram=not a.mmap, log=lambda s: print(s, flush=True), render_md=md, ckpt_dir=a.ckpt_dir)
    md.write_text(render(load_results(a.results_dir)))
    print(f"ran {n} configurations; wrote {md.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
