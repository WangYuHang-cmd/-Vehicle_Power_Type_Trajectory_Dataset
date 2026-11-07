#!/usr/bin/env python3
"""Evaluate the LLM-proposed physical features with the baseline classifiers and write experiments/LLM_FEATURES.md.

Models (same protocols and result format as the other baselines): feat_llm_logreg, feat_llm_hgb (LLM features only),
feat_llmstats_hgb (LLM features + the 180 plain statistics). Permutation importance of the LLM-only gradient-boosting
model is measured on the validation split of the headline scheme. The report lists every proposed feature with its lens,
rationale, training-sample statistics, audit verdict and importance, then the result tables next to the plain-statistics
baseline and the best deep model.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402
from experiments.data import TASKS, WindowData  # noqa: E402
from experiments.features_baseline import run as run_baseline  # noqa: E402
from experiments.run_all import NAMES as MODEL_NAMES, load_results  # noqa: E402
from experiments.train import EXP_ROOT, macro_f1  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TDIR = Path(C.DATA_ROOT) / "work" / "tensors"
PKG = REPO / "experiments" / "llm_features"
SPLIT_COLS = ("split_platform", "split_device")
TASK_LIST = ("size", "bev_ice")


def permutation_importance(feats: np.ndarray, names: list[str], data: WindowData, task: str, split_col: str, seed: int = 0,
                           n_val: int = 20000, n_repeats: int = 3) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr = data.indices(task, split_col, "train")
    va = data.indices(task, split_col, "validation")
    rng = np.random.default_rng(seed)
    if len(va) > n_val:
        va = np.sort(rng.choice(va, size=n_val, replace=False))
    ytr, yva = data.labels(task, tr), data.labels(task, va)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                         class_weight="balanced", early_stopping=True, validation_fraction=0.1, random_state=seed)
    clf.fit(feats[tr], ytr)
    k = len(TASKS[task]["classes"])
    base = macro_f1(yva, clf.predict(feats[va]), k)
    rows = []
    Xva = feats[va].copy()
    for j, name in enumerate(names):
        drops = []
        for r in range(n_repeats):
            col = Xva[:, j].copy()
            Xva[:, j] = rng.permutation(Xva[:, j])
            drops.append(base - macro_f1(yva, clf.predict(Xva), k))
            Xva[:, j] = col
        rows.append(dict(feature=name, importance=float(np.mean(drops)), std=float(np.std(drops))))
    out = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    out.attrs["base_macro_f1"] = base
    return out


def result_row(results: list[dict], model: str, task: str, split_col: str) -> dict | None:
    rs = [r for r in results if r["model"] == model and r["task"] == task and r["split_col"] == split_col]
    if not rs:
        return None
    agg = lambda f: float(np.mean([f(r) for r in rs]))     # noqa: E731
    return dict(model=model, n=len(rs), full=agg(lambda r: r["protocols"]["test_full"]["macro_f1"]),
                balanced=agg(lambda r: r["protocols"]["test_balanced"]["macro_f1"]),
                human=agg(lambda r: r["protocols"]["test_human_only"]["macro_f1"]),
                drive=agg(lambda r: r["drive_level"]["test_full"]["macro_f1"]))


def best_deep(results: list[dict], task: str, split_col: str) -> dict | None:
    deep = ("resnet", "gru", "transformer", "mamba2", "mamba1")
    rows = [result_row(results, m, task, split_col) for m in deep]
    rows = [r for r in rows if r]
    return max(rows, key=lambda r: r["full"]) if rows else None


def _scrub(text: str) -> str:
    """Agent prose may quote local file paths; the report keeps only the file name."""
    t = re.sub(r"(?<![A-Za-z0-9_])" + "AD" + "AS" + r"(?![A-Za-z0-9_])", "DAS", str(text))   # public wording
    return re.sub(r"/(?:data|home|mnt|tmp)/[^\s`)\]]*/", "", t)


def render(manifest: list[dict], importances: dict, results: list[dict], audit: dict | None, n_windows: int) -> str:
    kept = [m for m in manifest if m["status"] == "kept"]
    L = ["# LLM-proposed physical features", "",
         "Six agents, each with one physics lens (longitudinal powertrain, transmission signature, braking / regeneration / "
         "coasting, lateral inertia and geometry, car-following interaction, speed-dependent capability), proposed window features "
         "with a physical rationale and numpy code while seeing only the channel definitions and a training-only sample "
         f"({n_windows:,} windows that are in the training split of both schemes). Six verifier agents ran the code on that sample, "
         "fixed bugs, measured class separation and dropped degenerate or duplicate features; one auditor screened the merged list "
         "for recording-artefact shortcuts. Everything below was fixed before any evaluation split was touched. The features are "
         "then fed to the same classifiers and protocols as the plain-statistics baseline.", "",
         f"Proposed: {len(manifest)}; registered after verification, audit and self-test: {len(kept)}.", "",
         "## Features", "",
         "| feature | lens | rationale | hypothesis | AUROC BEV vs ICE (train sample) | Spearman footprint | audit | importance size | importance BEV/ICE |",
         "|---|---|---|---|---|---|---|---|---|"]
    imp_size = importances.get("size", {}); imp_pt = importances.get("bev_ice", {})
    for m in sorted(kept, key=lambda m: -(imp_pt.get(m["name"], 0) + imp_size.get(m["name"], 0))):
        au = m.get("auroc_bev_vs_ice"); sp = m.get("spearman_footprint")
        L.append(f"| `{m['name']}` | {m['lens']} | {_scrub(m['rationale']).strip()} | {_scrub(m['hypothesis']).strip()} | "
                 f"{au:.2f} | {sp:+.2f} | {m.get('audit_verdict') or 'keep'} | {100 * imp_size.get(m['name'], 0):+.1f} | {100 * imp_pt.get(m['name'], 0):+.1f} |"
                 if au is not None and sp is not None else
                 f"| `{m['name']}` | {m['lens']} | {_scrub(m['rationale']).strip()} | {_scrub(m['hypothesis']).strip()} | - | - | {m.get('audit_verdict') or 'keep'} | "
                 f"{100 * imp_size.get(m['name'], 0):+.1f} | {100 * imp_pt.get(m['name'], 0):+.1f} |")
    L += ["", "Importance = drop of the validation macro-F1 (percentage points) when the feature is permuted, gradient boosting on the "
          "LLM features alone, headline split scheme; AUROC and Spearman were measured by the verifier agents on the training sample.", ""]
    dropped = [m for m in manifest if m["status"] != "kept"]
    if dropped:
        L += ["### Not used", "", "| feature | lens | status | reason |", "|---|---|---|---|"]
        for m in dropped:
            why = m.get("audit_reason") if m["status"] == "excluded_by_audit" else m.get("reason", "")
            L.append(f"| `{m['name']}` | {m['lens']} | {m['status']} | {_scrub(str(why)).strip()} |")
        L.append("")
    if audit:
        L += ["### Auditor's summary", "", _scrub(audit.get("summary", "")).strip(), ""]
        if audit.get("gaps"):
            L += ["Physical cues no feature covers: " + _scrub("; ".join(audit["gaps"])) + ".", ""]
    L += ["## Results (macro-F1, percent; mean over seeds)", ""]
    for split_col, title in (("split_platform", "car models held out (headline)"), ("split_device", "devices held out, models shared")):
        L += [f"### {title}", "", "| task | model | window full | window balanced | window human-only | drive full |", "|---|---|---|---|---|---|"]
        for task, tname in (("size", "size"), ("bev_ice", "BEV vs ICE")):
            rows = [result_row(results, m, task, split_col) for m in ("feat_hgb", "feat_llm_logreg", "feat_llm_hgb", "feat_llmstats_hgb")]
            bd = best_deep(results, task, split_col)
            for r in rows + ([bd] if bd else []):
                if r:
                    L.append(f"| {tname} | {MODEL_NAMES.get(r['model'], r['model'])} | {100 * r['full']:.1f} | {100 * r['balanced']:.1f} | "
                             f"{100 * r['human']:.1f} | {100 * r['drive']:.1f} |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--skip-eval", action="store_true", help="only render the report from existing results")
    a = ap.parse_args(argv)
    data = WindowData(in_ram=False)
    llm = np.load(TDIR / "features_llm_f32.npy")
    names = json.loads((TDIR / "features_llm_names.json").read_text())
    stats = np.load(TDIR / "features_f32.npy")
    if llm.shape[0] != len(data.win) or stats.shape[0] != len(data.win):
        raise SystemExit("feature matrices do not match the windows index")
    both = np.concatenate([llm, stats], axis=1)
    out_dir = EXP_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not a.skip_eval:
        for split_col in SPLIT_COLS:
            for task in TASK_LIST:
                for seed in [int(s) for s in a.seeds.split(",")]:
                    for name, feats in (("feat_llm_logreg", llm), ("feat_llm_hgb", llm), ("feat_llmstats_hgb", both)):
                        if (out_dir / f"{split_col}__{task}__{name}__s{seed}.json").exists():
                            continue
                        run_baseline(name, task, split_col, data, feats, seed, out_dir)
    importances = {}
    for task in TASK_LIST:
        imp = permutation_importance(llm, names, data, task, "split_platform")
        importances[task] = dict(zip(imp["feature"], imp["importance"]))
        imp.to_csv(EXP_ROOT / f"llm_feature_importance_{task}.csv", index=False)
        print(f"importance {task}: base val macro-F1 {imp.attrs['base_macro_f1']:.3f}; top: " +
              ", ".join(f"{r.feature} {100 * r.importance:+.1f}" for r in imp.head(8).itertuples()), flush=True)
    manifest = json.loads((PKG / "_manifest.json").read_text())
    llm_dir = Path(C.DATA_ROOT) / "experiments" / "llm_feat"
    audit_path = llm_dir / "proposals.json"
    audit = json.loads(audit_path.read_text()).get("audit") if audit_path.exists() else None
    n_sample = int(np.load(llm_dir / "train_sample.npz", allow_pickle=True)["X"].shape[0])
    (REPO / "experiments" / "LLM_FEATURES.md").write_text(render(manifest, importances, load_results(out_dir), audit, n_sample))
    print("wrote experiments/LLM_FEATURES.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
