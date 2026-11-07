#!/usr/bin/env python3
"""Run the experiment grid (split schemes x tasks x models x seeds) and render experiments/RESULTS.md.

Finished configurations (a results JSON exists) are skipped, so the grid can be resumed. The tensor cache is loaded
once (about 5 GB of RAM). Baseline results written by experiments/features_baseline.py are picked up by the table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.data import PROTOCOLS, WindowData  # noqa: E402
from experiments.train import EXP_ROOT, run  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ORDER = ["feat_logreg", "feat_hgb", "feat_llm_logreg", "feat_llm_hgb", "feat_llmstats_hgb", "resnet", "gru", "transformer", "mamba2", "mamba1"]
NAMES = {"feat_logreg": "hand-crafted features + logistic regression", "feat_hgb": "hand-crafted features + gradient boosting",
         "feat_llm_logreg": "LLM physical features + logistic regression", "feat_llm_hgb": "LLM physical features + gradient boosting",
         "feat_llmstats_hgb": "LLM physical + hand-crafted features + gradient boosting",
         "resnet": "1-D ResNet (10 Hz)", "gru": "bi-GRU", "transformer": "Transformer encoder", "mamba2": "Mamba-2 (SSD), bidirectional",
         "mamba1": "Mamba-1 (S6), bidirectional"}


def load_results(results_dir: Path) -> list[dict]:
    out = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except ValueError:
            continue
    return out


def _agg(vals: list[float]) -> str:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return "-"
    if len(v) == 1:
        return f"{100 * v[0]:.1f}"
    return f"{100 * np.mean(v):.1f} ± {100 * np.std(v):.1f}"


def render(results: list[dict]) -> str:
    L = ["# Classification results (60-s windows, official input channels only)", "",
         "Macro-F1 in percent; mean ± std over seeds when several seeds ran. `window` = one prediction per 60-s window, "
         "`drive` = mean class probability over the windows of a drive. Protocols follow docs/PROTOCOLS.md: `full` = every "
         "eligible test window, `balanced` = the published class-balanced list, `human` = windows without any driving "
         "automation activity (strict). Chance level for macro-F1 is about 33 (size) and 50 (BEV vs ICE); the always-majority "
         "predictor scores below that.", ""]
    for split_col, title in (("split_platform", "Headline: car models held out (`split_platform`)"),
                             ("split_device", "Secondary: devices held out, models shared (`split_device`)")):
        L += [f"## {title}", ""]
        for task, tname in (("size", "Size: SMALL / MID / LARGE"), ("bev_ice", "Powertrain: BEV vs ICE")):
            sub = [r for r in results if r["split_col"] == split_col and r["task"] == task]
            if not sub:
                continue
            L += [f"### {tname}", "", "| model | params | window full | window balanced | window human-only | drive full | drive human-only | best val | seeds |",
                  "|---|---|---|---|---|---|---|---|---|"]
            by_model: dict[str, list[dict]] = {}
            for r in sub:
                by_model.setdefault(r["model"], []).append(r)
            for m in sorted(by_model, key=lambda x: ORDER.index(x) if x in ORDER else 99):
                rs = by_model[m]
                cell = lambda f: _agg([f(r) for r in rs])   # noqa: E731
                L.append(f"| {NAMES.get(m, m)} | {rs[0].get('n_params', 0):,} | "
                         f"{cell(lambda r: r['protocols']['test_full']['macro_f1'])} | "
                         f"{cell(lambda r: r['protocols']['test_balanced']['macro_f1'])} | "
                         f"{cell(lambda r: r['protocols']['test_human_only']['macro_f1'])} | "
                         f"{cell(lambda r: r['drive_level']['test_full']['macro_f1'])} | "
                         f"{cell(lambda r: r['drive_level']['test_human_only']['macro_f1'])} | "
                         f"{cell(lambda r: r.get('val_macro_f1'))} | {len(rs)} |")
            L.append("")
            # per-class recall of the full protocol for the best model
            best = max(sub, key=lambda r: r["protocols"]["test_full"]["macro_f1"])
            rec = best["protocols"]["test_full"]["per_class_recall"]
            L.append(f"Best single run on `full`: {NAMES.get(best['model'], best['model'])} (seed {best['seed']}), per-class recall: "
                     + ", ".join(f"{c} {100 * v:.1f}" for c, v in rec.items() if v is not None) + ".")
            L.append("")
    L += ["## Set-up", "",
          "- inputs: the 20 official channels (vtt.config.OFFICIAL_INPUTS) of each 600-sample window, clipped per channel, "
          "standardised with training-split statistics, NaN -> 0 (lead channels carry `lead_valid`);",
          "- sequence models share a 2-layer convolutional stem (stride 4 -> 150 tokens of width 128); 4 layers for the Transformer "
          "and the Mamba variants, 2 bidirectional GRU layers; the ResNet works on the 10 Hz signal;",
          "- training: class-balanced sampling ((N / K n_k)^0.5, the dataset's train_uniform rule), AdamW, warm-up + cosine, label "
          "smoothing 0.05, batch 256, at most 25 epochs, early stopping (patience 6) on the validation macro-F1 of the same split scheme;",
          "- the Mamba blocks are pure-PyTorch implementations of the published algorithms (chunked state-space duality scan for "
          "Mamba-2, selective scan for Mamba-1), bidirectional (forward and reversed scans summed) as usual for classification.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="resnet,gru,transformer,mamba2")
    ap.add_argument("--tasks", default="size,bev_ice")
    ap.add_argument("--split-cols", default="split_platform,split_device")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args(argv)
    results_dir = EXP_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if not a.render_only:
        data = WindowData()
        print(f"windows={len(data.win)} channels={len(data.channels)}", flush=True)
        grid = list(product(a.split_cols.split(","), a.tasks.split(","), a.models.split(","), [int(s) for s in a.seeds.split(",")]))
        for i, (split_col, task, model, seed) in enumerate(grid, 1):
            tag = f"{split_col}__{task}__{model}__s{seed}"
            if (results_dir / f"{tag}.json").exists():
                print(f"[{i}/{len(grid)}] skip {tag} (done)", flush=True)
                continue
            print(f"[{i}/{len(grid)}] start {tag} {time.strftime('%H:%M')}", flush=True)
            run(task, split_col, model, seed, data=data, log=lambda s: print(s, flush=True), epochs=a.epochs)
            (REPO / "experiments" / "RESULTS.md").write_text(render(load_results(results_dir)))
    (REPO / "experiments" / "RESULTS.md").write_text(render(load_results(results_dir)))
    print("wrote experiments/RESULTS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
