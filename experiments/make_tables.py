#!/usr/bin/env python3
"""Render the experiment results as LaTeX (booktabs) tables for a paper draft.

Reads <data-root>/experiments/results/*.json (written by train.py / features_baseline.py) and writes
<out-dir>/results_<split_col>.tex (one table per split scheme: rows = models, column groups = task x protocol) and
<out-dir>/results_summary.tex (headline table: both tasks on split_platform, window and drive level). Numbers are macro-F1
in percent, mean (and std when several seeds ran). The output folder defaults to <repo>/paper/tables, which is not
part of the public repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_all import NAMES, ORDER, load_results  # noqa: E402
from experiments.train import EXP_ROOT  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TEX_NAMES = {"feat_logreg": "Statistics + logistic regression", "feat_hgb": "Statistics + gradient boosting",
             "feat_llm_logreg": "LLM physical features + logistic regression", "feat_llm_hgb": "LLM physical features + gradient boosting",
             "feat_llmstats_hgb": "LLM physical + statistics + gradient boosting",
             "resnet": "1-D ResNet", "gru": "Bi-GRU", "transformer": "Transformer encoder", "mamba2": "Mamba-2 (SSD)",
             "mamba1": "Mamba-1 (S6)"}
PROTOCOL_TEX = {"test_full": "full", "test_balanced": "balanced", "test_human_only": "human-only"}


def _cell(vals: list[float]) -> str:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return "--"
    if len(v) == 1:
        return f"{100 * v[0]:.1f}"
    return f"{100 * np.mean(v):.1f} $\\pm$ {100 * np.std(v):.1f}"


def _bold_max(cells: list[str]) -> list[str]:
    """Bold the largest mean in a column of rendered cells."""
    vals = []
    for c in cells:
        try:
            vals.append(float(c.split()[0]))
        except (ValueError, IndexError):
            vals.append(-1.0)
    m = max(vals) if vals else -1.0
    return [f"\\textbf{{{c}}}" if v == m and v >= 0 else c for c, v in zip(cells, vals)]


def table_for_split(results: list[dict], split_col: str) -> str:
    sub = [r for r in results if r["split_col"] == split_col]
    models = sorted({r["model"] for r in sub}, key=lambda x: ORDER.index(x) if x in ORDER else 99)
    tasks = [("size", "Size (3 classes)"), ("bev_ice", "BEV vs ICE")]
    prots = ["test_full", "test_balanced", "test_human_only"]
    cols = []     # list of (task, protocol or 'drive')
    for t, _ in tasks:
        cols += [(t, p) for p in prots] + [(t, "drive")]
    body: dict[str, list[str]] = {m: [] for m in models}
    for t, p in cols:
        rendered = []
        for m in models:
            rs = [r for r in sub if r["model"] == m and r["task"] == t]
            if p == "drive":
                rendered.append(_cell([r["drive_level"]["test_full"]["macro_f1"] for r in rs]))
            else:
                rendered.append(_cell([r["protocols"][p]["macro_f1"] for r in rs]))
        for m, c in zip(models, _bold_max(rendered)):
            body[m].append(c)
    title = {"split_platform": "car models held out (headline)", "split_device": "devices held out, models shared"}[split_col]
    L = ["\\begin{table*}[t]", "\\centering", "\\small",
         f"\\caption{{Macro-F1 (\\%) on the test split with {title}. Window-level results under the three protocols and "
         "drive-level results (mean class probability over the windows of a drive, protocol full).}",
         f"\\label{{tab:results_{split_col}}}",
         "\\begin{tabular}{l" + "cccc" * len(tasks) + "}", "\\toprule"]
    L.append("Model & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{name}}}" for _, name in tasks) + " \\\\")
    L.append(" ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(tasks))))
    L.append(" & " + " & ".join(" & ".join([PROTOCOL_TEX[p] for p in prots] + ["drive"]) for _ in tasks) + " \\\\")
    L.append("\\midrule")
    for m in models:
        L.append(f"{TEX_NAMES.get(m, m)} & " + " & ".join(body[m]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(L)


def summary_table(results: list[dict]) -> str:
    sub = [r for r in results if r["split_col"] == "split_platform"]
    models = sorted({r["model"] for r in sub}, key=lambda x: ORDER.index(x) if x in ORDER else 99)
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Headline results: macro-F1 (\\%) on held-out car models (protocol full), per 60-s window and per drive.}",
         "\\label{tab:headline}", "\\begin{tabular}{lcccc}", "\\toprule",
         " & \\multicolumn{2}{c}{Size} & \\multicolumn{2}{c}{BEV vs ICE} \\\\", "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
         "Model & window & drive & window & drive \\\\", "\\midrule"]
    cols = []
    for t in ("size", "bev_ice"):
        for lvl in ("window", "drive"):
            rendered = []
            for m in models:
                rs = [r for r in sub if r["model"] == m and r["task"] == t]
                key = (lambda r: r["protocols"]["test_full"]["macro_f1"]) if lvl == "window" else (lambda r: r["drive_level"]["test_full"]["macro_f1"])
                rendered.append(_cell([key(r) for r in rs]))
            cols.append(_bold_max(rendered))
    for i, m in enumerate(models):
        L.append(f"{TEX_NAMES.get(m, m)} & " + " & ".join(c[i] for c in cols) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=EXP_ROOT / "results")
    ap.add_argument("--out-dir", type=Path, default=REPO / "paper" / "tables")
    a = ap.parse_args(argv)
    results = load_results(a.results_dir)
    if not results:
        print("no results")
        return 2
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for split_col in ("split_platform", "split_device"):
        if any(r["split_col"] == split_col for r in results):
            (a.out_dir / f"results_{split_col}.tex").write_text(table_for_split(results, split_col) + "\n")
    (a.out_dir / "results_summary.tex").write_text(summary_table(results) + "\n")
    (a.out_dir / "results.json").write_text(json.dumps(
        [{k: r[k] for k in ("tag", "task", "split_col", "model", "seed", "n_params", "val_macro_f1")} |
         {"test_full": r["protocols"]["test_full"]["macro_f1"], "test_balanced": r["protocols"]["test_balanced"]["macro_f1"],
          "test_human_only": r["protocols"]["test_human_only"]["macro_f1"], "drive_full": r["drive_level"]["test_full"]["macro_f1"]}
         for r in results], indent=1))
    print(f"wrote {len(list(a.out_dir.glob('*.tex')))} tex tables to {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
