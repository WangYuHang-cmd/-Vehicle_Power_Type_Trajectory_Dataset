#!/usr/bin/env python3
"""Error analysis of the saved test predictions: per held-out car model, per size x powertrain cell, and by distance of
the vehicle's footprint to the class thresholds. Reads <data-root>/experiments/results/*_probs.npz and the windows
index; writes experiments/ANALYSIS.md (model names, counts and rates only).
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
from experiments.data import TASKS  # noqa: E402
from experiments.run_all import NAMES, ORDER  # noqa: E402
from experiments.train import EXP_ROOT, macro_f1  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def load_probs(results_dir: Path) -> list[dict]:
    out = []
    for p in sorted(results_dir.glob("*_probs.npz")):
        tag = p.name[: -len("_probs.npz")]
        meta = results_dir / f"{tag}.json"
        if not meta.exists():
            continue
        r = json.loads(meta.read_text())
        z = np.load(p, allow_pickle=True)
        out.append(dict(tag=tag, task=r["task"], split_col=r["split_col"], model=r["model"], seed=r["seed"],
                        window_id=z["window_id"].astype(str), probs=z["probs"], y=z["y"]))
    return out


def per_model_table(df: pd.DataFrame, classes: tuple) -> pd.DataFrame:
    g = df.groupby("car_model")
    rows = []
    for m, sub in g:
        true = sub["y"].iloc[0]
        pred_share = sub["pred"].value_counts(normalize=True)
        rows.append(dict(car_model=m, true_class=classes[true], n_windows=len(sub), accuracy=float((sub["pred"] == sub["y"]).mean()),
                         dominant_prediction=classes[int(pred_share.index[0])], dominant_share=float(pred_share.iloc[0]),
                         footprint=float(sub["footprint_m2"].median()) if sub["footprint_m2"].notna().any() else np.nan))
    return pd.DataFrame(rows).sort_values(["true_class", "accuracy"]).reset_index(drop=True)


def boundary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy by distance of the footprint to the nearest size threshold (size task only)."""
    t1, t2 = C.FOOTPRINT_THRESHOLDS_M2
    d = np.minimum((df["footprint_m2"] - t1).abs(), (df["footprint_m2"] - t2).abs())
    bins = pd.cut(d, [0, 0.1, 0.2, 0.4, 0.8, 10], include_lowest=True)
    d2 = df.assign(bin=bins, hit=(df["pred"] == df["y"]).astype(float))
    g = d2.groupby("bin", observed=True)
    return pd.DataFrame({"windows": g.size(), "car_models": g["car_model"].nunique(), "accuracy": g["hit"].mean()}).reset_index()


def render(sections: list[str]) -> str:
    return "\n".join(["# Error analysis of the test predictions", "",
                      "Per held-out car model (headline split), per size x powertrain cell and by distance to the footprint thresholds. "
                      "`dominant prediction` is the class the model assigns most often to that car model's windows.", ""] + sections)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=EXP_ROOT / "results")
    ap.add_argument("--models", default="mamba2,transformer,gru,resnet,feat_hgb")
    ap.add_argument("--split-col", default="split_platform")
    ap.add_argument("--out", type=Path, default=REPO / "experiments" / "ANALYSIS.md")
    a = ap.parse_args(argv)
    win = pd.read_parquet(Path(C.DATA_ROOT) / "build" / "index" / "windows.parquet",
                          columns=["window_id", "drive_id", "model", "brand", "size_class", "powertrain", "footprint_m2", "human_only_strict"])
    win = win.rename(columns={"model": "car_model"})
    runs = [r for r in load_probs(a.results_dir) if r["split_col"] == a.split_col and r["model"] in a.models.split(",")]
    if not runs:
        print("no predictions found")
        return 2
    sections: list[str] = []
    for task in ("size", "bev_ice"):
        classes = TASKS[task]["classes"]
        task_runs = [r for r in runs if r["task"] == task]
        if not task_runs:
            continue
        sections += [f"## Task {task} ({a.split_col})", ""]
        # best model by test_full macro-F1 among the requested ones, seed 0 preferred
        best = None
        for r in sorted(task_runs, key=lambda r: (ORDER.index(r["model"]) if r["model"] in ORDER else 99, r["seed"])):
            f1 = macro_f1(r["y"], r["probs"].argmax(1), len(classes))
            if best is None or f1 > best[0]:
                best = (f1, r)
        f1, r = best
        df = pd.DataFrame({"window_id": r["window_id"], "y": r["y"], "pred": r["probs"].argmax(1)}).merge(win, on="window_id", how="left")
        sections += [f"Model analysed: {NAMES.get(r['model'], r['model'])} (seed {r['seed']}), window macro-F1 {100 * f1:.1f}.", ""]
        # per cell
        cell = df.groupby(["size_class", "powertrain"], dropna=False).apply(lambda s: pd.Series(dict(windows=len(s), car_models=s["car_model"].nunique(), accuracy=float((s["pred"] == s["y"]).mean()))), include_groups=False).reset_index()
        sections += ["### Accuracy per size x powertrain cell", "", "| size | powertrain | windows | car models | accuracy |", "|---|---|---|---|---|"]
        for c in cell.itertuples():
            sections.append(f"| {c.size_class} | {c.powertrain} | {int(c.windows)} | {int(c.car_models)} | {100 * c.accuracy:.1f} % |")
        sections.append("")
        # human-only vs automated windows
        for flag, name in ((True, "human-only (strict)"), (False, "with automation activity")):
            s = df[df["human_only_strict"].fillna(False).astype(bool) == flag]
            if len(s):
                sections.append(f"- {name}: {len(s)} windows, accuracy {100 * float((s['pred'] == s['y']).mean()):.1f} %, macro-F1 {100 * macro_f1(s['y'].to_numpy(), s['pred'].to_numpy(), len(classes)):.1f}")
        sections.append("")
        pm = per_model_table(df, classes)
        sections += ["### Held-out car models", "", "| car model | true class | windows | accuracy | dominant prediction (share) | footprint m^2 |", "|---|---|---|---|---|---|"]
        for m in pm.itertuples():
            fp = f"{m.footprint:.2f}" if np.isfinite(m.footprint) else "-"
            sections.append(f"| {m.car_model} | {m.true_class} | {m.n_windows} | {100 * m.accuracy:.1f} % | {m.dominant_prediction} ({100 * m.dominant_share:.0f} %) | {fp} |")
        sections.append("")
        if task == "size" and C.FOOTPRINT_THRESHOLDS_M2:
            bt = boundary_table(df.dropna(subset=["footprint_m2"]))
            sections += ["### Accuracy by distance of the footprint to the nearest threshold", "", "| distance (m^2) | windows | car models | accuracy |", "|---|---|---|---|"]
            for b in bt.itertuples():
                sections.append(f"| {b.bin} | {int(b.windows)} | {int(b.car_models)} | {100 * b.accuracy:.1f} % |")
            sections.append("")
    a.out.write_text(render(sections))
    print(f"wrote {a.out.relative_to(REPO)} ({len(runs)} runs considered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
