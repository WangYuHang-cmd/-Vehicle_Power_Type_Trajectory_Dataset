#!/usr/bin/env python3
"""Cluster-bootstrap 95 % intervals (over car models) for the unit-level protocols of saved test predictions.

Reads <data-root>/experiments/results/<tag>_probs.npz (window_id, probs, y) of the listed runs, scores drive_full,
drive_human_only, vehicle_full and vehicle_full_multi with starter_kit.eval.evaluate_units (which carries the bootstrap of
docs/PROTOCOLS.md: car models resampled with replacement, 1,000 draws, seed 0) and writes paper/tables/ci.json plus a
markdown table on stdout. Per configuration the point estimate is the mean over seeds; the interval is reported per
seed and summarised as the mean of the seeds' lower and upper bounds. CPU only; reads the public index tables.
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
from starter_kit.eval import evaluate_units  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = Path(C.DATA_ROOT) / "experiments" / "results"
PROTOCOLS = ("drive_full", "drive_human_only", "vehicle_full", "vehicle_full_multi")
DEFAULT_RUNS = [
    # (label, task, split_col, tag prefix, seeds)
    ("bi-GRU (reference)", "size", "split_platform", "split_platform__size__gru", (0, 1, 2)),
    ("ResNet (reference)", "size", "split_platform", "split_platform__size__resnet", (0, 1, 2)),
    ("Transformer (reference)", "size", "split_platform", "split_platform__size__transformer", (0, 1, 2)),
    ("Mamba-2 (reference)", "size", "split_platform", "split_platform__size__mamba2", (0, 1, 2)),
    ("bi-GRU S9", "size", "split_platform", "split_platform__size__gru__S9", (0, 1, 2)),
    ("ResNet S9", "size", "split_platform", "split_platform__size__resnet__S9", (0, 1, 2)),
    ("episode MIL", "size", "split_platform", "split_platform__size__episode_mil", (0, 1, 2)),
    ("bi-GRU (reference)", "bev_ice", "split_platform", "split_platform__bev_ice__gru", (0, 1, 2)),
    ("ResNet (reference)", "bev_ice", "split_platform", "split_platform__bev_ice__resnet", (0, 1, 2)),
    ("Transformer (reference)", "bev_ice", "split_platform", "split_platform__bev_ice__transformer", (0, 1, 2)),
    ("Mamba-2 (reference)", "bev_ice", "split_platform", "split_platform__bev_ice__mamba2", (0, 1, 2)),
    ("bi-GRU B5", "bev_ice", "split_platform", "split_platform__bev_ice__gru__B5", (0, 1, 2)),
    ("ResNet B5", "bev_ice", "split_platform", "split_platform__bev_ice__resnet__B5", (0, 1, 2)),
    ("episode MIL", "bev_ice", "split_platform", "split_platform__bev_ice__episode_mil", (0, 1, 2)),
    ("bi-GRU S9", "size", "split_device", "split_device__size__gru__S9", (0, 1, 2)),
    ("ResNet S9", "size", "split_device", "split_device__size__resnet__S9", (0, 1, 2)),
    ("bi-GRU B5", "bev_ice", "split_device", "split_device__bev_ice__gru__B5", (0, 1, 2)),
    ("ResNet B5", "bev_ice", "split_device", "split_device__bev_ice__resnet__B5", (0, 1, 2)),
]


def load_tables(root: Path):
    win = pd.read_parquet(root / "build" / "index" / "windows.parquet")
    drives = pd.read_parquet(root / "build" / "index" / "drives.parquet")
    return win, drives


def score(win, drives, task, split_col, tag, n_boot, seed):
    z = np.load(RESULTS / f"{tag}_probs.npz", allow_pickle=True)
    probs = pd.DataFrame(z["probs"].astype(np.float64), index=pd.Index(z["window_id"].astype(str), name="window_id"))
    out = {}
    for prot in PROTOCOLS:
        r = evaluate_units(win, probs, task, prot, split_col=split_col, split="test", n_boot=n_boot, seed=seed, drives=drives)
        out[prot] = dict(macro_f1=100 * r["macro_f1"], ci=[100 * x for x in r["ci95_macro_f1"]] if r["ci95_macro_f1"] else None,
                         n_units=r["n_units"], n_models=r["n_models"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "paper" / "tables" / "ci.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    win, drives = load_tables(Path(C.DATA_ROOT))
    rows = []
    for label, task, split_col, prefix, seeds in DEFAULT_RUNS:
        per_seed = {}
        for s in seeds:
            tag = f"{prefix}__s{s}"
            if not (RESULTS / f"{tag}_probs.npz").exists():
                continue
            per_seed[s] = score(win, drives, task, split_col, tag, a.n_boot, a.seed)
        if not per_seed:
            print(f"# no predictions for {prefix}", file=sys.stderr)
            continue
        summary = {}
        for prot in PROTOCOLS:
            vals = [per_seed[s][prot]["macro_f1"] for s in per_seed]
            cis = [per_seed[s][prot]["ci"] for s in per_seed if per_seed[s][prot]["ci"]]
            summary[prot] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)) if len(vals) > 1 else 0.0,
                                 ci_lo=float(np.mean([c[0] for c in cis])) if cis else None,
                                 ci_hi=float(np.mean([c[1] for c in cis])) if cis else None,
                                 ci_per_seed={str(s): per_seed[s][prot]["ci"] for s in per_seed},
                                 n_units=per_seed[next(iter(per_seed))][prot]["n_units"],
                                 n_models=per_seed[next(iter(per_seed))][prot]["n_models"])
        rows.append(dict(label=label, task=task, split_col=split_col, prefix=prefix, seeds=sorted(per_seed), protocols=summary))
        d, v = summary["drive_full"], summary["vehicle_full"]
        print(f"| {task} | {split_col} | {label} | {d['mean']:.1f} [{d['ci_lo']:.1f}, {d['ci_hi']:.1f}] (n={d['n_units']}, models={d['n_models']}) "
              f"| {summary['drive_human_only']['mean']:.1f} [{summary['drive_human_only']['ci_lo']:.1f}, {summary['drive_human_only']['ci_hi']:.1f}] "
              f"| {v['mean']:.1f} [{v['ci_lo']:.1f}, {v['ci_hi']:.1f}] (n={v['n_units']}) "
              f"| {summary['vehicle_full_multi']['mean']:.1f} [{summary['vehicle_full_multi']['ci_lo']:.1f}, {summary['vehicle_full_multi']['ci_hi']:.1f}] (n={summary['vehicle_full_multi']['n_units']}) |", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
