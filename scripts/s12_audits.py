#!/usr/bin/env python3
"""S12: shortcut audits of the build (nuisance metadata -> labels / splits; official-input statistics -> brand).

Reads <data-root>/build/index/drives.parquet and windows.parquet, writes under <out> (default
<data-root>/build/qc): audit_nuisance.csv, audit_inputs_brand.csv, audit_numbers.json and AUDITS.md (the bullets
for the data card's limitations section). Prints counts only. Exit 2 when an input table is missing or empty.
Idempotent: outputs are rewritten from scratch on every run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from vtt import config as C
from vtt.audits import inputs_brand as IB
from vtt.audits import nuisance as NU
from vtt.private import write_json_atomic, write_text_atomic


def build_paths(data_root: Path) -> dict[str, Path]:
    idx = Path(data_root) / "build" / "index"
    return {"drives": idx / "drives.parquet", "windows": idx / "windows.parquet", "qc": Path(data_root) / "build" / "qc"}


def numbers(nuis: pd.DataFrame, brand: pd.DataFrame) -> dict:
    """Headline numbers for the data card (joint nuisance key per target; nearest-centroid brand / platform)."""
    out: dict[str, float | int] = {}
    for r in nuis[nuis.predictor == NU.ALL_KEY].itertuples(index=False):
        out[f"nuisance_all_{r.target}_macro_f1"] = round(float(r.macro_f1), 3)
        out[f"nuisance_all_{r.target}_chance_f1"] = round(float(r.chance_permuted_f1), 3)
    for r in brand[brand.method == "nearest_centroid:all"].itertuples(index=False):
        out[f"inputs_{r.target}_macro_f1"] = round(float(r.macro_f1), 3)
        out[f"inputs_{r.target}_chance_f1"] = round(float(r.chance_permuted_f1), 3)
        out[f"inputs_{r.target}_n_classes"] = int(r.n_classes)
    return out


def render_md(nuis: pd.DataFrame, brand: pd.DataFrame, n_drives: int, n_windows: int) -> str:
    lines = ["# Shortcut audits", "",
             f"Inputs: {n_drives} drives, {n_windows} windows (eligible windows subsampled for the input audit).",
             "Classifiers are deliberately simple (per-value majority vote, nearest centroid) under group-disjoint",
             "cross-validation; macro-F1 is compared with the permutation chance level of the same label set.", "",
             "## Nuisance metadata -> labels and splits", ""]
    lines += NU.summary_lines(nuis) or ["* (no target with enough labelled drives)"]
    lines += ["", "## Official-input window statistics -> brand / platform / labels", ""]
    lines += IB.summary_lines(brand) or ["* (no target with enough windows)"]
    lines += ["", "Tables: `audit_nuisance.csv`, `audit_inputs_brand.csv`; numbers: `audit_numbers.json`.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default <data-root>/build/qc)")
    ap.add_argument("--seed", type=int, default=C.SPLIT_SEED)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--max-windows", type=int, default=200000, help="subsample of eligible windows for the input audit")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    paths = build_paths(a.data_root)
    out = a.out or paths["qc"]
    missing = [k for k in ("drives", "windows") if not paths[k].exists()]
    if missing:
        print(f"missing inputs: {len(missing)} ({', '.join(missing)})")
        return 2
    drives = pd.read_parquet(paths["drives"]); windows = pd.read_parquet(paths["windows"])
    print(f"drives={len(drives)} windows={len(windows)} eligible={int(windows['eligible'].fillna(False).astype(bool).sum()) if 'eligible' in windows else -1}")
    if drives.empty or windows.empty:
        print("empty input table")
        return 2
    if a.dry_run:
        print(f"dry-run: would write 4 files under {out}")
        return 0
    nuis = NU.audit(drives, n_folds=a.n_folds, seed=a.seed)
    brand = IB.audit(windows, n_folds=a.n_folds, seed=a.seed, max_rows=a.max_windows)
    out.mkdir(parents=True, exist_ok=True)
    nuis.to_csv(out / "audit_nuisance.csv", index=False)
    brand.to_csv(out / "audit_inputs_brand.csv", index=False)
    write_json_atomic(numbers(nuis, brand), out / "audit_numbers.json")
    write_text_atomic(render_md(nuis, brand, len(drives), len(windows)), out / "AUDITS.md")
    print(f"nuisance rows={len(nuis)} inputs rows={len(brand)} targets={nuis.target.nunique()}/{brand.target.nunique()} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
