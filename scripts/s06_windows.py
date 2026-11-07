#!/usr/bin/env python3
"""S6: build the 60-s window index (build/index/windows.parquet, build/index/human_only_windows.parquet).

Reads build/data/<vehicle_id>/<drive_id>.parquet and build/index/drives.parquet; per-vehicle parts are cached
under work/windows_parts (resumable; --force rebuilds them). Prints counts only. Exit 2 when the acceptance
checks fail (vtt.windows.acceptance_problems or the public schema check).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from vtt import config as C
from vtt import schema as S
from vtt import windows as W


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="rebuild cached per-vehicle parts")
    ap.add_argument("--limit", type=int, default=0, help="only the first N drives of the index (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="compute and print counts, write nothing")
    a = ap.parse_args(argv)
    roots = W.Roots(a.data_root)
    if not roots.drives.exists():
        print(f"missing drives index under {roots.index}; run S5 first")
        return 1
    drives = pd.read_parquet(roots.drives)
    if a.limit:
        drives = drives.head(a.limit).copy()
    windows, counts = W.build_windows(drives, roots, workers=max(1, a.workers), force=a.force, dry_run=a.dry_run)
    print(" ".join(f"{k}={v}" for k, v in counts.items()))
    problems = W.acceptance_problems(windows, drives)
    if a.dry_run:
        print("dry-run: nothing written" + (f"; problems: {problems}" if problems else ""))
        return 2 if problems else 0
    W.write_windows(windows, roots.windows)
    human = W.human_only_subset(windows)
    W.write_windows(human, roots.human_only_windows)
    for path in (roots.windows, roots.human_only_windows):
        problems += [f"{path.name}: {p}" for p in S.check_public_parquet(path, S.WINDOWS_INDEX_COLUMNS, W.WINDOW_TABLE)]
    reasons = windows.loc[~windows["eligible"].astype(bool), "inelig_reason"].fillna("").value_counts()
    print("ineligible by reason: " + ", ".join(f"{k or 'none'}={v}" for k, v in reasons.items()))
    print(f"wrote windows={len(windows)} human_only_windows={len(human)} eligible_hours={counts['eligible'] * C.WINDOW_SAMPLES * C.NOMINAL_DT_S / 3600:.1f}")
    if problems:
        print("ACCEPTANCE FAILED: " + "; ".join(problems))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
