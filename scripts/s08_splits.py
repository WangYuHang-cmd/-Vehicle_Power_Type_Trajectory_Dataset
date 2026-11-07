#!/usr/bin/env python3
"""S8: splits (device-disjoint + platform-disjoint), balanced evaluation lists, class weights, split summary.

Refuses to run (exit 2) unless config.FOOTPRINT_THRESHOLDS_M2 is set and sha256(docs/SIZE_CLASSES.md) equals
config.SIZE_CLASSES_SHA256 (S3c lock). Reads build/index/{drives,windows}.parquet and private/inhouse_groups.csv;
writes build/splits/{splits.json, eval_balanced_*.parquet, train_uniform_*.parquet}, build/index/split_summary.csv and
the split columns of drives.parquet / windows.parquet. Exit 2 (and build/splits/splits_FAILED.json) when an asserted
constraint of a supported cell is violated. Prints counts only. Deterministic for a given seed; idempotent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vtt import config as C
from vtt import splits as SP
from vtt.private import mask


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--seed", type=int, default=C.SPLIT_SEED)
    ap.add_argument("--size-classes-doc", type=Path, default=None, help="default: <repo>/docs/SIZE_CLASSES.md")
    ap.add_argument("--no-inhouse", action="store_true",
                    help="run without private/inhouse_groups.csv (no train-only lab groups); otherwise a missing or empty list is exit 2")
    ap.add_argument("--dry-run", action="store_true", help="compute, print counts, write nothing")
    a = ap.parse_args(argv)
    try:
        counts = SP.run(a.data_root, a.size_classes_doc, seed=a.seed, dry_run=a.dry_run, require_inhouse=not a.no_inhouse)
    except SP.LockError as e:
        print(f"s08 REFUSED: {e}")
        return 2
    except (FileNotFoundError, ValueError) as e:
        print("s08 FAILED:", mask(str(e)))
        return 2
    print(mask(json.dumps(counts, indent=1, sort_keys=True, default=str)))
    print(f"s08 lab groups (train-only): {counts.get('n_lab_groups', 0)}; held-out platforms: {counts.get('n_heldout_platforms', 0)}; "
          f"excluded drives: {counts.get('n_excluded_drives', 0)}")
    if counts.get("violations"):
        print(f"s08 FAILED: {len(counts['violations'])} asserted constraint(s) violated (see splits_FAILED.json)")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
