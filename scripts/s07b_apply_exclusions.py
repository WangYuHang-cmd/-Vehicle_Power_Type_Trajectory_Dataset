#!/usr/bin/env python3
"""S7b: exclude duplicate recordings (and listed drives) from the window index before the splits.

The S7 QC marks drives whose speed trace is byte-identical to an earlier drive (`duplicate_of` in build/qc/drives_qc.csv).
Cross-vehicle matches are stationary recordings (speed constantly 0) and have no eligible window anyway; same-vehicle
matches are the same drive logged twice. Every flagged copy loses its windows: `eligible = False`,
`inelig_reason = 'duplicate_recording'` (appended), and the drive's `label_note` records the exclusion. Extra drive ids can be
listed in <data-root>/private/exclude_drives.csv (column drive_id, reason). Idempotent; prints counts only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtt import config as C          # noqa: E402
from vtt import schema as S          # noqa: E402

REASON = "duplicate_recording"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.data_root)
    qc = pd.read_csv(root / "build" / "qc" / "drives_qc.csv", dtype=str).fillna("")
    dup = qc[qc["duplicate_of"].str.len() > 0][["drive_id", "duplicate_of"]]
    reasons = {d: f"{REASON}: same speed trace as {o}" for d, o in zip(dup.drive_id, dup.duplicate_of)}
    extra = root / "private" / "exclude_drives.csv"
    n_extra = 0
    if extra.exists():
        e = pd.read_csv(extra, dtype=str).fillna("")
        for d, r in zip(e.get("drive_id", []), e.get("reason", [])):
            if d and d not in reasons:
                reasons[d] = f"excluded: {r or 'listed'}"
                n_extra += 1
    windows_p = root / "build" / "index" / "windows.parquet"
    human_p = root / "build" / "index" / "human_only_windows.parquet"
    drives_p = root / "build" / "index" / "drives.parquet"
    w = pd.read_parquet(windows_p)
    hit = w.drive_id.isin(reasons)
    already = hit & w.inelig_reason.fillna("").str.contains(REASON)
    n_windows = int(hit.sum()); n_elig_before = int((hit & w.eligible.astype(bool)).sum())
    print(f"flagged drives: {len(reasons)} (duplicates {len(dup)}, listed {n_extra}); windows affected {n_windows}, "
          f"eligible before {n_elig_before}, already marked {int(already.sum())}")
    if a.dry_run:
        return 0
    w.loc[hit, "eligible"] = False
    w.loc[hit, "human_only"] = False
    w.loc[hit, "human_only_strict"] = False
    tag = w.loc[hit & ~already, "inelig_reason"].fillna("").astype(str)
    w.loc[hit & ~already, "inelig_reason"] = [(t + "+" if t else "") + REASON for t in tag]
    S.write_public_parquet(w, windows_p, S.WINDOWS_INDEX_COLUMNS, "windows")
    if human_p.exists():
        h = pd.read_parquet(human_p)
        h = h[~h.drive_id.isin(reasons)]
        S.write_public_parquet(h, human_p, S.WINDOWS_INDEX_COLUMNS, "windows")
    d = pd.read_parquet(drives_p)
    m = d.drive_id.isin(reasons)
    notes = d.loc[m, "label_note"].fillna("").astype(str)
    new = [n if REASON in n else ((n + "; " if n else "") + reasons[k]) for n, k in zip(notes, d.loc[m, "drive_id"])]
    d.loc[m, "label_note"] = new
    S.write_public_parquet(d, drives_p, S.DRIVES_INDEX_COLUMNS, "drives")
    pd.DataFrame({"drive_id": list(reasons), "reason": list(reasons.values())}).to_csv(root / "build" / "qc" / "exclusions.csv", index=False)
    print(f"windows now eligible: {int(w.eligible.sum())}; wrote build/qc/exclusions.csv ({len(reasons)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
