#!/usr/bin/env python3
"""S13 (1): export the official input channels of every eligible 60-s window into one float16 array.

reads   build/index/drives.parquet, build/index/windows.parquet (eligible rows), build/data/<vehicle>/<drive>.parquet
writes  work/tensors/X_f16.npy        (N, WINDOW_SAMPLES, len(OFFICIAL_INPUTS)) float16, NaN kept as NaN, bools as 0/1
        work/tensors/windows.parquet  (window_id, drive_id, vehicle_id, group_id, w_idx, tensor_idx) in array order
        work/tensors/channels.json    channel names in column order

The array is a modelling cache (never released); labels and splits are joined from build/index/windows.parquet at
training time by window_id, so a relabel or a new split never needs a re-export. Workers write their slices straight into
the memory-mapped file. Prints counts only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402

CHANNELS = list(C.OFFICIAL_INPUTS)
KEEP = ["window_id", "drive_id", "vehicle_id", "group_id", "w_idx", "tensor_idx"]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None, help="default <data-root>/work/tensors")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-drives", type=int, default=0, help="smoke test: only the first N drives")
    ap.add_argument("--nice", type=int, default=10)
    return ap.parse_args(argv)


def drive_matrix(path: Path) -> np.ndarray:
    """(rows, channels) float32 of one drive table; bools -> 0/1, everything else numeric (NaN kept)."""
    t = pq.read_table(path, columns=CHANNELS).to_pandas()
    arr = np.empty((len(t), len(CHANNELS)), dtype=np.float32)
    for j, c in enumerate(CHANNELS):
        col = t[c]
        if str(col.dtype) in ("bool", "boolean"):
            arr[:, j] = col.fillna(False).astype(np.float32).to_numpy()
        else:
            arr[:, j] = pd.to_numeric(col, errors="coerce").astype(np.float32).to_numpy()
    return arr


def _work(job) -> tuple[int, int]:
    path, rows, out_path, n_samples = job
    try:
        arr = drive_matrix(Path(path))
    except Exception:          # noqa: BLE001  a damaged file: every window of the drive stays NaN
        return 0, len(rows)
    mm = np.lib.format.open_memmap(out_path, mode="r+")
    n_ok = n_bad = 0
    for idx, r0, r1 in rows:
        if r1 - r0 != n_samples or r1 > len(arr):
            n_bad += 1
            continue
        mm[idx] = arr[r0:r1].astype(np.float16)
        n_ok += 1
    mm.flush()
    del mm
    return n_ok, n_bad


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.nice > 0:
        try:
            os.nice(a.nice)
        except OSError:
            pass
    root = Path(a.data_root)
    out_dir = a.out_dir or root / "work" / "tensors"
    out_dir.mkdir(parents=True, exist_ok=True)
    drives = pd.read_parquet(root / "build" / "index" / "drives.parquet", columns=["drive_id", "vehicle_id", "file"])
    win = pd.read_parquet(root / "build" / "index" / "windows.parquet",
                          columns=["window_id", "drive_id", "vehicle_id", "group_id", "w_idx", "row0", "row1", "eligible"])
    win = win[win["eligible"].astype(bool)].copy()
    win = win.sort_values(["vehicle_id", "drive_id", "w_idx"], kind="stable").reset_index(drop=True)
    if a.limit_drives > 0:
        keep = win["drive_id"].drop_duplicates().head(a.limit_drives)
        win = win[win["drive_id"].isin(set(keep))].reset_index(drop=True)
    win["tensor_idx"] = np.arange(len(win), dtype=np.int64)
    n = len(win)
    shape = (n, C.WINDOW_SAMPLES, len(CHANNELS))
    print(f"windows={n} drives={win.drive_id.nunique()} shape={shape} bytes={2 * int(np.prod(shape))}")
    if n == 0:
        print("nothing to export")
        return 2
    out_path = out_dir / "X_f16.npy"
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=shape)
    mm[:] = np.float16(np.nan)
    mm.flush()
    del mm
    files = dict(zip(drives["drive_id"], drives["file"]))
    jobs = []
    for did, g in win.groupby("drive_id", sort=False):
        f = files.get(did)
        if f is None:
            continue
        rows = list(zip(g["tensor_idx"].astype(int), g["row0"].astype(int), g["row1"].astype(int)))
        jobs.append((str(root / "build" / f), rows, str(out_path), C.WINDOW_SAMPLES))
    t0 = time.time()
    ok = bad = 0
    with Pool(processes=max(1, a.workers)) as pool:
        for k, (n_ok, n_bad) in enumerate(pool.imap_unordered(_work, jobs, chunksize=4), 1):
            ok += n_ok
            bad += n_bad
            if k % 1000 == 0:
                print(f"progress: drives={k}/{len(jobs)} windows_ok={ok} elapsed_s={time.time() - t0:.0f}")
    win[KEEP].to_parquet(out_dir / "windows.parquet", index=False)
    (out_dir / "channels.json").write_text(json.dumps(CHANNELS))
    print(f"done: windows_ok={ok} windows_bad={bad} drives={len(jobs)} elapsed_s={time.time() - t0:.0f}")
    return 0 if ok == n else 2


if __name__ == "__main__":
    sys.exit(main())
