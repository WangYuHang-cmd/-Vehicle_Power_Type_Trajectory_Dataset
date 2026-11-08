#!/usr/bin/env python3
"""Build the episode cache <data-root>/work/episodes/ (contract (1) of the method-research plan) from the drive tables.

Reads build/index/drives.parquet, build/data/<vehicle_id>/<drive_id>.parquet (through vtt.windows.resolve_drive_path) and
build/index/windows.parquet (for the window_id join). Per vehicle, experiments.episodes.extract_drive runs over every drive
and the result is cached as <out-dir>/parts/<vehicle_id>.npz (E, G, the index as parquet bytes, the drive ids and the
detector-parameter hash); a part is reused when it is newer than every drive file of the vehicle, lists every drive the
index assigns to the vehicle and was built with the same parameters (--force rebuilds). Assembly writes E_f16.npy
(N, 180, 22) and G_f16.npy (N_g, 72, 6) through np.lib.format.open_memmap, episodes.parquet (global episode_id = row
order, global grid_idx, window_id joined on (drive_id, w_idx)), channels.json ({"crop", "grid", "types"}) and params.json.
Rows of drives that are no longer in the drives index (or moved to another vehicle) are dropped at assembly; the stale part
of a vehicle whose rebuild failed, and any part whose stored parameter hash differs from the current one, stays out of the
assembly (counted as parts_skipped; exit code 2 when a rebuild failed). No label or split column is stored. Prints counts
only: episodes per type, human_strict share per type, drives without episodes.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402
from vtt import windows as W  # noqa: E402
from experiments import episodes as EP  # noqa: E402

PART_SUFFIX = ".npz"


def out_root(data_root: Path | str | None = None) -> Path:
    return Path(data_root or C.DATA_ROOT) / "work" / "episodes"


def part_path(parts_dir: Path, vehicle_id: str) -> Path:
    return parts_dir / f"{vehicle_id}{PART_SUFFIX}"


# ----------------------------------------------------------------------------- parts
def write_part(path: Path, E: np.ndarray, G: np.ndarray, idx: pd.DataFrame, drive_ids, phash: str) -> None:
    """Atomic write of one per-vehicle part (the index travels as parquet bytes inside the npz)."""
    buf = io.BytesIO()
    idx.to_parquet(buf, index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{PART_SUFFIX}")
    np.savez(tmp, E=E, G=G, idx=np.frombuffer(buf.getvalue(), dtype=np.uint8),
             drive_ids=np.array(sorted(str(d) for d in drive_ids), dtype=str), params_hash=np.array(phash))
    os.replace(tmp, path)


def read_part(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, set[str], str]:
    """(E, G, idx, drive_ids, params_hash) of a cached part."""
    z = np.load(path, allow_pickle=False)
    idx = pd.read_parquet(io.BytesIO(z["idx"].tobytes()))
    return z["E"], z["G"], idx, set(z["drive_ids"].tolist()), str(z["params_hash"].item())


def part_meta(path: Path) -> tuple[set[str], str] | None:
    """(drive_ids, params_hash) stored in a part without loading its arrays (None when unreadable)."""
    try:
        z = np.load(path, allow_pickle=False)
        return set(z["drive_ids"].tolist()), str(z["params_hash"].item())
    except Exception:                           # noqa: BLE001 - an unreadable part is simply stale
        return None


def part_is_fresh(part: Path, drive_paths: list[Path], drive_ids, phash: str) -> bool:
    """vtt.windows.part_is_fresh (newer than every drive file) plus: covers every current drive of the vehicle and was
    built with the same detector parameters."""
    if not W.part_is_fresh(part, drive_paths):
        return False
    meta = part_meta(part)
    return meta is not None and set(drive_ids) <= meta[0] and meta[1] == phash


def build_vehicle_part(task: tuple) -> tuple[str, int, int, int, str]:
    """Worker: episodes of every drive of one vehicle -> parts_dir/<vehicle_id>.npz.
    task = (vehicle_id, [(drive_id, group_id, path_str), ...], parts_dir_str, params). Returns
    (vehicle_id, n_drives, n_episodes, n_drives_without_episodes, err) where err is an exception class name or ''."""
    vehicle_id, items, parts_dir, p = task
    try:
        Es, Gs, idxs, zero, n_g = [], [], [], 0, 0
        for drive_id, group_id, path in items:
            E, G, idx = EP.extract_drive(Path(path), drive_id, vehicle_id, group_id, p)
            if len(idx) == 0:
                zero += 1
            gi = idx["grid_idx"].to_numpy().copy()
            gi[gi >= 0] += n_g                                   # part-local grid rows
            idx["grid_idx"] = gi.astype(np.int32)
            n_g += len(G)
            Es.append(E)
            Gs.append(G)
            idxs.append(idx)
        E = np.concatenate(Es) if Es else np.zeros((0, EP.L_STORE, len(EP.EP_CHANNELS)), np.float16)
        G = np.concatenate(Gs) if Gs else np.zeros((0, len(EP.SPEED_GRID), len(EP.GRID_CHANNELS)), np.float16)
        idx = pd.concat(idxs, ignore_index=True) if idxs else EP.coerce_index(pd.DataFrame())
        idx["episode_id"] = np.arange(len(idx), dtype=np.int64)
        write_part(part_path(Path(parts_dir), vehicle_id), E, G, idx, [d for d, _, _ in items], EP.params_hash(p))
        return vehicle_id, len(items), len(idx), zero, ""
    except Exception as e:                      # noqa: BLE001 - reported as a count, never an identifier
        return vehicle_id, len(items), 0, 0, type(e).__name__


# ----------------------------------------------------------------------------- assembly
def _save_array(path: Path, shape: tuple, fill) -> None:
    """Write an .npy of `shape` float16 at `path` (atomically) by streaming part arrays through `fill(memmap)`."""
    tmp = path.with_name(f".{path.name}.tmp")
    if shape[0] == 0:
        with open(tmp, "wb") as f:                       # a file object: np.save must not append '.npy' to the temp name
            np.save(f, np.zeros(shape, np.float16))
    else:
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=shape)
        fill(mm)
        mm.flush()
        del mm
    os.replace(tmp, path)


def assemble(part_paths: list[Path], drives: pd.DataFrame, out_dir: Path, windows_path: Path | None,
             p: dict = EP.PARAMS, log=print) -> tuple[pd.DataFrame, dict]:
    """Concatenate the parts into E_f16.npy / G_f16.npy / episodes.parquet (+ channels.json, params.json) under out_dir.
    Rows whose (drive_id, vehicle_id) pair is not in the drives index are dropped; a part built with other detector
    parameters (stored hash != params_hash(p)) is skipped and counted. Returns (index, counts)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    phash = EP.params_hash(p)
    veh = drives.drop_duplicates("drive_id").set_index("drive_id")["vehicle_id"]
    metas, nE, nG, dropped, skipped = [], 0, 0, 0, 0
    for path in part_paths:
        z = np.load(path, allow_pickle=False)
        if str(z["params_hash"].item()) != phash:
            skipped += 1
            continue
        idx = pd.read_parquet(io.BytesIO(z["idx"].tobytes()))
        want = veh.reindex(idx["drive_id"]).to_numpy(object)
        keep = np.array([isinstance(w, str) and w == v for w, v in zip(want, idx["vehicle_id"].to_numpy(object))], bool)
        dropped += int((~keep).sum())
        idx = idx.loc[keep].reset_index(drop=True)
        n_grid = int((idx["grid_idx"].to_numpy() >= 0).sum())
        metas.append((path, keep, idx, nE, nG))
        nE += len(idx)
        nG += n_grid
    frames: list[pd.DataFrame] = []

    def fill_E(mm):
        for path, keep, idx, e0, _ in metas:
            if len(idx):
                mm[e0:e0 + len(idx)] = np.load(path, allow_pickle=False)["E"][keep]

    def fill_G(mm):
        for path, keep, idx, _, g0 in metas:
            gi = idx["grid_idx"].to_numpy()
            has = gi >= 0
            if has.any():
                mm[g0:g0 + int(has.sum())] = np.load(path, allow_pickle=False)["G"][gi[has]]

    _save_array(out_dir / "E_f16.npy", (nE, EP.L_STORE, len(EP.EP_CHANNELS)), fill_E)
    _save_array(out_dir / "G_f16.npy", (nG, len(EP.SPEED_GRID), len(EP.GRID_CHANNELS)), fill_G)
    for path, keep, idx, e0, g0 in metas:
        gi = idx["grid_idx"].to_numpy()
        has = gi >= 0
        new = np.full(len(idx), -1, np.int32)
        new[has] = g0 + np.arange(int(has.sum()), dtype=np.int32)
        idx["grid_idx"] = new
        frames.append(idx)
    index = pd.concat(frames, ignore_index=True) if frames else EP.coerce_index(pd.DataFrame())
    index["episode_id"] = np.arange(len(index), dtype=np.int64)
    n_window_ids = 0
    if windows_path is not None and Path(windows_path).exists() and len(index):
        win = pd.read_parquet(windows_path, columns=["window_id", "drive_id", "w_idx"])
        win = win.drop_duplicates(["drive_id", "w_idx"])
        index = index.drop(columns=["window_id"]).merge(win, on=["drive_id", "w_idx"], how="left", validate="many_to_one")
        n_window_ids = int(index["window_id"].notna().sum())
    index = EP.coerce_index(index)
    index.to_parquet(out_dir / "episodes.parquet", index=False)
    (out_dir / "channels.json").write_text(json.dumps({"crop": list(EP.EP_CHANNELS), "grid": list(EP.GRID_CHANNELS),
                                                       "types": list(EP.TYPES)}, indent=1))
    (out_dir / "params.json").write_text(json.dumps(dict(p, params_hash=EP.params_hash(p), l_store=EP.L_STORE,
                                                         pre_samples=dict(PRE_DEFAULT=EP.PRE_DEFAULT, **EP.PRE_SAMPLES),
                                                         speed_grid=dict(start=0.0, stop=36.0, step=0.5, n=len(EP.SPEED_GRID))),
                                                    indent=1, default=list))
    counts = dict(episodes=int(len(index)), grids=int(nG), stale_rows_dropped=int(dropped), window_ids_joined=n_window_ids,
                  drives_with_episodes=int(index["drive_id"].nunique()), parts_skipped=int(skipped))
    return index, counts


# ----------------------------------------------------------------------------- driver
def build_episodes(drives: pd.DataFrame, roots: W.Roots, out_dir: Path, p: dict = EP.PARAMS, workers: int = 1,
                   force: bool = False, limit_vehicles: int = 0, log=print) -> tuple[pd.DataFrame, dict]:
    """Episode cache for every drive of `drives` (optionally the first `limit_vehicles` vehicles in sorted order).
    Returns (index, counts)."""
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    phash = EP.params_hash(p)
    vehicles = sorted(drives["vehicle_id"].astype(str).unique().tolist())
    if limit_vehicles:
        vehicles = vehicles[:limit_vehicles]
        drives = drives[drives["vehicle_id"].isin(vehicles)]
    tasks, reused, missing = [], [], 0
    for vehicle_id, grp in drives.groupby("vehicle_id", sort=True):
        items = []
        for _, row in grp.iterrows():
            path = W.resolve_drive_path(row, roots)
            if not path.exists():
                missing += 1
                continue
            gid = row.get("group_id") if hasattr(row, "get") else None
            items.append((str(row["drive_id"]), None if gid is None or gid != gid else str(gid), str(path)))
        if not items:
            continue
        part = part_path(parts_dir, str(vehicle_id))
        if not force and part_is_fresh(part, [Path(x[2]) for x in items], [x[0] for x in items], phash):
            reused.append(part)
        else:
            tasks.append((str(vehicle_id), items, str(parts_dir), p))
    log(f"vehicles={len(vehicles)} drives={len(drives)} missing_files={missing} parts_reused={len(reused)} parts_to_build={len(tasks)}")
    built = zero = 0
    failed: set[str] = set()
    t0 = time.time()
    if tasks and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(build_vehicle_part, tasks, chunksize=2))
    else:
        results = [build_vehicle_part(task) for task in tasks]
    for vehicle_id, _, n_ep, n_zero, err in results:
        built += n_ep
        zero += n_zero
        if err:
            failed.add(vehicle_id)
    errors = len(failed)
    if tasks:
        log(f"parts_built={len(tasks) - errors} part_errors={errors} episodes_in_new_parts={built} minutes={(time.time() - t0) / 60:.1f}")
    # a failed rebuild leaves the vehicle's old part (stale or built with other parameters) on disk: keep it out of the assembly
    fresh_new = {part_path(parts_dir, t[0]) for t in tasks if t[0] not in failed and part_path(parts_dir, t[0]).exists()}
    stale_kept = sum(part_path(parts_dir, v).exists() for v in failed)
    part_paths = sorted(set(reused) | fresh_new)
    index, counts = assemble(part_paths, drives, out_dir, roots.windows if roots.windows.exists() else None, p, log)
    counts["parts_skipped"] = int(counts.get("parts_skipped", 0) + stale_kept)
    counts.update(vehicles=int(len(vehicles)), drives=int(len(drives)), missing_files=int(missing), parts_reused=int(len(reused)),
                  parts_built=int(len(tasks) - errors), part_errors=int(errors),
                  drives_without_episodes=int(len(set(drives["drive_id"].astype(str)) - set(index["drive_id"].astype(str)))))
    return index, counts


def summary_lines(index: pd.DataFrame, counts: dict) -> list[str]:
    """Count lines for the terminal: totals, per-type counts and human_strict shares."""
    tc = EP.type_counts(index)
    lines = [" ".join(f"{k}={v}" for k, v in counts.items()),
             "episodes by type: " + " ".join(f"{k}={v}" for k, v in tc.items())]
    if len(index):
        hs = index.groupby("type_id")["human_strict"].mean()
        lines.append("human_strict share by type: " + " ".join(
            f"{t}={hs.get(i, float('nan')):.3f}" if tc[t] else f"{t}=-" for i, t in enumerate(EP.TYPES)))
        lines.append(f"episodes with a gap in the crop: {int(index['gap_in_crop'].sum())}; "
                     f"with a window_id: {int(index['window_id'].notna().sum())}; grids: {int((index['grid_idx'] >= 0).sum())}")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None, help="default <data-root>/work/episodes")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nice", type=int, default=10, help="os.nice increment applied to this process and its workers")
    ap.add_argument("--limit-vehicles", type=int, default=0, help="only the first N vehicles (sorted ids; smoke test)")
    ap.add_argument("--force", action="store_true", help="rebuild cached per-vehicle parts")
    ap.add_argument("--no-turn", action="store_true", help="skip TURN episodes (changes the parameter hash)")
    a = ap.parse_args(argv)
    if a.nice:
        try:
            os.nice(a.nice)
        except OSError:
            pass
    roots = W.Roots(a.data_root)
    if not roots.drives.exists():
        print("missing drives index; run the S5 build first")
        return 1
    p = dict(EP.PARAMS)
    if a.no_turn:
        p["turn_enabled"] = False
    out_dir = a.out_dir or out_root(a.data_root)
    drives = pd.read_parquet(roots.drives)
    index, counts = build_episodes(drives, roots, out_dir, p, workers=max(1, a.workers), force=a.force,
                                   limit_vehicles=a.limit_vehicles)
    for line in summary_lines(index, counts):
        print(line)
    return 0 if counts["part_errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
