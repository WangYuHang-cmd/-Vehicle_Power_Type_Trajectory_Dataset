"""Release assembly (stage S9): build/ -> release/<version>.

Order: index tables -> splits -> schema -> data (hardlinks) -> window shards -> readme numbers + README -> LICENSE
-> CHECKSUMS.sha256 -> pre-upload gate -> .gate_ok. Every step is idempotent: existing hardlinks are kept, a shard is
kept only when its content fingerprint (planned index rows + the size / mtime of the drive tables it reads, see
`shard_fingerprint`) and its own size / mtime match the entry in `.cache/vtt_shards.json`; everything else is
rewritten deterministically. Only counts are printed. Units: hours = nominal rows / 10 / 3600.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from vtt import config as C
from vtt import private as PV
from vtt import readme as RM
from vtt import schema as S
from vtt.gate import rules as R
from vtt.gate import scan as SC

DATASET_NAME = "Vehicle_Power_Type_Trajectory_Dataset"
HF_URL = f"https://huggingface.co/datasets/{C.HF_REPO}"
SHARD_MAX_BYTES = 500 * 1024 * 1024
SHARD_ROW_GROUP = 1000
WINDOW_HOURS = C.WINDOW_SAMPLES * C.NOMINAL_DT_S / 3600.0
TASKS = {"size": ("size_class", "task_size_eligible"), "bev_ice": ("powertrain", "task_bev_ice_eligible")}
SPLIT_COLS = ("split_platform", "split_device")
CHECKSUM_CACHE = ".cache/vtt_checksums.json"
SHARD_CACHE = ".cache/vtt_shards.json"           # shard rel path -> [fingerprint, size, mtime_ns] of the written file


class AcceptanceError(RuntimeError):
    """A release acceptance check failed (the script exits 2)."""


@dataclass
class Roots:
    data: Path
    build: Path
    release: Path
    private: Path

    @classmethod
    def from_data_root(cls, data_root: Path, build: Path | None = None, release: Path | None = None,
                       private: Path | None = None) -> "Roots":
        d = Path(data_root)
        return cls(d, Path(build or d / "build"), Path(release or d / "release" / C.RELEASE_VERSION), Path(private or d / "private"))


def log(msg: str) -> None:
    print(PV.mask(msg), flush=True)


# ------------------------------------------------------------------------------------------------ inputs
def resolve_drive_file(build: Path, file: str, vehicle_id: str, drive_id: str) -> Path:
    """Drive table of the build: index `file` relative to build/, then under its data folder, else data/<vehicle>/<drive>.parquet."""
    for cand in (build / file, build / "data" / file, build / "data" / vehicle_id / f"{drive_id}.parquet"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"drive table missing for {vehicle_id}/{drive_id}")


def load_build(build: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    drives = pd.read_parquet(build / "index" / "drives.parquet")
    windows = pd.read_parquet(build / "index" / "windows.parquet")
    for col in ("vehicle_id", "drive_id", "group_id", "file"):
        if col not in drives.columns:
            raise AcceptanceError(f"drives index lacks column {col}")
    for col in ("window_id", "drive_id", "row0", "row1", "eligible", "task_size_eligible", "task_bev_ice_eligible"):
        if col not in windows.columns:
            raise AcceptanceError(f"windows index lacks column {col}")
    if getattr(C, "DROP_UNKNOWN_MODEL", False) and "model" in drives.columns:
        # user ruling: routes whose car model is unknown are not part of the release
        unknown = drives["model"].isna() | (drives["model"].astype(str) == "Unknown")
        if unknown.any():
            log(f"s09: dropping {int(unknown.sum())} drives with an unknown car model")
            drives = drives[~unknown].reset_index(drop=True)
            windows = windows[windows["drive_id"].isin(set(drives["drive_id"]))].reset_index(drop=True)
    sj = build / "splits" / "splits.json"
    splits_json = json.loads(sj.read_text()) if sj.exists() else None
    return drives, windows, splits_json


# ------------------------------------------------------------------------------------------------ index tables
def release_file_col(drives: pd.DataFrame) -> pd.Series:
    return "data/" + drives["vehicle_id"].astype(str) + "/" + drives["drive_id"].astype(str) + ".parquet"


def _first(g, cols):
    return g[list(cols)].first()


def vehicles_table(drives: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    g = drives.sort_values("drive_id").groupby("vehicle_id", sort=True)
    out = _first(g, ("group_id", "platform", "brand", "model", "body", "size_class", "powertrain", "powertrain_raw", "region",
                     "split_device", "split_platform", "low_support_cell"))
    out["n_drives"] = g.size()
    out["duration_h"] = g["duration_s"].sum() / 3600.0
    out["moving_h"] = g["moving_s"].sum() / 3600.0
    out["das_any_h"] = g["das_any_s"].sum() / 3600.0
    elig = windows[windows["eligible"].fillna(False).astype(bool)].groupby("vehicle_id").size()
    out["n_windows_eligible"] = elig.reindex(out.index).fillna(0).astype(int)
    return out.reset_index()[S.names(R.VEHICLES_COLUMNS)]


def platforms_table(drives: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    d = drives[drives["platform"].notna()].sort_values("drive_id")
    g = d.groupby("platform", sort=True)
    out = _first(g, ("brand", "model", "body", "size_code", "wheelbase_mm", "track_front_mm", "track_rear_mm", "footprint_m2",
                     "curb_mass_kg", "size_class", "powertrain", "powertrain_raw"))
    out["n_vehicles"] = g["vehicle_id"].nunique()
    out["n_groups"] = g["group_id"].nunique()
    out["n_drives"] = g.size()
    out["duration_h"] = g["duration_s"].sum() / 3600.0
    w = windows[windows["eligible"].fillna(False).astype(bool)]
    out["n_windows_eligible"] = w.groupby("platform").size().reindex(out.index).fillna(0).astype(int)
    splits = g["split_platform"].agg(lambda s: s.dropna().unique().tolist())
    out["split_platform"] = [_platform_split_value(v) for v in splits.reindex(out.index)]
    return out.reset_index()[S.names(R.PLATFORMS_COLUMNS)]


def _platform_split_value(values) -> str | None:
    """A platform's split from its drives' split_platform values: the single value of the non-excluded drives, 'mixed'
    when they span several, 'excluded' when every drive is excluded (S8: the drive's group also drove a held-out
    platform), None without any value."""
    vals = [v for v in (values if isinstance(values, list) else []) if v is not None]
    kept = sorted({v for v in vals if v != "excluded"})
    if len(kept) == 1:
        return kept[0]
    if len(kept) > 1:
        return "mixed"
    return "excluded" if vals else None


def compute_split_summary(windows: pd.DataFrame) -> pd.DataFrame:
    """Hours / windows / groups / platforms per task x class x split for both split columns (fallback when S8 wrote none)."""
    rows = []
    w = windows[windows["eligible"].fillna(False).astype(bool)]
    tasks = dict(TASKS)
    tasks["powertrain4"] = ("powertrain", None)
    for task, (label, elig) in tasks.items():
        sub = w[w[elig].fillna(False).astype(bool)] if elig else w[w[label].notna()]
        for split_col in SPLIT_COLS:
            if split_col not in sub.columns:
                continue
            for (cls, split), grp in sub.groupby([label, split_col], dropna=False):
                rows.append({"task": task, "protocol": split_col, "class": cls, "split": split, "windows": int(len(grp)),
                             "hours": round(len(grp) * WINDOW_HOURS, 3), "groups": int(grp["group_id"].nunique()),
                             "platforms": int(grp["platform"].nunique())})
    return pd.DataFrame(rows, columns=["task", "protocol", "class", "split", "windows", "hours", "groups", "platforms"])


def write_index_tables(roots: Roots, drives: pd.DataFrame, windows: pd.DataFrame) -> dict:
    idx = roots.release / "index"
    idx.mkdir(parents=True, exist_ok=True)
    rel_drives = drives.copy()
    rel_drives["file"] = release_file_col(rel_drives)
    S.write_public_parquet(rel_drives, idx / "drives.parquet", S.DRIVES_INDEX_COLUMNS, "drives")
    S.write_public_parquet(windows, idx / "windows_60s.parquet", S.WINDOWS_INDEX_COLUMNS, "windows")
    veh = vehicles_table(drives, windows)
    S.write_public_parquet(veh, idx / "vehicles.parquet", R.VEHICLES_COLUMNS, "vehicles")
    plat = platforms_table(drives, windows)
    S.write_public_parquet(plat, idx / "platforms.parquet", R.PLATFORMS_COLUMNS, "platforms")
    src_summary = roots.build / "index" / "split_summary.csv"
    if src_summary.exists():
        shutil.copyfile(src_summary, idx / "split_summary.csv")
        summary_source = "build"
    else:
        compute_split_summary(windows).to_csv(idx / "split_summary.csv", index=False)
        summary_source = "computed"
    return {"n_drives": int(len(drives)), "n_windows": int(len(windows)), "n_vehicles": int(len(veh)), "n_platforms": int(len(plat)),
            "split_summary": summary_source}


# ------------------------------------------------------------------------------------------------ splits / schema
def write_splits(roots: Roots) -> dict:
    """Copy build/splits: JSON verbatim, parquet re-written with the two metadata keys and without pandas index columns."""
    src = roots.build / "splits"
    dst = roots.release / "splits"
    if not src.is_dir():
        return {"n_files": 0}
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.iterdir()):
        if p.suffix == ".json":
            shutil.copyfile(p, dst / p.name)
        elif p.suffix == ".parquet":
            t = pq.read_table(p)
            drop = [c for c in t.column_names if c.startswith("__index_level")]
            if drop:
                t = t.drop_columns(drop)
            t = t.replace_schema_metadata({b"vtt_table": p.stem.encode(), b"vtt_schema_version": S.SCHEMA_VERSION.encode()})
            tmp = dst / f".{p.name}.tmp"
            pq.write_table(t, tmp, compression="zstd")
            tmp.replace(dst / p.name)
        else:
            continue
        n += 1
    for p in dst.iterdir():                          # stale files from an earlier build
        if p.name not in {q.name for q in src.iterdir()}:
            p.unlink()
    return {"n_files": n}


def schema_json(name: str, spec, extra: list[dict] | None = None) -> dict:
    cols = []
    for t in spec:
        if len(t) == 5:
            n, dt, unit, desc, origin = t
        else:
            n, dt, desc = t
            unit, origin = "", ""
        cols.append({"name": n, "dtype": dt, "unit": unit, "description": desc, "origin": origin})
    cols += extra or []
    return {"table": name, "schema_version": S.SCHEMA_VERSION, "config_version": C.CONFIG_VERSION, "columns": cols}


def write_schema(roots: Roots) -> dict:
    out = roots.release / "schema"
    out.mkdir(parents=True, exist_ok=True)
    units = {t[0]: (t[2], t[3]) for t in S.DRIVE_COLUMNS}
    seq = [{"name": n, "dtype": f"fixed_size_list<float32>[{C.WINDOW_SAMPLES}]", "unit": units[n][0],
            "description": f"{C.WINDOW_SAMPLES}-step sequence (rows row0..row1 of the drive table) of: {units[n][1]}; booleans as 0/1",
            "origin": f"drive table column {n}"} for n in C.OFFICIAL_INPUTS]
    tables = {"drive": schema_json("drive", S.DRIVE_COLUMNS), "drives": schema_json("drives", S.DRIVES_INDEX_COLUMNS),
              "windows_60s": schema_json("windows", S.WINDOWS_INDEX_COLUMNS), "vehicles": schema_json("vehicles", R.VEHICLES_COLUMNS),
              "platforms": schema_json("platforms", R.PLATFORMS_COLUMNS),
              "windows_shard": schema_json(R.SHARD_TABLE, S.WINDOWS_INDEX_COLUMNS, seq)}
    for name, obj in tables.items():
        PV.write_json_atomic(obj, out / f"{name}.json")
    PV.write_json_atomic({"official_inputs": list(C.OFFICIAL_INPUTS), "window_samples": C.WINDOW_SAMPLES, "nominal_hz": C.NOMINAL_HZ,
                          "not_inputs": [t[0] for t in S.DRIVE_COLUMNS if t[0] not in C.OFFICIAL_INPUTS]}, out / "official_inputs.json")
    return {"n_files": len(tables) + 1}


# ------------------------------------------------------------------------------------------------ data
def _same_file(a: Path, b: Path) -> bool:
    """Same inode (hardlink) or an exact copy (size and nanosecond mtime, as shutil.copy2 preserves them)."""
    try:
        sa, sb = a.stat(), b.stat()
    except OSError:
        return False
    return (sa.st_ino == sb.st_ino and sa.st_dev == sb.st_dev) or (sa.st_size == sb.st_size and sa.st_mtime_ns == sb.st_mtime_ns)


def link_data(roots: Roots, drives: pd.DataFrame, copy: bool = False, workers: int = 4) -> dict:
    """Hardlink (or copy) every drive table into release/data; remove release/data files no longer indexed."""
    dst_root = roots.release / "data"
    dst_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for r in drives.itertuples(index=False):
        src = resolve_drive_file(roots.build, str(r.file), str(r.vehicle_id), str(r.drive_id))
        jobs.append((src, dst_root / str(r.vehicle_id) / f"{r.drive_id}.parquet"))
    counts = {"linked": 0, "copied": 0, "kept": 0, "removed": 0, "bytes": 0}

    def do(job):
        src, dst = job
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if _same_file(src, dst):
                return "kept", dst.stat().st_size
            dst.unlink()
        if not copy:
            try:
                os.link(src, dst)
                return "linked", dst.stat().st_size
            except OSError:
                pass
        shutil.copy2(src, dst)
        return "copied", dst.stat().st_size

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for kind, size in ex.map(do, jobs):
            counts[kind] += 1
            counts["bytes"] += size
    wanted = {dst for _, dst in jobs}
    for p in dst_root.rglob("*.parquet"):
        if p not in wanted:
            p.unlink()
            counts["removed"] += 1
    return counts


# ------------------------------------------------------------------------------------------------ shards
@dataclass
class Shard:
    split: str
    index: int
    n_total: int
    rows: pd.DataFrame

    @property
    def name(self) -> str:
        return f"{self.split}-{self.index:05d}-of-{self.n_total:05d}.parquet"


def select_shard_windows(windows: pd.DataFrame, split_col: str) -> pd.DataFrame:
    """Eligible windows that are eligible for at least one task and have a split; sorted (split, drive_id, w_idx)."""
    if split_col not in windows.columns:
        raise AcceptanceError(f"windows index lacks split column {split_col}")
    m = windows["eligible"].fillna(False).astype(bool)
    m &= windows["task_size_eligible"].fillna(False).astype(bool) | windows["task_bev_ice_eligible"].fillna(False).astype(bool)
    m &= windows[split_col].isin(list(R.SPLIT_VALUES))
    sel = windows[m].copy()
    sel["_split"] = sel[split_col]
    return sel.sort_values(["_split", "drive_id", "w_idx"], kind="mergesort").reset_index(drop=True)


def plan_shards(sel: pd.DataFrame, per_shard: int) -> list[Shard]:
    plan = []
    for split in sorted(R.SPLIT_VALUES):
        part = sel[sel["_split"] == split]
        if not len(part):
            continue
        n_total = int(math.ceil(len(part) / per_shard))
        for i in range(n_total):
            plan.append(Shard(split, i, n_total, part.iloc[i * per_shard:(i + 1) * per_shard]))
    return plan


def shard_schema() -> pa.Schema:
    fields = list(S.arrow_schema(S.WINDOWS_INDEX_COLUMNS, R.SHARD_TABLE))
    fields += [pa.field(n, pa.list_(pa.float32(), C.WINDOW_SAMPLES), nullable=True) for n in C.OFFICIAL_INPUTS]
    return pa.schema(fields, metadata={b"vtt_table": R.SHARD_TABLE.encode(), b"vtt_schema_version": S.SCHEMA_VERSION.encode()})


def _read_inputs(path: Path) -> dict[str, np.ndarray]:
    t = pq.read_table(path, columns=list(C.OFFICIAL_INPUTS))
    out = {}
    for n in C.OFFICIAL_INPUTS:
        col = t[n]
        if pa.types.is_boolean(col.type):
            out[n] = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float32)
        else:
            out[n] = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float32)
    return out


def build_shard_table(shard: Shard, files: dict[str, Path], workers: int = 4) -> pa.Table:
    rows = shard.rows
    drive_ids = list(dict.fromkeys(rows["drive_id"].tolist()))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        arrays = dict(zip(drive_ids, ex.map(lambda d: _read_inputs(files[d]), drive_ids)))
    n = len(rows)
    seqs = {name: np.empty((n, C.WINDOW_SAMPLES), dtype=np.float32) for name in C.OFFICIAL_INPUTS}
    for k, r in enumerate(rows.itertuples(index=False)):
        r0, r1 = int(r.row0), int(r.row1)
        if r1 - r0 != C.WINDOW_SAMPLES:
            raise AcceptanceError(f"window {r.window_id} spans {r1 - r0} rows, expected {C.WINDOW_SAMPLES}")
        arr = arrays[r.drive_id]
        for name in C.OFFICIAL_INPUTS:
            if len(arr[name]) < r1:
                raise AcceptanceError(f"window {r.window_id} exceeds its drive table")
            seqs[name][k] = arr[name][r0:r1]
    idx = S.coerce(rows.drop(columns=["_split"]), S.WINDOWS_INDEX_COLUMNS)
    table = pa.Table.from_pandas(idx, schema=S.arrow_schema(S.WINDOWS_INDEX_COLUMNS, R.SHARD_TABLE), preserve_index=False, safe=False)
    for name in C.OFFICIAL_INPUTS:
        flat = pa.array(seqs[name].reshape(-1), type=pa.float32())
        table = table.append_column(pa.field(name, pa.list_(pa.float32(), C.WINDOW_SAMPLES)), pa.FixedSizeListArray.from_arrays(flat, C.WINDOW_SAMPLES))
    return table.replace_schema_metadata({b"vtt_table": R.SHARD_TABLE.encode(), b"vtt_schema_version": S.SCHEMA_VERSION.encode()})


def shard_fingerprint(shard: Shard, files: dict[str, Path]) -> str:
    """sha256 over everything a shard's content depends on: the schema names and sequence length, every index value of
    its planned rows (labels, splits, membership, order) and the size + mtime of each drive table it reads."""
    h = hashlib.sha256()
    head = {"names": shard_schema().names, "samples": C.WINDOW_SAMPLES, "schema": S.SCHEMA_VERSION, "inputs": list(C.OFFICIAL_INPUTS)}
    h.update(json.dumps(head, sort_keys=True).encode())
    idx = S.coerce(shard.rows.drop(columns=["_split"]), S.WINDOWS_INDEX_COLUMNS)
    h.update(pd.util.hash_pandas_object(idx, index=False).to_numpy().tobytes())
    for d in dict.fromkeys(shard.rows["drive_id"].astype(str).tolist()):
        st = files[d].stat()
        h.update(f"{d}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _shard_complete(path: Path, n_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        md = pq.read_metadata(path)
        sch = pq.read_schema(path)
    except Exception:                                 # noqa: BLE001 - a broken file is rebuilt
        return False
    return md.num_rows == n_rows and list(sch.names) == shard_schema().names


def _shard_reusable(path: Path, n_rows: int, fingerprint: str, entry) -> bool:
    """An existing shard is reused only when its cache entry carries the current fingerprint and the file on disk is
    the one that was written (size and mtime), besides the row count and column names."""
    if not entry or not isinstance(entry, list) or len(entry) != 3 or entry[0] != fingerprint:
        return False
    try:
        st = path.stat()
    except OSError:
        return False
    if st.st_size != entry[1] or st.st_mtime_ns != entry[2]:
        return False
    return _shard_complete(path, n_rows)


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
    except Exception:                                 # noqa: BLE001
        return {}
    return obj if isinstance(obj, dict) else {}


def write_shards(roots: Roots, drives: pd.DataFrame, windows: pd.DataFrame, split_col: str, per_shard: int, workers: int = 4,
                 force: bool = False) -> dict:
    sel = select_shard_windows(windows, split_col)
    plan = plan_shards(sel, per_shard)
    files = {str(r.drive_id): resolve_drive_file(roots.build, str(r.file), str(r.vehicle_id), str(r.drive_id))
             for r in drives.itertuples(index=False)}
    root = roots.release / "windows"
    cache_path = roots.release / SHARD_CACHE
    cache = _load_cache(cache_path)
    new_cache: dict[str, list] = {}
    wanted = set()
    per_split: dict[str, dict] = {}
    counts = {"n_shards": len(plan), "n_windows": int(len(sel)), "written": 0, "kept": 0, "removed": 0, "bytes": 0}
    for shard in plan:
        dst = root / shard.split / shard.name
        rel = f"windows/{shard.split}/{shard.name}"
        wanted.add(dst)
        fp = shard_fingerprint(shard, files)
        if not force and _shard_reusable(dst, len(shard.rows), fp, cache.get(rel)):
            counts["kept"] += 1
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            table = build_shard_table(shard, files, workers)
            tmp = dst.with_name(f".{dst.name}.tmp")
            pq.write_table(table, tmp, compression="zstd", row_group_size=SHARD_ROW_GROUP)
            tmp.replace(dst)
            counts["written"] += 1
        st = dst.stat()
        new_cache[rel] = [fp, st.st_size, st.st_mtime_ns]
        PV.write_json_atomic({**cache, **new_cache}, cache_path)      # resumable: every finished shard is recorded at once
        size = st.st_size
        if size > SHARD_MAX_BYTES:
            raise AcceptanceError(f"shard {shard.name} is {size} bytes (> {SHARD_MAX_BYTES}); lower --shard-windows")
        counts["bytes"] += size
        ps = per_split.setdefault(shard.split, {"split": shard.split, "n_files": 0, "windows": 0, "bytes": 0})
        ps["n_files"] += 1
        ps["windows"] += len(shard.rows)
        ps["bytes"] += size
    if root.exists():
        for p in root.rglob("*.parquet"):
            if p not in wanted:
                p.unlink()
                counts["removed"] += 1
        for d in sorted((d for d in root.iterdir() if d.is_dir()), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
    PV.write_json_atomic(new_cache, cache_path)                       # entries of removed shards dropped
    counts["per_split"] = [per_split[s] for s in sorted(per_split)]
    return counts


# ------------------------------------------------------------------------------------------------ readme numbers
def size_category(n: int) -> str:
    for lim, name in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"), (1_000_000, "100K<n<1M"),
                      (10_000_000, "1M<n<10M"), (100_000_000, "10M<n<100M")):
        if n < lim:
            return name
    return "100M<n<1B"


def _pct(num: float, den: float) -> float:
    return round(100.0 * num / den, 1) if den else 0.0


def _bool(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def class_rows(windows: pd.DataFrame) -> list[dict]:
    w = windows[_bool(windows["eligible"])]
    rows = []
    for task, label, classes in (("size", "size_class", list(C.SIZE_CLASSES)), ("powertrain", "powertrain", list(C.POWERTRAIN4))):
        for cls in classes + [None]:
            grp = w[w[label].isna()] if cls is None else w[w[label] == cls]
            rows.append({"task": task, "class": cls if cls is not None else "null", "windows": int(len(grp)),
                         "hours": round(len(grp) * WINDOW_HOURS, 1), "vehicles": int(grp["vehicle_id"].nunique()),
                         "platforms": int(grp["platform"].nunique())})
    return rows


def split_rows(windows: pd.DataFrame) -> list[dict]:
    w = windows[_bool(windows["eligible"])]
    rows = []
    for task, (label, elig) in TASKS.items():
        sub = w[_bool(w[elig])]
        for split_col in SPLIT_COLS:
            if split_col not in sub.columns:
                continue
            for split in ("train", "validation", "test"):
                grp = sub[sub[split_col] == split]
                rows.append({"task": task, "protocol": split_col, "split": split, "windows": int(len(grp)),
                             "hours": round(len(grp) * WINDOW_HOURS, 1), "groups": int(grp["group_id"].nunique()),
                             "platforms": int(grp["platform"].nunique())})
    return rows


def compute_readme_numbers(drives: pd.DataFrame, windows: pd.DataFrame, splits_json: dict | None, shard_counts: dict,
                           data_bytes: int, split_col: str, contact_email: str) -> dict:
    """Every number of the dataset card (JSON-serializable; display formatting happens in vtt.readme.fmt)."""
    w_el = windows[_bool(windows["eligible"])]
    dur = float(drives["duration_s"].sum())
    t1, t2 = (C.FOOTPRINT_THRESHOLDS_M2 or (None, None))
    heldout = None
    if splits_json and isinstance(splits_json.get("heldout_platforms"), list):
        heldout = len(splits_json["heldout_platforms"])
    if heldout is None:
        heldout = int(drives.loc[drives["split_platform"].isin(["validation", "test"]), "platform"].nunique()) if "split_platform" in drives else 0
    n_shard_windows = int(shard_counts.get("n_windows", 0))
    nums = {
        # data-derived
        "n_drives": int(len(drives)), "n_vehicles": int(drives["vehicle_id"].nunique()), "n_groups": int(drives["group_id"].nunique()),
        "n_platforms": int(drives["platform"].nunique()), "n_brands": int(drives["brand"].nunique()), "n_models": int(drives["model"].nunique()),
        "total_hours": round(dur / 3600.0, 1), "moving_hours": round(float(drives["moving_s"].sum()) / 3600.0, 1),
        "das_share_pct": _pct(float(drives["das_s"].sum()), dur), "das_any_share_pct": _pct(float(drives["das_any_s"].sum()), dur),
        "n_windows_total": int(len(windows)), "n_windows_eligible": int(len(w_el)),
        "n_windows_human_only_strict": int(_bool(w_el["human_only_strict"]).sum()),
        "n_windows_size_task": int(_bool(w_el["task_size_eligible"]).sum()),
        "n_windows_bev_ice_task": int(_bool(w_el["task_bev_ice_eligible"]).sum()),
        "n_windows_pt4_task": int(w_el["powertrain"].notna().sum()),
        "n_shard_windows": n_shard_windows, "n_shards": int(shard_counts.get("n_shards", 0)),
        "data_gb": round(data_bytes / 1e9, 2), "shards_gb": round(shard_counts.get("bytes", 0) / 1e9, 2),
        "size_category": size_category(n_shard_windows),
        "footprint_t1": f"{t1:.2f}" if t1 is not None else "n/a", "footprint_t2": f"{t2:.2f}" if t2 is not None else "n/a",
        "n_heldout_platforms": int(heldout),
        "pt_null_pct": _pct(float(drives["powertrain"].isna().sum()), len(drives)),
        "radar_unavailable_pct": _pct(float(_bool(drives["radar_unavailable"]).sum()), len(drives)),
        "gas_dead_pct": _pct(float((~_bool(drives["gas_alive"])).sum()), len(drives)),
        "brake_dead_pct": _pct(float((~_bool(drives["brake_alive"])).sum()), len(drives)),
        "class_rows": class_rows(windows), "split_rows": split_rows(windows), "shard_rows": shard_counts.get("per_split", []),
        # configuration constants
        "hz": C.NOMINAL_HZ, "window_s": int(round(C.WINDOW_SAMPLES * C.NOMINAL_DT_S)), "n_samples": C.WINDOW_SAMPLES,
        "gap_s": C.CS_GAP_S, "thw_cf_s": f"{C.THW_CF_S:g}", "thw_v_min": f"{C.THW_V_MIN_MPS:g}", "moving_v": f"{C.MOVING_V_MPS:g}",
        "min_moving_pct": int(round(100 * C.WINDOW_MIN_MOVING_FRAC)), "human_guard_s": f"{C.HUMAN_GUARD_S:g}",
        "lead_max_age_s": f"{C.LEAD_MAX_AGE_S:g}", "pose_max_age_s": f"{C.POSE_MAX_AGE_S:g}", "torque_q_pct": int(round(100 * C.TORQUE_NORM_Q)),
        "train_pct": int(round(100 * C.SPLIT_FRACS["train"])), "val_pct": int(round(100 * C.SPLIT_FRACS["validation"])),
        "test_pct": int(round(100 * C.SPLIT_FRACS["test"])), "holdout_pct": int(round(100 * C.PLATFORM_HOLDOUT_FRAC)),
        "low_support_hours": f"{C.LOW_SUPPORT['min_test_hours']:g}", "low_support_groups": C.LOW_SUPPORT["min_groups"],
        "low_support_platforms": C.LOW_SUPPORT["min_platforms"],
        "bal_group_pct": int(round(100 * C.BALANCED_EVAL["max_group_share"])), "bal_platform_pct": int(round(100 * C.BALANCED_EVAL["max_platform_share"])),
        "cw_power": f"{C.CLASS_WEIGHT_POWER:g}", "cw_cap": f"{C.CLASS_WEIGHT_CAP:g}", "split_seed": str(C.SPLIT_SEED),
        "split_col": split_col, "shard_max_mb": SHARD_MAX_BYTES // (1024 * 1024),
        "release_version": C.RELEASE_VERSION, "schema_version": S.SCHEMA_VERSION, "config_version": C.CONFIG_VERSION,
        "hf_repo": C.HF_REPO, "github_repo": C.GITHUB_REPO, "contact_email": contact_email,
    }
    return nums


def render_numbers(nums: dict) -> dict:
    """Numbers + generated blocks ready for vtt.readme.render (tables built from the row lists)."""
    out = dict(nums)
    out["class_table"] = RM.rows_table(nums["class_rows"], [("task", "task"), ("class", "class"), ("windows", "eligible windows"),
                                                            ("hours", "hours"), ("vehicles", "vehicles"), ("platforms", "platforms")])
    out["split_table"] = RM.rows_table(nums["split_rows"], [("task", "task"), ("protocol", "split column"), ("split", "split"),
                                                            ("windows", "eligible windows"), ("hours", "hours"), ("groups", "groups"), ("platforms", "platforms")])
    shard_rows = [{**r, "gb": round(r["bytes"] / 1e9, 2)} for r in nums["shard_rows"]]
    out["shard_table"] = RM.rows_table(shard_rows, [("split", "split"), ("n_files", "files"), ("windows", "windows"), ("gb", "GB")])
    out.update(RM.blocks())
    return out


def write_readme(roots: Roots, nums: dict) -> Path:
    PV.write_json_atomic(nums, roots.release / "index" / "readme_numbers.json")
    text = RM.render(RM.template_text(), render_numbers(nums))
    return PV.write_text_atomic(text, roots.release / "README.md")


def write_license(roots: Roots) -> Path:
    return PV.write_text_atomic(RM.license_text(DATASET_NAME, HF_URL), roots.release / "LICENSE")


# ------------------------------------------------------------------------------------------------ checksums
def write_checksums(roots: Roots, workers: int = 4, use_cache: bool = True) -> dict:
    """CHECKSUMS.sha256 over every release file except .cache/, itself and .gate_ok; (size, mtime) cache under .cache/."""
    rel = roots.release
    files = SC.checksum_members(SC.list_release_files(rel))
    cache_path = rel / CHECKSUM_CACHE
    cache = {}
    if use_cache and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:                             # noqa: BLE001
            cache = {}
    todo, digests = [], {}
    for r in files:
        st = (rel / r).stat()
        c = cache.get(r)
        if c and c[0] == st.st_size and c[1] == st.st_mtime_ns:
            digests[r] = c[2]
        else:
            todo.append(r)
    digests.update(SC.compute_checksums(rel, todo, workers))
    new_cache = {r: [(rel / r).stat().st_size, (rel / r).stat().st_mtime_ns, digests[r]] for r in files}
    if use_cache:
        PV.write_json_atomic(new_cache, cache_path)
    PV.write_text_atomic(SC.format_checksums(digests), rel / "CHECKSUMS.sha256")
    return {"n_files": len(files), "hashed": len(todo), "cached": len(files) - len(todo)}


# ------------------------------------------------------------------------------------------------ driver
def package(roots: Roots, split_col: str = "split_platform", per_shard: int = 5000, workers: int = 4, copy: bool = False,
            force: bool = False, dry_run: bool = False, run_gate: bool = True, allow_no_splits: bool = False,
            contact_email: str | None = None) -> dict:
    """Assemble the release and run the gate; returns a summary dict with `accepted`."""
    drives, windows, splits_json = load_build(roots.build)
    sel = select_shard_windows(windows, split_col)
    plan = plan_shards(sel, per_shard)
    summary = {"n_drives": int(len(drives)), "n_windows": int(len(windows)), "n_shard_windows": int(len(sel)), "n_shards": len(plan),
               "split_col": split_col, "accepted": False}
    log(f"s09: drives={len(drives)} windows={len(windows)} shard_windows={len(sel)} shards={len(plan)} split_col={split_col}")
    if not len(sel) and not allow_no_splits:
        raise AcceptanceError("no eligible window carries a split assignment (run S8 first or pass --allow-no-splits)")
    if dry_run:
        summary["accepted"] = True
        return summary
    roots.release.mkdir(parents=True, exist_ok=True)
    summary["index"] = write_index_tables(roots, drives, windows)
    log(f"s09: index tables written ({summary['index']})")
    summary["splits"] = write_splits(roots)
    summary["schema"] = write_schema(roots)
    summary["data"] = link_data(roots, drives, copy=copy, workers=workers)
    log(f"s09: data {summary['data']}")
    summary["shards"] = write_shards(roots, drives, windows, split_col, per_shard, workers=workers, force=force)
    log(f"s09: shards written={summary['shards']['written']} kept={summary['shards']['kept']} bytes={summary['shards']['bytes']}")
    email = contact_email if contact_email is not None else R.load_allow()["email"]
    nums = compute_readme_numbers(drives, windows, splits_json, summary["shards"], summary["data"]["bytes"], split_col, email)
    write_readme(roots, nums)
    write_license(roots)
    summary["checksums"] = write_checksums(roots, workers=workers)
    log(f"s09: checksums {summary['checksums']}")
    n_data = sum(1 for _ in (roots.release / "data").rglob("*.parquet"))
    if n_data != len(drives):
        raise AcceptanceError(f"release/data holds {n_data} tables for {len(drives)} drives")
    if run_gate:
        from vtt.gate import pre_upload
        rep, report_path = pre_upload.run(roots.release, data_root=roots.data, private_root=roots.private, workers=workers)
        SC.print_summary(rep, report_path)
        summary["gate_findings"] = len(rep.findings)
        summary["gate_report"] = str(report_path)
        summary["accepted"] = rep.ok
    else:
        summary["accepted"] = True
    return summary
