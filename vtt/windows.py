"""S6: the 60-s window index over the public drive tables (plan §3 'window'; docs/PIPELINE.md S6).

Windows are consecutive, non-overlapping blocks of config.WINDOW_SAMPLES rows of a drive table starting at row 0;
the trailing partial block is listed too (ineligible, reason 'short') so every sample belongs to exactly one window.
Eligibility (schema 'eligible'): the full 600 rows, no gap row inside the window (public tables carry nominal time
only, so a gap-free block of 600 consecutive 10-Hz samples is the 60-s span guarantee; span_s is the nominal
(rows - 1) x 0.1 s and config.WINDOW_SPAN_S is not tested here), moving fraction (v >= config.MOVING_V_MPS) >=
config.WINDOW_MIN_MOVING_FRAC. inelig_reason joins the failed tests in the order short+gap+stationary.
Statistics per schema.WINDOWS_INDEX_COLUMNS; episode counts come from vtt.features_core (shared with the
starter kit). human_only = no das_any sample; human_only_strict = human_only and no das_any sample in the
config.HUMAN_GUARD_S seconds of the drive before the window (vacuously true for the first window of a drive).
Only build/ is read; labels are copied from the drives index at assembly time.
Resume cache: work/windows_parts/<vehicle_id>.parquet holds the statistics of one vehicle's drives; a part is reused
only when it is newer than every drive file of that vehicle AND lists every drive the index currently assigns to
the vehicle (a `--limit` smoke part is rebuilt by the next full run). At assembly every part row whose
(drive_id, vehicle_id) pair is not in the drives index is dropped, so drives removed from the index, renamed, or
moved to another vehicle never reach the window tables (acceptance_problems checks this too).
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vtt import config as C
from vtt import features_core as F
from vtt import schema as S

STAT_COLS: tuple[str, ...] = ("gap", "v_mps", "a_mps2", "standstill", "das_enabled", "das_any", "op_long_active",
                              "op_lat_active", "lead_valid", "thw_s", "lead_speed_mps", "lead_a_slope_mps2", "yaw_valid",
                              "yaw_rate_radps", "curv_1pm", "steer_pressed", "gas_pressed", "brake_pressed")
LABEL_COLS: tuple[str, ...] = ("group_id", "size_class", "powertrain", "powertrain_raw", "platform", "brand", "model", "body",
                               "size_code", "footprint_m2", "split_device", "split_platform", "task_size_eligible",
                               "task_bev_ice_eligible", "low_support_cell")
INELIG_REASONS = ("short", "gap", "stationary")
WINDOW_TABLE = "windows"


@dataclass(frozen=True)
class Roots:
    """Build-tree paths under one data root (mirrors vtt.paths.P for a --data-root override)."""
    data: Path

    @property
    def build(self) -> Path: return self.data / "build"
    @property
    def data_dir(self) -> Path: return self.build / "data"
    @property
    def index(self) -> Path: return self.build / "index"
    @property
    def drives(self) -> Path: return self.index / "drives.parquet"
    @property
    def windows(self) -> Path: return self.index / "windows.parquet"
    @property
    def human_only_windows(self) -> Path: return self.index / "human_only_windows.parquet"
    @property
    def drive_names(self) -> Path: return self.build / "drive_names.parquet"
    @property
    def qc(self) -> Path: return self.build / "qc"
    @property
    def review_md(self) -> Path: return self.qc / "REVIEW.md"
    @property
    def drives_qc(self) -> Path: return self.qc / "drives_qc.csv"
    @property
    def label_overrides(self) -> Path: return self.build / "label_overrides.csv"
    @property
    def private(self) -> Path: return self.data / "private"
    @property
    def private_qc(self) -> Path: return self.private / "qc"
    @property
    def route_audit(self) -> Path: return self.private / "route_audit.parquet"
    @property
    def pseudonym_map(self) -> Path: return self.private / "pseudonym_map.parquet"
    @property
    def private_drives_qc(self) -> Path: return self.private_qc / "drives_private_qc.parquet"
    @property
    def label_audit_private(self) -> Path: return self.private_qc / "label_audit_private.csv"
    @property
    def work(self) -> Path: return self.data / "work"
    @property
    def windows_parts(self) -> Path: return self.work / "windows_parts"
    @property
    def qc_parts(self) -> Path: return self.work / "qc_parts"
    @property
    def tmp(self) -> Path: return self.data / "tmp"


# ----------------------------------------------------------------------------- reading
def resolve_drive_path(row, roots: Roots) -> Path:
    """Path of a drive table from its drives-index row: `file` relative to build/ (the release layout), else
    relative to build/data, else data/<vehicle_id>/<drive_id>.parquet."""
    rel = row.get("file") if hasattr(row, "get") else None
    if isinstance(rel, str) and rel:
        for base in (roots.build, roots.data_dir):
            p = base / rel
            if p.exists():
                return p
    return roots.data_dir / str(row["vehicle_id"]) / f"{row['drive_id']}.parquet"


def read_drive(path: Path, columns=STAT_COLS) -> dict[str, np.ndarray]:
    """The columns of a drive table needed for window statistics, as numpy arrays (bool -> bool, float -> float64);
    columns missing from the file (older build) come back as NaN / False."""
    sch = pq.read_schema(path)
    present = [c for c in columns if c in sch.names]
    tab = pq.read_table(path, columns=present)
    n = tab.num_rows
    spec = S.dtypes(S.DRIVE_COLUMNS)
    out: dict[str, np.ndarray] = {}
    for c in columns:
        is_bool = spec.get(c, "float32") == "bool"
        if c not in present:
            out[c] = np.zeros(n, bool) if is_bool else np.full(n, np.nan)
            continue
        col = tab.column(c)
        if is_bool:
            out[c] = col.fill_null(False).to_numpy(zero_copy_only=False).astype(bool)
        else:
            out[c] = col.to_numpy(zero_copy_only=False).astype(float)
    out["__n__"] = np.array([n])
    return out


# ----------------------------------------------------------------------------- per-window statistics
def window_bounds(n_rows: int, size: int = C.WINDOW_SAMPLES) -> list[tuple[int, int]]:
    """[(row0, row1_exclusive), ...]: consecutive blocks of `size` rows from row 0, the trailing partial block last."""
    if n_rows <= 0:
        return []
    starts = range(0, n_rows, size)
    return [(r0, min(r0 + size, n_rows)) for r0 in starts]


def _frac(mask) -> float:
    m = np.asarray(mask, bool)
    return float(m.mean()) if len(m) else float("nan")


def _nanmean(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def _nanstd(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.std()) if len(x) else float("nan")


def _nanmin(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.min()) if len(x) else float("nan")


def _nanmax(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.max()) if len(x) else float("nan")


def _nanmedian(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def window_kind(das_any_frac: float) -> str:
    """human when no das_any sample, das when every sample, else mixed."""
    if das_any_frac <= 0.0:
        return "human"
    if das_any_frac >= 1.0:
        return "das"
    return "mixed"


def window_stats(d: dict[str, np.ndarray], row0: int, row1: int, guard_rows: int) -> dict:
    """Statistics of rows [row0, row1) of a drive (arrays from read_drive). `guard_rows` = HUMAN_GUARD_S / dt."""
    sl = slice(row0, row1)
    n = row1 - row0
    v = d["v_mps"][sl]
    a = d["a_mps2"][sl]
    gap_inside = int(d["gap"][row0 + 1:row1].sum())        # a gap flag on the first row marks missing data before the window
    moving = np.nan_to_num(v, nan=-np.inf) >= C.MOVING_V_MPS
    das_any = d["das_any"][sl]
    lead = d["lead_valid"][sl]
    thw = d["thw_s"][sl]
    yaw_valid = d["yaw_valid"][sl]
    yaw = d["yaw_rate_radps"][sl]
    span_s = (n - 1) * C.NOMINAL_DT_S
    reasons = []
    if n < C.WINDOW_SAMPLES:
        reasons.append("short")
    if gap_inside > 0:
        reasons.append("gap")
    moving_frac = _frac(moving)
    if not moving_frac >= C.WINDOW_MIN_MOVING_FRAC:
        reasons.append("stationary")
    das_any_frac = _frac(das_any)
    human_only = bool(das_any_frac <= 0.0)
    pre = d["das_any"][max(0, row0 - guard_rows):row0]
    human_only_strict = bool(human_only and not pre.any())
    n_lead = int(lead.sum())
    cf = float(np.sum(lead & (np.nan_to_num(thw, nan=np.inf) <= C.THW_CF_S)) / n_lead) if n_lead else float("nan")
    out = dict(
        row0=row0, row1=row1, t0_s=row0 * C.NOMINAL_DT_S, span_s=span_s,
        eligible=not reasons, inelig_reason="+".join(reasons) if reasons else None,
        kind=window_kind(das_any_frac),
        das_frac=_frac(d["das_enabled"][sl]), das_any_frac=das_any_frac,
        op_long_frac=_frac(d["op_long_active"][sl]), op_lat_frac=_frac(d["op_lat_active"][sl]),
        moving_frac=moving_frac, standstill_frac=_frac(d["standstill"][sl]), gap_count=gap_inside,
        v_mean=_nanmean(v), v_max=_nanmax(v), v_std=_nanstd(v),
        a_min=_nanmin(a), a_max=_nanmax(a), a_std=_nanstd(a),
        n_accel_ep=len(F.accel_episodes(v, C.NOMINAL_DT_S, **C.ACCEL_EP)),
        n_decel_ep=len(F.decel_episodes(v, C.NOMINAL_DT_S, **C.ACCEL_EP)),
        lead_frac=_frac(lead), cf_frac=cf,
        thw_med=_nanmedian(thw[lead]) if n_lead else float("nan"),
        lead_speed_mean=_nanmean(d["lead_speed_mps"][sl][lead]) if n_lead else float("nan"),
        n_lead_accel_events=int(len(F.lead_accel_events(d["lead_a_slope_mps2"][sl], lead, C.NOMINAL_DT_S,
                                                        **C.LEAD_ACCEL_EVENT))),
        yaw_valid_frac=_frac(yaw_valid),
        yaw_abs_mean=_nanmean(np.abs(yaw[yaw_valid])) if yaw_valid.any() else float("nan"),
        curv_abs_mean=_nanmean(np.abs(d["curv_1pm"][sl])),
        steer_pressed_frac=_frac(d["steer_pressed"][sl]), gas_pressed_frac=_frac(d["gas_pressed"][sl]),
        brake_pressed_frac=_frac(d["brake_pressed"][sl]),
        human_only=human_only, human_only_strict=human_only_strict,
    )
    return out


def drive_windows(d: dict[str, np.ndarray], drive_id: str, vehicle_id: str) -> pd.DataFrame:
    """All windows of one drive (statistics + ids, no labels)."""
    n = int(d["__n__"][0])
    guard_rows = int(round(C.HUMAN_GUARD_S / C.NOMINAL_DT_S))
    rows = []
    for w_idx, (r0, r1) in enumerate(window_bounds(n)):
        st = window_stats(d, r0, r1, guard_rows)
        st.update(window_id=f"{drive_id}_w{w_idx}", drive_id=drive_id, vehicle_id=vehicle_id, w_idx=w_idx)
        rows.append(st)
    cols = [c for c in S.names(S.WINDOWS_INDEX_COLUMNS) if c not in LABEL_COLS]
    return pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------------------------------- assembly
def attach_labels(parts: pd.DataFrame, drives: pd.DataFrame) -> pd.DataFrame:
    """Join group_id and the label / split columns of the drives index onto the window statistics."""
    lab = [c for c in LABEL_COLS if c in drives.columns]
    merged = parts.merge(drives[["drive_id", *lab]], on="drive_id", how="left", validate="many_to_one")
    for c in LABEL_COLS:
        if c not in merged.columns:
            merged[c] = None
    return merged[S.names(S.WINDOWS_INDEX_COLUMNS)]


def human_only_subset(windows: pd.DataFrame) -> pd.DataFrame:
    """Eligible windows with human_only_strict (the human_only_windows table)."""
    m = windows["eligible"].astype(bool) & windows["human_only_strict"].astype(bool)
    return windows.loc[m].reset_index(drop=True)


def write_windows(windows: pd.DataFrame, path: Path) -> Path:
    return S.write_public_parquet(windows, path, S.WINDOWS_INDEX_COLUMNS, WINDOW_TABLE)


def _part_path(parts_dir: Path, vehicle_id: str) -> Path:
    return parts_dir / f"{vehicle_id}.parquet"


def part_drive_ids(part: Path) -> set[str] | None:
    """The drive_id set stored in a cached part (None when it cannot be read)."""
    try:
        return set(pq.read_table(part, columns=["drive_id"]).column("drive_id").to_pylist())
    except Exception:                           # noqa: BLE001 - an unreadable part is simply stale
        return None


def part_is_fresh(part: Path, drive_paths: list[Path], drive_ids=None) -> bool:
    """A cached per-vehicle part is reused when it is newer than every drive file it summarises and, when
    `drive_ids` is given, it holds rows for every one of those drives (a superset is fine: rows of drives that left
    the index are dropped at assembly by index_rows)."""
    if not part.exists():
        return False
    try:
        pm = part.stat().st_mtime
        if not all(p.exists() and p.stat().st_mtime <= pm for p in drive_paths):
            return False
    except OSError:
        return False
    if drive_ids is not None:
        have = part_drive_ids(part)
        if have is None or not set(drive_ids) <= have:
            return False
    return True


def index_rows(parts: pd.DataFrame, drives: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep only part rows whose (drive_id, vehicle_id) pair is in the drives index (ghost rows of drives removed,
    renamed or moved to another vehicle are dropped). Returns (rows, n_dropped)."""
    if parts.empty:
        return parts, 0
    veh = drives.drop_duplicates("drive_id").set_index("drive_id")["vehicle_id"]
    if "vehicle_id" in parts.columns:
        want = veh.reindex(parts["drive_id"]).to_numpy(object)
        keep = np.array([isinstance(w, str) and w == v for w, v in zip(want, parts["vehicle_id"].to_numpy(object))], bool)
    else:
        keep = parts["drive_id"].isin(set(veh.index)).to_numpy()
    out = parts.loc[keep].drop_duplicates(["drive_id", "w_idx"] if "w_idx" in parts.columns else ["drive_id"])
    return out.reset_index(drop=True), int(len(parts) - len(out))


def build_vehicle_part(task: tuple) -> tuple[str, int, int, str]:
    """Worker: windows of every drive of one vehicle -> parts_dir/<vehicle_id>.parquet.
    task = (vehicle_id, [(drive_id, path_str), ...], parts_dir_str, dry_run). Returns (vehicle_id, n_drives, n_windows, err)."""
    vehicle_id, drives, parts_dir, dry_run = task
    frames = []
    try:
        for drive_id, path in drives:
            d = read_drive(Path(path))
            frames.append(drive_windows(d, drive_id, vehicle_id))
        cols = [c for c in S.names(S.WINDOWS_INDEX_COLUMNS) if c not in LABEL_COLS]
        part = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
        if not dry_run:
            out = _part_path(Path(parts_dir), vehicle_id)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(f".{out.name}.tmp")
            part.to_parquet(tmp, index=False)
            os.replace(tmp, out)
        return vehicle_id, len(drives), len(part), ""
    except Exception as e:                      # noqa: BLE001 - reported as a count, never an identifier
        return vehicle_id, len(drives), 0, type(e).__name__


def build_windows(drives: pd.DataFrame, roots: Roots, workers: int = 1, force: bool = False,
                  dry_run: bool = False, log=print) -> tuple[pd.DataFrame, dict]:
    """Window index for every drive of the drives index. Per-vehicle parts are cached under work/windows_parts
    (reused when newer than the drive files and covering the vehicle's current drives, unless force); rows of drives
    that are no longer in the index are dropped at assembly. Returns (windows, counts)."""
    parts_dir = roots.windows_parts
    tasks, reused, missing = [], [], 0
    for vehicle_id, grp in drives.groupby("vehicle_id", sort=True):
        items = []
        for _, row in grp.iterrows():
            p = resolve_drive_path(row, roots)
            if not p.exists():
                missing += 1
                continue
            items.append((str(row["drive_id"]), str(p)))
        if not items:
            continue
        part = _part_path(parts_dir, str(vehicle_id))
        if not force and part_is_fresh(part, [Path(p) for _, p in items], [d for d, _ in items]):
            reused.append(part)
        else:
            tasks.append((str(vehicle_id), items, str(parts_dir), dry_run))
    log(f"vehicles={drives['vehicle_id'].nunique()} drives={len(drives)} missing_files={missing} "
        f"parts_reused={len(reused)} parts_to_build={len(tasks)}")
    built, errors, built_frames = 0, 0, []
    if tasks and dry_run:                        # dry run: keep the frames in memory instead of writing parts
        for vehicle_id, items, _, _ in tasks:
            try:
                built_frames += [drive_windows(read_drive(Path(p)), did, vehicle_id) for did, p in items]
            except Exception:                    # noqa: BLE001 - reported as a count, never an identifier
                errors += 1
    elif tasks:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for vehicle_id, n_d, n_w, err in ex.map(build_vehicle_part, tasks, chunksize=4):
                    built += n_w
                    errors += bool(err)
        else:
            for task in tasks:
                vehicle_id, n_d, n_w, err = build_vehicle_part(task)
                built += n_w
                errors += bool(err)
    frames = [pd.read_parquet(p) for p in reused]
    if not dry_run:
        frames += [pd.read_parquet(_part_path(parts_dir, t[0])) for t in tasks if _part_path(parts_dir, t[0]).exists()]
    else:
        frames += built_frames
    cols = [c for c in S.names(S.WINDOWS_INDEX_COLUMNS) if c not in LABEL_COLS]
    parts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    parts, dropped = index_rows(parts, drives)
    windows = attach_labels(parts, drives)
    windows = windows.sort_values(["drive_id", "w_idx"], kind="stable").reset_index(drop=True)
    counts = dict(drives=int(len(drives)), drives_with_windows=int(windows["drive_id"].nunique()), windows=int(len(windows)),
                  eligible=int(windows["eligible"].astype(bool).sum()), part_errors=int(errors), missing_files=int(missing),
                  stale_rows_dropped=int(dropped), human_only_strict_eligible=int(len(human_only_subset(windows))))
    return windows, counts


def acceptance_problems(windows: pd.DataFrame, drives: pd.DataFrame) -> list[str]:
    """Contract checks on the assembled window index (empty list = OK)."""
    problems = []
    if len(drives) and windows.empty:
        problems.append("no windows built")
    if windows["window_id"].duplicated().any():
        problems.append("duplicate window_id")
    el = windows["eligible"].astype(bool)
    full = (windows["row1"] - windows["row0"]) == C.WINDOW_SAMPLES
    if (el & ~full).any():
        problems.append("eligible window without 600 rows")
    if (el & (windows["gap_count"] > 0)).any():
        problems.append("eligible window with a gap")
    if (el & ~(windows["moving_frac"] >= C.WINDOW_MIN_MOVING_FRAC)).any():
        problems.append("eligible window below the moving fraction")
    if (windows["human_only_strict"].astype(bool) & ~windows["human_only"].astype(bool)).any():
        problems.append("human_only_strict outside human_only")
    idx = drives.drop_duplicates("drive_id").set_index("drive_id")
    in_index = windows["drive_id"].isin(set(idx.index)).to_numpy()
    if (~in_index).any():
        problems.append(f"{int((~in_index).sum())} windows of drives that are not in the index (stale work/windows_parts; rerun with --force)")
    if in_index.any():
        w = windows.loc[in_index]
        veh = idx["vehicle_id"].reindex(w["drive_id"]).to_numpy(object)
        if (w["vehicle_id"].to_numpy(object) != veh).any():
            problems.append("windows whose vehicle_id differs from the index")
        if "group_id" in idx.columns:
            gid = idx["group_id"].reindex(w["drive_id"])
            if (w["group_id"].isna().to_numpy() & gid.notna().to_numpy()).any():
                problems.append("windows without group_id although the index has one")
    have_rows = set(drives.loc[drives["n_rows"].fillna(0).astype(int) > 0, "drive_id"])
    got = set(windows["drive_id"])
    lost = have_rows - got
    if lost:
        problems.append(f"{len(lost)} drives with rows but no windows (missing drive files or stale work/windows_parts; rerun with --force)")
    return problems
