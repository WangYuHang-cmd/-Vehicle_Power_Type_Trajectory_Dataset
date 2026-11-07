"""S7: quality control and label audit (docs/PIPELINE.md S7; plan §4 S7).

Per-drive checks over the public drive tables and the drives index: carState rate (from the private per-drive QC
frame when present, else the rows-per-segment consistency of the public index), radar / pose availability,
dead pedal / torque channels (overall and per platform; the build is expected to show gas dead ~68 % and brake
dead ~41 % of drives), lead_source distribution by radar_unavailable (S5 derives lead_source with the route's
radarUnavailable flag, so a per-drive mismatch cannot occur by construction; the distribution is reported instead),
all-zero yaw, duplicate drives (same n_rows and sha1 of the v_mps bytes), devices logging several platforms.
Label audit against the private probe private/route_audit.parquet (joined through build/drive_names.parquet):
  BEV  with engine_rpm_pos_frac > 0.05            -> conflict_bev_engine_rpm   (rpm == 0 is consistent -> ok)
  ICE  with regen_any or charging_any             -> conflict_ice_electric     (independent of the rpm channel)
  ICE / HEV / PHEV with engine_rpm_pos_frac == 0  -> no_rpm_signal (many car ports never fill carState.engineRpm, so
                                                     rpm == 0 while moving is an unpopulated channel, not an engine-off fact)
  ICE  with engine_off_moving_frac > 0.3          -> suspect_mild_hybrid
  HEV / PHEV with the engine never off while moving (engine_off_moving_frac <= 0.001) -> suspect_hybrid_engine_always_on
  no probe row -> no_audit; probe error / NaN numbers -> no_audit_signal; null powertrain -> na.
Verdicts are appended to the public label_note as 'label_audit=<verdict>' (idempotent; '/resolved' when an override
row with resolve=1 covers the drive); the private CSV keeps the numbers. build/label_overrides.csv (optional,
columns scope,key,powertrain,label_note,resolve) re-labels drives / vehicles / platforms and resolves conflicts.
Outputs: build/qc/drives_qc.csv (pseudonyms only), build/qc/REVIEW.md, private/qc/label_audit_private.csv,
label_note (and overridden powertrain / task_bev_ice_eligible) written back into build/index/drives.parquet and,
for changed powertrains, into the window tables. Nothing here prints an identifier.
"""
from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vtt import config as C
from vtt import schema as S
from vtt.private import HEX12, HEX16, mask
from vtt.windows import Roots, index_rows, part_is_fresh, resolve_drive_path

SCAN_COLS = ("v_mps", "yaw_rate_radps", "lead_valid", "lead_source")
EXPECTED_DEAD_SHARE = {"gas": 0.68, "brake": 0.41}
RATE_LOW_HZ = C.NOMINAL_HZ * 0.8
RATE_HIGH_HZ = C.NOMINAL_HZ * 1.2
ROWS_PER_SEG_MIN = 0.8 * C.WINDOW_SAMPLES       # a 60-s segment holds 600 rows at 10 Hz
MIN_SEGMENTS_FOR_ROW_CHECK = 3
LEAD_SUSPECT_MIN_VALID = 1000                   # valid-lead samples needed before 'radar never used' is flagged
AUDIT_TAG = "label_audit="
OVERRIDE_TAG = "override:"
CONFLICT_PREFIX = "conflict_"
SUSPECT_PREFIX = "suspect_"
VERDICTS = ("ok", "conflict_bev_engine_rpm", "conflict_ice_electric", "suspect_mild_hybrid",
            "suspect_hybrid_engine_always_on", "no_audit", "no_audit_signal", "no_rpm_signal", "na")
SCAN_COLUMNS = ("drive_id", "vehicle_id", "n_rows_file", "v_sha1", "yaw_n_finite", "yaw_all_zero", "lead_n_valid",
                "lead_radar_share", "lead_vision_share")
OVERRIDE_COLS = ("scope", "key", "powertrain", "label_note", "resolve")
OVERRIDE_SCOPES = {"drive": "drive_id", "vehicle": "vehicle_id", "platform": "platform"}
AUDIT_NUMBERS = ("engine_rpm_pos_frac", "engine_off_moving_frac", "regen_any", "charging_any", "has_hybrid_ecu",
                 "transmission", "n_samples", "n_segments_read", "err")
PRIVATE_QC_RATE_COLS = ("cs_rate_hz", "rate_carState", "carstate_hz", "cs_hz", "rate_hz")
PRIVATE_QC_DT_COLS = ("cs_dt_med_s", "dt_med_s", "dt_median_s", "cs_dt_median_s", "dt_med", "cs_dt_median")
PRIVATE_QC_SPAN_COLS = ("real_span_s", "span_real_s", "duration_real_s", "real_duration_s", "span_s")


# ----------------------------------------------------------------------------- per-drive scan (cached per vehicle)
def scan_drive(path: Path) -> dict:
    """Cheap per-drive facts from four columns: sha1 of the v_mps float32 bytes (duplicate detection), yaw all-zero,
    lead-source shares among valid-lead samples."""
    sch = pq.read_schema(path)
    cols = [c for c in SCAN_COLS if c in sch.names]
    tab = pq.read_table(path, columns=cols)
    n = tab.num_rows
    v = tab.column("v_mps").to_numpy(zero_copy_only=False).astype(np.float32) if "v_mps" in cols else np.zeros(0, np.float32)
    yaw = tab.column("yaw_rate_radps").to_numpy(zero_copy_only=False).astype(float) if "yaw_rate_radps" in cols else np.full(n, np.nan)
    yf = yaw[np.isfinite(yaw)]
    lead = tab.column("lead_valid").fill_null(False).to_numpy(zero_copy_only=False).astype(bool) if "lead_valid" in cols else np.zeros(n, bool)
    src = tab.column("lead_source").fill_null(0).to_numpy(zero_copy_only=False).astype(int) if "lead_source" in cols else np.zeros(n, int)
    n_lead = int(lead.sum())
    return dict(n_rows_file=n, v_sha1=hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest(),
                yaw_n_finite=int(len(yf)), yaw_all_zero=bool(len(yf) > 0 and np.all(yf == 0.0)),
                lead_n_valid=n_lead,
                lead_radar_share=float(np.mean(src[lead] == 1)) if n_lead else float("nan"),
                lead_vision_share=float(np.mean(src[lead] == 2)) if n_lead else float("nan"))


def scan_vehicle_frame(vehicle_id: str, drives: list[tuple[str, str]]) -> pd.DataFrame:
    """SCAN_COLUMNS rows of every drive of one vehicle (in memory)."""
    rows = []
    for drive_id, path in drives:
        r = scan_drive(Path(path))
        r["drive_id"] = drive_id
        r["vehicle_id"] = vehicle_id
        rows.append(r)
    return pd.DataFrame(rows, columns=list(SCAN_COLUMNS))


def scan_vehicle(task: tuple) -> tuple[str, int, str]:
    """Worker: scan every drive of one vehicle -> parts_dir/<vehicle_id>.parquet. Returns (vehicle_id, n_drives, err)."""
    vehicle_id, drives, parts_dir = task
    try:
        frame = scan_vehicle_frame(vehicle_id, drives)
        out = Path(parts_dir) / f"{vehicle_id}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.tmp")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        return vehicle_id, len(drives), ""
    except Exception as e:                      # noqa: BLE001
        return vehicle_id, len(drives), type(e).__name__


def scan_all(drives: pd.DataFrame, roots: Roots, workers: int = 1, force: bool = False, dry_run: bool = False,
             log=print) -> pd.DataFrame:
    """Per-drive scan of every drive table; per-vehicle parts cached under work/qc_parts (reused when newer than the
    drive files and covering the vehicle's current drives, unless force; same rule as S6). Rows of drives no longer
    in the index are dropped at assembly. With dry_run nothing is written (stale parts are scanned in memory)."""
    parts_dir = roots.qc_parts
    tasks, reused, missing = [], [], 0
    for vehicle_id, grp in drives.groupby("vehicle_id", sort=True):
        items = []
        for _, row in grp.iterrows():
            p = resolve_drive_path(row, roots)
            if p.exists():
                items.append((str(row["drive_id"]), str(p)))
            else:
                missing += 1
        if not items:
            continue
        part = parts_dir / f"{vehicle_id}.parquet"
        if not force and part_is_fresh(part, [Path(p) for _, p in items], [d for d, _ in items]):
            reused.append(part)
        else:
            tasks.append((str(vehicle_id), items, str(parts_dir)))
    log(f"scan: vehicles={drives['vehicle_id'].nunique()} missing_files={missing} parts_reused={len(reused)} parts_to_build={len(tasks)}")
    errors = 0
    frames = [pd.read_parquet(p) for p in reused]
    if tasks and dry_run:
        for vehicle_id, items, _ in tasks:
            try:
                frames.append(scan_vehicle_frame(vehicle_id, items))
            except Exception:                   # noqa: BLE001 - reported as a count, never an identifier
                errors += 1
    elif tasks:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for _, _, err in ex.map(scan_vehicle, tasks, chunksize=4):
                    errors += bool(err)
        else:
            for t in tasks:
                errors += bool(scan_vehicle(t)[2])
        frames += [pd.read_parquet(parts_dir / f"{t[0]}.parquet") for t in tasks if (parts_dir / f"{t[0]}.parquet").exists()]
    if errors:
        log(f"scan: part_errors={errors}")
    cols = list(SCAN_COLUMNS)
    if not frames:
        return pd.DataFrame(columns=cols)
    parts = pd.concat(frames, ignore_index=True)
    if "vehicle_id" not in parts.columns:       # parts written before vehicle_id was stored
        parts["vehicle_id"] = None
    parts, dropped = index_rows(parts, drives)
    if dropped:
        log(f"scan: stale_rows_dropped={dropped}")
    return parts.reindex(columns=cols)


# ----------------------------------------------------------------------------- per-drive checks
def _first_col(df: pd.DataFrame | None, candidates) -> str | None:
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def rate_table(drives: pd.DataFrame, private_qc: pd.DataFrame | None = None) -> pd.DataFrame:
    """drive_id, cs_rate_hz (private real-time statistics when available), rows_per_seg, rate_flag.
    Without the private frame the check is the public consistency n_rows / (n_segments - 1) >= 480 for drives with
    >= 3 segments (the last segment of a route is usually short, so it is left out of the denominator)."""
    out = drives[["drive_id", "n_rows", "n_segments"]].copy()
    out["cs_rate_hz"] = np.nan
    if private_qc is not None and "drive_id" in private_qc.columns:
        pq_ = private_qc.drop_duplicates("drive_id").set_index("drive_id")
        rate_c, dt_c, span_c = (_first_col(pq_, PRIVATE_QC_RATE_COLS), _first_col(pq_, PRIVATE_QC_DT_COLS),
                                _first_col(pq_, PRIVATE_QC_SPAN_COLS))
        ids = out["drive_id"]
        if rate_c:
            out["cs_rate_hz"] = pd.to_numeric(pq_[rate_c].reindex(ids), errors="coerce").to_numpy()
        elif dt_c:
            dt = pd.to_numeric(pq_[dt_c].reindex(ids), errors="coerce").to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out["cs_rate_hz"] = np.where(dt > 0, 1.0 / dt, np.nan)
        elif span_c:
            span = pd.to_numeric(pq_[span_c].reindex(ids), errors="coerce").to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out["cs_rate_hz"] = np.where(span > 0, out["n_rows"].to_numpy(float) / span, np.nan)
    nseg = out["n_segments"].fillna(0).astype(int).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rows_per_seg"] = np.where(nseg >= MIN_SEGMENTS_FOR_ROW_CHECK, out["n_rows"].to_numpy(float) / np.maximum(nseg - 1, 1), np.nan)
    rate = out["cs_rate_hz"].to_numpy(float)
    flag = np.where(np.isfinite(rate) & (rate < RATE_LOW_HZ), "rate_low",
                    np.where(np.isfinite(rate) & (rate > RATE_HIGH_HZ), "rate_high",
                             np.where(~np.isfinite(rate) & (out["rows_per_seg"].to_numpy(float) < ROWS_PER_SEG_MIN), "rows_per_seg_low", "")))
    out["rate_flag"] = flag
    return out[["drive_id", "cs_rate_hz", "rows_per_seg", "rate_flag"]]


def dead_channel_summary(drives: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Overall dead shares (gas / brake / torque, fraction of drives) and a per-platform table sorted by drives."""
    overall = {}
    for ch in ("gas", "brake", "torque"):
        col = f"{ch}_alive"
        alive = drives[col].astype(bool) if col in drives.columns else pd.Series(True, index=drives.index)
        overall[ch] = float(1.0 - alive.mean()) if len(drives) else float("nan")
    g = drives.groupby("platform", dropna=False)
    per = pd.DataFrame({"drives": g.size(),
                        "gas_dead": 1.0 - g["gas_alive"].mean() if "gas_alive" in drives.columns else np.nan,
                        "brake_dead": 1.0 - g["brake_alive"].mean() if "brake_alive" in drives.columns else np.nan,
                        "torque_dead": 1.0 - g["torque_alive"].mean() if "torque_alive" in drives.columns else np.nan})
    per = per.sort_values("drives", ascending=False).reset_index()
    return overall, per


def _as_bool(series: pd.Series) -> np.ndarray:
    """True only where the value is exactly True (None / NaN / object -> False); no pandas downcasting involved."""
    return series.eq(True).to_numpy(bool)


def lead_source_flags(drives: pd.DataFrame, scan: pd.DataFrame) -> pd.Series:
    """Per drive_id: 'lead_radar_never' (radar available but never the source over >= 1000 valid-lead samples),
    'no_lead' (no valid lead at all while moving more than 10 min), else ''. A radar-vs-camera source mismatch
    against radar_unavailable cannot occur (S5 derives lead_source with that flag), so it is not a flag; the
    distribution is tabulated by lead_source_table instead."""
    m = drives[["drive_id", "radar_unavailable", "moving_s"]].merge(scan[["drive_id", "lead_n_valid", "lead_radar_share"]],
                                                                  on="drive_id", how="left")
    radar_unavail = _as_bool(m["radar_unavailable"])
    rs = m["lead_radar_share"].to_numpy(float)
    nv = np.nan_to_num(m["lead_n_valid"].to_numpy(float), nan=0.0)
    moving = np.nan_to_num(m["moving_s"].to_numpy(float), nan=0.0)
    flag = np.full(len(m), "", dtype=object)
    flag[~radar_unavail & (nv >= LEAD_SUSPECT_MIN_VALID) & (np.nan_to_num(rs, nan=1.0) == 0)] = "lead_radar_never"
    flag[(nv == 0) & (moving > 600)] = "no_lead"
    return pd.Series(flag, index=m["drive_id"].to_numpy(), name="lead_flag")


def lead_source_table(drives: pd.DataFrame, scan: pd.DataFrame) -> pd.DataFrame:
    """Lead-source distribution by radar availability: per radar_unavailable value the drives, the drives with a
    valid lead, the valid-lead samples and the sample-weighted radar / camera source shares."""
    m = drives[["drive_id", "radar_unavailable"]].merge(scan[["drive_id", "lead_n_valid", "lead_radar_share", "lead_vision_share"]],
                                                        on="drive_id", how="left")
    m["radar_unavailable"] = _as_bool(m["radar_unavailable"])
    nv = np.nan_to_num(m["lead_n_valid"].to_numpy(float), nan=0.0)
    m["n_valid"] = nv
    m["n_radar"] = nv * np.nan_to_num(m["lead_radar_share"].to_numpy(float), nan=0.0)
    m["n_vision"] = nv * np.nan_to_num(m["lead_vision_share"].to_numpy(float), nan=0.0)
    rows = []
    for ru, g in m.groupby("radar_unavailable", sort=True):
        tot = float(g["n_valid"].sum())
        rows.append(dict(radar_unavailable=bool(ru), drives=int(len(g)), drives_with_lead=int((g["n_valid"] > 0).sum()),
                         valid_lead_samples=int(tot), radar_share=float(g["n_radar"].sum() / tot) if tot else float("nan"),
                         camera_share=float(g["n_vision"].sum() / tot) if tot else float("nan")))
    return pd.DataFrame(rows, columns=["radar_unavailable", "drives", "drives_with_lead", "valid_lead_samples", "radar_share", "camera_share"])


def duplicate_of(scan: pd.DataFrame) -> pd.Series:
    """drive_id -> the first drive_id (sorted) of its duplicate group (same n_rows and v_mps sha1); '' when unique."""
    s = scan.sort_values("drive_id")
    key = s["n_rows_file"].astype(str) + ":" + s["v_sha1"].astype(str)
    first = s.groupby(key)["drive_id"].transform("first")
    size = s.groupby(key)["drive_id"].transform("size")
    out = np.where((size > 1) & (first != s["drive_id"]), first, "")
    return pd.Series(out, index=s["drive_id"].to_numpy(), name="duplicate_of")


def multi_platform(drives: pd.DataFrame, pseudonym_map: pd.DataFrame | None = None) -> tuple[int, pd.Series, str]:
    """(number of devices with several platforms, per-drive flag, basis). Device = dongle_log of the private
    pseudonym map when available (never printed), else the public group_id (identity cluster)."""
    if pseudonym_map is not None and {"dongle_log", "platform", "drive_id"} <= set(pseudonym_map.columns):
        pm = pseudonym_map[["dongle_log", "platform", "drive_id"]].dropna(subset=["dongle_log"])
        n_plat = pm.groupby("dongle_log")["platform"].nunique()
        multi_dev = set(n_plat.index[n_plat > 1])
        flagged = set(pm.loc[pm["dongle_log"].isin(multi_dev), "drive_id"])
        basis = "device"
        n_multi = len(multi_dev)
    else:
        n_plat = drives.groupby("group_id")["platform"].nunique()
        multi = set(n_plat.index[n_plat > 1])
        flagged = set(drives.loc[drives["group_id"].isin(multi), "drive_id"])
        basis = "group"
        n_multi = len(multi)
    flag = pd.Series(np.where(drives["drive_id"].isin(flagged), f"multi_platform_{basis}", ""),
                     index=drives["drive_id"].to_numpy(), name="multi_platform")
    return n_multi, flag, basis


# ----------------------------------------------------------------------------- label audit
def audit_verdict(powertrain, row: pd.Series | None) -> str:
    """The audit rule for one drive (see module docstring). `row` is its private/route_audit.parquet row or None."""
    if row is None:
        return "no_audit"
    if powertrain is None or (isinstance(powertrain, float) and np.isnan(powertrain)) or powertrain not in C.POWERTRAIN4:
        return "na"
    err = row.get("err")
    rpm = pd.to_numeric(pd.Series([row.get("engine_rpm_pos_frac")]), errors="coerce").iloc[0]
    off = pd.to_numeric(pd.Series([row.get("engine_off_moving_frac")]), errors="coerce").iloc[0]
    regen = bool(row.get("regen_any")) if row.get("regen_any") is not None and row.get("regen_any") == row.get("regen_any") else False
    charging = bool(row.get("charging_any")) if row.get("charging_any") is not None and row.get("charging_any") == row.get("charging_any") else False
    if isinstance(err, str) and err.strip():
        return "no_audit_signal"
    if powertrain == "BEV":
        if np.isfinite(rpm) and rpm > 0.05:
            return "conflict_bev_engine_rpm"
        return "ok" if np.isfinite(rpm) else "no_audit_signal"      # rpm == 0 is what a BEV should report
    if powertrain == "ICE" and (regen or charging):
        return "conflict_ice_electric"                             # electric-drive flags are evidence on their own
    if not np.isfinite(rpm):
        return "no_audit_signal"
    if rpm <= 0.0:
        return "no_rpm_signal"                                     # channel never positive: this port does not fill engineRpm
    if powertrain == "ICE":
        if np.isfinite(off) and off > 0.3:
            return "suspect_mild_hybrid"
        return "ok"
    # HEV / PHEV
    if np.isfinite(off) and off <= 0.001:
        return "suspect_hybrid_engine_always_on"
    return "ok"


def label_audit(drives: pd.DataFrame, route_audit: pd.DataFrame | None, names: pd.DataFrame | None) -> pd.DataFrame:
    """Private per-drive audit table: drive_id, vehicle_id, route_uid, platform, powertrain, powertrain_raw, the probe
    numbers and the verdict. Without the probe or the route_uid -> drive_id map every verdict is no_audit."""
    base = drives[["drive_id", "vehicle_id", "platform", "powertrain", "powertrain_raw"]].copy()
    base["route_uid"] = None
    for c in AUDIT_NUMBERS:
        base[c] = None
    if route_audit is None or names is None or not {"route_uid", "drive_id"} <= set(names.columns):
        base["verdict"] = "no_audit"
        return base
    nm = names[["route_uid", "drive_id"]].drop_duplicates("drive_id")
    ra = route_audit.drop_duplicates("route_uid").set_index("route_uid")
    base = base.drop(columns=["route_uid"]).merge(nm, on="drive_id", how="left")
    verdicts = []
    for i, r in base.iterrows():
        uid = r["route_uid"]
        row = ra.loc[uid] if isinstance(uid, str) and uid in ra.index else None
        if row is not None:
            for c in AUDIT_NUMBERS:
                if c in ra.columns:
                    base.at[i, c] = row[c]
        verdicts.append(audit_verdict(r["powertrain"], row))
    base["verdict"] = verdicts
    cols = ["drive_id", "vehicle_id", "route_uid", "platform", "powertrain", "powertrain_raw", *AUDIT_NUMBERS, "verdict"]
    return base[cols]


# ----------------------------------------------------------------------------- overrides and label_note write-back
def load_overrides(path: Path) -> pd.DataFrame:
    """build/label_overrides.csv (scope,key,powertrain,label_note,resolve); missing file -> empty frame.
    label_note text is masked and may not carry identifier-like tokens (they would reach a public table)."""
    if not path.exists():
        return pd.DataFrame(columns=OVERRIDE_COLS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in OVERRIDE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"label_overrides.csv lacks columns {missing}")
    df = df[list(OVERRIDE_COLS)].copy()
    df["scope"] = df["scope"].str.strip().str.lower()
    bad = df.loc[~df["scope"].isin(OVERRIDE_SCOPES), "scope"]
    if len(bad):
        raise ValueError(f"label_overrides.csv: unknown scope values ({len(bad)} rows)")
    pt = df["powertrain"].str.strip()
    ok_pt = pt.isin(["", "null", "none", "NULL", "None", *C.POWERTRAIN4])
    if (~ok_pt).any():
        raise ValueError(f"label_overrides.csv: invalid powertrain values ({int((~ok_pt).sum())} rows)")
    df["powertrain"] = pt
    notes = df["label_note"].fillna("").map(lambda s: mask(str(s)).strip())
    if notes.map(lambda s: bool(HEX16.search(s) or HEX12.search(s) or "<" in s)).any():
        raise ValueError("label_overrides.csv: label_note carries identifier-like tokens")
    df["label_note"] = notes
    df["resolve"] = df["resolve"].fillna("").str.strip().str.lower().isin(["1", "true", "yes", "y"])
    return df


def _split_note(note) -> list[str]:
    if note is None or (isinstance(note, float) and np.isnan(note)):
        return []
    return [t.strip() for t in str(note).split(";") if t.strip()]


def strip_generated(note) -> list[str]:
    """Tokens of a label_note without the ones S7 generates (label_audit=..., override:...)."""
    return [t for t in _split_note(note) if not (t.startswith(AUDIT_TAG) or t.startswith(OVERRIDE_TAG))]


def compose_note(base_tokens: list[str], override_notes: list[str], verdict: str | None, resolved: bool) -> str | None:
    toks = list(base_tokens) + [f"{OVERRIDE_TAG}{n}" for n in override_notes if n]
    if verdict:
        toks.append(f"{AUDIT_TAG}{verdict}" + ("/resolved" if resolved and verdict.startswith(CONFLICT_PREFIX) else ""))
    return "; ".join(toks) if toks else None


def apply_overrides(drives: pd.DataFrame, overrides: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]], set[str], set[str]]:
    """Apply override rows in file order. Returns (drives, override notes per drive_id, drives with a changed
    powertrain, drives whose conflicts are resolved)."""
    out = drives.copy()
    notes: dict[str, list[str]] = {}
    changed: set[str] = set()
    resolved: set[str] = set()
    for _, o in overrides.iterrows():
        col = OVERRIDE_SCOPES[o["scope"]]
        m = out[col].astype(str) == str(o["key"]).strip()
        if not m.any():
            continue
        ids = out.loc[m, "drive_id"].tolist()
        pt = o["powertrain"]
        if pt:
            new = None if pt.lower() in ("null", "none") else pt
            before = out.loc[m, "powertrain"].tolist()
            out.loc[m, "powertrain"] = new
            out.loc[m, "task_bev_ice_eligible"] = new in C.MAIN_POWERTRAIN_TASK
            changed.update(d for d, b in zip(ids, before) if (b if isinstance(b, str) else None) != new)
        if o["label_note"]:
            for d in ids:
                notes.setdefault(d, []).append(o["label_note"])
        if bool(o["resolve"]):
            resolved.update(ids)
    return out, notes, changed, resolved


def write_back_labels(drives: pd.DataFrame, verdicts: pd.Series, notes: dict[str, list[str]], resolved: set[str]) -> pd.DataFrame:
    """label_note := original tokens + override notes + label_audit=<verdict>[/resolved] (idempotent)."""
    out = drives.copy()
    v = verdicts.reindex(out["drive_id"]).to_numpy(object)
    new_notes = []
    for d, note, verdict in zip(out["drive_id"], out["label_note"], v):
        verdict = verdict if isinstance(verdict, str) else "no_audit"
        new_notes.append(compose_note(strip_generated(note), notes.get(d, []), verdict, d in resolved))
    out["label_note"] = new_notes
    return out


def unresolved_conflicts(verdicts: pd.Series, resolved: set[str]) -> list[str]:
    """drive_ids whose verdict is a conflict and that no override resolves."""
    return [d for d, v in verdicts.items() if isinstance(v, str) and v.startswith(CONFLICT_PREFIX) and d not in resolved]


def table_name(path: Path, default: str) -> str:
    try:
        return (pq.read_schema(path).metadata or {}).get(b"vtt_table", default.encode()).decode()
    except Exception:              # noqa: BLE001
        return default


def propagate_to_windows(roots: Roots, drives: pd.DataFrame, changed: set[str]) -> int:
    """Copy powertrain / task_bev_ice_eligible of changed drives into windows.parquet and human_only_windows.parquet."""
    if not changed:
        return 0
    n = 0
    lab = drives.set_index("drive_id")[["powertrain", "task_bev_ice_eligible"]]
    for path in (roots.windows, roots.human_only_windows):
        if not path.exists():
            continue
        w = pd.read_parquet(path)
        m = w["drive_id"].isin(changed)
        if m.any():
            w.loc[m, "powertrain"] = lab["powertrain"].reindex(w.loc[m, "drive_id"]).to_numpy(object)
            w.loc[m, "task_bev_ice_eligible"] = lab["task_bev_ice_eligible"].reindex(w.loc[m, "drive_id"]).astype(bool).to_numpy()
            S.write_public_parquet(w, path, S.WINDOWS_INDEX_COLUMNS, table_name(path, "windows"))
            n += int(m.sum())
    return n


# ----------------------------------------------------------------------------- summaries and REVIEW.md
def _cell(df: pd.DataFrame) -> pd.Series:
    return df["size_class"].fillna("null").astype(str) + " x " + df["powertrain"].fillna("null").astype(str)


def eligible_hours_by_drive(windows: pd.DataFrame | None) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(n_windows, n_eligible, n_eligible_human_strict) per drive_id from the window index (empty without it)."""
    if windows is None or windows.empty:
        e = pd.Series(dtype=float)
        return e, e, e
    g = windows.groupby("drive_id")
    el = windows["eligible"].astype(bool)
    hs = el & windows["human_only_strict"].astype(bool)
    return g.size(), el.groupby(windows["drive_id"]).sum(), hs.groupby(windows["drive_id"]).sum()


def cell_table(drives: pd.DataFrame, windows: pd.DataFrame | None) -> pd.DataFrame:
    """cell (size_class x powertrain): drives, vehicles, groups, platforms, hours, top platform and its hour share,
    DAS share (das_any hours / hours), eligible-window hours, human-only-strict window hours."""
    d = drives.copy()
    d["cell"] = _cell(d)
    d["hours"] = d["duration_s"].astype(float) / 3600.0
    d["das_h"] = d["das_any_s"].astype(float) / 3600.0
    _, n_el, n_hs = eligible_hours_by_drive(windows)
    wh = C.WINDOW_SAMPLES * C.NOMINAL_DT_S / 3600.0
    d["elig_h"] = n_el.reindex(d["drive_id"]).fillna(0).to_numpy() * wh
    d["human_strict_h"] = n_hs.reindex(d["drive_id"]).fillna(0).to_numpy() * wh
    rows = []
    for cell, g in d.groupby("cell", sort=True):
        ph = g.groupby("platform")["hours"].sum().sort_values(ascending=False)
        hours = float(g["hours"].sum())
        rows.append(dict(cell=cell, drives=len(g), vehicles=g["vehicle_id"].nunique(), groups=g["group_id"].nunique(),
                         platforms=g["platform"].nunique(), hours=hours,
                         top_platform=str(ph.index[0]) if len(ph) else "", top_platform_share=float(ph.iloc[0] / hours) if (hours > 0 and len(ph)) else float("nan"),
                         das_share=float(g["das_h"].sum() / hours) if hours > 0 else float("nan"),
                         eligible_window_h=float(g["elig_h"].sum()), human_strict_window_h=float(g["human_strict_h"].sum())))
    return pd.DataFrame(rows)


def region_table(drives: pd.DataFrame) -> pd.DataFrame:
    """Hours per cell (rows) x region (columns)."""
    d = drives.copy()
    d["cell"] = _cell(d)
    d["hours"] = d["duration_s"].astype(float) / 3600.0
    pv = d.pivot_table(index="cell", columns=d["region"].fillna(C.REGION_OTHER), values="hours", aggfunc="sum", fill_value=0.0)
    pv["total"] = pv.sum(axis=1)
    return pv.reset_index()


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}", max_rows: int | None = None) -> str:
    """Markdown table without external dependencies (floats formatted, NaN -> '')."""
    if df is None or len(df) == 0:
        return "_(none)_\n"
    d = df if max_rows is None else df.head(max_rows)
    cols = [str(c) for c in d.columns]

    def fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        if isinstance(x, (float, np.floating)):
            return floatfmt.format(float(x))
        if isinstance(x, (bool, np.bool_)):
            return "yes" if x else "no"
        return str(x)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in d.columns) + " |")
    if max_rows is not None and len(df) > max_rows:
        lines.append(f"\n_{len(df) - max_rows} more rows in drives_qc.csv_")
    return "\n".join(lines) + "\n"


def drives_qc_table(drives: pd.DataFrame, scan: pd.DataFrame, rates: pd.DataFrame, lead_flag: pd.Series, dup: pd.Series,
                    multi: pd.Series, audit: pd.DataFrame, resolved: set[str], windows: pd.DataFrame | None) -> pd.DataFrame:
    """The public per-drive QC table (pseudonyms, platform names and counts only)."""
    keep = ["drive_id", "vehicle_id", "group_id", "platform", "brand", "model", "powertrain", "powertrain_raw", "size_class", "region",
            "software_era", "n_rows", "duration_s", "n_segments", "n_gaps", "gas_alive", "brake_alive", "torque_alive", "has_pose",
            "radar_unavailable", "lead_valid_frac"]
    t = drives[[c for c in keep if c in drives.columns]].merge(rates, on="drive_id", how="left")
    t = t.merge(scan[["drive_id", "yaw_all_zero", "lead_radar_share", "lead_vision_share"]], on="drive_id", how="left")
    t["yaw_all_zero"] = _as_bool(t["yaw_all_zero"])           # a drive without a scan row counts as not all-zero
    ids = t["drive_id"]
    t["lead_flag"] = lead_flag.reindex(ids).fillna("").to_numpy()
    t["duplicate_of"] = dup.reindex(ids).fillna("").to_numpy()
    t["multi_platform"] = multi.reindex(ids).fillna("").to_numpy()
    n_w, n_el, _ = eligible_hours_by_drive(windows)
    t["n_windows"] = n_w.reindex(ids).fillna(0).astype(int).to_numpy()
    t["n_eligible_windows"] = n_el.reindex(ids).fillna(0).astype(int).to_numpy()
    t["label_audit"] = audit.set_index("drive_id")["verdict"].reindex(ids).fillna("no_audit").to_numpy()
    t["label_resolved"] = ids.isin(resolved).to_numpy()
    flags = []
    for _, r in t.iterrows():
        f = [x for x in (r["rate_flag"], r["lead_flag"], r["multi_platform"]) if isinstance(x, str) and x]
        if r["duplicate_of"]:
            f.append("duplicate")
        if bool(r["yaw_all_zero"]):
            f.append("yaw_all_zero")
        for ch in ("gas", "brake", "torque"):
            col = f"{ch}_alive"
            if col in t.columns and not bool(r[col]):
                f.append(f"{ch}_dead")
        if not bool(r["has_pose"]) if "has_pose" in t.columns else False:
            f.append("no_pose")
        if str(r["label_audit"]).startswith((CONFLICT_PREFIX, SUSPECT_PREFIX)):
            f.append(str(r["label_audit"]) + ("/resolved" if r["label_resolved"] else ""))
        flags.append(";".join(f))
    t["flags"] = flags
    return t.sort_values("drive_id").reset_index(drop=True)


def flag_counts(col: pd.Series) -> pd.Series:
    """value_counts of a string flag column without the empty / missing entries (no replace / fillna downcasting)."""
    s = col[col.notna()]
    s = s[s.astype(str) != ""]
    return s.astype(str).value_counts()


def review_markdown(drives: pd.DataFrame, windows: pd.DataFrame | None, qc: pd.DataFrame, dead: tuple[dict, pd.DataFrame],
                    audit: pd.DataFrame, resolved: set[str], n_multi: int, multi_basis: str, private_qc_used: bool,
                    audit_available: bool, lead_sources: pd.DataFrame | None = None) -> str:
    """REVIEW.md: summary tables for the user's sign-off (pseudonyms and platform names only). `lead_sources` is
    lead_source_table(drives, scan) when the scan is available."""
    overall, per_plat = dead
    hours = float(drives["duration_s"].astype(float).sum() / 3600.0)
    cells = cell_table(drives, windows)
    verd = audit["verdict"].value_counts()
    conflicts = audit[audit["verdict"].str.startswith(CONFLICT_PREFIX)].copy()
    conflicts["resolved"] = conflicts["drive_id"].isin(resolved)
    unresolved = conflicts[~conflicts["resolved"]]
    suspects = audit[audit["verdict"].str.startswith(SUSPECT_PREFIX)]
    flagged = qc[qc["flags"].str.contains("rate_|lead_|duplicate|yaw_all_zero|multi_platform|conflict|suspect", regex=True)]
    L = []
    L.append("# VTT build review (S7)\n")
    L.append("Pseudonyms only. 'DAS' = driving automation system (das_any samples). Hours are nominal (rows / 10 / 3600).\n")
    L.append(f"- drives: {len(drives)}; vehicles: {drives['vehicle_id'].nunique()}; groups: {drives['group_id'].nunique()}; "
             f"platforms: {drives['platform'].nunique()}; hours: {hours:.1f}")
    if windows is not None and len(windows):
        el = int(windows["eligible"].astype(bool).sum())
        hs = int((windows["eligible"].astype(bool) & windows["human_only_strict"].astype(bool)).sum())
        L.append(f"- windows: {len(windows)}; eligible: {el} ({el / 60:.1f} h); eligible human-only strict: {hs} ({hs / 60:.1f} h)")
    else:
        L.append("- windows: index not present (run S6)")
    L.append(f"- carState rate source: {'private per-drive statistics' if private_qc_used else 'public rows-per-segment consistency'}; "
             f"label audit probe: {'present' if audit_available else 'absent (every verdict no_audit)'}")
    L.append(f"- unresolved label conflicts: {len(unresolved)}; resolved by overrides: {int(conflicts['resolved'].sum())}; suspects: {len(suspects)}\n")
    L.append("## 1. Cells (size_class x powertrain)\n")
    L.append(md_table(cells))
    L.append("\n## 2. Hours by cell and region\n")
    L.append(md_table(region_table(drives)))
    L.append("\n## 3. DAS share and eligible-window hours per cell\n")
    L.append(md_table(cells[["cell", "hours", "das_share", "eligible_window_h", "human_strict_window_h"]]))
    L.append("\n## 4. Dead channels (share of drives)\n")
    L.append(f"- gas dead: {overall['gas']:.1%} (expected about {EXPECTED_DEAD_SHARE['gas']:.0%}); brake dead: {overall['brake']:.1%} "
             f"(expected about {EXPECTED_DEAD_SHARE['brake']:.0%}); torque dead: {overall['torque']:.1%}\n")
    L.append(md_table(per_plat, max_rows=40))
    L.append("\n## 5. Signal checks\n")
    for name, col in (("rate", "rate_flag"), ("lead", "lead_flag"), ("multi-platform", "multi_platform")):
        vc = flag_counts(qc[col])
        L.append(f"- {name}: " + (", ".join(f"{k}={v}" for k, v in vc.items()) if len(vc) else "no flags"))
    dup_flag = qc["duplicate_of"].notna() & (qc["duplicate_of"].astype(str) != "")
    L.append(f"- duplicate drives: {int(dup_flag.sum())} (duplicate groups: {qc.loc[dup_flag, 'duplicate_of'].nunique()})")
    L.append(f"- yaw all-zero: {int(_as_bool(qc['yaw_all_zero']).sum())}; no pose: {int((~_as_bool(qc['has_pose'])).sum()) if 'has_pose' in qc.columns else 0}")
    L.append(f"- devices / groups with several platforms ({multi_basis} basis): {n_multi}\n")
    if lead_sources is not None and len(lead_sources):
        L.append("Lead source by radar availability (sample-weighted shares; the camera model is the only source when the "
                 "stack reports no radar, so a mismatch cannot occur by construction):\n")
        L.append(md_table(lead_sources, floatfmt="{:.3f}"))
    L.append("## 6. Label audit\n")
    L.append("- verdicts: " + ", ".join(f"{k}={v}" for k, v in verd.items()) + "\n")
    if len(conflicts):
        by_plat = conflicts.groupby(["platform", "powertrain", "verdict"], dropna=False).agg(
            drives=("drive_id", "size"), resolved=("resolved", "sum")).reset_index()
        L.append("### Conflicts by platform\n")
        L.append(md_table(by_plat))
        L.append("\n### Conflicting drives\n")
        L.append(md_table(conflicts[["drive_id", "vehicle_id", "platform", "powertrain", "verdict", "engine_rpm_pos_frac",
                                     "engine_off_moving_frac", "regen_any", "charging_any", "resolved"]], max_rows=300))
    if len(suspects):
        L.append("\n### Suspect drives\n")
        L.append(md_table(suspects[["drive_id", "vehicle_id", "platform", "powertrain", "verdict", "engine_rpm_pos_frac",
                                    "engine_off_moving_frac"]], max_rows=300))
    L.append("\n## 7. Flagged drives\n")
    L.append(md_table(flagged[["drive_id", "vehicle_id", "platform", "powertrain", "flags"]], max_rows=500))
    L.append("\n## 8. How to resolve\n")
    L.append("Edit `build/label_overrides.csv` (columns `scope,key,powertrain,label_note,resolve`; scope = drive | vehicle | platform; "
             "powertrain blank = keep, ICE/HEV/PHEV/BEV = relabel, null = clear; resolve = 1 marks the matched conflicts as reviewed) "
             "and re-run `scripts/s07_qc.py`. Duplicate drives and signal flags are informational; exclude drives through the private "
             "exclude list before S8 if needed.\n")
    return "\n".join(L)
