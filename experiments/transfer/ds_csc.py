#!/usr/bin/env python3
"""commaSteeringControl (comma.ai, Hugging Face `commaai/commaSteeringControl`) -> transfer windows of the
experiments/transfer contract (common.save_windows), dataset name `csc`.

Source: <external>/csc/data/<ARCHIVE>.zip, one archive per 2023 openpilot platform name; inside `<ARCHIVE>/NNNNN.csv`, one
CSV per ~60-s segment on an exact 10 Hz grid (598-601 rows, a few shorter or longer), 13 columns: vEgo, aEgo,
steeringAngleDeg, steeringPressed, steer, steerFiltered, latActive, roll, t, latAccelSteeringAngle, latAccelDesired,
latAccelLocalizer, epsFwVersion. The data carry no device, route, time or driver identifiers.

Per archive (one worker process; CSVs are read straight from the zip with zipfile + pandas, nothing is extracted or
executed): (1) an inventory pass over every member (row count = newline count, EPS firmware hash); (2) the members with
>= 598 rows are shuffled (numpy default_rng(seed)) and read in that order until `cap` windows are kept. A segment becomes
exactly one 600-row window: 598-599 rows -> the last row is repeated (note 'padded k'), > 600 rows -> trailing rows dropped
(note 'trimmed k': the task rule for 601 rows, applied as well to the rare 602+ files whose t is on the grid); off-grid t,
NaN in a required column, unexpected columns or < 20 % moving samples -> discarded (counted by reason).
Channels (DS_CSC.md section 2): v_mps = vEgo, a_mps2 = aEgo, steer_angle_deg = steeringAngleDeg (measured);
steer_rate_degps = central difference of the angle at 10 Hz (derived; the lab channel is the CAN rate sample);
lat_acc_mps2 = -latAccelLocalizer (measured; the data set's lateral acceleration is right-positive, the lab's yaw and
lateral acceleration are left-positive: `validate` checks the sign against the steering angle on every platform);
yaw_rate_radps = lat_acc_mps2 / v for v >= 1 m/s, NaN below (derived); curv_1pm = yaw / v for v >= 1 m/s (lab rule);
steer_pressed = steeringPressed; das_enabled = latActive (lateral control active: a proxy for the lab's controller-enabled
OR cruise-enabled flag); standstill = v < 0.3 m/s (derived). gas_pressed, brake_pressed, steer_torque_norm (the `steer`
columns are the EPS torque command, not the driver's torque) and the six lead values are NaN with lead_valid = 0: the
training augmentation's state for the dropped lead / torque / pedal groups (= common.apply_channel_mask(X, "V1")).
Labels: archive name -> current opendbc platform through the MIGRATION table (ccs_platforms.load_opendbc_names: AST parse
of the opendbc tree, identical to opendbc.car.fingerprints.MIGRATION of the openpilot venv for all 36 archives) -> the lab
platform table (footprint = wheelbase x mean track, model, stratum seen / held-out / absent against
build/index/drives.parquet) through ccs_platforms.Labeller; powertrain from the archive identity (the 2023 naming kept
hybrids as separate `* HYBRID *` platforms, so every archive here is a BEV or a gasoline car), cross-checked against the
lab label of the folded platform and specs_external.csv.

Run (any cwd; -I keeps the interpreter from importing anything next to the data):
  source env.sh && nice -n 10 $CPY -I experiments/transfer/ds_csc.py build [--workers 6] [--cap 1500] [--seed 0]
  $CPY -I experiments/transfer/ds_csc.py validate           # contract + physical-range checks of the saved set
  $CPY -I experiments/transfer/ds_csc.py labels             # only the archive -> platform -> label table
  $CPY -I experiments/transfer/ds_csc.py inventory          # only the row-count inventory of the archives
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common as T  # noqa: E402
import ccs_platforms as P  # noqa: E402

DATASET = "csc"                              # default --dataset; the work dir is <external>/<dataset>/work
DATA_DIR = T.EXTERNAL_ROOT / DATASET / "data"
CAP = 1500                                   # windows per archive (random, seed 0)
SEED = 0
WORKERS = 6                                  # at most 6 worker processes (task rule)
NICE = 10
COLS = ("vEgo", "aEgo", "steeringAngleDeg", "steeringPressed", "steer", "steerFiltered", "latActive", "roll", "t",
        "latAccelSteeringAngle", "latAccelDesired", "latAccelLocalizer", "epsFwVersion")
REQUIRED_FINITE = ("vEgo", "aEgo", "steeringAngleDeg")       # NaN here -> the segment is discarded
MIN_ROWS = 598                               # shorter segments are discarded (task rule)
STANDSTILL_MPS = 0.3                         # standstill = v < 0.3 m/s (the lab flag is 0 at v >= 0.3)
LAT_SIGN = -1.0                              # latAccel* are right-positive (device frame); the lab's yaw / lat_acc are left-positive
MAX_MEMBER_BYTES = 50 * 1024 * 1024          # a CSV member above this is not a 60-s segment (zip-bomb guard)
GRID_TOL_S = 1e-6
BEV_NAME_TOKENS = ("ELECTRIC", "_EV_", "_EV6", "EV6_", "BOLT", "LEAF", "IONIQ_5")
NOT_SCORED_TOKENS = ("HYBRID", "PHEV", "PRIME", "MIRAI", "SYNTHETIC")
PRESENT = ("v_mps", "a_mps2", "steer_angle_deg", "steer_rate_degps", "yaw_rate_radps", "curv_1pm", "lat_acc_mps2",
           "steer_pressed", "standstill", "das_enabled", "lead_valid")
ABSENT = tuple(c for c in T.CHANNELS if c not in PRESENT)
NOTE_RULES = ("yaw=-latAccelLocalizer/v (v>=1 m/s, NaN below);lat_acc=-latAccelLocalizer;rate=central diff of angle at 10 Hz;"
              "das=latActive;standstill=v<0.3;lead/torque/pedals absent (lead_valid=0)")
HEX16 = re.compile(r"\b[0-9a-f]{16}\b")


# ----------------------------------------------------------------------------- small helpers
def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _round(o, nd: int = 4):
    if isinstance(o, dict):
        return {str(k): _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    if isinstance(o, (float, np.floating)):
        if not np.isfinite(o):
            return None
        return round(float(o), nd) if abs(o) >= 10.0 ** (-nd) or o == 0 else float(f"{float(o):.3g}")   # keep small fractions
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def archives(data_dir: Path, only: list[str] | None = None) -> list[Path]:
    zips = sorted(p for p in Path(data_dir).glob("*.zip") if not p.stem.upper().startswith("SYNTHETIC"))
    if only:
        want = set(only)
        zips = [p for p in zips if p.stem in want]
        miss = want - {p.stem for p in zips}
        if miss:
            raise FileNotFoundError(f"archives not found: {sorted(miss)}")
    if not zips:
        raise FileNotFoundError(f"no <PLATFORM>.zip archives in {data_dir}")
    return zips


def setup_log(work: Path, name: str = "csc_build.log") -> logging.Logger:
    work.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("ds_csc")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(work / name, mode="a")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


# ----------------------------------------------------------------------------- labels
def fold_archive(archive: str, migration: dict[str, str], platforms: set[str]) -> tuple[str | None, str]:
    """2023 archive name -> current opendbc platform. The legacy MIGRATION keys are the 2023 CAR strings with spaces,
    hyphens and parentheses ('HONDA CR-V 2017', 'HONDA CIVIC (BOSCH) 2019'); the archive names drop those, so both sides
    are compared without non-alphanumerics. Returns (platform, 'MIGRATION' | 'current name' | 'unmapped')."""
    nm: dict[str, set[str]] = {}
    for k, v in migration.items():
        nm.setdefault(_norm(k), set()).add(v)
    coll = {k: v for k, v in nm.items() if len(v) > 1}
    if coll:
        raise RuntimeError(f"MIGRATION keys collide after normalisation: {coll}")
    pn = {_norm(p): p for p in platforms}
    k = _norm(archive)
    if k in nm:
        return next(iter(nm[k])), "MIGRATION"
    if k in pn:
        return pn[k], "current name"
    return None, "unmapped"


def lab_powertrain_mix(drives: Path) -> tuple[dict[str, str], dict[str, str]]:
    """platform -> 'HEV 29 / ICE 26 / MIXED 7' (powertrain_raw counts of the lab drives) and -> split counts string."""
    if not Path(drives).is_file():
        return {}, {}
    d = pd.read_parquet(drives, columns=["platform", "powertrain_raw", "split_platform"])
    d = d[d["platform"].notna()]
    mix, spl = {}, {}
    for p, g in d.groupby("platform"):
        vc = g["powertrain_raw"].fillna("UNK").value_counts()
        mix[p] = " / ".join(f"{k} {v}" for k, v in vc.items())
        vs = g["split_platform"].fillna("none").value_counts()
        spl[p] = " / ".join(f"{k} {v}" for k, v in vs.items())
    return mix, spl


def label_archives(names: list[str], opendbc_root: Path | None = None) -> pd.DataFrame:
    """One row per archive: folded platform (how), model, footprint, size class, borderline, powertrain (archive rule), the
    lab label of the folded platform (mode and mix), the specs_external row (if any), stratum and notes."""
    root = P.find_opendbc_root(opendbc_root)
    migration, platforms = P.load_opendbc_names(root)
    L = P.Labeller()
    mix, spl = lab_powertrain_mix(P.DRIVES)
    ext_rows: dict[str, dict] = {}                       # specs_external rows tagged commaSteeringControl, by platform key
    if P.EXT_SPECS.is_file():
        e = pd.read_csv(P.EXT_SPECS, dtype=str, keep_default_na=False)
        for r in e.itertuples(index=False):
            if "commaSteeringControl" not in r.datasets:
                continue
            for k in [x.strip() for x in r.platform_keys.replace(";", " ").replace(",", " ").split() if x.strip()]:
                ext_rows.setdefault(k, r._asdict())
    rows = []
    for a in names:
        up = a.upper()
        bad = [t for t in NOT_SCORED_TOKENS if t in up]
        if bad:
            raise ValueError(f"{a}: archive carries {bad}; only BEV and gasoline archives are handled (extend the rule)")
        folded, how = fold_archive(a, migration, platforms)
        lab = L.label(folded, a) if folded else L.label(a)
        ext = ext_rows.get(a) or (ext_rows.get(folded) if folded else None)
        lab_pt = lab["powertrain_raw"]
        is_bev_name = any(t in up for t in BEV_NAME_TOKENS)
        if (lab_pt == "BEV") != is_bev_name:
            raise ValueError(f"{a}: lab powertrain {lab_pt!r} vs name tokens (BEV={is_bev_name}) disagree; resolve by hand")
        pt = "BEV" if lab_pt == "BEV" else "ICE"
        pt_note = ""
        if lab_pt not in ("BEV", "ICE"):
            pt_note = f"lab folded platform labelled {lab_pt} (mix {mix.get(folded, 'n/a')}); archive is the 2023 gasoline platform (hybrid twin separate) -> ICE"
        ext_fp = float(ext["footprint_m2"]) if ext and ext.get("footprint_m2") else float("nan")
        ext_pt = (ext.get("powertrain") or "").upper() if ext else ""
        ext_bl = str(ext.get("borderline", "")).strip().lower() == "true" if ext else False
        if ext and ext_pt and ext_pt != pt:
            pt_note = (pt_note + "; " if pt_note else "") + f"specs_external says {ext_pt}"
        if ext and np.isfinite(ext_fp) and np.isfinite(lab["footprint_m2"]) and abs(ext_fp - lab["footprint_m2"]) > 0.002:
            pt_note = (pt_note + "; " if pt_note else "") + f"specs_external footprint {ext_fp:.3f} differs from the lab table"
        borderline = bool(lab["borderline"]) or ext_bl
        bl_note = ""
        if ext_bl and not lab["borderline"]:
            bl_note = "borderline per specs_external (alternative published dimensions cross a cut)"
        rows.append(dict(archive=a, platform=folded, fold=how, model=lab["model"], brand=lab.get("brand"),
                         footprint_m2=lab["footprint_m2"], size_class=lab["size_class"], borderline=borderline,
                         powertrain=pt, powertrain_lab=lab_pt, lab_mix=mix.get(folded, ""), lab_splits=spl.get(folded, ""),
                         stratum=lab["stratum"], label_source=lab["label_source"], spec_confidence=L.lab.get(folded, {}).get("confidence"),
                         wheelbase_mm=L.lab.get(folded, {}).get("wheelbase_mm"), in_specs_external=bool(ext), note="; ".join(x for x in (pt_note, bl_note, lab["note"]) if x)))
    df = pd.DataFrame(rows)
    df.attrs["opendbc"] = P.opendbc_version(root)
    return df


# ----------------------------------------------------------------------------- one segment -> one window
def segment_window(df: pd.DataFrame, label: dict, archive: str, member: str, note_rules: str = NOTE_RULES) -> tuple[T.CutResult | None, dict]:
    """(CutResult with one window or None, qc dict). The qc dict always carries `status` ('ok' or the discard reason)."""
    base = Path(member).stem
    qc: dict = dict(archive=archive, member=base, n_rows_raw=int(len(df)), pad=0, trim=0, status="ok")
    if list(df.columns) != list(COLS):
        qc["status"] = "columns"
        return None, qc
    n = len(df)
    if n < MIN_ROWS:
        qc["status"] = "short"
        return None, qc
    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(t).all() or not np.allclose(t, np.arange(n) / T.HZ, atol=GRID_TOL_S, rtol=0.0):
        qc["status"] = "off-grid t"
        return None, qc
    num = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64) for c in
           ("vEgo", "aEgo", "steeringAngleDeg", "latAccelLocalizer", "latAccelSteeringAngle", "roll")}
    for c in num:
        qc[f"nan_{c}"] = int(np.isnan(num[c]).sum())
    if any(qc[f"nan_{c}"] for c in REQUIRED_FINITE):
        qc["status"] = "nan"
        return None, qc
    flags = {c: df[c].to_numpy() for c in ("steeringPressed", "latActive")}
    for c, arr in flags.items():
        if arr.dtype != bool:
            s = pd.Series(arr).astype(str).str.strip().str.lower()
            if not s.isin(("true", "false", "0", "1")).all():
                qc["status"] = "flag values"
                return None, qc
            flags[c] = s.isin(("true", "1")).to_numpy()
    W = T.WINDOW_SAMPLES
    if n > W:
        qc["trim"] = n - W
        sl = slice(0, W)
        num = {c: a[sl] for c, a in num.items()}
        flags = {c: a[sl] for c, a in flags.items()}
    elif n < W:
        qc["pad"] = W - n
        num = {c: np.concatenate([a, np.repeat(a[-1:], W - n)]) for c, a in num.items()}
        flags = {c: np.concatenate([a, np.repeat(a[-1:], W - n)]) for c, a in flags.items()}
    v, a, ang = num["vEgo"], num["aEgo"], num["steeringAngleDeg"]
    rate = np.gradient(ang) * T.HZ                                   # central difference, one-sided at the ends
    lat = LAT_SIGN * num["latAccelLocalizer"]                        # left-positive lateral acceleration (measured)
    yaw = np.full(W, np.nan)
    ok = (v >= T.MOVING_V_MPS) & np.isfinite(lat)
    np.divide(lat, v, out=yaw, where=ok)
    curv, _ = T.curvature_and_lat_acc(v, yaw)                        # yaw / v while v >= 1 m/s, NaN otherwise
    sp = flags["steeringPressed"].astype(np.float64)
    das = flags["latActive"].astype(np.float64)
    ss = (v < STANDSTILL_MPS).astype(np.float64)
    cmap = {"v_mps": v, "a_mps2": a, "steer_angle_deg": ang, "steer_rate_degps": rate, "yaw_rate_radps": yaw, "curv_1pm": curv,
            "lat_acc_mps2": lat, "steer_pressed": sp, "standstill": ss, "das_enabled": das, "lead_valid": np.zeros(W)}
    note = f"stratum={label['stratum']};label={label['label_source']};pt_lab={label['powertrain_lab']};{note_rules}"
    if qc["pad"]:
        note += f";padded {qc['pad']}"
    if qc["trim"]:
        note += f";trimmed {qc['trim']}"
    info = dict(dataset=label.get("dataset", DATASET), vehicle_id=archive, model=label["model"], platform=label["platform"],
                drive_id=f"{archive}_{hashlib.sha1(member.encode()).hexdigest()[:10]}", footprint_m2=label["footprint_m2"],
                size_class=label["size_class"], powertrain=label["powertrain"], borderline=bool(label["borderline"]), note=note)
    frame = pd.DataFrame({"v": v})                                   # only its length is used (arrays are passed directly)
    res = T.cut_windows(frame, cmap, t_col=None, info=info, align="drive")
    mv = v > 5.0
    qc.update(v_mean=float(v.mean()), v_max=float(v.max()), moving_frac=float((v >= T.MOVING_V_MPS).mean()),
              standstill_frac=float(ss.mean()), das_frac=float(das.mean()), steer_pressed_frac=float(sp.mean()),
              a_absmax=float(np.abs(a).max()), ang_absmax=float(np.abs(ang).max()), rate_absmax=float(np.abs(rate).max()),
              lat_absmax=float(np.nanmax(np.abs(lat))), yaw_absmax=float(np.nanmax(np.abs(yaw))) if np.isfinite(yaw).any() else np.nan,
              corr_steer_lat=float(np.corrcoef(ang[mv], lat[mv])[0, 1]) if mv.sum() > 100 and ang[mv].std() > 0.5 and np.isfinite(lat[mv]).all() else np.nan,
              corr_latSA_latLoc=float(np.corrcoef(num["latAccelSteeringAngle"][mv], num["latAccelLocalizer"][mv])[0, 1]) if mv.sum() > 100 and np.isfinite(num["latAccelSteeringAngle"][mv]).all() and np.isfinite(num["latAccelLocalizer"][mv]).all() and num["latAccelSteeringAngle"][mv].std() > 0.05 else np.nan,
              eps_fw=hashlib.sha1(str(df["epsFwVersion"].iloc[0]).encode()).hexdigest()[:8])
    if res.X.shape[0] == 0:
        qc["status"] = "stationary"
        return None, qc
    return res, qc


# ----------------------------------------------------------------------------- one archive (worker)
def _fw_hash_from_bytes(b: bytes) -> str:
    """sha1 prefix of the last column of the last data line (the EPS firmware string) without parsing the CSV."""
    tail = b.rstrip(b"\n").rsplit(b"\n", 1)[-1]
    return hashlib.sha1(tail.rsplit(b",", 1)[-1].strip().strip(b'"')).hexdigest()[:8]


def process_archive(zip_path: str, label: dict, cap: int, seed: int, parts_dir: str) -> dict:
    """Inventory every CSV member of one archive, then read a seeded random sample until `cap` windows are kept. Writes
    <parts_dir>/<archive>.npz (X) + .meta.parquet + .qc.parquet and returns the inventory / counts dict."""
    t0 = time.time()
    archive = Path(zip_path).stem
    out = dict(archive=archive, n_members=0, n_non_csv=0, n_oversize=0, n_eligible=0, n_processed=0, n_kept=0, cap=cap, seed=seed)
    with zipfile.ZipFile(zip_path) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        csvs = sorted((i for i in infos if i.filename.lower().endswith(".csv")), key=lambda i: i.filename)
        out["n_non_csv"] = len(infos) - len(csvs)
        big = [i for i in csvs if i.file_size > MAX_MEMBER_BYTES]
        out["n_oversize"] = len(big)
        csvs = [i for i in csvs if i.file_size <= MAX_MEMBER_BYTES]
        out["n_members"] = len(csvs)
        rows = np.zeros(len(csvs), dtype=np.int64)
        fw = set()
        for k, i in enumerate(csvs):
            b = z.read(i)
            rows[k] = b.count(b"\n") - 1 + (0 if b.endswith(b"\n") else 1)
            fw.add(_fw_hash_from_bytes(b))
        hist = Counter(rows.tolist())
        out["rows_hist"] = {int(k): int(v) for k, v in sorted(hist.items())}
        out["n_eps_fw_all"] = len(fw)
        out["inventory_s"] = round(time.time() - t0, 1)
        eligible = np.flatnonzero(rows >= MIN_ROWS)
        out["n_eligible"] = int(len(eligible))
        order = eligible[np.random.default_rng(seed).permutation(len(eligible))]
        kept: list[T.CutResult] = []
        qcs: list[dict] = []
        reasons: Counter = Counter()
        for k in order:
            if len(kept) >= cap:
                break
            i = csvs[k]
            try:
                df = pd.read_csv(io.BytesIO(z.read(i)))
            except Exception as e:  # noqa: BLE001  a malformed member is a discard, not a crash
                qcs.append(dict(archive=archive, member=Path(i.filename).stem, n_rows_raw=int(rows[k]), status=f"read error: {type(e).__name__}"))
                reasons["read error"] += 1
                continue
            res, qc = segment_window(df, label, archive, i.filename)
            qc["rank"] = len(qcs)
            qcs.append(qc)
            if res is None:
                reasons[qc["status"]] += 1
                continue
            kept.append(res)
    out["n_processed"] = len(qcs)
    out["n_kept"] = len(kept)
    out["discards"] = dict(reasons)
    out["n_padded"] = int(sum(1 for q in qcs if q.get("status") == "ok" and q.get("pad", 0) > 0))
    out["n_trimmed"] = int(sum(1 for q in qcs if q.get("status") == "ok" and q.get("trim", 0) > 0))
    parts = Path(parts_dir)
    parts.mkdir(parents=True, exist_ok=True)
    X, meta = T.concat_results(kept)
    np.savez(parts / f"{archive}.npz", X=X)
    meta.to_parquet(parts / f"{archive}.meta.parquet", index=False)
    pd.DataFrame(qcs).to_parquet(parts / f"{archive}.qc.parquet", index=False)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


# ----------------------------------------------------------------------------- validation of the saved set
def validate_saved(dataset: str, labels: pd.DataFrame | None = None) -> dict:
    X, meta = T.load_windows(dataset)
    n = X.shape[0]
    rep: dict = dict(n_windows=n, n_vehicles=int(meta["vehicle_id"].nunique()), n_models=int(meta["model"].nunique()),
                     n_platforms=int(meta["platform"].nunique()), n_drives=int(meta["drive_id"].nunique()),
                     hours=round(n * T.WINDOW_SAMPLES * T.DT_S / 3600.0, 2), problems=[])
    prob = rep["problems"]
    v = X[:, :, T.CH["v_mps"]]
    # presence pattern
    nan_frac = {c: float(np.isnan(X[:, :, T.CH[c]]).mean()) for c in T.CHANNELS}
    rep["nan_frac"] = nan_frac
    for c in ("v_mps", "a_mps2", "steer_angle_deg", "steer_rate_degps", "lat_acc_mps2", "steer_pressed", "standstill", "das_enabled", "lead_valid"):
        if nan_frac[c] > 0:
            prob.append(f"NaN inside present channel {c}: {nan_frac[c]:.4f}")
    for c in ABSENT:
        if nan_frac[c] < 1.0:
            prob.append(f"values inside absent channel {c}")
    yaw = X[:, :, T.CH["yaw_rate_radps"]]; curv = X[:, :, T.CH["curv_1pm"]]
    low = v < T.MOVING_V_MPS
    rep["yaw_nan_matches_v_below_1"] = bool(np.array_equal(np.isnan(yaw), low)) and bool(np.array_equal(np.isnan(curv), low))
    if not rep["yaw_nan_matches_v_below_1"]:
        prob.append("yaw / curv NaN pattern differs from v < 1 m/s")
    cp = sorted(set(meta["channels_present"].unique()))
    rep["channels_present"] = cp
    if cp != [",".join(PRESENT)]:
        prob.append(f"channels_present {cp} != {','.join(PRESENT)}")
    # flags
    for c in T.BINARY_CHANNELS:
        col = X[:, :, T.CH[c]]
        fin = col[np.isfinite(col)]
        if fin.size and not np.isin(fin, (0.0, 1.0)).all():
            prob.append(f"{c} not 0/1")
    lv = X[:, :, T.CH["lead_valid"]]
    if not (lv == 0).all():
        prob.append("lead_valid must be 0 everywhere (lead group absent)")
    ss = X[:, :, T.CH["standstill"]]
    rep["standstill_agrees_with_v"] = bool(np.array_equal(ss, (v < STANDSTILL_MPS).astype(np.float32)))
    if not rep["standstill_agrees_with_v"]:
        prob.append("standstill != (v < 0.3 m/s)")
    das = X[:, :, T.CH["das_enabled"]]
    rep["das_frac"] = dict(mean=float(das.mean()), q10=float(np.quantile(das.mean(1), 0.10)), median=float(np.median(das.mean(1))),
                           frac_windows_all_on=float((das.mean(1) == 1).mean()), frac_windows_all_off=float((das.mean(1) == 0).mean()))
    if np.abs(meta["das_frac"].to_numpy(np.float32) - das.mean(1)).max() > 1e-5:
        prob.append("meta.das_frac disagrees with X")
    # ranges
    a = X[:, :, T.CH["a_mps2"]]; ang = X[:, :, T.CH["steer_angle_deg"]]; rate = X[:, :, T.CH["steer_rate_degps"]]; lat = X[:, :, T.CH["lat_acc_mps2"]]
    q = lambda x, p: float(np.nanquantile(x, p))  # noqa: E731
    rep["v_mps"] = dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean()), p50=q(v, 0.5), p99=q(v, 0.99), frac_moving=float((v >= T.MOVING_V_MPS).mean()),
                        frac_below_0p3=float((v < STANDSTILL_MPS).mean()), frac_zero=float((v == 0).mean()), max_kmh=float(v.max() * 3.6))
    rep["a_mps2"] = dict(min=float(a.min()), max=float(a.max()), std=float(a.std()), p01=q(a, 0.01), p99=q(a, 0.99), frac_abs_gt_4=float((np.abs(a) > 4).mean()), frac_abs_gt_5=float((np.abs(a) > 5).mean()), frac_abs_gt_8=float((np.abs(a) > 8).mean()),
                         n_abs_gt_8=int((np.abs(a) > 8).sum()), n_windows_abs_gt_8=int((np.abs(a) > 8).any(1).sum()))
    rep["steer_angle_deg"] = dict(min=float(ang.min()), max=float(ang.max()), p01=q(ang, 0.01), p99=q(ang, 0.99), abs_p999=q(np.abs(ang), 0.999), mean=float(ang.mean()))
    rep["steer_rate_degps"] = dict(std=float(rate.std()), p01=q(rate, 0.01), p99=q(rate, 0.99), abs_p999=q(np.abs(rate), 0.999), abs_max=float(np.abs(rate).max()))
    rep["lat_acc_mps2"] = dict(std=float(lat.std()), p01=q(lat, 0.01), p99=q(lat, 0.99), abs_max=float(np.abs(lat).max()), mean=float(lat.mean()))
    rep["yaw_rate_radps"] = dict(std=float(np.nanstd(yaw)), p01=q(yaw, 0.01), p99=q(yaw, 0.99), abs_max=float(np.nanmax(np.abs(yaw))))
    rep["curv_1pm"] = dict(std=float(np.nanstd(curv)), p01=q(curv, 0.01), p99=q(curv, 0.99), abs_max=float(np.nanmax(np.abs(curv))))
    if v.min() < -0.5 or v.max() > 60:
        prob.append(f"v_mps outside [-0.5, 60]: min {v.min():.2f} max {v.max():.2f}")
    if rep["a_mps2"]["frac_abs_gt_8"] > 1e-4:                     # isolated aEgo spikes are clipped by the scorer; a tail above 0.01 % is a unit problem
        prob.append(f"|a_mps2| above 8 m/s^2 on {rep['a_mps2']['frac_abs_gt_8']:.2e} of the samples (max {np.abs(a).max():.2f})")
    if np.abs(lat).max() > 15:
        prob.append(f"|lat_acc| above 15 m/s^2: max {np.abs(lat).max():.2f}")
    if np.abs(ang).max() > 720:
        prob.append("steering angle above 720 deg")
    # window rule and meta consistency
    mf = (v >= T.MOVING_V_MPS).mean(1)
    rep["moving_frac"] = dict(min=float(mf.min()), median=float(np.median(mf)), mean=float(mf.mean()))
    if (mf < T.MIN_MOVING_FRAC).any():
        prob.append("windows below the moving rule")
    if np.abs(meta["moving_frac"].to_numpy(np.float32) - mf).max() > 1e-5:
        prob.append("meta.moving_frac disagrees with X")
    if meta.duplicated(["drive_id", "w_idx"]).any():
        prob.append("(drive_id, w_idx) not unique")
    if (meta.groupby("drive_id").size() != 1).any():
        prob.append("a drive_id carries more than one window")
    # labels
    for c in ("model", "platform", "size_class", "powertrain", "footprint_m2", "borderline"):
        k = meta.groupby("vehicle_id")[c].nunique(dropna=False)
        if (k > 1).any():
            prob.append(f"{int((k > 1).sum())} vehicles carry more than one {c}")
    if not meta["footprint_m2"].notna().all():
        prob.append("footprint missing on some windows")
    if not meta["powertrain"].isin(["BEV", "ICE"]).all():
        prob.append(f"powertrain outside BEV / ICE: {sorted(set(meta['powertrain']) - {'BEV', 'ICE'})}")
    derived = T.classify_footprint(meta["footprint_m2"].to_numpy(np.float64))
    if (derived != meta["size_class"].to_numpy(object)).any():
        prob.append("size_class disagrees with the footprint thresholds")
    rep["per_size_class"] = meta["size_class"].value_counts().to_dict()
    rep["per_powertrain"] = meta["powertrain"].value_counts().to_dict()
    strata = meta["note"].str.extract(r"stratum=([a-z-]+)")[0]
    rep["per_stratum_windows"] = strata.value_counts().to_dict()
    rep["per_stratum_vehicles"] = meta.assign(s=strata).groupby("s")["vehicle_id"].nunique().to_dict()
    rep["borderline_vehicles"] = sorted(meta.loc[meta["borderline"], "vehicle_id"].unique().tolist())
    rep["padded_windows"] = int(meta["note"].str.contains(";padded ").sum())
    rep["trimmed_windows"] = int(meta["note"].str.contains(";trimmed ").sum())
    # ids and strings
    bad = [s for s in pd.concat([meta["vehicle_id"], meta["drive_id"]]).unique() if not re.match(r"^[A-Za-z0-9][A-Za-z0-9 ._+()/-]*$", s)]
    if bad:
        prob.append(f"ids with unusual characters: {bad[:5]}")
    strings = pd.concat([meta["vehicle_id"], meta["drive_id"], meta["model"], meta["platform"].astype(str), meta["note"]]).unique()
    if any(HEX16.search(str(s)) for s in strings):
        prob.append("16-hex token inside meta strings")
    # per platform: sign of the lateral channel against the steering angle, ranges, counts
    per = []
    for vid, g in meta.groupby("vehicle_id", sort=True):
        idx = g.index.to_numpy()
        vv = v[idx]; aa = ang[idx]; yy = yaw[idx]; ll = lat[idx]
        m = (vv > 5.0) & np.isfinite(yy)
        c_sy = float(np.corrcoef(aa[m], yy[m])[0, 1]) if m.sum() > 1000 else np.nan
        c_sl = float(np.corrcoef(aa[m], ll[m])[0, 1]) if m.sum() > 1000 else np.nan
        per.append(dict(vehicle_id=vid, platform=g["platform"].iloc[0], model=g["model"].iloc[0], n_windows=int(len(g)), hours=round(len(g) / 60.0, 2),
                        size_class=g["size_class"].iloc[0], powertrain=g["powertrain"].iloc[0], footprint_m2=float(g["footprint_m2"].iloc[0]),
                        borderline=bool(g["borderline"].iloc[0]), stratum=strata.iloc[idx[0]] if len(idx) else None,
                        v_mean=float(vv.mean()), v_max=float(vv.max()), v_p50=float(np.median(vv)), moving_frac=float((vv >= 1).mean()),
                        standstill_frac=float((vv < STANDSTILL_MPS).mean()), das_frac=float(das[idx].mean()), steer_pressed_frac=float(X[idx][:, :, T.CH["steer_pressed"]].mean()),
                        ang_std=float(aa.std()), ang_abs_p99=float(np.quantile(np.abs(aa), 0.99)), rate_std=float(rate[idx].std()),
                        lat_std=float(ll.std()), yaw_std=float(np.nanstd(yy)), a_std=float(a[idx].std()),
                        corr_steer_yaw=c_sy, corr_steer_lat=c_sl, n_padded=int(g["note"].str.contains(";padded ").sum()), n_trimmed=int(g["note"].str.contains(";trimmed ").sum())))
    per = pd.DataFrame(per)
    rep["per_vehicle"] = per.to_dict(orient="records")
    neg = per[per["corr_steer_yaw"].notna() & (per["corr_steer_yaw"] < 0.5)]
    rep["sign_check"] = dict(min_corr_steer_yaw=float(per["corr_steer_yaw"].min()), median=float(per["corr_steer_yaw"].median()),
                             n_platforms_below_0p5=int(len(neg)), below=neg["vehicle_id"].tolist())
    if len(neg):
        prob.append(f"steering-angle / yaw correlation below 0.5 on {neg['vehicle_id'].tolist()} (sign convention?)")
    if labels is not None:
        lab = labels.set_index("archive")
        for r in per.itertuples():
            L = lab.loc[r.vehicle_id]
            if L["model"] != r.model or L["size_class"] != r.size_class or L["powertrain"] != r.powertrain or abs(float(L["footprint_m2"]) - r.footprint_m2) > 1e-3:
                prob.append(f"labels of {r.vehicle_id} differ between meta and the label table")
    return rep


# ----------------------------------------------------------------------------- markdown
def fmt1(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.1f}"


def pct(x) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{100.0 * x:.1f}"


def tables_md(labels: pd.DataFrame, inv: pd.DataFrame, rep: dict) -> str:
    L = ["# csc tables (written by ds_csc.py; numbers to one decimal, footprints to 3 decimals)", ""]
    # label table
    L.append("## Archive -> platform -> label")
    L.append("")
    L.append("| archive (2023 name) | folded platform | via | model | footprint m^2 | class | borderline | powertrain | lab label of folded platform (drive mix) | stratum |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in labels.sort_values("archive").itertuples():
        L.append(f"| {r.archive} | {r.platform} | {r.fold} | {r.model} | {r.footprint_m2:.3f} | {r.size_class} | {'yes' if r.borderline else 'no'} | {r.powertrain} | {r.powertrain_lab} ({r.lab_mix}) | {r.stratum} |")
    L.append("")
    # inventory / counts table
    per = pd.DataFrame(rep["per_vehicle"]).set_index("vehicle_id") if rep.get("per_vehicle") else None
    L.append("## Per archive: segments available, eligible, kept; window statistics")
    L.append("")
    L.append("| archive | class | powertrain | stratum | segments | rows 598 / 599 / 600 / 601 / other | eligible (>= 598) | read | kept | padded | trimmed | discards | EPS fw (all) | hours | v mean m/s | moving % | v<0.3 % | das % | steer pressed % | corr(steer, yaw) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    tot = Counter()
    for r in inv.sort_values("archive").itertuples():
        h = r.rows_hist if isinstance(r.rows_hist, dict) else json.loads(r.rows_hist)
        h = {int(k): int(v) for k, v in h.items()}
        other = sum(v for k, v in h.items() if k not in (598, 599, 600, 601))
        hist = f"{h.get(598, 0)} / {h.get(599, 0)} / {h.get(600, 0)} / {h.get(601, 0)} / {other}"
        disc = r.discards if isinstance(r.discards, dict) else json.loads(r.discards or "{}")
        disc_s = ", ".join(f"{k} {v}" for k, v in sorted(disc.items())) or "-"
        p = per.loc[r.archive] if per is not None and r.archive in per.index else None
        lab = labels.set_index("archive").loc[r.archive]
        L.append(f"| {r.archive} | {lab['size_class']} | {lab['powertrain']} | {lab['stratum']} | {r.n_members:,} | {hist} | {r.n_eligible:,} | {r.n_processed:,} | {r.n_kept:,} | {r.n_padded} | {r.n_trimmed} | {disc_s} | {r.n_eps_fw_all} | "
                 + (f"{p['hours']:.1f} | {p['v_mean']:.1f} | {pct(p['moving_frac'])} | {pct(p['standstill_frac'])} | {pct(p['das_frac'])} | {pct(p['steer_pressed_frac'])} | {p['corr_steer_yaw']:.2f} |" if p is not None else "- | - | - | - | - | - | - |"))
        tot["segments"] += r.n_members; tot["eligible"] += r.n_eligible; tot["read"] += r.n_processed; tot["kept"] += r.n_kept; tot["padded"] += r.n_padded; tot["trimmed"] += r.n_trimmed
    L.append(f"| total | | | | {tot['segments']:,} | | {tot['eligible']:,} | {tot['read']:,} | {tot['kept']:,} | {tot['padded']} | {tot['trimmed']} | | | {rep['hours']:.1f} | | | | | | |")
    L.append("")
    # class / powertrain / stratum summary
    m = pd.DataFrame(rep["per_vehicle"])
    L.append("## Windows and archives per class, powertrain and stratum")
    L.append("")
    L.append("| group | value | archives | models | windows | hours |")
    L.append("|---|---|---|---|---|---|")
    for col in ("size_class", "powertrain", "stratum"):
        for val, g in m.groupby(col):
            L.append(f"| {col} | {val} | {len(g)} | {g['model'].nunique()} | {int(g['n_windows'].sum()):,} | {g['n_windows'].sum() / 60.0:.1f} |")
    for (sc, pt), g in m.groupby(["size_class", "powertrain"]):
        L.append(f"| class x powertrain | {sc} {pt} | {len(g)} | {g['model'].nunique()} | {int(g['n_windows'].sum()):,} | {g['n_windows'].sum() / 60.0:.1f} |")
    L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------------- commands
def cmd_labels(args, log) -> pd.DataFrame:
    zips = archives(args.data_dir, args.archives)
    labels = label_archives([z.stem for z in zips], args.opendbc_root)
    args.work.mkdir(parents=True, exist_ok=True)
    out = args.work / "csc_platforms.csv"
    labels.assign(footprint_m2=labels["footprint_m2"].round(3)).to_csv(out, index=False)
    log.info("labels: %d archives -> %d platforms, %d models; opendbc %s; written %s", len(labels), labels["platform"].nunique(), labels["model"].nunique(), labels.attrs.get("opendbc"), out.name)
    for r in labels.itertuples():
        log.info("  %-36s -> %-26s [%s] %-24s fp %.3f %-5s bl=%d %s (lab %s) %s %s", r.archive, r.platform, r.fold, r.model, r.footprint_m2, r.size_class, r.borderline, r.powertrain, r.powertrain_lab, r.stratum, ("| " + r.note) if r.note else "")
    unmapped = labels[labels["platform"].isna()]
    if len(unmapped):
        raise RuntimeError(f"unmapped archives: {unmapped['archive'].tolist()}")
    if labels["footprint_m2"].isna().any():
        raise RuntimeError(f"no footprint for {labels.loc[labels['footprint_m2'].isna(), 'archive'].tolist()}")
    return labels


def cmd_build(args, log) -> int:
    t0 = time.time()
    try:
        if os.nice(0) < NICE:
            os.nice(NICE - os.nice(0))
    except OSError:
        pass
    labels = cmd_labels(args, log)
    zips = archives(args.data_dir, args.archives)
    parts = args.work / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    lab = {r["archive"]: dict(r, dataset=args.dataset) for r in labels.to_dict(orient="records")}
    todo = []
    for z in zips:
        if args.resume and all((parts / f"{z.stem}{sfx}").exists() for sfx in (".npz", ".meta.parquet", ".qc.parquet", ".inv.json")):
            log.info("resume: %s already built", z.stem)
            continue
        todo.append(z)
    # largest archives first so the pool tail is short
    todo.sort(key=lambda p: -p.stat().st_size)
    log.info("build: %d archives (%d to do), cap %d, seed %d, workers %d, %.1f GB zipped", len(zips), len(todo), args.cap, args.seed, args.workers, sum(p.stat().st_size for p in zips) / 1e9)
    with ProcessPoolExecutor(max_workers=max(1, min(WORKERS, args.workers))) as ex:
        futs = {ex.submit(process_archive, str(z), lab[z.stem], args.cap, args.seed, str(parts)): z.stem for z in todo}
        for f in as_completed(futs):
            name = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                log.error("%s failed: %s: %s", name, type(e).__name__, str(e)[:300])
                raise
            (parts / f"{name}.inv.json").write_text(json.dumps(_round(r), indent=1))
            log.info("%-36s members %6d eligible %6d read %5d kept %5d padded %4d trimmed %2d discards %s fw %3d  %.0f s (inventory %.0f s)",
                     r["archive"], r["n_members"], r["n_eligible"], r["n_processed"], r["n_kept"], r["n_padded"], r["n_trimmed"], r["discards"] or "-", r["n_eps_fw_all"], r["elapsed_s"], r["inventory_s"])
    # assemble
    inv_rows, Xs, metas, qcs = [], [], [], []
    for z in zips:
        inv_rows.append(json.loads((parts / f"{z.stem}.inv.json").read_text()))
        Xs.append(np.load(parts / f"{z.stem}.npz")["X"])
        metas.append(pd.read_parquet(parts / f"{z.stem}.meta.parquet"))
        qcs.append(pd.read_parquet(parts / f"{z.stem}.qc.parquet"))
    X = np.concatenate(Xs) if Xs else np.zeros((0, T.WINDOW_SAMPLES, T.N_CHANNELS), np.float32)
    meta = pd.concat(metas, ignore_index=True)
    qc = pd.concat(qcs, ignore_index=True)
    inv = pd.DataFrame(inv_rows)
    inv["rows_hist"] = inv["rows_hist"].map(json.dumps)
    inv["discards"] = inv["discards"].map(json.dumps)
    inv.to_csv(args.work / "csc_inventory.csv", index=False)
    qc.round(4).to_csv(args.work / "csc_segments_qc.csv", index=False)
    log.info("assembled %d windows from %d archives (%d segments read, %d discarded); %.1f h", X.shape[0], len(zips), len(qc), int((qc["status"] != "ok").sum()), X.shape[0] / 60.0)
    res = T.save_windows(args.dataset, X, meta)
    for w in res["warnings"]:
        log.info("save_windows warning: %s", w)
    rep = validate_saved(args.dataset, labels)
    rep.update(cap=args.cap, seed=args.seed, n_archives=len(zips), opendbc=labels.attrs.get("opendbc"),
               inventory_totals=dict(segments=int(inv["n_members"].sum()), eligible=int(inv["n_eligible"].sum()), read=int(inv["n_processed"].sum()), kept=int(inv["n_kept"].sum()),
                                     rows_hist={k: int(v) for k, v in sorted(Counter(sum((Counter({int(a): b for a, b in json.loads(h).items()}) for h in inv["rows_hist"]), Counter())).items())},
                                     discards=dict(sum((Counter(json.loads(d)) for d in inv["discards"]), Counter()))),
               save_warnings=res["warnings"], build_s=round(time.time() - t0, 1))
    (args.work / "csc_validation.json").write_text(json.dumps(_round(rep), indent=1, default=str))
    (args.work / "csc_tables.md").write_text(tables_md(labels, inv, rep))
    log.info("validate: problems: %s", rep["problems"] or "none")
    log.info("validate: %s", json.dumps(_round({k: rep[k] for k in ("n_windows", "n_vehicles", "n_models", "hours", "per_size_class", "per_powertrain", "per_stratum_windows", "per_stratum_vehicles", "borderline_vehicles", "padded_windows", "trimmed_windows", "sign_check", "v_mps", "das_frac")})))
    if not args.keep_parts:
        for z in zips:
            (parts / f"{z.stem}.npz").unlink(missing_ok=True)
        log.info("removed the per-archive X parts (meta / qc / inventory parts kept; --keep-parts retains them)")
    log.info("build done in %.0f s", time.time() - t0)
    return 0


def cmd_validate(args, log) -> int:
    labels = pd.read_csv(args.work / "csc_platforms.csv") if (args.work / "csc_platforms.csv").exists() else None
    rep = validate_saved(args.dataset, labels)
    (args.work / "csc_validation.json").write_text(json.dumps(_round(rep), indent=1, default=str))
    if (args.work / "csc_inventory.csv").exists() and labels is not None:
        (args.work / "csc_tables.md").write_text(tables_md(labels, pd.read_csv(args.work / "csc_inventory.csv"), rep))
    log.info("validate: problems: %s", rep["problems"] or "none")
    print(json.dumps(_round({k: v for k, v in rep.items() if k != "per_vehicle"}), indent=1, default=str))
    return 0 if not rep["problems"] else 1


def cmd_inventory(args, log) -> int:
    zips = archives(args.data_dir, args.archives)
    rows = []
    for z in zips:
        t0 = time.time()
        with zipfile.ZipFile(z) as zf:
            csvs = [i for i in zf.infolist() if not i.is_dir() and i.filename.lower().endswith(".csv")]
            cnt = Counter()
            for i in csvs:
                b = zf.read(i)
                cnt[b.count(b"\n") - 1 + (0 if b.endswith(b"\n") else 1)] += 1
        rows.append(dict(archive=z.stem, n_members=len(csvs), zip_mb=round(z.stat().st_size / 1e6, 1), rows_hist=json.dumps({int(k): int(v) for k, v in sorted(cnt.items())}), s=round(time.time() - t0, 1)))
        log.info("%-36s %6d members %s", z.stem, len(csvs), rows[-1]["rows_hist"])
    pd.DataFrame(rows).to_csv(args.work / "csc_inventory_only.csv", index=False)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "validate", "labels", "inventory"])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--dataset", default=DATASET, help="window-set name: outputs go to <external>/<dataset>/work (use another name for a smoke run)")
    ap.add_argument("--archives", nargs="*", default=None, help="subset of archive stems (default: every <PLATFORM>.zip except SYNTHETIC*)")
    ap.add_argument("--cap", type=int, default=CAP, help="windows kept per archive (random order, --seed)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=WORKERS, help="worker processes (capped at 6)")
    ap.add_argument("--resume", action="store_true", help="skip archives whose parts exist")
    ap.add_argument("--keep-parts", action="store_true", help="keep the per-archive X parts after the assembly")
    ap.add_argument("--opendbc-root", type=Path, default=None, help="opendbc source tree for MIGRATION (default: ccs_platforms.find_opendbc_root)")
    args = ap.parse_args()
    args.work = T.work_dir(args.dataset)
    log = setup_log(args.work)
    if args.cmd == "build":
        return cmd_build(args, log)
    if args.cmd == "validate":
        return cmd_validate(args, log)
    if args.cmd == "labels":
        cmd_labels(args, log)
        return 0
    return cmd_inventory(args, log)


if __name__ == "__main__":
    sys.exit(main())
