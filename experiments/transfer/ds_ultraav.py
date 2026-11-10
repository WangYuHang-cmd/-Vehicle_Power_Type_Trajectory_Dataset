#!/usr/bin/env python3
"""ULTra-AV adapter for the external zero-shot transfer test (experiments/transfer, window contract in common.py).

Input : the named-vehicle step3 subset of ULTra-AV (CATS Lab, figshare article 26339512 v4) in <external>/ultraav/raw/:
        step3_*.csv with the 14 harmonised car-following columns (Trajectory_ID, Time_Index, ID_LV, Type_LV, Pos_LV,
        Speed_LV, Acc_LV, ID_FAV, Pos_FAV, Speed_FAV, Acc_FAV, Spatial_Gap, Spatial_Headway, Speed_Diff); one gap-free
        episode per (file, Trajectory_ID, ID_FAV).
Output: <external>/ultraav/work/windows.npz + meta.parquet through common.save_windows, plus work/episodes.parquet (one
        row per source episode with its raw-run match), work/build_report.json and work/build_tables.md (the tables of
        DS_ULTRAAV.md).

Channel map (CANDIDATES.md 2.3):
  v_mps            Speed_FAV (measured: GNSS Doppler / OBD / lidar-GNSS depending on the source)
  a_mps2           0.5-s smoothed derivative of Speed_FAV (common.smooth_derivative; derived)
  standstill       v < STANDSTILL_V. Default 0.1 m/s: in the training tensors the car's standstill flag is 1 for 97.5 %
                   of the samples below 0.1 m/s and for 1.5 % / 0 % of the samples at 0.1-0.3 / >= 0.3 m/s, and the ULTra-AV
                   authors removed every row below 0.1 m/s, so the lab-equivalent flag is 0 everywhere; --standstill-v 1.0
                   gives the CANDIDATES wording (v < 1 m/s), which marks slow rolling as standstill (never seen in training).
  das_enabled      per sample from the raw OpenACC run when the episode is matched to it (Driver_k column: ACC -> 1,
                   Human -> 0; else the run header's ACC flag 0 / 1), otherwise the protocol value of the source (1 for ACC
                   followers, 0 for the manually driven Ohio Fusion)
  lead_valid       Spatial_Gap > 0.5 m (lab rule LEAD_VALID_DREL_MIN_M); the six lead channels are NaN where it is 0
  lead_d_m         Spatial_Gap (bumper to bumper, measured)
  lead_v_rel_mps   Speed_Diff = Speed_LV - Speed_FAV (lead minus ego, the training sign)
  lead_speed_mps, lead_a_slope_mps2, thw_s, ttc_s   common.lead_channels (lab definitions)
  steering / yaw / curvature / lateral acceleration / pedal and steering flags / torque   NaN (not recorded): score with the
                   longitudinal-only variant or --mask-channels V2.

Episode -> raw-run matching: the 10 Hz OpenACC-derived files (ZalaZone, AstaZero, Vicolungo, Casale) are matched back to
the raw JRC run files (exact equality of the follower speed sequence). That verifies the README's ID_FAV -> vehicle table
(the Casale ids are inverted there: CSV id 0 is the Hyundai Nexo), gives the per-sample Human / ACC flag or the run-level
ACC header for das_enabled and defines drive_id = one raw run (one contiguous recording). Without the raw folder
(--no-match) the adapter falls back to the protocol flags and drive_id = one ULTra-AV episode.

Usage:
  source env.sh && $CPY experiments/transfer/ds_ultraav.py                      # default build -> external/ultraav/work
  ... --standstill-v 1.0            CANDIDATES wording for the standstill flag
  ... --mhev-as ice                 Audi A4 Avant / Mercedes GLE 450 as ICE (lab convention) instead of HEV (CANDIDATES)
  ... --upsample-5hz                add the 5 Hz MicroSimACC Corolla episodes (linear upsampling, flagged in note)
  ... --include-ambiguous           add the Vicolungo Peugeot (3008 or 5008: footprint unknown, ICE) for the powertrain task
  ... --no-match                    do not use the raw OpenACC files
  ... --dataset NAME                write to <external>/NAME/work instead of ultraav
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.transfer import common as T  # noqa: E402
from vtt import config as C  # noqa: E402

DATASET = "ultraav"
RAW_DEFAULT = T.EXTERNAL_ROOT / "ultraav" / "raw"
OPENACC_RAW_DEFAULT = T.EXTERNAL_ROOT / "openacc" / "raw"
SPECS_CSV = HERE / "specs_external.csv"
STANDSTILL_V_DEFAULT = 0.1          # m/s, the measured meaning of the lab's standstill flag (see module docstring)
ACC_SMOOTH_S_DEFAULT = 0.5          # s, moving-average length of the speed derivative (CANDIDATES.md 2.3: ~0.5 s)
KEY_LEN = 8                         # consecutive speed samples hashed for the raw-run lookup
MATCH_TOL = 1e-6                    # m/s, equality tolerance of the verified sequence match

# ----------------------------------------------------------------------------- sources (one step3 file each)
# dt = nominal sample interval of the file; campaign = raw OpenACC folder used for the episode match (None = no raw data)
SOURCES: dict[str, dict] = {
    "step3_ZalaZone": dict(source="ZalaZone", dt=0.1, campaign="ZalaZone",
                           speed="GNSS (VBOX / u-blox 9) fused and interpolated to 10 Hz with 2nd-degree B-splines"),
    "step3_ASta": dict(source="AstaZero", dt=0.1, campaign="AstaZero",
                       speed="OxTS RT-Range Doppler speed, 100 Hz downsampled to 10 Hz"),
    "step3_Vicolungo": dict(source="Vicolungo", dt=0.1, campaign="Vicolungo",
                            speed="u-blox 8 GNSS Doppler at 3-5 Hz, cubic spline to 10 Hz"),
    "step3_Casale": dict(source="Casale", dt=0.1, campaign="Casale",
                         speed="u-blox 9 GNSS Doppler, cubic spline to 10 Hz"),
    "step3_ACC": dict(source="CATS-ACC", dt=0.1, campaign=None, speed="CATS Lab ACC data set (Tampa), 10 Hz"),
    "step3_UWM": dict(source="CATS-UWM", dt=0.1, campaign=None, speed="CATS UW-Madison data set, 10 Hz"),
    "step3_platoon": dict(source="CATS-platoon", dt=1.0, campaign=None, speed="CATS platoon data set, 1 Hz"),
    "step3_MicroSim_59770073": dict(source="MicroSimACC", dt=0.2, campaign=None, speed="MicroSimACC OBD / GNSS, 5 Hz"),
    "step3_two_vehicle_ACC": dict(source="Vanderbilt", dt=0.1, campaign=None, speed="Vanderbilt two-vehicle ACC, 10 Hz"),
    "step3_single_vehicle": dict(source="Ohio-SV1", dt=0.1, campaign=None, speed="Central Ohio single-vehicle, 10 Hz"),
    "step3_two_vehicle": dict(source="Ohio-SV2", dt=0.1, campaign=None, speed="Central Ohio two-vehicle, 10 Hz"),
}

# ----------------------------------------------------------------------------- labels: (source, ID_FAV) -> car
# model = row name in specs_external.csv (public model name); raw = vehicle name in the raw OpenACC header (verified by
# the sequence match); pt = powertrain label of the window set (BEV / ICE / HEV / PHEV / FCEV / unknown, MHEV resolved by
# --mhev-as); das = protocol value of das_enabled when no raw flag exists; use = how the car enters the window set:
#   "task"       footprint known, BEV or ICE -> both tasks
#   "size"       footprint known, powertrain outside BEV / ICE -> size task only (the scorer reports predicted shares)
#   "ambiguous"  named but the footprint is unresolved -> only with --include-ambiguous (powertrain task only)
#   "5hz"        5 Hz source -> only with --upsample-5hz
#   "exclude"    never (unnamed car, 1 Hz file)
LABELS: list[dict] = [
    # ZalaZone (OpenACC, Oct 2019; README ids = alphabetical raw names; verified by the sequence match)
    dict(source="ZalaZone", id_fav=0, raw="AUDI_A4", model="Audi A4 Avant", pt="MHEV", das=1, use="size",
         note="48-V mild hybrid: HEV in the OpenACC spec file and excluded from the powertrain score by CANDIDATES.md, ICE by the lab convention (--mhev-as)"),
    dict(source="ZalaZone", id_fav=1, raw="AUDI_E_TRON", model="Audi e-tron", pt="BEV", das=1, use="task", note=""),
    dict(source="ZalaZone", id_fav=2, raw="BMW_I3", model="BMW i3s REx", pt="PHEV", das=1, use="size",
         note="range-extended EV (series hybrid): outside the BEV-vs-ICE task"),
    dict(source="ZalaZone", id_fav=3, raw="JAGUAR_I_PACE", model="Jaguar I-Pace", pt="BEV", das=1, use="task", note=""),
    dict(source="ZalaZone", id_fav=4, raw="MAZDA_3", model="Mazda 3", pt="ICE", das=1, use="task",
         note="EU 2.0 Skyactiv-G with a 24-V M Hybrid starter-generator; ICE in the OpenACC spec file and in CANDIDATES.md"),
    dict(source="ZalaZone", id_fav=5, raw="MERCEDES_GLE450", model="Mercedes-Benz GLE 450 4MATIC", pt="MHEV", das=1, use="size",
         note="48-V EQ Boost mild hybrid: HEV in the OpenACC spec file and excluded from the powertrain score by CANDIDATES.md, ICE by the lab convention (--mhev-as)"),
    dict(source="ZalaZone", id_fav=8, raw="TESLA_MODEL3", model="Tesla Model 3", pt="BEV", das=1, use="task", note=""),
    dict(source="ZalaZone", id_fav=9, raw="TESLA_MODELS", model="Tesla Model S", pt="BEV", das=1, use="task", note=""),
    dict(source="ZalaZone", id_fav=10, raw="TESLA_MODELX", model="Tesla Model X", pt="BEV", das=1, use="task", note=""),
    dict(source="ZalaZone", id_fav=11, raw="TOYOTA_RAV4", model="Toyota RAV4 Hybrid", pt="HEV", das=1, use="size",
         note="full hybrid: outside the BEV-vs-ICE task"),
    # AstaZero (OpenACC, Jul 2019); id 1 = Audi A8 appears only as the leader
    dict(source="AstaZero", id_fav=0, raw="Audi(A6)", model="Audi A6", pt="ICE", das=1, use="task",
         note="C8 40 TDI (12-V mild hybrid starter-generator); ICE in the OpenACC spec file and in CANDIDATES.md"),
    dict(source="AstaZero", id_fav=2, raw="BMW(X5)", model="BMW X5", pt="ICE", das=1, use="task", note=""),
    dict(source="AstaZero", id_fav=3, raw="Mercedes(AClass)", model="Mercedes-Benz A-Class", pt="ICE", das=1, use="task", note=""),
    dict(source="AstaZero", id_fav=4, raw="Tesla(Model3)", model="Tesla Model 3", pt="BEV", das=1, use="task", note=""),
    # Vicolungo (OpenACC, Feb 2019); id 2 = Mini and id 4 = Space Star never follow
    dict(source="Vicolungo", id_fav=0, raw="Ford(S-Max)", model="Ford S-Max", pt="ICE", das=1, use="task", note=""),
    dict(source="Vicolungo", id_fav=1, raw="KIA(Niro)", model="Kia Niro PHEV", pt="PHEV", das=1, use="size",
         note="plug-in hybrid: outside the BEV-vs-ICE task"),
    dict(source="Vicolungo", id_fav=3, raw="Mitsubishi(OutlanderPHEV)", model="Mitsubishi Outlander PHEV", pt="PHEV", das=1, use="size",
         note="plug-in hybrid: outside the BEV-vs-ICE task"),
    dict(source="Vicolungo", id_fav=5, raw="Peugeot(3008GTLine)", model="Peugeot 3008", pt="ICE", das=1, use="ambiguous",
         note="3008 in the raw headers, 5008 in the OpenACC spec file: footprint unresolved (4.27 SMALL vs 4.54 MID), diesel either way"),
    dict(source="Vicolungo", id_fav=6, raw="VW(GolfE)", model="Volkswagen e-Golf", pt="BEV", das=1, use="task", note=""),
    # Casale (OpenACC, Oct 2020): the raw order is Rexton (lead) -> Hyundai (follower); the README lists the ids inverted
    dict(source="Casale", id_fav=0, raw="Hyundai", model="Hyundai Nexo", pt="FCEV", das=1, use="size",
         note="fuel-cell EV: outside the BEV-vs-ICE task; README id table inverted (CSV id 0 = Nexo)"),
    # CATS Lab (Lincoln MKZ research cars; powertrain not stated, the drive-by-wire MKZ platforms are hybrids)
    dict(source="CATS-ACC", id_fav=0, raw=None, model="Lincoln MKZ", vehicle="Lincoln MKZ 2016", pt="unknown", das=1, use="size",
         note="2016 black research car; powertrain not stated by the authors (CANDIDATES.md assumes hybrid): outside the BEV-vs-ICE task"),
    dict(source="CATS-ACC", id_fav=1, raw=None, model="Lincoln MKZ", vehicle="Lincoln MKZ 2017", pt="unknown", das=1, use="size",
         note="2017 red research car; powertrain not stated by the authors (CANDIDATES.md assumes hybrid): outside the BEV-vs-ICE task"),
    dict(source="CATS-UWM", id_fav=0, raw=None, model="Lincoln MKZ", vehicle="Lincoln MKZ 2017", pt="unknown", das=1, use="size",
         note="2017 red research car; powertrain not stated: outside the BEV-vs-ICE task"),
    dict(source="CATS-platoon", id_fav=0, raw=None, model="Lincoln MKZ", vehicle="Lincoln MKZ 2016", pt="unknown", das=1, use="exclude", note="1 Hz file"),
    dict(source="CATS-platoon", id_fav=1, raw=None, model="Lincoln MKZ", vehicle="Lincoln MKZ 2017", pt="unknown", das=1, use="exclude", note="1 Hz file"),
    # MicroSimACC (5 Hz)
    dict(source="MicroSimACC", id_fav=0, raw=None, model="Toyota Corolla", pt="ICE", das=1, use="5hz",
         note="Corolla LE 2020 (gasoline) at 5 Hz: linearly upsampled to 10 Hz (--upsample-5hz)"),
    # Vanderbilt: unnamed 2019 SUV
    dict(source="Vanderbilt", id_fav=0, raw=None, model="unnamed 2019 SUV", pt="unknown", das=1, use="exclude", note="make and model not disclosed"),
    # Central Ohio: Tesla sedan of undisclosed model (Model 3 or S: MID or LARGE), Ford Fusion driven manually
    dict(source="Ohio-SV1", id_fav=0, raw=None, model="Tesla sedan (undisclosed)", pt="BEV", das=1, use="exclude",
         note="model not disclosed (footprint unknown) and the automation state per sample is unknown"),
    dict(source="Ohio-SV2", id_fav=0, raw=None, model="Tesla sedan (undisclosed)", pt="BEV", das=1, use="exclude",
         note="model not disclosed (footprint unknown) and the automation state per sample is unknown"),
    dict(source="Ohio-SV2", id_fav=1, raw=None, model="Ford Fusion", pt="unknown", das=0, use="size",
         note="retrofitted Ford Fusion driven manually (das_enabled = 0); hybrid or not is not stated: outside the BEV-vs-ICE task"),
]
LABEL_INDEX = {(d["source"], d["id_fav"]): d for d in LABELS}


# ----------------------------------------------------------------------------- specs
def load_specs(path: Path = SPECS_CSV) -> dict[str, dict]:
    """specs_external.csv -> {model: row} with footprint_m2 (float), size_class, borderline (bool), powertrain, vtt_platform."""
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        fp = pd.to_numeric(r.get("footprint_m2"), errors="coerce")
        out[str(r["model"]).strip()] = dict(
            footprint_m2=float(fp) if np.isfinite(fp) else float("nan"),
            size_class=(str(r["size_class"]).strip() if isinstance(r.get("size_class"), str) else None),
            borderline=str(r.get("borderline", "")).strip().lower() == "true",
            powertrain=(str(r["powertrain"]).strip() if isinstance(r.get("powertrain"), str) else "unknown"),
            platform=(str(r["vtt_platform"]).strip() if isinstance(r.get("vtt_platform"), str) and str(r["vtt_platform"]).strip() else None),
            generation=str(r.get("generation_or_years", "")),
        )
    return out


# ----------------------------------------------------------------------------- raw OpenACC runs (for the match)
def read_openacc_run(path: Path) -> tuple[dict, pd.DataFrame]:
    """A platoon CSV: 5 metadata rows (Date, Vehicle_order, Number_of_vehicles, ACC, Distance_setting) then the table."""
    with open(path, newline="") as f:
        head = [next(f) for _ in range(5)]
    meta: dict[str, list[str]] = {}
    for line in head:
        parts = [p.strip() for p in line.rstrip("\r\n").split(",")]
        meta[parts[0]] = [p for p in parts[1:] if p != ""]
    df = pd.read_csv(path, skiprows=5, low_memory=False)
    return meta, df


def seq_keys(sp: np.ndarray, L: int = KEY_LEN) -> np.ndarray:
    """Rolling polynomial hash of L consecutive speeds rounded to 1e-6 (uint64, wrap-around arithmetic)."""
    q = np.round(np.asarray(sp, float) * 1e6)
    q = np.where(np.isfinite(q), q, -999999999.0).astype(np.int64).astype(np.uint64)
    n = len(q) - L + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(n, dtype=np.uint64)
    mult = np.uint64(1000003)
    with np.errstate(over="ignore"):
        for j in range(L):
            h = h * mult + q[j:j + n]
    return h


class RawIndex:
    """All runs of one raw OpenACC campaign folder with a sorted key index of every vehicle's speed sequence."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.runs: list[dict] = []
        keys, refs = [], []
        files = sorted(p for p in self.folder.glob("*.csv") if "spec" not in p.name.lower() and not p.name.startswith("Long setting"))
        for p in files:
            meta, df = read_openacc_run(p)
            order = meta.get("Vehicle_order", [])
            acc = meta.get("ACC", [])
            run = dict(stem=p.stem, meta=meta, order=order, acc=(acc[0] if acc else ""), n=len(df),
                       time=pd.to_numeric(df["Time"], errors="coerce").to_numpy(float) if "Time" in df.columns else None,
                       speed={}, driver={}, ivs={})
            for col in df.columns:
                if col.startswith("Speed"):
                    run["speed"][int(col[5:])] = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
                elif col.startswith("Driver"):
                    run["driver"][int(col[6:])] = df[col].astype(object).where(df[col].notna(), None).to_numpy(object)
                elif col.startswith("IVS"):
                    run["ivs"][int(col[3:])] = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            ri = len(self.runs)
            self.runs.append(run)
            for k, sp in run["speed"].items():
                h = seq_keys(sp)
                if len(h):
                    keys.append(h)
                    refs.append(np.stack([np.full(len(h), ri, np.int64), np.full(len(h), k, np.int64), np.arange(len(h), dtype=np.int64)], 1))
        if not keys:
            raise RuntimeError(f"no raw runs with speed columns under {folder}")
        self.keys = np.concatenate(keys)
        self.ref = np.concatenate(refs)
        o = np.argsort(self.keys, kind="stable")
        self.keys, self.ref = self.keys[o], self.ref[o]

    def _cands(self, key: np.uint64) -> np.ndarray:
        a = np.searchsorted(self.keys, key, side="left")
        b = np.searchsorted(self.keys, key, side="right")
        return self.ref[a:b]

    def locate(self, sp: np.ndarray) -> tuple[tuple[int, int, int] | None, str]:
        """(run index, vehicle column k, row offset) of the raw stretch equal to the speed sequence sp, and a status."""
        sp = np.asarray(sp, float)
        n = len(sp)
        if n < KEY_LEN:
            return None, "short"
        cands: set | None = None
        for off in sorted({0, n // 3, (2 * n) // 3, n - KEY_LEN}):
            h = seq_keys(sp[off:off + KEY_LEN])[0]
            c = {(int(r), int(k), int(i) - off) for r, k, i in self._cands(h)}
            cands = c if cands is None else (cands & c)
            if not cands:
                return None, "nomatch"
        verified = []
        for ri, k, i0 in sorted(cands):
            if i0 < 0:
                continue
            raw = self.runs[ri]["speed"][k]
            if i0 + n > len(raw):
                continue
            seg = raw[i0:i0 + n]
            ok = np.isfinite(seg)
            if ok.sum() < 0.9 * n:
                continue
            if np.max(np.abs(seg[ok] - sp[ok])) <= MATCH_TOL:
                verified.append((ri, k, i0))
        if len(verified) == 1:
            return verified[0], "ok"
        return None, ("unverified" if not verified else "ambiguous")


def das_from_run(run: dict, k: int, i0: int, n: int) -> tuple[np.ndarray, str]:
    """das_enabled samples of vehicle k over rows i0..i0+n of a raw run: Driver_k (ACC -> 1, Human -> 0) with the run's ACC
    header (0 / 1) filling samples without a flag; returns (values, how) where how names the rule used."""
    acc = str(run.get("acc", "")).strip()
    run_flag = {"0": 0.0, "1": 1.0}.get(acc, np.nan)
    out = np.full(n, np.nan)
    how = "none"
    if k in run["driver"]:
        d = run["driver"][k][i0:i0 + n]
        s = np.array([str(x).strip().upper() if x is not None else "" for x in d], dtype=object)
        out[s == "ACC"] = 1.0
        out[s == "HUMAN"] = 0.0
        how = "driver"
        unknown = ~np.isin(s, ("ACC", "HUMAN"))
        if unknown.any() and np.isfinite(run_flag):
            out[unknown] = run_flag
            how = "driver+header"
    elif np.isfinite(run_flag):
        out[:] = run_flag
        how = "header"
    return out, how


# ----------------------------------------------------------------------------- episodes
def episodes_of(df: pd.DataFrame):
    """Yield (Trajectory_ID, ID_FAV, frame) for every run of constant (Trajectory_ID, ID_FAV) in file order (one ZalaZone
    trajectory holds two followers in sequence)."""
    brk = (df["Trajectory_ID"].diff().fillna(0) != 0) | (df["ID_FAV"].diff().fillna(0) != 0)
    ep = brk.cumsum().to_numpy()
    for _, e in df.groupby(ep, sort=False):
        yield int(e["Trajectory_ID"].iloc[0]), int(e["ID_FAV"].iloc[0]), e


def resolve_powertrain(pt: str, mhev_as: str) -> str:
    if pt == "MHEV":
        return "HEV" if mhev_as == "hev" else "ICE"
    return T.normalise_powertrain(pt)


def build(raw_dir: Path, openacc_raw: Path | None, dataset: str, *, standstill_v: float, acc_smooth_s: float, mhev_as: str,
          upsample_5hz: bool, include_ambiguous: bool, align: str = "run", root: Path | None = None) -> dict:
    t_start = time.time()
    specs = load_specs()
    indexes: dict[str, RawIndex] = {}
    results: list[T.CutResult] = []
    ep_rows: list[dict] = []
    per_source: dict[str, dict] = {}
    raw_name_mismatch: list[str] = []
    acc_noise: dict[str, list[float]] = {}         # std of the first difference of a_mps2 per source (moving samples)
    acc_vs_src: dict[str, list[float]] = {}        # RMSE between a_mps2 and the 0.5-s smoothed Acc_FAV

    for stem, src in SOURCES.items():
        path = raw_dir / f"{stem}.csv"
        if not path.exists():
            print(f"[ultraav] missing {path.name}: skipped")
            continue
        name = src["source"]
        st = per_source.setdefault(name, dict(episodes=0, rows=0, used_episodes=0, windows=0, candidates=0, dropped_gap=0,
                                              dropped_stationary=0, matched=0, match_status={}, das_rule={}, skipped_use={},
                                              upsampled=0, lead_mismatch=0, gap_mismatch=0, lead_invalid_samples=0, samples=0))
        df = pd.read_csv(path)
        st["rows"] += int(len(df))
        if src["dt"] >= 0.9:                       # 1 Hz CATS platoon: never usable
            st["skipped_use"]["1hz"] = st["skipped_use"].get("1hz", 0) + df["Trajectory_ID"].nunique()
            st["episodes"] += int(df["Trajectory_ID"].nunique())
            print(f"[ultraav] {name}: {src['dt']} s sampling, skipped")
            continue
        idx = None
        if openacc_raw is not None and src["campaign"]:
            if name not in indexes:
                t0 = time.time()
                indexes[name] = RawIndex(openacc_raw / src["campaign"])
                print(f"[ultraav] indexed {len(indexes[name].runs)} raw {src['campaign']} runs, {len(indexes[name].keys):,} keys ({time.time() - t0:.1f} s)")
            idx = indexes[name]
        for tid, fav, e in episodes_of(df):
            st["episodes"] += 1
            lab = LABEL_INDEX.get((name, fav))
            if lab is None:
                st["skipped_use"]["unlabelled"] = st["skipped_use"].get("unlabelled", 0) + 1
                print(f"[ultraav] WARNING {name}: ID_FAV {fav} has no label row, skipped")
                continue
            use = lab["use"]
            if use == "exclude" or (use == "ambiguous" and not include_ambiguous) or (use == "5hz" and not upsample_5hz):
                st["skipped_use"][use] = st["skipped_use"].get(use, 0) + 1
                continue
            t = e["Time_Index"].to_numpy(float)
            v = e["Speed_FAV"].to_numpy(float)
            gap = e["Spatial_Gap"].to_numpy(float)
            vr = e["Speed_Diff"].to_numpy(float)
            v_lead_src = e["Speed_LV"].to_numpy(float)
            a_src = e["Acc_FAV"].to_numpy(float)
            # ---- das_enabled and drive id from the raw run
            das = np.full(len(t), float(lab["das"]))
            das_rule = "protocol"
            run_stem, k_col, i0, status = None, None, None, "no-raw"
            if idx is not None:
                hit, status = idx.locate(v)
                if hit is not None:
                    ri, k_col, i0 = hit
                    run = idx.runs[ri]
                    run_stem = run["stem"]
                    st["matched"] += 1
                    raw_name = run["order"][k_col - 1] if k_col - 1 < len(run["order"]) else f"veh{k_col}"
                    if lab["raw"] is not None and raw_name != lab["raw"]:
                        raw_name_mismatch.append(f"{name} ID_FAV {fav}: raw {raw_name!r} vs label {lab['raw']!r}")
                    d_raw, how = das_from_run(run, k_col, i0, len(t))
                    if np.isfinite(d_raw).all():
                        das = d_raw
                        das_rule = f"raw-{how}"
                    elif np.isfinite(d_raw).any():
                        das = np.where(np.isfinite(d_raw), d_raw, float(lab["das"]))
                        das_rule = f"raw-{how}+protocol"
                    # cross-checks of the leader columns (information only)
                    if (k_col - 1) in run["speed"]:
                        lv = run["speed"][k_col - 1][i0:i0 + len(t)]
                        ok = np.isfinite(lv)
                        if ok.any() and np.max(np.abs(lv[ok] - v_lead_src[ok])) > 1e-6:
                            st["lead_mismatch"] += 1
                    if (k_col - 1) in run["ivs"]:
                        iv = run["ivs"][k_col - 1][i0:i0 + len(t)]
                        ok = np.isfinite(iv)
                        if ok.any() and np.max(np.abs(iv[ok] - gap[ok])) > 1e-6:
                            st["gap_mismatch"] += 1
                st["match_status"][status] = st["match_status"].get(status, 0) + 1
            st["das_rule"][das_rule] = st["das_rule"].get(das_rule, 0) + 1
            # ---- 10 Hz series
            upsampled = False
            if src["dt"] > 0.15:                   # 5 Hz MicroSim -> linear upsampling onto the 10 Hz grid
                grid = T.make_grid(t[0], t[-1])
                _, v10 = T.resample_to_10hz(t, v, "linear", t_grid=grid)
                _, gap10 = T.resample_to_10hz(t, gap, "linear", t_grid=grid)
                _, vr10 = T.resample_to_10hz(t, vr, "linear", t_grid=grid)
                _, das10 = T.resample_to_10hz(t, das, "hold", t_grid=grid)
                _, a_src10 = T.resample_to_10hz(t, a_src, "linear", t_grid=grid)
                t, v, gap, vr, das, a_src = grid, v10, gap10, vr10, das10, a_src10
                upsampled = True
                st["upsampled"] += 1
            a = T.smooth_derivative(v, win_s=acc_smooth_s)
            standstill = (v < standstill_v).astype(float)
            lead_valid = np.isfinite(gap) & np.isfinite(vr) & (gap > float(C.LEAD_VALID_DREL_MIN_M))
            st["lead_invalid_samples"] += int((~lead_valid).sum())
            st["samples"] += int(len(t))
            lead = T.lead_channels(v, lead_valid, gap, vr, t)
            # acceleration diagnostics (moving samples)
            mv = v >= T.MOVING_V_MPS
            if mv.sum() > 20:
                acc_noise.setdefault(name, []).append(float(np.nanstd(np.diff(a)[mv[1:] & mv[:-1]])))
                k_ = max(1, int(round(acc_smooth_s * T.HZ)))
                a_src_sm = np.convolve(np.pad(np.nan_to_num(a_src), (k_ // 2, k_ - 1 - k_ // 2), mode="edge"), np.ones(k_) / k_, mode="valid")
                acc_vs_src.setdefault(name, []).append(float(np.sqrt(np.nanmean((a[mv] - a_src_sm[mv]) ** 2))))
            # ---- labels and meta constants
            sp = specs.get(lab["model"], {})
            fp = sp.get("footprint_m2", float("nan")) if use != "ambiguous" else float("nan")
            pt = resolve_powertrain(lab["pt"], mhev_as)
            vehicle = f"{name} {lab.get('vehicle', lab['model'])}"
            drive_id = f"{name}/{run_stem}" if run_stem else f"{name}/traj{tid:04d}"
            note_parts = [f"das: {das_rule}", f"speed: {src['speed']}", f"a_mps2: {acc_smooth_s:.1f}-s smoothed speed derivative",
                          f"standstill: v < {standstill_v:g} m/s"]
            if upsampled:
                note_parts.append("UPSAMPLED from 5 Hz (linear)")
            if lab["note"]:
                note_parts.append(lab["note"])
            if lab["pt"] == "MHEV":
                note_parts.append(f"mild hybrid labelled {pt} (--mhev-as {mhev_as})")
            info = dict(dataset=dataset, vehicle_id=vehicle, model=lab["model"], platform=sp.get("platform"), drive_id=drive_id,
                        footprint_m2=fp, size_class=(sp.get("size_class") if use != "ambiguous" else None), powertrain=pt,
                        borderline=bool(sp.get("borderline", False)) if use != "ambiguous" else False, note="; ".join(note_parts))
            df10 = pd.DataFrame(dict(t_s=t))
            cmap = {"v_mps": v, "a_mps2": a, "standstill": standstill, "das_enabled": das, **lead}
            res = T.cut_windows(df10, cmap, t_col="t_s", info=info, align=align)
            st["used_episodes"] += 1
            st["candidates"] += res.stats["n_candidates"]
            st["dropped_gap"] += res.stats["n_dropped_gap"]
            st["dropped_stationary"] += res.stats["n_dropped_stationary"]
            st["windows"] += res.stats["n_windows"]
            results.append(res)
            ep_rows.append(dict(source=name, trajectory_id=tid, id_fav=fav, model=lab["model"], vehicle_id=vehicle, use=use, n_rows=int(len(e)),
                                duration_s=float(t[-1] - t[0]), raw_run=run_stem, raw_vehicle_col=k_col, raw_row0=i0, match=status,
                                das_rule=das_rule, das_frac=float(np.mean(das)), v_min=float(np.nanmin(v)), v_max=float(np.nanmax(v)),
                                gap_min=float(np.nanmin(gap)), gap_max=float(np.nanmax(gap)), lead_invalid=int((~lead_valid).sum()),
                                upsampled=upsampled, n_windows=res.stats["n_windows"], drive_id=drive_id))
        print(f"[ultraav] {name}: episodes {st['episodes']} used {st['used_episodes']} windows {st['windows']} "
              f"(candidates {st['candidates']}, dropped gap {st['dropped_gap']}, stationary {st['dropped_stationary']}); "
              f"match {st['match_status']}; das {st['das_rule']}; skipped {st['skipped_use']}")

    if raw_name_mismatch:
        raise RuntimeError("raw vehicle name differs from the label table:\n  " + "\n  ".join(raw_name_mismatch[:20]))
    X, meta = T.concat_results(results)
    if len(meta):
        meta = meta.reset_index(drop=True)
        meta["w_idx"] = meta.groupby("drive_id", sort=False).cumcount().astype(np.int32)   # one counter per drive
    out = T.save_windows(dataset, X, meta, root=root)
    wd = T.work_dir(dataset, root)
    eps = pd.DataFrame(ep_rows)
    eps.to_parquet(wd / "episodes.parquet", index=False)
    report = dict(dataset=dataset, built=time.strftime("%Y-%m-%d %H:%M"), seconds=round(time.time() - t_start, 1),
                  options=dict(standstill_v=standstill_v, acc_smooth_s=acc_smooth_s, mhev_as=mhev_as, upsample_5hz=upsample_5hz,
                               include_ambiguous=include_ambiguous, align=align, raw_match=openacc_raw is not None),
                  per_source=per_source, n_windows=int(len(meta)),
                  acc_noise_std_diff=({k: round(float(np.median(v)), 4) for k, v in acc_noise.items()}),
                  acc_rmse_vs_smoothed_Acc_FAV=({k: round(float(np.median(v)), 4) for k, v in acc_vs_src.items()}),
                  save_warnings=out["warnings"])
    (wd / "build_report.json").write_text(json.dumps(report, indent=1, default=str))
    return dict(X=X, meta=meta, episodes=eps, report=report, work_dir=wd)


# ----------------------------------------------------------------------------- validation
RANGES = {"v_mps": (0.0, 60.0), "a_mps2": (-8.0, 8.0), "lead_d_m": (0.5, 200.0), "lead_v_rel_mps": (-30.0, 30.0),
          "lead_speed_mps": (-5.0, 60.0), "lead_a_slope_mps2": (-4.0, 4.0), "thw_s": (0.0, np.inf), "ttc_s": (0.0, np.inf)}


def validate(X: np.ndarray, meta: pd.DataFrame, episodes: pd.DataFrame, report: dict) -> tuple[list[str], dict]:
    """Checks of the task: NaN inside declared channels, physical ranges, per vehicle / model / class counts, speeds, moving
    share. Returns (problems, tables)."""
    problems: list[str] = []
    ch = T.CH
    v = X[:, :, ch["v_mps"]]
    lv = X[:, :, ch["lead_valid"]]
    for c in ("v_mps", "a_mps2", "standstill", "das_enabled", "lead_valid"):
        n_nan = int(np.isnan(X[:, :, ch[c]]).sum())
        if n_nan:
            problems.append(f"{c}: {n_nan} NaN samples")
    valid = lv == 1
    for c in ("lead_d_m", "lead_v_rel_mps", "lead_speed_mps"):
        col = X[:, :, ch[c]]
        n1 = int(np.isnan(col[valid]).sum()); n0 = int(np.isfinite(col[~valid]).sum())
        if n1:
            problems.append(f"{c}: {n1} NaN samples while lead_valid = 1")
        if n0:
            problems.append(f"{c}: {n0} finite samples while lead_valid = 0")
    # conditional lead channels
    d = X[:, :, ch["lead_d_m"]]; vr = X[:, :, ch["lead_v_rel_mps"]]
    thw = X[:, :, ch["thw_s"]]; ttc = X[:, :, ch["ttc_s"]]; slope = X[:, :, ch["lead_a_slope_mps2"]]
    # the rules were applied in float64 at build time; X is float32, so samples within 1e-5 of a threshold may fall either way
    eps = 1e-5
    v64, d64, vr64 = v.astype(np.float64), d.astype(np.float64), vr.astype(np.float64)
    must_thw = valid & (v64 > C.THW_V_MIN_MPS + eps) & (d64 > C.LEAD_VALID_DREL_MIN_M + eps)
    no_thw = ~valid | (v64 < C.THW_V_MIN_MPS - eps) | (d64 < C.LEAD_VALID_DREL_MIN_M - eps)
    n_bad = int((must_thw & ~np.isfinite(thw)).sum() + (no_thw & np.isfinite(thw)).sum())
    if n_bad:
        problems.append(f"thw_s: {n_bad} samples disagree with the thw rule")
    must_ttc = valid & (-vr64 > C.TTC_CLOSING_MIN_MPS + eps)
    no_ttc = ~valid | (-vr64 < C.TTC_CLOSING_MIN_MPS - eps)
    n_bad = int((must_ttc & ~np.isfinite(ttc)).sum() + (no_ttc & np.isfinite(ttc)).sum())
    if n_bad:
        problems.append(f"ttc_s: {n_bad} samples disagree with the ttc rule")
    slope_nan_valid = float(np.isnan(slope[valid]).mean()) if valid.any() else float("nan")
    for c, (lo, hi) in RANGES.items():
        col = X[:, :, ch[c]]
        fin = col[np.isfinite(col)]
        if fin.size and (fin.min() < lo or fin.max() > hi):
            problems.append(f"{c}: values outside [{lo}, {hi}] (min {fin.min():.3f}, max {fin.max():.3f})")
    for c in T.CHANNELS:
        if c not in ("v_mps", "a_mps2", "standstill", "das_enabled") + T.LEAD_CHANNELS:
            if np.isfinite(X[:, :, ch[c]]).any():
                problems.append(f"{c}: finite values although the dataset does not record it")
    # physical sanity: lead_speed = v + v_rel
    ls = X[:, :, ch["lead_speed_mps"]]
    m = valid & np.isfinite(ls)
    if m.any() and np.max(np.abs(ls[m] - (v[m] + vr[m]))) > 1e-3:
        problems.append("lead_speed_mps differs from v + v_rel")
    mf = (v >= T.MOVING_V_MPS).mean(1)
    if (mf < T.MIN_MOVING_FRAC).any():
        problems.append("windows below the moving rule present")
    bad_bin = [c for c in T.BINARY_CHANNELS if not np.isin(X[:, :, ch[c]][np.isfinite(X[:, :, ch[c]])], (0.0, 1.0)).all()]
    if bad_bin:
        problems.append(f"non-binary flags {bad_bin}")

    # ---- tables
    meta = meta.copy()
    meta["source"] = meta["drive_id"].str.split("/").str[0]
    meta["v_mean"] = np.nanmean(v, 1); meta["v_max"] = np.nanmax(v, 1)
    meta["a_std"] = np.nanstd(X[:, :, ch["a_mps2"]], 1)
    meta["lead_frac"] = np.nanmean(lv, 1)
    meta["thw_med"] = np.nanmedian(np.where(np.isfinite(thw), thw, np.nan), 1)
    g = meta.groupby(["source", "model"], sort=True)
    per_model = pd.DataFrame({
        "windows": g.size(), "vehicles": g["vehicle_id"].nunique(), "drives": g["drive_id"].nunique(),
        "hours": (g.size() / 60.0).round(2), "footprint_m2": g["footprint_m2"].first(), "size_class": g["size_class"].first(),
        "borderline": g["borderline"].first(), "powertrain": g["powertrain"].first(),
        "das_frac": g["das_frac"].mean().round(3), "moving_frac": g["moving_frac"].mean().round(3),
        "v_mean": g["v_mean"].mean().round(2), "v_max": g["v_max"].max().round(2), "a_std": g["a_std"].mean().round(3),
        "lead_frac": g["lead_frac"].mean().round(3), "thw_med": g["thw_med"].median().round(2),
    }).reset_index()
    per_source = meta.groupby("source").agg(windows=("w_idx", "size"), models=("model", "nunique"), vehicles=("vehicle_id", "nunique"),
                                            drives=("drive_id", "nunique"), das_frac=("das_frac", "mean"),
                                            das_windows_0=("das_frac", lambda s: int((s == 0).sum())),
                                            das_windows_mixed=("das_frac", lambda s: int(((s > 0) & (s < 1)).sum())),
                                            v_mean=("v_mean", "mean")).round(3).reset_index()
    task_rows = []
    for task, (col, classes) in T.TASK_LABEL.items():
        lab = meta[col].fillna("unknown")
        for cls in list(classes) + ["(outside task)"]:
            sel = lab.isin(classes) if cls == "(outside task)" else (lab == cls)
            sel = ~sel if cls == "(outside task)" else sel
            task_rows.append(dict(task=task, **{"class": cls}, windows=int(sel.sum()), models=int(meta.loc[sel, "model"].nunique()),
                                  vehicles=int(meta.loc[sel, "vehicle_id"].nunique()), drives=int(meta.loc[sel, "drive_id"].nunique())))
    per_class = pd.DataFrame(task_rows)
    size_nb = meta[~meta["borderline"].astype(bool)]
    tables = dict(per_model=per_model, per_source=per_source, per_class=per_class,
                  summary=dict(n_windows=int(len(meta)), n_models=int(meta["model"].nunique()), n_vehicles=int(meta["vehicle_id"].nunique()),
                               n_drives=int(meta["drive_id"].nunique()), hours=round(len(meta) / 60.0, 2),
                               v_min=float(np.nanmin(v)), v_max=float(np.nanmax(v)), v_mean=float(np.nanmean(v)),
                               moving_share=float((v >= T.MOVING_V_MPS).mean()), standstill_share=float(np.nanmean(X[:, :, ch["standstill"]])),
                               das_share=float(np.nanmean(X[:, :, ch["das_enabled"]])), lead_valid_share=float(np.nanmean(lv)),
                               slope_nan_share_within_lead=slope_nan_valid, a_abs_p99=float(np.nanquantile(np.abs(X[:, :, ch["a_mps2"]]), 0.99)),
                               a_abs_max=float(np.nanmax(np.abs(X[:, :, ch["a_mps2"]]))), thw_med=float(np.nanmedian(thw)),
                               ttc_med=float(np.nanmedian(ttc)), gap_med=float(np.nanmedian(d)),
                               size_windows_without_borderline=int(size_nb["size_class"].notna().sum()),
                               windows_das0=int((meta["das_frac"] == 0).sum()), windows_das_mixed=int(((meta["das_frac"] > 0) & (meta["das_frac"] < 1)).sum())))
    return problems, tables


def write_tables_md(path: Path, tables: dict, report: dict, problems: list[str]) -> None:
    s = tables["summary"]
    lines = [f"# ULTra-AV build tables ({report['built']}; options {json.dumps(report['options'])})", ""]
    lines += ["## Summary", "", T.md_table(["quantity", "value"], [[k, (round(v, 4) if isinstance(v, float) else v)] for k, v in s.items()]), ""]
    lines += ["## Per source", "", T.md_table(list(tables["per_source"].columns), tables["per_source"].values.tolist()), ""]
    lines += ["## Per class", "", T.md_table(list(tables["per_class"].columns), tables["per_class"].values.tolist()), ""]
    pm = tables["per_model"]
    lines += ["## Per model", "", T.md_table(list(pm.columns), pm.values.tolist()), ""]
    rows = []
    for name, st in report["per_source"].items():
        rows.append([name, st["episodes"], st["used_episodes"], st["windows"], st["candidates"], st["dropped_gap"], st["dropped_stationary"],
                     json.dumps(st["match_status"]), json.dumps(st["das_rule"]), json.dumps(st["skipped_use"]), st["lead_mismatch"], st["gap_mismatch"],
                     f"{st['lead_invalid_samples']}/{st['samples']}", report["acc_noise_std_diff"].get(name), report["acc_rmse_vs_smoothed_Acc_FAV"].get(name)])
    lines += ["## Per source build statistics", "", T.md_table(["source", "episodes", "used", "windows", "candidates", "dropped gap", "dropped stationary",
                                                               "raw match", "das rule", "skipped (use)", "lead speed mismatch", "gap mismatch",
                                                               "lead_valid=0 samples", "std(diff a) moving", "RMSE a vs 0.5-s Acc_FAV"], rows), ""]
    lines += ["## Validation", "", ("no problems" if not problems else "\n".join(f"* {p}" for p in problems)), ""]
    path.write_text("\n".join(lines))


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_DEFAULT, help="folder with the step3_*.csv files")
    ap.add_argument("--openacc-raw", type=Path, default=OPENACC_RAW_DEFAULT, help="raw OpenACC campaign folders (episode match)")
    ap.add_argument("--no-match", action="store_true", help="do not match episodes to the raw OpenACC runs")
    ap.add_argument("--dataset", default=DATASET, help="window set name (output folder <external>/<name>/work)")
    ap.add_argument("--root", type=Path, default=None, help=f"external root (default {T.EXTERNAL_ROOT})")
    ap.add_argument("--standstill-v", type=float, default=STANDSTILL_V_DEFAULT, help="standstill = v below this speed (m/s)")
    ap.add_argument("--acc-smooth-s", type=float, default=ACC_SMOOTH_S_DEFAULT, help="moving-average length of the speed derivative (s)")
    ap.add_argument("--mhev-as", choices=["hev", "ice"], default="hev", help="label of the 48-V mild hybrids A4 Avant / GLE 450")
    ap.add_argument("--upsample-5hz", action="store_true", help="include the 5 Hz MicroSimACC Corolla (linear upsampling)")
    ap.add_argument("--include-ambiguous", action="store_true", help="include the Vicolungo Peugeot 3008/5008 (footprint unknown, ICE)")
    ap.add_argument("--align", choices=["run", "drive"], default="run")
    ap.add_argument("--validate-only", action="store_true", help="re-run the checks and tables on the saved window set")
    args = ap.parse_args(argv)
    if args.validate_only:
        wd = T.work_dir(args.dataset, args.root)
        X, meta = T.load_windows(args.dataset, root=args.root)
        eps_df = pd.read_parquet(wd / "episodes.parquet")
        report = json.loads((wd / "build_report.json").read_text())
        problems, tables = validate(X, meta, eps_df, report)
        write_tables_md(wd / "build_tables.md", tables, report, problems)
        print(json.dumps(tables["summary"], indent=1, default=str))
        print("[ultraav] validation:", "no problems" if not problems else problems)
        return 1 if problems else 0

    openacc = None if args.no_match else args.openacc_raw
    if openacc is not None and not openacc.exists():
        print(f"[ultraav] raw OpenACC folder {openacc} not found: protocol flags only")
        openacc = None
    res = build(args.raw, openacc, args.dataset, standstill_v=args.standstill_v, acc_smooth_s=args.acc_smooth_s, mhev_as=args.mhev_as,
                upsample_5hz=args.upsample_5hz, include_ambiguous=args.include_ambiguous, align=args.align, root=args.root)
    X, meta = T.load_windows(args.dataset, root=args.root)                       # re-read: the saved files are what the scorer sees
    problems, tables = validate(X, meta, res["episodes"], res["report"])
    write_tables_md(res["work_dir"] / "build_tables.md", tables, res["report"], problems)
    s = tables["summary"]
    print(f"[ultraav] saved {s['n_windows']} windows, {s['n_models']} models, {s['n_vehicles']} vehicles, {s['n_drives']} drives, {s['hours']} h; "
          f"v {s['v_min']:.2f}-{s['v_max']:.2f} m/s (mean {s['v_mean']:.2f}), moving share {s['moving_share']:.3f}, das share {s['das_share']:.3f}, "
          f"lead_valid share {s['lead_valid_share']:.3f}")
    print(tables["per_class"].to_string(index=False))
    print(tables["per_source"].to_string(index=False))
    if problems:
        print("[ultraav] VALIDATION PROBLEMS:")
        for p in problems:
            print("  *", p)
        return 1
    print("[ultraav] validation: no problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
