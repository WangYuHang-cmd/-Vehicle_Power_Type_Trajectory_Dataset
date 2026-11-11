#!/usr/bin/env python3
"""Understeer gradient of the steering-derived yaw, fitted on the lab's released drive tables (IMU / pose yaw = truth).

Why. The commaCarSegments decoder (ccs_decode.py) replaces the lab's IMU / pose yaw by the kinematic bicycle model
yaw_b = v tan(delta_road) / L (delta_road = steering wheel angle / steerRatio, L = wheelbase). That model has no
understeer: on comma2k19 the ratio yaw_b / yawRate grows like 1 + k v^2 (CCS_PIPELINE.md, gap 9). The linear
single-track steady state is yaw = v tan(delta_road) / (L (1 + k v^2)) with k = K_us / L (K_us the understeer gradient
in rad per m/s^2), so one k per vehicle removes the bias. This script fits k per lab platform and reduces it to ONE global
k (no platform-specific lab calibration may leak into the external features), to be used by `--yaw derived-understeer`.

Sample rule (per 10 Hz row of a drive table): v >= 2 m/s, |delta_road| >= 0.3 deg, yaw_valid, not standstill,
|yaw_meas| < 1.5 rad/s, finite. Per platform (>= 2 vehicles with usable samples, >= 1 h of such samples):
  * one-parameter fit  yaw_b = (1 + k1 v^2) yaw_meas            (the decoder's form; k1 is what the global k is made of),
  * two-parameter fit  yaw_b = a (1 + k v^2) yaw_meas           (linear in (a, a k); a absorbs a steerRatio / wheelbase error),
  both least squares with one robust pass (residual outliers beyond 3 scaled MADs dropped, refit), R^2 of the final fit;
  * binned median ratio yaw_b / yaw_meas per speed bin (samples with |yaw_meas| >= 0.01 rad/s so the ratio is defined),
    and a cross-check k_bin from the bin medians (ratio - 1 ~ k v^2, weighted by samples).
Global k = hours-weighted median of k1 over the train / validation platforms (also: plain median, interquartile range,
hours-weighted mean; the same for the two-parameter k; and whether a sits away from 1). Split status per platform = the
modal split_platform of its drives (train / validation / held-out test / excluded).

Steering-angle offset (choice, --angle-variant): the lab tables carry the raw carState angle and several platforms sit
on a constant offset of a few degrees, which a no-intercept fit cannot absorb and which the decoder does not remove on the
external sets either. Both variants are fitted from one pass: 'offset' subtracts, per drive, the median raw angle during
straight driving (v >= 5 m/s, yaw_valid, |yaw_meas| < 0.02 rad/s, at least 300 such samples, else the drive is skipped);
'raw' keeps the angle as logged (what the decoder sees). The primary tables use 'offset'; the 'raw' numbers are reported
as the sensitivity. Drives whose yaw / steering correlation (v >= 2 m/s) is below 0.5 are skipped; a negative correlation
would mean a sign error of one of the two signals and is counted (none is expected: both are left-positive).
Lateral-controller samples: the fit is also run on the human-only subset (op_lat_active false) and on the controller-on
subset; the primary numbers use all samples (the external features are decoded from all samples too).

steerRatio: opendbc 0.3.1 CarSpecs parsed from the decode venv's source tree (vtt.specs AST parser, no import; the
same values the decoder takes from the logged / constructed carParams), else the 2024-08 opendbc checkout, else the
base platform of the lab name (suffix tokens such as _GEN1 / _2022_2023 / _PE stripped until a 0.3.1 name matches;
column sr_src says which). L: the 0.3.1 wheelbase when the platform is in 0.3.1 (the decoder's value), else the lab
spec (drives.parquet wheelbase_mm). A steerRatio error moves a and k1, not the two-parameter k.

comma2k19 validation (--c2k-npz-dir: the verifier's per-segment logged carState dumps, array `cs` with columns t, vEgo,
aEgo, steeringAngleDeg, ..., yawRate at index 14, and `ctrl` with t, controlsState.enabled; made with the lab LogReader):
corrected derived yaw with the global k against the logged yawRate, binned median ratio per speed bin, on 10 evenly
spaced segments per chunk and on all segments (plus the human-only samples of all segments); also with the two cars' own
lab k1, and with the (a, k) that 2018 openpilot's own single-track vehicle model implies for the two cars (the logged
2018 `carState.yawRate` is that model's output, not a sensor; the test 'yaw_is_model' measures how exactly the logged
value is reproduced by a v sa / (1 + k v^2) with the fitted (a, k)).

Usage (under $CPY, from the repository root, `source env.sh` first):
  nice -n 10 $CPY experiments/transfer/understeer_fit.py --workers 6 --out-dir $D/tmp/understeer \
      --json experiments/transfer/understeer_fit.json
Outputs: <out-dir>/platform_table.csv, platform_table.md, c2k_table.md, fit.json; the --json copy is the same JSON
(numbers and platform names only: no device / route identifiers, no machine paths).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vtt import config as C  # noqa: E402
from vtt import specs as S  # noqa: E402
from experiments.transfer import ccs_platforms as P  # noqa: E402

HZ = 10
V_MIN = 2.0                 # m/s
DELTA_MIN_DEG = 0.3         # road-wheel angle
YAW_MAX = 1.5               # rad/s
YAW_RATIO_MIN = 0.01        # rad/s, bin medians only
STRAIGHT_YAW = 0.02         # rad/s: straight driving for the steering-offset estimate
STRAIGHT_V = 5.0            # m/s
STRAIGHT_MIN_N = 300
CORR_MIN = 0.5              # drive-level corr(steering angle, yaw_meas) gate
MAD_K = 3.0
MIN_VEHICLES = 2
MIN_HOURS = 1.0
MIN_SAMPLES = 1000
BINS = [(1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40)]
SUBSETS = ("all", "human", "das")
VARIANTS = ("offset", "raw")
C2K = {"Chunk_1": dict(platform="TOYOTA_RAV4", steer_ratio=16.88, wheelbase_m=2.65),
       "Chunk_3": dict(platform="HONDA_CIVIC", steer_ratio=15.38, wheelbase_m=2.70)}     # the constructed carParams (CCS_PIPELINE.md)
# 2018 openpilot (0.5.x) CarParams of the two cars, as the interface built them: the logged carState.yawRate of comma2k19 is
# VehicleModel.yaw_rate(sa, v) = (1 - chi) v (sa / sR) / (L (1 - sf v^2)), sf = m (cF aF - cR aR) / (L^2 cF cR) (< 0),
# cF / cR scaled from the Civic's 192150 / 202500 N/rad by tire_stiffness_factor, mass and axle-load share.
LB = 0.45359237
STD_CARGO = 136.0
C2K_2018 = {"TOYOTA_RAV4": dict(mass=3650 * LB + STD_CARGO, wheelbase=2.65, cf_frac=0.44, steer_ratio=16.30, tsf=0.5533),
            "HONDA_CIVIC": dict(mass=2923 * LB + STD_CARGO, wheelbase=2.70, cf_frac=0.40, steer_ratio=14.63, tsf=1.0)}
SUFFIX_RE = re.compile(r"_(GEN\d|MK\d+|\d{4}(_\d{4})?|PE|MMR|HYBRID|HEV|PHEV|EV|NON_SCC|NON-SCC|PEDAL|TSS\d|FL|FACELIFT|TYPE_S|\d+(ST|ND|RD|TH)_GEN|[A-Z]+)$")


def model_2018(name: str) -> dict:
    """(a, k) implied by the 2018 vehicle model relative to the decoder's bicycle model (a = sR_2018 / sR_decoder, k = -sf)."""
    p = C2K_2018[name]; civ = C2K_2018["HONDA_CIVIC"]
    L = p["wheelbase"]; aF = p["cf_frac"] * L; aR = L - aF
    aF_c = civ["cf_frac"] * civ["wheelbase"]; aR_c = civ["wheelbase"] - aF_c
    cF = 192150 * p["tsf"] * p["mass"] / civ["mass"] * (aR / L) / (aR_c / civ["wheelbase"])
    cR = 202500 * p["tsf"] * p["mass"] / civ["mass"] * (aF / L) / (aF_c / civ["wheelbase"])
    sf = p["mass"] * (cF * aF - cR * aR) / (L ** 2 * cF * cR)
    dec = next(c for c in C2K.values() if c["platform"] == name)
    return dict(a=p["steer_ratio"] / dec["steer_ratio"], k=-sf, steer_ratio_2018=p["steer_ratio"], mass_2018=p["mass"], cF=cF, cR=cR)


# ----------------------------------------------------------------------------- steer ratio table
def steer_ratio_table(lab_platforms: list[str], lab_wb_mm: dict[str, float]) -> pd.DataFrame:
    """platform -> steer_ratio, wheelbase_m, sr_src, wb_src (see the module docstring)."""
    root031 = P.find_opendbc_root()
    t031 = S.parse_platforms(root031).set_index("platform")
    try:
        t24 = S.parse_platforms(Path(C.OPENDBC_LOCAL_FALLBACK)).set_index("platform")
    except Exception:
        t24 = pd.DataFrame(columns=t031.columns)
    rows = []
    for p in lab_platforms:
        sr = wb = None; sr_src = wb_src = "none"
        if p in t031.index and pd.notna(t031.loc[p, "steer_ratio"]):
            sr, sr_src = float(t031.loc[p, "steer_ratio"]), "opendbc_0.3.1"
            if pd.notna(t031.loc[p, "wheelbase_mm"]):
                wb, wb_src = float(t031.loc[p, "wheelbase_mm"]) / 1000.0, "opendbc_0.3.1"
        elif p in t24.index and pd.notna(t24.loc[p, "steer_ratio"]):
            sr, sr_src = float(t24.loc[p, "steer_ratio"]), "opendbc_2024-08"
        else:
            base = p
            while sr is None:
                m = SUFFIX_RE.search(base)
                if not m or m.start() == 0:
                    break
                base = base[:m.start()]
                for tab, src in ((t031, "opendbc_0.3.1"), (t24, "opendbc_2024-08")):
                    if base in tab.index and pd.notna(tab.loc[base, "steer_ratio"]):
                        sr, sr_src = float(tab.loc[base, "steer_ratio"]), f"{src}:{base}"
                        break
        if wb is None and p in lab_wb_mm and np.isfinite(lab_wb_mm[p]) and lab_wb_mm[p] > 0:
            wb, wb_src = lab_wb_mm[p] / 1000.0, "lab_spec"
        rows.append(dict(platform=p, steer_ratio=sr, wheelbase_m=wb, sr_src=sr_src, wb_src=wb_src))
    return pd.DataFrame(rows).set_index("platform")


# ----------------------------------------------------------------------------- per drive
def load_drive(args) -> tuple[str, dict | None]:
    """(platform, usable samples of one drive table): v, yaw_b (offset-corrected angle), yaw_b_raw, yaw_meas, op_lat, das."""
    path, platform, vehicle, sr, wb = args
    cols = ["v_mps", "steer_angle_deg", "yaw_rate_radps", "yaw_valid", "standstill", "das_enabled", "op_lat_active"]
    try:
        df = pd.read_parquet(path, columns=cols)
    except Exception:
        return platform, None
    v = df.v_mps.to_numpy(np.float64); sa = df.steer_angle_deg.to_numpy(np.float64); yaw = df.yaw_rate_radps.to_numpy(np.float64)
    base = np.isfinite(v) & np.isfinite(sa) & np.isfinite(yaw) & df.yaw_valid.to_numpy(bool) & ~df.standstill.to_numpy(bool)
    straight = base & (v >= STRAIGHT_V) & (np.abs(yaw) < STRAIGHT_YAW)
    if int(straight.sum()) < STRAIGHT_MIN_N:
        return platform, dict(skip="no_straight")
    offset = float(np.median(sa[straight]))
    mv = base & (v >= V_MIN)
    corr = float(np.corrcoef(sa[mv], yaw[mv])[0, 1]) if int(mv.sum()) > 100 and sa[mv].std() > 0 else np.nan
    if not corr >= CORR_MIN:
        return platform, dict(skip="neg_corr" if corr < 0 else "low_corr", corr=corr)
    delta = np.deg2rad((sa - offset) / sr); delta_raw = np.deg2rad(sa / sr)
    sel = mv & (np.abs(np.rad2deg(delta)) >= DELTA_MIN_DEG) & (np.abs(yaw) < YAW_MAX)
    if int(sel.sum()) < 10:
        return platform, dict(skip="no_samples")
    yaw_b = v[sel] * np.tan(delta[sel]) / wb; yaw_b_raw = v[sel] * np.tan(delta_raw[sel]) / wb
    return platform, dict(v=v[sel].astype(np.float32), yaw_b=yaw_b.astype(np.float32), yaw_b_raw=yaw_b_raw.astype(np.float32),
                          yaw_meas=yaw[sel].astype(np.float32), op_lat=df.op_lat_active.to_numpy(bool)[sel], das=df.das_enabled.to_numpy(bool)[sel],
                          vehicle=vehicle, corr=corr, offset_deg=offset, n_rows=int(len(df)))


# ----------------------------------------------------------------------------- fits
def _ls2(v, yb, ym):
    """yaw_b = c1 yaw_meas + c2 v^2 yaw_meas -> (a, k) = (c1, c2 / c1)."""
    X = np.column_stack([ym, v * v * ym])
    c, *_ = np.linalg.lstsq(X, yb, rcond=None)
    return float(c[0]), (float(c[1] / c[0]) if c[0] != 0 else np.nan), X @ c


def _ls1(v, yb, ym):
    """yaw_b - yaw_meas = k v^2 yaw_meas (no intercept, no scale)."""
    x = v * v * ym; y = yb - ym
    k = float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else np.nan
    return k, ym * (1 + k * v * v)


def robust_fit(v, yb, ym, fn):
    """One robust pass: fit, drop residuals beyond MAD_K scaled MADs from the median residual, refit."""
    out = fn(v, yb, ym); pred = out[-1]
    r = yb - pred; med = np.median(r); mad = 1.4826 * np.median(np.abs(r - med)) + 1e-12
    keep = np.abs(r - med) <= MAD_K * mad
    out = fn(v[keep], yb[keep], ym[keep]); pred = out[-1]
    ss_res = float(((yb[keep] - pred) ** 2).sum()); ss_tot = float(((yb[keep] - yb[keep].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return out[:-1], float(keep.mean()), r2


def bin_ratios(v, yb, ym) -> dict:
    ok = np.abs(ym) >= YAW_RATIO_MIN
    out = {}
    for lo, hi in BINS:
        m = ok & (v >= lo) & (v < hi)
        n = int(m.sum())
        if n >= 50:
            r = yb[m] / ym[m]
            out[f"{lo}-{hi}"] = dict(n=n, median=float(np.median(r)), q25=float(np.quantile(r, 0.25)), q75=float(np.quantile(r, 0.75)))
        else:
            out[f"{lo}-{hi}"] = dict(n=n, median=np.nan, q25=np.nan, q75=np.nan)
    return out


def k_from_bins(ratios: dict) -> float:
    """ratio - 1 = k v^2 through the bin medians (bin centre, weighted by n); the cross-check of the sample fits."""
    xs, ys, ws = [], [], []
    for key, b in ratios.items():
        if np.isfinite(b["median"]):
            lo, hi = map(float, key.split("-")); vc = (max(lo, V_MIN) + hi) / 2
            xs.append(vc * vc); ys.append(b["median"] - 1); ws.append(b["n"])
    if len(xs) < 2:
        return np.nan
    xs, ys, ws = map(np.asarray, (xs, ys, ws))
    return float((ws * xs * ys).sum() / (ws * xs * xs).sum())


def fit_platform(v, yb, ym, with_bins: bool = True) -> dict:
    v = v.astype(np.float64); yb = yb.astype(np.float64); ym = ym.astype(np.float64)
    (k1,), kept1, r2_1 = robust_fit(v, yb, ym, _ls1)
    (a, k), kept2, r2_2 = robust_fit(v, yb, ym, _ls2)
    out = dict(n=int(len(v)), hours=float(len(v) / HZ / 3600), k_1par=k1, r2_1par=r2_1, kept_1par=kept1, a=a, k=k, r2=r2_2, kept_frac=kept2,
               v_p50=float(np.median(v)), v_p90=float(np.quantile(v, 0.9)))
    if with_bins:
        ratios = bin_ratios(v, yb, ym)
        out["k_bin"] = k_from_bins(ratios); out["bins"] = ratios
    return out


def wmedian(x, w) -> float:
    x = np.asarray(x, float); w = np.asarray(w, float); o = np.argsort(x); x, w = x[o], w[o]
    cw = np.cumsum(w) / w.sum()
    return float(x[np.searchsorted(cw, 0.5)])


def spread(k: pd.Series, h: pd.Series) -> dict:
    k = k.astype(float); h = h.astype(float)
    if len(k) == 0:
        return dict(n=0)
    return dict(n=int(len(k)), wmedian=wmedian(k, h), median=float(k.median()), q25=float(k.quantile(0.25)), q75=float(k.quantile(0.75)),
                wmean=float((k * h).sum() / h.sum()), min=float(k.min()), max=float(k.max()), hours=float(h.sum()))


def global_stats(elig: pd.DataFrame) -> dict:
    """The global k and its spread (train / validation platforms), the same for the two-parameter k, and a's position."""
    tv = elig[elig.split.isin(["train", "validation"])]
    out = dict(k_global=wmedian(tv.k_1par, tv.hours) if len(tv) else np.nan,
               k_1par_train_val=spread(tv.k_1par, tv.hours), k_1par_all_eligible=spread(elig.k_1par, elig.hours),
               k_2par_train_val=spread(tv.k, tv.hours), k_2par_all_eligible=spread(elig.k, elig.hours),
               k_bin_train_val=spread(tv.k_bin.fillna(tv.k_1par), tv.hours) if "k_bin" in tv else None,
               a_train_val=spread(tv.a, tv.hours), a_all_eligible=spread(elig.a, elig.hours),
               a_frac_above_1_10=float((elig.a > 1.10).mean()) if len(elig) else np.nan, a_frac_below_0_90=float((elig.a < 0.90).mean()) if len(elig) else np.nan,
               n_platforms_eligible=int(len(elig)), n_platforms_train_val=int(len(tv)), hours_eligible=float(elig.hours.sum()), hours_train_val=float(tv.hours.sum()))
    out["v_char_mps"] = float(1 / np.sqrt(out["k_global"])) if out["k_global"] and out["k_global"] > 0 else np.nan
    # k against footprint / class (eligible platforms): Spearman rank correlation and class medians
    if len(elig) >= 4 and elig.footprint_m2.notna().sum() >= 4:
        e = elig[elig.footprint_m2.notna()]
        out["spearman_k1_footprint"] = float(e.k_1par.rank().corr(e.footprint_m2.rank()))
        out["spearman_k2_footprint"] = float(e.k.rank().corr(e.footprint_m2.rank()))
        out["spearman_k1_wheelbase"] = float(e.k_1par.rank().corr(e.wheelbase_m.rank()))
        out["k1_median_by_class"] = {c: float(g.k_1par.median()) for c, g in e.groupby("size_class")}
        out["k1_n_by_class"] = {c: int(len(g)) for c, g in e.groupby("size_class")}
        if "powertrain" in e:
            out["k1_median_by_powertrain"] = {c: float(g.k_1par.median()) for c, g in e.groupby("powertrain")}
    return out


# ----------------------------------------------------------------------------- comma2k19
def c2k_validate(npz_dir: Path, k_global: float, k_own: dict[str, float | None]) -> dict:
    """Corrected derived yaw vs the logged yawRate per chunk (10 evenly spaced segments, all segments, human-only of all)."""
    out = {}
    for chunk, cp in C2K.items():
        files = sorted(glob.glob(str(npz_dir / f"{chunk}__*.npz")))
        if not files:
            out[chunk] = dict(error="no npz dumps"); continue
        idx10 = np.unique(np.linspace(0, len(files) - 1, 10).round().astype(int))
        m18 = model_2018(cp["platform"])
        res = dict(platform=cp["platform"], n_segments=len(files), n_segments_subset=int(len(idx10)), model_2018=m18)
        loaded = {}
        for f in files:
            z = np.load(f)
            cs = z["cs"]
            if cs.ndim != 2 or cs.shape[1] < 15:
                continue
            ctrl = z["ctrl"] if "ctrl" in z.files else np.zeros((0, 3))
            if len(ctrl):
                j = np.clip(np.searchsorted(ctrl[:, 0], cs[:, 0]) - 1, 0, len(ctrl) - 1)
                enabled = ctrl[j, 1] > 0.5
            else:
                enabled = np.zeros(len(cs), bool)
            loaded[f] = (cs[:, 1], cs[:, 3], cs[:, 14], enabled)
        for name, sel, human in (("subset10", [files[i] for i in idx10], False), ("all", files, False), ("all_human", files, True)):
            parts = [loaded[f] for f in sel if f in loaded]
            v = np.concatenate([p[0] for p in parts]); sa = np.concatenate([p[1] for p in parts]); y = np.concatenate([p[2] for p in parts])
            en = np.concatenate([p[3] for p in parts])
            delta = np.deg2rad(sa / cp["steer_ratio"])
            m = np.isfinite(v) & np.isfinite(sa) & np.isfinite(y) & (v >= V_MIN) & (np.abs(np.rad2deg(delta)) >= DELTA_MIN_DEG) & (np.abs(y) < YAW_MAX)
            if human:
                m &= ~en
            v, delta, y = v[m], delta[m], y[m]
            yb = v * np.tan(delta) / cp["wheelbase_m"]
            r = dict(n=int(m.sum()), hours=float(m.sum() / 100 / 3600))
            r["uncorrected"] = bin_ratios(v, yb, y)
            r["global_k"] = bin_ratios(v, yb / (1 + k_global * v * v), y)
            ko = k_own.get(cp["platform"])
            r["own_k"] = bin_ratios(v, yb / (1 + ko * v * v), y) if ko is not None and np.isfinite(ko) else None
            r["model_2018_ak"] = bin_ratios(v, yb / (m18["a"] * (1 + m18["k"] * v * v)), y)
            r["residual_global_k"] = {b: (s["median"] - 1 if np.isfinite(s["median"]) else np.nan) for b, s in r["global_k"].items()}
            fit = fit_platform(v, yb, y, with_bins=False)
            r["fit_on_c2k"] = fit
            if name == "all":
                # is the logged yawRate a sensor? with the small-angle form a vehicle model computed from the same steering
                # angle is reproduced to rounding by a v (sa / sR) / (L (1 + k v^2)); an IMU would leave tens of per cent
                v64, y64 = v.astype(float), y.astype(float)
                ylin = v64 * delta / cp["wheelbase_m"]
                (a, k), _, _ = robust_fit(v64, ylin, y64, _ls2)
                pred = ylin / (a * (1 + k * v64 * v64))
                rel = np.abs(pred - y64) / np.maximum(np.abs(y64), 1e-3)
                r["yaw_is_model_test"] = dict(a=a, k=k, rel_err_median=float(np.median(rel)), rel_err_q90=float(np.quantile(rel, 0.9)),
                                              rel_err_q99=float(np.quantile(rel, 0.99)), corr_yaw_steer=float(np.corrcoef(ylin, y64)[0, 1]),
                                              steer_ratio_implied=cp["steer_ratio"] * a)
            res[name] = r
        out[chunk] = res
    return out


# ----------------------------------------------------------------------------- tables
def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, (float, np.floating)):
        return "-" if not np.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def bins_md(b: dict) -> str:
    return " / ".join(fmt(b[f"{lo}-{hi}"]["median"], 2) for lo, hi in BINS)


def platform_md(tab: pd.DataFrame) -> str:
    hdr = ("| platform | class | footprint m2 | split | vehicles | hours used | samples | k_1par (1e-3 s2/m2) | a | k_2par (1e-3) | k_bin (1e-3) | R2 (1-par / 2-par) "
           "| kept | steerRatio (src) | L m | median |offset| deg | median ratio yaw_b / yaw_meas per bin " + " / ".join(f"{lo}-{hi}" for lo, hi in BINS) + " m/s |")
    lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for p, r in tab.iterrows():
        lines.append(f"| {p} | {r.size_class} | {fmt(r.footprint_m2, 2)} | {r.split} | {r.n_vehicles_used} | {fmt(r.hours, 1)} | {int(r.n)} | "
                     f"{fmt(r.k_1par * 1000, 3)} | {fmt(r.a, 3)} | {fmt(r.k * 1000, 3)} | {fmt(r.k_bin * 1000, 3)} | {fmt(r.r2_1par, 3)} / {fmt(r.r2, 3)} | "
                     f"{fmt(r.kept_1par, 3)} | {fmt(r.steer_ratio, 2)} ({r.sr_src}) | {fmt(r.wheelbase_m, 2)} | {fmt(r.offset_abs_med_deg, 1)} | {r.bins_txt} |")
    return "\n".join(lines)


def c2k_md(c2k: dict) -> str:
    lines = ["| chunk / car | samples | yaw variant | median ratio derived / logged per bin " + " / ".join(f"{lo}-{hi}" for lo, hi in BINS) + " m/s |", "|---|---|---|---|"]
    for chunk, r in c2k.items():
        if "error" in r:
            lines.append(f"| {chunk} | - | {r['error']} | |"); continue
        for name in ("subset10", "all", "all_human"):
            rr = r[name]
            for var in ("uncorrected", "global_k", "own_k", "model_2018_ak"):
                if rr.get(var) is None:
                    lines.append(f"| {chunk} ({r['platform']}, {name}) | {rr['n']} | {var} | platform not fitted in the lab |"); continue
                lines.append(f"| {chunk} ({r['platform']}, {name}) | {rr['n']} | {var} | {bins_md(rr[var])} |")
    return "\n".join(lines)


def clean(o):
    """JSON-safe: numpy scalars to Python, non-finite floats to None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(C.DATA_ROOT))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", default=None, help="scratch directory (default <data-root>/tmp/understeer)")
    ap.add_argument("--json", default=None, help="numbers-only JSON copy (e.g. experiments/transfer/understeer_fit.json)")
    ap.add_argument("--c2k-npz-dir", default=None, help="verifier npz dumps (default <data-root>/tmp/comma2k19/verify/logged)")
    ap.add_argument("--angle-variant", choices=VARIANTS, default="offset", help="primary tables: per-drive offset removed, or the raw angle")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N drives only")
    a = ap.parse_args()
    root = Path(a.data_root); out = Path(a.out_dir or root / "tmp" / "understeer"); out.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(6, a.workers))

    ix = pd.read_parquet(root / "build" / "index" / "drives.parquet")
    ix = ix[ix.platform.notna()].copy()
    ix["hours"] = ix.duration_s / 3600
    lab_wb = ix.groupby("platform").wheelbase_mm.first().to_dict()
    plats = sorted(ix.platform.unique())
    srt = steer_ratio_table(plats, lab_wb)
    meta = ix.groupby("platform").agg(n_vehicles=("vehicle_id", "nunique"), hours_total=("hours", "sum"),
                                      split=("split_platform", lambda s: s.mode().iloc[0]), footprint_m2=("footprint_m2", "first"),
                                      size_class=("size_class", "first"), powertrain=("powertrain", "first"), model=("model", "first"),
                                      pose=("pose_source", lambda s: s.mode().iloc[0]))
    meta["split"] = meta.split.map({"train": "train", "validation": "validation", "test": "held-out test", "excluded": "excluded"}).fillna("absent")

    jobs = []
    for r in ix.itertuples(index=False):
        s = srt.loc[r.platform]
        if s.steer_ratio is None or s.wheelbase_m is None or not (s.steer_ratio > 0 and s.wheelbase_m > 0):
            continue
        jobs.append((str(root / "build" / r.file), r.platform, r.vehicle_id, float(s.steer_ratio), float(s.wheelbase_m)))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} drives, {len(plats)} platforms, {int(srt.steer_ratio.notna().sum())} with a steer ratio, workers {workers}", flush=True)
    acc: dict[str, list] = {}; skipped: dict[str, int] = {}; n_ok = 0
    with Pool(workers) as pool:
        for i, (p, d) in enumerate(pool.imap_unordered(load_drive, jobs, chunksize=4)):
            if d is None or "skip" in d:
                key = d["skip"] if d else "read_error"
                skipped[key] = skipped.get(key, 0) + 1
            else:
                acc.setdefault(p, []).append(d); n_ok += 1
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    print(f"{n_ok} drives usable, skipped {skipped}", flush=True)

    # per platform: the primary fit (angle variant of --angle-variant, all samples) with bins; the other (variant, subset) cells without
    rows, bins_all, cells = [], {}, {}
    for p, lst in acc.items():
        v = np.concatenate([d["v"] for d in lst]); ym = np.concatenate([d["yaw_meas"] for d in lst])
        yb_var = {"offset": np.concatenate([d["yaw_b"] for d in lst]), "raw": np.concatenate([d["yaw_b_raw"] for d in lst])}
        op = np.concatenate([d["op_lat"] for d in lst]); das = np.concatenate([d["das"] for d in lst])
        subset_mask = {"all": np.ones(len(v), bool), "human": ~op, "das": op}
        corr = float(np.nanmedian([d["corr"] for d in lst])); offs = np.asarray([d["offset_deg"] for d in lst])
        m = meta.loc[p]
        hours_used = float(len(v) / HZ / 3600); n_veh_used = len({d["vehicle"] for d in lst})
        base = dict(platform=p, n_vehicles=int(m.n_vehicles), n_vehicles_used=n_veh_used, hours_total=float(m.hours_total), split=m.split,
                    footprint_m2=float(m.footprint_m2), size_class=m.size_class, powertrain=m.powertrain, model=m.model, pose=m.pose,
                    steer_ratio=float(srt.loc[p, "steer_ratio"]), sr_src=srt.loc[p, "sr_src"], wheelbase_m=float(srt.loc[p, "wheelbase_m"]),
                    wb_src=srt.loc[p, "wb_src"], corr_yaw_steer=corr, n_drives=len(lst), offset_med_deg=float(np.median(offs)),
                    offset_abs_med_deg=float(np.median(np.abs(offs))), offset_abs_q90_deg=float(np.quantile(np.abs(offs), 0.9)),
                    op_lat_frac=float(op.mean()), das_frac=float(das.mean()), hours=hours_used, n=int(len(v)))
        if n_veh_used < MIN_VEHICLES or hours_used < MIN_HOURS or len(v) < MIN_SAMPLES:
            rows.append(dict(**base, eligible=False)); continue
        f = fit_platform(v, yb_var[a.angle_variant], ym)
        bins_all[p] = f.pop("bins")
        rows.append(dict(**{k: x for k, x in base.items() if k not in f}, eligible=True, bins_txt=bins_md(bins_all[p]), **f))
        cells[p] = {}
        for var in VARIANTS:
            for sub in SUBSETS:
                msk = subset_mask[sub]
                if var == a.angle_variant and sub == "all":
                    cells[p][f"{var}/{sub}"] = {k: f[k] for k in ("n", "hours", "k_1par", "a", "k", "r2", "r2_1par")}
                elif msk.sum() >= MIN_SAMPLES and msk.sum() / HZ / 3600 >= MIN_HOURS:
                    g = fit_platform(v[msk], yb_var[var][msk], ym[msk], with_bins=False)
                    cells[p][f"{var}/{sub}"] = {k: g[k] for k in ("n", "hours", "k_1par", "a", "k", "r2", "r2_1par")}
    tab = pd.DataFrame(rows).set_index("platform")
    elig = tab[tab.eligible].sort_values("k_1par")
    g = global_stats(elig)
    print(json.dumps(clean({k: v for k, v in g.items() if not isinstance(v, dict)}), indent=1), flush=True)

    # the other (variant, subset) cells: global statistics over the same train / validation platforms
    alt = {}
    for var in VARIANTS:
        for sub in SUBSETS:
            key = f"{var}/{sub}"
            rr = []
            for p, c in cells.items():
                if key in c:
                    rr.append(dict(platform=p, split=tab.loc[p, "split"], footprint_m2=tab.loc[p, "footprint_m2"], wheelbase_m=tab.loc[p, "wheelbase_m"],
                                   size_class=tab.loc[p, "size_class"], powertrain=tab.loc[p, "powertrain"], k_bin=np.nan, **c[key]))
            if rr:
                e = pd.DataFrame(rr).set_index("platform")
                alt[key] = global_stats(e)
                alt[key]["per_platform"] = {p: dict(k_1par=r.k_1par, a=r.a, k=r.k, hours=r.hours) for p, r in e.iterrows()}

    npz_dir = Path(a.c2k_npz_dir or root / "tmp" / "comma2k19" / "verify" / "logged")
    own = {cp["platform"]: (float(tab.loc[cp["platform"], "k_1par"]) if cp["platform"] in elig.index else None) for cp in C2K.values()}
    c2k = c2k_validate(npz_dir, g["k_global"], own) if npz_dir.is_dir() else {"error": "npz dir absent"}

    tab.to_csv(out / "platform_table.csv")
    (out / "platform_table.md").write_text(platform_md(elig) + "\n")
    (out / "c2k_table.md").write_text(c2k_md(c2k) + "\n" if "error" not in c2k else str(c2k))
    result = dict(primary=dict(angle_variant=a.angle_variant, samples="all"),
                  rule=dict(v_min_mps=V_MIN, delta_min_deg=DELTA_MIN_DEG, yaw_max_radps=YAW_MAX, mad_k=MAD_K, min_vehicles=MIN_VEHICLES,
                            min_hours=MIN_HOURS, min_samples=MIN_SAMPLES, bins_mps=BINS, straight_yaw_radps=STRAIGHT_YAW, straight_v_mps=STRAIGHT_V,
                            straight_min_n=STRAIGHT_MIN_N, corr_min=CORR_MIN, yaw_ratio_min_radps=YAW_RATIO_MIN),
                  n_drives=len(jobs), n_drives_usable=n_ok, n_drives_skipped=skipped, n_platforms_lab=len(plats),
                  n_platforms_with_steer_ratio=int(srt.steer_ratio.notna().sum()), n_platforms_with_samples=len(acc),
                  sr_sources=srt.sr_src.value_counts().to_dict(), **g, variants=alt,
                  platforms={p: {k: v for k, v in r.items() if k != "bins_txt"} for p, r in tab.to_dict("index").items()},
                  bins=bins_all, comma2k19=c2k)
    txt = json.dumps(clean(result), indent=1)
    (out / "fit.json").write_text(txt)
    if a.json:
        Path(a.json).write_text(txt)
    print(f"eligible platforms {len(elig)}, train/val {g['n_platforms_train_val']} -> k_global {g['k_global']:.6g} s^2/m^2")
    print(c2k_md(c2k) if "error" not in c2k else c2k)


if __name__ == "__main__":
    main()
