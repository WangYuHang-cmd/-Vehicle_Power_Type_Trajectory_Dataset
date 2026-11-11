#!/usr/bin/env python3
"""commaCarSegments: rlog.zst -> opendbc CarInterface / RadarInterface -> 10 Hz -> the 20 training channels -> windows.

Runs under the decode venv (PyPI opendbc 0.3.1, pycapnp, zstandard, numpy, pandas, pyarrow):
  $D/ext/venv_opendbc/bin/python3 experiments/transfer/ccs_decode.py \
      --segments-root $EXT/ccs/segments [--list $EXT/ccs/select.txt] --dataset ccs --workers 6
  ... --segments-root $D/scratch/ccs_verify/dl/smp --dataset ccs_sample \
      --copy-sample-to $EXT/ccs/work          # the 5 verification segments

Per segment (one 60-s file, 100 Hz CAN): the log is read with the cereal `log.capnp` of the openpilot checkout through a
private pycapnp SchemaParser (the full openpilot LogReader needs the openpilot runtime; opendbc's own car.capnp shares the
cereal schema ids, so the two schemas live in separate parsers); carParams is re-read with opendbc's CarParams, its
fingerprint folded with MIGRATION; every `can` message goes through CarInterface.update (CarState at 100 Hz) and, on radar
cars, RadarInterface.update (tracks at 20 Hz); `pandaStates.controlsAllowed` is held onto the grid. The 10 Hz grid starts
at the first CAN packet (600 samples); a grid sample takes the latest CarState (ZOH, max age 0.3 s = the lab gap rule) and
a segment with any stale sample is dropped. Channels (CANDIDATES.md 2.1 / CONTRACT.md): v = vEgo, a = aEgo, steering
angle / rate / torque, flags from CarState, das_enabled = cruiseState.enabled OR controlsAllowed, yaw = kinematic bicycle
model from steering (steerRatio, wheelbase of carParams; `--yaw measured` prefers CarState.yawRate where a brand provides
it), curv = yaw / v, lat_acc = v yaw, steer_torque_norm = torque / 95th percentile of |torque| over moving steer_pressed
samples of the whole route batch (all segments of the route in this run, >= 50 samples, else the torque group is absent),
lead from the radar tracks (nearest in-path track: |y - curv d^2 / 2| < 1.8 m + 1 cm/m, track seen >= 3 radar samples,
far stationary clutter rejected) or, on radar-less Hyundai / Kia / Genesis, the stock ACC object of SCC11 / SCC_CONTROL
(`--no-scc-lead` disables); vision-only platforms without either get lead_valid = 0. Windows: common.cut_windows with
align='drive' on the route's concatenated 10 Hz frame (t = seg x 60 + k / 10): exactly one window per complete segment,
w_idx = segment number; common.save_windows writes <external>/<dataset>/work/windows.npz + meta.parquet. Also written
there: segments_qc.csv (per-segment quality numbers, pseudonymous ids only) and decode_summary.json.

`--honda-gearbox-from-can` (off by default; the `ccs_honda` re-decode of the nine Honda platforms that the first run lost):
opendbc 0.3.1's Honda CarState reads the gearbox message named by CarParams.transmissionType (cvt -> GEARBOX_CVT 0x191,
anything else -> GEARBOX_AUTO 0x1A3) and its pure-Python parser holds canValid at 0 while a registered message has never
arrived. The logging openpilot versions never set transmissionType for Honda (logged `unknown`), so every traditional-CVT
Honda (Civic, Fit, HR-V, CR-V, Freed: 0x191 only) decoded with canValid 0. The option applies 0.3.1's own `_get_params`
rule to the pt-bus addresses of the log before the interfaces are built (0x191 present and not an Acura RDX -> cvt,
0x1A3 or neither -> automatic / manual) and records the result as `gearbox=<cvt|automatic|manual>` in `meta.note` and
`segments_qc.csv` (Honda only; everything else is untouched, the default run is byte-identical).

comma2k19 (2018 logs; all three options off by default, the ccs / ccs_honda paths stay byte-identical):
  `--layout comma2k19`: segments at <root>/Chunk_N/<dongle>|<route>/<seg>/raw_log.bz2 (dongle = the part before '|', route = the
  whole folder name; the list file, if any, holds Chunk_N/<folder>/<seg> lines). `--fingerprint-map Chunk_1=TOYOTA_RAV4,...`: a
  2018 route logged carParams in its first segment only, and the chunks hold middle segments, so the CarParams of a mapped
  segment is built the way openpilot builds it at start-up, with opendbc 0.3.1's `CarInterface.get_params(platform, fingerprint,
  car_fw=[], alpha_long=False, is_release=False, docs=False)` from the mapped platform and the segment's own live fingerprint
  (address -> payload length per bus, buses below 128, as `can_fingerprint` collects it); this sets transmissionType by 0.3.1's
  own Honda rule, the safety configuration, the specs and the flags. Every segment of a mapped folder uses the constructed
  CarParams (uniform provenance; `cp=constructed` in `meta.note`, `cp_source` in `segments_qc.csv`); a logged carParams, when one
  exists, is only compared with it (`cp_logged_diff` column: fingerprint fold, transmissionType, safety model / param, wheelbase,
  steerRatio, mass). `--das-from controlsState`: das_enabled = controlsState.enabled OR cruiseState.enabled (controlsState at
  100 Hz, held onto the grid like pandaStates; `das=controlsState` and `ctrl_enabled=` in `meta.note`, `controls_enabled_frac`
  in `segments_qc.csv`); the default `panda` keeps controlsAllowed (the 2018 `pandaStateDEPRECATED` runs at 1 Hz and is not read).
  `--ignore-absent-messages`: 2018 openpilot logged buses 0 and 1 only (no camera bus) and the 2017 RAV4 carries no PRE_COLLISION
  0x283 on bus 0, while 0.3.1's CarState parsers register the camera-bus HUD messages (Toyota LKAS_HUD; Honda ACC_HUD, LKAS_HUD,
  BRAKE_COMMAND) and PRE_COLLISION and hold canValid at 0 while a registered message has never arrived (the Honda gearbox case
  again). With the option, a registered message whose address never appears on the parser's bus in the segment's live
  fingerprint is marked `ignore_alive` (the parser's own rule for messages registered without a frequency, e.g. BLINKERS_STATE),
  also for messages the CarState registers lazily during the replay; checksum, counter and timeout checks of every message the
  log does carry stay in force. The waived messages feed stockAeb / stockFcw / HUD pass-through / cruiseState.speed only, none of
  the 20 channels; the list is written per segment as `absent_msgs` (`pt:NAME,...;cam:NAME,...`) to `segments_qc.csv`.

`--yaw derived-understeer --understeer-k K` (the understeer-corrected re-decodes `ccs_us`, `ccs_honda_us`, `comma2k19_us`; off by
  default, the `derived` path stays byte-identical): the kinematic bicycle yaw of `--yaw derived` overestimates the measured yaw
  rate with speed (ratio about 1 + k v^2 against the logged `carState.yawRate` of comma2k19, CCS_PIPELINE.md known gap 9: the
  understeer gradient the kinematic model ignores). The option divides the bicycle yaw by the linear single-track steady-state
  factor, yaw = v tan(angle / steerRatio) / (wheelbase (1 + K v^2)), with ONE global K in s^2/m^2 given on the command line
  (fitted on the lab's own IMU / pose yaw, validated on comma2k19; never per platform, so no lab calibration leaks into the
  external features), before curv = yaw / v and lat_acc = v yaw are derived exactly as today. `yaw_source` = `derived-us` in
  `segments_qc.csv`, `yaw=derived-us;k=K` in `meta.note`, `yaw` / `understeer_k` / `yaw_sources` in `decode_summary.json`. Nothing
  else changes (same grid, lead gate - which keeps the uncorrected bicycle curvature for its in-path test - torque scale, labels,
  pseudonyms), so a re-decode differs from the original in the three yaw-group channels only.
"""
from __future__ import annotations

import argparse
import bz2
import glob
import importlib
import json
import logging
import math
import os
import shutil
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.transfer import common as T  # noqa: E402
from experiments.transfer import ccs_platforms as P  # noqa: E402
from vtt import config as C  # noqa: E402

CEREAL_DIR = Path(os.environ.get("VTT_CEREAL_DIR", str(Path(__file__).resolve().parents[3] / "openpilot" / "cereal")))   # <repo parent>/openpilot/cereal
GRID_N = T.WINDOW_SAMPLES                     # 600 samples per segment
CS_MAX_AGE_S = T.GAP_S                        # 0.3 s: the lab gap rule for the CarState hold
CTRL_MAX_AGE_S = float(C.CTRL_MAX_AGE_S)      # 0.3 s: pandaStates hold
LEAD_MAX_AGE_S = float(C.LEAD_MAX_AGE_S)      # 0.75 s: radar lead hold
LEAD_DREL_MIN_M = float(C.LEAD_VALID_DREL_MIN_M)
LEAD_Y_GATE_M, LEAD_Y_GATE_PER_M = 1.8, 0.01  # in-path gate: |y_rel - curv d^2 / 2| < 1.8 m + 0.01 m per m of range
LEAD_MIN_TRACK_SAMPLES = 3                    # a track must have been seen in >= 3 consecutive radar samples
LEAD_STATIC_FAR_M, LEAD_STATIC_V_MPS, LEAD_STATIC_EGO_V_MPS = 60.0, 1.0, 5.0   # far stationary objects while driving = clutter
SCC_DIST_MAX_M, SCC_VREL_MAX_MPS = 200.0, 30.0   # SCC11 / SCC_CONTROL report 204.7 m / full-scale speed when no object is tracked
MIN_CAN_VALID_FRAC = 0.5
SEG_S = 60.0

HONDA_GEARBOX_CVT, HONDA_GEARBOX_AUTO = 0x191, 0x1A3   # GEARBOX_CVT / GEARBOX_AUTO on the Honda powertrain bus
LAYOUTS = ("ccs", "comma2k19")
DAS_SOURCES = ("panda", "controlsState")
YAW_UNDERSTEER = "derived-understeer"         # --yaw mode: bicycle yaw / (1 + K v^2); yaw_source 'derived-us', note 'yaw=derived-us;k=K'
YAW_MODES = ("derived", "measured", "none", YAW_UNDERSTEER)


def understeer_yaw(v, steer_angle_deg, steer_ratio: float, wheelbase_m: float, k: float) -> np.ndarray:
    """Understeer-corrected steering-derived yaw rate (rad/s): the kinematic bicycle yaw of common.yaw_from_steering divided by the
    linear single-track steady-state factor 1 + k v^2 (k in s^2/m^2 = understeer gradient / (g wheelbase); one global value)."""
    v = np.asarray(v, float)
    return T.yaw_from_steering(v, steer_angle_deg, steer_ratio, wheelbase_m) / (1.0 + float(k) * v * v)
CP_COMPARE_TOL = dict(wheelbase=1e-3, steerRatio=1e-2, mass=1.0)   # logged vs constructed CarParams: tolerances of the numeric fields

_W: dict = {}                                 # per-process state (schemas, opendbc modules, labeller)


# ----------------------------------------------------------------------------- worker set-up
def init_worker(cereal_dir: str, yaw_mode: str, use_scc: bool, scc_sign: float = 1.0, honda_gearbox: bool = False,
                das_from: str = "panda", fp_map: dict | None = None, ignore_absent: bool = False, understeer_k: float | None = None) -> None:
    import capnp
    import zstandard
    capnp.remove_import_hook()
    parser = capnp.SchemaParser()                                   # private parser: cereal's car.capnp ids clash with opendbc's
    _W["log"] = parser.load(str(Path(cereal_dir) / "log.capnp"))
    _W["zstd"] = zstandard
    from opendbc.car import structs
    from opendbc.car.fingerprints import MIGRATION
    from opendbc.car.can_definitions import CanData
    from opendbc.car.carlog import carlog
    carlog.setLevel(logging.CRITICAL)                               # the pure-Python CANParser logs every stale message otherwise
    _W.update(structs=structs, MIGRATION={k: str(v) for k, v in MIGRATION.items()}, CanData=CanData, labeller=P.Labeller(),
              yaw_mode=yaw_mode, use_scc=use_scc, scc_sign=float(scc_sign), honda_gearbox=bool(honda_gearbox), brand_of={},
              das_from=das_from, fp_map=dict(fp_map or {}), ignore_absent=bool(ignore_absent),
              understeer_k=(float(understeer_k) if understeer_k is not None else None))
    if yaw_mode == YAW_UNDERSTEER and not (_W["understeer_k"] is not None and _W["understeer_k"] > 0):
        raise SystemExit("--yaw derived-understeer needs --understeer-k K (s^2/m^2, > 0)")
    if _W["fp_map"]:
        from opendbc.car.values import PLATFORMS
        bad = sorted(v for v in _W["fp_map"].values() if v not in PLATFORMS)
        if bad:
            raise SystemExit(f"--fingerprint-map: unknown opendbc platform(s) {bad}")


def _brand_of(fp: str) -> str | None:
    if not _W["brand_of"]:
        from opendbc.car.values import BRANDS
        for CAR in BRANDS:
            for p in CAR:
                _W["brand_of"][str(p)] = CAR.__module__.split(".")[-2]
    return _W["brand_of"].get(fp)


def read_events(path: str):
    with open(path, "rb") as f:
        buf = f.read()
    if path.endswith(".zst"):
        raw = _W["zstd"].ZstdDecompressor().decompressobj().decompress(buf)
    elif path.endswith(".bz2"):
        raw = bz2.decompress(buf)
    else:
        raw = buf
    return list(_W["log"].Event.read_multiple_bytes(raw, traversal_limit_in_words=2 ** 61)), raw


def car_params(evs):
    """opendbc CarParams reader (fingerprint folded with MIGRATION), raw fingerprint, folded fingerprint; None without carParams."""
    ev = next((e for e in evs if _which(e) == "carParams"), None)
    if ev is None:
        return None, None, None
    with _W["structs"].CarParams.from_bytes(ev.carParams.as_builder().to_bytes()) as r:
        b = r.as_builder()
    raw_fp = str(b.carFingerprint)
    folded = _W["MIGRATION"].get(raw_fp, raw_fp)
    b.carFingerprint = folded
    if not str(b.brand):
        b.brand = _brand_of(folded) or ""
    return b.as_reader(), raw_fp, folded


def _which(e) -> str | None:
    try:
        return e.which()
    except Exception:                                               # union member unknown to the 2024 schema
        return None


def mapped_platform(path: str) -> str | None:
    """The platform of `--fingerprint-map` whose folder name is an ancestor of the segment path; None without a map or match."""
    fp_map = _W.get("fp_map") or {}
    if not fp_map:
        return None
    for part in Path(path).parts:
        if part in fp_map:
            return fp_map[part]
    return None


def live_fingerprint(evs) -> dict[int, dict[int, int]]:
    """{bus: {address: payload length}} over the segment's own CAN, buses below 128 (the panda's transmit echoes excluded), as
    opendbc's `can_fingerprint` collects it at start-up."""
    from opendbc.car import gen_empty_fingerprint
    finger = gen_empty_fingerprint()
    for e in evs:
        if _which(e) == "can":
            for c in e.can:
                src = int(c.src)
                if src < 128:
                    finger.setdefault(src, {})[int(c.address)] = len(c.dat)
    return finger


def construct_car_params(platform: str, finger: dict):
    """CarParams built the way openpilot builds it at start-up: opendbc 0.3.1 `CarInterface.get_params` of the platform's brand from
    the live fingerprint (no firmware versions, no alpha longitudinal, not a release build, not docs) -> reader."""
    brand = _brand_of(platform)
    if brand is None:
        raise KeyError(f"no opendbc brand for platform {platform}")
    CI_cls = importlib.import_module(f"opendbc.car.{brand}.interface").CarInterface
    b = CI_cls.get_params(platform, finger, [], False, False, False)
    if not str(b.brand):
        b.brand = brand
    return b.as_reader()


def compare_car_params(CP_logged, CP_built) -> str:
    """Differences between a logged CarParams (fingerprint already folded) and the constructed one; '' when equal on the compared
    fields (fingerprint, transmissionType, safety model / param, wheelbase, steerRatio, mass)."""
    diffs = []
    if str(CP_logged.carFingerprint) != str(CP_built.carFingerprint):
        diffs.append(f"fingerprint {CP_logged.carFingerprint}!={CP_built.carFingerprint}")
    if str(CP_logged.transmissionType) != str(CP_built.transmissionType):
        diffs.append(f"transmission {CP_logged.transmissionType}!={CP_built.transmissionType}")
    sl = [(str(c.safetyModel), int(c.safetyParam)) for c in CP_logged.safetyConfigs]
    if not sl and hasattr(CP_logged, "safetyModelDEPRECATED"):
        sl = [(str(CP_logged.safetyModelDEPRECATED), int(CP_logged.safetyParamDEPRECATED))]
    sb = [(str(c.safetyModel), int(c.safetyParam)) for c in CP_built.safetyConfigs]
    if sl != sb:
        diffs.append(f"safety {sl}!={sb}")
    for k, tol in CP_COMPARE_TOL.items():
        a, b = float(getattr(CP_logged, k)), float(getattr(CP_built, k))
        if abs(a - b) > tol:
            diffs.append(f"{k} {a:.3f}!={b:.3f}")
    return "; ".join(diffs)


def waive_absent_messages(CI, finger: dict) -> dict[str, list[str]]:
    """Mark every registered CarState parser message whose address never appears on the parser's bus in the segment's live
    fingerprint as `ignore_alive` (the parser's rule for messages registered without a frequency), now and at every later lazy
    registration; returns the dict that collects the waived names per parser (filled during the replay)."""
    waived: dict[str, list[str]] = {}
    for bus_key, cp in CI.can_parsers.items():
        present = finger.get(int(cp.bus), {})
        key = str(bus_key)

        def mark(cp=cp, present=present, key=key):
            for addr, st in cp.message_states.items():
                if addr not in present and not st.ignore_alive:
                    st.ignore_alive = True
                    waived.setdefault(key, []).append(str(st.name))

        def add(name_or_addr, freq=None, _orig=cp._add_message, _mark=mark):
            _orig(name_or_addr, freq)
            _mark()

        mark()
        cp._add_message = add                                       # VLDict.__getitem__ registers through this attribute
    return waived


def honda_gearbox_from_can(CP, evs, folded: str):
    """(CarParams with transmissionType set by opendbc's Honda `_get_params` rule applied to the logged pt-bus addresses, the
    rule's result as a string). GEARBOX_CVT 0x191 present (and not an Acura RDX) -> cvt; GEARBOX_AUTO 0x1A3 (or 0x191 on the RDX)
    -> automatic; neither -> manual. Honda only: other brands come back unchanged with None."""
    if str(CP.brand) != "honda":
        return CP, None
    from opendbc.car.honda.hondacan import CanBus
    from opendbc.car.honda.values import CAR
    TT = _W["structs"].CarParams.TransmissionType
    pt = CanBus(CP).pt
    seen = set()
    for e in evs:
        if _which(e) == "can":
            for c in e.can:
                if int(c.src) == pt and int(c.address) in (HONDA_GEARBOX_CVT, HONDA_GEARBOX_AUTO):
                    seen.add(int(c.address))
    if not seen:
        name = "manual"
    elif HONDA_GEARBOX_CVT in seen and folded != str(CAR.ACURA_RDX):
        name = "cvt"
    else:
        name = "automatic"
    b = CP.as_builder()
    b.transmissionType = getattr(TT, name)
    return b.as_reader(), name


def build_interfaces(CP):
    import importlib
    brand = str(CP.brand)
    CI = importlib.import_module(f"opendbc.car.{brand}.interface").CarInterface(CP)
    RI = None
    if not CP.radarUnavailable:
        try:
            RI = importlib.import_module(f"opendbc.car.{brand}.radar_interface").RadarInterface(CP)
        except Exception:
            RI = None
    return CI, RI


def scc_lead_parser(CP):
    """(CANParser, message name, address, sign) for the stock ACC object of radar-less Hyundai / Kia / Genesis cars; None elsewhere."""
    if not _W["use_scc"] or str(CP.brand) != "hyundai" or not CP.radarUnavailable:
        return None
    try:
        from opendbc.can import CANParser
        from opendbc.car import Bus
        from opendbc.car.hyundai.values import DBC, HyundaiFlags
        dbc = DBC[CP.carFingerprint][Bus.pt]
        if CP.flags & HyundaiFlags.CANFD:
            from opendbc.car.hyundai.hyundaicanfd import CanBus
            cb = CanBus(CP)
            bus = cb.CAM if CP.flags & HyundaiFlags.CANFD_CAMERA_SCC else cb.ECAN
            return CANParser(dbc, [("SCC_CONTROL", 50)], bus), "SCC_CONTROL", 0x1A0
        bus = 2 if CP.flags & HyundaiFlags.CAMERA_SCC else 0
        return CANParser(dbc, [("SCC11", 50)], bus), "SCC11", 0x420
    except Exception:
        return None


# ----------------------------------------------------------------------------- lead picker (radar tracks)
def pick_lead(points, v_ego: float, curv: float, ages: dict) -> tuple[bool, float, float, float]:
    """(valid, dRel, yRel, vRel) of the nearest in-path, persistent, measured radar track; far stationary clutter rejected."""
    seen = set()
    best = None
    for p in points:
        tid = int(p.trackId)
        seen.add(tid)
        ages[tid] = ages.get(tid, 0) + 1
        d = float(p.dRel)
        if not (d > LEAD_DREL_MIN_M) or not bool(p.measured):
            continue
        if ages[tid] < LEAD_MIN_TRACK_SAMPLES:
            continue
        y_path = 0.5 * curv * d * d if math.isfinite(curv) else 0.0
        if abs(float(p.yRel) - y_path) > LEAD_Y_GATE_M + LEAD_Y_GATE_PER_M * d:
            continue
        vr = float(p.vRel)
        if v_ego > LEAD_STATIC_EGO_V_MPS and abs(v_ego + vr) < LEAD_STATIC_V_MPS and d > LEAD_STATIC_FAR_M:
            continue
        if best is None or d < best[1]:
            best = (True, d, float(p.yRel), vr)
    for tid in list(ages):
        if tid not in seen:
            del ages[tid]
    return best if best is not None else (False, float("nan"), float("nan"), float("nan"))


# ----------------------------------------------------------------------------- one segment -> 10 Hz frame
def decode_segment(path: str) -> dict:
    """10 Hz arrays (600 samples) of one segment plus QC numbers; dict(skip=reason) when the segment yields no window."""
    t_start = time.time()
    evs, _raw = read_events(path)
    CP, raw_fp, folded = car_params(evs)
    cp_source, cp_logged_diff = "logged", None
    plat_map = mapped_platform(path)
    finger = live_fingerprint(evs) if (plat_map is not None or _W.get("ignore_absent")) else None
    if plat_map is not None:                                        # comma2k19: the CarParams of the mapped platform, built from the live fingerprint
        CP_built = construct_car_params(plat_map, finger)
        if CP is not None:
            cp_logged_diff = compare_car_params(CP, CP_built) or "none"
        CP, raw_fp, folded, cp_source = CP_built, plat_map, plat_map, "constructed"
    if CP is None:
        return dict(skip="no carParams")
    brand = str(CP.brand)
    gearbox = None
    if _W.get("honda_gearbox"):
        CP, gearbox = honda_gearbox_from_can(CP, evs, folded)
    elif cp_source == "constructed" and brand == "honda":
        gearbox = str(CP.transmissionType)                          # set by 0.3.1's own rule inside get_params
    try:
        CI, RI = build_interfaces(CP)
    except Exception as ex:
        return dict(skip=f"no CarInterface for {folded} ({type(ex).__name__})", platform=folded, raw_fp=raw_fp)
    waived = waive_absent_messages(CI, finger) if _W.get("ignore_absent") else None
    scc = scc_lead_parser(CP)
    CanData = _W["CanData"]
    steer_ratio, wheelbase = float(CP.steerRatio), float(CP.wheelbase)
    rows: list[tuple] = []
    radar: list[tuple] = []
    panda: list[tuple] = []
    ctrl_state: list[tuple] = []
    das_from = _W.get("das_from", "panda")
    ages: dict = {}
    n_radar_samples = 0
    last_v, last_curv = 0.0, 0.0
    for e in evs:
        w = _which(e)
        if w == "can":
            t = e.logMonoTime * 1e-9
            pkt = [(e.logMonoTime, [CanData(c.address, c.dat, c.src) for c in e.can])]
            CS = CI.update(pkt)
            v = float(CS.vEgo)
            sa = float(CS.steeringAngleDeg)
            rows.append((t, v, float(CS.aEgo), sa, float(CS.steeringRateDeg), float(CS.steeringTorque), float(CS.yawRate),
                         float(CS.gasPressed), float(CS.brakePressed), float(CS.steeringPressed), float(CS.standstill),
                         float(CS.cruiseState.enabled), float(CS.canValid)))
            if v >= T.MOVING_V_MPS and steer_ratio > 0 and wheelbase > 0:
                last_curv = math.tan(math.radians(sa / steer_ratio)) / wheelbase        # yaw / v of the bicycle model
            else:
                last_curv = 0.0
            last_v = v
            if RI is not None:
                rr = RI.update(pkt)
                if rr is not None:
                    n_radar_samples += 1
                    radar.append((t, *pick_lead(rr.points, last_v, last_curv, ages)))
            if scc is not None:
                try:
                    upd = scc[0].update(pkt)
                except Exception:
                    scc = None
                    continue
                if scc[2] in upd:
                    vl = scc[0].vl[scc[1]]
                    d, vr = float(vl["ACC_ObjDist"]), _W["scc_sign"] * float(vl["ACC_ObjRelSpd"])
                    # 204.7 m / full-scale relative speed are the 'no object' sentinels (seen on the CAN-FD Ioniq 5 sample)
                    valid = float(vl["ObjValid"]) > 0.5 and LEAD_DREL_MIN_M < d < SCC_DIST_MAX_M and abs(vr) < SCC_VREL_MAX_MPS
                    n_radar_samples += 1
                    radar.append((t, valid, d if valid else float("nan"), 0.0, vr if valid else float("nan")))
        elif w == "pandaStates":
            ps = e.pandaStates
            if len(ps):
                panda.append((e.logMonoTime * 1e-9, float(bool(ps[0].controlsAllowed))))
        elif w == "pandaState":
            panda.append((e.logMonoTime * 1e-9, float(bool(e.pandaState.controlsAllowed))))
        elif w == "controlsState" and das_from == "controlsState":
            ctrl_state.append((e.logMonoTime * 1e-9, float(bool(e.controlsState.enabled))))
    if len(rows) < 100:
        return dict(skip=f"only {len(rows)} can packets", platform=folded, raw_fp=raw_fp)
    A = np.asarray(rows, dtype=np.float64)
    t = A[:, 0]
    span = float(t[-1] - t[0])
    can_valid_frac = float(A[:, 12].mean())
    if can_valid_frac < MIN_CAN_VALID_FRAC:
        return dict(skip=f"canValid only {can_valid_frac:.2f}", platform=folded, raw_fp=raw_fp)
    grid = t[0] + np.arange(GRID_N) / T.HZ
    j = np.searchsorted(t, grid + 1e-6, side="right") - 1
    age = grid - t[np.clip(j, 0, len(t) - 1)]
    fresh = (j >= 0) & (age <= CS_MAX_AGE_S + 1e-9)
    if not fresh.all():
        return dict(skip=f"{int((~fresh).sum())} stale grid samples (span {span:.1f} s)", platform=folded, raw_fp=raw_fp)
    S = A[j]
    v, a, sa, sr, tq, yaw_meas = S[:, 1], S[:, 2], S[:, 3], S[:, 4], S[:, 5], S[:, 6]
    gas, brake, sp, ss, cruise = (S[:, k] for k in (7, 8, 9, 10, 11))
    # pandaStates.controlsAllowed (10 Hz) held onto the grid
    ctrl = np.zeros(GRID_N)
    if panda:
        Pn = np.asarray(panda, dtype=np.float64)
        _, held = T.resample_to_10hz(Pn[:, 0], Pn[:, 1], "hold", t_grid=grid, max_age_s=CTRL_MAX_AGE_S)
        ctrl = np.nan_to_num(held, nan=0.0)
    ctrl_en = np.zeros(GRID_N)
    if ctrl_state:                                                  # controlsState.enabled (100 Hz) held onto the grid
        Cn = np.asarray(ctrl_state, dtype=np.float64)
        _, held = T.resample_to_10hz(Cn[:, 0], Cn[:, 1], "hold", t_grid=grid, max_age_s=CTRL_MAX_AGE_S)
        ctrl_en = np.nan_to_num(held, nan=0.0)
    das = ((cruise > 0.5) | ((ctrl_en if das_from == "controlsState" else ctrl) > 0.5)).astype(np.float64)
    # yaw
    yaw_src = "none"
    yaw = np.full(GRID_N, np.nan)
    mode = _W["yaw_mode"]
    meas_frac = float((yaw_meas != 0).mean())
    if mode == "measured" and meas_frac > 0.5:
        yaw, yaw_src = yaw_meas.copy(), "measured"
    elif mode in ("derived", "measured") and steer_ratio > 0 and wheelbase > 0:
        yaw, yaw_src = T.yaw_from_steering(v, sa, steer_ratio, wheelbase), "derived"
    elif mode == YAW_UNDERSTEER and steer_ratio > 0 and wheelbase > 0:
        yaw, yaw_src = understeer_yaw(v, sa, steer_ratio, wheelbase, _W["understeer_k"]), "derived-us"
    # lead at the grid (hold the native samples, max age 0.75 s)
    lead_src = "none"
    lv = np.zeros(GRID_N)
    ld = np.full(GRID_N, np.nan)
    lvr = np.full(GRID_N, np.nan)
    if radar:
        R = np.asarray(radar, dtype=np.float64)
        jr = np.searchsorted(R[:, 0], grid + 1e-6, side="right") - 1
        fr = (jr >= 0) & ((grid - R[np.clip(jr, 0, len(R) - 1), 0]) <= LEAD_MAX_AGE_S + 1e-9)
        jr = np.clip(jr, 0, len(R) - 1)
        valid = fr & (R[jr, 1] > 0.5)
        lv = valid.astype(np.float64)
        ld = np.where(valid, R[jr, 2], np.nan)
        lvr = np.where(valid, R[jr, 4], np.nan)
        lead_src = ("scc" if (scc is not None and RI is None) else "radar") if valid.any() else "none"
    scc_sign_corr = float("nan")
    if radar and RI is None and scc is not None:
        R2 = np.asarray(radar, dtype=np.float64)
        m = R2[:, 1] > 0.5
        if m.sum() >= 30:
            dd = np.gradient(R2[m, 2], R2[m, 0])
            ok = np.isfinite(dd) & (np.abs(dd) < 20)
            if ok.sum() >= 30 and np.std(dd[ok]) > 0 and np.std(R2[m, 4][ok]) > 0:
                scc_sign_corr = float(np.corrcoef(dd[ok], R2[m, 4][ok])[0, 1])   # > 0 when ObjRelSpd is lead minus ego
    seg = int(Path(path).parent.name) if Path(path).parent.name.isdigit() else -1
    qc = dict(platform=folded, raw_fp=raw_fp, brand=brand, radar_unavailable=bool(CP.radarUnavailable), seg=seg, n_can=int(len(rows)),
              span_s=round(span, 2), can_valid_frac=round(can_valid_frac, 3), v_min=round(float(v.min()), 2), v_max=round(float(v.max()), 2),
              v_mean=round(float(v.mean()), 2), moving_frac=round(float((v >= T.MOVING_V_MPS).mean()), 3), a_min=round(float(a.min()), 2),
              a_max=round(float(a.max()), 2), steer_min=round(float(sa.min()), 1), steer_max=round(float(sa.max()), 1),
              steer_rate_nonzero=round(float((sr != 0).mean()), 3), torque_nonzero=round(float((tq != 0).mean()), 3),
              yaw_meas_nonzero=round(meas_frac, 3), gas_frac=round(float(gas.mean()), 3), brake_frac=round(float(brake.mean()), 3),
              steer_pressed_frac=round(float(sp.mean()), 3), standstill_frac=round(float(ss.mean()), 3), cruise_frac=round(float(cruise.mean()), 3),
              controls_allowed_frac=round(float(ctrl.mean()), 3), das_frac=round(float(das.mean()), 3), lead_source=lead_src,
              n_radar_samples=int(n_radar_samples), lead_valid_frac=round(float(lv.mean()), 3), scc_sign_corr=(round(scc_sign_corr, 2) if math.isfinite(scc_sign_corr) else None), yaw_source=yaw_src,
              steer_ratio=round(steer_ratio, 2), wheelbase_m=round(wheelbase, 3), decode_s=round(time.time() - t_start, 2))
    if _W.get("honda_gearbox") or gearbox is not None:
        qc["gearbox"] = gearbox
    if cp_source != "logged" or das_from != "panda":
        qc.update(cp_source=cp_source, cp_logged_diff=cp_logged_diff, das_source=das_from, controls_enabled_frac=round(float(ctrl_en.mean()), 3))
    if waived is not None:
        qc["absent_msgs"] = ";".join(f"{k}:{','.join(v)}" for k, v in sorted(waived.items())) or "none"
    return dict(seg=seg, t=grid, v=v, a=a, sa=sa, sr=sr, tq=tq, yaw=yaw, gas=gas, brake=brake, sp=sp, ss=ss, das=das,
                lv=lv, ld=ld, lvr=lvr, qc=qc, platform=folded, raw_fp=raw_fp, lead_src=lead_src, yaw_src=yaw_src)


# ----------------------------------------------------------------------------- one route (several segments) -> windows
def process_route(task: tuple) -> dict:
    """task = (dongle, route, [(seg, path), ...]) -> dict(X, meta, qc rows, skipped)."""
    dongle, route, segs = task
    out = dict(X=None, meta=None, qc=[], skipped=[])
    try:
        frames = []
        for seg, path in sorted(segs):
            try:
                r = decode_segment(path)
            except Exception as ex:                                 # one bad file must not kill the route
                r = dict(skip=f"exception {type(ex).__name__}: {str(ex)[:120]}")
            if "skip" in r:
                out["skipped"].append(dict(seg=seg, reason=r["skip"], platform=r.get("platform")))
                continue
            r["seg"] = seg if r["seg"] < 0 else r["seg"]
            frames.append(r)
        if not frames:
            return out
        plats = {f["platform"] for f in frames}
        if len(plats) > 1:                                          # a route always has one car; keep the majority platform
            top = max(plats, key=lambda p: sum(f["platform"] == p for f in frames))
            for f in frames:
                if f["platform"] != top:
                    out["skipped"].append(dict(seg=f["seg"], reason=f"platform {f['platform']} differs from the route's {top}", platform=f["platform"]))
            frames = [f for f in frames if f["platform"] == top]
        platform = frames[0]["platform"]
        lab = _W["labeller"].label(platform, frames[0]["raw_fp"])
        frames.sort(key=lambda f: f["seg"])
        cat = lambda k: np.concatenate([f[k] for f in frames])  # noqa: E731
        t_s = np.concatenate([f["seg"] * SEG_S + np.arange(GRID_N) / T.HZ for f in frames])
        v, tq, sp = cat("v"), cat("tq"), cat("sp")
        scale, tq_norm = T.torque_norm(tq, v, sp > 0.5)
        torque_ok = math.isfinite(scale) and (tq != 0).any()
        yaw = cat("yaw")
        curv, lat = T.curvature_and_lat_acc(v, yaw)
        lead = T.lead_channels(v, cat("lv"), cat("ld"), cat("lvr"), t=t_s)
        has_lead_src = any(f["lead_src"] != "none" for f in frames)
        has_yaw = any(f["yaw_src"] != "none" for f in frames)
        df = pd.DataFrame(dict(t_s=t_s, v_mps=v, a_mps2=cat("a"), steer_angle_deg=cat("sa"), steer_rate_degps=cat("sr"),
                               steer_torque_norm=tq_norm, yaw_rate_radps=yaw, curv_1pm=curv, lat_acc_mps2=lat, gas_pressed=cat("gas"),
                               brake_pressed=cat("brake"), steer_pressed=sp, standstill=cat("ss"), das_enabled=cat("das"), **lead))
        cmap = {c: c for c in T.CHANNELS}
        if not torque_ok:
            cmap["steer_torque_norm"] = None
        if not has_yaw:
            for c in ("yaw_rate_radps", "curv_1pm", "lat_acc_mps2"):
                cmap[c] = None
        if not has_lead_src:                                        # vision-only car: lead_valid = 0, the lead values are absent
            for c in T.LEAD_CHANNELS[1:]:
                cmap[c] = None
        vid, did = P.pseudonym("v", dongle), P.pseudonym("d", dongle, route)
        info = dict(dataset=_W["dataset"], vehicle_id=vid, model=lab["model"] or platform, platform=platform, drive_id=did,
                    footprint_m2=lab["footprint_m2"], size_class=lab["size_class"], powertrain=lab["powertrain"],
                    borderline=lab["borderline"], note="")
        res = T.cut_windows(df, cmap, t_col="t_s", info=info, align="drive")
        meta = res.meta.copy()
        by_seg = {f["seg"]: f for f in frames}
        notes, widx = [], []
        for t0 in meta["t0_s"].to_numpy():
            seg = int(round(t0 / SEG_S))
            f = by_seg.get(seg)
            q = f["qc"] if f else {}
            widx.append(seg)
            notes.append(";".join([f"stratum={lab['stratum']}", f"label={lab['label_source']}", f"lead={q.get('lead_source', 'none')}",
                                   f"yaw={q.get('yaw_source', 'none')}"]
                                  + ([f"k={_W['understeer_k']:g}"] if q.get("yaw_source") == "derived-us" else [])
                                  + [f"torque={'route-q95' if torque_ok else 'absent'}",
                                   f"can_valid={q.get('can_valid_frac', float('nan')):.2f}", f"ctrl_allowed={q.get('controls_allowed_frac', float('nan')):.2f}",
                                   f"cruise={q.get('cruise_frac', float('nan')):.2f}"]
                                  + ([f"gearbox={q['gearbox']}"] if q.get("gearbox") else [])
                                  + ([f"cp={q['cp_source']}"] if q.get("cp_source") not in (None, "logged") else [])
                                  + ([f"das={q['das_source']}", f"ctrl_enabled={q.get('controls_enabled_frac', float('nan')):.2f}"] if q.get("das_source") not in (None, "panda") else [])
                                  + ([lab["note"]] if lab["note"] else [])))
        meta["w_idx"] = np.asarray(widx, dtype=np.int32)
        meta["note"] = notes
        for f in frames:
            q = dict(f["qc"])
            q.update(vehicle_id=vid, drive_id=did, model=lab["model"], stratum=lab["stratum"], torque_scale=(round(scale, 2) if torque_ok else None),
                     window=bool(f["seg"] in set(widx)))
            q.pop("raw_fp", None)
            out["qc"].append(q)
        out["X"], out["meta"] = res.X, meta
        out["stats"] = res.stats
    except Exception:
        out["skipped"].append(dict(seg=-1, reason="route exception: " + traceback.format_exc().splitlines()[-1][:160], platform=None))
    return out


def _init(cereal_dir: str, yaw_mode: str, use_scc: bool, scc_sign: float, dataset: str, honda_gearbox: bool = False,
          das_from: str = "panda", fp_map: dict | None = None, ignore_absent: bool = False, understeer_k: float | None = None) -> None:
    init_worker(cereal_dir, yaw_mode, use_scc, scc_sign, honda_gearbox, das_from, fp_map, ignore_absent, understeer_k)
    _W["dataset"] = dataset


# ----------------------------------------------------------------------------- driver
def find_segments(root: Path, list_file: Path | None, layout: str = "ccs") -> list[tuple[str, str, list[tuple[int, str]]]]:
    """[(dongle, route, [(seg, path)])] of the log files under root (restricted to a list of <dir>/<dir>/<seg> lines).
    layout ccs: root/<dongle>/<route>/<seg>/rlog.zst|rlog.bz2; layout comma2k19: root/Chunk_N/<dongle>|<route>/<seg>/raw_log.bz2
    (dongle = the part before '|', route = the whole folder name)."""
    want = None
    if list_file:
        want = {ln.strip().rstrip("/") for ln in Path(list_file).read_text().splitlines() if ln.strip()}
    by_route: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    names = ("raw_log.bz2",) if layout == "comma2k19" else ("rlog.zst", "rlog.bz2")
    for p in [q for n in names for q in glob.glob(str(root / "*" / "*" / "*" / n))]:
        pp = Path(p)
        seg_dir, route_dir, top_dir = pp.parent, pp.parent.parent, pp.parent.parent.parent
        key = f"{top_dir.name}/{route_dir.name}/{seg_dir.name}"
        if want is not None and key not in want:
            continue
        if not seg_dir.name.isdigit():
            continue
        if layout == "comma2k19":
            dongle, route = route_dir.name.split("|", 1)[0], route_dir.name
        else:
            dongle, route = top_dir.name, route_dir.name
        by_route[(dongle, route)].append((int(seg_dir.name), p))
    return [(d, r, sorted(v)) for (d, r), v in sorted(by_route.items())]


def parse_fingerprint_map(spec: str | None) -> dict[str, str]:
    """'Chunk_1=TOYOTA_RAV4,Chunk_3=HONDA_CIVIC' -> {folder name: opendbc platform}."""
    out: dict[str, str] = {}
    for item in (spec or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--fingerprint-map: expected <folder>=<PLATFORM>, got {item!r}")
        k, v = (x.strip() for x in item.split("=", 1))
        out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments-root", type=Path, required=True, help="folder holding <dongle>/<route>/<seg>/rlog.zst")
    ap.add_argument("--list", type=Path, default=None, help="restrict to the <dongle>/<route>/<seg> lines of this file (select.txt)")
    ap.add_argument("--dataset", default="ccs", help="dataset name for common.save_windows (ccs, ccs_sample, ...)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N routes (smoke)")
    ap.add_argument("--yaw", choices=YAW_MODES, default="derived",
                    help="derived = kinematic bicycle model from steering (default); measured = CarState.yawRate where a brand provides it, else derived; none = yaw group absent; "
                         "derived-understeer = the bicycle yaw divided by 1 + K v^2 (needs --understeer-k; module docstring)")
    ap.add_argument("--understeer-k", type=float, default=None, metavar="K",
                    help="understeer factor K in s^2/m^2 of --yaw derived-understeer (one global value, e.g. 0.001 = characteristic speed 32 m/s); required for that mode, refused otherwise")
    ap.add_argument("--no-scc-lead", action="store_true", help="do not use the stock ACC object of radar-less Hyundai / Kia / Genesis as lead")
    ap.add_argument("--scc-sign", type=float, default=1.0, choices=(1.0, -1.0),
                    help="sign applied to ACC_ObjRelSpd (default +1 = lead minus ego; check the scc_sign_corr column of segments_qc.csv)")
    ap.add_argument("--honda-gearbox-from-can", action="store_true",
                    help="Honda: set CarParams.transmissionType from the gearbox message seen on the pt bus (0x191 cvt / 0x1A3 automatic) before building "
                         "the interfaces; recovers the traditional-CVT platforms that opendbc 0.3.1 decodes with canValid 0 (module docstring)")
    ap.add_argument("--layout", choices=LAYOUTS, default="ccs",
                    help="ccs (default): <root>/<dongle>/<route>/<seg>/rlog.zst; comma2k19: <root>/Chunk_N/<dongle>|<route>/<seg>/raw_log.bz2")
    ap.add_argument("--fingerprint-map", default=None, metavar="FOLDER=PLATFORM,...",
                    help="build the CarParams of every segment under the named folders with opendbc 0.3.1 CarInterface.get_params from the platform and the "
                         "segment's live fingerprint (routes whose carParams event sits in a segment that is not present; module docstring)")
    ap.add_argument("--das-from", choices=DAS_SOURCES, default="panda",
                    help="second term of das_enabled next to cruiseState.enabled: panda = pandaStates.controlsAllowed (default), controlsState = controlsState.enabled")
    ap.add_argument("--ignore-absent-messages", action="store_true",
                    help="registered CarState parser messages that never appear on their bus in the segment's live fingerprint do not hold canValid at 0 "
                         "(2018 logs without the camera bus; the waived names go to segments_qc.csv absent_msgs; module docstring)")
    ap.add_argument("--cereal-dir", type=Path, default=CEREAL_DIR, help="directory with cereal log.capnp (openpilot checkout)")
    ap.add_argument("--copy-sample-to", type=Path, default=None, help="also copy the result as sample_windows.npz / sample_meta.parquet into this folder")
    ap.add_argument("--root", type=Path, default=None, help="external data root for save_windows (default: common.EXTERNAL_ROOT)")
    a = ap.parse_args(argv)

    if a.yaw == YAW_UNDERSTEER and not (a.understeer_k is not None and a.understeer_k > 0):
        ap.error("--yaw derived-understeer needs --understeer-k K (s^2/m^2, > 0)")
    if a.yaw != YAW_UNDERSTEER and a.understeer_k is not None:
        ap.error("--understeer-k applies to --yaw derived-understeer only")
    fp_map = parse_fingerprint_map(a.fingerprint_map)
    tasks = find_segments(a.segments_root, a.list, a.layout)
    if a.limit:
        tasks = tasks[:a.limit]
    n_seg = sum(len(t[2]) for t in tasks)
    print(f"[ccs_decode] {n_seg} segments in {len(tasks)} routes under {a.segments_root} -> dataset {a.dataset!r}; workers={a.workers} yaw={a.yaw} scc_lead={not a.no_scc_lead}"
          + (f" understeer_k={a.understeer_k:g}" if a.yaw == YAW_UNDERSTEER else "")
          + (" honda_gearbox_from_can=True" if a.honda_gearbox_from_can else "")
          + (f" layout={a.layout} das_from={a.das_from} fingerprint_map={fp_map} ignore_absent_messages={a.ignore_absent_messages}"
             if (a.layout != "ccs" or a.das_from != "panda" or fp_map or a.ignore_absent_messages) else ""), flush=True)
    if not tasks:
        return 2
    t0 = time.time()
    results = []
    init_args = (str(a.cereal_dir), a.yaw, not a.no_scc_lead, a.scc_sign, a.dataset, a.honda_gearbox_from_can, a.das_from, fp_map, a.ignore_absent_messages,
                 a.understeer_k)
    if a.workers <= 1:
        _init(*init_args)
        for i, task in enumerate(tasks, 1):
            results.append(process_route(task))
            if i % 50 == 0 or i == len(tasks):
                print(f"  routes {i}/{len(tasks)}  windows so far {sum(r['X'].shape[0] for r in results if r['X'] is not None)}  {time.time() - t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=init_args) as ex:
            for i, r in enumerate(ex.map(process_route, tasks, chunksize=1), 1):
                results.append(r)
                if i % 50 == 0 or i == len(tasks):
                    print(f"  routes {i}/{len(tasks)}  windows so far {sum(r['X'].shape[0] for r in results if r['X'] is not None)}  {time.time() - t0:.0f}s", flush=True)
    Xs = [r["X"] for r in results if r["X"] is not None and r["X"].shape[0]]
    metas = [r["meta"] for r in results if r["meta"] is not None and len(r["meta"])]
    qc = pd.DataFrame([q for r in results for q in r["qc"]])
    skipped = [s for r in results for s in r["skipped"]]
    wd = T.work_dir(a.dataset, a.root)
    wd.mkdir(parents=True, exist_ok=True)
    if len(qc):
        qc.to_csv(wd / "segments_qc.csv", index=False)
    n_windows = 0
    saved = None
    if Xs:
        X = np.concatenate(Xs)
        meta = pd.concat(metas, ignore_index=True)
        saved = T.save_windows(a.dataset, X, meta, root=a.root)
        n_windows = int(X.shape[0])
        if a.copy_sample_to:
            a.copy_sample_to.mkdir(parents=True, exist_ok=True)
            shutil.copy2(saved["npz"], a.copy_sample_to / "sample_windows.npz")
            shutil.copy2(saved["parquet"], a.copy_sample_to / "sample_meta.parquet")
            print(f"[ccs_decode] copied -> {a.copy_sample_to / 'sample_windows.npz'} (+ sample_meta.parquet)")
    summary = dict(dataset=a.dataset, segments_found=n_seg, routes=len(tasks), windows=n_windows, segments_decoded=int(len(qc)),
                   skipped=len(skipped), skip_reasons=pd.Series([s["reason"].split(" (")[0] for s in skipped]).value_counts().to_dict() if skipped else {},
                   seconds=round(time.time() - t0, 1), yaw=a.yaw, scc_lead=not a.no_scc_lead, scc_sign=a.scc_sign,
                   scc_sign_corr_median=(float(qc["scc_sign_corr"].median()) if len(qc) and qc["scc_sign_corr"].notna().any() else None), workers=a.workers,
                   per_platform=(qc.groupby("platform").agg(segments=("seg", "size"), windows=("window", "sum"), vehicles=("vehicle_id", "nunique"),
                                                            v_max=("v_max", "max"), moving_frac=("moving_frac", "mean"), gas=("gas_frac", "mean"),
                                                            brake=("brake_frac", "mean"), steer_pressed=("steer_pressed_frac", "mean"), das=("das_frac", "mean"),
                                                            lead_valid=("lead_valid_frac", "mean"), can_valid=("can_valid_frac", "mean"),
                                                            decode_s=("decode_s", "mean")).round(3).reset_index().to_dict(orient="records") if len(qc) else []))
    if a.yaw == YAW_UNDERSTEER:
        summary.update(understeer_k=a.understeer_k, yaw_formula="v tan(steer_angle / steerRatio) / (wheelbase (1 + K v^2))",
                       yaw_sources=({str(k): int(v) for k, v in qc["yaw_source"].value_counts(dropna=False).items()} if len(qc) and "yaw_source" in qc else {}))
    if a.honda_gearbox_from_can:
        summary["honda_gearbox_from_can"] = True
        summary["gearbox_counts"] = ({str(k): int(v) for k, v in qc["gearbox"].value_counts(dropna=False).items()} if len(qc) and "gearbox" in qc else {})
    if a.layout != "ccs" or a.das_from != "panda" or fp_map or a.ignore_absent_messages:
        summary.update(layout=a.layout, das_from=a.das_from, fingerprint_map=fp_map, ignore_absent_messages=a.ignore_absent_messages,
                       absent_msgs=({str(k): int(v) for k, v in qc["absent_msgs"].value_counts(dropna=False).items()} if len(qc) and "absent_msgs" in qc else {}),
                       cp_sources=({str(k): int(v) for k, v in qc["cp_source"].value_counts(dropna=False).items()} if len(qc) and "cp_source" in qc else {}),
                       cp_logged_diffs=({str(k): int(v) for k, v in qc["cp_logged_diff"].value_counts(dropna=False).items()} if len(qc) and "cp_logged_diff" in qc else {}),
                       gearbox_counts=({str(k): int(v) for k, v in qc["gearbox"].value_counts(dropna=False).items()} if len(qc) and "gearbox" in qc else {}),
                       controls_enabled_mean=(round(float(qc["controls_enabled_frac"].mean()), 3) if len(qc) and "controls_enabled_frac" in qc else None))
    (wd / "decode_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    # ---- validation printout
    print(f"[ccs_decode] decoded {len(qc)} segments -> {n_windows} windows; skipped {len(skipped)}; {summary['seconds']} s")
    for s in skipped[:20]:
        print(f"    skipped seg {s['seg']} ({s.get('platform')}): {s['reason']}")
    if len(qc):
        cols = ["platform", "seg", "n_can", "span_s", "can_valid_frac", "v_min", "v_max", "moving_frac", "a_min", "a_max", "steer_min", "steer_max",
                "gas_frac", "brake_frac", "steer_pressed_frac", "standstill_frac", "cruise_frac", "controls_allowed_frac", "das_frac",
                "lead_source", "lead_valid_frac", "scc_sign_corr", "yaw_source", "torque_scale", "window", "decode_s"]
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(qc[cols].to_string(index=False))
    if saved is not None:
        meta = pd.read_parquet(saved["parquet"])
        print("[ccs_decode] channels present per platform:")
        for plat, g in meta.groupby("platform"):
            chans = sorted({c for s in g["channels_present"] for c in s.split(",") if c}, key=T.CH.get)
            absent = [c for c in T.CHANNELS if c not in chans]
            print(f"    {plat}: {len(g)} windows, model={g['model'].iloc[0]}, {g['size_class'].iloc[0]}/{g['powertrain'].iloc[0]}, absent channels: {absent or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
