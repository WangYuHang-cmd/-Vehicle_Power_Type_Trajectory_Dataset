"""Synthetic extractor output for the S5 tests (not collected by pytest: the name does not end in _tests).

Frames are shaped like the ADAS-TO-CF writer: boot-clock logMonoTime (int64 ns, origin ~97 s after boot), a jittered
10-Hz carState with one gap, carControl / controlsState / selfdriveState offset by a few ms after the carState sample,
a 4-Hz radarState with one lead episode (lead accelerating at +1 m/s^2) and a stale stretch, a 5-Hz livePose stream
(or 4-Hz liveLocationKalman) with a stale stretch and a left turn (angularVelocityDevice_z negative), a dead gas channel
and a live brake channel. All identifiers are fake and generated here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

T0_NS = 97_312_456_789                 # boot clock origin of the first carState sample (ns)
N = 1200                               # carState samples (~120 s)
GAP_AT, GAP_S = 600, 1.7               # row with a 1.7-s hole before it
SEG_NUMBERS = (3, 7)                   # source segment numbers (ordinals must become 0, 1)
ENGAGE = (300, 700)                    # carControl / selfdriveState flags True for these carState indices
CC_DROP = (650, 700)                   # carControl samples removed -> stale flags
TURN = (400, 500)                      # left turn: steering +25 deg, yaw +YAW rad/s (z = -YAW)
YAW = 0.15
LEAD_T = (20.0, 60.0)                  # lead episode in real seconds from the first carState sample
LEAD_D0, LEAD_DDOT, LEAD_V0, LEAD_ACC = 40.0, -0.3, 8.0, 1.0     # lead starts 2 m/s slower than ego (closing), then pulls away
RADAR_DROP_T = (45.0, 50.0)            # radar samples removed -> stale lead
POSE_DROP_T = (70.0, 75.0)             # pose samples removed -> stale yaw
GAS_PRESSED = (60, 120)                # gasPressed fires while gas magnitude stays 0 -> dead channel
BRAKE = (1000, 1100)                   # brake 0.3 with brakePressed
BLINK = (380, 400)
VCRUISE_KPH = 100.0                    # set speed while engaged; 255 (unset sentinel) otherwise


def fake_uid(k: int) -> str:
    """12-hex internal route uid (fake)."""
    return "f" * 6 + "%06x" % k


def fake_dongle(k: int) -> str:
    """16-hex device id (fake)."""
    return "f" * 10 + "%06x" % k


def make_signals(seed: int = 0, n: int = N, pose_source: str = "livePose", radar_flag_nan: bool = False,
                 drop_topics: tuple = (), radar_unavailable: bool = False, version: str = "0.9.7"):
    """(sig, meta, truth): sig = {topic: raw frame}, meta = route_meta dict, truth = dict(t real seconds, v, k)."""
    rng = np.random.default_rng(seed)
    dt = 0.1 + rng.normal(0.0, 0.0015, n); dt[0] = 0.0; dt[GAP_AT] = GAP_S
    t = np.cumsum(dt)
    ns = T0_NS + np.round(t * 1e9).astype(np.int64)
    k = np.arange(n)
    v = np.zeros(n)
    ramp = (k >= 50) & (k < 350); v[ramp] = 20.0 * (k[ramp] - 50) / 300.0
    v[(k >= 350) & (k < 1000)] = 20.0
    dec = k >= 1000; v[dec] = np.maximum(0.0, 20.0 - 20.0 * (k[dec] - 1000) / 200.0)
    a = np.gradient(v, t)
    turn = (k >= TURN[0]) & (k < TURN[1])
    angle = np.where(turn, 25.0, 0.0) + rng.normal(0, 0.1, n)
    rate = np.gradient(angle, t)
    pressed = turn.copy()
    extra = rng.choice(np.flatnonzero((v >= 1.0) & ~turn), 100, replace=False); pressed[extra] = True
    torque = np.where(pressed, rng.uniform(0.5, 2.0, n) * np.where(angle > 0, 1.0, -1.0), rng.normal(0, 0.02, n))
    gas = np.zeros(n); gas_pressed = (k >= GAS_PRESSED[0]) & (k < GAS_PRESSED[1])
    brake_pressed = (k >= BRAKE[0]) & (k < BRAKE[1]); brake = np.where(brake_pressed, 0.3, 0.0)
    engaged = (k >= ENGAGE[0]) & (k < ENGAGE[1])
    seg = np.where(k < GAP_AT, SEG_NUMBERS[0], SEG_NUMBERS[1]).astype(np.int64)
    cs = pd.DataFrame(dict(
        logMonoTime=ns, seg=seg, vEgo=v, aEgo=a, vEgoRaw=v + rng.normal(0, 0.01, n), standstill=v < 0.05,
        steeringAngleDeg=angle, steeringRateDeg=rate, steeringTorque=torque, steeringPressed=pressed, gas=gas,
        gasPressed=gas_pressed, brake=brake, brakePressed=brake_pressed, leftBlinker=(k >= BLINK[0]) & (k < BLINK[1]),
        rightBlinker=np.zeros(n, bool), cruiseState_enabled=np.zeros(n, bool), cruiseState_available=k >= 50,
        cruiseState_speed=np.zeros(n), vCruise=np.where(engaged, VCRUISE_KPH, 255.0), vCruiseCluster=np.full(n, np.nan)))

    def ctrl(offset_ns: int, enabled, drop=None):
        keep = np.ones(n, bool)
        if drop is not None:
            keep[drop[0]:drop[1]] = False
        return pd.DataFrame(dict(logMonoTime=ns[keep] + offset_ns, seg=seg[keep], enabled=enabled[keep], active=enabled[keep],
                                 state=np.where(enabled[keep], "enabled", "disabled"), personality=np.full(keep.sum(), "standard"),
                                 experimentalMode=np.zeros(keep.sum(), bool), vCruise=cs["vCruise"].to_numpy()[keep],
                                 vCruiseCluster=np.full(keep.sum(), np.nan)))

    cc = ctrl(10_000_000, engaged, CC_DROP)
    cc = cc[["logMonoTime", "seg", "enabled"]].assign(latActive=cc["enabled"].to_numpy(), longActive=cc["enabled"].to_numpy())
    ctrl_state = ctrl(20_000_000, np.zeros(n, bool))          # dead controlsState.enabled (newer software)
    sds = ctrl(30_000_000, engaged)

    tr = np.arange(0.05, t[-1], 0.25)
    tr = tr[~((tr >= RADAR_DROP_T[0]) & (tr < RADAR_DROP_T[1]))]
    in_ep = (tr >= LEAD_T[0]) & (tr < LEAD_T[1]); tau = np.where(in_ep, tr - LEAD_T[0], 0.0)
    v_at = np.interp(tr, t, v)
    d = np.where(in_ep, LEAD_D0 + LEAD_DDOT * tau, 0.0)
    vlead = np.where(in_ep, LEAD_V0 + LEAD_ACC * tau, 0.0)
    vrel = np.where(in_ep, vlead - v_at, 0.0)
    m = len(tr)
    radar_flag = in_ep.copy(); tid = np.where(in_ep, 7, -1).astype(np.int64)
    rs = pd.DataFrame(dict(
        logMonoTime=T0_NS + np.round(tr * 1e9).astype(np.int64), seg=np.where(tr < t[GAP_AT], SEG_NUMBERS[0], SEG_NUMBERS[1]).astype(np.int64),
        leadOne_status=in_ep, leadOne_dRel=d, leadOne_yRel=np.where(in_ep, 0.2, 0.0), leadOne_vRel=vrel, leadOne_aRel=np.zeros(m),
        leadOne_vLead=vlead, leadOne_radar=radar_flag, leadOne_radarTrackId=tid, leadOne_modelProb=np.where(in_ep, 0.95, 0.1),
        leadTwo_status=np.zeros(m, bool), leadTwo_dRel=np.zeros(m), leadTwo_yRel=np.zeros(m), leadTwo_vRel=np.zeros(m),
        leadTwo_radar=np.zeros(m, bool), leadTwo_radarTrackId=np.full(m, -1, np.int64)))
    if radar_flag_nan:                                         # dtype drift of older schemas: all-missing -> float64 NaN
        for c in ("leadOne_radar", "leadOne_radarTrackId", "leadTwo_radar", "leadTwo_radarTrackId"):
            rs[c] = np.full(m, np.nan)

    step = 0.2 if pose_source == "livePose" else 0.25
    tp = np.arange(0.03, t[-1], step)
    tp = tp[~((tp >= POSE_DROP_T[0]) & (tp < POSE_DROP_T[1]))]
    yaw = np.where((tp >= t[TURN[0]]) & (tp < t[TURN[1]]), YAW, 0.0) + rng.normal(0, 1e-4, len(tp))
    pose = pd.DataFrame(dict(
        logMonoTime=T0_NS + np.round(tp * 1e9).astype(np.int64), seg=np.where(tp < t[GAP_AT], SEG_NUMBERS[0], SEG_NUMBERS[1]).astype(np.int64),
        angularVelocityDevice_x=rng.normal(0, 1e-3, len(tp)), angularVelocityDevice_y=rng.normal(0, 1e-3, len(tp)),
        angularVelocityDevice_z=-yaw, angularVelocityDevice_valid=np.ones(len(tp), bool), inputsOK=np.ones(len(tp), bool),
        posenetOK=np.ones(len(tp), bool), pose_source=np.full(len(tp), pose_source)))

    sig = {"carState": cs, "carControl": cc, "controlsState": ctrl_state, "selfdriveState": sds, "radarState": rs, "pose": pose}
    for tpc in drop_topics:
        sig.pop(tpc, None)
    meta = dict(route_uid=fake_uid(seed), dongle_log=fake_dongle(seed), version=version, device_type="tici", git_commit="0" * 8,
                car_fp="SYNTH CAR", opLong=False, altExp=0, pcmCruise=True, radarUnavailable=radar_unavailable, vin_hash="0" * 40,
                n_seg=2, pose_source=pose_source if "pose" in sig else "none",
                segments=[dict(seg=SEG_NUMBERS[0], src_kind="qlog", unknown_union_n=5, t0_mono_ns=int(ns[0]), t1_mono_ns=int(ns[GAP_AT - 1])),
                          dict(seg=SEG_NUMBERS[1], src_kind="rlog_dec", unknown_union_n=2, t0_mono_ns=int(ns[GAP_AT]), t1_mono_ns=int(ns[-1]))],
                extractor_version="s02-v4", config_version="v1.19")
    truth = dict(t=t, v=v, k=k, ns=ns, engaged=engaged, turn=turn, cc_times=(cc["logMonoTime"].to_numpy() - T0_NS) / 1e9,
                 cc_enabled=cc["enabled"].to_numpy())
    return sig, meta, truth


def write_signals(sig: dict, meta: dict, sig_dir: Path) -> Path:
    """Write the frames and route_meta.json exactly like the extractor (pandas parquet, no index)."""
    sig_dir = Path(sig_dir); sig_dir.mkdir(parents=True, exist_ok=True)
    for topic, df in sig.items():
        df.to_parquet(sig_dir / f"{topic}.parquet", index=False)
    (sig_dir / "route_meta.json").write_text(json.dumps(meta, default=str))
    return sig_dir
