"""Minimal synthetic openpilot qlog writer for the VTT S2 tests.

Writes capnp Event streams through the parse schema of the reused extractor (tocf.capnp_io.schema): initData + carParams at
the head of every segment, then carState (10 Hz), controlsState / carControl / selfdriveState (10 Hz), radarState (4 Hz) and
livePose (5 Hz) at loggerd's qlog rates. No video, no tocf.video, no real identifiers: dongle ids are 'f' * 10 + 6 hex digits,
route ids are counter-style '<8 hex>--<10 hex>' built from small integers. Nothing here touches the raw library.
Units: times in ns on a fake boot clock, speeds m/s, engineRpm rpm, fuelGauge 0..1, yawRate rad/s.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import zstandard

from vtt.tocf_bridge import tocf

K = tocf("capnp_io")
RAWLIB = tocf("rawlib")
TSCHEMA = tocf("schema")

SEG_S = 60.0                                   # one real-length segment (the acceptance judges >= 55 s segments)
T0_NS = 1_000_000_000_000                      # fake boot clock at route start (1000 s)
HZ = {"carState": 10, "controlsState": 10, "carControl": 10, "selfdriveState": 10, "radarState": 4, "livePose": 5}
OFFSET_NS = {t: (i + 1) * 1_000 for i, t in enumerate(HZ)}    # distinct logMonoTime per topic
UNKNOWN_DISCRIMINANT = 60000                   # a union member the parse schema does not know
TEST_SALT = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


def fake_dongle(n: int) -> str:
    """16-hex fake dongle id that can never be a real one ('ffffffffff' prefix + counter)."""
    return "f" * 10 + f"{n:06x}"


def fake_route(n: int) -> str:
    """Counter-style fake route id (matches rawlib's 'counter' pattern)."""
    return f"{n:08x}--{n:010x}"


def new_event(t: int, which: str):
    ev = K.schema().Event.new_message()
    ev.logMonoTime = int(t)
    return ev, ev.init(which)


def unknown_bytes(t: int) -> bytes:
    """A valid carState event whose union discriminant is patched to UNKNOWN_DISCRIMINANT (counted, never parsed)."""
    ev, m = new_event(t, "carState")
    m.vEgo = 1.0
    raw = bytearray(ev.to_bytes())
    disc = K.schema().Event.schema.node.struct.discriminantOffset
    hdr, ptr = K._root(bytes(raw))
    data_start = hdr + 8 + 8 * ((ptr >> 2) & 0x3FFFFFFF)
    struct.pack_into("<H", raw, data_start + 2 * disc, UNKNOWN_DISCRIMINANT)
    return bytes(raw)


# ----------------------------------------------------------------------------- vehicle profiles (what the audit probe sees)
PROFILES = {
    # engine_rpm: rpm while moving (0 = electric); regen / charging flags; fuel 0..1; gear names cycled; ecus; transmission
    "ice": dict(engine_rpm=1800.0, regen=False, charging=False, fuel=0.6, gears=("park", "drive"),
                ecus=("engine", "transmission", "eps", "fwdRadar"), transmission="automatic", mass=1600.0, wheelbase=2.70, steer_ratio=15.0),
    "bev": dict(engine_rpm=0.0, regen=True, charging=False, fuel=0.8, gears=("drive",),
                ecus=("eps", "fwdCamera"), transmission="direct", mass=2100.0, wheelbase=3.00, steer_ratio=13.5),
    "hev": dict(engine_rpm=1200.0, regen=True, charging=False, fuel=0.5, gears=("drive", "park"),
                ecus=("engine", "hybrid", "eps", "fwdCamera"), transmission="cvt", mass=1500.0, wheelbase=2.60, steer_ratio=14.0),
}


def _fill(topic: str, m, k: int, prof: dict, v: float):
    on = (k % 200) < 100
    if topic == "carState":
        m.vEgo, m.vEgoRaw, m.aEgo, m.standstill = v, v, 0.1, v < 0.1
        m.steeringAngleDeg, m.steeringRateDeg, m.steeringTorque, m.steeringPressed = float(k % 7), 0.5, 10.0 * (k % 3), (k % 3) == 0
        m.steeringTorqueEps = 5.0 if (k % 2) else 0.0
        m.gas, m.gasPressed, m.brake, m.brakePressed = 0.2, (k % 50) == 0, 0.0, (k % 90) == 0
        m.cruiseState.enabled, m.cruiseState.available, m.cruiseState.speed = on, True, 25.0
        m.vCruise, m.vCruiseCluster = 90.0, 92.0
        m.engineRpm = prof["engine_rpm"] if v > 0.1 else 0.0
        m.regenBraking = bool(prof["regen"] and (k % 10) == 0)
        m.charging = bool(prof["charging"])
        m.fuelGauge = float(prof["fuel"])
        m.gearShifter = prof["gears"][k % len(prof["gears"])]
        m.yawRate = 0.01 if (k % 4) else 0.0
    elif topic == "controlsState":
        m.enabledDEPRECATED, m.activeDEPRECATED, m.stateDEPRECATED = on, on, "enabled" if on else "disabled"
    elif topic == "selfdriveState":
        m.enabled, m.active, m.state, m.personality = on, on, "enabled" if on else "disabled", "standard"
    elif topic == "carControl":
        m.enabled, m.latActive, m.longActive = on, on, False
    elif topic == "radarState":
        l1 = m.leadOne
        l1.status, l1.dRel, l1.vRel, l1.yRel, l1.vLead = True, 30.0 - k * 0.01, -1.0, 0.2, v - 1.0
        l1.radar, l1.radarTrackId, l1.modelProb = True, 7, 0.9
    elif topic == "livePose":
        av = m.angularVelocityDevice
        av.x, av.y, av.z, av.valid = 0.001, -0.002, float(0.05 * np.sin(2 * np.pi * k / 40.0)), (k % 50) != 7
        m.inputsOK, m.posenetOK = True, True


def head_bytes(t0: int, dongle_log: str, version: str, car_fp: str, prof: dict, vin: str = "") -> list[bytes]:
    ev, i = new_event(T0_NS - 500_000_000, "initData")
    i.dongleId, i.version, i.deviceType, i.gitCommit = dongle_log, version, "tici", "0000000000000000000000000000000000000000"
    ev2, cp = new_event(t0 + 100_000, "carParams")
    cp.carFingerprint, cp.carVin, cp.openpilotLongitudinalControl = car_fp, vin, False
    cp.alternativeExperience, cp.pcmCruise, cp.radarUnavailable = 1, True, False
    cp.mass, cp.wheelbase, cp.steerRatio = prof["mass"], prof["wheelbase"], prof["steer_ratio"]
    cp.transmissionType = prof["transmission"]
    fw = cp.init("carFw", len(prof["ecus"]))
    for j, e in enumerate(prof["ecus"]):
        fw[j].ecu = e
    return [ev.to_bytes(), ev2.to_bytes()]


def segment_bytes(seg: int, seg_s: float, prof: dict, topics=None, unknown_n: int = 0, v_mps: float = 15.0) -> bytes:
    """All messages of one segment in time order (head first). topics: subset of HZ to write (default all)."""
    t0 = T0_NS + int(seg * SEG_S * 1e9)
    msgs = []
    for topic, hz in HZ.items():
        if topics is not None and topic not in topics:
            continue
        for k in range(int(round(seg_s * hz))):
            t = t0 + int(k * 1e9 / hz) + OFFSET_NS[topic]
            ev, m = new_event(t, topic)
            _fill(topic, m, k + seg * 1000, prof, v_mps)
            msgs.append((t, ev.to_bytes()))
    for j in range(unknown_n):
        msgs.append((t0 + 10 + j, unknown_bytes(t0 + 10 + j)))
    msgs.sort(key=lambda x: x[0])
    return b"".join(b for _, b in msgs)


def write_route(route_dir: Path, dongle_log: str, car_fp: str, n_seg: int, profile: str = "ice", version: str = "0.9.7",
                last_seg_s: float = SEG_S, topics=None, unknown_n: int = 0, truncate_seg: int | None = None,
                zst_segs=(), v_mps: float = 15.0, vin: str = "") -> dict:
    """<route_dir>/<n>--qlog[.zst] for n in range(n_seg). last_seg_s shortens the last segment (a partial minute);
    truncate_seg cuts that segment's file at 60 % (an interrupted upload). Returns {seg: seconds}."""
    route_dir.mkdir(parents=True, exist_ok=True)
    prof = PROFILES[profile]
    out = {}
    for seg in range(n_seg):
        seg_s = last_seg_s if seg == n_seg - 1 else SEG_S
        t0 = T0_NS + int(seg * SEG_S * 1e9)
        blob = b"".join(head_bytes(t0, dongle_log, version, car_fp, prof, vin)) + segment_bytes(seg, seg_s, prof, topics, unknown_n if seg == 0 else 0, v_mps)
        if truncate_seg == seg:
            blob = blob[: int(len(blob) * 0.6)]
        if seg in zst_segs:
            (route_dir / f"{seg}--qlog.zst").write_bytes(zstandard.ZstdCompressor().compress(blob))
        else:
            (route_dir / f"{seg}--qlog").write_bytes(blob)
        out[seg] = seg_s
    return out


# ----------------------------------------------------------------------------- the private side (what the extractor reads)
def write_private(private_root: Path, routes: list[dict], salt: str = TEST_SALT) -> pd.DataFrame:
    """private_root/{salt.txt, uid_map.parquet} (tocf.schema.PRIVATE_UID_MAP columns) for routes
    [{car_dir, dongle_dir, dongle_log, route_id, vin?}]. Returns the uid_map frame."""
    private_root.mkdir(parents=True, exist_ok=True)
    (private_root / "salt.txt").write_text(salt)
    rows = []
    for r in routes:
        rows.append({"route_uid": RAWLIB.route_uid(salt, r["dongle_log"], r["route_id"]), "car_dir": r["car_dir"],
                     "dongle_dir": r["dongle_dir"], "dongle_log": r["dongle_log"], "route_id": r["route_id"],
                     "identity_uid": RAWLIB.identity_uid(salt, r["dongle_log"]), "vin_hash": RAWLIB.vin_hash(salt, r.get("vin"))})
    um = pd.DataFrame(rows, columns=list(TSCHEMA.PRIVATE_UID_MAP))
    um.to_parquet(private_root / "uid_map.parquet", index=False)
    return um
