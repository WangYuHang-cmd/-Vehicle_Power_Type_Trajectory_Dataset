"""Synthetic raw library for the S1 tests: qlog-only routes written with pycapnp through the tocf parse schema
(initData + sentinel + carParams + a few carState messages per segment; no video, no real identifiers).

Layout <car>/<dongle folder>/<route>/<n>--qlog[.zst] | <n>--rlog, with the library's quirks that S1 must handle: two
fingerprints, a misfiled extra copy of a route under the other car dir, a route under a non-hex (person-name) folder
whose logs carry the device id, a non-hex folder with no readable initData (unresolved), a 16-hex folder holding another
device's logs, a route whose segment has an rlog but no qlog, the same route id under two dongle folders with identical
content (no initData -> both fall back to their folder) and with different content, a date-style id, a route longer than
config.LONG_ROUTE_SEGMENTS, a 0-byte log and a segment-less junk dir. Every id is generated here (fake_dongle), nothing is
copied from the real library. Artefacts go under the directory the caller passes (tests use $D/tmp)."""
from __future__ import annotations

from pathlib import Path

import zstandard

from vtt import config as C
from vtt.tocf_bridge import tocf

CAR_A, CAR_B = "ACME_ALPHA", "ACME_BETA"
NAME_X, NAME_Y = "personx", "persony"          # non-hex dongle folders
ROUTE_T0_NS = 1_000_000_000_000


def fake_dongle(i: int) -> str:
    """A 16-hex id that cannot be a real one: ten 'f' + 6 hex digits."""
    return "f" * 10 + f"{i:06x}"


def counter_route(i: int) -> str:
    return f"{i:08x}--{'ab' * 5}"


def date_route(day: int) -> str:
    return f"2024-01-{day:02d}--10-00-00"


def _event(t_ns: int, which: str):
    K = tocf("capnp_io")
    ev = K.schema().Event.new_message()
    ev.logMonoTime = int(t_ns)
    return ev, ev.init(which)


def qlog_bytes(seg: int, dongle_log: str | None, car_fp: str, version: str = "0.9.7", device_type: str = "tizi",
               vin: str = "", n_carstate: int = 20, with_carparams: bool = True, salt_byte: int = 0) -> bytes:
    """One segment log. dongle_log None writes NO initData (probe -> no_initdata, dongle falls back to the folder);
    salt_byte changes the carState values so two logs differ in content."""
    t0 = ROUTE_T0_NS + seg * 60_000_000_000
    msgs = []
    if dongle_log is not None:
        ev, i = _event(ROUTE_T0_NS - 500_000_000, "initData")
        i.dongleId, i.version, i.deviceType, i.gitCommit = dongle_log, version, device_type, "0000000"
        msgs.append(ev.to_bytes())
    ev, s = _event(t0, "sentinel")
    s.type = "startOfRoute" if seg == 0 else "startOfSegment"
    msgs.append(ev.to_bytes())
    if with_carparams:
        ev, cp = _event(t0 + 100_000, "carParams")
        cp.carFingerprint, cp.carVin, cp.openpilotLongitudinalControl = car_fp, vin, False
        cp.alternativeExperience, cp.pcmCruise, cp.radarUnavailable = 1, True, False
        msgs.append(ev.to_bytes())
    for k in range(n_carstate):
        ev, cs = _event(t0 + 1_000_000 + k * 100_000_000, "carState")
        cs.vEgo, cs.aEgo = 10.0 + k * 0.1 + salt_byte, 0.1
        msgs.append(ev.to_bytes())
    return b"".join(msgs)


def write_route(route_dir: Path, n_seg: int, dongle_log: str | None, car_fp: str, version: str = "0.9.7",
                device_type: str = "tizi", vin: str = "", rlog_only: tuple[int, ...] = (), zst: tuple[int, ...] = (),
                empty: tuple[int, ...] = (), n_carstate: int = 20, with_carparams: bool = True, salt_byte: int = 0) -> None:
    """Route dir with n_seg segments: '<n>--qlog' (or .zst for segments in `zst`), '<n>--rlog' only for segments in
    `rlog_only`, a 0-byte qlog for segments in `empty`."""
    route_dir.mkdir(parents=True, exist_ok=True)
    for seg in range(n_seg):
        if seg in empty:
            (route_dir / f"{seg}--qlog").write_bytes(b"")
            continue
        blob = qlog_bytes(seg, dongle_log, car_fp, version, device_type, vin, n_carstate, with_carparams, salt_byte)
        if seg in rlog_only:
            (route_dir / f"{seg}--rlog").write_bytes(blob)
        elif seg in zst:
            (route_dir / f"{seg}--qlog.zst").write_bytes(zstandard.ZstdCompressor().compress(blob))
        else:
            (route_dir / f"{seg}--qlog").write_bytes(blob)


def build_tree(root: Path) -> dict:
    """Write the tree under root/raw and return ids + expectations (ids are fake; keep them out of any public file)."""
    raw = Path(root) / "raw"
    D = {k: fake_dongle(i) for i, k in enumerate(("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"), start=1)}
    R = {"R1": counter_route(1), "R2": date_route(2), "R3": counter_route(3), "R4": counter_route(4), "R5": counter_route(5),
         "R5b": counter_route(15), "R6": counter_route(6), "R7": counter_route(7), "R8": counter_route(8), "R9": counter_route(9),
         "R10": counter_route(10)}
    long_n = int(C.LONG_ROUTE_SEGMENTS) + 1
    # R1: CAR_A / D1, 3 segs (seg 2 zst) + misfiled extra copy under CAR_B (2 segs, fingerprint still CAR_A)
    write_route(raw / CAR_A / D["D1"] / R["R1"], 3, D["D1"], CAR_A, "0.9.7", "tizi", vin="VINA00001", zst=(2,))
    write_route(raw / CAR_B / D["D1"] / R["R1"], 2, D["D1"], CAR_A, "0.9.7", "tizi", vin="VINA00001")
    # R2: date-style id, 0.8.13 tici
    write_route(raw / CAR_A / D["D1"] / R["R2"], 2, D["D1"], CAR_A, "0.8.13", "tici", vin="VINA00001")
    # R3: CAR_B / D2, seg 2 rlog only, 0.10.0 mici
    write_route(raw / CAR_B / D["D2"] / R["R3"], 3, D["D2"], CAR_B, "0.10.0", "mici", rlog_only=(2,))
    # R4: non-hex folder, logs carry D3, calendar version 2026
    write_route(raw / CAR_B / NAME_X / R["R4"], 2, D["D3"], CAR_B, "2026.03.14", "tizi")
    # R5: non-hex folder with no initData anywhere -> unresolved; R5b: 0-byte log only -> log-less
    write_route(raw / CAR_B / NAME_Y / R["R5"], 1, None, CAR_B)
    write_route(raw / CAR_B / NAME_Y / R["R5b"], 1, None, CAR_B, empty=(0,))
    # R6: identical content (no initData) under two 16-hex folders D4 / D5 -> identical_copy, keep D4
    write_route(raw / CAR_A / D["D4"] / R["R6"], 2, None, CAR_A)
    write_route(raw / CAR_A / D["D5"] / R["R6"], 2, None, CAR_A)
    # R7: same route id under D6 (CAR_A) and D7 (CAR_B) with different content -> shared_route_id, both kept
    write_route(raw / CAR_A / D["D6"] / R["R7"], 2, D["D6"], CAR_A, "0.9.9", "tici", salt_byte=0)
    write_route(raw / CAR_B / D["D7"] / R["R7"], 3, D["D7"], CAR_B, "0.9.9", "tici", salt_byte=1)
    # R8: long route (minimal segments)
    write_route(raw / CAR_A / D["D1"] / R["R8"], long_n, D["D1"], CAR_A, "0.9.7", "tizi", n_carstate=1)
    # R9: 0.11 era
    write_route(raw / CAR_B / D["D2"] / R["R9"], 2, D["D2"], CAR_B, "0.11.0", "tici")
    # R10: 16-hex folder D8 holding device D9's logs (misfiled id)
    write_route(raw / CAR_B / D["D8"] / R["R10"], 2, D["D9"], CAR_B, "0.10.0", "tizi")
    # junk: segment-less dir
    (raw / CAR_A / D["D1"] / "rlogs").mkdir(parents=True, exist_ok=True)
    return {"raw": raw, "D": D, "R": R, "n_dirs": 15, "n_logless": 2, "n_uid_map": 10, "n_excluded_unresolved": 1,
            "long_n": long_n, "all_ids": list(D.values()) + list(R.values()) + [NAME_X, NAME_Y]}
