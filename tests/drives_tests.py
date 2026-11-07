"""vtt.drives + scripts/s05_build_drives.py on synthetic extractor output (tests/synth_signals.py)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from tests import synth_signals as SY
from vtt import config as C
from vtt import drives as D
from vtt import schema as S
from vtt import signals as SG

REPO = Path(__file__).resolve().parents[1]
TMP_BASE = Path(os.environ.get("VTT_DATA_ROOT", str(C.DATA_ROOT))) / "tmp"
HEX16 = re.compile(r"(?<![0-9a-zA-Z])[0-9a-f]{16}(?![0-9a-zA-Z])")


def _build(**kw):
    sig_raw, meta, truth = SY.make_signals(**kw)
    sig = {t: SG.normalise(df, t) for t, df in sig_raw.items()}
    frame, info = D.build_drive(sig, meta)
    return frame, info, truth, meta, sig_raw


@pytest.fixture(scope="module")
def built():
    return _build()


@pytest.fixture()
def tmp_root():
    TMP_BASE.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=TMP_BASE, prefix="s5test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------------------------------------- timeline
def test_rows_nominal_time_segments_and_gap(built):
    frame, info, truth, _, _ = built
    assert len(frame) == SY.N == info["n_rows"]
    assert list(frame.columns) == S.names(S.DRIVE_COLUMNS)
    assert np.array_equal(frame["t_s"].to_numpy(), np.arange(SY.N) / 10.0)
    seg = frame["seg"].to_numpy()
    assert (seg[:SY.GAP_AT] == 0).all() and (seg[SY.GAP_AT:] == 1).all() and info["n_segments"] == 2
    gap = frame["gap"].to_numpy()
    assert gap.sum() == 1 and gap[SY.GAP_AT] and not gap[0]
    assert info["n_gaps"] == 1 and abs(info["max_gap_s"] - SY.GAP_S) < 1e-6
    assert abs(info["dt_med_s"] - 0.1) < 0.002 and abs(info["drift_s"]) < 0.5 and info["n_seg_rlog_dec"] == 1
    assert info["unknown_union_n"] == 7 and info["n_seg_meta"] == 2


def test_zoh_is_causal_and_bounded_by_max_age(built):
    frame, _, truth, _, _ = built
    lat = frame["op_lat_active"].to_numpy()
    assert not lat[SY.ENGAGE[0]] and lat[SY.ENGAGE[0] + 1]          # the carControl sample 10 ms after the row is not used
    held = np.ones(SY.N, bool); held[SY.GAP_AT] = False                # at the gap row the last sample is 1.7 s old -> stale
    assert lat[SY.ENGAGE[0] + 1:SY.CC_DROP[0] + 2][held[SY.ENGAGE[0] + 1:SY.CC_DROP[0] + 2]].all()   # held while <= 0.3 s old
    assert not lat[SY.GAP_AT]
    assert not lat[SY.CC_DROP[0] + 4:].any()                           # stale (> 0.3 s) -> False, and off after ENGAGE
    # independent re-derivation: value of the last carControl sample at or before the row, None when older than 0.3 s
    expect = np.zeros(SY.N, bool)
    for i, tq in enumerate(truth["t"]):
        idx = np.searchsorted(truth["cc_times"], tq, side="right") - 1
        if idx >= 0 and tq - truth["cc_times"][idx] <= C.CTRL_MAX_AGE_S + 1e-9:
            expect[i] = truth["cc_enabled"][idx]
    assert np.array_equal(lat, expect)
    assert frame["das_enabled"].sum() == 0                             # controlsState.enabled dead, cruiseState.enabled False
    das_any = frame["das_any"].to_numpy()
    on = das_any[SY.ENGAGE[0] + 1:SY.ENGAGE[1] + 1]
    assert on[held[SY.ENGAGE[0] + 1:SY.ENGAGE[1] + 1]].all() and not das_any[SY.GAP_AT]
    assert das_any.sum() == SY.ENGAGE[1] - SY.ENGAGE[0] - 1            # selfdriveState is stale at the gap row too
    assert not das_any[:SY.ENGAGE[0] + 1].any() and not das_any[SY.ENGAGE[1] + 1:].any()


# ----------------------------------------------------------------------------- lead
def test_lead_columns(built):
    frame, info, truth, _, _ = built
    t = truth["t"]; v = truth["v"]
    valid = frame["lead_valid"].to_numpy()
    in_ep = (t >= SY.LEAD_T[0] + 0.3) & (t < SY.LEAD_T[1])
    stale = (t >= SY.RADAR_DROP_T[0] + C.LEAD_MAX_AGE_S + 0.1) & (t < SY.RADAR_DROP_T[1])
    near_drop = (t >= SY.RADAR_DROP_T[0]) & (t < SY.RADAR_DROP_T[1] + 0.3)      # hold runs out / resumes at the edges
    assert valid[in_ep & ~near_drop].all()
    assert not valid[stale].any() and np.isnan(frame["lead_d_m"].to_numpy()[stale]).all()
    assert np.isnan(frame["qc_lead_age_s"].to_numpy()[stale]).all()
    assert not valid[t < SY.LEAD_T[0]].any() and not valid[t > SY.LEAD_T[1] + 0.3].any()
    d = frame["lead_d_m"].to_numpy(); vrel = frame["lead_v_rel_mps"].to_numpy(); speed = frame["lead_speed_mps"].to_numpy()
    assert np.allclose(speed[valid], v[valid] + vrel[valid]) and np.isnan(speed[~valid]).all()
    slope = frame["lead_a_slope_mps2"].to_numpy()
    early = valid & (t >= SY.LEAD_T[0] + 2.0) & (t < SY.RADAR_DROP_T[0])
    late = valid & (t >= SY.RADAR_DROP_T[1] + 1.5) & (t < SY.LEAD_T[1])
    assert np.isfinite(slope[early]).all() and abs(np.median(slope[early]) - SY.LEAD_ACC) < 0.15
    assert np.isfinite(slope[late]).all() and abs(np.median(slope[late]) - SY.LEAD_ACC) < 0.15
    assert np.isnan(slope[~valid]).all()
    src = frame["lead_source"].to_numpy()
    assert (src[valid] == 1).all() and (src[~valid] == 0).all()
    thw = frame["thw_s"].to_numpy(); m = valid & (v > C.THW_V_MIN_MPS)
    assert np.allclose(thw[m], d[m] / v[m]) and np.isnan(thw[~m]).all()
    ttc = frame["ttc_s"].to_numpy(); closing = -vrel
    mc = valid & (closing > C.TTC_CLOSING_MIN_MPS)
    assert mc.any() and np.allclose(ttc[mc], d[mc] / closing[mc]) and np.isnan(ttc[~mc]).all()
    assert np.allclose(frame["lead_prob"].to_numpy()[valid], 0.95)
    assert not frame["lead2_valid"].any() and np.isnan(frame["lead2_d_m"]).all()
    assert info["lead_radar_frac"] == 1.0 and info["lead_vision_frac"] == 0.0 and info["cf_frac"] == 1.0
    assert abs(info["lead_valid_frac"] - valid.mean()) < 1e-9


def test_lead_source_vision_fallback_on_drifted_radar_flag():
    frame, _, truth, _, _ = _build(radar_flag_nan=True, radar_unavailable=True)
    valid = frame["lead_valid"].to_numpy()
    assert valid.any() and (frame["lead_source"].to_numpy()[valid] == 2).all()
    frame2, _, _, _, _ = _build(radar_flag_nan=True, radar_unavailable=False)
    assert (frame2["lead_source"].to_numpy()[frame2["lead_valid"].to_numpy()] == 1).all()


# ----------------------------------------------------------------------------- pose
def test_yaw_sign_validity_and_staleness(built):
    frame, info, truth, _, _ = built
    t = truth["t"]
    yaw = frame["yaw_rate_radps"].to_numpy(); yv = frame["yaw_valid"].to_numpy(); ps = frame["pose_source"].to_numpy()
    turn = slice(SY.TURN[0] + 4, SY.TURN[1])                            # 5-Hz hold delay at the start of the turn
    assert (yaw[turn] > 0.1).all() and (frame["curv_1pm"].to_numpy()[turn] > 0).all() and (frame["lat_acc_mps2"].to_numpy()[turn] > 1.0).all()
    assert np.allclose(frame["curv_1pm"].to_numpy()[turn], yaw[turn] / truth["v"][turn])
    assert yv[turn].all() and (ps[turn] == 1).all()
    quiet = (t > 5.0) & (t < 35.0)
    assert (np.abs(yaw[quiet]) < 0.01).all()
    stale = (t >= SY.POSE_DROP_T[0] + C.POSE_MAX_AGE_S + 0.1) & (t < SY.POSE_DROP_T[1])
    assert np.isnan(yaw[stale]).all() and not yv[stale].any() and (ps[stale] == 0).all()
    assert np.isnan(frame["qc_yaw_age_s"].to_numpy()[stale]).all() and np.isnan(frame["curv_1pm"].to_numpy()[stale]).all()
    near_drop = (t >= SY.POSE_DROP_T[0]) & (t < SY.POSE_DROP_T[1] + 0.3)       # hold runs out / resumes at the edges
    fresh = ~near_drop & (t > 1.0)
    assert np.isfinite(yaw[fresh]).all() and (frame["qc_yaw_age_s"].to_numpy()[fresh] <= C.POSE_MAX_AGE_S + 1e-9).all()
    assert info["pose_source"] == "livePose" and info["has_pose"] and info["yaw_valid_frac"] > 0.9
    assert np.isnan(frame["curv_1pm"].to_numpy()[truth["v"] < C.MOVING_V_MPS]).all()


def test_live_location_kalman_source():
    frame, info, _, _, _ = _build(pose_source="liveLocationKalman")
    ps = frame["pose_source"].to_numpy()
    assert set(np.unique(ps).tolist()) <= {0, 2} and (ps == 2).mean() > 0.9 and info["pose_source"] == "liveLocationKalman"


# ----------------------------------------------------------------------------- channels
def test_dead_gas_live_brake_and_torque_norm(built):
    frame, info, truth, _, raw = built
    assert np.isnan(frame["gas"]).all() and not info["gas_alive"]
    assert frame["gas_pressed"].sum() == SY.GAS_PRESSED[1] - SY.GAS_PRESSED[0]      # the flag itself is kept
    brake = frame["brake"].to_numpy()
    assert info["brake_alive"] and np.allclose(brake[SY.BRAKE[0]:SY.BRAKE[1]], 0.3) and (brake[:SY.BRAKE[0]] == 0).all()
    cs = raw["carState"]
    torque = cs["steeringTorque"].to_numpy(); sel = (cs["vEgo"].to_numpy() >= C.MOVING_V_MPS) & cs["steeringPressed"].to_numpy()
    scale = np.quantile(np.abs(torque[sel]), C.TORQUE_NORM_Q)
    assert info["torque_alive"] and abs(info["steer_torque_scale"] - scale) < 1e-9
    assert np.allclose(frame["steer_torque_norm"].to_numpy(), torque / scale)
    assert np.array_equal(frame["steer_torque_raw"].to_numpy(), torque)


def test_dead_channel_rule():
    z = np.zeros(100); pressed = np.zeros(100, bool)
    assert D.dead_channel(np.full(100, np.nan), pressed)                          # all NaN
    assert not D.dead_channel(z, pressed)                                         # zeros but no evidence of use
    pressed[:C.DEAD_CHANNEL_MIN_PRESSED] = True
    assert D.dead_channel(z, pressed)                                             # zeros while pressed fired often enough
    assert not D.dead_channel(z, pressed[::-1][:100] & (np.arange(100) < C.DEAD_CHANNEL_MIN_PRESSED - 1))
    live = z.copy(); live[5] = 0.2
    assert not D.dead_channel(live, pressed)
    scale, norm = D.torque_norm(np.zeros(100), np.full(100, 5.0), pressed)
    assert np.isnan(scale) and np.isnan(norm).all()                               # too few samples / zero scale


def test_set_speed_sentinel_and_units(built):
    frame, info, truth, _, _ = built
    vs = frame["v_set_mps"].to_numpy()
    assert np.allclose(vs[SY.ENGAGE[0]:SY.ENGAGE[1]], SY.VCRUISE_KPH / 3.6, atol=1e-6)
    # 255 km/h sentinel -> NaN; the row right after disengagement still holds the previous (fresh) controlsState sample
    assert np.isnan(vs[:SY.ENGAGE[0]]).all() and np.isnan(vs[SY.ENGAGE[1] + 1:]).all()
    assert abs(vs[SY.ENGAGE[1]] - SY.VCRUISE_KPH / 3.6) < 1e-6
    assert np.allclose(D.valid_set_speed_kph(np.array([255.0, 0.0, 90.0, np.nan]))[2], 25.0) and \
        np.isnan(D.valid_set_speed_kph(np.array([255.0, 0.0]))).all()
    assert np.allclose(D.valid_set_speed_mps(np.array([29.0])), 29.0) and np.isnan(D.valid_set_speed_mps(np.array([0.0, 70.0]))).all()


def test_missing_topics_are_tolerated():
    frame, info, truth, meta, _ = _build(drop_topics=("carControl", "controlsState", "selfdriveState", "radarState", "pose"))
    assert len(frame) == SY.N and not info["has_pose"] and info["pose_source"] == "none"
    assert not frame["lead_valid"].any() and np.isnan(frame["lead_d_m"]).all() and (frame["lead_source"] == 0).all()
    assert np.isnan(frame["yaw_rate_radps"]).all() and not frame["yaw_valid"].any() and (frame["pose_source"] == 0).all()
    assert not frame["das_any"].any() and not frame["op_lat_active"].any()
    assert np.isfinite(frame["v_set_mps"].to_numpy()[SY.ENGAGE[0]:SY.ENGAGE[1]]).all()    # carState.vCruise still carries
    assert info["n_radarState"] == 0 and info["n_pose"] == 0 and info["n_carControl"] == 0
    assert np.isnan(info["lead_valid_frac"]) or info["lead_valid_frac"] == 0.0


def test_software_era_fallback():
    for ver, era in (("0.9.7", "0.9"), ("0.10.1-release", "0.10"), ("0.8.16", "0.8"), ("0.11.0", "0.11"), ("2026.003.000", "2026"),
                     ("2025.12.1", "2025"), ("2023.4", "other"), (None, "other"), ("garbage", "other"), ("0.7.10", "other")):
        assert D.software_era(ver) == era, ver


# ----------------------------------------------------------------------------- schema
def test_public_schema_roundtrip(built, tmp_root):
    frame, info, truth, meta, _ = built
    path = tmp_root / "vehicle_0001" / "drive_000001.parquet"
    S.write_public_parquet(frame, path, S.DRIVE_COLUMNS, "drive")
    assert S.check_public_parquet(path, S.DRIVE_COLUMNS, "drive") == []
    sch = pq.read_schema(path)
    types = dict(zip(sch.names, (str(t) for t in sch.types)))
    assert types["t_s"] == "double" and types["seg"] == "int16" and types["gap"] == "bool" and types["v_mps"] == "float"
    assert types["pose_source"] == "int8" and types["lead_source"] == "int8" and types["das_any"] == "bool"
    back = pd.read_parquet(path)
    assert len(back) == SY.N and back["seg"].max() == 1 and np.isnan(back["gas"]).all()
    assert np.isfinite(back["yaw_rate_radps"]).sum() > 0.9 * SY.N
    names = dict(vehicle_id="vehicle_0001", drive_id="drive_000001", group_id="group_0001")
    labels = dict(brand="SynthBrand", model="Synth", platform="SYNTH_PLATFORM", body="sedan", size_code="D", powertrain="BEV",
                  powertrain_raw="BEV", label_source="fingerprint", label_note=None, region="europe", software_era="0.9",
                  wheelbase_mm=2700.0, track_front_mm=1550.0, track_rear_mm=1560.0, footprint_m2=4.2, curb_mass_kg=1800.0,
                  spec_confidence="A", size_class="MID")
    row = D.index_row(info, names, labels, meta)
    assert set(row) == set(S.names(S.DRIVES_INDEX_COLUMNS))
    assert row["file"] == "data/vehicle_0001/drive_000001.parquet" and row["n_rows"] == SY.N and row["duration_s"] == SY.N / 10
    assert row["task_bev_ice_eligible"] and row["task_size_eligible"] and row["split_device"] is None
    assert row["software_era"] == "0.9" and row["device_type"] == "tici" and row["pose_source"] == "livePose"
    assert row["radar_unavailable"] is False and row["op_long"] is False and row["n_seg_rlog_dec"] == 1
    ipath = tmp_root / "drives.parquet"
    S.write_public_parquet(pd.DataFrame([row]), ipath, S.DRIVES_INDEX_COLUMNS, "drives")
    assert S.check_public_parquet(ipath, S.DRIVES_INDEX_COLUMNS, "drives") == []
    row2 = D.index_row(info, names, None, dict(meta, version="2026.003.000"))
    assert row2["software_era"] == "2026" and row2["brand"] is None and not row2["task_bev_ice_eligible"]
    qc = D.qc_row(info, names, meta, SY.fake_uid(0), row["software_era"])
    assert set(qc) == set(D.PRIVATE_QC_COLUMNS) and qc["n_seg_rlog_dec"] == 1 and qc["dt_med_s"] == info["dt_med_s"]


def test_roots_match_paths():
    from vtt.paths import P
    r = D.Roots(C.DATA_ROOT)
    assert r.signals == P.SIGNALS and r.drives == P.DRIVES and r.drive_names == P.DRIVE_NAMES and r.routes_labels == P.ROUTES_LABELS
    assert r.routes_all == P.ROUTES_ALL and r.data_dir == P.DATA_DIR and r.private_qc == P.PRIVATE_QC / "drives_private_qc.parquet"


# ----------------------------------------------------------------------------- the script, end to end
def _write_inputs(root: Path):
    """Three buildable routes (+ one listed without signals, one with signals but no name); returns (uids, names frame)."""
    sigroot = root / "work" / "signals"
    specs = [dict(seed=1), dict(seed=2, drop_topics=("pose",)), dict(seed=3, pose_source="liveLocationKalman", version="2026.003.000")]
    uids = []
    for sp in specs:
        sig, meta, _ = SY.make_signals(**sp)
        SY.write_signals(sig, meta, sigroot / meta["route_uid"])
        uids.append(meta["route_uid"])
    sig, meta, _ = SY.make_signals(seed=5)                       # signals without a drive name
    SY.write_signals(sig, meta, sigroot / meta["route_uid"])
    unnamed = meta["route_uid"]; unsignalled = SY.fake_uid(4)
    names = pd.DataFrame(dict(route_uid=uids + [unsignalled], vehicle_id=["vehicle_0002", "vehicle_0002", "vehicle_0001", "vehicle_0003"],
                              drive_id=["drive_000011", "drive_000012", "drive_000013", "drive_000014"],
                              group_id=["group_0001", "group_0001", "group_0002", "group_0003"]))
    (root / "build").mkdir(parents=True, exist_ok=True)
    names.to_parquet(root / "build" / "drive_names.parquet", index=False)
    labels = pd.DataFrame(dict(route_uid=uids[:2], platform=["P1", "P1"], brand=["SynthBrand"] * 2, model=["Synth"] * 2, body=["suv"] * 2,
                               size_code=["D"] * 2, powertrain_raw=["ICE", "BEV"], powertrain=["ICE", "BEV"], label_source=["fingerprint"] * 2,
                               label_grade=["A"] * 2, label_note=[None, None], region=["north_america"] * 2, software_era=["0.9", "0.9"],
                               wheelbase_mm=[2800.0] * 2, track_front_mm=[1600.0] * 2, track_rear_mm=[1600.0] * 2, footprint_m2=[4.48] * 2,
                               curb_mass_kg=[2000.0] * 2, spec_confidence=["A"] * 2, size_class=[None, "LARGE"]))
    labels.to_parquet(root / "build" / "routes_labels.parquet", index=False)
    pd.DataFrame(dict(route_uid=uids + [unsignalled, unnamed])).to_csv(root / "build" / "routes_all.csv", index=False)
    return uids, names


def _run(root: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(REPO / "scripts" / "s05_build_drives.py"), "--data-root", str(root), "--workers", "2",
           "--flush-every", "1", "--nice", "0", *extra]
    env = dict(os.environ, PYTHONPATH=str(REPO))
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)


def test_script_builds_resumes_and_accepts(tmp_root):
    uids, names = _write_inputs(tmp_root)
    dry = _run(tmp_root, "--dry-run")
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert "to_build=3" in dry.stdout and "no_name=1" in dry.stdout and "no_signals=1" in dry.stdout
    assert not (tmp_root / "build" / "index" / "drives.parquet").exists()
    r1 = _run(tmp_root)
    assert r1.returncode == 0, r1.stdout + r1.stderr
    assert "ok=3" in r1.stdout and "accept: ok" in r1.stdout and not HEX16.search(r1.stdout + r1.stderr)
    idx_path = tmp_root / "build" / "index" / "drives.parquet"
    assert S.check_public_parquet(idx_path, S.DRIVES_INDEX_COLUMNS, "drives") == []
    idx = pd.read_parquet(idx_path)
    assert len(idx) == 3 and idx["drive_id"].tolist() == ["drive_000013", "drive_000011", "drive_000012"]   # sorted by vehicle, drive
    for f in idx["file"]:
        p = tmp_root / "build" / f
        assert p.exists() and S.check_public_parquet(p, S.DRIVE_COLUMNS, "drive") == []
    assert idx["n_rows"].tolist() == [SY.N] * 3
    by = idx.set_index("drive_id")
    assert by.loc["drive_000011", "brand"] == "SynthBrand" and by.loc["drive_000011", "software_era"] == "0.9"
    assert by.loc["drive_000013", "software_era"] == "2026" and by.loc["drive_000013", "brand"] is None
    assert by.loc["drive_000013", "pose_source"] == "liveLocationKalman" and by.loc["drive_000012", "pose_source"] == "none"
    assert bool(by.loc["drive_000012", "task_bev_ice_eligible"]) and bool(by.loc["drive_000012", "task_size_eligible"])
    assert not bool(by.loc["drive_000011", "task_size_eligible"]) and not bool(by.loc["drive_000012", "has_pose"])
    assert by["split_device"].isna().all() and by["gas_alive"].tolist() == [False] * 3 and by["brake_alive"].tolist() == [True] * 3
    qc_path = tmp_root / "private" / "qc" / "drives_private_qc.parquet"
    qc = pd.read_parquet(qc_path)
    assert list(qc.columns) == list(D.PRIVATE_QC_COLUMNS) and len(qc) == 3 and set(qc["route_uid"]) == set(uids)
    assert (os.stat(qc_path).st_mode & 0o077) == 0 and (os.stat(tmp_root / "private").st_mode & 0o077) == 0
    assert not (tmp_root / "work" / "logs" / "s05_failures.csv").exists()
    # resume: nothing to rebuild, outputs unchanged
    r2 = _run(tmp_root)
    assert r2.returncode == 0 and "skipped_resume=3" in r2.stdout and "to_build=0" in r2.stdout, r2.stdout + r2.stderr
    idx2 = pd.read_parquet(idx_path)
    pd.testing.assert_frame_equal(idx, idx2)
    assert len(pd.read_parquet(qc_path)) == 3
    # --limit and --no-resume rebuild one drive; the index keeps all three
    r3 = _run(tmp_root, "--no-resume", "--limit", "1")
    assert r3.returncode == 0 and "ok=1" in r3.stdout, r3.stdout + r3.stderr
    assert len(pd.read_parquet(idx_path)) == 3 and len(pd.read_parquet(qc_path)) == 3
    # a damaged route fails the acceptance (1 of 1 attempted) and is logged with a masked error, good drives stay
    sig, meta, _ = SY.make_signals(seed=6)
    SY.write_signals(sig, meta, tmp_root / "work" / "signals" / meta["route_uid"])
    (tmp_root / "work" / "signals" / meta["route_uid"] / "carState.parquet").write_bytes(b"not a parquet file")
    names2 = pd.concat([names, pd.DataFrame(dict(route_uid=[meta["route_uid"]], vehicle_id=["vehicle_0004"], drive_id=["drive_000015"],
                                                  group_id=["group_0004"]))], ignore_index=True)
    names2.to_parquet(tmp_root / "build" / "drive_names.parquet", index=False)
    pd.DataFrame(dict(route_uid=uids + [meta["route_uid"]])).to_csv(tmp_root / "build" / "routes_all.csv", index=False)
    r4 = _run(tmp_root)
    assert r4.returncode == 2 and "error=1" in r4.stdout and "accept: FAILED" in r4.stdout, r4.stdout + r4.stderr
    fails = pd.read_csv(tmp_root / "work" / "logs" / "s05_failures.csv")
    assert len(fails) == 1 and fails["status"].iloc[0] == "error" and not HEX16.search(str(fails["err"].iloc[0]))
    assert len(pd.read_parquet(idx_path)) == 3
    # --routes override restricts the run to the listed routes
    (tmp_root / "one.csv").write_text("route_uid\n" + uids[0] + "\n")
    r5 = _run(tmp_root, "--routes", str(tmp_root / "one.csv"), "--no-resume")
    assert r5.returncode == 0 and "listed=1" in r5.stdout and "ok=1" in r5.stdout, r5.stdout + r5.stderr
    assert len(pd.read_parquet(idx_path)) == 3
