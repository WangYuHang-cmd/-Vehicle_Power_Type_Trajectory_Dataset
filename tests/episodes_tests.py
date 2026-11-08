"""experiments/episodes.py, episodes_build.py and window_embed.py on a synthetic 10-Hz drive.

The drive (180 s): standstill, launch to 10 m/s, a 3-s turn, cruise, lift-off deceleration (no pedal), a small speed-up,
braked stop to standstill, second launch, a lead in car-following range whose slope steps up (the ego answers 1.5 s later),
a driving-automation segment, an acceleration right after it, a braked deceleration whose crop spans the drive's one gap
row. Truth anchors are the construction rows. The window-embedding test needs torch (skipped without it).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vtt import config as C
from vtt import features_core as F
from vtt import schema as S
from vtt import windows as W
from experiments import episodes as EP
from experiments import episodes_build as EB
from tests import synth_drive as SD

TMP_BASE = Path(C.DATA_ROOT) / "tmp"
DT = C.NOMINAL_DT_S
N = 1800
GAP_ROW = 1600
DAS_ROWS = (1300, 1400)
TRUTH = {"LAUNCH": [100, 750], "DEC_LIFT": [400], "DEC_BRK": [600, 1560], "STOP": [649], "LEAD_ACC": [1000],
         "TURN": [200], "ACC": [100, 750, 1420], "LEAD_DEC": []}


def _ramp(v: np.ndarray, r0: int, r1: int, v0: float, v1: float) -> None:
    k = np.arange(1, r1 - r0 + 1)
    v[r0:r1] = v0 + (v1 - v0) * k / (r1 - r0)


def synth_frame() -> pd.DataFrame:
    """The synthetic drive as a public drive table (every schema column)."""
    v = np.zeros(N)
    _ramp(v, 100, 160, 0.0, 10.0); v[160:400] = 10.0                 # launch, cruise (turn at 200-230)
    _ramp(v, 400, 500, 10.0, 6.0); v[500:520] = 6.0                    # lift-off deceleration
    _ramp(v, 520, 540, 6.0, 7.0); v[540:600] = 7.0                     # small speed-up (dv < 2: no episode)
    _ramp(v, 600, 650, 7.0, 0.0); v[650:750] = 0.0                     # braked stop (standstill from row 649)
    _ramp(v, 750, 810, 0.0, 10.0); v[810:1015] = 10.0                  # second launch, cruise with a lead
    _ramp(v, 1015, 1035, 10.0, 11.0); v[1035:1200] = 11.0              # ego answer to the lead acceleration
    _ramp(v, 1200, 1220, 11.0, 10.0); v[1220:1420] = 10.0              # small slow-down (no episode), automation 1300-1400
    _ramp(v, 1420, 1460, 10.0, 14.0); v[1460:1560] = 14.0              # acceleration right after the automation
    _ramp(v, 1560, 1600, 14.0, 10.0); v[1600:] = 10.0                  # braked deceleration, gap row at 1600
    das = np.zeros(N, bool); das[DAS_ROWS[0]:DAS_ROWS[1]] = True
    lead = np.zeros(N, bool); lead[900:1200] = True
    slope = np.where(lead, 0.0, np.nan); slope[1000:1020] = 1.0
    lead_d = np.where(lead, 20.0, np.nan)                               # THW = 2 s at 10 m/s
    yaw = np.zeros(N); yaw[200:230] = 0.2                               # curvature 0.02 1/m at 10 m/s
    df = SD.make_drive(N, v=v, das_any=das, gap_rows=[GAP_ROW], lead_valid=lead, lead_a_slope=slope, lead_d=lead_d, yaw=yaw)
    brake = np.zeros(N, bool); brake[600:650] = True; brake[1560:1600] = True
    df["brake_pressed"] = brake
    df["gas_pressed"] = (df["a_mps2"].to_numpy() > 0.05) & ~brake
    return df


def frame_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """The drive dict exactly as experiments.episodes.load_drive_arrays builds it, straight from the frame."""
    spec = S.dtypes(S.DRIVE_COLUMNS)
    d = {c: (df[c].to_numpy(bool) if spec[c] == "bool" else df[c].to_numpy(float)) for c in EP.READ_COLUMNS}
    d["__n__"] = np.array([len(df)])
    return d


@pytest.fixture(scope="module")
def drive():
    df = synth_frame()
    d = frame_arrays(df)
    eps = EP.detect_episodes(d, EP.PARAMS)
    return df, d, eps


def _by_type(eps):
    out: dict[str, list] = {t: [] for t in EP.TYPES}
    for e in eps:
        out[e.type].append(e)
    return out


def test_constants_match_the_contract():
    assert len(EP.TYPES) == 8 and EP.TYPES[0] == "ACC" and EP.TYPES[3] == "STOP" and EP.TYPES[7] == "TURN"
    assert len(EP.EP_CHANNELS) == 22 and EP.EP_CHANNELS[:20] == tuple(C.OFFICIAL_INPUTS) and EP.EP_CHANNELS[20:] == ("a_sg", "valid")
    assert EP.GRID_CHANNELS == ("a_sg", "gas_pressed", "brake_pressed", "das_enabled", "valid", "t_from_onset_s")
    assert EP.L_STORE == 180 and EP.PRE_DEFAULT == 40 and EP.PRE_SAMPLES == {"STOP": 140}
    assert len(EP.SPEED_GRID) == 72 and EP.SPEED_GRID[0] == 0.0 and EP.SPEED_GRID[-1] == 35.5 and EP.SPEED_GRID[1] == 0.5
    assert EP.PARAMS["accel"]["trim_win_s"] == 2.0 and EP.PARAMS["lead"]["trans_s"] == 0.5 and EP.PARAMS["lead_decel_slope_on"] == 0.6
    assert EP.PARAMS["thw_cf_s"] == C.THW_CF_S and EP.PARAMS["human_guard_s"] == C.HUMAN_GUARD_S and EP.PARAMS["lead_standstill_d_m"] == 30.0
    assert EP.PARAMS["stop_bound_at_runs"] is True and EP.PARAMS["stop_lookback_s"] == 15.0 and EP.PARAMS["stop_v_min"] == 2.0
    assert [k for k, _ in EP.INDEX_COLUMNS] == [
        "episode_id", "drive_id", "vehicle_id", "group_id", "type_id", "anchor_row", "i0", "i1", "crop_row0", "v_anchor", "dv",
        "dur_s", "mean_a", "peak_a", "resp_delay_s", "das_frac", "das_any_frac", "human_strict", "lead_valid_frac", "thw_anchor",
        "gap_in_crop", "w_idx", "window_id", "grid_idx"]
    assert len(EP.params_hash()) == 8 and EP.params_hash(dict(EP.PARAMS, turn_enabled=False)) != EP.params_hash()


def test_type_counts_and_anchors(drive):
    _, d, eps = drive
    by = _by_type(eps)
    counts = {t: len(v) for t, v in by.items()}
    assert counts == {"ACC": 3, "DEC_LIFT": 1, "DEC_BRK": 2, "STOP": 1, "LAUNCH": 2, "LEAD_ACC": 1, "LEAD_DEC": 0, "TURN": 1}
    for t, truth in TRUTH.items():
        got = sorted(e.anchor_row for e in by[t])
        assert len(got) == len(truth), t
        for g, tr in zip(got, truth):
            assert abs(g - tr) <= 2, (t, g, tr)
    # kinematics: half-open [i0, i1) in time order, speeds and signs as constructed
    for e in eps:
        assert 0 <= e.i0 <= e.anchor_row < e.i1 <= N and e.dur_s == pytest.approx((e.i1 - e.i0) * DT)
    lift = by["DEC_LIFT"][0]
    assert lift.dv == pytest.approx(-4.0, abs=0.3) and lift.mean_a < -0.3 and lift.peak_a == pytest.approx(-0.4, abs=0.05)
    assert lift.v_anchor == pytest.approx(10.0, abs=0.1) and np.isnan(lift.resp_delay_s) and np.isnan(lift.thw_anchor)
    stop = by["STOP"][0]
    assert stop.i1 == stop.anchor_row + 1 and stop.dv == pytest.approx(-2.0, abs=0.4) and d["standstill"][stop.anchor_row]
    launch = by["LAUNCH"][0]
    assert launch.i0 == launch.anchor_row and not d["standstill"][launch.anchor_row] and d["standstill"][launch.anchor_row - 1]
    assert d["v_mps"][launch.i1 - 1] > 2.0 and d["v_mps"][launch.i1 - 2] <= 2.0
    lead = by["LEAD_ACC"][0]
    assert lead.thw_anchor == pytest.approx(2.0, abs=1e-6) and 1.0 <= lead.resp_delay_s <= 2.0 and lead.i1 - lead.i0 == 80
    turn = by["TURN"][0]
    assert turn.i1 - turn.i0 == 30 and turn.v_anchor == pytest.approx(10.0)
    acc = sorted(by["ACC"], key=lambda e: e.anchor_row)[-1]
    assert acc.dv == pytest.approx(4.0, abs=0.3) and acc.peak_a == pytest.approx(1.0, abs=0.05)


def test_flags_human_strict_and_gap(drive):
    _, d, eps = drive
    by = _by_type(eps)
    flags = {e.anchor_row: EP.episode_flags(d, e) for e in eps}
    late_acc = sorted(by["ACC"], key=lambda e: e.anchor_row)[-1]           # 2 s after the automation segment
    assert not flags[late_acc.anchor_row]["human_strict"] and flags[late_acc.anchor_row]["das_any_frac"] == 0.0
    for e in eps:
        if e is late_acc:
            continue
        assert flags[e.anchor_row]["human_strict"], e.type              # everything else is > 10 s from the automation
        assert flags[e.anchor_row]["das_frac"] == 0.0
    gap_dec = sorted(by["DEC_BRK"], key=lambda e: e.anchor_row)[-1]
    assert flags[gap_dec.anchor_row]["gap_in_crop"] and sum(f["gap_in_crop"] for f in flags.values()) == 1
    lead = by["LEAD_ACC"][0]
    assert flags[lead.anchor_row]["lead_valid_frac"] == 1.0 and flags[by["STOP"][0].anchor_row]["lead_valid_frac"] == 0.0
    # the guard also covers automation that is still active at the anchor
    d2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    d2["das_any"][by["STOP"][0].anchor_row - 5:by["STOP"][0].anchor_row + 5] = True
    assert not EP.episode_flags(d2, by["STOP"][0])["human_strict"]


def test_crop_shape_valid_and_gap(drive):
    _, d, eps = drive
    by = _by_type(eps)
    for e in eps:
        x = EP.crop_episode(d, e)
        assert x.shape == (180, 22) and x.dtype == np.float16
        pre = 140 if e.type == "STOP" else 40
        valid = x[:, 21].astype(float)
        assert set(np.unique(valid)) <= {0.0, 1.0} and valid[pre] == 1.0                      # the anchor sample is valid
        assert np.isfinite(x[valid == 1.0][:, [0, 20]].astype(np.float32)).all()        # v_mps and a_sg always known
        assert np.isnan(x[valid == 0.0][:, :21].astype(np.float32)).all()              # NaN of lead / curvature channels is kept
        lead_cols = [EP.EP_CHANNELS.index(c) for c in ("lead_valid", "thw_s")]
        assert np.array_equal(np.isnan(x[valid == 1.0][:, lead_cols[1]].astype(np.float32)),
                              np.isnan(d["thw_s"][max(e.crop_row0, 0):max(e.crop_row0, 0) + int(valid.sum())]))
        assert float(x[pre, 0]) == pytest.approx(d["v_mps"][e.anchor_row], abs=0.02)           # channel 0 = v_mps
        assert e.crop_row0 == e.anchor_row - pre
    gap_dec = sorted(by["DEC_BRK"], key=lambda e: e.anchor_row)[-1]
    x = EP.crop_episode(d, gap_dec)
    k = GAP_ROW - gap_dec.crop_row0
    assert (x[:k, 21] == 1).all() and (x[k:, 21] == 0).all() and 0 < k < 180
    # a crop that starts before the drive: NaN with valid = 0 there
    early = EP.Episode(type_id=EP.TYPE_ID["ACC"], anchor_row=20, i0=20, i1=60, v_anchor=0.0, dv=0.0, dur_s=4.0, mean_a=0.0, peak_a=0.0)
    x = EP.crop_episode(d, early)
    assert (x[:20, 21] == 0).all() and (x[20:, 21] == 1).all() and np.isnan(x[:20, :21].astype(np.float32)).all()
    assert EP.episode_flags(d, early)["gap_in_crop"] is False
    # a STOP crop keeps 14 s before the anchor and the valid part ends at the drive end when the crop runs past it
    late_stop = EP.Episode(type_id=EP.TYPE_ID["STOP"], anchor_row=N - 10, i0=N - 30, i1=N - 9, v_anchor=0.0, dv=-2.0, dur_s=2.0, mean_a=-1.0, peak_a=-1.0)
    x = EP.crop_episode(d, late_stop)
    assert (x[:150, 21] == 1).all() and (x[150:, 21] == 0).all()
    # a_sg channel equals the smoothed derivative of the speed at the anchor of the lift-off episode
    lift = by["DEC_LIFT"][0]
    x = EP.crop_episode(d, lift)
    assert float(x[40 + 20, 20]) == pytest.approx(-0.4, abs=0.05)


def test_speed_grid_monotone_coverage(drive):
    _, d, eps = drive
    by = _by_type(eps)
    for e in eps:
        g = EP.speed_grid(d, e)
        if e.type not in ("ACC", "DEC_LIFT", "DEC_BRK"):
            assert g is None
            continue
        assert g.shape == (72, 6) and g.dtype == np.float16
        valid = g[:, 4].astype(float)
        cov = np.flatnonzero(valid == 1.0)
        assert len(cov) >= 4 and (np.diff(cov) == 1).all()                                       # one contiguous speed range
        assert np.isnan(g[valid == 0.0][:, [0, 1, 2, 3, 5]].astype(np.float32)).all()
        assert np.isfinite(g[cov].astype(np.float32)).all()
        t = g[cov, 5].astype(np.float32)
        if e.type == "ACC":
            assert (np.diff(t) > 0).all()                                                           # time grows with speed
        else:
            assert (np.diff(t) < 0).all()                                                           # time grows as speed falls
    lift = by["DEC_LIFT"][0]
    g = EP.speed_grid(d, lift)
    cov = np.flatnonzero(g[:, 4] == 1)
    assert EP.SPEED_GRID[cov[0]] == pytest.approx(6.0, abs=0.5) and EP.SPEED_GRID[cov[-1]] == pytest.approx(10.0, abs=0.5)
    mid = cov[2:-2]
    assert np.allclose(g[mid, 0].astype(np.float32), -0.4, atol=0.05) and (g[mid, 2] == 0).all() and (g[mid, 3] == 0).all()
    acc = sorted(by["ACC"], key=lambda e: e.anchor_row)[-1]
    g = EP.speed_grid(d, acc)
    cov = np.flatnonzero(g[:, 4] == 1)
    assert EP.SPEED_GRID[cov[0]] == pytest.approx(10.0, abs=0.5) and EP.SPEED_GRID[cov[-1]] == pytest.approx(14.0, abs=0.5)
    assert (g[cov[2:-2], 1] == 1).all()                                                             # gas pressed while accelerating


def test_episode_context_and_matrix(drive):
    _, d, eps = drive
    rows = []
    for e in eps:
        c = EP.episode_context(d, e)
        assert c.shape == (9,) and c.dtype == np.float32 and np.isfinite(c).all()
        assert c[0] == pytest.approx(e.v_anchor / 30.0, abs=1e-6) and c[1] == pytest.approx(e.dv / 10.0, abs=1e-6)
        assert c[7] == (0.0 if np.isnan(e.thw_anchor) else pytest.approx(e.thw_anchor / 10.0, abs=1e-6))
        rows.append(EP.episode_row(d, e))
    idx = EP.coerce_index(pd.DataFrame(rows))
    m = EP.context_matrix(idx)
    assert m.shape == (len(eps), 9) and np.allclose(m, np.stack([EP.episode_context(d, e) for e in eps]), atol=1e-6)
    assert (idx["w_idx"].to_numpy() == idx["anchor_row"].to_numpy() // 600).all()


def test_lead_onsets_transition_and_spacing():
    s = np.zeros(120)
    s[30:32] = (0.2, 0.4)                                  # 2 transition samples (0.2 is still quiet, 0.4 is neither)
    s[32:60] = 1.0
    valid = np.ones(120, bool)
    lp = EP.PARAMS["lead"]
    assert EP.lead_onsets(s, valid, DT, **lp).tolist() == [32]
    assert EP.lead_onsets(s, valid, DT, **dict(lp, trans_s=0.0)).tolist() == []           # strict rule: no transition allowed
    assert EP.lead_onsets(-s, valid, DT, **dict(lp, slope_on=0.6)).tolist() == []          # sign matters
    assert EP.lead_onsets(-s, valid, DT, **dict(lp, slope_on=0.6)).size == 0 and EP.lead_onsets(s * -1.0 * -1.0, valid, DT, **lp).tolist() == [32]
    valid[20] = False                                       # a lead drop-out inside the quiet window kills the onset
    assert EP.lead_onsets(s, valid, DT, **lp).tolist() == []
    # two onsets closer than the spacing: only the first survives in detect_episodes
    n = 900
    v = np.full(n, 10.0)
    slope = np.zeros(n); slope[300:320] = 1.0; slope[350:370] = 1.0; slope[700:720] = 1.0
    df = SD.make_drive(n, v=v, lead_valid=np.ones(n, bool), lead_a_slope=slope, lead_d=np.full(n, 20.0))
    eps = EP.detect_episodes(frame_arrays(df), EP.PARAMS)
    assert sorted(e.anchor_row for e in eps if e.type == "LEAD_ACC") == [300, 700]
    # out of car-following range (THW 8 s): no event
    df = SD.make_drive(n, v=v, lead_valid=np.ones(n, bool), lead_a_slope=slope, lead_d=np.full(n, 80.0))
    assert not [e for e in EP.detect_episodes(frame_arrays(df), EP.PARAMS) if e.type == "LEAD_ACC"]
    # a deceleration of the lead (slope -1) is a LEAD_DEC
    df = SD.make_drive(n, v=v, lead_valid=np.ones(n, bool), lead_a_slope=-slope, lead_d=np.full(n, 20.0))
    assert sorted(e.anchor_row for e in EP.detect_episodes(frame_arrays(df), EP.PARAMS) if e.type == "LEAD_DEC") == [300, 700]
    # the ego stands behind the lead (THW undefined) and creeps off within 8 s: kept by the distance rule; dropped when the
    # lead is far, when the rule is off, or when the ego never moves (parked car, other lane)
    creep = np.zeros(n); creep[330:400] = 1.0; creep[730:800] = 1.0                 # 3 s after each onset (no episode of its own)
    df = SD.make_drive(n, v=creep, lead_valid=np.ones(n, bool), lead_a_slope=slope, lead_d=np.full(n, 15.0))
    kept = [e for e in EP.detect_episodes(frame_arrays(df), EP.PARAMS) if e.type == "LEAD_ACC"]
    assert sorted(e.anchor_row for e in kept) == [300, 700] and all(np.isnan(e.thw_anchor) and e.v_anchor == 0.0 for e in kept)
    assert not [e for e in EP.detect_episodes(frame_arrays(df), dict(EP.PARAMS, lead_standstill_d_m=0.0)) if e.type == "LEAD_ACC"]
    df = SD.make_drive(n, v=creep, lead_valid=np.ones(n, bool), lead_a_slope=slope, lead_d=np.full(n, 50.0))
    assert not [e for e in EP.detect_episodes(frame_arrays(df), EP.PARAMS) if e.type == "LEAD_ACC"]
    df = SD.make_drive(n, v=np.zeros(n), lead_valid=np.ones(n, bool), lead_a_slope=slope, lead_d=np.full(n, 15.0))
    assert not [e for e in EP.detect_episodes(frame_arrays(df), EP.PARAMS) if e.type == "LEAD_ACC"]


def test_stop_and_go_one_stop_per_approach():
    """Approach, 5-s stop, 3-s creep at 1 m/s, 5-s stop, departure -> one STOP (first standstill onset) and one LAUNCH
    (last standstill end), neither interval containing the other standstill run. A creep above 2 m/s between the stops
    makes two STOP / LAUNCH pairs with distinct intervals; a sub-second standstill blip inside the approach does not split
    it (the STOP sits at the qualifying stop, i0 in the approach); stop_bound_at_runs=False restores the unbounded rule."""
    n = 700

    def profile(creep: float) -> np.ndarray:
        v = np.zeros(n)
        v[:100] = 10.0; _ramp(v, 100, 150, 10.0, 0.0)                    # approach: standstill from row 149
        v[150:200] = 0.0; v[200:230] = creep; v[230:280] = 0.0             # stop, creep, stop
        _ramp(v, 280, 340, 0.0, 10.0); v[340:] = 10.0                     # departure
        return v

    d = frame_arrays(SD.make_drive(n, v=profile(1.0)))
    runs = [(r0, r1) for r0, r1 in F.true_runs(d["standstill"]) if r1 - r0 >= 10]
    assert runs == [(149, 200), (230, 280)]
    eps = EP.detect_episodes(d, EP.PARAMS)
    stops = [e for e in eps if e.type == "STOP"]
    launches = [e for e in eps if e.type == "LAUNCH"]
    assert [e.anchor_row for e in stops] == [149] and [e.anchor_row for e in launches] == [280]
    assert stops[0].i0 == 138 and stops[0].i1 == 150 and launches[0].i0 == 280 and launches[0].i1 > 280
    for e in stops + launches:
        assert not any(e.i0 < r0 and r1 <= e.i1 for r0, r1 in runs)                     # no interval spans a standstill run
    # the unbounded rule duplicates the approach (same i0) and lets the LAUNCH of the first stop span the second
    old = EP.detect_episodes(d, dict(EP.PARAMS, stop_bound_at_runs=False))
    old_stops = [e for e in old if e.type == "STOP"]
    old_launch = [e for e in old if e.type == "LAUNCH"]
    assert [e.anchor_row for e in old_stops] == [149, 230] and old_stops[0].i0 == old_stops[1].i0 == 138
    assert [e.anchor_row for e in old_launch] == [200, 280] and old_launch[0].i0 < 230 and old_launch[0].i1 > 280
    # a creep above stop_v_min separates the stops: two approaches, two departures, disjoint intervals
    d3 = frame_arrays(SD.make_drive(n, v=profile(3.0)))
    eps3 = EP.detect_episodes(d3, EP.PARAMS)
    stops3 = [e for e in eps3 if e.type == "STOP"]
    launch3 = [e for e in eps3 if e.type == "LAUNCH"]
    assert [e.anchor_row for e in stops3] == [149, 230] and [e.i0 for e in stops3] == [138, 229]
    assert [(e.anchor_row, e.i1) for e in launch3] == [(200, 201), (280, launches[0].i1)]
    # a 0.5-s standstill blip inside the approach is a hesitation: one STOP at the 5-s stop, i0 still in the approach
    v = np.zeros(n)
    v[:100] = 10.0; _ramp(v, 100, 150, 10.0, 0.0)
    v[150:155] = 0.0; v[155:175] = 1.0; v[175:225] = 0.0; _ramp(v, 225, 285, 0.0, 10.0); v[285:] = 10.0
    db = frame_arrays(SD.make_drive(n, v=v))
    assert [(r0, r1) for r0, r1 in F.true_runs(db["standstill"])] == [(149, 155), (175, 225)]
    epsb = EP.detect_episodes(db, EP.PARAMS)
    assert [(e.anchor_row, e.i0) for e in epsb if e.type == "STOP"] == [(175, 138)]
    assert [e.anchor_row for e in epsb if e.type == "LAUNCH"] == [225]


def test_turn_flag_and_segments(drive):
    _, d, _ = drive
    eps = EP.detect_episodes(d, dict(EP.PARAMS, turn_enabled=False))
    assert not [e for e in eps if e.type == "TURN"] and len(eps) == 10
    assert EP.segments(d) == [(0, GAP_ROW), (GAP_ROW, N)]
    assert not any(e.i0 < GAP_ROW < e.i1 for e in eps)                      # no episode spans the gap
    empty = {k: v[:0] for k, v in d.items() if isinstance(v, np.ndarray) and k != "__n__"}
    empty["__n__"] = np.array([0])
    assert EP.detect_episodes(empty) == [] and EP.segments(empty) == []


def _tree(prefix: str) -> tuple[Path, pd.DataFrame]:
    TMP_BASE.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=prefix, dir=TMP_BASE))
    df = synth_frame()
    SD.write_drive(df, root, "vehicle_0001", "drive_000001")
    SD.write_drives_index([SD.drives_index_row(df, "vehicle_0001", "drive_000001", "group_0001")], root)
    return root, df


def test_load_drive_arrays_and_extract_drive(drive):
    df, d, eps = drive
    root, _ = _tree("vtt_episodes_extract_")
    try:
        path = root / "build" / "data" / "vehicle_0001" / "drive_000001.parquet"
        d2 = EP.load_drive_arrays(path)
        assert set(EP.READ_COLUMNS) <= set(d2) and int(d2["__n__"][0]) == N
        for c in EP.READ_COLUMNS:
            np.testing.assert_allclose(d2[c].astype(float), d[c].astype(float), atol=1e-5, equal_nan=True)
        E, G, idx = EP.extract_drive(path, "drive_000001", "vehicle_0001", "group_0001")
        assert list(idx.columns) == [k for k, _ in EP.INDEX_COLUMNS]
        assert E.shape == (len(eps), 180, 22) and E.dtype == np.float16 and len(idx) == len(eps) == 11
        n_grid = sum(e.type in ("ACC", "DEC_LIFT", "DEC_BRK") for e in eps)
        assert G.shape == (n_grid, 72, 6) and (idx["grid_idx"] >= 0).sum() == n_grid == 6
        assert idx["episode_id"].tolist() == list(range(len(eps))) and idx["window_id"].isna().all()
        assert idx["type_id"].dtype == np.int8 and idx["anchor_row"].dtype == np.int32 and idx["human_strict"].dtype == bool
        assert sorted(idx.loc[idx["grid_idx"] >= 0, "grid_idx"]) == list(range(n_grid))
        assert (idx["drive_id"] == "drive_000001").all() and (idx["group_id"] == "group_0001").all()
        assert EP.type_counts(idx) == {"ACC": 3, "DEC_LIFT": 1, "DEC_BRK": 2, "STOP": 1, "LAUNCH": 2, "LEAD_ACC": 1, "LEAD_DEC": 0, "TURN": 1}
        # a crop in E equals the function output for the same episode
        np.testing.assert_array_equal(E[0].astype(np.float32), EP.crop_episode(d, eps[0]).astype(np.float32))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_parts_cache_reuse_force_and_assembly(capsys):
    root, df = _tree("vtt_episodes_build_")
    try:
        roots = W.Roots(root)
        drives = pd.read_parquet(roots.drives)
        windows, _ = W.build_windows(drives, roots, workers=1, log=lambda *a: None)
        W.write_windows(windows, roots.windows)
        out = root / "work" / "episodes"
        logs: list[str] = []
        index, counts = EB.build_episodes(drives, roots, out, workers=1, log=logs.append)
        assert "parts_reused=0 parts_to_build=1" in logs[0] and counts["parts_built"] == 1 and counts["part_errors"] == 0
        assert counts["episodes"] == 11 and counts["grids"] == 6 and counts["drives_without_episodes"] == 0 and counts["window_ids_joined"] == 11
        assert (out / "parts" / "vehicle_0001.npz").exists()
        E = np.load(out / "E_f16.npy", mmap_mode="r")
        G = np.load(out / "G_f16.npy", mmap_mode="r")
        assert E.shape == (11, 180, 22) and G.shape == (6, 72, 6) and E.dtype == np.float16
        idx = pd.read_parquet(out / "episodes.parquet")
        assert list(idx.columns) == [k for k, _ in EP.INDEX_COLUMNS] and idx["episode_id"].tolist() == list(range(11))
        assert sorted(idx.loc[idx["grid_idx"] >= 0, "grid_idx"]) == list(range(6))
        lead = idx[idx["type_id"] == EP.TYPE_ID["LEAD_ACC"]].iloc[0]
        assert lead["window_id"] == "drive_000001_w1" and lead["w_idx"] == 1 and 990 <= lead["anchor_row"] <= 1010
        ch = json.loads((out / "channels.json").read_text())
        assert ch == {"crop": list(EP.EP_CHANNELS), "grid": list(EP.GRID_CHANNELS), "types": list(EP.TYPES)}
        pj = json.loads((out / "params.json").read_text())
        assert pj["params_hash"] == EP.params_hash() and pj["turn_enabled"] is True and pj["accel"]["trim_win_s"] == 2.0
        assert set(idx.columns).isdisjoint({"size_class", "powertrain", "split_platform", "split_device"})
        # the stored crop of the gap-spanning deceleration has valid = 0 after the gap row
        gap_rows = idx[idx["gap_in_crop"]]
        assert len(gap_rows) == 1
        k = GAP_ROW - int(gap_rows.iloc[0]["crop_row0"])
        e = np.asarray(E[int(gap_rows.iloc[0]["episode_id"])])
        assert (e[:k, 21] == 1).all() and (e[k:, 21] == 0).all()
        # second run reuses the part
        logs.clear()
        index2, counts2 = EB.build_episodes(drives, roots, out, workers=1, log=logs.append)
        assert "parts_reused=1 parts_to_build=0" in logs[0] and counts2["episodes"] == 11
        pd.testing.assert_frame_equal(index2, index)
        # --force rebuilds, a parameter change (no TURN) also rebuilds and drops the TURN rows
        logs.clear()
        EB.build_episodes(drives, roots, out, workers=1, force=True, log=logs.append)
        assert "parts_to_build=1" in logs[0]
        logs.clear()
        index3, counts3 = EB.build_episodes(drives, roots, out, dict(EP.PARAMS, turn_enabled=False), workers=1, log=logs.append)
        assert "parts_to_build=1" in logs[0] and counts3["episodes"] == 10 and EP.type_counts(index3)["TURN"] == 0
        assert np.load(out / "E_f16.npy", mmap_mode="r").shape[0] == 10
        # the part is stale again for the default parameters; the CLI rebuilds it and prints counts only
        rc = EB.main(["--data-root", str(root), "--out-dir", str(out), "--workers", "1", "--nice", "0"])
        text = capsys.readouterr().out
        assert rc == 0 and "parts_to_build=1" in text and "episodes by type: ACC=3 DEC_LIFT=1 DEC_BRK=2 STOP=1 LAUNCH=2 LEAD_ACC=1 LEAD_DEC=0 TURN=1" in text
        assert "human_strict share by type:" in text and "drives_without_episodes=0" in text
        from vtt.private import mask
        assert mask(text) == text
        # a stale row (drive moved out of the index) is dropped at assembly
        ghost = drives.copy()
        ghost["drive_id"] = "drive_000099"
        ghost["file"] = "data/vehicle_0001/drive_000099.parquet"
        shutil.copy(root / "build" / "data" / "vehicle_0001" / "drive_000001.parquet", root / "build" / "data" / "vehicle_0001" / "drive_000099.parquet")
        both = pd.concat([drives, ghost], ignore_index=True)
        logs.clear()
        index4, counts4 = EB.build_episodes(both, roots, out, workers=1, log=logs.append)
        assert "parts_to_build=1" in logs[0] and counts4["episodes"] == 22 and counts4["stale_rows_dropped"] == 0
        assert set(index4["drive_id"]) == {"drive_000001", "drive_000099"} and index4["episode_id"].tolist() == list(range(22))
        assert sorted(index4.loc[index4["grid_idx"] >= 0, "grid_idx"]) == list(range(12))
        logs.clear()
        index5, counts5 = EB.build_episodes(drives, roots, out, workers=1, log=logs.append)      # the part lists both drives
        assert "parts_reused=1" in logs[0] and set(index5["drive_id"]) == {"drive_000001"} and counts5["stale_rows_dropped"] == 11
        assert counts5["episodes"] == 11 and np.load(out / "E_f16.npy", mmap_mode="r").shape[0] == 11
        assert np.load(out / "G_f16.npy", mmap_mode="r").shape[0] == 6 and index5["episode_id"].tolist() == list(range(11))
        pd.testing.assert_frame_equal(index5, index)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_failed_rebuild_keeps_stale_part_out_of_assembly(capsys):
    """A vehicle whose rebuild fails (its drive file became unreadable) still has its old part on disk: the assembly leaves
    that part out (parts_skipped), the error is counted (part_errors, exit code 2) and the other vehicles are assembled as
    usual; assemble itself skips a part whose stored parameter hash differs from the current parameters."""
    TMP_BASE.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="vtt_episodes_fail_", dir=TMP_BASE))
    try:
        df = synth_frame()
        SD.write_drive(df, root, "vehicle_0001", "drive_000001")
        SD.write_drive(df, root, "vehicle_0002", "drive_000002")
        SD.write_drives_index([SD.drives_index_row(df, "vehicle_0001", "drive_000001", "group_0001"),
                               SD.drives_index_row(df, "vehicle_0002", "drive_000002", "group_0002")], root)
        roots = W.Roots(root)
        drives = pd.read_parquet(roots.drives)
        out = root / "work" / "episodes"
        index, counts = EB.build_episodes(drives, roots, out, workers=2, log=lambda *a: None)     # the process-pool path
        assert counts["episodes"] == 22 and counts["part_errors"] == 0 and counts["parts_skipped"] == 0 and counts["parts_built"] == 2
        assert set(index["vehicle_id"]) == {"vehicle_0001", "vehicle_0002"}
        part2 = out / "parts" / "vehicle_0002.npz"
        bad = root / "build" / "data" / "vehicle_0002" / "drive_000002.parquet"
        bad.write_bytes(b"not a parquet file")                                                   # newer than the part: stale
        t = part2.stat().st_mtime + 10
        os.utime(bad, (t, t))
        logs: list[str] = []
        index2, counts2 = EB.build_episodes(drives, roots, out, workers=1, log=logs.append)
        assert "parts_reused=1 parts_to_build=1" in logs[0] and "parts_built=0 part_errors=1" in logs[1]
        assert counts2["part_errors"] == 1 and counts2["parts_skipped"] == 1 and counts2["parts_built"] == 0
        assert counts2["episodes"] == 11 and set(index2["vehicle_id"]) == {"vehicle_0001"} and counts2["drives_without_episodes"] == 1
        assert np.load(out / "E_f16.npy", mmap_mode="r").shape[0] == 11 and np.load(out / "G_f16.npy", mmap_mode="r").shape[0] == 6
        assert part2.exists()                                                                   # kept for a later retry
        rc = EB.main(["--data-root", str(root), "--out-dir", str(out), "--workers", "1", "--nice", "0"])
        text = capsys.readouterr().out
        assert rc == 2 and "part_errors=1" in text and "parts_skipped=1" in text and "episodes=11" in text
        # assemble skips parts built with other parameters (both parts carry the default hash here)
        _, counts3 = EB.assemble([out / "parts" / "vehicle_0001.npz", part2], drives, out, roots.windows,
                                 dict(EP.PARAMS, turn_enabled=False), log=lambda *a: None)
        assert counts3["parts_skipped"] == 2 and counts3["episodes"] == 0
        _, counts4 = EB.assemble([out / "parts" / "vehicle_0001.npz", part2], drives, out, roots.windows, EP.PARAMS, log=lambda *a: None)
        assert counts4["parts_skipped"] == 0 and counts4["episodes"] == 22
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_assembly_without_episodes_writes_empty_contract_files():
    """A drive with nothing to detect still yields the contract files (empty arrays, empty index with every column)."""
    TMP_BASE.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="vtt_episodes_empty_", dir=TMP_BASE))
    try:
        df = SD.make_drive(700, v=np.full(700, 15.0))                     # steady cruise, no lead, no turn
        SD.write_drive(df, root, "vehicle_0001", "drive_000001")
        SD.write_drives_index([SD.drives_index_row(df, "vehicle_0001", "drive_000001", "group_0001")], root)
        roots = W.Roots(root)
        drives = pd.read_parquet(roots.drives)
        out = root / "work" / "episodes"
        index, counts = EB.build_episodes(drives, roots, out, workers=1, log=lambda *a: None)
        assert counts["episodes"] == 0 and counts["grids"] == 0 and counts["drives_without_episodes"] == 1 and counts["part_errors"] == 0
        assert len(index) == 0 and list(index.columns) == [k for k, _ in EP.INDEX_COLUMNS]
        assert np.load(out / "E_f16.npy").shape == (0, 180, 22) and np.load(out / "G_f16.npy").shape == (0, 72, 6)
        assert list(pd.read_parquet(out / "episodes.parquet").columns) == [k for k, _ in EP.INDEX_COLUMNS]
        assert set(json.loads((out / "channels.json").read_text())) == {"crop", "grid", "types"}
        E, G, idx = EB.read_part(out / "parts" / "vehicle_0001.npz")[:3]
        assert E.shape[0] == 0 and G.shape[0] == 0 and len(idx) == 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_window_embeddings_hook_matches_head_input():
    torch = pytest.importorskip("torch")
    from experiments import window_embed as WE
    from experiments.data import CLIP
    from experiments.models import build_model
    torch.manual_seed(0)
    channels = list(C.OFFICIAL_INPUTS)
    model = build_model("gru", len(channels), 3, d=16, n_layers=1, dropout=0.0)
    mean = np.zeros(len(channels), np.float32)
    std = np.ones(len(channels), np.float32)
    ck = dict(state={k: v.clone() for k, v in model.state_dict().items()}, mean=mean, std=std,
              cfg=dict(d=16, n_layers=1, dropout=0.0), task="size", split_col="split_platform", model="gru",
              channels=channels, classes=("SMALL", "MID", "LARGE"))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(37, 600, len(channels))).astype(np.float16)
    X[:, :50, 13:] = np.nan                                            # lead channels start unknown

    class Data:
        def __init__(self):
            self.X = X
            self.channels = channels
            self.clip_lo = np.array([CLIP.get(c, (-np.inf, np.inf))[0] for c in channels], np.float32)
            self.clip_hi = np.array([CLIP.get(c, (-np.inf, np.inf))[1] for c in channels], np.float32)

        def batch(self, idx):
            return torch.from_numpy(np.ascontiguousarray(self.X[idx]))

    data = Data()
    TMP_BASE.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="vtt_wemb_", dir=TMP_BASE))
    try:
        path = tmp / "split_platform__size__gru__s0.pt"
        torch.save(ck, path)
        logs: list[str] = []
        emb = WE.window_embeddings(path, data, "cpu", batch=16, log=logs.append)
        assert emb.shape == (37, 64) and emb.dtype == np.float16 and np.isfinite(emb.astype(np.float32)).all()
        assert "embedded n=37 dim=64" in logs[0]
        part = WE.window_embeddings(path, data, "cpu", batch=16, limit=5, log=lambda *a: None)
        assert part.shape == (5, 64) and np.array_equal(part, emb[:5])
        # the hook captured the pooled features that enter the head's LayerNorm
        model2, prep = WE.model_from_checkpoint(ck, data.clip_lo, data.clip_hi, torch.device("cpu"))
        with torch.no_grad():
            h, _ = model2.gru(model2.stem(prep(data.batch(np.arange(37)))))
            z = torch.cat([h.mean(1), h.amax(1)], dim=-1).numpy()
        assert np.allclose(emb.astype(np.float32), z, atol=2e-2, rtol=1e-2)
        assert WE.embedding_dim(model2) == 64 and WE.output_paths(tmp, "t", 0)[0].name == "wemb_t.npy"
        assert WE.output_paths(tmp, "t", 2000)[1].name == "wemb_t_limit2000.json" and WE.ckpt_path(str(path)) == path
        # a recipe checkpoint carries auxiliary heads (CORAL / footprint / powertrain / domain) in its state, as train.py
        # writes them: the network is rebuilt with the stored heads, the state loads strictly and the hooked feature is the
        # one of the plain trunk; heads={} (a plain run) and a checkpoint without the key behave like the base case
        heads = {"coral": 3, "fp": 1, "pt4": 4, "dom": 5}
        model_h = build_model("gru", len(channels), 3, d=16, n_layers=1, dropout=0.0, heads=heads)
        missing, unexpected = model_h.load_state_dict(model.state_dict(), strict=False)
        assert not unexpected and missing and all(k.startswith(("heads.", "domain.")) for k in missing)
        for extra, name in ((heads, "fold1__size__gru__r2__s0"), ({}, "split_platform__size__gru__s1")):
            state = (model_h if extra else model).state_dict()
            ck2 = dict(ck, state={k: v.clone() for k, v in state.items()}, heads=extra, aux="coral+fp+pt4" if extra else "cls")
            torch.save(ck2, tmp / f"{name}.pt")
            emb2 = WE.window_embeddings(tmp / f"{name}.pt", data, "cpu", batch=16, log=lambda *a: None)
            assert emb2.shape == emb.shape and np.allclose(emb2.astype(np.float32), emb.astype(np.float32), atol=1e-3)
            m3, _ = WE.model_from_checkpoint(ck2, data.clip_lo, data.clip_hi, torch.device("cpu"))
            assert (m3.heads is not None and m3.domain is not None) if extra else (m3.heads is None and m3.domain is None)
        with pytest.raises(RuntimeError):                      # the trunk alone cannot take a state with head parameters
            build_model("gru", len(channels), 3, d=16, n_layers=1, dropout=0.0).load_state_dict(model_h.state_dict())
        # tag resolution: experiments/ckpt first, then experiments/recipe/ckpt (fold runs); a missing tag names the first
        rec = tmp / "experiments" / "recipe" / "ckpt" / "fold1__size__gru__r2__s0.pt"
        rec.parent.mkdir(parents=True)
        rec.write_bytes(b"")
        assert WE.ckpt_path("fold1__size__gru__r2__s0", tmp) == rec
        main_ck = tmp / "experiments" / "ckpt" / "fold1__size__gru__r2__s0.pt"
        main_ck.parent.mkdir(parents=True)
        main_ck.write_bytes(b"")
        assert WE.ckpt_path("fold1__size__gru__r2__s0", tmp) == main_ck
        assert WE.ckpt_path("nope", tmp) == tmp / "experiments" / "ckpt" / "nope.pt" and not WE.ckpt_path("nope", tmp).exists()
        # a --limit smoke run never defaults into work/tensors, where the MIL loader globs the full embeddings
        assert WE.default_out_dir(tmp, 0) == tmp / "work" / "tensors" and WE.default_out_dir(tmp, 2000) == tmp / "tmp"
        bad = Data()
        bad.channels = channels[::-1]
        with pytest.raises(ValueError):
            WE.window_embeddings(path, bad, "cpu")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_one_row_segment_does_not_break_derivatives():
    """A gap row directly followed by another gap row leaves a one-row gap-free segment: the smoothed derivative must
    skip it (NaN) instead of raising, and detection must still run on the other segments."""
    import numpy as np
    from experiments import episodes as EP
    n = 400
    t = np.arange(n) * 0.1
    v = np.clip(t - 5.0, 0, 12.0).astype(np.float64)            # standstill, then a ramp to 12 m/s
    d = {c: np.zeros(n, dtype=np.float64) for c in EP.EP_CHANNELS if c not in ("a_sg", "valid")}
    d["v_mps"] = v
    d["a_mps2"] = np.gradient(v, 0.1)
    d["standstill"] = (v < 0.05).astype(np.float64)
    d["gap"] = np.zeros(n, dtype=bool)
    d["gap"][200] = True
    d["gap"][201] = True                                          # rows 200..200 form a one-row segment
    d["das_any"] = np.zeros(n, dtype=bool)
    d["yaw_valid"] = np.ones(n, dtype=bool)
    d["lead_valid"] = np.zeros(n, dtype=np.float64)
    for c in ("lead_d_m", "lead_v_rel_mps", "lead_speed_mps", "lead_a_slope_mps2", "thw_s", "ttc_s"):
        d[c] = np.full(n, np.nan)
    d["__n__"] = np.array([n])
    EP.ensure_derived(d)
    assert np.isnan(d["a_sg"][200]) and np.isfinite(d["a_sg"][:199]).all()
    eps = EP.detect_episodes(d)
    assert any(e.type_id == EP.TYPES.index("ACC") for e in eps)
