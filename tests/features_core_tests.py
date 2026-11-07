"""vtt.features_core: Savitzky-Golay kernel, speed episodes on ramps, lead-acceleration events, ego response."""
from __future__ import annotations

import numpy as np

from vtt import config as C
from vtt import features_core as F

DT = C.NOMINAL_DT_S


def test_savgol_kernel_matches_classic_coefficients():
    c = F.savgol_kernel(5, 2)
    np.testing.assert_allclose(c, np.array([-3, 12, 17, 12, -3]) / 35.0, atol=1e-12)
    assert abs(c.sum() - 1.0) < 1e-12
    # the first-derivative kernel of a linear ramp returns its slope
    x = 2.0 * np.arange(21) * DT
    d = F.savgol_smooth(x, 5, 2, deriv=1, dt=DT)
    np.testing.assert_allclose(d, 2.0, atol=1e-9)


def test_savgol_smooth_preserves_polynomials_and_handles_nan():
    t = np.arange(50) * DT
    x = 1.0 + 3.0 * t - 0.5 * t ** 2
    np.testing.assert_allclose(F.savgol_smooth(x, 5, 2), x, atol=1e-9)
    y = x.copy()
    y[10] = np.nan
    y[0] = np.nan
    s = F.savgol_smooth(y, 5, 2)
    assert np.isfinite(s).all()
    assert abs(s[10] - x[10]) < 0.05
    assert np.isnan(F.savgol_smooth(np.full(10, np.nan), 5, 2)).all()
    assert len(F.savgol_smooth(np.array([1.0, 2.0]), 5, 2)) == 2


def test_monotone_runs_tolerate_eps_dips():
    v = np.array([0, 1, 2, 1.9, 3, 4, 3.0, 2.0, 1.0, 0.5])
    runs = F.monotone_runs(v, eps=0.3, increasing=True)
    assert runs[0].tolist() == [0, 5]          # the 1.9 dip (0.1 below the running max) is tolerated, 3.0 after 4 is not
    dec = F.monotone_runs(v, eps=0.3, increasing=False)
    assert dec[-1].tolist() == [5, 9]
    assert len(F.monotone_runs(np.array([np.nan, np.nan]), 0.3)) == 0


def _ramp(segments):
    """segments = [(duration_s, dv_total)] -> speed trace starting at 10 m/s."""
    v = [10.0]
    for dur, dv in segments:
        n = int(round(dur / DT))
        base = v[-1]
        v.extend([base + dv * (k + 1) / n for k in range(n)])
    return np.array(v)


def test_accel_episode_on_ramp():
    # 20 s cruise, 5 s ramp +5 m/s (1 m/s^2), 20 s cruise, 4 s ramp -8 m/s, 10 s cruise
    v = _ramp([(20, 0.0), (5, 5.0), (20, 0.0), (4, -8.0), (10, 0.0)])
    acc = F.accel_episodes(v, DT, **C.ACCEL_EP)
    dec = F.decel_episodes(v, DT, **C.ACCEL_EP)
    assert len(acc) == 1 and len(dec) == 1
    i0, i1, dv, mean_a = acc[0]
    assert abs(dv - 5.0) < 0.2 and mean_a > 0.1 and i0 == 0 and i1 > 250   # literal rule: the leading flat cruise joins the monotone run
    j0, j1, dv2, mean_a2 = dec[0]
    assert dv2 < -7.5 and mean_a2 < -0.1 and 240 <= j0 <= 460 and j1 == len(v)   # the detectors are independent: the decel run starts at the speed plateau
    # with the trim switch the same manoeuvres are found, bounded to the ramps (+- half the trim window)
    trimmed = F.accel_episodes(v, DT, **dict(C.ACCEL_EP, trim_win_s=2.0))
    assert len(trimmed) == 1 and 185 <= trimmed[0][0] <= 201 and 250 <= trimmed[0][1] <= 266 and abs(trimmed[0][2] - 5.0) < 0.2
    assert trimmed[0][3] > 0.7                                                # 5 m/s over about 5-7 s instead of 25 s
    dtr = F.decel_episodes(v, DT, **dict(C.ACCEL_EP, trim_win_s=2.0))
    assert len(dtr) == 1 and 435 <= dtr[0][0] <= 451 and 490 <= dtr[0][1] <= 506


def _cruise_with_ramp(pre_s: float, ramp_s: float = 3.0, dv: float = 2.5, n: int = 600, v0: float = 25.0) -> np.ndarray:
    t = np.arange(n) * DT
    v = np.full(n, v0)
    m = (t >= pre_s) & (t < pre_s + ramp_s)
    v[m] = v0 + dv * (t[m] - pre_s) / ramp_s
    v[t >= pre_s + ramp_s] = v0 + dv
    return v


def test_cruise_dilution_and_trim_switch():
    """A 3-s +2.5 m/s acceleration (0.83 m/s^2) inside a 60-s steady cruise: the literal rule (trim_win_s=0) tests the
    whole monotone run (cruise included, mean 0.04 m/s^2) and finds nothing; with the trim switch the manoeuvre
    alone is tested and counted, wherever it sits in the window."""
    for pre_s in (5, 20, 40, 50):
        v = _cruise_with_ramp(pre_s)
        assert F.accel_episodes(v, DT, **dict(C.ACCEL_EP, trim_win_s=0.0)) == []
        ep = F.accel_episodes(v, DT, **dict(C.ACCEL_EP, trim_win_s=2.0))
        assert len(ep) == 1, pre_s
        i0, i1, dv, mean_a = ep[0]
        assert abs(dv - 2.5) < 0.15 and mean_a >= 0.4 and (i1 - i0) * DT <= 5.5
        assert abs(i0 * DT - pre_s) <= 1.05 and abs(i1 * DT - (pre_s + 3.0)) <= 1.15
    # the current config default is the literal rule
    assert C.ACCEL_EP.get("trim_win_s", 0.0) == 0.0
    # a plain steady cruise with small noise never yields an episode under either rule
    rng = np.random.default_rng(1)
    v = 25.0 + rng.normal(0, 0.03, 600)
    assert F.accel_episodes(v, DT, **dict(C.ACCEL_EP, trim_win_s=2.0)) == [] and F.decel_episodes(v, DT, **dict(C.ACCEL_EP, trim_win_s=2.0)) == []
    # slow 40-s creep of 2.5 m/s (0.06 m/s^2) stays rejected with the trim: the slope never reaches min_mean_a
    assert F.accel_episodes(_ramp([(40, 2.5)]), DT, **dict(C.ACCEL_EP, trim_win_s=2.0)) == []


def test_active_slope_mask():
    v = _cruise_with_ramp(20.0, ramp_s=4.0, dv=4.0, n=400)
    vs = F.savgol_smooth(v, 5, 2)
    act = F.active_slope_mask(vs, DT, 1, 0.1, 2.0)
    assert not act[:190].any() and act[205:235].all() and not act[255:].any()
    assert not F.active_slope_mask(vs, DT, -1, 0.1, 2.0).any()
    assert len(F.active_slope_mask(np.zeros(0), DT, 1, 0.1, 2.0)) == 0


def test_episode_thresholds_reject_small_slow_or_short_changes():
    # +1.5 m/s only (below min_dv), then 2.5 m/s spread over 40 s (mean a 0.06 < 0.1)
    v1 = _ramp([(5, 1.5), (10, 0.0)])
    assert F.accel_episodes(v1, DT, **C.ACCEL_EP) == []
    v2 = _ramp([(40, 2.5)])
    assert F.accel_episodes(v2, DT, **C.ACCEL_EP) == []
    # a strong but 2-s long change: 2.4 m/s in 2 s with nothing else -> fewer than min_s samples
    v3 = _ramp([(2, 2.4)])
    assert F.accel_episodes(v3, DT, **C.ACCEL_EP) == []
    # noise below eps does not break a real episode
    rng = np.random.default_rng(0)
    v4 = _ramp([(10, 0.0), (6, 4.0), (10, 0.0)]) + rng.normal(0, 0.03, 261)
    assert len(F.accel_episodes(v4, DT, **C.ACCEL_EP)) == 1


def test_lead_accel_events_need_quiet_then_hold():
    n = 300
    valid = np.ones(n, bool)
    slope = np.zeros(n)
    slope[100:130] = 1.0                   # 3 s of +1 m/s^2 after 10 s quiet
    ev = F.lead_accel_events(slope, valid, DT, **C.LEAD_ACCEL_EVENT)
    assert ev.tolist() == [100]
    # same slope but the quiet period is broken by a -0.5 blip 1 s before the onset
    s2 = slope.copy()
    s2[90] = -0.5
    assert F.lead_accel_events(s2, valid, DT, **C.LEAD_ACCEL_EVENT).tolist() == []
    # hold too short (0.5 s)
    s3 = np.zeros(n)
    s3[100:105] = 1.0
    assert F.lead_accel_events(s3, valid, DT, **C.LEAD_ACCEL_EVENT).tolist() == []
    # an episode break (lead lost) inside the quiet period prevents the onset
    v4 = valid.copy()
    v4[95] = False
    assert F.lead_accel_events(slope, v4, DT, **C.LEAD_ACCEL_EVENT).tolist() == []
    # NaN slope (first samples of an episode) counts as not quiet
    s5 = slope.copy()
    s5[85] = np.nan
    assert F.lead_accel_events(s5, valid, DT, **C.LEAD_ACCEL_EVENT).tolist() == []
    # horizon requirement: the episode must continue 8 s after the onset
    v6 = valid.copy()
    v6[150:] = False
    assert F.lead_accel_events(slope, v6, DT, horizon_s=8.0).tolist() == []
    assert F.lead_accel_events(slope, valid, DT, horizon_s=8.0).tolist() == [100]
    # two separated events in one episode
    s7 = np.zeros(n)
    s7[50:65] = 0.8
    s7[200:215] = 0.8
    assert F.lead_accel_events(s7, valid, DT, **C.LEAD_ACCEL_EVENT).tolist() == [50, 200]


def test_ego_response_delay():
    a = np.zeros(200)
    a[120:140] = 0.5                       # response 2 s after an onset at 100
    assert abs(F.ego_response_delay(a, 100, DT, **C.LEAD_ACCEL_EVENT) - 2.0) < 1e-9
    a2 = np.zeros(200)
    a2[190:] = 0.5                         # beyond the 8 s horizon
    assert np.isnan(F.ego_response_delay(a2, 100, DT, **C.LEAD_ACCEL_EVENT))
    a3 = np.zeros(200)
    a3[110:113] = 0.5                      # 0.3 s only, below resp_hold_s
    assert np.isnan(F.ego_response_delay(a3, 100, DT, **C.LEAD_ACCEL_EVENT))
