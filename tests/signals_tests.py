"""vtt.signals: normalisation of extractor frames, causal hold, sample-time statistics, segment ordinal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vtt import signals as SG


def test_signal_columns_match_the_extractor():
    try:
        from vtt.tocf_bridge import tocf
        upstream = tocf("schema").SIGNAL_COLUMNS
    except Exception:  # noqa: BLE001
        pytest.skip("tocf not importable")
    for topic in SG.TOPICS:
        assert list(upstream[topic]) == SG._SIGNAL_COLUMNS_MIRROR[topic] == SG.signal_columns(topic)


def test_normalise_adds_missing_columns_casts_and_sorts_by_time():
    df = pd.DataFrame(dict(logMonoTime=np.array([30, 10, 20], np.int64), seg=np.array([1, 0, 0], np.int64),
                           leadOne_status=[True, False, True], leadOne_dRel=[3.0, 1.0, 2.0],
                           leadOne_radar=np.full(3, np.nan)))                   # all-missing float column (dtype drift)
    out = SG.normalise(df, "radarState")
    assert list(out.columns) == ["logMonoTime", "seg"] + SG.signal_columns("radarState")
    assert out["logMonoTime"].tolist() == [10, 20, 30] and out["seg"].tolist() == [0, 0, 1]
    assert out["logMonoTime"].dtype == np.int64 and out["seg"].dtype == np.int64
    assert out["leadOne_status"].tolist() == [0.0, 1.0, 1.0] and out["leadOne_status"].dtype == np.float64
    assert out["leadOne_dRel"].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(out["leadOne_radar"]).all() and np.isnan(out["leadOne_modelProb"]).all()   # drifted / absent -> NaN
    assert out["leadTwo_radarTrackId"].dtype == np.float64


def test_normalise_strings_and_nan_times():
    df = pd.DataFrame(dict(logMonoTime=[5.0, np.nan, 1.0], seg=[0, 0, 0], pose_source=["livePose", None, "liveLocationKalman"],
                           angularVelocityDevice_z=[0.1, 0.2, 0.3], inputsOK=["True", "False", None]))
    out = SG.normalise(df, "pose")
    assert len(out) == 2 and out["logMonoTime"].tolist() == [1, 5]
    assert out["pose_source"].tolist() == ["liveLocationKalman", "livePose"]
    assert out["inputsOK"].tolist()[1] == 1.0 and np.isnan(out["inputsOK"].tolist()[0])
    empty = SG.normalise(None, "carState")
    assert len(empty) == 0 and "vEgo" in empty.columns


def test_as_bool_and_seg_ordinal():
    assert SG.as_bool(np.array([1.0, 0.0, np.nan, 0.7])).tolist() == [True, False, False, True]
    assert SG.as_bool(np.array([True, False])).tolist() == [True, False]
    assert SG.seg_ordinal(np.array([5, 5, 9, 9, 2, -1, 9])).tolist() == [0, 0, 1, 1, 2, -1, 1]


def test_hold_is_causal_with_max_age():
    t_src = np.array([1.0, 2.0, 3.0]); vals = np.array([10.0, np.nan, 30.0]); flags = np.array([1.0, np.nan, 0.0])
    t_q = np.array([0.5, 1.0, 1.2, 2.5, 3.9])
    j, age, fresh = SG.hold(t_src, t_q, 0.5)
    assert fresh.tolist() == [False, True, True, True, False]
    assert np.allclose(age[fresh], [0.0, 0.2, 0.5]) and np.isnan(age[~fresh]).all()
    held = SG.hold_float(vals, j, fresh)
    assert held[1] == 10.0 and held[2] == 10.0 and np.isnan(held[3]) and np.isnan(held[0]) and np.isnan(held[4])
    assert SG.hold_bool(flags, j, fresh).tolist() == [False, True, True, False, False]
    j0, age0, fresh0 = SG.hold(np.zeros(0), t_q, 0.5)
    assert not fresh0.any() and np.isnan(age0).all() and np.isnan(SG.hold_float(np.zeros(0), j0, fresh0)).all()


def test_dt_stats_gap_and_drift():
    dt = np.full(100, 0.1003); dt[0] = 0.0; dt[50] += 2.0
    t = np.cumsum(dt)
    s = SG.dt_stats(t)
    assert s["n_gaps"] == 1 and abs(s["max_gap_s"] - 2.1003) < 1e-9
    assert abs(s["dt_med_s"] - 0.1003) < 1e-9 and abs(s["gap_total_s"] - 2.0003) < 1e-9
    # the gap row's whole excess over the nominal dt counts as gap time, so the drift is carried by the other 98 intervals
    assert abs(s["drift_s"] - 98 * 0.0003) < 1e-6 and abs(s["nominal_span_s"] - 9.9) < 1e-9
    assert abs(s["real_span_s"] - (99 * 0.1003 + 2.0)) < 1e-9
    one = SG.dt_stats(np.array([0.0]))
    assert one["n_gaps"] == 0 and np.isnan(one["dt_med_s"])
