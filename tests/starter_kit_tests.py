"""Starter kit on synthetic data: the synthetic sample itself (schema, size), window features (episodes,
lift-off deceleration, steering gain, catch-up events on a hand-built scenario), protocol masks and metrics,
loading and window tensors."""
from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starter_kit import eval as EV
from starter_kit import features as F
from starter_kit import load as L
from vtt import config as C
from vtt import schema as S
from vtt.paths import P
from vtt.synth_sample import PROFILES, synth_drive, write_sample


@pytest.fixture(scope="module")
def sample():
    P.TMP.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="vtt_sk_", dir=P.TMP))
    info = write_sample(d / "sample")
    yield d / "sample", info
    if not os.environ.get("VTT_KEEP_TMP"):
        shutil.rmtree(d, ignore_errors=True)


def test_sample_schema_and_size(sample):
    root, info = sample
    assert info["n_drives"] == 4 and info["n_windows"] == 8 and info["bytes"] <= 500_000
    for p in (root / "data").rglob("*.parquet"):
        assert S.check_public_parquet(p, S.DRIVE_COLUMNS, "drive") == []
    assert S.check_public_parquet(root / "index" / "drives.parquet", S.DRIVES_INDEX_COLUMNS, "drives") == []
    assert S.check_public_parquet(root / "index" / "windows.parquet", S.WINDOWS_INDEX_COLUMNS, "windows") == []
    drives = pd.read_parquet(root / "index" / "drives.parquet")
    assert set(drives.powertrain) == {"BEV", "ICE"} and set(drives.split_platform) == {"train", "test"}
    assert (drives.file.str.startswith("data/")).all()


def test_features_table_keys_and_physics(sample):
    root, _ = sample
    windows = L.load_index("windows", root=root); drives = L.load_index("drives", root=root)
    ft = F.features_table(windows, lambda d: L.load_drive(d, root=root, drives=drives))
    assert list(ft.columns) == ["window_id", *F.FEATURE_NAMES] and len(ft) == len(windows)
    assert ft.v_mean.between(0, 40).all() and ft.moving_frac.between(0, 1).all() and ft.lead_frac.between(0, 1).all()
    # per drive (more samples than one window): steering gain recovers the profile constant, BEV lifts off harder
    k, coast = {}, {}
    for r in drives.itertuples(index=False):
        df = L.load_drive(r.drive_id, root=root, drives=drives)
        k.setdefault(r.powertrain, []).append(F.k_steer(df))
        coast.setdefault(r.powertrain, []).append(F.coast_features(df)["coast_decel_mean"])
    assert abs(np.mean(k["BEV"]) - PROFILES["bev"]["k_steer_degpm"]) / PROFILES["bev"]["k_steer_degpm"] < 0.1
    assert abs(np.mean(k["ICE"]) - PROFILES["ice"]["k_steer_degpm"]) / PROFILES["ice"]["k_steer_degpm"] < 0.1
    assert np.nanmean(coast["BEV"]) < np.nanmean(coast["ICE"]) < 0           # every synthetic drive has a lift-off phase


def test_accel_episode_rules():
    dt = 0.1
    v = np.concatenate([np.zeros(20), np.linspace(0, 10, 51)[1:], np.full(80, 10.0)])   # 5 s ramp at 2 m/s^2, then flat
    eps = F.accel_episodes(v, 1)
    assert len(eps) == 1 and F.accel_episodes(v, -1) == []
    i0, i1 = eps[0]
    assert i0 <= 20 and i1 == len(v)                      # the windowed-max rule keeps the trailing plateau inside the run
    st = F.episode_stats(v, i0, i1, 1)
    assert 9.0 <= st["dv"] <= 10.5 and 1.6 <= st["peak_a"] <= 2.4 and 1.5 <= st["t_peak_s"] <= 7.5 and st["dur_s"] == pytest.approx((i1 - i0) * dt)
    assert 0.2 <= st["plateau_frac"] <= 0.45                 # about the 5 s ramp out of the 15 s run
    assert F.accel_episodes(np.full(300, 12.0), 1) == [] and F.accel_episodes(v[::-1], -1)                   # mirrored ramp = deceleration
    k = F.sg_kernel(5, 2)
    assert np.allclose(k, np.array([-3, 12, 17, 12, -3]) / 35.0)
    assert np.allclose(F.smooth(np.arange(50.0), 5, 2), np.arange(50.0))                                    # linear signals pass unchanged
    assert np.allclose(F.smooth(np.arange(50.0) * 0.1, 5, 2, deriv=1, dt=dt), 1.0)                          # slope 0.1 / 0.1 s = 1 m/s^2


def test_coast_features_hand_built_and_profiles():
    n = 100
    df = pd.DataFrame({"v_mps": np.full(n, 8.0), "a_mps2": np.full(n, -0.6), "gas_pressed": False, "brake_pressed": False,
                       "das_any": False, "standstill": False})
    df.loc[:9, "gas_pressed"] = True                        # pedal samples are excluded
    df.loc[10:19, "a_mps2"] = 0.0                           # not decelerating -> not coasting
    out = F.coast_features(df)
    assert out["coast_decel_mean"] == pytest.approx(-0.6) and out["coast_frac"] == pytest.approx(0.8)
    assert np.isnan(F.coast_features(df.head(12))["coast_decel_mean"])
    bev = [F.coast_features(synth_drive(s, 1200, "bev"))["coast_decel_mean"] for s in range(4)]
    ice = [F.coast_features(synth_drive(s, 1200, "ice"))["coast_decel_mean"] for s in range(4)]
    assert np.isfinite(bev).all() and np.isfinite(ice).all() and np.mean(bev) < np.mean(ice) < 0


def catchup_frame(delay_rows: int = 8, thw: float = 2.0, das: bool = False, n: int = 600, onset: int = 300) -> pd.DataFrame:
    """Lead at constant speed, then +3 m/s over 3 s (slope 1 m/s^2); ego accelerates 0.5 m/s^2 delay_rows after the onset."""
    v_lead = 10.0 + np.clip((np.arange(n) - onset) * 0.1 / 3.0, 0, 1) * 3.0
    slope = np.zeros(n); slope[onset:onset + 30] = 1.0
    a = np.zeros(n); a[onset + delay_rows:onset + delay_rows + 25] = 0.5
    d = 20.0 + np.cumsum(np.concatenate([[0.0], np.diff(v_lead)])) * 0.0 + np.concatenate([[0.0], np.cumsum(v_lead[1:] - 10.0) * 0.1])
    return pd.DataFrame({"lead_valid": True, "lead_a_slope_mps2": slope, "a_mps2": a, "lead_d_m": d, "thw_s": thw,
                         "das_any": das, "v_mps": 10.0})


def test_catchup_events_hand_built():
    df = catchup_frame()
    ev = F.catchup_events(df)
    assert len(ev) == 1
    e = ev[0]
    assert e["onset_row"] == 300 and e["responded"] and abs(e["delay_s"] - 0.8) < 1e-9
    assert abs(e["gap_change_5s_m"] - (df.lead_d_m[350] - df.lead_d_m[300])) < 1e-9 and e["gap_change_5s_m"] > 0
    assert abs(e["accel_ratio"] - 0.5) < 1e-6 and e["das_at_onset"] is False
    assert F.catchup_events(catchup_frame(das=True)) == [] and len(F.catchup_events(catchup_frame(das=True), human_only=False)) == 1
    assert F.catchup_events(catchup_frame(thw=8.0)) == []
    late = F.catchup_events(catchup_frame(delay_rows=120))[0]                                 # beyond the 8 s horizon
    assert late["responded"] is False and np.isnan(late["delay_s"])
    feats = F.catchup_features(df)
    assert feats["n_catchup_events"] == 1 and feats["catchup_responded_frac"] == 1.0 and abs(feats["catchup_delay_mean_s"] - 0.8) < 1e-9


def test_lead_accel_onsets_transition_allowance():
    n = 400
    slope = np.zeros(n); slope[200:203] = (0.15, 0.35, 0.45); slope[203:230] = 1.0      # gradual climb through the (0.3, 0.5) band
    valid = np.ones(n, bool)
    assert list(F.lead_accel_onsets(slope, valid, C.LEAD_ACCEL_EVENT)) == [203]
    assert list(F.lead_accel_onsets(slope, valid, C.LEAD_ACCEL_EVENT, trans_s=0.0)) == []     # the strict rule misses it
    valid[150] = False                                                                       # a gap inside the quiet window
    assert list(F.lead_accel_onsets(slope, valid, C.LEAD_ACCEL_EVENT)) == [203]
    valid[190] = False
    assert list(F.lead_accel_onsets(slope, valid, C.LEAD_ACCEL_EVENT)) == []


def windows_frame() -> pd.DataFrame:
    rows = []
    for i in range(12):
        rows.append(dict(window_id=f"drive_{i // 2 + 1:06d}_w{i % 2}", drive_id=f"drive_{i // 2 + 1:06d}", group_id=f"group_{i // 4 + 1:04d}",
                         platform=["P_A", "P_B", "P_C"][i // 4], eligible=i != 11, task_size_eligible=i < 10, task_bev_ice_eligible=i % 3 != 0,
                         size_class=["SMALL", "MID", "LARGE"][i % 3], powertrain=["BEV", "ICE", "HEV", "BEV"][i % 4],
                         split_platform="test" if i < 8 else "train", split_device="test" if i % 2 else "train",
                         human_only_strict=i % 2 == 0, low_support_cell=i in (4, 5)))
    return pd.DataFrame(rows)


def test_protocol_masks():
    w = windows_frame()
    full = EV.protocol_mask(w, "size", "test_full")
    assert full.sum() == 8 and not full[8:].any()                                               # eligible & size-eligible & test
    assert EV.protocol_mask(w, "size", "test_full", split_col="split_device").sum() == 5     # odd i < 10
    human = EV.protocol_mask(w, "size", "test_human_only")
    assert human.sum() == 4 and (human <= full).all()
    bal = EV.protocol_mask(w, "size", "test_balanced", balanced_ids={w.window_id[0], w.window_id[3], "missing"})
    assert bal.sum() == 2
    with pytest.raises(ValueError):
        EV.protocol_mask(w, "size", "test_balanced")
    bi = EV.protocol_mask(w, "bev_ice", "test_full")
    assert bi.sum() == sum(1 for i in range(8) if i % 3 != 0 and ["BEV", "ICE", "HEV", "BEV"][i % 4] in ("BEV", "ICE"))
    assert EV.protocol_mask(w, "powertrain4", "test_full", split="train").sum() == 3
    with pytest.raises(ValueError):
        EV.protocol_mask(w, "size", "nope")


def test_metrics_and_evaluate(tmp_path_factory):
    y_true = ["A", "A", "B", "B", "C"]
    y_pred = ["A", "B", "B", "B", "A"]
    cm = EV.confusion_matrix(y_true, y_pred, ["A", "B", "C"])
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    f1 = EV.macro_f1(y_true, y_pred, ["A", "B", "C"])
    assert abs(f1 - np.mean([2 * 1 / (2 * 1 + 1 + 1), 2 * 2 / (2 * 2 + 1 + 0), 0.0])) < 1e-9
    assert np.isnan(EV.macro_f1([], [], ["A"])) and EV.accuracy(y_true, y_pred) == 0.6
    w = windows_frame()
    out = Path(tempfile.mkdtemp(prefix="vtt_eval_", dir=P.TMP))
    pred = pd.Series(w.size_class.to_numpy(), index=w.window_id)                                    # perfect predictions
    res = EV.evaluate(w, pred, "size", "test_full", out_dir=out)
    assert res["n_windows"] == 8 and res["macro_f1"] == 1.0 and res["macro_f1_supported"] == 1.0 and res["accuracy"] == 1.0
    assert Path(res["confusion_csv"]).exists() and (out / "size_test_full_split_platform_test_per_cell.csv").exists()
    assert set(res["per_cell"].columns) == {"size_class", "powertrain", "n", "accuracy", "low_support_cell"} and res["per_cell"].n.sum() == 8
    wrong = {wid: "MID" for wid in w.window_id}                                                     # constant prediction
    res2 = EV.evaluate(w, wrong, "size", "test_full")
    assert res2["macro_f1"] < 0.5 and res2["majority_f1"] <= res2["macro_f1"] + 1e-9
    sup = w[EV.protocol_mask(w, "size", "test_full") & ~w.low_support_cell]
    assert abs(res2["macro_f1_supported"] - EV.macro_f1(sup.size_class, ["MID"] * len(sup), list(C.SIZE_CLASSES))) < 1e-9
    with pytest.raises(ValueError):
        EV.evaluate(w, ["MID"] * 3, "size", "test_full")
    shutil.rmtree(out, ignore_errors=True)


def test_loading_and_tensors(sample):
    root, _ = sample
    windows = L.load_index("windows", root=root); drives = L.load_index("drives", root=root)
    assert len(windows) == 8 and len(drives) == 4 and isinstance(L.load_index("splits", root=root), dict)
    with pytest.raises(FileNotFoundError):
        L.load_index("vehicles", root=root)
    ids = L.load_balanced_ids("bev_ice", "test", root=root)
    assert ids and ids <= set(windows.window_id)
    d1 = L.load_drive("drive_000001", root=root, drives=drives)
    d2 = L.load_drive(drives.file.iloc[0], root=root)
    assert d1.equals(d2) and len(d1) == 1200
    with pytest.raises(KeyError):
        L.load_drive("drive_999999", root=root, drives=drives)
    x = L.window_tensor(d1, 0, 600)
    assert x.shape == (600, len(C.OFFICIAL_INPUTS)) and x.dtype == np.float32
    j = C.OFFICIAL_INPUTS.index("das_enabled")
    assert set(np.unique(x[:, j])) <= {0.0, 1.0}
    with pytest.warns(UserWarning):
        y = L.window_tensor(d1, 0, 10, channels=("v_mps", "not_a_channel"))
    assert np.isnan(y[:, 1]).all() and np.allclose(y[:, 0], d1.v_mps.to_numpy()[:10])
    got = list(L.iter_windows("test", "test_human_only", "bev_ice", root=root))
    assert len(got) == 2 and all(t.shape == (600, len(C.OFFICIAL_INPUTS)) for _, t in got)
    assert all(bool(row["human_only_strict"]) for row, _ in got)
    assert len(list(L.iter_windows("train", "test_full", "size", root=root, limit=3))) == 3
    assert len(list(L.iter_windows("test", "test_balanced", "bev_ice", root=root))) == len(ids)


def test_balanced_lists_per_split_scheme(sample):
    root, _ = sample
    assert L.balanced_list_path("size", "test", "split_device") == "splits/eval_balanced_size_test_device.parquet"
    assert L.balanced_list_path("bev_ice") == "splits/eval_balanced_bev_ice_test.parquet"
    with pytest.raises(ValueError):
        L.balanced_list_path("size", "test", "nope")
    ids_p = L.load_balanced_ids("bev_ice", "test", root=root)
    ids_d = L.load_balanced_ids("bev_ice", "test", root=root, split_col="split_device")
    assert ids_p and ids_p == ids_d                                                  # the sample's two schemes coincide
    got = list(L.iter_windows("test", "test_balanced", "bev_ice", root=root, split_col="split_device"))
    assert len(got) == len(ids_d) and all(row["split_device"] == "test" for row, _ in got)
    # a synthetic splits/ folder whose two schemes differ (S8 naming: no suffix = split_platform, _device = split_device)
    d = Path(tempfile.mkdtemp(prefix="vtt_bal_", dir=P.TMP))
    w = windows_frame()
    (d / "splits").mkdir()
    plat = w[(w.split_platform == "test") & w.eligible & w.task_size_eligible][["window_id"]]
    dev = w[(w.split_device == "test") & w.eligible & w.task_size_eligible][["window_id"]]
    plat.to_parquet(d / "splits" / "eval_balanced_size_test.parquet", index=False)
    dev.to_parquet(d / "splits" / "eval_balanced_size_test_device.parquet", index=False)
    assert L.load_balanced_ids("size", root=d) == set(plat.window_id) != set(dev.window_id)
    assert L.load_balanced_ids("size", root=d, split_col="split_device") == set(dev.window_id)
    with warnings.catch_warnings():                                                    # matching scheme: no warning, exact ids
        warnings.simplefilter("error")
        m = EV.protocol_mask(w, "size", "test_balanced", split_col="split_device", balanced_ids=set(dev.window_id))
    assert set(w.window_id[m]) == set(dev.window_id)
    with pytest.warns(UserWarning):                                                    # the other scheme's list: train windows dropped
        m2 = EV.protocol_mask(w, "size", "test_balanced", split_col="split_device", balanced_ids=set(plat.window_id))
    assert (w.split_device[m2] == "test").all() and 0 < m2.sum() < len(plat)
    shutil.rmtree(d, ignore_errors=True)
