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


# ----------------------------------------------------------------------------- drive- and vehicle-level protocols
def unit_frame() -> pd.DataFrame:
    """Hand-built windows index: drive A (vehicle V1, two eligible human-only windows), drive B (V1, two eligible windows with
    automation), drive C (V2, one ineligible window), drive D (V3, train split). Labels are consistent per drive."""
    rows = []

    def add(drive, vehicle, group, w, eligible, human, powertrain, split="test"):
        rows.append(dict(window_id=f"{drive}_w{w}", drive_id=drive, vehicle_id=vehicle, group_id=group, model=f"car_{vehicle}",
                         eligible=eligible, task_size_eligible=True, task_bev_ice_eligible=True, size_class="MID", powertrain=powertrain,
                         split_platform=split, split_device=split, human_only_strict=human, low_support_cell=False))
    add("drive_A", "V1", "G1", 0, True, True, "ICE"); add("drive_A", "V1", "G1", 1, True, True, "ICE")
    add("drive_B", "V1", "G1", 0, True, False, "ICE"); add("drive_B", "V1", "G1", 1, True, False, "ICE")
    add("drive_C", "V2", "G2", 0, False, True, "BEV")
    add("drive_D", "V3", "G3", 0, True, True, "BEV", split="train")
    return pd.DataFrame(rows)


def unit_drives() -> pd.DataFrame:
    """Drives index matching unit_frame plus drive E (vehicle V4, test, no window at all)."""
    return pd.DataFrame([dict(drive_id="drive_A", vehicle_id="V1", group_id="G1", model="car_V1", powertrain="ICE", size_class="MID"),
                         dict(drive_id="drive_B", vehicle_id="V1", group_id="G1", model="car_V1", powertrain="ICE", size_class="MID"),
                         dict(drive_id="drive_C", vehicle_id="V2", group_id="G2", model="car_V2", powertrain="BEV", size_class="MID"),
                         dict(drive_id="drive_D", vehicle_id="V3", group_id="G3", model="car_V3", powertrain="BEV", size_class="MID"),
                         dict(drive_id="drive_E", vehicle_id="V4", group_id="G4", model="car_V4", powertrain="BEV", size_class="MID")]
                        ).assign(split_platform=lambda d: np.where(d.drive_id == "drive_D", "train", "test"),
                                 split_device=lambda d: np.where(d.drive_id == "drive_D", "train", "test"),
                                 task_size_eligible=True, task_bev_ice_eligible=True)


def test_unit_protocol_populations_and_tie_rule():
    w = unit_frame()
    sub = w[EV.protocol_mask(w, "bev_ice", "test_full")]
    assert sorted(sub.drive_id.unique()) == ["drive_A", "drive_B"]
    # drive A leans ICE, drive B is an exact tie -> earlier class (BEV) by the tie rule
    probs = pd.DataFrame({"BEV": [0.2, 0.3, 0.5, 0.5], "ICE": [0.8, 0.7, 0.5, 0.5]}, index=sub.window_id)
    res = EV.evaluate_units(w, probs, "bev_ice", "drive_full")
    assert res["protocol"] == "drive_full" and res["window_protocol"] == "test_full" and res["unit_col"] == "drive_id"
    assert res["n_units"] == 2 and res["n_units_without_windows"] == 1 and res["n_windows_used"] == 4 and res["n_models"] == 1
    assert res["confusion"] == [[0, 0], [1, 1]] and res["accuracy"] == 0.5 and res["support"] == {"BEV": 0, "ICE": 2}
    assert res["macro_f1"] == pytest.approx(2 * 1 / (2 * 1 + 0 + 1)) and res["per_class_recall"] == {"BEV": None, "ICE": 0.5}
    assert res["ci95_macro_f1"] is None and res["cluster_col"] == "model"              # one car model: no interval
    with_drives = EV.evaluate_units(w, probs, "bev_ice", "drive_full", drives=unit_drives())
    assert with_drives["n_units_without_windows"] == 2                                  # drive E has no listed window at all
    human = EV.evaluate_units(w, probs, "bev_ice", "drive_human_only", drives=unit_drives())
    assert human["n_units"] == 1 and human["n_units_without_windows"] == 3 and human["n_windows_used"] == 2 and human["accuracy"] == 1.0
    veh = EV.evaluate_units(w, probs, "bev_ice", "vehicle_full", drives=unit_drives())
    assert veh["n_units"] == 1 and veh["n_drives_used"] == 2 and veh["n_windows_used"] == 4 and veh["n_units_without_windows"] == 2
    assert veh["accuracy"] == 1.0                                                          # the pooled mean leans ICE
    multi = EV.evaluate_units(w, probs, "bev_ice", "vehicle_full_multi", drives=unit_drives())
    assert multi["n_units"] == 1 and multi["n_units_without_windows"] == 0 and multi["n_drives_used"] == 2
    flipped = probs.copy(); flipped.loc["drive_A_w0"] = [0.9, 0.1]                     # vehicle-level tie: 0.5 / 0.5 over 4 windows
    tie = EV.evaluate_units(w, flipped, "bev_ice", "vehicle_full")
    assert abs(tie["accuracy"]) == 0.0 and tie["confusion"] == [[0, 0], [1, 0]]
    with pytest.raises(ValueError):
        EV.evaluate_units(w, probs.iloc[:3], "bev_ice", "drive_full")                    # a selected window without probabilities
    with pytest.raises(ValueError):
        EV.evaluate_units(w, probs, "bev_ice", "nope")
    assert list(EV.UNIT_PROTOCOLS) == ["drive_full", "drive_human_only", "vehicle_full", "vehicle_full_multi"]


def test_unit_protocols_on_synthetic_sample(sample, tmp_path):
    root, _ = sample
    windows = L.load_index("windows", root=root); drives = L.load_index("drives", root=root)
    sub = windows[EV.protocol_mask(windows, "bev_ice", "test_full")]
    assert sub.vehicle_id.nunique() == 1 and sub.drive_id.nunique() == 2 and (sub.powertrain == "ICE").all()
    tie = pd.DataFrame({"BEV": 0.5, "ICE": 0.5}, index=sub.window_id)
    res = EV.evaluate_units(windows, tie, "bev_ice", "drive_full", drives=drives, out_dir=tmp_path, name="tie")
    assert res["n_units"] == 2 and res["n_units_without_windows"] == 0 and res["accuracy"] == 0.0 and res["support"] == {"BEV": 0, "ICE": 2}
    assert Path(res["units_csv"]).exists() and Path(res["confusion_csv"]).exists()
    units = pd.read_csv(res["units_csv"])
    assert list(units.columns[:5]) == ["unit", "truth", "pred", "n_windows", "n_drives"] and set(units.pred) == {"BEV"}
    ice = pd.DataFrame({"BEV": 0.49, "ICE": 0.51}, index=sub.window_id)
    for prot, n_units in (("drive_full", 2), ("vehicle_full", 1), ("vehicle_full_multi", 1)):
        r = EV.evaluate_units(windows, ice, "bev_ice", prot, drives=drives)
        assert r["n_units"] == n_units and r["macro_f1"] == 1.0 and r["n_windows_used"] == len(sub) and r["macro_f1_supported"] == 1.0
    hum = EV.evaluate_units(windows, ice, "bev_ice", "drive_human_only", drives=drives)
    hsub = windows[EV.protocol_mask(windows, "bev_ice", "test_human_only")]
    assert hum["n_units"] == hsub.drive_id.nunique() and hum["n_windows_used"] == len(hsub) and 0 < hum["n_units"] <= 2


def test_one_hot_fallback_and_probs_matrix():
    w = unit_frame()
    sub = w[EV.protocol_mask(w, "bev_ice", "test_full")]
    labels = pd.Series(sub.powertrain.to_numpy(), index=sub.window_id)
    M = EV.probs_matrix(sub, labels, ("BEV", "ICE"))
    assert M.shape == (4, 2) and M.dtype == np.float64 and (M.sum(1) == 1).all() and (M[:, 1] == 1).all()
    assert (EV.probs_matrix(sub, dict(zip(sub.window_id, sub.powertrain)), ("BEV", "ICE")) == M).all()      # dict of labels
    assert (EV.probs_matrix(sub, list(sub.powertrain), ("BEV", "ICE")) == M).all()                            # aligned label list
    assert (EV.probs_matrix(sub, pd.DataFrame({"p_BEV": M[:, 0], "p_ICE": M[:, 1]}, index=sub.window_id), ("BEV", "ICE")) == M).all()
    assert (EV.probs_matrix(sub, pd.DataFrame(M, index=sub.window_id), ("BEV", "ICE")) == M).all()             # columns 0..k-1
    by_label = EV.evaluate_units(w, labels, "bev_ice", "drive_full")
    by_table = EV.evaluate_units(w, pd.DataFrame(M, index=sub.window_id, columns=["BEV", "ICE"]), "bev_ice", "drive_full")
    by_array = EV.evaluate_units(w, M, "bev_ice", "drive_full")
    assert by_label["macro_f1"] == by_table["macro_f1"] == by_array["macro_f1"] == 1.0
    assert by_label["confusion"] == by_table["confusion"] == by_array["confusion"] == [[0, 0], [0, 2]]
    with pytest.raises(ValueError):
        EV.one_hot(["BEV", "nope"], ("BEV", "ICE"))
    with pytest.raises(ValueError):
        EV.probs_matrix(sub, pd.DataFrame({"x": 1.0}, index=sub.window_id), ("BEV", "ICE"))
    with pytest.raises(ValueError):
        EV.probs_matrix(sub, M[:3], ("BEV", "ICE"))
    with pytest.raises(ValueError):
        EV.probs_matrix(sub, pd.DataFrame(M, index=["a", "a", "b", "c"], columns=["BEV", "ICE"]), ("BEV", "ICE"))   # duplicate ids


def test_aggregate_windows_variants():
    sub = pd.DataFrame({"window_id": [f"d1_w{i}" for i in range(3)] + ["d2_w0"], "drive_id": ["d1"] * 3 + ["d2"], "vehicle_id": ["v1"] * 4,
                        "powertrain": ["BEV"] * 3 + ["ICE"]})
    P = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.3, 0.7]])
    mean = EV.aggregate_windows(sub, P, "drive_id", ("BEV", "ICE"))
    assert list(mean.columns) == ["unit", "truth", "pred", "n_windows", "n_drives", "p_BEV", "p_ICE"]
    assert mean.unit.tolist() == ["d1", "d2"] and mean.truth.tolist() == ["BEV", "ICE"] and mean.pred.tolist() == ["BEV", "ICE"]
    assert mean.n_windows.tolist() == [3, 1] and mean.n_drives.tolist() == [1, 1] and mean.p_BEV[0] == pytest.approx(1.7 / 3)
    vote = EV.aggregate_windows(sub, P, "drive_id", ("BEV", "ICE"), how="vote")
    assert vote.p_BEV[0] == pytest.approx(2 / 3) and vote.pred[0] == "BEV"
    med = EV.aggregate_windows(sub, P, "drive_id", ("BEV", "ICE"), how="median")
    assert med.p_BEV[0] == pytest.approx(0.6)
    lo = EV.aggregate_windows(sub, P, "drive_id", ("BEV", "ICE"), how="logodds")
    logit = lambda p: np.log(p / (1 - p))                                                   # noqa: E731
    assert lo.p_BEV[0] == pytest.approx(1 / (1 + np.exp(-(logit(0.9) + logit(0.6) + logit(0.2)) / 3)))
    conf = EV.aggregate_windows(sub, P, "drive_id", ("BEV", "ICE"), how="confidence")       # weights 0.4, 0.1, 0.3
    assert conf.p_BEV[0] == pytest.approx((0.9 * 0.4 + 0.6 * 0.1 + 0.2 * 0.3) / 0.8)
    trim3 = EV.aggregate_windows(sub, P, "drive_id", ("BEV", "ICE"), how="trimmed")       # 3 windows: nothing trimmed
    assert trim3.p_BEV[0] == pytest.approx(mean.p_BEV[0])
    sub5 = pd.DataFrame({"window_id": [f"d_w{i}" for i in range(5)], "drive_id": "d", "powertrain": "BEV"})
    P5 = np.array([[0.0, 1.0], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [1.0, 0.0]])
    trim5 = EV.aggregate_windows(sub5, P5, "drive_id", ("BEV", "ICE"), how="trimmed")       # drops the 0.0 and the 1.0
    assert trim5.p_BEV[0] == pytest.approx(0.5) and trim5.pred[0] == "BEV"                  # exact tie -> earlier class
    veh = EV.aggregate_windows(sub, P, "vehicle_id", ("BEV", "ICE"))
    assert veh.n_drives.tolist() == [2] and veh.n_windows.tolist() == [4] and veh.truth.tolist() == ["BEV"]
    with pytest.raises(ValueError):
        EV.aggregate_windows(sub, P, "drive_id", ("BEV", "ICE"), how="nope")
    assert EV.AGGREGATIONS == ("mean", "logodds", "median", "confidence", "trimmed", "vote")


def test_unit_metrics_and_cluster_bootstrap():
    y = ["A", "A", "B", "B", "C", "C"]
    p = ["A", "B", "B", "B", "C", "A"]
    m = EV.unit_metrics(y, p, ["A", "B", "C"])
    assert m["n"] == 6 and m["confusion"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]] and m["classes"] == ["A", "B", "C"]
    assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3) and m["balanced_acc"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert m["accuracy"] == pytest.approx(4 / 6) and m["support"] == {"A": 2, "B": 2, "C": 2} and m["per_class_recall"]["B"] == 1.0
    assert EV.unit_metrics(["A", "A"], ["A", "B"], ["A", "B", "C"])["macro_f1"] == pytest.approx(2 / 3)   # only supported classes count
    assert EV.macro_f1_from_confusion(np.array(m["confusion"])) == pytest.approx(m["macro_f1"])
    clusters = ["m1", "m1", "m2", "m2", "m3", "m3"]
    assert EV.cluster_bootstrap_ci(y, y, clusters, ["A", "B", "C"], n_boot=50) == [1.0, 1.0]
    assert EV.cluster_bootstrap_ci(y, p, ["m1"] * 6, ["A", "B", "C"]) is None                 # one cluster: no interval
    ci = EV.cluster_bootstrap_ci(y, p, clusters, ["A", "B", "C"], n_boot=200, seed=0)
    assert 0.0 <= ci[0] <= ci[1] <= 1.0 and ci == EV.cluster_bootstrap_ci(y, p, clusters, ["A", "B", "C"], n_boot=200, seed=0)


def test_evaluate_units_reproduces_saved_drive_level():
    import json
    results = C.DATA_ROOT / "experiments" / "results"
    windows_path = C.DATA_ROOT / "build" / "index" / "windows.parquet"
    saved = sorted(results.glob("*_probs.npz")) if results.exists() else []
    if not windows_path.exists() or not saved:
        pytest.skip("saved runs or the windows index are not available")
    preferred = results / "split_platform__size__gru__s0_probs.npz"
    path = preferred if preferred.exists() else saved[0]
    tag = path.name[: -len("_probs.npz")]
    r = json.loads((results / f"{tag}.json").read_text())
    z = np.load(path, allow_pickle=True)
    classes = list(EV.TASKS[r["task"]]["classes"])
    probs = pd.DataFrame(z["probs"], index=z["window_id"].astype(str), columns=classes)
    windows = pd.read_parquet(windows_path)
    for protocol, key in (("drive_full", "test_full"), ("drive_human_only", "test_human_only")):
        res = EV.evaluate_units(windows, probs, r["task"], protocol, split_col=r["split_col"], n_boot=100)
        ref = r["drive_level"][key]
        assert abs(res["macro_f1"] - ref["macro_f1"]) < 1e-9 and res["n_units"] == ref["n_drives"] and res["confusion"] == ref["confusion"]
        assert abs(res["balanced_acc"] - ref["balanced_acc"]) < 1e-9 and res["n"] == ref["n"]
        assert res["ci95_macro_f1"] is not None and 0.0 <= res["ci95_macro_f1"][0] <= res["ci95_macro_f1"][1] <= 1.0
        assert res["n_models"] > 1 and res["n_windows_used"] > res["n_units"] and res["n_units_supported"] <= res["n_units"]


def test_regen_probs_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from experiments import regen_probs as RP
    from experiments.models import build_model
    torch.manual_seed(0)
    channels = list(C.OFFICIAL_INPUTS)
    m = build_model("transformer", len(channels), 3, d=32, n_layers=1, dropout=0.0).eval()
    ck = dict(state={k: v.clone() for k, v in m.state_dict().items()}, mean=np.zeros(len(channels), np.float32), std=np.ones(len(channels), np.float32),
              cfg=dict(d=32, n_layers=1, dropout=0.0), task="size", split_col="split_platform", model="transformer", channels=channels,
              classes=("SMALL", "MID", "LARGE"))
    path = tmp_path / "split_platform__size__transformer__s9.pt"
    torch.save(ck, path)
    torch.save(ck, tmp_path / "split_device__size__gru__s0.pt")
    assert RP.list_checkpoints(tmp_path, "split_platform__size__*") == [path] and len(RP.list_checkpoints(tmp_path)) == 2
    model, prep, ck2 = RP.load_checkpoint(path, torch.device("cpu"))
    assert ck2["task"] == "size" and not model.training
    x = torch.randn(2, 600, len(channels))
    with torch.no_grad():
        assert torch.allclose(model(prep(x)), m(prep(x)))
    lo, hi = RP.clip_bounds(channels)
    assert lo[channels.index("v_mps")] == -1.0 and hi[channels.index("v_mps")] == 60.0 and np.isinf(lo[channels.index("gas_pressed")])
    assert RP.SUFFIX == "_valprobs.npz"


def test_oracle_helpers():
    from experiments import oracles as O
    lo, hi = O.wilson_ci(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77 and np.isnan(O.wilson_ci(0, 0)[0]) and O.wilson_ci(10, 10)[1] == 1.0
    assert O.bin_label((1, 2)) == "1-2" and O.bin_label((65, None)) == ">64" and O.bin_label((0, 0)) == "0"
    assert O.in_bin(np.array([0, 1, 2, 3]), (1, 2)).tolist() == [False, True, True, False] and O.in_bin(np.array([64, 65]), (65, None)).tolist() == [False, True]
    fp = np.array([4.0, 4.5, 5.0, 4.43, 4.88, 4.9])
    assert O.size_class_of(fp, (4.43, 4.88)).tolist() == [0, 1, 2, 1, 1, 2]
    truth = O.size_class_of(fp, (4.43, 4.88))
    df = O.footprint_noise_oracle(fp, np.arange(len(fp)), truth, (4.43, 4.88), sigmas=(0.0, 0.3), taus=(0.0, 0.3), n_rep=5, seed=0)
    assert len(df) == 4 and df.loc[(df.sigma == 0) & (df.tau == 0), "macro_f1_mean"].item() == 1.0 and (df.macro_f1_mean <= 1.0).all()
    mass = np.array([1000.0, 1100.0, 1500.0, 1600.0, 2000.0, 2100.0])
    y = np.array([0, 0, 1, 1, 2, 2])
    c1, c2, f1 = O.fit_mass_cuts(mass, y)
    assert 1100 < c1 < 1500 and 1600 < c2 < 2000 and f1 == 1.0 and O.apply_mass_cuts(mass, c1, c2).tolist() == y.tolist()
    assert np.isnan(O.fit_mass_cuts(np.array([1.0, 1.0]), np.array([0, 1]))[0])
    units = pd.DataFrame({"unit": ["a", "b", "c", "d", "e"], "truth": ["X", "X", "Y", "Y", "Y"], "pred": ["X", "Y", "Y", "X", "Y"]})
    car = pd.Series({"a": "m1", "b": "m1", "c": "m2", "d": "m2", "e": "m2"})
    assert O.majority_prediction(units, car, ["X", "Y"]).tolist() == ["X", "X", "Y", "Y", "Y"]      # m1 tie -> earlier class X
    vote = O.vote_over_members([units, units.assign(pred=["Y", "Y", "Y", "X", "X"]), units.assign(pred=["X", "X", "X", "X", "X"])], ["X", "Y"])
    assert vote.pred.tolist() == ["X", "Y", "Y", "X", "X"]
    sep = O.separability_table(pd.DataFrame([dict(car_model="H1", size_class="MID", footprint=4.5, n_drives=3, accuracy=0.5, majority_correct=True)]),
                               pd.DataFrame([dict(car_model="T1", size_class="MID", footprint=4.52), dict(car_model="T2", size_class="SMALL", footprint=4.4),
                                             dict(car_model="T3", size_class="LARGE", footprint=4.95)]))
    assert sep.nearest_other_class_model.tolist() == ["T2"] and sep.footprint_gap.iloc[0] == pytest.approx(0.1)
    sub = pd.DataFrame({"window_id": ["w0", "w1", "w2"], "drive_id": ["d", "d", "e"], "moving_frac": [0.6, 0.2, 0.9], "n_accel_ep": [1, 0, 0],
                        "n_decel_ep": [0, 0, 0], "human_only_strict": [True, False, False]})
    assert O.window_mask(sub, "moving").tolist() == [True, False, True] and O.window_mask(sub, "episodes").tolist() == [True, False, False]
    assert O.window_mask(sub, "human").tolist() == [True, False, False] and O.window_mask(sub, "all").all()
    assert O.window_f1(np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([0, 1]), 2) == 1.0
