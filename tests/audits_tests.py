"""Shortcut audits on synthetic frames: Cramer's V, group-disjoint folds, majority-vote and nearest-centroid
predictability against chance, the two audit tables and the S12 script end to end."""
from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vtt.audits import common as CM
from vtt.audits import inputs_brand as IB
from vtt.audits import nuisance as NU
from vtt.paths import P
from vtt.synth_sample import write_sample


def load_s12():
    path = Path(__file__).resolve().parents[1] / "scripts" / "s12_audits.py"
    spec = importlib.util.spec_from_file_location("s12_audits", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def root():
    P.TMP.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="vtt_audit_", dir=P.TMP))
    yield d
    if not os.environ.get("VTT_KEEP_TMP"):
        shutil.rmtree(d, ignore_errors=True)


def drives_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Powertrain is fully determined by device_type; region is independent noise; splits are null (pre-S8)."""
    rng = np.random.default_rng(seed)
    pt = rng.choice(["BEV", "ICE"], n)
    return pd.DataFrame(dict(
        drive_id=[f"drive_{i + 1:06d}" for i in range(n)], group_id=[f"group_{i % 24 + 1:04d}" for i in range(n)],
        device_type=np.where(pt == "BEV", "tici", "mici"), region=rng.choice(["europe", "asia", "north_america"], n),
        software_era=rng.choice(["0.9", "0.10"], n), pose_source=rng.choice(["livePose", "liveLocationKalman"], n),
        radar_unavailable=rng.random(n) < 0.5, op_long=rng.random(n) < 0.5, gas_alive=rng.random(n) < 0.5, brake_alive=True,
        torque_alive=True, has_pose=True, n_seg_rlog_dec=rng.integers(0, 3, n), powertrain=pt,
        size_class=rng.choice(["SMALL", "MID", "LARGE"], n), split_platform=None, split_device=None))


def test_cramers_v_and_folds():
    df = drives_frame()
    assert abs(NU.cramers_v(df.device_type, df.powertrain) - 1.0) < 1e-9
    assert NU.cramers_v(df.region, df.powertrain) < 0.35
    assert np.isnan(NU.cramers_v(["a"] * 10, list("ababababab")))
    folds = CM.group_folds(df.group_id, 5, seed=1)
    assert len(np.unique(folds)) == 5
    assert (pd.Series(folds).groupby(df.group_id.to_numpy()).nunique() == 1).all()         # a group never spans folds
    assert len(np.unique(CM.group_folds(["g1", "g1", "g2"], 5))) == 2                           # fewer groups than folds


def test_macro_f1_and_chance():
    assert CM.macro_f1(["a", "b", "b"], ["a", "b", "a"]) == pytest.approx(np.mean([2 / 3, 2 / 3]))
    assert np.isnan(CM.macro_f1([], []))
    ch = CM.chance_levels(["a"] * 50 + ["b"] * 50, n_perm=10, seed=0)
    assert 0.4 < ch["chance_permuted_f1"] < 0.6 and ch["chance_majority_f1"] == pytest.approx(1 / 3)


def test_majority_vote_and_centroid_cv():
    df = drives_frame()
    perfect = CM.majority_vote_cv(df.device_type, df.powertrain, df.group_id)
    assert perfect["macro_f1"] == 1.0 and perfect["n"] == 120 and perfect["n_groups"] == 24 and perfect["n_keys"] == 2
    noise = CM.majority_vote_cv(df.region, df.powertrain, df.group_id)
    assert noise["macro_f1"] < 0.75
    X = np.column_stack([np.where(df.powertrain == "BEV", 3.0, 2.0) + np.random.default_rng(1).normal(0, 0.2, len(df)),
                         np.random.default_rng(2).normal(0, 1, len(df))])
    X[5, 1] = np.nan
    nc = CM.nearest_centroid_cv(X, df.powertrain, df.group_id)
    assert nc["macro_f1"] > 0.9 and nc["n"] == 120
    keys = CM.joint_key(df, ["device_type", "software_era"])
    assert len(set(keys)) <= 4 and all("|" in k for k in keys)
    bins = CM.quantile_bins(np.arange(100.0), np.array([0.0, 50.0, 99.0, np.nan]), 4)
    assert list(bins) == ["0", "2", "3", "NA"]


def test_nuisance_audit_table():
    df = drives_frame()
    tab = NU.audit(df, n_folds=5, seed=0)
    assert set(tab.target) == {"powertrain", "size_class"}                                      # null splits are skipped
    assert set(tab.columns) == set(NU.COLUMNS)
    dev = tab[(tab.target == "powertrain") & (tab.predictor == "device_type")].iloc[0]
    assert dev.macro_f1 == 1.0 and dev.cramers_v == pytest.approx(1.0) and dev.lift_vs_permuted > 0.3
    allrow = tab[(tab.target == "powertrain") & (tab.predictor == NU.ALL_KEY)].iloc[0]
    assert allrow.n == 120 and allrow.n_levels >= 2
    assert "rlog_dec_any" in set(tab.predictor)
    lines = NU.summary_lines(tab)
    assert len(lines) == 2 and "`device_type`" in lines[0]
    assert NU.audit(df.head(10)).empty                                                             # below min_rows


def windows_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    brand = rng.choice(["BrandA", "BrandB"], n)
    df = pd.DataFrame(dict(window_id=[f"drive_{i // 4 + 1:06d}_w{i % 4}" for i in range(n)],
                           group_id=[f"group_{i % 20 + 1:04d}" for i in range(n)], eligible=rng.random(n) < 0.95,
                           brand=brand, platform=brand + "_P", powertrain=np.where(brand == "BrandA", "BEV", "ICE"),
                           size_class=rng.choice(["MID", "LARGE"], n)))
    for c in IB.STAT_COLS:
        df[c] = rng.normal(0, 1, n)
    df["a_max"] = np.where(brand == "BrandA", 3.0, 2.0) + rng.normal(0, 0.15, n)                 # the one informative statistic
    df.loc[3, "v_mean"] = np.nan
    return df


def test_inputs_brand_audit_table():
    w = windows_frame()
    tab = IB.audit(w, n_folds=5, seed=0, max_rows=300)
    assert set(tab.columns) == set(IB.COLUMNS) and set(tab.target) == {"brand", "platform", "powertrain", "size_class"}
    best = tab[(tab.target == "brand") & (tab.method == "binned_majority:a_max")].iloc[0]
    assert best.macro_f1 > 0.9 and best.n <= 300 and best.n_classes == 2
    nc = tab[(tab.target == "brand") & (tab.method == "nearest_centroid:all")].iloc[0]
    assert nc.macro_f1 > nc.chance_permuted_f1
    size = tab[(tab.target == "size_class") & (tab.method == "binned_majority:a_max")].iloc[0]
    assert size.macro_f1 < 0.75                                                                   # size is unrelated to a_max here
    lines = IB.summary_lines(tab)
    assert len(lines) == 4 and "`a_max`" in lines[0]
    assert len(IB.select_windows(w, max_rows=50)) == 50 and IB.select_windows(w).eligible.all()


def test_s12_script_end_to_end(root, capsys):
    S12 = load_s12()
    assert S12.main(["--data-root", str(root)]) == 2                                               # inputs missing
    write_sample(root / "build", n_vehicles=8, n_drives=3)
    assert S12.main(["--data-root", str(root), "--dry-run"]) == 0
    assert not (root / "build" / "qc" / "AUDITS.md").exists()
    assert S12.main(["--data-root", str(root), "--n-folds", "4"]) == 0
    qc = root / "build" / "qc"
    for name in ("audit_nuisance.csv", "audit_inputs_brand.csv", "audit_numbers.json", "AUDITS.md"):
        assert (qc / name).exists(), name
    nuis = pd.read_csv(qc / "audit_nuisance.csv")
    assert {"powertrain", "size_class", "split_platform", "split_device"} <= set(nuis.target)
    md = (qc / "AUDITS.md").read_text()
    assert "## Nuisance metadata" in md and "`brand` from" in md
    out = capsys.readouterr().out
    assert "drives=24" in out and "nuisance rows=" in out
    assert S12.main(["--data-root", str(root)]) == 0                                               # idempotent re-run


def _loop_macro_f1(y_true, y_pred) -> float:
    """The original per-class loop (reference for the integer-coded implementation)."""
    y_true = np.asarray(list(y_true), dtype=object); y_pred = np.asarray(list(y_pred), dtype=object)
    f1s = []
    for lab in sorted(set(y_true.tolist())):
        tp = int(np.sum((y_true == lab) & (y_pred == lab))); fn = int(np.sum((y_true == lab) & (y_pred != lab)))
        fp = int(np.sum((y_true != lab) & (y_pred == lab)))
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s)) if f1s else np.nan


def test_macro_f1_vectorised_matches_loop_and_chance_is_shared():
    rng = np.random.default_rng(3)
    yt = rng.choice([f"c{i}" for i in range(12)], 500); yp = rng.choice([f"c{i}" for i in range(15)], 500)   # predicted labels absent from truth
    assert CM.macro_f1(yt, yp) == pytest.approx(_loop_macro_f1(yt, yp), abs=1e-12)
    assert CM.macro_f1(["a", "b", "b"], ["a", "b", "a"]) == pytest.approx(_loop_macro_f1(["a", "b", "b"], ["a", "b", "a"]))
    assert CM.macro_f1([None, np.nan, "a"], ["NA", "NA", "a"]) == 1.0 and list(CM.to_str([None, np.nan, True, 1.5])) == ["NA", "NA", "True", "1.5"]
    with pytest.raises(ValueError):
        CM.macro_f1(["a"], ["a", "b"])
    big_t = rng.choice(np.arange(100).astype(str), 200_000); big_p = rng.choice(np.arange(100).astype(str), 200_000)
    t0 = time.perf_counter(); v = CM.macro_f1(big_t, big_p); ch = CM.chance_levels(big_t, n_perm=20); dt = time.perf_counter() - t0
    assert 0.0 <= v <= 0.03 and 0.0 <= ch["chance_permuted_f1"] <= 0.03 and dt < 1.5          # was 2.2 s per macro_f1 call before
    y = CM.to_str(["a"] * 50 + ["b"] * 30 + ["c"] * 20)                                        # permuting codes == permuting labels
    perm = np.random.default_rng(0)
    assert CM.chance_levels(y, n_perm=5, seed=0)["chance_permuted_f1"] == pytest.approx(np.mean([_loop_macro_f1(y, perm.permutation(y)) for _ in range(5)]))
    nuis = NU.audit(drives_frame(), n_folds=5, seed=0)                                          # one chance level per target
    for _, g in nuis.groupby("target"):
        assert g.chance_permuted_f1.nunique() == 1 and g.chance_majority_f1.nunique() == 1
    brand = IB.audit(windows_frame(), n_folds=5, seed=0, max_rows=300)
    for _, g in brand.groupby("target"):
        assert g.chance_permuted_f1.nunique() == 1 and g.chance_majority_f1.nunique() == 1
