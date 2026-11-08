"""experiments/aggregate.py and experiments/run_recipe.py on toy data: aggregation rules, weights, temperature, gate,
stacking and member selection, run selection, the unit-level evaluation table (protocol reference and the two-stage
vehicle candidate) and the recipe grid (dry run / render / report path / execute with fakes).

Only the comparisons with experiments.train (drive_level, metrics) and the execute() test need torch (skipped without
it); the rules themselves are numpy / pandas, the gate and the stacker need scikit-learn.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import aggregate as A
from experiments import run_recipe as RR

CLASSES = ("SMALL", "MID", "LARGE")


def _toy_run(n_drives: int = 12, per_drive: int = 7, k: int = 3, seed: int = 0, n_models: int = 4):
    """Window probabilities of n_drives drives (per_drive windows each) with a constant label per drive, and the matching
    windows index (drive / vehicle / car model keys plus the statistics the weights and the gate use)."""
    rng = np.random.default_rng(seed)
    rows = []
    P, y = [], []
    for d in range(n_drives):
        vehicle = f"v{d // 2:03d}"           # two drives per vehicle, one label per vehicle (as in the real index)
        lab = (d // 2) % k
        drive = f"d{d:03d}"
        car = f"car{d % n_models}"
        for w in range(per_drive):
            z = rng.normal(size=k)
            z[lab] += 1.5
            P.append(A.softmax_rows(z[None])[0])
            y.append(lab)
            rows.append(dict(window_id=f"{drive}_w{w}", drive_id=drive, vehicle_id=vehicle, car_model=car, human_only_strict=bool(w % 2 == 0),
                             das_any_frac=float(rng.uniform(0, 1)), n_accel_ep=int(rng.integers(0, 4)), n_decel_ep=int(rng.integers(0, 4)),
                             n_lead_accel_events=int(rng.integers(0, 3)), steer_pressed_frac=float(rng.uniform(0, 1)),
                             gas_pressed_frac=float(rng.uniform(0, 1)), brake_pressed_frac=float(rng.uniform(0, 0.3)), moving_frac=0.9,
                             v_mean=float(rng.uniform(5, 30)), v_std=1.0, a_std=0.5, lead_frac=0.5, cf_frac=0.2))
    win = pd.DataFrame(rows)
    P, y = np.array(P), np.array(y, dtype=np.int64)
    run = dict(tag="toy__size__gru__s0", task="size", split_col="split_platform", model="gru", seed=0, kind="test_posthoc", fold=None,
               recipe_id="", window_id=win["window_id"].to_numpy(str), probs=P, y=y)
    return run, win


def _row(df: pd.DataFrame, level: str, method: str, weights: str = A.UNIFORM) -> pd.Series:
    sub = df[(df["level"] == level) & (df["method"] == method) & (df["weights"] == weights)]
    assert len(sub) == 1, (level, method, weights, len(sub))
    return sub.iloc[0]


# ----------------------------------------------------------------------------- aggregation rules
def test_aggregate_mean_matches_drive_level():
    torch = pytest.importorskip("torch")   # noqa: F841  (experiments.train imports torch)
    from experiments.train import drive_level
    run, win = _toy_run()
    idx = np.arange(len(win))
    ref = drive_level(win, idx, run["probs"], run["y"], CLASSES)
    units, agg = A.aggregate(run["probs"], win["drive_id"].to_numpy(object), "mean")
    u2, yu = A.unit_labels(win["drive_id"].to_numpy(object), run["y"])
    assert np.array_equal(units, u2)
    m = A.unit_metrics(units, yu, agg, CLASSES)
    assert m["macro_f1"] == pytest.approx(ref["macro_f1"], abs=1e-12)
    assert m["balanced_acc"] == pytest.approx(ref["balanced_acc"], abs=1e-12)
    assert m["accuracy"] == pytest.approx(ref["accuracy"], abs=1e-12)
    assert m["confusion"] == ref["confusion"] and m["n_units"] == ref["n_drives"] == 12


def test_metrics_match_experiments_train():
    pytest.importorskip("torch")
    from experiments.train import metrics as train_metrics
    y = np.array([0, 0, 1, 1, 2, 2, 1])
    p = np.array([0, 1, 1, 1, 2, 0, 0])
    assert A.metrics(y, p, CLASSES) == train_metrics(y, p, CLASSES)
    assert A.macro_f1(y, p, 3) == pytest.approx((0.4 + 2 / 3 + 2 / 3) / 3)      # class 0: tp 1, fp 2, fn 1
    assert A.metrics(np.array([0, 0]), np.array([0, 1]), CLASSES)["macro_f1"] == pytest.approx(2 / 3)   # only supported classes


def test_aggregate_rules_on_toy_probabilities():
    P = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8], [0.5, 0.3, 0.2]])
    keys = np.array(["b", "a", "a", "b"], dtype=object)
    units, mean = A.aggregate(P, keys, "mean")
    assert units.tolist() == ["a", "b"]
    assert np.allclose(mean, [[0.15, 0.35, 0.5], [0.6, 0.25, 0.15]])
    _, lo = A.aggregate(P, keys, "logodds")
    gm = np.sqrt(P[1] * P[2])
    assert np.allclose(lo[0], gm / gm.sum())                             # geometric mean, renormalised
    _, med = A.aggregate(P, keys, "median")
    assert np.allclose(med[0], np.median(P[[1, 2]], axis=0) / np.median(P[[1, 2]], axis=0).sum())
    _, vote = A.aggregate(P, keys, "vote")
    assert np.allclose(vote, [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0]])
    assert vote[0].argmax() == 1                                          # tie -> earlier class
    # weights: a zero-weight window is ignored; a unit with only zero weights falls back to uniform weights
    _, wm = A.aggregate(P, keys, "mean", w=np.array([1.0, 0.0, 1.0, 1.0]))
    assert np.allclose(wm[0], P[2])
    _, wz = A.aggregate(P, keys, "mean", w=np.array([1.0, 0.0, 0.0, 1.0]))
    assert np.allclose(wz[0], mean[0])
    # footprint mean cut at the size thresholds
    _, fp = A.aggregate(np.array([4.2, 4.6, 5.0, 4.4]), keys, "fp_mean", thresholds=(4.43, 4.88))
    assert fp.tolist() == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    with pytest.raises(ValueError):
        A.aggregate(P, keys, "max")
    with pytest.raises(ValueError):
        A.aggregate(P, keys, "mean", w=np.array([1.0, -1.0, 1.0, 1.0]))


def test_unit_labels_rejects_mixed_units():
    with pytest.raises(ValueError):
        A.unit_labels(np.array(["a", "a"], dtype=object), np.array([0, 1]))


def test_weight_grid_shapes_and_formula():
    meta = pd.DataFrame({"das_any_frac": [0.5, 0.0, np.nan], "n_accel_ep": [1, 0, 2], "n_decel_ep": [0, 2, 1], "steer_pressed_frac": [0.2, 0.0, 1.0]})
    grid = A.weight_grid(meta)
    assert len(grid) == 27 and all(v.shape == (3,) for v in grid.values())
    assert np.allclose(grid[A.UNIFORM], 1.0)
    assert np.allclose(grid["das1_ep1_steer1"], [(0.6) * 2 * 0.25, 1.1 * 3 * 0.05, 1.1 * 4 * 1.05])   # NaN das counts as 0
    assert np.allclose(grid["das0.5_ep0_steer0"], np.sqrt([0.6, 1.1, 1.1]))
    assert all(np.isfinite(v).all() and (v > 0).all() for v in grid.values())


def test_vehicle_level_weights_drives_by_sqrt_windows():
    drives = np.array(["d0", "d1"], dtype=object)
    agg = np.array([[0.8, 0.2], [0.3, 0.7]])
    units, veh, keep = A.vehicle_level(drives, agg, {"d0": "v0", "d1": "v0"}, {"d0": 4, "d1": 1})
    assert units.tolist() == ["v0"] and keep.all()
    w = np.array([2.0, 1.0])                                            # n_windows ** 0.5
    expect = A.softmax_rows(((np.log(agg) * w[:, None]).sum(0) / w.sum())[None])[0]
    assert np.allclose(veh[0], expect)
    units_m, _, keep_m = A.vehicle_level(drives, agg, {"d0": "v0", "d1": "v1"}, {"d0": 4, "d1": 1}, multi=True)
    assert len(units_m) == 0 and not keep_m.any()


# ----------------------------------------------------------------------------- temperature, gate, stacking
def test_fit_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(20000, 4))
    p_true = A.softmax_rows(z)
    y = np.array([rng.choice(4, p=p) for p in p_true])
    assert A.fit_temperature(A.softmax_rows(2.0 * z), y) == pytest.approx(2.0, abs=0.15)
    assert A.fit_temperature(A.softmax_rows(0.5 * z), y) == pytest.approx(0.5, abs=0.05)
    P = A.softmax_rows(z[:5])
    assert np.allclose(A.apply_temperature(P, 1.0), P, atol=1e-5)
    assert np.allclose(A.apply_temperature(P, 3.0).sum(1), 1.0)


def test_gate_learns_correctness_from_window_statistics():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(1)
    n = 3000
    das = rng.uniform(0, 1, n)
    correct = rng.uniform(0, 1, n) < (0.9 - 0.7 * das)                   # windows with automation are wrong more often
    P = np.where(correct[:, None], [[0.8, 0.1, 0.1]], [[0.1, 0.8, 0.1]])
    y = np.zeros(n, dtype=np.int64)
    meta = pd.DataFrame({"das_any_frac": das, "n_accel_ep": rng.integers(0, 3, n), "steer_pressed_frac": rng.uniform(0, 1, n)})
    st = A.window_stats(P, meta)
    assert {"das_any_frac", "n_accel_ep", "steer_pressed_frac", "max_prob", "entropy"} <= set(st.columns)
    gate = A.fit_gate(st.iloc[:2000].assign(correct=correct[:2000]))
    w = A.apply_gate(gate, st.iloc[2000:])
    assert w.shape == (1000,) and (w >= 0).all() and (w <= 1).all()
    assert np.corrcoef(w, das[2000:])[0, 1] < -0.8                        # the gate down-weights automation windows
    assert w[correct[2000:]].mean() > w[~correct[2000:]].mean()
    const = A.fit_gate(st.iloc[:10].assign(correct=True))
    assert np.allclose(A.apply_gate(const, st.iloc[:3]), 1.0)


def test_stack_improves_over_the_worst_member():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(2)
    n, k = 4000, 3
    y = rng.integers(0, k, n)
    ids = np.array([f"w{i}" for i in range(n)], dtype=object)
    onehot = np.eye(k)[y]
    good = A.softmax_rows(2.0 * onehot + rng.normal(size=(n, k)))
    bad = A.softmax_rows(0.3 * onehot + rng.normal(size=(n, k)))
    tr, te = np.arange(n) < n // 2, np.arange(n) >= n // 2
    members_tr = {"good": (ids[tr], good[tr]), "bad": (ids[tr], bad[tr])}
    stk = A.stack(members_tr, pd.Series(y[tr], index=ids[tr]), n_classes=k)
    assert stk["members"] == ["good", "bad"] and stk["n_fit"] == n // 2
    pids, ps = A.predict_stack(stk, {"good": (ids[te], good[te]), "bad": (ids[te], bad[te])})
    assert np.array_equal(pids, np.array(sorted(ids[te]), dtype=object)) and ps.shape == (te.sum(), k)
    assert np.allclose(ps.sum(1), 1.0)
    yte = pd.Series(y, index=ids).reindex(pids).to_numpy()
    acc = lambda P: float((P.argmax(1) == yte).mean())   # noqa: E731
    pos = pd.Series(np.arange(n), index=ids).reindex(pids).to_numpy()
    assert acc(ps) > acc(bad[pos]) + 0.1
    assert acc(ps) >= acc(good[pos]) - 0.02


def test_crossfit_leaves_the_group_out():
    groups = np.array(["a", "a", "b", "b", "c"], dtype=object)
    seen = []
    out = A.crossfit(groups, lambda tr: seen.append(tuple(np.flatnonzero(tr))) or tr.sum(), lambda m, te: np.full(te.sum(), float(m)))
    assert out.tolist() == [3.0, 3.0, 3.0, 3.0, 4.0]
    assert all(len(s) + (2 if i < 2 else 1) == 5 for i, s in enumerate(seen))
    with pytest.raises(ValueError):
        A.crossfit(np.array(["a", "a"], dtype=object), lambda tr: 0, lambda m, te: np.zeros(te.sum()))


def test_select_members_drops_members_without_shared_windows():
    ids = np.array([f"w{i}" for i in range(100)], dtype=object)
    probe = np.array([f"p{i}" for i in range(20)], dtype=object)
    # two pooled runs (same windows), one probe (disjoint) and one single-fold run (a third of the pooled windows)
    kept, dropped, common = A.select_members({"pooled_a": ids, "pooled_b": ids, "probe": probe, "fold0": ids[:30]})
    assert kept == ["pooled_a", "pooled_b"] and dropped == ["probe", "fold0"] and len(common) == 100
    kept, dropped, common = A.select_members({"a": ids, "b": ids[::-1]})
    assert kept == ["a", "b"] and dropped == [] and common.tolist() == sorted(ids.tolist())
    kept, dropped, common = A.select_members({"a": ids, "probe": probe})          # disjoint pair: the smaller one goes
    assert kept == ["a"] and dropped == ["probe"] and len(common) == 100
    kept, dropped, common = A.select_members({"a": ids, "half": ids[:50]}, min_cover=0.5)   # exactly at the bound: kept
    assert kept == ["a", "half"] and len(common) == 50


# ----------------------------------------------------------------------------- evaluation table and report
def test_evaluate_aggregations_table_and_summary():
    pytest.importorskip("sklearn")
    run, win = _toy_run()
    df = A.evaluate_aggregations([run], win, "size", "split_platform", ["drive", "vehicle", "vehicle_multi"], "full")
    assert set(df["level"]) == {"drive", "vehicle", "vehicle_multi"}
    assert {"macro_f1", "balanced_acc", "n_units", "method", "weights", "kind", "recipe_id"} <= set(df.columns)
    d = df[df["level"] == "drive"]
    assert set(d["method"]) == {"mean", "logodds", "median", "vote", "mean_T", "gate_mean", "gate_logodds"}
    assert len(d[d["method"] == "mean"]) == 27 and (d["n_units"] == 12).all() and (d["n_windows"] == 84).all()
    v = df[df["level"] == "vehicle"]
    assert (v["n_units"] == 6).all()
    assert set(v["method"]) == set(d["method"]) | {m + A.DRIVES_SUFFIX for m in set(d["method"])}   # pooled windows + two-stage
    ref = d[(d["method"] == "mean") & (d["weights"] == A.UNIFORM)].iloc[0]
    units, agg = A.aggregate(run["probs"], win["drive_id"].to_numpy(object), "mean")
    _, yu = A.unit_labels(win["drive_id"].to_numpy(object), run["y"])
    assert ref["accuracy"] == pytest.approx(float((agg.argmax(1) == yu).mean()))
    h = A.evaluate_aggregations([run], win, "size", "split_platform", "drive", "human_only")
    assert (h["n_windows"] == 48).all()
    s = A.summarise(d)
    assert list(s.columns) == ["method", "weights", "n_runs", "macro_f1_mean", "macro_f1_std", "delta_mean", "improved", "n_units"]
    r = s[(s["method"] == "mean") & (s["weights"] == A.UNIFORM)].iloc[0]
    assert r["delta_mean"] == pytest.approx(0.0) and r["improved"] == 0
    assert s["macro_f1_mean"].is_monotonic_decreasing
    with pytest.raises(ValueError):
        A.evaluate_aggregations([run], win, "size", "split_platform", "fleet")
    text = "\n".join(A.render_task("size", "split_platform", "test_posthoc", {("drive", "full"): d, ("vehicle", "full"): v}, None))
    assert "POST-HOC" in text and "| mean | uniform |" in text and "kinds: test_posthoc" in text
    assert "mean / uniform (protocol)" in text and f"mean{A.DRIVES_SUFFIX} / uniform" in text and "### Vehicle level, protocol full" in text


def test_vehicle_rows_protocol_reference_and_two_stage_candidate():
    pytest.importorskip("sklearn")
    run, win = _toy_run(n_drives=13)                        # vehicle v006 has a single drive (d012)
    df = A.evaluate_aggregations([run], win, "size", "split_platform", ["drive", "vehicle", "vehicle_multi"], "full")
    P, y = run["probs"], run["y"]
    drive_keys, veh_keys = win["drive_id"].to_numpy(object), win["vehicle_id"].to_numpy(object)
    # `mean / uniform` at the vehicle level is the protocol: equal-weight mean over the vehicle's pooled windows
    v_units, v_agg = A.aggregate(P, veh_keys, "mean")
    _, y_v = A.unit_labels(veh_keys, y)
    prot = A.unit_metrics(v_units, y_v, v_agg, CLASSES)
    row = _row(df, "vehicle", "mean")
    assert row["n_units"] == 7 and row["macro_f1"] == pytest.approx(prot["macro_f1"]) and row["accuracy"] == pytest.approx(prot["accuracy"])
    # `mean>drives / uniform` is the plan's two-stage candidate: drive means -> log-odds mean weighted by n_windows^0.5
    d_units, d_agg = A.aggregate(P, drive_keys, "mean")
    dv = dict(zip(win["drive_id"], win["vehicle_id"]))
    dn = win.groupby("drive_id").size().to_dict()
    u2, agg2, _ = A.vehicle_level(d_units, d_agg, dv, dn)
    two = A.unit_metrics(u2, y_v, agg2, CLASSES)
    row2 = _row(df, "vehicle", "mean" + A.DRIVES_SUFFIX)
    assert row2["n_units"] == 7 and row2["macro_f1"] == pytest.approx(two["macro_f1"]) and row2["accuracy"] == pytest.approx(two["accuracy"])
    # the two families differ in general (log-odds over drives is not the pooled-window mean)
    assert not np.allclose(agg2, v_agg)
    # vehicle_multi keeps the vehicles with >= 2 evaluated drives, in both families
    m = df[df["level"] == "vehicle_multi"]
    assert (m["n_units"] == 6).all() and len(m) == len(df[df["level"] == "vehicle"])
    sel = win["vehicle_id"].to_numpy() != "v006"
    mu, magg = A.aggregate(P[sel], veh_keys[sel], "mean")
    _, ym = A.unit_labels(veh_keys[sel], y[sel])
    assert _row(df, "vehicle_multi", "mean")["macro_f1"] == pytest.approx(A.unit_metrics(mu, ym, magg, CLASSES)["macro_f1"])
    # the reference of the vehicle summary is the protocol row; the two-stage row's delta is measured against it
    s = A.summarise(df[df["level"] == "vehicle"])
    ref = s[(s["method"] == "mean") & (s["weights"] == A.UNIFORM)].iloc[0]
    cand = s[(s["method"] == "mean" + A.DRIVES_SUFFIX) & (s["weights"] == A.UNIFORM)].iloc[0]
    assert ref["delta_mean"] == pytest.approx(0.0) and cand["delta_mean"] == pytest.approx(two["macro_f1"] - prot["macro_f1"])


def test_select_runs_defaults_and_filters():
    mk = lambda tag, kind, rid, fold, model="gru": dict(tag=tag, kind=kind, recipe_id=rid, fold=fold, model=model, task="size")   # noqa: E731
    pooled = [mk("pooled__size__gru__S5__s0", "oof_pooled", "S5", None), mk("pooled__size__resnet__S5__s0", "oof_pooled", "S5", None, "resnet")]
    sweep = [mk(f"fold0__size__gru__S{i}__s0", "oof", f"S{i}", 0) for i in range(3)]
    probe = [mk("fold2__size__gru__L3_S5__s0", "oof", "L3_S5", 2)]
    other = [mk("fold1__size__gru__S5__s0", "oof", "S5", 1)]
    post = [mk("split_platform__size__gru__s0", "test_posthoc", "", None)]
    kept, dropped = A.select_runs(pooled + sweep + probe + other + post)
    assert [r["tag"] for r in kept] == [r["tag"] for r in pooled] and dropped == {"kind": 6}
    kept, dropped = A.select_runs(sweep + probe + other)                       # no pooled runs: the most frequent fold
    assert [r["fold"] for r in kept] == [0, 0, 0] and dropped == {"probe": 1, "fold": 1}
    kept, _ = A.select_runs(sweep + probe + other, include_probes=True, kinds=("oof",))
    assert len(kept) == 5                                                       # explicit kinds: every fold, probes kept
    kept, dropped = A.select_runs(pooled + sweep, kinds=("oof_pooled",), models=("gru",))
    assert len(kept) == 1 and dropped == {"kind": 3, "model": 1}
    kept, dropped = A.select_runs(sweep, recipes=("S1", "S2"))
    assert [r["recipe_id"] for r in kept] == ["S1", "S2"] and dropped == {"recipe": 1}
    kept, _ = A.select_runs(post + pooled, kinds=("test_posthoc",))
    assert len(kept) == 1 and kept[0]["kind"] == "test_posthoc"
    assert A.is_probe("L1_S5") and A.is_probe("L12") and not A.is_probe("S5") and not A.is_probe("") and not A.is_probe(None)


def test_stack_table_and_pooling(tmp_path):
    pytest.importorskip("sklearn")
    run, win = _toy_run()
    run2 = dict(run, tag="toy__size__resnet__s0", model="resnet", probs=A.softmax_rows(np.log(run["probs"]) + 0.1))
    t = A.stack_table([run, run2], win.set_index("window_id"), "size", "split_platform")
    assert t["member"].tolist() == [run["tag"], run2["tag"], "ensemble_mean", "stack_crossfit"]
    assert (t["n_windows"] == 84).all() and t["window_macro_f1"].between(0, 1).all()
    # a probe run (disjoint windows) is dropped from the stacking members instead of emptying the common window set
    probe = dict(run, tag="fold0__size__gru__L1_S5__s0", kind="oof", fold=0, recipe_id="L1_S5",
                 window_id=np.array([f"p{i}" for i in range(len(run["y"]))], dtype=object))
    msgs = []
    t2 = A.stack_table([probe, run, run2], win.set_index("window_id"), "size", "split_platform", log=msgs.append)
    assert t2["member"].tolist() == t["member"].tolist() and (t2["n_windows"] == 84).all()
    assert any("1 members dropped" in m and probe["tag"] in m for m in msgs)
    assert A.stack_table([probe, run], win.set_index("window_id"), "size", "split_platform", log=msgs.append).empty
    folds = [dict(run, tag=f"fold{k}__size__gru__S0__s0", kind="oof", fold=k, recipe_id="S0") for k in range(2)]
    pooled = A.pool_folds(folds + [run])
    assert len(pooled) == 2 and {r["kind"] for r in pooled} == {"oof_pooled", "test_posthoc"}
    pr = next(r for r in pooled if r["kind"] == "oof_pooled")
    assert pr["tag"] == "pooled__size__gru__S0__s0" and len(pr["y"]) == 2 * len(run["y"]) and set(pr["groups"]) == {"fold0", "fold1"}
    # load_runs reads <tag>_oof.npz + <tag>.json (C1 layout: model = <arch>__<recipe>, arch = <arch>) and falls back to *_probs.npz
    oof = tmp_path / "oof"; oof.mkdir()
    fp_mu = np.full(len(run["y"]), np.log(4.6))             # the footprint head predicts ln(footprint m^2), already de-standardised
    fp_mu[0] = np.nan
    np.savez(oof / "fold0__size__gru__S2__s0_oof.npz", window_id=run["window_id"], probs=run["probs"], y=run["y"], fp_mu=fp_mu)
    (oof / "fold0__size__gru__S2__s0.json").write_text(json.dumps(dict(task="size", split_col="split_platform", model="gru__S2", arch="gru", seed=0, fold=0,
                                                                          recipe_id="S2", cfg={"lr": 1e-3})))
    loaded = A.load_runs(oof, tmp_path / "none")
    assert len(loaded) == 1 and loaded[0]["kind"] == "oof" and loaded[0]["fold"] == 0 and loaded[0]["model"] == "gru" and loaded[0]["recipe_id"] == "S2"
    fp = loaded[0]["fp_m2"]
    assert np.isnan(fp[0]) and np.allclose(fp[1:], 4.6)
    _, buckets = A.aggregate(np.nan_to_num(fp), win["drive_id"].to_numpy(object), "fp_mean", w=np.isfinite(fp).astype(float), thresholds=(4.43, 4.88))
    assert (buckets.argmax(1) == 1).all()                    # every drive lands in the MID bucket
    msgs = []
    df = A.evaluate_aggregations(loaded, win, "size", "split_platform", ["drive", "vehicle"], "full", log=msgs.append)
    fpm = df[df["method"] == "fp_mean"]
    assert set(fpm["level"]) == {"drive", "vehicle"} and np.allclose(fpm["accuracy"].to_numpy(), 1 / 3)   # MID is one of three labels
    assert not (df["method"] == "fp_mean" + A.DRIVES_SUFFIX).any() and any("1 windows without a finite footprint" in m for m in msgs)
    assert A.load_runs(tmp_path / "empty", oof) == []            # fallback looks for *_probs.npz only
    md = tmp_path / "AGG.md"
    md.write_text("# my header\n\n" + A.RENDER_MARK + "\n\nold tables\n")
    A.write_report(md, ["## new"])
    text = md.read_text()
    assert text.startswith("# my header") and "old tables" not in text and "## new" in text


# ----------------------------------------------------------------------------- recipe grid
def test_run_recipe_dry_run_lists_expected_ids(capsys):
    RR.main(["--stage", "sweep", "--task", "size", "--dry-run"])
    out = capsys.readouterr().out
    assert all(f"fold0__size__gru__S{i}__s0" in out for i in range(10)) and "10 configurations" in out
    RR.main(["--stage", "sweep", "--task", "bev_ice", "--models", "gru,resnet", "--dry-run"])
    out = capsys.readouterr().out
    assert all(f"fold0__bev_ice__{m}__B{i}__s0" in out for i in range(6) for m in ("gru", "resnet")) and "12 configurations" in out
    RR.main(["--stage", "probes", "--task", "size", "--probe-base", "S5", "--dry-run"])
    out = capsys.readouterr().out
    assert all(f"fold{k}__size__gru__L{k + 1}_S5__s0" in out for k in range(4))
    assert "Honda Odyssey" in out and "Tesla Model 3" in out
    RR.main(["--stage", "folds", "--task", "size", "--recipes", "S3", "--dry-run"])
    out = capsys.readouterr().out
    assert all(f"fold{k}__size__{m}__S3__s0" in out for k in range(3) for m in ("gru", "resnet")) and "6 configurations" in out


def test_recipe_configs_follow_contract_values():
    for task in RR.RECIPES:
        cfgs = RR.configs("sweep", task, 0, ("gru",))
        assert [c["id"] for c in cfgs] == [rid for rid, _, _ in RR.RECIPES[task]]
        for c in cfgs:
            RR.check_kwargs(c["kwargs"])
            assert c["kwargs"]["select_on"] == "inner_drive_f1" and c["folds"] is None
    size = {c["id"]: c["kwargs"] for c in RR.configs("sweep", "size", 0, ("gru",))}
    assert "aux" not in size["S0"] and size["S1"] == {"sampling": "class_model", "select_on": "inner_drive_f1"}
    assert size["S5"]["augment"] == "flip,mask,chdrop,noise" and size["S6"]["augment"].endswith(",cutmix")
    assert size["S7"]["das_beta"] == 0.5 and size["S8"]["adv_lambda"] == 0.1 and size["S9"]["dro"] is True
    assert all("coral" not in c["kwargs"].get("aux", "") for c in RR.configs("sweep", "bev_ice", 0, ("gru",)))
    with pytest.raises(ValueError):
        RR.check_kwargs(dict(aux="cls+body"))
    with pytest.raises(ValueError):
        RR.check_kwargs(dict(augment="flip,scale"))
    smoke = RR.configs("folds", "size", 1, ("gru",), ["S0"], epochs=2, limit_train=2000)
    assert len(smoke) == 1 and smoke[0]["kwargs"]["epochs"] == 2 and smoke[0]["kwargs"]["limit_train"] == 2000 and smoke[0]["fold"] == 1


def test_probe_folds_follow_fold_contract():
    drives = pd.DataFrame({"drive_id": [f"d{i}" for i in range(8)], "group_id": ["g0", "g0", "g1", "g2", "g3", "g4", "g5", "g5"],
                           "model": ["Honda Odyssey", "Other", "Other", "Other", "Tesla Model 3", "Other", "Other", "Honda Odyssey"],
                           "split_platform": ["train", "train", "validation", "train", "train", "test", "train", "test"]})
    pf = RR.probe_folds("size", drives)
    assert pf["split_col"] == "split_platform" and pf["n_folds"] == 4 and [f["fold"] for f in pf["folds"]] == [0, 1, 2, 3]
    f0 = pf["folds"][0]
    # held out: d0; d1 shares group g0 and d6 shares group g5 with the test-split Odyssey drive d7 -> training = d2, d3, d4
    assert f0["heldout_models"] == ["Honda Odyssey"] and f0["n_heldout_drives"] == 1 and f0["n_train_drives"] == 3
    f3 = pf["folds"][3]
    assert f3["heldout_models"] == ["Tesla Model 3"] and f3["n_heldout_drives"] == 1 and f3["n_train_drives"] == 5
    assert RR.probe_folds("size")["folds"][3]["n_heldout_drives"] == -1


def test_report_path_defaults(tmp_path):
    assert RR.report_path(RR.RESULTS_DIR) == RR.REPO / "experiments" / "RECIPE.md"
    assert RR.report_path(tmp_path / "smoke") == tmp_path / "smoke" / "RECIPE.md"          # a smoke folder keeps its own table
    assert RR.report_path(tmp_path / "smoke", tmp_path / "x.md") == tmp_path / "x.md"


def test_render_recipe_md_from_results(tmp_path):
    m = lambda f1, rec: dict(macro_f1=f1, balanced_acc=f1, accuracy=f1, per_class_recall={"SMALL": 0.8, "MID": 0.6, "LARGE": rec})   # noqa: E731
    rs = []
    for rid, f1 in (("S0", 0.61), ("S1", 0.66)):
        # the layout experiments.train writes for fold runs: model = <arch>__<recipe_id>, arch = <arch>
        r = dict(tag=f"fold0__size__gru__{rid}__s0", task="size", split_col="split_platform", model=f"gru__{rid}", arch="gru", seed=0, fold=0, recipe_id=rid,
                 inner_val_metrics={"window": m(f1 - 0.1, 0.3), "drive_full": m(f1, 0.4)}, decision="softmax", best_epoch=7, epochs_run=13, train_minutes=5.5)
        rs.append(r)
        (tmp_path / f"{r['tag']}.json").write_text(json.dumps(r))
    rs.append(dict(tag="fold0__size__gru__L1_S1__s0", task="size", split_col="split_platform", model="gru__L1_S1", arch="gru", seed=0, fold=0, recipe_id="L1_S1",
                   inner_val_metrics={"window": m(0.5, 0.2), "drive_full": m(0.55, 0.3)}, best_epoch=3, train_minutes=2.0))
    (tmp_path / "broken.json").write_text("{not json")
    loaded = RR.load_results(tmp_path)
    assert len(loaded) == 2 and RR.best_recipe(loaded, "size") == "S1" and RR.best_recipe([], "size") is None
    assert RR.best_recipe(loaded, "size", model="gru") == "S1" and RR.best_recipe(loaded, "size", model="resnet") is None
    text = RR.render(rs)
    assert text.startswith("# Recipe grid") and RR.RENDER_MARK in text
    assert "### gru, fold 0" in text and "gru__S1" not in text
    assert "| S1 | sampling by car model inside each class (sampling=class_model) | 66.0 | 56.0 | 40.0 | softmax | 7 | 6 | 1 |" in text
    assert "Best recipe so far by inner drive_full macro-F1: `S1`" in text
    assert "| L1_S1 | Honda Odyssey | gru | 55.0 | 50.0 | 3 | 2 |" in text
    assert "No fold runs yet" in RR.render([])


def test_execute_skips_finished_runs_and_passes_fold_kwargs(tmp_path, monkeypatch):
    pytest.importorskip("torch")                           # execute() imports experiments.data / experiments.train
    import experiments.data as D
    import experiments.train as T
    calls = []

    class FakeData:
        def __init__(self, in_ram: bool = True):
            self.win = pd.DataFrame({"window_id": ["w0", "w1"]})
            self.channels = ["v_mps"]

    def fake_run(task, split_col, model_name, seed=0, data=None, log=print, out_dir=None, *, fold=None, folds=None,
                 select_on="val_window_f1", eval_test=True, recipe_id="", ckpt_dir=None, **hp):
        calls.append(dict(task=task, split_col=split_col, model=model_name, seed=seed, data=data, out_dir=out_dir, fold=fold, folds=folds,
                          select_on=select_on, eval_test=eval_test, recipe_id=recipe_id, ckpt_dir=ckpt_dir, hp=hp))
        tag = RR.make_tag(fold, task, model_name, recipe_id, seed)
        r = dict(tag=tag, task=task, split_col=split_col, model=f"{model_name}__{recipe_id}", arch=model_name, seed=seed, fold=fold, recipe_id=recipe_id,
                 inner_val_metrics={"window": {"macro_f1": 0.5}, "drive_full": {"macro_f1": 0.6}}, best_epoch=1, train_minutes=0.1)
        (Path(out_dir) / f"{tag}.json").write_text(json.dumps(r))
        return r

    monkeypatch.setattr(D, "WindowData", FakeData)
    monkeypatch.setattr(T, "run", fake_run)
    results = tmp_path / "results"
    results.mkdir()
    cfgs = RR.configs("folds", "size", None, ("gru",), ["S2"], epochs=2, limit_train=500)
    assert [c["fold"] for c in cfgs] == [0, 1, 2]
    (results / f"{cfgs[1]['tag']}.json").write_text(json.dumps(dict(tag=cfgs[1]["tag"], task="size", model="gru__S2", arch="gru", fold=1, recipe_id="S2")))
    assert RR.execute(cfgs, results, log=lambda s: None) == 2            # fold 1 is finished and skipped
    assert [c["fold"] for c in calls] == [0, 2]
    c = calls[0]
    assert c["task"] == "size" and c["split_col"] == "split_platform" and c["model"] == "gru" and c["seed"] == 0 and c["out_dir"] == results
    assert isinstance(c["data"], FakeData) and c["eval_test"] is False and c["recipe_id"] == "S2" and c["folds"] is None and c["ckpt_dir"] is None
    assert c["select_on"] == "inner_drive_f1" and c["hp"] == {"sampling": "class_model", "aux": "cls+fp", "epochs": 2, "limit_train": 500}
    assert (results / "RECIPE.md").exists() and "| S2 |" in (results / "RECIPE.md").read_text()   # rendered next to the smoke results
    assert RR.execute(cfgs, results, log=lambda s: None) == 0 and len(calls) == 2                # everything finished now
    calls.clear()
    probes = RR.configs("probes", "size", 0, ("gru",), probe_base="S0")
    assert RR.execute(probes, results, log=lambda s: None, ckpt_dir=tmp_path / "ckpt") == 1
    c = calls[0]
    assert c["fold"] == 0 and c["recipe_id"] == "L1_S0" and c["ckpt_dir"] == tmp_path / "ckpt"
    assert c["folds"]["folds"][0]["heldout_models"] == ["Honda Odyssey"] and c["folds"]["n_folds"] == 4 and c["hp"] == {}
