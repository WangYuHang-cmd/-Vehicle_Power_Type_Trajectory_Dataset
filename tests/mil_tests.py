"""experiments/mil_*: episode bags, the multiple-instance model and its trainer on synthetic data (no cache needed).

The torch-dependent tests are skipped in an interpreter without torch; run them with the miniconda interpreter
(`$CPY -m pytest tests/mil_tests.py`). Every test works on experiments.mil_data.make_synthetic().
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from experiments import mil_data as MD  # noqa: E402
from experiments import mil_models as MM  # noqa: E402
from experiments import mil_train as MT  # noqa: E402
from experiments import run_all  # noqa: E402
from experiments import train as TR  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def data():
    return MD.make_synthetic(40, np.random.default_rng(0))


def _batch(data, n=5, K=12, K_w=2, seed=1, train=False):
    rng = np.random.default_rng(seed)
    ids = sorted(data.drive_pool("split_platform", "train"))[:n]
    bags = [data.sample_bag(d, K, K_w, rng) for d in ids]
    tb = MD.to_torch(data.collate_bags(bags, ids, train=train, rng=rng))
    tb["x"] = torch.nan_to_num(tb["x"].float())
    tb["grid"] = torch.nan_to_num(tb["grid"].float())
    return tb, ids, bags


# ----------------------------------------------------------------------------- data
def test_ctx_matrix_follows_the_contract():
    ep = pd.DataFrame(dict(v_anchor=[30.0], dv=[5.0], dur_s=[10.0], mean_a=[0.5], peak_a=[1.0], das_frac=[0.25], lead_valid_frac=[0.75],
                           thw_anchor=[np.nan], gap_in_crop=[True]))
    c = MD.ctx_matrix(ep)
    assert c.shape == (1, MD.CTX_DIM) and c.dtype == np.float32
    assert c[0].tolist() == pytest.approx([1.0, 0.5, 0.5, 0.5, 1.0, 0.25, 0.75, 0.0, 1.0])


def test_drive_targets_masks_and_indices():
    d = pd.DataFrame(dict(drive_id=["a", "b", "c", "d", "e"], vehicle_id=["v"] * 5, group_id=["g"] * 5, model=["m"] * 5,
                          size_class=["LARGE", None, "SMALL", "MID", "MID"], powertrain=["BEV", "HEV", None, "ICE", "ICE"],
                          powertrain_raw=["BEV", "HEV", "MIXED", "ICE", "ICE"], footprint_m2=[5.0, np.nan, 4.0, 4.6, 4.6],
                          curb_mass_kg=[2000.0, 0.0, np.nan, 9.0, 3600.0], wheelbase_mm=[2900.0, np.nan, 2600.0, np.nan, 406.0],
                          split_platform=["train"] * 5, split_device=["train"] * 5, task_size_eligible=[True, False, True, True, True],
                          task_bev_ice_eligible=[True, False, False, True, True]))
    t = MD.drive_targets(d)
    assert t["y_size"].tolist() == [2, -1, 0, 1, 1] and t["y_bev"].tolist() == [0, -1, -1, 1, 1] and t["y_pt4"].tolist() == [3, 1, -1, 0, 0]
    assert t["elig_size"].tolist() == [True, False, True, True, True] and t["elig_bev"].tolist() == [True, False, False, True, True]
    # PLAUSIBLE (experiments.data): curb mass 800-3500 kg, footprint target needs a 2200-4000 mm wheelbase
    assert t["fp_mask"].tolist() == [True, False, True, False, False] and t["mass_mask"].tolist() == [True, False, False, False, False]
    assert t.loc["a", "log_fp"] == pytest.approx(np.log(5.0)) and t.loc["b", "log_fp"] == 0.0 and t.loc["c", "log_mass"] == 0.0
    assert t.loc["d", "log_mass"] == 0.0 and t.loc["e", "log_fp"] == 0.0
    # without a wheelbase column every footprint target is masked (the window recipe does the same)
    t2 = MD.drive_targets(d.drop(columns=["wheelbase_mm"]))
    assert not t2["fp_mask"].any() and t2["mass_mask"].tolist() == t["mass_mask"].tolist()


def test_drive_sets_lift_window_eligibility(data):
    for task in ("size", "bev_ice"):
        full = data.drive_sets(task, "split_platform", "test", "test_full")
        human = data.drive_sets(task, "split_platform", "test", "test_human_only")
        assert human <= full and len(full) > 0
        m = data.window_mask(task, "split_platform", "test", "test_full")
        assert full == set(data.win_drive[m])
        # a drive is in the set iff it has >= 1 eligible test window with a label of the task
        w = data.win
        expect = set(w.loc[(w["split_platform"] == "test") & w["eligible"] & w[MD.TASKS[task]["eligible"]]
                           & w[MD.TASKS[task]["label"]].isin(MD.TASKS[task]["classes"]), "drive_id"])
        assert full == expect
        assert data.drive_sets(task, "split_platform", "test", "test_balanced") <= full
        # the unit universe of contract (4): labelled task-eligible test drives, a superset of the drives with windows
        universe = set(data.task_drives(task, "split_platform", "test"))
        d = data.drives_df
        assert full <= universe
        assert universe == set(d.loc[(d["split_platform"] == "test") & d[MD.TASKS[task]["eligible"]].fillna(False).astype(bool)
                                     & d[MD.TASKS[task]["label"]].isin(MD.TASKS[task]["classes"]), "drive_id"].astype(str))
    with pytest.raises(ValueError):
        data.window_mask("size", "split_platform", "test", "nope")


def test_fold_sets_are_model_disjoint_and_group_clean(data):
    for split_col, fold, mode in (("split_platform", 0, "auto"), ("split_platform", 2, "auto"), ("split_device", 0, "models")):
        tr, va, info = data.fold_sets(split_col, fold, seed=0, mode=mode)
        assert tr and va and not (tr & va)
        m_tr = set(data.tgt.loc[sorted(tr), "model"]); m_va = set(data.tgt.loc[sorted(va), "model"])
        assert not (m_tr & m_va)
        pool = data.drive_pool(split_col, "train") | data.drive_pool(split_col, "validation")
        assert tr <= pool and va <= pool
        held_groups = set(data.tgt.loc[data.tgt["model"].isin(m_va), "group_id"])
        assert not (set(data.tgt.loc[sorted(tr), "group_id"]) & held_groups)
        assert info["source"] in ("fallback_carve", "internal_folds") and info["mode"] == "models"
    tr1, va1, _ = data.fold_sets("split_platform", 0, seed=0)
    tr2, va2, _ = data.fold_sets("split_platform", 0, seed=0)
    assert tr1 == tr2 and va1 == va2                      # deterministic


def test_inner_validation_modes_for_split_device(data):
    # default for split_device = the official validation (device-disjoint, the protocol of its test), models may be shared
    tr, va, info = data.fold_sets("split_device", 0, seed=0)
    assert info["source"] == "official_validation" and info["mode"] == "official"
    assert tr == data.drive_pool("split_device", "train") and va == data.drive_pool("split_device", "validation") and not (tr & va)
    # explicit device-disjoint carve of identity groups
    tr, va, info = data.fold_sets("split_device", 1, seed=0, mode="groups")
    assert info["source"] == "group_carve" and info["n_heldout_groups"] >= 1 and tr and va and not (tr & va)
    assert not (set(data.tgt.loc[sorted(tr), "group_id"]) & set(data.tgt.loc[sorted(va), "group_id"]))
    pool = data.drive_pool("split_device", "train") | data.drive_pool("split_device", "validation")
    assert tr <= pool and va <= pool
    assert MD.resolve_inner_val("split_platform") == "models" and MD.resolve_inner_val("split_device") == "official"
    assert MD.resolve_inner_val("split_device", "groups") == "groups"
    with pytest.raises(ValueError):
        MD.resolve_inner_val("split_platform", "nope")
    d = pd.DataFrame(dict(drive_id=[f"d{i}" for i in range(8)], group_id=["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"]))
    tr, va, held = MD.carve_groups(d, seed=1, frac=0.25)
    assert len(held) == 1 and va == set(d.loc[d["group_id"].isin(held), "drive_id"]) and tr | va == set(d["drive_id"]) and not (tr & va)


def test_carve_models_rule():
    d = pd.DataFrame(dict(drive_id=[f"d{i}" for i in range(8)], model=["A", "A", "B", "B", "C", "C", "D", "D"],
                          group_id=["g1", "g1", "g2", "g1", "g3", "g3", "g4", "g4"]))
    tr, va, held = MD.carve_models(d, seed=3, frac=0.25)
    assert len(held) == 1 and va == set(d.loc[d["model"].isin(held), "drive_id"])
    tainted = set(d.loc[d["model"].isin(held), "group_id"])
    assert not (set(d.loc[d["drive_id"].isin(tr), "group_id"]) & tainted) and not (tr & va)


def test_norm_stats_shapes_and_flags_untouched(data):
    ids = sorted(data.drive_pool("split_platform", "train"))
    s = data.norm_stats(ids, max_episodes=100, seed=0)
    assert s["crop_mean"].shape == (len(MD.CROP_CHANNELS),) and s["grid_std"].shape == (len(MD.GRID_CHANNELS),)
    assert np.all(np.isfinite(s["crop_mean"])) and np.all(s["crop_std"] > 0) and s["n_episodes"] == 100
    for c in MD.BINARY_CHANNELS:
        i = MD.CROP_CHANNELS.index(c)
        assert s["crop_mean"][i] == 0.0 and s["crop_std"][i] == 1.0
    assert s["grid_mean"][MD.GRID_CHANNELS.index("valid")] == 0.0
    t = data.target_stats(ids)
    assert set(t) == {"log_fp_mu", "log_fp_sd", "log_mass_mu", "log_mass_sd"} and t["log_fp_sd"] > 0


def test_sampling_weights_balance_classes_and_models(data):
    ids = sorted(data.drive_pool("split_platform", "train"))
    w = data.sampling_weights(ids, "size")
    assert w.shape == (len(ids),) and w.mean() == pytest.approx(1.0) and (w > 0).all()
    y = data.tgt.loc[ids, "y_size"].to_numpy()
    tot = {c: w[y == c].sum() for c in np.unique(y) if c >= 0}
    assert max(tot.values()) / min(tot.values()) < max(np.bincount(y[y >= 0])) / max(1, min(np.bincount(y[y >= 0])))   # flatter than raw


def test_sample_bag_caps_and_priorities(data):
    rng = np.random.default_rng(0)
    # a drive with many priority episodes: build one by pointing the type array at a fake drive
    d = data.ep_drive[0]
    idx = data._ep_by_drive[d]
    saved = data.type_id[idx].copy()
    data.type_id[idx] = np.array([MD.PRIORITY_TYPES[i % 3] for i in range(len(idx))])      # all LEAD_ACC / LEAD_DEC / TURN
    try:
        ep, tw = data.sample_bag(d, K=64, K_w=8, rng=rng)
        assert len(ep) <= 64 and len(set(ep.tolist())) == len(ep) and set(ep.tolist()) <= set(idx.tolist())
        ep2, _ = data.sample_bag(d, K=2, K_w=0, rng=rng)
        assert len(ep2) == 2 and len(_) == 0
    finally:
        data.type_id[idx] = saved
    # caps: a drive with > 16 episodes of one priority type contributes at most 16 of it, other types fill the rest
    big = max(data._ep_by_drive, key=lambda k: len(data._ep_by_drive[k]))
    idx = data._ep_by_drive[big]
    saved = data.type_id[idx].copy()
    data.type_id[idx] = 7                                                                   # everything TURN
    data.type_id[idx[:3]] = 0                                                               # three ACC episodes
    try:
        ep, _ = data.sample_bag(big, K=64, K_w=0, rng=rng)
        assert (data.type_id[ep] == 7).sum() == min(16, len(idx) - 3) + max(0, min(64, len(idx)) - 3 - 16) or len(idx) <= 19
        ep, _ = data.sample_bag(big, K=18, K_w=0, rng=rng)
        n_turn = int((data.type_id[idx] == 7).sum())
        # priority phase: min(16, n_turn) TURN episodes; fill phase: 2 more from everything left (TURN or ACC)
        assert len(ep) == min(18, len(idx)) and (data.type_id[ep] == 7).sum() >= min(16, n_turn) and (data.type_id[ep] == 0).sum() <= 3
        ep, _ = data.sample_bag(big, K=16, K_w=0, rng=rng)
        assert len(ep) == min(16, len(idx)) and (data.type_id[ep] == 7).sum() == min(16, n_turn)       # the cap fills K exactly
    finally:
        data.type_id[idx] = saved
    # human_only keeps human_strict episodes and human_only_strict windows only
    for d in list(data._ep_by_drive)[:10]:
        ep, tw = data.sample_bag(d, K=64, K_w=8, rng=rng, human_only=True)
        assert data.human_strict[ep].all()
        rows = np.flatnonzero(np.isin(data.win_tidx, tw))
        assert data.win_human[rows].all() and len(tw) <= 8
        ep_all, tw_all = data.sample_bag(d, K=64, K_w=8, rng=rng)
        assert len(ep_all) == min(64, len(data._ep_by_drive[d])) and len(tw_all) <= min(8, len(data._win_by_drive.get(d, [])))


def test_bag_for_window_uses_episodes_anchored_inside(data):
    rows = data.window_rows("size", "split_platform", "test", "test_full")
    for r in rows[:10]:
        ep, tw = data.bag_for_window(int(r))
        wid = data.win["window_id"].iat[int(r)]
        assert all(data.ep["window_id"].iat[int(e)] == wid for e in ep)
        w_idx = int(data.win["w_idx"].iat[int(r)])
        assert all(data.anchor_row[e] // 600 == w_idx for e in ep)
        assert tw.tolist() == [int(data.win_tidx[r])] if data.win_tidx[r] >= 0 else len(tw) == 0
        assert len(data.bag_for_window(int(r), use_windows=False)[1]) == 0


def test_iterate_drive_batches_respects_max_model_share():
    rng = np.random.default_rng(0)
    ids = [f"d{i}" for i in range(200)]
    model_of = {d: f"m{i % 20}" for i, d in enumerate(ids)}
    w = rng.uniform(0.5, 2.0, size=200)
    batches = list(MD.iterate_drive_batches(ids, w, model_of, 24, rng, 0.10))
    assert len(batches) == int(np.ceil(200 / 24)) and all(len(b) == 24 for b in batches)
    for b in batches:
        counts = pd.Series([model_of[d] for d in b]).value_counts()
        assert counts.max() <= 2                                  # floor(0.10 * 24) = 2
    # aligned array form of model_of and a pool with too few models (cap raised)
    few = ids[:10]
    bs = list(MD.iterate_drive_batches(few, None, np.array([model_of[d] for d in few], dtype=object), 24, rng, 0.10))
    assert all(len(b) == 24 for b in bs) and len(bs) == 1
    # weights steer the draws
    w0 = np.zeros(200); w0[5] = 1.0
    assert set(np.concatenate(list(MD.iterate_drive_batches(ids, w0, model_of, 4, rng, 1.0))).tolist()) == {"d5"}


def test_collate_on_synthetic_round_trips_shapes(data):
    rng = np.random.default_rng(3)
    ids = sorted(data.drive_pool("split_platform", "train"))[:6]
    bags = [data.sample_bag(d, 10, 3, rng) for d in ids]
    b = data.collate_bags(bags, ids, train=True, rng=rng)
    M, Mw = sum(len(e) for e, _ in bags), sum(len(t) for _, t in bags)
    assert b["x"].shape == (M, MD.CROP_OUT, len(MD.CROP_CHANNELS)) and b["x"].dtype == np.float16
    assert b["ctx"].shape == (M, MD.CTX_DIM) and b["type"].shape == (M,) and b["bag_idx"].shape == (M,) and b["pos"].shape == (M,)
    assert b["grid"].shape[1:] == (MD.GRID_LEN, len(MD.GRID_CHANNELS)) and b["grid"].shape[0] == int((b["grid_map"] >= 0).sum())
    assert (b["grid_map"] >= 0).sum() == int((data.grid_idx[np.concatenate([e for e, _ in bags])] >= 0).sum())
    assert b["wemb"].shape == (Mw, data.wemb_dim) and b["wemb_bag_idx"].shape == (Mw,) and b["wemb_pos"].shape == (Mw,)
    assert b["mask"].shape == (6, max(len(e) + len(t) for e, t in bags)) and int(b["mask"].sum()) == M + Mw
    for k in ("y_size", "y_bev", "y_pt4", "log_fp", "fp_mask", "log_mass", "mass_mask", "elig_size", "elig_bev", "n_inst"):
        assert b[k].shape == (6,)
    # positions are unique within a bag and the slots exactly cover the mask
    slots = set(zip(b["bag_idx"].tolist(), b["pos"].tolist())) | set(zip(b["wemb_bag_idx"].tolist(), b["wemb_pos"].tolist()))
    assert len(slots) == M + Mw and all(b["mask"][i, j] for i, j in slots)
    # eval collate: fixed offset 10; train collate: jitter within 0..20
    b0 = data.collate_bags(bags, ids, train=False)
    e0 = bags[0][0][0]
    assert np.array_equal(np.nan_to_num(b0["x"][0]), np.nan_to_num(np.asarray(data.E[e0][MD.CROP_OFFSET:MD.CROP_OFFSET + MD.CROP_OUT])))
    found = any(np.array_equal(np.nan_to_num(b["x"][0]), np.nan_to_num(np.asarray(data.E[e0][o:o + MD.CROP_OUT]))) for o in range(MD.JITTER_MAX + 1))
    assert found
    tb = MD.to_torch(b)
    assert tb["x"].dtype == torch.float16 and tb["mask"].dtype == torch.bool and isinstance(tb["drive_id"], np.ndarray)


def test_write_and_reload_synthetic_cache(tmp_path, data):
    wemb = MD.write_synthetic(data, tmp_path)
    d2 = MD.EpisodeData(tmp_path, wemb=wemb)
    assert d2.summary() == data.summary() and d2.wemb_tags == ["synthetic"]
    assert d2.balanced_ids("size", "test", "split_platform") == data.balanced_ids("size", "test", "split_platform")
    assert d2.drive_sets("bev_ice", "split_platform", "test") == data.drive_sets("bev_ice", "split_platform", "test")
    assert np.array_equal(d2.win_tidx, data.win_tidx)
    assert (tmp_path / "work" / "episodes" / "channels.json").exists() and (tmp_path / "work" / "tensors" / "wemb_synthetic.json").exists()


# ----------------------------------------------------------------------------- model
@pytest.mark.parametrize("pool", ["gated_attn", "mean_logit"])
def test_forward_shapes_of_every_head(data, pool):
    torch.manual_seed(0)
    tb, ids, bags = _batch(data)
    m = MM.DriveMIL(d=32, pool=pool, wemb_dim=data.wemb_dim)
    out = m(tb)
    B, K = tb["mask"].shape
    M_all = tb["bag_idx"].numel() + tb["wemb_bag_idx"].numel()
    assert out["size"].shape == (B, 3) and out["bev"].shape == (B, 2) and out["pt4"].shape == (B, 4)
    assert out["footprint"].shape == (B,) and out["logmass"].shape == (B,) and out["ordinal"].shape == (B, 2)
    assert out["inst_size"].shape == (M_all, 3) and out["inst_bev"].shape == (M_all, 2) and out["attn"].shape == (B, K, 2)
    assert all(torch.isfinite(v).all() for v in out.values() if v.dtype.is_floating_point)
    h, mask = m.encode(tb)
    assert h.shape == (B, K, 32) and mask.shape == (B, K)
    sum(v.float().sum() for k, v in out.items() if k in MM.HEADS).backward()
    assert m.enc.stem.c1.weight.grad is not None and m.enc.type_emb.weight.grad is not None
    # ablations: no grid / no windows still run
    m2 = MM.DriveMIL(d=32, pool=pool, use_grid=False, use_windows=False, wemb_dim=data.wemb_dim)
    out2 = m2(tb)
    assert out2["inst_size"].shape == (tb["bag_idx"].numel(), 3) and out2["size"].shape == (B, 3)


def test_padding_invariance(data):
    torch.manual_seed(0)
    tb, _, _ = _batch(data)
    for pool in ("gated_attn", "mean_logit"):
        m = MM.DriveMIL(d=32, pool=pool, wemb_dim=data.wemb_dim).eval()
        with torch.no_grad():
            ref = m(tb)
            wide = dict(tb)
            wide["mask"] = torch.cat([tb["mask"], torch.zeros(tb["mask"].shape[0], 3, dtype=torch.bool)], 1)   # extra padded slots
            out = m(wide)
        for k in MM.HEADS + ("inst_size", "inst_bev"):
            assert torch.allclose(ref[k], out[k], atol=1e-5), (pool, k)
        assert torch.allclose(out["attn"][:, : tb["mask"].shape[1]], ref["attn"], atol=1e-6) and (out["attn"][:, tb["mask"].shape[1]:] == 0).all()
    # the pooling module itself ignores the content of padded rows
    pool_mod = MM.GatedAttentionPool(16, 2)
    h = torch.randn(3, 7, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    z1, a1 = pool_mod(h, mask)
    h2 = h.clone()
    h2[~mask] = torch.randn(int((~mask).sum()), 16) * 100
    z2, a2 = pool_mod(h2, mask)
    assert torch.allclose(z1, z2, atol=1e-5) and torch.allclose(a1, a2, atol=1e-6)
    # an empty bag pools to zero without NaN
    z3, a3 = pool_mod(h, torch.zeros(3, 7, dtype=torch.bool))
    assert torch.isfinite(z3).all() and (z3 == 0).all() and (a3 == 0).all()


def test_attention_sums_to_one_over_valid_instances(data):
    torch.manual_seed(0)
    tb, _, _ = _batch(data)
    m = MM.DriveMIL(d=32, pool="gated_attn", wemb_dim=data.wemb_dim).eval()
    with torch.no_grad():
        out = m(tb)
    a = out["attn"]
    assert torch.allclose(a.sum(1), torch.ones(a.shape[0], 2), atol=1e-5)
    assert (a[~tb["mask"]] == 0).all() and (a[tb["mask"]] > 0).all()


def test_mean_logit_equals_weighted_mean_of_instance_logits(data):
    torch.manual_seed(0)
    tb, _, _ = _batch(data)
    m = MM.DriveMIL(d=32, pool="mean_logit", wemb_dim=data.wemb_dim).eval()
    with torch.no_grad():
        out = m(tb)
    B, K = tb["mask"].shape
    g = out["inst_g"]
    assert (g > 0).all()
    bag, pos = out["inst_bag_idx"], out["inst_pos"]
    for b in range(B):
        sel = bag == b
        gw = g[sel]
        expect = (gw[:, 0:1] * out["inst_size"][sel]).sum(0) / gw[:, 0].sum()
        assert torch.allclose(out["size"][b], expect, atol=1e-5)
        expect_b = (gw[:, 1:2] * out["inst_bev"][sel]).sum(0) / gw[:, 1].sum()
        assert torch.allclose(out["bev"][b], expect_b, atol=1e-5)
        assert torch.allclose(out["attn"][b, pos[sel], 0], gw[:, 0] / gw[:, 0].sum(), atol=1e-6)
    assert torch.allclose(out["attn"].sum(1), torch.ones(B, 2), atol=1e-5)


def test_eligibility_masks_zero_the_gradient_of_ineligible_bags(data):
    torch.manual_seed(0)
    tb, _, _ = _batch(data, n=4)
    tb["elig_size"] = torch.tensor([False, True, True, True])
    tb["elig_bev"] = torch.tensor([True, False, True, True])
    tb["fp_mask"] = torch.tensor([False, True, True, True])
    tb["mass_mask"] = torch.tensor([False, True, True, True])
    tb["y_pt4"] = torch.tensor([-1, 0, 1, 3])
    tb["y_size"] = tb["y_size"].clamp_min(0); tb["y_bev"] = tb["y_bev"].clamp_min(0)
    m = MM.DriveMIL(d=32, pool="gated_attn", wemb_dim=data.wemb_dim).eval()
    out = m(tb)
    leaf = {k: out[k].detach().clone().requires_grad_(True) for k in MM.HEADS + ("inst_size", "inst_bev")}
    fake = dict(out); fake.update(leaf)
    tstats = dict(log_fp_mu=1.5, log_fp_sd=0.1, log_mass_mu=7.4, log_mass_sd=0.2)
    total, parts = MT.mil_loss(fake, tb, tstats, aux=True)
    total.backward()
    assert set(parts) >= {"ce_size", "ce_bev", "ce_inst_size", "ce_inst_bev", "huber_fp", "huber_mass", "ce_pt4", "bce_ordinal", "total"}
    for k in ("size", "footprint", "logmass", "ordinal", "pt4"):
        assert (leaf[k].grad[0] == 0).all(), k                       # bag 0 ineligible for size / fp / mass / pt4
        assert leaf[k].grad[2].abs().sum() > 0, k
    assert (leaf["bev"].grad[1] == 0).all() and leaf["bev"].grad[0].abs().sum() > 0
    bag = out["inst_bag_idx"]
    assert (leaf["inst_size"].grad[bag == 0] == 0).all() and leaf["inst_size"].grad[bag == 2].abs().sum() > 0
    assert (leaf["inst_bev"].grad[bag == 1] == 0).all() and leaf["inst_bev"].grad[bag == 0].abs().sum() > 0
    # instance dropout mask removes instances from the instance loss
    keep = torch.ones_like(bag, dtype=torch.bool); keep[bag == 2] = False
    leaf2 = {k: out[k].detach().clone().requires_grad_(True) for k in ("inst_size", "inst_bev")}
    fake2 = dict(out); fake2.update(leaf2)
    t2, _ = MT.mil_loss(fake2, tb, tstats, aux=False, inst_keep=keep)
    t2.backward()
    assert (leaf2["inst_size"].grad[bag == 2] == 0).all() and leaf2["inst_size"].grad[bag == 3].abs().sum() > 0


def test_batch_prep_augmentation_keeps_bags_non_empty(data):
    torch.manual_seed(0)
    tb, ids, bags = _batch(data, n=6, train=True)
    stats = data.norm_stats(ids)
    prep = MT.BatchPrep(data, stats, torch.device("cpu"), aug=dict(p_flip=1.0, inst_drop=0.9, lead_drop=0.5, noise=0.0), seed=0)
    raw = MD.to_torch(data.collate_bags(bags, ids, train=False))
    x_raw = raw["x"].float().clone()
    out = prep(dict(raw), train=True)
    assert out["mask"].any(1).all() and out["mask"].shape == raw["mask"].shape and int(out["mask"].sum()) <= int(raw["mask"].sum())
    assert torch.isfinite(out["x"]).all() and torch.isfinite(out["grid"]).all()
    # a sign flip with p = 1 negates the lateral channels (compare the standardised values of a flipped vs unflipped copy)
    prep0 = MT.BatchPrep(data, stats, torch.device("cpu"), aug=dict(p_flip=0.0, inst_drop=0.0, lead_drop=0.0, noise=0.0), seed=0)
    ref = prep0(MD.to_torch(data.collate_bags(bags, ids, train=False)), train=False)
    side_idx = torch.as_tensor(data.lateral_idx)
    mean = torch.as_tensor(stats["crop_mean"])[side_idx]; std = torch.as_tensor(stats["crop_std"])[side_idx]
    flipped = (-x_raw[:, :, side_idx].clamp(torch.as_tensor(data.crop_lo)[side_idx], torch.as_tensor(data.crop_hi)[side_idx]) - mean) / std
    flipped = torch.nan_to_num(flipped)
    assert torch.allclose(out["x"][:, :, side_idx], flipped, atol=1e-4)
    other = [i for i in range(len(MD.CROP_CHANNELS)) if i not in data.lateral_idx]
    assert torch.allclose(out["x"][:, :, other], ref["x"][:, :, other], atol=1e-5)


def test_vehicle_level_and_drive_from_windows_definitions():
    classes = ("A", "B")
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]], dtype=np.float32)
    veh = np.array(["v1", "v1", "v2", "v3"], dtype=object)
    y = np.array([0, 0, 1, 1])
    models = np.array(["m1", "m1", "m2", "m2"], dtype=object)
    # direct-bag vehicle rule (mean drive log-probability), one row per drive
    vf = MT.vehicle_level_bag(veh, probs, y, models, classes, 1)
    assert vf["n_units"] == 3 and vf["n_vehicles"] == 3 and vf["n_models"] == 2 and vf["n_drives_used"] == 4 and "n_drives" not in vf
    assert vf["confusion"] == [[1, 0], [1, 1]]                      # v1: mean log-prob favours A; v2 -> A (wrong); v3 -> B
    vm = MT.vehicle_level_bag(veh, probs, y, models, classes, 2)
    assert vm["n_units"] == 1 and vm["accuracy"] == 1.0
    # window-mean drive rule (contract (4)) with the unit universe
    drives = np.array(["d1", "d1", "d2", "d2"], dtype=object)
    dw = MT.drive_level_from_windows(drives, probs, y, classes, models, all_drives=["d1", "d2", "d3"])
    assert dw["n_units"] == 2 and dw["n_drives"] == 2 and dw["n_windows_used"] == 4 and dw["n_models"] == 2 and dw["n_units_without_windows"] == 1
    assert dw["confusion"] == [[1, 0], [0, 1]] and "n_vehicles" not in dw
    # window-mean vehicle rule: windows pooled per vehicle, multi = vehicles with >= 2 distinct drives
    vw = MT.vehicle_level_from_windows(veh, drives, probs, y, classes, models, 1)
    assert vw["n_units"] == 3 and vw["n_vehicles"] == 3 and vw["n_windows_used"] == 4 and "n_drives" not in vw
    assert vw["confusion"] == [[1, 0], [1, 1]]                      # v1 mean (0.55, 0.45) -> A; v2 -> A (wrong); v3 -> B
    vw2 = MT.vehicle_level_from_windows(veh, drives, probs, y, classes, models, 2)
    assert vw2["n_units"] == 0 and vw2["n_windows_used"] == 0        # v1 has one drive (d1) only
    veh2 = np.array(["v1", "v1", "v1", "v3"], dtype=object)
    vw3 = MT.vehicle_level_from_windows(veh2, drives, probs, y, classes, models, 2)
    assert vw3["n_units"] == 1 and vw3["n_windows_used"] == 3
    um = MT.unit_metrics(np.array([0, 1]), np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32), classes, ["m"], n_windows_used=2)
    assert um["confusion"] == [[1, 0], [1, 0]] and um["n_windows_used"] == 2 and um["n_models"] == 1   # ties go to the earlier class
    empty = MT.drive_level_from_windows(np.zeros(0, object), np.zeros((0, 2), np.float32), np.zeros(0, np.int64), classes, [], all_drives=["d1"])
    assert empty["n_units"] == 0 and empty["n_units_without_windows"] == 1 and np.isnan(empty["macro_f1"])


def test_window_mean_aggregation_equals_train_unit_level():
    rng = np.random.default_rng(5)
    n, classes = 300, ("SMALL", "MID", "LARGE")
    drives = np.array([f"d{i}" for i in rng.integers(0, 40, n)], dtype=object)
    veh_of = {f"d{i}": f"v{i // 3}" for i in range(40)}
    model_of = {f"d{i}": f"m{i // 5}" for i in range(40)}
    vehicles = np.array([veh_of[d] for d in drives], dtype=object)
    models = np.array([model_of[d] for d in drives], dtype=object)
    probs = rng.dirichlet(np.ones(3), size=n).astype(np.float32)
    y = np.array([int(d[1:]) % 3 for d in drives], dtype=np.int64)
    win = pd.DataFrame(dict(drive_id=drives, vehicle_id=vehicles, model=models))
    idx = np.arange(n)
    universe = [f"d{i}" for i in range(45)]
    ref_d = TR.unit_level(win, idx, probs, y, classes, "drive_id", all_units=universe)
    got_d = MT.drive_level_from_windows(drives, probs, y, classes, models, vehicles, all_drives=universe)
    for k in ("macro_f1", "balanced_acc", "accuracy", "confusion", "n_units", "n_windows_used", "n_models", "n_units_without_windows"):
        assert got_d[k] == ref_d[k], k
    assert got_d["macro_f1"] == TR.drive_level(win, idx, probs, y, classes, how="mean")["macro_f1"]
    ref_v = TR.unit_level(win, idx, probs, y, classes, "vehicle_id")
    got_v = MT.vehicle_level_from_windows(vehicles, drives, probs, y, classes, models, 1)
    for k in ("macro_f1", "confusion", "n_units", "n_windows_used", "n_models"):
        assert got_v[k] == ref_v[k], k
    n_dr = pd.Series(drives).groupby(vehicles).nunique()
    multi = set(n_dr[n_dr >= 2].index)
    sel = np.array([v in multi for v in vehicles])
    ref_m = TR.unit_level(win[sel].reset_index(drop=True), np.arange(int(sel.sum())), probs[sel], y[sel], classes, "vehicle_id")
    got_m = MT.vehicle_level_from_windows(vehicles, drives, probs, y, classes, models, 2)
    assert got_m["confusion"] == ref_m["confusion"] and got_m["n_units"] == ref_m["n_units"] and got_m["n_windows_used"] == ref_m["n_windows_used"]


def test_discover_wemb_provenance(tmp_path):
    tdir = tmp_path / "work" / "tensors"
    tdir.mkdir(parents=True)
    def put(tag, meta):
        np.save(tdir / f"wemb_{tag}.npy", np.zeros((4, 2), np.float16))
        if meta is not None:
            (tdir / f"wemb_{tag}.json").write_text(json.dumps(meta))
    put("split_platform__size__gru__s0", dict(ckpt="split_platform__size__gru__s0", dim=2, limit=0, split_col="split_platform", task="size"))
    put("split_platform__bev_ice__resnet__s1", None)                                    # no sidecar: the tag prefix decides
    put("split_device__size__gru__s0", dict(dim=2, limit=0, split_col="split_device", task="size"))
    put("split_platform__size__gru__s0_limit500", dict(dim=2, limit=500, split_col="split_platform", task="size"))
    put("fold1__size__gru__S1__s0", dict(dim=2, limit=0, split_col="split_platform", task="size"))
    put("fold2__size__gru__S1__s0", None)
    quiet = lambda s: None  # noqa: E731
    kept, info = MT.discover_wemb(tmp_path, None, "split_platform", fold=None, log=quiet)
    assert set(kept) == {"split_platform__size__gru__s0", "split_platform__bev_ice__resnet__s1", "fold1__size__gru__S1__s0", "fold2__size__gru__S1__s0"}
    assert info["dropped"] == {"other_split": 1, "smoke": 1} and info["n_candidates"] == 6 and info["fold_rule"] is False
    kept, info = MT.discover_wemb(tmp_path, None, "split_device", fold=None, log=quiet)
    assert set(kept) == {"split_device__size__gru__s0"} and info["dropped"]["other_split"] == 4
    # model-disjoint inner validation of fold 1: only fold-1 checkpoints
    kept, info = MT.discover_wemb(tmp_path, None, "split_platform", fold=1, log=quiet)
    assert set(kept) == {"fold1__size__gru__S1__s0"} and info["dropped"] == {"other_split": 1, "smoke": 1, "not_fold_checkpoint": 3}
    kept, _ = MT.discover_wemb(tmp_path, None, "split_platform", fold=0, log=quiet)
    assert kept == {}
    kept, _ = MT.discover_wemb(tmp_path, None, "split_platform", fold=0, allow_full=True, log=quiet)
    assert len(kept) == 4
    # explicit tags must all pass
    kept, _ = MT.discover_wemb(tmp_path, "split_platform__size__gru__s0", "split_platform", log=quiet)
    assert list(kept) == ["split_platform__size__gru__s0"]
    with pytest.raises(ValueError):
        MT.discover_wemb(tmp_path, "split_platform__size__gru__s0_limit500", "split_platform", log=quiet)
    with pytest.raises(ValueError):
        MT.discover_wemb(tmp_path, "split_platform__size__gru__s0", "split_platform", fold=1, log=quiet)
    with pytest.raises(ValueError):
        MT.discover_wemb(tmp_path, "split_device__size__gru__s0", "split_platform", log=quiet)
    assert MT.wemb_tag_allowed("fold1__size__gru__S1__s0", {}, "split_platform", fold=1) == (True, "ok")
    assert MT.wemb_tag_allowed("fold1__size__gru__S1__s0", {}, "split_device") == (False, "other_split")


REQUIRED_KEYS = {"tag", "task", "split_col", "model", "arch", "seed", "n_params", "cfg", "n_train", "n_val", "n_test", "epochs_run", "best_epoch",
                 "val_macro_f1", "val_metrics", "inner_val_metrics", "train_minutes", "protocols", "drive_level", "unit_level", "vehicle_level",
                 "drive_level_bag", "unit_level_bag", "vehicle_level_bag", "instance_type_counts", "per_type_instance_f1", "history", "loss_parts",
                 "fold_info", "fold", "recipe_id", "decision", "select_on"}
UNIT_KEYS = {"n", "macro_f1", "balanced_acc", "accuracy", "per_class_recall", "support", "confusion", "classes", "n_units", "n_windows_used", "n_models"}


@pytest.mark.parametrize("pool", ["gated_attn", "mean_logit"])
def test_run_synthetic_writes_results(tmp_path, pool):
    res = MT.run(split_col="split_platform", seed=0, pool=pool, epochs=1, synthetic=True, out_dir=tmp_path, log=lambda s: None,
                 n_synthetic=30, d=16, K=8, K_w=2, batch_drives=8, K_eval=12, K_w_eval=2, eval_bags=16)
    assert set(res) == {"size", "bev_ice"}
    for task, r in res.items():
        name = "episode_mil" if pool == "gated_attn" else "episode_mil_meanlogit"
        tag = f"split_platform__{task}__{name}__s0"
        assert r["tag"] == tag and r["model"] == name and REQUIRED_KEYS <= set(r)
        js = json.loads((tmp_path / f"{tag}.json").read_text())
        assert REQUIRED_KEYS <= set(js) and js["epochs_run"] == 1
        assert set(js["protocols"]) == {"test_full", "test_balanced", "test_human_only"}
        assert js["recipe_id"] == "mil" and js["fold"] is None and js["decision"] == "softmax" and js["arch"] == name
        assert js["cfg"]["select_on"] == "inner_bag_f1" and js["cfg"]["inner_val"] == "models" and js["cfg"]["wemb_inner_val_clean"] is False
        assert js["fold_info"]["wemb_inner_val_clean"] is False                   # synthetic embeddings are not fold-0 checkpoints
        # contract (4): the drive_level / unit_level keys hold the window-mean protocols (the same windows as `protocols`)
        assert set(js["drive_level"]) == {"test_full", "test_human_only"} and "n_drives" in js["drive_level"]["test_full"]
        assert set(js["unit_level"]) == {"drive_full", "drive_human_only", "vehicle_full", "vehicle_full_multi"}
        for m in js["unit_level"].values():
            assert UNIT_KEYS <= set(m)
        assert js["unit_level"]["drive_full"] == js["drive_level"]["test_full"] and js["unit_level"]["drive_human_only"] == js["drive_level"]["test_human_only"]
        assert js["unit_level"]["drive_full"]["n_windows_used"] == js["protocols"]["test_full"]["n"]
        assert js["unit_level"]["drive_human_only"]["n_windows_used"] == js["protocols"]["test_human_only"]["n"]
        assert "n_units_without_windows" in js["unit_level"]["drive_full"] and "n_vehicles" in js["unit_level"]["vehicle_full"]
        assert "n_drives" not in js["unit_level"]["vehicle_full"] and "n_vehicles" not in js["unit_level"]["drive_full"]
        assert js["unit_level"]["vehicle_full_multi"]["n_units"] <= js["unit_level"]["vehicle_full"]["n_units"]
        assert set(js["vehicle_level"]) == {"vehicle_full", "vehicle_full_multi"} and js["vehicle_level"]["vehicle_full"] == js["unit_level"]["vehicle_full"]
        # the model's own direct-bag prediction is listed separately
        assert set(js["drive_level_bag"]) == {"test_full", "test_human_only"} and set(js["unit_level_bag"]) == set(js["unit_level"])
        for m in js["unit_level_bag"].values():
            assert {"n_units", "n_models", "macro_f1", "confusion"} <= set(m) and "n_windows_used" not in m
        assert "n_units_without_instances" in js["drive_level_bag"]["test_full"] and "n_instances_used" in js["drive_level_bag"]["test_full"]
        assert set(js["inner_val_metrics"]) == {"window", "drive_full", "drive_full_bag"}
        assert UNIT_KEYS <= set(js["inner_val_metrics"]["drive_full"]) and "n_units_without_windows" in js["inner_val_metrics"]["drive_full"]
        assert js["val_metrics"] == js["inner_val_metrics"]["drive_full_bag"]
        assert set(js["instance_type_counts"]) == set(MT.TYPE_NAMES) and set(js["per_type_instance_f1"]) == set(MT.TYPE_NAMES)
        assert len(js["history"]) == 1 and "parts" in js["history"][0]
        z = np.load(tmp_path / f"{tag}_probs.npz", allow_pickle=True)
        k = len(MD.TASKS[task]["classes"])
        assert set(z.files) == {"window_id", "probs", "y"} and z["probs"].shape == (js["protocols"]["test_full"]["n"], k)
        assert np.allclose(z["probs"].sum(1), 1.0, atol=1e-4)
        zd = np.load(tmp_path / f"{tag}_drive_probs.npz", allow_pickle=True)
        assert set(zd.files) == {"drive_id", "vehicle_id", "probs", "y", "n_episodes", "n_human_episodes"}
        assert zd["probs"].shape == (js["unit_level_bag"]["drive_full"]["n_units"], k) and (zd["n_human_episodes"] <= zd["n_episodes"]).all()
        assert (tmp_path / MT.CKPT_DIRNAME / f"{tag}.pt").exists() and not (tmp_path / f"{tag}_oof.npz").exists()
    ck = torch.load(tmp_path / MT.CKPT_DIRNAME / f"split_platform__size__{'episode_mil' if pool == 'gated_attn' else 'episode_mil_meanlogit'}__s0.pt",
                    weights_only=False)
    assert {"state", "mean", "std", "cfg", "task", "split_col", "model", "channels", "classes"} <= set(ck) and ck["kind"] == MT.CKPT_KIND
    # the final-run JSONs render with the existing table code next to window-model results
    md = run_all.render(run_all.load_results(tmp_path))
    assert any(run_all.NAMES.get(k, k) in md for k in ("episode_mil", "episode_mil_meanlogit")) and "split_platform" in md


def test_run_official_validation_and_ablations(tmp_path):
    res = MT.run(split_col="split_device", seed=1, pool="gated_attn", val_mode="official", epochs=1, synthetic=True, out_dir=tmp_path,
                 log=lambda s: None, n_synthetic=30, d=16, K=8, K_w=2, batch_drives=8, K_eval=12, K_w_eval=2, eval_bags=16,
                 use_grid=False, use_windows=False, aux=False, human_only_train=True, variant="abl", eval_test=False)
    r = res["size"]
    assert r["model"] == "episode_mil_abl" and r["fold_info"]["source"] == "official_validation" and r["protocols"] == {}
    assert r["cfg"]["use_windows"] is False and r["cfg"]["aux"] is False and r["cfg"]["human_only_train"] is True
    assert r["cfg"]["wemb_inner_val_clean"] is True and r["cfg"]["inner_val"] == "official"
    assert set(r["loss_parts"]) == {"ce_size", "ce_bev", "ce_inst_size", "ce_inst_bev", "total"}
    # a selection run (no test) carries the fold tag even without --fold
    assert r["tag"] == "fold0__size__episode_mil_abl__mil__s1" and (tmp_path / "fold0__bev_ice__episode_mil_abl__mil__s1.json").exists()
    assert (tmp_path / "fold0__size__episode_mil_abl__mil__s1_oof.npz").exists()


def test_run_locations_and_fold_run_outputs(tmp_path, monkeypatch):
    exp = tmp_path / "experiments"
    monkeypatch.setattr(MT, "EXP_ROOT", exp)
    monkeypatch.setattr(MT, "RECIPE_ROOT", exp / "recipe")
    # final run: train.py tag in experiments/results, checkpoints in the sibling ckpt_mil directory
    loc = MT.run_locations("split_platform", "episode_mil", 0, None, True)
    assert loc["tags"]["size"] == "split_platform__size__episode_mil__s0" and loc["out_dir"] == exp / "results"
    assert loc["ckpt_dir"] == exp / MT.CKPT_DIRNAME and loc["is_fold_run"] is False and loc["recipe_id"] == "mil"
    # fold / selection runs: recipe location with the contract (5) tag
    loc = MT.run_locations("split_platform", "episode_mil", 2, 1, False, recipe_id="milA")
    assert loc["tags"]["bev_ice"] == "fold1__bev_ice__episode_mil__milA__s2" and loc["out_dir"] == exp / "recipe" / "results"
    assert loc["ckpt_dir"] == exp / "recipe" / MT.CKPT_DIRNAME and loc["is_fold_run"] is True
    loc = MT.run_locations("split_platform", "episode_mil", 0, 2, True)          # a fold with the test evaluated is still a fold run
    assert loc["tags"]["size"].startswith("fold2__") and loc["out_dir"] == exp / "recipe" / "results"
    assert MT.run_locations("split_platform", "episode_mil", 0, None, True, synthetic=True)["out_dir"].name == "mil_smoke"
    assert MT.run_locations("split_platform", "m", 0, None, True, out_dir=tmp_path / "x", ckpt_dir=tmp_path / "ck")["ckpt_dir"] == tmp_path / "ck"
    assert MT.run_locations("split_platform", "m", 0, None, True, out_dir=tmp_path / "x")["ckpt_dir"] == tmp_path / "x" / MT.CKPT_DIRNAME
    # a run without protocols never lands in experiments/results
    with pytest.raises(ValueError):
        MT.run_locations("split_platform", "episode_mil", 0, None, False, out_dir=exp / "results")
    with pytest.raises(ValueError):
        MT.run(split_col="split_platform", synthetic=True, eval_test=False, out_dir=exp / "results", log=lambda s: None, n_synthetic=20, epochs=1)
    assert not (exp / "results").exists()
    # a fold run with the default locations
    res = MT.run(split_col="split_platform", seed=0, fold=1, eval_test=False, synthetic=True, epochs=1, log=lambda s: None, n_synthetic=30,
                 d=16, K=8, K_w=2, batch_drives=8, K_eval=12, K_w_eval=2, eval_bags=16, out_dir=exp / "recipe" / "results", recipe_id="milA",
                 select_on="inner_drive_f1")
    r = res["size"]
    assert r["tag"] == "fold1__size__episode_mil__milA__s0" and r["fold"] == 1 and r["recipe_id"] == "milA" and r["protocols"] == {}
    assert r["fold_info"]["fold"] == 1 and r["fold_info"]["source"] in ("fallback_carve", "internal_folds")
    assert set(r["inner_val_metrics"]) == {"window", "drive_full", "drive_full_bag"} and r["val_metrics"] == r["inner_val_metrics"]["drive_full"]
    assert r["val_macro_f1"] == pytest.approx(r["inner_val_metrics"]["drive_full"]["macro_f1"])
    assert (exp / "recipe" / "results" / f"{r['tag']}.json").exists() and (exp / "recipe" / "results" / MT.CKPT_DIRNAME / f"{r['tag']}.pt").exists()
    z = np.load(exp / "recipe" / "results" / f"{r['tag']}_oof.npz", allow_pickle=True)
    assert set(z.files) == {"window_id", "probs", "y"} and z["probs"].shape == (r["inner_val_metrics"]["window"]["n"], 3)
    assert not (exp / "results").exists() and not list((exp / "recipe" / "results").glob("*_probs.npz"))
    # fixed epochs: no early stopping, the last state is kept
    res2 = MT.run(split_col="split_platform", seed=0, fold=0, eval_test=False, synthetic=True, fixed_epochs=2, patience=0, log=lambda s: None,
                  n_synthetic=30, d=16, K=8, K_w=2, batch_drives=8, K_eval=12, K_w_eval=2, eval_bags=16, out_dir=exp / "recipe" / "results")
    assert res2["size"]["epochs_run"] == 2 and res2["size"]["best_epoch"] == 2 and res2["size"]["cfg"]["fixed_epochs"] == 2
