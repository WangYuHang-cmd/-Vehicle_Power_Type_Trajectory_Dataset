"""experiments/: the Mamba-2 chunked scan against a naive recurrence, model shapes, heads (CORAL, multi-head, domain),
auxiliary targets, sampling weights, augmentation, internal folds (group rule, fleet rule, the built fold files),
aggregation / unit-level protocols, losses (windows without a class label), decision rules, run paths, features and
metrics, plus two short training smokes on the real tensor cache when it is present (a fold run and a final run).

The torch-dependent tests are skipped in an interpreter without torch (the build venv); run them with the
miniconda interpreter (`$CPY -m pytest tests/experiments_tests.py`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from vtt import config as C  # noqa: E402
from experiments import internal_folds as IF  # noqa: E402
from experiments import models as M  # noqa: E402
from experiments.data import (BINARY_CHANNELS, FLIP_CHANNELS, Augment, Preprocess, WindowData, channel_keep, iterate_batches)  # noqa: E402
from experiments.features_baseline import STATS, window_features  # noqa: E402
from experiments.train import (EXP_ROOT, NOTEST_ROOT, PROTOCOLS, RECIPE_ROOT, UNIT_PROTOCOLS, GroupDRO, aggregate_probs,  # noqa: E402
                               balanced_accuracy, class_probs_from_regression, confusion, decide, drive_level, macro_f1, metrics,
                               multitask_loss, run_paths, unit_level, unit_protocols)

CHANNELS = list(C.OFFICIAL_INPUTS)
DATA_ROOT = Path(C.DATA_ROOT)
HAVE_CACHE = (DATA_ROOT / "work" / "tensors" / "X_f16.npy").exists() and (DATA_ROOT / "build" / "index" / "drives.parquet").exists()


# ----------------------------------------------------------------------------- helpers
def toy_window_data(win: pd.DataFrame, channels=CHANNELS) -> WindowData:
    """A WindowData shell around a hand-made windows frame (no tensors)."""
    d = WindowData.__new__(WindowData)
    d.win = win.reset_index(drop=True)
    d.channels = list(channels)
    d.model_codes = {m: i for i, m in enumerate(sorted(d.win["model"].dropna().astype(str).unique()))}
    return d


def toy_drives() -> pd.DataFrame:
    """Groups g1..g4; g3 drove models A and C (shared group), model C is in the pool, model T is the test split."""
    rows = [("d1", "g1", "A", "train"), ("d2", "g1", "A", "train"), ("d3", "g2", "B", "validation"), ("d4", "g3", "A", "train"),
            ("d5", "g3", "C", "train"), ("d6", "g4", "C", "validation"), ("d7", "g5", "T", "test"), ("d8", "g6", "B", "excluded")]
    return pd.DataFrame(rows, columns=["drive_id", "group_id", "model", "split_platform"])


def old_drive_level(win, idx, probs, y, classes):
    """The pre-recipe implementation, kept verbatim as the reference."""
    df = pd.DataFrame(probs, columns=[f"p{i}" for i in range(len(classes))])
    df["drive_id"] = win["drive_id"].to_numpy(object)[idx]
    df["y"] = y
    g = df.groupby("drive_id")
    p = g[[f"p{i}" for i in range(len(classes))]].mean().to_numpy()
    yt = g["y"].first().to_numpy()
    m = metrics(yt, p.argmax(1), classes)
    m["n_drives"] = int(len(yt))
    return m


# ----------------------------------------------------------------------------- Mamba pieces and model shapes
def test_ssd_chunked_matches_sequential_recurrence():
    torch.manual_seed(0)
    b, l, h, p, n = 2, 30, 3, 4, 5
    X = torch.randn(b, l, h, p)
    A = -torch.rand(b, l, h) * 0.8
    B = torch.randn(b, l, h, n)
    Cm = torch.randn(b, l, h, n)
    Y = M.ssd_chunked(X, A, B, Cm, block=10)
    state = torch.zeros(b, h, p, n)
    ys = []
    for t in range(l):
        state = torch.exp(A[:, t])[..., None, None] * state + X[:, t][..., None] * B[:, t][:, :, None, :]
        ys.append((state * Cm[:, t][:, :, None, :]).sum(-1))
    Yn = torch.stack(ys, 1)
    assert float((Y - Yn).abs().max()) < 1e-4


def test_segsum_is_segment_sum_with_minus_inf_above_diagonal():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    s = M.segsum(x)[0]
    assert s[0, 0] == 0 and s[1, 0] == 2 and s[2, 0] == 5 and s[2, 1] == 3 and s[1, 1] == 0
    assert torch.isinf(s[0, 1]) and s[0, 1] < 0


@pytest.mark.parametrize("name", ["resnet", "gru", "transformer", "mamba2", "mamba1"])
def test_models_forward_shape_cpu(name):
    torch.manual_seed(0)
    m = M.build_model(name, in_ch=20, n_classes=3, d=32, n_layers=1)
    x = torch.randn(2, 600, 20)
    out = m(x)
    assert out.shape == (2, 3) and torch.isfinite(out).all()
    assert M.n_params(m) > 0 and m.head.fc2.out_features == 3 and m.n_classes == 3
    assert m.heads is None and m.domain is None
    out.sum().backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad and p.numel() > 0]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)   # every parameter takes part in the forward pass


@pytest.mark.parametrize("name", ["resnet", "gru", "transformer", "mamba2"])
def test_forward_all_returns_requested_heads_and_matches_forward(name):
    torch.manual_seed(0)
    heads = {"coral": 3, "fp": 1, "mass": 1, "pt4": 4, "dom": 5}
    m = M.build_model(name, in_ch=20, n_classes=3, d=32, n_layers=1, heads=heads).eval()
    x = torch.randn(2, 600, 20)
    out = m.forward_all(x)
    assert set(out) == {"cls", "emb", "coral", "fp", "mass", "pt4", "dom"}
    assert out["cls"].shape == (2, 3) and out["coral"].shape == (2, 2) and out["fp"].shape == (2, 1) and out["pt4"].shape == (2, 4) and out["dom"].shape == (2, 5)
    assert torch.allclose(out["cls"], m(x))                       # forward(x) stays the plain classifier
    assert torch.allclose(out["emb"], m.embed(x)) and out["emb"].shape[1] == m.head.fc1.out_features
    # the hook point of the window embeddings: Head.norm receives the pooled (mean + max) feature
    seen = {}
    h = m.head.norm.register_forward_hook(lambda mod, inp, o: seen.setdefault("in", inp[0]))
    m(x)
    h.remove()
    assert seen["in"].shape == (2, 2 * m.head.fc1.in_features // 2)
    with pytest.raises(ValueError):
        M.build_model(name, 20, 3, d=32, n_layers=1, heads={"bogus": 1})


def test_pool_and_multihead_shapes():
    torch.manual_seed(0)
    pool = M.Pool(16, 8, dropout=0.0)
    z = pool(torch.randn(3, 10, 16))
    assert z.shape == (3, 8)
    mh = M.MultiHead(8, {"fp": 1, "coral": 4, "pt4": 4})
    out = mh(z)
    assert out["fp"].shape == (3, 1) and out["coral"].shape == (3, 3) and out["pt4"].shape == (3, 4)
    assert isinstance(mh.heads["coral"], M.CoralHead)


# ----------------------------------------------------------------------------- CORAL and gradient reversal
def test_coral_rank_consistent_and_prediction():
    torch.manual_seed(0)
    head = M.CoralHead(6, 4)
    with torch.no_grad():
        head.delta.copy_(torch.tensor([0.7, -0.2]))
        head.b0.fill_(0.3)
    z = torch.randn(50, 6) * 3
    logits = head(z)
    assert logits.shape == (50, 3)
    s = torch.sigmoid(logits)
    assert (s[:, 1:] <= s[:, :-1] + 1e-6).all()                   # P(y > j) non-increasing in j for every input
    b = head.biases().detach()
    assert (b[1:] < b[:-1]).all() and float(b[0]) == pytest.approx(0.3)
    p = M.coral_predict(logits)
    assert p.shape == (50, 4) and torch.allclose(p.sum(1), torch.ones(50), atol=1e-5) and (p >= 0).all()
    for j in range(3):                                              # P(y > j) is recovered from the class probabilities
        assert torch.allclose(p[:, j + 1:].sum(1), s[:, j], atol=1e-5)
    clear = M.coral_predict(torch.tensor([[3.0, 2.0, 1.0], [3.0, -2.0, -4.0], [-3.0, -4.0, -5.0]]))
    assert clear.argmax(1).tolist() == [3, 1, 0]                     # clear margins: the class is the number of passed thresholds
    t = M.coral_targets(torch.tensor([0, 1, 3]), 4)
    assert t.tolist() == [[0, 0, 0], [1, 0, 0], [1, 1, 1]]
    with pytest.raises(ValueError):
        M.CoralHead(4, 1)


def test_grad_reverse_and_conditional_domain_head():
    x = torch.randn(4, 3, requires_grad=True)
    y = M.grad_reverse(x, 0.5)
    assert torch.equal(y, x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.5))
    head = M.ConditionalDomainHead(3, n_models=6, n_classes=2)
    z = torch.randn(4, 3, requires_grad=True)
    out = head(z, torch.randn(4, 2), lam=1.0)
    assert out.shape == (4, 6)
    out.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


# ----------------------------------------------------------------------------- data: aux targets, sampling weights, augmentation
def test_aux_targets_masks_and_codes():
    win = pd.DataFrame(dict(footprint_m2=[4.5, 5.0, np.nan, 4.0, 6.0], wheelbase_mm=[2800, 3000, 2700, 406, 3100],
                            curb_mass_kg=[1500, 2000, 1700, 9, np.nan], powertrain=["ICE", "BEV", None, "HEV", "PHEV"],
                            model=["A", "B", "A", None, "C"], das_any_frac=[0.0, 0.5, np.nan, 1.0, 0.2]))
    d = toy_window_data(win)
    a = d.aux_targets(np.arange(5))
    assert a["fp_mask"].tolist() == [True, True, False, False, True]          # NaN footprint and a 406 mm wheelbase are masked
    assert a["mass_mask"].tolist() == [True, True, True, False, False]        # 9 kg and NaN are masked
    assert a["log_fp"][~a["fp_mask"]].tolist() == [0.0, 0.0] and abs(float(a["log_fp"][a["fp_mask"]].mean())) < 1e-6
    assert a["pt4"].tolist() == [0, 3, -1, 1, 2] and a["model_id"].tolist() == [0, 1, 0, -1, 2]
    assert a["das_any_frac"].tolist() == pytest.approx([0.0, 0.5, 0.0, 1.0, 0.2])
    b = d.aux_targets(np.array([1, 4]), stats=a["stats"])                    # reusing stats keeps the standardisation
    assert b["log_fp"].tolist() == pytest.approx(a["log_fp"][[1, 4]].tolist())


def test_sampling_weights_class_sums_cap_and_unit_balance():
    rng = np.random.default_rng(0)
    y = np.array([0] * 90 + [1] * 10 + [2] * 50)
    models = np.array(["big"] * 81 + ["small"] * 9 + ["m1"] * 5 + ["m2"] * 5 + ["x"] * 50, dtype=object)
    win = pd.DataFrame(dict(model=models, das_any_frac=rng.random(len(y))))
    d = toy_window_data(win)
    idx = np.arange(len(y))
    base = d.class_sampling_weights(y, 3)
    assert np.allclose(d.sampling_weights(idx, y, 3, unit=None), base)          # unit=None reproduces the class rule
    w = d.sampling_weights(idx, y, 3, unit="model", unit_power=0.5)
    for c in range(3):                                                           # class totals are kept
        assert w[y == c].sum() == pytest.approx(base[y == c].sum())
    share_small = w[(y == 0) & (models == "small")].sum() / w[y == 0].sum()
    assert share_small == pytest.approx(np.sqrt(9) / (np.sqrt(81) + np.sqrt(9)))   # hours^0.5 shares inside the class
    assert share_small > 9 / 90
    ratio = w[(y == 0) & (models == "small")][0] / base[y == 0][0]
    assert ratio == pytest.approx((np.sqrt(9) / (np.sqrt(81) + np.sqrt(9))) / (9 / 90))
    w_cap = d.sampling_weights(idx, y, 3, unit="model", unit_power=0.0, cap=2.0)   # power 0 -> equal model shares, cap 2
    base2 = d.class_sampling_weights(y, 3, cap=2.0)                                  # the cap also bounds the class weight
    assert w_cap[(y == 0) & (models == "small")][0] / base2[y == 0][0] <= 2.0 + 1e-9   # the per-window ratio is capped
    assert w_cap[(y == 0) & (models == "small")][0] / base2[y == 0][0] == pytest.approx(2.0)
    for c in (1, 2):                                                             # classes without a binding model cap keep their totals
        assert w_cap[y == c].sum() == pytest.approx(base2[y == c].sum())
    assert w_cap[y == 0].sum() <= base2[y == 0].sum() + 1e-9
    w_das = d.sampling_weights(idx, y, 3, unit=None, das_beta=0.5)
    assert np.allclose(w_das, base * (0.5 + 0.5 * (1 - win["das_any_frac"].to_numpy())))
    y_u = np.array([0, 0, 1, -1, -1])                                            # windows without a class label (hybrids)
    d_u = toy_window_data(pd.DataFrame(dict(model=["a", "a", "b", "b", "a"], das_any_frac=[0.0] * 5)))
    w_u = d_u.class_sampling_weights(y_u, 2)
    assert np.allclose(w_u[:3], [(3 / 4) ** 0.5, (3 / 4) ** 0.5, (3 / 2) ** 0.5]) and np.allclose(w_u[3:], w_u[:3].mean())
    w_um = d_u.sampling_weights(np.arange(5), y_u, 2, unit="model")
    assert np.allclose(w_um[3:], w_u[3:]) and w_um[:3].sum() == pytest.approx(w_u[:3].sum())


def test_extra_mask_selects_known_labels_outside_the_task_classes():
    win = pd.DataFrame(dict(split_platform=["train"] * 5 + ["test"], eligible=[True, True, False, True, True, True],
                            task_bev_ice_eligible=[True, False, False, False, False, False], powertrain=["BEV", "HEV", "PHEV", "PHEV", None, "HEV"],
                            task_size_eligible=[True] * 6, size_class=["SMALL", "MID", "MID", None, "LARGE", "MID"], model=["A"] * 6))
    d = toy_window_data(win)
    assert d.extra_mask("bev_ice", "split_platform", "train").tolist() == [False, True, False, True, False, False]   # HEV / PHEV, eligible, train
    assert d.extra_indices("bev_ice", "split_platform", "test").tolist() == [5]
    assert not d.extra_mask("size", "split_platform", "train").any()           # every known size label is a class


def test_flip_augmentation_flips_exactly_six_channels_and_keeps_labels():
    torch.manual_seed(0)
    aug = Augment(CHANNELS, p_flip=1.0, p_mask=0.0, p_chdrop=0.0, noise=0.0).train()
    x = torch.randn(4, 600, len(CHANNELS))
    y = torch.tensor([0, 1, 2, 1])
    out = aug(x, y=y)
    flip = [CHANNELS.index(c) for c in FLIP_CHANNELS]
    keep = [i for i in range(len(CHANNELS)) if i not in flip]
    assert len(flip) == 6
    assert torch.equal(out[:, :, flip], -x[:, :, flip]) and torch.equal(out[:, :, keep], x[:, :, keep])
    assert torch.equal(y, torch.tensor([0, 1, 2, 1]))
    aug.eval()
    assert torch.equal(aug(x), x)                                    # identity outside training


def test_mask_chdrop_noise_cutmix_augmentations():
    torch.manual_seed(0)
    mean = np.zeros(len(CHANNELS), np.float32); std = np.ones(len(CHANNELS), np.float32)
    mean[CHANNELS.index("lead_valid")] = 0.6; std[CHANNELS.index("lead_valid")] = 0.5
    x = torch.ones(3, 600, len(CHANNELS))
    aug = Augment(CHANNELS, mean, std, p_flip=0.0, p_mask=1.0, mask_s=(3.0, 3.0), p_chdrop=0.0, noise=0.0).train()
    out = aug(x)
    lv = CHANNELS.index("lead_valid")
    other = [i for i in range(len(CHANNELS)) if i != lv]
    zeroed = (out[:, :, other] == 0).all(-1)
    assert zeroed.sum(1).tolist() == [30, 30, 30]                   # exactly one 3-s span per window
    assert torch.allclose(out[:, :, lv][zeroed], torch.full((90,), (0.0 - 0.6) / 0.5))   # lead_valid -> standardised 0
    aug = Augment(CHANNELS, mean, std, p_flip=0.0, p_mask=0.0, p_chdrop=1.0, noise=0.0).train()
    out = aug(x)
    dropped = (out == 0).all(1).sum(1) + (out[:, :, lv] == (0.0 - 0.6) / 0.5).all(1).int()
    assert set(dropped.tolist()) <= {7, 1, 2}                      # whole lead group (7), torque (1) or pedals (2)
    aug = Augment(CHANNELS, p_flip=0.0, p_mask=0.0, p_chdrop=0.0, noise=0.1).train()
    out = aug(x)
    binary = [CHANNELS.index(c) for c in BINARY_CHANNELS]
    assert torch.equal(out[:, :, binary], x[:, :, binary]) and not torch.equal(out, x)
    aug = Augment(CHANNELS, p_flip=0.0, p_mask=0.0, p_chdrop=0.0, noise=0.0, p_cutmix=1.0, mask_s=(5.0, 5.0)).train()
    xa = torch.stack([torch.full((600, len(CHANNELS)), float(i)) for i in range(4)])
    out = aug(xa, y=torch.tensor([0, 0, 1, 1]), model_id=torch.tensor([1, 2, 3, 3]))
    assert (out[0] == 1.0).any() and (out[1] == 0.0).any()         # same-class partners of another model were spliced in
    assert torch.equal(out[2], xa[2]) and torch.equal(out[3], xa[3])   # same model -> no partner -> unchanged
    out = aug(xa, y=torch.tensor([-1, -1, 1, 1]), model_id=torch.tensor([1, 2, 3, 4]))
    assert torch.equal(out[0], xa[0]) and torch.equal(out[1], xa[1])   # windows without a class label are neither cut nor partners
    assert (out[2] == 3.0).any() and (out[3] == 2.0).any()


# ----------------------------------------------------------------------------- opt-in channel mask (reduced-input variants)
V3 = ["v_mps", "a_mps2", "standstill"]


def test_channel_keep_mask_names_and_errors():
    keep = channel_keep(CHANNELS, V3)
    assert keep.dtype == bool and keep.sum() == 3 and [c for c, k in zip(CHANNELS, keep) if k] == V3
    assert np.array_equal(channel_keep(CHANNELS, "v_mps,a_mps2,standstill,"), keep)      # comma string, trailing comma ignored
    assert np.array_equal(channel_keep(CHANNELS, V3 + ["v_mps"]), keep)                  # duplicates ignored
    with pytest.raises(ValueError):
        channel_keep(CHANNELS, ["v_mps", "speed"])                                      # unknown name
    with pytest.raises(ValueError):
        channel_keep(CHANNELS, [])                                                      # nothing kept


def test_preprocess_channel_mask_makes_masked_channels_missing_and_default_unchanged():
    torch.manual_seed(0)
    C_ = len(CHANNELS)
    mean = np.random.default_rng(0).normal(size=C_).astype(np.float32)
    std = np.random.default_rng(1).uniform(0.5, 2.0, size=C_).astype(np.float32)
    lo = np.full(C_, -3.0, np.float32); hi = np.full(C_, 3.0, np.float32)
    x = torch.randn(2, 600, C_) * 2
    x[0, :10, 0] = float("nan")                                                        # NaN input stays the missing value 0
    ref = Preprocess(mean, std, lo, hi)(x)
    manual = torch.nan_to_num((torch.clamp(x, -3.0, 3.0) - torch.as_tensor(mean)) / torch.as_tensor(std), nan=0.0)
    assert torch.allclose(ref, manual) and (ref[0, :10, 0] == 0).all()                  # keep=None: the published behaviour
    keep = channel_keep(CHANNELS, V3)
    out = Preprocess(mean, std, lo, hi, keep=keep)(x)
    kept = torch.as_tensor(keep)
    assert (out[:, :, ~kept] == 0).all()                                               # masked channels -> exactly 0 on every sample
    assert torch.equal(out[:, :, kept], ref[:, :, kept])                               # kept channels untouched
    assert out.shape == ref.shape and torch.isfinite(out).all()


def test_augment_keep_mask_resets_masked_channels_and_default_unchanged():
    torch.manual_seed(0)
    C_ = len(CHANNELS)
    mean = np.zeros(C_, np.float32); std = np.ones(C_, np.float32)
    mean[CHANNELS.index("lead_valid")] = 0.6; std[CHANNELS.index("lead_valid")] = 0.5
    keep = channel_keep(CHANNELS, ["v_mps", "a_mps2", "standstill", "das_enabled", "steer_angle_deg", "yaw_rate_radps"])
    kept = torch.as_tensor(keep)
    x = torch.ones(8, 600, C_)
    x[:, :, ~kept] = 0.0                                                               # what Preprocess(keep) hands over
    aug = Augment(CHANNELS, mean, std, p_flip=0.5, p_mask=1.0, mask_s=(3.0, 3.0), p_chdrop=1.0, noise=0.1, keep=keep).train()
    out = aug(x)
    assert (out[:, :, ~kept] == 0).all()                                               # no noise, no lead_valid fill-in on masked channels
    assert not torch.equal(out[:, :, kept], x[:, :, kept])                             # kept channels are still augmented
    torch.manual_seed(0)
    ref = Augment(CHANNELS, mean, std, p_flip=0.5, p_mask=1.0, mask_s=(3.0, 3.0), p_chdrop=1.0, noise=0.1).train()(x)
    assert (ref[:, :, ~kept] != 0).any()                                               # keep=None: masked-by-hand channels get noise as before
    lv = CHANNELS.index("lead_valid")
    assert ((ref[:, :, lv] == (0.0 - 0.6) / 0.5).sum(1) >= 30).all()                   # ... and the lead_valid fill-in (span or group drop)


def test_run_paths_tag_prefix_replaces_the_split_column():
    tag, key, out, ck = run_paths("size", "split_platform", "gru", 1, recipe_id="V3", tag_prefix="transfer")
    assert tag == "transfer__size__gru__V3__s1" and key == "gru__V3" and out == EXP_ROOT / "results" and ck == EXP_ROOT / "ckpt"
    assert run_paths("size", "split_platform", "gru", 1)[0] == "split_platform__size__gru__s1"       # default unchanged
    assert run_paths("size", "split_platform", "gru", 1, recipe_id="V3", fold=0, eval_test=False, tag_prefix="transfer")[0] == "fold0__size__gru__V3__s1"
    for bad in ("", "a__b", "x/y"):
        with pytest.raises(ValueError):
            run_paths("size", "split_platform", "gru", 1, tag_prefix=bad)


# ----------------------------------------------------------------------------- internal folds
def test_fold_drive_sets_disjoint_and_group_exclusion():
    drives = toy_drives()
    folds = {"split_col": "split_platform", "folds": [{"fold": 0, "heldout_models": ["C"]}]}
    train, val = IF.fold_drive_sets(drives, folds, 0)
    assert val == {"d5", "d6"}                                     # pool drives of the held-out model
    assert "d4" not in train                                       # g3 also drove C -> excluded from training
    assert train == {"d1", "d2", "d3"} and not (train & val)
    assert "d7" not in train and "d7" not in val and "d8" not in train     # test / excluded drives are outside the pool
    win = pd.DataFrame(dict(drive_id=["d1", "d4", "d5", "d6", "d7"], group_id=["g1", "g3", "g3", "g4", "g5"],
                            model=["A", "A", "C", "C", "T"], split_platform=["train", "train", "train", "validation", "test"]))
    tr_m, va_m = IF.fold_window_masks(win, folds, 0, drives_df=drives)
    assert tr_m.tolist() == [True, False, False, False, False] and va_m.tolist() == [False, False, True, True, False]
    tr2, va2 = IF.fold_window_masks(win, folds, 0)                 # derived from the windows table alone
    assert tr2.tolist() == tr_m.tolist() and va2.tolist() == va_m.tolist()
    # fleet rule: a group spanning >= fleet_min_models car models stays on the training side
    rows = [(f"f{i}", "fleet", m, "train") for i, m in enumerate(["A", "B", "C", "D", "E", "C"])]      # 5 car models, two drives of C
    drives2 = pd.concat([drives, pd.DataFrame(rows, columns=drives.columns)], ignore_index=True)
    assert IF.fleet_groups(drives2, 5) == {"fleet"} and IF.fleet_groups(drives2, 6) == set() and IF.fleet_groups(drives2, 0) == set()
    train5, val5 = IF.fold_drive_sets(drives2, dict(folds, params={"fleet_min_models": 5}), 0)
    assert val5 == {"d5", "d6"} and {"f0", "f1", "f3", "f4"} <= train5           # the fleet's drives of C are dropped, its others train
    assert not ({"f2", "f5", "d4"} & train5) and not ({"f2", "f5"} & val5)
    assert IF.fold_drive_sets(drives2, folds, 0) == (train5, val5)               # the PARAMS default applies without params
    train_lit, val_lit = IF.fold_drive_sets(drives2, dict(folds, params={"fleet_min_models": 0}), 0)   # literal group rule
    assert val_lit == {"d5", "d6", "f2", "f5"} and train_lit == {"d1", "d2", "d3"}
    assert IF.fold_drive_sets(drives2, dict(folds, params={"fleet_min_models": 6}), 0) == (train_lit, val_lit)


def test_model_table_and_assign_folds_on_toy_windows():
    rng = np.random.default_rng(1)
    rows = []
    spec = {"Dom": ("LARGE", "ICE", 6.0, 400, 5), "Large2": ("LARGE", "ICE", 4.9, 150, 2), "Large4": ("LARGE", "ICE", 5.5, 150, 2),
            "Mid1": ("MID", "ICE", 4.5, 150, 3), "Mid3": ("MID", "ICE", 4.65, 150, 2), "OneGroup": ("MID", "ICE", 4.6, 150, 1),
            "Test": ("MID", "ICE", 4.7, 150, 3), "Mid2": ("MID", "BEV", 4.85, 150, 3), "Mid4": ("MID", "BEV", 4.7, 150, 2),
            "Mid5": ("MID", "BEV", 4.6, 150, 2), "Small1": ("SMALL", "ICE", 4.3, 150, 2), "Small3": ("SMALL", "ICE", 4.4, 150, 2),
            "Small5": ("SMALL", "ICE", 4.1, 150, 2), "Tiny": ("SMALL", "ICE", 4.1, 50, 2), "Small2": ("SMALL", "BEV", 3.9, 150, 2),
            "Small4": ("SMALL", "BEV", 4.0, 150, 2), "Small6": ("SMALL", "BEV", 4.2, 150, 2), "Large3": ("LARGE", "BEV", 5.2, 150, 2),
            "Large5": ("LARGE", "BEV", 5.3, 150, 2), "Large6": ("LARGE", "BEV", 5.6, 150, 2)}
    for m, (sz, pt, fp, n, ng) in spec.items():
        for i in range(n):
            rows.append(dict(window_id=f"{m}_{i}", drive_id=f"{m}_d{i % (3 * ng)}", group_id=f"{m}_g{i % ng}", model=m, size_class=sz,
                             powertrain=pt, footprint_m2=fp, eligible=True, task_size_eligible=True, task_bev_ice_eligible=True,
                             split_platform="test" if m == "Test" else ("validation" if i % 5 == 0 else "train")))
    win = pd.DataFrame(rows)
    params = dict(IF.PARAMS, n_folds=3, min_boundary_models=1)
    table = IF.model_table("size", win, params)
    cand = set(table.loc[table["candidate"], "model"])
    assert "Dom" not in cand and "Tiny" not in cand and "OneGroup" not in cand and "Test" not in cand   # share / windows / groups / pool
    assert cand == set(spec) - {"Dom", "Tiny", "OneGroup", "Test"} and len(cand) == 16
    assert float(table.set_index("model").loc["Dom", "cell_share"]) == pytest.approx(400 / 700)
    assert set(table.loc[table["boundary"], "model"]) == {"Mid1", "Mid2", "Small1", "Small3", "Large2"}
    folds = IF.assign_folds(table, seed=0, params=params)
    assert sorted(sum(folds, [])) == sorted(cand)                  # every candidate in exactly one fold
    assert max(len(f) for f in folds) - min(len(f) for f in folds) <= 2
    assert folds == IF.assign_folds(table, seed=0, params=params)  # deterministic
    with pytest.raises(ValueError):                                # 5 near-threshold models cannot give 3 folds >= 2 each
        IF.assign_folds(table, seed=0, params=dict(params, min_boundary_models=2))
    drives = win.drop_duplicates("drive_id")[["drive_id", "group_id", "model", "split_platform"]]
    res, _ = IF.build_folds("size", drives, win, seed=0, params=params)
    assert res["n_folds"] == 3 and res["split_col"] == "split_platform" and len(res["folds"]) == 3
    for f in res["folds"]:
        assert f["n_heldout_drives"] > 0 and f["n_train_drives"] > 0 and f["n_boundary_models"] >= 1
        assert f["n_dropped_drives"] >= 0 and f["n_fleet_dropped_windows"] == 0 and f["n_val_windows"] + f["n_train_windows"] + f["n_dropped_windows"] == res["params"]["n_pool_windows"]
    p = res["params"]
    assert p["fleet_min_models"] == 5 and p["n_fleet_pool_groups"] == 0 and p["never_trained_share"] == p["never_trained_share_literal_rule"]
    assert p["n_pool_windows_never_trained"] == p["n_pool_windows_never_trained_literal_rule"] and p["n_pool_windows_in_no_fold"] >= 0
    md = IF.render_md({"size": (res, table)})
    assert "Fold 0" in md and "Never held out" in md and "|BEV" not in md.replace("{BEV", "")


@pytest.mark.skipif(not (DATA_ROOT / "experiments" / "internal_folds_size.json").exists(), reason="fold file not built")
def test_built_folds_satisfy_the_contract():
    drives = pd.read_parquet(DATA_ROOT / "build" / "index" / "drives.parquet", columns=["drive_id", "group_id", "model", "split_platform"])
    win = pd.read_parquet(DATA_ROOT / "build" / "index" / "windows.parquet",
                          columns=["drive_id", "group_id", "model", "split_platform", "eligible", "task_size_eligible", "task_bev_ice_eligible",
                                   "size_class", "powertrain"])
    for task in ("size", "bev_ice"):
        f = IF.load_folds(task)
        assert f["task"] == task and f["split_col"] == "split_platform" and f["n_folds"] == 3 and len(f["folds"]) == 3
        held = [set(k["heldout_models"]) for k in f["folds"]]
        assert not (held[0] & held[1]) and not (held[0] & held[2]) and not (held[1] & held[2])
        assert all(k["n_boundary_models"] >= 2 and 10 <= len(k["heldout_models"]) <= 25 for k in f["folds"])
        assert not any("Rivian" in m or "Civic" in m for m in set().union(*held))
        p = f["params"]
        assert p["fleet_min_models"] == 5 and p["n_fleet_pool_groups"] == 1 and p["n_fleet_pool_windows"] > 0
        pw = IF.pool_windows(task, win)
        pw_drive = pw["drive_id"].astype(str)
        assert len(pw) == p["n_pool_windows"]
        group_of = drives.set_index(drives["drive_id"].astype(str))["group_id"].astype(str)
        model_of = drives.set_index(drives["drive_id"].astype(str))["model"]
        trained = np.zeros(len(pw), dtype=bool)
        used = np.zeros(len(pw), dtype=bool)
        for k in range(3):
            tr, va = IF.fold_drive_sets(drives, f, k)
            assert tr and va and not (tr & va)
            assert len(va) == f["folds"][k]["n_heldout_drives"] and len(tr) == f["folds"][k]["n_train_drives"]
            assert not (set(group_of[list(tr)]) & set(group_of[list(va)]))                 # no identity group on both sides
            assert set(model_of[list(va)]) <= held[k] and not (set(model_of[list(tr)]) & held[k])
            in_tr, in_va = pw_drive.isin(tr).to_numpy(), pw_drive.isin(va).to_numpy()
            assert int(in_tr.sum()) == f["folds"][k]["n_train_windows"] and int(in_va.sum()) == f["folds"][k]["n_val_windows"]
            assert int((~in_tr & ~in_va).sum()) == f["folds"][k]["n_dropped_windows"]
            trained |= in_tr
            used |= in_tr | in_va
        assert float((~trained).mean()) == pytest.approx(p["never_trained_share"]) and p["never_trained_share"] <= 0.02   # fleet rule
        assert p["never_trained_share_literal_rule"] > 0.10 and p["n_never_trained_literal_rule_in_fleet"] == p["n_pool_windows_never_trained_literal_rule"]
        assert int((~used).sum()) == p["n_pool_windows_in_no_fold"]


# ----------------------------------------------------------------------------- aggregation, unit protocols, losses, decisions
def test_drive_level_mean_equals_old_behaviour_and_other_rules_run():
    rng = np.random.default_rng(0)
    n, k = 300, 3
    probs = rng.dirichlet(np.ones(k), size=n).astype(np.float32)
    drives = rng.choice([f"drive_{i:06d}" for i in range(40)], size=n)
    y_drive = {d: i % k for i, d in enumerate(sorted(set(drives)))}
    y = np.array([y_drive[d] for d in drives])
    win = pd.DataFrame(dict(drive_id=drives, vehicle_id=[d[:-1] for d in drives], model=[f"m{y_drive[d] % 4}" for d in drives]))
    idx = np.arange(n)
    classes = ("A", "B", "C")
    new = drive_level(win, idx, probs, y, classes)
    old = old_drive_level(win, idx, probs, y, classes)
    assert new == old
    assert drive_level(win, idx, probs, y, classes, how="mean") == old
    uid, p = aggregate_probs(probs, drives, "mean")
    uid_w, p_w = aggregate_probs(probs, drives, "mean", weights=np.ones(n))
    assert uid.tolist() == uid_w.tolist() and np.allclose(p, p_w, atol=1e-6)
    for how in ("logodds", "median", "vote"):
        u2, p2 = aggregate_probs(probs, drives, how)
        assert u2.tolist() == uid.tolist() and p2.shape == p.shape and (p2 >= 0).all() and (p2 <= 1).all()
        assert how == "median" or np.allclose(p2.sum(1), 1.0)       # per-class medians need not sum to one
        assert drive_level(win, idx, probs, y, classes, how=how)["n_drives"] == len(uid)
    w = rng.random(n)
    _, p_w2 = aggregate_probs(probs, drives, "mean", weights=w)
    d0 = drives == uid[0]
    assert np.allclose(p_w2[0], (probs[d0] * w[d0, None]).sum(0) / w[d0].sum())
    _, pv = aggregate_probs(np.array([[0.6, 0.4], [0.6, 0.4], [0.1, 0.9]]), np.array(["a", "a", "a"], dtype=object), "vote")
    assert pv.tolist() == [[2 / 3, 1 / 3]]
    with pytest.raises(ValueError):
        aggregate_probs(probs, drives, "max")


def test_unit_level_counts_and_vehicle_multi():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.3, 0.7], [0.8, 0.2]], dtype=np.float32)
    win = pd.DataFrame(dict(drive_id=["d1", "d1", "d2", "d3", "d4"], vehicle_id=["v1", "v1", "v1", "v2", "v3"], model=["A", "A", "A", "B", "B"]))
    y = np.array([0, 0, 0, 1, 1])
    idx = np.arange(5)
    m = unit_level(win, idx, probs, y, ("X", "Y"), "drive_id", all_units=np.array(["d1", "d2", "d3", "d4", "d9"]))
    assert m["n_units"] == 4 and m["n_windows_used"] == 5 and m["n_models"] == 2 and m["n_units_without_windows"] == 1
    assert m["confusion"] == [[1, 1], [1, 1]]                      # d1 mean (0.55, 0.45) -> X; d2 -> Y; d3 -> Y; d4 -> X
    v = unit_level(win, idx, probs, y, ("X", "Y"), "vehicle_id")
    assert v["n_units"] == 3 and v["confusion"] == [[1, 0], [1, 1]]     # v1 mean (0.5, 0.5) -> tie -> earlier class X

    class Shell:
        pass
    data = Shell()
    data.win = win
    data.drives = pd.DataFrame(dict(drive_id=["d1", "d2", "d3", "d4", "d5"], split_platform=["test"] * 5, task_size_eligible=[True] * 5,
                                    size_class=["SMALL", "SMALL", "MID", "MID", "LARGE"]))
    human = np.array([0, 2, 3])
    out = unit_protocols(data, "size", "split_platform", idx, probs, y, human, probs[human], y[human], ("X", "Y"))
    assert set(out) == {"drive_full", "drive_human_only", "vehicle_full", "vehicle_full_multi"}
    assert out["drive_full"]["n_units_without_windows"] == 1 and out["drive_human_only"]["n_units"] == 3
    assert out["vehicle_full_multi"]["n_units"] == 1 and out["vehicle_full_multi"]["n_windows_used"] == 3     # only v1 has 2 drives


def test_multitask_loss_parts_masks_and_sample_weights():
    torch.manual_seed(0)
    n, k = 6, 3
    out = {"cls": torch.randn(n, k), "coral": torch.randn(n, k - 1), "fp": torch.randn(n, 1), "mass": torch.randn(n, 1),
           "pt4": torch.randn(n, 4), "dom": torch.randn(n, 5)}
    targets = {"y": torch.tensor([0, 1, 2, 0, 1, 2]), "log_fp": torch.randn(n), "fp_mask": torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool),
               "log_mass": torch.randn(n), "mass_mask": torch.zeros(n, dtype=torch.bool), "pt4": torch.tensor([0, 1, 2, 3, -1, -1]),
               "model_id": torch.tensor([0, 1, 2, 3, 4, -1])}
    loss, parts = multitask_loss(out, targets, label_smoothing=0.05)
    assert set(parts) == {"cls", "coral", "fp", "mass", "pt4", "dom", "total"}
    assert parts["mass"] == 0.0                                    # fully masked -> no contribution
    expected = parts["cls"] + 0.5 * parts["coral"] + 1.0 * parts["fp"] + 0.5 * parts["mass"] + 0.3 * parts["pt4"] + 1.0 * parts["dom"]
    assert float(loss) == pytest.approx(expected, rel=1e-5)
    fp_only = torch.nn.functional.smooth_l1_loss(out["fp"].reshape(n)[:3], targets["log_fp"][:3])
    assert parts["fp"] == pytest.approx(float(fp_only), rel=1e-5)
    loss_w, parts_w = multitask_loss(out, targets, weights={"coral": 0.0, "fp": 2.0}, label_smoothing=0.0)
    assert float(loss_w) == pytest.approx(parts_w["cls"] + 2.0 * parts_w["fp"] + 0.5 * parts_w["mass"] + 0.3 * parts_w["pt4"] + parts_w["dom"], rel=1e-5)
    sw = torch.tensor([1.0, 0, 0, 0, 0, 0])
    loss_s, parts_s = multitask_loss({"cls": out["cls"]}, {"y": targets["y"]}, sample_weights=sw)
    assert float(loss_s) == pytest.approx(float(torch.nn.functional.cross_entropy(out["cls"][:1], targets["y"][:1])), rel=1e-5)
    _, _, vec = multitask_loss({"cls": out["cls"]}, {"y": targets["y"]}, return_per_sample=True)
    assert vec.shape == (n,)
    dro = GroupDRO(5, eta=1.0)
    w1 = dro.weights(torch.tensor([1.0, 1.0, 5.0, 5.0, 1.0, 1.0]), torch.tensor([0, 0, 1, 1, 2, -1]))
    assert w1.shape == (n,) and float(w1.mean()) == pytest.approx(1.0) and w1[2] > w1[0]      # the worse group counts more


def test_class_probs_from_regression_and_decide():
    thr = (4.43, 4.88)
    mu = np.log(np.array([4.0, 4.43, 4.65, 4.88, 6.0]))
    p = class_probs_from_regression(mu, sigma=0.05, thresholds=thr)
    assert p.shape == (5, 3) and np.allclose(p.sum(1), 1.0)
    assert p[0].argmax() == 0 and p[2].argmax() == 1 and p[4].argmax() == 2
    assert p[1, 0] == pytest.approx(0.5, abs=1e-6) and p[3, 2] == pytest.approx(0.5, abs=1e-6)   # on a threshold: half / half
    p_wide = class_probs_from_regression(mu, sigma=5.0, thresholds=thr)
    assert np.abs(p_wide - p_wide.mean()).max() < np.abs(p - p.mean()).max()                        # a larger sigma flattens
    assert class_probs_from_regression(np.array([np.nan]), 0.1, thr).tolist() == [[1 / 3, 1 / 3, 1 / 3]]
    pred = {"probs": np.array([[0.2, 0.5, 0.3]], np.float32), "coral": np.array([[0.1, 0.1, 0.8]], np.float32), "fp_mu": np.log(np.array([4.0]))}
    assert decide(pred, "softmax") is pred["probs"] and decide(pred, "coral") is pred["coral"]
    fp = decide(pred, "fp", sigma=0.05)
    assert fp.argmax(1).tolist() == [0]
    avg = decide(pred, "avg", sigma=0.05)
    assert np.allclose(avg, 0.5 * (pred["probs"] + fp)) and np.allclose(avg.sum(1), 1.0)
    with pytest.raises(ValueError):
        decide({"probs": pred["probs"]}, "coral")
    with pytest.raises(ValueError):
        decide({"probs": np.array([[0.4, 0.6]]), "fp_mu": np.array([1.5])}, "fp", sigma=0.1)     # two classes: not the size task
    with pytest.raises(ValueError):
        decide(pred, "majority")
    with pytest.raises(ValueError):
        decide(pred, "fp", sigma=float("nan"))                      # a NaN sigma would silently predict class 0 everywhere
    with pytest.raises(ValueError):
        decide(pred, "avg", sigma=float("inf"))
    assert decide(pred, "fp", sigma=None).shape == (1, 3)           # None keeps the default sigma


def test_multitask_loss_ignores_windows_without_a_class_label():
    torch.manual_seed(0)
    out = {"cls": torch.randn(4, 3), "coral": torch.randn(4, 2), "pt4": torch.randn(4, 4)}
    y = torch.tensor([0, 2, -1, 1])
    targets = {"y": y, "pt4": torch.tensor([0, 3, 1, 2])}
    loss, parts = multitask_loss(out, targets, label_smoothing=0.05)
    lab = y >= 0
    assert parts["cls"] == pytest.approx(float(torch.nn.functional.cross_entropy(out["cls"][lab], y[lab], label_smoothing=0.05)), rel=1e-5)
    coral_ref = torch.nn.functional.binary_cross_entropy_with_logits(out["coral"][lab], M.coral_targets(y[lab], 3))
    assert parts["coral"] == pytest.approx(float(coral_ref), rel=1e-5)
    assert parts["pt4"] == pytest.approx(float(torch.nn.functional.cross_entropy(out["pt4"], targets["pt4"])), rel=1e-5)   # all four count
    _, _, vec = multitask_loss(out, targets, return_per_sample=True)
    assert vec[2] == pytest.approx(0.3 * float(torch.nn.functional.cross_entropy(out["pt4"][2:3], targets["pt4"][2:3])), rel=1e-5)
    full, parts_full = multitask_loss({"cls": out["cls"]}, {"y": torch.tensor([0, 2, 1, 1])})
    assert parts_full["cls"] == pytest.approx(float(torch.nn.functional.cross_entropy(out["cls"], torch.tensor([0, 2, 1, 1]))), rel=1e-6)


def test_run_paths_keep_published_layout_and_route_runs_without_test_split():
    tag, key, out, ck = run_paths("size", "split_platform", "gru", 1)
    assert tag == "split_platform__size__gru__s1" and key == "gru" and out == EXP_ROOT / "results" and ck == EXP_ROOT / "ckpt"
    tag, key, out, ck = run_paths("size", "split_platform", "gru", 1, recipe_id="rev", eval_test=False)
    assert tag == "split_platform__size__gru__rev__s1" and key == "gru__rev" and out == NOTEST_ROOT and ck == EXP_ROOT / "ckpt"
    tag, key, out, ck = run_paths("bev_ice", "split_platform", "resnet", 0, recipe_id="S1", fold=2, eval_test=False)
    assert tag == "fold2__bev_ice__resnet__S1__s0" and key == "resnet__S1" and out == RECIPE_ROOT / "results" and ck == RECIPE_ROOT / "ckpt"
    assert run_paths("size", "split_platform", "gru", 0, eval_test=False, out_dir=Path("x"), ckpt_dir=Path("y"))[2:] == (Path("x"), Path("y"))
    with pytest.raises(ValueError):                                 # RESULTS.md is rendered from experiments/results
        run_paths("size", "split_platform", "gru", 0, eval_test=False, out_dir=EXP_ROOT / "results")


# ----------------------------------------------------------------------------- features, metrics, batches
def test_window_features_nan_aware():
    x = np.full((1, 10, 2), np.nan, dtype=np.float32)
    x[0, :, 0] = np.arange(10, dtype=np.float32)          # full channel
    x[0, :4, 1] = [1.0, 3.0, 2.0, 4.0]                    # 4 finite samples
    f = window_features(x).reshape(2, len(STATS))
    s = dict(zip(STATS, f[0]))
    assert s["mean"] == pytest.approx(4.5) and s["min"] == 0 and s["max"] == 9 and s["finite"] == 1.0 and s["absdiff"] == pytest.approx(1.0)
    t = dict(zip(STATS, f[1]))
    assert t["mean"] == pytest.approx(2.5) and t["min"] == 1 and t["max"] == 4 and t["finite"] == pytest.approx(0.4)
    assert t["q50"] in (2.0, 3.0)
    g = window_features(np.full((1, 10, 1), np.nan, dtype=np.float32)).reshape(len(STATS))
    assert np.isnan(g[0]) and g[-1] == 0.0


def test_metrics_match_definitions():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 1, 2, 0])
    cm = confusion(y, p, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    # F1: class0 tp1 fp1 fn1 -> 0.5; class1 tp2 fp1 fn0 -> 0.8; class2 tp1 fp0 fn1 -> 2/3
    assert macro_f1(y, p, 3) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert balanced_accuracy(y, p, 3) == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    m = metrics(y, p, ("A", "B", "C"))
    assert m["n"] == 6 and m["support"] == {"A": 2, "B": 2, "C": 2} and m["per_class_recall"]["B"] == 1.0
    assert macro_f1(np.array([0, 0]), np.array([0, 1]), 3) == pytest.approx(2 / 3)   # only the supported class counts


def test_iterate_batches_weighted_and_sequential():
    idx = np.arange(10)
    seq = list(iterate_batches(idx, 4))
    assert [len(b) for b in seq] == [4, 4, 2] and np.concatenate(seq).tolist() == list(range(10))
    rng = np.random.default_rng(0)
    w = np.zeros(10); w[3] = 1.0
    draws = np.concatenate(list(iterate_batches(idx, 4, rng=rng, weights=w)))
    assert len(draws) == 10 and set(draws.tolist()) == {3}


# ----------------------------------------------------------------------------- training smoke on the real cache
@pytest.mark.skipif(not HAVE_CACHE or not (DATA_ROOT / "experiments" / "internal_folds_size.json").exists(), reason="tensor cache / folds absent")
def test_run_smoke_fold_with_recipe_options(tmp_path):
    from experiments.train import run
    data = WindowData(in_ram=False)
    assert all(c in data.win for c in ("curb_mass_kg", "wheelbase_mm", "body")) and len(data.drives) > 0
    r = run("size", "split_platform", "gru", seed=0, data=data, out_dir=tmp_path, ckpt_dir=tmp_path, log=lambda s: None,
            aux="cls+coral+fp", sampling="class_model", augment="flip,mask", fold=0, select_on="inner_drive_f1", decision="avg",
            eval_test=False, limit_train=1500, limit_val=400, recipe_id="smoke", epochs=1, d=32, n_layers=1, eval_batch=512)
    assert r["tag"] == "fold0__size__gru__smoke__s0" and r["fold"] == 0 and r["recipe_id"] == "smoke" and r["decision"] == "avg"
    assert r["n_train"] == 1500 and r["n_val"] == 400 and r["n_test"] == 0 and r["epochs_run"] == 1 and r["best_epoch"] == 1
    assert set(r["inner_val_metrics"]) == {"window", "drive_full"} and r["protocols"] == {} and r["drive_level"] == {} and r["unit_level"] == {}
    assert {"cls", "coral", "fp", "total"} <= set(r["loss_parts"]) and np.isfinite(r["fp_sigma"])
    assert len(r["heldout_models"]) == len(IF.load_folds("size")["folds"][0]["heldout_models"])
    for key in ("tag", "task", "split_col", "model", "seed", "n_params", "cfg", "n_train", "n_val", "n_test", "epochs_run", "best_epoch",
                "val_macro_f1", "val_metrics", "train_minutes", "protocols", "drive_level", "history"):
        assert key in r                                            # every pre-recipe key is still there
    saved = json.loads((tmp_path / f"{r['tag']}.json").read_text())
    assert saved["tag"] == r["tag"]
    z = np.load(tmp_path / f"{r['tag']}_oof.npz")
    assert set(z.files) >= {"window_id", "probs", "y", "softmax", "coral", "fp_mu"} and z["probs"].shape == (400, 3)
    assert np.allclose(z["probs"].sum(1), 1.0, atol=1e-4)
    ck = torch.load(tmp_path / f"{r['tag']}.pt", map_location="cpu", weights_only=False)
    m = M.build_model("gru", len(ck["channels"]), len(ck["classes"]), d=ck["cfg"]["d"], n_layers=ck["cfg"]["n_layers"], heads=ck["heads"])
    m.load_state_dict(ck["state"], strict=True)
    assert ck["heads"] == {"coral": 3, "fp": 1}


@pytest.mark.skipif(not HAVE_CACHE, reason="tensor cache absent")
def test_run_smoke_final_with_test_protocols_hybrids_dro_and_adversary(tmp_path):
    from experiments.train import run
    data = WindowData(in_ram=False)
    r = run("bev_ice", "split_platform", "gru", seed=0, data=data, out_dir=tmp_path, ckpt_dir=tmp_path, log=lambda s: None,
            aux="cls+mass+pt4", sampling="class_model", augment="flip,cutmix", include_hybrids=True, adv_lambda=0.1, dro=True,
            fixed_epochs=1, eval_test=True, save_emb=True, limit_train=1000, limit_val=300, recipe_id="smoke", d=32, n_layers=1, eval_batch=1024)
    assert r["tag"] == "split_platform__bev_ice__gru__smoke__s0" and r["model"] == "gru__smoke" and r["arch"] == "gru" and r["fold"] is None
    assert r["n_train"] == 1000 and r["n_val"] == 300 and r["n_test"] > 0 and r["epochs_run"] == 1 and r["best_epoch"] == 1 and r["fixed_epochs"] == 1
    assert r["include_hybrids"] is True and 0 < r["n_train_extra"] < 1000 and r["dro"] is True and r["adv_lambda"] == 0.1
    assert set(r["protocols"]) == set(PROTOCOLS) and set(r["drive_level"]) == {"test_full", "test_human_only"} and set(r["unit_level"]) == set(UNIT_PROTOCOLS)
    for key in ("macro_f1", "confusion", "n"):                       # contract (4): drive_full is the old drive_level on test_full
        assert r["unit_level"]["drive_full"][key] == r["drive_level"]["test_full"][key]
        assert r["unit_level"]["drive_human_only"][key] == r["drive_level"]["test_human_only"][key]
    assert r["unit_level"]["drive_full"]["n_units"] == r["drive_level"]["test_full"]["n_drives"] and r["unit_level"]["vehicle_full_multi"]["n_units"] <= r["unit_level"]["vehicle_full"]["n_units"]
    assert {"cls", "mass", "pt4", "dom", "total"} <= set(r["loss_parts"])
    z = np.load(tmp_path / f"{r['tag']}_probs.npz")
    assert set(z.files) == {"window_id", "probs", "y", "mass_mu", "emb"} and z["probs"].shape == (r["n_test"], 2) and z["emb"].dtype == np.float16
    assert np.allclose(z["probs"].sum(1), 1.0, atol=1e-4) and (z["y"] >= 0).all()
    assert not (tmp_path / f"{r['tag']}_oof.npz").exists() and json.loads((tmp_path / f"{r['tag']}.json").read_text())["tag"] == r["tag"]
    ck = torch.load(tmp_path / f"{r['tag']}.pt", map_location="cpu", weights_only=False)
    assert ck["heads"] == {"mass": 1, "pt4": 4, "dom": len(data.model_codes)} and ck["recipe_id"] == "smoke" and ck["fold"] is None
    m = M.build_model("gru", len(ck["channels"]), len(ck["classes"]), d=32, n_layers=1, heads=ck["heads"])
    m.load_state_dict(ck["state"], strict=True)


@pytest.mark.skipif(not HAVE_CACHE, reason="tensor cache absent")
def test_run_smoke_channel_subset_variant_records_mask_and_reloads(tmp_path):
    from experiments.regen_probs import load_checkpoint
    from experiments.run_all import load_results
    from experiments.train import predict_probs, run
    data = WindowData(in_ram=False)
    r = run("size", "split_platform", "gru", seed=0, data=data, out_dir=tmp_path, ckpt_dir=tmp_path, log=lambda s: None,
            aux="cls+fp", augment="flip,mask,chdrop,noise", fixed_epochs=1, eval_test=True, limit_train=800, limit_val=300,
            recipe_id="smoke", channels=",".join(V3), tag_prefix="transfer", d=32, n_layers=1, eval_batch=1024)
    assert r["tag"] == "transfer__size__gru__smoke__s0" and r["split_col"] == "split_platform" and r["channels"] == V3
    assert r["cfg"]["channels"] == V3 and set(r["protocols"]) == set(PROTOCOLS) and r["unit_level"]["vehicle_full"]["n_units"] > 0
    ck = torch.load(tmp_path / f"{r['tag']}.pt", map_location="cpu", weights_only=False)
    assert ck["cfg"]["channels"] == V3 and list(ck["channels"]) == CHANNELS                 # subset in cfg, tensor order unchanged
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, prep, _ = load_checkpoint(tmp_path / f"{r['tag']}.pt", device)                 # strict load with the fp head
    assert prep.drop is not None and prep.drop.sum().item() == len(CHANNELS) - 3
    z = np.load(tmp_path / f"{r['tag']}_probs.npz")
    te = data.indices("size", "split_platform", "test")[:256]
    again = predict_probs(model, prep, data, te, device, 256, amp=False)
    assert np.allclose(again, z["probs"][:256], atol=2e-3)                                 # the reloaded mask reproduces the saved probabilities
    x = prep(data.batch(te[:8]).to(device))
    assert (x[:, :, [i for i, c in enumerate(CHANNELS) if c not in V3]] == 0).all()
    assert load_results(tmp_path) == [] and len(load_results(tmp_path, official_only=False)) == 1   # kept out of the official tables
