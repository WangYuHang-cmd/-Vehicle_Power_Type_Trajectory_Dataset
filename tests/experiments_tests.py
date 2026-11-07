"""experiments/: the Mamba-2 chunked scan against a naive recurrence, model shapes, features and metrics.

The torch-dependent tests are skipped in an interpreter without torch (the build venv); run them with the
miniconda interpreter (`$CPY -m pytest tests/experiments_tests.py`).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments import models as M  # noqa: E402
from experiments.data import iterate_batches  # noqa: E402
from experiments.features_baseline import STATS, window_features  # noqa: E402
from experiments.train import balanced_accuracy, confusion, macro_f1, metrics  # noqa: E402


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
    assert M.n_params(m) > 0 and m.head.fc2.out_features == 3
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad and p.numel() > 0 and p.grad is not None)


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
