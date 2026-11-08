#!/usr/bin/env python3
"""Frozen 60-s window embeddings from a saved checkpoint (contract (2) of the method-research plan).

Reads <data-root>/experiments/ckpt/<tag>.pt, or experiments/recipe/ckpt/<tag>.pt for a fold run (state, mean, std, cfg,
model, channels, classes and the auxiliary `heads` as written by experiments/train.py) and the tensor cache through
experiments.data.WindowData (memory-mapped); rebuilds the network with experiments.models.build_model (with the stored
heads, so recipe checkpoints with CORAL / footprint / mass / powertrain / domain heads load strictly) and the checkpoint's
Preprocess statistics, runs it over every window in tensor-cache order and captures the input of model.head.norm (the
mean + max pooled features, dim = 2 x token width) with a forward hook; the extra heads never touch that feature.
Writes <data-root>/work/tensors/wemb_<tag>.npy ((N, dim) float16) and wemb_<tag>.json ({"ckpt", "dim", "hook", ...});
with --limit N only the first N windows are embedded, both files carry a _limit<N> suffix and, unless --out-dir is given,
they go to <data-root>/tmp instead of work/tensors (the MIL loader globs work/tensors/wemb_*.npy, and a partial file there
would be concatenated with the full ones). Mixed precision (bfloat16) is used on CUDA except for the GRU (cuDNN GRU stays
in float32, as in training). Prints counts only.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402
from experiments.data import Preprocess, WindowData  # noqa: E402
from experiments.models import build_model  # noqa: E402

HOOK = "input of model.head.norm"
NO_AMP = ("gru",)                    # mirrors experiments.train.NO_AMP: the cuDNN GRU runs in float32
F16_MAX = 65000.0


def ckpt_path(tag_or_path: str, data_root: Path | str | None = None) -> Path:
    """A checkpoint given as an existing .pt path, or as a tag: <data-root>/experiments/ckpt/<tag>.pt (final runs), else
    <data-root>/experiments/recipe/ckpt/<tag>.pt (fold runs of experiments/train.py); the first location when neither
    exists (so the caller's existence check still names it)."""
    p = Path(tag_or_path)
    if p.suffix == ".pt" and p.exists():
        return p
    exp = Path(data_root or C.DATA_ROOT) / "experiments"
    candidates = [exp / "ckpt" / f"{tag_or_path}.pt", exp / "recipe" / "ckpt" / f"{tag_or_path}.pt"]
    return next((c for c in candidates if c.exists()), candidates[0])


def default_out_dir(data_root: Path | str, limit: int = 0) -> Path:
    """<data-root>/work/tensors for a full run; a --limit smoke run defaults to <data-root>/tmp so that no partial
    wemb_*.npy ever lands where experiments/mil_train.discover_wemb globs the full embeddings."""
    root = Path(data_root)
    return root / "tmp" if limit else root / "work" / "tensors"


def load_checkpoint(path: Path | str) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def model_from_checkpoint(ck: dict, clip_lo: np.ndarray, clip_hi: np.ndarray, device) -> tuple[torch.nn.Module, Preprocess]:
    """The network and pre-processing exactly as the checkpoint says (eval mode, on `device`): the architecture from cfg
    and the auxiliary heads from ck["heads"] ({} or absent for a plain classifier), so the state dict loads strictly."""
    cfg = ck["cfg"]
    model = build_model(ck["model"], len(ck["channels"]), len(ck["classes"]), d=cfg["d"], n_layers=cfg["n_layers"],
                        dropout=cfg["dropout"], heads=dict(ck.get("heads") or {}) or None)
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    prep = Preprocess(np.asarray(ck["mean"], np.float32), np.asarray(ck["std"], np.float32), clip_lo, clip_hi).to(device)
    return model, prep


def embedding_dim(model: torch.nn.Module) -> int:
    return int(model.head.norm.normalized_shape[0])


def output_paths(out_dir: Path, tag: str, limit: int = 0) -> tuple[Path, Path]:
    stem = f"wemb_{tag}" + (f"_limit{limit}" if limit else "")
    return out_dir / f"{stem}.npy", out_dir / f"{stem}.json"


@torch.no_grad()
def window_embeddings(ckpt: Path | str | dict, data, device="cuda", batch: int = 1024, limit: int = 0, log=print) -> np.ndarray:
    """(N, dim) float16 pooled features (input of model.head.norm) of the first N = len(data.X) (or `limit`) windows in
    tensor-cache order. `data` provides channels, clip_lo / clip_hi, X and batch(idx) (experiments.data.WindowData)."""
    ck = ckpt if isinstance(ckpt, dict) else load_checkpoint(ckpt)
    if list(ck["channels"]) != list(data.channels):
        raise ValueError("checkpoint channel order differs from the tensor cache")
    device = torch.device(device)
    model, prep = model_from_checkpoint(ck, data.clip_lo, data.clip_hi, device)
    amp = ck["model"] not in NO_AMP and device.type == "cuda"
    n_all = int(data.X.shape[0])
    n = min(limit, n_all) if limit else n_all
    dim = embedding_dim(model)
    out = np.zeros((n, dim), dtype=np.float16)
    store: dict[str, torch.Tensor] = {}
    handle = model.head.norm.register_forward_hook(lambda m, inp, o: store.__setitem__("z", inp[0]))
    t0 = time.time()
    try:
        for i in range(0, n, batch):
            idx = np.arange(i, min(n, i + batch))
            x = prep(data.batch(idx).to(device, non_blocking=True))
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()
            with ctx:
                model(x)
            z = store["z"].float().cpu().numpy()
            out[i:i + len(idx)] = np.clip(z, -F16_MAX, F16_MAX).astype(np.float16)
    finally:
        handle.remove()
    log(f"embedded n={n} dim={dim} model={ck['model']} amp={amp} minutes={(time.time() - t0) / 60:.1f}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint tag (experiments/ckpt/<tag>.pt or experiments/recipe/ckpt/<tag>.pt) or a .pt path")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N windows (smoke test; output gets a _limit<N> suffix and defaults to <data-root>/tmp)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None, help="default <data-root>/work/tensors (<data-root>/tmp with --limit)")
    a = ap.parse_args(argv)
    path = ckpt_path(a.ckpt, a.data_root)
    if not path.exists():
        print("checkpoint not found")
        return 1
    tag = path.stem
    ck = load_checkpoint(path)
    data = WindowData(a.data_root, in_ram=False)
    emb = window_embeddings(ck, data, a.device, batch=a.batch, limit=a.limit)
    # TODO(contract (2), package D): experiments/mil_train.discover_wemb could also skip '*_limit*' files under work/tensors
    out_dir = a.out_dir or default_out_dir(a.data_root, a.limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy, js = output_paths(out_dir, tag, a.limit)
    np.save(npy, emb)
    js.write_text(json.dumps(dict(ckpt=tag, dim=int(emb.shape[1]), hook=HOOK, n=int(emb.shape[0]), limit=int(a.limit),
                                  model=ck["model"], task=ck["task"], split_col=ck["split_col"], dtype="float16",
                                  order="tensor-cache order (work/tensors/windows.parquet tensor_idx)"), indent=1))
    print(f"wrote {npy.name} shape={tuple(emb.shape)} finite_frac={float(np.isfinite(emb.astype(np.float32)).mean()):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
