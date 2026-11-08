#!/usr/bin/env python3
"""Regenerate the validation-split window probabilities of every saved checkpoint (no training).

Reads <data-root>/experiments/ckpt/<tag>.pt (dict: state, mean, std, cfg, task, split_col, model, channels, classes)
and the tensor cache through experiments.data.WindowData; rebuilds the network with experiments.models.build_model,
loads the weights, applies the stored normalisation (experiments.data.Preprocess with the training clip bounds) and
writes <data-root>/experiments/results/<tag>_valprobs.npz (window_id, probs, y) for the validation split of the
checkpoint's split scheme and task. The recomputed validation macro-F1 is compared with the `val_macro_f1` stored in the
run's results JSON as a consistency check (the best epoch was selected on exactly this number). Mixed precision is off
for the GRU, as in training. Prints counts and metrics only.

Smoke: `--tags 'split_platform__size__transformer__s0' --limit 500` processes the first 500 validation windows of one
checkpoint (the stored value is then not comparable and is printed as a reference only).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402
from experiments.data import CLIP, Preprocess, WindowData  # noqa: E402
from experiments.models import build_model  # noqa: E402
from experiments.train import macro_f1, predict_probs  # noqa: E402

try:
    from experiments.train import NO_AMP  # noqa: E402
except ImportError:                                   # pragma: no cover - contract (5): train.py keeps NO_AMP
    NO_AMP = ("gru",)

EXP_ROOT = Path(C.DATA_ROOT) / "experiments"
SUFFIX = "_valprobs.npz"


def list_checkpoints(ckpt_dir: Path, pattern: str = "*") -> list[Path]:
    """Checkpoint files whose tag (file stem) matches the glob pattern, sorted by name."""
    return sorted(p for p in Path(ckpt_dir).glob("*.pt") if fnmatch.fnmatch(p.stem, pattern))


def clip_bounds(channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (lo, hi) clip bounds of experiments.data.CLIP in the order of `channels` (inf where unclipped)."""
    lo = np.array([CLIP.get(c, (-np.inf, np.inf))[0] for c in channels], dtype=np.float32)
    hi = np.array([CLIP.get(c, (-np.inf, np.inf))[1] for c in channels], dtype=np.float32)
    return lo, hi


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, Preprocess, dict]:
    """Rebuild the network of a checkpoint, load its weights (strict; the non-strict fallback prints key counts) and
    its pre-processing; returns (model in eval mode on `device`, prep, checkpoint dict)."""
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = build_model(ck["model"], len(ck["channels"]), len(ck["classes"]), d=cfg["d"], n_layers=cfg["n_layers"],
                        dropout=cfg["dropout"])
    try:
        model.load_state_dict(ck["state"], strict=True)
    except RuntimeError:
        # TODO(contract 5): if C1 adds heads to the networks, old checkpoints lack their keys; counts are printed, not keys
        info = model.load_state_dict(ck["state"], strict=False)
        print(f"[{Path(path).stem}] non-strict load: missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}", flush=True)
    lo, hi = clip_bounds(list(ck["channels"]))
    prep = Preprocess(np.asarray(ck["mean"], dtype=np.float32), np.asarray(ck["std"], dtype=np.float32), lo, hi).to(device)
    return model.to(device).eval(), prep, ck


def regenerate(path: Path, data: WindowData, device: torch.device, out_dir: Path, limit: int = 0, batch: int = 1024) -> dict:
    """Validation probabilities of one checkpoint -> <out_dir>/<tag>_valprobs.npz; returns counts and the macro-F1."""
    model, prep, ck = load_checkpoint(path, device)
    tag = Path(path).stem
    if list(ck["channels"]) != list(data.channels):
        raise ValueError(f"[{tag}] checkpoint channels differ from the tensor cache")
    task, split_col, classes = ck["task"], ck["split_col"], tuple(ck["classes"])
    va = data.indices(task, split_col, "validation")
    if limit:
        va = va[:limit]
    y = data.labels(task, va)
    amp = ck["model"] not in NO_AMP and device.type == "cuda"
    t0 = time.time()
    probs = predict_probs(model, prep, data, va, device, batch, amp)
    f1 = macro_f1(y, probs.argmax(1), len(classes))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}{SUFFIX}"
    np.savez_compressed(out, window_id=data.win["window_id"].to_numpy(object)[va].astype(str), probs=probs, y=y)
    meta = out_dir / f"{tag}.json"
    stored = json.loads(meta.read_text()).get("val_macro_f1") if meta.exists() else None
    res = dict(tag=tag, task=task, split_col=split_col, model=ck["model"], n=int(len(va)), limited=bool(limit), amp=amp,
               macro_f1=f1, stored_val_macro_f1=stored, seconds=time.time() - t0, out=str(out))
    res["consistent"] = (None if (stored is None or limit) else bool(abs(f1 - stored) < 0.01))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", default="*", help="glob over checkpoint tags, e.g. 'split_platform__*__gru__*'")
    ap.add_argument("--limit", type=int, default=0, help="smoke: only the first N validation windows")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--ckpt-dir", type=Path, default=EXP_ROOT / "ckpt")
    ap.add_argument("--out-dir", type=Path, default=EXP_ROOT / "results")
    ap.add_argument("--in-ram", action="store_true", help="load the tensor cache into RAM (default: memory map)")
    ap.add_argument("--overwrite", action="store_true", help="recompute files that already exist")
    a = ap.parse_args(argv)
    paths = list_checkpoints(a.ckpt_dir, a.tags)
    if not paths:
        print("no checkpoint matches")
        return 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = WindowData(in_ram=a.in_ram)
    print(f"checkpoints={len(paths)} windows={len(data.win)} device={device.type} limit={a.limit}", flush=True)
    n_done = n_skip = n_inconsistent = 0
    for i, p in enumerate(paths, 1):
        if not a.overwrite and (a.out_dir / f"{p.stem}{SUFFIX}").exists():
            n_skip += 1
            print(f"[{i}/{len(paths)}] skip {p.stem} (exists)", flush=True)
            continue
        r = regenerate(p, data, device, a.out_dir, limit=a.limit, batch=a.batch)
        n_done += 1
        n_inconsistent += r["consistent"] is False
        stored = "-" if r["stored_val_macro_f1"] is None else f"{r['stored_val_macro_f1']:.4f}"
        print(f"[{i}/{len(paths)}] {r['tag']} n={r['n']} val_f1={r['macro_f1']:.4f} stored={stored}"
              f"{' (subset, not comparable)' if r['limited'] else ''} amp={r['amp']} seconds={r['seconds']:.0f}", flush=True)
    print(f"done={n_done} skipped={n_skip} inconsistent={n_inconsistent}")
    return 0 if n_inconsistent == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
