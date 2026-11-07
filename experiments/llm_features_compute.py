#!/usr/bin/env python3
"""Compute every registered LLM-proposed feature for all windows of the tensor cache (parallel over window chunks).

writes work/tensors/features_llm_f32.npy (N, F) and work/tensors/features_llm_names.json; a feature that raises on a
window yields NaN for that window. Prints counts and the per-feature finite fraction.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402

TDIR = Path(C.DATA_ROOT) / "work" / "tensors"
_X = None
_CH = None
_FEATS = None


def _init(x_path: str, channels: list[str]):
    global _X, _CH, _FEATS
    from experiments.llm_features import FEATURES
    _X = np.load(x_path, mmap_mode="r")
    _CH = channels
    _FEATS = list(FEATURES.items())


def _chunk(bounds: tuple[int, int]) -> tuple[int, np.ndarray]:
    i0, i1 = bounds
    x = np.asarray(_X[i0:i1], dtype=np.float32)
    out = np.full((i1 - i0, len(_FEATS)), np.nan, dtype=np.float32)
    for i in range(i1 - i0):
        w = {c: x[i, :, k] for k, c in enumerate(_CH)}
        for j, (_, fn) in enumerate(_FEATS):
            try:
                v = fn(w)
                out[i, j] = float(v) if v is not None else np.nan
            except Exception:           # noqa: BLE001  one bad window must not stop the run
                out[i, j] = np.nan
    return i0, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N windows")
    a = ap.parse_args(argv)
    from experiments.llm_features import NAMES
    channels = json.loads((TDIR / "channels.json").read_text())
    X = np.load(TDIR / "X_f16.npy", mmap_mode="r")
    n = a.limit or X.shape[0]
    bounds = [(i, min(i + a.chunk, n)) for i in range(0, n, a.chunk)]
    out = np.full((n, len(NAMES)), np.nan, dtype=np.float32)
    t0 = time.time()
    with Pool(processes=max(1, a.workers), initializer=_init, initargs=(str(TDIR / "X_f16.npy"), channels)) as pool:
        for k, (i0, arr) in enumerate(pool.imap_unordered(_chunk, bounds), 1):
            out[i0:i0 + len(arr)] = arr
            if k % 20 == 0:
                print(f"progress: chunks={k}/{len(bounds)} elapsed_s={time.time() - t0:.0f}", flush=True)
    suffix = "" if not a.limit else f"_smoke{a.limit}"
    np.save(TDIR / f"features_llm_f32{suffix}.npy", out)
    (TDIR / f"features_llm_names{suffix}.json").write_text(json.dumps(NAMES))
    fin = np.isfinite(out).mean(0)
    print(f"done: windows={n} features={len(NAMES)} elapsed_s={time.time() - t0:.0f}")
    print("finite fraction per feature: " + ", ".join(f"{nm}={f:.2f}" for nm, f in zip(NAMES, fin)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
