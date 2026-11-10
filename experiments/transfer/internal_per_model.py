#!/usr/bin/env python3
"""Per-car-model decisions of the final models and the channel-subset variants on OUR OWN held-out test models
(split_platform test split), for comparison with the external named-vehicle tables: for every held-out car model the
label, counts, the drive-level accuracy and the model-level decision (mean probability over all its test windows ->
arg-max) of the seed ensembles full / V1 / V2 / V3 and of the full model under the external masks (cached probabilities
of internal_reference.py). Writes internal_per_model.md next to this file (public car-model names only).

  source env.sh && $CPY experiments/transfer/internal_per_model.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.data import WindowData  # noqa: E402
from experiments.transfer import common as T  # noqa: E402
from experiments.transfer.score import aggregate  # noqa: E402
from experiments.transfer.internal_reference import SETS, CLASSES, CACHE  # noqa: E402

HERE = Path(__file__).resolve().parent


def ens(tag: str, seeds, masked: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    P, wid, y = [], None, None
    for s in seeds:
        f = (CACHE / f"probs__{tag}__s{s}__mask-{masked}.npz") if masked else (T.RESULTS_DIR / f"{tag}__s{s}_probs.npz")
        z = np.load(f, allow_pickle=True)
        w = z["window_id"].astype(str)
        if wid is None:
            wid = w
        else:
            assert np.array_equal(w, wid)
        if "y" in z:
            y = z["y"].astype(int)
        P.append(z["probs"].astype(np.float64))
    return np.mean(P, axis=0), wid, y


def main() -> int:
    data = WindowData(in_ram=False)
    win = data.win.set_index("window_id")
    L = ["# Internal per-car-model decisions (split_platform test = 41 held-out car models)", "",
         f"Generated {time.strftime('%-d %b %Y %H:%M')} by experiments/transfer/internal_per_model.py. For every held-out car model: label, test windows / drives / "
         "vehicles, and per model the drive-level accuracy (%) and the model-level decision (mean probability over all its windows -> arg-max; P of the "
         "true class in brackets). `full+Vk` = the full model with the external mask Vk applied. These are the in-distribution counterparts of the external "
         "named-vehicle tables in RESULTS.md.", ""]
    for task, sets in SETS.items():
        classes = CLASSES[task]; k = len(classes)
        cols = []
        for name, (tag, seeds) in sets.items():
            cols.append((name.split(" (")[0], ens(tag, seeds)))
        full_tag, full_seeds = sets[[n for n in sets if n.startswith("full")][0]]
        for mk in ("V1", "V2", "V3"):
            cols.append((f"full+{mk}", ens(full_tag, full_seeds, masked=mk)))
        P0, wid, y = cols[0][1]
        for i, (n, (P, w, yy)) in enumerate(cols):
            if yy is None:
                cols[i] = (n, (P, w, y))
        m = win.loc[wid]
        models = m["model"].astype(str).to_numpy(); drive = m["drive_id"].astype(str).to_numpy(); veh = m["vehicle_id"].astype(str).to_numpy()
        fp = m["footprint_m2"].to_numpy(float) if "footprint_m2" in m else np.full(len(m), np.nan)
        rows = []
        for model in sorted(set(models)):
            sel = models == model
            lab = int(np.bincount(y[sel], minlength=k).argmax())
            cells = [model, classes[lab] + (f" ({np.nanmean(fp[sel]):.2f})" if np.isfinite(np.nanmean(fp[sel])) else ""), int(sel.sum()), int(len(np.unique(drive[sel]))), int(len(np.unique(veh[sel])))]
            for n, (P, w, yy) in cols:
                uid, Pd, cnt, inv = aggregate(P[sel], drive[sel])
                dacc = float((Pd.argmax(1) == lab).mean())
                pm = P[sel].mean(0)
                cells.append(f"{100*dacc:.0f} / {classes[int(pm.argmax())]} ({pm[lab]:.2f})" + ("" if int(pm.argmax()) == lab else " x"))
            rows.append(cells)
        L.append(f"## {task} ({' / '.join(classes)}); cells = drive acc % / model-level decision (P true), x = miss"); L.append("")
        L.append(T.md_table(["held-out model", "label (footprint)", "win", "drives", "veh"] + [n for n, _ in cols], rows))
        L.append("")
        # per-class summary of model-level decisions
        rows = []
        for n, (P, w, yy) in cols:
            hit = {c: [0, 0] for c in range(k)}
            for model in set(models):
                sel = models == model
                lab = int(np.bincount(y[sel], minlength=k).argmax())
                hit[lab][1] += 1; hit[lab][0] += int(P[sel].mean(0).argmax() == lab)
            rows.append([n] + [f"{hit[c][0]}/{hit[c][1]}" for c in range(k)] + [f"{sum(h[0] for h in hit.values())}/{sum(h[1] for h in hit.values())}"])
        L.append(f"Model-level decisions correct per class ({task}):"); L.append("")
        L.append(T.md_table(["model"] + [f"{c} models correct" for c in classes] + ["all"], rows)); L.append("")
    (HERE / "internal_per_model.md").write_text("\n".join(L) + "\n")
    print("wrote", HERE / "internal_per_model.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
