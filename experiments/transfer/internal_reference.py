#!/usr/bin/env python3
"""In-distribution reference numbers for the zero-shot transfer report (experiments/transfer/RESULTS.md).

For the final models (split_platform__size__gru__S9, split_platform__bev_ice__resnet__B5) and the channel-subset variants
(transfer__size__gru__V1/V2/V3, transfer__bev_ice__resnet__V1/V2/V3) this script pools the saved test probabilities
(results/<tag>_probs.npz, test_full windows of the held-out car models) into the seed ensemble (mean probability) and
evaluates it exactly as score.py evaluates an external set: window -> drive (mean probability per drive) -> vehicle
(all drives of a vehicle), accuracy, macro-F1, per-class recall, plus the same on two sub-populations that matter for the
external sets: windows with the driving automation engaged (das_frac >= 0.5; the ACC-driven external sets) and the
strict human-only windows (test_human_only). The full models are also run on the same test windows with the external
channel masks V1 / V2 / V3 applied (common.apply_channel_mask, the "missing channels at the missing value" reference of
the secondary external numbers); their probabilities are cached under <external>/_internal_ref/. Writes
internal_reference.json + .md next to this file (no identifiers: counts and metrics only).

  source env.sh && $CPY experiments/transfer/internal_reference.py
"""
from __future__ import annotations

import json
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
from experiments.transfer.score import aggregate, cm_metrics, confusion, run_checkpoint, pick_device  # noqa: E402

SETS = {
    "size": {"full (GRU S9, 20 ch)": ("split_platform__size__gru__S9", (0, 1, 2)),
             "V1 lateral + speed (10 ch)": ("transfer__size__gru__V1", (0, 1)),
             "V2 longitudinal-only (11 ch)": ("transfer__size__gru__V2", (0, 1)),
             "V3 speed-only (3 ch)": ("transfer__size__gru__V3", (0, 1))},
    "bev_ice": {"full (ResNet B5, 20 ch)": ("split_platform__bev_ice__resnet__B5", (0, 1, 2)),
                "V1 lateral + speed (10 ch)": ("transfer__bev_ice__resnet__V1", (0, 1)),
                "V2 longitudinal-only (11 ch)": ("transfer__bev_ice__resnet__V2", (0, 1)),
                "V3 speed-only (3 ch)": ("transfer__bev_ice__resnet__V3", (0, 1))},
}
MASKED_FULL = ("V1", "V2", "V3")                       # full model + external mask, computed live on the test windows
CLASSES = {"size": ("SMALL", "MID", "LARGE"), "bev_ice": ("BEV", "ICE")}
CACHE = T.EXTERNAL_ROOT / "_internal_ref"
SUBSETS = ("all", "das", "human")


def level_metrics(p: np.ndarray, y: np.ndarray, units: np.ndarray | None, k: int) -> dict:
    if units is None:
        pred = p.argmax(1); yy = y
    else:
        uid, P, cnt, inv = aggregate(p, units)
        pred = P.argmax(1)
        yy = np.array([np.bincount(y[inv == u], minlength=k).argmax() for u in range(len(uid))])
    r = cm_metrics(confusion(yy, pred, k))
    return dict(n=int(len(yy)), accuracy=r["accuracy"], macro_f1=r["macro_f1"], per_class_recall=r["per_class_recall"],
                support=r["support"], chance_majority=r["chance_majority"])


def _round_floats(o, nd: int = 6):
    """Round every float in a nested structure (JSON outputs: short digit runs, no identifier-like tokens)."""
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round_floats(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round_floats(v, nd) for v in o]
    return o


def main() -> int:
    data = WindowData(in_ram=False)
    win = data.win.set_index("window_id")
    out: dict = {"generated": time.strftime("%-d %b %Y %H:%M"), "tasks": {}}
    for task, sets in SETS.items():
        k = len(CLASSES[task])
        out["tasks"][task] = {}
        for name, (tag, seeds) in sets.items():
            probs, y, wid = [], None, None
            for s in seeds:
                z = np.load(T.RESULTS_DIR / f"{tag}__s{s}_probs.npz", allow_pickle=True)
                w = z["window_id"].astype(str)
                if wid is None:
                    wid, y = w, z["y"].astype(int)
                else:
                    assert np.array_equal(w, wid), f"{tag} seeds disagree on the window list"
                probs.append(z["probs"].astype(np.float64))
            P = np.mean(probs, axis=0)
            m = win.loc[wid]
            drive = m["drive_id"].astype(str).to_numpy(); veh = m["vehicle_id"].astype(str).to_numpy()
            das = m["das_frac"].to_numpy(float) >= 0.5
            human = m["human_only_strict"].fillna(False).to_numpy(bool)
            res = {"tag": tag, "seeds": list(seeds), "n_windows": int(len(wid)), "n_drives": int(len(np.unique(drive))),
                   "n_vehicles": int(len(np.unique(veh))), "n_models": int(m["model"].nunique()), "subsets": {}}
            for sub, sel in (("all", np.ones(len(wid), bool)), ("das", das), ("human", human)):
                res["subsets"][sub] = {"n_windows": int(sel.sum()),
                                       "window": level_metrics(P[sel], y[sel], None, k),
                                       "drive": level_metrics(P[sel], y[sel], drive[sel], k),
                                       "vehicle": level_metrics(P[sel], y[sel], veh[sel], k)}
            out["tasks"][task][name] = res
            if name.startswith("full"):
                full_tag, full_seeds, full_wid, full_y = tag, seeds, wid, y
            print(f"[{task}] {name}: window acc {100*res['subsets']['all']['window']['accuracy']:.1f} F1 {100*res['subsets']['all']['window']['macro_f1']:.1f} | "
                  f"drive acc {100*res['subsets']['all']['drive']['accuracy']:.1f} F1 {100*res['subsets']['all']['drive']['macro_f1']:.1f} | "
                  f"vehicle acc {100*res['subsets']['all']['vehicle']['accuracy']:.1f} F1 {100*res['subsets']['all']['vehicle']['macro_f1']:.1f}", flush=True)
        # the full model with the external channel masks on the same windows
        for mk in MASKED_FULL:
            name = f"{[n for n in sets if n.startswith('full')][0]} + mask-{mk}"
            P = masked_full_probs(data, full_tag, full_seeds, full_wid, mk)
            m = win.loc[full_wid]
            drive = m["drive_id"].astype(str).to_numpy(); veh = m["vehicle_id"].astype(str).to_numpy()
            das = m["das_frac"].to_numpy(float) >= 0.5
            human = m["human_only_strict"].fillna(False).to_numpy(bool)
            res = {"tag": f"{full_tag} + mask-{mk}", "seeds": list(full_seeds), "n_windows": int(len(full_wid)), "n_drives": int(len(np.unique(drive))),
                   "n_vehicles": int(len(np.unique(veh))), "n_models": int(m["model"].nunique()), "subsets": {}}
            for sub, sel in (("all", np.ones(len(full_wid), bool)), ("das", das), ("human", human)):
                res["subsets"][sub] = {"n_windows": int(sel.sum()),
                                       "window": level_metrics(P[sel], full_y[sel], None, k),
                                       "drive": level_metrics(P[sel], full_y[sel], drive[sel], k),
                                       "vehicle": level_metrics(P[sel], full_y[sel], veh[sel], k)}
            out["tasks"][task][name] = res
            print(f"[{task}] {name}: window acc {100*res['subsets']['all']['window']['accuracy']:.1f} F1 {100*res['subsets']['all']['window']['macro_f1']:.1f} | "
                  f"drive acc {100*res['subsets']['all']['drive']['accuracy']:.1f} F1 {100*res['subsets']['all']['drive']['macro_f1']:.1f} | "
                  f"vehicle acc {100*res['subsets']['all']['vehicle']['accuracy']:.1f} F1 {100*res['subsets']['all']['vehicle']['macro_f1']:.1f}", flush=True)
    here = Path(__file__).resolve().parent
    (here / "internal_reference.json").write_text(json.dumps(_round_floats(out), indent=1))
    (here / "internal_reference.md").write_text(render(out))
    print(f"wrote {here / 'internal_reference.md'}")
    return 0


def masked_full_probs(data, tag: str, seeds, wid: np.ndarray, mk: str) -> np.ndarray:
    """Seed-ensemble probabilities of the full checkpoints on the test windows `wid` with the external mask `mk` applied
    before the training Preprocess (cached per checkpoint under CACHE)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    pos = {w: i for i, w in enumerate(data.win["window_id"].to_numpy(object))}
    tidx = np.array([pos[w] for w in wid], dtype=np.int64)
    X = None
    device = pick_device()
    probs = []
    for s in seeds:
        f = CACHE / f"probs__{tag}__s{s}__mask-{mk}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            if np.array_equal(z["window_id"].astype(str), wid):
                probs.append(z["probs"].astype(np.float64)); continue
        if X is None:
            order = np.argsort(tidx)
            X = np.asarray(data.X[tidx[order]], dtype=np.float32)[np.argsort(order)]
            X = T.apply_channel_mask(X, T.resolve_mask(mk))
        p, info = run_checkpoint(T.CKPT_DIR / f"{tag}__s{s}.pt", X, device)
        np.savez_compressed(f, window_id=wid, probs=p, mask=mk, checkpoint=f"{tag}__s{s}")
        probs.append(p.astype(np.float64))
    return np.mean(probs, axis=0)


def _p(x) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:.1f}"


def render(out: dict) -> str:
    L = [f"# In-distribution reference of the seed ensembles (split_platform test, held-out car models)", "",
         f"Generated {out['generated']} by experiments/transfer/internal_reference.py from results/<tag>_probs.npz. Ensemble = mean of the seeds' "
         "window probabilities (the same rule score.py applies to external sets); drive = mean probability per drive, vehicle = all drives "
         "of a vehicle pooled; accuracy and macro-F1 in percent, recall per class in the class order. `all` = test_full; `das` = windows "
         "with the driving automation engaged on >= 50 % of the samples (the reference for ACC-driven external sets); `human` = strict "
         "human-only windows (test_human_only). Rows `full ... + mask-Vk` = the full 20-channel model run on the same windows with the external "
         "channel mask Vk applied (missing channels at the training missing value), the reference for the secondary external numbers.", ""]
    for task, sets in out["tasks"].items():
        classes = CLASSES[task]
        L.append(f"## {task} ({' / '.join(classes)})"); L.append("")
        rows = []
        for name, r in sets.items():
            for sub in SUBSETS:
                s = r["subsets"][sub]
                for lv in ("window", "drive", "vehicle"):
                    v = s[lv]
                    rows.append([name, sub, lv, v["n"], _p(v["accuracy"]), _p(v["macro_f1"]), _p(v["chance_majority"]),
                                 " / ".join(f"{c} {_p(x)}" for c, x in zip(classes, v["per_class_recall"])), "/".join(str(x) for x in v["support"])])
        L.append(T.md_table(["model", "subset", "level", "units", "accuracy %", "macro-F1 %", "majority %", "recall per class %", "support"], rows))
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
