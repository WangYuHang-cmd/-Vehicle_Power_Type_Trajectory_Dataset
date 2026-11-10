#!/usr/bin/env python3
"""Diagnostic for the transfer report: does replacing the CAN acceleration (aEgo) by the external adapters' derived
acceleration (common.smooth_derivative of the 10 Hz speed, 0.5-s or 1.0-s window) change the decisions of the reduced-input
variants on OUR OWN held-out test windows? Every external set except commaCarSegments carries a derived acceleration, so
this separates "the car is different" from "the acceleration channel is different". Also tests a 1 Hz speed hold
(VED-like) and a GNSS-like smoothing of the speed itself. Runs the V2 and V3 variants (both tasks) on test_full; writes
diag_derived_accel.json / .md next to this file.

  source env.sh && $CPY experiments/transfer/diag_derived_accel.py
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

CLASSES = {"size": ("SMALL", "MID", "LARGE"), "bev_ice": ("BEV", "ICE")}
SETS = {"size": [("V3 speed-only", "transfer__size__gru__V3", (0, 1)), ("V2 longitudinal-only", "transfer__size__gru__V2", (0, 1))],
        "bev_ice": [("V3 speed-only", "transfer__bev_ice__resnet__V3", (0, 1)), ("V2 longitudinal-only", "transfer__bev_ice__resnet__V2", (0, 1))]}
HERE = Path(__file__).resolve().parent
N_MAX = 8000          # random test windows per task (speed: the derivative loop is per window)


def derived_accel(X: np.ndarray, win_s: float) -> np.ndarray:
    Y = X.copy()
    iv, ia = T.CH["v_mps"], T.CH["a_mps2"]
    for i in range(len(Y)):
        Y[i, :, ia] = T.smooth_derivative(Y[i, :, iv].astype(np.float64), hz=10, win_s=win_s)
    return Y


def hold_1hz(X: np.ndarray, win_s: float = 1.0) -> np.ndarray:
    """VED-like: speed known once per second (linear interpolation between the 1-s nodes), acceleration derived with a 1-s window."""
    Y = X.copy()
    iv, ia = T.CH["v_mps"], T.CH["a_mps2"]
    t = np.arange(600) / 10.0
    nodes = np.arange(0, 60.0, 1.0)
    for i in range(len(Y)):
        v = Y[i, :, iv].astype(np.float64)
        vn = np.interp(nodes, t, v)
        Y[i, :, iv] = np.interp(t, nodes, vn)
        Y[i, :, ia] = T.smooth_derivative(Y[i, :, iv].astype(np.float64), hz=10, win_s=win_s)
    return Y


def smooth_speed(X: np.ndarray, win_s: float = 0.5) -> np.ndarray:
    """GNSS-spline-like: the speed itself smoothed (0.5-s moving average), acceleration derived from it."""
    Y = X.copy()
    iv, ia = T.CH["v_mps"], T.CH["a_mps2"]
    k = int(round(win_s * 10)); ker = np.ones(k) / k
    for i in range(len(Y)):
        v = Y[i, :, iv].astype(np.float64)
        vp = np.pad(v, (k // 2, k - 1 - k // 2), mode="edge")
        Y[i, :, iv] = np.convolve(vp, ker, mode="valid")
        Y[i, :, ia] = T.smooth_derivative(Y[i, :, iv].astype(np.float64), hz=10, win_s=0.5)
    return Y


VARIANTS = [("CAN aEgo (as trained)", lambda X: X),
            ("a := smooth_derivative(v), 0.5 s (ULTra-AV / OpenACC / microSIM / UGA / A2D2 / Tesla)", lambda X: derived_accel(X, 0.5)),
            ("a := smooth_derivative(v), 1.0 s", lambda X: derived_accel(X, 1.0)),
            ("v smoothed 0.5 s + derived a (GNSS-spline-like)", smooth_speed),
            ("v held at 1 Hz + derived a, 1.0 s (VED-like)", hold_1hz)]


def metrics(P: np.ndarray, y: np.ndarray, drive: np.ndarray, k: int) -> dict:
    out = {}
    pred = P.argmax(1)
    r = cm_metrics(confusion(y, pred, k))
    out["window"] = dict(accuracy=r["accuracy"], macro_f1=r["macro_f1"], recall=r["per_class_recall"], pred_share=list(np.bincount(pred, minlength=k) / len(pred)))
    uid, Pd, cnt, inv = aggregate(P, drive)
    yd = np.array([np.bincount(y[inv == u], minlength=k).argmax() for u in range(len(uid))])
    r = cm_metrics(confusion(yd, Pd.argmax(1), k))
    out["drive"] = dict(n=int(len(uid)), accuracy=r["accuracy"], macro_f1=r["macro_f1"], recall=r["per_class_recall"], pred_share=list(np.bincount(Pd.argmax(1), minlength=k) / len(uid)))
    return out


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
    device = pick_device()
    rng = np.random.default_rng(0)
    out = {"generated": time.strftime("%-d %b %Y %H:%M"), "n_max": N_MAX, "tasks": {}}
    for task, sets in SETS.items():
        k = len(CLASSES[task])
        idx = data.indices(task, "split_platform", "test")
        if len(idx) > N_MAX:
            idx = np.sort(rng.choice(idx, N_MAX, replace=False))
        X0 = np.asarray(data.X[idx], dtype=np.float32)
        y = data.labels(task, idx)
        drive = data.win["drive_id"].astype(str).to_numpy()[idx]
        out["tasks"][task] = {"n_windows": int(len(idx)), "n_drives": int(len(np.unique(drive))), "variants": {}}
        Xs = {name: fn(X0) for name, fn in VARIANTS}
        for sname, tag, seeds in sets:
            out["tasks"][task]["variants"][sname] = {}
            for name in Xs:
                P = np.mean([run_checkpoint(T.CKPT_DIR / f"{tag}__s{s}.pt", Xs[name], device)[0] for s in seeds], axis=0)
                m = metrics(P, y, drive, k)
                out["tasks"][task]["variants"][sname][name] = m
                print(f"[{task}] {sname} | {name}: window acc {100*m['window']['accuracy']:.1f} F1 {100*m['window']['macro_f1']:.1f} "
                      f"recall {[round(100*x,1) for x in m['window']['recall']]} | drive acc {100*m['drive']['accuracy']:.1f} F1 {100*m['drive']['macro_f1']:.1f} "
                      f"recall {[round(100*x,1) for x in m['drive']['recall']]}", flush=True)
    (HERE / "diag_derived_accel.json").write_text(json.dumps(_round_floats(out), indent=1))
    (HERE / "diag_derived_accel.md").write_text(render(out))
    print("wrote", HERE / "diag_derived_accel.md")
    return 0


def render(out: dict) -> str:
    L = ["# Diagnostic: derived acceleration and resampled speed on our own held-out test windows", "",
         f"Generated {out['generated']} by experiments/transfer/diag_derived_accel.py. {out['n_max']} random test_full windows per task (seed 0) of the "
         "split_platform test split (held-out car models), scored by the seed ensembles of the reduced-input variants after replacing the input "
         "channels the way the external adapters have to build them. `recall` in class order; `drive` = mean probability per drive -> arg-max.", ""]
    for task, tk in out["tasks"].items():
        classes = CLASSES[task]
        L.append(f"## {task} ({' / '.join(classes)}): {tk['n_windows']} windows, {tk['n_drives']} drives"); L.append("")
        rows = []
        for sname, vs in tk["variants"].items():
            for name, m in vs.items():
                w, d = m["window"], m["drive"]
                rows.append([sname, name, f"{100*w['accuracy']:.1f}", f"{100*w['macro_f1']:.1f}", " / ".join(f"{c} {100*x:.1f}" for c, x in zip(classes, w["recall"])),
                             " / ".join(f"{c} {100*x:.1f}" for c, x in zip(classes, w["pred_share"])),
                             f"{100*d['accuracy']:.1f}", f"{100*d['macro_f1']:.1f}", " / ".join(f"{c} {100*x:.1f}" for c, x in zip(classes, d["recall"]))])
        L.append(T.md_table(["variant", "input treatment", "window acc %", "window F1 %", "window recall %", "window predicted share %", "drive acc %", "drive F1 %", "drive recall %"], rows))
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
