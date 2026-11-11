#!/usr/bin/env python3
"""Held-out stratum of commaCarSegments, original decode (`ccs`) against the understeer-corrected re-decode (`ccs_us`):
per held-out car model the device-level size accuracy and the model-level size decision before -> after, from the saved
ensemble probabilities of the full size model (probs__split_platform__size__gru__S9__ens__full.npz) and the window metas,
with the same rules as collect_scores.py (device = mean probability over the device's labelled windows -> arg-max; model =
mean probability over the model's labelled windows -> arg-max; truth per unit = majority label). The internal result of the
same model (internal_per_model.md, column `full`) is listed next to it. No device or route identifier is printed.

  source env.sh && $CPY experiments/transfer/us_heldout_compare.py [--orig ccs] [--us ccs_us] [--out experiments/transfer/us_heldout_compare.md]

RESULTS.md section 2.15, table (d).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.transfer import common as T  # noqa: E402

HERE = Path(__file__).resolve().parent
EXT = T.EXTERNAL_ROOT
CLASSES = ("SMALL", "MID", "LARGE")
PROBS = "probs__split_platform__size__gru__S9__ens__full.npz"


def load(ds: str):
    z = np.load(EXT / ds / "work" / PROBS, allow_pickle=True)
    m = pd.read_parquet(EXT / ds / "work" / "meta.parquet")
    if len(m) != len(z["y"]):
        raise SystemExit(f"{ds}: {len(m)} meta rows against {len(z['y'])} probability rows")
    keys = m["vehicle_id"].astype(str) + "|" + m["drive_id"].astype(str) + "|" + m["w_idx"].astype(str)
    zk = pd.Series(z["vehicle_id"]).astype(str) + "|" + pd.Series(z["drive_id"]).astype(str) + "|" + pd.Series(z["w_idx"]).astype(str)
    if not (keys.to_numpy() == zk.to_numpy()).all():
        raise SystemExit(f"{ds}: the probability file and the meta are not row-aligned")
    stratum = m["note"].astype(str).str.extract(r"stratum=([^;]+)")[0].fillna("-")
    return z["probs"].astype(np.float64), z["y"].astype(int), m["vehicle_id"].astype(str).to_numpy(), m["model"].astype(str).to_numpy(), stratum.to_numpy(), keys.to_numpy()


def unit_decisions(P, y, units):
    lab = y >= 0
    uid, inv = np.unique(units[lab], return_inverse=True)
    S = np.zeros((len(uid), P.shape[1])); np.add.at(S, inv, P[lab]); cnt = np.bincount(inv, minlength=len(uid))
    Pu = S / cnt[:, None]
    yu = np.array([np.bincount(y[lab][inv == u], minlength=P.shape[1]).argmax() for u in range(len(uid))])
    return uid, Pu, yu, Pu.argmax(1)


def internal_full() -> dict[str, str]:
    """{held-out model: 'drive acc / decision (P)'} of the full size model from internal_per_model.md."""
    f = HERE / "internal_per_model.md"
    out: dict[str, str] = {}
    if not f.exists():
        return out
    task, cols = None, None
    for line in f.read_text().splitlines():
        if line.startswith("## "):
            task, cols = line[3:].split(" (")[0].strip(), None
            continue
        if task != "size" or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "held-out model":
            cols = cells
            continue
        if cols is None or len(cells) != len(cols) or set(cells[0]) <= {"-"}:
            continue
        row = dict(zip(cols, cells))
        if re.match(r"(\d+) / (\w+) \(([\d.]+)\)( x)?$", row["full"]):
            out[row["held-out model"]] = row["full"].replace(" x", " miss")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="ccs")
    ap.add_argument("--us", default="ccs_us")
    ap.add_argument("--out", default=str(HERE / "us_heldout_compare.md"))
    a = ap.parse_args()
    Po, yo, vo, mo, so, ko = load(a.orig)
    Pu, yu, vu, mu, su, ku = load(a.us)
    if not (ko == ku).all() or not (yo == yu).all() or not (mo == mu).all() or not (so == su).all():
        raise SystemExit("the two sets are not aligned (rows, labels, models or strata differ)")
    ho = (so == "held-out") & (yo >= 0)
    ipm = internal_full()
    rows, recs = [], []
    tot = dict(dev_before=0, dev_after=0, dev=0, mod_before=0, mod_after=0, mod=0, rec=0, lost=0)
    for model in sorted(set(mo[ho])):
        sel = np.nonzero(ho & (mo == model))[0]
        yt = int(np.bincount(yo[sel], minlength=3).argmax()); lab = CLASSES[yt]
        pa, pb = Po[sel].mean(0), Pu[sel].mean(0)
        da, db = CLASSES[int(pa.argmax())], CLASSES[int(pb.argmax())]
        _, _, yua, dua = unit_decisions(Po[sel], yo[sel], vo[sel])
        _, _, yub, dub = unit_decisions(Pu[sel], yu[sel], vu[sel])
        ra, rb, n = int((dua == yua).sum()), int((dub == yub).sum()), int(len(yua))
        kind = "same" if da == db else ("recovered" if db == lab else ("lost" if da == lab else "wrong->wrong"))
        tot["dev_before"] += ra; tot["dev_after"] += rb; tot["dev"] += n; tot["mod"] += 1
        tot["mod_before"] += int(da == lab); tot["mod_after"] += int(db == lab)
        tot["rec"] += int(kind == "recovered"); tot["lost"] += int(kind == "lost")
        rows.append([model, lab, n, int(len(sel)), f"{ra}/{n} ({100 * ra / n:.1f}) -> {rb}/{n} ({100 * rb / n:.1f})", f"{100 * (rb - ra) / n:+.1f}",
                     f"{da} ({pa[yt]:.2f}) -> {db} ({pb[yt]:.2f})" + (" OK" if db == lab else " miss"), kind, ipm.get(model, "-")])
        recs.append(dict(model=model, label=lab, n_vehicles=n, n_windows=int(len(sel)), vehicles_right=[ra, rb], decision=[da, db],
                         p_true=[round(float(pa[yt]), 3), round(float(pb[yt]), 3)], kind=kind))
    hdr = ["held-out model", "label", "devices", "windows", "devices right (acc %) before -> after", "delta (points)",
           "model decision (P true) before -> after", "flip", "internal: drive acc / decision (P true)"]
    summary = (f"Held-out stratum, full size model, `{a.orig}` -> `{a.us}`: {tot['mod']} models; devices right {tot['dev_before']} -> {tot['dev_after']} of {tot['dev']} "
               f"({100 * tot['dev_before'] / tot['dev']:.1f} -> {100 * tot['dev_after'] / tot['dev']:.1f} %); models right {tot['mod_before']} -> {tot['mod_after']}; "
               f"recovered {tot['rec']}, lost {tot['lost']}.")
    text = "\n".join([f"# Held-out models of `{a.orig}` against `{a.us}` (full size model; us_heldout_compare.py)", "",
                      "Device = mean probability over the device's labelled windows -> arg-max; model = mean probability over the model's windows -> arg-max; "
                      "truth = majority label; `internal` = the lab's own test windows of that model (internal_per_model.md, column `full`).", "",
                      T.md_table(hdr, rows), "", summary, ""])
    Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
