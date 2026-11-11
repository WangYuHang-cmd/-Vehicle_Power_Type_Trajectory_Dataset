#!/usr/bin/env python3
"""Reviewer recomputation of the headline numbers of experiments/transfer/RESULTS.md straight from the saved ensemble
probability files (probs__<set>__ens__<mask>.npz) and the window metas (meta.parquet), independent of score.py /
collect_scores.py. Prints what it recomputes next to the figure quoted in RESULTS.md. No external identifiers are
printed (vehicle ids are only counted).

  source env.sh && $CPY experiments/transfer/review_recompute.py [--phase 1|2|3|all] [--raw-sign-check] [--no-windows] [--sections 20,22]

Phase 3b (sections 34-41, added by the phase-3b review on 10 Nov, evening; run with `--phase 3`, after sections 20-33): the two sets decoded
after the phase-3 write-up, `ccs_honda` (the nine recovered Honda platforms) and `comma2k19` (two cars on 2018 hardware): the decode
accounting against decode_summary.json / segments_qc.csv (and the budget list for the Honda platforms), every table of RESULTS.md 2.13
(a)-(e) and the results table of 2.14 re-rendered cell by cell and searched in the file, the figures of their Reading bullets and
caveats (the possible-hybrid split, the automation split, the masks, the variants), the updated pooled totals of sections 0 / 3.1 / 3.2
(phase-3 pairs over ccs + ccs_honda, all sets with the comma2k19 devices), the labels and strata of both sets, the window tensors
(the sample-0 and oncoming-lead artefacts, the identities), the identifier hygiene of both work folders, and the derived-yaw
understeer ratios of 2.14 against the output of comma2k19_yaw_check.py (run separately under the lab venv: yaw_check.json /
yaw_check_all.json in the comma2k19 work folder). Same log as phase 3.

Phase 3 (sections 20-33, added by the phase-3 review on 10 Nov): the commaCarSegments device-disjoint selection (`ccs`): the decode
accounting (decode_summary.json, segments_qc.csv, select_summary.json: skips by reason and platform, the nine Honda platforms, the
strata), the set description of 2.12, every table (a)-(h) of RESULTS.md 2.12 re-rendered cell by cell in the report's format and
searched in the file (FOUND / NOT FOUND), the figures of the Reading bullets (OK / MISMATCH), the pooled totals of sections 0 / 3.1 /
3.2 with the device x car-model pairs, the labels (footprints against the lab table and specs_external.csv with the 4.43 / 4.88 cuts,
strata against the drives index, powertrain against the lab platform table, the lab-library device check by intersection count only),
the internal references of 2.12 against internal_reference.json / internal_per_model.md, the window tensor identities and the
identifier hygiene of the phase-3 outputs. Log: <external>/logs/review_recompute_phase3.log.

Phase 2 (sections 10-19, added by the phase-2 review on 10 Nov): commaSteeringControl (`csc`) and MoreTorque (`moretorque`,
`moretorque_dasany`) numbers of RESULTS.md 0 / 2.10 / 2.11 / 3.1 / 3.2, the label mappings (strata against the drives index,
footprints against the lab table, hybrids excluded), the sign conventions (yaw vs steering, lead relative speed), the
598-599-row padding rule and the identifier hygiene of the phase-2 outputs. `--raw-sign-check` additionally reads a few
CSVs straight from two commaSteeringControl zips (zipfile + pandas, read-only) to show the raw lateral-acceleration sign.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.transfer import common as T  # noqa: E402

EXT = T.EXTERNAL_ROOT
TAG = {("bev_ice", "V2"): "transfer__bev_ice__resnet__V2", ("bev_ice", "V3"): "transfer__bev_ice__resnet__V3",
       ("bev_ice", "full"): "split_platform__bev_ice__resnet__B5", ("size", "V3"): "transfer__size__gru__V3",
       ("size", "V2"): "transfer__size__gru__V2", ("size", "full"): "split_platform__size__gru__S9",
       ("size", "V1"): "transfer__size__gru__V1", ("bev_ice", "V1"): "transfer__bev_ice__resnet__V1"}
PRIMARY = {  # dataset -> {task: variant key} (RESULTS.md 1.3 reading rules)
    "ultraav": {"bev_ice": "V2", "size": "V3"}, "openacc": {"bev_ice": "V2", "size": "V3"},
    "microsim_ev": {"bev_ice": "V2", "size": "V3"}, "uga_ev": {"bev_ice": "V2", "size": "V3"},
    "tesla_yuan": {"bev_ice": "V3", "size": "V3"}, "openacc_500x": {"bev_ice": "V3", "size": "V3"},
    "ved": {"bev_ice": "V3", "size": "V3"}, "a2d2": {"bev_ice": "full", "size": "full"}, "ccs_sample": {"bev_ice": "full", "size": "full"},
}


def load(ds: str, task: str, key: str, mask: str = "full"):
    z = np.load(EXT / ds / "work" / f"probs__{TAG[(task, key)]}__ens__{mask}.npz", allow_pickle=True)
    m = pd.read_parquet(EXT / ds / "work" / "meta.parquet")
    P = z["probs"].astype(np.float64)
    assert len(m) == len(P), (ds, len(m), len(P))
    assert (m["vehicle_id"].astype(str).to_numpy() == z["vehicle_id"].astype(str)).all(), "probs / meta misaligned"
    y_saved = z["y"]
    y = T.task_labels(m, task)
    assert (y == y_saved).all(), "saved y differs from the meta labels"
    return P, y, m, [str(c) for c in z["classes"]]


def unit_eval(P, y, units, k):
    lab = y >= 0
    if not lab.any():
        return dict(n=0, acc=np.nan, f1=np.nan, rec=np.full(k, np.nan), sup=np.zeros(k, int), cm=np.zeros((k, k), int))
    P, y, units = P[lab], y[lab], np.asarray(units)[lab].astype(str)
    uid, inv = np.unique(units, return_inverse=True)
    S = np.zeros((len(uid), k)); np.add.at(S, inv, P); S /= np.bincount(inv)[:, None]
    yu = np.array([np.bincount(y[inv == u], minlength=k).argmax() for u in range(len(uid))])
    pred = S.argmax(1)
    cm = np.zeros((k, k)); np.add.at(cm, (yu, pred), 1)
    tp = np.diag(cm); sup = cm.sum(1); pr = cm.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(sup > 0, tp / sup, np.nan)
        den = 2 * tp + (pr - tp) + (sup - tp)
        f1 = np.where(den > 0, 2 * tp / den, 0.0)
    has = sup > 0
    return dict(n=len(uid), acc=tp.sum() / cm.sum(), f1=(f1[has].mean() if has.sum() >= 2 else np.nan), rec=rec, sup=sup.astype(int), cm=cm.astype(int))


def pct(x):
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:.1f}"


def show(title, r, classes):
    print(f"  {title}: n={r['n']} acc={pct(r['acc'])} F1={pct(r['f1'])} recall " + " / ".join(f"{c} {pct(x)}" for c, x in zip(classes, r["rec"]))
          + f" support {r['sup'].tolist()}")


def phase1():
    print("## 1. OpenACC platoons, BEV vs ICE, V2 variant (RESULTS.md 2.2: window 68.2/67.6, drive 80.3/76.3 (93.9/55.5), vehicle 73.7/73.7 (100/58.3), model 72.2/72.1)")
    P, y, m, cl = load("openacc", "bev_ice", "V2")
    lab = y >= 0
    show("window", dict(n=int(lab.sum()), acc=(P[lab].argmax(1) == y[lab]).mean(), f1=unit_eval(P, y, np.arange(len(y)), 2)["f1"],
                        rec=unit_eval(P, y, np.arange(len(y)), 2)["rec"], sup=unit_eval(P, y, np.arange(len(y)), 2)["sup"]), cl)
    for lv, col in (("drive", "drive_id"), ("vehicle", "vehicle_id"), ("model", "model")):
        show(lv, unit_eval(P, y, m[col].to_numpy(), 2), cl)
    print("  labels outside the task (should be HEV only, shares):", m.loc[~lab, "powertrain"].value_counts().to_dict())
    print("  per-car decision (V2):")
    for model, g in m[lab].groupby("model"):
        idx = g.index.to_numpy()
        pm = P[idx].mean(0)
        print(f"    {model:32s} truth {cl[y[idx[0]]]:3s} P(BEV) {100 * pm[0]:.1f} -> {cl[pm.argmax()]} ({'OK' if pm.argmax() == y[idx[0]] else 'MISS'}); DAS {100 * g['das_frac'].mean():.1f} %")

    print("\n## 2. ULTra-AV, BEV vs ICE, V2 variant (RESULTS.md 2.1: window 72.1/63.3, drive 85.7/46.2, vehicle 75.0/69.7 (100/40), model 72.7/68.6)")
    P, y, m, cl = load("ultraav", "bev_ice", "V2")
    lab = y >= 0
    w = unit_eval(P, y, np.arange(len(y)), 2)
    show("window", dict(n=int(lab.sum()), acc=(P[lab].argmax(1) == y[lab]).mean(), f1=w["f1"], rec=w["rec"], sup=w["sup"]), cl)
    for lv, col in (("drive", "drive_id"), ("vehicle", "vehicle_id"), ("model", "model")):
        show(lv, unit_eval(P, y, m[col].to_numpy(), 2), cl)
    print("  labels outside the task:", m.loc[~lab, "powertrain"].value_counts().to_dict())

    print("\n## 3. VED, V3 variant (RESULTS.md 2.7: 258 of 264 ICE cars ICE (97.7 %); Leaf 16.4 % of 317 trips BEV, 0 of 3 cars; vehicle acc 96.6, F1 49.1)")
    P, y, m, cl = load("ved", "bev_ice", "V3")
    v = unit_eval(P, y, m["vehicle_id"].to_numpy(), 2)
    show("vehicle", v, cl)
    print(f"  ICE cars called ICE: {v['cm'][1, 1]} of {v['sup'][1]}; BEV cars called BEV: {v['cm'][0, 0]} of {v['sup'][0]}")
    leaf = (m["powertrain"] == "BEV").to_numpy()
    d = unit_eval(P, np.where(leaf, y, -1), m["drive_id"].to_numpy(), 2)
    print(f"  Leaf trips: {d['n']} , called BEV {d['cm'][0, 0]} ({pct(d['cm'][0, 0] / d['n'])} %)")
    print("  powertrain label counts (windows):", m["powertrain"].value_counts().to_dict(), "; windows in the task:", int((y >= 0).sum()))
    # Spearman weight bin vs ordinal score (size V3)
    Ps, ys, ms, cls = load("ved", "size", "V3")
    wb = pd.to_numeric(ms["note"].astype(str).str.extract(r"weight_bin_lb=(\d+)")[0], errors="coerce")
    score = (Ps * np.array([1, 2, 3])).sum(1)
    veh = pd.DataFrame({"v": ms["vehicle_id"], "wb": wb, "s": score}).groupby("v").agg(wb=("wb", "first"), s=("s", "mean")).dropna()
    print(f"  Spearman(weight bin, ordinal score) over vehicles: {veh[['wb', 's']].corr(method='spearman').iloc[0, 1]:.3f} (n = {len(veh)}); RESULTS.md: -0.125, n = 368")
    print(f"  Leaf size: 3 cars -> {unit_eval(Ps, np.where(leaf, ys, -1), ms['vehicle_id'].to_numpy(), 3)['cm'].tolist()} ; drive recall {pct(unit_eval(Ps, np.where(leaf, ys, -1), ms['drive_id'].to_numpy(), 3)['rec'][0])} (RESULTS.md 62.5)")

    print("\n## 4. OpenACC one-vehicle Fiat 500X, V3 (RESULTS.md 2.6: ICE recall window 87.9, 99.1 % of 429 trips, P(ICE) 81.8; size SMALL 46.2 % of trips, P 47.2/38.5/14.3)")
    P, y, m, cl = load("openacc_500x", "bev_ice", "V3")
    d = unit_eval(P, y, m["drive_id"].to_numpy(), 2)
    print(f"  window ICE recall {pct((P.argmax(1) == 1).mean())}; trips {d['n']}, ICE {d['cm'][1, 1]} ({pct(d['cm'][1, 1] / d['n'])} %); mean P(ICE) {100 * P[:, 1].mean():.1f}")
    Ps, ys, ms, cls = load("openacc_500x", "size", "V3")
    d = unit_eval(Ps, ys, ms["drive_id"].to_numpy(), 3)
    print(f"  size: trips SMALL {d['cm'][0, 0]} of {d['n']} ({pct(d['cm'][0, 0] / d['n'])} %), window SMALL recall {pct((Ps.argmax(1) == 0).mean())}, mean P {np.round(100 * Ps.mean(0), 1).tolist()}")

    print("\n## 5. Tesla Yuan, V3 (RESULTS.md 2.5: BEV window recall 62.3, 80.2 % of 106 trips (85), P(BEV) 60.8; size MID window 63.7, 78.3 % of trips (83), P 18.8/50.0/31.2)")
    P, y, m, cl = load("tesla_yuan", "bev_ice", "V3")
    d = unit_eval(P, y, m["drive_id"].to_numpy(), 2)
    print(f"  window BEV recall {pct((P.argmax(1) == 0).mean())}; trips {d['n']}, BEV {d['cm'][0, 0]} ({pct(d['cm'][0, 0] / d['n'])} %); mean P(BEV) {100 * P[:, 0].mean():.1f}")
    Ps, ys, ms, cls = load("tesla_yuan", "size", "V3")
    d = unit_eval(Ps, ys, ms["drive_id"].to_numpy(), 3)
    print(f"  size: trips MID {d['cm'][1, 1]} of {d['n']} ({pct(d['cm'][1, 1] / d['n'])} %), window MID recall {pct((Ps.argmax(1) == 1).mean())}, mean P {np.round(100 * Ps.mean(0), 1).tolist()}")
    print("  secondary (full + V3 mask) trips BEV:", end=" ")
    Pm, ym, mm, _ = load("tesla_yuan", "bev_ice", "full", "mask-V3")
    dm = unit_eval(Pm, ym, mm["drive_id"].to_numpy(), 2)
    print(f"{dm['cm'][0, 0]} of {dm['n']} (RESULTS.md: 0 of 106)")

    print("\n## 6. Pooled vehicle-level decisions, primary runs (RESULTS.md 3.1 / 3.2)")
    for task, k in (("bev_ice", 2), ("size", 3)):
        cls = T.TASK_LABEL[task][1]
        for scope, dss in (("all sets", list(PRIMARY)), ("without VED", [d for d in PRIMARY if d != "ved"]),
                           ("V2/V3 sets", [d for d in PRIMARY if d not in ("a2d2", "ccs_sample")])):
            hit = {c: [0, 0] for c in cls}
            for ds in dss:
                P, y, m, _ = load(ds, task, PRIMARY[ds][task])
                r = unit_eval(P, y, m["vehicle_id"].to_numpy(), k)
                if r["n"] == 0:
                    continue
                for i, c in enumerate(cls):
                    hit[c][0] += int(r["cm"][i, i]); hit[c][1] += int(r["sup"][i])
            print(f"  {task:8s} {scope:12s}: " + "  ".join(f"{c} {hit[c][0]}/{hit[c][1]}" for c in cls)
                  + f"  all {sum(h[0] for h in hit.values())}/{sum(h[1] for h in hit.values())}")
    print("  RESULTS.md: bev_ice all 23/27 BEV, 268/284 ICE, 291/311; without VED 23/24, 10/20, 33/44; V2/V3 sets 20/24, 268/283, 288/307")
    print("  RESULTS.md: size SMALL 25/27, MID 2/20, LARGE 0/13; V2/V3 sets SMALL 24/25, MID 1/19, LARGE 0/11")

    print("\n## 7. Hygiene of the labels in every set")
    for ds in PRIMARY:
        m = pd.read_parquet(EXT / ds / "work" / "meta.parquet")
        yb = T.task_labels(m, "bev_ice")
        out = m.loc[yb < 0, "powertrain"].value_counts().to_dict()
        bad_in = m.loc[yb >= 0, "powertrain"].unique().tolist()
        print(f"  {ds:13s}: windows {len(m):6d}, moving_frac min {m['moving_frac'].min():.3f}, das_frac mean {np.nanmean(m['das_frac']):.3f}, "
              f"in task {sorted(bad_in)}, excluded {out}, size labels {m['size_class'].fillna('none').value_counts().to_dict()}")

    print("\n## 8. Ensemble = mean of the member seeds (one check per task family)")
    for ds, tag, mask in (("openacc", "transfer__bev_ice__resnet__V2", "full"), ("ved", "transfer__size__gru__V3", "full"), ("a2d2", "split_platform__size__gru__S9", "full")):
        z = np.load(EXT / ds / "work" / f"probs__{tag}__ens__{mask}.npz", allow_pickle=True)
        mem = [np.load(EXT / ds / "work" / f"probs__{s}__{mask}.npz")["probs"] for s in z["members"]]
        print(f"  {ds}/{tag}: members {len(mem)}, max |ens - mean(members)| = {np.abs(z['probs'] - np.mean(mem, 0)).max():.2e}")

    print("\n## 9. Same car, ACC vs manual windows, OpenACC V2 (RESULTS.md 2.2: A6 71/76, X5 61/73, A-Class 49/71, XC40 77/47, Mazda 3 55/47, S-Max 42/47)")
    P, y, m, cl = load("openacc", "bev_ice", "V2")
    das = m["das_frac"].to_numpy() >= 0.5
    for model, g in m[y >= 0].groupby("model"):
        idx = g.index.to_numpy(); a = idx[das[idx]]; h = idx[~das[idx]]
        if len(a) >= 5 and len(h) >= 5:
            print(f"    {model:32s} P(BEV) ACC {100 * P[a, 0].mean():.0f} ({len(a)} w)  manual {100 * P[h, 0].mean():.0f} ({len(h)} w)")
    lab = y >= 0
    for name, sel in (("ACC", das), ("manual", ~das)):
        s = sel & lab
        pred = P[s].argmax(1); yy = y[s]
        print(f"  pooled {name}: ICE recall {pct((pred[yy == 1] == 1).mean())}, BEV recall {pct((pred[yy == 0] == 0).mean())}  (RESULTS.md ICE 46.4 / 62.0, BEV 86.3 / 86.2)")
    return 0


# ============================================================================= phase 2: csc + moretorque
TAG.update({("size", "mask"): "split_platform__size__gru__S9", ("bev_ice", "mask"): "split_platform__bev_ice__resnet__B5"})
PRIMARY2 = {"csc": {"bev_ice": "V3", "size": "V1"}, "moretorque": {"bev_ice": "full", "size": "full"}}   # RESULTS.md 1.3


def note_field(m: pd.DataFrame, key: str) -> pd.Series:
    return m["note"].astype(str).str.extract(rf"{key}=([^;]+)")[0]


def hits(r, classes) -> str:
    return ", ".join(f"{c} {int(r['cm'][i, i])}/{int(r['sup'][i])}" for i, c in enumerate(classes))


def levels(P, y, m, k, classes, label, units=(("window", None), ("drive", "drive_id"), ("vehicle", "vehicle_id"), ("model", "model"))):
    out = {}
    for lv, col in units:
        r = unit_eval(P, y, np.arange(len(y)) if col is None else m[col].to_numpy(), k)
        out[lv] = r
        show(f"{label:28s} {lv:8s}", r, classes)
    return out


def strata_eval(P, y, m, k, classes, label):
    strata = note_field(m, "stratum")
    for s in ("seen", "held-out", "absent"):
        sel = (strata == s).to_numpy()
        if not sel.any():
            continue
        w = unit_eval(P[sel], y[sel], np.arange(int(sel.sum())), k)
        v = unit_eval(P[sel], y[sel], m["vehicle_id"].to_numpy()[sel], k)
        mo = unit_eval(P[sel], y[sel], m["model"].to_numpy()[sel], k)
        print(f"     {label} {s:9s}: window {pct(w['acc'])} / {pct(w['f1'])}; vehicles {v['n']}: {pct(v['acc'])} / {pct(v['f1'])} [{hits(v, classes)}]; "
              f"models {mo['n']}: {int(np.trace(mo['cm']))}/{mo['n']} ({pct(mo['acc'])}) [{hits(mo, classes)}]")


def unit_table(P, y, m, k, classes, by="vehicle_id", extra=None):
    """mean P per unit, truth, decision; sorted by class then footprint."""
    rows = []
    for uid, g in m.groupby(by, sort=True):
        idx = g.index.to_numpy()
        pm = P[idx].mean(0)
        truth = int(y[idx[0]])
        rows.append((classes[truth], float(g["footprint_m2"].iloc[0]), uid, str(g["model"].iloc[0]), len(idx), pm, classes[int(pm.argmax())], truth == pm.argmax(),
                     str(note_field(g, "stratum").iloc[0]), extra(idx) if extra else ""))
    rows.sort(key=lambda r: (classes.index(r[0]), r[1]))
    return rows


def sec10_csc_size():
    print("## 10. commaSteeringControl size (RESULTS.md 2.10): V1 window 48.4/44.1 (68.1/29.3/35.1), archive 63.9/62.3 (88.2/30.8/66.7), "
          "no-borderline 65.6/60.9 (32: 17/10/5), model 64.0/64.3 (90.0/33.3/66.7); V3 window 47.0/42.5, archive 55.6/44.9 (100/0/50), model 52.0/43.1; "
          "full+V1 mask window 47.6/29.7 (94.8/4.9/9.6), archive 47.2/21.4 (100/0/0)")
    print("   strata (RESULTS.md): V1 seen 61.5/59.8 [S 10/11, M 3/10, L 3/5] models 11/18, window 48.1/44.7; held-out 70.0/72.3 [5/6, 1/3, 1/1] models 5/7, window 49.1/41.8;"
          " V3 seen 53.8/46.6 [11/11, 0/10, 3/5] 9/18, window 45.2/42.9; held-out 60.0/25.0 [6/6, 0/3, 0/1] 4/7, window 52.1/38.2")
    tabs = {}
    for key, mask, label in (("V1", "full", "V1 variant"), ("V3", "full", "V3 variant"), ("mask", "mask-V1", "full + V1 mask")):
        P, y, m, cl = load("csc", "size", key, mask)
        r = levels(P, y, m, 3, cl, label, units=(("window", None), ("archive", "vehicle_id"), ("model", "model")))
        print("     archives right per class:", hits(r["archive"], cl), "; models right per class:", hits(r["model"], cl))
        if key == "V1":
            nb = ~m["borderline"].to_numpy(bool)
            show(f"{'V1 archive, no borderline':28s} archive ", unit_eval(P[nb], y[nb], m["vehicle_id"].to_numpy()[nb], 3), cl)
            print(f"     borderline archives: {sorted(m.loc[~nb, 'vehicle_id'].unique().tolist())}")
        strata_eval(P, y, m, 3, cl, label)
        tabs[key] = unit_table(P, y, m, 3, cl)
    print("  per archive, V1 size (truth, footprint, archive, n, P(S, M, L) %, decision) | V3 size decision:")
    v3 = {r[2]: r for r in tabs["V3"]}
    for r in tabs["V1"]:
        print(f"    {r[0]:5s} {r[1]:.2f} {r[2]:36s} {r[8]:8s} n={r[4]:5d} P=({', '.join(f'{100 * x:.1f}' for x in r[5])}) -> {r[6]:5s} {'OK  ' if r[7] else 'MISS'} | V3 {v3[r[2]][6]:5s} {'OK' if v3[r[2]][7] else 'MISS'}")


def sec11_csc_pt():
    print("\n## 11. commaSteeringControl BEV vs ICE (RESULTS.md 2.10): V3 window 75.4/69.0 (63.9/78.9), archive 97.2/96.1 (88.9/100), model 96.0/95.2 (87.5/100); "
          "V1 window 79.4/61.7 (24.4/96.1), archive 75.0/42.9 (0/100), model 68.0/40.5; full+V1 mask window 77.8/57.3 (18.3/95.8), archive 75.0/42.9")
    print("   strata (RESULTS.md): V3 seen 100/100 [4/4, 22/22] 18/18, window 78.6/68.7; held-out 90.0/89.9 [4/5, 5/5] 6/7, window 66.6/65.7; "
          "V1 seen 84.6/45.8 [0/4, 22/22] 14/18, window 85.6/66.4; held-out 50.0/33.3 [0/5, 5/5] 3/7, window 62.1/51.4")
    tabs = {}
    for key, mask, label in (("V3", "full", "V3 variant"), ("V1", "full", "V1 variant"), ("mask", "mask-V1", "full + V1 mask")):
        P, y, m, cl = load("csc", "bev_ice", key, mask)
        r = levels(P, y, m, 2, cl, label, units=(("window", None), ("archive", "vehicle_id"), ("model", "model")))
        print("     archives right per class:", hits(r["archive"], cl), "; models right per class:", hits(r["model"], cl))
        strata_eval(P, y, m, 2, cl, label)
        tabs[key] = unit_table(P, y, m, 2, cl)
    print("  per archive (truth, archive, P(BEV) % V3 -> decision | V1 -> decision | full+V1 mask -> decision):")
    v1 = {r[2]: r for r in tabs["V1"]}; mk = {r[2]: r for r in tabs["mask"]}
    for r in tabs["V3"]:
        a = r[2]
        print(f"    {r[0]:3s} {a:36s} {r[8]:8s} V3 {100 * r[5][0]:5.1f} -> {r[6]:3s} {'OK  ' if r[7] else 'MISS'} | V1 {100 * v1[a][5][0]:5.1f} -> {v1[a][6]:3s} {'OK  ' if v1[a][7] else 'MISS'} | mask {100 * mk[a][5][0]:5.1f} -> {mk[a][6]:3s} {'OK' if mk[a][7] else 'MISS'}")


def _mt_runs(task):
    return [("full", "full", "full 20-ch"), ("V1", "full", "V1 variant"), ("V2", "full", "V2 variant"), ("V3", "full", "V3 variant"),
            ("mask", "mask-V1", "full + V1 mask"), ("mask", "mask-V2", "full + V2 mask"), ("mask", "mask-V3", "full + V3 mask"),
            ("mask", "mask-lead", "full + lead mask"), ("mask", "mask-yaw", "full + yaw mask"), ("mask", "mask-3ch-b54e96", "full + torque/pedal mask")]


def sec12_moretorque():
    print("\n## 12. MoreTorque (RESULTS.md 2.11). Size full: window 43.4/41.1 (81.1/13.4), drive 42.7/37.6 (85.6/9.8), vehicle 66.7/40.0 (100/0), model 50.0/33.3; "
          "V1 window 44.2/41.7, drive 40.7/35.1, vehicle 58.3/36.8 (87.5/0); V2 window 27.9/31.7, vehicle 41.7/31.2 (62.5/0); V3 window 36.2/34.0, drive 36.8/30.9, vehicle 66.7/40.0; "
          "mask-V1 window 51.5/49.4 (88.4/22.0), vehicle 66.7/40.0; mask-V2 window 44.2/39.1; mask-V3 window 43.4/33.1 (93.6/3.4); yaw mask window acc 24.3")
    print("   Powertrain full: window 72.4/45.2 (8.0/81.3), drive 85.9/48.5 (4.5/92.7), vehicle 91.7/47.8 (0/100), model 75.0/42.9; V1 window 74.6/46.5 (8.7/83.8), drive 84.6/48.0 (4.5/91.3); "
          "V2 window 77.7/55.1 (27.7/84.6), drive 87.5/55.7 (18.0/93.3); V3 window 73.3/54.9 (38.9/78.1), drive 80.6/51.8 (21.3/85.6); masks V1/V2/V3 drive 86.0/88.6/88.4, F1 47.4/48.4/46.9, BEV 2.2/2.2/0.0, ICE 93.0/95.8/95.8")
    for task, k in (("size", 3), ("bev_ice", 2)):
        for key, mask, label in _mt_runs(task):
            P, y, m, cl = load("moretorque", task, key, mask)
            levels(P, y, m, k, cl, f"{task} {label}")
    # per platform and per car, full models
    print("  per platform / per car, full models (RESULTS.md per-platform table: Sonata P(S,M,L) 52/23/25 drive acc 13.8, P(BEV) 31 drive acc 90.6; CX-50 64/22/14 6.4, 11 95.5; "
          "Mazda 3 75/18/7 91.7 7/7 cars, 24 91.7; Omoda 55/24/20 57.3, 20 4.5; variants P(MID) V1/V3: Sonata 28/24, CX-50 23/27, Mazda 3 66/53, Omoda 59/42; P(BEV) V1/V2/V3: 37/30/32, 8/18/30, 6/20/22, 17/34/41)")
    Ps, ys, m, cls = load("moretorque", "size", "full", "full")
    Pb, yb, _, clb = load("moretorque", "bev_ice", "full", "full")
    var = {v: (load("moretorque", "size", v, "full")[0], load("moretorque", "bev_ice", v, "full")[0]) for v in ("V1", "V2", "V3")}
    lat = pd.to_numeric(note_field(m, "lat_only"), errors="coerce").fillna(0.0).to_numpy(float)
    for model, g in m.groupby("model", sort=True):
        idx = g.index.to_numpy()
        ds = unit_eval(Ps[idx], ys[idx], g["drive_id"].to_numpy(), 3); db = unit_eval(Pb[idx], yb[idx], g["drive_id"].to_numpy(), 2)
        vs = unit_eval(Ps[idx], ys[idx], g["vehicle_id"].to_numpy(), 3); vb = unit_eval(Pb[idx], yb[idx], g["vehicle_id"].to_numpy(), 2)
        ps, pb = Ps[idx].mean(0), Pb[idx].mean(0)
        print(f"    {model:14s} cars {g['vehicle_id'].nunique():2d} drives {g['drive_id'].nunique():4d} win {len(idx):5d} | size P=({', '.join(f'{100 * x:.1f}' for x in ps)}) -> {cls[ps.argmax()]} "
              f"drive acc {pct(ds['acc'])} cars right {int(np.trace(vs['cm']))}/{vs['n']} | pt P(BEV) {100 * pb[0]:.1f} -> {clb[pb.argmax()]} drive acc {pct(db['acc'])} cars right {int(np.trace(vb['cm']))}/{vb['n']}"
              f" | P(MID) V1/V3 {100 * var['V1'][0][idx].mean(0)[1]:.1f}/{100 * var['V3'][0][idx].mean(0)[1]:.1f}; V1 size cars right {int(np.trace(unit_eval(var['V1'][0][idx], ys[idx], g['vehicle_id'].to_numpy(), 3)['cm']))}/{vs['n']}"
              f" | P(BEV) V1/V2/V3 {100 * var['V1'][1][idx].mean(0)[0]:.1f}/{100 * var['V2'][1][idx].mean(0)[0]:.1f}/{100 * var['V3'][1][idx].mean(0)[0]:.1f}")
    print("  per car (RESULTS.md per-car table):")
    for vid, g in m.groupby("vehicle_id", sort=True):
        idx = g.index.to_numpy()
        ds = unit_eval(Ps[idx], ys[idx], g["drive_id"].to_numpy(), 3); db = unit_eval(Pb[idx], yb[idx], g["drive_id"].to_numpy(), 2)
        ps, pb = Ps[idx].mean(0), Pb[idx].mean(0)
        print(f"    {vid} {g['model'].iloc[0]:14s} {g['size_class'].iloc[0]:5s}/{g['powertrain'].iloc[0]:3s} {note_field(g, 'stratum').iloc[0]:6s} drives {g['drive_id'].nunique():3d} win {len(idx):5d} "
              f"DAS {100 * g['das_frac'].mean():.0f} lat-only {100 * lat[idx].mean():.0f} | size ({', '.join(f'{100 * x:.0f}' for x in ps)}) -> {cls[ps.argmax()]} {'OK' if ps.argmax() == ys[idx[0]] else 'MISS'} drive acc {pct(ds['acc'])} | pt P(BEV) {100 * pb[0]:.0f} -> {clb[pb.argmax()]} {'OK' if pb.argmax() == yb[idx[0]] else 'MISS'} drive acc {pct(db['acc'])}")
    # masks (RESULTS.md sensitivity paragraph)
    print("  sensitivity (RESULTS.md: lead mask Sonata P(ICE) 0.71, CX-50 0.90, Omoda P(BEV) 0.21, Mazda 3 SMALL 7/7; torque+pedal mask Omoda P(BEV) 0.20, Sonata P(MID) 0.26, CX-50 0.13; "
          "yaw mask Sonata LARGE 0.47, CX-50 LARGE 0.48, Mazda 3 SMALL 4/7 (P(SMALL) 0.43), Omoda SMALL 0.41, window acc 24.3, Omoda P(BEV) 0.22)")
    for mask in ("mask-lead", "mask-3ch-b54e96", "mask-yaw"):
        Pm, _, _, _ = load("moretorque", "size", "mask", mask); Pbm, _, _, _ = load("moretorque", "bev_ice", "mask", mask)
        parts = []
        for model, g in m.groupby("model", sort=True):
            idx = g.index.to_numpy(); ps = Pm[idx].mean(0); pb = Pbm[idx].mean(0)
            vs = unit_eval(Pm[idx], ys[idx], g["vehicle_id"].to_numpy(), 3)
            parts.append(f"{model}: size ({', '.join(f'{x:.2f}' for x in ps)}) -> {cls[ps.argmax()]} cars {int(np.trace(vs['cm']))}/{vs['n']}, P(BEV) {pb[0]:.2f}")
        print(f"    {mask:16s} window size acc {pct((Pm.argmax(1) == ys).mean())}: " + " | ".join(parts))
    # automation state and speed regime (needs lat_only from the note and X for the mean speed)
    das = m["das_frac"].to_numpy(float)
    state = np.where(das >= 0.5, "engaged", np.where(lat >= 0.5, "lateral-only", "human"))
    print("  automation state, full models (RESULTS.md: Sonata human 234/29.1/0.35/0.08, lateral-only 839/25.1/0.28/0.07, engaged 1,853/6.5/0.19/0.45; CX-50 356/12.1/0.22/0.11, 568/16.2/0.26/0.07, 147/0.0/0.07/0.27; "
          "Mazda 3 316/76.3/0.29/0.13, 293/84.6/0.23/0.27, 1,710/90.4/0.15/0.26; Omoda 614/63.7/0.26/0.22, 9/88.9/0.07/0.33, 251/62.5/0.22/0.14)")
    for model, g in m.groupby("model", sort=True):
        parts = []
        for st in ("human", "lateral-only", "engaged"):
            idx = g.index[state[g.index.to_numpy()] == st].to_numpy()
            if len(idx):
                parts.append(f"{st} n={len(idx)} acc {pct((Ps[idx].argmax(1) == ys[idx]).mean())} P(MID) {Ps[idx][:, 1].mean():.2f} P(BEV) {Pb[idx][:, 0].mean():.2f}")
        print(f"    {model:14s} " + " | ".join(parts))
    if not ARGS.no_windows:
        X, _ = T.load_windows("moretorque")
        vmean = np.nanmean(X[:, :, T.CH["v_mps"]], axis=1)
        print("  P(MID) by window mean speed 0-10 / 10-20 / 20-30 / 30+ (RESULTS.md: Sonata 0.37/0.29/0.20/0.18, CX-50 0.30/0.21/0.14/0.18, Mazda 3 0.30/0.21/0.12/0.16, Omoda 0.34/0.08/0.11/-)")
        for model, g in m.groupby("model", sort=True):
            cells = []
            for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 99)):
                idx = g.index[(vmean[g.index.to_numpy()] >= lo) & (vmean[g.index.to_numpy()] < hi)].to_numpy()
                cells.append(f"{Ps[idx][:, 1].mean():.2f} (n={len(idx)})" if len(idx) >= 10 else f"- (n={len(idx)})")
            print(f"    {model:14s} " + " / ".join(cells))
        sec16_moretorque_signs(X, m)
        del X
    # das any re-cut
    print("  --das any re-cut (RESULTS.md: das_frac 91 % Sonata, 87 % Mazda 3, 63 % CX-50, 31 % Omoda; Sonata P(MID) 0.23, CX-50 0.32 (drive acc 23.2), Mazda 3 SMALL 6/7 cars, Omoda P(BEV) 0.19; window acc 42.4 size / 71.6 pt)")
    Pd, yd, md, _ = load("moretorque_dasany", "size", "full", "full"); Pdb, ydb, _, _ = load("moretorque_dasany", "bev_ice", "full", "full")
    print(f"    window acc size {pct((Pd.argmax(1) == yd).mean())}, pt {pct((Pdb.argmax(1) == ydb).mean())}; windows {len(md)} (moretorque {len(m)})")
    for model, g in md.groupby("model", sort=True):
        idx = g.index.to_numpy(); ps = Pd[idx].mean(0); pb = Pdb[idx].mean(0)
        ds = unit_eval(Pd[idx], yd[idx], g["drive_id"].to_numpy(), 3); vs = unit_eval(Pd[idx], yd[idx], g["vehicle_id"].to_numpy(), 3)
        print(f"    {model:14s} das_frac {100 * g['das_frac'].mean():.0f} % | size P=({', '.join(f'{x:.2f}' for x in ps)}) drive acc {pct(ds['acc'])} cars right {int(np.trace(vs['cm']))}/{vs['n']} | P(BEV) {pb[0]:.2f}")


def sec13_pooled_all():
    print("\n## 13. Pooled vehicle-level decisions over every set, primary runs (RESULTS.md 0 / 3.1 / 3.2: BEV 31/37, ICE 306/322, all 337/359; SMALL 48/52, MID 6/37, LARGE 4/19)")
    prim = dict(PRIMARY); prim.update(PRIMARY2)
    for task, k in (("bev_ice", 2), ("size", 3)):
        cls = T.TASK_LABEL[task][1]
        for scope, dss in (("csc only", ["csc"]), ("moretorque only", ["moretorque"]), ("all sets", list(prim))):
            hit = {c: [0, 0] for c in cls}
            for ds in dss:
                P, y, m, _ = load(ds, task, prim[ds][task])
                r = unit_eval(P, y, m["vehicle_id"].to_numpy(), k)
                if r["n"] == 0:
                    continue
                for i, c in enumerate(cls):
                    hit[c][0] += int(r["cm"][i, i]); hit[c][1] += int(r["sup"][i])
            print(f"  {task:8s} {scope:16s}: " + "  ".join(f"{c} {hit[c][0]}/{hit[c][1]}" for c in cls) + f"  all {sum(h[0] for h in hit.values())}/{sum(h[1] for h in hit.values())}")


def sec14_labels():
    print("\n## 14. Labels: strata vs the drives index, footprints vs the lab table, hybrids, borderline, counts")
    d = pd.read_parquet(T.DATA_ROOT / "build" / "index" / "drives.parquet", columns=["model", "split_platform"])
    d = d[d["model"].notna() & (d["model"] != "Unknown")]
    splits = d.groupby("model")["split_platform"].agg(lambda s: set(s.dropna()))
    seen = {k for k, v in splits.items() if v & {"train", "validation"}}; held = {k for k, v in splits.items() if "test" in v}
    print(f"  drives index: {len(seen)} seen models, {len(held)} held-out (test) models, models in both: {sorted(seen & held)}")
    specs = pd.read_csv(REPO / "vtt" / "specs_data" / "platform_specs.csv", dtype=str, keep_default_na=False).set_index("platform")
    for ds in ("csc", "moretorque"):
        m = pd.read_parquet(EXT / ds / "work" / "meta.parquet")
        strata = note_field(m, "stratum")
        exp = m["model"].map(lambda x: "seen" if x in seen else ("held-out" if x in held else "absent"))
        bad = m.loc[strata != exp, ["vehicle_id", "model"]].drop_duplicates()
        print(f"  {ds}: windows {len(m)}, vehicles {m['vehicle_id'].nunique()}, models {m['model'].nunique()}, hours {len(m) / 60:.1f}; strata (vehicles) {m.assign(s=strata).groupby('s')['vehicle_id'].nunique().to_dict()}, "
              f"stratum mismatches vs the drives index: {len(bad)}" + (f" {bad.values.tolist()}" if len(bad) else ""))
        fp = m["footprint_m2"].to_numpy(float)
        cls_ok = (T.classify_footprint(fp) == m["size_class"].to_numpy(object)).all()
        bl_ok = (T.is_borderline(fp) == m["borderline"].to_numpy(bool)) | m["borderline"].to_numpy(bool)
        print(f"    size_class == thresholds({T.FOOTPRINT_THRESHOLDS if hasattr(T, 'FOOTPRINT_THRESHOLDS') else '4.43/4.88'}) on every window: {cls_ok}; borderline flags: {sorted(m.loc[m['borderline'], 'vehicle_id'].unique().tolist())}; "
              f"strict-rule borderline (|fp - cut| < 0.05): {sorted(m.loc[T.is_borderline(fp), 'vehicle_id'].unique().tolist())}; windows with a borderline flag but not by the strict rule: {int((m['borderline'].to_numpy(bool) & ~T.is_borderline(fp)).sum())} ({sorted(m.loc[m['borderline'].to_numpy(bool) & ~T.is_borderline(fp), 'vehicle_id'].unique().tolist())})")
        print(f"    powertrain vocabulary: {sorted(m['powertrain'].unique().tolist())}; per class windows: size {m['size_class'].value_counts().to_dict()}, powertrain {m['powertrain'].value_counts().to_dict()}")
        per_v = m.groupby("vehicle_id").agg(model=("model", "nunique"), fp=("footprint_m2", "nunique"), sc=("size_class", "nunique"), pt=("powertrain", "nunique"))
        print(f"    vehicles with more than one model / footprint / class / powertrain: {int((per_v > 1).any(axis=1).sum())}")
        # footprint vs lab table for platforms that exist there
        diffs = []
        for plat, g in m.groupby("platform"):
            if plat in specs.index:
                r = specs.loc[plat]
                try:
                    lab_fp = float(r["wheelbase_mm"]) / 1000 * (float(r["track_front_mm"]) + float(r["track_rear_mm"])) / 2 / 1000
                except ValueError:
                    continue
                diffs.append((plat, lab_fp, float(g["footprint_m2"].iloc[0])))
        worst = max(diffs, key=lambda t: abs(t[1] - t[2])) if diffs else None
        print(f"    footprint == lab table (wheelbase x mean track) for {len(diffs)} platforms in the lab table; max |diff| {abs(worst[1] - worst[2]):.4f} m^2 ({worst[0]})" if worst else "    no platform in the lab table")
        if ds == "csc":
            names = sorted(m["vehicle_id"].unique().tolist())
            hyb = [n for n in names if any(t in n.upper() for t in ("HYBRID", "PHEV", "PRIME", "MIRAI"))]
            print(f"    archives with a hybrid token in the name: {hyb}; BEV archives: {sorted(m.loc[m['powertrain'] == 'BEV', 'vehicle_id'].unique().tolist())}")
            print(f"    models per class: " + "; ".join(f"{c} {m.loc[m['size_class'] == c, 'model'].nunique()}" for c in T.SIZE_CLASSES) + "; " + "; ".join(f"{c} {m.loc[m['powertrain'] == c, 'model'].nunique()}" for c in ("BEV", "ICE")))
            print(f"    archives per class: " + "; ".join(f"{c} {m.loc[m['size_class'] == c, 'vehicle_id'].nunique()}" for c in T.SIZE_CLASSES) + "; " + "; ".join(f"{c} {m.loc[m['powertrain'] == c, 'vehicle_id'].nunique()}" for c in ("BEV", "ICE")))
            pl = note_field(m, "pt_lab")
            print(f"    archives whose lab platform label is not the archive powertrain (pt_lab != powertrain): {sorted(m.loc[pl != m['powertrain'], 'vehicle_id'].unique().tolist())}")
            print(f"    one window per drive: {(m.groupby('drive_id').size() == 1).all()}; das_frac mean {m['das_frac'].mean():.3f}, moving_frac min {m['moving_frac'].min():.3f}")
        else:
            print(f"    channels_present per model: " + "; ".join(f"{mo} {g['channels_present'].str.count(',').iloc[0] + 1}" for mo, g in m.groupby('model')))
            print(f"    drives with windows {m['drive_id'].nunique()}, single-window drives {(m.groupby('drive_id').size() == 1).sum()}, median windows per drive {m.groupby('drive_id').size().median():.0f}; "
                  f"das_frac mean per model " + ", ".join(f"{mo} {g['das_frac'].mean():.3f}" for mo, g in m.groupby("model")))
            lat = pd.to_numeric(note_field(m, "lat_only"), errors="coerce")
            print(f"    lat_only (window mean) per model " + ", ".join(f"{mo} {lat[g.index].mean():.3f}" for mo, g in m.groupby("model")))


def sec15_csc_windows():
    print("\n## 15. commaSteeringControl windows: padding rule (598-599 rows -> last row repeated), sign of yaw vs steering, derived-channel identities")
    m = pd.read_parquet(EXT / "csc" / "work" / "meta.parquet")
    pad = pd.to_numeric(m["note"].str.extract(r"padded (\d+)")[0], errors="coerce").fillna(0).astype(int).to_numpy()
    trim = pd.to_numeric(m["note"].str.extract(r"trimmed (\d+)")[0], errors="coerce").fillna(0).astype(int).to_numpy()
    print(f"  padded windows {int((pad > 0).sum())} ({pct((pad > 0).mean())} %; by 1: {int((pad == 1).sum())}, by 2: {int((pad == 2).sum())}), trimmed {int((trim > 0).sum())} (RESULTS.md: 26,202 = 50.9 %, 25,770 / 432; 320)")
    inv = pd.read_csv(EXT / "csc" / "work" / "csc_inventory.csv")
    import json as _json
    from collections import Counter as _C
    tot = _C()
    for h in inv["rows_hist"]:
        tot.update({int(k): v for k, v in _json.loads(h).items()})
    elig = sum(v for k, v in tot.items() if k >= 598)
    print(f"  inventory: {sum(tot.values()):,} segments, eligible {elig:,}; shares of the eligible: 598 rows {pct(tot[598] / elig)} %, 599 {pct(tot[599] / elig)} %, 600 {pct(tot[600] / elig)} %, 601 {pct(tot[601] / elig)} %, > 601 {pct(sum(v for k, v in tot.items() if k > 601) / elig)} %"
          f" -> expected padded share of a random sample {pct((tot[598] + tot[599]) / elig)} % (observed {pct((pad > 0).mean())} %)")
    if ARGS.no_windows:
        print("  (window tensor checks skipped: --no-windows)")
        return
    X, m2 = T.load_windows("csc")
    assert len(m2) == len(m)
    v = X[:, :, T.CH["v_mps"]]; a = X[:, :, T.CH["a_mps2"]]; ang = X[:, :, T.CH["steer_angle_deg"]]; lat = X[:, :, T.CH["lat_acc_mps2"]]
    yaw = X[:, :, T.CH["yaw_rate_radps"]]; curv = X[:, :, T.CH["curv_1pm"]]; rate = X[:, :, T.CH["steer_rate_degps"]]
    for k in (1, 2):
        idx = np.flatnonzero(pad == k)
        if len(idx) == 0:
            continue
        same = np.ones(len(idx), bool)
        for ch in (v, a, ang, lat, X[:, :, T.CH["steer_pressed"]], X[:, :, T.CH["das_enabled"]]):
            tail = ch[idx][:, 600 - k - 1:]
            same &= np.all(tail == tail[:, :1], axis=1)
        print(f"  padded {k}: last {k + 1} rows identical in v / a / angle / lat_acc / flags on {int(same.sum())} of {len(idx)} windows; derived rate at the last row = 0 on {int((rate[idx][:, -1] == 0).sum())} of {len(idx)}")
    idx = np.flatnonzero(pad == 0)[:2000]
    same = np.all(v[idx][:, -2:] == v[idx][:, -1:], axis=1) & np.all(ang[idx][:, -2:] == ang[idx][:, -1:], axis=1)
    print(f"  unpadded sample ({len(idx)} windows): last two rows identical in v and angle on {int(same.sum())} (chance coincidences only)")
    # identities
    mv = v >= 1.0
    print(f"  yaw NaN exactly where v < 1 m/s: {bool(np.array_equal(np.isnan(yaw), ~mv))}; curv NaN likewise: {bool(np.array_equal(np.isnan(curv), ~mv))}")
    print(f"  max |yaw * v - lat_acc| where defined: {np.nanmax(np.abs(yaw * v - lat)):.2e}; max |curv * v - yaw|: {np.nanmax(np.abs(curv * v - yaw)):.2e}; max |rate - gradient(angle) x 10| (interior): {np.abs(rate[:, 1:-1] - (ang[:, 2:] - ang[:, :-2]) * 5).max():.2e}")
    print(f"  standstill == (v < 0.3): {bool(np.array_equal(X[:, :, T.CH['standstill']], (v < 0.3).astype(np.float32)))}; lead_valid all 0: {bool((X[:, :, T.CH['lead_valid']] == 0).all())}; "
          f"absent channels all NaN: {all(np.isnan(X[:, :, T.CH[c]]).all() for c in ('steer_torque_norm', 'gas_pressed', 'brake_pressed', 'lead_d_m', 'lead_v_rel_mps', 'lead_speed_mps', 'lead_a_slope_mps2', 'thw_s', 'ttc_s'))}")
    # sign per archive
    cors = []
    for arch, g in m.groupby("vehicle_id", sort=True):
        idx = g.index.to_numpy()
        sel = (v[idx] > 5.0) & np.isfinite(yaw[idx])
        cors.append((arch, float(np.corrcoef(ang[idx][sel], yaw[idx][sel])[0, 1]), float(np.corrcoef(ang[idx][sel], lat[idx][sel])[0, 1])))
    c = np.array([x[1] for x in cors])
    print(f"  corr(steering angle, yaw) above 5 m/s per archive: min {c.min():.3f} ({cors[int(c.argmin())][0]}), median {np.median(c):.3f}, max {c.max():.3f}; all positive (left-positive yaw): {bool((c > 0).all())} (RESULTS.md / DS_CSC.md: min 0.84, median 0.91)")
    print(f"  speed: mean {v.mean():.2f} m/s, below 0.3 m/s {pct((v < 0.3).mean())} %, max {v.max():.1f}; das_frac mean {X[:, :, T.CH['das_enabled']].mean():.3f}; a_mps2 std {a.std():.3f}, |a| > 8 on {int((np.abs(a) > 8).sum())} samples")
    del X


def sec16_moretorque_signs(X, m):
    print("  MoreTorque window tensor: yaw sign, lead identities (lead_v_rel = lead minus ego; lead_speed = v + v_rel; THW = d / v), curvature")
    v = X[:, :, T.CH["v_mps"]]; ang = X[:, :, T.CH["steer_angle_deg"]]; yaw = X[:, :, T.CH["yaw_rate_radps"]]; lat = X[:, :, T.CH["lat_acc_mps2"]]; curv = X[:, :, T.CH["curv_1pm"]]
    lv = X[:, :, T.CH["lead_valid"]]; d = X[:, :, T.CH["lead_d_m"]]; vr = X[:, :, T.CH["lead_v_rel_mps"]]; ls = X[:, :, T.CH["lead_speed_mps"]]; thw = X[:, :, T.CH["thw_s"]]; ttc = X[:, :, T.CH["ttc_s"]]
    cors = []
    for vid, g in m.groupby("vehicle_id", sort=True):
        idx = g.index.to_numpy(); sel = (v[idx] > 3.0) & np.isfinite(yaw[idx])
        cors.append((vid, float(np.corrcoef(ang[idx][sel], yaw[idx][sel])[0, 1])))
    c = np.array([x[1] for x in cors])
    print(f"    corr(steering angle, yaw) above 3 m/s per car: min {c.min():.3f}, median {np.median(c):.3f}, max {c.max():.3f}; all positive: {bool((c > 0).all())} (DS_MORETORQUE.md: +0.87 to +0.98)")
    ok = lv > 0.5
    print(f"    lead values NaN exactly where lead_valid = 0: {bool(np.array_equal(np.isnan(d), ~ok)) and bool(np.array_equal(np.isnan(vr), ~ok))}; max |lead_speed - (v + v_rel)| {np.nanmax(np.abs(ls - (v + vr))):.2e}; "
          f"max |thw - d / v| where thw defined {np.nanmax(np.abs(thw - d / v)):.2e}; ttc defined only while closing (v_rel < -0.1): {bool(np.all(vr[np.isfinite(ttc)] < -0.1))}; max |ttc - d / (-v_rel)| {np.nanmax(np.abs(ttc - d / (-vr))):.2e}")
    # sign of the relative speed: over a 1-s lag inside a valid lead run the gap should grow when v_rel > 0 (lead faster than ego)
    L = 10
    n1 = ok.shape[1] - L
    run = np.ones((ok.shape[0], n1), bool)
    for k in range(L + 1):
        run &= ok[:, k:k + n1]
    dd = (d[:, L:] - d[:, :-L]) / (L / 10.0)
    vm = np.nanmean(np.stack([vr[:, k:k + n1] for k in range(L + 1)]), axis=0)
    sel = run & np.isfinite(dd) & np.isfinite(vm) & (np.abs(dd) < 15)
    big = sel & (np.abs(vm) > 1.0)
    print(f"    1-s lag inside valid lead runs ({int(sel.sum()):,} samples): corr(d(lead_d)/dt, lead_v_rel) {np.corrcoef(dd[sel], vm[sel])[0, 1]:.3f}, LS slope {np.polyfit(vm[sel], dd[sel], 1)[0]:.3f} (+1 = lead minus ego, vision lead noisy), "
          f"sign agreement when |v_rel| > 1 m/s {np.mean(np.sign(dd[big]) == np.sign(vm[big])):.3f} ({int(big.sum()):,} samples)")
    mv = (v >= 1.0) & np.isfinite(yaw)
    print(f"    max |curv * v - yaw| where curv defined: {np.nanmax(np.abs(curv * v - yaw)):.2e}; curv NaN wherever v < 1 or yaw NaN: {bool(np.all(np.isnan(curv[~mv])))}; max |lat_acc - v * yaw| {np.nanmax(np.abs(lat - v * yaw)):.2e}")
    for mo, g in m.groupby("model", sort=True):
        idx = g.index.to_numpy()
        nan = {c: pct(np.isnan(X[idx][:, :, T.CH[c]]).mean()) for c in ("steer_torque_norm", "gas_pressed", "brake_pressed", "steer_rate_degps", "yaw_rate_radps")}
        print(f"    {mo:14s} NaN share %: {nan}; mean speed {np.nanmean(v[idx]):.1f} m/s; lead_valid share {pct(lv[idx].mean())} %")


def sec17_raw_sign():
    print("\n## 17. Raw commaSteeringControl CSVs (zipfile + pandas, read-only): sign of the lateral-acceleration columns against the steering angle")
    import io as _io, zipfile as _zip
    data = EXT / "csc" / "data"
    for arch in ("CHEVROLET_BOLT_EUV_2022", "RAM_1500_5TH_GEN", "TOYOTA_SIENNA_2018"):
        zp = data / f"{arch}.zip"
        if not zp.exists():
            print(f"  {arch}: zip not found"); continue
        with _zip.ZipFile(zp) as z:
            csvs = sorted((i for i in z.infolist() if i.filename.lower().endswith(".csv") and i.file_size < 50 * 1024 * 1024), key=lambda i: i.filename)
            rng = np.random.default_rng(1)
            pick = rng.choice(len(csvs), size=min(6, len(csvs)), replace=False)
            rows = []
            for k in pick:
                df = pd.read_csv(_io.BytesIO(z.read(csvs[k])))
                mv = df["vEgo"].to_numpy(float) > 5.0
                if mv.sum() < 100 or df["steeringAngleDeg"][mv].std() < 0.5:
                    continue
                rows.append((len(df), float(np.corrcoef(df["steeringAngleDeg"][mv], df["latAccelLocalizer"][mv])[0, 1]), float(np.corrcoef(df["steeringAngleDeg"][mv], df["latAccelSteeringAngle"][mv])[0, 1]),
                             float(np.allclose(df["t"].to_numpy(float), np.arange(len(df)) / 10.0, atol=1e-6, rtol=0.0))))
            print(f"  {arch}: columns {list(df.columns)}")
            for n, c1, c2, grid in rows:
                print(f"    rows {n}: corr(steer, latAccelLocalizer) {c1:+.3f}, corr(steer, latAccelSteeringAngle) {c2:+.3f}, t on the 10 Hz grid {bool(grid)}  (negative = right-positive raw -> the adapter's sign flip is right)")


def sec18_ensembles():
    print("\n## 18. Ensemble = mean of the member seeds (phase-2 files)")
    for ds, tag, mask in (("csc", "transfer__size__gru__V1", "full"), ("csc", "transfer__bev_ice__resnet__V3", "full"), ("csc", "split_platform__size__gru__S9", "mask-V1"),
                          ("moretorque", "split_platform__size__gru__S9", "full"), ("moretorque", "split_platform__bev_ice__resnet__B5", "mask-yaw"), ("moretorque", "transfer__bev_ice__resnet__V3", "full")):
        z = np.load(EXT / ds / "work" / f"probs__{tag}__ens__{mask}.npz", allow_pickle=True)
        mem = [np.load(EXT / ds / "work" / f"probs__{s}__{mask}.npz")["probs"] for s in z["members"]]
        print(f"  {ds}/{tag}/{mask}: members {len(mem)}, max |ens - mean(members)| = {np.abs(z['probs'] - np.mean(mem, 0)).max():.2e}, rows sum to 1: {np.abs(z['probs'].sum(1) - 1).max():.1e}")


def sec19_hygiene():
    print("\n## 19. Identifier hygiene of the phase-2 outputs (16-hex tokens with a letter, route stamps, home paths) in markdown / CSV / JSON / file names")
    import re as _re
    hex16 = _re.compile(r"(?<![0-9a-fA-F.])[0-9a-f]{16}(?![0-9a-fA-F])")
    route1 = _re.compile(r"\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}"); route2 = _re.compile(r"\b[0-9a-f]{8}--[0-9a-f]{10}\b"); home = _re.compile("/" + r"home/\w+")
    files = [HERE_ / f for f in ("RESULTS.md", "RESULTS_TABLES.md", "results_summary.json", "DS_CSC.md", "DS_MORETORQUE.md")]
    for ds in ("csc", "moretorque", "moretorque_dasany"):
        files += [p for p in (EXT / ds / "work").glob("*") if p.suffix in (".csv", ".json", ".md")]
    bad = 0
    for f in files:
        t = f.read_text(errors="replace")
        h = [x for x in hex16.findall(t) if _re.search(r"[a-f]", x)]
        r = route1.findall(t) + route2.findall(t); hm = home.findall(t)
        if h or r or hm:
            bad += 1
            print(f"  {f.name}: hex16 {len(h)}, route stamps {len(r)}, home paths {len(hm)}")
    names = [p.name for ds in ("csc", "moretorque", "moretorque_dasany") for p in (EXT / ds / "work").rglob("*")]
    badn = [n for n in names if (hex16.search(n) and _re.search(r"[a-f]", hex16.search(n).group(0))) or route1.search(n) or route2.search(n)]
    print(f"  files scanned: {len(files)} (flagged {bad}); file names scanned: {len(names)} (flagged {len(badn)}); meta strings: "
          + "; ".join(f"{ds} {sum(bool(hex16.search(str(x)) and _re.search(r'[a-f]', hex16.search(str(x)).group(0))) for col in ('vehicle_id', 'drive_id', 'model', 'note') for x in pd.read_parquet(EXT / ds / 'work' / 'meta.parquet')[col].unique())} hex16 tokens" for ds in ("csc", "moretorque")))



# ============================================================================= phase 3: commaCarSegments device-disjoint selection (ccs)
# Sections 20-32, added by the phase-3 review on 10 Nov. Every table of RESULTS.md 2.12 is recomputed from the probs__*__ens__*.npz
# files and meta.parquet, re-rendered in the report's own cell format and searched in RESULTS.md ("FOUND" / "NOT FOUND"); the prose
# figures are printed next to the quoted value with "OK" / "MISMATCH". No device or route identifier is printed (pseudonyms are
# only counted; the raw dongle ids under <external>/ccs/segments are only intersected and counted).
import re as _re3

CCS_STRATA = ("seen", "held-out", "absent")
CCS_RUNS_SIZE = (("full", "full", "full (20 ch)"), ("V1", "full", "V1 lateral + speed"), ("V2", "full", "V2 longitudinal-only"), ("V3", "full", "V3 speed-only"),
                 ("mask", "mask-V1", "full + mask-V1"), ("mask", "mask-V2", "full + mask-V2"), ("mask", "mask-V3", "full + mask-V3"),
                 ("mask", "mask-yaw", "full + mask-yaw (sensitivity)"), ("mask", "mask-lead", "full + mask-lead (sensitivity)"))
_RESULTS_TXT = None
_SEC212 = None
_NFOUND = [0, 0]
_NCHK = [0, 0]


def results_text() -> str:
    global _RESULTS_TXT, _SEC212
    if _RESULTS_TXT is None:
        t = (HERE_ / "RESULTS.md").read_text()
        _RESULTS_TXT = _re3.sub(r"[ \t]+", " ", t)
        i = t.find("### 2.12 "); j = t.find("## 3. Cross-dataset summary")
        _SEC212 = _re3.sub(r"[ \t]+", " ", t[i:j]) if i >= 0 and j > i else _RESULTS_TXT
    return _RESULTS_TXT


def found(label: str, cell: str, where: str = "2.12") -> bool:
    """Is the re-rendered table cell / row present verbatim in RESULTS.md (2.12 or the whole file)?"""
    results_text()
    hay = _SEC212 if where == "2.12" else _RESULTS_TXT
    ok = _re3.sub(r"[ \t]+", " ", cell) in hay
    _NFOUND[0 if ok else 1] += 1
    print(f"    {'FOUND    ' if ok else 'NOT FOUND'} {label}: {cell}")
    return ok


def chk(label: str, computed, quoted) -> bool:
    """Print a recomputed prose figure next to the quoted one."""
    c, q = str(computed), str(quoted)
    ok = c == q
    _NCHK[0 if ok else 1] += 1
    print(f"    {'OK      ' if ok else 'MISMATCH'} {label}: recomputed {c} | report {q}")
    return ok


def fmt_int(n) -> str:
    return f"{int(n):,}"


def ccs_meta() -> pd.DataFrame:
    m = pd.read_parquet(EXT / "ccs" / "work" / "meta.parquet").reset_index(drop=True)
    m["stratum"] = note_field(m, "stratum").fillna("absent")
    m["lead"] = note_field(m, "lead").fillna("none")
    m["label_src"] = note_field(m, "label").fillna("none")
    m["torque"] = note_field(m, "torque").fillna("absent")
    mixed = m["note"].astype(str).str.contains("merges gasoline")
    pt = m["powertrain"].fillna("unknown").astype(str)
    m["pt_label"] = np.where(pt.eq("unknown") & mixed.to_numpy(), "MIXED", pt)
    m["n_ch"] = m["channels_present"].astype(str).str.count(",") + 1
    m["has_torque"] = m["channels_present"].astype(str).str.contains("steer_torque_norm")
    m["has_lead"] = m["channels_present"].astype(str).str.contains("lead_d_m")
    m["pair"] = m["vehicle_id"].astype(str) + "|" + m["model"].astype(str)
    return m


def ccs_qc() -> pd.DataFrame:
    return pd.read_csv(EXT / "ccs" / "work" / "segments_qc.csv")


def ccs_load(task: str, key: str, mask: str = "full"):
    P, y, _, cl = load("ccs", task, key, mask)
    return P, y, cl


def unit_dec(P, y, units):
    """(unit ids, mean P per unit, majority truth, decision, count) over the labelled windows."""
    lab = y >= 0
    uid, inv, cnt = np.unique(np.asarray(units)[lab].astype(str), return_inverse=True, return_counts=True)
    k = P.shape[1]
    S = np.zeros((len(uid), k)); np.add.at(S, inv, P[lab]); Pu = S / cnt[:, None]
    yl = y[lab]
    yu = np.zeros(len(uid), int)
    for u in range(len(uid)):
        yu[u] = np.bincount(yl[inv == u], minlength=k).argmax()
    return uid, Pu, yu, Pu.argmax(1), cnt


def boot_ci(P, y, m, k, level, B=300, seed=0):
    """score.py level_eval's vehicle-cluster bootstrap (same RNG, same resampling rule) for the vehicle / model level."""
    lab = y >= 0
    mm = m.loc[lab].reset_index(drop=True); p = P[lab].astype(np.float64); yy = y[lab]
    veh_ids, veh_idx = np.unique(mm["vehicle_id"].astype(str).to_numpy(), return_inverse=True); V = len(veh_ids)
    rng = np.random.default_rng(seed)
    mult = np.zeros((B, V))
    for b in range(B):
        mult[b] = np.bincount(rng.integers(0, V, V), minlength=V)
    units = mm["vehicle_id" if level == "vehicle" else "model"].astype(str).to_numpy()
    uid, inv, cnt = np.unique(units, return_inverse=True, return_counts=True)
    S = np.zeros((len(uid), k)); np.add.at(S, inv, p); pred_u = (S / cnt[:, None]).argmax(1)
    y_u = np.array([np.bincount(yy[inv == u], minlength=k).argmax() for u in range(len(uid))])
    if level == "vehicle":
        veh_u = np.zeros(len(uid), int); veh_u[inv] = veh_idx
        cm_v = np.zeros((V, k, k)); np.add.at(cm_v, (veh_u, y_u, pred_u), 1.0)
        cm_b = (mult @ cm_v.reshape(V, k * k)).reshape(B, k, k)
    else:
        M = len(uid)
        Sm = np.zeros((M, V, k)); N = np.zeros((M, V))
        np.add.at(Sm, (inv, veh_idx), p); np.add.at(N, (inv, veh_idx), 1.0)
        Pb = (mult @ Sm.transpose(1, 0, 2).reshape(V, M * k)).reshape(B, M, k)
        Nb = mult @ N.T
        pred_b = Pb.argmax(2)
        cm_b = np.zeros((B, k, k))
        bb, mi = np.nonzero(Nb > 0)
        np.add.at(cm_b, (bb, y_u[mi], pred_b[bb, mi]), 1.0)
    tp = np.einsum("bii->bi", cm_b); sup = cm_b.sum(2); pr = cm_b.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = tp.sum(1) / sup.sum(1)
        den = 2 * tp + (pr - tp) + (sup - tp)
        f1 = np.where(den > 0, 2 * tp / den, 0.0); has = sup > 0
        f1m = np.where(has.sum(1) >= 2, (f1 * has).sum(1) / np.maximum(has.sum(1), 1), np.nan)

    def ci(v):
        v = v[np.isfinite(v)]
        return None if len(v) < 20 else (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return ci(acc), ci(f1m)


def ci_str(c) -> str:
    return "" if c is None else f" [{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def f1_or_dash(r) -> str:
    return pct(r["f1"]) if np.isfinite(r["f1"]) else "-"


def sec20_ccs_decode():
    import json as _json
    print("## 20. commaCarSegments decode accounting (RESULTS.md 2.12 first paragraph, CCS_PIPELINE.md 'Decode result') against decode_summary.json, segments_qc.csv and select_summary.json")
    ds = _json.load(open(EXT / "ccs" / "work" / "decode_summary.json"))
    qc = ccs_qc(); m = ccs_meta()
    sel = _json.load(open(EXT / "ccs" / "select_summary.json"))
    chk("segments found", fmt_int(ds["segments_found"]), "49,563"); chk("routes", fmt_int(ds["routes"]), "23,302")
    chk("decoded", fmt_int(ds["segments_decoded"]), "47,006"); chk("decoded == segments_qc rows", len(qc) == ds["segments_decoded"], True)
    chk("windows", fmt_int(ds["windows"]), "45,763"); chk("windows == qc window flags == meta rows", int(qc["window"].sum()) == ds["windows"] == len(m), True)
    chk("dropped by the moving rule", fmt_int(ds["segments_decoded"] - ds["windows"]), "1,243")
    mv = qc.loc[~qc["window"], "moving_frac"]
    chk("moving-rule drops all below 0.2 / median", f"{bool((mv < 0.2).all())} / {mv.median():.1f}", "True / 0.0")
    chk("skipped", fmt_int(ds["skipped"]), "2,557"); chk("found == decoded + skipped", ds["segments_found"] == ds["segments_decoded"] + ds["skipped"], True)
    sr = ds["skip_reasons"]
    cv = {k: v for k, v in sr.items() if k.startswith("canValid")}; st = {k: v for k, v in sr.items() if "stale" in k}; ci_ = {k: v for k, v in sr.items() if "CarInterface" in k}
    chk("canValid skips (all / exactly 0.00 / other values)", f"{fmt_int(sum(cv.values()))} / {fmt_int(cv.get('canValid only 0.00', 0))} / {sorted(k.split()[-1] for k in cv if k != 'canValid only 0.00')}", "2,460 / 2,457 / ['0.03', '0.35', '0.40']")
    stale_n = np.repeat([int(k.split()[0]) for k in st], [v for v in st.values()])
    chk("stale-grid skips (n / min-max / median)", f"{sum(st.values())} / {stale_n.min()}-{stale_n.max()} / {int(np.median(stale_n))}", "73 / 1-570 / 24")
    chk("no-CarInterface skips", f"{sum(ci_.values())} " + str({k.split()[-1]: v for k, v in ci_.items()}), "24 {'HYUNDAI_TUCSON_4TH_GEN': 20, 'HONDA_CRV_HYBRID_6G': 3, 'HYUNDAI_SONATA_HEV_2024': 1}")
    chk("skip reasons sum to skipped", sum(sr.values()) == ds["skipped"], True)
    chk("wall time h / CPU-hours of CAN replay / s per segment", f"{ds['seconds'] / 3600:.1f} / {qc['decode_s'].sum() / 3600:.1f} / {qc['decode_s'].mean():.2f}", "3.2 / 17.9 / 1.37")
    chk("workers / yaw / scc lead / sign", f"{ds['workers']} / {ds['yaw']} / {ds['scc_lead']} / {ds['scc_sign']:+.0f}", "6 / derived / True / +1")
    scc = qc[qc["lead_source"] == "scc"]
    chk("median scc_sign_corr / stock-ACC windows with a value (decoded segments with a value)", f"{scc['scc_sign_corr'].median():+.2f} / {fmt_int((scc['window'] & scc['scc_sign_corr'].notna()).sum())} ({fmt_int(scc['scc_sign_corr'].notna().sum())})", "+0.70 / 4,702 (4,745)")
    chk("decode_summary median", f"{ds['scc_sign_corr_median']:+.2f}", "+0.70")
    # per-platform attribution: budget list (n_budget) minus decoded
    pl = pd.DataFrame(sel["platforms"])
    bud = pl.set_index("platform")["n_budget"].fillna(0).astype(int)
    bud = bud[bud > 0]
    dec = qc.groupby("platform").size()
    chk("platforms in the budget list / decoded / in the window set", f"{len(bud)} / {dec.index.nunique()} / {m['platform'].nunique()}", "172 / 161 / 161")
    chk("budget list total == segments found", int(bud.sum()) == ds["segments_found"], True)
    loss = (bud - dec.reindex(bud.index).fillna(0).astype(int)).sort_values(ascending=False)
    no_ci = {k.split()[-1] for k in ci_}
    whole = loss[(loss > 0) & (dec.reindex(loss.index).fillna(0) == 0) & ~loss.index.isin(no_ci)]
    print(f"    platforms losing every segment: {whole.to_dict()} -> {fmt_int(whole.sum())} segments (report: nine Honda platforms, 2,198)")
    chk("nine Honda platforms, whole loss", f"{len(whole)} / {fmt_int(whole.sum())} / {bool(all(p.startswith('HONDA') for p in whole.index))}", "9 / 2,198 / True")
    part = loss[(loss > 0) & (dec.reindex(loss.index).fillna(0) > 0)]
    print(f"    partial losses (selected -> lost): " + ", ".join(f"{p} {v} of {bud[p]}" for p, v in part.head(6).items()) + f"; remaining range {part.iloc[6:].min()}-{part.iloc[6:].max()} over {len(part) - 6} platforms")
    chk("RAM_HD_5TH_GEN / CHRYSLER_PACIFICA_2018 / SKODA_OCTAVIA_MK3 / TOYOTA_PRIUS lost of selected",
        f"{part.get('RAM_HD_5TH_GEN', 0)} of {bud.get('RAM_HD_5TH_GEN', 0)}, {part.get('CHRYSLER_PACIFICA_2018', 0)} of {bud.get('CHRYSLER_PACIFICA_2018', 0)}, {part.get('SKODA_OCTAVIA_MK3', 0)} of {bud.get('SKODA_OCTAVIA_MK3', 0)}, {part.get('TOYOTA_PRIUS', 0)} of {bud.get('TOYOTA_PRIUS', 0)}",
        "64 of 143, 42 of 389, 31 of 322, 14 of 400")
    chk("sum of per-platform losses == skipped", int(loss.sum()) == ds["skipped"], True)
    tuc = bud.get("HYUNDAI_TUCSON_4TH_GEN", 0) - loss.get("HYUNDAI_TUCSON_4TH_GEN", 0)
    chk("HYUNDAI_TUCSON_4TH_GEN decoded (report: 'the platform's other 345 decoded')", int(tuc), 345)
    # strata: selected vs in the set, models
    sel_models = pl[pl["platform"].isin(bud.index)].groupby("stratum")["model"].nunique()
    set_models = m.groupby("stratum")["model"].nunique()
    chk("models selected seen / held-out / absent", "/".join(str(sel_models.get(s, 0)) for s in CCS_STRATA), "72/30/15")
    chk("models in the set seen / held-out / absent", "/".join(str(set_models.get(s, 0)) for s in CCS_STRATA), "70/29/14")
    missing = sorted(set(pl.loc[pl["platform"].isin(bud.index), "model"].dropna()) - set(m["model"].unique()))
    chk("selected models missing from the set", missing, "['Honda Civic', 'Honda Fit', 'Honda Freed', 'Honda HR-V']")
    crv = sorted(m.loc[m["model"] == "Honda CR-V", "platform"].unique())
    chk("Honda CR-V present only through", crv, "['HONDA_CRV_HYBRID']")
    print(f"    CR-V hybrid devices: {m.loc[m['platform'] == 'HONDA_CRV_HYBRID', 'vehicle_id'].nunique()} (report 13)")
    chk("hours", f"{len(m) / 60:.1f}", "762.7")
    chk("devices / drives / models / platforms", f"{fmt_int(m['vehicle_id'].nunique())} / {fmt_int(m['drive_id'].nunique())} / {m['model'].nunique()} / {m['platform'].nunique()}", "4,061 / 21,987 / 113 / 161")
    print(f"    selection summary: devices selected {sel['budget']['devices']:,}, routes {sel['budget']['routes']:,} (report 4,334 / 23,302); shared devices removed {sel['shared_devices']} ({sel['shared_segments_removed']:,} segments); index {sel['index_segments']:,} segments / {sel['index_keys']} keys -> {sel['folded_platforms']} folded platforms")
    # per-platform quality (CCS_PIPELINE 'Decode result')
    chk("can_valid_frac min / mean (decoded)", f"{qc['can_valid_frac'].min():.2f} / {qc['can_valid_frac'].mean():.3f}", "0.81 / 0.992")
    pm = qc.groupby("platform")["can_valid_frac"].mean().sort_values()
    print(f"    lowest platform means of can_valid_frac: " + ", ".join(f"{p} {v:.3f}" for p, v in pm.head(4).items()) + " (report: Toyota / Lexus TSS2 0.983-0.986)")
    chk("lead source, decoded segments radar / scc / none", "/".join(fmt_int(qc['lead_source'].value_counts().get(k, 0)) for k in ("radar", "scc", "none")), "15,949/4,804/26,253")
    chk("lead source, windows radar / scc / none (meta.note)", "/".join(fmt_int(m['lead'].value_counts().get(k, 0)) for k in ("radar", "scc", "none")), "15,539/4,740/25,484")
    rad = m[m["lead"] == "radar"]
    brand = rad["platform"].str.split("_").str[0]
    tl = brand.isin(["TOYOTA", "LEXUS"]).sum(); ho = brand.isin(["HONDA", "ACURA"]).sum(); fo = (brand == "FORD").sum(); hy = brand.isin(["HYUNDAI", "KIA", "GENESIS"]).sum(); ri = (brand == "RIVIAN").sum()
    chk("radar windows Toyota/Lexus, Honda/Acura, Ford, Hyundai group, Rivian", f"{fmt_int(tl)}, {fmt_int(ho)}, {fmt_int(fo)}, {fmt_int(hy)}, {fmt_int(ri)}", "12,483, 1,286, 915, 794, 61")
    print(f"    radar brands present: {sorted(brand.unique().tolist())}; sum {tl + ho + fo + hy + ri} of {len(rad)}")
    for src, quoted in (("radar", "0.63"), ("scc", "0.69")):
        chk(f"lead_valid_frac mean on {src} segments", f"{qc.loc[qc['lead_source'] == src, 'lead_valid_frac'].mean():.2f}", quoted)
    chk("stock-ACC platforms", qc.loc[qc["lead_source"] == "scc", "platform"].nunique(), 33)
    chk("scc windows written lead=none (no object in the minute) but lead channels present", fmt_int(((m["lead"] == "none") & m["has_lead"]).sum()), "1,271")
    scc_plats = set(qc.loc[qc["lead_source"] == "scc", "platform"])
    lhl = m.loc[(m["lead"] == "none") & m["has_lead"], "platform"]
    chk("...of which on stock-ACC platforms / on radar platforms (report before the review: all 1,271 stock-ACC)", f"{fmt_int(lhl.isin(scc_plats).sum())} / {fmt_int((~lhl.isin(scc_plats)).sum())}", "928 / 343")
    chk("yaw source derived on every segment", bool((qc["yaw_source"] == "derived").all()), True)
    ford = qc["brand"].str.lower() == "ford"
    chk("CarState.yawRate non-zero share on Ford samples / Ford share of the windows", f"{100 * qc.loc[ford & qc['window'], 'yaw_meas_nonzero'].mean():.1f} % / {pct(ford[qc['window']].mean())} %", "98.1 % / 4.4 %")
    chk("yawRate non-zero on non-Ford segments (max share)", f"{qc.loc[~ford, 'yaw_meas_nonzero'].max():.2f}", "0.00")
    sr_b = qc[qc["window"]].groupby(qc["brand"].str.lower())["steer_rate_nonzero"].mean()
    print("    CAN steering-rate non-zero share by brand (windows): " + ", ".join(f"{b} {100 * v:.0f}" for b, v in sr_b.sort_values(ascending=False).items()))
    ww = qc["window"]
    chk("torque_nonzero overall / Mazda / Ford (windows)", f"{qc.loc[ww, 'torque_nonzero'].mean():.2f} / {qc.loc[ww & (qc['brand'].str.lower() == 'mazda'), 'torque_nonzero'].mean():.2f} / {qc.loc[ww & ford, 'torque_nonzero'].mean():.2f}", "0.97 / 0.84 / 0.93")
    chk("torque scale absent: decoded segments / share / windows", f"{fmt_int(qc['torque_scale'].isna().sum())} / {pct(qc['torque_scale'].isna().mean())} % / {fmt_int((~m['has_torque']).sum())}", "7,434 / 15.8 % / 7,039")
    ta = m.groupby("stratum")["has_torque"].apply(lambda s: 100 * (~s).mean())
    print(f"    torque absent per stratum (windows) %: " + ", ".join(f"{s} {ta[s]:.1f}" for s in CCS_STRATA) + " (report: evenly, 14.4-15.6 %)")
    w = qc[qc["window"]]
    chk("das_frac / cruise_frac / controls_allowed_frac means (windows; all decoded segments in brackets)", f"{w['das_frac'].mean():.2f} / {w['cruise_frac'].mean():.2f} / {w['controls_allowed_frac'].mean():.2f} ({qc['das_frac'].mean():.2f} / {qc['cruise_frac'].mean():.2f} / {qc['controls_allowed_frac'].mean():.2f})", "0.39 / 0.39 / 0.38 (0.38 / 0.38 / 0.37)")
    chk("engaged (das_frac >= 0.5) share of the windows", pct((m["das_frac"] >= 0.5).mean()) + " %", "39.3 %")
    chk("mean speed / moving share / standstill / gas / brake / steer pressed (windows, qc means)", f"{w['v_mean'].mean():.1f} / {w['moving_frac'].mean():.2f} / {100 * w['standstill_frac'].mean():.1f} % / {100 * w['gas_frac'].mean():.0f} % / {100 * w['brake_frac'].mean():.0f} % / {100 * w['steer_pressed_frac'].mean():.0f} %", "18.0 / 0.93 / 5.7 % / 39 % / 13 % / 22 %")
    bv = w.groupby(w["brand"].str.lower())["v_mean"].mean()
    print(f"    brand mean speed range: {bv.idxmin()} {bv.min():.1f} .. {bv.idxmax()} {bv.max():.1f} m/s (report: Volkswagen group 16.7 .. GM 19.5)")
    rpd = qc.groupby("drive_id").size()
    chk("routes with decoded segments / segments per route min-max / median / 75th pct", f"{fmt_int(len(rpd))} / {rpd.min()}-{rpd.max()} / {rpd.median():.0f} / {rpd.quantile(0.75):.0f}", "22,238 / 1-49 / 1 / 2")
    per_v = m.groupby("vehicle_id").agg(np_=("platform", "nunique"), nm=("model", "nunique"), nc=("size_class", "nunique"))
    chk("devices under two platforms / two models / two size classes", f"{int((per_v['np_'] > 1).sum())} / {int((per_v['nm'] > 1).sum())} / {int((per_v['nc'] > 1).sum())}", "91 / 88 / 60")
    chk("devices under two / three platforms; under two / three models", f"{int((per_v['np_'] == 2).sum())} / {int((per_v['np_'] == 3).sum())}; {int((per_v['nm'] == 2).sum())} / {int((per_v['nm'] == 3).sum())}", "88 / 3; 85 / 3")
    # the 4,334-device claim and the shared-device removal (ids only intersected and counted)
    seg_devs = {p.name for p in (EXT / "ccs" / "segments").iterdir() if p.is_dir()}
    shared = {ln.strip() for ln in (EXT / "ccs" / "shared_dongles.txt").read_text().splitlines() if ln.strip()}
    chk("device folders under segments / in shared_dongles.txt", f"{fmt_int(len(seg_devs))} / {len(seg_devs & shared)}", "4,334 / 0")
    pm_ = pd.read_parquet(T.DATA_ROOT / "private" / "pseudonym_map.parquet", columns=["dongle_log"])
    lab_devs = set(pm_["dongle_log"].dropna().astype(str))
    chk("downloaded ccs devices that are lab-library devices (private pseudonym map) / shared-list devices in the pseudonym map", f"{len(seg_devs & lab_devs)} / {len(shared & lab_devs)} of {len(shared)}", "0 / 461 of 464")
    pseud = {P_.pseudonym("v", d) for d in shared} if (P_ := _ccs_platforms()) else set()
    chk("pseudonyms of the shared devices among the meta vehicle ids", len(pseud & set(m["vehicle_id"].astype(str))) if pseud else "n/a", 0)
    chk("meta vehicle ids == pseudonyms of the device folders that produced windows (count)", f"{len(set(m['vehicle_id'].astype(str)) - {P_.pseudonym('v', d) for d in seg_devs})}" if pseud else "n/a", "0")


def _ccs_platforms():
    try:
        from experiments.transfer import ccs_platforms as P_
        return P_
    except Exception as e:  # noqa: BLE001
        print(f"    (ccs_platforms import failed: {e})")
        return None


def sec21_ccs_set():
    print("\n## 21. commaCarSegments set description (RESULTS.md 2.12 paragraphs 1-3, 1.1 row, 1.3 bullet)")
    m = ccs_meta()
    nch = m["n_ch"].value_counts()
    chk("channels per window 20 / 19 / 14 / 13", " / ".join(fmt_int(nch.get(k, 0)) for k in (20, 19, 14, 13)), "18,323 / 3,227 / 20,401 / 3,812")
    chk("other channel counts present", sorted(set(nch.index) - {20, 19, 14, 13}), [])
    chk("torque present / absent (share)", f"{fmt_int(m['has_torque'].sum())} ({pct(m['has_torque'].mean())} %) / {fmt_int((~m['has_torque']).sum())} ({pct((~m['has_torque']).mean())} %)", "38,724 (84.6 %) / 7,039 (15.4 %)")
    chk("lead channels present (share)", f"{fmt_int(m['has_lead'].sum())} ({pct(m['has_lead'].mean())} %)", "21,550 (47.1 %)")
    chk("lead note shares radar / scc / none %", " / ".join(pct((m['lead'] == k).mean()) for k in ("radar", "scc", "none")), "34.0 / 10.4 / 55.7")
    ls = m["label_src"].value_counts()
    chk("label source lab / public windows (share)", f"{fmt_int(ls.get('lab', 0))} ({pct(ls.get('lab', 0) / len(m))} %) / {fmt_int(ls.get('public', 0))} ({pct(ls.get('public', 0) / len(m))} %)", "43,195 (94.4 %) / 2,568 (5.6 %)")
    chk("other label sources", sorted(set(ls.index) - {"lab", "public"}), [])
    pub = m[m["label_src"] == "public"]
    chk("public-spec platforms in the set / absent models with a public footprint (lab-labelled absent platform)", f"{pub['platform'].nunique()} / {len(set(m.loc[m['stratum'] == 'absent', 'model']) & set(pub['model']))} of 14 ({sorted(m.loc[(m['stratum'] == 'absent') & (m['label_src'] == 'lab'), 'platform'].unique().tolist())})", "22 / 13 of 14 (['VOLKSWAGEN_TOURAN_MK2'])")
    chk("public-spec platforms that are not absent", int(pub.loc[pub["stratum"] != "absent", "platform"].nunique()), 9)
    sc = m["size_class"].value_counts()
    mo_s = unit_eval(np.ones((len(m), 3)) / 3, T.task_labels(m, "size"), m["model"].to_numpy(), 3)["sup"]
    chk("size windows SMALL / MID / LARGE (models by majority label)", " / ".join(f"{fmt_int(sc[c])} ({mo_s[i]})" for i, c in enumerate(T.SIZE_CLASSES)), "20,036 (50) / 15,159 (33) / 10,568 (30)")
    span = m.groupby("model")["size_class"].nunique()
    print(f"    models with platforms in two size classes: {sorted(span[span > 1].index.tolist())}")
    chk("size label on every window", bool(m["size_class"].notna().all()), True)
    yb = T.task_labels(m, "bev_ice")
    chk("BEV vs ICE labelled windows", fmt_int((yb >= 0).sum()), "29,585")
    mo_b = unit_eval(np.ones((len(m), 2)) / 2, yb, m["model"].to_numpy(), 2)["sup"]; ve_b = unit_eval(np.ones((len(m), 2)) / 2, yb, m["vehicle_id"].to_numpy(), 2)["sup"]
    for i, (c, q) in enumerate((("BEV", "5,420 (15 models, 621 devices)"), ("ICE", "24,165 (73 models, 1,891 devices)"))):
        g = m[m["pt_label"] == c]
        chk(f"{c} windows (models, devices by majority label)", f"{fmt_int(len(g))} ({mo_b[i]} models, {fmt_int(ve_b[i])} devices)", q)
    pt = m["pt_label"].value_counts()
    chk("HEV / PHEV / FCEV / MIXED+unknown windows", f"{fmt_int(pt.get('HEV', 0))} / {fmt_int(pt.get('PHEV', 0))} / {fmt_int(pt.get('FCEV', 0))} / {fmt_int(pt.get('MIXED', 0) + pt.get('unknown', 0))}", "9,386 / 2,392 / 296 / 4,104")
    print(f"    MIXED {pt.get('MIXED', 0):,} + unknown {pt.get('unknown', 0):,}; unknown platforms: {sorted(m.loc[m['pt_label'] == 'unknown', 'platform'].unique().tolist())}")
    bl = m[m["borderline"].astype(bool)]
    chk("borderline models (windows)", f"{bl['model'].nunique()} ({fmt_int(len(bl))})", "16 (5,335)")
    for s, q in (("seen", "25,667 / 3,199 / 11,627 / 70 / 100"), ("held-out", "18,710 / 864 / 9,697 / 29 / 47"), ("absent", "1,386 / 36 / 663 / 14 / 14")):
        g = m[m["stratum"] == s]
        chk(f"stratum {s}: windows / devices / drives / models / platforms", f"{fmt_int(len(g))} / {fmt_int(g['vehicle_id'].nunique())} / {fmt_int(g['drive_id'].nunique())} / {g['model'].nunique()} / {g['platform'].nunique()}", q)
    ab = m[m["stratum"] == "absent"].groupby("model").agg(sc=("size_class", "first"), pt=("pt_label", "first"))
    chk("absent models by size class", ab["sc"].value_counts().to_dict(), {"SMALL": 10, "LARGE": 3, "MID": 1})
    chk("absent MID / LARGE models", f"{sorted(ab.index[ab['sc'] == 'MID'])} / {sorted(ab.index[ab['sc'] == 'LARGE'])}", "['Buick Regal Sportback'] / ['Cadillac Escalade', 'Dodge Durango', 'Hyundai Staria']")
    chk("absent powertrains", sorted(ab["pt"].value_counts().to_dict().items()), sorted({"ICE": 10, "HEV": 2, "FCEV": 1, "PHEV": 1}.items()))
    chk("absent unlabelled models", sorted(ab.index[~ab["pt"].isin(["BEV", "ICE"])]), ["Hyundai Ioniq Plug-in", "Lexus CT 200h", "Toyota Mirai", "Toyota Prius v"])
    das = m.groupby("stratum")["das_frac"].mean()
    chk("das_frac mean all (seen, held-out, absent)", f"{m['das_frac'].mean():.2f} ({', '.join(f'{das[s]:.2f}' for s in CCS_STRATA)})", "0.39 (0.40, 0.38, 0.39)")
    eng = (m["das_frac"] >= 0.5)
    chk("engaged share / engaged / human windows", f"{pct(eng.mean())} % / {fmt_int(eng.sum())} / {fmt_int((~eng).sum())}", "39.3 % / 18,004 / 27,759")
    qc = ccs_qc(); w = qc[qc["window"]].copy()
    key_q = w["drive_id"].astype(str) + "#" + w["seg"].astype(int).astype(str); key_m = m["drive_id"].astype(str) + "#" + m["w_idx"].astype(int).astype(str)
    chk("qc window rows join meta rows one to one (drive_id # segment)", f"{key_q.nunique()} / {key_m.nunique()} / {len(set(key_q) & set(key_m))}", f"{len(m)} / {len(m)} / {len(m)}")
    vm = w.set_index(key_q)["v_mean"].reindex(key_m).to_numpy()
    chk("mean speed all (seen, held-out, absent) m/s", f"{np.nanmean(vm):.1f} ({', '.join(f'{np.nanmean(vm[(m['stratum'] == s).to_numpy()]):.1f}' for s in CCS_STRATA)})", "18.0 (18.2, 17.8, 16.5)")
    chk("moving share (meta moving_frac mean)", f"{m['moving_frac'].mean():.2f}", "0.93")
    ev = m.groupby("vehicle_id").size()
    es = m.groupby(["stratum", "vehicle_id"]).size()
    chk("windows per device median all (seen, held-out, absent)", f"{ev.median():.0f} ({', '.join(f'{es[s].median():.0f}' for s in CCS_STRATA)})", "5 (4, 15, 49)")
    chk("devices with 1 / 2-4 / 5-9 / 10+ windows", f"{(ev == 1).sum()} / {fmt_int(((ev >= 2) & (ev <= 4)).sum())} / {((ev >= 5) & (ev <= 9)).sum()} / {fmt_int((ev >= 10).sum())}", "513 / 1,371 / 822 / 1,355")
    dw = m.groupby("drive_id").size()
    chk("drives with one window %", pct((dw == 1).mean()) + " %", "55.0 %")
    chk("device x model pairs size / powertrain", f"{fmt_int(m['pair'].nunique())} / {fmt_int(m.loc[yb >= 0, 'pair'].nunique())}", "4,152 / 2,547")
    chk("one window per segment: w_idx unique per drive", bool((m.groupby('drive_id')['w_idx'].nunique() == dw).all()), True)
    chk("1.1 row: 4,334 devices downloaded -> 4,061 with windows; 49,563 -> 47,006 -> 45,763", f"{fmt_int(m['vehicle_id'].nunique())}", "4,061")


def _strata_sel(m):
    return [("all", np.ones(len(m), bool))] + [(s, (m["stratum"] == s).to_numpy()) for s in CCS_STRATA]


def sec22_ccs_levels():
    print("\n## 22. Table (a) of 2.12: full models per stratum and level (window / drive / device / model; 300-resample vehicle-cluster intervals) + chance, no-borderline, the two model misses")
    m = ccs_meta()
    for task, k, cl_q in (("size", 3, "SMALL / MID / LARGE"), ("bev_ice", 2, "BEV / ICE")):
        P, y, cl = ccs_load(task, "full")
        print(f"  {task}:")
        for s, sel in _strata_sel(m):
            idx = np.flatnonzero(sel & (y >= 0))
            if len(idx) == 0:
                continue
            mm = m.iloc[idx].reset_index(drop=True); Pp = P[idx]; yy = y[idx]
            w = unit_eval(Pp, yy, np.arange(len(yy)), k); d = unit_eval(Pp, yy, mm["drive_id"].to_numpy(), k)
            v = unit_eval(Pp, yy, mm["vehicle_id"].to_numpy(), k); mo = unit_eval(Pp, yy, mm["model"].to_numpy(), k)
            civ = boot_ci(Pp, yy, mm, k, "vehicle"); cim = boot_ci(Pp, yy, mm, k, "model")
            if task == "size":
                row = (f"| {s} | {fmt_int(w['n'])}: {pct(w['acc'])} / {pct(w['f1'])} | {fmt_int(d['n'])}: {pct(d['acc'])} / {pct(d['f1'])} | {fmt_int(v['n'])}: {pct(v['acc'])}{ci_str(civ[0])} / {pct(v['f1'])}{ci_str(civ[1])} "
                       f"| {mo['n']}: {pct(mo['acc'])}{ci_str(cim[0])} / {pct(mo['f1'])}{ci_str(cim[1])} | {', '.join(f'{int(mo['cm'][i, i])}/{int(mo['sup'][i])}' for i in range(k))} | {' / '.join(pct(x) for x in v['rec'])} |")
            else:
                mr = ", ".join(("-" if mo["sup"][i] == 0 else f"{int(mo['cm'][i, i])}/{int(mo['sup'][i])}") for i in range(k))
                vr = " / ".join(("-" if v["sup"][i] == 0 else pct(x)) for i, x in enumerate(v["rec"]))
                row = (f"| {s} | {fmt_int(w['n'])}: {pct(w['acc'])} / {f1_or_dash(w)} | {fmt_int(d['n'])}: {pct(d['acc'])} / {f1_or_dash(d)} | {fmt_int(v['n'])}: {pct(v['acc'])}{ci_str(civ[0])} / {f1_or_dash(v)}{ci_str(civ[1]) if np.isfinite(v['f1']) else ''} "
                       f"| {mo['n']}: {pct(mo['acc'])}{ci_str(cim[0])} / {f1_or_dash(mo)}{ci_str(cim[1]) if np.isfinite(mo['f1']) else ''} | {mr} | {vr} |")
            found(f"(a) {task} {s}", row)
            if s == "all":
                maj = lambda r: pct(r["sup"].max() / r["n"])  # noqa: E731
                chk(f"{task} majority share windows / devices / models", f"{maj(w)} / {maj(v)} / {maj(mo)}", "43.8 / 43.4 / 44.2" if task == "size" else "81.7 / 75.3 / 83.0")
            if s == "absent" and task == "size":
                chk("absent stratum SMALL share of windows", pct(w["sup"][0] / w["n"]) + " %", "92.0 %")
        if task == "size":
            nb = ~m["borderline"].astype(bool).to_numpy()
            mm = m[nb].reset_index(drop=True); v = unit_eval(P[nb], y[nb], mm["vehicle_id"].to_numpy(), 3); mo = unit_eval(P[nb], y[nb], mm["model"].to_numpy(), 3)
            civ = boot_ci(P[nb], y[nb], mm, 3, "vehicle")
            chk("without the 16 borderline models: devices acc [CI] / F1; models acc / F1", f"{fmt_int(v['n'])} devices {pct(v['acc'])}{ci_str(civ[0])} / {pct(v['f1'])}, {mo['n']} models {pct(mo['acc'])} / {pct(mo['f1'])}", "3,678 devices 57.3 [55.7, 59.3] / 49.5, 99 models 56.6 / 47.3")
        else:
            for model, q in (("Honda e", "1 device, 2 windows, P(BEV) 0.34"), ("Cadillac Escalade", "1 device, 1 windows, P(BEV) 0.93")):
                g = m[(m["model"] == model) & (y >= 0)]
                chk(f"{model} miss", f"{g['vehicle_id'].nunique()} device, {len(g)} windows, P(BEV) {P[g.index.to_numpy(), 0].mean():.2f}", q)
            esc = m[(m["model"] == "Cadillac Escalade")]
            print(f"    Escalade window: das_frac {esc['das_frac'].iloc[0]:.2f} (report: one engaged highway window)")
            mo = unit_eval(P, y, m["model"].to_numpy(), 2)
            miss = [cl_ for cl_, ok in zip(np.unique(m.loc[y >= 0, 'model']), (mo['cm'].argmax(1) if False else None) or []) if not ok]  # placeholder, replaced below
            uid, Pu, yu, du, _ = unit_dec(P, y, m["model"].to_numpy())
            chk("model-level misses (powertrain, full)", sorted(uid[du != yu].tolist()), ["Cadillac Escalade", "Honda e"])


def sec23_ccs_runs():
    print("\n## 23. Table (b) of 2.12: variants and masks, device and model level per stratum (+ the V3 / mask-lead / mask-yaw figures quoted in the Reading bullets)")
    m = ccs_meta()
    store = {}
    for task, k in (("size", 3), ("bev_ice", 2)):
        print(f"  {task}:")
        for key, mask, label in CCS_RUNS_SIZE:
            P, y, cl = ccs_load(task, key, mask)
            cells = []
            for s, sel in _strata_sel(m):
                idx = np.flatnonzero(sel & (y >= 0))
                mm = m.iloc[idx].reset_index(drop=True)
                v = unit_eval(P[idx], y[idx], mm["vehicle_id"].to_numpy(), k); mo = unit_eval(P[idx], y[idx], mm["model"].to_numpy(), k)
                store[(task, mask if key == "mask" else key, s)] = (v, mo, P[idx], y[idx], mm)
                if task == "size":
                    cells.append(f"{pct(v['acc'])} / {pct(v['f1'])}; {'/'.join(str(int(mo['cm'][i, i])) for i in range(3))}")
                elif s == "absent":
                    cells.append(f"{pct(v['acc'])}; {int(mo['cm'][1, 1])}/{int(mo['sup'][1])}")
                else:
                    cells.append(f"{pct(v['acc'])} / {pct(v['f1'])}; {int(mo['cm'][0, 0])}/{int(mo['sup'][0])}, {int(mo['cm'][1, 1])}/{int(mo['sup'][1])}")
            found(f"(b) {task} {label}", f"| {label} | " + " | ".join(cells) + " |")
    v, mo, P, y, mm = store[("size", "V3", "all")]
    civ = boot_ci(P, y, mm, 3, "vehicle")
    chk("size V3 devices acc [CI] / F1; model F1; LARGE device recall; LARGE models", f"{pct(v['acc'])}{ci_str(civ[0])} / {pct(v['f1'])}; {pct(mo['f1'])}; {pct(v['rec'][2])}; {int(mo['cm'][2, 2])}/{int(mo['sup'][2])}", "53.7 [52.2, 55.3] / 50.0; 50.2; 44.5; 14/30")
    uid, Pu, yu, du, _ = unit_dec(P, y, mm["model"].to_numpy())
    print(f"    V3 LARGE model hits: {sorted(uid[(yu == 2) & (du == 2)].tolist())} (report: pickups, Traverse, Pacifica, Rivian, Ranger, Durango, Staria, Crafter, both Escalades)")
    v1 = store[("size", "V1", "all")]
    chk("size V1 MID device recall / MID models / LARGE models / SMALL models", f"{pct(v1[0]['rec'][1])} / {int(v1[1]['cm'][1, 1])}/33 / {int(v1[1]['cm'][2, 2])}/30 / {int(v1[1]['cm'][0, 0])}/50", "54.6 / 18/33 / 4/30 / 39/50")
    chk("size full MID device recall", pct(store[("size", "full", "all")][0]["rec"][1]), "27.6")
    yw = store[("size", "mask-yaw", "all")]
    chk("size mask-yaw device recall S / M / L; LARGE models", f"{' / '.join(pct(x) for x in yw[0]['rec'])}; {int(yw[1]['cm'][2, 2])}/{int(yw[1]['sup'][2])}", "25.7 / 20.4 / 88.3; 30/30")
    chk("size mask-yaw: share of all devices called LARGE (88.3 is the LARGE recall)", pct(yw[0]["cm"][:, 2].sum() / yw[0]["n"]) + " %", "69.6 %")
    ld = store[("size", "mask-lead", "all")]
    chk("size mask-lead devices / models right (full: 54.1, 63)", f"{pct(ld[0]['acc'])}, {int(np.trace(ld[1]['cm']))}", "53.8, 63")
    chk("size V3 absent models all right (10/1/3)", "/".join(str(int(store[('size', 'V3', 'absent')][1]['cm'][i, i])) for i in range(3)), "10/1/3")
    v, mo, P, y, mm = store[("bev_ice", "V3", "all")]
    civ = boot_ci(P, y, mm, 2, "vehicle")
    chk("powertrain V3 devices acc [CI] / F1; models", f"{pct(v['acc'])}{ci_str(civ[0])} / {pct(v['f1'])}; {int(np.trace(mo['cm']))}/{mo['n']}", "92.6 [91.5, 93.6] / 90.1; 87/88")
    ld = store[("bev_ice", "mask-lead", "all")]; fu = store[("bev_ice", "full", "all")]
    chk("powertrain mask-lead devices; ICE recall full -> mask; BEV recall full -> mask; models", f"{pct(ld[0]['acc'])}; {pct(fu[0]['rec'][1])} -> {pct(ld[0]['rec'][1])}; {pct(fu[0]['rec'][0])} -> {pct(ld[0]['rec'][0])}; {int(np.trace(ld[1]['cm']))}/88", "91.3; 88.2 -> 93.3; 86.6 -> 85.3; 86/88")
    yw = store[("bev_ice", "mask-yaw", "all")]
    chk("powertrain mask-yaw devices / models", f"{pct(yw[0]['acc'])} / {int(np.trace(yw[1]['cm']))}/88", "86.2 / 87/88")
    chk("powertrain V2 devices / models", f"{pct(store[('bev_ice', 'V2', 'all')][0]['acc'])} / {int(np.trace(store[('bev_ice', 'V2', 'all')][1]['cm']))}/88", "84.6 / 86/88")
    chk("powertrain V1 models BEV / ICE; held-out ICE", f"{int(store[('bev_ice', 'V1', 'all')][1]['cm'][0, 0])}/15, {int(store[('bev_ice', 'V1', 'all')][1]['cm'][1, 1])}/73; {int(store[('bev_ice', 'V1', 'held-out')][1]['cm'][1, 1])}/21", "15/15, 49/73; 11/21")
    for mk in ("mask-V2", "mask-V3"):
        chk(f"powertrain {mk} BEV models", f"{int(store[('bev_ice', mk, 'all')][1]['cm'][0, 0])}/15", "0/15")
    chk("size mask-V2 / mask-V3 MID+LARGE models right (3.4: 0 of 63)", f"{int(store[('size', 'mask-V2', 'all')][1]['cm'][1, 1] + store[('size', 'mask-V2', 'all')][1]['cm'][2, 2])} / {int(store[('size', 'mask-V3', 'all')][1]['cm'][1, 1] + store[('size', 'mask-V3', 'all')][1]['cm'][2, 2])}", "0 / 1")
    chk("3.4: mask-V1 BEV models / MID+LARGE models", f"{int(store[('bev_ice', 'mask-V1', 'all')][1]['cm'][0, 0])}/15 / {int(store[('size', 'mask-V1', 'all')][1]['cm'][1, 1] + store[('size', 'mask-V1', 'all')][1]['cm'][2, 2])}/63", "3/15 / 1/63")
    # Leaf / Lightning / Kona / Ioniq / Model Y / GV60 (Reading bullet 4)
    Pf, yf, _ = ccs_load("bev_ice", "full")
    for model, q in (("Nissan Leaf", "22/28 / 22, P(BEV) 0.63"), ("Ford F-150 Lightning", "4/6 / 4, P(BEV) 0.52"), ("Hyundai Kona", "18/20 / 16, P(BEV) 0.75"), ("Hyundai Ioniq", "5/5 / 5, P(BEV) 0.83"), ("Tesla Model Y", "20/21 / 20, P(BEV) 0.84"), ("Genesis GV60", "2/2 / 2, P(BEV) 0.78")):
        idx = np.flatnonzero((m["model"] == model).to_numpy() & (yf >= 0))
        uid, Pu, yu, du, _ = unit_dec(Pf[idx], yf[idx], m["vehicle_id"].to_numpy()[idx])
        chk(f"{model} full: devices right / devices BEV, P(BEV)", f"{int((du == yu).sum())}/{len(uid)} / {int((du == 0).sum())}, P(BEV) {Pf[idx, 0].mean():.2f}", q)
    for key in ("V3", "V2"):
        Pv, yv, _ = ccs_load("bev_ice", key)
        idx = np.flatnonzero((m["platform"] == "NISSAN_LEAF").to_numpy())
        print(f"    NISSAN_LEAF platform decision {key}: P(BEV) {Pv[idx, 0].mean():.2f} -> {'BEV' if Pv[idx, 0].mean() >= 0.5 else 'ICE'} (report: V3 and V2 call it ICE)")


def _internal_per_model_rows(task: str) -> dict:
    """{model: (label, drive_acc %, decision, p_true, hit)} of the `full` column of internal_per_model.md."""
    out, cur, cols = {}, None, None
    for line in (HERE_ / "internal_per_model.md").read_text().splitlines():
        if line.startswith("## "):
            cur = line[3:].split(" (")[0].strip(); cols = None; continue
        if cur != task or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "held-out model":
            cols = cells; continue
        if cols is None or len(cells) != len(cols) or "full" not in cols or set(cells[0]) <= {"-"}:
            continue
        row = dict(zip(cols, cells))
        hit = _re3.match(r"(\d+) / (\w+) \(([\d.]+)\)( x)?$", row["full"])
        if hit:
            out[row["held-out model"]] = (row["label (footprint)"].split(" (")[0], int(hit.group(1)), hit.group(2), float(hit.group(3)), hit.group(4) is None)
    return out


def sec24_ccs_heldout():
    print("\n## 24. Table (c) of 2.12: the 29 held-out models against internal_per_model.md (full models), and the agreement counts")
    m = ccs_meta()
    ho = (m["stratum"] == "held-out").to_numpy()
    models = sorted(set(m.loc[ho, "model"]))
    chk("held-out models present", len(models), 29)
    agree = {"size": [0, 0, 0, 0], "bev_ice": [0, 0]}  # size: ext hits, int hits, both hit, both miss
    lost, gained = [], []
    for task, k in (("size", 3), ("bev_ice", 2)):
        P, y, cl = ccs_load(task, "full"); ipm = _internal_per_model_rows(task)
        for model in models:
            idx = np.flatnonzero(ho & (m["model"] == model).to_numpy() & (y >= 0))
            line = next((ln for ln in _SEC212.splitlines() if ln.startswith(f"| {model} |")), None) if results_text() else None
            if len(idx) == 0:
                g = m[ho & (m["model"] == model).to_numpy()]
                cell = f"- ({len(g)} windows, unlabelled)" if task == "bev_ice" and len(g) <= 10 else "- (unlabelled)"
                ok = line is not None and (cell in line or "- (unlabelled)" in line)
                _NFOUND[0 if ok else 1] += 1
                print(f"    {'FOUND    ' if ok else 'NOT FOUND'} (c) {task} {model}: {cell} (pt label {g['pt_label'].iloc[0]})")
                continue
            uid, Pu, yu, du, _ = unit_dec(P[idx], y[idx], m["vehicle_id"].to_numpy()[idx])
            yt = int(np.bincount(y[idx], minlength=k).argmax()); pm = P[idx].mean(0); d = cl[int(pm.argmax())]
            ext = f"{int((du == yu).sum())}/{len(uid)} ({pct((du == yu).mean())}), {d} ({pm[yt]:.2f}) {'OK' if d == cl[yt] else 'miss'}"
            ir = ipm.get(model)
            internal = f"{ir[1]} / {ir[2]} ({ir[3]:.2f})" + ("" if ir[4] else " miss") if ir else "-"
            ok = line is not None and ext in line and internal in line
            _NFOUND[0 if ok else 1] += 1
            print(f"    {'FOUND    ' if ok else 'NOT FOUND'} (c) {task} {model}: | {cl[yt]} | {ext} | {internal} |" + ("" if ir else "  (no internal row)") + ("" if (ir is None or ir[0] == cl[yt] or model == 'Hyundai Kona') else f"  LABEL DIFFERS internally: {ir[0]}"))
            if ir:
                eh = d == cl[yt]; ih = ir[4]
                if task == "size":
                    agree["size"][0] += eh; agree["size"][1] += ih; agree["size"][2] += eh and ih; agree["size"][3] += (not eh) and (not ih)
                    if ih and not eh: lost.append(model)
                    if eh and not ih: gained.append(model)
                else:
                    agree["bev_ice"][0] += eh; agree["bev_ice"][1] += ih
    a = agree["size"]
    chk("size: external hits / internal hits / agree (both hit, both miss)", f"{a[0]} / {a[1]} / {a[2] + a[3]} ({a[2]}, {a[3]})", "13 / 19 / 19 (11, 8)")
    chk("size: lost externally", sorted(lost), sorted(["Ford Ranger", "Genesis GV70", "Genesis GV80", "Hyundai Azera", "Hyundai Santa Cruz", "Kia Carnival", "Kia Optima", "Toyota Sienna"]))
    chk("size: gained externally", sorted(gained), ["Jeep Grand Cherokee", "Kia Stinger"])
    chk("powertrain: external hits / internal hits of the labelled held-out models (each on its own label)", f"{agree['bev_ice'][0]} / {agree['bev_ice'][1]}", "27 / 27")


def sec25_ccs_seen_absent():
    print("\n## 25. Table (d) of 2.12: seen models (model-level hits, median device accuracy, device x model pairs, ten lowest P(true), the MID / LARGE hits, the ICE MID list), powertrain seen, absent models")
    m = ccs_meta()
    veh = m["vehicle_id"].to_numpy()
    recs = {}
    for task, k in (("size", 3), ("bev_ice", 2)):
        P, y, cl = ccs_load(task, "full")
        for st in ("seen", "absent"):
            rows = []
            for model in sorted(set(m.loc[(m["stratum"] == st) & (y >= 0), "model"])):
                idx = np.flatnonzero((m["stratum"] == st).to_numpy() & (m["model"] == model).to_numpy() & (y >= 0))
                uid, Pu, yu, du, _ = unit_dec(P[idx], y[idx], veh[idx])
                yt = int(np.bincount(y[idx], minlength=k).argmax()); pm = P[idx].mean(0); d = cl[int(pm.argmax())]
                rows.append(dict(model=model, label=cl[yt], border=bool(m.loc[idx, "borderline"].astype(bool).any()), n=len(uid), right=int((du == yu).sum()), win=len(idx), p=float(pm[yt]), dec=d, hit=d == cl[yt], ptrue=pm[yt], pdec=pm.max(), yt=yt))
            recs[(task, st)] = rows
    r = recs[("size", "seen")]
    hits = [x for x in r if x["hit"]]
    chk("size seen: models right (S/M/L); median per-model device acc; pairs right / pairs (share)",
        f"{len(hits)}/{len(r)} ({sum(x['label'] == 'SMALL' for x in hits)}/{sum(x['label'] == 'SMALL' for x in r)}, {sum(x['label'] == 'MID' for x in hits)}/{sum(x['label'] == 'MID' for x in r)}, {sum(x['label'] == 'LARGE' for x in hits)}/{sum(x['label'] == 'LARGE' for x in r)}); "
        f"{pct(np.median([x['right'] / x['n'] for x in r]))} %; {fmt_int(sum(x['right'] for x in r))} of {fmt_int(sum(x['n'] for x in r))} ({pct(sum(x['right'] for x in r) / sum(x['n'] for x in r))} %)",
        "40/70 (31/32, 4/20, 5/18); 59.6 %; 1,743 of 3,245 (53.7 %)")
    for x in sorted(r, key=lambda x: x["p"])[:10]:
        found("(d) seen lowest", f"| {x['model']} | {x['label']}{' (border)' if x['border'] else ''} | {x['n']} | {fmt_int(x['win'])} | {x['right']}/{x['n']} ({pct(x['right'] / x['n'])}) | {x['p']:.2f} | {x['dec']} {'OK' if x['hit'] else 'miss'} |")
    mid_hits = [(x["model"], f"{x['right']}/{x['n']}", f"{x['p']:.2f}") for x in r if x["hit"] and x["label"] == "MID"]
    print(f"    seen MID hits: {mid_hits} (report: Model 3 17/28 0.53, Ioniq 5 46/101 0.49, EV6 70/87 0.63, Mach-E 7/12 0.46)")
    lg_hits = [(x["model"], f"{x['right']}/{x['n']}", f"{x['p']:.2f}") for x in r if x["hit"] and x["label"] == "LARGE"]
    print(f"    seen LARGE hits: {lg_hits} (report: Silverado 1500 31/52 0.42, RAM 1500 57/99 0.47, RAM 2500 / 3500 HD 4/5 0.49, F-150 25/42 0.44, Traverse 1/1 0.46)")
    want = {"Honda Accord": "55/120", "Toyota Highlander": "42/225", "Toyota Camry": "12/167", "Lexus ES": "21/68", "Hyundai Sonata": "4/56", "Hyundai Tucson": "3/27", "Acura RDX": "3/21 (0 called MID)", "Subaru Ascent": "0/34", "Skoda Superb": "0/10", "Genesis G70": "0/4", "Hyundai Ioniq 6": "7/24",
            "Hyundai Palisade": "24/63", "Mazda CX-9": "4/13", "Chrysler Pacifica": "27/184", "Ford Explorer": "5/44", "Ford Maverick": "2/32", "Honda Odyssey": "2/96", "Honda Pilot": "2/77", "Honda Ridgeline": "2/56", "Volkswagen Transporter": "0/2",
            "Tesla Model 3": "17/28", "Hyundai Ioniq 5": "46/101", "Kia EV6": "70/87", "Ford Mustang Mach-E": "7/12", "Chevrolet Silverado 1500": "31/52", "RAM 1500": "57/99", "Ford F-150": "25/42"}
    by = {x["model"]: x for x in r}
    for model, q in want.items():
        x = by.get(model)
        if model == "Acura RDX" and x:
            idx = np.flatnonzero((m["stratum"] == "seen").to_numpy() & (m["model"] == model).to_numpy()); _, _, _, du_, _ = unit_dec(P, y, veh[idx]) if False else unit_dec(ccs_load("size", "full")[0][idx], ccs_load("size", "full")[1][idx], veh[idx])
            chk(f"seen {model} devices right by own label (devices called MID; report quotes the MID count)", f"{x['right']}/{x['n']} ({int((du_ == 1).sum())} called MID)", q)
            continue
        chk(f"seen {model} devices right ({x['label'] if x else '?'})", f"{x['right']}/{x['n']}" if x else "absent from the seen stratum", q)
    print("    RAM HD model name in the set:", [x["model"] for x in r if "RAM" in x["model"]])
    r = recs[("bev_ice", "seen")]
    hits = [x for x in r if x["hit"]]
    chk("powertrain seen: models right (BEV/ICE); median; pairs", f"{len(hits)}/{len(r)} ({sum(x['label'] == 'BEV' for x in hits)}/{sum(x['label'] == 'BEV' for x in r)}, {sum(x['label'] == 'ICE' for x in hits)}/{sum(x['label'] == 'ICE' for x in r)}); {pct(np.median([x['right'] / x['n'] for x in r]))} %; {fmt_int(sum(x['right'] for x in r))} of {fmt_int(sum(x['n'] for x in r))} ({pct(sum(x['right'] for x in r) / sum(x['n'] for x in r))} %)",
        "50/51 (8/9, 42/42); 96.4 %; 1,772 of 2,059 (86.1 %)")
    print("    lowest P(true), powertrain seen: " + "; ".join(f"{x['model']} ({x['label']}, {x['right']}/{x['n']}, {x['p']:.2f})" for x in sorted(r, key=lambda x: x["p"])[:10]))
    print("    report: Honda e 0.34; Bronco Sport ICE 7/8 0.62; Highlander ICE 148/225 0.64; Escalade ESV 0.67; RAV4 ICE 224/307 0.68; Focus 0.69; Transporter 0.72; Lexus IS 14/16 0.72; Rogue 0.73; Bolt EUV BEV 118/160 0.73")
    bolt = m[(m["model"] == "Chevrolet Bolt EUV")]
    chk("Bolt EUV windows / devices / median windows per device", f"{len(bolt)} / {bolt['vehicle_id'].nunique()} / {bolt.groupby('vehicle_id').size().median():.0f}", "380 / 160 / 2")
    # RAV4 TSS2 / Highlander ICE device accuracy (Reading bullet 1: 65 of 113, 148 of 225) and the three ICE platforms' device accuracy (caveat: 65.8 / 73.0 / 87.0)
    P, y, cl = ccs_load("bev_ice", "full")
    for plat_or_model, col, q in (("TOYOTA_RAV4_TSS2", "platform", "65/113"), ("Toyota Highlander", "model", "148/225"), ("Toyota RAV4", "model", "224/307")):
        idx = np.flatnonzero((m[col] == plat_or_model).to_numpy() & (y >= 0))
        uid, Pu, yu, du, _ = unit_dec(P[idx], y[idx], veh[idx])
        chk(f"{plat_or_model} ICE devices right", f"{int((du == yu).sum())}/{len(uid)}", q)
    for model, q in (("Toyota Highlander", "65.8"), ("Toyota RAV4", "73.0"), ("Lexus RX", "87.0")):
        idx = np.flatnonzero((m["model"] == model).to_numpy() & (y >= 0))
        uid, Pu, yu, du, _ = unit_dec(P[idx], y[idx], veh[idx])
        chk(f"caveat: {model} ICE device accuracy %", pct((du == yu).mean()), q)
    # absent table
    ra = {x["model"]: x for x in recs[("size", "absent")]}; rb = {x["model"]: x for x in recs[("bev_ice", "absent")]}
    chk("absent size models / powertrain-labelled absent models", f"{len(ra)} / {len(rb)}", "14 / 10")
    for model, x in sorted(ra.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        xb = rb.get(model)
        if xb:
            ptc = f"{xb['right']}/{xb['n']}, {xb['dec']} ({xb['p']:.2f}) {'OK' if xb['hit'] else 'miss'}"
        else:
            idx = np.flatnonzero((m["model"] == model).to_numpy())
            uid, Pu, _, _, _ = unit_dec(P[idx], np.zeros(len(idx), int), veh[idx])
            ptc = f"unlabelled: {pct((P[idx].argmax(1) == 0).mean())} % ... {int((Pu.argmax(1) == 0).sum())}/{len(uid)}"
        row = f"| {model} | {x['label']}{' (border)' if x['border'] else ''} / {m.loc[m['model'] == model, 'pt_label'].iloc[0]} | {x['n']} | {x['win']} | {x['right']}/{x['n']}, {x['dec']} ({x['p']:.2f}) {'OK' if x['hit'] else 'miss'} | {ptc} |"
        if xb:
            found("(d) absent", row)
        else:
            line = next((ln for ln in _SEC212.splitlines() if ln.startswith(f"| {model} |")), "")
            ok = row.split(" | unlabelled")[0] in line and ptc.split(" ...")[0] in line and ptc.split("... ")[1] in line
            _NFOUND[0 if ok else 1] += 1
            print(f"    {'FOUND    ' if ok else 'NOT FOUND'} (d) absent {model}: {row}")
    chk("absent: SMALL models right / devices", f"{sum(1 for x in ra.values() if x['label'] == 'SMALL' and x['hit'])}/{sum(1 for x in ra.values() if x['label'] == 'SMALL')} / {sum(x['right'] for x in ra.values() if x['label'] == 'SMALL')}/{sum(x['n'] for x in ra.values() if x['label'] == 'SMALL')}", "10/10 / 32/32")
    chk("absent: powertrain models right", f"{sum(1 for x in rb.values() if x['hit'])}/{len(rb)}", "9/10")


def sec26_ccs_lead_state():
    print("\n## 26. Table (e) of 2.12: lead source and automation state per stratum (full models), the other powertrain runs by state, mean P(BEV) on ICE windows")
    m = ccs_meta()
    das = m["das_frac"].to_numpy(float); veh = m["vehicle_id"].to_numpy()
    Ps, ys, _ = ccs_load("size", "full"); Pb, yb, _ = ccs_load("bev_ice", "full")
    state = np.where(das >= 0.5, "engaged", "human")
    for factor, values, arr in (("lead", ("radar", "stock ACC", "none"), m["lead"].to_numpy(str)), ("state", ("human", "engaged"), state)):
        for val in values:
            v_arr = "scc" if val == "stock ACC" else val
            for s, sel in _strata_sel(m):
                cell = sel & (arr == v_arr)
                cells = []
                for P, y in ((Ps, ys), (Pb, yb)):
                    idx = np.flatnonzero(cell & (y >= 0))
                    uid, Pu, yu, du, _ = unit_dec(P[idx], y[idx], veh[idx])
                    cells.append(f"{fmt_int(len(idx))} / {fmt_int(len(uid))} | {pct((P[idx].argmax(1) == y[idx]).mean())} | {pct((du == yu).mean())} ({fmt_int((du == yu).sum())}/{fmt_int(len(uid))})")
                row = f"| {cells[0]} | {cells[1]} |"
                if s == "absent":
                    print(f"    (e) absent {factor} {val}: {row}  [RESULTS_TABLES.md only; report: radar 12 devices, stock ACC 4, none 25; human 34, engaged 30]")
                else:
                    found(f"(e) {factor} {val} {s}", row)
    hum = state == "human"; eng = ~hum
    for key, mask, label, q in (("full", "full", "full", "92.2 / 64.8 and 93.8 / 70.9"), ("V3", "full", "V3", "86.0 / 73.5 and 94.1 / 84.3"), ("V2", "full", "V2", "78.3 / 68.8 and 88.7 / 73.9"), ("mask", "mask-lead", "full + mask-lead", "93.2 / 73.3 and 94.8 / 81.8")):
        P, y, _ = ccs_load("bev_ice", key, mask)
        out = []
        for selst in (hum, eng):
            idx = np.flatnonzero(selst & (y >= 0))
            out.append(pct((P[idx].argmax(1) == y[idx]).mean()))
        for selst in (hum, eng):
            idx = np.flatnonzero(selst & (y >= 0))
            uid, Pu, yu, du, _ = unit_dec(P[idx], y[idx], veh[idx])
            out.append(pct((du == yu).mean()))
        chk(f"powertrain {label}: window acc human / engaged and device acc human / engaged", f"{out[0]} / {out[1]} and {out[2]} / {out[3]}", q)
        ice = y == 1
        if label == "V2":
            print(f"    (info) powertrain V2: mean P(BEV) on ICE windows human / engaged {P[ice & hum, 0].mean():.2f} / {P[ice & eng, 0].mean():.2f} (not quoted in the report)")
        else:
            chk(f"powertrain {label}: mean P(BEV) on ICE windows human / engaged", f"{P[ice & hum, 0].mean():.2f} / {P[ice & eng, 0].mean():.2f}", {"full": "0.12 / 0.39", "V3": "0.17 / 0.31", "full + mask-lead": "0.10 / 0.32"}[label])
    chk("radar rows share of Toyota / Lexus windows", pct(m.loc[m['lead'] == 'radar', 'platform'].str.startswith(('TOYOTA', 'LEXUS')).mean()) + " %", "80.3 %")


def sec27_ccs_evidence_confusion():
    print("\n## 27. Tables (f) and (g) of 2.12: evidence per device, predicted-share confusion (full and yaw-masked), the absent rows")
    m = ccs_meta(); veh = m["vehicle_id"].to_numpy()
    buckets = (("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10+", 10, 10 ** 9))
    for task, k in (("size", 3), ("bev_ice", 2)):
        P, y, cl = ccs_load(task, "full")
        for s, sel in _strata_sel(m):
            idx = np.flatnonzero(sel & (y >= 0))
            uid, Pu, yu, du, cnt = unit_dec(P[idx], y[idx], veh[idx])
            cells = []
            for name, lo, hi in buckets:
                b = (cnt >= lo) & (cnt <= hi)
                cells.append(f"{fmt_int((du[b] == yu[b]).sum())}/{fmt_int(b.sum())} ({pct((du[b] == yu[b]).mean())})" if b.any() else "-")
            found(f"(f) {task} {s}", f"| {'powertrain' if task == 'bev_ice' else task} | {s} | {fmt_int(len(uid))} | " + " | ".join(cells) + " |")
    Pf, y, cl = ccs_load("size", "full"); Py, _, _ = ccs_load("size", "mask", "mask-yaw")
    for run, P in (("full", Pf), ("full + mask-yaw", Py)):
        for s, sel in _strata_sel(m):
            for c in range(3):
                idx = np.flatnonzero(sel & (y == c))
                share = np.bincount(P[idx].argmax(1), minlength=3) / len(idx); pm = P[idx].mean(0)
                row = f"| {run} | {s} | {cl[c]} | {fmt_int(len(np.unique(veh[idx])))} | {fmt_int(len(idx))} | {', '.join(pct(x) for x in share)} | {', '.join(f'{x:.2f}' for x in pm)} |"
                if s == "absent":
                    if run == "full":
                        print(f"    (g) absent {cl[c]}: shares {' / '.join(pct(x) for x in share)} devices {len(np.unique(veh[idx]))} (report: SMALL 85.2 / 13.8 / 1.0, MID (1 device) 68.0 / 32.0 / 0.0, LARGE (3 devices) 75.4 / 19.7 / 4.9)")
                else:
                    found(f"(g) {run} {s} {cl[c]}", row)


def sec28_ccs_unlabelled():
    print("\n## 28. Table (h) of 2.12: unlabelled powertrains per platform (>= 100 windows) and the pooled shares")
    m = ccs_meta(); veh = m["vehicle_id"].to_numpy()
    P, y, cl = ccs_load("bev_ice", "full")
    unl = y < 0
    n_rows = 0
    for (p, lab), g in m.loc[unl].groupby(["platform", "pt_label"], sort=True):
        idx = g.index.to_numpy()
        uid, Pu, _, _, _ = unit_dec(P[idx], np.zeros(len(idx), int), veh[idx])
        row = f"| {p} | {g['model'].iloc[0]} | {lab} | {g['stratum'].iloc[0]} | {len(uid)} | {len(idx)} | {pct((P[idx].argmax(1) == 0).mean())} | {int((Pu.argmax(1) == 0).sum())}/{len(uid)} | {P[idx, 0].mean():.2f} |"
        if len(idx) >= 100:
            found("(h)", row); n_rows += 1
    chk("rows with >= 100 windows", n_rows, 45)
    chk("unlabelled platforms (pooled count)", m.loc[unl, "platform"].nunique(), 53)
    for lab, q in (("PHEV", "44.6 % of 2,392 (69 of 214)"), ("HEV", "32.0 % of 9,386 (245 of 1,093)"), ("FCEV", "36.5 % of 296 (0 of 6)"), ("MIXED", "19.9 % of 3,867 (23 of 275)")):
        idx = np.flatnonzero(unl & (m["pt_label"] == lab).to_numpy())
        uid, Pu, _, _, _ = unit_dec(P[idx], np.zeros(len(idx), int), veh[idx])
        chk(f"pooled {lab}: windows BEV % of windows (devices BEV of devices)", f"{pct((P[idx].argmax(1) == 0).mean())} % of {fmt_int(len(idx))} ({int((Pu.argmax(1) == 0).sum())} of {fmt_int(len(uid))})", q)
    chk("labelled ICE windows called BEV / labelled BEV windows called BEV %", f"{pct((P[y == 1].argmax(1) == 0).mean())} / {pct((P[y == 0].argmax(1) == 0).mean())}", "17.5 / 78.7")
    idx = np.flatnonzero(unl & (m["pt_label"] == "unknown").to_numpy())
    print(f"    non-MIXED unknown windows (not in the pooled MIXED count): {len(idx)} on {sorted(m.loc[idx, 'platform'].unique().tolist())}")


def sec29_ccs_pooled():
    print("\n## 29. Pooled decisions (RESULTS.md 0 / 3.1 / 3.2): phase-3 device x car-model pairs and the all-sets totals")
    m = ccs_meta()
    prim = dict(PRIMARY); prim.update(PRIMARY2)
    for task, k, q3, qall in (("bev_ice", 2, "BEV 544/630  ICE 1,688/1,917  all 2,232/2,547", "BEV 575/667  ICE 1,994/2,239  all 2,569/2,906"),
                              ("size", 3, "SMALL 1,635/1,798  MID 360/1,301  LARGE 250/1,053  all 2,245/4,152", "SMALL 1,683/1,850  MID 366/1,338  LARGE 254/1,072  all 2,303/4,260")):
        cls = T.TASK_LABEL[task][1]
        P, y, _ = ccs_load(task, "full")
        r = unit_eval(P, y, m["pair"].to_numpy(), k)
        s3 = "  ".join(f"{c} {fmt_int(r['cm'][i, i])}/{fmt_int(r['sup'][i])}" for i, c in enumerate(cls)) + f"  all {fmt_int(np.trace(r['cm']))}/{fmt_int(r['n'])}"
        chk(f"{task} phase 3 (device x model pairs)", s3, q3)
        rv = unit_eval(P, y, m["vehicle_id"].to_numpy(), k)
        print(f"    {task} phase 3 at device level (majority label): " + "  ".join(f"{c} {fmt_int(rv['cm'][i, i])}/{fmt_int(rv['sup'][i])}" for i, c in enumerate(cls)) + f"  all {fmt_int(np.trace(rv['cm']))}/{fmt_int(rv['n'])}  (2.12 Reading: SMALL 1,606/1,762, MID 351/1,274, LARGE 241/1,025)")
        hit = {c: [int(r["cm"][i, i]), int(r["sup"][i])] for i, c in enumerate(cls)}
        for ds in prim:
            Pd, yd, md, _ = load(ds, task, prim[ds][task])
            rd = unit_eval(Pd, yd, md["vehicle_id"].to_numpy(), k)
            if rd["n"] == 0:
                continue
            for i, c in enumerate(cls):
                hit[c][0] += int(rd["cm"][i, i]); hit[c][1] += int(rd["sup"][i])
        sall = "  ".join(f"{c} {fmt_int(hit[c][0])}/{fmt_int(hit[c][1])}" for c in cls) + f"  all {fmt_int(sum(h[0] for h in hit.values()))}/{fmt_int(sum(h[1] for h in hit.values()))}"
        chk(f"{task} all sets", sall, qall)
        for c in cls:
            chk(f"{task} all sets {c} share", pct(hit[c][0] / hit[c][1]) + " %", {"BEV": "86.2 %", "ICE": "89.1 %", "SMALL": "91.0 %", "MID": "27.4 %", "LARGE": "23.7 %"}[c])
        chk(f"{task} all sets overall share", pct(sum(h[0] for h in hit.values()) / sum(h[1] for h in hit.values())) + " %", "88.4 %" if task == "bev_ice" else "54.1 %")
        chk(f"{task} phase-3 shares", "  ".join(pct(r['cm'][i, i] / r['sup'][i]) for i in range(k)) + "  " + pct(np.trace(r['cm']) / r['n']), "86.3  88.1  87.6" if task == "bev_ice" else "90.9  27.7  23.7  54.1")


def sec30_ccs_labels():
    print("\n## 30. Labels and strata of the ccs set: footprints against the lab table / specs_external.csv with the 4.43 / 4.88 cuts, strata against build/index/drives.parquet, powertrain against the lab platform table")
    m = ccs_meta()
    chk("cuts", T.THRESHOLDS if hasattr(T, "THRESHOLDS") else None, (4.43, 4.88))
    fp = m["footprint_m2"].to_numpy(float)
    chk("size_class == classify_footprint(footprint) on every window", bool((T.classify_footprint(fp) == m["size_class"].to_numpy(object)).all()), True)
    chk("borderline flag == |fp - cut| < 0.05 on every window", bool((T.is_borderline(fp) == m["borderline"].astype(bool).to_numpy()).all()), True)
    specs = pd.read_csv(REPO / "vtt" / "specs_data" / "platform_specs.csv", dtype=str, keep_default_na=False).set_index("platform")
    ext = pd.read_csv(HERE_ / "specs_external.csv", dtype=str, keep_default_na=False)
    ext_keys = {}
    for r in ext.itertuples(index=False):
        for k in [x.strip() for x in r.platform_keys.replace(";", " ").replace(",", " ").split() if x.strip()]:
            ext_keys.setdefault(k, r)
    n_lab = n_pub = 0; worst_lab = worst_pub = (None, 0.0); bad_src = []; model_mism = []
    for plat, g in m.groupby("platform"):
        f0 = float(g["footprint_m2"].iloc[0]); src = g["label_src"].iloc[0]
        if plat in specs.index and src == "lab":
            r = specs.loc[plat]
            try:
                lf = float(r["wheelbase_mm"]) / 1000 * (float(r["track_front_mm"]) + float(r["track_rear_mm"])) / 2 / 1000
            except ValueError:
                lf = float("nan")
            if np.isfinite(lf):
                n_lab += 1
                if abs(lf - f0) > worst_lab[1]:
                    worst_lab = (plat, abs(lf - f0))
            if r["model"] and r["model"] != g["model"].iloc[0]:
                model_mism.append((plat, r["model"], g["model"].iloc[0]))
        elif plat in ext_keys and src == "public":
            r = ext_keys[plat]
            try:
                pf = float(r.wheelbase_mm) / 1000 * (float(r.track_front_mm) + float(r.track_rear_mm)) / 2 / 1000
            except ValueError:
                pf = float(r.footprint_m2) if r.footprint_m2 else float("nan")
            n_pub += 1
            if abs(pf - f0) > worst_pub[1]:
                worst_pub = (plat, abs(pf - f0))
            if r.model != g["model"].iloc[0]:
                model_mism.append((plat, r.model, g["model"].iloc[0]))
        else:
            bad_src.append((plat, src, plat in specs.index, plat in ext_keys))
    chk("lab-labelled platforms checked against the lab table (max |diff| m^2)", f"{n_lab} ({worst_lab[1]:.4f})", f"{m.loc[m['label_src'] == 'lab', 'platform'].nunique()} (0.0000)")
    chk("public-labelled platforms checked against specs_external.csv (max |diff| m^2)", f"{n_pub} ({worst_pub[1]:.4f})", "22 (0.0000)")
    chk("platforms whose label source does not match the table that holds them", bad_src, [])
    chk("model-name mismatches between the tables and the meta", model_mism, [])
    print(f"    public-spec platforms: {sorted(m.loc[m['label_src'] == 'public', 'platform'].unique().tolist())}")
    # strata vs drives index
    d = pd.read_parquet(T.DATA_ROOT / "build" / "index" / "drives.parquet", columns=["model", "platform", "split_platform", "powertrain_raw"])
    dm = d[d["model"].notna() & (d["model"] != "Unknown")]
    splits = dm.groupby("model")["split_platform"].agg(lambda s: set(s.dropna()))
    seen = {k for k, v in splits.items() if v & {"train", "validation"}}; held = {k for k, v in splits.items() if "test" in v}
    chk("drives index: seen models / held-out test models / models in both", f"{len(seen)} / {len(held)} / {sorted(seen & held)}", f"{len(seen)} / 41 / []")
    exp = m["model"].map(lambda x: "seen" if x in seen else ("held-out" if x in held else "absent"))
    chk("stratum mismatches against the drives index (windows)", int((exp != m["stratum"]).sum()), 0)
    chk("held-out models present of the 41", f"{len(set(m.loc[m['stratum'] == 'held-out', 'model']))} / 41; missing: {sorted(held - set(m['model']))}", "29 / 41; missing: " + str(sorted(held - set(m["model"]))))
    excl_only = {k for k, v in splits.items() if v and v <= {"excluded"}}
    print(f"    absent models that exist in the lab index only as excluded drives: {sorted(set(m.loc[m['stratum'] == 'absent', 'model']) & excl_only)}; absent models in the lab inventory at all: {sorted(set(m.loc[m['stratum'] == 'absent', 'model']) & set(dm['model']))}")
    # powertrain vs the lab platform table (drives index mode per platform)
    lab_pt = d[d["platform"].notna()].groupby("platform")["powertrain_raw"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else "UNK")
    vocab = {"ICE": "ICE", "HEV": "HEV", "PHEV": "PHEV", "BEV": "BEV", "FCEV": "FCEV"}
    mism = []
    for plat, g in m.groupby("platform"):
        if plat in lab_pt.index:
            want = vocab.get(lab_pt[plat], "MIXED" if lab_pt[plat] == "MIXED" else "unknown")
            have = g["pt_label"].iloc[0]
            if want != have:
                mism.append((plat, lab_pt[plat], have))
    chk("powertrain mismatches against the lab drives index (platforms present there)", mism, [])
    n_in = sum(p in lab_pt.index for p in m["platform"].unique())
    print(f"    platforms present in the lab drives index: {n_in} of {m['platform'].nunique()}; the rest take the taxonomy / public label")
    for plat in ("TOYOTA_HIGHLANDER_TSS2", "TOYOTA_HIGHLANDER", "TOYOTA_RAV4_TSS2", "TOYOTA_RAV4_TSS2_2023", "LEXUS_RX_TSS2", "LEXUS_RX"):
        g = d[d["platform"] == plat]
        print(f"    lab drives of {plat}: {g['powertrain_raw'].value_counts(dropna=False).to_dict()} -> label {m.loc[m['platform'] == plat, 'pt_label'].iloc[0] if (m['platform'] == plat).any() else '(not in ccs)'} (caveat of 2.12: hybrid-majority mixes in some years)")
    print(f"    ccs ICE-labelled Toyota / Lexus platforms: {sorted(m.loc[(m['pt_label'] == 'ICE') & m['platform'].str.startswith(('TOYOTA', 'LEXUS')), 'platform'].unique().tolist())}")
    # vocabulary and pt per platform constant
    per_p = m.groupby("platform").agg(pt=("pt_label", "nunique"), sc=("size_class", "nunique"), mo=("model", "nunique"), st=("stratum", "nunique"))
    chk("platforms with more than one powertrain / size class / model / stratum", int((per_p > 1).any(axis=1).sum()), 0)
    chk("powertrain vocabulary", sorted(m["pt_label"].unique().tolist()), ["BEV", "FCEV", "HEV", "ICE", "MIXED", "PHEV", "unknown"])


def sec31_ccs_internal_refs():
    import json as _json
    print("\n## 31. Internal references quoted in 2.12 (a) / (b) and 1.2 against internal_reference.json and internal_per_model.md")
    r = _json.load(open(HERE_ / "internal_reference.json"))["tasks"]
    names = {("size", "full"): "full (GRU S9, 20 ch)", ("size", "V1"): "V1 lateral + speed (10 ch)", ("size", "V2"): "V2 longitudinal-only (11 ch)", ("size", "V3"): "V3 speed-only (3 ch)",
             ("size", "mask-V1"): "full (GRU S9, 20 ch) + mask-V1", ("size", "mask-V2"): "full (GRU S9, 20 ch) + mask-V2", ("size", "mask-V3"): "full (GRU S9, 20 ch) + mask-V3",
             ("bev_ice", "full"): "full (ResNet B5, 20 ch)", ("bev_ice", "V1"): "V1 lateral + speed (10 ch)", ("bev_ice", "V2"): "V2 longitudinal-only (11 ch)", ("bev_ice", "V3"): "V3 speed-only (3 ch)",
             ("bev_ice", "mask-V1"): "full (ResNet B5, 20 ch) + mask-V1", ("bev_ice", "mask-V2"): "full (ResNet B5, 20 ch) + mask-V2", ("bev_ice", "mask-V3"): "full (ResNet B5, 20 ch) + mask-V3"}
    quoted_b = {("size", "full"): "63.5 / 59.7", ("size", "V1"): "62.6 / 58.0", ("size", "V2"): "43.2 / 32.8", ("size", "V3"): "54.4 / 45.6", ("size", "mask-V1"): "60.8 / 50.8", ("size", "mask-V2"): "48.0 / 23.2", ("size", "mask-V3"): "49.2 / 25.3",
                ("bev_ice", "full"): "93.9 / 93.4", ("bev_ice", "V1"): "75.8 / 74.6", ("bev_ice", "V2"): "91.4 / 90.4", ("bev_ice", "V3"): "91.0 / 90.1", ("bev_ice", "mask-V1"): "69.3 / 61.2", ("bev_ice", "mask-V2"): "69.3 / 56.8", ("bev_ice", "mask-V3"): "65.2 / 49.8"}
    for (task, key), nm in names.items():
        a = r[task][nm]["subsets"]["all"]["vehicle"]
        chk(f"(b) internal vehicle {task} {key}", f"{pct(a['accuracy'])} / {pct(a['macro_f1'])}", quoted_b[(task, key)])
    for task, q in (("size", "60.0 / 54.3 window, 70.8 / 67.3 drive, 63.5 / 59.7 vehicle"), ("bev_ice", "82.5 / 80.7 window, 93.7 / 93.3 drive, 93.9 / 93.4 vehicle")):
        s = r[task][names[(task, "full")]]["subsets"]["all"]
        chk(f"(a) internal {task} full", f"{pct(s['window']['accuracy'])} / {pct(s['window']['macro_f1'])} window, {pct(s['drive']['accuracy'])} / {pct(s['drive']['macro_f1'])} drive, {pct(s['vehicle']['accuracy'])} / {pct(s['vehicle']['macro_f1'])} vehicle", q)
    s = r["bev_ice"][names[("bev_ice", "full")]]["subsets"]
    chk("Reading bullet 2: internal full powertrain vehicle acc all / das (engaged)", f"{pct(s['all']['vehicle']['accuracy'])} / {pct(s['das']['vehicle']['accuracy'])}", "93.9 / 93.2")
    sz = r["size"][names[("size", "full")]]["subsets"]["all"]["vehicle"]
    chk("Reading bullet 5: internal MID / LARGE vehicle recall", f"{pct(sz['per_class_recall'][1])} / {pct(sz['per_class_recall'][2])}", "47.3 / 50.6")
    v3 = r["size"][names[("size", "V3")]]["subsets"]["all"]["vehicle"]; fu = r["size"][names[("size", "full")]]["subsets"]["all"]["vehicle"]
    chk("'nine vehicle points' full over V3 internally (size)", f"{100 * (fu['accuracy'] - v3['accuracy']):.1f}", "9.1")
    v3b = r["bev_ice"][names[("bev_ice", "V3")]]["subsets"]["all"]["vehicle"]; fub = r["bev_ice"][names[("bev_ice", "full")]]["subsets"]["all"]["vehicle"]
    chk("'+2.9 vehicle points' full over V3 internally (powertrain)", f"{100 * (fub['accuracy'] - v3b['accuracy']):+.1f}", "+2.9")
    # internal_per_model.md model counts
    txt = (HERE_ / "internal_per_model.md").read_text()
    def counts(task):
        blk = txt.split(f"Model-level decisions correct per class ({task}):")[1].split("##")[0]
        out = {}
        for ln in blk.splitlines():
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            if len(c) >= 4 and c[0] not in ("model", "") and not set(c[0]) <= {"-"}:
                out[c[0]] = c[1:]
        return out
    cs = counts("size"); cb = counts("bev_ice")
    chk("internal models right, size full / V1 / V2 / V3 / +V1 / +V2 / +V3", " / ".join(cs[k][-1] for k in ("full", "V1 lateral + speed", "V2 longitudinal-only", "V3 speed-only", "full+V1", "full+V2", "full+V3")), "28/41 / 25/41 / 17/41 / 20/41 / 23/41 / 15/41 / 15/41")
    chk("internal models right, powertrain full / V1 / V2 / V3 / +V1 / +V2 / +V3", " / ".join(cb[k][-1] for k in ("full", "V1 lateral + speed", "V2 longitudinal-only", "V3 speed-only", "full+V1", "full+V2", "full+V3")), "38/39 / 33/39 / 38/39 / 38/39 / 32/39 / 30/39 / 31/39")
    chk("internal size full per class (14/15, 6/14, 8/12)", " ".join(cs["full"][:3]), "14/15 6/14 8/12")
    chk("internal powertrain full per class", " ".join(cb["full"][:2]), "8/9 30/30")
    ipm = _internal_per_model_rows("size")
    chk("internal Carnival / Sienna / Yukon drive acc % (Reading bullet 6: 77 / 45 / 100)", f"{ipm['Kia Carnival'][1]} / {ipm['Toyota Sienna'][1]} / {ipm['GMC Yukon'][1]}", "77 / 45 / 100")
    ipb = _internal_per_model_rows("bev_ice")
    chk("internal Leaf V2 / V3 drive acc % (Reading bullet 4: 18-21 %)", "see per-model: " + " ".join(ln.split("|")[8].strip().split(" / ")[0] + "/" + ln.split("|")[9].strip().split(" / ")[0] + " (V2 / V3)" for ln in txt.splitlines() if ln.startswith("| Nissan Leaf |") and "BEV" in ln), "see per-model: 21/18 (V2 / V3)")
    chk("internal Lightning full drive acc % (bullet 4: 86)", ipb["Ford F-150 Lightning"][1], 86)


def sec32_ccs_windows():
    print("\n## 32. ccs window tensor (windows.npz): derived-yaw identities, absent-channel pattern, flag shares, steering-yaw sign")
    if ARGS.no_windows:
        print("  (skipped: --no-windows)"); return
    X, m2 = T.load_windows("ccs")
    m = ccs_meta(); assert len(m2) == len(m)
    v = X[:, :, T.CH["v_mps"]]; ang = X[:, :, T.CH["steer_angle_deg"]]; yaw = X[:, :, T.CH["yaw_rate_radps"]]; lat = X[:, :, T.CH["lat_acc_mps2"]]; curv = X[:, :, T.CH["curv_1pm"]]
    chk("mean speed m/s (tensor)", f"{np.nanmean(v):.1f}", "18.0")
    chk("standstill share % / gas / brake / steer pressed (tensor)", f"{pct(np.nanmean(X[:, :, T.CH['standstill']]))} / {100 * np.nanmean(X[:, :, T.CH['gas_pressed']]):.0f} / {100 * np.nanmean(X[:, :, T.CH['brake_pressed']]):.0f} / {100 * np.nanmean(X[:, :, T.CH['steer_pressed']]):.0f}", "5.7 / 39 / 13 / 22")
    chk("das_enabled sample mean", f"{np.nanmean(X[:, :, T.CH['das_enabled']]):.2f}", "0.39")
    chk("moving share (v >= 1 m/s)", f"{np.nanmean(v >= 1.0):.2f}", "0.93")
    print(f"    max |curv * v - yaw| where defined: {np.nanmax(np.abs(curv * v - yaw)):.2e}; max |lat - v * yaw|: {np.nanmax(np.abs(lat - v * yaw)):.2e}; yaw NaN share {pct(np.isnan(yaw).mean())} % (defined where v >= 1: {bool(np.array_equal(np.isnan(yaw), v < 1.0))})")
    nan_t = np.isnan(X[:, :, T.CH["steer_torque_norm"]]).all(1); nan_l = np.isnan(X[:, :, T.CH["lead_d_m"]]).all(1)
    chk("torque channel all-NaN exactly where channels_present lacks it", bool(np.array_equal(nan_t, ~m["has_torque"].to_numpy())), True)
    chk("lead_d_m all-NaN windows = no lead group + group present but no object in the minute", f"{int(nan_l.sum())} = {int((~m['has_lead']).sum())} + {int((nan_l & m['has_lead'].to_numpy()).sum())}", "25484 = 24213 + 1271")
    lv = X[:, :, T.CH["lead_valid"]]
    chk("lead_valid = 0 on every sample of the windows without the lead group", bool((lv[~m["has_lead"].to_numpy()] == 0).all()), True)
    chk("lead=none in the note but lead group present: lead_valid all 0 in those windows", bool((lv[((m['lead'] == 'none') & m['has_lead']).to_numpy()] == 0).all()), True)
    cors = []
    for plat, g in m.groupby("platform", sort=True):
        idx = g.index.to_numpy(); sel = (v[idx] > 5.0) & np.isfinite(yaw[idx])
        if sel.sum() > 1000 and np.nanstd(ang[idx][sel]) > 0.5:
            cors.append((plat, float(np.corrcoef(ang[idx][sel], yaw[idx][sel])[0, 1])))
    c = np.array([x[1] for x in cors])
    print(f"    corr(steering angle, derived yaw) above 5 m/s per platform: min {c.min():.3f} ({cors[int(c.argmin())][0]}), median {np.median(c):.3f}; all positive: {bool((c > 0).all())} ({len(cors)} platforms)")
    rate = X[:, :, T.CH["steer_rate_degps"]]
    zero_rate = (np.nan_to_num(rate) == 0).all(1)
    br = m["platform"].str.split("_").str[0]
    print("    windows whose CAN steering rate is 0 throughout, by brand (%): " + ", ".join(f"{b} {100 * zero_rate[(br == b).to_numpy()].mean():.0f}" for b in ("FORD", "NISSAN", "VOLKSWAGEN", "HYUNDAI", "TOYOTA", "TESLA", "RIVIAN")))
    del X


def sec33_ccs_hygiene():
    print("\n## 33. Hygiene: 16-hex / 12-hex tokens, route stamps, absolute paths, ISO dates, the four-letter acronym, multi-decimal percentages in 2.12 / 3.6")
    hex16 = _re3.compile(r"(?<![0-9a-zA-Z_])[0-9a-f]{16}(?![0-9a-zA-Z])"); hex12 = _re3.compile(r"(?<![0-9a-zA-Z_])[0-9a-f]{12}(?![0-9a-zA-Z])")
    route1 = _re3.compile(r"\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}"); route2 = _re3.compile(r"\b[0-9a-f]{8}--[0-9a-f]{10}\b")
    abspath = _re3.compile(r"(?<![\w.])/(?:home|data|mnt|tmp)/"); iso = _re3.compile(r"\b\d{4}-\d{2}-\d{2}\b"); acr = _re3.compile(r"(?<![A-Za-z])" + "A" + "D" + "A" + "S(?![A-Za-z])", _re3.I)
    repo_files = [HERE_ / f for f in ("RESULTS.md", "RESULTS_TABLES.md", "results_summary.json", "CCS_PIPELINE.md", "CCS_SELECTION.md", "review_recompute.py")]
    work_files = [p for p in (EXT / "ccs" / "work").glob("*") if p.suffix in (".md", ".csv", ".json")]
    flagged = 0
    for f in repo_files + work_files:
        t = f.read_text(errors="replace")
        h16 = [x for x in hex16.findall(t) if _re3.search(r"[a-f]", x)]; h12 = [x for x in hex12.findall(t) if _re3.search(r"[a-f]", x)]
        rt = route1.findall(t) + route2.findall(t); ap = abspath.findall(t) if f in repo_files else []; isod = iso.findall(t) if f in repo_files else []; ac = acr.findall(t)
        if h16 or h12 or rt or ap or isod or ac:
            flagged += 1
            print(f"    FLAG {f.name}: hex16 {len(h16)} hex12 {len(h12)} route stamps {len(rt)} abs paths {len(ap)} {sorted(set(ap))} iso dates {len(isod)} {sorted(set(isod))[:5]} acronym {len(ac)}")
    names = [p.name for p in (EXT / "ccs" / "work").iterdir()]
    badn = [n for n in names if hex16.search(n) or hex12.search(n) or route1.search(n) or route2.search(n)]
    print(f"    files scanned {len(repo_files) + len(work_files)} (flagged {flagged}); work-folder names scanned {len(names)} (flagged {len(badn)})")
    m = ccs_meta()
    toks = sum(bool(hex16.search(str(x)) or hex12.search(str(x)) or route1.search(str(x)) or route2.search(str(x))) for col in ("vehicle_id", "drive_id", "model", "platform", "note") for x in m[col].unique())
    chk("meta.parquet strings with a long hex / route token", toks, 0)
    chk("vehicle / drive pseudonym format ccs_v_<10 hex> / ccs_d_<10 hex>", bool(m["vehicle_id"].str.fullmatch(r"ccs_v_[0-9a-f]{10}").all() and m["drive_id"].str.fullmatch(r"ccs_d_[0-9a-f]{10}").all()), True)
    qc = ccs_qc()
    chk("segments_qc.csv id columns are pseudonyms only", bool(qc["vehicle_id"].str.fullmatch(r"ccs_v_[0-9a-f]{10}").all() and qc["drive_id"].str.fullmatch(r"ccs_d_[0-9a-f]{10}").all()), True)
    results_text()
    t = (HERE_ / "RESULTS.md").read_text(); i = t.find("### 2.12 "); j = t.find("## 3. Cross-dataset summary"); k = t.find("### 3.6 "); l_ = t.find("## 4. What the owner")
    new = t[i:j] + t[k:l_]
    multi = sorted(set(_re3.findall(r"\d+\.\d{2,} %", new)))
    chk("percentages with two or more decimals in 2.12 / 3.6 prose", multi, [])
    print(f"    checks: {_NCHK[0]} OK / {_NCHK[1]} MISMATCH; table cells: {_NFOUND[0]} FOUND / {_NFOUND[1]} NOT FOUND")


# ============================================================================= phase 3b: ccs_honda + comma2k19 (sections 34-41)
# Added by the phase-3b review on 10 Nov (evening). Every table of RESULTS.md 2.13 / 2.14 is recomputed from the probs__*__ens__*.npz
# files and meta.parquet of <external>/{ccs_honda,comma2k19}/work, re-rendered in the report's own cell format and searched in the
# section ("FOUND" / "NOT FOUND"); prose figures are printed next to the quoted value ("OK" / "MISMATCH"). The decode accounting is
# checked against decode_summary.json / segments_qc.csv (and the budget list of select_summary.json for the Honda platforms), the
# pooled totals of sections 0 / 3.1 / 3.2 against every set, the labels against both spec tables and the drives index, the window
# tensors for the two shared artefacts (sample 0, oncoming lead), the derived-yaw ratios of 2.14 against comma2k19_yaw_check.py's
# output (yaw_check.json / yaw_check_all.json, run under the lab venv), and the identifier hygiene of the two work folders.
_SEC_TXT: dict = {}


def sec_text(where: str) -> str:
    """Whitespace-normalised text of a RESULTS.md section ('2.12', '2.13', '2.14', '0', '1.1', '3.1', '3.2', '3.6', '4') or the whole file."""
    results_text()
    if not _SEC_TXT:
        t = (HERE_ / "RESULTS.md").read_text()
        marks = {"0": ("## 0. Headline", "## 1. Set-up"), "1.1": ("### 1.1 ", "### 1.2 "), "1.3": ("### 1.3 ", "## 2. Results"),
                 "2.12": ("### 2.12 ", "### 2.13 "), "2.13": ("### 2.13 ", "### 2.14 "), "2.14": ("### 2.14 ", "## 3. Cross-dataset summary"),
                 "3.1": ("### 3.1 ", "### 3.2 "), "3.2": ("### 3.2 ", "### 3.3 "), "3.6": ("### 3.6 ", "## 4. What the owner"), "4": ("## 4. What the owner", "## 5. Files")}
        for k, (a, b) in marks.items():
            i = t.find(a); j = t.find(b, i + 1)
            _SEC_TXT[k] = _re3.sub(r"[ \t]+", " ", t[i:j]) if i >= 0 and j > i else ""
    return _SEC_TXT.get(where, _RESULTS_TXT)


def found_in(label: str, cell: str, where: str) -> bool:
    """found() for any section (2.12 keeps its own narrower text in found())."""
    hay = sec_text(where)
    ok = _re3.sub(r"[ \t]+", " ", cell) in hay
    _NFOUND[0 if ok else 1] += 1
    print(f"    {'FOUND    ' if ok else 'NOT FOUND'} {label}: {cell}")
    return ok


def set_meta(ds: str) -> pd.DataFrame:
    m = pd.read_parquet(EXT / ds / "work" / "meta.parquet").reset_index(drop=True)
    m["stratum"] = note_field(m, "stratum").fillna("absent")
    m["lead"] = note_field(m, "lead").fillna("none")
    m["label_src"] = note_field(m, "label").fillna("none")
    m["gearbox"] = note_field(m, "gearbox").fillna("-")
    m["pt_label"] = m["powertrain"].fillna("unknown").astype(str)
    m["n_ch"] = m["channels_present"].astype(str).str.count(",") + 1
    m["has_torque"] = m["channels_present"].astype(str).str.contains("steer_torque_norm")
    m["has_lead"] = m["channels_present"].astype(str).str.contains("lead_d_m")
    m["pair"] = m["vehicle_id"].astype(str) + "|" + m["model"].astype(str)
    m["state"] = np.where(m["das_frac"].to_numpy(float) >= 0.5, "engaged", "human")
    return m


def set_qc(ds: str) -> pd.DataFrame:
    return pd.read_csv(EXT / ds / "work" / "segments_qc.csv")


def set_load(ds: str, task: str, key: str, mask: str = "full"):
    P, y, _, cl = load(ds, task, key, mask)
    return P, y, cl


def pfmt(x: float) -> str:
    return f"{x:.2f}"


HONDA9 = ("HONDA_FIT", "HONDA_CIVIC", "HONDA_CIVIC_BOSCH", "HONDA_CIVIC_2022", "HONDA_CRV", "HONDA_CRV_5G", "HONDA_HRV", "HONDA_HRV_3G", "HONDA_FREED")


def sec34_honda_decode():
    import json as _json
    print("\n## 34. ccs_honda decode accounting and set description (RESULTS.md 2.13 first paragraph, 1.1 row, CCS_PIPELINE.md 'Honda re-decode') against decode_summary.json / segments_qc.csv / select_summary.json / meta")
    ds = _json.load(open(EXT / "ccs_honda" / "work" / "decode_summary.json")); qc = set_qc("ccs_honda"); m = set_meta("ccs_honda")
    sel = _json.load(open(EXT / "ccs" / "select_summary.json"))
    bud = pd.DataFrame(sel["platforms"]).set_index("platform")["n_budget"].fillna(0).astype(int)
    chk("segments found / routes / devices (qc)", f"{fmt_int(ds['segments_found'])} / {fmt_int(ds['routes'])} / {fmt_int(qc['vehicle_id'].nunique())}", "2,198 / 931 / 230")
    chk("budget-list segments of the nine platforms == found", f"{fmt_int(sum(bud.get(p, 0) for p in HONDA9))}", "2,198")
    chk("decoded == qc rows", f"{fmt_int(ds['segments_decoded'])} ({len(qc) == ds['segments_decoded']})", "2,189 (True)")
    chk("skipped / found == decoded + skipped", f"{ds['skipped']} / {ds['segments_found'] == ds['segments_decoded'] + ds['skipped']}", "9 / True")
    sr = ds["skip_reasons"]; st = sorted(int(k.split()[0]) for k in sr if "stale" in k)
    chk("skip reasons: canValid 0.00 / stale (n, samples)", f"{sr.get('canValid only 0.00', 0)} / {len(st)} ({st[0]}-{st[-1]})", "5 / 4 (17-31)")
    loss = {p: int(bud.get(p, 0) - (qc["platform"] == p).sum()) for p in HONDA9}
    chk("segments lost per platform (list - decoded; report: 1 Fit + 4 CR-V 5G canValid, 3 Civic + 1 HR-V stale)", sorted((k, v) for k, v in loss.items() if v), [("HONDA_CIVIC", 3), ("HONDA_CRV_5G", 4), ("HONDA_FIT", 1), ("HONDA_HRV", 1)])
    print("    (the device count of the five canValid skips, 'of 3 devices', is not recorded in the work folder or the decode log: not verifiable here)")
    mv = qc.loc[~qc["window"], "moving_frac"]
    chk("moving-rule drops / moving fraction range", f"{int((~qc['window']).sum())} / {mv.min():.0f}-{mv.max():.3f}", "57 / 0-0.175")
    chk("windows == qc window flags == meta rows", f"{fmt_int(ds['windows'])} ({int(qc['window'].sum()) == ds['windows'] == len(m)})", "2,132 (True)")
    chk("hours / devices / drives / models / platforms", f"{len(m) / 60:.1f} / {m['vehicle_id'].nunique()} / {m['drive_id'].nunique()} / {m['model'].nunique()} / {m['platform'].nunique()}", "35.5 / 230 / 913 / 5 / 9")
    chk("seconds / workers / CPU s per segment", f"{ds['seconds']:.0f} / {ds['workers']} / {qc['decode_s'].mean():.1f}", "467 / 6 / 1.3")
    chk("gearbox option on / counts cvt, automatic (decode_summary = qc)", f"{ds['honda_gearbox_from_can']} / {fmt_int(ds['gearbox_counts']['cvt'])}, {ds['gearbox_counts']['automatic']} ({qc['gearbox'].value_counts().to_dict() == ds['gearbox_counts']})", "True / 2,086, 103 (True)")
    auto = m[m["gearbox"] == "automatic"]
    chk("automatic windows / platforms / devices", f"{len(auto)} / {sorted(auto['platform'].unique())} / {auto['vehicle_id'].nunique()}", "103 / ['HONDA_CIVIC_2022'] / 9")
    per_v = m.groupby("vehicle_id").agg(np_=("platform", "nunique"), nm=("model", "nunique"))
    multi = per_v.index[per_v["np_"] > 1]
    chk("devices under two or more platforms inside the set / under two models", f"{len(multi)} / {int((per_v['nm'] > 1).sum())}", "2 / 2")
    auto_dev = set(auto["vehicle_id"])
    for v in multi:
        g = m[m["vehicle_id"] == v]
        print(f"    multi-platform device ({'one of the 9 automatic devices' if v in auto_dev else 'cvt only'}): " + ", ".join(f"{p} {gb} x{n}" for (p, gb), n in g.groupby(['platform', 'gearbox']).size().items()))
    chk("automatic devices that also carry cvt windows under other platforms (2.13 paragraph 1 before the review: 'two of the nine')", int(sum(v in auto_dev for v in multi)), 1)
    chk("devices under three platforms", int((per_v["np_"] == 3).sum()), 1)
    chk("CIVIC_2022 devices total / cvt-only", f"{m.loc[m['platform'] == 'HONDA_CIVIC_2022', 'vehicle_id'].nunique()} / {m.loc[(m['platform'] == 'HONDA_CIVIC_2022') & (m['gearbox'] == 'cvt'), 'vehicle_id'].nunique()}", "36 / 27")
    # per-platform rows of CCS_PIPELINE 'Honda re-decode' (in list / decoded / windows / devices / drives)
    quoted = {"HONDA_CIVIC": "400/397/385/33/186", "HONDA_CIVIC_BOSCH": "400/400/395/55/154", "HONDA_CIVIC_2022": "400/400/393/36/148", "HONDA_CRV_5G": "400/396/380/92/154",
              "HONDA_FIT": "282/281/273/7/116", "HONDA_CRV": "106/106/106/4/46", "HONDA_HRV": "100/99/97/2/62", "HONDA_HRV_3G": "60/60/55/3/17", "HONDA_FREED": "50/50/48/1/30"}
    for p, q in quoted.items():
        g = m[m["platform"] == p]
        chk(f"{p} in list / decoded / windows / devices / drives", f"{bud.get(p, 0)}/{(qc['platform'] == p).sum()}/{len(g)}/{g['vehicle_id'].nunique()}/{g['drive_id'].nunique()}", q)
    chk("can_valid_frac mean / min (decoded)", f"{qc['can_valid_frac'].mean():.3f} / {qc['can_valid_frac'].min():.3f}", "0.999 / 0.967")
    for s, q in (("seen", "1,932 / 224 / 804 / 3 / 6"), ("held-out", "152 / 5 / 79 / 1 / 2"), ("absent", "48 / 1 / 30 / 1 / 1")):
        g = m[m["stratum"] == s]
        chk(f"stratum {s}: windows / devices / drives / models / platforms", f"{fmt_int(len(g))} / {g['vehicle_id'].nunique()} / {g['drive_id'].nunique()} / {g['model'].nunique()} / {g['platform'].nunique()}", q)
    dm = m.groupby("model").agg(v=("vehicle_id", "nunique"), p=("platform", "nunique"), w=("w_idx", "size"))
    chk("Civic devices / platforms; CR-V; Fit; HR-V; Freed", f"{dm.loc['Honda Civic', 'v']} / {dm.loc['Honda Civic', 'p']}; {dm.loc['Honda CR-V', 'v']} / {dm.loc['Honda CR-V', 'p']}; {dm.loc['Honda Fit', 'v']}; {dm.loc['Honda HR-V', 'v']} / {dm.loc['Honda HR-V', 'p']}; {dm.loc['Honda Freed', 'v']}", "123 / 3; 96 / 2; 7; 5 / 2; 1")
    chk("footprint range / borderline / size labels", f"{m['footprint_m2'].min():.2f}-{m['footprint_m2'].max():.2f} / {int(m['borderline'].astype(bool).sum())} / {m['size_class'].value_counts().to_dict()}", "3.74-4.28 / 0 / {'SMALL': 2132}")
    chk("powertrain labels (windows)", m["pt_label"].value_counts().to_dict(), {"ICE": 2084, "unknown": 48})
    chk("label source lab / public platforms", f"{m.loc[m['label_src'] == 'lab', 'platform'].nunique()} / {sorted(m.loc[m['label_src'] == 'public', 'platform'].unique())}", "8 / ['HONDA_FREED']")
    lead = m["lead"].value_counts()
    chk("lead radar / none windows (share %)", f"{lead.get('radar', 0)} ({pct(lead.get('radar', 0) / len(m))}) / {lead.get('none', 0)} ({pct(lead.get('none', 0) / len(m))}); other: {sorted(set(lead.index) - {'radar', 'none'})}", "887 (41.6) / 1245 (58.4); other: []")
    rp = sorted(m.loc[m["lead"] == "radar", "platform"].unique()); npl = sorted(set(m["platform"]) - set(rp))
    chk("radar platforms / radarless platforms", f"{rp} / {npl}", "['HONDA_CIVIC', 'HONDA_CRV', 'HONDA_FIT', 'HONDA_FREED', 'HONDA_HRV'] / ['HONDA_CIVIC_2022', 'HONDA_CIVIC_BOSCH', 'HONDA_CRV_5G', 'HONDA_HRV_3G']")
    lvp = pd.Series({r["platform"]: r["lead_valid"] for r in ds["per_platform"] if r["lead_valid"] > 0})
    chk("lead_valid per radar platform range (decode_summary, all decoded segments of the platform)", f"{lvp.min():.2f}-{lvp.max():.2f}", "0.39-0.77")
    rw = m[m["platform"].isin(rp)]
    chk("windows on radar platforms (share %) / with an in-path object in the minute (lead=radar) / on the radarless ports (share %)", f"{len(rw)} ({pct(len(rw) / len(m))}) / {int((rw['lead'] == 'radar').sum())} / {len(m) - len(rw)} ({pct((len(m) - len(rw)) / len(m))})", "909 (42.6) / 887 / 1223 (57.4)")
    ta = ~m["has_torque"]
    chk("torque absent windows (share) / among them every HONDA_CRV window", f"{int(ta.sum())} ({pct(ta.mean())} %) / {bool(ta[m['platform'] == 'HONDA_CRV'].all())} ({int((m['platform'] == 'HONDA_CRV').sum())})", "406 (19.0 %) / True (106)")
    chk("HONDA_CRV steer_pressed never set (qc)", f"{qc.loc[qc['platform'] == 'HONDA_CRV', 'steer_pressed_frac'].max():.3f}", "0.000")
    nch = m["n_ch"].value_counts().sort_index(ascending=False).to_dict()
    chk("channels per window (20 = lead + torque, 19 = no torque, 14 = no lead, 13 = neither)", nch, {20: 676, 19: 233, 14: 1050, 13: 173})
    chk("20 + 19 == windows with the lead group; 19 + 13 == torque absent", f"{nch.get(20, 0) + nch.get(19, 0)} == {int(m['has_lead'].sum())}; {nch.get(19, 0) + nch.get(13, 0)} == {int((~m['has_torque']).sum())}", f"{int(m['has_lead'].sum())} == {int(m['has_lead'].sum())}; {int((~m['has_torque']).sum())} == {int((~m['has_torque']).sum())}")
    chk("yaw derived everywhere", bool((note_field(m, "yaw") == "derived").all()), True)
    das_p = m.groupby("platform")["das_frac"].mean() * 100
    chk("das_frac mean / per-platform range % / engaged / human windows", f"{m['das_frac'].mean():.2f} / {das_p.min():.0f}-{das_p.max():.0f} / {int((m['state'] == 'engaged').sum())} / {int((m['state'] == 'human').sum())}", "0.42 / 20-56 / 902 / 1230")
    chk("1.1 row: engaged share %", pct((m["state"] == "engaged").mean()), "42.3")
    w = qc[qc["window"]]
    chk("mean speed / gas / brake / steer pressed % (qc, window set; 2.13 before the review quoted the all-decoded-segment means 17.7 / 36 / 16 / 18)", f"{w['v_mean'].mean():.1f} / {100 * w['gas_frac'].mean():.0f} / {100 * w['brake_frac'].mean():.0f} / {100 * w['steer_pressed_frac'].mean():.0f}", "18.2 / 37 / 15 / 18")
    chk("the same over all 2,189 decoded segments (with the 57 near-stationary ones)", f"{qc['v_mean'].mean():.1f} / {100 * qc['gas_frac'].mean():.0f} / {100 * qc['brake_frac'].mean():.0f} / {100 * qc['steer_pressed_frac'].mean():.0f}", "17.7 / 36 / 16 / 18")
    ev = m.groupby("vehicle_id").size()
    chk("windows per device median / 1 / 2-4 / 5-9 / 10+", f"{ev.median():.0f} / {(ev == 1).sum()} / {((ev >= 2) & (ev <= 4)).sum()} / {((ev >= 5) & (ev <= 9)).sum()} / {(ev >= 10).sum()}", "6 / 17 / 79 / 61 / 73")
    dw = m.groupby("drive_id").size()
    chk("drives with one window %", pct((dw == 1).mean()), "46.1")
    mc = pd.read_parquet(EXT / "ccs" / "work" / "meta.parquet", columns=["vehicle_id", "platform"])
    shared = set(m["vehicle_id"]) & set(mc["vehicle_id"])
    chk("devices shared with ccs / all of them under other platforms there", f"{len(shared)} / {bool(not (set(mc.loc[mc['vehicle_id'].isin(shared), 'platform']) & set(m['platform'])))}", "9 / True")
    chk("wheelSpeedFactor caveat: CR-V 5G + CR-V 2016 + HR-V (2016) windows", int(m["platform"].isin(["HONDA_CRV_5G", "HONDA_CRV", "HONDA_HRV"]).sum()), 583)
    chk("device x model pairs size / powertrain", f"{m['pair'].nunique()} / {m.loc[T.task_labels(m, 'bev_ice') >= 0, 'pair'].nunique()}", "232 / 231")
    chk("one window per segment (w_idx unique per drive)", bool((m.groupby("drive_id")["w_idx"].nunique() == dw).all()), True)


def _plat_row_honda(m, P_s, y_s, P_b, y_b, Pv1, Pv3, Pb3, Pb2, p):
    g = m[m["platform"] == p]; idx = g.index.to_numpy(); veh = m["vehicle_id"].to_numpy()
    radar = 100 * (g["lead"] == "radar").mean()
    lead = f"radar {radar:.0f} %" if radar > 0 else "none"
    pm = P_s[idx].mean(0); ds_ = T.SIZE_CLASSES[int(pm.argmax())]
    uid, Pu, yu, du, _ = unit_dec(P_s[idx], y_s[idx], veh[idx]); d = unit_eval(P_s[idx], y_s[idx], g["drive_id"].to_numpy(), 3)
    size_cell = f"{', '.join(f'{100 * x:.0f}' for x in pm)} -> {ds_} {'OK' if ds_ == 'SMALL' else 'miss'}, {int((du == yu).sum())}/{len(uid)}, {pct(d['acc'])}"
    v1 = T.SIZE_CLASSES[int(Pv1[idx].mean(0).argmax())]; v3 = T.SIZE_CLASSES[int(Pv3[idx].mean(0).argmax())]
    v13 = f"{v1}{'' if v1 == 'SMALL' else ' miss'} / {v3}{'' if v3 == 'SMALL' else ' miss'}"
    pb = P_b[idx].mean(0); db = T.BEV_ICE[int(pb.argmax())]
    if (y_b[idx] >= 0).any():
        uidb, _, yub, dub, _ = unit_dec(P_b[idx], y_b[idx], veh[idx])
        pt_cell = f"{100 * pb[0]:.0f} -> {db} {'OK' if db == 'ICE' else 'miss'}, {int((dub == yub).sum())}/{len(uidb)}"
    else:
        pt_cell = f"{100 * pb[0]:.0f} -> {db} (unlabelled)"
    b3 = T.BEV_ICE[int(Pb3[idx].mean(0).argmax())]; b2 = T.BEV_ICE[int(Pb2[idx].mean(0).argmax())]
    return (f"| {g['stratum'].iloc[0]} | {lead} | {g['vehicle_id'].nunique()} | {g['drive_id'].nunique()} | {len(g)} | {100 * g['das_frac'].mean():.1f} | {size_cell} | {v13} | {pt_cell} | {b3} / {b2} |")


def sec35_honda_tables():
    print("\n## 35. Tables (a)-(e) of 2.13 re-rendered cell by cell, plus the figures of its Reading bullets and caveats")
    m = set_meta("ccs_honda"); veh = m["vehicle_id"].to_numpy()
    Ps, ys, _ = set_load("ccs_honda", "size", "full"); Pb, yb, _ = set_load("ccs_honda", "bev_ice", "full")
    Pv1, _, _ = set_load("ccs_honda", "size", "V1"); Pv3, _, _ = set_load("ccs_honda", "size", "V3")
    Pb3, _, _ = set_load("ccs_honda", "bev_ice", "V3"); Pb2, _, _ = set_load("ccs_honda", "bev_ice", "V2")
    print("  (a) decisions per platform (the first column's label text is the writer's; the cells from `stratum` on are searched):")
    for p in HONDA9:
        found_in(f"(a) {p}", _plat_row_honda(m, Ps, ys, Pb, yb, Pv1, Pv3, Pb3, Pb2, p), "2.13")
    print("  (b) every run at device and model level per stratum:")
    import json as _json
    ref = _json.load(open(HERE_ / "internal_reference.json"))["tasks"]
    ref_names = {"size": {"full": "full (GRU S9, 20 ch)", "V1": "V1 lateral + speed (10 ch)", "V2": "V2 longitudinal-only (11 ch)", "V3": "V3 speed-only (3 ch)",
                          "mask-V1": "full (GRU S9, 20 ch) + mask-V1", "mask-V2": "full (GRU S9, 20 ch) + mask-V2", "mask-V3": "full (GRU S9, 20 ch) + mask-V3"},
                 "bev_ice": {"full": "full (ResNet B5, 20 ch)", "V1": "V1 lateral + speed (10 ch)", "V2": "V2 longitudinal-only (11 ch)", "V3": "V3 speed-only (3 ch)",
                             "mask-V1": "full (ResNet B5, 20 ch) + mask-V1", "mask-V2": "full (ResNet B5, 20 ch) + mask-V2", "mask-V3": "full (ResNet B5, 20 ch) + mask-V3"}}
    store = {}
    for task, k in (("size", 3), ("bev_ice", 2)):
        for key, mask, label in CCS_RUNS_SIZE:
            P, y, cl = set_load("ccs_honda", task, key, mask)
            cells = []
            for s, sel in _strata_sel(m):
                idx = np.flatnonzero(sel & (y >= 0))
                if len(idx) == 0:
                    continue
                mm = m.iloc[idx].reset_index(drop=True)
                v = unit_eval(P[idx], y[idx], mm["vehicle_id"].to_numpy(), k); mo = unit_eval(P[idx], y[idx], mm["model"].to_numpy(), k)
                store[(task, mask if key == "mask" else key, s)] = (v, mo, P[idx], y[idx], mm)
                right, n = int(np.trace(v["cm"])), v["n"]; models = int(np.trace(mo["cm"]))
                ci = boot_ci(P[idx], y[idx], mm, k, "vehicle")[0] if (s == "all" or (s == "seen" and key == "full")) else None
                cis = "" if (ci is None or (abs(ci[0] - 1) < 1e-9 and abs(ci[1] - 1) < 1e-9 and right == n)) else ci_str(ci)
                if s in ("all", "seen"):
                    cells.append(f"{pct(v['acc'])}{cis} ({right}/{n}); {models}")
                elif s == "held-out":
                    cells.append(f"{pct(v['acc'])} ({right}/{n}); {models}")
                else:
                    cells.append(f"{right}/{n}; {models}")
            w = unit_eval(P, y, np.arange(len(y)), k)
            rk = mask if key == "mask" else key
            internal = "-" if rk in ("mask-yaw", "mask-lead") else (lambda a: f"{pct(a['accuracy'])} / {pct(a['macro_f1'])}")(ref[task][ref_names[task][rk]]["subsets"]["all"]["vehicle"])
            lab_ = label + (" (secondary)" if rk in ("mask-V1", "mask-V2", "mask-V3") else "")
            row = f"| {lab_} | " + " | ".join(cells) + f" | {pct(w['acc'])} | {internal} |"
            found_in(f"(b) {task} {lab_}", row, "2.13")
    print("  (c) the held-out HR-V against internal_per_model.md:")
    ho = (m["stratum"] == "held-out").to_numpy()
    cells = []
    for task, k, P, y in (("size", 3, Ps, ys), ("bev_ice", 2, Pb, yb)):
        idx = np.flatnonzero(ho & (y >= 0)); cl = T.TASK_LABEL[task][1]
        uid, Pu, yu, du, _ = unit_dec(P[idx], y[idx], veh[idx]); pm = P[idx].mean(0); yt = int(np.bincount(y[idx], minlength=k).argmax()); d = cl[int(pm.argmax())]
        dr = unit_eval(P[idx], y[idx], m["drive_id"].to_numpy()[idx], k)
        ir = _internal_per_model_rows(task).get("Honda HR-V")
        cells.append(f"{int((du == yu).sum())}/{len(uid)} ({pct((du == yu).mean())}), {d} ({pm[yt]:.2f}) {'OK' if d == cl[yt] else 'miss'}; drive acc {pct(dr['acc'])} % | {ir[1]}.0 / {ir[2]} ({ir[3]:.2f})")
    found_in("(c) HR-V", f"| Honda HR-V (HONDA_HRV, HONDA_HRV_3G) | SMALL / ICE | {cells[0]} | {cells[1]} |", "2.13")
    print("  (d) lead source and automation state (all stratum):")
    for factor, values, arr in (("lead", ("radar", "none"), m["lead"].to_numpy(str)), ("state", ("human", "engaged"), m["state"].to_numpy(str))):
        for val in values:
            cells = []
            for P, y in ((Ps, ys), (Pb, yb)):
                idx = np.flatnonzero((arr == val) & (y >= 0))
                uid, Pu, yu, du, _ = unit_dec(P[idx], y[idx], veh[idx])
                cells.append(f"{fmt_int(len(idx))} / {len(uid)} | {pct((P[idx].argmax(1) == y[idx]).mean())} | {pct((du == yu).mean())} ({int((du == yu).sum())}/{len(uid)})")
            found_in(f"(d) {factor} {val}", f"| {factor} | {val} | {cells[0]} | {cells[1]} |", "2.13")
    hum = (m["state"] == "human").to_numpy(); eng = ~hum; ice = yb == 1
    chk("(d) mean P(BEV) on ICE windows human / engaged; mean P(SMALL) human / engaged", f"{Pb[ice & hum, 0].mean():.2f} / {Pb[ice & eng, 0].mean():.2f}; {Ps[hum, 0].mean():.2f} / {Ps[eng, 0].mean():.2f}", "0.08 / 0.29; 0.73 / 0.65")
    for task, P, y in (("size", Ps, ys), ("bev_ice", Pb, yb)):
        out = []
        for selst in (hum, eng):
            idx = np.flatnonzero(ho & selst & (y >= 0)); uid, Pu, yu, du, _ = unit_dec(P[idx], y[idx], veh[idx]); out.append(f"{int((du == yu).sum())}/{len(uid)}")
        chk(f"(d) held-out HR-V devices right from human / engaged minutes ({task})", " / ".join(out), "5/5 / 4/5")
    print("  (e) evidence per device:")
    buckets = (("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10+", 10, 10 ** 9))
    for task, P, y in (("size", Ps, ys), ("powertrain", Pb, yb)):
        idx = np.flatnonzero(y >= 0); uid, Pu, yu, du, cnt = unit_dec(P[idx], y[idx], veh[idx])
        cells = [f"{int((du[b] == yu[b]).sum())}/{int(b.sum())} ({pct((du[b] == yu[b]).mean())})" for name, lo, hi in buckets for b in [(cnt >= lo) & (cnt <= hi)]]
        found_in(f"(e) {task}", f"| {task} | {len(uid)} | " + " | ".join(cells) + " |", "2.13")
    # ---- Reading bullets
    print("  Reading bullets:")
    uid, Pu, yu, du, cnt = unit_dec(Ps, ys, veh); dev_s = pd.DataFrame({"v": uid, "ok": du == yu, "n": cnt, "ps": Pu[:, 0]})
    uidb, Pub, yub, dub, cntb = unit_dec(Pb, yb, veh); dev_b = pd.DataFrame({"v": uidb, "ok": dub == yub, "n": cntb, "pbev": Pub[:, 0]})
    mo_s = unit_eval(Ps, ys, m["model"].to_numpy(), 3); mo_b = unit_eval(Pb, yb, m["model"].to_numpy(), 2)
    chk("bullet 1: size devices / models; powertrain devices / models; Freed P(BEV)", f"{int(dev_s['ok'].sum())}/{len(dev_s)} / {int(np.trace(mo_s['cm']))}/{mo_s['n']}; {int(dev_b['ok'].sum())}/{len(dev_b)} / {int(np.trace(mo_b['cm']))}/{mo_b['n']}; {Pb[(m['platform'] == 'HONDA_FREED').to_numpy(), 0].mean():.2f}", "226/230 / 5/5; 227/229 / 4/4; 0.20")
    civ = boot_ci(Ps, ys, m, 3, "vehicle")[0]; cib = boot_ci(Pb, yb, m, 2, "vehicle")[0]
    chk("bullet 1: device CIs size / powertrain (300 resamples)", f"{ci_str(civ).strip()} / {ci_str(cib).strip()}", "[96.5, 99.6] / [97.8, 100.0]")
    crv = m["model"] == "Honda CR-V"
    uidc, _, yuc, duc, _ = unit_dec(Pb[crv.to_numpy()], yb[crv.to_numpy()], veh[crv.to_numpy()])
    chk("bullet 1: CR-V gasoline ports devices right (powertrain)", f"{int((duc == yuc).sum())} of {len(uidc)}", "95 of 96")
    mc = ccs_meta(); Pcs, ycs, _ = ccs_load("size", "full"); Pcb, ycb, _ = ccs_load("bev_ice", "full")
    rs = unit_eval(Pcs, ycs, mc["vehicle_id"].to_numpy(), 3); rb = unit_eval(Pcb, ycb, mc["vehicle_id"].to_numpy(), 2)
    chk("bullet 1: ccs SMALL device recall / ICE device recall", f"{pct(rs['rec'][0])} / {pct(rb['rec'][1])}", "91.1 / 88.2")
    chk("bullet 1: human-driven share of the minutes %", f"{100 * hum.mean():.0f}", "58")
    chk("bullet 1: window acc size / powertrain", f"{pct((Ps.argmax(1) == ys).mean())} / {pct((Pb[yb >= 0].argmax(1) == yb[yb >= 0]).mean())}", "83.8 / 89.3")
    # bullet 2: the possible hybrids
    c22 = (m["platform"] == "HONDA_CIVIC_2022").to_numpy()
    uid22, Pu22, yu22, du22, cnt22 = unit_dec(Pb[c22], yb[c22], veh[c22])
    auto_dev = set(m.loc[m["gearbox"] == "automatic", "vehicle_id"])
    miss22 = [(u, float(Pu22[i, 0]), int(cnt22[i])) for i, u in enumerate(uid22) if du22[i] != yu22[i]]
    chk("bullet 2: CIVIC_2022 devices right at platform level", f"{int((du22 == yu22).sum())}/{len(uid22)}", "34/36")
    chk("bullet 2: both platform-level misses are automatic devices (P(BEV) over windows)", f"{all(u in auto_dev for u, _, _ in miss22)} " + ", ".join(f"{p:.2f} over {n}" for _, p, n in sorted(miss22, key=lambda x: -x[2])), "True 0.62 over 12, 0.56 over 6")
    is_auto22 = np.array([u in auto_dev for u in uid22])
    w_auto = c22 & m["vehicle_id"].isin(auto_dev).to_numpy(); w_cvt22 = c22 & ~w_auto
    w9 = m["vehicle_id"].isin(auto_dev).to_numpy()          # every window of the nine devices (109: 103 automatic + the 6 cvt windows of the device under three platforms)
    dm9 = np.array([Pb[(m["vehicle_id"] == u).to_numpy(), 0].mean() for u in sorted(auto_dev)])
    chk("bullet 2: mean P(BEV) of the 9 automatic devices (mean of the device means over all their windows; over their windows) vs the 27 cvt devices of the platform (device means; windows)", f"{dm9.mean():.2f} ({len(np.flatnonzero(w9))} windows: {Pb[w9, 0].mean():.2f}) vs {Pu22[~is_auto22, 0].mean():.2f} ({Pb[w_cvt22, 0].mean():.2f})", "0.38 (109 windows: 0.36) vs 0.14 (0.16)")
    rest = ~w9 & (yb >= 0)
    chk("bullet 2: window acc of the nine devices' windows vs the rest of the set", f"{pct((Pb[w9].argmax(1) == 1).mean())} vs {pct((Pb[rest].argmax(1) == yb[rest]).mean())}", "72.5 vs 90.2")
    dev_auto = dev_b[dev_b["v"].isin(auto_dev)]
    chk("bullet 2: automatic devices right at platform level / at device level (all their windows pooled)", f"{int((du22[is_auto22] == yu22[is_auto22]).sum())} of 9 / {int(dev_auto['ok'].sum())} of {len(dev_auto)}", "7 of 9 / 8 of 9")
    multi_auto = [u for u in auto_dev if m.loc[m["vehicle_id"] == u, "platform"].nunique() > 1]
    for u in multi_auto:
        g = m[m["vehicle_id"] == u]
        chk("bullet 2: the automatic device that also carries cvt windows: other windows / P(BEV) pooled (ICE by a hair)", f"{int((g['gearbox'] == 'cvt').sum())} under {g.loc[g['gearbox'] == 'cvt', 'platform'].nunique()} other platforms / {Pb[g.index.to_numpy(), 0].mean():.3f}", "6 under 2 other platforms / 0.485")
    keep = ~m["vehicle_id"].isin(auto_dev).to_numpy() & (yb >= 0)
    uidk, Puk, yuk, duk, _ = unit_dec(Pb[keep], yb[keep], veh[keep])
    k22 = keep & c22; uid22k, _, yu22k, du22k, _ = unit_dec(Pb[k22], yb[k22], veh[k22])
    chk("bullet 2: without the nine: devices right / CIVIC_2022 / platform P(ICE) without vs with", f"{int((duk == yuk).sum())}/{len(uidk)} ({pct((duk == yuk).mean())}) / {int((du22k == yu22k).sum())}/{len(uid22k)} / {1 - Pb[k22, 0].mean():.2f} vs {1 - Pb[c22, 0].mean():.2f}", "219/220 (99.5) / 27/27 / 0.84 vs 0.79")
    other_miss = dev_b[~dev_b["ok"] & ~dev_b["v"].isin(auto_dev)]
    for u in other_miss["v"]:
        g = m[m["vehicle_id"] == u]
        chk("bullet 2: the remaining device-level powertrain miss", f"{g['platform'].iloc[0]} with {len(g)} window(s), das_frac {g['das_frac'].iloc[0]:.2f}, P(BEV) {Pb[g.index.to_numpy(), 0].mean():.2f}", "HONDA_CRV_5G with 1 window(s), das_frac 1.00, P(BEV) 0.67")
    sm = dev_s[~dev_s["ok"]]
    chk("size device misses: n / models / any possible hybrid", f"{len(sm)} / {sorted(m.set_index('vehicle_id').loc[sm['v'], 'model'].groupby(level=0).first().value_counts().to_dict().items())} / {bool(sm['v'].isin(auto_dev).any())}", "4 / [('Honda CR-V', 1), ('Honda Civic', 2), ('Honda Fit', 1)] / False")
    # bullet 3: yaw / lead masks
    yw = store[("size", "mask-yaw", "all")]; ld = store[("size", "mask-lead", "all")]
    Py, _, _ = set_load("ccs_honda", "size", "mask", "mask-yaw")
    chk("bullet 3: yaw mask size devices / models; windows LARGE %; P(LARGE); powertrain devices", f"{int(np.trace(yw[0]['cm']))}/{yw[0]['n']} ({pct(yw[0]['acc'])}) / {int(np.trace(yw[1]['cm']))}/{yw[1]['n']}; {pct((Py.argmax(1) == 2).mean())}; {Py[:, 2].mean():.2f}; {pct(store[('bev_ice', 'mask-yaw', 'all')][0]['acc'])}", "33/230 (14.3) / 0/5; 70.7; 0.58; 98.7")
    chk("bullet 3: lead mask devices size / powertrain", f"{pct(ld[0]['acc'])} / {pct(store[('bev_ice', 'mask-lead', 'all')][0]['acc'])}", "98.7 / 99.1")
    # bullet 4: engaged
    out = []
    for task, P, y in (("size", Ps, ys), ("pt", Pb, yb)):
        for selst in (eng, hum):
            idx = np.flatnonzero(selst & (y >= 0)); out.append(pct((P[idx].argmax(1) == y[idx]).mean()))
    chk("bullet 4: window acc engaged / human, size then powertrain", " / ".join(out[:2]) + " and " + " / ".join(out[2:]), "73.7 / 91.1 and 80.7 / 95.6")
    out = []
    for task, P, y in (("size", Ps, ys), ("pt", Pb, yb)):
        for selst in (eng, hum):
            idx = np.flatnonzero(selst & (y >= 0)); uid_, _, yu_, du_, _ = unit_dec(P[idx], y[idx], veh[idx]); out.append(pct((du_ == yu_).mean()))
    chk("bullet 4: device acc from engaged / human minutes, size then powertrain", " / ".join(out[:2]) + " and " + " / ".join(out[2:]), "84.8 / 98.5 and 94.1 / 99.5")
    # bullet 5: variants
    chk("bullet 5: V1 / V2 / V3 device acc size, powertrain", " / ".join(f"{pct(store[('size', k, 'all')][0]['acc'])} / {pct(store[('bev_ice', k, 'all')][0]['acc'])}" for k in ("V1", "V2", "V3")), "47.8 / 86.0 / 73.0 / 97.8 / 86.5 / 98.3")
    v1_mid = [p for p in HONDA9 if T.SIZE_CLASSES[int(Pv1[(m['platform'] == p).to_numpy()].mean(0).argmax())] == "MID"]
    chk("bullet 5: platforms V1 calls MID", v1_mid, ["HONDA_FIT", "HONDA_CIVIC", "HONDA_CRV_5G", "HONDA_HRV", "HONDA_HRV_3G"])
    ps_on = [Ps[(m['platform'] == p).to_numpy(), 0].mean() for p in v1_mid]
    chk("bullet 5: full-model P(SMALL) range on those platforms; V1 held-out devices; V1 window acc", f"{min(ps_on):.2f}-{max(ps_on):.2f}; {int(np.trace(store[('size', 'V1', 'held-out')][0]['cm']))} of 5; {pct((Pv1.argmax(1) == 0).mean())}", "0.60-0.75; 1 of 5; 49.6")
    v3_all = all(T.SIZE_CLASSES[int(Pv3[(m['platform'] == p).to_numpy()].mean(0).argmax())] == "SMALL" for p in HONDA9)
    chk("bullet 5: V3 all nine SMALL; devices size / powertrain", f"{v3_all}; {int(np.trace(store[('size', 'V3', 'all')][0]['cm']))}/230 / {int(np.trace(store[('bev_ice', 'V3', 'all')][0]['cm']))}/229", "True; 199/230 / 225/229")
    # caveats: masked full models 97-100 %
    mk = [pct(store[('size', k, 'all')][0]['acc']) for k in ("mask-V1", "mask-V2", "mask-V3")] + [pct(store[('bev_ice', k, 'all')][0]['acc']) for k in ("mask-V1", "mask-V2", "mask-V3")]
    chk("caveat: masked full models device acc size V1/V2/V3, powertrain V1/V2/V3", " / ".join(mk), "98.7 / 96.5 / 97.0 / 98.3 / 99.6 / 100.0")
    chk("CCS_PIPELINE: full drive acc size / powertrain", f"{pct(unit_eval(Ps, ys, m['drive_id'].to_numpy(), 3)['acc'])} / {pct(unit_eval(Pb, yb, m['drive_id'].to_numpy(), 2)['acc'])}", "88.2 / 94.1")


def sec36_comma2k19():
    import json as _json
    print("\n## 36. comma2k19 (RESULTS.md 2.14, 1.1 row, CCS_PIPELINE.md 'comma2k19'): decode accounting, set description, the results table, the per-car automation split")
    ds = _json.load(open(EXT / "comma2k19" / "work" / "decode_summary.json")); qc = set_qc("comma2k19"); m = set_meta("comma2k19")
    chk("found / routes / decoded (== qc rows) / skipped / windows (== meta rows)", f"{ds['segments_found']} / {ds['routes']} / {ds['segments_decoded']} ({len(qc) == ds['segments_decoded']}) / {ds['skipped']} / {ds['windows']} ({ds['windows'] == len(m) == int(qc['window'].sum())})", "390 / 42 / 387 (True) / 3 / 381 (True)")
    st = sorted(int(k.split()[0]) for k in ds["skip_reasons"] if "stale" in k)
    chk("skip reasons: stale samples", f"{st} (other reasons: {[k for k in ds['skip_reasons'] if 'stale' not in k]})", "[77, 110, 381] (other reasons: [])")
    print("    (the CAN spans of the three skipped segments, 52.0 / 48.7 / 21.5 s, and their cars, two Civic / one RAV4, are in the decode log only)")
    chk("moving-rule drops per car (Civic, RAV4)", f"{int((~qc['window'] & (qc['platform'] == 'HONDA_CIVIC')).sum())}, {int((~qc['window'] & (qc['platform'] == 'TOYOTA_RAV4')).sum())}", "4, 2")
    chk("options: layout / das_from / fingerprint map / ignore absent / cp sources / gearbox", f"{ds['layout']} / {ds['das_from']} / {ds['fingerprint_map']} / {ds['ignore_absent_messages']} / {ds['cp_sources']} / {ds['gearbox_counts']}", "comma2k19 / controlsState / {'Chunk_1': 'TOYOTA_RAV4', 'Chunk_3': 'HONDA_CIVIC'} / True / {'constructed': 387} / {'cvt': 200, 'nan': 187}")
    chk("absent (waived) messages per car", ds["absent_msgs"], {"cam:ACC_HUD,LKAS_HUD,BRAKE_COMMAND": 200, "cam:LKAS_HUD;pt:PRE_COLLISION": 187})
    chk("seconds / workers", f"{ds['seconds']:.0f} / {ds['workers']}", "153 / 6")
    # route folders: sizes and gaps from the segment folders (names are not printed)
    root = EXT / "comma2k19" / "segments"
    sizes = {}; gaps = {}
    for chunk in sorted(p for p in root.iterdir() if p.is_dir()):
        for rd in sorted(p for p in chunk.iterdir() if p.is_dir()):
            segs = sorted(int(p.name) for p in rd.iterdir() if p.is_dir() and p.name.isdigit() and (p / "raw_log.bz2").exists())
            sizes.setdefault(chunk.name, []).append(len(segs)); gaps.setdefault(chunk.name, 0)
            gaps[chunk.name] += int(segs[-1] - segs[0] + 1 != len(segs))
    chk("segments per chunk (Chunk_1 RAV4, Chunk_3 Civic) / folders per chunk", f"{sum(sizes['Chunk_1'])} / {sum(sizes['Chunk_3'])} / {len(sizes['Chunk_1'])}, {len(sizes['Chunk_3'])}", "188 / 202 / 21, 21")
    chk("route sizes all / Chunk 1 / Chunk 3; folders with gaps all (Chunk 1)", f"{min(sizes['Chunk_1'] + sizes['Chunk_3'])}-{max(sizes['Chunk_1'] + sizes['Chunk_3'])} / {min(sizes['Chunk_1'])}-{max(sizes['Chunk_1'])} / {min(sizes['Chunk_3'])}-{max(sizes['Chunk_3'])}; {gaps['Chunk_1'] + gaps['Chunk_3']} ({gaps['Chunk_1']})", "1-21 / 2-15 / 1-21; 8 (7)")
    g1, g3 = m[m["platform"] == "TOYOTA_RAV4"], m[m["platform"] == "HONDA_CIVIC"]
    chk("windows RAV4 / Civic; hours (two decimals; the report rounds 6.35 up); devices; drives; models", f"{len(g1)} / {len(g3)}; {len(m) / 60:.2f}; {m['vehicle_id'].nunique()}; {m['drive_id'].nunique()}; {m['model'].nunique()}", "185 / 196; 6.35; 2; 42; 2")
    w = qc[qc["window"]]
    for p, name, q, q_all in (("TOYOTA_RAV4", "RAV4", "0.987 (0.983) / 0.78 / 0.56 / 22.8 / 0.26 / 0.12 / 0.15", "22.6 / 0.26 / 0.13 / 0.15"), ("HONDA_CIVIC", "Civic", "0.997 (0.995) / 0.87 / 0.62 / 20.9 / 0.17 / 0.12 / 0.16", "20.5 / 0.17 / 0.14 / 0.16")):
        s = qc[qc["platform"] == p]; ww = w[w["platform"] == p]; mm = m[m["platform"] == p]
        chk(f"{name}: can_valid mean (min, decoded) / lead_valid (decoded) / das_frac (windows) / speed / gas / brake / steer_pressed (window set)", f"{s['can_valid_frac'].mean():.3f} ({s['can_valid_frac'].min():.3f}) / {s['lead_valid_frac'].mean():.2f} / {mm['das_frac'].mean():.2f} / {ww['v_mean'].mean():.1f} / {ww['gas_frac'].mean():.2f} / {ww['brake_frac'].mean():.2f} / {ww['steer_pressed_frac'].mean():.2f}", q)
        print(f"    (info) {name}: speed / gas / brake / steer_pressed over all decoded segments {s['v_mean'].mean():.2f} / {s['gas_frac'].mean():.3f} / {s['brake_frac'].mean():.3f} / {s['steer_pressed_frac'].mean():.3f} (2.14 and CCS_PIPELINE.md quoted {q_all} before the review)")
        chk(f"{name}: radar segments / lead source", f"{int((s['lead_source'] == 'radar').sum())} of {len(s)} / {sorted(s['lead_source'].unique())}", "185 of 187 / ['none', 'radar']" if name == "RAV4" else "199 of 200 / ['none', 'radar']")
        chk(f"{name}: cruise_frac / controls_enabled_frac means (max |diff| per segment) / gearbox", f"{s['cruise_frac'].mean():.2f} / {s['controls_enabled_frac'].mean():.2f} ({(s['cruise_frac'] - s['controls_enabled_frac']).abs().max():.3f}) / {sorted(s['gearbox'].astype(str).unique())}", "0.55 / 0.55 (0.013) / ['nan']" if name == "RAV4" else "0.61 / 0.61 (0.004) / ['cvt']")
        chk(f"{name}: torque scale range (never absent)", f"{s['torque_scale'].min():.0f}-{s['torque_scale'].max():.0f} ({s['torque_scale'].isna().sum()} absent)", "192-366 (0 absent)" if name == "RAV4" else "2061-3070 (0 absent)")
    chk("engaged / human windows RAV4, Civic", f"{int((g1['state'] == 'engaged').sum())} / {int((g1['state'] == 'human').sum())}, {int((g3['state'] == 'engaged').sum())} / {int((g3['state'] == 'human').sum())}", "105 / 80, 121 / 75")
    chk("1.1 row: engaged share %", pct((m["state"] == "engaged").mean()), "59.3")
    chk("all 20 channels on every window / torque present everywhere / lead note (none = radar segment without an object in the minute)", f"{bool((m['n_ch'] == 20).all())} / {bool(m['has_torque'].all())} / {m['lead'].value_counts().to_dict()}", "True / True / {'radar': 378, 'none': 3}")
    chk("note fields: cp / das source / gearbox (Civic)", f"{sorted(note_field(m, 'cp').unique())} / {sorted(note_field(m, 'das').unique())} / {sorted(m.loc[m['platform'] == 'HONDA_CIVIC', 'gearbox'].unique())}", "['constructed'] / ['controlsState'] / ['cvt']")
    chk("steerRatio / wheelbase RAV4, Civic (qc)", f"{qc.loc[qc['platform'] == 'TOYOTA_RAV4', 'steer_ratio'].iloc[0]} / {qc.loc[qc['platform'] == 'TOYOTA_RAV4', 'wheelbase_m'].iloc[0]}, {qc.loc[qc['platform'] == 'HONDA_CIVIC', 'steer_ratio'].iloc[0]} / {qc.loc[qc['platform'] == 'HONDA_CIVIC', 'wheelbase_m'].iloc[0]}", "16.88 / 2.65, 15.38 / 2.7")
    chk("labels: footprints / size / powertrain / strata", f"{sorted((p, round(float(v), 3)) for p, v in m.groupby('platform')['footprint_m2'].first().items())} / {m['size_class'].unique().tolist()} / {m['pt_label'].unique().tolist()} / {m['stratum'].unique().tolist()}", "[('HONDA_CIVIC', 4.197), ('TOYOTA_RAV4', 4.134)] / ['SMALL'] / ['ICE'] / ['seen']")
    # ---- results table of 2.14
    print("  results table (window / drive [CI from the score file, 1,000 resamples] / devices / per car):")
    veh = m["vehicle_id"].to_numpy(); drv = m["drive_id"].to_numpy()
    car = {"TOYOTA_RAV4": (m["platform"] == "TOYOTA_RAV4").to_numpy(), "HONDA_CIVIC": (m["platform"] == "HONDA_CIVIC").to_numpy()}
    hum = (m["state"] == "human").to_numpy(); eng = ~hum

    def drive_ci(task, key, mask):
        f = EXT / "comma2k19" / "work" / f"score__{TAG[(task, key)]}__{mask}.json"
        lv = _json.load(open(f))["levels"]["drive"]
        return f"{pct(lv['accuracy'])}{ci_str(tuple(lv['ci95_accuracy']))}"

    def level_cells(task, key, mask):
        P, y, cl = set_load("comma2k19", task, key, mask); k = len(cl)
        w = pct((P.argmax(1) == y).mean()); d = unit_eval(P, y, drv, k); v = unit_eval(P, y, veh, k)
        per = {}
        for p, sel in car.items():
            pm = P[sel].mean(0); dd = unit_eval(P[sel], y[sel], drv[sel], k)
            per[p] = (cl[int(pm.argmax())], pm, pct(dd["acc"]))
        return P, y, cl, w, d, v, per

    P, y, cl, w, d, v, per = level_cells("bev_ice", "full", "full")
    r, c = per["TOYOTA_RAV4"], per["HONDA_CIVIC"]
    found_in("pt full", f"| powertrain | full (20 ch) | {w} | {drive_ci('bev_ice', 'full', 'full')} | {int(np.trace(v['cm']))}/{v['n']} ICE | {r[0]} ({r[1][1]:.2f}), {r[2]} | {c[0]} ({c[1][1]:.2f}), {c[2]} |", "2.14")
    chk("pt full: window / drive / device", f"{w} / {pct(d['acc'])} / {int(np.trace(v['cm']))}/{v['n']}", "70.9 / 88.1 / 2/2")
    chk("pt full: Civic P(ICE) and drive acc; RAV4", f"{c[1][1]:.2f} {c[2]}; {r[1][1]:.2f} {r[2]}", "0.71 90.5; 0.68 85.7")
    P3, y3, _, w3, d3, v3, per3 = level_cells("bev_ice", "V3", "full"); r3, c3 = per3["TOYOTA_RAV4"], per3["HONDA_CIVIC"]
    found_in("pt V3", f"| powertrain | V3 speed-only | {w3} | {drive_ci('bev_ice', 'V3', 'full')} | {int(np.trace(v3['cm']))}/{v3['n']} | {r3[0]} ({r3[1][1]:.2f}), {r3[2]} | {c3[0]} ({c3[1][1]:.2f}), {c3[2]} |", "2.14")
    v12 = [level_cells("bev_ice", k, "full") for k in ("V1", "V2")]
    found_in("pt V1 / V2", f"| powertrain | V1 / V2 | {v12[0][3]} / {v12[1][3]} | {pct(v12[0][4]['acc'])} / {pct(v12[1][4]['acc'])} | {int(np.trace(v12[0][5]['cm']))}/2 / {int(np.trace(v12[1][5]['cm']))}/2 | {v12[0][6]['TOYOTA_RAV4'][0]} / {v12[1][6]['TOYOTA_RAV4'][0]} | {v12[0][6]['HONDA_CIVIC'][0]} / {v12[1][6]['HONDA_CIVIC'][0]} |", "2.14")
    mk = [level_cells("bev_ice", "mask", f"mask-{x}") for x in ("V1", "V2", "V3")]
    found_in("pt masks", f"| powertrain | full + mask-V1 / V2 / V3 (secondary) | {' / '.join(x[3] for x in mk)} | {' / '.join(pct(x[4]['acc']) for x in mk)} | 2/2 each | ICE | ICE |", "2.14")
    chk("pt masks: every device ICE", all(int(np.trace(x[5]["cm"])) == 2 for x in mk), True)
    yl = [level_cells("bev_ice", "mask", f"mask-{x}") for x in ("yaw", "lead")]
    found_in("pt yaw / lead masks", f"| powertrain | full + mask-yaw / mask-lead | {yl[0][3]} / {yl[1][3]} | {pct(yl[0][4]['acc'])} / {pct(yl[1][4]['acc'])} | 2/2 each | ICE | ICE |", "2.14")
    chk("pt yaw / lead masks: every device ICE", all(int(np.trace(x[5]["cm"])) == 2 for x in yl), True)
    for name, sel, q in (("human-driven", hum, "100.0 (mean P(BEV) 0.06) | - | - | P(BEV) 0.06 (80 windows) | 0.05 (75)"), ("engaged", eng, "50.9 (mean P(BEV) 0.47) | - | - | P(BEV) 0.51 (105) | 0.44 (121)")):
        cell = f"{pct((P[sel].argmax(1) == y[sel]).mean())} (mean P(BEV) {P[sel, 0].mean():.2f}) | - | - | P(BEV) {P[sel & car['TOYOTA_RAV4'], 0].mean():.2f} ({int((sel & car['TOYOTA_RAV4']).sum())}{' windows' if name == 'human-driven' else ''}) | {P[sel & car['HONDA_CIVIC'], 0].mean():.2f} ({int((sel & car['HONDA_CIVIC']).sum())})"
        found_in(f"pt {name} windows", f"| powertrain | full, {name} windows ({int(sel.sum())}) | {cell} |", "2.14")
    Ps, ys, cls, ws, dsz, vs, pers = level_cells("size", "full", "full"); r, c = pers["TOYOTA_RAV4"], pers["HONDA_CIVIC"]
    found_in("size full", f"| size | full (20 ch) | {ws} | {drive_ci('size', 'full', 'full')} | {int(np.trace(vs['cm']))}/{vs['n']} | {r[0]} ({', '.join(pfmt(x) for x in r[1])}), {r[2]} | {c[0]} ({c[1][0]:.3f}, {c[1][1]:.3f}, {c[1][2]:.2f}) miss, {c[2]} |", "2.14")
    chk("size full: Civic P(SMALL) / P(MID) to three decimals; RAV4 P(SMALL)", f"{c[1][0]:.3f} / {c[1][1]:.3f}; {r[1][0]:.2f}", "0.485 / 0.491; 0.54")
    P3s, _, _, w3s, d3s, v3s, per3s = level_cells("size", "V3", "full"); r3, c3 = per3s["TOYOTA_RAV4"], per3s["HONDA_CIVIC"]
    found_in("size V3", f"| size | V3 speed-only | {w3s} | {drive_ci('size', 'V3', 'full')} | {int(np.trace(v3s['cm']))}/{v3s['n']} | {r3[0]} ({', '.join(pfmt(x) for x in r3[1])}) | {c3[0]} ({', '.join(pfmt(x) for x in c3[1])}) |", "2.14")
    s12 = [level_cells("size", k, "full") for k in ("V1", "V2")]
    found_in("size V1 / V2", f"| size | V1 / V2 | {s12[0][3]} / {s12[1][3]} | {pct(s12[0][4]['acc'])} / {pct(s12[1][4]['acc'])} | {int(np.trace(s12[0][5]['cm']))}/2 / {int(np.trace(s12[1][5]['cm']))}/2 | {s12[0][6]['TOYOTA_RAV4'][0]} / {s12[1][6]['TOYOTA_RAV4'][0]} | {s12[0][6]['HONDA_CIVIC'][0]} / {s12[1][6]['HONDA_CIVIC'][0]} |", "2.14")
    chk("size V1: P(MID) RAV4 / Civic", f"{s12[0][6]['TOYOTA_RAV4'][1][1]:.2f} / {s12[0][6]['HONDA_CIVIC'][1][1]:.2f}", "0.53 / 0.53")
    mks = [level_cells("size", "mask", f"mask-{x}") for x in ("V1", "V2", "V3")]
    found_in("size masks", f"| size | full + mask-V1 / V2 / V3 (secondary) | {' / '.join(x[3] for x in mks)} | {' / '.join(pct(x[4]['acc']) for x in mks)} | 2/2 each | SMALL | SMALL |", "2.14")
    chk("size masks: every device SMALL", all(int(np.trace(x[5]["cm"])) == 2 for x in mks), True)
    yls = [level_cells("size", "mask", f"mask-{x}") for x in ("yaw", "lead")]
    found_in("size yaw / lead masks", f"| size | full + mask-yaw / mask-lead | {yls[0][3]} / {yls[1][3]} | {pct(yls[0][4]['acc'])} / {pct(yls[1][4]['acc'])} | {int(np.trace(yls[0][5]['cm']))}/2 ({yls[0][6]['TOYOTA_RAV4'][0]}) / {int(np.trace(yls[1][5]['cm']))}/2 | {yls[0][6]['TOYOTA_RAV4'][0]} / {yls[1][6]['TOYOTA_RAV4'][0]} | {yls[0][6]['HONDA_CIVIC'][0]} / {yls[1][6]['HONDA_CIVIC'][0]} |", "2.14")
    chk("size yaw mask: P(LARGE) range over the two cars / windows SMALL %", f"{min(yls[0][6][p][1][2] for p in car):.2f}-{max(yls[0][6][p][1][2] for p in car):.2f} / {yls[0][3]}", "0.82-0.83 / 3.4")
    for name, sel, q in (("human-driven", hum, None), ("engaged", eng, None)):
        rr = Ps[sel & car["TOYOTA_RAV4"]].mean(0); cc = Ps[sel & car["HONDA_CIVIC"]].mean(0)
        found_in(f"size {name} windows", f"| size | full, {name} windows ({int(sel.sum())}) | {pct((Ps[sel].argmax(1) == ys[sel]).mean())} | - | - | P(SMALL) {rr[0]:.2f}, P(MID) {rr[1]:.2f} | {cc[0]:.2f}, {cc[1]:.2f} |", "2.14")
    chk("Reading: human windows SMALL P range / engaged MID P range (both cars)", f"{min(Ps[hum & car[p], 0].mean() for p in car):.2f}-{max(Ps[hum & car[p], 0].mean() for p in car):.2f} / {min(Ps[eng & car[p], 1].mean() for p in car):.2f}-{max(Ps[eng & car[p], 1].mean() for p in car):.2f}", "0.68-0.72 / 0.58-0.61")
    chk("CCS_PIPELINE: window acc RAV4 / Civic (size full)", f"{pct((Ps[car['TOYOTA_RAV4']].argmax(1) == 0).mean())} / {pct((Ps[car['HONDA_CIVIC']].argmax(1) == 0).mean())}", "55.7 / 48.0")
    chk("CCS_PIPELINE: V3 size drive / window; mask-V3 window / drive", f"{pct(d3s['acc'])} / {w3s}; {mks[2][3]} / {pct(mks[2][4]['acc'])}", "97.6 / 82.2; 96.6 / 100.0")


def sec37_pooled_3b():
    print("\n## 37. Pooled decisions after the two addenda (RESULTS.md 0 / 3.1 / 3.2): the phase-3 scope (ccs + ccs_honda, device x car-model pairs with the pair's own label) and the all-sets totals (+ comma2k19 devices)")
    mc = ccs_meta(); mh = set_meta("ccs_honda"); m2 = set_meta("comma2k19")
    prim = dict(PRIMARY); prim.update(PRIMARY2)
    for task, k, q_ccs, q_h, q_3, q_all, q_pct in (
            ("bev_ice", 2, "BEV 544/630  ICE 1,688/1,917  all 2,232/2,547", "BEV 0/0  ICE 228/231  all 228/231", "BEV 544/630  ICE 1,916/2,148  all 2,460/2,778", "BEV 575/667  ICE 2,224/2,472  all 2,799/3,139", "86.2  90.0  89.2"),
            ("size", 3, "SMALL 1,635/1,798  MID 360/1,301  LARGE 250/1,053  all 2,245/4,152", "SMALL 228/232  MID 0/0  LARGE 0/0  all 228/232", "SMALL 1,863/2,030  MID 360/1,301  LARGE 250/1,053  all 2,473/4,384", "SMALL 1,912/2,084  MID 366/1,338  LARGE 254/1,072  all 2,532/4,494", "91.7  27.4  23.7  56.3")):
        cls = T.TASK_LABEL[task][1]
        tot = {c: [0, 0] for c in cls}
        parts = {}
        if task == "bev_ice":
            parts_b = parts
        for name, m, ds in (("ccs", mc, "ccs"), ("ccs_honda", mh, "ccs_honda")):
            P, y, _ = set_load(ds, task, "full")
            r = unit_eval(P, y, m["pair"].to_numpy(), k)
            parts[name] = r
            for i, c in enumerate(cls):
                tot[c][0] += int(r["cm"][i, i]); tot[c][1] += int(r["sup"][i])
        fmt = lambda h: "  ".join(f"{c} {fmt_int(h[c][0])}/{fmt_int(h[c][1])}" for c in cls) + f"  all {fmt_int(sum(x[0] for x in h.values()))}/{fmt_int(sum(x[1] for x in h.values()))}"  # noqa: E731
        chk(f"{task} ccs pairs", fmt({c: [int(parts['ccs']['cm'][i, i]), int(parts['ccs']['sup'][i])] for i, c in enumerate(cls)}), q_ccs)
        chk(f"{task} ccs_honda pairs", fmt({c: [int(parts['ccs_honda']['cm'][i, i]), int(parts['ccs_honda']['sup'][i])] for i, c in enumerate(cls)}), q_h)
        chk(f"{task} phase 3 pooled (ccs + ccs_honda)", fmt(tot), q_3)
        p3 = dict(tot)
        chk(f"{task} phase-3 shares per class and overall", "  ".join(pct(p3[c][0] / p3[c][1]) for c in cls if p3[c][1]) + "  " + pct(sum(x[0] for x in p3.values()) / sum(x[1] for x in p3.values())), "86.3  89.2  88.6" if task == "bev_ice" else "91.8  27.7  23.7  56.4")
        # comma2k19 at device level
        P, y, _ = set_load("comma2k19", task, "full"); r2 = unit_eval(P, y, m2["vehicle_id"].to_numpy(), k)
        chk(f"{task} comma2k19 devices", "  ".join(f"{c} {int(r2['cm'][i, i])}/{int(r2['sup'][i])}" for i, c in enumerate(cls)), "BEV 0/0  ICE 2/2" if task == "bev_ice" else "SMALL 1/2  MID 0/0  LARGE 0/0")
        for i, c in enumerate(cls):
            tot[c][0] += int(r2["cm"][i, i]); tot[c][1] += int(r2["sup"][i])
        for ds in prim:
            Pd, yd, md, _ = load(ds, task, prim[ds][task]); rd = unit_eval(Pd, yd, md["vehicle_id"].to_numpy(), k)
            if rd["n"] == 0:
                continue
            for i, c in enumerate(cls):
                tot[c][0] += int(rd["cm"][i, i]); tot[c][1] += int(rd["sup"][i])
        chk(f"{task} all sets", fmt(tot), q_all)
        chk(f"{task} all sets shares", "  ".join(pct(tot[c][0] / tot[c][1]) for c in cls) + "  " + pct(sum(x[0] for x in tot.values()) / sum(x[1] for x in tot.values())), q_pct)
    # the device-level view of ccs_honda used in 2.13 / 3.1 bullets (device majority label) vs pairs
    Pb, yb, _ = set_load("ccs_honda", "bev_ice", "full"); rv = unit_eval(Pb, yb, mh["vehicle_id"].to_numpy(), 2)
    chk("ccs_honda powertrain at device level (majority label) vs pairs", f"{int(rv['cm'][1, 1])}/{int(rv['sup'][1])} vs {int(parts_b['ccs_honda']['cm'][1, 1])}/{int(parts_b['ccs_honda']['sup'][1])}", "227/229 vs 228/231")
    print("    (RESULTS_TABLES.md pooled-decision rows: ccs + ccs_honda 544/630, 1916/2148, 2460/2778 and 1863/2030, 360/1301, 250/1053, 2473/4384; comma2k19 2/2 and 1/2; all sets 575/667, 2224/2472, 2799/3139 and 1912/2084, 366/1338, 254/1072, 2532/4494)")


def sec38_labels_3b():
    print("\n## 38. Labels and strata of ccs_honda and comma2k19: footprints against vtt/specs_data/platform_specs.csv and specs_external.csv, size classes against the cuts, strata against build/index/drives.parquet, powertrain against the lab table")
    specs = pd.read_csv(REPO / "vtt" / "specs_data" / "platform_specs.csv", dtype=str, keep_default_na=False).set_index("platform")
    ext = pd.read_csv(HERE_ / "specs_external.csv", dtype=str, keep_default_na=False)
    ext_keys = {}
    for r in ext.itertuples(index=False):
        for k in [x.strip() for x in r.platform_keys.replace(";", " ").replace(",", " ").split() if x.strip()]:
            ext_keys.setdefault(k, r)
    d = pd.read_parquet(T.DATA_ROOT / "build" / "index" / "drives.parquet", columns=["model", "platform", "split_platform", "powertrain_raw"])
    dm = d[d["model"].notna() & (d["model"] != "Unknown")]
    splits = dm.groupby("model")["split_platform"].agg(lambda s: set(s.dropna()))
    seen = {k for k, v in splits.items() if v & {"train", "validation"}}; held = {k for k, v in splits.items() if "test" in v}
    lab_pt = d[d["platform"].notna()].groupby("platform")["powertrain_raw"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else "UNK")
    for ds in ("ccs_honda", "comma2k19"):
        m = set_meta(ds); fp = m["footprint_m2"].to_numpy(float)
        chk(f"{ds}: size_class == classify_footprint / borderline == |fp - cut| < 0.05 on every window", f"{bool((T.classify_footprint(fp) == m['size_class'].to_numpy(object)).all())} / {bool((T.is_borderline(fp) == m['borderline'].astype(bool).to_numpy()).all())}", "True / True")
        rows = []
        for plat, g in m.groupby("platform"):
            f0 = float(g["footprint_m2"].iloc[0]); src = g["label_src"].iloc[0]; model = g["model"].iloc[0]
            if plat in specs.index and src == "lab":
                r = specs.loc[plat]; lf = float(r["wheelbase_mm"]) / 1000 * (float(r["track_front_mm"]) + float(r["track_rear_mm"])) / 2 / 1000
                rows.append((plat, "lab", round(lf, 3), round(f0, 3), r["model"] == model))
            elif plat in ext_keys and src == "public":
                r = ext_keys[plat]; pf = float(r.wheelbase_mm) / 1000 * (float(r.track_front_mm) + float(r.track_rear_mm)) / 2 / 1000
                rows.append((plat, "public", round(pf, 3), round(f0, 3), r.model == model))
            else:
                rows.append((plat, f"{src}: NOT IN ITS TABLE", None, round(f0, 3), False))
        chk(f"{ds}: platforms whose footprint matches its table to 1e-3 and whose model name agrees", f"{sum(1 for r in rows if r[2] is not None and abs(r[2] - r[3]) < 1e-3 and r[4])} of {len(rows)}", f"{len(rows)} of {len(rows)}")
        print("    " + "; ".join(f"{p} {src} {t} vs {f} {'OK' if (t is not None and abs(t - f) < 1e-3 and ok) else 'MISMATCH'}" for p, src, t, f, ok in rows))
        exp = m["model"].map(lambda x: "seen" if x in seen else ("held-out" if x in held else "absent"))
        chk(f"{ds}: stratum mismatches against the drives index (windows)", int((exp != m["stratum"]).sum()), 0)
        print("    strata: " + ", ".join(f"{mo} {s}" for mo, s in m.groupby("model")["stratum"].first().items()))
        mism = []
        for plat, g in m.groupby("platform"):
            have = g["pt_label"].iloc[0]
            if plat in lab_pt.index:
                want = {"ICE": "ICE", "HEV": "HEV", "PHEV": "PHEV", "BEV": "BEV", "FCEV": "FCEV"}.get(lab_pt[plat], "unknown")
                if want != have:
                    mism.append((plat, lab_pt[plat], have))
            elif have != "unknown":
                mism.append((plat, "not in the lab index", have))
        chk(f"{ds}: powertrain mismatches against the lab drives index (platforms absent from it must be unknown)", mism, [])
        per_p = m.groupby("platform").agg(pt=("pt_label", "nunique"), sc=("size_class", "nunique"), mo=("model", "nunique"), st=("stratum", "nunique"))
        chk(f"{ds}: platforms with more than one powertrain / size class / model / stratum", int((per_p > 1).any(axis=1).sum()), 0)
    print(f"    lab drives of the nine Honda platforms + TOYOTA_RAV4: " + ", ".join(f"{p} {d.loc[d['platform'] == p, 'powertrain_raw'].value_counts().to_dict()}" for p in HONDA9 + ("TOYOTA_RAV4",) if (d['platform'] == p).any()))
    print(f"    HONDA_FREED in the lab index: {bool((d['platform'] == 'HONDA_FREED').any())}; Honda Freed model in the index: {bool((dm['model'] == 'Honda Freed').any())} (absent stratum, public spec)")
    chk("HONDA_HRV / HONDA_HRV_3G are split_platform test platforms (held-out)", sorted(set(d.loc[d["platform"].isin(["HONDA_HRV", "HONDA_HRV_3G"]), "split_platform"].dropna())), ["test"])


def sec39_windows_3b():
    print("\n## 39. Window tensors of ccs_honda and comma2k19: the two shared artefacts (sample 0 all-zero; oncoming traffic through the lead gate), identities, ranges, the absent-channel pattern")
    if ARGS.no_windows:
        print("  (skipped: --no-windows)"); return
    for ds, q in (("ccs_honda", dict(v0="100.0", vrel30="1.15", lsp2="2.2", vmin="-63.0", nneg="537", speed="18.2", flags="37 / 15 / 18", das="0.42")),
                  ("comma2k19", dict(v0="100.0", vrel30="0.05", lsp2="0.3", vmin="-61.9", nneg="98", speed=None, flags=None, das=None))):
        X, m2 = T.load_windows(ds); m = set_meta(ds); assert len(m2) == len(m)
        v = X[:, :, T.CH["v_mps"]]; a = X[:, :, T.CH["a_mps2"]]; yaw = X[:, :, T.CH["yaw_rate_radps"]]; lat = X[:, :, T.CH["lat_acc_mps2"]]; curv = X[:, :, T.CH["curv_1pm"]]; ang = X[:, :, T.CH["steer_angle_deg"]]
        lv = X[:, :, T.CH["lead_valid"]] > 0.5; vrel = X[:, :, T.CH["lead_v_rel_mps"]]; lsp = X[:, :, T.CH["lead_speed_mps"]]
        print(f"  {ds}: {X.shape}")
        chk(f"{ds}: windows with v = 0 at sample 0 (%) / mean |v| at k = 0 / mean v at k = 1", f"{pct((v[:, 0] == 0).mean())} / {np.nanmean(np.abs(v[:, 0])):.3f} / {np.nanmean(v[:, 1]):.1f}", f"{q['v0']} / 0.000 / " + f"{np.nanmean(v[:, 1]):.1f}")
        cont = [c for c in T.CHANNELS if c not in T.BINARY_CHANNELS]
        nz0 = {c: round(float((np.nan_to_num(X[:, 0, T.CH[c]]) != 0).mean()), 3) for c in T.CHANNELS if (np.nan_to_num(X[:, 0, T.CH[c]]) != 0).any()}
        chk(f"{ds}: every continuous channel is 0 (or absent) at sample 0; channels with a non-zero value at k = 0 (share of windows)", f"{bool(all((np.nan_to_num(X[:, 0, T.CH[c]]) == 0).all() for c in cont))}; {nz0}", "True; " + ("{'standstill': 1.0, 'das_enabled': 0.005}" if ds == "ccs_honda" else "{'gas_pressed': 0.486, 'standstill': 1.0, 'das_enabled': 0.236}"))
        n_valid = int(lv.sum())
        chk(f"{ds}: oncoming lead: v_rel < -30 m/s % of valid lead samples / lead speed < -2 m/s % / min v_rel / windows with a negative lead speed", f"{100 * (lv & (vrel < -30)).sum() / n_valid:.2f} / {100 * (lv & (lsp < -2)).sum() / n_valid:.1f} / {np.nanmin(vrel[lv]):.1f} / {int(((lv & (lsp < 0)).any(1)).sum())}", f"{q['vrel30']} / {q['lsp2']} / {q['vmin']} / {q['nneg']}")
        print(f"    (lead speed < 0 on {100 * (lv & (lsp < 0)).sum() / n_valid:.1f} % of the valid lead samples; valid lead samples {n_valid:,} = {pct(n_valid / lv.size)} % of all samples)")
        chk(f"{ds}: max |curv v - yaw| below 1e-6 / max |lat - v yaw| below 1e-5", f"{bool(np.nanmax(np.abs(curv * v - yaw)) < 1e-6)} / {bool(np.nanmax(np.abs(lat - v * yaw)) < 1e-5)}", "True / True")
        print(f"    (max |curv v - yaw| {np.nanmax(np.abs(curv * v - yaw)):.1e}, max |lat - v yaw| {np.nanmax(np.abs(lat - v * yaw)):.1e}; yaw NaN share {pct(np.isnan(yaw).mean())} %)")
        sel = (v > 5) & np.isfinite(yaw) & (np.abs(ang) > 2)
        chk(f"{ds}: yaw sign == steering sign on moving samples with |angle| > 2 deg (%)", pct((np.sign(yaw[sel]) == np.sign(ang[sel])).mean()), "100.0")
        nan_t = np.isnan(X[:, :, T.CH["steer_torque_norm"]]).all(1); nan_l = np.isnan(X[:, :, T.CH["lead_d_m"]]).all(1)
        chk(f"{ds}: torque all-NaN exactly where channels_present lacks it / lead_d_m all-NaN windows (= lead=none in the note)", f"{bool(np.array_equal(nan_t, ~m['has_torque'].to_numpy()))} / {int(nan_l.sum())} (no lead group {int((~m['has_lead']).sum())} + group present, no object {int((nan_l & m['has_lead'].to_numpy()).sum())}) == {int((m['lead'] == 'none').sum())}", "True / " + ("1245 (no lead group 1223 + group present, no object 22) == 1245" if ds == "ccs_honda" else "3 (no lead group 0 + group present, no object 3) == 3"))
        if q["speed"]:
            chk(f"{ds}: mean speed / gas / brake / steer pressed % / das (tensor = window set)", f"{np.nanmean(v):.1f} / {100 * np.nanmean(X[:, :, T.CH['gas_pressed']]):.0f} / {100 * np.nanmean(X[:, :, T.CH['brake_pressed']]):.0f} / {100 * np.nanmean(X[:, :, T.CH['steer_pressed']]):.0f} / {np.nanmean(X[:, :, T.CH['das_enabled']]):.2f}", f"{q['speed']} / {q['flags']} / {q['das']}")
        if ds == "comma2k19":
            k = np.arange(X.shape[1])[None, :]
            chk("comma2k19: tensor ranges v / a (samples with a > 4 at k <= 2; max a at k >= 3) / steering / yaw / torque_norm", f"{np.nanmin(v):.1f}..{np.nanmax(v):.1f} / {np.nanmin(a):.1f}..{np.nanmax(a):.1f} ({int(((a > 4) & (k <= 2)).sum())} of {int((a > 4).sum())}; {np.nanmax(a[:, 3:]):.1f}) / {np.nanmin(ang):.0f}..{np.nanmax(ang):.0f} / {np.nanmin(yaw):.2f}..{np.nanmax(yaw):.2f} / {np.nanmin(X[:, :, T.CH['steer_torque_norm']]):.1f}..{np.nanmax(X[:, :, T.CH['steer_torque_norm']]):.1f}", "-0.1..46.6 / -3.5..8.0 (7 of 7; 3.6) / -482..385 / -0.52..0.53 / -1.4..2.2")
            gb = X[:, :, T.CH["gas_pressed"]] > 0.5; bb = X[:, :, T.CH["brake_pressed"]] > 0.5; ss = X[:, :, T.CH["standstill"]] > 0.5
            chk("comma2k19: gas and brake both pressed (samples) / standstill with v > 0.5 m/s", f"{int((gb & bb).sum())} of {gb.size:,} / {int((ss & (v > 0.5)).sum())}", "0 of 228,600 / 0")
        del X


def sec40_hygiene_3b():
    print("\n## 40. Hygiene of the phase-3b outputs: 16-hex / 12-hex tokens, route stamps, absolute paths, ISO dates, the four-letter acronym (repository files + both work folders, names and contents); multi-decimal percentages in the new prose")
    hex16 = _re3.compile(r"(?<![0-9a-zA-Z_])[0-9a-f]{16}(?![0-9a-zA-Z])"); hex12 = _re3.compile(r"(?<![0-9a-zA-Z_])[0-9a-f]{12}(?![0-9a-zA-Z])")
    route1 = _re3.compile(r"\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}"); route2 = _re3.compile(r"\b[0-9a-f]{8}--[0-9a-f]{10}\b")
    abspath = _re3.compile(r"(?<![\w.])/(?:home|data|mnt|tmp)/"); iso = _re3.compile(r"\b\d{4}-\d{2}-\d{2}\b"); acr = _re3.compile(r"(?<![A-Za-z])" + "A" + "D" + "A" + "S(?![A-Za-z])", _re3.I)
    repo_files = [HERE_ / f for f in ("RESULTS.md", "RESULTS_TABLES.md", "results_summary.json", "CCS_PIPELINE.md", "review_recompute.py", "comma2k19_yaw_check.py", "ccs_honda_check.py", "ccs_honda_list.py", "ccs_decode.py") if (HERE_ / f).exists()]
    work_files = [p for ds in ("ccs_honda", "comma2k19") for p in (EXT / ds / "work").glob("*") if p.suffix in (".md", ".csv", ".json")]
    flagged = 0
    for f in repo_files + work_files:
        t = f.read_text(errors="replace")
        h16 = [x for x in hex16.findall(t) if _re3.search(r"[a-f]", x)]; h12 = [x for x in hex12.findall(t) if _re3.search(r"[a-f]", x)]
        rt = route1.findall(t) + route2.findall(t); ap = abspath.findall(t) if f in repo_files else []; isod = iso.findall(t) if f in repo_files else []; ac = acr.findall(t)
        if h16 or h12 or rt or ap or isod or ac:
            flagged += 1
            print(f"    FLAG {f.name}: hex16 {len(h16)} hex12 {len(h12)} route stamps {len(rt)} abs paths {len(ap)} {sorted(set(ap))} iso dates {len(isod)} {sorted(set(isod))[:5]} acronym {len(ac)}")
    names = [p.name for ds in ("ccs_honda", "comma2k19") for p in (EXT / ds / "work").iterdir()]
    badn = [n for n in names if hex16.search(n) or hex12.search(n) or route1.search(n) or route2.search(n)]
    chk("files flagged (repo + work folders) / work-folder names flagged", f"{flagged} of {len(repo_files) + len(work_files)} / {len(badn)} of {len(names)}", f"0 of {len(repo_files) + len(work_files)} / 0 of {len(names)}")
    for ds in ("ccs_honda", "comma2k19"):
        m = set_meta(ds); qc = set_qc(ds)
        toks = sum(bool(hex16.search(str(x)) or hex12.search(str(x)) or route1.search(str(x)) or route2.search(str(x))) for col in ("vehicle_id", "drive_id", "model", "platform", "note") for x in m[col].unique())
        chk(f"{ds}: meta strings with a long hex / route token; pseudonym format; qc ids pseudonymous", f"{toks}; {bool(m['vehicle_id'].str.fullmatch(r'ccs_v_[0-9a-f]{10}').all() and m['drive_id'].str.fullmatch(r'ccs_d_[0-9a-f]{10}').all())}; {bool(qc['vehicle_id'].str.fullmatch(r'ccs_v_[0-9a-f]{10}').all() and qc['drive_id'].str.fullmatch(r'ccs_d_[0-9a-f]{10}').all())}", "0; True; True")
    new = "".join(sec_text(k) for k in ("2.13", "2.14", "3.6")) + sec_text("0").split("Phase 2 (two sets")[0] + sec_text("4")
    multi = sorted(set(_re3.findall(r"\d+\.\d{2,} %", new)))
    chk("percentages with two or more decimals in 2.13 / 2.14 / 3.6 / the phase-3 blocks of 0 / 4", multi, [])
    cp = sec_text("2.13") + sec_text("2.14")
    chk("2.13 / 2.14: the four-letter driver-assistance acronym does not appear", len(acr.findall(cp)), 0)
    pipe = (HERE_ / "CCS_PIPELINE.md").read_text()
    chk("CCS_PIPELINE.md: known gaps 9 and 10 present", f"{'9. Understeer bias' in pipe} / {'10. Sample 0 of every window' in pipe}", "True / True")


def sec41_yaw_check():
    import json as _json
    print("\n## 41. The derived-yaw understeer ratios of 2.14 / CCS_PIPELINE.md gap 9 against comma2k19_yaw_check.py (lab LogReader; decoder channel vs logged carState.yawRate by speed bin)")
    quoted = {"TOYOTA_RAV4": {"1-5": 0.98, "10-15": 1.13, "20-25": 1.54, "25-30": 1.85, "30-40": 2.06}, "HONDA_CIVIC": {"1-5": 0.87, "10-15": 1.03, "20-25": 1.53, "25-30": 1.78, "30-40": 2.01}}
    for fname in ("yaw_check.json", "yaw_check_all.json"):
        f = EXT / "comma2k19" / "work" / fname
        if not f.exists():
            print(f"    ({fname} not found: run `source env.sh && nice -n 10 $PY experiments/transfer/comma2k19_yaw_check.py{' --all' if 'all' in fname else ''}`)"); continue
        r = _json.load(open(f))
        print(f"  {fname}: {r['segments_per_car']} segments per car")
        for plat, car in r["cars"].items():
            dec = car["decoder_vs_logged"]; lg = car["bicycle_from_log_vs_logged"]
            row = []
            for b, qv in quoted[plat].items():
                s = dec.get(b, {}); s2 = lg.get(b, {})
                sl = s.get("slope"); md = s.get("median_ratio")
                tag = "n/a" if sl is None else ("OK" if abs(sl - qv) <= 0.05 or (md is not None and abs(md - qv) <= 0.05) else "DIFF")
                row.append(f"{b}: slope {sl if sl is None else f'{sl:.2f}'} median {md if md is None else f'{md:.2f}'} (n {s.get('n', 0)}; bicycle-from-log slope {s2.get('slope') if s2.get('slope') is None else f'{s2.get('slope'):.2f}'}) vs quoted {qv:.2f} {tag}")
            print(f"    {plat} ({car['n_segments']} segments): " + " | ".join(row))
            print(f"      k fit decoder {car['k_fit_decoder']:.5f} / bicycle-from-log {car['k_fit_log']:.5f} (quoted ~0.001); corr all moving samples {dec['all'].get('corr', float('nan')):.3f}, within-bin corr min {min(s.get('corr', 1.0) for s in dec.values() if s.get('n', 0) >= 50):.3f}; same sign (|yaw| > 0.02) min {min(s.get('same_sign', 1.0) for s in dec.values() if s.get('n', 0) >= 50):.3f}; nearest-match dt max {car['match_dt_max_s'] * 1e3:.0f} ms")
            if "v_absdiff_max" in car["segments"][0]:
                sg = car["segments"]
                print(f"      vs the logged carState on these segments: |dv| max {min(s['v_absdiff_max'] for s in sg):.2f}-{max(s['v_absdiff_max'] for s in sg):.2f} m/s, p99 {min(s['v_absdiff_p99'] for s in sg):.2f}-{max(s['v_absdiff_p99'] for s in sg):.2f}, median ratio {min(s['v_ratio_median'] for s in sg if s['v_ratio_median']):.4f}-{max(s['v_ratio_median'] for s in sg if s['v_ratio_median']):.4f}; "
                      f"steering angle |diff| max {min(s['angle_absdiff_max'] for s in sg):.1f}-{max(s['angle_absdiff_max'] for s in sg):.1f} deg, p99 {min(s['angle_absdiff_p99'] for s in sg):.1f}-{max(s['angle_absdiff_p99'] for s in sg):.1f}, offset {min(s['angle_offset_median'] for s in sg):.1f}-{max(s['angle_offset_median'] for s in sg):.1f}, corr {min(s['angle_corr'] for s in sg):.4f}-{max(s['angle_corr'] for s in sg):.4f} "
                      f"(2.14 on the writer's five: vEgo max <= 0.22, p99 <= 0.11, ratio 1.0000; angle 0.2-0.4 / <= 0.3 deg, corr 0.9996-1.0000)")
        bc = r["both_cars"]
        print(f"    both cars: k fit {bc['k_fit_decoder']:.5f} (decoder) / {bc['k_fit_log']:.5f} (bicycle from the log); " + ", ".join(f"{b} {bc['decoder_vs_logged'][b].get('slope', float('nan')):.2f}" for b in r["bins"]))


def phase3b():
    secs = {34: sec34_honda_decode, 35: sec35_honda_tables, 36: sec36_comma2k19, 37: sec37_pooled_3b, 38: sec38_labels_3b, 39: sec39_windows_3b, 40: sec40_hygiene_3b, 41: sec41_yaw_check}
    want = [int(x) for x in ARGS.sections.split(",") if int(x) in secs] if ARGS.sections else list(secs)
    for k in want:
        secs[k]()
    print(f"    checks so far: {_NCHK[0]} OK / {_NCHK[1]} MISMATCH; table cells: {_NFOUND[0]} FOUND / {_NFOUND[1]} NOT FOUND")
    return 0


def phase3():
    secs = {20: sec20_ccs_decode, 21: sec21_ccs_set, 22: sec22_ccs_levels, 23: sec23_ccs_runs, 24: sec24_ccs_heldout, 25: sec25_ccs_seen_absent, 26: sec26_ccs_lead_state,
            27: sec27_ccs_evidence_confusion, 28: sec28_ccs_unlabelled, 29: sec29_ccs_pooled, 30: sec30_ccs_labels, 31: sec31_ccs_internal_refs, 32: sec32_ccs_windows, 33: sec33_ccs_hygiene}
    want = [int(x) for x in ARGS.sections.split(",") if int(x) in secs] if ARGS.sections else list(secs)
    for k in want:
        secs[k]()
    if not ARGS.sections or any(34 <= int(x) <= 41 for x in ARGS.sections.split(",")):
        print("\n" + "-" * 100 + "\n# PHASE 3b (ccs_honda, comma2k19)\n")
        phase3b()
    return 0

HERE_ = Path(__file__).resolve().parent
ARGS = None


def phase2():
    secs = {10: sec10_csc_size, 11: sec11_csc_pt, 12: sec12_moretorque, 13: sec13_pooled_all, 14: sec14_labels, 15: sec15_csc_windows,
            17: sec17_raw_sign, 18: sec18_ensembles, 19: sec19_hygiene}
    want = [int(x) for x in ARGS.sections.split(",") if int(x) in secs] if ARGS.sections else [k for k in secs if k != 17 or ARGS.raw_sign_check]
    for k in want:
        secs[k]()
    return 0


def main():
    import argparse
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["1", "2", "3", "all"], default="all")
    ap.add_argument("--raw-sign-check", action="store_true", help="also read a few CSVs from two csc zips for the raw lateral-acceleration sign")
    ap.add_argument("--no-windows", action="store_true", help="skip the checks that load windows.npz (csc: 2.5 GB in RAM)")
    ap.add_argument("--sections", default="", help="phase-2 / phase-3 sections to run, e.g. 10,11,13 or 20,22 (default: all; 17 only with --raw-sign-check)")
    ARGS = ap.parse_args()
    if ARGS.phase in ("1", "all"):
        phase1()
    if ARGS.phase in ("2", "all"):
        if ARGS.phase == "all":
            print("\n" + "=" * 100 + "\n# PHASE 2 (csc, moretorque)\n")
        phase2()
    if ARGS.phase in ("3", "all"):
        if ARGS.phase == "all":
            print("\n" + "=" * 100 + "\n# PHASE 3 (ccs)\n")
        phase3()
    return 0


if __name__ == "__main__":
    sys.exit(main())
