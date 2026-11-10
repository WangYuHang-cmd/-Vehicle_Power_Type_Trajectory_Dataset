#!/usr/bin/env python3
"""Collect the zero-shot scores of every external window set into the tables of experiments/transfer/RESULTS.md.

Reads <external>/<dataset>/work/score__<set>__<mask>.json (+ probs__<set>__ens__<mask>.npz and meta.parquet for the
sub-analyses) and internal_reference.json (the in-distribution numbers of the same variants, held-out car models) and
writes RESULTS_TABLES.md (every table, generated) and results_summary.json next to this file. Numbers in percent with one
decimal; external vehicle ids never appear, only public car-model names and dataset / campaign names.

  source env.sh && $CPY experiments/transfer/collect_scores.py

Phase 2 (9 Nov, evening): the commaSteeringControl set (`csc`, V1 channel list, 36 archives = 36 pseudo-vehicles) and the
MoreTorque rlog set (`moretorque`, all 20 channels, 12 cars) are added with their own sub-analyses (per archive and per
stratum for csc; per vehicle, per automation state and per speed regime for moretorque). The pooled-decision scopes of
phase 1 are kept unchanged (PHASE1) and the new sets appear as extra scopes.

Phase 3: the commaCarSegments device-disjoint selection (`ccs`, all 20 channels, kind "ccs_full": full models primary,
V1 / V2 / V3 variants, the full models under the V1 / V2 / V3 masks and two sensitivity masks, yaw and lead) with its
own tables (`ccs_tables`: per platform, per stratum with cluster-bootstrap intervals, held-out models against the
internal per-model result, seen / absent models, lead source and automation state, evidence per vehicle, predicted-share
confusion, unlabelled powertrains). Every ccs table returns nothing while <external>/ccs/work has no meta.parquet or no
full-model probabilities, so the collector runs unchanged before the decode is finished. `ccs_selftest.py` builds a
small synthetic set (`_selftest_ccs`) and checks the tables; main(out_dir=...) writes elsewhere for such tests.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.transfer import common as T  # noqa: E402

HERE = Path(__file__).resolve().parent
EXT = T.EXTERNAL_ROOT
CLASSES = {"size": ("SMALL", "MID", "LARGE"), "bev_ice": ("BEV", "ICE")}
FULL = {"size": "split_platform__size__gru__S9", "bev_ice": "split_platform__bev_ice__resnet__B5"}
VAR = {("size", "V1"): "transfer__size__gru__V1", ("size", "V2"): "transfer__size__gru__V2", ("size", "V3"): "transfer__size__gru__V3",
       ("bev_ice", "V1"): "transfer__bev_ice__resnet__V1", ("bev_ice", "V2"): "transfer__bev_ice__resnet__V2", ("bev_ice", "V3"): "transfer__bev_ice__resnet__V3"}
REF_NAME = {"full": {"size": "full (GRU S9, 20 ch)", "bev_ice": "full (ResNet B5, 20 ch)"},
            "V1": "V1 lateral + speed (10 ch)", "V2": "V2 longitudinal-only (11 ch)", "V3": "V3 speed-only (3 ch)"}
# (set, mask, label, reference entry of internal_reference.json)
RUNS = {
    "V2": [("V2", "full", "V2 variant (longitudinal-only, 11 ch)", "V2"),
           ("V3", "full", "V3 variant (speed-only, 3 ch)", "V3"),
           ("full", "mask-V2", "full 20-ch model, V2 mask (secondary)", "full+mask-V2"),
           ("full", "mask-V3", "full 20-ch model, V3 mask (secondary)", "full+mask-V3")],
    "V3": [("V3", "full", "V3 variant (speed-only, 3 ch)", "V3"),
           ("full", "mask-V3", "full 20-ch model, V3 mask (secondary)", "full+mask-V3")],
    "V0": [("full", "full", "full 20-ch model (lead + torque absent in the data)", "full"),
           ("V1", "full", "V1 variant (lateral + speed, 10 ch)", "V1"),
           ("full", "mask-V1", "full 20-ch model, V1 mask (secondary)", "full+mask-V1"),
           ("V3", "full", "V3 variant (speed-only, 3 ch)", "V3"),
           ("full", "mask-V3", "full 20-ch model, V3 mask (secondary)", "full+mask-V3")],
    "ccs": [("full", "full", "full 20-ch model", "full")],
    "V1": [("V1", "full", "V1 variant (lateral + speed, 10 ch)", "V1"),
           ("full", "mask-V1", "full 20-ch model, V1 mask (secondary)", "full+mask-V1"),
           ("V3", "full", "V3 variant (speed-only, 3 ch)", "V3")],
    "V0mt": [("full", "full", "full 20-ch model (vision lead, localizer yaw)", "full"),
             ("V1", "full", "V1 variant (lateral + speed, 10 ch)", "V1"),
             ("V2", "full", "V2 variant (longitudinal-only, 11 ch)", "V2"),
             ("V3", "full", "V3 variant (speed-only, 3 ch)", "V3"),
             ("full", "mask-V1", "full 20-ch model, V1 mask (secondary)", "full+mask-V1"),
             ("full", "mask-V2", "full 20-ch model, V2 mask (secondary)", "full+mask-V2"),
             ("full", "mask-V3", "full 20-ch model, V3 mask (secondary)", "full+mask-V3")],
    "ccs_full": [("full", "full", "full 20-ch model (radar / stock-ACC / no lead by platform; steering-derived yaw)", "full"),
                 ("V1", "full", "V1 variant (lateral + speed, 10 ch)", "V1"),
                 ("V2", "full", "V2 variant (longitudinal-only, 11 ch)", "V2"),
                 ("V3", "full", "V3 variant (speed-only, 3 ch)", "V3"),
                 ("full", "mask-V1", "full 20-ch model, V1 mask (secondary)", "full+mask-V1"),
                 ("full", "mask-V2", "full 20-ch model, V2 mask (secondary)", "full+mask-V2"),
                 ("full", "mask-V3", "full 20-ch model, V3 mask (secondary)", "full+mask-V3"),
                 ("full", "mask-yaw", "full 20-ch model, yaw group masked (sensitivity: steering-derived yaw)", "full+mask-yaw"),
                 ("full", "mask-lead", "full 20-ch model, lead group masked (sensitivity: vision-only platforms)", "full+mask-lead")],
}
DATASETS = {
    "ultraav": dict(title="ULTra-AV, named-vehicle step3 subset", kind="V2", ref="das", primary={"bev_ice": ("V2", "full"), "size": ("V3", "full")}),
    "openacc": dict(title="OpenACC platoon campaigns (raw JRC files)", kind="V2", ref="das", primary={"bev_ice": ("V2", "full"), "size": ("V3", "full")}),
    "microsim_ev": dict(title="microSIM-ACC / EV", kind="V2", ref="das", primary={"bev_ice": ("V2", "full"), "size": ("V3", "full")}),
    "uga_ev": dict(title="UGA EV platoon", kind="V2", ref="das", primary={"bev_ice": ("V2", "full"), "size": ("V3", "full")}),
    "tesla_yuan": dict(title="Yuan et al. 2024, Tesla Model 3 trips", kind="V3", ref="human", primary={"bev_ice": ("V3", "full"), "size": ("V3", "full")}),
    "openacc_500x": dict(title="OpenACC one-vehicle campaign (Fiat 500X, 20 drivers)", kind="V3", ref="human", primary={"bev_ice": ("V3", "full"), "size": ("V3", "full")}),
    "ved": dict(title="VED, Vehicle Energy Dataset", kind="V3", ref="human", primary={"bev_ice": ("V3", "full"), "size": ("V3", "full")}),
    "a2d2": dict(title="A2D2 bus signals (Audi Q7 e-tron)", kind="V0", ref="human", primary={"bev_ice": ("full", "full"), "size": ("full", "full")}),
    "ccs_sample": dict(title="commaCarSegments, five verification segments", kind="ccs", ref="all", primary={"bev_ice": ("full", "full"), "size": ("full", "full")}),
    "csc": dict(title="commaSteeringControl, 36 platform archives (lateral + speed = the V1 channel list, one window per segment)", kind="V1", ref="das",
                primary={"bev_ice": ("V3", "full"), "size": ("V1", "full")}),   # powertrain primary = V3 (RESULTS.md 1.3: V1 is the weakest powertrain variant)
    "moretorque": dict(title="MoreTorque/rlogs, four community-fork platforms (all 20 channels; vision lead, localizer yaw)", kind="V0mt", ref="all",
                       primary={"bev_ice": ("full", "full"), "size": ("full", "full")}),
    "ccs": dict(title="commaCarSegments, device-disjoint selection (all 20 channels; radar / stock-ACC / no lead by platform; steering-derived yaw)",
                kind="ccs_full", ref="all", primary={"bev_ice": ("full", "full"), "size": ("full", "full")}),
}
PHASE1 = ("ultraav", "openacc", "microsim_ev", "uga_ev", "tesla_yuan", "openacc_500x", "ved", "a2d2", "ccs_sample")   # the sets of RESULTS.md 2.1-2.9
PHASE2 = ("csc", "moretorque")                                                                                      # RESULTS.md 2.10-2.11
PHASE3 = ("ccs",)                                                                                                   # the device-disjoint commaCarSegments set
PAIR_SETS = PHASE3           # sets whose devices appear under several platforms / car models: pooled decisions per device x car-model pair (`_pair_hits`)
LEVELS = ("window", "drive", "vehicle", "model")
CCS_BOOTSTRAP = 300          # vehicle-cluster bootstrap resamples of the per-stratum vehicle / model intervals of ccs_tables (score.py: 1,000 on the pooled set)
CCS_STRATA = ("seen", "held-out", "absent")
CCS_TABLE_TITLES = (         # the section titles ccs_tables renders, in order (ccs_selftest.py checks every one of them)
    "decisions per platform", "pooled per stratum", "held-out models against the internal per-model result", "seen and absent models",
    "lead source and automation state per stratum", "evidence per vehicle", "predicted-share confusion per stratum", "unlabelled powertrains")


def _p(x) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * float(x):.1f}"


def _ci(ci) -> str:
    return "" if not ci else f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def set_tag(task: str, key: str) -> str:
    return FULL[task] if key == "full" else VAR[(task, key)]


def load_score(ds: str, task: str, key: str, mask: str) -> dict | None:
    p = EXT / ds / "work" / f"score__{set_tag(task, key)}__{mask}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_probs(ds: str, task: str, key: str, mask: str) -> np.ndarray | None:
    p = EXT / ds / "work" / f"probs__{set_tag(task, key)}__ens__{mask}.npz"
    return np.load(p, allow_pickle=True)["probs"] if p.exists() else None


def load_meta(ds: str) -> pd.DataFrame:
    return pd.read_parquet(EXT / ds / "work" / "meta.parquet")


def ref_entry(REF: dict, task: str, key: str) -> dict | None:
    """internal_reference.json entry of a run key: 'V2' -> variant, 'full' -> full model, 'full+mask-V2' -> full model masked."""
    tasks = REF["tasks"][task]
    if key == "full":
        return tasks.get(REF_NAME["full"][task])
    if key.startswith("full+mask-"):
        return tasks.get(f"{REF_NAME['full'][task]} + {key.split('+')[1]}")
    return tasks.get(REF_NAME[key])


def recall_str(classes, rec) -> str:
    return " / ".join(f"{c} {_p(x)}" for c, x in zip(classes, rec))


# ----------------------------------------------------------------------------- per dataset tables
def dataset_tables(ds: str, cfg: dict, REF: dict, out: dict) -> list[str]:
    L = []
    for task in ("bev_ice", "size"):
        classes = CLASSES[task]
        rows = []
        per_model_md = None
        unl_md = None
        for key, mask, label, refkey in RUNS[cfg["kind"]]:
            R = load_score(ds, task, key, mask)
            if R is None:
                continue
            ref = ref_entry(REF, task, refkey)
            for lv in LEVELS:
                r = R["levels"].get(lv) or {}
                if not r.get("n_units"):
                    continue
                refs = "" if lv == "model" else "-"               # '-' when internal_reference.json has no entry for the run (e.g. the yaw / lead masks)
                if ref is not None and lv != "model":
                    a = ref["subsets"]["all"][lv]; s = ref["subsets"][cfg["ref"]][lv]
                    refs = f"{_p(a['accuracy'])} / {_p(a['macro_f1'])}"
                    if cfg["ref"] != "all":
                        refs += f"; {cfg['ref']} {_p(s['accuracy'])} / {_p(s['macro_f1'])}"
                rows.append([label if lv == "window" else "", lv, r["n_units"], _p(r["accuracy"]) + _ci(r.get("ci95_accuracy")),
                             _p(r["macro_f1"]) + _ci(r.get("ci95_macro_f1")), f"{_p(1 / len(classes))} / {_p(r['chance_majority'])}",
                             recall_str(classes, r["per_class_recall"]), refs])
            out.setdefault(ds, {}).setdefault(task, {})[f"{key}|{mask}"] = {
                "label": label, "n_windows": R["n_windows"], "n_labelled": R["n_labelled"], "n_vehicles": R["n_vehicles"], "n_drives": R["n_drives"],
                "n_models": R["n_models"], "label_counts": R["label_counts"],
                "levels": {lv: {k_: R["levels"][lv].get(k_) for k_ in ("n_units", "accuracy", "macro_f1", "per_class_recall", "support", "chance_majority", "ci95_accuracy", "ci95_macro_f1")}
                           for lv in LEVELS if R["levels"].get(lv, {}).get("n_units")},
                "levels_no_borderline": {lv: {k_: R["levels_no_borderline"][lv].get(k_) for k_ in ("n_units", "accuracy", "macro_f1", "support")}
                                         for lv in LEVELS if R.get("levels_no_borderline", {}).get(lv, {}).get("n_units")} if R.get("levels_no_borderline") else None,
                "per_model": R["per_model"], "unlabelled": R["unlabelled"], "borderline_models": R.get("borderline_models", [])}
            if (key, mask) == cfg["primary"][task]:
                per_model_md = per_model_table(R, classes)
                unl_md = unlabelled_table(R, classes)
                if R.get("levels_no_borderline"):
                    nb = []
                    for lv in LEVELS:
                        r = R["levels_no_borderline"].get(lv) or {}
                        if r.get("n_units"):
                            nb.append([lv, r["n_units"], _p(r["accuracy"]) + _ci(r.get("ci95_accuracy")), _p(r["macro_f1"]) + _ci(r.get("ci95_macro_f1")),
                                       recall_str(classes, r["per_class_recall"]), "/".join(str(x) for x in r["support"])])
                    unl_md = (unl_md or "") + f"\n\nPrimary run without the borderline models ({', '.join(R['borderline_models'])}):\n\n" + \
                        T.md_table(["level", "units", "accuracy % [95% CI]", "macro-F1 % [95% CI]", "recall per class %", "support"], nb)
        if not rows:
            continue
        R0 = None
        for key, mask, label, refkey in RUNS[cfg["kind"]]:
            R0 = R0 or load_score(ds, task, key, mask)
        L.append(f"### {ds}: {task} ({' / '.join(classes)})")
        L.append("")
        L.append(f"Windows {R0['n_windows']} ({R0['n_labelled']} with a {task} label), vehicles {R0['n_vehicles']}, drives {R0['n_drives']}, car models {R0['n_models']}; "
                 f"label counts {R0['label_counts']}. Internal reference = seed ensemble of the same variant on our held-out car models "
                 f"(accuracy / macro-F1; `{cfg['ref']}` = the matching sub-population, see internal_reference.md).")
        L.append("")
        L.append(T.md_table(["model", "level", "units", "accuracy % [95% CI]", "macro-F1 % [95% CI]", "chance uniform / majority %", "recall per class %",
                             f"internal reference acc / F1 (all{'; ' + cfg['ref'] if cfg['ref'] != 'all' else ''})"], rows))
        if per_model_md:
            L.append("")
            labels = {(k_, m_): l_ for k_, m_, l_, _ in RUNS[cfg["kind"]]}
            L.append(f"Per car model, primary run ({labels[cfg['primary'][task]]}):")
            L.append("")
            L.append(per_model_md)
        if unl_md:
            L.append("")
            L.append(unl_md)
        L.append("")
    return L


def per_model_table(R: dict, classes) -> str:
    rows = []
    for r in R["per_model"]:
        rows.append([r["model"], r["label"], f"{r['footprint_m2']:.3f}" if r["footprint_m2"] is not None else "-", "yes" if r["borderline"] else "",
                     r["powertrain"], r["n_vehicles"], r["n_drives"], r["n_windows"], _p(r["das_frac"]) if r["das_frac"] is not None else "-",
                     f"{r['mean_p_true']:.2f}", _p(r["window_acc"]), _p(r["drive_acc"]), f"{r['vehicles_correct']}/{r['n_vehicles']}",
                     f"{r['model_pred']} ({', '.join(f'{x:.2f}' for x in r['model_probs'])})" + (" OK" if r["model_pred"] == r["label"] else " MISS")])
    return T.md_table(["model", "label", "footprint", "border", "powertrain", "veh", "drives", "win", "DAS %", "mean P(true)", "win acc %", "drive acc %",
                       "veh correct", "model-level decision (P)"], rows)


def unlabelled_table(R: dict, classes) -> str | None:
    if not R["unlabelled"]:
        return None
    rows = [[r["model"], r["label"], r["n_vehicles"], r["n_windows"], ", ".join(f"{c} {_p(s)}" for c, s in zip(classes, r["pred_share"])),
             ", ".join(f"{x:.2f}" for x in r["mean_probs"])] for r in R["unlabelled"]]
    return f"Windows outside the {R['task']} classes (scored, not in the metrics), primary run:\n\n" + \
        T.md_table(["model", "label", "veh", "win", "predicted share %", "mean P"], rows)


# ----------------------------------------------------------------------------- cross-dataset summaries
def summary_table(REF: dict, out: dict) -> list[str]:
    L = []
    for task in ("bev_ice", "size"):
        classes = CLASSES[task]
        rows = []
        for ds, cfg in DATASETS.items():
            key, mask = cfg["primary"][task]
            R = out.get(ds, {}).get(task, {}).get(f"{key}|{mask}")
            if not R:
                continue
            lv = R["levels"]
            ref = ref_entry(REF, task, key if mask == "full" else f"full+{mask}")
            refs = "-"
            if ref is not None:
                a = ref["subsets"]["all"]; s = ref["subsets"][cfg["ref"]]
                refs = f"{_p(a['drive']['accuracy'])} / {_p(a['drive']['macro_f1'])} / {_p(a['vehicle']['accuracy'])} / {_p(a['vehicle']['macro_f1'])}"
                if cfg["ref"] != "all":
                    refs += f" ({cfg['ref']}: {_p(s['drive']['accuracy'])} / {_p(s['drive']['macro_f1'])} / {_p(s['vehicle']['accuracy'])} / {_p(s['vehicle']['macro_f1'])})"
            w, d, v = lv.get("window", {}), lv.get("drive", {}), lv.get("vehicle", {})
            rows.append([ds, R["label"].split(" (")[0], f"{R['n_labelled']} / {d.get('n_units', '-')} / {v.get('n_units', '-')} / {R['n_models']}",
                         f"{_p(w.get('accuracy'))} / {_p(w.get('macro_f1'))}", f"{_p(d.get('accuracy'))}{_ci(d.get('ci95_accuracy'))} / {_p(d.get('macro_f1'))}",
                         f"{_p(v.get('accuracy'))}{_ci(v.get('ci95_accuracy'))} / {_p(v.get('macro_f1'))}",
                         recall_str(classes, v.get("per_class_recall", [None] * len(classes))) if v else "-",
                         f"{_p(1 / len(classes))} / {_p(w.get('chance_majority'))}", refs])
        L.append(f"### {task} ({' / '.join(classes)}): primary run per dataset")
        L.append("")
        L.append(T.md_table(["dataset", "model", "labelled windows / drives / vehicles / models", "window acc / F1 %", "drive acc [95% CI] / F1 %",
                             "vehicle acc [95% CI] / F1 %", "vehicle-level recall per class %", "chance uniform / majority %",
                             "internal reference drive acc / F1 / vehicle acc / F1 (same variant)"], rows))
        L.append("")
    return L


def named_vehicle_tables(out: dict) -> list[str]:
    L = []
    for task in ("bev_ice", "size"):
        classes = CLASSES[task]
        rows = []
        for ds, cfg in DATASETS.items():
            key, mask = cfg["primary"][task]
            R = out.get(ds, {}).get(task, {}).get(f"{key}|{mask}")
            if not R:
                continue
            for r in R["per_model"]:
                rows.append([ds, r["model"], r["label"] + (" (border)" if r["borderline"] else ""), r["powertrain"],
                             f"{r['footprint_m2']:.2f}" if r["footprint_m2"] is not None else "-", r["n_vehicles"], r["n_windows"],
                             _p(r["das_frac"]) if r["das_frac"] is not None else "-", R["label"].split(" (")[0],
                             ", ".join(f"{x:.2f}" for x in r["model_probs"]), r["model_pred"], "OK" if r["model_pred"] == r["label"] else "MISS",
                             f"{r['vehicles_correct']}/{r['n_vehicles']}", _p(r["drive_acc"])])
        L.append(f"### {task}: every named external car model, primary run of its dataset (mean probability over all windows of the model -> arg-max)")
        L.append("")
        L.append(T.md_table(["dataset", "car model", "label", "powertrain", "footprint", "veh", "win", "DAS %", "model", f"P({'/'.join(classes)})", "decision", "hit",
                             "vehicles correct", "drive acc %"], rows))
        L.append("")
    return L


# ----------------------------------------------------------------------------- sub-analyses on the probability files
def _pair_hits(ds: str, task: str, key: str, mask: str) -> dict[str, list[int]]:
    """Per class [units right, units] of one run, one unit = one device x car-model pair judged by its own label (mean ensemble
    probability over the pair's labelled windows -> arg-max). The per-model rows of score.py judge a device against its car
    model's majority label, which is not the device's own platform label when a model spans platforms of two classes (Kona EV
    and gasoline ports; Acura RDX and Lexus IS, whose platforms fall in two size classes). Here the label is part of the unit
    key, so a pair whose platforms carried two labels would count once per label (none in the present set: the 2,547 / 4,152
    pairs of RESULTS.md 2.12 and 3.1)."""
    P = load_probs(ds, task, key, mask)
    if P is None:
        raise FileNotFoundError(f"{ds}: no probability file for the primary run {key} / {mask} of {task} (re-run score.py)")
    m = load_meta(ds)
    if len(P) != len(m):
        raise ValueError(f"{ds}: probability file and meta.parquet differ in length (re-run score.py after a new decode)")
    y = T.task_labels(m, task)
    classes = CLASSES[task]
    if not (y >= 0).any():
        return {c: [0, 0] for c in classes}
    units = np.array([f"{v}|{mo}|{c}" for v, mo, c in zip(m["vehicle_id"].astype(str), m["model"].astype(str), y)])
    _, _, yu, du, _ = _unit_decisions(P, y, units)
    return {c: [int(((du == yu) & (yu == i)).sum()), int((yu == i).sum())] for i, c in enumerate(classes)}


def pooled_decisions(out: dict) -> list[str]:
    """Unit-level hits pooled over the external sets (primary runs): per class, how many units were called right, every unit
    judged against its own label. Unit = the physical car (phase-1 sets, MoreTorque) or the archive (commaSteeringControl),
    taken from the per-model rows of score.py (exact there: every vehicle of those sets carries one car model and one label),
    and the device x car-model pair of commaCarSegments (`_pair_hits` on the probability file: a device under several car
    models counts once per model, each judged by its own platform label)."""
    L = ["### Pooled vehicle-level decisions over the external sets (primary run of each set; one physical car, archive or device x car model = one decision, "
         "judged by its own label)", ""]
    for task in ("bev_ice", "size"):
        classes = CLASSES[task]
        rows = []
        for scope, dss in (("phase-1 sets (2.1-2.9)", [d for d in PHASE1 if d in DATASETS]),
                           ("phase-1 without VED (264 ICE cars would dominate)", [d for d in PHASE1 if d in DATASETS and d != "ved"]),
                           ("longitudinal / speed-only sets only (V2 / V3 variants)", [d for d in PHASE1 if d in DATASETS and DATASETS[d]["kind"] in ("V2", "V3")]),
                           ("commaSteeringControl archives (primary run per task; one archive = one pseudo-vehicle)", ["csc"]),
                           ("MoreTorque cars (full model)", ["moretorque"]),
                           ("commaCarSegments devices (full model; one device x car model = one decision, judged by the device's own platform label; "
                            "a device under more than one car model counts once per model)", list(PHASE3)),
                           ("all sets", list(DATASETS))):
            if not any(out.get(ds, {}).get(task) for ds in dss):      # a scope whose sets are not scored yet (e.g. ccs before the decode) is left out
                continue
            hit = {c: [0, 0] for c in classes}
            for ds in dss:
                key, mask = DATASETS[ds]["primary"][task]
                R = out.get(ds, {}).get(task, {}).get(f"{key}|{mask}")
                if not R:
                    continue
                if ds in PAIR_SETS:                                   # devices under several platforms: own-label pairs, not the model majority
                    for c, (r_, n_) in _pair_hits(ds, task, key, mask).items():
                        hit[c][0] += r_; hit[c][1] += n_
                    continue
                for r in R["per_model"]:
                    hit[r["label"]][0] += r["vehicles_correct"]; hit[r["label"]][1] += r["n_vehicles"]
            tot = [sum(h[0] for h in hit.values()), sum(h[1] for h in hit.values())]
            rows.append([scope] + [f"{hit[c][0]}/{hit[c][1]} ({_p(hit[c][0] / hit[c][1]) if hit[c][1] else '-'})" for c in classes] + [f"{tot[0]}/{tot[1]} ({_p(tot[0] / tot[1]) if tot[1] else '-'})"])
        L.append(f"{task}:"); L.append("")
        L.append(T.md_table(["scope"] + [f"{c} cars correct" for c in classes] + ["all cars"], rows)); L.append("")
    return L


def das_split(out: dict) -> list[str]:
    """Powertrain decision vs automation state inside the sets that hold both (window share with das_frac >= 0.5 vs < 0.5)."""
    L = ["### Automation-engaged vs human-driven windows (primary powertrain run; window accuracy and P(BEV) per sub-population)", ""]
    rows = []
    for ds in ("ultraav", "openacc", "uga_ev", "microsim_ev", "openacc_500x", "tesla_yuan", "ved", "a2d2", "ccs_sample", "csc", "moretorque") + PHASE3:
        cfg = DATASETS[ds]
        key, mask = cfg["primary"]["bev_ice"]
        P = load_probs(ds, "bev_ice", key, mask)
        if P is None:
            continue
        m = load_meta(ds)
        y = T.task_labels(m, "bev_ice")
        das = m["das_frac"].to_numpy(float) >= 0.5
        for name, sel in (("DAS engaged", das), ("human-driven", ~das)):
            lab = sel & (y >= 0)
            if lab.sum() == 0:
                continue
            pred = P[lab].argmax(1); yy = y[lab]
            acc = (pred == yy).mean()
            rec = [float((pred[yy == c] == c).mean()) if (yy == c).any() else None for c in range(2)]
            pbev_ice = float(P[lab][yy == 1, 0].mean()) if (yy == 1).any() else None
            pbev_bev = float(P[lab][yy == 0, 0].mean()) if (yy == 0).any() else None
            rows.append([ds, name, int(lab.sum()), int((yy == 0).sum()), int((yy == 1).sum()), _p(acc), _p(rec[0]), _p(rec[1]),
                         "-" if pbev_bev is None else f"{pbev_bev:.2f}", "-" if pbev_ice is None else f"{pbev_ice:.2f}"])
    L.append(T.md_table(["dataset", "windows", "n", "BEV", "ICE", "accuracy %", "BEV recall %", "ICE recall %", "mean P(BEV) on BEV", "mean P(BEV) on ICE"], rows))
    L.append("")
    # same car, both states (OpenACC and ULTra-AV cars with manual and ACC windows)
    rows = []
    for ds in ("openacc", "ultraav", "uga_ev"):
        cfg = DATASETS[ds]
        key, mask = cfg["primary"]["bev_ice"]
        P = load_probs(ds, "bev_ice", key, mask)
        if P is None:
            continue
        m = load_meta(ds)
        y = T.task_labels(m, "bev_ice")
        das = m["das_frac"].to_numpy(float) >= 0.5
        for model, g in m.groupby("model", sort=True):
            idx = g.index.to_numpy()
            if y[idx[0]] < 0:
                continue
            a = idx[das[idx]]; h = idx[~das[idx]]
            if len(a) < 5 or len(h) < 5:
                continue
            rows.append([ds, model, classes_name(y[idx[0]]), len(a), f"{P[a, 0].mean():.2f}", len(h), f"{P[h, 0].mean():.2f}"])
    if rows:
        L.append("Same car, both automation states (cars with >= 5 windows in each state): mean P(BEV)")
        L.append("")
        L.append(T.md_table(["dataset", "car model", "label", "DAS windows", "P(BEV) DAS", "human windows", "P(BEV) human"], rows))
        L.append("")
    return L


def classes_name(c: int) -> str:
    return CLASSES["bev_ice"][c]


def openacc_campaigns(out: dict) -> list[str]:
    """OpenACC per campaign: the vehicles are confounded with the campaigns, so within-campaign tables are the fair view."""
    L = ["### OpenACC by campaign (primary runs: bev_ice V2 variant, size V3 variant; model-level = mean probability over the model's windows of that campaign)", ""]
    m = load_meta("openacc")
    camp = m["drive_id"].astype(str).str.split("/").str[0]
    Pb = load_probs("openacc", "bev_ice", *DATASETS["openacc"]["primary"]["bev_ice"])
    Ps = load_probs("openacc", "size", *DATASETS["openacc"]["primary"]["size"])
    if Pb is None or Ps is None:
        return []
    yb = T.task_labels(m, "bev_ice"); ys = T.task_labels(m, "size")
    rows = []
    for c in sorted(camp.unique()):
        for model, g in m[camp == c].groupby("model", sort=True):
            idx = g.index.to_numpy()
            pb = Pb[idx].mean(0); ps = Ps[idx].mean(0)
            lb = CLASSES["bev_ice"][yb[idx[0]]] if yb[idx[0]] >= 0 else str(g["powertrain"].iloc[0])
            ls = CLASSES["size"][ys[idx[0]]] if ys[idx[0]] >= 0 else "-"
            rows.append([c, model, len(idx), _p(g["das_frac"].mean()), _p((m.loc[idx, "channels_present"].str.contains("lead_d_m")).mean()) if False else "",
                         lb, f"{pb[0]:.2f}", CLASSES["bev_ice"][int(pb.argmax())] + (" OK" if yb[idx[0]] >= 0 and int(pb.argmax()) == yb[idx[0]] else (" MISS" if yb[idx[0]] >= 0 else " (shares only)")),
                         f"{ls} ({g['footprint_m2'].iloc[0]:.2f})", ", ".join(f"{x:.2f}" for x in ps), CLASSES["size"][int(ps.argmax())] + (" OK" if int(ps.argmax()) == ys[idx[0]] else " MISS")])
    rows = [[r[0], r[1], r[2], r[3], r[5], r[6], r[7], r[8], r[9], r[10]] for r in rows]
    L.append(T.md_table(["campaign", "car model", "win", "DAS %", "powertrain", "P(BEV)", "decision", "size label (footprint)", "P(S, M, L)", "size decision"], rows))
    L.append("")
    return L


def ved_tables(out: dict) -> list[str]:
    """VED: weight-bin ordinal check of the size shares and the powertrain decision per vehicle class."""
    L = ["### VED: size shares per weight bin (V3 size variant; the only ordinal check possible without footprints) and powertrain decisions per vehicle", ""]
    m = load_meta("ved")
    Ps = load_probs("ved", "size", *DATASETS["ved"]["primary"]["size"])
    Pb = load_probs("ved", "bev_ice", *DATASETS["ved"]["primary"]["bev_ice"])
    if Ps is None or Pb is None:
        return []
    note = m["note"].astype(str)
    wb = note.str.extract(r"weight_bin_lb=(\d+)")[0]
    m = m.assign(weight_bin=pd.to_numeric(wb, errors="coerce"))
    rows = []
    for w, g in m.groupby("weight_bin", sort=True, dropna=False):
        idx = g.index.to_numpy()
        share = np.bincount(Ps[idx].argmax(1), minlength=3) / len(idx)
        pm = Ps[idx].mean(0)
        rows.append(["n/a" if pd.isna(w) else f"{int(w)} lb", g["vehicle_id"].nunique(), len(idx), ", ".join(_p(s) for s in share), ", ".join(f"{x:.3f}" for x in pm),
                     f"{(pm * np.array([1, 2, 3])).sum():.3f}"])
    L.append("Pooled over powertrains; `ordinal score` = 1 x P(SMALL) + 2 x P(MID) + 3 x P(LARGE) of the mean probability (rises with size if the model reads size):")
    L.append("")
    L.append(T.md_table(["weight bin", "vehicles", "windows", "predicted share SMALL, MID, LARGE %", "mean P(SMALL, MID, LARGE)", "ordinal score"], rows))
    L.append("")
    # Spearman between weight bin and ordinal score over vehicles
    veh = m.groupby("vehicle_id").agg(weight_bin=("weight_bin", "first"), powertrain=("powertrain", "first"))
    vs = pd.DataFrame({"vehicle_id": m["vehicle_id"], "score": (Ps * np.array([1, 2, 3])).sum(1)}).groupby("vehicle_id")["score"].mean()
    veh = veh.join(vs).dropna(subset=["weight_bin"])
    rho = veh[["weight_bin", "score"]].corr(method="spearman").iloc[0, 1]
    L.append(f"Vehicle-level Spearman correlation between the weight bin and the ordinal score: {rho:.3f} (n = {len(veh)} vehicles with a weight bin).")
    L.append("")
    rows = []
    yb = T.task_labels(m, "bev_ice")
    for pt, g in m.groupby("powertrain", sort=True):
        idx = g.index.to_numpy()
        vid = g["vehicle_id"].to_numpy()
        uid, inv = np.unique(vid, return_inverse=True)
        Pv = np.zeros((len(uid), 2)); np.add.at(Pv, inv, Pb[idx]); Pv /= np.bincount(inv)[:, None]
        did = g["drive_id"].to_numpy(); duid, dinv = np.unique(did, return_inverse=True)
        Pd = np.zeros((len(duid), 2)); np.add.at(Pd, dinv, Pb[idx]); Pd /= np.bincount(dinv)[:, None]
        rows.append([pt, len(uid), len(duid), len(idx), _p((Pb[idx].argmax(1) == 0).mean()), _p((Pd.argmax(1) == 0).mean()), _p((Pv.argmax(1) == 0).mean()),
                     f"{Pb[idx][:, 0].mean():.2f}", f"{np.median(Pv[:, 0]):.2f}"])
    L.append("Powertrain (V3 variant): share of windows / drives / vehicles called BEV, per true powertrain (BEV and ICE are scored; HEV and PHEV are shares only):")
    L.append("")
    L.append(T.md_table(["true powertrain", "vehicles", "drives", "windows", "windows called BEV %", "drives called BEV %", "vehicles called BEV %", "mean P(BEV)", "median vehicle P(BEV)"], rows))
    L.append("")
    return L


# ----------------------------------------------------------------------------- phase 2: commaSteeringControl and MoreTorque
def _cm_stats(y: np.ndarray, pred: np.ndarray, k: int) -> dict:
    """accuracy, macro-F1 (classes with support; None with < 2), recall per class, support and hits per class."""
    cm = np.zeros((k, k)); np.add.at(cm, (y, pred), 1.0)
    tp = np.diag(cm); sup = cm.sum(1); pr = cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = [float(tp[i] / sup[i]) if sup[i] else None for i in range(k)]
        den = 2 * tp + (pr - tp) + (sup - tp)
        f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    has = sup > 0
    return dict(acc=float(tp.sum() / cm.sum()) if cm.sum() else None, f1=float(f1[has].mean()) if has.sum() >= 2 else None, recall=rec,
                support=sup.astype(int).tolist(), hits=tp.astype(int).tolist(), n=int(cm.sum()))


def _unit_decisions(P: np.ndarray, y: np.ndarray, units: np.ndarray):
    """(unit ids, mean P per unit, truth per unit (majority), decision per unit) over the labelled windows."""
    lab = y >= 0
    uid, inv = np.unique(units[lab].astype(str), return_inverse=True)
    k = P.shape[1]
    S = np.zeros((len(uid), k)); np.add.at(S, inv, P[lab]); cnt = np.bincount(inv, minlength=len(uid))
    Pu = S / cnt[:, None]
    yu = np.array([np.bincount(y[lab][inv == u], minlength=k).argmax() for u in range(len(uid))])
    return uid, Pu, yu, Pu.argmax(1), cnt


def _note_field(meta: pd.DataFrame, key: str) -> pd.Series:
    return meta["note"].astype(str).str.extract(rf"{key}=([^;]+)")[0]


def _dec(classes, p) -> str:
    return classes[int(np.argmax(p))]


def csc_tables(out: dict) -> list[str]:
    """commaSteeringControl: one window per segment and no device ids, so vehicle = archive (= 2023 platform). Per archive
    decisions of every run, and the window / archive / model levels pooled per stratum (seen / held-out vs the lab's
    split_platform split, from meta.note)."""
    if not (EXT / "csc" / "work" / "meta.parquet").exists():
        return []
    m = load_meta("csc")
    m["stratum"] = _note_field(m, "stratum")
    runs = {"size": [("V1", "full", "V1"), ("full", "mask-V1", "full+V1 mask"), ("V3", "full", "V3")],
            "bev_ice": [("V3", "full", "V3"), ("V1", "full", "V1"), ("full", "mask-V1", "full+V1 mask")]}
    P = {(t, lab): load_probs("csc", t, k, mk) for t in runs for k, mk, lab in runs[t]}
    if any(v is None for v in P.values()):
        return []
    ys = T.task_labels(m, "size"); yb = T.task_labels(m, "bev_ice")
    L = ["### commaSteeringControl (`csc`): decisions per archive (= pseudo-vehicle = 2023 openpilot platform), all runs", "",
         "Mean probability over the archive's windows -> arg-max; `OK` / `MISS` against the archive label; stratum = the car model's status in the "
         "lab's `split_platform` split (seen = training / validation model, held-out = one of the 41 test models). Size runs: V1 variant (primary), "
         "full model under the V1 mask (secondary), V3 speed-only variant; powertrain runs: V3 speed-only (primary, RESULTS.md 1.3), V1 (the matched "
         "input list), full + V1 mask.", ""]
    rows = []
    veh = m.groupby("vehicle_id", sort=True)
    for arch, g in veh:
        idx = g.index.to_numpy()
        fp = float(g["footprint_m2"].iloc[0]); bl = bool(g["borderline"].iloc[0])
        cs, cb = str(g["size_class"].iloc[0]), str(g["powertrain"].iloc[0])
        cells = [arch, str(g["model"].iloc[0]), f"{fp:.2f}" + (" (border)" if bl else ""), cs, cb, str(g["stratum"].iloc[0]), len(idx), _p(g["das_frac"].mean())]
        for k, mk, lab in runs["size"]:
            pm = P[("size", lab)][idx].mean(0); d = _dec(CLASSES["size"], pm)
            cells.append((", ".join(f"{100 * x:.0f}" for x in pm) + f" -> {d} " + ("OK" if d == cs else "MISS")) if lab == "V1" else (d + (" OK" if d == cs else " MISS")))
        for k, mk, lab in runs["bev_ice"]:
            pm = P[("bev_ice", lab)][idx].mean(0); d = _dec(CLASSES["bev_ice"], pm)
            cells.append(f"{100 * pm[0]:.0f} -> {d} " + ("OK" if d == cb else "MISS"))
        rows.append(cells)
    L.append(T.md_table(["archive", "model", "footprint", "size", "pt", "stratum", "win", "DAS %", "size V1: P(S, M, L) % -> decision", "size full+V1 mask",
                         "size V3", "BEV vs ICE V3: P(BEV) % -> decision", "V1: P(BEV) % -> decision", "full+V1 mask: P(BEV) % -> decision"], rows))
    L.append("")
    # pooled per stratum
    L.append("### commaSteeringControl: pooled per stratum (window = segment = drive here; archive = pseudo-vehicle; model = car model)")
    L.append("")
    strata = [("all", np.ones(len(m), bool)), ("seen", (m["stratum"] == "seen").to_numpy()), ("held-out", (m["stratum"] == "held-out").to_numpy())]
    for task, y in (("size", ys), ("bev_ice", yb)):
        classes = CLASSES[task]
        rows = []
        for k, mk, lab in runs[task]:
            Pt = P[(task, lab)]
            for sname, sel in strata:
                idx = np.nonzero(sel & (y >= 0))[0]
                w = _cm_stats(y[idx], Pt[idx].argmax(1), len(classes))
                _, _, yu, du, _ = _unit_decisions(Pt[idx], y[idx], m["vehicle_id"].to_numpy()[idx])
                v = _cm_stats(yu, du, len(classes))
                _, _, ym, dm, _ = _unit_decisions(Pt[idx], y[idx], m["model"].to_numpy()[idx])
                mo = _cm_stats(ym, dm, len(classes))
                rows.append([lab if sname == "all" else "", sname, f"{len(idx)} / {v['n']} / {mo['n']}", f"{_p(w['acc'])} / {_p(w['f1'])}",
                             recall_str(classes, w["recall"]), f"{_p(v['acc'])} / {_p(v['f1'])}",
                             " / ".join(f"{c} {h}/{s}" for c, h, s in zip(classes, v["hits"], v["support"])),
                             f"{_p(mo['acc'])}", " / ".join(f"{c} {h}/{s}" for c, h, s in zip(classes, mo["hits"], mo["support"])),
                             f"{_p(1 / len(classes))} / {_p(max(w['support']) / w['n'])}"])
        L.append(f"{task} ({' / '.join(classes)}):"); L.append("")
        L.append(T.md_table(["run", "stratum", "windows / archives / models", "window acc / F1 %", "window recall per class %", "archive acc / F1 %",
                             "archives right per class", "model acc %", "models right per class", "chance uniform / majority %"], rows))
        L.append("")
        out.setdefault("csc", {}).setdefault(task, {})["strata"] = rows
    # predicted shares per stratum (size, V1): does the model use the whole label space?
    rows = []
    Pt = P[("size", "V1")]
    for sname, sel in strata:
        for c_true in range(3):
            idx = np.nonzero(sel & (ys == c_true))[0]
            if len(idx) == 0:
                continue
            share = np.bincount(Pt[idx].argmax(1), minlength=3) / len(idx)
            rows.append([sname, CLASSES["size"][c_true], m["vehicle_id"].to_numpy()[idx].size and len(np.unique(m["vehicle_id"].to_numpy()[idx])), len(idx),
                         ", ".join(_p(x) for x in share), ", ".join(f"{x:.2f}" for x in Pt[idx].mean(0))])
    L.append("Size, V1 variant: predicted class shares of the windows per true class and stratum (confusion as shares):"); L.append("")
    L.append(T.md_table(["stratum", "true class", "archives", "windows", "predicted share SMALL, MID, LARGE %", "mean P(SMALL, MID, LARGE)"], rows))
    L.append("")
    return L


def moretorque_tables(out: dict) -> list[str]:
    """MoreTorque: per-vehicle decisions (full models), automation-state split (human / lateral-only fork automation / engaged)
    and speed-regime split, both from the full-model probabilities and meta.note (lat_only=)."""
    if not (EXT / "moretorque" / "work" / "meta.parquet").exists():
        return []
    m = load_meta("moretorque")
    Ps = load_probs("moretorque", "size", "full", "full"); Pb = load_probs("moretorque", "bev_ice", "full", "full")
    if Ps is None or Pb is None:
        return []
    ys = T.task_labels(m, "size"); yb = T.task_labels(m, "bev_ice")
    lat = pd.to_numeric(_note_field(m, "lat_only"), errors="coerce").fillna(0.0).to_numpy(float)
    das = m["das_frac"].to_numpy(float)
    stratum = _note_field(m, "stratum")
    L = ["### MoreTorque (`moretorque`): decisions per vehicle, full 20-channel models", "",
         "One row per physical car (pseudonymous ids); `lat-only %` = share of samples with the fork's lateral-only automation active "
         "(`latActive` without `enabled`, not counted in `das_enabled`); drive = run of consecutive segments.", ""]
    rows = []
    for vid, g in m.groupby("vehicle_id", sort=True):
        idx = g.index.to_numpy()
        cs, cb = str(g["size_class"].iloc[0]), str(g["powertrain"].iloc[0])
        ps = Ps[idx].mean(0); pb = Pb[idx].mean(0)
        _, _, yu, du, _ = _unit_decisions(Ps[idx], ys[idx], g["drive_id"].to_numpy()); dacc_s = float((du == yu).mean())
        _, _, yu2, du2, _ = _unit_decisions(Pb[idx], yb[idx], g["drive_id"].to_numpy()); dacc_b = float((du2 == yu2).mean())
        rows.append([vid, str(g["model"].iloc[0]), f"{cs} / {cb}", str(stratum.iloc[idx[0]]), g["drive_id"].nunique(), len(idx), _p(das[idx].mean()), _p(lat[idx].mean()),
                     ", ".join(f"{100 * x:.0f}" for x in ps) + f" -> {_dec(CLASSES['size'], ps)} " + ("OK" if _dec(CLASSES["size"], ps) == cs else "MISS"), _p(dacc_s),
                     f"{100 * pb[0]:.0f} -> {_dec(CLASSES['bev_ice'], pb)} " + ("OK" if _dec(CLASSES["bev_ice"], pb) == cb else "MISS"), _p(dacc_b)])
    L.append(T.md_table(["vehicle", "model", "size / pt", "stratum", "drives", "win", "DAS %", "lat-only %", "size: P(S, M, L) % -> decision", "size drive acc %",
                         "BEV vs ICE: P(BEV) % -> decision", "pt drive acc %"], rows))
    L.append("")
    # automation state split per model
    L.append("### MoreTorque: automation state per model (full models; window-level accuracy and mean probabilities)")
    L.append("")
    L.append("`human` = das_frac < 0.5 and lat-only < 0.5; `lateral-only` = lat-only >= 0.5 (fork steering while `das_enabled` = 0); "
             "`engaged` = das_frac >= 0.5 (controller or stock cruise enabled).")
    L.append("")
    state = np.where(das >= 0.5, "engaged", np.where(lat >= 0.5, "lateral-only", "human"))
    rows = []
    for model, g in m.groupby("model", sort=True):
        for st in ("human", "lateral-only", "engaged"):
            idx = g.index[state[g.index.to_numpy()] == st].to_numpy()
            if len(idx) == 0:
                continue
            cs = ys[idx[0]]; cb = yb[idx[0]]
            rows.append([model, st, len(idx), _p((Ps[idx].argmax(1) == cs).mean()), ", ".join(f"{x:.2f}" for x in Ps[idx].mean(0)),
                         _p((Pb[idx].argmax(1) == cb).mean()), f"{Pb[idx][:, 0].mean():.2f}"])
    L.append(T.md_table(["model", "state", "windows", "size window acc %", "mean P(S, M, L)", "pt window acc %", "mean P(BEV)"], rows))
    L.append("")
    # speed regime split per model (window mean speed from X)
    try:
        X, _ = T.load_windows("moretorque")
        vmean = np.nanmean(X[:, :, T.CH["v_mps"]], axis=1)
        del X
    except Exception as e:  # pragma: no cover
        L.append(f"(speed-regime table skipped: {e})"); L.append("")
        return L
    bins = [(0, 10), (10, 20), (20, 30), (30, 99)]
    L.append("### MoreTorque: speed regime per model (window mean speed; full models)")
    L.append("")
    rows = []
    for model, g in m.groupby("model", sort=True):
        for lo, hi in bins:
            idx = g.index[(vmean[g.index.to_numpy()] >= lo) & (vmean[g.index.to_numpy()] < hi)].to_numpy()
            if len(idx) < 10:
                continue
            cs = ys[idx[0]]; cb = yb[idx[0]]
            rows.append([model, f"{lo}-{hi if hi < 99 else ''} m/s", len(idx), _p((Ps[idx].argmax(1) == cs).mean()), ", ".join(f"{x:.2f}" for x in Ps[idx].mean(0)),
                         _p((Pb[idx].argmax(1) == cb).mean()), f"{Pb[idx][:, 0].mean():.2f}"])
    L.append(T.md_table(["model", "mean speed", "windows (>= 10)", "size window acc %", "mean P(S, M, L)", "pt window acc %", "mean P(BEV)"], rows))
    L.append("")
    out.setdefault("moretorque", {})["speed_bins"] = rows
    return L


# ----------------------------------------------------------------------------- phase 3: commaCarSegments (device-disjoint selection)
def _ccs_probs(ds: str, task: str) -> dict[str, np.ndarray]:
    """{run key ('full', 'V1', ..., 'full+mask-yaw'): ensemble probabilities} of the ccs_full runs that have a probability file."""
    P = {}
    for key, mask, _label, refkey in RUNS["ccs_full"]:
        p = load_probs(ds, task, key, mask)
        if p is not None:
            P[refkey] = p
    return P


def _internal_per_model() -> dict[str, dict[str, dict]]:
    """{task: {held-out model: dict(label, n_windows, n_drives, n_vehicles, drive_acc, decision, p_true, hit)}} of the full model,
    parsed from internal_per_model.md (column `full`); {} when the file is missing."""
    f = HERE / "internal_per_model.md"
    if not f.exists():
        return {}
    out: dict[str, dict[str, dict]] = {}
    task, cols = None, None
    for line in f.read_text().splitlines():
        if line.startswith("## "):
            task, cols = line[3:].split(" (")[0].strip(), None
            continue
        if task is None or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "held-out model":
            cols = cells
            continue
        if cols is None or len(cells) != len(cols) or "full" not in cols or set(cells[0]) <= {"-"}:
            continue
        row = dict(zip(cols, cells))
        hit = re.match(r"(\d+) / (\w+) \(([\d.]+)\)( x)?$", row["full"])
        if not hit:
            continue
        out.setdefault(task, {})[row["held-out model"]] = dict(
            label=row["label (footprint)"].split(" (")[0], n_windows=int(row["win"]), n_drives=int(row["drives"]), n_vehicles=int(row["veh"]),
            drive_acc=int(hit.group(1)) / 100.0, decision=hit.group(2), p_true=float(hit.group(3)), hit=hit.group(4) is None)
    return out


def _hits_str(classes, hits, support) -> str:
    return " / ".join(f"{c} {h}/{s}" for c, h, s in zip(classes, hits, support))


def _veh_acc(P: np.ndarray, y: np.ndarray, units: np.ndarray) -> tuple[int, int]:
    """(units right, units) at the unit level (mean probability per unit -> arg-max) over the labelled windows."""
    if not (y >= 0).any():
        return 0, 0
    _, _, yu, du, _ = _unit_decisions(P, y, units)
    return int((du == yu).sum()), int(len(yu))


def _ccs_strata_rows(m: pd.DataFrame, P: dict[str, np.ndarray], y: np.ndarray, classes, strata, level_eval, B: int, store: dict) -> list[list]:
    """Table rows (run, stratum, level, ...) of every run of one task; vehicle / model levels carry cluster-bootstrap intervals."""
    k = len(classes)
    rows = []
    for run in [r[3] for r in RUNS["ccs_full"] if r[3] in P]:
        Pt = P[run]
        first = True
        for sname, sel in strata:
            idx = np.nonzero(sel & (y >= 0))[0]
            if len(idx) == 0:
                continue
            res = {}
            for lv in LEVELS:
                if lv == "window":
                    r = _cm_stats(y[idx], Pt[idx].argmax(1), k)
                    res[lv] = dict(n_units=r["n"], accuracy=r["acc"], macro_f1=r["f1"], per_class_recall=r["recall"], support=r["support"], hits=r["hits"],
                                   ci95_accuracy=None, ci95_macro_f1=None)
                elif lv == "drive":
                    _, _, yu, du, _ = _unit_decisions(Pt[idx], y[idx], m["drive_id"].to_numpy()[idx])
                    r = _cm_stats(yu, du, k)
                    res[lv] = dict(n_units=r["n"], accuracy=r["acc"], macro_f1=r["f1"], per_class_recall=r["recall"], support=r["support"], hits=r["hits"],
                                   ci95_accuracy=None, ci95_macro_f1=None)
                else:
                    r = level_eval(m.iloc[idx].reset_index(drop=True), Pt[idx], y[idx], k, lv, B=B, seed=0)
                    cm = np.asarray(r["confusion"])
                    res[lv] = dict(n_units=r["n_units"], accuracy=r["accuracy"], macro_f1=r["macro_f1"], per_class_recall=r["per_class_recall"], support=r["support"],
                                   hits=[int(cm[i, i]) for i in range(k)], ci95_accuracy=r.get("ci95_accuracy"), ci95_macro_f1=r.get("ci95_macro_f1"))
                r2 = res[lv]
                r2["chance_majority"] = (max(r2["support"]) / r2["n_units"]) if r2["n_units"] else None
                rows.append([run if first else "", sname, lv, r2["n_units"], _p(r2["accuracy"]) + _ci(r2["ci95_accuracy"]), _p(r2["macro_f1"]) + _ci(r2["ci95_macro_f1"]),
                             recall_str(classes, r2["per_class_recall"]), _hits_str(classes, r2["hits"], r2["support"]), f"{_p(1 / k)} / {_p(r2['chance_majority'])}"])
                first = False
            store.setdefault(run, {})[sname] = res
    return rows


def ccs_tables(out: dict, ds: str = "ccs") -> list[str]:
    """commaCarSegments device-disjoint selection (all 20 channels; vehicle = device pseudonym, drive = device x route, one
    window per segment): per-platform decisions, pooled levels per stratum (seen / held-out / absent against the lab's
    split_platform split, with vehicle-cluster bootstrap intervals), held-out models against the internal per-model
    result, seen / absent models, lead source and automation state, evidence per vehicle, predicted-share confusion and
    the unlabelled powertrains. Returns [] while the set has no meta.parquet or no full-model probabilities."""
    wd = EXT / ds / "work"
    if not (wd / "meta.parquet").exists():
        return []
    P = {t: _ccs_probs(ds, t) for t in ("size", "bev_ice")}
    if any("full" not in P[t] for t in P):
        return []
    from experiments.transfer.score import level_eval  # noqa: PLC0415  (torch import only once the set exists)
    m = load_meta(ds).reset_index(drop=True)
    if any(len(p) != len(m) for t in P for p in P[t].values()):
        raise ValueError(f"{ds}: probability files and meta.parquet differ in length (re-run score.py after a new decode)")
    m["stratum"] = _note_field(m, "stratum").fillna("absent")
    m["label_src"] = _note_field(m, "label").fillna("none")
    m["lead_src"] = _note_field(m, "lead").fillna("none")
    mixed = m["note"].astype(str).str.contains("merges gasoline")
    pt_raw = m["powertrain"].fillna("unknown").astype(str)
    m["pt_label"] = np.where(pt_raw.eq("unknown") & mixed.to_numpy(), "MIXED", pt_raw)
    m["borderline"] = m["borderline"].fillna(False).astype(bool)
    y = {"size": T.task_labels(m, "size"), "bev_ice": T.task_labels(m, "bev_ice")}
    das = m["das_frac"].to_numpy(float)
    veh = m["vehicle_id"].astype(str).to_numpy(); drv = m["drive_id"].astype(str).to_numpy(); mod = m["model"].astype(str).to_numpy()
    strata = [("all", np.ones(len(m), bool))] + [(s, (m["stratum"] == s).to_numpy()) for s in CCS_STRATA]
    order = {s: i for i, s in enumerate(CCS_STRATA)}
    Sz, Pt = P["size"], P["bev_ice"]
    S = out.setdefault(ds, {}).setdefault("tables", {})
    S["bootstrap"] = CCS_BOOTSTRAP
    S["n_windows"], S["n_vehicles"], S["n_drives"], S["n_models"], S["n_platforms"] = int(len(m)), int(m["vehicle_id"].nunique()), int(m["drive_id"].nunique()), int(m["model"].nunique()), int(m["platform"].nunique())
    L: list[str] = []

    # a. per platform
    L += [f"### commaCarSegments (`{ds}`): decisions per platform, full 20-channel models (V1 / V3 / V2 variants for comparison)", "",
          "One row per openpilot platform; the devices are disjoint from the lab library (vehicle = device pseudonym, drive = device x route, one window "
          "per segment). Sorted by stratum (seen = training / validation model of the lab's `split_platform` split, held-out = one of the 41 test models, "
          "absent = not in the lab inventory), then footprint. `label` = source of footprint and model (lab table / public specs); `lead` = share of the "
          "windows per lead source (radar tracks / stock-ACC object / none = vision-only platform); `DAS %` = mean share of automation-engaged samples "
          "(driving automation system: stock cruise or controls allowed). Model-level decision = mean probability over the platform's labelled windows -> "
          "arg-max (`OK` / `MISS`); `veh right` = devices whose mean probability picks the label; HEV / PHEV / FCEV / MIXED platforms are unlabelled for "
          "the powertrain task and show the BEV probability and the predicted class only.", ""]
    rows = []; plat = []
    for p, g in m.groupby("platform", sort=False):
        idx = g.index.to_numpy()
        fp = g["footprint_m2"].iloc[0]
        plat.append((order.get(str(g["stratum"].iloc[0]), 3), float(fp) if pd.notna(fp) else 99.0, str(p), idx))
    for _, _, p, idx in sorted(plat, key=lambda t: (t[0], t[1], t[2])):
        g = m.loc[idx]
        fp = g["footprint_m2"].iloc[0]; bl = bool(g["borderline"].iloc[0])
        cs = g["size_class"].iloc[0]; cs = str(cs) if pd.notna(cs) else None
        ptl = str(g["pt_label"].iloc[0]); pt_in = ptl in CLASSES["bev_ice"]
        lead = g["lead_src"].value_counts(normalize=True)
        lead_s = " / ".join(f"{k_} {100 * v:.0f}" for k_, v in lead.items())
        cells = [p, str(g["model"].iloc[0]), (f"{fp:.2f}" if pd.notna(fp) else "-") + (" (border)" if bl else ""), cs or "-", ptl if pt_in else f"unlabelled: {ptl}",
                 str(g["stratum"].iloc[0]), str(g["label_src"].iloc[0]), lead_s, int(g["vehicle_id"].nunique()), int(g["drive_id"].nunique()), len(idx), _p(np.nanmean(das[idx]))]
        rec = dict(platform=p, model=str(g["model"].iloc[0]), footprint_m2=(float(fp) if pd.notna(fp) else None), borderline=bl, size_class=cs, powertrain=ptl,
                   stratum=str(g["stratum"].iloc[0]), label_source=str(g["label_src"].iloc[0]), lead_share={str(k_): round(float(v), 3) for k_, v in lead.items()},
                   n_vehicles=int(g["vehicle_id"].nunique()), n_drives=int(g["drive_id"].nunique()), n_windows=int(len(idx)), das_frac=float(np.nanmean(das[idx])))
        # size
        ys = y["size"][idx]; li = idx[ys >= 0]
        if len(li) and cs:
            pm = Sz["full"][li].mean(0); d = _dec(CLASSES["size"], pm)
            vr, vn = _veh_acc(Sz["full"][li], y["size"][li], veh[li]); dr, dn = _veh_acc(Sz["full"][li], y["size"][li], drv[li])
            cells += [", ".join(f"{100 * x:.0f}" for x in pm) + f" -> {d} " + ("OK" if d == cs else "MISS"), f"{vr}/{vn}", _p(dr / dn if dn else None)]
            rec.update(size_full_probs=[round(float(x), 3) for x in pm], size_full_decision=d, size_full_hit=(d == cs), size_vehicles_right=[vr, vn], size_drive_acc=(dr / dn if dn else None))
            for run in ("V1", "V3"):
                if run in Sz:
                    d2 = _dec(CLASSES["size"], Sz[run][li].mean(0)); cells.append(d2 + (" OK" if d2 == cs else " MISS")); rec[f"size_{run}_decision"] = d2
                else:
                    cells.append("-")
        else:
            cells += ["-", "-", "-", "-", "-"]
        # powertrain
        yb = y["bev_ice"][idx]; lb = idx[yb >= 0]
        if pt_in and len(lb):
            pm = Pt["full"][lb].mean(0); d = _dec(CLASSES["bev_ice"], pm)
            vr, vn = _veh_acc(Pt["full"][lb], y["bev_ice"][lb], veh[lb])
            cells += [f"{100 * pm[0]:.0f} -> {d} " + ("OK" if d == ptl else "MISS"), f"{vr}/{vn}"]
            rec.update(pt_full_p_bev=round(float(pm[0]), 3), pt_full_decision=d, pt_full_hit=(d == ptl), pt_vehicles_right=[vr, vn])
            for run in ("V3", "V2"):
                if run in Pt:
                    d2 = _dec(CLASSES["bev_ice"], Pt[run][idx[yb >= 0]].mean(0)); cells.append(d2 + (" OK" if d2 == ptl else " MISS")); rec[f"pt_{run}_decision"] = d2
                else:
                    cells.append("-")
        else:
            pm = Pt["full"][idx].mean(0); d = _dec(CLASSES["bev_ice"], pm)
            cells += [f"{100 * pm[0]:.0f} -> {d} (unlabelled)", "-"]
            rec.update(pt_full_p_bev=round(float(pm[0]), 3), pt_full_decision=d, pt_full_hit=None)
            for run in ("V3", "V2"):
                cells.append((_dec(CLASSES["bev_ice"], Pt[run][idx].mean(0)) + " (unlabelled)") if run in Pt else "-")
        rows.append(cells)
        S.setdefault("per_platform", []).append(rec)
    L.append(T.md_table(["platform", "model", "footprint", "size", "powertrain", "stratum", "label", "lead %", "veh", "drives", "win", "DAS %",
                         "size full: P(S, M, L) % -> decision", "size veh right", "size drive acc %", "size V1", "size V3",
                         "BEV vs ICE full: P(BEV) % -> decision", "pt veh right", "pt V3", "pt V2"], rows))
    L.append("")

    # b. pooled per stratum
    L += [f"### commaCarSegments (`{ds}`): pooled per stratum, every run (window / drive / vehicle = device / model = car model)", "",
          f"Accuracy and macro-F1 in percent; the vehicle and model levels carry 95 % intervals from a vehicle-cluster bootstrap of {CCS_BOOTSTRAP:,} resamples "
          "over the stratum's devices (score.py `level_eval` on the stratum subset; the pooled score files use 1,000). `units right per class` = diagonal "
          "of the confusion matrix / support; chance = 1 / K and the majority share of the level. Runs: full = the 20-channel models; V1 / V2 / V3 = the "
          "reduced-input variants; full+mask-Vk = the full models with that preset masked; full+mask-yaw / full+mask-lead = the sensitivity runs "
          "(steering-derived yaw group masked; lead group masked).", ""]
    S["strata"] = {}
    for task in ("size", "bev_ice"):
        classes = CLASSES[task]
        S["strata"][task] = {}
        rows = _ccs_strata_rows(m, P[task], y[task], classes, strata, level_eval, CCS_BOOTSTRAP, S["strata"][task])
        L.append(f"{task} ({' / '.join(classes)}):"); L.append("")
        L.append(T.md_table(["run", "stratum", "level", "units", "accuracy % [95% CI]", "macro-F1 % [95% CI]", "recall per class %", "units right per class",
                             "chance uniform / majority %"], rows))
        L.append("")

    # c. held-out models against the internal per-model result
    ipm = _internal_per_model()
    L += [f"### commaCarSegments (`{ds}`): held-out models against the internal per-model result (full models)", "",
          "Every held-out `split_platform` test model present in the set: the external vehicle-level accuracy (devices right / devices) and the model-level "
          "decision with P(true), next to the internal result on the lab's own test windows of that model (internal_per_model.md, column `full`: drive "
          "accuracy % / decision (P true)). A model may span several platforms (e.g. gasoline and hybrid ports); `-` = no label for that task or no internal row.", ""]
    rows = []; S["heldout_vs_internal"] = []
    ho = (m["stratum"] == "held-out").to_numpy()
    for model in sorted(set(mod[ho])):
        sel = ho & (mod == model)
        g = m.loc[sel]
        cells = [model, int(g["platform"].nunique()), int(g["vehicle_id"].nunique()), int(sel.sum())]
        rec = dict(model=model, n_platforms=int(g["platform"].nunique()), n_vehicles=int(g["vehicle_id"].nunique()), n_windows=int(sel.sum()))
        for task, Pr in (("size", Sz), ("bev_ice", Pt)):
            classes = CLASSES[task]
            idx = np.nonzero(sel & (y[task] >= 0))[0]
            if len(idx) == 0:
                cells += ["-", "-", "-", "-"]
                continue
            yt = int(np.bincount(y[task][idx], minlength=len(classes)).argmax()); lab = classes[yt]
            pm = Pr["full"][idx].mean(0); d = _dec(classes, pm)
            vr, vn = _veh_acc(Pr["full"][idx], y[task][idx], veh[idx])
            ir = ipm.get(task, {}).get(model)
            cells += [lab, f"{vr}/{vn} ({_p(vr / vn)})", f"{d} ({pm[yt]:.2f})" + (" OK" if d == lab else " MISS"),
                      (f"{_p(ir['drive_acc'])} / {ir['decision']} ({ir['p_true']:.2f})" + ("" if ir["hit"] else " x")) if ir else "-"]
            rec[task] = dict(label=lab, vehicles_right=[vr, vn], decision=d, p_true=round(float(pm[yt]), 3), hit=(d == lab),
                             internal=(dict(drive_acc=ir["drive_acc"], decision=ir["decision"], p_true=ir["p_true"], hit=ir["hit"]) if ir else None))
        rows.append(cells); S["heldout_vs_internal"].append(rec)
    L.append(T.md_table(["held-out model", "platforms", "veh", "win", "size label", "size: veh right (acc %)", "size: model decision (P true)", "size: internal drive acc / decision (P true)",
                         "pt label", "pt: veh right (acc %)", "pt: model decision (P true)", "pt: internal drive acc / decision (P true)"], rows))
    L.append("")

    # d. seen and absent models
    L += [f"### commaCarSegments (`{ds}`): seen and absent models (full models; seen platform, new device)", "",
          "Per car model of the seen stratum (training / validation models of the lab split, here on devices the lab never saw) and of the absent stratum "
          "(models outside the lab inventory): model-level hits, the median per-model vehicle accuracy, then the ten seen models with the lowest P(true) "
          "and every absent model.", ""]
    S["seen_absent"] = {}
    for task, Pr in (("size", Sz), ("bev_ice", Pt)):
        classes = CLASSES[task]
        for st in ("seen", "absent"):
            sel0 = (m["stratum"] == st).to_numpy() & (y[task] >= 0)
            recs = []
            for model in sorted(set(mod[sel0])):
                idx = np.nonzero(sel0 & (mod == model))[0]
                yt = int(np.bincount(y[task][idx], minlength=len(classes)).argmax()); pm = Pr["full"][idx].mean(0); d = _dec(classes, pm)
                vr, vn = _veh_acc(Pr["full"][idx], y[task][idx], veh[idx])
                recs.append(dict(model=model, label=classes[yt], n_platforms=int(m.loc[idx, "platform"].nunique()), n_vehicles=vn, n_windows=int(len(idx)),
                                 vehicles_right=vr, vehicle_acc=(vr / vn if vn else None), p_true=float(pm[yt]), decision=d, hit=(d == classes[yt])))
            if not recs:
                continue
            n_hit = sum(r["hit"] for r in recs); med = float(np.median([r["vehicle_acc"] for r in recs if r["vehicle_acc"] is not None]))
            vr_all = sum(r["vehicles_right"] for r in recs); vn_all = sum(r["n_vehicles"] for r in recs)
            L.append(f"{task} ({' / '.join(classes)}), {st} models: {n_hit} / {len(recs)} right at model level; median per-model vehicle accuracy {_p(med)} %; "
                     f"devices right {vr_all} / {vn_all} ({_p(vr_all / vn_all if vn_all else None)} %).")
            L.append("")
            shown = sorted(recs, key=lambda r: r["p_true"])[:10] if st == "seen" else sorted(recs, key=lambda r: r["model"])
            L.append(T.md_table(["model", "label", "platforms", "veh", "win", "veh right (acc %)", "P(true)", "model decision"],
                                [[r["model"], r["label"], r["n_platforms"], r["n_vehicles"], r["n_windows"], f"{r['vehicles_right']}/{r['n_vehicles']} ({_p(r['vehicle_acc'])})",
                                  f"{r['p_true']:.2f}", r["decision"] + (" OK" if r["hit"] else " MISS")] for r in shown]))
            L.append("")
            S["seen_absent"][f"{task}|{st}"] = dict(n_models=len(recs), n_models_right=int(n_hit), median_vehicle_acc=med, vehicles_right=[vr_all, vn_all],
                                                    models=[{k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in r.items()} for r in recs])

    # e. lead source and automation state per stratum
    L += [f"### commaCarSegments (`{ds}`): lead source and automation state per stratum (full models)", "",
          "`lead` = the platform's lead source (radar tracks / stock-ACC object / none = vision-only); `state`: human = das_frac < 0.5, engaged = das_frac >= 0.5 "
          "(the driving automation system or stock cruise active for most of the minute). Vehicle accuracy inside a cell uses the vehicle's windows of that cell only.", ""]
    state = np.where(das >= 0.5, "engaged", np.where(das < 0.5, "human", "unknown"))
    rows = []; S["lead_state"] = []
    for factor, values, arr in (("lead", ("radar", "scc", "none"), m["lead_src"].to_numpy(str)), ("state", ("human", "engaged"), state)):
        for val in values:
            for sname, sel in strata:
                cell = sel & (arr == val)
                if not cell.any():
                    continue
                c = [factor, val, sname]; rec = dict(factor=factor, value=val, stratum=sname)
                for task, Pr in (("size", Sz), ("bev_ice", Pt)):
                    idx = np.nonzero(cell & (y[task] >= 0))[0]
                    if len(idx) == 0:
                        c += ["0 / 0", "-", "-"]; rec[task] = None; continue
                    wacc = float((Pr["full"][idx].argmax(1) == y[task][idx]).mean()); vr, vn = _veh_acc(Pr["full"][idx], y[task][idx], veh[idx])
                    c += [f"{len(idx)} / {vn}", _p(wacc), f"{_p(vr / vn)} ({vr}/{vn})"]
                    rec[task] = dict(n_windows=int(len(idx)), n_vehicles=vn, window_acc=wacc, vehicles_right=[vr, vn])
                rows.append(c); S["lead_state"].append(rec)
    L.append(T.md_table(["factor", "value", "stratum", "size: win / veh", "size win acc %", "size veh acc % (right/n)", "pt: win / veh", "pt win acc %", "pt veh acc % (right/n)"], rows))
    L.append("")

    # f. evidence per vehicle
    L += [f"### commaCarSegments (`{ds}`): evidence per vehicle (full models; vehicle-level accuracy by labelled windows per device)", "",
          "How many minutes of a new device are needed: devices bucketed by their number of labelled windows (1, 2-4, 5-9, 10+); cells = devices right / devices (accuracy %).", ""]
    buckets = (("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10+", 10, 10 ** 9))
    rows = []; S["evidence"] = []
    for task, Pr in (("size", Sz), ("bev_ice", Pt)):
        for sname, sel in strata:
            idx = np.nonzero(sel & (y[task] >= 0))[0]
            if len(idx) == 0:
                continue
            uid, Pu, yu, du, cnt = _unit_decisions(Pr["full"][idx], y[task][idx], veh[idx])
            c = [task, sname, int(len(uid))]; rec = dict(task=task, stratum=sname, n_vehicles=int(len(uid)), buckets={})
            for name, lo, hi in buckets:
                b = (cnt >= lo) & (cnt <= hi)
                n = int(b.sum()); r = int((du[b] == yu[b]).sum())
                c.append(f"{r}/{n} ({_p(r / n)})" if n else "-"); rec["buckets"][name] = [r, n]
            rows.append(c); S["evidence"].append(rec)
    L.append(T.md_table(["task", "stratum", "devices", "1 window", "2-4 windows", "5-9 windows", "10+ windows"], rows))
    L.append("")

    # g. predicted-share confusion per stratum (size): full and yaw-masked full model
    L += [f"### commaCarSegments (`{ds}`): predicted-share confusion per stratum (size; full model and full model with the yaw group masked)", "",
          "Predicted class shares of the windows per true class and stratum (the confusion matrix as shares) and the mean probabilities; the yaw-masked run shows "
          "how much of the decision rests on the steering-derived yaw / curvature / lateral-acceleration group.", ""]
    rows = []; S["share_confusion"] = []
    for run in ("full", "full+mask-yaw"):
        if run not in Sz:
            continue
        for sname, sel in strata:
            for c_true in range(3):
                idx = np.nonzero(sel & (y["size"] == c_true))[0]
                if len(idx) == 0:
                    continue
                share = np.bincount(Sz[run][idx].argmax(1), minlength=3) / len(idx); pm = Sz[run][idx].mean(0)
                rows.append([run, sname, CLASSES["size"][c_true], int(len(np.unique(veh[idx]))), int(len(idx)), ", ".join(_p(x) for x in share), ", ".join(f"{x:.2f}" for x in pm)])
                S["share_confusion"].append(dict(run=run, stratum=sname, true_class=CLASSES["size"][c_true], n_vehicles=int(len(np.unique(veh[idx]))), n_windows=int(len(idx)),
                                                 pred_share=[round(float(x), 4) for x in share], mean_probs=[round(float(x), 4) for x in pm]))
    L.append(T.md_table(["run", "stratum", "true class", "veh", "win", "predicted share SMALL, MID, LARGE %", "mean P(SMALL, MID, LARGE)"], rows))
    L.append("")

    # h. unlabelled powertrains per platform
    L += [f"### commaCarSegments (`{ds}`): unlabelled powertrains (HEV / PHEV / FCEV / MIXED platforms), predicted BEV share per platform (full model)", "",
          "Scored but outside the BEV / ICE metrics: share of the platform's windows and devices called BEV and the mean P(BEV) (MIXED = the platform merges gasoline "
          "and electrified variants upstream, as in training).", ""]
    rows = []; S["unlabelled"] = []
    unl = y["bev_ice"] < 0
    for (p, lab), g in m.loc[unl].groupby(["platform", "pt_label"], sort=True):
        idx = g.index.to_numpy()
        share = float((Pt["full"][idx].argmax(1) == 0).mean()); pm = float(Pt["full"][idx][:, 0].mean())
        uid, Pu, _, _, _ = _unit_decisions(Pt["full"][idx], np.zeros(len(idx), dtype=np.int64), veh[idx])
        vb = int((Pu.argmax(1) == 0).sum())
        rows.append([p, str(g["model"].iloc[0]), lab, str(g["stratum"].iloc[0]), int(len(uid)), int(len(idx)), _p(share), f"{vb}/{len(uid)}", f"{pm:.2f}"])
        S["unlabelled"].append(dict(platform=p, model=str(g["model"].iloc[0]), powertrain=lab, stratum=str(g["stratum"].iloc[0]), n_vehicles=int(len(uid)), n_windows=int(len(idx)),
                                    bev_share=round(share, 4), vehicles_called_bev=[vb, int(len(uid))], mean_p_bev=round(pm, 4)))
    L.append(T.md_table(["platform", "model", "powertrain", "stratum", "veh", "win", "windows called BEV %", "devices called BEV", "mean P(BEV)"], rows) if rows else "(none)")
    L.append("")
    return L


def _round_floats(o, nd: int = 6):
    """Round every float in a nested structure (JSON outputs: short digit runs, no identifier-like tokens)."""
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round_floats(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round_floats(v, nd) for v in o]
    return o


def main(out_dir: Path | str | None = None) -> int:
    """Collect every set into RESULTS_TABLES.md + results_summary.json (in `out_dir`, default: next to this file)."""
    out_dir = Path(out_dir) if out_dir else HERE
    REF = json.loads((HERE / "internal_reference.json").read_text())
    out: dict = {}
    L = ["# Zero-shot transfer: generated tables", "",
         f"Generated {time.strftime('%-d %b %Y %H:%M')} by experiments/transfer/collect_scores.py from the score__*.json / probs__*__ens__*.npz files of "
         "score.py and internal_reference.json. Accuracy and macro-F1 in percent (one decimal); 95 % intervals from the vehicle-cluster bootstrap "
         "(1,000 resamples; absent when a set has fewer than about 20 bootstrap-able clusters or one vehicle). Chance = 1 / K and the majority share.", ""]
    per_ds = []
    for ds, cfg in DATASETS.items():
        if not (EXT / ds / "work" / "meta.parquet").exists():
            continue
        per_ds += [f"## {ds}: {cfg['title']}", ""] + dataset_tables(ds, cfg, REF, out)
    L += ["## Cross-dataset summary", ""] + summary_table(REF, out)
    L += named_vehicle_tables(out)
    L += pooled_decisions(out)
    L += das_split(out)
    L += openacc_campaigns(out)
    L += ved_tables(out)
    L += csc_tables(out)
    L += moretorque_tables(out)
    L += ccs_tables(out)
    L += ["## Per dataset", ""] + per_ds
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "RESULTS_TABLES.md").write_text("\n".join(L) + "\n")
    (out_dir / "results_summary.json").write_text(json.dumps(_round_floats(out), indent=1, default=lambda o: round(float(o), 6) if isinstance(o, np.floating) else str(o)))
    print(f"wrote {out_dir / 'RESULTS_TABLES.md'} and results_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
