#!/usr/bin/env python3
"""Self-test of the commaCarSegments tables (collect_scores.ccs_tables, kind "ccs_full") on a synthetic window set.

Builds <external>/_selftest_ccs/work/{windows.npz, meta.parquet} by tiling the four verification windows of `ccs_sample`
(small noise on the continuous channels) into 14 synthetic platforms that cover the three strata (seen / held-out / absent),
both label sources (lab / public), the three lead sources (radar / scc / none), labelled (BEV / ICE) and unlabelled
(HEV / PHEV / FCEV / MIXED) powertrains and borderline footprints, with several devices per platform (1-12 windows each)
and multi-window drives; runs score.py for every run of the ccs_full kind on it (small bootstrap, GPU when free); renders
ccs_tables and the standard dataset_tables on it and checks them (every table present, strata sum to the `all` rows,
units right <= support, pooled `all` rows equal the score.py files, JSON-serialisable summary, no long hex token); then
runs the collector's main() into a temporary folder. Never touches <external>/ccs/work.

Understeer-corrected re-decodes (collect_scores.yaw_correction_tables): a synthetic `_selftest_ccs_us` copy of the synthetic set in
which only the three yaw-group channels are changed (divided by 1 + k v^2 with k = 0.001, as `ccs_decode.py --yaw derived-understeer`
would; `yaw=derived-us;k=0.001` in the note) is built and scored with the same nine runs, and the comparison tables are rendered
against `_selftest_ccs` and checked (alignment OK, k read back, deltas = corrected - original everywhere, strata sum to `all`,
platform flips consistent with the platforms-right counts, the yaw-masked runs of both decodes identical). Two deliberately
misaligned copies (`_selftest_ccs_us_bad`: rows permuted; `_selftest_ccs_us_bad2`: one speed sample changed) must be refused with
the ABORTED line and no tables. Finally the collector's main() is run twice into temporary folders, with and without the synthetic
pair registered, and everything before the "Understeer-corrected yaw re-decodes" section must be byte-identical (the re-decodes
stay out of every pre-existing table).

  source env.sh && $CPY experiments/transfer/ccs_selftest.py [--skip-build] [--skip-score] [--out <folder under $D/tmp>]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.transfer import collect_scores as CS  # noqa: E402
from experiments.transfer import common as T  # noqa: E402

DS = "_selftest_ccs"
DS_US = "_selftest_ccs_us"            # the synthetic understeer-corrected copy (yaw group changed, everything else identical)
DS_BAD = "_selftest_ccs_us_bad"       # rows permuted: must be refused by the alignment check
DS_BAD2 = "_selftest_ccs_us_bad2"     # one non-yaw sample changed: must be refused by the alignment check
US_K = 0.001
SAMPLE = "ccs_sample"
V = {"V1": "transfer__bev_ice__resnet__V1__s*,transfer__size__gru__V1__s*", "V2": "transfer__bev_ice__resnet__V2__s*,transfer__size__gru__V2__s*",
     "V3": "transfer__bev_ice__resnet__V3__s*,transfer__size__gru__V3__s*"}
SCORE_RUNS = [[], ["--ckpts", V["V1"]], ["--ckpts", V["V2"]], ["--ckpts", V["V3"]], ["--mask-channels", "V1"], ["--mask-channels", "V2"],
              ["--mask-channels", "V3"], ["--mask-channels", "yaw"], ["--mask-channels", "lead"]]        # = the ccs case of run_scoring.sh
# platform, model, stratum, footprint, powertrain (meta vocabulary), MIXED flag, label source, lead source, base sample platform, devices
PLATFORMS = [
    ("ST_COROLLA_HEV", "Toyota Corolla", "held-out", 4.093, "HEV", False, "lab", "radar", "TOYOTA_RAV4_TSS2", 5),
    ("ST_COROLLA_ICE", "Toyota Corolla", "held-out", 4.139, "ICE", False, "lab", "radar", "TOYOTA_RAV4_TSS2", 4),
    ("ST_LEAF", "Nissan Leaf", "held-out", 4.160, "BEV", False, "lab", "none", "CHEVROLET_BOLT_EUV", 4),
    ("ST_SIENNA", "Toyota Sienna", "held-out", 5.212, "ICE", False, "lab", "radar", "TOYOTA_RAV4_TSS2", 3),
    ("ST_MODEL3", "Tesla Model 3", "seen", 4.571, "BEV", False, "lab", "none", "TESLA_MODEL_3", 4),
    ("ST_RIVIAN", "Rivian R1T/R1S", "seen", 5.578, "BEV", False, "lab", "none", "RIVIAN_R1", 3),
    ("ST_BOLT", "Chevrolet Bolt EUV", "seen", 3.986, "BEV", False, "lab", "none", "CHEVROLET_BOLT_EUV", 4),
    ("ST_IONIQ5", "Hyundai Ioniq 5", "seen", 4.826, "BEV", False, "lab", "scc", "TOYOTA_RAV4_TSS2", 3),
    ("ST_CAMRY", "Toyota Camry", "seen", 4.468, "HEV", False, "lab", "radar", "TOYOTA_RAV4_TSS2", 3),
    ("ST_EXPLORER", "Ford Explorer", "seen", 5.139, "unknown", True, "lab", "radar", "TOYOTA_RAV4_TSS2", 3),
    ("ST_OCTAVIA", "Skoda Octavia", "absent", 4.149, "ICE", False, "public", "none", "TESLA_MODEL_3", 3),
    ("ST_MIRAI", "Toyota Mirai", "absent", 4.281, "FCEV", False, "public", "radar", "TOYOTA_RAV4_TSS2", 2),
    ("ST_DURANGO", "Dodge Durango", "absent", 4.946, "ICE", False, "public", "none", "RIVIAN_R1", 2),
    ("ST_IONIQ_PHEV", "Hyundai Ioniq Plug-in", "absent", 4.217, "PHEV", False, "public", "scc", "TOYOTA_RAV4_TSS2", 2),
]
WINDOWS_PER_DEVICE = (1, 2, 3, 4, 6, 8, 12, 2, 5, 1, 9, 3)       # cycled over the devices: every bucket of the evidence table is populated
HEX_RE = re.compile(r"(?<![0-9a-zA-Z])[0-9a-fA-F]{12,}(?![0-9a-zA-Z])")   # 12- and 16-hex identifier tokens


def build(rng_seed: int = 0) -> dict:
    """Tile the ccs_sample windows into the synthetic set; returns the save_windows summary."""
    X0, m0 = T.load_windows(SAMPLE)
    base = {str(p): i for i, p in enumerate(m0["platform"].to_numpy(object))}
    rng = np.random.default_rng(rng_seed)
    cont = [c for c in T.CHANNELS if c not in T.BINARY_CHANNELS]
    Xs, rows = [], []
    vi, di, k = 0, 0, 0
    for plat, model, stratum, fp, pt, mixed, label_src, lead_src, base_plat, n_dev in PLATFORMS:
        xb = X0[base[base_plat]]
        torque_ok = np.isfinite(xb[:, T.CH["steer_torque_norm"]]).any()
        sd = {c: (float(np.nanstd(xb[:, T.CH[c]])) if np.isfinite(xb[:, T.CH[c]]).any() else 0.0) for c in cont}
        for d in range(n_dev):
            vid = f"ccs_v_st{vi:03d}"; vi += 1
            n_win = WINDOWS_PER_DEVICE[k % len(WINDOWS_PER_DEVICE)]; k += 1
            engaged = (vi + d) % 3 != 0                                  # about two thirds of the devices mostly engaged
            seg = 0
            while n_win > 0:
                run = min(n_win, int(rng.integers(1, 6)))               # a drive = 1-5 consecutive segments
                did = f"ccs_d_st{di:04d}"; di += 1
                for j in range(run):
                    x = xb.copy()
                    for c in cont:
                        col = x[:, T.CH[c]]
                        fin = np.isfinite(col)
                        if fin.any() and sd[c] > 0:
                            col[fin] += rng.normal(0.0, 0.02 * sd[c], size=int(fin.sum())).astype(np.float32)
                    das = 1.0 if engaged else 0.0
                    x[:, T.CH["das_enabled"]] = das
                    Xs.append(x)
                    note = [f"stratum={stratum}", f"label={label_src}", f"lead={lead_src}", "yaw=derived", f"torque={'route-q95' if torque_ok else 'absent'}",
                            "can_valid=0.99", f"ctrl_allowed={das:.2f}", f"cruise={das:.2f}"]
                    if mixed:
                        note.append("platform merges gasoline and electrified variants")
                    rows.append(dict(dataset=DS, vehicle_id=vid, model=model, platform=plat, drive_id=did, w_idx=seg + j, t0_s=60.0 * (seg + j),
                                     das_frac=das, footprint_m2=fp, powertrain=pt, channels_present=str(m0["channels_present"].iloc[base[base_plat]]),
                                     note=";".join(note)))
                seg += run + 2
                n_win -= run
    X = np.stack(Xs).astype(np.float32)
    meta = pd.DataFrame(rows)
    return T.save_windows(DS, X, meta, strict=True)


def build_us(k: float = US_K) -> dict:
    """The synthetic understeer-corrected copy of the synthetic set: yaw_rate / curv / lat_acc divided by 1 + k v^2 (the NaN pattern of
    the curvature is kept), every other channel and every meta column identical, `yaw=derived` -> `yaw=derived-us;k=<k>` in the note."""
    X, m = T.load_windows(DS)
    X = X.copy()
    v = X[:, :, T.CH["v_mps"]].astype(np.float64)
    f = (1.0 + k * v * v).astype(np.float32)
    for c in T.CHANNEL_GROUPS["yaw"]:
        X[:, :, T.CH[c]] = X[:, :, T.CH[c]] / f
    m = m.copy()
    m["dataset"] = DS_US
    m["note"] = m["note"].astype(str).str.replace("yaw=derived;", f"yaw=derived-us;k={k:g};", regex=False)
    assert m["note"].str.contains("yaw=derived-us;k=").all()
    return T.save_windows(DS_US, X, m, strict=True)


def build_bad() -> None:
    """Two misaligned copies of the synthetic corrected set: rows permuted (DS_BAD) and one speed sample changed (DS_BAD2); the
    probability / score files of DS_US are copied next to them so that only the alignment check can refuse them."""
    X, m = T.load_windows(DS_US)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(m))
    mb = m.iloc[perm].reset_index(drop=True).assign(dataset=DS_BAD)
    T.save_windows(DS_BAD, X[perm], mb, strict=True)
    X2 = X.copy(); X2[0, 10, T.CH["v_mps"]] += 0.5
    T.save_windows(DS_BAD2, X2, m.assign(dataset=DS_BAD2), strict=True)
    src = T.work_dir(DS_US)
    for ds in (DS_BAD, DS_BAD2):
        dst = T.work_dir(ds)
        for f in list(src.glob("probs__*.npz")) + list(src.glob("score__*")):
            (dst / f.name).write_bytes(f.read_bytes())


def score(log: Path, bootstrap: int = 50, ds: str = DS) -> list[dict]:
    """score.py for every run of the ccs_full kind on the synthetic set (sequential; GPU when available)."""
    res = []
    with log.open("a") as fh:
        for extra in SCORE_RUNS:
            cmd = [sys.executable, str(HERE / "score.py"), "--dataset", ds, "--bootstrap", str(bootstrap)] + extra
            t0 = time.time()
            fh.write(f"### {' '.join(cmd[2:])}\n"); fh.flush()
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO), check=False)
            res.append(dict(args=extra, returncode=r.returncode, seconds=round(time.time() - t0, 1)))
            print(f"[selftest] {ds}: score.py {' '.join(extra) or '(full models)'}: rc={r.returncode} {res[-1]['seconds']} s", flush=True)
    return res


def check_us(out_dir: Path) -> dict:
    """Render yaw_correction_tables on the synthetic pair and verify it; the two misaligned copies must be refused."""
    rep: dict = {}
    out: dict = {}
    L = CS.yaw_correction_tables(out, pairs={DS_US: DS})
    assert L and L[0] == CS.YAW_CORRECTION_TITLE, "yaw_correction_tables returned nothing or no title"
    text = "\n".join(L)
    assert "ABORTED" not in text, text[:500]
    rec = out["yaw_correction"][DS_US]
    al = rec["alignment"]
    assert al["ok"] and al["nonyaw_max_abs_diff"] == 0 and al["meta_equal_except_note"] and al["note_equal_after_yaw_strip"], al
    assert all(al["yaw_max_abs_diff"][c] > 0 for c in CS.YAW_GROUP), al["yaw_max_abs_diff"]
    assert rec["k"] == f"{US_K:g}", rec["k"]
    assert rec["original"] == DS
    # (ii) delta arithmetic, strata sum to 'all', hits consistent with the accuracies
    n_cells = 0
    for task, runs in rec["strata"].items():
        assert set(runs) == set(CS.YAW_RUNS), (task, sorted(runs))
        for run, by_s in runs.items():
            for lv in CS.YAW_LEVELS:
                tot = by_s["all"][lv]["n_units"]
                assert tot == sum(by_s[s_][lv]["n_units"] for s_ in CS.CCS_STRATA if s_ in by_s), (task, run, lv)
                for sname, res in by_s.items():
                    r = res[lv]
                    a, b = r["accuracy"]
                    assert abs(r["delta_accuracy"] - (b - a)) < 1e-12, (task, run, sname, lv)
                    if r["macro_f1"][0] is not None and r["macro_f1"][1] is not None:
                        assert abs(r["delta_macro_f1"] - (r["macro_f1"][1] - r["macro_f1"][0])) < 1e-12
                    for j in (0, 1):
                        assert abs(r["accuracy"][j] - sum(r["hits"][j]) / r["n_units"]) < 1e-9, (task, run, sname, lv, j)
                        assert all(h <= sp for h, sp in zip(r["hits"][j], r["support"]))
                    n_cells += 1
    rep["strata_cells_checked"] = n_cells
    # the original side equals the ccs_tables / score.py numbers of the original set (same decision rule)
    for task in ("size", "bev_ice"):
        R = CS.load_score(DS, task, "full", "full")
        for lv in ("window", "vehicle", "model"):
            assert abs(R["levels"][lv]["accuracy"] - rec["strata"][task]["full"]["all"][lv]["accuracy"][0]) < 1e-9, (task, lv)
    # (iii) flips consistent with the platforms-right counts
    for task, fl in rec["flips"].items():
        b0, b1 = fl["platforms_right"]
        assert b1 - b0 == sum(fl["recovered"].values()) - sum(fl["lost"].values()), (task, fl["platforms_right"], fl["recovered"], fl["lost"])
        assert len(fl["flipped"]) == sum(1 for f in fl["flipped"] if f["decision_before"] != f["decision_after"])
        assert sum(1 for f in fl["flipped"] if f["kind"] == "recovered") == sum(fl["recovered"].values())
        assert sum(1 for f in fl["flipped"] if f["kind"] == "lost") == sum(fl["lost"].values())
        for c, (d0, d1, n) in fl["devices_right_per_class"].items():
            assert 0 <= d0 <= n and 0 <= d1 <= n
        # the device counts per class of the full run equal the 'all' / vehicle hits of (ii)
        hits = rec["strata"][task]["full"]["all"]["vehicle"]["hits"]
        assert [v[0] for v in fl["devices_right_per_class"].values()] == hits[0] and [v[1] for v in fl["devices_right_per_class"].values()] == hits[1], task
    rep["flipped_platforms"] = {t: len(fl["flipped"]) for t, fl in rec["flips"].items()}
    # (iv) the yaw-masked runs cannot see the changed channels
    assert rec["mask_yaw"] and all(v["probs_equal"] for v in rec["mask_yaw"].values()), rec["mask_yaw"]
    for task, mk in rec["mask_yaw"].items():
        for lv in CS.YAW_LEVELS:
            assert mk[lv]["accuracy"][0] == mk[lv]["accuracy"][1], (task, lv, mk[lv])
    # (v) summary line present, no hex token, JSON round trip
    assert rec["summary"].startswith(f"{DS_US} (K = {US_K:g}) against {DS}") and "yaw-masked runs agree: yes" in rec["summary"]
    assert f"Summary: {rec['summary']}" in text
    assert not HEX_RE.findall(text), HEX_RE.findall(text)[:5]
    js = json.dumps(CS._round_floats(out), default=str)
    assert '"yaw_correction"' in js and "numpy" not in js
    # misaligned copies are refused with the ABORTED line and no tables
    for ds_bad, expect in ((DS_BAD, "vehicle_id"), (DS_BAD2, "non-yaw channels differ: v_mps")):
        out_b: dict = {}
        Lb = CS.yaw_correction_tables(out_b, pairs={ds_bad: DS})
        tb = "\n".join(Lb)
        assert "ABORTED" in tb and expect in tb, tb[-600:]
        assert "original -> corrected per stratum" not in tb and "decision flips" not in tb and "Summary:" not in tb
        assert not out_b["yaw_correction"][ds_bad]["alignment"]["ok"] and "strata" not in out_b["yaw_correction"][ds_bad]
        rep[f"refused_{ds_bad}"] = out_b["yaw_correction"][ds_bad]["alignment"]["reason"]
    (out_dir / "yaw_correction_selftest.md").write_text(f"# yaw_correction_tables self-test on `{DS_US}` against `{DS}`\n\n"
                                                        f"Generated {time.strftime('%-d %b %Y %H:%M')} by experiments/transfer/ccs_selftest.py.\n\n" + text + "\n")
    return rep


def check_untouched(out_dir: Path) -> dict:
    """The collector's main() with the synthetic pair registered (kind ccs_us) and without it: everything before the understeer section
    must be byte-identical (the re-decodes stay out of every pre-existing table) and the section must hold the synthetic set."""
    gen = re.compile(r"^Generated .*$", re.M)
    saved = (dict(CS.DATASETS), dict(CS.YAW_US))
    try:
        CS.DATASETS[DS_US] = dict(title="synthetic understeer-corrected copy (self-test)", kind="ccs_us", ref="all",
                                  primary={"bev_ice": ("full", "full"), "size": ("full", "full")})
        CS.YAW_US[DS_US] = DS
        assert CS.main(out_dir=out_dir / "collector_with_us") == 0
    finally:
        CS.DATASETS.clear(); CS.DATASETS.update(saved[0]); CS.YAW_US.clear(); CS.YAW_US.update(saved[1])
    assert CS.main(out_dir=out_dir / "collector") == 0
    with_us = gen.sub("Generated <t>", (out_dir / "collector_with_us" / "RESULTS_TABLES.md").read_text())
    without = gen.sub("Generated <t>", (out_dir / "collector" / "RESULTS_TABLES.md").read_text())
    assert CS.YAW_CORRECTION_TITLE in with_us, "understeer section missing"
    head, tail = with_us.split(CS.YAW_CORRECTION_TITLE, 1)
    if CS.YAW_CORRECTION_TITLE in without:                        # real re-decodes present: compare up to the section in both
        without_head = without.split(CS.YAW_CORRECTION_TITLE, 1)[0]
    else:
        without_head = without
    assert head == without_head, "a pre-existing table changed when the synthetic understeer pair was registered"
    assert DS_US not in head and f"### {DS_US} against {DS}" in tail
    js = json.loads((out_dir / "collector_with_us" / "results_summary.json").read_text())
    assert DS_US in js["yaw_correction"] and DS_US not in {k for k in js if k != "yaw_correction"}
    js0 = json.loads((out_dir / "collector" / "results_summary.json").read_text())
    assert {k: v for k, v in js.items() if k != "yaw_correction"} == {k: v for k, v in js0.items() if k != "yaw_correction"}, "results_summary.json differs outside yaw_correction"
    return dict(head_bytes=len(head), section_bytes=len(tail))


def check(out_dir: Path) -> dict:
    """Render the tables on the synthetic set and verify them; writes the rendered text to out_dir."""
    rep: dict = {}
    out: dict = {}
    L = CS.ccs_tables(out, ds=DS)
    assert L, "ccs_tables returned nothing"
    text = "\n".join(L)
    titles = [ln for ln in text.splitlines() if ln.startswith("### ")]        # md_table returns one multi-line string per table
    for t in CS.CCS_TABLE_TITLES:
        assert any(t in ln for ln in titles), f"table missing: {t}"
    rep["n_tables"] = len(titles)
    S = out[DS]["tables"]
    # strata sum to the 'all' row and units right <= support, at every level of every run
    n_checked = 0
    for task, runs in S["strata"].items():
        for run, by_stratum in runs.items():
            for lv in CS.LEVELS:
                tot = by_stratum["all"][lv]["n_units"]
                parts = sum(by_stratum[s][lv]["n_units"] for s in CS.CCS_STRATA if s in by_stratum)
                assert tot == parts, (task, run, lv, tot, parts)
                for s, res in by_stratum.items():
                    r = res[lv]
                    assert all(h <= sp for h, sp in zip(r["hits"], r["support"])), (task, run, s, lv, r["hits"], r["support"])
                    assert sum(r["support"]) == r["n_units"] and sum(r["hits"]) <= r["n_units"], (task, run, s, lv)
                    if r["accuracy"] is not None and r["n_units"]:
                        assert abs(r["accuracy"] - sum(r["hits"]) / r["n_units"]) < 1e-9, (task, run, s, lv)
                    n_checked += 1
    rep["strata_cells_checked"] = n_checked
    assert len(S["strata"]["size"]) == len(CS.RUNS["ccs_full"]), f"runs in the strata table: {sorted(S['strata']['size'])}"
    # the pooled 'all' rows equal the score.py result files (same ensemble, same decision rule)
    for task in ("size", "bev_ice"):
        R = CS.load_score(DS, task, "full", "full")
        for lv in CS.LEVELS:
            a, b = R["levels"][lv]["accuracy"], S["strata"][task]["full"]["all"][lv]["accuracy"]
            assert abs(a - b) < 1e-9, (task, lv, a, b)
            assert R["levels"][lv]["n_units"] == S["strata"][task]["full"]["all"][lv]["n_units"]
    # per-platform counts
    for r in S["per_platform"]:
        for key in ("size_vehicles_right", "pt_vehicles_right"):
            if key in r:
                assert 0 <= r[key][0] <= r[key][1] <= r["n_vehicles"], (r["platform"], key, r[key])
    assert {r["stratum"] for r in S["per_platform"]} == set(CS.CCS_STRATA)
    assert {r["label_source"] for r in S["per_platform"]} == {"lab", "public"}
    assert {k for r in S["per_platform"] for k in r["lead_share"]} == {"radar", "scc", "none"}
    assert any(r["powertrain"] == "MIXED" for r in S["per_platform"]) and any(r["borderline"] for r in S["per_platform"])
    assert len(S["heldout_vs_internal"]) == 3 and all(r["size"]["internal"] for r in S["heldout_vs_internal"]), "internal per-model rows not matched"
    assert all(len(r["buckets"]) == 4 for r in S["evidence"])
    assert S["unlabelled"] and {r["powertrain"] for r in S["unlabelled"]} == {"HEV", "PHEV", "FCEV", "MIXED"}
    # identifiers: pseudonyms only, no long hex token
    hits = HEX_RE.findall(text)
    assert not hits, f"hex-like tokens in the rendered tables: {hits[:5]}"
    # JSON round trip of the summary block
    js = json.dumps(CS._round_floats(out), default=lambda o: round(float(o), 6) if isinstance(o, np.floating) else str(o))
    assert '"strata"' in js and "numpy" not in js
    rep["summary_json_bytes"] = len(js)
    # the standard per-dataset tables with the new kind: unlabelled table rendered, '-' reference for the sensitivity masks
    REF = json.loads((HERE / "internal_reference.json").read_text())
    out2: dict = {}
    L2 = CS.dataset_tables(DS, CS.DATASETS["ccs"], REF, out2)
    t2 = "\n".join(L2)
    assert "Windows outside the bev_ice classes" in t2, "unlabelled powertrain table not rendered"
    sens = [ln for ln in t2.splitlines() if ln.startswith("| full 20-ch model, yaw group masked") or ln.startswith("| full 20-ch model, lead group masked")]
    assert len(sens) == 4, f"{len(sens)} sensitivity rows (expected 2 masks x 2 tasks)"
    assert sens and all(ln.rstrip().endswith("| - |") for ln in sens), "reference column of the yaw / lead masks must show '-'"
    assert not HEX_RE.findall(t2)
    rep["dataset_tables_lines"] = len(L2)
    (out_dir / "ccs_tables_selftest.md").write_text(f"# ccs_tables self-test on `{DS}` (synthetic set tiled from the ccs_sample windows)\n\n"
                                                    f"Generated {time.strftime('%Y-%m-%d %H:%M')} by experiments/transfer/ccs_selftest.py.\n\n" + text + "\n\n"
                                                    f"## dataset_tables({DS}, kind ccs_full)\n\n" + t2 + "\n")
    (out_dir / "ccs_selftest_summary.json").write_text(json.dumps(CS._round_floats(out), indent=1, default=str))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-build", action="store_true", help="re-use the existing synthetic set")
    ap.add_argument("--skip-score", action="store_true", help="re-use the existing probability files")
    ap.add_argument("--bootstrap", type=int, default=50, help="score.py bootstrap resamples (smoke test)")
    ap.add_argument("--out", type=Path, default=T.DATA_ROOT / "tmp" / "ccs_selftest", help="folder for the rendered tables, log and the collector's test output")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"dataset": DS}
    t0 = time.time()
    if not a.skip_build:
        s = build()
        rep["build"] = dict(n_windows=s["n"], warnings=s["warnings"])
        m = pd.read_parquet(s["parquet"])
        assert m["platform"].nunique() == len(PLATFORMS) and m["vehicle_id"].nunique() == sum(p[-1] for p in PLATFORMS)
        assert (m.groupby("drive_id").size() > 1).any() and (m.groupby("vehicle_id").size().max() >= 10)
        print(f"[selftest] built {s['n']} windows, {m['vehicle_id'].nunique()} devices, {m['drive_id'].nunique()} drives, {m['platform'].nunique()} platforms", flush=True)
        s_us = build_us()
        rep["build_us"] = dict(n_windows=s_us["n"], warnings=s_us["warnings"], k=US_K)
        print(f"[selftest] built the understeer-corrected copy {DS_US}: {s_us['n']} windows (k = {US_K:g})", flush=True)
    if not a.skip_score:
        rep["score"] = score(a.out / "score.log", a.bootstrap)
        assert all(r["returncode"] == 0 for r in rep["score"]), "a score.py run failed (see score.log)"
        rep["score_us"] = score(a.out / "score.log", a.bootstrap, ds=DS_US)
        assert all(r["returncode"] == 0 for r in rep["score_us"]), f"a score.py run failed on {DS_US} (see score.log)"
    build_bad()
    rep["check"] = check(a.out)
    rep["check_us"] = check_us(a.out)
    rep["check_untouched"] = check_untouched(a.out)
    for ds in (DS_BAD, DS_BAD2):                                      # the misaligned copies served their purpose
        shutil.rmtree(T.work_dir(ds).parent, ignore_errors=True)
    assert (a.out / "collector" / "RESULTS_TABLES.md").exists() and (a.out / "collector" / "results_summary.json").exists()
    rep["collector_main"] = "ok"
    rep["seconds"] = round(time.time() - t0, 1)
    (a.out / "ccs_selftest_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[selftest] PASS: {rep['check']['n_tables']} tables, {rep['check']['strata_cells_checked']} strata cells checked; understeer pair: "
          f"{rep['check_us']['strata_cells_checked']} cells, flips {rep['check_us']['flipped_platforms']}, misaligned copies refused; collector main() ok and "
          f"unchanged before the understeer section; tables -> {a.out / 'ccs_tables_selftest.md'}, {a.out / 'yaw_correction_selftest.md'} ({rep['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
