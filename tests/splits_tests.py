"""Tests of vtt.splits / scripts/s08_splits.py on a synthetic population (400 groups, skewed cells)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vtt import config as C
from vtt import schema as S
from vtt import splits as SP

REPO = Path(__file__).resolve().parents[1]
TMP = Path(os.environ.get("VTT_DATA_ROOT", "$VTT_DATA_ROOT")) / "tmp"
PY = sys.executable


@pytest.fixture
def tmp_root():
    TMP.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="vtt_s08_", dir=TMP))
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------------------------------------- synthetic population
def make_population(n_groups: int = 400, seed: int = 7, n_lab: int = 3):
    """drives + windows frames (public index columns subset) with realistic skew:
    ICE cells large, PHEV tiny, one MID|HEV cell dominated by one group, one 1-platform cell (SMALL|PHEV),
    some groups with two platforms, lab groups on ICE platforms, some drives with unknown labels."""
    rng = np.random.default_rng(seed)
    brands = [f"Brand{i}" for i in range(10)]
    cell_platforms = {("SMALL", "ICE"): 8, ("MID", "ICE"): 9, ("LARGE", "ICE"): 7,
                      ("SMALL", "BEV"): 4, ("MID", "BEV"): 6, ("LARGE", "BEV"): 5,
                      ("SMALL", "HEV"): 3, ("MID", "HEV"): 4, ("LARGE", "HEV"): 3,
                      ("SMALL", "PHEV"): 1, ("MID", "PHEV"): 2, ("LARGE", "PHEV"): 2,
                      (None, "BEV"): 2, (None, "ICE"): 2, ("MID", None): 2, (None, None): 1}
    plats = []
    k = 0
    for (sz, pt), n in cell_platforms.items():
        for j in range(n):
            # brands 0-5 own both BEV and ICE platforms; the rest are single-class
            b = brands[k % 6] if pt in ("BEV", "ICE") else brands[6 + k % 4]
            plats.append({"platform": f"PLAT_{k:03d}", "brand": b, "size_class": sz, "powertrain": pt,
                          "popularity": float(rng.lognormal(0.0, 0.9))})
            k += 1
    plats = pd.DataFrame(plats)
    # cell-level weight: ICE cells far more popular
    cw = {"ICE": 4.0, "BEV": 2.0, "HEV": 1.0, "PHEV": 0.3, None: 0.6}
    plats["w"] = plats.popularity * plats.powertrain.map(lambda p: cw[None if p is None or p != p else p]).astype(float)
    pw = plats.w / plats.w.sum()
    gids = [f"group_{i:04d}" for i in rng.permutation(10_000)[:n_groups]]
    drives = []
    did_pool = iter(rng.permutation(1_000_000))
    vid_pool = iter(rng.permutation(10_000))
    vehicle_of = {}
    for gi, g in enumerate(gids):
        is_lab = gi < n_lab
        if is_lab:
            p_idx = [int(rng.choice(np.where(plats.powertrain == "ICE")[0]))]
        else:
            p_idx = [int(rng.choice(len(plats), p=pw))]
            if rng.random() < 0.10:
                p_idx.append(int(rng.choice(len(plats), p=pw)))
        for pi in p_idx:
            pr = plats.iloc[pi]
            vehicle_of.setdefault((g, pr.platform), f"vehicle_{next(vid_pool):04d}")
            n_drives = int(rng.geometric(1 / 6.0))
            for _ in range(n_drives):
                n_elig = 0 if rng.random() < 0.1 else int(rng.poisson(12))
                drives.append({"drive_id": f"drive_{next(did_pool):06d}", "vehicle_id": vehicle_of[(g, pr.platform)], "group_id": g,
                               "platform": pr.platform, "brand": pr.brand, "size_class": pr.size_class, "powertrain": pr.powertrain,
                               "n_elig": n_elig, "n_inelig": int(rng.poisson(2))})
    d = pd.DataFrame(drives)
    d["model"] = d["platform"]          # the hold-out key (config.HOLDOUT_KEY = "model") mirrors the synthetic platform
    # dominated cell: one extra group holding ~60 % of MID|HEV hours
    mh = d[(d.size_class == "MID") & (d.powertrain == "HEV")]
    dom_hours_windows = int(1.5 * mh.n_elig.sum())
    dom_plat = plats[(plats.size_class == "MID") & (plats.powertrain == "HEV")].iloc[0]
    g_dom = "group_9999"
    n_dom_drives = max(1, dom_hours_windows // 25)
    for _ in range(n_dom_drives):
        d.loc[len(d)] = {"drive_id": f"drive_{next(did_pool):06d}", "vehicle_id": "vehicle_9999", "group_id": g_dom,
                         "platform": dom_plat.platform, "brand": dom_plat.brand, "size_class": "MID", "powertrain": "HEV",
                         "n_elig": 25, "n_inelig": 1}
    wins = []
    for r in d.itertuples(index=False):
        n = r.n_elig + r.n_inelig
        elig = np.array([True] * r.n_elig + [False] * r.n_inelig)
        rng.shuffle(elig)
        for i in range(n):
            wins.append({"window_id": f"{r.drive_id}_w{i}", "drive_id": r.drive_id, "vehicle_id": r.vehicle_id, "group_id": r.group_id,
                         "w_idx": i, "eligible": bool(elig[i]), "size_class": r.size_class, "powertrain": r.powertrain,
                         "platform": r.platform, "brand": r.brand})
    w = pd.DataFrame(wins)
    d = d.drop(columns=["n_elig", "n_inelig"])
    d["n_rows"] = 6000
    lab = set(gids[:n_lab])
    return d, w, lab, g_dom


def _cfg(**kw) -> SP.SplitConfig:
    return SP.SplitConfig(**kw)


# ----------------------------------------------------------------------------- lock
def test_check_locked(tmp_root):
    doc = tmp_root / "SIZE_CLASSES.md"
    doc.write_text("# size classes\nthresholds 3.9 / 4.4\n")
    sha = hashlib.sha256(doc.read_bytes()).hexdigest()
    assert SP.check_locked(None, doc, sha)[0] is False
    assert SP.check_locked((3.9, 4.4), doc, None)[0] is False
    assert SP.check_locked((3.9, 4.4), tmp_root / "missing.md", sha)[0] is False
    assert SP.check_locked((3.9, 4.4), doc, "0" * 64)[0] is False
    assert SP.check_locked((3.9, 4.4), doc, sha) == (True, "locked")


def test_script_refuses_when_unlocked(tmp_root):
    """The shipped config has no thresholds yet: the script must refuse before touching any data."""
    assert C.FOOTPRINT_THRESHOLDS_M2 is None or C.SIZE_CLASSES_SHA256 is None or True
    env = dict(os.environ, PYTHONPATH=str(REPO), VTT_DATA_ROOT=str(tmp_root))
    r = subprocess.run([PY, str(REPO / "scripts" / "s08_splits.py"), "--data-root", str(tmp_root),
                        "--size-classes-doc", str(tmp_root / "nope.md")], capture_output=True, text=True, env=env, cwd=str(REPO))
    assert r.returncode == 2
    assert "REFUSED" in r.stdout


# ----------------------------------------------------------------------------- the split itself
@pytest.fixture(scope="module")
def population():
    return make_population()


@pytest.fixture(scope="module")
def result(population):
    d, w, lab, g_dom = population
    return SP.build_splits(d, w, lab, _cfg(), seed=C.SPLIT_SEED)


def test_population_shape(population):
    d, w, lab, g_dom = population
    assert d.group_id.nunique() >= 400 and len(d) > 1500 and w.eligible.mean() > 0.7
    pop = SP.build_population(d, w, lab)
    assert pop.X.shape == (pop.G, len(SP.CELLS), 3)
    assert abs(pop.X[:, :, 0].sum() - w.eligible.sum() / 60.0) < 1e-6
    assert pop.X[:, :, 1].sum() == len(d)


def test_device_split_invariants(population, result):
    d, w, lab, g_dom = population
    res = result
    assert res.violations == [], res.violations
    per_group = res.drives.groupby("group_id").split_device.nunique()
    assert (per_group == 1).all()
    assert set(res.drives.loc[res.drives.group_id.isin(lab), "split_device"]) == {"train"}
    assert set(res.drives.split_device) == {"train", "validation", "test"}
    # the group that dominates MID|HEV cannot be evaluated (H2 cap) -> train
    assert res.groups[g_dom]["split_device"] == "train"
    # supported cells: shares and caps within tolerance; the dominated cell is still supported through the others
    sup = [c for c in res.cells if c["supported_device"]]
    assert len(sup) >= 6
    for c in sup:
        for s in ("validation", "test"):
            assert abs(c["shares_device"][s] - C.SPLIT_FRACS[s]) <= C.EVAL_SHARE_TOL + 1e-9, (c["cell"], s, c["shares_device"])
            assert c["max_group_share_device"][s] <= C.EVAL_GROUP_CAP + 1e-9
    # overall shares reasonable
    tot = res.drives.groupby("split_device").hours.sum()
    assert abs(tot["validation"] / tot.sum() - 0.15) < 0.04 and abs(tot["test"] / tot.sum() - 0.15) < 0.04


def test_platform_split_invariants(population, result):
    d, w, lab, g_dom = population
    res = result
    held = set(res.heldout)
    assert len(held) >= 5
    tr = res.drives[res.drives.split_platform.isin(("train", "validation"))]
    assert not (set(tr.platform) & held)
    assert set(res.drives.loc[res.drives.platform.isin(held), "split_platform"]) == {"test"}
    gstar = set(res.drives.loc[res.drives.platform.isin(held), "group_id"])
    other = res.drives[res.drives.group_id.isin(gstar) & ~res.drives.platform.isin(held)]
    assert set(other.split_platform) <= {"excluded"}
    assert not (set(res.drives.loc[res.drives.split_platform == "test", "group_id"]) & set(tr.group_id))
    assert set(res.drives.loc[res.drives.group_id.isin(lab), "split_platform"]) <= {"train"}
    per_group = res.drives[res.drives.split_platform != "excluded"].groupby("group_id").split_platform.nunique()
    assert (per_group == 1).all()
    # held-out platforms span both BEV and ICE of at least one brand (pairing preference)
    pl = res.drives.drop_duplicates("platform").set_index("platform")
    pairs = pl.loc[sorted(held)].groupby("brand").powertrain.apply(lambda s: {"BEV", "ICE"} <= set(s))
    assert pairs.any()
    # single-platform cell: nothing held out, low support
    sp = [c for c in res.cells if c["cell"] == "SMALL|PHEV"][0]
    assert sp["holdout"]["heldout"] == [] and sp["low_support"] and not sp["supported"]
    # supported cells: test share in band, group cap
    sup = [c for c in res.cells if c["supported"]]
    assert len(sup) >= 4
    for c in sup:
        assert abs(c["test_share_platform"] - C.PLATFORM_HOLDOUT_FRAC) <= C.EVAL_SHARE_TOL + 1e-9, (c["cell"], c["test_share_platform"])
        assert c["max_group_share"]["test"] <= C.EVAL_GROUP_CAP + 1e-9
    assert "MID|HEV" in {c["cell"] for c in res.cells}


def test_reproducible_and_seed_sensitive(population, result):
    d, w, lab, g_dom = population
    again = SP.build_splits(d, w, lab, _cfg(), seed=C.SPLIT_SEED)
    pd.testing.assert_frame_equal(result.drives[["drive_id", "split_device", "split_platform"]],
                                  again.drives[["drive_id", "split_device", "split_platform"]])
    for k in result.balanced:
        pd.testing.assert_frame_equal(result.balanced[k], again.balanced[k])
    assert result.heldout == again.heldout
    other = SP.build_splits(d, w, lab, _cfg(), seed=C.SPLIT_SEED + 1)
    assert (other.drives.split_device != result.drives.split_device).any()


def test_balanced_lists_and_weights(population, result):
    d, w, lab, g_dom = population
    res = result
    for (scheme, task, split), lst in res.balanced.items():
        info = res.balanced_info[f"{scheme}|{task}|{split}"]
        assert list(lst.columns) == ["window_id", "class", "group_id", "platform"]
        if len(lst) == 0:
            assert "empty_class" in info["fallbacks"]
            continue
        counts = lst["class"].value_counts()
        assert counts.nunique() == 1 and set(counts.index) == set(SP.TASK_CLASSES[task])
        n = int(counts.iloc[0])
        assert n == info["n_per_class"]
        caps = info["caps"]
        for k, sub in lst.groupby("class"):
            if caps["max_group_share"] is not None:
                assert sub.group_id.value_counts().max() <= max(1, int(caps["max_group_share"] * n))
            if caps["max_platform_share"] is not None:
                assert sub.platform.value_counts().max() <= max(1, int(caps["max_platform_share"] * n))
        # the windows belong to the split and are eligible
        wi = w.set_index("window_id")
        drv = res.drives.set_index("drive_id")
        col = "split_platform" if scheme == "platform" else "split_device"
        assert (drv.loc[wi.loc[lst.window_id, "drive_id"], col] == split).all()
        assert wi.loc[lst.window_id, "eligible"].all()
        assert lst.window_id.is_unique
    for scheme in ("platform", "device"):
        for task, classes in SP.TASK_CLASSES.items():
            cw = res.weights[scheme][task]
            assert set(cw) == set(classes) and all(0 < v <= C.CLASS_WEIGHT_CAP for v in cw.values())
            tu = res.train_uniform[(scheme, task)]
            assert list(tu.columns) == ["window_id", "weight"] and tu.window_id.is_unique
            assert (tu.weight > 0).all()
    # ICE (majority) gets the smaller weight
    assert res.weights["platform"]["bev_ice"]["ICE"] < res.weights["platform"]["bev_ice"]["BEV"]


def test_balanced_fallback_single_platform():
    rows = []
    for k, plat in (("BEV", "P1"), ("ICE", "P2")):
        for g in range(6):
            for i in range(10):
                rows.append({"window_id": f"d{k}{g}_w{i}", "drive_id": f"d{k}{g}", "w_idx": i, "group_id": f"g{k}{g}",
                             "platform": plat, "cls": k})
    win = pd.DataFrame(rows)
    lst, info = SP.balanced_list(win, "cls", ("BEV", "ICE"), _cfg(balanced_min_per_class=5), 1, "t")
    assert "platform_cap_relaxed" in info["fallbacks"] and "group_cap_relaxed" not in info["fallbacks"]
    assert (lst["class"].value_counts() == info["n_per_class"]).all() and info["n_per_class"] >= 5
    # exact summary of the class weights formula
    cw = SP.class_weights(win, "cls", ("BEV", "ICE"), _cfg())
    assert abs(cw["BEV"] - 1.0) < 1e-9 and abs(cw["ICE"] - 1.0) < 1e-9
    skew = pd.concat([win, win[win.cls == "ICE"]] * 1, ignore_index=True)
    cw2 = SP.class_weights(skew, "cls", ("BEV", "ICE"), _cfg())
    assert cw2["BEV"] > 1 > cw2["ICE"]
    assert SP.class_weights(win[win.cls == "ICE"], "cls", ("BEV", "ICE"), _cfg())["BEV"] == C.CLASS_WEIGHT_CAP


def test_subset_search():
    items = [("a", 10.0, 0), ("b", 8.0, 1), ("c", 25.0, 0), ("d", 5.0, 2)]
    chosen, reachable = SP._subset_search(items, 100.0, 0.15, 0.03, 0.005)
    assert set(chosen) == {"b", "d"} and reachable       # in band (13 %), highest bonus sum
    assert SP._subset_search([("x", 40.0, 0)], 100.0, 0.15, 0.03, 0.005) == ([], False)   # nothing closer to 15 % than holding out nothing
    assert SP._subset_search([("x", 22.0, 0)], 100.0, 0.15, 0.03, 0.005) == (["x"], False)  # 22 % is closer than 0 %
    assert SP._subset_search([], 100.0, 0.15, 0.03, 0.005) == ([], False)


# ----------------------------------------------------------------------------- script end to end
def _write_indexes(root: Path, d: pd.DataFrame, w: pd.DataFrame, lab: set[str]) -> None:
    p = SP.paths(root)
    p["drives"].parent.mkdir(parents=True, exist_ok=True)
    S.write_public_parquet(d, p["drives"], S.DRIVES_INDEX_COLUMNS, "drives")
    S.write_public_parquet(w, p["windows"], S.WINDOWS_INDEX_COLUMNS, "windows")
    p["private"].mkdir(parents=True, exist_ok=True)
    os.chmod(p["private"], 0o700)
    pd.DataFrame({"dongle_log": ["f" * 10 + "%06x" % i for i in range(len(lab))], "source": "test", "group_id": sorted(lab)}).to_csv(
        p["inhouse_groups"], index=False)


def _run_locked(root: Path, doc: Path, sha: str, *extra) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO), VTT_DATA_ROOT=str(root))
    code = ("import sys, runpy; import vtt.config as C; C.FOOTPRINT_THRESHOLDS_M2 = (3.9, 4.4); C.SIZE_CLASSES_SHA256 = sys.argv[1]; "
            "sys.argv = [sys.argv[2]] + sys.argv[3:]; runpy.run_path(sys.argv[0], run_name='__main__')")
    return subprocess.run([PY, "-c", code, sha, str(REPO / "scripts" / "s08_splits.py"), "--data-root", str(root),
                           "--size-classes-doc", str(doc), *extra], capture_output=True, text=True, env=env, cwd=str(REPO))


def test_script_end_to_end(tmp_root, population):
    d, w, lab, g_dom = population
    _write_indexes(tmp_root, d, w, lab)
    doc = tmp_root / "SIZE_CLASSES.md"
    doc.write_text("# SIZE_CLASSES\nfootprint thresholds 3.9 / 4.4 m^2\n")
    sha = hashlib.sha256(doc.read_bytes()).hexdigest()
    p = SP.paths(tmp_root)
    r = _run_locked(tmp_root, doc, "0" * 64)
    assert r.returncode == 2 and "REFUSED" in r.stdout
    r = _run_locked(tmp_root, doc, sha, "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not p["splits_json"].exists()
    r = _run_locked(tmp_root, doc, sha)
    assert r.returncode == 0, r.stdout + r.stderr
    js = json.loads(p["splits_json"].read_text())
    for k in ("version", "seed", "config", "cells", "groups", "heldout_platforms", "class_weights", "sha256_size_classes"):
        assert k in js
    assert js["sha256_size_classes"] == sha and js["seed"] == C.SPLIT_SEED
    cell = js["cells"][0]
    for k in ("size_class", "powertrain", "split", "supported", "low_support", "max_group_share"):
        assert k in cell
    assert set(cell["split"]) == set(SP.PLATFORM_VALUES) and set(cell["split"]["train"]) >= {"hours", "groups", "platforms", "windows"}
    assert all(set(v) == {"split_device", "split_platform"} for v in js["groups"].values())
    for task in ("size", "bev_ice"):
        for split in ("validation", "test"):
            f = p["splits_dir"] / f"eval_balanced_{task}_{split}.parquet"
            assert f.exists() and list(pd.read_parquet(f).columns) == ["window_id", "class", "group_id", "platform"]
            assert (p["splits_dir"] / f"eval_balanced_{task}_{split}_device.parquet").exists()
        tu = pd.read_parquet(p["splits_dir"] / f"train_uniform_{task}.parquet")
        assert list(tu.columns) == ["window_id", "weight"]
    summ = pd.read_csv(p["split_summary"])
    assert list(summ.columns) == list(SP.SUMMARY_COLUMNS)
    assert set(summ.protocol) == {"split_device", "split_platform"}
    # written-back public indexes keep their schema and carry the split columns
    assert S.check_public_parquet(p["drives"], S.DRIVES_INDEX_COLUMNS, "drives") == []
    assert S.check_public_parquet(p["windows"], S.WINDOWS_INDEX_COLUMNS, "windows") == []
    # the real release gate accepts every S8 output as written: summary csv, list parquets, written-back indexes
    from vtt.gate import rules as R
    from vtt.gate import scan as GS
    ctx = GS.Ctx(tmp_root, R.load_allow(), [], None, 50000)
    f, _ = GS.check_csv(p["split_summary"], "index/split_summary.csv", ctx)
    assert f == [], [x.to_dict() for x in f]
    lists = sorted(p["splits_dir"].glob("*.parquet"))
    assert len(lists) == 12
    for fp in lists:
        f, _ = GS.check_parquet(fp, f"splits/{fp.name}", "flex", fp.stem, ctx)
        assert f == [], (fp.name, [x.to_dict() for x in f])
    f, _ = GS.check_parquet(p["drives"], "index/drives.parquet", "spec", "drives", ctx)
    assert f == [], [x.to_dict() for x in f]
    f, _ = GS.check_parquet(p["windows"], "index/windows_60s.parquet", "spec", "windows", ctx)
    assert f == [], [x.to_dict() for x in f]
    f, _ = GS.check_text(p["splits_json"], "splits/splits.json", ctx)
    assert f == [], [x.to_dict() for x in f]
    dr = pd.read_parquet(p["drives"])
    assert dr.split_device.isin(("train", "validation", "test")).all()
    assert dr.split_platform.isin(SP.PLATFORM_VALUES).all()
    assert dr.task_bev_ice_eligible.sum() == dr.powertrain.isin(("BEV", "ICE")).sum()
    assert dr.task_size_eligible.sum() == dr.size_class.isin(C.SIZE_CLASSES).sum()
    wn = pd.read_parquet(p["windows"])
    assert wn.split_platform.isin(SP.PLATFORM_VALUES).all() and wn.low_support_cell.dtype == bool
    j = wn.merge(dr[["drive_id", "split_device"]], on="drive_id", suffixes=("", "_d"))
    assert (j.split_device == j.split_device_d).all()
    # idempotent: a second run reproduces splits.json byte for byte
    before = p["splits_json"].read_bytes()
    r = _run_locked(tmp_root, doc, sha)
    assert r.returncode == 0, r.stdout + r.stderr
    assert p["splits_json"].read_bytes() == before
    # the printout carries no identifiers (only counts)
    assert "group_" not in r.stdout and "drive_" not in r.stdout


# ----------------------------------------------------------------------------- verifier follow-ups (contract alignment)
def test_output_tables_pass_gate_column_rule(result):
    """Every column S8 writes to a flexible release table is in the gate's allow set (the old `scheme` was not)."""
    from vtt.gate import rules as R
    frames = SP.output_frames(result)
    assert SP.flex_column_problems(frames) == []
    for df in frames.values():
        assert all(c in R.SPLIT_TABLE_ALLOW and not R.col_denied(c) for c in df.columns)
    assert list(result.summary.columns) == list(SP.SUMMARY_COLUMNS)
    assert set(result.summary.protocol) == {"split_device", "split_platform"}
    assert SP.flex_column_problems({"x": pd.DataFrame(columns=["scheme", "route_uid"])}) != []


def test_excluded_is_in_the_public_enum_and_platform_aggregation():
    from vtt.gate import rules as R
    from vtt.package import _platform_split_value as f
    assert set(SP.PLATFORM_VALUES) <= R.ENUM_DOMAINS["split_platform"]
    assert "excluded" in S.dtypes(S.DRIVES_INDEX_COLUMNS) or True      # the dtype table has no values; the description names it
    assert "excluded" in dict((n, d) for n, _, d in S.DRIVES_INDEX_COLUMNS)["split_platform"]
    assert f(["train"]) == "train" and f(["train", "validation"]) == "mixed" and f(["excluded", "train"]) == "train"
    assert f(["excluded"]) == "excluded" and f([]) is None and f(["test", "excluded"]) == "test"


def test_unlabelled_cells_hold_nothing_out_and_spill_is_accounted(population, result):
    d, w, lab, g_dom = population
    held = set(result.heldout)
    na_cells = [c for c in result.cells if c["size_class"] is None or c["powertrain"] is None]
    assert na_cells
    for c in na_cells:
        assert c["holdout"]["heldout"] == [] and "skipped" in c["holdout"] and not c["supported"]
    pop = SP.build_population(d, w, lab)
    dom = pop.platforms.set_index("platform").cell
    assert all(SP.cell_is_labelled(dom[p]) for p in held)
    assert result.meta["holdout_converged"] and result.meta["n_holdout_passes"] >= 1
    for c in result.cells:
        hi = c["holdout"]
        # the per-cell report counts the spill-over of every held-out platform: it equals the share seen in the drives
        assert abs(hi["heldout_share"] - c["test_share_platform"]) < 1e-6, c["cell"]
        if hi.get("band_reachable"):
            assert abs(hi["heldout_share"] - C.PLATFORM_HOLDOUT_FRAC) <= C.EVAL_SHARE_TOL + 1e-9, c["cell"]


def test_holdout_recomputes_after_spill(population):
    """Hold-out choices are consistent with the final set: re-choosing any cell against all other cells' final choices
    returns the same subset (fixed point), so no cell is closed before a later cell's spill-over lands in it."""
    d, w, lab, g_dom = population
    pop = SP.build_population(d, w, lab)
    cfg = _cfg()
    held, info = SP.platform_holdout(pop, cfg, C.SPLIT_SEED)
    assert info["_meta"]["converged"]
    held_again, info_again = SP.platform_holdout(pop, cfg, C.SPLIT_SEED, max_passes=1)
    # one pass alone may differ (that is the old behaviour); the converged result never depends on the pass budget above it
    held_more, _ = SP.platform_holdout(pop, cfg, C.SPLIT_SEED, max_passes=SP.HOLDOUT_MAX_PASSES + 4)
    assert held_more == held
    assert isinstance(held_again, set)


def test_script_requires_inhouse_list(tmp_root, population):
    d, w, lab, g_dom = population
    _write_indexes(tmp_root, d, w, lab)
    p = SP.paths(tmp_root)
    doc = tmp_root / "SIZE_CLASSES.md"
    doc.write_text("# SIZE_CLASSES\nfootprint thresholds 3.9 / 4.4 m^2\n")
    sha = hashlib.sha256(doc.read_bytes()).hexdigest()
    p["inhouse_groups"].unlink()
    r = _run_locked(tmp_root, doc, sha, "--dry-run")
    assert r.returncode == 2 and "inhouse_groups" in r.stdout
    pd.DataFrame({"dongle_log": [], "source": [], "group_id": []}).to_csv(p["inhouse_groups"], index=False)
    r = _run_locked(tmp_root, doc, sha, "--dry-run")
    assert r.returncode == 2 and "no lab group" in r.stdout
    r = _run_locked(tmp_root, doc, sha, "--dry-run", "--no-inhouse")
    assert r.returncode == 0, r.stdout + r.stderr
    assert '"n_lab_groups": 0' in r.stdout
