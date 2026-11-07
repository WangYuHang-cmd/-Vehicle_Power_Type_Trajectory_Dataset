"""S1 tests: vtt.manifest helpers and scripts/s01_probe_uidmap.py end to end on the synthetic tree of tests/synth_tree.py
(two fingerprints, misfiled extra copy, non-hex folders resolved / unresolved, a 16-hex folder with another device's logs,
identical and different same-route-id copies, an rlog-only segment, a date-style id, a long route, a 0-byte log, junk dir).
Artefacts live under $VTT_DATA_ROOT/tmp and are removed at the end. No real identifier is used anywhere."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from vtt import config as C  # noqa: E402
from vtt import manifest as M  # noqa: E402
import synth_tree as T  # noqa: E402

TMP_BASE = Path(os.environ.get("VTT_DATA_ROOT", "$VTT_DATA_ROOT")) / "tmp"
HEX16 = re.compile(r"(?<![0-9a-zA-Z])[0-9a-f]{16}(?![0-9a-zA-Z])")
ROUTE_TOKEN = re.compile(r"\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}|[0-9a-f]{8}--[0-9a-f]{10}")


def load_script_main():
    spec = importlib.util.spec_from_file_location("s01_probe_uidmap", REPO / "scripts" / "s01_probe_uidmap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


def new_data_root(root: Path, name: str) -> Path:
    data = root / name
    (data / "private" / "seeds").mkdir(parents=True)
    os.chmod(data / "private", 0o700)
    (data / "private" / "seeds" / "salt.txt").write_text("f" * 64 + "\n")
    return data


def make_tree(prefix: str, workers: str = "2") -> dict:
    """Synthetic tree + a seeded data root; the first S1 run is executed here (tests may run in any order)."""
    TMP_BASE.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=prefix, dir=TMP_BASE))
    info = T.build_tree(root)
    info["root"], info["data"] = root, new_data_root(root, "droot")
    assert run(info, workers=workers) == 0
    return info


@pytest.fixture(scope="module")
def tree():
    """Read-only after its first run: shared by the structural tests."""
    info = make_tree("s1_manifest_")
    yield info
    shutil.rmtree(info["root"], ignore_errors=True)


@pytest.fixture()
def tree_mut():
    """A private tree for the test that changes the raw library between runs (probed with the sequential worker path)."""
    info = make_tree("s1_manifest_mut_", workers="1")
    yield info
    shutil.rmtree(info["root"], ignore_errors=True)


def run(info, *extra, data=None, workers="2") -> int:
    argv = ["--data-root", str(data or info["data"]), "--raw-root", str(info["raw"]), "--workers", workers, *extra]
    return load_script_main()(argv)


def ensure_first_run(info):
    assert (info["data"] / "private" / "uid_map.parquet").exists()


def outputs(data: Path) -> dict:
    priv, build = data / "private", data / "build"
    return {
        "uid_map": pd.read_parquet(priv / "uid_map.parquet"),
        "resolved": pd.read_parquet(priv / "routes_resolved.parquet"),
        "routes_all": pd.read_csv(build / "routes_all.csv", dtype=str),
        "smoke": pd.read_csv(build / "routes_smoke20.csv", dtype=str),
        "misfiled": json.loads((priv / "misfiled_dirs.json").read_text()),
        "detail": json.loads((priv / "misfiled_dirs_detail.json").read_text()),
        "merges": pd.read_csv(priv / "identity_merges.csv", dtype=str),
        "summary": json.loads((build / "logs" / "s01_summary.json").read_text()),
        "chunks": sorted(p.name for p in (build / "chunks").glob("routes_chunk_*.csv")),
    }


# ----------------------------------------------------------------------------- unit
@pytest.mark.parametrize("version, era", [
    ("0.8.13", "0.8"), ("0.9.7", "0.9"), ("0.9.9-release", "0.9"), ("0.10.0", "0.10"), ("0.11.1", "0.11"),
    ("2025.1.0", "2025"), ("2026.03.14", "2026"), ("2026.03.14-release2", "2026"),
    (None, "other"), ("", "other"), ("garbage", "other"), ("0.7.10", "other"), ("0.12.0", "other"), ("1.2.3", "other"),
])
def test_software_era(version, era):
    assert M.software_era(version) == era
    assert era in C.SOFTWARE_ERAS


def test_is_hex16_and_tri_state():
    assert M.is_hex16(T.fake_dongle(1)) and not M.is_hex16("personx") and not M.is_hex16(None) and not M.is_hex16("F" * 16)
    assert list(M.tri_state(pd.Series([True, False, None, float("nan"), 1.0], dtype=object))) == [True, False, None, None, True]


def test_scan_public_text():
    TMP_BASE.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="s1_scan_", dir=TMP_BASE))
    try:
        good, bad = d / "good.csv", d / "bad.csv"
        good.write_text("route_uid,n\n0123456789ab,3\n")
        bad.write_text(f"route_uid,x\n0123456789ab,{T.fake_dongle(7)}\nabc,{T.date_route(3)}\n")
        assert M.scan_public_text([good]) == {}
        assert M.scan_public_text([good, bad]) == {"bad.csv": 2}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_acceptance_detects_problems(tree):
    ensure_first_run(tree)
    o = outputs(tree["data"])
    build = tree["data"] / "build"
    ok = M.acceptance(o["uid_map"], o["resolved"], M.routes_all_public(o["resolved"]), M.smoke_select(o["resolved"]),
                      build / "chunks", [build / "routes_all.csv", build / "routes_smoke20.csv"])
    assert ok == []
    bad = o["uid_map"].copy()
    bad.loc[0, "dongle_log"] = "personx"
    bad.loc[1, "route_uid"] = bad.loc[2, "route_uid"]
    problems = M.acceptance(bad, o["resolved"], M.routes_all_public(o["resolved"]), M.smoke_select(o["resolved"]),
                            build / "chunks", [build / "routes_all.csv"])
    assert any("non-hex" in p for p in problems) and any("duplicated route_uid" in p for p in problems)


# ----------------------------------------------------------------------------- end to end
def test_first_run_structure(tree):
    ensure_first_run(tree)
    o = outputs(tree["data"])
    D, R = tree["D"], tree["R"]
    um, rr = o["uid_map"], o["resolved"]
    S = M.tocf("schema")
    assert list(um.columns) == list(S.PRIVATE_UID_MAP)
    assert len(um) == tree["n_uid_map"] and um.route_uid.is_unique and um.dongle_log.map(M.is_hex16).all()
    assert set(um.route_uid) == set(rr.route_uid) and len(rr) == len(um)
    assert o["summary"]["accepted"] is True and o["summary"]["n_dirs"] == tree["n_dirs"]
    assert o["summary"]["n_logless_dirs"] == tree["n_logless"] and o["summary"]["n_routes_excluded_unresolved_folder"] == 1
    by_route = rr.set_index(["dongle_log", "route_id"])
    # misfiled extra copy: 2 copies, the fingerprint-matching car dir chosen, dup_rule stays none
    r1 = by_route.loc[(D["D1"], R["R1"])]
    assert r1.n_copies == 2 and r1.car_dir == T.CAR_A and r1.dup_rule == "none" and len(json.loads(r1.copies_rel)) == 2
    assert r1.n_segments == 3 and r1.n_qlog == 3 and r1.software_era == "0.9" and r1.device_type == "tizi"
    # non-hex folder resolved through its logs; the unresolved one excluded and recorded
    r4 = by_route.loc[(D["D3"], R["R4"])]
    assert r4.dongle_dir == T.NAME_X and r4.software_era == "2026"
    assert R["R5"] not in set(rr.route_id) and R["R5b"] not in set(rr.route_id)
    assert o["misfiled"][T.NAME_X] == D["D3"] and o["misfiled"][T.NAME_Y] == T.NAME_Y and o["misfiled"][D["D8"]] == D["D9"]
    assert o["detail"][T.NAME_Y]["status"] == "unresolved" and o["detail"][D["D8"]]["kind"] == "wrong_id"
    assert by_route.loc[(D["D9"], R["R10"])].dongle_dir == D["D8"]
    # identical content under two folders without initData: one row kept (lexically first folder), copies merged
    r6 = rr[rr.route_id == R["R6"]]
    assert len(r6) == 1 and r6.iloc[0].dup_rule == "identical_copy" and r6.iloc[0].dongle_dir == D["D4"] and r6.iloc[0].n_copies == 2
    assert r6.iloc[0].software_era == "other" and r6.iloc[0].probe_status == "no_initdata"
    # different content under two dongles: both kept + one identity-merge pair
    r7 = rr[rr.route_id == R["R7"]]
    assert len(r7) == 2 and set(r7.dup_rule) == {"shared_route_id"} and set(r7.dongle_log) == {D["D6"], D["D7"]}
    assert len(o["merges"]) == 1 and tuple(o["merges"].iloc[0][["dongle_log_a", "dongle_log_b", "route_id", "evidence"]]) == (D["D6"], D["D7"], R["R7"], "counter_id")
    # rlog-only segment, date-style id, long route
    r3 = by_route.loc[(D["D2"], R["R3"])]
    assert r3.n_segments == 3 and r3.n_qlog == 2 and r3.n_rlog == 1 and r3.n_seg_rlog_only == 1 and r3.software_era == "0.10"
    assert (rr.n_seg_rlog_only > 0).sum() == 1 and o["summary"]["n_rlog_only_routes"] == 1
    assert by_route.loc[(D["D1"], R["R2"])].route_style == "date" and by_route.loc[(D["D1"], R["R2"])].software_era == "0.8"
    r8 = by_route.loc[(D["D1"], R["R8"])]
    assert bool(r8.is_long) and r8.n_segments == tree["long_n"]
    assert set(rr.software_era) == {"0.8", "0.9", "0.10", "0.11", "2026", "other"}
    assert set(o["summary"]["dup_rules"]) == {"none", "identical_copy", "shared_route_id"}


def test_public_files(tree):
    ensure_first_run(tree)
    o = outputs(tree["data"])
    build = tree["data"] / "build"
    ra, smoke = o["routes_all"], o["smoke"]
    assert list(ra.columns) == M.ROUTES_ALL_COLS and set(ra.route_uid) == set(o["uid_map"].route_uid)
    assert set(ra.device_type) >= {"tizi", "tici", "mici", "unknown"} and set(ra.op_long.dropna()) == {"False"}
    assert (ra.is_long == "True").sum() == 1
    # chunks: one short chunk + the long file, disjoint cover of uid_map
    assert o["chunks"] == ["routes_chunk_00.csv", "routes_chunk_long.csv"]
    short = pd.read_csv(build / "chunks" / "routes_chunk_00.csv", dtype=str)
    long_ = pd.read_csv(build / "chunks" / "routes_chunk_long.csv", dtype=str)
    assert list(short.columns) == ["route_uid"] and len(short) == tree["n_uid_map"] - 1 and len(long_) == 1
    assert set(short.route_uid) | set(long_.route_uid) == set(o["uid_map"].route_uid)
    assert long_.route_uid.iloc[0] == o["resolved"][o["resolved"].route_id == tree["R"]["R8"]].route_uid.iloc[0]
    # smoke: <= 20, stratified, every special stratum present
    assert len(smoke) <= 20 and smoke.route_uid.is_unique and set(smoke.route_uid) <= set(ra.route_uid)
    assert list(smoke.columns) == M.ROUTES_ALL_COLS + ["stratum"]
    assert set(M.SMOKE_SPECIAL) <= set(smoke.stratum)
    rr = o["resolved"].set_index("route_uid")
    s = smoke.set_index("stratum").route_uid
    assert not M.is_hex16(rr.loc[s["nonhex_folder"]].dongle_dir)
    assert rr.loc[s["misfiled_dir"]].dongle_dir != rr.loc[s["misfiled_dir"]].dongle_log and M.is_hex16(rr.loc[s["misfiled_dir"]].dongle_dir)
    assert rr.loc[s["date_style"]].route_style == "date" and rr.loc[s["rlog_only"]].n_seg_rlog_only > 0
    assert rr.loc[s["multi_copy"]].n_copies > 1 and rr.loc[s["dup_rule"]].dup_rule != "none" and bool(rr.loc[s["long"]].is_long)
    assert any(x.startswith("era_") for x in smoke.stratum)
    # no identifier-like token in any public csv (fake dongle ids, route ids, folder names)
    for p in list(build.glob("*.csv")) + list((build / "chunks").glob("*.csv")):
        text = p.read_text()
        assert not HEX16.search(text) and not ROUTE_TOKEN.search(text), p.name
        assert not any(tok in text for tok in tree["all_ids"]), p.name
    # private files are mode 600 under a 700 root
    priv = tree["data"] / "private"
    assert stat.S_IMODE(priv.stat().st_mode) == 0o700
    for f in ("uid_map.parquet", "routes_resolved.parquet", "raw_manifest.csv", "route_probe.parquet", "misfiled_dirs.json", "identity_merges.csv"):
        assert stat.S_IMODE((priv / f).stat().st_mode) == 0o600, f
    man = pd.read_csv(priv / "raw_manifest.csv", dtype=str)
    assert len(man) == tree["n_dirs"]


def test_rerun_is_idempotent(tree):
    ensure_first_run(tree)
    before = outputs(tree["data"])
    assert run(tree, workers="1") == 0
    after = outputs(tree["data"])
    pd.testing.assert_frame_equal(before["uid_map"], after["uid_map"])
    pd.testing.assert_frame_equal(before["routes_all"], after["routes_all"])
    pd.testing.assert_frame_equal(before["smoke"], after["smoke"])
    assert after["summary"]["n_probed_now"] == 0 and after["summary"]["n_identity_merges_new"] == 0
    assert len(after["merges"]) == 1 and after["chunks"] == before["chunks"]


def test_dry_run_writes_no_outputs(tree):
    data = new_data_root(tree["root"], "droot_dry")
    assert run(tree, "--dry-run", data=data) == 0
    assert not (data / "private" / "uid_map.parquet").exists() and not (data / "build").exists()
    assert not (data / "private" / "route_probe.parquet").exists()


def test_stale_reprobe_and_learned_rekey(tree_mut):
    """A log appears in the formerly empty route of the unresolved folder: the stale 'no_logs' row is probed again, the
    folder learns its id and its initData-less route is re-keyed through the folder table."""
    tree = tree_mut
    D, R = tree["D"], tree["R"]
    before = outputs(tree["data"])
    assert before["summary"]["n_routes_excluded_unresolved_folder"] == 1 and before["misfiled"][T.NAME_Y] == T.NAME_Y
    rd = tree["raw"] / T.CAR_B / T.NAME_Y / R["R5b"]
    (rd / "0--qlog").write_bytes(T.qlog_bytes(0, D["D3"], T.CAR_B, "0.9.7", "tizi"))
    assert run(tree) == 0
    o = outputs(tree["data"])
    assert o["summary"]["n_stale_reprobed"] == 1 and o["summary"]["n_probed_now"] == 1
    assert len(o["uid_map"]) == tree["n_uid_map"] + 2 and o["misfiled"][T.NAME_Y] == D["D3"]
    rr = o["resolved"].set_index("route_id")
    assert rr.loc[R["R5"]].dongle_log == D["D3"] and rr.loc[R["R5b"]].dongle_log == D["D3"]
    assert o["summary"]["n_routes_excluded_unresolved_folder"] == 0 and o["summary"]["accepted"] is True
    # --no-reprobe-stale keeps the cache as it is (nothing new to probe now anyway)
    assert run(tree, "--no-reprobe-stale") == 0 and outputs(tree["data"])["summary"]["n_probed_now"] == 0
