"""Tests of vtt.pseudonyms / scripts/s04_pseudonyms.py on a synthetic population (all ids generated here)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vtt import config as C
from vtt import pseudonyms as PS

REPO = Path(__file__).resolve().parents[1]
TMP = Path(os.environ.get("VTT_DATA_ROOT", "$VTT_DATA_ROOT")) / "tmp"
PY = sys.executable


def dongle(i: int) -> str:
    return "f" * 10 + "%06x" % i


def counter_route(i: int) -> str:
    return "%08x--%010x" % (i, i * 7919)


def date_route(i: int) -> str:
    return "2023-01-%02d--%02d-%02d-%02d" % (1 + i % 28, i % 24, (i * 7) % 60, (i * 13) % 60)


def vin(i: int) -> str:
    return hashlib.sha256(f"fake-vin-{i}".encode()).hexdigest()


def ruid(d: str, r: str) -> str:
    return hashlib.sha1(f"test|{d}|{r}".encode()).hexdigest()[:12]


@pytest.fixture
def tmp_root():
    TMP.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="vtt_s04_", dir=TMP))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def make_routes(n_devices: int = 40, routes_per_device: int = 5, seed: int = 0) -> pd.DataFrame:
    """Synthetic routes_resolved: device i has routes; VINs: devices 0-1 share vin(100) (merge), devices 2..7 share vin(200)
    (placeholder, 6 devices), device 8 and 9 share a counter route id (merge), devices 10 and 11 share a date-style id
    in different car dirs (no merge), device 12 lives in a person-name folder."""
    rows = []
    for i in range(n_devices):
        d = dongle(i)
        for k in range(routes_per_device):
            rid = counter_route(i * 100 + k)
            car = "CAR_A" if i % 2 == 0 else "CAR_B"
            vh = ""
            if i in (0, 1):
                vh = vin(100)
            elif 2 <= i <= 7:
                vh = vin(200)
            else:
                vh = vin(1000 + i)
            dd = d
            dl = d
            if i == 12:
                dd = "SomePerson"
            rows.append(dict(route_uid=ruid(d, rid), car_dir=car, dongle_dir=dd, dongle_log=dl, route_id=rid, identity_uid="",
                             vin_hash=vh, rel_path=f"{car}/{dd}/{rid}", copies_rel=json.dumps([f"{car}/{dd}/{rid}"]),
                             n_copies=1, n_segments=10, n_qlog=10, n_rlog=0, version="0.9.7", device_type="tici",
                             car_fp=f"FP_{i % 6}", opLong=False, radarUnavailable=False, pcmCruise=True, probe_status="ok",
                             dup_rule="none", is_long=False))
    # shared counter route id across devices 8 and 9
    shared = counter_route(999_999)
    for i in (8, 9):
        d = dongle(i)
        rows.append(dict(rows[-1], route_uid=ruid(d, shared), dongle_dir=d, dongle_log=d, route_id=shared, vin_hash=vin(1000 + i),
                         rel_path=f"CAR_A/{d}/{shared}", copies_rel=json.dumps([f"CAR_A/{d}/{shared}"]), car_dir="CAR_A",
                         car_fp=f"FP_{i % 6}"))
    # shared date-style id in different car dirs (10 -> CAR_A, 11 -> CAR_B): no merge
    shared_d = date_route(5)
    for i, car in ((10, "CAR_A"), (11, "CAR_B")):
        d = dongle(i)
        rows.append(dict(rows[-1], route_uid=ruid(d, shared_d), dongle_dir=d, dongle_log=d, route_id=shared_d, vin_hash=vin(1000 + i),
                         rel_path=f"{car}/{d}/{shared_d}", copies_rel=json.dumps([f"{car}/{d}/{shared_d}"]), car_dir=car,
                         car_fp=f"FP_{i % 6}"))
    df = pd.DataFrame(rows)
    assert not df.route_uid.duplicated().any()
    return df


def make_labels(routes: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"route_uid": routes.route_uid, "platform": routes.car_fp.map(lambda f: f"PLATFORM_{f[-1]}")})


# ----------------------------------------------------------------------------- clusters
def test_build_groups_rules():
    routes = make_routes()
    misfiled = {"SomePerson": dongle(12)}
    r = PS.prepare_routes(routes, misfiled)
    assert (r.loc[r.dongle_dir == "SomePerson", "_device"] == dongle(12)).all()
    lab = {dongle(30), dongle(31), dongle(12)}
    gk, st = PS.build_groups(r, merges=[(dongle(20), dongle(21))], inhouse=lab, misfiled=misfiled, placeholder_max=5)
    g = dict(zip(r.route_uid, gk))
    first = lambda i: g[ruid(dongle(i), counter_route(i * 100))]  # noqa: E731
    assert first(0) == first(1)                       # shared vin
    assert len({first(i) for i in range(2, 8)}) == 6  # placeholder vin ignored
    assert st["n_placeholders_ignored"]["vin"] == 1
    assert first(8) == first(9)                       # shared counter route id
    assert first(10) != first(11)                     # date-style id across car dirs: no merge
    assert first(20) == first(21)                     # explicit
    assert first(30) == first(31) == first(12)        # one lab cluster incl. the person-name folder device
    assert st["n_lab_routes"] == 15
    assert first(13) != first(14)
    assert st["n_merges"]["lab"] == 2 and st["n_merges"]["explicit"] == 1 and st["n_merges"]["vin"] == 1
    assert st["n_merges"]["route_id"] == 1


def test_inhouse_sources():
    routes = make_routes()
    misfiled = {"SomePerson": dongle(12), dongle(13): dongle(14)}      # person folder (lab) + hex re-key (not lab)
    r = PS.prepare_routes(routes, misfiled)
    rm = pd.DataFrame({"rel_path": [r.rel_path.iloc[0], r.rel_path.iloc[7]], "in_house": ["True", "False"]})
    ih = PS.inhouse_devices(r, rm, misfiled, {"tocf_lab": [dongle(30)], "tocf_t0_extra": [dongle(30), dongle(31)]})
    src = dict(zip(ih.dongle_log, ih.source))
    assert src[dongle(0)] == "route_master"
    assert src[dongle(12)] == "misfiled_name_dir"
    assert src[dongle(30)] == "tocf_lab+tocf_t0_extra"
    assert src[dongle(31)] == "tocf_t0_extra"
    assert dongle(14) not in src and dongle(13) not in src
    assert list(ih.columns) == ["dongle_log", "source"]


# ----------------------------------------------------------------------------- pseudonyms
def _table(routes, labels, lab=()):
    r = PS.prepare_routes(routes, {})
    r["platform"] = r.route_uid.map(dict(zip(labels.route_uid, labels.platform)))
    gk, _ = PS.build_groups(r, inhouse=set(lab))
    return pd.DataFrame({"route_uid": r.route_uid, "dongle_log": r["_device"], "platform": r.platform, "group_key": gk,
                         "route_id": r.route_id, "rel_path": r.rel_path, "vin_hash": r.vin_hash})


def test_assign_formats_unique_deterministic():
    routes = make_routes()
    t = _table(routes, make_labels(routes))
    seed = "ab" * 16
    new, state, counts = PS.assign_pseudonyms(t, None, PS.empty_state(seed), seed)
    assert list(new.columns) == list(PS.MAP_COLUMNS)
    assert new.vehicle_id.str.fullmatch(r"vehicle_\d{4}").all()
    assert new.drive_id.str.fullmatch(r"drive_\d{6}").all()
    assert new.group_id.str.fullmatch(r"group_\d{4}").all()
    assert new.drive_id.is_unique and new.route_uid.is_unique
    assert not (set(new.vehicle_id) & set(new.group_id))
    assert PS.check_map(new) == []
    # random order: ids are not sorted like the keys
    assert not new.sort_values("dongle_log").drive_id.is_monotonic_increasing
    again, _, _ = PS.assign_pseudonyms(t.sample(frac=1, random_state=3), None, PS.empty_state(seed), seed)
    pd.testing.assert_frame_equal(new, again)
    other, _, _ = PS.assign_pseudonyms(t, None, PS.empty_state("cd" * 16), "cd" * 16)
    assert (other.drive_id != new.drive_id).mean() > 0.9
    assert state["namespaces"]["drive"]["n_assigned"] == len(new)
    assert state["namespaces"]["vehicle"]["n_assigned"] == new.vehicle_id.nunique()


def test_append_only_and_regroup():
    routes = make_routes()
    labels = make_labels(routes)
    seed = "ab" * 16
    dev_idx = routes.dongle_log.map(lambda d: int(d[10:], 16))
    t1 = _table(routes[dev_idx < 30], labels)                 # devices 0-29 incl. their shared-route rows
    m1, s1, _ = PS.assign_pseudonyms(t1, None, PS.empty_state(seed), seed)
    t2 = _table(routes, labels)
    m2, s2, c2 = PS.assign_pseudonyms(t2, m1, s1, seed)
    assert PS.check_map(m2, m1) == []
    j = m1.merge(m2, on="route_uid", suffixes=("_1", "_2"))
    assert (j.drive_id_1 == j.drive_id_2).all() and (j.vehicle_id_1 == j.vehicle_id_2).all() and (j.group_id_1 == j.group_id_2).all()
    assert c2["n_new_routes"] == len(t2) - len(t1) and c2["n_group_changed_routes"] == 0
    assert set(m2.drive_id) > set(m1.drive_id)
    # a later explicit merge of two clusters: refused without allow_regroup, accepted with it; retired id never reused
    r = PS.prepare_routes(routes, {})
    r["platform"] = r.route_uid.map(dict(zip(labels.route_uid, labels.platform)))
    gk, _ = PS.build_groups(r, merges=[(dongle(13), dongle(14))])
    t3 = t2.assign(group_key=gk.to_numpy())
    with pytest.raises(AssertionError):
        PS.assign_pseudonyms(t3, m2, s2, seed)
    m3, s3, c3 = PS.assign_pseudonyms(t3, m2, s2, seed, allow_regroup=True)
    assert c3["n_group_ids_retired"] == 1 and c3["n_group_merged_ids"] == 1
    retired = set(m2.group_id) - set(m3.group_id)
    assert len(retired) == 1 and retired <= set(s3["namespaces"]["group"]["retired"])
    assert m3.loc[m3.dongle_log == dongle(13), "group_id"].iloc[0] == m3.loc[m3.dongle_log == dongle(14), "group_id"].iloc[0]
    # a new group afterwards never gets the retired id
    extra = t3.iloc[:1].copy()
    extra["route_uid"] = ["0" * 12]
    extra["dongle_log"] = [dongle(900)]
    extra["group_key"] = [dongle(900)]
    m4, s4, _ = PS.assign_pseudonyms(pd.concat([t3, extra], ignore_index=True), m3, s3, seed)
    assert not (set(m4.group_id) & retired)
    assert PS.check_map(m4, m3) == []
    # tampering with the state is detected
    bad = json.loads(json.dumps(s4))
    bad["namespaces"]["drive"]["n_assigned"] -= 1
    with pytest.raises(ValueError):
        PS.assign_pseudonyms(t3, m4, bad, seed)


def test_vehicle_rekey_keeps_old_vehicle_ids():
    routes = make_routes()
    labels = make_labels(routes)
    seed = "ab" * 16
    t1 = _table(routes, labels)
    m1, s1, _ = PS.assign_pseudonyms(t1, None, PS.empty_state(seed), seed)
    labels2 = labels.copy()
    labels2.loc[labels2.route_uid == t1.route_uid.iloc[0], "platform"] = "PLATFORM_NEW"
    t2 = _table(routes, labels2)
    m2, s2, c2 = PS.assign_pseudonyms(t2, m1, s1, seed)
    assert c2["n_vehicle_rekeyed"] == 1 and c2["n_new_vehicles"] == 1
    assert PS.check_map(m2, m1) == []
    kept = m1[m1.route_uid != t1.route_uid.iloc[0]].merge(m2, on="route_uid", suffixes=("_1", "_2"))
    assert (kept.vehicle_id_1 == kept.vehicle_id_2).all()


def test_namespace_capacity_and_permutation():
    assert PS.namespace_capacity("vehicle_{:04d}") == 10_000
    assert PS.namespace_capacity("drive_{:06d}") == 1_000_000
    p = PS.permutation("x" * 32, "drive", 1_000_000)
    assert len(p) == 1_000_000 and len(np.unique(p[:5000])) == 5000
    assert (PS.permutation("x" * 32, "drive", 100) == PS.permutation("x" * 32, "drive", 100)).all()


# ----------------------------------------------------------------------------- script end to end
def _write_inputs(root: Path, routes: pd.DataFrame, labels: pd.DataFrame | None = True) -> Path:
    p = PS.paths(root)
    p["private"].mkdir(parents=True, exist_ok=True)
    os.chmod(p["private"], 0o700)
    routes.to_parquet(p["routes_resolved"], index=False)
    if labels is not None:
        p["routes_labels"].parent.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(p["routes_labels"], index=False)
    (p["private"] / "misfiled_dirs.json").write_text(json.dumps({"SomePerson": dongle(12)}))
    pd.DataFrame({"a": [dongle(20)], "b": [dongle(21)]}).to_csv(p["identity_merges"], index=False)
    rm = root / "route_master.csv"
    pd.DataFrame({"rel_path": [routes.rel_path.iloc[0], routes.rel_path.iloc[5]], "in_house": ["True", "True"],
                  "dongle": ["x", "y"]}).to_csv(rm, index=False)
    return rm


def _run(root: Path, rm: Path, *extra) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO), VTT_DATA_ROOT=str(root))
    return subprocess.run([PY, str(REPO / "scripts" / "s04_pseudonyms.py"), "--data-root", str(root), "--route-master", str(rm),
                          "--no-tocf", *extra], capture_output=True, text=True, env=env, cwd=str(REPO))


def test_script_end_to_end(tmp_root):
    routes = make_routes()
    labels = make_labels(routes)
    rm = _write_inputs(tmp_root, routes, labels)
    r = _run(tmp_root, rm, "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    p = PS.paths(tmp_root)
    assert not p["drive_names"].exists() and not p["pseudonym_map"].exists()
    r = _run(tmp_root, rm)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert not re.search(r"(?<![0-9a-zA-Z])[0-9a-f]{16}(?![0-9a-zA-Z])", out)
    assert "SomePerson" not in out
    names = pd.read_parquet(p["drive_names"])
    assert list(names.columns) == ["route_uid", "vehicle_id", "drive_id", "group_id"]
    assert len(names) == len(routes) and names.drive_id.is_unique
    ih = pd.read_csv(p["inhouse_groups"], dtype=str).fillna("")
    assert set(ih.dongle_log) == {dongle(0), dongle(1), dongle(12)}
    assert ih.group_id.nunique() == 1 and ih.group_id.iloc[0].startswith("group_")
    m = pd.read_parquet(p["pseudonym_map"])
    assert list(m.columns) == list(PS.MAP_COLUMNS)
    assert m.loc[m.dongle_log == dongle(20), "group_id"].iloc[0] == m.loc[m.dongle_log == dongle(21), "group_id"].iloc[0]
    assert (m.assigned_release == C.RELEASE_VERSION).all()
    assert oct(os.stat(p["pseudonym_map"]).st_mode & 0o777) == "0o600"
    assert p["release_seed"].exists() and (tmp_root / "private" / PS.STATE_FILE).exists()
    # idempotent re-run
    r = _run(tmp_root, rm)
    assert r.returncode == 0, r.stdout + r.stderr
    pd.testing.assert_frame_equal(names, pd.read_parquet(p["drive_names"]))
    # append: more routes, old ids unchanged
    more = make_routes(n_devices=45)
    _write_inputs(tmp_root, more, make_labels(more))
    r = _run(tmp_root, rm)
    assert r.returncode == 0, r.stdout + r.stderr
    names2 = pd.read_parquet(p["drive_names"])
    j = names.merge(names2, on="route_uid", suffixes=("_1", "_2"))
    assert len(j) == len(names) and (j.drive_id_1 == j.drive_id_2).all() and (j.vehicle_id_1 == j.vehicle_id_2).all()
    # a merge of two clusters without --allow-regroup exits 2 and writes nothing new
    before = p["pseudonym_map"].stat().st_mtime_ns
    pd.DataFrame({"a": [dongle(20), dongle(13)], "b": [dongle(21), dongle(14)]}).to_csv(p["identity_merges"], index=False)
    r = _run(tmp_root, rm)
    assert r.returncode == 2
    assert p["pseudonym_map"].stat().st_mtime_ns == before
    r = _run(tmp_root, rm, "--allow-regroup")
    assert r.returncode == 0, r.stdout + r.stderr


def test_script_missing_labels(tmp_root):
    routes = make_routes()
    rm = _write_inputs(tmp_root, routes, labels=None)
    r = _run(tmp_root, rm)
    assert r.returncode == 2
    r = _run(tmp_root, rm, "--allow-missing-labels")
    assert r.returncode == 0, r.stdout + r.stderr


# ----------------------------------------------------------------------------- verifier follow-ups
def test_one_route_vehicle_rekey_then_rerun():
    """A (device, platform) key with ONE route changes platform: the old vehicle_id is retired (never reused) and a third
    run with unchanged inputs succeeds (it used to fail the namespace check)."""
    routes = make_routes(n_devices=20, routes_per_device=1)
    labels = make_labels(routes)
    seed = "ab" * 16
    t1 = _table(routes, labels)
    m1, s1, _ = PS.assign_pseudonyms(t1, None, PS.empty_state(seed), seed)
    target = t1.route_uid.iloc[0]
    orphan = m1.loc[m1.route_uid == target, "vehicle_id"].iloc[0]
    assert (m1.vehicle_id == orphan).sum() == 1
    labels2 = labels.copy()
    labels2.loc[labels2.route_uid == target, "platform"] = "PLATFORM_NEW"
    t2 = _table(routes, labels2)
    m2, s2, c2 = PS.assign_pseudonyms(t2, m1, s1, seed)
    assert c2["n_vehicle_rekeyed"] == 1 and c2["n_new_vehicles"] == 1 and c2["n_vehicle_ids_retired"] == 1
    assert orphan not in set(m2.vehicle_id) and s2["namespaces"]["vehicle"]["retired"] == [orphan]
    assert PS.check_map(m2, m1) == []
    m3, s3, c3 = PS.assign_pseudonyms(t2, m2, s2, seed)                 # unchanged inputs: must not raise
    pd.testing.assert_frame_equal(m2, m3)
    assert s3["namespaces"]["vehicle"] == s2["namespaces"]["vehicle"] and c3["n_vehicle_ids_retired"] == 0
    extra = t2.iloc[:1].copy()
    extra["route_uid"], extra["dongle_log"], extra["group_key"], extra["platform"] = ["0" * 12], [dongle(900)], [dongle(900)], ["PLATFORM_X"]
    m4, s4, c4 = PS.assign_pseudonyms(pd.concat([t2, extra], ignore_index=True), m3, s3, seed)
    assert c4["n_new_vehicles"] == 1 and orphan not in set(m4.vehicle_id) and PS.check_map(m4, m3) == []
    assert s4["namespaces"]["vehicle"]["n_assigned"] == m4.vehicle_id.nunique() + 1


def test_load_merges_evidence(tmp_root):
    p = tmp_root / "identity_merges.csv"
    pd.DataFrame({"dongle_log_a": [dongle(1), dongle(3), dongle(5)], "dongle_log_b": [dongle(2), dongle(4), dongle(6)],
                  "route_id": ["r1", "r2", "r3"], "rule": "shared_route_id",
                  "evidence": ["counter_id", "date_id_other_car", "date_id_same_car"]}).to_csv(p, index=False)
    pairs, counts = PS.load_merges(p)
    assert pairs == [(dongle(1), dongle(2)), (dongle(5), dongle(6))]
    assert counts["n_rows"] == 3 and counts["n_used"] == 2 and counts["n_skipped"] == 1
    assert counts["by_evidence"] == {"counter_id": 1, "date_id_other_car": 1, "date_id_same_car": 1}
    assert counts["skipped_evidence"] == sorted(C.MERGE_SKIP_EVIDENCE)
    pairs_all, c_all = PS.load_merges(p, skip_evidence=())
    assert len(pairs_all) == 3 and c_all["n_skipped"] == 0
    pd.DataFrame({"a": [dongle(7)], "b": [dongle(8)]}).to_csv(p, index=False)       # two columns, no evidence: all used
    pairs, counts = PS.load_merges(p)
    assert pairs == [(dongle(7), dongle(8))] and counts["by_evidence"] == {"unspecified": 1}
    assert PS.load_merges(tmp_root / "missing.csv")[0] == []


def test_misfiled_markers_and_nonhex_lab_devices():
    raw = {"SomePerson": dongle(12), "OtherPerson": "OtherPerson", dongle(13): dongle(14), "Nobody": "", dongle(15): "x"}
    clean, n = PS.filter_misfiled(raw)
    assert clean == {"SomePerson": dongle(12), dongle(13): dongle(14)} and n == 3
    routes = make_routes()
    r = PS.prepare_routes(routes, clean)
    ih = PS.inhouse_devices(r, None, {**clean, "OtherPerson": "OtherPerson"}, {"tocf_lab": ["notahexid", dongle(30)]})
    assert set(ih.dongle_log) == {dongle(12), dongle(30)} and ih.attrs["n_ignored_nonhex"] == 2
    assert all(PS.HEX16.match(d) for d in ih.dongle_log)


def test_script_route_master_required(tmp_root):
    routes = make_routes()
    _write_inputs(tmp_root, routes, make_labels(routes))
    r = _run(tmp_root, tmp_root / "nope.csv")
    assert r.returncode == 2 and "route_master" in r.stdout
    r = _run(tmp_root, tmp_root / "nope.csv", "--no-route-master")
    assert r.returncode == 0, r.stdout + r.stderr
    ih = pd.read_csv(PS.paths(tmp_root)["inhouse_groups"], dtype=str).fillna("")
    assert set(ih.dongle_log) == {dongle(12)}                       # the person-name folder's device only
    assert "in-house devices: 1" in r.stdout


def test_script_rekey_rerun(tmp_root):
    """Script path of the re-key case: relabel the only route of a device, run twice more, both exit 0."""
    routes = make_routes(n_devices=20, routes_per_device=1)
    labels = make_labels(routes)
    rm = _write_inputs(tmp_root, routes, labels)
    r = _run(tmp_root, rm)
    assert r.returncode == 0, r.stdout + r.stderr
    p = PS.paths(tmp_root)
    names1 = pd.read_parquet(p["drive_names"])
    target = routes.route_uid.iloc[0]
    labels2 = labels.copy()
    labels2.loc[labels2.route_uid == target, "platform"] = "PLATFORM_NEW"
    labels2.to_parquet(p["routes_labels"], index=False)
    r = _run(tmp_root, rm)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "vehicle ids retired: 1" in r.stdout
    r = _run(tmp_root, rm)
    assert r.returncode == 0, r.stdout + r.stderr
    st = json.loads((p["private"] / PS.STATE_FILE).read_text())
    assert len(st["namespaces"]["vehicle"]["retired"]) == 1
    names3 = pd.read_parquet(p["drive_names"])
    j = names1.merge(names3, on="route_uid", suffixes=("_1", "_3"))
    assert len(j) == len(names1) and (j.drive_id_1 == j.drive_id_3).all() and int((j.vehicle_id_1 != j.vehicle_id_3).sum()) == 1
    assert st["namespaces"]["vehicle"]["retired"][0] not in set(names3.vehicle_id)
