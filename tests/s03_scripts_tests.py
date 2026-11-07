"""scripts/s03_platforms.py, s03_labels.py, s03_inventory.py end to end on a synthetic data root (fake ids only).

The module fixture runs the whole S3a chain once (platforms -> user fills specs / hand-maps a spelling -> platforms again
-> labels); every test re-runs what it needs and restores the state it changes, so the order of tests does not matter
(pytest-randomly is installed in the venv)."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pandas as pd

from vtt import labels as L
import pytest

from tests import s3_synth as SY
from vtt import config as C
from vtt import specs as S
from vtt.private import mask

R = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, R / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def common_args(env) -> list[str]:
    return ["--data-root", str(env["data_root"]), "--taxonomy-csv", str(env["data"]["taxonomy_csv"]),
            "--taxonomy-dir", str(env["taxdir"]), "--specs-dir", str(env["specs_dir"])]


def platform_args(env) -> list[str]:
    return common_args(env) + ["--opendbc-root", str(env["opendbc"])]


def label_args(env) -> list[str]:
    return common_args(env) + ["--route-master", str(env["data"]["master_csv"])]


def fill_specs_by_hand(env) -> None:
    """What the user does between the two s03_platforms runs: tracks for four platforms, one hand-mapped spelling."""
    sp = S.read_platform_specs_raw(env["specs_dir"] / "platform_specs.csv").set_index("platform")
    sp.loc["FAKE_SEDAN", ["track_front_mm", "track_rear_mm", "track_source", "spec_confidence"]] = ["1580", "1590", "oem", "A"]
    sp.loc["FAKE_SUV_EV", ["track_front_mm", "track_rear_mm", "spec_confidence"]] = ["1600", "1610", "B"]
    sp.loc["FAKE_TRUCK", ["track_front_mm", "track_rear_mm", "spec_confidence"]] = ["1720", "1720", "B"]
    sp.loc["TOYOTA_FAKE_TSS2", ["wheelbase_mm", "track_front_mm", "track_rear_mm", "spec_confidence"]] = ["2690", "1595", "1615", "B"]
    S.write_csv_sorted(sp.reset_index(), env["specs_dir"] / "platform_specs.csv")
    mm = S.read_fp2platform(env["specs_dir"] / "fingerprint_to_platform.csv").set_index("fingerprint")
    mm.loc["TOTALLY_NEW_CAR", ["platform", "note"]] = ["FAKE_SUV_EV", "hand: fork spelling of the SUV"]
    S.write_csv_sorted(mm.reset_index(), env["specs_dir"] / "fingerprint_to_platform.csv")


@pytest.fixture(scope="module")
def env():
    root = SY.mkdtemp("vtt_s03_")
    opendbc = SY.make_opendbc(root / "opendbc")
    taxdir = SY.make_taxonomy_dir(root / "taxonomy")
    data = SY.make_data_root(root / "data", taxdir)
    e = dict(root=root, opendbc=opendbc, taxdir=taxdir, data=data, specs_dir=root / "specs", data_root=root / "data")
    s03p, s03l = load_script("s03_platforms"), load_script("s03_labels")
    assert s03p.main(platform_args(e) + ["--dry-run"]) == 0
    assert not (e["specs_dir"] / "fingerprint_to_platform.csv").exists()
    assert s03p.main(platform_args(e)) == 0
    e["specs_first"] = S.read_platform_specs_raw(e["specs_dir"] / "platform_specs.csv").set_index("platform")
    e["mapping_first"] = S.read_fp2platform(e["specs_dir"] / "fingerprint_to_platform.csv").set_index("fingerprint")
    fill_specs_by_hand(e)
    assert s03p.main(platform_args(e)) == 0
    assert s03l.main(label_args(e) + ["--dry-run"]) == 0
    assert not (e["data_root"] / "build" / "routes_labels.parquet").exists()
    assert s03l.main(label_args(e)) == 0
    yield e
    SY.rmtree(root)


# -------------------------------------------------------------------------------------------------- s03_platforms
def test_platforms_script_first_run(env):
    m, sp = env["mapping_first"], env["specs_first"]
    tax_fps = set(env["data"]["taxonomy"]["fingerprint"].dropna())
    probe_fps = set(env["data"]["routes"]["car_fp"].dropna())
    assert tax_fps <= set(m.index) and probe_fps <= set(m.index)
    assert m.loc["FAKE_SEDAN", "method"] == "exact" and m.loc["FAKE_TRUCK_MMR", "platform"] == "FAKE_TRUCK"
    assert m.loc["TOTALLY_NEW_CAR", "method"] == "manual" and m.loc["TOTALLY_NEW_CAR", "note"] == S.UNRESOLVED
    assert m.loc["mock", "platform"] == "MOCK" and sp.loc["MOCK", "model"] == ""        # mock routes never vote a model
    assert set(m["platform"]) == set(sp.index)
    folders = set(env["data"]["taxonomy"].query("source in ('directory', 'fw-match')")["car_dir"]) - {"Unknown"}
    assert folders and folders <= set(m.index)                                            # folder spellings get rows too
    assert sp.loc["FAKE_SEDAN", "model"] == "Fake Sedan" and sp.loc["FAKE_SEDAN", "wheelbase_mm"] == "2820"
    assert sp.loc["FAKE_SEDAN", "curb_mass_kg"] == "1590" and sp.loc["FAKE_SEDAN", "mass_source"] == S.OPENDBC_SOURCE
    assert sp.loc["FAKE_TRUCK", "body"] == "pickup" and sp.loc["TOYOTA_FAKE_TSS2", "brand"] == "Toyota"
    assert sp.loc["MOCK", "wheelbase_mm"] == "" and sp.loc["MOCK", "brand"] == ""
    assert (sp["spec_confidence"] == "C").all() and (sp["track_front_mm"] == "").all()


def test_platforms_script_rerun_keeps_hand_work(env, capsys):
    """A third run must keep the filled cells and the hand-resolved manual row (idempotent merge)."""
    s03p = load_script("s03_platforms")
    assert s03p.main(platform_args(env)) == 0
    out = capsys.readouterr().out
    assert "method counts" in out and "hours by method" in out and mask(out) == out
    sp = S.read_platform_specs_raw(env["specs_dir"] / "platform_specs.csv").set_index("platform")
    assert sp.loc["FAKE_SEDAN", "track_front_mm"] == "1580" and sp.loc["FAKE_SEDAN", "spec_confidence"] == "A"
    assert sp.loc["FAKE_SEDAN", "track_source"] == "oem" and sp.loc["FAKE_SEDAN", "wheelbase_mm"] == "2820"
    assert sp.loc["TOYOTA_FAKE_TSS2", "wheelbase_mm"] == "2690"
    assert sp.loc["FAKE_TRUCK", "wheelbase_mm"] == "3745"
    m = S.read_fp2platform(env["specs_dir"] / "fingerprint_to_platform.csv").set_index("fingerprint")
    assert m.loc["TOTALLY_NEW_CAR", "platform"] == "FAKE_SUV_EV" and m.loc["TOTALLY_NEW_CAR", "note"].startswith("hand")
    assert m.loc["FAKE_SEDAN", "method"] == "exact"


def test_platforms_script_missing_opendbc(env):
    s03p = load_script("s03_platforms")
    assert s03p.main(common_args(env) + ["--opendbc-root", str(env["root"] / "nowhere"), "--dry-run"]) == 2


# ----------------------------------------------------------------------------------------------------- s03_labels
def test_labels_script_outputs(env, capsys):
    s03l = load_script("s03_labels")
    assert s03l.main(label_args(env)) == 0                     # idempotent re-run
    out = capsys.readouterr().out
    assert "size_class x powertrain" in out and "label_source" in out and mask(out) == out
    pub = pd.read_parquet(env["data_root"] / "build" / "routes_labels.parquet")
    priv = pd.read_parquet(env["data_root"] / "private" / "labels_private.parquet")
    assert list(pub.columns) == s03l.PUBLIC_COLUMNS
    assert list(priv.columns) == s03l.PUBLIC_COLUMNS + s03l.PRIVATE_EXTRA
    assert "lat" not in priv.columns and "lon" not in priv.columns
    assert (env["data_root"] / "private" / "labels_private.parquet").stat().st_mode & 0o077 == 0
    routes = env["data"]["routes"]
    assert len(pub) == len(routes) and set(pub["route_uid"]) == set(routes["route_uid"])
    p = priv.set_index("route_uid")
    uids = routes["route_uid"].tolist()

    r0 = p.loc[uids[0]]           # taxonomy row via the chosen rel_path, specs filled
    assert r0.label_source == "fingerprint" and r0.label_grade == "A" and r0.platform == "FAKE_SEDAN"
    assert r0.model == "Fake Sedan" and r0.brand == "Fake" and r0.body == "sedan" and r0.size_code == "D"
    assert r0.powertrain_raw == "ICE" and r0.powertrain == "ICE" and r0.region == "north_america" and r0.software_era == "0.9"
    assert r0.wheelbase_mm == 2820 and r0.track_front_mm == 1580 and r0.spec_confidence == "A"
    assert abs(r0.footprint_m2 - 2.82 * 1.585) < 1e-9
    assert r0.size_class == L.size_class(r0.footprint_m2)      # None while unlocked, the configured cut once S3c locked
    assert bool(r0.in_house) is False and r0.country == "Fakeland"

    r1 = p.loc[uids[1]]           # taxonomy row found through another copy's rel_path
    assert r1.label_source == "fingerprint" and "another copy" in r1.label_note and r1.software_era == "0.8"

    r2 = p.loc[uids[2]]           # no taxonomy row: resolved from the probe fingerprint
    assert r2.label_source == "resolved_new" and r2.label_grade == "A" and r2.powertrain == "BEV"
    assert r2.platform == "FAKE_SUV_EV" and r2.region == "europe" and r2.software_era == "2025"
    assert r2.footprint_m2 == pytest.approx(2.9 * 1.605)

    r3 = p.loc[uids[3]]           # unknown fingerprint: hand-mapped platform but Unknown model, grade C
    assert r3.label_source == "unmapped" and r3.label_grade == "C" and r3.model == "Unknown" and r3.powertrain is None
    assert r3.powertrain_raw == "UNK" and r3.region == C.REGION_OTHER and r3.platform == "FAKE_SUV_EV"

    r4 = p.loc[uids[4]]           # Toyota MIXED + hybrid ECU from the audit
    assert r4.powertrain_raw == "HEV" and r4.powertrain == "HEV" and "toyota_hybrid_ecu" in r4.label_note
    assert "gas and hybrid" in r4.label_note and r4.region == "asia" and r4.wheelbase_mm == 2690
    r5 = p.loc[uids[5]]           # Honda Accord MIXED + transmission ECU (JSON list)
    assert r5.powertrain_raw == "ICE" and "honda_engine_ecu" in r5.label_note and r5.software_era == "0.11"
    r6 = p.loc[uids[6]]           # VW Golf MIXED + direct transmission
    assert r6.powertrain_raw == "BEV" and r6.powertrain == "BEV" and "vw_direct_transmission" in r6.label_note

    r7 = p.loc[uids[7]]           # fingerprint unreadable: folder-name taxonomy row, platform from the folder name
    assert r7.label_source == "directory" and r7.label_grade == "B" and r7.platform == "FAKE_TRUCK"
    assert "folder name" in r7.label_note and r7.fingerprint_raw is None and r7.region == "north_america"
    assert "mild hybrid trims" not in (r7.label_note or "")        # no fingerprint -> no curator note
    r8 = p.loc[uids[8]]           # alias spelling, fork suffix mapped to the base platform, lab device
    assert r8.label_source == "alias" and r8.platform == "FAKE_TRUCK" and r8.model == "Fake Truck" and bool(r8.in_house) is True
    assert r8.footprint_m2 == pytest.approx(3.745 * 1.72) and "mild hybrid trims" not in (r8.label_note or "")
    r9 = p.loc[uids[9]]           # mock fingerprint inside a real car folder: folder-name labels and platform, grade B
    assert r9.model == "Fake Sedan" and r9.powertrain == "ICE" and r9.platform == "FAKE_SEDAN" and r9.region == C.REGION_OTHER
    assert r9.label_source == "directory" and r9.label_grade == "B" and "mock fingerprint" in r9.label_note
    assert r9.fingerprint_raw == "mock"
    r10 = p.loc[uids[10]]         # firmware match in the Unknown folder: model from the match, platform = the model's only platform
    assert r10.label_source == "fw-match" and r10.label_grade == "A" and r10.model == "Fake SUV" and r10.powertrain == "BEV"
    assert r10.platform == "FAKE_SUV_EV" and "taxonomy model" in r10.label_note and "mock fingerprint" in r10.label_note
    assert r10.footprint_m2 == pytest.approx(2.9 * 1.605)

    assert set(pub["label_grade"]) == {"A", "B", "C"}
    if L.C.FOOTPRINT_THRESHOLDS_M2 is None:
        assert pub["size_class"].isna().all()
    else:                                                   # S3c locked: every known footprint is cut, unknown stays null
        known = pub["footprint_m2"].notna()
        assert pub.loc[known, "size_class"].isin(L.C.SIZE_CLASSES).all() and pub.loc[~known, "size_class"].isna().all()
    assert set(pub["powertrain"].dropna()) <= set(C.POWERTRAIN4)


def test_labels_script_overrides_and_rerun(env, capsys):
    s03l = load_script("s03_labels")
    uids = env["data"]["routes"]["route_uid"].tolist()
    ov = env["data_root"] / "build" / "label_overrides_routes.csv"          # S3a per-route overrides
    s7 = env["data_root"] / "build" / "label_overrides.csv"                 # S7 table: another schema, never read here
    try:
        pd.DataFrame([dict(scope="platform", key="FAKE_SEDAN", powertrain="HEV", label_note="s7", resolve="1")]).to_csv(s7, index=False)
        pd.DataFrame([
            {"route_uid": uids[0], "field": "powertrain_raw", "value": "HEV", "reason": "owner confirmed hybrid"},
            {"route_uid": uids[2], "field": "wheelbase_mm", "value": "3000", "reason": "spec fix"},
            {"route_uid": uids[4], "field": "powertrain", "value": "null", "reason": "undecidable"},
            {"route_uid": uids[5], "field": "powertrain", "value": "HEV", "reason": "explicit class first"},
            {"route_uid": uids[5], "field": "powertrain_raw", "value": "ICE", "reason": "raw code after it"},
            {"route_uid": uids[6], "field": "track_front_mm", "value": "1500", "reason": "explicit cell before the platform"},
            {"route_uid": uids[6], "field": "platform", "value": "FAKE_TRUCK", "reason": "re-merge specs"},
            {"route_uid": "0" * 12, "field": "model", "value": "X", "reason": "unknown route"},
        ]).to_csv(ov, index=False)
        assert s03l.main(label_args(env)) == 0
        out = capsys.readouterr().out
        assert "'applied': 7" in out and "'unknown_uid': 1" in out
        pub = pd.read_parquet(env["data_root"] / "build" / "routes_labels.parquet").set_index("route_uid")
        assert pub.loc[uids[0], "powertrain_raw"] == "HEV" and pub.loc[uids[0], "powertrain"] == "HEV"
        assert "override: owner confirmed hybrid" in pub.loc[uids[0], "label_note"]
        assert pub.loc[uids[2], "wheelbase_mm"] == 3000 and pub.loc[uids[2], "footprint_m2"] == pytest.approx(3.0 * 1.605)
        assert pub.loc[uids[4], "powertrain"] is None and pub.loc[uids[4], "powertrain_raw"] == "HEV"
        assert pub.loc[uids[5], "powertrain"] == "HEV" and pub.loc[uids[5], "powertrain_raw"] == "ICE"   # explicit class wins
        r6 = pub.loc[uids[6]]                      # platform override re-merges the specs row; the explicit track cell wins
        assert r6.platform == "FAKE_TRUCK" and r6.wheelbase_mm == 3745 and r6.track_rear_mm == 1720 and r6.track_front_mm == 1500
        assert r6.curb_mass_kg == 2500 and r6.spec_confidence == "B" and r6.footprint_m2 == pytest.approx(3.745 * 1.61)
        assert r6.model == "Volkswagen Golf" and r6.powertrain == "BEV"           # non-spec labels untouched
        assert pub.loc[uids[9], "powertrain"] == "ICE"                            # the S7 table was not applied
        pd.DataFrame([{"route_uid": uids[0], "field": "colour", "value": "red", "reason": ""}]).to_csv(ov, index=False)
        assert s03l.main(label_args(env)) == 2                 # unknown field fails acceptance
        pd.DataFrame([dict(scope="platform", key="FAKE_SEDAN", powertrain="HEV", label_note="", resolve="")]).to_csv(ov, index=False)
        assert s03l.main(label_args(env)) == 2                 # S7-format file at the S3 path: refused, no traceback
        out = capsys.readouterr().out
        assert "ACCEPTANCE FAILED" in out and "required columns missing" in out and mask(out) == out
        pd.DataFrame([{"route_uid": uids[0], "field": "model", "value": "Fake SUV"}]).to_csv(ov, index=False)
        assert s03l.main(label_args(env)) == 0                 # the reason column is optional
    finally:
        for f in (ov, s7):
            if f.exists():
                f.unlink()
        assert s03l.main(label_args(env)) == 0                 # restore the computed labels
    pub2 = pd.read_parquet(env["data_root"] / "build" / "routes_labels.parquet").set_index("route_uid")
    assert pub2.loc[uids[0], "powertrain_raw"] == "ICE" and pub2.loc[uids[6], "platform"] == "VOLKSWAGEN_GOLF_MK7"


def test_labels_script_size_class_when_locked(env, monkeypatch):
    s03l = load_script("s03_labels")
    uids = env["data"]["routes"]["route_uid"].tolist()
    monkeypatch.setattr(C, "FOOTPRINT_THRESHOLDS_M2", (4.5, 5.5))
    try:
        assert s03l.main(label_args(env)) == 0
        pub = pd.read_parquet(env["data_root"] / "build" / "routes_labels.parquet").set_index("route_uid")
        assert pub.loc[uids[0], "size_class"] == "SMALL" and pub.loc[uids[2], "size_class"] == "MID"
        assert pub.loc[uids[8], "size_class"] == "LARGE" and pub.loc[uids[5], "size_class"] is None
    finally:
        monkeypatch.setattr(C, "FOOTPRINT_THRESHOLDS_M2", None)
        assert s03l.main(label_args(env)) == 0


def test_labels_script_missing_mapping_row_fails(env, capsys):
    """A fingerprint that s03_platforms has not seen yet must stop the labels stage (exit 2)."""
    s03l = load_script("s03_labels")
    rr = env["data_root"] / "private" / "routes_resolved.parquet"
    routes = pd.read_parquet(rr)
    extra = routes.iloc[[0]].copy()
    extra["route_uid"] = "%012x" % 0xABCFFF
    extra["car_fp"] = "BRAND_NEW_SPELLING_2031"
    extra["rel_path"] = SY.fake_rel("FAKE_SEDAN", 999)
    extra["copies_rel"] = '["' + SY.fake_rel("FAKE_SEDAN", 999) + '"]'
    try:
        pd.concat([routes, extra]).to_parquet(rr, index=False)
        assert s03l.main(label_args(env)) == 2
        assert "without a mapping row" in capsys.readouterr().out
    finally:
        routes.to_parquet(rr, index=False)
        assert s03l.main(label_args(env)) == 0


def test_labels_script_missing_inputs(env):
    s03l = load_script("s03_labels")
    empty = SY.mkdtemp("vtt_s03_empty_")
    try:
        assert s03l.main(["--data-root", str(empty), "--specs-dir", str(env["specs_dir"])]) == 2
    finally:
        SY.rmtree(empty)


# -------------------------------------------------------------------------------------------------- s03_inventory
def test_inventory_script(env, capsys):
    s03i = load_script("s03_inventory")
    args = ["--data-root", str(env["data_root"]), "--specs-dir", str(env["specs_dir"])]
    md_path = env["data_root"] / "build" / "SIZE_INVENTORY.md"
    if md_path.exists():
        md_path.unlink()
    assert s03i.main(args + ["--dry-run"]) == 0
    assert not md_path.exists()
    assert s03i.main(args + ["--thresholds", "5.5,4.5"]) == 0
    out = capsys.readouterr().out
    assert "scheme tertiles" in out and "scheme natural_breaks" in out and "scheme user: 4.500 / 5.500" in out
    md = md_path.read_text()
    assert mask(md) == md and "ffffffffff" not in md and not re.search(r"\d{4}-\d{2}-\d{2}--", md)
    assert "## Platforms by footprint" in md and "## Scheme: tertiles" in md and "## Scheme: natural_breaks" in md
    assert "## Scheme: user" in md and "SMALL < 4.500 <= MID < 5.500 <= LARGE" in md
    assert "| FAKE_SEDAN | Fake Sedan | Fake | sedan | 4.47 |" in md
    assert "## Platforms without footprint" in md and "| HONDA_ACCORD |" in md and "| VOLKSWAGEN_GOLF_MK7 |" in md
    assert "platforms within +-0.1 m^2 of" in md
    assert md.index("| FAKE_SEDAN |") < md.index("| FAKE_SUV_EV |") < md.index("| FAKE_TRUCK |")   # footprint order


def test_inventory_requires_labels(env):
    s03i = load_script("s03_inventory")
    empty = SY.mkdtemp("vtt_s03_empty2_")
    try:
        assert s03i.main(["--data-root", str(empty)]) == 2
    finally:
        SY.rmtree(empty)
