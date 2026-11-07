"""vtt.labels: resolve() port, ECU rules, codes, buckets, footprint and size-class cut. Fake tables only."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from vtt import config as C
from vtt import labels as L

MODELS = {"Fake Sedan": ("Fake", "sedan", "D"), "Fake SUV": ("Fake", "suv", "SUV-C"), "Toyota Fake": ("Toyota", "suv", "SUV-C"),
          "Honda Accord": ("Honda", "sedan", "D"), "Volkswagen Golf": ("Volkswagen", "hatch", "C"), "Unknown": ("Unknown", "unknown", "UNK")}
DIRS = {"FAKE_SEDAN": ("Fake Sedan", "ICE", ""), "FAKE SEDAN 2017": ("Fake Sedan", "ICE", "legacy spelling"),
        "FAKE_SUV_EV": ("Fake SUV", "BEV", ""), "TOYOTA_FAKE_TSS2": ("Toyota Fake", "MIXED", "gas and hybrid"),
        "HONDA_ACCORD": ("Honda Accord", "MIXED", ""), "VOLKSWAGEN_GOLF_MK7": ("Volkswagen Golf", "MIXED", ""), "Unknown": ("Unknown", "UNK", "")}
ALIAS = {"FAKE_SEDAN_FL": ("Fake Sedan", "ICE")}


@pytest.fixture(scope="module")
def tables():
    return L.make_tables(MODELS, DIRS, ALIAS)


# ------------------------------------------------------------------------------------------------------- resolve
def test_resolve_exact_alias_prefix_unmapped_none(tables):
    assert L.resolve("FAKE_SEDAN", tables) == ("Fake Sedan", "ICE", "fingerprint")
    assert L.resolve("fake sedan 2017", tables) == ("Fake Sedan", "ICE", "fingerprint")       # normalized spelling
    assert L.resolve("FAKE_SEDAN_FL", tables) == ("Fake Sedan", "ICE", "alias")
    assert L.resolve("FAKE_SEDAN_2021_NON_SCC", tables) == ("Fake Sedan", "ICE", "prefix")
    assert L.resolve("SOMETHING_ELSE", tables) == ("?SOMETHING_ELSE", "UNK", "unmapped")
    assert L.resolve("mock", tables) is None and L.resolve("", tables) is None and L.resolve(None, tables) is None
    assert L.resolve(float("nan"), tables) is None


def test_resolve_prefix_generic_powertrain_override(tables):
    assert L.resolve("FAKE_SEDAN_HYBRID_2020", tables) == ("Fake Sedan", "HEV", "prefix")
    assert L.resolve("FAKE_SEDAN_PHEV", tables) == ("Fake Sedan", "PHEV", "prefix")
    assert L.resolve("FAKE_SEDAN_ELECTRIC_2022", tables) == ("Fake Sedan", "BEV", "prefix")
    assert L.resolve("FAKE_SUV_EV_2023", tables) == ("Fake SUV", "BEV", "prefix")                 # token already in the prefix


def test_resolve_full_and_notes(tables):
    rf = L.resolve_full("TOYOTA_FAKE_TSS2", tables)
    assert rf["model"] == "Toyota Fake" and rf["brand"] == "Toyota" and rf["body"] == "suv" and rf["size_code"] == "SUV-C"
    assert rf["powertrain_raw"] == "MIXED" and rf["how"] == "fingerprint" and rf["note"] == "gas and hybrid"
    un = L.resolve_full("NOT_A_CAR", tables)
    assert un["model"] == "Unknown" and un["how"] == "unmapped" and un["powertrain_raw"] == "UNK"
    assert L.resolve_full(None, tables)["how"] == "none"
    assert L.model_attrs("Mystery Car", tables) == ("Mystery", "unknown", "UNK")
    assert L.dir_note("FAKE SEDAN 2017", tables) == "legacy spelling" and L.dir_note("FAKE_SEDAN_FL", tables) == ""


# ------------------------------------------------------------------------------------------------------- ECU rules
def test_ecu_set_formats():
    assert L.ecu_set(None) == set() and L.ecu_set(float("nan")) == set() and L.ecu_set("") == set()
    assert L.ecu_set("eps;hybrid;fwdCamera") == {"eps", "hybrid", "fwdCamera"}
    assert L.ecu_set(json.dumps(["eps", "transmission"])) == {"eps", "transmission"}
    assert L.ecu_set(["eps"]) == {"eps"} and L.ecu_set(np.array(["eps", "abs"])) == {"eps", "abs"}
    assert L.ecu_set("[not json") == set()


def test_ecu_rules_toyota_honda_vw():
    assert L.ecu_resolve("Toyota Fake", "TOYOTA_FAKE_TSS2", "MIXED", "eps;hybrid") == ("HEV", "toyota_hybrid_ecu")
    assert L.ecu_resolve("Lexus Fake", "LEXUS_X", "MIXED", '["eps", "transmission"]') == ("ICE", "toyota_transmission_ecu")
    assert L.ecu_resolve("Toyota Fake", "TOYOTA_FAKE_TSS2", "MIXED", "eps;fwdCamera") == ("MIXED", "")
    assert L.ecu_resolve("Honda Accord", "HONDA_ACCORD", "MIXED", "eps;transmission") == ("ICE", "honda_engine_ecu")
    assert L.ecu_resolve("Honda Accord", "HONDA ACCORD 2018", "MIXED", "programmedFuelInjection") == ("ICE", "honda_engine_ecu")
    assert L.ecu_resolve("Honda Accord", "HONDA_INSPIRE", "MIXED", "eps;shiftByWire") == ("HEV", "honda_hybrid_ecu_pattern")
    assert L.ecu_resolve("Honda Accord", "HONDA_ACCORD_11G", "MIXED", "eps;shiftByWire") == ("MIXED", "")   # other platform
    assert L.ecu_resolve("Honda Accord", "HONDA_ACCORD", "MIXED", "") == ("MIXED", "")                    # no ECU list
    assert L.ecu_resolve("Volkswagen Golf", "VOLKSWAGEN_GOLF_MK7", "MIXED", None, "direct") == ("BEV", "vw_direct_transmission")
    assert L.ecu_resolve("Volkswagen Golf", "VOLKSWAGEN_GOLF_MK7", "MIXED", None, "automatic") == ("MIXED", "")
    assert L.ecu_resolve("Ford F-150", "FORD_F_150_MK14", "MIXED", "eps;hybrid") == ("MIXED", "")           # no rule for Ford
    assert L.ecu_resolve("Toyota Fake", "X", "ICE", "eps;hybrid") == ("ICE", "")                           # non-MIXED untouched


# ------------------------------------------------------------------------------------------------- codes / buckets
def test_powertrain_map_and_grade():
    for p in C.POWERTRAIN4:
        assert L.powertrain_map(p) == p
    assert L.powertrain_map("MIXED") is None and L.powertrain_map("UNK") is None and L.powertrain_map("FCEV") is None
    assert L.powertrain_map(None) is None and L.powertrain_map("weird") is None and L.powertrain_map(" bev ") == "BEV"
    assert L.label_grade("fingerprint") == "A" and L.label_grade("resolved_new") == "A" and L.label_grade("alias") == "A"
    assert L.label_grade("directory") == "B"
    assert L.label_grade("unmapped") == "C" and L.label_grade(None) == "C" and L.label_grade("none") == "C"


def test_region_bucket():
    assert L.region_bucket("US") == "north_america" and L.region_bucket("Canada") == "north_america"
    assert L.region_bucket("Europe") == "europe" and L.region_bucket("Asia") == "asia"
    for v in ("Other", "Unknown", None, float("nan"), ""):
        assert L.region_bucket(v) == C.REGION_OTHER


def test_software_era():
    assert L.software_era("0.9.7") == "0.9" and L.software_era("0.8.16") == "0.8"
    assert L.software_era("0.10.1-release") == "0.10" and L.software_era("0.11.0") == "0.11"
    assert L.software_era("2025.3.1") == "2025" and L.software_era("2026.1.0") == "2026"
    assert L.software_era("0.7.1") == "other" and L.software_era(None) == "other" and L.software_era("garbage") == "other"
    for v in ("0.9.7", "2025.3.1", "x"):
        assert L.software_era(v) in C.SOFTWARE_ERAS


def test_label_note():
    assert "Prius Prime" in L.label_note("TOYOTA_PRIUS_TSS2")
    assert "R1S" in L.label_note("RIVIAN_R1_GEN1") and "R1S" in L.label_note(None, "RIVIAN_R1")
    assert "Passport" in L.label_note("HONDA_PILOT_4G") and "mild hybrid" in L.label_note("RAM_1500_5TH_GEN")
    assert L.label_note("HONDA_CIVIC") is None and L.label_note(None, None) is None
    note = L.label_note("HONDA_CIVIC", None, "curator; override: x")
    assert note == "curator; override: x"
    both = L.label_note("TOYOTA_PRIUS", None, "gas and hybrid")
    assert both.startswith("platform shared with Prius Prime") and both.endswith("gas and hybrid")


# ------------------------------------------------------------------------------------------ footprint / size class
def test_footprint():
    assert math.isclose(L.footprint_m2(2700, 1550, 1560), 2.7 * 1.555, rel_tol=1e-9)
    assert math.isnan(L.footprint_m2(None, 1550, 1560)) and math.isnan(L.footprint_m2(2700, float("nan"), 1560))
    assert math.isnan(L.footprint_m2(0, 1550, 1560)) and math.isnan(L.footprint_m2("x", 1550, 1560))
    df = pd.DataFrame({"wheelbase_mm": [2700, 2800, None], "track_front_mm": [1550, 1600, 1500], "track_rear_mm": [1560, 1600, 1500],
                       "footprint_m2": [None, 9.9, None]})
    fs = L.footprint_series(df)
    assert math.isclose(fs[0], 2.7 * 1.555) and fs[1] == 9.9 and math.isnan(fs[2])


def test_size_class_none_until_locked(monkeypatch):
    monkeypatch.setattr(C, "FOOTPRINT_THRESHOLDS_M2", None)
    assert L.size_class(4.0) is None
    assert L.cut_size_classes([3.0, 4.0, 5.0]).tolist() == [None, None, None]


def test_size_class_cut(monkeypatch):
    thr = (3.9, 4.6)
    assert L.size_class(3.0, thr) == "SMALL" and L.size_class(3.899, thr) == "SMALL"
    assert L.size_class(3.9, thr) == "MID" and L.size_class(4.599, thr) == "MID"
    assert L.size_class(4.6, thr) == "LARGE" and L.size_class(9.0, thr) == "LARGE"
    assert L.size_class(float("nan"), thr) is None and L.size_class(None, thr) is None and L.size_class("x", thr) is None
    assert L.size_class(4.0, (4.6, 3.9)) == "MID"                                     # unordered thresholds tolerated
    cut = L.cut_size_classes(pd.Series([3.0, 3.9, 4.6, None, 5.0]), thr).tolist()
    assert cut == ["SMALL", "MID", "LARGE", None, "LARGE"]
    monkeypatch.setattr(C, "FOOTPRINT_THRESHOLDS_M2", thr)
    assert L.size_class(4.0) == "MID" and L.cut_size_classes([3.0]).tolist() == ["SMALL"]


def test_helpers():
    assert L.parse_copies('["a/b/c", "d/e/f"]') == ["a/b/c", "d/e/f"]
    assert L.parse_copies(None) == [] and L.parse_copies("") == [] and L.parse_copies("{bad") == [] and L.parse_copies('{"x": 1}') == []
    assert L.parse_copies(np.array(["a/b/c"])) == ["a/b/c"]
    assert L.hours_from_segments([60, 30, None]).tolist() == [1.0, 0.5, 0.0]
    from vtt.paths import P
    assert L.rebase(P.ROUTES_LABELS, None) == P.ROUTES_LABELS
    assert L.rebase(P.ROUTES_LABELS, C.DATA_ROOT / "tmp" / "x") == C.DATA_ROOT / "tmp" / "x" / "build" / "routes_labels.parquet"


def test_platform_from_model(tables):
    specs = pd.DataFrame({"platform": ["FAKE_SEDAN", "FAKE_SEDAN_HEV_2020", "FAKE_SUV_EV", "MOCK", "FAKE_TRUCK", "FAKE_TRUCK_MMR"],
                          "model": ["Fake Sedan", "Fake Sedan", "Fake SUV", "", "Fake Truck", "Fake Truck"]})
    idx = L.model_platform_index(specs)
    assert idx["Fake SUV"] == ["FAKE_SUV_EV"] and idx["Fake Sedan"] == ["FAKE_SEDAN", "FAKE_SEDAN_HEV_2020"] and "" not in idx
    assert L.platform_from_model("Fake SUV", "BEV", idx, tables) == ("FAKE_SUV_EV", 1)
    assert L.platform_from_model("Fake SUV", "UNK", idx, tables) == ("FAKE_SUV_EV", 1)
    # two platforms share the model: the route's powertrain picks the one whose own powertrain agrees
    assert L.platform_from_model("Fake Sedan", "HEV", idx, tables) == ("FAKE_SEDAN_HEV_2020", 1)
    assert L.platform_from_model("Fake Sedan", "ICE", idx, tables) == ("FAKE_SEDAN", 1)
    assert L.platform_from_model("Fake Sedan", "MIXED", idx, tables) == (None, 2)          # no powertrain to split them
    assert L.platform_from_model("Fake Truck", "ICE", idx, tables) == (None, 2)            # both candidates are ICE
    assert L.platform_from_model("Unknown", "ICE", idx, tables) == (None, 0)
    assert L.platform_from_model("?WEIRD", "ICE", idx, tables) == (None, 0) and L.platform_from_model(None, "ICE", idx, tables) == (None, 0)
    assert L.platform_from_model("Fake Sedan", "ICE", {}, tables) == (None, 0)
    assert L.model_platform_index(None) == {} and L.model_platform_index(pd.DataFrame({"platform": ["X"]})) == {}


# ---------------------------------------------------------------------------------------------------- real tables
@pytest.mark.skipif(not (C.TAXONOMY_DIR / "taxonomy_tables.py").exists(), reason="taxonomy tables not on this machine")
def test_real_taxonomy_tables_consistent():
    T = L.taxonomy_tables()
    assert len(T.MODELS) > 100 and len(T.DIRS) > 300 and len(T.ALIAS) > 50
    bad = [k for k, v in T.DIRS.items() if v[0] not in T.MODELS]
    assert bad == []
    bad_alias = [k for k, v in T.ALIAS.items() if v[0] not in T.MODELS]
    assert bad_alias == []
    assert L.resolve("TOYOTA_RAV4_TSS2")[2] == "fingerprint" and L.resolve("TOYOTA_RAV4_TSS2")[1] == "MIXED"
    assert L.resolve("KIA_SPORTAGE_HYBRID_5TH_GEN")[2] == "alias"
    assert set(v[1] for v in T.DIRS.values()) <= set(C.POWERTRAIN_MAP)


def test_apply_model_merges_renames_only_the_model_and_notes_it():
    import pandas as pd
    from vtt.labels import apply_model_merges
    df = pd.DataFrame({"model": ["Rivian R1S", "Rivian R1T/R1S", "Honda Civic"], "platform": ["RIVIAN_R1S", "RIVIAN_R1", "HONDA_CIVIC"],
                       "label_note": [None, "x", None]})
    merges = pd.DataFrame({"from_model": ["Rivian R1S"], "to_model": ["Rivian R1T/R1S"], "reason": ["same platform family"]})
    out, counts = apply_model_merges(df, merges)
    assert counts == {"Rivian R1S": 1}
    assert out["model"].tolist() == ["Rivian R1T/R1S", "Rivian R1T/R1S", "Honda Civic"]
    assert out["platform"].tolist() == ["RIVIAN_R1S", "RIVIAN_R1", "HONDA_CIVIC"]          # platform untouched
    assert out["label_note"].iloc[0] == "model merged into Rivian R1T/R1S (same platform family)"
    assert out["label_note"].iloc[1] == "x" and out["label_note"].iloc[2] is None
    same, c0 = apply_model_merges(df, None)
    assert c0 == {} and same["model"].tolist() == df["model"].tolist()
