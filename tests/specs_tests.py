"""vtt.specs: opendbc AST parsing, fingerprint -> platform mapping methods, specs skeleton merge; plus a read-only
check that the committed fingerprint_to_platform.csv covers every fingerprint of the real routes_taxonomy.csv."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from tests import s3_synth as SY
from vtt import config as C
from vtt import specs as S


@pytest.fixture(scope="module")
def synth():
    root = SY.mkdtemp("vtt_specs_")
    SY.make_opendbc(root / "opendbc")
    pl = S.parse_platforms(root / "opendbc")
    mig = S.parse_migration(root / "opendbc")
    yield dict(root=root, platforms=pl, migration=mig)
    SY.rmtree(root)


# ------------------------------------------------------------------------------------------------------- parsing
def test_parse_platforms(synth):
    pl = synth["platforms"].set_index("platform")
    assert set(pl.index) == {"FAKE_SEDAN", "FAKE_SEDAN_2020", "FAKE_SEDAN_HEV_2020", "FAKE_SUV_EV", "FAKE_SUV_EV_2ND_GEN",
                             "FAKE_TRUCK", "FAKE_TRUCK_IC", "FAKE_TRUCK_LT", "FAKE_TRUCK_HD"}
    assert math.isclose(pl.loc["FAKE_SEDAN", "mass_kg"], 3505 * 0.453592, rel_tol=1e-9)        # lb expression evaluated
    assert pl.loc["FAKE_SEDAN", "wheelbase_mm"] == 2820.0 and pl.loc["FAKE_SEDAN", "steer_ratio"] == 14.8
    assert pl.loc["FAKE_SEDAN_2020", "wheelbase_mm"] == 2820.0                                  # OTHER.specs reference
    assert pl.loc["FAKE_SUV_EV", "mass_kg"] == 2100.0 and pl.loc["FAKE_SUV_EV", "wheelbase_mm"] == 2900.0   # specs= keyword
    assert pl.loc["FAKE_TRUCK", "mass_kg"] == 2500.0                                             # GMCarSpecs subclass
    assert pl.loc["FAKE_TRUCK_IC", "wheelbase_mm"] == 3745.0                                     # X.override(...) inherits
    assert pl.loc["FAKE_TRUCK_IC", "docs"] == "Fake Truck 2019-21"
    assert pl.loc["FAKE_SEDAN", "docs"] == "Fake Sedan 2017-19 | Fake Sedan Hybrid 2017-19"
    assert pl.loc["FAKE_SEDAN", "years"] == "2017-2019" and pl.loc["FAKE_TRUCK_LT", "brand_dir"] == "gmfake"


def test_parse_migration(synth):
    assert synth["migration"] == {"FAKE SEDAN 2017": "FAKE_SEDAN", "FAKE TRUCK LEGACY": "FAKE_TRUCK"}
    assert S.parse_migration(synth["root"] / "nowhere") == {}


def test_doc_years():
    assert S.doc_years("Toyota RAV4 2019-21") == {2019, 2020, 2021}
    assert S.doc_years("Hyundai Kona 2017-2019") == {2017, 2018, 2019}
    assert S.doc_years("Fake Car 2022") == {2022} and S.doc_years("comma body") == set()
    assert S.doc_years("Nissan Leaf 2018-23") == set(range(2018, 2024))


def test_eval_safety():
    import ast
    assert S._eval(ast.parse("3505. * CV.LB_TO_KG", mode="eval").body) == pytest.approx(3505 * 0.453592)
    assert S._eval(ast.parse("1500 + STD_CARGO_KG", mode="eval").body) == 1636.0
    assert S._eval(ast.parse("-2 * 3", mode="eval").body) == -6.0
    assert S._eval(ast.parse("UNKNOWN_NAME * 2", mode="eval").body) is None
    assert S._eval(ast.parse("__import__('os')", mode="eval").body) is None
    assert S._eval(ast.parse("True", mode="eval").body) is None


# ------------------------------------------------------------------------------------------------------- mapping
def test_normalize():
    assert S.normalize(" fake  sedan 2017 ") == "FAKE_SEDAN_2017" and S.normalize("FAKE_SEDAN") == "FAKE_SEDAN"


@pytest.mark.parametrize("fp,platform,method,note_part", [
    ("FAKE_SEDAN", "FAKE_SEDAN", "exact", ""),
    ("fake sedan", "FAKE_SEDAN", "exact", "normalized"),
    ("FAKE SEDAN 2017", "FAKE_SEDAN", "migration", ""),
    ("FAKE_SEDAN_2017", "FAKE_SEDAN", "migration", "normalized"),
    ("FAKE_TRUCK_MMR", "FAKE_TRUCK", "suffix", "stripped _MMR"),
    ("FAKE_TRUCK_MMR_PEDAL", "FAKE_TRUCK", "suffix", "stripped _MMR_PEDAL"),
    ("FAKE TRUCK LEGACY NON SCC", "FAKE_TRUCK", "suffix", "migration"),
    ("FAKE_SEDAN_2021", "FAKE_SEDAN_2020", "suffix", "year 2021"),             # docs 2020-22; HEV twin excluded
    ("FAKE_SEDAN_HEV_2021", "FAKE_SEDAN_HEV_2020", "suffix", "year 2021"),
    ("FAKE_SEDAN_2018", "FAKE_SEDAN", "suffix", "year 2018"),
    ("FAKE_SUV_EV_2023", "FAKE_SUV_EV", "suffix", "kept base"),               # two candidates incl. the base
    ("FAKE_TRUCK_2022", "FAKE_TRUCK_2022", "manual", "UNRESOLVED; year candidates: FAKE_TRUCK_HD|FAKE_TRUCK_LT"),
    ("FAKE_SEDAN_2030", "FAKE_SEDAN_2030", "manual", "UNRESOLVED"),
    ("TOTALLY_NEW_CAR", "TOTALLY_NEW_CAR", "manual", "UNRESOLVED"),
    ("totally new car", "TOTALLY_NEW_CAR", "manual", "UNRESOLVED"),
])
def test_map_fingerprint(synth, fp, platform, method, note_part):
    p, m, note = S.map_fingerprint(fp, synth["platforms"], synth["migration"])
    assert (p, m) == (platform, method), (p, m, note)
    assert note_part in note
    assert m in S.METHODS


def test_map_fingerprint_with_plain_set(synth):
    assert S.map_fingerprint("FAKE_SEDAN", {"FAKE_SEDAN"}, {}) == ("FAKE_SEDAN", "exact", "")
    assert S.map_fingerprint("FAKE_SEDAN_2021", {"FAKE_SEDAN", "FAKE_SEDAN_2020"}, {})[1] == "manual"   # no docs -> no year rule


def test_map_fingerprints_table_and_manual_preservation(synth):
    fps = ["FAKE_SEDAN", "TOTALLY_NEW_CAR", "FAKE_TRUCK_2022", "", None, float("nan"), " FAKE_SEDAN "]
    m = S.map_fingerprints(fps, synth["platforms"], synth["migration"])
    assert list(m.columns) == list(S.FP2PLATFORM_COLUMNS)
    assert m["fingerprint"].tolist() == ["FAKE_SEDAN", "FAKE_TRUCK_2022", "TOTALLY_NEW_CAR"]        # sorted, deduplicated, no empties
    existing = pd.DataFrame([
        {"fingerprint": "TOTALLY_NEW_CAR", "platform": "FAKE_SUV_EV", "method": "manual", "note": "hand: fork name of the SUV"},
        {"fingerprint": "FAKE_TRUCK_2022", "platform": "FAKE_TRUCK_2022", "method": "manual", "note": "UNRESOLVED; year candidates: x"},
        {"fingerprint": "FAKE_SEDAN", "platform": "WRONG", "method": "exact", "note": ""},
    ])
    m2 = S.map_fingerprints(fps, synth["platforms"], synth["migration"], existing).set_index("fingerprint")
    assert m2.loc["TOTALLY_NEW_CAR", "platform"] == "FAKE_SUV_EV" and m2.loc["TOTALLY_NEW_CAR", "note"].startswith("hand")
    assert m2.loc["FAKE_TRUCK_2022", "note"].startswith(S.UNRESOLVED)                                  # recomputed, still unresolved
    assert m2.loc["FAKE_SEDAN", "platform"] == "FAKE_SEDAN"                                             # automatic rows are recomputed


def test_fp2platform_dict_roundtrip(synth):
    root = synth["root"]
    m = S.map_fingerprints(["FAKE_SEDAN", "fake sedan 2017"], synth["platforms"], synth["migration"])
    p = S.write_csv_sorted(m, root / "specs" / "fingerprint_to_platform.csv")
    d = S.fp2platform_dict(p)
    assert d["FAKE_SEDAN"] == "FAKE_SEDAN" and d["fake sedan 2017"] == "FAKE_SEDAN" and d["FAKE_SEDAN_2017"] == "FAKE_SEDAN"
    assert S.read_fp2platform(root / "nope.csv").empty


# ------------------------------------------------------------------------------------------------------- skeleton
def test_skeleton_and_merge(synth):
    mapping = S.map_fingerprints(["FAKE_SEDAN", "FAKE_TRUCK_MMR", "TOTALLY_NEW_CAR"], synth["platforms"], synth["migration"])
    pm = {"FAKE_SEDAN": ("Fake", "Fake Sedan", "sedan", "D"), "FAKE_TRUCK": ("Fake", "Fake Truck", "pickup", "PU-F")}
    sk = S.skeleton_specs(mapping, synth["platforms"], pm)
    assert list(sk.columns) == list(S.PLATFORM_SPECS_COLUMNS)
    assert sk["platform"].tolist() == ["FAKE_SEDAN", "FAKE_TRUCK", "TOTALLY_NEW_CAR"]
    row = sk.set_index("platform").loc["FAKE_SEDAN"]
    assert row["brand"] == "Fake" and row["model"] == "Fake Sedan" and row["body"] == "sedan"
    assert row["wheelbase_mm"] == "2820" and row["wheelbase_source"] == S.OPENDBC_SOURCE
    assert row["curb_mass_kg"] == "1590" and row["mass_source"] == S.OPENDBC_SOURCE
    assert row["track_front_mm"] == "" and row["spec_confidence"] == "C" and "Fake Sedan 2017-19" in row["generation_note"]
    new = sk.set_index("platform").loc["TOTALLY_NEW_CAR"]
    assert new["wheelbase_mm"] == "" and new["brand"] == "" and new["spec_confidence"] == "C"

    existing = pd.DataFrame([
        {"platform": "FAKE_SEDAN", "brand": "Fake", "model": "Fake Sedan", "body": "sedan", "generation_note": "", "wheelbase_mm": "2825",
         "track_front_mm": "1580", "track_rear_mm": "1590", "footprint_m2": "", "footprint_min_m2": "", "footprint_max_m2": "",
         "curb_mass_kg": "", "mass_source": "", "wheelbase_source": "oem", "track_source": "oem", "spec_confidence": "A",
         "source_urls": "https://example.org/spec", "note": "hand filled"},
        {"platform": "OLD_ONLY", "brand": "Old", "model": "Old Car", "body": "sedan", "generation_note": "", "wheelbase_mm": "2500",
         "track_front_mm": "", "track_rear_mm": "", "footprint_m2": "", "footprint_min_m2": "", "footprint_max_m2": "",
         "curb_mass_kg": "", "mass_source": "", "wheelbase_source": "", "track_source": "", "spec_confidence": "",
         "source_urls": "", "note": ""},
    ])
    merged = S.merge_specs(existing, sk).set_index("platform")
    assert merged.index.tolist() == ["FAKE_SEDAN", "FAKE_TRUCK", "OLD_ONLY", "TOTALLY_NEW_CAR"]
    fs = merged.loc["FAKE_SEDAN"]
    assert fs["wheelbase_mm"] == "2825" and fs["wheelbase_source"] == "oem" and fs["spec_confidence"] == "A"   # filled cells win
    assert fs["track_front_mm"] == "1580" and fs["note"] == "hand filled"
    assert fs["curb_mass_kg"] == "1590" and fs["mass_source"] == S.OPENDBC_SOURCE                               # empty cells filled
    assert "Fake Sedan 2017-19" in fs["generation_note"]
    assert merged.loc["OLD_ONLY", "wheelbase_mm"] == "2500" and merged.loc["OLD_ONLY", "spec_confidence"] == "C"
    assert merged.loc["FAKE_TRUCK", "wheelbase_mm"] == "3745"
    assert S.merge_specs(None, sk).equals(sk.sort_values("platform").reset_index(drop=True))


def test_read_platform_specs_numeric(synth):
    root = synth["root"]
    df = pd.DataFrame([{c: "" for c in S.PLATFORM_SPECS_COLUMNS}])
    df.loc[0, ["platform", "wheelbase_mm", "track_front_mm", "track_rear_mm", "spec_confidence"]] = ["X", "2700", "1550", "", "B"]
    p = S.write_csv_sorted(df, root / "specs" / "platform_specs.csv")
    r = S.read_platform_specs(p)
    assert r.loc[0, "wheelbase_mm"] == 2700.0 and r.loc[0, "track_front_mm"] == 1550.0 and math.isnan(r.loc[0, "track_rear_mm"])
    assert r.loc[0, "spec_confidence"] == "B"
    assert S.read_platform_specs(root / "none.csv").empty


# ------------------------------------------------------------------------------------------------------ real data
REAL_TAX = C.TAXONOMY_DIR / "csv" / "routes_taxonomy.csv"


@pytest.mark.skipif(not (REAL_TAX.exists() and S.FP2PLATFORM_CSV.exists()), reason="real taxonomy csv or mapping csv absent")
def test_real_mapping_covers_every_taxonomy_fingerprint():
    """Read-only: every distinct fingerprint spelling of routes_taxonomy.csv has a mapping row; reports method counts and
    the hours share that is still unresolved (run pytest -s to see the numbers)."""
    t = pd.read_csv(REAL_TAX, usecols=["fingerprint", "dur_min", "source", "car_dir"], low_memory=False)
    m = S.read_fp2platform()
    assert list(m.columns) == list(S.FP2PLATFORM_COLUMNS)
    assert not m["fingerprint"].duplicated().any()
    assert set(m["method"]) <= set(S.METHODS)
    assert (m["platform"].str.strip() != "").all()
    fps = {str(x).strip() for x in t["fingerprint"].dropna() if str(x).strip()}
    missing = fps - set(m["fingerprint"])
    assert missing == set(), f"{len(missing)} taxonomy fingerprints without a mapping row"
    folders = {str(x).strip() for x in t.loc[t["source"].isin(["directory", "fw-match"]), "car_dir"].dropna()
               if str(x).strip() and S.normalize(x) not in ("UNKNOWN", "MOCK")}
    missing_dirs = folders - set(m["fingerprint"])
    assert missing_dirs == set(), f"{len(missing_dirs)} folder spellings of directory / fw-match rows without a mapping row"
    hours = t.dropna(subset=["fingerprint"]).groupby("fingerprint")["dur_min"].sum() / 60.0
    method = m.set_index("fingerprint")["method"]
    note = m.set_index("fingerprint")["note"]
    unresolved = [fp for fp in hours.index if method.get(fp) == "manual" and str(note.get(fp, "")).startswith(S.UNRESOLVED)]
    share = float(hours[unresolved].sum()) / float(hours.sum())
    counts = m["method"].value_counts().to_dict()
    print(f"\nfingerprint_to_platform: {len(m)} rows; methods {counts}; taxonomy fingerprints {len(fps)}; "
          f"unresolved {len(unresolved)} spellings = {100 * share:.1f} % of hours")
    assert 0.0 <= share <= 1.0


@pytest.mark.skipif(not S.PLATFORM_SPECS_CSV.exists(), reason="platform_specs.csv absent")
def test_real_platform_specs_consistent():
    sp = S.read_platform_specs()
    m = S.read_fp2platform()
    assert list(sp.columns) == list(S.PLATFORM_SPECS_COLUMNS)
    assert not sp["platform"].duplicated().any()
    assert set(m["platform"]) <= set(sp["platform"])                       # every mapped platform has a specs row
    assert set(sp["spec_confidence"].dropna()) <= {"A", "B", "C"}
    filled = sp.dropna(subset=["wheelbase_mm", "track_front_mm", "track_rear_mm"])
    if len(filled):
        fp = filled["wheelbase_mm"] / 1000 * (filled["track_front_mm"] + filled["track_rear_mm"]) / 2000
        assert ((fp > 2.0) & (fp < 8.0)).all()                              # plausible footprints (m^2)
