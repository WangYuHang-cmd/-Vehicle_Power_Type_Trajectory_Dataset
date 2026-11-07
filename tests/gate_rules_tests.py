"""Positive and negative tests for every gate rule. Every identifier-like token is built at run time, so this file
itself passes the pre-push gate (which allow-lists nothing under tests/)."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from vtt.gate import rules as R
from vtt.paths import P


def hex16(n: int = 1) -> str:
    return "f" * 10 + "%06x" % n


def hex12(n: int = 1) -> str:
    return "e" * 6 + "%06x" % n


def route_date(i: int = 0) -> str:
    return f"{2020 + i:04d}-{1:02d}-{2:02d}--{3:02d}-{4:02d}-{5:02d}"


def route_num() -> str:
    return "a" * 8 + "--" + "b" * 10


def iso() -> str:
    return f"{2024}-{5:02d}-{6:02d}"


def vin() -> str:
    return "1HGBH41JXMN" + "109186"


def acronym() -> str:
    return "AD" + "AS"


def a_path() -> str:
    return "/ho" + "me/user/x.txt"


# ------------------------------------------------------------------------------------------------ hex / route ids
def test_hex16_positive_and_negative():
    assert R.hits_hex16(f"id {hex16()} here") == 1
    assert R.hits_hex16("n=" + "1234567890" + "123456" + " end") == 1          # all-digit 16 chars is still an id shape
    assert R.hits_hex16(hex16(1) + "\n" + hex16(2)) == 2
    assert R.hits_hex16("1234567890123456.0") == 0                         # float with a decimal point
    assert R.hits_hex16("5.69731309" + "9517487e-05") == 0                   # exponent float
    assert R.hits_hex16("a" * 64) == 0                                       # sha256 digest
    assert R.hits_hex16("x" + hex16()) == 0                                  # glued to a letter
    assert R.hits_hex16("f" * 15) == 0


def test_hex16_json_numeric_context():
    j = '{"id": "' + hex16() + '", "t": ' + "1234567890" + "123456" + ', "x": ' + "1234567890" + "123456" + '.0}'
    assert R.hits_hex16(j, is_json=True) == 1                                # the quoted id only
    assert R.hits_hex16(j, is_json=False) == 2                               # in prose the long int is reported too


def test_hex12_positive_and_negative():
    assert R.hits_hex12(f"uid {hex12()}") == 1
    assert R.hits_hex12(hex16()) == 0                                        # part of a longer token
    assert R.hits_hex12("1234.5678901234567") == 0
    assert R.hits_hex12(hex12(1) + " " + hex12(2)) == 2


def test_route_patterns():
    assert R.hits_route_date(f"x {route_date()} y") == 1
    assert R.hits_route_date(iso()) == 0
    assert R.hits_route_num(route_num()) == 1
    assert R.hits_route_num("a" * 8 + "--" + "b" * 9) == 0


def test_iso_date_and_allow():
    assert R.hits_iso_date(f"on {iso()}") == 1
    assert R.hits_iso_date(route_date()) == 0                                # counted under ROUTE_DATE instead
    assert R.hits_iso_date("20240506") == 0
    assert "ISO_DATE" in R.scan_text(iso())
    assert "ISO_DATE" not in R.scan_text(iso(), iso_allowed=True)
    assert "ROUTE_DATE" in R.scan_text(route_date(), iso_allowed=True)      # hard rule survives the allow


# ------------------------------------------------------------------------------------------------ words
def test_paths():
    assert R.hits_paths(a_path()) == 1
    assert R.hits_paths("home/user/x") == 0
    assert R.hits_paths("/dat" + "a/x") == 1
    assert R.hits_paths("/tmpfs/x") == 0
    assert R.hits_paths("build/dat" + "a/vehicle_0001/x.parquet") == 0          # relative fragment
    assert R.hits_paths("<root>/tm" + "p/x and $D/tm" + "p/y and ./dat" + "a/z") == 0   # placeholder paths
    assert R.hits_paths("root=/dat" + "a/x `/ho" + "me/u/y` (/mn" + "t/z)") == 3        # still absolute


def test_coord_words_in_text():
    for w in ("la" + "t", "Lo" + "n", "G" + "PS", "ge" + "o", "coun" + "try", "us_" + "state", "in_" + "house", "lati" + "tude", "ln" + "g"):
        assert R.hits_coord(f"the {w} value") == 1, w
    for w in ("lateral", "along", "latency", "geometry", "platform", "lat_acc_mps2", "longitudinal", "contrary"):
        assert R.hits_coord(w) == 0, w


def test_coord_column_names():
    for name in ("la" + "t", "start_" + "la" + "t", "la" + "t_deg", "lo" + "n_mean", "g" + "ps_speed", "in_" + "house", "ge" + "o_region", "us_" + "state"):
        assert R.col_coord(name), name
    for name in ("lat_acc_mps2", "op_lat_active", "op_lat_frac", "platform", "lead_d_m", "op_long_active", "v_mps"):
        assert not R.col_coord(name), name


def test_vin_tokens():
    assert R.hits_vin(f"vin {vin()} x") == 1
    assert R.hits_vin(vin().lower()) == 1
    assert R.hits_vin("A" * 17) == 0                                         # no digit
    assert R.hits_vin("1" * 17) == 0                                         # no letter
    assert R.hits_vin(vin()[:-1] + "I") == 0                                 # forbidden letter
    assert R.hits_vin(vin() + "7") == 0                                      # 18 chars
    assert R.hits_vin("drive_000001_w12 vehicle_0001") == 0


def test_column_name_rules():
    assert R.col_vin("vin_hash") and R.col_vin("VIN") and R.col_vin("vin")
    assert not R.col_vin("driving") and not R.col_vin("vinyl")
    for name in ("logMonoTime", "t_ns", "timestamp_s", "mono"):
        assert R.col_mono(name), name
    for name in ("t_s", "seg", "duration_s", "n_segments"):
        assert not R.col_mono(name), name
    for name in ("route_uid", "uid", "git_commit", "version", "fingerprint_raw", "dongle_log", "route_id", "rel_path", "salt", "identity_uid"):
        assert R.col_denied(name), name
    for name in ("window_id", "weight", "platform", "software_era"):
        assert not R.col_denied(name), name
    hits = R.scan_names(["t_s", "logMonoTime", "vin_hash", "la" + "t"])
    assert hits == {"MONO": 1, "VIN": 1, "COORD": 1}


def test_email_rule():
    allowed = "contact@" + "example.org"
    other = "someone@" + "example.net"
    assert R.hits_email(f"mail {allowed}", allowed) == 0
    assert R.hits_email(f"mail {allowed.upper()}", allowed) == 0
    assert R.hits_email(f"mail {other}", allowed) == 1
    assert R.hits_email(f"{allowed} {other}", allowed) == 1
    assert R.hits_email(other, "") == 1


def test_anonymous_authors():
    assert R.hits_anon_auth("by Anonymous " + "Authors") == 1
    assert R.hits_anon_auth("anonymous  " + "author") == 1
    assert R.hits_anon_auth("anonymized authors") == 0


def test_acronym_rule():
    assert R.hits_acronym(f"{acronym()} engaged") == 1
    assert R.hits_acronym(f"({acronym()})") == 1
    assert R.hits_acronym(acronym().lower()) == 1
    assert R.hits_acronym(acronym() + "-TO") == 0
    assert R.hits_acronym(acronym() + "-TO-CF extractor") == 0
    assert R.hits_acronym(acronym() + "_WORD") == 0
    fence = "intro\n\n```bibtex\n@article{x,\n  title = {" + acronym() + "-TO: " + acronym() + " Engagement},\n}\n```\n"
    assert R.hits_acronym(fence, readme=True) == 0
    assert R.hits_acronym(fence, readme=False) == 1
    assert R.hits_acronym(fence + fence, readme=True) >= 1                   # a second fence is not exempt


def test_forbidden_tokens():
    tokens = ["Zorblax", "@fake.example"]
    assert R.hits_forbidden("by zorblax today", tokens) == 1
    assert R.hits_forbidden("zorblaxes", tokens) == 0
    assert R.hits_forbidden("mail x@" + "fake.example now", tokens) == 1
    assert R.hits_forbidden("nothing here", tokens) == 0
    assert R.hits_forbidden("zorblax", None) == 0


def test_exact_private_tokens():
    private = frozenset({hex16(3), "secretfolder"})
    assert R.hits_exact(f"see {hex16(3)}", private) == 1
    assert R.hits_exact("CAR_X/" + hex16(3) + "/" + route_num(), private) >= 1
    assert R.hits_exact("SecretFolder/x", private) == 1
    assert R.hits_exact(f"see {hex16(4)}", private) == 0
    assert R.hits_exact("anything", None) == 0


def test_scan_text_aggregate_and_clean():
    assert R.scan_text("A clean sentence about vehicles and speed.") == {}
    hits = R.scan_text(f"{hex16()} {hex12()} {route_date()} {route_num()} {iso()} {a_path()} {vin()} {acronym()}")
    for rule in ("HEX16", "HEX12", "ROUTE_DATE", "ROUTE_NUM", "ISO_DATE", "PATHS", "VIN", "ACRONYM"):
        assert hits[rule] == 1, rule


def test_gate_mask_replaces_everything():
    out = R.gate_mask(f"{hex16()} {hex12()} {route_date()} {route_num()} x@{'ex.org'} {vin()}")
    assert hex16() not in out and hex12() not in out and route_date() not in out and vin() not in out
    for tag in ("<hex16>", "<hex12>", "<route_date>", "<route_num>", "<email>", "<vin>"):
        assert tag in out, tag


def test_id_formats():
    assert R.id_format_ok("vehicle_id", "vehicle_0001") and R.id_format_ok("drive_id", "drive_000123")
    assert R.id_format_ok("group_id", "group_0007") and R.id_format_ok("window_id", "drive_000123_w12")
    assert not R.id_format_ok("vehicle_id", hex16()) and not R.id_format_ok("drive_id", "drive_1")
    assert R.id_format_ok("platform", "ANY_NAME")


def test_load_allow_and_specs():
    allow = R.load_allow()
    assert allow["email"] and "@" in allow["email"]
    assert isinstance(allow["iso_date_files"], list)
    assert R.scan_text(R.ALLOW_PATH.read_text(), is_json=True, allowed_email=allow["email"]) == {}
    for spec in R.TABLE_SPECS.values():
        names = [t[0] for t in spec]
        assert len(names) == len(set(names))
    assert set(R.SHARD_SEQ_COLUMNS) == set(R.SHARD_SEQ_COLUMNS) and "window_id" in R.SPLIT_TABLE_ALLOW and "weight" in R.SPLIT_TABLE_ALLOW
    assert all(r not in R.SOFT_RULES for r in R.HARD_RULES)


def test_private_tokens_from_fake_private_root():
    P.TMP.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="vtt_priv_", dir=P.TMP))
    (root / "forbidden_tokens.txt").write_text("# comment\nZorblax\n\n")
    pd.DataFrame({"route_uid": [hex12(9)], "car_dir": ["CAR_X"], "dongle_dir": [hex16(9)], "dongle_log": [hex16(9)],
                  "route_id": [route_num()], "identity_uid": [hex12(8)], "vin_hash": [None]}).to_parquet(root / "uid_map.parquet", index=False)
    toks = R.private_tokens(root)
    assert toks is not None
    assert hex16(9) in toks and hex12(9) in toks and hex12(8) in toks and route_num() in toks and "zorblax" in toks
    assert "car_x" not in toks                                                # car folder names are public model names
    assert R.private_tokens(None) is None
    assert R.private_tokens(root / "missing") is None
    assert json.dumps(sorted(toks)) and R.hits_exact(f"found {hex16(9)}", toks) == 1
    shutil.rmtree(root, ignore_errors=True)


def test_private_tokens_keep_folders_and_routes_but_not_car_model_folders():
    """rel_path / copies_rel contribute the whole path, the dongle folders (possible person names) and the route ids,
    never the first component: a public platform spelling equal to the raw car folder must not block the release."""
    P.TMP.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="vtt_priv2_", dir=P.TMP))
    plat = "SYNTH_PLATFORM_X"
    rel = f"{plat}/{hex16(11)}/{route_num()}"
    copies = json.dumps([rel, f"{plat}/PersonFolder/{route_date(1)}"])
    pd.DataFrame({"route_uid": [hex12(11)], "dongle_log": [hex16(11)], "rel_path": [rel], "copies_rel": [copies],
                  "car_dir": [plat]}).to_parquet(root / "routes_resolved.parquet", index=False)
    toks = R.private_tokens(root)
    assert plat.lower() not in toks and "car_dir" not in toks
    assert rel.lower() in toks and hex16(11) in toks and route_num() in toks
    assert "personfolder" in toks and route_date(1).lower() in toks
    assert R.hits_exact(f"platform {plat} in data/vehicle_0001/drive_000001.parquet", toks) == 0
    assert R.hits_exact(f"folder PersonFolder and {hex16(11)}", toks) == 2
    assert R.hits_exact("CAR/" + hex16(11) + "/" + route_num(), toks) == 2
    shutil.rmtree(root, ignore_errors=True)


def test_private_tokens_drop_structural_alphabetic_path_components_only():
    """A purely alphabetic path component shared by many paths (a log sub-folder of the raw layout) is layout, not an
    identifier, and must not block a public file that uses the same word; rare components and identifier columns stay."""
    P.TMP.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="vtt_priv3_", dir=P.TMP))
    n = R.STRUCTURAL_MIN_PATHS + 5
    dongles = [f"{i + 1:016x}" for i in range(n)]
    rels = [f"SYNTH_CAR/{d}/{i:08x}--{i + 7:010x}/logsub" for i, d in enumerate(dongles)]
    rels[0] = rels[0].replace("/logsub", "/rarefolder")
    pd.DataFrame({"route_uid": [f"{i + 1:012x}" for i in range(n)], "dongle_log": ["frequentword"] * n,
                  "rel_path": rels, "copies_rel": [json.dumps([r]) for r in rels], "car_dir": ["SYNTH_CAR"] * n}
                 ).to_parquet(root / "routes_resolved.parquet", index=False)
    toks = R.private_tokens(root)
    assert "logsub" not in toks and "rarefolder" not in toks     # plain-word sub-folders are layout, frequent or not
    assert "frequentword" in toks                     # identifier column: kept whatever its frequency
    assert all(d in toks for d in dongles) and rels[3].lower() in toks
    assert R.hits_exact("files under logsub are read first", toks) == 0
    assert R.hits_exact("see rarefolder", toks) == 0 and R.hits_exact(rels[3], toks) >= 1
    shutil.rmtree(root, ignore_errors=True)


def test_private_tokens_route_ids_and_sub_folders_need_a_digit():
    """A route folder or a deeper path component without any digit is raw-library layout (a stray log sub-folder),
    never an identifier; dongle-position folders (hex or person names) and real route ids are kept."""
    P.TMP.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="vtt_priv4_", dir=P.TMP))
    dongle = f"{77:016x}"
    rel_ok = f"SYNTH_CAR/{dongle}/{5:08x}--{9:010x}/logsub"
    rel_odd = f"SYNTH_CAR/PersonName/strayfolder"
    pd.DataFrame({"route_uid": [f"{1:012x}", f"{2:012x}"], "dongle_log": [dongle, dongle], "route_id": [f"{5:08x}--{9:010x}", "strayfolder"],
                  "rel_path": [rel_ok, rel_odd], "copies_rel": ["", ""], "car_dir": ["SYNTH_CAR", "SYNTH_CAR"]}
                 ).to_parquet(root / "routes_resolved.parquet", index=False)
    toks = R.private_tokens(root)
    assert "strayfolder" not in toks and "logsub" not in toks
    assert "personname" in toks and dongle in toks and f"{5:08x}--{9:010x}" in toks
    assert rel_odd.lower() in toks                     # the whole odd path still is a private string
    assert R.hits_exact("the strayfolder and logsub words are public", toks) == 0
    shutil.rmtree(root, ignore_errors=True)
